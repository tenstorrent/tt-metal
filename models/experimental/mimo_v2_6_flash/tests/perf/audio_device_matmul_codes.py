# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Would a device tokenizer encoder keep the RVQ codes? Every matmul on the device, everything else exact.

Blackhole's fp32 elementwise ops (add, sub, mul, exp, rsqrt, reciprocal, division, typecast with round-to-nearest-even,
``ttnn.sum``) are exact to ~1e-7, but its matmuls are not: products of bf16 operands are exact and the accumulation
across K tiles in an fp32 dest is exact, yet each 32-term tile dot product keeps only ~13-14 bits below its largest
product, so an fp32-output matmul has ~3e-4 relative error (measured on one chip; no math fidelity or packer setting
changes it). This script runs the fp32 tokenizer encoder of reference/audio.py with every matmul (linears, the three
convolutions, q.k and P.v) executed by ``ttnn.matmul`` on one chip and every other op on the host in fp32, i.e. the
numerics of a device port whose elementwise ops are exact, and counts the 25 Hz code frames that change against the
fp32 reference, with their RVQ decision margins (tests/perf/audio_codes_margin.py).

Schemes (``--schemes``): ``split`` (activations as two bf16 parts: 2 matmuls per weight matmul, 3 for q.k and P.v;
the precision scheme of audio_device_precision.py), ``bf16`` (bf16 activations, 1 matmul each).

    TT_VISIBLE_DEVICES=<one chip> python models/experimental/mimo_v2_6_flash/tests/perf/audio_device_matmul_codes.py \\
        --clips <dir> [--clip speech_5s speech_30s ...] [--schemes split bf16]
"""

import argparse
import contextlib
import os
import time

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tests.perf.audio_codes_margin import margins
from models.experimental.mimo_v2_6_flash.tests.perf.audio_host_latency import load_mel

_linear, _conv1d = F.linear, F.conv1d


class DeviceMatmul:
    """``a @ b`` with fp32 ``a`` and bf16-exact ``b`` (weights) or fp32 ``b`` (activations), on one chip."""

    def __init__(self, mesh, split):
        self.mesh, self.split = mesh, split
        self.cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._weights = {}
        self.calls = 0

    def _dev(self, t):
        return ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mm(self, a16, b16_dev):
        self.calls += 1
        out = ttnn.matmul(self._dev(a16), b16_dev, dtype=ttnn.float32, compute_kernel_config=self.cfg)
        return ttnn.to_torch(out).float()

    def weight(self, w):
        """A linear / conv weight ([N, K] or [N, Cin, k]) as a device [K, N] bf16 matrix, uploaded once."""
        key = (w.data_ptr(), tuple(w.shape))
        if key not in self._weights:
            assert torch.equal(w.to(torch.bfloat16).float(), w.float()), "weights must be bf16-exact"
            self._weights[key] = self._dev(w.reshape(w.shape[0], -1).t().contiguous())
        return self._weights[key]

    def parts(self, x):
        hi = x.to(torch.bfloat16).float()
        return (hi, (x - hi).to(torch.bfloat16).float()) if self.split else (hi,)

    def with_weight(self, x, w_dev):
        """x [M, K] fp32 @ device weight -> fp32 [M, N]; split parts summed in fp32 (an exact SFPU add)."""
        out = None
        for p in self.parts(x):
            y = self._mm(p, w_dev)
            out = y if out is None else out + y
        return out

    def batched(self, a, b):
        """a [B, M, K] fp32 @ b [B, K, N] fp32: 3 matmuls (hi.hi + hi.lo + lo.hi) when split, else 1."""
        pa, pb = self.parts(a), self.parts(b)
        pairs = [(0, 0), (0, 1), (1, 0)] if self.split else [(0, 0)]
        out = None
        for i, j in pairs:
            y = self._mm(pa[i], self._dev(pb[j]))
            out = y if out is None else out + y
        return out


@contextlib.contextmanager
def on_device(dm):
    def linear(x, w, b=None):
        shape = x.shape
        y = dm.with_weight(x.reshape(-1, shape[-1]).float(), dm.weight(w))
        y = y.reshape(*shape[:-1], -1)
        return y + b if b is not None else y

    def conv1d(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation == 1 and groups == 1
        bsz, cin, _ = x.shape
        cout, _, k = w.shape
        xp = F.pad(x, (padding, padding))
        cols = xp.unfold(-1, k, stride)  # [B, Cin, L_out, k]
        L = cols.shape[2]
        a = cols.permute(0, 2, 1, 3).reshape(bsz * L, cin * k)
        y = dm.with_weight(a.float(), dm.weight(w))
        y = y.reshape(bsz, L, cout).permute(0, 2, 1)
        return y + b[None, :, None] if b is not None else y

    F.linear, F.conv1d = linear, conv1d
    try:
        yield
    finally:
        F.linear, F.conv1d = _linear, _conv1d


class DeviceAttentionEncoder(ref.AudioTokenizerEncoder):
    dm = None

    def _attention(self, q, k, v, window, masks):
        key = (q.shape[2], window)
        if key not in masks:
            masks[key] = ref._attn_mask(q.shape[2], self.cfg.encoder_causal, window, q.dtype)
        s = self.dm.batched(q[0], k[0].transpose(-1, -2).contiguous()) * q.shape[-1] ** -0.5
        if masks[key] is not None:
            s = s + masks[key]
        p = torch.softmax(s, dim=-1)
        return self.dm.batched(p, v[0]).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--clips", default=os.environ.get("MIMO_AUDIO_CLIPS", "/data/ttuser-agentx/mm/audio/clips"))
    parser.add_argument("--clip", nargs="+", default=["speech_5s", "speech_30s", "speech_65s"])
    parser.add_argument("--schemes", nargs="+", default=["split", "bf16"], choices=["split", "bf16"])
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    weights = ref.load_audio_weights(args.model, patch=False)
    cfg = ref.MiMoAudioConfig.from_model_path(args.model)
    base = ref.AudioTokenizerEncoder(weights, cfg)
    enc = DeviceAttentionEncoder(weights, cfg)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        dms = {s: DeviceMatmul(mesh, s == "split") for s in args.schemes}
        for clip in args.clip:
            mel = load_mel(os.path.join(args.clips, f"{clip}.wav"), cfg)
            lens = torch.tensor(ref.segment_lengths(mel.shape[0], cfg.segment_size))
            with torch.inference_mode():
                h0, _ = base.features(mel, lens)
                c0 = base.quantize(h0)
                m = margins(base, h0)
                for scheme, dm in dms.items():
                    enc.dm = dm
                    t = time.time()
                    with on_device(dm):
                        h, _ = enc.features(mel, lens)
                    d = (base.quantize(h) != c0).any(0)
                    mm = m[d].sort().values
                    print(
                        f"{clip} {scheme}: {int(d.sum())}/{d.numel()} code frames changed ({100 * d.float().mean():.2f} %), "
                        f"hidden rel err {((h - h0).norm() / h0.norm()).item():.1e}; changed frames' margins "
                        f"median {mm.median().item() if len(mm) else 0:.1e}, max {mm.max().item() if len(mm) else 0:.1e} "
                        f"(all frames: median {m.median().item():.1e}); {time.time() - t:.0f} s",
                        flush=True,
                    )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
