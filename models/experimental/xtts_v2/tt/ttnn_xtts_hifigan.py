# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the XTTS-v2 HiFi-GAN vocoder generator (Block 4).

Mirrors reference/xtts_hifigan_ref.py op-for-op, running ENTIRELY on device: the signal stays a
[1,1,L,C] channels-last device tensor through the whole generator (~34 convs + leaky_relu / add /
tanh), so there is no host round-trip between ops — only z in and the waveform out. conv1d runs on
device; the (missing) conv_transpose1d is done via ttnn.conv_transpose2d with a singleton width dim.
Weights are uploaded once in __init__. Large convs slice along length in DRAM (Conv2dSliceConfig /
Op2DSliceConfig) so each slice fits L1. fp32 + fp32 accumulation to hold accuracy through the deep
stack (bf16 compounds to ~0.91 PCC; fp32 gives ~0.999).

Drop-in for coqui's HifiganGenerator (the HifiDecoder.waveform_decoder): __call__(z, g) -> waveform.

Validate + time vs the reference:
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/xtts_v2/tt/ttnn_xtts_hifigan.py
"""

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_hifigan_ref import (
    DEFAULT_CKPT,
    LRELU,
    RES_D,
    RES_K,
    UPS,
    _pad,
    load_hifigan_state,
)

COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
DTYPE = ttnn.float32
# conv1d auto-slices (num_slices=0 -> ttnn picks the minimal count that fits L1). The transpose can't:
# with too few slices its circular buffers clash with L1, so it keeps a small budget -> more slices.
_TR_BUDGET = 200_000  # per-slice output-footprint budget (bytes) for conv_transpose only


def _num_slices(L_out, C, budget):
    return int(max(1, -(-L_out * C * 4 // budget)))  # ceil(footprint_bytes / budget)


class TtHifiganGenerator(torch.nn.Module):
    """On-device TTNN HiFi-GAN generator. Drop-in for HifiDecoder.waveform_decoder: forward(z, g) -> wav.
    Subclasses nn.Module only so it can be assigned as a child of coqui's HifiDecoder (no torch params)."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT):
        super().__init__()
        self.device = device
        w = load_hifigan_state(ckpt_path)  # weight-norm folded torch weights
        # Upload all weights to device ONCE. Transpose weights (ups.*) are [Cin,Cout,k] -> [Cin,Cout,k,1];
        # conv weights [Cout,Cin,k] stay as-is; biases -> [1,1,1,Cout].
        self.wd = {}
        for key, v in w.items():
            if key.endswith(".weight"):
                if key.startswith("ups."):
                    Cin, Cout, k = v.shape
                    self.wd[key] = ttnn.from_torch(v.reshape(Cin, Cout, k, 1).contiguous(), dtype=DTYPE)
                else:
                    self.wd[key] = ttnn.from_torch(v, dtype=DTYPE)
            elif key.endswith(".bias"):
                self.wd[key] = ttnn.from_torch(v.reshape(1, 1, 1, v.shape[0]), dtype=DTYPE)

    def remove_weight_norm(self):  # no-op: weights are pre-folded at load
        pass

    def _to_dev(self, x):  # torch [1,C,L] -> device [1,1,L,C] TILE channels-last
        return ttnn.from_torch(
            x.permute(0, 2, 1).reshape(1, 1, x.shape[2], x.shape[1]).contiguous(),
            dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device,
        )

    def _conv(self, o, wkey, stride=1, padding=0, dilation=1, transpose=False):
        """o: device [1,1,L,Cin] TILE -> device [1,1,Lout,Cout] TILE. Runs on device; no host."""
        _, _, L, Cin = o.shape
        wt = self.wd[wkey + ".weight"]
        bt = self.wd.get(wkey + ".bias")
        Cout = wt.shape[1] if transpose else wt.shape[0]
        k = wt.shape[2]
        o_in = ttnn.reshape(ttnn.to_layout(o, ttnn.ROW_MAJOR_LAYOUT), [1, L, 1, Cin])
        if transpose:
            Lout = (L - 1) * stride - 2 * padding + k
            kw = {}
            n = _num_slices(Lout, max(Cin, Cout), _TR_BUDGET)
            if n > 1:
                kw["dram_slice_config"] = ttnn.Op2DSliceConfig(slice_type=ttnn.Op2DDRAMSliceHeight, num_slices=n)
            out, (Lo, _) = ttnn.conv_transpose2d(
                input_tensor=o_in, weight_tensor=wt, bias_tensor=bt, in_channels=Cin, out_channels=Cout,
                device=self.device, kernel_size=(k, 1), stride=(stride, 1), padding=(padding, 0), output_padding=(0, 0),
                batch_size=1, input_height=L, input_width=1, groups=1, compute_config=COMPUTE_CONFIG,
                return_output_dim=True, return_weights_and_bias=False, **kw,
            )
        else:
            # num_slices=0 -> ttnn auto-picks the minimal DRAM slice count that fits L1 (faster than a
            # fixed budget, which over-slices). Length is the WIDTH axis for conv1d.
            kw = {"slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0)}
            out, Lo = ttnn.conv1d(
                input_tensor=o_in, weight_tensor=wt, bias_tensor=bt, in_channels=Cin, out_channels=Cout,
                device=self.device, kernel_size=k, stride=stride, padding=padding, dilation=dilation,
                batch_size=1, input_length=L, groups=1, compute_config=COMPUTE_CONFIG,
                return_output_dim=True, return_weights_and_bias=False, **kw,
            )
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out)
        return ttnn.reshape(out, [1, 1, Lo, Cout])  # [1,1,Lout,Cout] TILE, channels-last

    def _resblock(self, x, ridx, k, dils):  # ResBlock1, all on device
        for j in range(3):
            xt = ttnn.leaky_relu(x, negative_slope=LRELU)
            xt = self._conv(xt, f"resblocks.{ridx}.convs1.{j}", dilation=dils[j], padding=_pad(k, dils[j]))
            xt = ttnn.leaky_relu(xt, negative_slope=LRELU)
            xt = self._conv(xt, f"resblocks.{ridx}.convs2.{j}", dilation=1, padding=_pad(k, 1))
            x = ttnn.add(xt, x)
        return x

    @torch.no_grad()
    def forward(self, z, g=None):  # z torch [1,1024,L] (or [1024,L]), g torch [1,512,1] -> torch [1,1,L*256]
        if z.dim() == 2:  # HifiDecoder.forward squeezes the batch dim before waveform_decoder
            z = z.unsqueeze(0)
        if g.dim() == 2:
            g = g.unsqueeze(0)
        o = self._to_dev(z)
        gd = self._to_dev(g)
        o = self._conv(o, "conv_pre", padding=3)
        o = ttnn.add(o, self._conv(gd, "cond_layer"))  # k1, broadcasts over time
        for i in range(len(UPS)):
            k, s, p = UPS[i]
            o = ttnn.leaky_relu(o, negative_slope=LRELU)
            o = self._conv(o, f"ups.{i}", stride=s, padding=p, transpose=True)
            o = ttnn.add(o, self._conv(gd, f"conds.{i}"))  # per-layer d-vector conditioning
            z_sum = None
            for j in range(len(RES_K)):
                r = self._resblock(o, i * len(RES_K) + j, RES_K[j], RES_D[j])
                z_sum = r if z_sum is None else ttnn.add(z_sum, r)
            o = ttnn.multiply(z_sum, 1.0 / len(RES_K))  # MRF average
        o = ttnn.leaky_relu(o, negative_slope=0.01)  # DEFAULT slope here (coqui quirk), not LRELU
        o = self._conv(o, "conv_post", padding=3)  # conv_post has no bias key -> bias None
        o = ttnn.tanh(o)
        wav = ttnn.to_torch(o).reshape(1, o.shape[2], o.shape[3]).permute(0, 2, 1).float()  # [1,1,L*256]
        return wav


def main():
    import time

    from models.experimental.xtts_v2.reference import xtts_hifigan_ref as ref
    from models.experimental.xtts_v2.reference.xtts_gpt_ref import pcc

    device = ttnn.open_device(device_id=0, l1_small_size=131072)
    try:
        gen = TtHifiganGenerator(device)
        z, g = ref.make_synthetic_inputs(32)
        ref_wav = ref.generator(z, ref.load_hifigan_state(), g)
        got = gen(z, g)
        print(f"[hifigan] PCC vs reference: {pcc(got, ref_wav):.5f}  (wav {tuple(got.shape)})")
        for L in (435, 2633):  # typical ~5s utterance and max ~28s utterance
            zt = torch.randn(1, 1024, L)
            gen(zt, g)  # warm
            t0 = time.perf_counter()
            wav = gen(zt, g)
            dt = time.perf_counter() - t0
            print(f"[hifigan] z len {L:5d} -> wav {wav.shape[-1]:7d} ({wav.shape[-1]/24000:.1f}s audio): {dt:.2f}s")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
