# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How close to a tie are the fp32 reference's RVQ codes? CPU only, no device.

For each clip: the RVQ decision margin of every 25 Hz frame (over the 20 stages, the smallest gap between the best
and the second-best codeword distance, relative to the best distance; fp64 RVQ on the fp32 encoder's hidden
states), and the code frames that change against the fp32 reference when the tokenizer encoder runs

  fp64            in fp64 throughout (more exact than the reference)
  fp32 reordered  in fp32, with the windowed layers' softmax and P @ V summed in another order (each 256-query block
                  attends to keys from its window's start only)
  act16x2         the split-precision device scheme of audio_device_precision.py (activations as two bf16 parts,
                  bf16 weights, fp32 everything else; elementwise ops exact)

Measured on bhg2 (2026-11-07): the smallest margins are 7e-8 to 4e-7, at fp32 rounding level, and both fp64 and
act16x2 change frames whose margin is below 1e-6 (5 / 30 / 65 / 120 / 300 s: fp64 0 / 1 / 1 / 1 / 1 frames,
act16x2 0 / 0 / 0 / 1 / 2). So no implementation that is not bit-identical to the fp32 CPU reference, a device
port of the tokenizer encoder included, can promise identical codes on every clip.

    python models/experimental/mimo_v2_6_flash/tests/perf/audio_codes_margin.py --clips <dir> \\
        [--clip speech_5s speech_30s ...] [--threads 16]
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tests.perf import audio_device_precision as emu
from models.experimental.mimo_v2_6_flash.tests.perf.audio_host_latency import load_mel


class Reordered(ref.AudioTokenizerEncoder):
    """fp32; windowed layers attend per 256-query block to keys from the window's start: another summation order."""

    def _attention(self, q, k, v, window, masks):
        n = q.shape[2]
        if window <= 0 or n <= 256:
            return super()._attention(q, k, v, window, masks)
        key = (n, window)
        if key not in masks:
            masks[key] = ref._attn_mask(n, self.cfg.encoder_causal, window, q.dtype)
        out = torch.empty_like(q)
        for q0 in range(0, n, 256):
            q1, k0 = min(n, q0 + 256), max(0, q0 - window)
            out[:, :, q0:q1] = F.scaled_dot_product_attention(
                q[:, :, q0:q1],
                k[:, :, k0:q1],
                v[:, :, k0:q1],
                attn_mask=masks[key][q0:q1, k0:q1],
                scale=q.shape[-1] ** -0.5,
            )
        return out


def margins(enc, h):
    """Per frame: min over the RVQ stages of (second-best - best distance) / best distance (fp64)."""
    residual = h.double()
    out = None
    for embed in enc.codebooks:
        e = embed.double().t()
        dist = residual.pow(2).sum(1, keepdim=True) - 2 * residual @ e + e.pow(2).sum(0, keepdim=True)
        top = dist.topk(2, dim=-1, largest=False).values
        m = (top[:, 1] - top[:, 0]) / top[:, 0].abs().clamp_min(1e-30)
        out = m if out is None else torch.minimum(out, m)
        residual = residual - F.embedding(dist.argmin(-1), embed.double())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--clips", default=os.environ.get("MIMO_AUDIO_CLIPS", "/data/ttuser-agentx/mm/audio/clips"))
    parser.add_argument(
        "--clip", nargs="+", default=["speech_5s", "speech_30s", "speech_65s", "speech_120s", "speech_300s"]
    )
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    weights = ref.load_audio_weights(args.model, patch=False)
    cfg = ref.MiMoAudioConfig.from_model_path(args.model)
    base = ref.AudioTokenizerEncoder(weights, cfg)
    f64 = ref.AudioTokenizerEncoder(weights, cfg, torch.float64)
    reordered = Reordered(weights, cfg)
    emulated = emu.EmulatedEncoder(weights, cfg)
    emulated.mode = "act16x2"

    def act16x2(mel, lens):
        with emu.rounded_matmuls("act16x2"):
            return emulated.features(mel, lens)[0]

    for clip in args.clip:
        mel = load_mel(os.path.join(args.clips, f"{clip}.wav"), cfg)
        lens = torch.tensor(ref.segment_lengths(mel.shape[0], cfg.segment_size))
        with torch.inference_mode():
            t = time.time()
            h0, _ = base.features(mel, lens)
            c0 = base.quantize(h0)
            m = margins(base, h0)
            res = []
            variants = (
                ("fp64", lambda: f64.quantize(f64.features(mel.double(), lens)[0])),
                ("fp32 reordered", lambda: base.quantize(reordered.features(mel, lens)[0])),
                ("act16x2", lambda: base.quantize(act16x2(mel, lens))),
            )
            for name, codes in variants:
                d = (codes() != c0).any(0)
                res.append(
                    f"{name}: {int(d.sum())} frames (margins {', '.join(f'{x:.1e}' for x in m[d][:4].tolist())})"
                )
        print(
            f"{clip}: {c0.shape[1]} frames, {time.time() - t:.0f} s; smallest margins "
            f"{', '.join(f'{x:.1e}' for x in m.sort().values[:5].tolist())}; changed vs fp32: " + " | ".join(res),
            flush=True,
        )


if __name__ == "__main__":
    main()
