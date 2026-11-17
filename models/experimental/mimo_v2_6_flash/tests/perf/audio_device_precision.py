# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Would a device port of the audio tokenizer encoder keep the RVQ codes? CPU emulation, no device.

The checkpoint's encoder weights are bf16, so a bf16 matmul with fp32 accumulation (HiFi4, fp32 dest) only rounds
the activation operand. Each variant rounds activations the way a device program would and counts the 25 Hz code
frames whose 20 codes differ from the fp32 reference, plus the embeddings' PCC (fp32 patch encoder on both):

  act16           every matmul / conv activation input rounded to bf16, fp32 accumulation; q, k, v rounded to
                  bf16, softmax and P @ V fp32; residual stream, LayerNorm, GELU fp32
  act16_p16       act16 with the softmax probabilities rounded to bf16 before P @ V (a bf16 SDPA)
  act16x2         activations split into two bf16 parts, hi + lo: 2 matmuls per weight matmul, and q.k / P.v as
                  hi.hi + hi.lo + lo.hi (3 matmuls); everything else fp32
  act16x2_sdpa16  act16x2 for the weight matmuls, attention as act16_p16
  bf16_all        the reference's bf16 mode: everything bf16, including the residual stream (vLLM's GPU path)

Elementwise ops (LayerNorm, GELU, softmax exp) run exactly in fp32 here; a device's approximations of them are not
modelled.

    python models/experimental/mimo_v2_6_flash/tests/perf/audio_device_precision.py --clips <dir> \\
        [--clip speech_5s speech_30s speech_65s] [--modes act16 act16x2 ...]
"""

import argparse
import contextlib
import os

import torch
import torch.nn.functional as F

from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tests.perf.audio_host_latency import load_mel

MODES = ("act16", "act16_p16", "act16x2", "act16x2_sdpa16", "bf16_all")
_linear, _conv1d = F.linear, F.conv1d


def r16(x):
    return x.to(torch.bfloat16).to(x.dtype)


def split(x):
    hi = r16(x)
    return hi, r16(x - hi)


def attention(q, k, v, mask, scale, mode):
    """[1, H, n, d] fp32 -> [1, H, n, d], with the mode's operand rounding."""
    if mode == "act16x2":
        (qh, ql), (kh, kl) = split(q), split(k)
        s = (qh @ kh.transpose(-1, -2) + qh @ kl.transpose(-1, -2) + ql @ kh.transpose(-1, -2)) * scale
    else:
        s = (r16(q) @ r16(k).transpose(-1, -2)) * scale
    if mask is not None:
        s = s + mask
    p = torch.softmax(s, dim=-1)
    if mode == "act16x2":
        (ph, pl), (vh, vl) = split(p), split(v)
        return ph @ vh + ph @ vl + pl @ vh
    if mode in ("act16_p16", "act16x2_sdpa16"):
        return r16(p) @ r16(v)
    return p @ r16(v)


class EmulatedEncoder(ref.AudioTokenizerEncoder):
    mode = "act16"

    def _attention(self, q, k, v, window, masks):
        key = (q.shape[2], window)
        if key not in masks:
            masks[key] = ref._attn_mask(q.shape[2], self.cfg.encoder_causal, window, q.dtype)
        heads = [
            attention(q[:, h : h + 4], k[:, h : h + 4], v[:, h : h + 4], masks[key], q.shape[-1] ** -0.5, self.mode)
            for h in range(0, q.shape[1], 4)  # 4 heads at a time bounds the [n, n] scores
        ]
        return torch.cat(heads, dim=1)


@contextlib.contextmanager
def rounded_matmuls(mode):
    """Patch F.linear / F.conv1d (the reference's weight matmuls) to round or split their activation input."""
    two = mode.startswith("act16x2")

    def linear(x, w, b=None):
        if two:
            hi, lo = split(x)
            y = _linear(hi, w) + _linear(lo, w)
        else:
            y = _linear(r16(x), w)
        return y + b if b is not None else y

    def conv1d(x, w, b=None, *args, **kwargs):
        if two:
            hi, lo = split(x)
            y = _conv1d(hi, w, None, *args, **kwargs) + _conv1d(lo, w, None, *args, **kwargs)
        else:
            y = _conv1d(r16(x), w, None, *args, **kwargs)
        return y + b[None, :, None] if b is not None else y

    F.linear, F.conv1d = linear, conv1d
    try:
        yield
    finally:
        F.linear, F.conv1d = _linear, _conv1d


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--clips", default=os.environ.get("MIMO_AUDIO_CLIPS", "/data/ttuser-agentx/mm/audio/clips"))
    parser.add_argument("--clip", nargs="+", default=["speech_5s", "speech_30s", "speech_65s"])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    weights = ref.load_audio_weights(args.model)
    cfg = ref.MiMoAudioConfig.from_model_path(args.model)
    base = ref.AudioTokenizerEncoder(weights, cfg)
    emulated = EmulatedEncoder(weights, cfg)
    bf16 = ref.AudioTokenizerEncoder(weights, cfg, torch.bfloat16)
    patch = ref.AudioPatchEncoder(weights, cfg)
    for clip in args.clip:
        mel = load_mel(os.path.join(args.clips, f"{clip}.wav"), cfg)
        lens = torch.tensor(ref.segment_lengths(mel.shape[0], cfg.segment_size))
        with torch.inference_mode():
            h0, _ = base.features(mel, lens)
            c0 = base.quantize(h0)
            e0 = patch(ref.pad_and_group_codes(c0.t(), cfg))
        print(f"== {clip}: {c0.shape[1]} code frames", flush=True)
        for mode in args.modes:
            with torch.inference_mode():
                if mode == "bf16_all":
                    h, _ = bf16.features(mel, lens)
                else:
                    emulated.mode = mode
                    with rounded_matmuls(mode):
                        h, _ = emulated.features(mel, lens)
                c = base.quantize(h)
                e = patch(ref.pad_and_group_codes(c.t(), cfg))
            diff = c != c0
            print(
                f"  {mode:>15}: hidden rel err {((h.float() - h0).norm() / h0.norm()).item():.2e}; frames with a "
                f"changed code {diff.any(0).float().mean():.4f} (codebook 0 {diff[0].float().mean():.4f}, "
                f"19 {diff[-1].float().mean():.4f}); embeddings PCC {pcc(e, e0):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
