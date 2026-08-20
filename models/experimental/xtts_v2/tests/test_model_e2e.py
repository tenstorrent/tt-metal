# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end determinism check for the XttsV2 request path (tt/ttnn_xtts_model.py).

Both the GPT decode step and the vocoder (once per VOC_BUCKETS shape) are replayed from Metal
Traces, and a trace writes to buffer addresses baked in at capture. Anything allocated later
that lands on those addresses is corrupted on the next replay, silently — so this drives the
real serving pattern and requires bit-identical repeats: every bucket through `generate`, and a
`compute_voice` for a NEW reference-clip length (which compiles and allocates fresh conv shapes
after the captures) in between.

Reproducibility alone is not enough — consistent corruption is still reproducible, so it also
finishes against the CPU reference waveform once the decode trace has replayed. `generate` samples
and so has no fixed waveform to compare against; the check instead runs reference GPT latents
through everything it does after decode — both host interpolates and the traced vocoder.

Opens its own device, like `demo/`, so the model's real trace-region and L1_SMALL sizes are the
ones under test.
"""
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.reference.xtts_hifigan_ref import HifiganReference
from models.experimental.xtts_v2.tests.reference_helpers import gpt_reference
from models.experimental.xtts_v2.tt.ttnn_xtts_model import (
    AR_COMP,
    HOP,
    ISR,
    OSR,
    VOC_BUCKETS,
    XttsV2,
    _voc_bucket,
    _voc_input,
    _voc_pad,
)

TARGET_PCC_WAV = 0.99  # waveform vs the CPU reference, same gate as test_hifigan_pcc

# max_new_tokens picks the bucket: a code is 1024 samples @22.05kHz resampled to 24 kHz, so
# frames = codes * (1024/256) * (24000/22050) ~= codes * 4.354.
PROMPTS = {
    "one_line": "Hello from Tenstorrent.",  # 47 tokens, and it stops on its own after ~50 codes
    "paragraph": "Text to speech synthesis on dedicated accelerator hardware requires careful "
    "attention to both numerical precision and memory layout. " * 4,  # 304 tokens, under the 402 cap
}
# (prompt, max_new_tokens). The prompt picks the PREFILL shape, and a new one compiles programs
# after the capture; max_new_tokens caps the AUDIO and so picks the vocoder bucket. Nearly every
# row is the paragraph because the cap only bites while the model is still generating — one_line
# ends by itself, so it cannot be stretched to reach the upper buckets. Both matter here: two
# prompt lengths, and every bucket reused after other buckets have replayed.
ORDER = (
    ("paragraph", 550),
    ("one_line", 100),
    ("paragraph", 90),
    ("paragraph", 120),
    ("paragraph", 180),
    ("paragraph", 250),
    ("paragraph", 400),
    ("paragraph", 550),
    ("one_line", 100),
    ("paragraph", 90),
    ("paragraph", 120),
    ("paragraph", 180),
    ("paragraph", 250),
    ("paragraph", 400),
)


def _latents_for(Lb):
    """Reference GPT latents tiled to land inside bucket Lb, then the two interpolates `generate`
    runs — only the length varies here."""
    lat = gpt_reference()["latents"]  # [1,T,1024]
    n = int((Lb - 7) / ((AR_COMP / HOP) * (OSR / ISR)))  # inside the bucket, so the replay pads
    lat = lat.repeat(1, n // lat.shape[1] + 1, 1)[:, :n]
    return _voc_input(lat)


def _clip(seconds, seed):  # a reference clip's LENGTH is what picks the conv shapes, not its content
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * 22050), generator=g) * 0.1


def run_model_e2e(verbose=True):
    tts = XttsV2()
    try:
        tts.warmup()
        voice = tts.compute_voice(_clip(6.0, 0), 22050)  # 6 s: the shape warmup compiled
        # A new reference-clip length compiles + allocates conv shapes AFTER every capture.
        other = tts.compute_voice(_clip(4.0, 1), 22050)

        first, buckets, worst = {}, set(), 0.0
        for i, key in enumerate(ORDER):
            prompt, n = key
            wav = tts.generate(PROMPTS[prompt], voice, seed=0, max_new_tokens=n)
            Lb = _voc_bucket(wav.shape[-1] // 256)
            buckets.add(Lb)
            assert torch.isfinite(wav).all(), f"request {i} ({prompt}, {n}) produced non-finite samples"
            if key not in first:
                first[key] = wav
                tts.generate(PROMPTS["one_line"], other, seed=0, max_new_tokens=60)  # interpose the other voice
            else:
                d = (first[key] - wav).abs().max().item() if first[key].shape == wav.shape else float("inf")
                worst = max(worst, d)
                if verbose:
                    print(f"  {i}: {prompt:9s} max_new={n:4d} bucket {Lb:5d} repeat maxabs {d:.3e}")

        # Every bucket against CPU: each has its own trace and slots, and the repeat check above
        # reports maxabs 0 for a bucket that is consistently wrong just as it does for a clean one.
        # The reference pads as _vocode does: conv_pre has a bias, so zero frames past L are not
        # silence and they perturb the trailing waveform.
        g, reference, scored = voice.speaker_embedding, HifiganReference(), []
        for Lb in VOC_BUCKETS:
            z = _latents_for(Lb)
            L = z.shape[-1]
            assert _voc_bucket(L) == Lb, f"z of {L} frames does not land in bucket {Lb}"
            gold = reference(_voc_pad(z, Lb), g)[:, :, : L * HOP]
            ok, pcc = comp_pcc(gold, tts._vocode(z, g), pcc=TARGET_PCC_WAV)
            if verbose:
                print(f"  bucket {Lb:5d} (L={L}) latents -> waveform vs CPU reference  pcc: {pcc}")
            scored.append((Lb, ok, pcc))
        worst_Lb, _, worst_pcc = min(scored, key=lambda r: r[2])
        ref_ok = all(r[1] for r in scored)
    finally:
        tts.close()

    msg = f"buckets {sorted(buckets)}, worst repeat maxabs {worst:.3e}, worst bucket {worst_Lb} pcc {worst_pcc}"
    return worst == 0.0 and buckets == set(VOC_BUCKETS) and ref_ok, msg


def test_model_e2e():
    passed, msg = run_model_e2e()
    assert passed, f"request path is not reproducible, or drifted from the reference: {msg}"


if __name__ == "__main__":
    import sys

    ok, msg = run_model_e2e()
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
