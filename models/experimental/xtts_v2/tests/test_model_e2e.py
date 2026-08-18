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

Opens its own device, like `demo/`, so the model's real trace-region and L1_SMALL sizes are the
ones under test.
"""
import torch

from models.experimental.xtts_v2.tt.ttnn_xtts_model import VOC_BUCKETS, XttsV2, _voc_bucket

# max_new_tokens picks the bucket: a code is 1024 samples @22.05kHz resampled to 24 kHz, so
# frames = codes * (1024/256) * (24000/22050) ~= codes * 4.354.
TEXTS = {
    "short": "Hello from Tenstorrent.",
    "long": "Text to speech synthesis on dedicated accelerator hardware requires careful attention "
    "to both numerical precision and memory layout. " * 4,  # 304 tokens, under the 402 cap
}
# Two prompt lengths (a NEW one compiles prefill programs post-capture) and every bucket reused
# after other buckets have replayed — both are what past corruption needed to show itself.
ORDER = (
    ("long", 550),
    ("short", 100),
    ("long", 250),
    ("long", 400),
    ("long", 550),
    ("short", 100),
    ("long", 250),
    ("long", 400),
)


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
            text, n = key
            wav = tts.generate(TEXTS[text], voice, seed=0, max_new_tokens=n)
            Lb = _voc_bucket(wav.shape[-1] // 256)
            buckets.add(Lb)
            assert torch.isfinite(wav).all(), f"request {i} ({text}, {n}) produced non-finite samples"
            if key not in first:
                first[key] = wav
                tts.generate(TEXTS["short"], other, seed=0, max_new_tokens=60)  # interpose the other voice
            else:
                d = (first[key] - wav).abs().max().item() if first[key].shape == wav.shape else float("inf")
                worst = max(worst, d)
                if verbose:
                    print(f"  {i}: {text:5s} max_new={n:4d} bucket {Lb:5d} repeat maxabs {d:.3e}")
    finally:
        tts.close()

    msg = f"buckets {sorted(buckets)}, worst repeat maxabs {worst:.3e}"
    return worst == 0.0 and buckets == set(VOC_BUCKETS), msg


def test_model_e2e():
    passed, msg = run_model_e2e()
    assert passed, f"request path is not reproducible across buckets and voices: {msg}"


if __name__ == "__main__":
    import sys

    ok, msg = run_model_e2e()
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
