# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Repeatability and trace-safety check for the XttsV2 request path (tt/ttnn_xtts_model.py).

Both the GPT decode step and the vocoder (once per VOC_BUCKETS shape) are replayed from Metal
Traces, and a trace writes to buffer addresses baked in at capture. Anything allocated later that
lands on those addresses is corrupted on the next replay, silently — so this drives the real
serving pattern and requires bit-identical repeats, with a `compute_voice` for a NEW
reference-clip length (which compiles and allocates fresh conv shapes after the captures) in
between. Every vocoder trace is then replayed once the requests have left the device in its
served state; driving those from chosen latent lengths keeps coverage independent of what the
model happens to sample.

This compares the path to ITSELF, which a consistently wrong result passes. Correctness against an
independent implementation lives elsewhere on purpose: test_model_teacher_forced_pcc drives real
text and compares GPT latents and waveforms to the CPU, and test_vocoder_request_path checks every
bucket's trace against the CPU vocoder.

Opens its own device, like demo/, so the model's real trace-region and L1_SMALL sizes are the ones
under test.

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_request_path_repeatability.py
"""
import torch

from models.experimental.xtts_v2.tt.ttnn_xtts_model import VOC_BUCKETS, XttsV2

# max_new_tokens picks the bucket: a code is 1024 samples @22.05kHz resampled to 24 kHz, so
# frames = codes * (1024/256) * (24000/22050) ~= codes * 4.354.
PROMPTS = {
    "one_line": "Hello from Tenstorrent.",  # 47 tokens, and it stops on its own after ~50 codes
    "paragraph": "Text to speech synthesis on dedicated accelerator hardware requires careful "
    "attention to both numerical precision and memory layout. " * 4,  # 304 tokens, under the 402 cap
}
# (prompt, max_new_tokens). The prompt picks the PREFILL shape, and a new one compiles programs
# after the capture; max_new_tokens caps the audio and so varies the vocoder shape a request
# reaches. Each row runs twice: the first records, the second must reproduce it.
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


def _clip(seconds, seed):  # a reference clip's LENGTH is what picks the conv shapes, not its content
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * 22050), generator=g) * 0.1


def _replay_every_bucket(tts, speaker_embedding, verbose):
    """Replay each vocoder trace from latents whose length selects its bucket -> (buckets, worst)."""
    covered, worst = set(), 0.0
    for b in VOC_BUCKETS:
        z = torch.randn(1, 1024, b, generator=torch.Generator().manual_seed(b))
        first = tts._vocode(z, speaker_embedding)
        again = tts._vocode(z, speaker_embedding)
        covered.add(tts.last_timings["voc_bucket"])
        d = (first - again).abs().max().item()
        worst = max(worst, d)
        if verbose:
            print(f"  bucket {b:5d} replay maxabs {d:.3e}")
    return covered, worst


def run_request_path_repeatability(verbose=True):
    tts = XttsV2()
    try:
        tts.warmup()
        voice = tts.compute_voice(_clip(6.0, 0), 22050)  # 6 s: the shape warmup compiled
        # A new reference-clip length compiles + allocates conv shapes AFTER every capture.
        other = tts.compute_voice(_clip(4.0, 1), 22050)

        first, reached, worst = {}, set(), 0.0
        for i, key in enumerate(ORDER):
            prompt, n = key
            wav = tts.generate(PROMPTS[prompt], voice, seed=0, max_new_tokens=n)
            # The vocoder picks its bucket from the latents, before trimming shortens the wav,
            # so re-deriving it from the returned wav can name a bucket that never ran.
            Lb = tts.last_timings["voc_bucket"]
            reached.add(Lb)
            assert torch.isfinite(wav).all(), f"request {i} ({prompt}, {n}) produced non-finite samples"
            if key not in first:
                first[key] = wav
                tts.generate(PROMPTS["one_line"], other, seed=0, max_new_tokens=60)  # interpose the other voice
            else:
                d = (first[key] - wav).abs().max().item() if first[key].shape == wav.shape else float("inf")
                worst = max(worst, d)
                if verbose:
                    print(f"  {i}: {prompt:9s} max_new={n:4d} bucket {Lb:5d} repeat maxabs {d:.3e}")
        covered, voc_worst = _replay_every_bucket(tts, voice.speaker_embedding, verbose)
    finally:
        tts.close()

    worst = max(worst, voc_worst)
    msg = f"requests reached {sorted(reached)}, replayed {sorted(covered)}, worst repeat maxabs {worst:.3e}"
    return worst == 0.0 and covered == set(VOC_BUCKETS), msg


def test_request_path_repeatability():
    passed, msg = run_request_path_repeatability()
    assert passed, f"the request path is not bit-reproducible, or missed a bucket: {msg}"


if __name__ == "__main__":
    import sys

    ok, msg = run_request_path_repeatability()
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
