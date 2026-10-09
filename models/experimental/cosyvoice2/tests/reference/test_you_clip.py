# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The "you" clip's regression test, the reference-venv half (notes: B28). Host only, reference venv only.

260-123440-0010 streamed ended in "... gently smiling jaws you" in 5 of 11 noise draws while the final HiFT call was
padded with silence: the last ~25 ms went silent and Whisper decoded a stray "you" from them. The device half
(tests/e2e/test_streaming.py::test_device_you_clip_noise_draws) renders the 11 draws into COSYVOICE2_YOU_OUT; this
transcribes them with the corpus scorer (scripts/eval_wer_sim.py, unchanged) and fails if any ends in a "you" the
text doesn't have.

The reference venv has no pytest, so run this file there as a plain script (it exits non-zero on failure):

    COSYVOICE2_YOU_OUT=<dir> $COSYVOICE2_REF_ENV/bin/python tests/reference/test_you_clip.py

Under python_env's pytest it skips: Whisper isn't installed there.
"""
from __future__ import annotations

import json
import os
import sys

try:
    import pytest
except ImportError:  # the reference venv
    pytest = None

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
)

YOU_OUT = os.environ.get("COSYVOICE2_YOU_OUT", "")


def _skip(why: str):
    if pytest is not None:
        pytest.skip(why)
    raise SystemExit(why)


def test_no_draw_of_the_you_clip_ends_in_you():
    if not YOU_OUT:
        _skip("set COSYVOICE2_YOU_OUT (the device half's output)")
    try:
        import eval_wer_sim as ews
        import whisper  # noqa: F401
    except ImportError as e:
        _skip(f"reference venv only ({e})")
    with open(os.path.join(YOU_OUT, "results.json")) as fh:
        results = json.load(fh)["results"]
    assert len(results) == 11, len(results)
    asr = ews.ASR()
    stray = []
    for r in results:
        hyp = asr.transcribe(os.path.join(YOU_OUT, r["wav"]), r["lang"])
        ref_words, hyp_words = ews.normalize(r["text"], r["lang"]), ews.normalize(hyp, r["lang"])
        errors = ews.edit_distance(ref_words, hyp_words)[0]
        print(f"  {r['case_id']:<44} errors {errors}  ...{hyp[-40:]!r}", flush=True)
        if hyp_words[-1:] == ["you"] and ref_words[-1:] != ["you"]:
            stray.append(r["case_id"])
    assert not stray, f"a stray trailing 'you' in {len(stray)} of {len(results)} draws: {stray}"


if __name__ == "__main__":
    test_no_draw_of_the_you_clip_ends_in_you()
    print("PASSED test_no_draw_of_the_you_clip_ends_in_you")
