# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blocks 1+2 end to end: do device and reference emit the same integer codes?

Teacher-forced -- both loops are fed the REFERENCE's codes each step, so every frame is an
independent measurement and a single divergence cannot compound. Feeding each loop its own codes
would compare two diverging sequences instead.

WHY THIS RUNS 64 FRAMES ON EVERY PROMPT AND NOT 8 ON THREE. The 8-frame version asserted that the
semantic code never differs and that acoustic codes are never more than one FSQ level apart. Both
are false, and the horizon is why: semantic flips occur on ~2% of frames, so 24 frames expects 0.4
of them and observing none proved nothing. Over 960 frames there are 16, and acoustic deltas reach
19 of 21 levels. The absolutes are replaced by measured RATES.

What the longer horizon also established, and it is the reassuring half: nothing accumulates. Over a
full utterance the figures are as good as or better than over 64 frames -- case 2 reads 5.12% across
447 frames against 6.47% across its first 64 -- so Block 1's error does not compound through Block 2
across a real request.

A "delta" is how many of the 21 FSQ levels separate a device acoustic code from the reference's.
Delta 1 is the smallest possible disagreement and is what boundary rounding looks like; a delta of 7
is a third of the range and means the two computed different values. The rate of frames carrying any
delta above 1 is therefore gated separately from the overall mismatch percentage.

SEMANTIC AND ACOUSTIC DIVERGE INDEPENDENTLY. It is tempting to assume a big acoustic delta is the
downstream effect of a flipped semantic code, since both come from one frame. Measured, it is not:
there are frames whose semantic code matches while 25 of 36 acoustic codes differ by up to 7, and
frames whose semantic code flips while every acoustic code is exact. Do not fold these two gates
into one.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_model_teacher_forced_pcc.py
    pytest -svv ... -k "not full_utterance"      # the 64-frame breadth alone, ~18 min
"""

from collections import Counter

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import END_AUDIO_ID  # noqa: E402
from models.experimental.voxtral_tts.tests.gates import compare_codes_frame  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    case_ids,
    fixture_embeds,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (  # noqa: E402
    CFG_ALPHA,
    TtVoxtralPipeline,
    open_device,
)

N_FRAMES = 64          # reaches frames 40 and 55, the two the decode work found hardest
LONG_CASES = (2, 3)    # the two prompts whose natural utterance is ~450 frames
LONG_CAP = 480

# MEASURED over 960 frames (15 prompts x 64), then 3x with a floor, the same rule the WER ceilings
# use. Rates, not absolutes: see the module docstring for why the absolutes were wrong.
MAX_SEMANTIC_FLIP_PCT = 5.0   # measured 1.67% (16 of 960 frames)
MAX_BIG_DELTA_FRAME_PCT = 7.5   # measured 2.50% (24 of 960 frames)
# Unchanged from the 8-frame version and still generous: measured 6.43% over 64 frames and 5.12% /
# 5.29% over the two full utterances.
MAX_ACOUSTIC_MISMATCH_PCT = 15.0


@pytest.fixture(scope="module")
def pipe():
    d = open_device()
    p = TtVoxtralPipeline(d)
    yield p
    ttnn.close_device(d)


def _chain(pipe, embeds, n_frames, cfg_alpha=CFG_ALPHA, stop_on_end=False):
    """Teacher-forced chain -> dict of counts. Shared by the breadth and full-utterance tests.

    `stop_on_end` matters for the long case: the reference GENERATES the codes here rather than
    replaying a capture, so running past its [END_AUDIO] would measure the model beyond the end of
    its own utterance, which is off-distribution.
    """
    wf = fref.load_flow_state()
    ref_dec = bref.IncrementalBackbone(pipe.wb)
    h_ref = ref_dec.prefill(embeds)
    pipe.backbone.reset()
    h_dev = pipe.backbone.prefill_last(embeds)

    r = {"frames": 0, "sem_bad": 0, "ac_bad": 0, "big_delta_frames": 0, "deltas": Counter(),
         "worst_frame": (0, -1)}
    for i in range(n_frames):
        torch.manual_seed(1000 + i)  # same noise draw both sides, so only the model differs
        c_ref = fref.reference_frame(h_ref[:, 0], wf, cfg_alpha=cfg_alpha)
        if stop_on_end and int(c_ref[0, 0]) == END_AUDIO_ID:
            break
        torch.manual_seed(1000 + i)
        c_dev = pipe.flow(h_dev[:, 0], cfg_alpha=cfg_alpha)
        m = compare_codes_frame(c_ref, c_dev)
        r["frames"] += 1
        r["sem_bad"] += 0 if m["sem_ok"] else 1
        r["ac_bad"] += m["n_diff"]
        if m["deltas"] and max(m["deltas"]) > 1:
            r["big_delta_frames"] += 1
        for v in m["deltas"]:
            r["deltas"][v] += 1
        if m["n_diff"] > r["worst_frame"][1]:
            r["worst_frame"] = (i, m["n_diff"])
        # teacher forcing: BOTH advance on the REFERENCE's codes
        emb = bref.embed_frame(pipe.wb, c_ref[0])
        h_ref = ref_dec.step(emb)
        h_dev = pipe.backbone.step(emb).reshape(1, 1, -1)
    return r


def _assert_rates(label, tot):
    """The three measured rates. Max delta is REPORTED, never asserted: we have observed 19 of 21
    levels, and there is no defensible bound on the magnitude -- only on how often it happens."""
    n_frames, n_ac = tot["frames"], tot["frames"] * 36
    sem_pct = tot["sem_bad"] / max(n_frames, 1) * 100
    ac_pct = tot["ac_bad"] / max(n_ac, 1) * 100
    big_pct = tot["big_delta_frames"] / max(n_frames, 1) * 100
    print(f"\n  {label}: {n_frames} frames | semantic {tot['sem_bad']} ({sem_pct:.2f}%) | "
          f"acoustic {tot['ac_bad']}/{n_ac} ({ac_pct:.2f}%) | frames with delta>1 "
          f"{tot['big_delta_frames']} ({big_pct:.2f}%) | max delta "
          f"{max(tot['deltas']) if tot['deltas'] else 0}", flush=True)
    assert sem_pct <= MAX_SEMANTIC_FLIP_PCT, (
        f"{label}: semantic flips {sem_pct:.2f}% of {n_frames} frames, above "
        f"{MAX_SEMANTIC_FLIP_PCT}% -- a wrong semantic code changes the audio outright")
    assert ac_pct < MAX_ACOUSTIC_MISMATCH_PCT, (
        f"{label}: acoustic mismatch {ac_pct:.2f}% above {MAX_ACOUSTIC_MISMATCH_PCT}%")
    assert big_pct <= MAX_BIG_DELTA_FRAME_PCT, (
        f"{label}: {big_pct:.2f}% of frames carry an acoustic delta above one FSQ level, above "
        f"{MAX_BIG_DELTA_FRAME_PCT}% -- a delta of one is boundary rounding, more is a different value")


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_model_teacher_forced_codes(pipe):
    """Every prompt, 64 frames: the breadth that shows the rates rather than one lucky window."""
    tot = {"frames": 0, "sem_bad": 0, "ac_bad": 0, "big_delta_frames": 0, "deltas": Counter()}
    for ci in range(len(case_ids())):
        embeds, case = fixture_embeds(ci, pipe.wb)
        r = _chain(pipe, embeds, N_FRAMES)
        for k in ("frames", "sem_bad", "ac_bad", "big_delta_frames"):
            tot[k] += r[k]
        tot["deltas"].update(r["deltas"])
        print(f"  case {ci:>2} ({case['voice']:<16} P={embeds.shape[1]:>3}): semantic "
              f"{r['sem_bad']}, acoustic {r['ac_bad']}/{r['frames'] * 36} "
              f"({r['ac_bad'] / (r['frames'] * 36) * 100:.2f}%), worst frame "
              f"f{r['worst_frame'][0]} at {r['worst_frame'][1]}/36", flush=True)
    print(f"\n  acoustic |delta| histogram: {dict(sorted(tot['deltas'].items()))}")
    _assert_rates("64-frame breadth", tot)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("ci", LONG_CASES)
def test_model_teacher_forced_full_utterance(pipe, ci):
    """One whole utterance, ~450 frames: the horizon a request actually runs.

    Held separately from the breadth test because it answers a different question -- not "what are
    the rates" but "do they grow over 36 s". They do not.
    """
    embeds, case = fixture_embeds(ci, pipe.wb)
    r = _chain(pipe, embeds, LONG_CAP, stop_on_end=True)
    assert r["frames"] > 300, f"case {ci} ended at {r['frames']} frames -- too short to be a horizon test"
    print(f"  case {ci} ({case['voice']}): {r['frames']} frames = {r['frames'] / 12.5:.1f}s, "
          f"deltas {dict(sorted(r['deltas'].items()))}")
    _assert_rates(f"case {ci} full utterance", r)
