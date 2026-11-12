# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The backbone and flow model end to end: do device and reference emit the same integer codes?

Teacher-forced (both loops advance on the reference's codes), 64 frames on every prompt plus two
full utterances, gated on mismatch rates rather than exact agreement.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/pcc/test_model_teacher_forced_pcc.py
    pytest -svv ... -k "not full_utterance"      # the 64-frame breadth alone
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
    needs_checkpoint,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (  # noqa: E402
    CFG_ALPHA,
    TtVoxtralPipeline,
    open_device,
)

pytestmark = needs_checkpoint

N_FRAMES = 64  # reaches frames 40 and 55, the hardest for decode
LONG_CASES = (2, 3)  # the two prompts with a full-length natural utterance
LONG_CAP = 480

# Rate ceilings: a multiple of what the fixture prompts reach, with a floor.
MAX_SEMANTIC_FLIP_PCT = 5.0
MAX_BIG_DELTA_FRAME_PCT = 7.5
MAX_ACOUSTIC_MISMATCH_PCT = 15.0


@pytest.fixture(scope="module")
def pipe():
    d = open_device()
    p = TtVoxtralPipeline(d)
    yield p
    ttnn.close_device(d)


def _chain(pipe, embeds, n_frames, cfg_alpha=CFG_ALPHA, stop_on_end=False):
    """Teacher-forced chain -> dict of counts. `stop_on_end` stops at the reference's [END_AUDIO],
    beyond which it would be off-distribution.
    """
    wf = fref.load_flow_state()
    ref_dec = bref.IncrementalBackbone(pipe.wb)
    h_ref = ref_dec.prefill(embeds)
    pipe.backbone.reset()
    h_dev = pipe.backbone.prefill_last(embeds)

    r = {"frames": 0, "sem_bad": 0, "ac_bad": 0, "big_delta_frames": 0, "deltas": Counter(), "worst_frame": (0, -1)}
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
    """Assert the three rates. Max delta is reported, never asserted: only its frequency has a
    defensible bound."""
    n_frames, n_ac = tot["frames"], tot["frames"] * 36
    sem_pct = tot["sem_bad"] / max(n_frames, 1) * 100
    ac_pct = tot["ac_bad"] / max(n_ac, 1) * 100
    big_pct = tot["big_delta_frames"] / max(n_frames, 1) * 100
    print(
        f"\n  {label}: {n_frames} frames | semantic {tot['sem_bad']} ({sem_pct:.2f}%) | "
        f"acoustic {tot['ac_bad']}/{n_ac} ({ac_pct:.2f}%) | frames with delta>1 "
        f"{tot['big_delta_frames']} ({big_pct:.2f}%) | max delta "
        f"{max(tot['deltas']) if tot['deltas'] else 0}",
        flush=True,
    )
    assert sem_pct <= MAX_SEMANTIC_FLIP_PCT, (
        f"{label}: semantic flips {sem_pct:.2f}% of {n_frames} frames, above "
        f"{MAX_SEMANTIC_FLIP_PCT}% -- a wrong semantic code changes the audio outright"
    )
    assert (
        ac_pct < MAX_ACOUSTIC_MISMATCH_PCT
    ), f"{label}: acoustic mismatch {ac_pct:.2f}% above {MAX_ACOUSTIC_MISMATCH_PCT}%"
    assert big_pct <= MAX_BIG_DELTA_FRAME_PCT, (
        f"{label}: {big_pct:.2f}% of frames carry an acoustic delta above one FSQ level, above "
        f"{MAX_BIG_DELTA_FRAME_PCT}% -- a delta of one is boundary rounding, more is a different value"
    )


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
        print(
            f"  case {ci:>2} ({case['voice']:<16} P={embeds.shape[1]:>3}): semantic "
            f"{r['sem_bad']}, acoustic {r['ac_bad']}/{r['frames'] * 36} "
            f"({r['ac_bad'] / (r['frames'] * 36) * 100:.2f}%), worst frame "
            f"f{r['worst_frame'][0]} at {r['worst_frame'][1]}/36",
            flush=True,
        )
    print(f"\n  acoustic |delta| histogram: {dict(sorted(tot['deltas'].items()))}")
    _assert_rates("64-frame breadth", tot)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("ci", LONG_CASES)
def test_model_teacher_forced_full_utterance(pipe, ci):
    """One whole utterance, the horizon a request actually runs: do the rates grow with length?"""
    embeds, case = fixture_embeds(ci, pipe.wb)
    r = _chain(pipe, embeds, LONG_CAP, stop_on_end=True)
    assert r["frames"] > 300, f"case {ci} ended at {r['frames']} frames -- too short to be a horizon test"
    print(
        f"  case {ci} ({case['voice']}): {r['frames']} frames = {r['frames'] / 12.5:.1f}s, "
        f"deltas {dict(sorted(r['deltas'].items()))}"
    )
    _assert_rates(f"case {ci} full utterance", r)
