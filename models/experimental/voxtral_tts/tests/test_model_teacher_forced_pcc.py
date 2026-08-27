# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blocks 1+2 end to end: do device and reference emit the same integer codes?

Teacher-forced -- both loops are fed the reference's codes each step, so every frame is an
independent measurement. Feeding each loop its own codes compares diverging sequences instead.

Synthetic and real-prompt figures are both printed and are not comparable; only the real-prompt
block is asserted.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_model_teacher_forced_pcc.py
"""

from collections import Counter

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref  # noqa: E402
from models.experimental.voxtral_tts.tests.gates import compare_codes_frame  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import fixture_embeds  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (  # noqa: E402
    CFG_ALPHA,
    TtVoxtralPipeline,
    open_device,
)

REAL_CASES = (0, 2, 3)
N_FRAMES = 8
# Loose tripwires, not the measured levels. Measured on the shipped build: semantic exact,
# 5.2% of acoustic codes off by one FSQ level of 21.
MAX_ACOUSTIC_MISMATCH_PCT = 15.0


@pytest.fixture(scope="module")
def pipe():
    d = open_device()
    p = TtVoxtralPipeline(d)
    yield p
    ttnn.close_device(d)


def _compare_codes(pipe, embeds, n_frames=N_FRAMES, cfg_alpha=CFG_ALPHA, seed=0):
    """-> (sem_bad, ac_bad, total_ac, delta histogram).

    The semantic code is reported apart from the 36 acoustic ones: a wrong semantic code changes the
    audio outright, while an acoustic code is one of 21 FSQ levels."""
    wf = fref.load_flow_state()
    ref_dec = bref.IncrementalBackbone(pipe.wb)
    torch.manual_seed(seed)
    h_ref = ref_dec.prefill(embeds)
    h_dev = pipe.backbone.prefill_last(embeds)

    sem_bad = ac_bad = total_ac = 0
    deltas = Counter()
    for i in range(n_frames):
        torch.manual_seed(1000 + i)  # same noise draw both sides, so only the model differs
        c_ref = fref.reference_frame(h_ref[:, 0], wf, cfg_alpha=cfg_alpha)
        torch.manual_seed(1000 + i)
        c_dev = pipe.flow(h_dev[:, 0], cfg_alpha=cfg_alpha)
        m = compare_codes_frame(c_ref, c_dev)
        sem_bad += 0 if m["sem_ok"] else 1
        ac_bad += m["n_diff"]
        total_ac += 36
        for v in m["deltas"]:
            deltas[v] += 1
        # teacher forcing: BOTH advance on the REFERENCE's codes
        emb = bref.embed_frame(pipe.wb, c_ref[0])
        h_ref = ref_dec.step(emb)
        h_dev = pipe.backbone.step(emb).reshape(1, 1, -1)
    return sem_bad, ac_bad, total_ac, deltas


@pytest.mark.slow
def test_model_teacher_forced_codes(pipe):
    """Real-prompt code agreement: the number that predicts audio."""
    tot_sem = tot_bad = tot_n = 0
    all_deltas = Counter()
    for ci in REAL_CASES:
        embeds, case = fixture_embeds(ci, pipe.wb)
        sem, bad, n, deltas = _compare_codes(pipe, embeds)
        all_deltas.update(deltas)
        tot_sem += sem
        tot_bad += bad
        tot_n += n
        print(
            f"\n  case {ci} ({case['voice']}, P={embeds.shape[1]}): "
            f"semantic {sem} wrong of {N_FRAMES}, acoustic {bad}/{n} ({bad/max(n,1)*100:.1f}%)"
        )
    pct = tot_bad / max(tot_n, 1) * 100
    print(
        f"\n  REAL-PROMPT TOTAL: semantic {tot_sem} wrong, acoustic {tot_bad}/{tot_n} ({pct:.1f}%)"
        f"\n  acoustic |delta| histogram: {dict(sorted(all_deltas.items()))}"
    )
    assert tot_sem == 0, f"{tot_sem} semantic codes differ -- a wrong semantic code changes the audio outright"
    assert pct < MAX_ACOUSTIC_MISMATCH_PCT, f"acoustic mismatch {pct:.1f}% exceeds {MAX_ACOUSTIC_MISMATCH_PCT}%"
    assert set(all_deltas) <= {1}, f"acoustic deltas beyond one FSQ level: {dict(sorted(all_deltas.items()))}"
