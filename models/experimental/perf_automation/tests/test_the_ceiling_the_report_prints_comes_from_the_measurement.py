# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The number on the page, against the measurement it claims to be derived from.

WHY THIS FILE EXISTS. Every other test in this suite hands one function a dict its author wrote and
checks what comes back. That is worth having and it is not enough: it cannot see a JOINT. Three
joints in the ceiling chain were broken at once on 2026-08-16, the suite was green throughout, and
the report printed a decode floor of 14.11 ms against a 2.89 ms measurement -- 2496.7 GB/s, 487% of
a 512 GB/s part.

    census 1.718 GB ──X── anchor 7.223 GB ──X── share ──> floor 14.11 ms ──> the row
                      1                    2

    1. THE ANCHOR. The pre-census guess had moved from params x 1.0 to params x <declared dtype>,
       and _anchor_is_placeholder still recognised only x 1.0. So the guess stopped being
       recognised as a guess and the census could not replace it. The guarding test passed the
       whole time, because it constructs an anchor of params x 1.0 -- the value the code no longer
       produces. It encoded the old world and kept passing in the new one.

    2. THE SHARE. stage_roots joins the block count the PROBE observed against the depths the
       checkpoint declares -- but the probe runs depth-capped by design, reporting 2 where the
       model has 32 and 30. It returned {} on every real run and had never once fired, so encode
       got no memory ceiling at all and printed "not modelled" beside a 12.80 ms measurement.

And a third, in the same table: encode published 345.7 tok/s/u, which is 1000/2.8926 -- DECODE's
rate, on encode's row, in a unit encode does not use. The rate was handed to any stage retiring one
item per unit, which an encoder pass is as much as a decoded token.

So these tests start from voxtral's REAL facts and assert the END of the chain: the bytes the floor
divides by are the bytes the census measured, apportioned by the tower the stage actually runs.
Nothing here is a unit test of a helper -- each one fails if any joint between the measurement and
the printed number comes apart.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_PEAK = 512e9

# Voxtral-Mini-3B-2507, as run 5 recorded it on 2026-08-16. device_weight_bytes and
# device_section_bytes are the census's own output; the sections are what the numel join produces.
_FACTS = {
    "total_params": 3611483136,
    "dominant_dtype": "bfloat16",
    "weight_bytes": 9356474312,
    "layers": 32,
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "kv_heads": 8,
    "head_dim": 128,
    "device_weight_bytes": 1718081696,
    "device_census_complete": True,
    "bytes_per_param": 1.3228,
    # A device split that is NOT the disk split: 1.718 GB resident, apportioned as the chip holds it
    # rather than as the bf16 file does (85.8% / 13.7% / 0.5%). If these ever agree by accident the
    # tests below cannot tell the two sources apart, which is how the first version of this file
    # passed while reading the disk ratio.
    "device_section_bytes": {"language_model": 1300000000, "audio_tower": 350081696, "multi_modal_projector": 68000000},
    "stage_roots": {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"},
}
_ANCHOR_BF16 = 7222966272  # 3611483136 x 2.0 -- the anchor run 5 actually pinned


def _floor_ms(nbytes) -> float:
    return 1e3 * float(nbytes) / _PEAK


# ------------------------------------------------------------------ joint 1: census -> anchor


def test_the_bf16_guess_is_recognised_as_a_guess():
    """THE REGRESSION, stated as arithmetic. params x 2.0 is a prediction of what the loader would
    do, made before it did it -- exactly what params x 1.0 was, and superseded for the same reason."""
    from agent.perf_target import _anchor_is_placeholder

    assert _anchor_is_placeholder(_ANCHOR_BF16, _FACTS) is True


def test_every_width_the_tool_can_guess_with_is_recognised():
    """Not a second special case. Whatever dtype a checkpoint declares, params x that width is the
    same kind of guess, and pinning the recogniser to one of them is how this broke."""
    from agent.perf_target import BYTES_PER_ELEM, _anchor_is_placeholder

    for w in {2.0, 1.0625, 0.5625, 1.0, 4.0} & set(BYTES_PER_ELEM.values()):
        anchor = round(_FACTS["total_params"] * w)
        assert _anchor_is_placeholder(anchor, _FACTS) is True, w


def test_evidence_is_still_never_overridden():
    """The narrowness is the point: only a guess is replaceable. A checkpoint total, a measured
    figure, or a previous census stays pinned exactly as it was."""
    from agent.perf_target import _anchor_is_placeholder

    assert _anchor_is_placeholder(_FACTS["weight_bytes"], _FACTS) is False
    assert _anchor_is_placeholder(_FACTS["device_weight_bytes"], _FACTS) is False


def test_the_ceiling_divides_by_what_the_census_measured():
    """END OF THE CHAIN. Given the anchor run 5 pinned, the target must come back with the censused
    bytes -- not 7.223 GB, and not the checkpoint's 9.356 GB."""
    from agent.perf_target import compute_target

    t = compute_target(_FACTS, {"dram_bw_gbps": 512.0}, bytes_per_unit=_ANCHOR_BF16)
    assert t.active_bytes == _FACTS["device_weight_bytes"]
    assert abs(_floor_ms(t.active_bytes) - 3.36) < 0.05, _floor_ms(t.active_bytes)


def test_the_printed_floor_is_no_longer_four_times_the_truth():
    """The report said 14.11 ms. The measurement it was printed beside was 2.89 ms."""
    from agent.perf_target import compute_target

    t = compute_target(_FACTS, {"dram_bw_gbps": 512.0}, bytes_per_unit=_ANCHOR_BF16)
    assert _floor_ms(t.active_bytes) < 0.5 * 14.11


# ------------------------------------------------------------------ joint 2: stage -> tower


def test_every_declared_stage_has_a_tower():
    """encode's ceiling was refused outright -- "not modelled" -- because no stage had a root."""
    from cc_optimize.summary import _roofline_stage_share

    for stage in ("encode", "prefill", "decode"):
        assert _roofline_stage_share(_FACTS, stage) > 0.0, stage


def test_a_stage_is_priced_from_its_own_towers_measured_bytes():
    """Not the whole model, and not the checkpoint's proportions."""
    from cc_optimize.summary import _roofline_stage_share

    res = float(_FACTS["device_weight_bytes"])
    assert abs(_roofline_stage_share(_FACTS, "decode") - 1300000000 / res) < 1e-9
    assert abs(_roofline_stage_share(_FACTS, "encode") - 350081696 / res) < 1e-9


def test_the_audio_tower_is_not_priced_at_the_backbones_bytes():
    """The failure this whole chain exists to prevent: one tower charged at another's weight."""
    from cc_optimize.summary import _roofline_stage_share

    assert _roofline_stage_share(_FACTS, "encode") < _roofline_stage_share(_FACTS, "decode")


def test_the_disk_ratio_is_not_what_gets_used():
    """85.8% is the language tower's share of the bf16 FILE. The chip is mixed precision, so the
    two differ -- and if the report ever prints the disk figure again, this catches it."""
    from cc_optimize.summary import _roofline_stage_share

    assert abs(_roofline_stage_share(_FACTS, "decode") - 0.858) > 0.01


# ------------------------------------------------------------------ joint 3: the rate on the row


def test_a_self_timed_stage_publishes_its_own_rate():
    """encode printed 345.7 tok/s/u -- 1000/2.8926, decode's rate, on encode's row. A stage the run
    timed separately has a rate of its own; only a stage with no timing of its own borrows one."""
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index("_own_ms = _ms is not None")
    body = src[i : i + 1400]
    assert "_mrate = (1000.0 / _ms) if (_own_ms and _ms)" in body, "the headline rate is handed out again"
    j = src.index("if _ms is None and per_unit_ms")
    assert i < j, "_own_ms is read after the fallback, so every stage looks self-timed"


# ------------------------------------------------------------------ batch, on every stage


def test_batch_scales_the_per_user_terms_and_not_the_weights():
    """8 users share one weight read and carry their own KV -- which is the whole reason batching
    pays, and the reason a per-user ceiling falls only by the KV term."""
    from agent.perf_target import active_bytes

    one = active_bytes(_FACTS, regime="decode", seq_len=128, batch=1)
    eight = active_bytes(_FACTS, regime="decode", seq_len=128, batch=8)
    assert eight > one
    assert eight < 8 * one, "the weights were multiplied by the batch"
    kv_one = one - _FACTS["weight_bytes"]
    assert abs((eight - one) - 7 * kv_one) <= 1, "the KV term did not scale with the batch"
