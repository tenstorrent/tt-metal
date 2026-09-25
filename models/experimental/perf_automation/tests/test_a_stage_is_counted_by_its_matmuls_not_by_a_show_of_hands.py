# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The item count that sets a stage's compute roof comes from its matmuls, not from a majority vote.

THE SHAPE OF THE BUG. `_stage_items_observed` used "does this op's shape parse as MxK @ KxN" as its
test for "is this op a matmul", and that is not what the parse means: a LayerNorm, an SDPA and a
BinaryNg publish the same fingerprint and parse just as cleanly. Every one of them voted, and they
outnumber the matmuls.

That is harmless while both populations agree, which is why it went unseen. It bites where a stage
folds its batch into the matmul while its elementwise ops stay per-request. Measured on
voxtral_mini_3b_2507 prefill: 25 non-matmul votes at 416 rows against 15 matmuls at 3328 (416 x
batch 8). The mode returned 416, the compute roof came out 8x small, and the stage was reported
memory-bound -- the exact failure the mode's own comment warns under-counting causes. Encode (1500
either way) and decode (8 either way) were untouched, because their two populations happen to agree;
luck, not a property, and the next model batching a different stage inherits the bug.

The class is already recorded on the bucket, by the same taxonomy the rest of the tool classifies
with. This asserts the count reads it.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _bucket(cls, ops):
    return {"id": cls, "top_ops": [{"shape": s, "rows": r, "count": c} for s, r, c in ops]}


def _profile(stage, buckets):
    return {"stage_buckets": {stage: buckets}}


def test_the_matmuls_set_the_count_even_when_outvoted():
    """The measured voxtral prefill shape: 15 matmuls at 3328 against 25 other ops at 416."""
    from agent.opclass import MATMUL_OP_CLASS
    from cc_optimize.summary import _stage_items_observed

    prof = _profile(
        "s",
        [
            _bucket(MATMUL_OP_CLASS, [("3328x3072 @ 3072x4096", 3328, 15)]),
            _bucket("reduction", [("416x3072 @ 3072x3072", 416, 25)]),
            _bucket("eltwise", [("416x3072 @ 3072x3072", 416, 6)]),
        ],
    )
    assert _stage_items_observed("s", prof) == 3328


def test_a_stage_whose_populations_agree_is_unchanged():
    """Encode and decode: the non-matmuls carry the same row count, so nothing moves either way."""
    from agent.opclass import MATMUL_OP_CLASS
    from cc_optimize.summary import _stage_items_observed

    prof = _profile(
        "s",
        [
            _bucket(MATMUL_OP_CLASS, [("1500x1280 @ 1280x1280", 1500, 8)]),
            _bucket("reduction", [("1500x1280 @ 1280x1280", 1500, 20)]),
        ],
    )
    assert _stage_items_observed("s", prof) == 1500


def test_the_mode_still_beats_a_tile_padded_outlier():
    """Why this stays a mode over the matmuls and does not become max-by-FLOPs: a decode step's
    vocab head runs at a tile-padded 32 rows for a batch of 8, and must not set the count."""
    from agent.opclass import MATMUL_OP_CLASS
    from cc_optimize.summary import _stage_items_observed

    prof = _profile(
        "s",
        [_bucket(MATMUL_OP_CLASS, [("8x3072 @ 3072x3072", 8, 19), ("32x3072 @ 3072x131072", 32, 1)])],
    )
    assert _stage_items_observed("s", prof) == 8


def test_a_profile_with_no_classed_buckets_is_priced_as_before():
    """Profiles written before buckets carried a class must keep their old number, not drop to 0
    and be priced at one item."""
    from cc_optimize.summary import _stage_items_observed

    prof = _profile(
        "s",
        [
            {"top_ops": [{"shape": "3328x3072 @ 3072x4096", "rows": 3328, "count": 15}]},
            {"top_ops": [{"shape": "416x3072 @ 3072x3072", "rows": 416, "count": 25}]},
        ],
    )
    assert _stage_items_observed("s", prof) == 416


def test_the_true_shape_wins_even_when_outnumbered_by_a_smaller_chunked_population():
    """Measured on voxtral_mini_3b_2507 prefill, batch 32: 16 chunked matmuls at 3328 rows outnumber
    7 full-sequence ones at the true 13312 (416 x 32), and the mode used to pick the smaller,
    outvoted-but-wrong value -- under-counting the compute roof 4x. Weighting by total rows moved
    (rows x count) picks 13312 (13312x7=93184 beats 3328x16=53248) without needing to know which
    population is "real"; it just weighs actual arithmetic, not occurrence count."""
    from agent.opclass import MATMUL_OP_CLASS
    from cc_optimize.summary import _stage_items_observed

    prof = _profile(
        "s",
        [
            _bucket(
                MATMUL_OP_CLASS,
                [
                    ("3328x3072 @ 3072x4096", 3328, 16),
                    ("13312x3072 @ 3072x4096", 13312, 7),
                    ("32x3072 @ 3072x3072", 32, 4),
                    ("1024x3072 @ 3072x3072", 1024, 4),
                ],
            )
        ],
    )
    assert _stage_items_observed("s", prof) == 13312


def test_the_mechanism_holds_at_a_different_batch_scale():
    """Same shape of bug, scaled to a smaller batch than the measured case -- chunked matmuls must
    not outvote full-sequence ones just because there happen to be more of them, regardless of the
    absolute row counts involved."""
    from agent.opclass import MATMUL_OP_CLASS
    from cc_optimize.summary import _stage_items_observed

    prof = _profile(
        "s",
        [_bucket(MATMUL_OP_CLASS, [("832x3072 @ 3072x4096", 832, 16), ("3328x3072 @ 3072x4096", 3328, 7)])],
    )
    assert _stage_items_observed("s", prof) == 3328


def test_a_padded_fallback_outlier_still_cannot_win_by_raw_size():
    """The OTHER path into `_rows`: an op with `rows` absent falls back to its parsed (padded) M.
    Weighting by total volume must keep this outlier from outranking real work even though its
    single value (32) is larger than the real one (8) -- the exact case the mode was written to
    protect, reached through the fallback rather than an explicit rows=32."""
    from agent.opclass import MATMUL_OP_CLASS
    from cc_optimize.summary import _stage_items_observed

    prof = _profile(
        "s",
        [
            _bucket(
                MATMUL_OP_CLASS,
                [
                    ("8x3072 @ 3072x3072", 8, 19),
                    ("32x3072 @ 3072x131072", None, 1),  # rows absent -> parsed padded M = 32
                ],
            )
        ],
    )
    assert _stage_items_observed("s", prof) == 8


def test_equal_total_volume_ties_break_to_the_larger_row_count():
    """Two populations of equal total arithmetic: the tie-break still prefers the larger row count,
    matching the existing preference for over- over under-counting the compute roof."""
    from agent.opclass import MATMUL_OP_CLASS
    from cc_optimize.summary import _stage_items_observed

    prof = _profile(
        "s",
        [_bucket(MATMUL_OP_CLASS, [("100x1 @ 1x1", 100, 2), ("200x1 @ 1x1", 200, 1)])],  # both volume 200
    )
    assert _stage_items_observed("s", prof) == 200


def test_a_stage_that_ran_no_arithmetic_states_nothing():
    from cc_optimize.summary import _stage_items_observed

    assert _stage_items_observed("s", {}) == 0
    assert _stage_items_observed("s", None) == 0
    assert _stage_items_observed("s", _profile("s", [_bucket("datamove", [("?x? @ ?x?", 0, 3)])])) == 0


def test_the_class_is_read_from_the_taxonomy_not_typed_here():
    """MATMUL_OP_CLASS must stay whatever OP_CLASS_MAP emits for a matmul op code."""
    from agent.opclass import MATMUL_OP_CLASS, classify_op

    assert MATMUL_OP_CLASS == classify_op("Matmul")
    assert MATMUL_OP_CLASS == classify_op("Linear")
    assert classify_op("LayerNorm") != MATMUL_OP_CLASS
