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
