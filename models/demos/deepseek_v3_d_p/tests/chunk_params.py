# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Per-chip prefill depths for parametrized tests.

A 4x4 stage (SP=4) runs a 5120-token chunk at 1280 rows per chip, against 640 on the 8x4 stage (SP=8).
Tests keep their 640 rows on every mesh and, through ``with_sp4_depth``, gain a 1280 twin of each row
that collects only on the 4x4 mesh.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP, SP4_CHUNK_TOKENS_PER_CHIP


def is_4x4(**params) -> bool:
    """``uncollect_if`` predicate: the row runs on the 4x4 stage mesh."""
    return tuple(params.get("mesh_device") or ()) == (4, 4)


def _not_4x4(**params) -> bool:
    return not is_4x4(**params)


def with_sp4_depth(rows, ids=None):
    """``rows`` (tuples or ``pytest.param``s whose first value is the per-chip depth) unchanged, each followed
    by a twin at ``SP4_CHUNK_TOKENS_PER_CHIP`` that is deselected on every mesh but 4x4 (``uncollect_if``).

    Original ids stay as they were, so existing ``-k`` filters keep selecting the same cases; a twin's id is
    its row's id plus ``-sp4_1280``. Pass ``ids`` when the parametrize used an ``ids=`` list: the result
    carries its ids itself, so drop that ``ids=`` argument."""
    out = []
    for i, row in enumerate(rows):
        values, marks, row_id = (row.values, list(row.marks), row.id) if hasattr(row, "values") else (row, [], None)
        if ids is not None:
            row_id = ids[i]
        assert values[0] == PREFILL_CHUNK_TOKENS_PER_CHIP, f"row {i}: first value {values[0]} is not the per-chip depth"
        out.append(pytest.param(*values, marks=marks, id=row_id))
        out.append(
            pytest.param(
                SP4_CHUNK_TOKENS_PER_CHIP,
                *values[1:],
                marks=[*marks, pytest.mark.uncollect_if(pred=_not_4x4)],
                id=None if row_id is None else f"{row_id}-sp4_1280",
            )
        )
    return out
