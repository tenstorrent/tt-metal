# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device) tests of the packed decode-input planner (``tt/packed_inputs.py``,
``GLM53_DECODE_PACKED_INPUTS``): the field offsets / widths / dtypes of 1-row and 8-row meshes with the user
partition on and off and with / without ``dense_positions``, the segment order (one per row, or one per chip in
row-major chip order), the round trip pack -> host unpack against an independent reimplementation of the direct
path's host values (``decode_position_tensors`` / ``_dense_bound`` / ``SparseDecodeMeta.host_values``), and the
input validation.

    PYTHONPATH=<tt-metal root> python -m pytest models/demos/glm53_flash/tests/test_decode_packed_inputs_host.py \\
        --noconftest -q -p no:cacheprovider
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.tt.packed_inputs import (
    COL_STRIDE,
    INT32,
    UINT32,
    PackedDecodeLayout,
    position_fields,
    sparse_host_values,
)

P, K, D = (
    4,
    512,
    2050,
)  # index_kpool, index_select_pools and the dense bound (index_topk + index_kpool - 2) of GLM-5.3-Flash
VOCAB = 154880


@contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host test runs with ``--noconftest`` because the root conftest imports ttnn at collection."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


def _layout(rows, cols, B, partition=False, dense=False, per_user=False) -> PackedDecodeLayout:
    return PackedDecodeLayout(
        rows=rows,
        cols=cols,
        users_per_row=B,
        index_kpool=P,
        index_select_pools=K,
        dense_max_pos=D,
        partition=partition,
        dense=dense,
        per_user_valid_len=per_user,
    )


def _reference(positions: list[int]) -> dict[str, list[int]]:
    """The direct path's per-user host values, written out independently of the planner (tt/dsa.py's docstring:
    n complete pools, t tail tokens; a dense user has n < K)."""
    out = {name: [] for name in ("positions", "pool_ids", "ring_slots", "dense_positions", "valid", "tail", "cur")}
    for a in positions:
        n, t = (a + 1) // P, (a + 1) % P
        out["positions"].append(a)
        out["pool_ids"].append(a // P)
        out["ring_slots"].append(a % P)
        out["dense_positions"].append(min(a, D))
        out["valid"].append(max(n, K))
        out["tail"].append(n)
        out["cur"].append(a if n < K else P * K - 1 + t)
    return out


def _ladder(U: int, seed: int = 0) -> list[int]:
    """Positions over the mesh-wide user list: the dense range, the pool boundary 2047 -> 2048 (the first sparse
    user), the dense bound 2050 -> 2051 and positions far beyond (a 1 M context)."""
    pool = [0, 1, 3, 4, 5, 600, 2044, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2100, 4097, 262144, 1048575]
    g = torch.Generator().manual_seed(seed)
    return [pool[int(i)] for i in torch.randint(0, len(pool), (U,), generator=g)]


# ------------------------------------------------------------------------------------------------ layouts
def test_single_chip_layout():
    lay = _layout(1, 1, 4)
    f = lay.fields
    assert lay.segments == 1 and lay.users_per_chip == 4 and lay.segment_cols == 1
    assert lay.row_fields == ("ids", "positions", "pool_ids", "ring_slots", "cur_pos")
    assert lay.column_block == 5 and lay.height == 9 and lay.width == 2 * COL_STRIDE == 32
    assert lay.segment_shape == (1, 1, 9, 32) and lay.host_shape == (1, 1, 9, 32)
    assert f["ids"].start == (0, 0, 0, 0) and f["ids"].end == (1, 1, 1, 4) and f["ids"].dtype == UINT32
    for r, name in enumerate(("positions", "pool_ids", "ring_slots", "cur_pos"), start=1):
        assert f[name].start == (0, 0, r, 0) and f[name].end == (1, 1, r + 1, 4) and f[name].dtype == INT32
        assert not f[name].column and f[name].n == 4
    assert f["tail_pool"].start == (0, 0, 5, 0) and f["tail_pool"].end == (1, 1, 9, 1) and f["tail_pool"].column
    assert f["valid_len_rows"].start == (0, 0, 5, COL_STRIDE) and f["valid_len_rows"].end == (1, 1, 9, COL_STRIDE + 1)
    assert f["tail_pool"].dtype == f["valid_len_rows"].dtype == UINT32 and f["valid_len_rows"].n == 4
    assert "dense_positions" not in f and not [n for n in f if n.startswith("valid_len[")]
    assert lay.segment(0, 0) == 0 and lay.chip_users(0, 0) == (0, 4)


def test_galaxy_layout_partition_dense_per_user():
    lay = _layout(8, 4, 32, partition=True, dense=True, per_user=True)
    f = lay.fields
    assert lay.segment_cols == 4 and lay.users_per_chip == 8 and lay.segments == 32 and lay.total_users == 256
    assert lay.row_fields == ("ids", "positions", "pool_ids", "ring_slots", "dense_positions", "cur_pos")
    assert lay.column_block == 6 and lay.height == 14 and lay.width == 32
    assert lay.host_shape == (1, 1, 32 * 14, 32)
    assert f["ids"].end == (1, 1, 1, 32)  # the row's 32 ids on every chip of the row
    assert f["dense_positions"].start == (0, 0, 4, 0) and f["dense_positions"].end == (1, 1, 5, 8)
    assert f["cur_pos"].start == (0, 0, 5, 0) and f["cur_pos"].end == (1, 1, 6, 8)
    assert f["tail_pool"].end == (1, 1, 14, 1) and f["valid_len_rows"].end == (1, 1, 14, COL_STRIDE + 1)
    for u in range(8):
        g = f[f"valid_len[{u}]"]
        assert g.start == (0, 0, 6 + u, COL_STRIDE) and g.end == (1, 1, 7 + u, COL_STRIDE + 1)
        assert g.n == 1 and g.column and g.dtype == UINT32
    assert "valid_len[8]" not in f
    for r in range(8):
        for c in range(4):
            assert lay.segment(r, c) == r * 4 + c and lay.chip_users(r, c) == ((r * 4 + c) * 8, (r * 4 + c) * 8 + 8)
    assert len(f) == 6 + 2 + 8


def test_galaxy_layout_without_partition_replicates_the_row_segment():
    lay = _layout(8, 4, 32)
    assert lay.segment_cols == 1 and lay.users_per_chip == 32 and lay.segments == 8 and lay.height == 5 + 32
    for r in range(8):
        assert {lay.segment(r, c) for c in range(4)} == {r}
        assert lay.chip_users(r, 3) == (r * 32, (r + 1) * 32)
    assert lay.fields["cur_pos"].end == (1, 1, 5, 32) and lay.fields["tail_pool"].end == (1, 1, 37, 1)


def test_width_and_validation():
    assert _layout(1, 1, 1).width == 32 and _layout(1, 1, 64).width == 64 and _layout(2, 4, 33).width == 33
    with expect_error(ValueError, "multiple of cols"):
        _layout(2, 4, 6, partition=True)
    _layout(2, 1, 6, partition=True)  # one column: the partition is a no-op
    with expect_error(ValueError, "at least one row"):
        _layout(0, 4, 8)
    with expect_error(ValueError, "users_per_row 0"):
        _layout(1, 1, 0)
    with expect_error(ValueError, "outside"):
        _layout(2, 4, 8).segment(2, 0)


# ---------------------------------------------------------------------------------------------- values
def test_formulas_match_the_direct_path_at_the_boundaries():
    pos = [0, 1, 2, 3, 4, 2046, 2047, 2048, 2050, 2051, 2052, 2055, 1048575]
    ref = _reference(pos)
    valid, tail, cur = sparse_host_values(pos, P, K)
    assert (valid, tail, cur) == (ref["valid"], ref["tail"], ref["cur"])
    pf = position_fields(pos, P, D)
    assert pf == {k: ref[k] for k in ("positions", "pool_ids", "ring_slots", "dense_positions")}
    # the dense / sparse switch: n = 512 complete pools from position 2047 on; cur jumps to the compact rows
    assert dict(zip(pos, cur))[2046] == 2046 and dict(zip(pos, cur))[2047] == P * K - 1 == 2047
    assert dict(zip(pos, cur))[2050] == 2050 and dict(zip(pos, cur))[2051] == 2047 and dict(zip(pos, cur))[2052] == 2048
    assert dict(zip(pos, valid))[2046] == 512 and dict(zip(pos, valid))[2051] == 513
    assert dict(zip(pos, pf["dense_positions"]))[2052] == 2050


@pytest.mark.parametrize(
    "rows,cols,B,partition,dense",
    [
        (1, 1, 4, False, False),
        (1, 1, 1, False, True),
        (1, 4, 4, False, True),
        (1, 4, 8, True, False),
        (2, 4, 4, False, False),
        (2, 4, 4, True, False),
        (8, 4, 32, False, True),
        (8, 4, 32, True, False),
        (8, 4, 32, True, True),
    ],
)
@pytest.mark.parametrize("per_user", [False, True])
def test_pack_round_trip(rows, cols, B, partition, dense, per_user):
    lay = _layout(rows, cols, B, partition, dense, per_user)
    U = rows * B
    g = torch.Generator().manual_seed(rows * 100 + cols * 10 + B)
    ids = torch.randint(0, VOCAB, (U,), generator=g).long()
    positions = _ladder(U, seed=B)
    packed = lay.pack(ids, positions)
    assert packed.dtype == torch.int32 and tuple(packed.shape) == lay.host_shape
    assert int(packed.min()) >= 0 and int(packed.max()) < 2**31  # exact as uint32 after the bitcast
    ref = _reference(positions)
    Bc = lay.users_per_chip
    for r in range(rows):
        for c in range(cols):
            seg = lay.segment(r, c)
            got = lay.unpack_host(packed, seg)
            assert got == lay.expected_chip_fields(ids, positions, r, c), (r, c)
            first, end = lay.chip_users(r, c)
            assert end - first == Bc and first == seg * Bc
            assert got["ids"] == ids[r * B : (r + 1) * B].tolist()
            for name in ("positions", "pool_ids", "ring_slots"):
                assert got[name] == ref[name][first:end], (name, r, c)
            if dense:
                assert got["dense_positions"] == ref["dense_positions"][first:end]
            else:
                assert "dense_positions" not in got
            assert got["cur_pos"] == ref["cur"][first:end] and got["tail_pool"] == ref["tail"][first:end]
            assert got["valid_len_rows"] == ref["valid"][first:end]
            if per_user:
                for u in range(Bc):
                    assert got[f"valid_len[{u}]"] == [ref["valid"][first + u]]
    # the host tensor is segment-major on dim 2 and every column of a non-partitioned row reads the same segment
    for seg in range(lay.segments):
        assert torch.equal(lay.segment_of(packed, seg), packed[:, :, seg * lay.height : (seg + 1) * lay.height])
    if not partition or cols == 1:
        assert all(lay.segment(r, c) == r for r in range(rows) for c in range(cols))
    # nothing but the fields is written
    used = sum(f.n for name, f in lay.fields.items() if not name.startswith("valid_len["))
    assert int((packed != 0).sum()) <= used * lay.segments


def test_pack_validates_its_inputs():
    lay = _layout(2, 4, 4)
    ids = torch.arange(8)
    with expect_error(ValueError, "positions for 8 user slots"):
        lay.pack(ids, [0] * 7)
    with expect_error(ValueError, "ids for 8 user slots"):
        lay.pack(torch.arange(9), [0] * 8)
    with expect_error(ValueError, r"positions: values must be in \[0, 2\*\*31\)"):
        lay.pack(ids, [0] * 7 + [-1])
    with expect_error(ValueError, "ids: values must be"):
        lay.pack(torch.tensor([2**31] + [0] * 7), [0] * 8)
    with expect_error(ValueError, "is not"):
        lay.segment_of(torch.zeros(1, 1, 3, 32, dtype=torch.int32), 0)
    assert lay.pack(ids.reshape(2, 4), [5] * 8).shape == lay.host_shape  # any id shape, row r's users first


def test_host_fields_are_the_mesh_wide_lists():
    lay = _layout(2, 4, 4, partition=True, dense=True)
    ids = torch.arange(8) + 100
    positions = [0, 2046, 2047, 2051, 5, 2100, 8, 262144]
    vals = lay.host_fields(ids, positions)
    ref = _reference(positions)
    assert vals["ids"] == ids.tolist() and vals["positions"] == positions
    assert vals["cur_pos"] == ref["cur"] and vals["tail_pool"] == ref["tail"] and vals["valid_len_rows"] == ref["valid"]
    assert vals["dense_positions"] == ref["dense_positions"]
    assert set(vals) == {
        "ids",
        "positions",
        "pool_ids",
        "ring_slots",
        "dense_positions",
        "cur_pos",
        "tail_pool",
        "valid_len_rows",
    }
