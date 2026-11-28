# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of ckv_bfp4_rows (golden spec: ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/unit/test_ckv_bfp4_rows.py).

The golden suite runs on the bring-up spec's mesh (galaxy 8x4), replicated inputs, ``max_grid`` = the grid capped at
11 x 10, its axis ``layout`` in {interleaved, nd_sharded} ([1, 1, 16, 512] shards round-robin over the DRAM banks) and
its calls ``(src_row0, n, row0)`` in CALLS on a [4, 1, 4096, 512] source and a 9216-row table. Every axis value and
every golden call (the smallest, one band, and the largest, 8192 rows) appear here on that mesh, checked with the
golden suite's own reference (``tests/reference.encode_rows``) and its tolerance: byte-exact records on every chip,
every other row untouched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.ckv_bfp4_rows.pre_migration import ckv_rows_reference as own_ref
from tests.ttnn.unit_tests.operations.ckv_bfp4_rows.pre_migration.test_acceptance import (  # noqa: F401
    ckv_bfp4_rows_op,
)

GOLDEN_REFERENCE = Path(ttnn.__file__).parent / "bringup" / "ckv_bfp4_ttnn" / "tests" / "reference.py"


def _golden_encode():
    """The golden suite's reference function (byte rule); this suite's own encoder, proven equal to it, if the
    golden file is not importable here."""
    if GOLDEN_REFERENCE.exists():
        spec = importlib.util.spec_from_file_location("ckv_bfp4_golden_reference", GOLDEN_REFERENCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.encode_rows
    return own_ref.encode_rows


encode_rows = _golden_encode()

FILL = 0xAB
BATCHES, R, TABLE_ROWS = 4, 4096, 8192 + 1024
# the golden CALLS: one band (smallest), a slab batch at an offset, more bands than cores (largest), records past
# the source row
CALLS = [(0, 32, 0), (2048 + 64, 640, 96), (4096, 8192, 0), (1024, 4096, 4096)]


def _golden_inputs(n, seed):
    # the golden suite's input pattern (its _inputs)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 512, generator=g) * torch.pow(2.0, torch.randint(-12, 6, (n, 1), generator=g).float())
    x[:, :16] *= 50.0
    x[1, 32:48] = 0.0
    x[2] = 0.0
    return x.bfloat16()


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return (min(g.x, 11), min(g.y, 10))


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("layout", ["interleaved", "nd_sharded"])
def test_golden_cut(mesh_device, ckv_bfp4_rows_op, layout):
    mesh = mesh_device
    mesh.enable_program_cache()
    mesh.clear_program_cache()
    x = _golden_inputs(BATCHES * R, 7 + (layout == "nd_sharded"))
    mc = own_ref.nd_sharded_dram(mesh) if layout == "nd_sharded" else ttnn.DRAM_MEMORY_CONFIG
    src = own_ref.to_src(mesh, x, (BATCHES, 1, R, 512), mc)
    mg = _grid(mesh)
    for i, (src_row0, n, row0) in enumerate(CALLS):
        table = own_ref.to_table(mesh, (1, 1, TABLE_ROWS, 320), fill=FILL)
        before = mesh.num_program_cache_entries()
        result = ckv_bfp4_rows_op(src, table, src_row0, n, row0, max_grid=mg)
        delta = mesh.num_program_cache_entries() - before
        assert result is None or result.buffer_address() == table.buffer_address()
        if i == 0:
            assert delta in ((1,) if ckv_bfp4_rows_op.mode == "native" else (0, 1)), f"first call: {delta} programs"
        else:
            assert delta == 0, f"call {(src_row0, n, row0)}: one program serves every (src_row0, n, row0)"
        expected = encode_rows(x[src_row0 : src_row0 + n])
        chips = own_ref.table_host(table)
        assert len(chips) == mesh.get_num_devices()
        for chip, got in enumerate(chips):
            bad = (got[row0 : row0 + n] != expected).any(-1).nonzero().flatten()
            assert not len(
                bad
            ), f"{layout} {(src_row0, n, row0)} chip {chip}: {len(bad)} rows differ {bad[:5].tolist()}"
            assert bool((got[:row0] == FILL).all()) and bool((got[row0 + n :] == FILL).all()), "rows outside written"
        ttnn.deallocate(table)
