# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for quasar TM ops."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)


def _explicit_height_shard_config(device, ncores, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _run_quasar_slice(shape, begins, ends, step, imc, omc, device):
    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=imc)
    result = ttnn.experimental.quasar.slice(ttnn_in, list(begins), list(ends), list(step), memory_config=omc)

    actual = result.memory_config()
    assert actual.memory_layout == omc.memory_layout
    if omc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        assert actual.shard_spec is not None

    slices = tuple(slice(b, e, s) for b, e, s in zip(begins, ends, step))
    ref = x[slices]
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_ulp(ref, got, ulp_threshold=0)


@pytest.mark.parametrize(
    "shape, begins, ends, step, in_shard, out_shard",
    [
        pytest.param(
            (1, 1, 52, 64), (0, 0, 0, 0), (1, 1, 26, 64), (1, 1, 1, 1), (4, 13, 64), (2, 13, 64), id="A_coalesced"
        ),
        pytest.param(
            (1, 1, 32, 64), (0, 0, 0, 16), (1, 1, 32, 32), (1, 1, 1, 1), (4, 8, 64), (4, 8, 16), id="F_w_begin_aligned"
        ),
        pytest.param(
            (1, 1, 32, 52),
            (0, 0, 0, 0),
            (1, 1, 26, 52),
            (1, 1, 1, 1),
            (4, 8, 52),
            (2, 13, 52),
            id="H_w_unaligned_stride",
        ),
        # HS→L1 fallback with misaligned W-begin. HS→HS output triggers an unrelated Quasar
        # SliceRmProgramFactory bug in the misalignment+HS-output path; interleaved output still
        # exercises the misaligned-begin fallback route through the same predicate.
        pytest.param(
            (1, 1, 32, 64),
            (0, 0, 0, 1),
            (1, 1, 32, 32),
            (1, 1, 1, 1),
            (4, 8, 64),
            None,
            id="G_w_begin_misaligned_routes_to_rm_fallback",
        ),
    ],
)
def test_quasar_slice_row_major_height_sharded_nontile_aligned(shape, begins, ends, step, in_shard, out_shard, device):
    imc = _explicit_height_shard_config(device, *in_shard)
    omc = _explicit_height_shard_config(device, *out_shard) if out_shard is not None else L1_INTERLEAVED
    _run_quasar_slice(shape, begins, ends, step, imc, omc, device)
