# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""select_final_carry must return the separated tail's owner state, or the prefix carry, on every trace replay."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_xy_device_params
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY, _VALUE = 640, 24, 128, 128


def _bounds(sp_size: int, bounded: bool) -> list[tuple[int, int | None]]:
    """Bounds replayed through one trace: aligned, rotated, split, last-rank and wrapped starts, and intervals ending
    before, exactly at and past the start of the separated tail (span - 32 rows after a 32-row offset)."""
    span = sp_size * _ROWS
    if not bounded:
        return [(start, None) for start in (0, _ROWS, _ROWS + 32, span - 32, span + 32, 32)]
    split = _ROWS + 32
    return [(split, split + span), (split, split + span - _ROWS), (0, span // 2), (32, span), (32, span - 32)]


def _meshes():
    return [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        ),
        pytest.param(
            (2, 4),
            fabric_1d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="SP2xTP4",
        ),
    ]


def _expected(rank_final: torch.Tensor, prefix: torch.Tensor, sp_size: int, start: int, end: int | None):
    """Chronology by hand: the tail separates from the prefix when the interval wraps back into its first rank."""
    offset = start % _ROWS
    first_rank = (start // _ROWS) % sp_size
    separated = sp_size > 1 and offset != 0
    if separated and end is not None:
        separated = end - start > (_ROWS - offset) + (sp_size - 1) * _ROWS
    return rank_final[first_rank][:, -1] if separated else prefix


@pytest.mark.parametrize("mesh_device,device_params", _meshes(), indirect=True)
@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
@pytest.mark.parametrize("groups", [1, 4], ids=["final", "grouped"])
def test_select_final_carry_follows_bounds_across_replays(mesh_device, bounded, groups):
    torch.manual_seed(groups)
    sp_size, tp_size = tuple(mesh_device.shape)
    # Every rank holds distinct group states, so a wrong owner, group or stale copy cannot match by chance.
    rank_final = torch.randn(sp_size, _HEADS, groups, _KEY, tp_size * _VALUE)
    prefix = torch.randn(_HEADS, _KEY, tp_size * _VALUE)
    rank_host = rank_final.reshape(sp_size * _HEADS, groups, _KEY, tp_size * _VALUE)
    if groups == 1:
        rank_host = rank_host[:, 0]
    rank_final_tt = ttnn.from_torch(
        rank_host,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, rank_host.dim() - 1), mesh_shape=(sp_size, tp_size)),
    )
    prefix_tt = ttnn.from_torch(
        prefix,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=(sp_size, tp_size)),
    )
    bounds = _bounds(sp_size, bounded)
    start_tt = make_actual_start(mesh_device, bounds[0][0])
    end_tt = make_actual_start(mesh_device, bounds[0][1]) if bounded else None

    def run():
        return ttnn.experimental.kda.select_final_carry(
            rank_final_tt,
            prefix_tt,
            actual_start=start_tt,
            actual_end=end_tt,
            local_rows=_ROWS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            sequence_parallel_axis=0,
        )

    # Compile, then capture once; every replay below only changes the bounds on device.
    ttnn.deallocate(run())
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = run()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for start, end in bounds:
            for target, value in ((start_tt, start), (end_tt, end)):
                if target is not None:
                    source = make_actual_start(mesh_device, value)
                    ttnn.copy(source, target)
                    ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            expected = _expected(rank_final, prefix, sp_size, start, end)
            got = ttnn.to_torch(
                output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 2), mesh_shape=(sp_size, tp_size)),
            )
            for rank in range(sp_size):
                assert torch.equal(
                    got[rank * _HEADS : (rank + 1) * _HEADS], expected
                ), f"rank {rank} carry differs for start={start}, end={end}"
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (output, rank_final_tt, prefix_tt, start_tt, end_tt):
            if tensor is not None:
                ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device,device_params", _meshes(), indirect=True)
@pytest.mark.parametrize(
    "case, error",
    [
        pytest.param("rank_final_dtype", "rank_final must be FLOAT32", id="rank-final-must-be-fp32"),
        pytest.param("prefix_final_dtype", "prefix_final must be FLOAT32", id="prefix-final-must-be-fp32"),
        pytest.param("prefix_final_layout", "prefix_final must use TILE layout", id="prefix-final-must-be-tiled"),
        pytest.param("prefix_final_rank", r"prefix_final must be \[B\*H, K, V\]", id="prefix-final-must-be-rank-3"),
        pytest.param("same_tensor", "must be different tensors", id="rank-final-must-not-be-prefix-final"),
        pytest.param("state_shape", r"rank_final must be \[B\*H, K, V\], or", id="rank-final-must-match-prefix-final"),
        pytest.param(
            "group_shape", r"rank_final must be \[B\*H, K, V\], or", id="grouped-rank-final-must-match-prefix-final"
        ),
        pytest.param("actual_start_dtype", "actual_start must be UINT32", id="actual-start-must-be-uint32"),
        pytest.param("local_rows", "local_rows must be positive and 32-aligned", id="local-rows-must-be-tile-aligned"),
        pytest.param("axis", "sequence_parallel_axis 2 is out of range", id="sp-axis-must-be-in-range"),
        pytest.param("num_links", "exceeds the", id="num-links-must-fit-the-axis"),
    ],
)
def test_select_final_carry_rejects_contract_violations(mesh_device, case, error, expect_error):
    def to_device(shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    state_shape = (_HEADS, _KEY, _VALUE)
    rank_final = to_device(state_shape)
    prefix_final = to_device(state_shape)
    actual_start = make_actual_start(mesh_device)
    kwargs = {"local_rows": _ROWS, "sequence_parallel_axis": 0}
    if case == "rank_final_dtype":
        rank_final = to_device(state_shape, dtype=ttnn.bfloat16)
    elif case == "prefix_final_dtype":
        prefix_final = to_device(state_shape, dtype=ttnn.bfloat16)
    elif case == "prefix_final_layout":
        prefix_final = to_device(state_shape, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "prefix_final_rank":
        rank_final, prefix_final = to_device((1, *state_shape)), to_device((1, *state_shape))
    elif case == "same_tensor":
        rank_final = prefix_final
    elif case == "state_shape":
        rank_final = to_device((_HEADS, _KEY, 2 * _VALUE))
    elif case == "group_shape":
        rank_final = to_device((_HEADS, 4, _KEY, 2 * _VALUE))
    elif case == "actual_start_dtype":
        actual_start = to_device((1,), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "local_rows":
        kwargs["local_rows"] = _ROWS + 16
    elif case == "axis":
        kwargs["sequence_parallel_axis"] = 2
    else:
        kwargs["num_links"] = 1000

    with expect_error(RuntimeError, error):
        ttnn.experimental.kda.select_final_carry(rank_final, prefix_final, actual_start=actual_start, **kwargs)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("projection", [False, True], ids=["carry", "with-projection"])
def test_select_final_carry_from_the_last_rank(mesh_device, projection):
    """prefix_on_last_rank: only the last chronological rank holds the prefix's final state and multicasts it, on the
    caller's cores and semaphores; with_projection also runs a ttnn.linear-equal projection in the same dispatch."""
    torch.manual_seed(0)
    sp_size, tp_size = tuple(mesh_device.shape)
    width = tp_size * _VALUE
    rank_final = torch.randn(sp_size, _HEADS, _KEY, width)
    # Every rank holds a different prefix state, so a carry from any rank but the last cannot match.
    prefix = torch.randn(sp_size, _HEADS, _KEY, width)
    shard = lambda t: ttnn.from_torch(
        t.reshape(sp_size * _HEADS, _KEY, width),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 2), mesh_shape=(sp_size, tp_size)),
    )
    rank_final_tt, prefix_tt = shard(rank_final), shard(prefix)
    grid = mesh_device.compute_with_storage_grid_size()
    # The carry on the bottom of the last worker column; the projection on the columns to its left.
    cores = [ttnn.CoreCoord(grid.x - 1, y) for y in range(grid.y - 4, grid.y)]
    core_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    semaphores = [ttnn.create_global_semaphore(mesh_device, core_set, 0) for _ in range(2)]
    ttnn.synchronize_device(mesh_device)
    replicate = lambda t: ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x, w = replicate(torch.randn(1, 128, 256)), replicate(torch.randn(256, 512) * 0.05)
    config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=2,
        out_block_h=1,
        out_block_w=2,
        per_core_M=1,
        per_core_N=2,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
    )
    expected_projection = ttnn.to_torch(
        ttnn.linear(x, w, program_config=config, compute_kernel_config=compute_config),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    try:
        for start in (0, _ROWS, _ROWS + 32, sp_size * _ROWS - 32):
            start_tt = make_actual_start(mesh_device, start)
            kwargs = dict(
                actual_start=start_tt,
                local_rows=_ROWS,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                sequence_parallel_axis=0,
                cores=cores,
                prefix_on_last_rank=True,
                semaphores=semaphores,
            )
            if projection:
                carry, projected = ttnn.experimental.kda.select_final_carry_with_projection(
                    rank_final_tt,
                    prefix_tt,
                    x,
                    w,
                    projection_program_config=config,
                    projection_compute_kernel_config=compute_config,
                    **kwargs,
                )
                got = ttnn.to_torch(projected, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
                assert torch.equal(got, expected_projection), f"projection differs from ttnn.linear for start={start}"
                ttnn.deallocate(projected)
            else:
                carry = ttnn.experimental.kda.select_final_carry(rank_final_tt, prefix_tt, **kwargs)
            first_rank = (start // _ROWS) % sp_size
            separated = start % _ROWS != 0
            expected = rank_final[first_rank] if separated else prefix[(first_rank + sp_size - 1) % sp_size]
            got = ttnn.to_torch(
                carry, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 2), mesh_shape=(sp_size, tp_size))
            )
            for rank in range(sp_size):
                assert torch.equal(
                    got[rank * _HEADS : (rank + 1) * _HEADS], expected
                ), f"rank {rank} carry differs for start={start}"
            ttnn.deallocate(carry)
            ttnn.deallocate(start_tt)
    finally:
        for tensor in (rank_final_tt, prefix_tt, x, w):
            ttnn.deallocate(tensor)
