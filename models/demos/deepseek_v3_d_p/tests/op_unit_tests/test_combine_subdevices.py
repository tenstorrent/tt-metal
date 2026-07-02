# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Local-only test: combine op under edge-row / edge-column worker subdevices.

The combine kernel lays its sender + untilizer cores along a single 1-D line of the
worker grid.  When ``subdevice_id`` is None it uses the first row of the full grid
(legacy behavior).  When an explicit subdevice is given, the kernel must lay the cores
along whatever line that subdevice spans — a first/last ROW (varies in x) or a
first/last COLUMN (varies in y).  This test confines combine to each of the four edge
lines (plus the None default) and checks the output still matches the torch reference,
exercising both the row-oriented and the new column-oriented core layouts.

This file MUST NOT run in CI.  It is collected by every *DeepSeek_PREFILL_OP_TESTS job
(some of which run the op_unit_tests/ folder unfiltered, e.g. bh_p150 / bh_p300), so a
``-k`` filter cannot exclude it.  Instead it skips itself in CI via the is_ci_env /
is_ci_v2_env fixtures (same mechanism as test_sub_device_load_clear_timing.py): all CI
jobs collect it but report it as skipped, while it still runs locally with no env var.

Run locally (needs an 8-chip box for the mesh-4x2 config):

    pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_combine_subdevices.py -vvv --tb=short
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_output_shape,
    log_combine_mismatch_details,
    log_per_chip_statistics,
    validate_combine_output,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results

# A single representative 2-D mesh config (8 chips). The subdevice placement under test is
# a property of the per-chip worker grid, independent of the mesh topology, so one config
# is enough to exercise all four edge lines.
MESH_CONFIGS = [p for p in ALL_MESH_CONFIGS if p.id == "mesh-4x2"]

# Small PCC-checked data case (same shape as test_prefill_combine.py's "pcc" case).
SEQ_LEN_PER_CHIP = 128
EMB_DIM = 7 * 1024
NUM_ROUTED_EXPERTS = 16
NUM_EXPERTS_PER_TOK = 4
DISPATCH_BUFFER_CAPACITY_FACTOR = 4

# The five placements the combine op must support, plus one it must reject: a 2-D subdevice
# block (>1 row AND >1 column, smaller than the full grid) has no single-line core layout, so
# combine must TT_FATAL on it.
SUBDEVICE_EDGES = ["none", "first_row", "last_row", "first_col", "last_col", "grid_2d"]


def _edge_core_range_set(grid_x, grid_y, edge):
    """Build a CoreRangeSet covering exactly one edge line of the worker grid.

    CoreCoord is (x=column, y=row); CoreRange is an inclusive rectangle. A single row
    collapses the y axis; a single column collapses the x axis.
    """
    if edge == "first_row":
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, 0))})
    if edge == "last_row":
        return ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, grid_y - 1), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
        )
    if edge == "first_col":
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, grid_y - 1))})
    if edge == "last_col":
        return ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(grid_x - 1, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
        )
    if edge == "grid_2d":
        # Rejected case: a true 2-D block (rows 0..1 across every column, so >1 row AND >1
        # column) that is strictly smaller than the full grid. The combine kernel has no
        # single-line layout for this, so the op must TT_FATAL.
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, 1))})
    raise ValueError(f"unknown edge: {edge}")


def _enumerate_cores(core_range_set):
    """Expand a CoreRangeSet into the explicit sorted list of (x, y) cores it contains."""
    cores = []
    for r in core_range_set.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                cores.append((x, y))
    return sorted(cores)


def _assert_edge_geometry(edge, core_range_set, grid_x, grid_y):
    """Prove (and log) that ``core_range_set`` is exactly the named edge of the worker grid.

    This is the geometric half of the proof: it confirms the subdevice we hand the op is
    literally row 0 / the last row / column 0 / the last column. Combined with the op's
    ``worker_cores(TENSIX, sub_device_id)`` contract (it lays its sender + untilizer kernels
    on exactly this set), it pins down which cores combine actually runs on.
    """
    cores = _enumerate_cores(core_range_set)
    xs = sorted({x for x, _ in cores})
    ys = sorted({y for _, y in cores})
    logger.info(f"[proof] subdevice='{edge}' worker_grid=({grid_x}x{grid_y}) n_cores={len(cores)} cores={cores}")

    if edge == "first_row":
        assert ys == [0], f"first_row must lie on y==0, got rows {ys}"
        assert xs == list(range(grid_x)), f"first_row must span every column 0..{grid_x - 1}, got {xs}"
        which = "row y=0 (first row)"
    elif edge == "last_row":
        assert ys == [grid_y - 1], f"last_row must lie on y=={grid_y - 1}, got rows {ys}"
        assert xs == list(range(grid_x)), f"last_row must span every column 0..{grid_x - 1}, got {xs}"
        which = f"row y={grid_y - 1} (last row)"
    elif edge == "first_col":
        assert xs == [0], f"first_col must lie on x==0, got cols {xs}"
        assert ys == list(range(grid_y)), f"first_col must span every row 0..{grid_y - 1}, got {ys}"
        which = "column x=0 (first column)"
    elif edge == "last_col":
        assert xs == [grid_x - 1], f"last_col must lie on x=={grid_x - 1}, got cols {xs}"
        assert ys == list(range(grid_y)), f"last_col must span every row 0..{grid_y - 1}, got {ys}"
        which = f"column x={grid_x - 1} (last column)"
    else:
        raise ValueError(f"unknown edge: {edge}")
    logger.info(f"[proof] ✓ subdevice='{edge}' is exactly {which}")


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("subdevice_edge", SUBDEVICE_EDGES)
def test_combine_subdevice_placement(
    mesh_device, device_params, num_links, topology, subdevice_edge, is_ci_env, is_ci_v2_env
):
    """Run combine confined to an edge-row/column worker subdevice and validate vs torch.

    The four edge subdevices overlap at the grid corners, so they cannot share one
    sub-device manager; each placement gets its own manager, created and torn down here.
    """
    # Local-only: this file is collected by every DeepSeek_PREFILL_OP_TESTS job (some
    # unfiltered, e.g. bh_p150/bh_p300), so a -k filter cannot exclude it. Skip in CI the
    # same way test_sub_device_load_clear_timing.py does; runs locally with no env var.
    if is_ci_env or is_ci_v2_env:
        pytest.skip("Local-only combine subdevice test; skipped in CI")

    torch.manual_seed(42)
    num_devices = mesh_device.get_num_devices()

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(
        f"subdevice_edge={subdevice_edge} mesh={tuple(mesh_device.shape)} worker_grid=({grid.x}x{grid.y}) "
        f"dispatch_group_size={dispatch_group_size} num_dispatch_groups={num_dispatch_groups}"
    )

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        SEQ_LEN_PER_CHIP,
        NUM_ROUTED_EXPERTS,
        NUM_EXPERTS_PER_TOK,
        num_devices,
        dispatch_group_size,
        DISPATCH_BUFFER_CAPACITY_FACTOR,
    )

    # --- Torch inputs (isolate combine via the torch dispatch reference) ---
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size,
        SEQ_LEN_PER_CHIP,
        EMB_DIM,
        NUM_ROUTED_EXPERTS,
        NUM_EXPERTS_PER_TOK,
        max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=NUM_ROUTED_EXPERTS,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        NUM_ROUTED_EXPERTS,
        experts_per_chip,
        SEQ_LEN_PER_CHIP,
        NUM_EXPERTS_PER_TOK,
        expert_dispatch_table=expert_dispatch_table,
    )

    torch_dispatch_module = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=SEQ_LEN_PER_CHIP,
        emb_dim=EMB_DIM,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )
    dispatched_buffer, dispatched_metadata = torch_dispatch_module(x, weights, indices, expert_offsets)

    # --- Host -> device ---
    mesh_mapper = get_ep_mesh_mapper(mesh_device)
    tt_dispatched_buffer = ttnn.from_torch(
        dispatched_buffer,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_dispatched_metadata = ttnn.from_torch(
        dispatched_metadata,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=get_expert_token_counts_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_expert_region_offsets = ttnn.from_torch(
        expert_region_offsets,
        mesh_mapper=get_expert_token_counts_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # --- Torch golden ---
    torch_combine = TorchCombineModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        seq_len_per_chip=SEQ_LEN_PER_CHIP,
        num_dispatch_groups=num_dispatch_groups,
    )
    torch_output = torch_combine(dispatched_buffer, dispatched_metadata, expert_token_counts, expert_region_offsets)

    combine_kwargs = dict(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        seq_len_per_chip=SEQ_LEN_PER_CHIP,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=False,
    )

    # --- Rejected case: a 2-D subdevice block must make combine TT_FATAL ---
    if subdevice_edge == "grid_2d":
        if grid.y < 3:
            pytest.skip(f"grid_2d needs a worker grid taller than 2 rows; got {grid.x}x{grid.y}")
        edge_cores = _edge_core_range_set(grid.x, grid.y, "grid_2d")
        xs = sorted({x for x, _ in _enumerate_cores(edge_cores)})
        ys = sorted({y for _, y in _enumerate_cores(edge_cores)})
        assert len(xs) > 1 and len(ys) > 1, f"grid_2d must span >1 row AND >1 column, got cols={xs} rows={ys}"
        logger.info(f"[proof] subdevice='grid_2d' is a {len(xs)}x{len(ys)} block (cols={xs}, rows={ys})")
        sd_manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([edge_cores])], 0)
        mesh_device.load_sub_device_manager(sd_manager)
        try:
            with pytest.raises(RuntimeError, match="2-D subdevice core grid"):
                ttnn.experimental.deepseek_prefill.combine(
                    tt_dispatched_buffer,
                    tt_dispatched_metadata,
                    tt_expert_token_counts,
                    tt_expert_region_offsets,
                    subdevice_id=ttnn.SubDeviceId(0),
                    **combine_kwargs,
                )
        finally:
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(sd_manager)
        logger.info("✅ combine correctly rejected a 2-D subdevice core grid")
        return

    # --- Build the edge subdevice (None == legacy first-row-of-full-grid path) ---
    sd_manager = None
    subdevice_id = None
    if subdevice_edge == "none":
        # Legacy path: no subdevice -> combine uses the first row of the full worker grid.
        logger.info(f"[proof] subdevice='none' -> legacy path: first row (y=0) of the full {grid.x}x{grid.y} grid")
    else:
        edge_cores = _edge_core_range_set(grid.x, grid.y, subdevice_edge)
        # Prove the subdevice we pass is exactly the named edge before handing it to the op.
        _assert_edge_geometry(subdevice_edge, edge_cores, grid.x, grid.y)
        # Each edge line shares one manager-of-one; the four edges overlap at corners so
        # they cannot coexist in a single manager.
        sd_manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([edge_cores])], 0)
        mesh_device.load_sub_device_manager(sd_manager)
        subdevice_id = ttnn.SubDeviceId(0)

    try:
        tt_output = ttnn.experimental.deepseek_prefill.combine(
            tt_dispatched_buffer,
            tt_dispatched_metadata,
            tt_expert_token_counts,
            tt_expert_region_offsets,
            subdevice_id=subdevice_id,
            **combine_kwargs,
        )

        mesh_composer = get_ep_mesh_composer(mesh_device)
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)
    finally:
        if sd_manager is not None:
            mesh_device.clear_loaded_sub_device_manager()
            mesh_device.remove_sub_device_manager(sd_manager)

    # --- Validate (EP-rank-aware; bf16 -> allclose, same as test_prefill_combine.py) ---
    assert_output_shape(tt_output_torch, num_dispatch_groups, dispatch_group_size, "combine output")
    result = validate_combine_output(
        torch_output,
        tt_output_torch,
        indices,
        num_dispatch_groups,
        NUM_ROUTED_EXPERTS,
        use_pcc=False,
        verbose=True,
        expert_dispatch_table=expert_dispatch_table,
        expert_token_counts=expert_token_counts,
        experts_per_chip=experts_per_chip,
    )
    log_validation_results(
        results=[result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title=f"Combine Validation Results (subdevice={subdevice_edge})",
    )
    if not result.passed:
        log_combine_mismatch_details(result.mismatches, torch_output, tt_output_torch)
        log_per_chip_statistics(result.mismatches, dispatch_group_size, SEQ_LEN_PER_CHIP, NUM_EXPERTS_PER_TOK)
    result.assert_passed(f"Combine data mismatch (subdevice={subdevice_edge})")

    logger.info(f"✅ combine matches torch reference under subdevice={subdevice_edge}")
