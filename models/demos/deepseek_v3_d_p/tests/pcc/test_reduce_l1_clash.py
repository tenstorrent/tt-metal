# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for L1 memory clash in deepseek_moe_post_combine_reduce.

The fused reduce kernel uses ~1MB of static CBs per core. On linear-8 mesh
topology, something allocates ~459KB of L1 per core BEFORE the reduce runs,
causing a clash. This test isolates the problem by progressively adding
infrastructure to find which layer causes the allocation.

Tests:
  1. bare_device: Just the reduce op on a single device (no mesh, no CCL)
  2. with_global_semaphores: Add GlobalSemaphores like TT_CCL does
  3. with_sub_device_manager: Add sub-device manager with local_l1_size=0
  4. full_pipeline_single_device: Run combine + reduce on single device
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config

# ============================================================
# Helpers
# ============================================================


def create_combine_output_and_weights(device, seq_len, topk, emb_dim):
    """Create tensors matching real pipeline shapes, placed in DRAM."""
    # combine_output: [1, 1, seq_len, topk, emb_dim] ROW_MAJOR bfloat16
    torch_combine = torch.randn(1, 1, seq_len, topk, emb_dim, dtype=torch.bfloat16)
    tt_combine = ttnn.from_torch(
        torch_combine,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # weights: [1, 1, seq_len, topk, 1] ROW_MAJOR bfloat16
    torch_weights = torch.randn(1, 1, seq_len, topk, 1, dtype=torch.bfloat16)
    tt_weights = ttnn.from_torch(
        torch_weights,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return tt_combine, tt_weights, torch_combine, torch_weights


def run_reduce(device, tt_combine, tt_weights, expert_dim=3):
    """Run the fused reduce op and return output."""
    logger.info(
        f"  combine_output: shape={tt_combine.shape} mem={tt_combine.memory_config().buffer_type} addr={tt_combine.buffer_address()}"
    )
    logger.info(
        f"  weights:        shape={tt_weights.shape} mem={tt_weights.memory_config().buffer_type} addr={tt_weights.buffer_address()}"
    )

    result = ttnn.experimental.deepseek_moe_post_combine_reduce(
        tt_combine,
        tt_weights,
        expert_dim=expert_dim,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"  output: shape={result.shape} layout={result.layout}")
    return result


# ============================================================
# Test 1: Bare single device — no mesh, no CCL, no fabric
# ============================================================


@pytest.mark.parametrize(
    "seq_len, topk, emb_dim",
    [
        pytest.param(1600, 8, 7168, id="deepseek-v3"),
        pytest.param(32, 8, 2048, id="small"),
    ],
)
def test_reduce_bare_device(device, seq_len, topk, emb_dim):
    """Test reduce op on a fresh single device with no preceding operations.

    If this passes, the 1MB static CBs fit fine when L1 is clean.
    If this fails, the kernel itself has a fundamental L1 sizing issue.
    """
    logger.info(f"=== Test 1: bare device, seq_len={seq_len}, topk={topk}, emb_dim={emb_dim} ===")

    tt_combine, tt_weights, torch_combine, torch_weights = create_combine_output_and_weights(
        device, seq_len, topk, emb_dim
    )

    result = run_reduce(device, tt_combine, tt_weights, expert_dim=3)

    # Basic sanity: output shape should be [1, 1, seq_len, emb_dim]
    expected_shape = [1, 1, seq_len, emb_dim]
    assert list(result.shape) == expected_shape, f"Expected shape {expected_shape}, got {list(result.shape)}"

    # PCC check against torch reference
    torch_result = (torch_weights * torch_combine).sum(dim=3)  # sum over topk dim
    tt_result = ttnn.to_torch(result)
    from tests.ttnn.utils_for_testing import comp_pcc

    _, pcc = comp_pcc(torch_result.float(), tt_result.float())
    logger.info(f"  PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} too low"


# ============================================================
# Test 2: Single device + GlobalSemaphores (like TT_CCL)
# ============================================================


@pytest.mark.parametrize(
    "seq_len, topk, emb_dim, num_semaphores",
    [
        pytest.param(1600, 8, 7168, 38, id="deepseek-v3-38sem"),
        pytest.param(1600, 8, 7168, 2, id="deepseek-v3-2sem"),
    ],
)
def test_reduce_with_global_semaphores(device, seq_len, topk, emb_dim, num_semaphores):
    """Test reduce op after allocating GlobalSemaphores in L1.

    TT_CCL.__init__ creates ~38 GlobalSemaphores on the full compute grid.
    Each one is tiny (16 bytes/core aligned), but let's verify they don't clash.
    """
    logger.info(f"=== Test 2: {num_semaphores} GlobalSemaphores, seq_len={seq_len} ===")

    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

    # Allocate GlobalSemaphores (they persist as long as we hold references)
    semaphores = [ttnn.create_global_semaphore(device, cores, 0) for _ in range(num_semaphores)]
    logger.info(f"  Allocated {len(semaphores)} GlobalSemaphores")

    tt_combine, tt_weights, _, _ = create_combine_output_and_weights(device, seq_len, topk, emb_dim)
    result = run_reduce(device, tt_combine, tt_weights, expert_dim=3)

    expected_shape = [1, 1, seq_len, emb_dim]
    assert list(result.shape) == expected_shape

    # Keep semaphores alive until test ends
    del semaphores


# ============================================================
# Test 3: Single device + sub-device manager (local_l1_size=0)
# ============================================================


@pytest.mark.parametrize(
    "seq_len, topk, emb_dim",
    [
        pytest.param(1600, 8, 7168, id="deepseek-v3"),
    ],
)
def test_reduce_with_sub_device_manager(device, seq_len, topk, emb_dim):
    """Test reduce op after creating a sub-device manager with local_l1_size=0.

    This mimics TT_CCL.__init__ which creates:
      sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
      mesh_device.load_sub_device_manager(sub_device_manager)
    """
    logger.info(f"=== Test 3: sub-device manager (local_l1_size=0), seq_len={seq_len} ===")

    grid = device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([worker_cores])
    sub_device_manager = device.create_sub_device_manager([worker_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)

    logger.info(f"  Loaded sub-device manager with local_l1_size=0")

    # Also add semaphores like TT_CCL does
    semaphores = [ttnn.create_global_semaphore(device, worker_cores, 0) for _ in range(38)]
    logger.info(f"  Allocated 38 GlobalSemaphores")

    tt_combine, tt_weights, _, _ = create_combine_output_and_weights(device, seq_len, topk, emb_dim)
    result = run_reduce(device, tt_combine, tt_weights, expert_dim=3)

    expected_shape = [1, 1, seq_len, emb_dim]
    assert list(result.shape) == expected_shape

    del semaphores


# ============================================================
# Test 4: Simulate preceding ops — create + deallocate L1 tensors
# ============================================================


@pytest.mark.parametrize(
    "seq_len, topk, emb_dim",
    [
        pytest.param(1600, 8, 7168, id="deepseek-v3"),
    ],
)
def test_reduce_after_l1_allocations(device, seq_len, topk, emb_dim):
    """Test reduce op after creating/deallocating L1 tensors to simulate pipeline.

    If the allocator doesn't properly reclaim L1 after deallocation, this could
    leave phantom allocations that clash with the reduce CBs.
    """
    logger.info(f"=== Test 4: L1 alloc/dealloc then reduce, seq_len={seq_len} ===")

    # Simulate some L1 tensor allocations (like preceding ops might create)
    l1_tensors = []
    for i in range(4):
        t = ttnn.from_torch(
            torch.randn(1, 1, 32, emb_dim, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        l1_tensors.append(t)
        logger.info(f"  L1 tensor[{i}]: addr={t.buffer_address()}")

    # Deallocate all L1 tensors
    for t in l1_tensors:
        ttnn.deallocate(t)
    l1_tensors.clear()
    logger.info("  Deallocated all L1 tensors")

    tt_combine, tt_weights, _, _ = create_combine_output_and_weights(device, seq_len, topk, emb_dim)
    result = run_reduce(device, tt_combine, tt_weights, expert_dim=3)

    expected_shape = [1, 1, seq_len, emb_dim]
    assert list(result.shape) == expected_shape


# ============================================================
# Test 5: Persistent L1 tensor (NOT deallocated) — should clash
# ============================================================


@pytest.mark.parametrize(
    "seq_len, topk, emb_dim, l1_alloc_kb",
    [
        pytest.param(1600, 8, 7168, 414, id="deepseek-v3-414KB"),
        pytest.param(1600, 8, 7168, 200, id="deepseek-v3-200KB"),
    ],
)
def test_reduce_with_persistent_l1(device, seq_len, topk, emb_dim, l1_alloc_kb):
    """Test reduce op with a persistent L1 allocation still alive.

    The reduce kernel needs 1MB of static CBs.
    Available L1 = 1,572,864 - 111,104 = 1,461,760 bytes.
    If L1 allocation > 1,461,760 - 1,048,576 = 413,184 bytes, it SHOULD clash.

    This test intentionally creates a persistent L1 tensor of l1_alloc_kb KB
    to verify the clash threshold.
    """
    logger.info(f"=== Test 5: persistent {l1_alloc_kb}KB L1 alloc + reduce ===")

    # Create a persistent L1 tensor that stays alive during reduce
    alloc_elements = (l1_alloc_kb * 1024) // 2  # bfloat16 = 2 bytes
    # Need shape compatible with TILE_LAYOUT (multiples of 32)
    rows = 32
    cols = (alloc_elements // rows // 32) * 32  # round down to tile-aligned
    if cols == 0:
        cols = 32

    persistent_l1 = ttnn.from_torch(
        torch.randn(1, 1, rows, cols, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    actual_kb = persistent_l1.buffer_address()  # just for logging
    logger.info(f"  Persistent L1 tensor: shape={persistent_l1.shape} addr={actual_kb}")

    tt_combine, tt_weights, _, _ = create_combine_output_and_weights(device, seq_len, topk, emb_dim)

    if l1_alloc_kb > 413:
        # This should clash — expect an error
        with pytest.raises(Exception, match="clash|L1"):
            run_reduce(device, tt_combine, tt_weights, expert_dim=3)
        logger.info("  Expected clash occurred!")
    else:
        # This should fit
        result = run_reduce(device, tt_combine, tt_weights, expert_dim=3)
        expected_shape = [1, 1, seq_len, emb_dim]
        assert list(result.shape) == expected_shape

    ttnn.deallocate(persistent_l1)


# ============================================================
# Test 6: Mesh device + CCL setup (no preceding ops)
# ============================================================


@pytest.mark.parametrize(
    "seq_len, topk, emb_dim",
    [
        pytest.param(1600, 8, 7168, id="deepseek-v3"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_reduce_mesh_no_preceding_ops(mesh_device, device_params, seq_len, topk, emb_dim):
    """Test reduce on mesh with CCL setup but NO preceding dispatch/combine.

    This tests if the fabric + TT_CCL infrastructure alone causes the L1 clash.
    """
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache, get_tt_ccl

    logger.info(f"=== Test 6: mesh {mesh_device.shape}, CCL setup, no preceding ops ===")

    # Clear any cached CCL state
    clear_tt_ccl_cache()

    # Set up TT_CCL (sub-device manager + global semaphores) — same as production
    tt_ccl = get_tt_ccl(mesh_device)
    logger.info(f"  TT_CCL initialized: sub_device_crs={tt_ccl.sub_device_crs}")

    # Create tensors on mesh — each device gets the full tensor (replicated)
    torch_combine = torch.randn(1, 1, seq_len, topk, emb_dim, dtype=torch.bfloat16)
    tt_combine = ttnn.from_torch(
        torch_combine,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_weights = torch.randn(1, 1, seq_len, topk, 1, dtype=torch.bfloat16)
    tt_weights = ttnn.from_torch(
        torch_weights,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"  combine: shape={tt_combine.shape} mem={tt_combine.memory_config().buffer_type}")
    logger.info(f"  weights: shape={tt_weights.shape} mem={tt_weights.memory_config().buffer_type}")

    # Run fused reduce op directly
    result = ttnn.experimental.deepseek_moe_post_combine_reduce(
        tt_combine,
        tt_weights,
        expert_dim=3,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"  output: shape={result.shape}")

    expected_shape = [1, 1, seq_len, emb_dim]
    assert list(result.shape) == expected_shape

    clear_tt_ccl_cache()


# ============================================================
# Test 7: Dispatch + Combine THEN Reduce (proper data)
# ============================================================


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_reduce_mesh_after_dispatch_combine(mesh_device, device_params):
    """Reproduce the L1 clash: dispatch → combine → reduce.

    Uses the same infrastructure as test_ttnn_dispatch_combine to create
    valid dispatch/combine data, then runs the reduce op.
    """
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
        ExpertMapping,
        compute_constants,
        extract_mesh_config,
        get_dispatch_input_mesh_mapper,
        initialize_test_inputs,
    )
    from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
    from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache, get_tt_ccl

    torch.manual_seed(42)

    seq_len_per_chip = 1600
    emb_dim = 7168
    num_routed_experts = 64
    num_experts_per_tok = 8
    capacity_factor = 2
    num_links = 1
    topology = ttnn.Topology.Linear

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.info(f"=== Test 7: dispatch→combine→reduce on {mesh_device.shape} ===")
    logger.info(f"  {dispatch_group_size=} {num_dispatch_groups=} {sp_axis=}")

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        capacity_factor,
    )
    logger.info(f"  {experts_per_chip=} {metadata_len=} {max_dispatched_tokens_per_expert=}")

    # --- CCL setup ---
    clear_tt_ccl_cache()
    tt_ccl = get_tt_ccl(mesh_device)

    # --- Generate valid test inputs ---
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    mesh_mapper_dispatch_inputs = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)
    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_weights = ttnn.from_torch(
        weights,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_indices = ttnn.from_torch(
        indices,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # --- Routing setup ---
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    tt_moe_routing_setup = TtMoERoutingSetup(
        mesh_device=mesh_device, expert_dispatch_table=expert_dispatch_table, num_links=num_links
    )
    tt_dispatch_offsets, tt_expert_token_counts, _ = tt_moe_routing_setup(
        ttnn_top_k_experts_indices=indices,
        num_routed_experts=num_routed_experts,
        seq_len_per_chip=seq_len_per_chip,
        num_experts_per_tok=num_experts_per_tok,
    )

    # --- Dispatch ---
    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
    )

    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)
    logger.info("  Running dispatch...")
    tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
        tt_x, tt_weights, tt_indices, tt_dispatch_offsets, tt_expert_dispatch_table
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"  dispatched_buffer: shape={tt_dispatched_buffer.shape} mem={tt_dispatched_buffer.memory_config().buffer_type}"
    )

    # --- Combine ---
    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=True,
    )

    logger.info("  Running combine...")
    combined_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_expert_token_counts)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"  combined_output: shape={combined_output.shape} mem={combined_output.memory_config().buffer_type}")

    # --- Reduce (this is where the clash should happen) ---
    # Create weights for reduce: [1, 1, seq_len, topk, 1]
    torch_reduce_weights = torch.randn(1, 1, seq_len_per_chip, num_experts_per_tok, 1, dtype=torch.bfloat16)
    tt_reduce_weights = ttnn.from_torch(
        torch_reduce_weights,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info("  Running reduce op (expecting possible L1 clash)...")
    result = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combined_output,
        tt_reduce_weights,
        expert_dim=3,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"  reduce output: shape={result.shape}")

    expected_shape = [1, 1, seq_len_per_chip, emb_dim]
    assert list(result.shape) == expected_shape
    logger.info("  PASSED — no L1 clash!")

    clear_tt_ccl_cache()


# ============================================================
# Test 8: Dispatch + Expert FFN + Combine + Reduce (full pipeline)
# ============================================================


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_reduce_mesh_full_pipeline(mesh_device, device_params):
    """Full pipeline: dispatch -> expert_ffn -> combine -> reduce.

    This is the minimal repro that should trigger the L1 clash.
    """
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
        ExpertMapping,
        compute_constants,
        extract_mesh_config,
        get_dispatch_input_mesh_mapper,
        initialize_test_inputs,
    )
    from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
    from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache, get_tt_ccl

    torch.manual_seed(42)

    seq_len_per_chip = 1600
    emb_dim = 7168
    hidden_dim = 2048
    num_routed_experts = 64
    num_experts_per_tok = 8
    capacity_factor = 2
    num_links = 1
    topology = ttnn.Topology.Linear

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.info(f"=== Test 8: FULL pipeline on {mesh_device.shape} ===")

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        capacity_factor,
    )
    logger.info(f"  {experts_per_chip=} {max_dispatched_tokens_per_expert=}")

    clear_tt_ccl_cache()
    tt_ccl = get_tt_ccl(mesh_device)

    # --- Inputs ---
    x, weights_gate, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    mesh_mapper = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)
    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights_gate = ttnn.from_torch(
        weights_gate, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    # --- Routing ---
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    tt_moe_routing = TtMoERoutingSetup(
        mesh_device=mesh_device, expert_dispatch_table=expert_dispatch_table, num_links=num_links
    )
    tt_offsets, tt_token_counts, _ = tt_moe_routing(
        ttnn_top_k_experts_indices=indices,
        num_routed_experts=num_routed_experts,
        seq_len_per_chip=seq_len_per_chip,
        num_experts_per_tok=num_experts_per_tok,
    )

    # --- Dispatch ---
    tt_dispatch = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
    )
    tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)
    logger.info("  Running dispatch...")
    tt_dispatched, tt_metadata = tt_dispatch(tt_x, tt_weights_gate, tt_indices, tt_offsets, tt_table)
    ttnn.synchronize_device(mesh_device)

    # --- Expert FFN ---
    logger.info("  Running expert FFN...")
    routed_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=None,  # Random weights
    )

    dispatched_squeezed = ttnn.squeeze(tt_dispatched, dim=0)
    dispatched_squeezed = ttnn.squeeze(dispatched_squeezed, dim=0)
    dispatched_tiled = ttnn.to_layout(dispatched_squeezed, ttnn.TILE_LAYOUT)

    expert_outputs = routed_expert(dispatched_tiled, tt_token_counts)
    logger.info(f"  expert_outputs: shape={expert_outputs.shape} mem={expert_outputs.memory_config().buffer_type}")

    expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
    expert_outputs = ttnn.unsqueeze(expert_outputs, dim=0)
    expert_outputs_rm = ttnn.to_layout(expert_outputs, ttnn.ROW_MAJOR_LAYOUT)

    # --- Combine ---
    logger.info("  Running combine...")
    tt_combine = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=True,
    )
    combined_output = tt_combine(expert_outputs_rm, tt_metadata, tt_token_counts)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"  combined_output: shape={combined_output.shape} mem={combined_output.memory_config().buffer_type}")

    # --- Deallocate what we can before reduce ---
    ttnn.deallocate(tt_token_counts)

    # --- Reduce ---
    torch_reduce_weights = torch.randn(1, 1, seq_len_per_chip, num_experts_per_tok, 1, dtype=torch.bfloat16)
    tt_reduce_weights = ttnn.from_torch(
        torch_reduce_weights,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info("  Running reduce (expecting possible L1 clash)...")
    result = ttnn.experimental.deepseek_moe_post_combine_reduce(
        combined_output,
        tt_reduce_weights,
        expert_dim=3,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    logger.info(f"  reduce output: shape={result.shape}")
    logger.info("  PASSED - no L1 clash!")

    clear_tt_ccl_cache()
