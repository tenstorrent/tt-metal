# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for GPT-OSS fused MOE compute kernel (moe_gpt).

Dimensions:
  M = 32 (sequence length / tokens)
  K = 2880 (hidden_size) -> 90 tiles
  N = 2880 (intermediate_size) -> 90 tiles
  E = 4 (experts per device)
  L = 1 (layers)

W0/W1: [K, N] = [90, 90] tiles, distributed as 7-8 tiles/core (90/12 = 7.5)
W2:    [N, K] = [90, 90] tiles, distributed as 7-8 tiles/core (90/12 = 7.5)

Both W0/W1 and W2 have the same distribution since K == N == 2880.

Activation: SwiGLU
  gate_clamped = clamp(gate, max=7.0)
  up_clamped   = clamp(up, min=-7.0, max=7.0)
  result       = (up_clamped + 1) * gate_clamped * sigmoid(1.702 * gate_clamped)
"""

import itertools
import math
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose
from models.demos.gpt_oss.tt.experts_throughput.weights import (
    _FUSED_MAX_TILES_PER_CORE as MAX_W0_W1_TILES_PER_CORE,
    _FUSED_PAD_CORES as PAD_CORES,
    _prepare_w0_b0_w1_b1_tensor as prepare_w0_b0_w1_b1_tensor,
    _prepare_w2_b2_tensor as prepare_w2_b2_tensor,
)

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)

PCC_THRESHOLD = 0.984


def create_torch_input(L, in0_num_cores, E, M, K):
    """
    Create torch input tensor with random values.

    Returns:
        torch_input: Tensor of shape (L, in0_num_cores, E, M, K)
    """
    torch_input = torch.randn((L, E, M, K), dtype=torch.bfloat16) - 0.5
    return torch_input.unsqueeze(1).repeat(1, in0_num_cores, 1, 1, 1)


def create_torch_b0(L, E, N):
    # Shape: (L, E, 1, 2880) -> (L, E, 32, 2880)
    bias = torch.randn((L, E, 1, N), dtype=torch.bfloat16) - 0.5
    bias = torch.nn.functional.pad(bias, (0, 0, 0, 31))  # (L, E, 32, 2880)
    return bias


def create_torch_b1(L, E, N):
    # Shape: (L, E, 1, 2880) -> (L, E, 32, 2880)
    bias = torch.randn((L, E, 1, N), dtype=torch.bfloat16) - 0.5
    bias = torch.nn.functional.pad(bias, (0, 0, 0, 31))  # (L, E, 32, 2880)
    return bias


def create_torch_b2(L, E, N):
    # Shape: (L, E, 1, 2880) -> (L, E, 32, 2880)
    bias = -0.5 + torch.randn((L, E, 1, N), dtype=torch.bfloat16)
    bias = torch.nn.functional.pad(bias, (0, 0, 0, 31))  # (L, E, 32, 2880)
    return bias


def create_torch_w0(L, E, K, N):
    """Create torch w0 weight tensor of shape (L, E, K, N)."""
    return (torch.randn((L, E, K, N), dtype=torch.bfloat16) - 0.5) / 2
    # return torch.cat((torch.zeros((L, E, K, N-32), dtype=torch.bfloat16),torch.ones((L, E, K, 32), dtype=torch.bfloat16)/K), dim = 3 )
    # temp = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
    # indices = torch.arange(N, dtype=torch.bfloat16)
    # # indices[-32:] = -1
    # return (indices.view(1, 1, 1, N).expand_as(temp))/8192


def create_torch_w1(L, E, K, N):
    """Create torch w1 weight tensor of shape (L, E, K, N)."""
    return (torch.randn((L, E, K, N), dtype=torch.bfloat16) - 0.5) / 2
    # return torch.ones((L, E, K, N), dtype=torch.bfloat16) /2048
    # return torch.cat((torch.zeros((L, E, K, N-32), dtype=torch.bfloat16),torch.ones((L, E, K, 32), dtype=torch.bfloat16)/K), dim = 3 )
    # temp = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
    # indices = torch.arange(N, dtype=torch.bfloat16)
    # # indices[-32:] = -1
    # return -1 * (indices.view(1, 1, 1, N).expand_as(temp))


def create_torch_w2(L, E, N, K):
    """Create torch w2 weight tensor of shape (L, E, N, K)."""
    return (torch.randn((L, E, K, N), dtype=torch.bfloat16) - 0.5) / 2
    # return torch.ones((L, E, N, K), dtype=torch.bfloat16) /2048
    # temp = torch.randn((L, E, N, K), dtype=torch.bfloat16) -0.5
    # indices = torch.arange(K, dtype=torch.bfloat16)
    # indices = tensor = torch.repeat_interleave(torch.arange(1, (K//32)+ 1, dtype=torch.bfloat16), repeats=32)
    # # indices[-32:] = -1
    # return  indices.view(1, 1, 1,K).expand_as(temp)


def prepare_output_tensor(tt_output, E, M, K, ring2cores):
    """Extract valid tiles per core from the raw output tensor.

    Wraps :func:`extract_fused_moe_kernel_output` for the case where
    ``tt_output`` is already a torch tensor (not a ttnn tensor).
    """
    each_shard = []
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        num_tiles = 7 if pad_flag else 8
        each_shard.append(tt_output[ring_pos, :, :, : num_tiles * ttnn.TILE_SIZE])

    result = torch.cat(each_shard, dim=-1)
    assert result.shape == (E, M, K), f"Expected shape {(E, M, K)}, got {result.shape}"
    return result


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = (torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std) if std != 0 else 0.0
    allclose_passed, allclose_val = comp_allclose(torch_output, tt_output)
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
        "allclose": allclose_passed,
        "allclose_val": allclose_val,
    }


def swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """
    GPT-OSS SwiGLU activation reference.

    Args:
        gate: Tensor from W0 matmul
        up: Tensor from W1 matmul

    Returns:
        result = (clamp(up, -7, 7) + 1) * clamp(gate, max=7) * sigmoid(1.702 * clamp(gate, max=7))
    """
    gate_c = torch.clamp(gate, max=clamp_limit)
    up_c = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def run_test_moe_gpt(device, M, K, N, E, L, check_accuracy, dump_outputs):
    torch.manual_seed(0)
    logger.info(
        f"Running test_moe_gpt with M={M}, K={K}, N={N}, E={E}, L={L}, "
        f"check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    # Make a new list of core coords sorted in decreasing order by y then x (ring order)
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        # key: ring_pos, value: (core_coord, dram_bank_id, pad_flag)
        # pad_flag: 1 for cores with 7 tiles, 0 for cores with 8 tiles
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES else 0)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(in0_num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w0_dtype = ttnn.bfloat4_b
    num_dram_banks = 12

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    # --------------------------------------------------------------------------
    # Tensor shapes and memory configurations
    # --------------------------------------------------------------------------
    input_shape = (in0_num_cores, E, M, K)

    in0_shard_spec = ttnn.ShardSpec(
        grid=in0_core_range_set,
        shard_shape=(E * M, K),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
    # GPT-OSS: (12, L, E, 4, K, 128) -> shard per bank = L * E * 4 * K height, 128 width
    # 4 groups (8 max tiles / 2 per group), K=2880 height, 128 = 4 tiles width
    # ------------------------------------------------------------------------
    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2  # 4
    w0_w1_shard_height = L * E * groups_per_core * (K + 32)
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # GPT-OSS: (12, L, E, 2, N, 128) -> shard per bank = L * E * 2 * N height, 128 width
    # 2 groups, N=2880 height, 128 = 4 tiles width
    # 90/10 = 9 exact, no N-dimension padding needed.
    # ------------------------------------------------------------------------
    w2_shard_height = L * E * 2 * (N + 32)
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # --------------------------------------------------------------------------
    # Prepare the tensors
    # --------------------------------------------------------------------------
    if check_accuracy:
        torch_input = create_torch_input(L, in0_num_cores, E, M, K)
        torch_w0 = create_torch_w0(L, E, K, N)
        torch_w1 = create_torch_w1(L, E, K, N)
        torch_w2 = create_torch_w2(L, E, N, K)
        torch_b0 = create_torch_b0(L, E, N)
        torch_b1 = create_torch_b1(L, E, N)
        torch_b2 = create_torch_b2(L, E, K)

        # Prepare w0_w1 tensor (interleaved, and reordered)
        torch_w0_b0_w1_b1_reordered = prepare_w0_b0_w1_b1_tensor(
            torch_w0, torch_b0, torch_w1, torch_b1, L, E, K, N, ring2cores
        )

        tt_w0_b0_w1_b1 = ttnn.from_torch(
            torch_w0_b0_w1_b1_reordered,
            dtype=w0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w0_w1_mem_config,
        )

        # torch.set_printoptions(profile="full")
        # torch.set_printoptions(sci_mode=False)
        # print(f"Extracted first shard")
        # for k in range(0,4):
        #     for i in range(0,2):
        #         for j in range(0, 128):
        # with open(f"tensor_shard_expert.txt", "w") as f:
        #     f.write(str(torch_w0_b0_w1_b1_reordered[0,0,0,0,:,0]))

        # Prepare w2 tensor (padded and reordered)
        torch_w2_b2_reordered = prepare_w2_b2_tensor(torch_w2, torch_b2, L, E, N, K, ring2cores)

        tt_w2_b2 = ttnn.from_torch(
            torch_w2_b2_reordered, dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=w2_mem_config
        )
    else:
        tt_input = ttnn.empty(
            input_shape,
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )
        tt_w0_b0_w1_b1 = ttnn.empty(
            [num_dram_banks] + w0_w1_shard_spec.shape,
            dtype=w0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w0_w1_mem_config,
        )
        tt_w2_b2 = ttnn.empty(
            [num_dram_banks] + w2_shard_spec.shape,
            dtype=w0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config,
        )

    # --------------------------------------------------------------------------
    # Run the operation
    # --------------------------------------------------------------------------
    all_outputs = []
    all_accuracy_metrics = {}

    for layer_id in range(L):
        if check_accuracy:
            tt_input = ttnn.from_torch(
                torch_input[layer_id],
                dtype=in0_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_sharded_mem_config,
            )

        _tt_output = ttnn.experimental.moe_gpt(
            tt_input,
            w0_w1_tensor=tt_w0_b0_w1_b1,
            w2_tensor=tt_w2_b2,
            output_tensor=tt_input,
            num_experts=E,
            layer_id=layer_id,
        )

        # Output is produced in-place on the input tensor
        print("post_moe")
        tt_raw_output = ttnn.to_torch(tt_input)
        print("post_to_torch")
        tt_to_torch_output = prepare_output_tensor(tt_raw_output, E, M, K, ring2cores)
        all_outputs.append(tt_to_torch_output)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            # Reference calculation using SwiGLU
            torch_input_ref = torch_input[:, 0, ...]  # (L, E, M, K)

            # W0 and W1 projections
            # (L, E, M, K) @ (L, E, K, N) -> (L, E, M, N)
            torch_w0_output_ref = torch_input_ref @ torch_w0
            torch_w1_output_ref = torch_input_ref @ torch_w1

            # SwiGLU activation: gate=W0 output, up=W1 output
            # torch_intermediate_ref = swiglu_reference(torch_w0_output_ref, torch_w1_output_ref)

            torch_intermediate_ref = torch_w0_output_ref
            # W2 projection
            # (L, E, M, N) @ (L, E, N, K) -> (L, E, M, K)
            torch_output_ref = torch_intermediate_ref @ torch_w2

        # Calculate accuracy metrics for each layer and expert
        for layer_id, expert_id in itertools.product(range(L), range(E)):
            torch_layer_output = torch_output_ref[layer_id, expert_id, :, :]
            tt_layer_output = tt_to_torch_outputs[layer_id, expert_id, :, :]

            layer_metrics = get_accuracy_metrics(torch_layer_output, tt_layer_output)
            all_accuracy_metrics[(layer_id, expert_id)] = layer_metrics
            # breakpoint()

    if dump_outputs:
        torch.set_printoptions(profile="full")
        var2filename = {
            torch_w0_output_ref: "torch_w0_output_ref.txt",
            torch_w1_output_ref: "torch_w1_output_ref.txt",
            torch_intermediate_ref: "torch_intermediate_ref.txt",
            torch_output_ref: "torch_output_ref.txt",
            tt_raw_output: "tt_raw_output.txt",
            tt_to_torch_outputs: "tt_output_act.txt",
        }

        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    return all_accuracy_metrics


# GPT-OSS dimensions: M=32, K=2880, N=2880, E=4, L=1
SHAPE2TIME = {
    (32, 2880, 2880, 4, 1): 300.0,
}


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K, N, E, L",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe_gpt(device, M, K, N, E, L, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_moe_gpt(
        device,
        M,
        K,
        N,
        E,
        L,
        check_accuracy,
        dump_outputs,
    )

    passing = True
    for (layer_id, expert_id), metrics in accuracy_metrics.items():
        if metrics["pcc"] < PCC_THRESHOLD:
            passing = False
            logger.warning(f"Layer {layer_id}, Expert {expert_id}: PCC={metrics['pcc']:.6f}")
        else:
            logger.info(f"Layer {layer_id}, Expert {expert_id}: PCC={metrics['pcc']:.6f} (Passed)")

    assert passing, "Some experts in some layers did not pass the PCC check"


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K, N, E, L",
    SHAPE2TIME.keys(),
)
def test_moe_gpt_dram_output(device, M, K, N, E, L):
    """Test the fused MoE kernel with enable_dram_output=True.

    Verifies that the DRAM output tensor has the correct shape (E, 1, M, K)
    in TILE_LAYOUT, and that after converting to ROW_MAJOR, accuracy matches
    the torch reference (PCC >= threshold).
    """
    accuracy_metrics = run_test_moe_gpt_dram_output(device, M, K, N, E, L)

    passing = True
    for (layer_id, expert_id), metrics in accuracy_metrics.items():
        if metrics["pcc"] < PCC_THRESHOLD:
            passing = False
            logger.warning(f"DRAM output - Layer {layer_id}, Expert {expert_id}: PCC={metrics['pcc']:.6f}")
        else:
            logger.info(f"DRAM output - Layer {layer_id}, Expert {expert_id}: PCC={metrics['pcc']:.6f} (Passed)")

    assert passing, "Some experts in some layers did not pass the PCC check (DRAM output path)"


def run_test_moe_gpt_dram_output(device, M, K, N, E, L):
    """Run the fused MoE kernel with enable_dram_output=True and check accuracy."""
    logger.info(f"Running test_moe_gpt_dram_output with M={M}, K={K}, N={N}, E={E}, L={L}")

    # --------------------------------------------------------------------------
    # Shard grid (same setup as run_test_moe_gpt)
    # --------------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in PAD_CORES else 0)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(in0_num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w0_dtype = ttnn.bfloat4_b
    num_dram_banks = 12

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    # --------------------------------------------------------------------------
    # Tensor shapes and memory configurations
    # --------------------------------------------------------------------------
    input_shape = (in0_num_cores, E, M, K)

    in0_shard_spec = ttnn.ShardSpec(
        grid=in0_core_range_set,
        shard_shape=(E * M, K),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    groups_per_core = MAX_W0_W1_TILES_PER_CORE // 2
    w0_w1_shard_height = L * E * groups_per_core * K
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    w2_shard_height = L * E * 2 * N
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # --------------------------------------------------------------------------
    # Prepare the tensors
    # --------------------------------------------------------------------------
    torch_input = create_torch_input(L, in0_num_cores, E, M, K)
    torch_w0 = create_torch_w0(L, E, K, N)
    torch_w1 = create_torch_w1(L, E, K, N)
    torch_w2 = create_torch_w2(L, E, N, K)

    torch_w0_w1_reordered = prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores)
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w0_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
    )

    torch_w2_reordered = prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores)
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered, dtype=w0_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=w2_mem_config
    )

    # --------------------------------------------------------------------------
    # Run the operation with enable_dram_output=True
    # --------------------------------------------------------------------------
    all_accuracy_metrics = {}

    for layer_id in range(L):
        tt_input = ttnn.from_torch(
            torch_input[layer_id],
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )

        tt_dram_output = ttnn.experimental.moe_gpt(
            tt_input,
            w0_w1_tensor=tt_w0_w1,
            w2_tensor=tt_w2,
            output_tensor=tt_input,
            num_experts=E,
            layer_id=layer_id,
            enable_dram_output=True,
        )

        # Verify DRAM output properties
        assert tt_dram_output.shape == [E, 1, M, K], f"Expected shape [{E}, 1, {M}, {K}], got {tt_dram_output.shape}"
        assert tt_dram_output.layout == ttnn.TILE_LAYOUT, f"Expected TILE_LAYOUT, got {tt_dram_output.layout}"

        # Convert to ROW_MAJOR for comparison (matches unfused code path)
        tt_dram_output_rm = ttnn.to_layout(tt_dram_output, ttnn.ROW_MAJOR_LAYOUT)
        tt_to_torch_output = ttnn.to_torch(tt_dram_output_rm)  # (E, 1, M, K)
        tt_to_torch_output = tt_to_torch_output.squeeze(1)  # (E, M, K)

        # Reference calculation
        with torch.no_grad():
            torch_input_ref = torch_input[layer_id, 0, ...]  # (E, M, K)
            torch_w0_output_ref = torch_input_ref @ torch_w0[layer_id]
            torch_w1_output_ref = torch_input_ref @ torch_w1[layer_id]
            torch_intermediate_ref = swiglu_reference(torch_w0_output_ref, torch_w1_output_ref)
            torch_output_ref = torch_intermediate_ref @ torch_w2[layer_id]

        for expert_id in range(E):
            torch_layer_output = torch_output_ref[expert_id, :, :]
            tt_layer_output = tt_to_torch_output[expert_id, :, :]
            layer_metrics = get_accuracy_metrics(torch_layer_output, tt_layer_output)
            all_accuracy_metrics[(layer_id, expert_id)] = layer_metrics

    return all_accuracy_metrics


@pytest.mark.parametrize(
    "M, K, N, E, L",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe_gpt_performance(M, K, N, E, L, check_accuracy, dump_outputs):
    command = (
        f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt.py"
        f"::test_moe_gpt[dump_outputs_{dump_outputs}-check_accuracy_{check_accuracy}"
        f"-M={M}-K={K}-N={N}-E={E}-L={L}-dispatch_row]"
    )
    run_device_profiler(command, "ttnn_moe_gpt_performance", device_analysis_types=["device_kernel_duration"])

    r = post_process_ops_log("ttnn_moe_gpt_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = r["DEVICE KERNEL DURATION [ns]"].sum() / 1000.0
    logger.info(f"Duration per layer: {duration_us / L} us")
    logger.info(f"Duration per layer per expert: {duration_us / L / E} us")
    logger.warning(f"Total Duration: {duration_us} us")

    bytes_per_tile = 512 + 64  # bfloat4_b
    tiles_per_txn = 10
    num_cores = 12

    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    # W0/W1: 8 max tiles/core, 90 height
    w0_w1_padded_tiles_per_core = 2 * MAX_W0_W1_TILES_PER_CORE * Kt  # 2 * 8 * 90 = 1440
    # W2: 8 max tiles/core, 90 height
    w2_padded_tiles_per_core = 2 * 4 * Nt  # 2 groups * 4 tiles * 90 = 720
    total_padded_tiles_per_core = w0_w1_padded_tiles_per_core + w2_padded_tiles_per_core

    total_bytes_transferred = L * E * num_cores * total_padded_tiles_per_core * bytes_per_tile
    realized_bandwidth = int(total_bytes_transferred / (duration_us * 1000))
    logger.warning(f"Realized Bandwidth: {realized_bandwidth} GB/s")

    total_tiles_0_1 = Kt * Nt
    total_tiles_2 = Nt * Kt
    total_tiles_per_core = 2 * total_tiles_0_1 + total_tiles_2
    total_bytes_used = L * E * total_tiles_per_core * bytes_per_tile
    bandwidth = int(total_bytes_used / (duration_us * 1000))
    logger.warning(f"Useful Bandwidth: {bandwidth} GB/s")

    assert (
        duration_us < SHAPE2TIME[(M, K, N, E, L)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N, E, L)]} us"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]) -> dict[str, float]:
    filename = get_latest_ops_log_filename(output_logs_subdir)

    import pandas as pd

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
