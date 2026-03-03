#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill MoE compute kernel tests.

Tests BFP4_b weight matmul with custom kernels via generic_op + ProgramDescriptor.
Validates gate_up and down matmul patterns used in the MoE expert compute pipeline.

Run:
    cd /data/sraizada_2/tt-metal
    pytest tests/ttnn/unit_tests/operations/experimental/prefill_moe/test_prefill_moe_compute.py -v -s
"""

import pytest
import torch
import ttnn
from loguru import logger

# Tile dimensions
TILE = 32

# Tile sizes in bytes
BF16_TILE_BYTES = 2048  # 32 * 32 * 2
BFP4_TILE_BYTES = 576  # BFP4_b tile size

# Kernel file paths (relative to tt-metal root)
KERNEL_DIR = "tests/ttnn/unit_tests/operations/experimental/prefill_moe/kernels"
ACT_READER_KERNEL = f"{KERNEL_DIR}/activation_reader.cpp"
WEIGHT_RW_KERNEL = f"{KERNEL_DIR}/weight_reader_writer.cpp"
COMPUTE_KERNEL = f"{KERNEL_DIR}/compute_gate_up.cpp"
SWIGLU_COMPUTE_KERNEL = f"{KERNEL_DIR}/compute_gate_up_swiglu.cpp"
WEIGHT_SWIGLU_RW_KERNEL = f"{KERNEL_DIR}/weight_reader_swiglu_writer.cpp"
GENERIC_COMPUTE_KERNEL = f"{KERNEL_DIR}/compute_matmul.cpp"


def run_matmul_kernel(device, P, K, N, num_cores, grid_x, grid_y):
    """Run custom matmul kernel: [P, K] × [K, N] → [P, N] with BFP4_b weights.

    Returns (output_torch, reference) as float32 tensors.
    """
    torch.manual_seed(42)

    k_tiles = K // TILE
    n_tiles = N // TILE
    n_per_core = n_tiles // num_cores
    assert n_tiles % num_cores == 0, f"N_TILES={n_tiles} not divisible by {num_cores}"

    # Create test data
    act_torch = torch.randn(P, K, dtype=torch.bfloat16)
    weight_torch = torch.randn(K, N, dtype=torch.bfloat16)

    # Upload to device
    act_tensor = ttnn.from_torch(
        act_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weight_tensor = ttnn.from_torch(
        weight_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, P, N]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    io_tensors = [act_tensor, weight_tensor, output_tensor]

    # Core grid
    assert grid_x * grid_y == num_cores
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])

    # CB Descriptors
    cb0_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1_desc = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    # Compile-time args
    act_reader_ct_args = ttnn.TensorAccessorArgs(act_tensor).get_compile_time_args()

    weight_rw_ct_args = list(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    weight_rw_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [k_tiles, n_per_core]

    # Runtime args (per-core)
    act_reader_rt_args = ttnn.RuntimeArgs()
    weight_rw_rt_args = ttnn.RuntimeArgs()

    core_idx = 0
    for y in range(grid_y):
        for x in range(grid_x):
            core_n_offset = core_idx * n_per_core

            act_reader_rt_args[x][y] = [
                act_tensor.buffer_address(),
                k_tiles,
                0,
            ]

            weight_rw_rt_args[x][y] = [
                weight_tensor.buffer_address(),
                output_tensor.buffer_address(),
                k_tiles,
                n_per_core,
                n_tiles,
                core_n_offset,
                core_n_offset,  # out_start_tile
            ]

            core_idx += 1

    # Kernel Descriptors
    act_reader_kernel = ttnn.KernelDescriptor(
        kernel_source=ACT_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=act_reader_ct_args,
        runtime_args=act_reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    weight_rw_kernel = ttnn.KernelDescriptor(
        kernel_source=WEIGHT_RW_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=weight_rw_ct_args,
        runtime_args=weight_rw_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        ),
    )

    program_desc = ttnn.ProgramDescriptor(
        kernels=[act_reader_kernel, weight_rw_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb0_desc, cb1_desc, cb2_desc],
    )

    logger.info(
        f"Running matmul: [{P}, {K}] × [{K}, {N}] on {num_cores} cores (K_tiles={k_tiles}, N_per_core={n_per_core})"
    )

    output = ttnn.generic_op(io_tensors, program_desc)
    ttnn.synchronize_device(device)

    output_torch = ttnn.to_torch(output).squeeze().float()

    # Reference: use dequantized BFP4_b weights for fair comparison
    weight_dequant = ttnn.to_torch(weight_tensor).squeeze().float()
    reference = act_torch.float() @ weight_dequant

    return output_torch, reference, n_per_core


@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize(
    "K, N, num_cores, grid_x, grid_y, test_name",
    [
        # gate_up: [32, 2880] × [2880, 5760] → [32, 5760], 12 cores
        (2880, 5760, 12, 6, 2, "gate_up"),
        # down: [32, 2880] × [2880, 2880] → [32, 2880], 6 cores
        (2880, 2880, 6, 6, 1, "down"),
    ],
    ids=["gate_up_matmul", "down_matmul"],
)
def test_matmul(device, K, N, num_cores, grid_x, grid_y, test_name):
    """Test BFP4_b matmul kernel at GPT-OSS MoE dimensions."""
    P = 32

    output_torch, reference, n_per_core = run_matmul_kernel(device, P, K, N, num_cores, grid_x, grid_y)

    # Compute PCC
    pcc = torch.corrcoef(torch.stack([output_torch.flatten(), reference.flatten()]))[0, 1].item()

    logger.info(f"[{test_name}] PCC: {pcc:.6f}")
    logger.info(f"[{test_name}] Max abs error: {(output_torch - reference).abs().max().item():.4f}")
    logger.info(f"[{test_name}] Mean abs error: {(output_torch - reference).abs().mean().item():.4f}")

    # Per-core PCC
    for core_idx in range(num_cores):
        col_start = core_idx * n_per_core * TILE
        col_end = col_start + n_per_core * TILE
        core_out = output_torch[:, col_start:col_end].flatten()
        core_ref = reference[:, col_start:col_end].flatten()
        core_pcc = torch.corrcoef(torch.stack([core_out, core_ref]))[0, 1].item()
        logger.info(f"[{test_name}] Core {core_idx:2d} (cols {col_start:4d}-{col_end:4d}): PCC={core_pcc:.6f}")

    assert pcc >= 0.98, f"[{test_name}] PCC {pcc:.6f} < 0.98 threshold"
    logger.info(f"[{test_name}] PASSED")


def swiglu_torch(gate_up_output, d_ff):
    """GPT-OSS SwiGLU: (up + 1) * gate * sigmoid(alpha * gate).

    gate_up_output: [P, D_FF] where first D_FF/2 cols are gate, last D_FF/2 are up.
    Returns: [P, D_FF/2]
    """
    alpha = 1.702
    clamp_limit = 7.0
    half = d_ff // 2
    gate = gate_up_output[:, :half].float()
    up = gate_up_output[:, half:].float()
    gate_clamped = gate.clamp(-clamp_limit, clamp_limit)
    return (up + 1.0) * gate * torch.sigmoid(alpha * gate_clamped)


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_expert_pipeline(device):
    """End-to-end single-expert compute: gate_up → SwiGLU → down.

    SwiGLU is applied on host (torch) as a placeholder.
    This validates the full data path: activation → gate_up matmul → intermediate →
    SwiGLU → down matmul → expert output.
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2  # 2880
    P = 32

    torch.manual_seed(42)

    # Create weights
    gate_up_w_torch = torch.randn(D, D_FF, dtype=torch.bfloat16)
    down_w_torch = torch.randn(D_FF_HALF, D, dtype=torch.bfloat16)
    act_torch = torch.randn(P, D, dtype=torch.bfloat16)

    # === Step 1: gate_up matmul on device ===
    act_tensor = ttnn.from_torch(
        act_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gate_up_w_tensor = ttnn.from_torch(
        gate_up_w_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Use ttnn.matmul for gate_up (our custom kernel is proven equivalent)
    gate_up_out = ttnn.matmul(act_tensor, gate_up_w_tensor)
    ttnn.synchronize_device(device)

    gate_up_torch = ttnn.to_torch(gate_up_out).squeeze()  # [P, D_FF]
    logger.info(f"gate_up output shape: {gate_up_torch.shape}")

    # === Step 2: SwiGLU on host ===
    swiglu_out = swiglu_torch(gate_up_torch, D_FF)  # [P, D_FF/2] float32
    logger.info(f"SwiGLU output shape: {swiglu_out.shape}")

    # === Step 3: down matmul on device ===
    # Upload SwiGLU output as activation for down matmul
    swiglu_tensor = ttnn.from_torch(
        swiglu_out.bfloat16().unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_w_tensor = ttnn.from_torch(
        down_w_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run down matmul using our custom kernel (6 cores, same as test_matmul[down])
    down_out_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, P, D]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    num_cores = 6
    grid_x, grid_y = 6, 1
    k_tiles = D_FF_HALF // TILE  # 90
    n_tiles = D // TILE  # 90
    n_per_core = n_tiles // num_cores  # 15

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])

    cb0_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1_desc = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    act_reader_ct_args = ttnn.TensorAccessorArgs(swiglu_tensor).get_compile_time_args()
    weight_rw_ct_args = list(ttnn.TensorAccessorArgs(down_w_tensor).get_compile_time_args())
    weight_rw_ct_args.extend(ttnn.TensorAccessorArgs(down_out_tensor).get_compile_time_args())

    act_reader_rt_args = ttnn.RuntimeArgs()
    weight_rw_rt_args = ttnn.RuntimeArgs()
    core_idx = 0
    for y in range(grid_y):
        for x in range(grid_x):
            core_n_offset = core_idx * n_per_core
            act_reader_rt_args[x][y] = [
                swiglu_tensor.buffer_address(),
                k_tiles,
                0,
            ]
            weight_rw_rt_args[x][y] = [
                down_w_tensor.buffer_address(),
                down_out_tensor.buffer_address(),
                k_tiles,
                n_per_core,
                n_tiles,
                core_n_offset,
                core_n_offset,
            ]
            core_idx += 1

    program_desc = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=ACT_READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=act_reader_ct_args,
                runtime_args=act_reader_rt_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WEIGHT_RW_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=weight_rw_ct_args,
                runtime_args=weight_rw_rt_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=[k_tiles, n_per_core],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=True,
                ),
            ),
        ],
        semaphores=[],
        cbs=[cb0_desc, cb1_desc, cb2_desc],
    )

    logger.info(f"Running down matmul: [{P}, {D_FF_HALF}] × [{D_FF_HALF}, {D}] on {num_cores} cores")
    down_output = ttnn.generic_op([swiglu_tensor, down_w_tensor, down_out_tensor], program_desc)
    ttnn.synchronize_device(device)

    down_out_torch = ttnn.to_torch(down_output).squeeze().float()  # [P, D]
    logger.info(f"Expert output shape: {down_out_torch.shape}")

    # === Torch reference for full pipeline ===
    # Use dequantized weights for fair comparison
    gate_up_w_dequant = ttnn.to_torch(gate_up_w_tensor).squeeze().float()
    down_w_dequant = ttnn.to_torch(down_w_tensor).squeeze().float()

    ref_gate_up = act_torch.float() @ gate_up_w_dequant
    ref_swiglu = swiglu_torch(ref_gate_up.bfloat16(), D_FF)
    ref_down = ref_swiglu.bfloat16().float() @ down_w_dequant

    pcc = torch.corrcoef(torch.stack([down_out_torch.flatten(), ref_down.flatten()]))[0, 1].item()
    logger.info(f"[expert_pipeline] PCC: {pcc:.6f}")
    logger.info(f"[expert_pipeline] Max abs error: {(down_out_torch - ref_down).abs().max().item():.4f}")
    logger.info(f"[expert_pipeline] Mean abs error: {(down_out_torch - ref_down).abs().mean().item():.4f}")

    assert pcc >= 0.97, f"[expert_pipeline] PCC {pcc:.6f} < 0.97 threshold"
    logger.info("[expert_pipeline] PASSED: full gate_up → SwiGLU → down pipeline verified")


def swiglu_sfpu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """Matches SFPU swiglu_sfpu.h: calculate_swiglu.

    gate_clamped = clamp(gate, max=clamp_limit)          -- only max clamp
    up_clamped   = clamp(up, -clamp_limit, clamp_limit)  -- both sides
    result       = (up_clamped + 1) * gate_clamped * sigmoid(alpha * gate_clamped)
    """
    gate_c = gate.float().clamp(max=clamp_limit)
    up_c = up.float().clamp(-clamp_limit, clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_gate_up_swiglu(device):
    """Test gate_up matmul + on-device SwiGLU on 15 cores with pre-shuffled weights.

    Step 2.1 of Phase 2: validates that the compute kernel correctly performs
    gate_up matmul followed by SwiGLU on dest registers, producing [P, D_FF/2]
    intermediate output.

    Weights are pre-shuffled so each core's 12 columns = [6 gate, 6 up],
    enabling per-core SwiGLU without cross-core data exchange.
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2  # 2880
    P = 32
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles = D // TILE  # 90
    n_weight_tiles = D_FF // TILE  # 180
    n_weight_per_core = n_weight_tiles // NUM_CORES  # 12
    n_out_tiles = D_FF_HALF // TILE  # 90
    n_out_per_core = n_out_tiles // NUM_CORES  # 6

    torch.manual_seed(42)

    # Create test data
    act_torch = torch.randn(P, D, dtype=torch.bfloat16)
    gate_up_w = torch.randn(D, D_FF, dtype=torch.bfloat16)

    # Pre-shuffle weights: interleave gate/up columns per core
    # Original: cols 0..D_FF/2-1 = gate, cols D_FF/2..D_FF-1 = up
    # Shuffled: each core's 12 tiles = [6 gate tiles, 6 up tiles]
    gate_cols = gate_up_w[:, :D_FF_HALF]  # [D, 2880]
    up_cols = gate_up_w[:, D_FF_HALF:]  # [D, 2880]

    cols_per_core = n_weight_per_core * TILE  # 384
    half_cols = n_out_per_core * TILE  # 192

    shuffled_w = torch.empty_like(gate_up_w)
    for c in range(NUM_CORES):
        shuffled_w[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[
            :, c * half_cols : (c + 1) * half_cols
        ]
        shuffled_w[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
            :, c * half_cols : (c + 1) * half_cols
        ]

    # Upload to device
    act_tensor = ttnn.from_torch(
        act_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shuffled_w_tensor = ttnn.from_torch(
        shuffled_w.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, P, D_FF_HALF]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    io_tensors = [act_tensor, shuffled_w_tensor, output_tensor]

    # Core grid
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    # CB Descriptors
    # CB0: 1 BF16 activation tile
    cb0_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    # CB1: 12 BFP4_b weight tiles per K iteration
    cb1_desc = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_weight_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    # CB2: 6 BF16 SwiGLU output tiles
    cb2_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_out_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    # Compile-time args
    act_reader_ct_args = ttnn.TensorAccessorArgs(act_tensor).get_compile_time_args()

    weight_rw_ct_args = list(ttnn.TensorAccessorArgs(shuffled_w_tensor).get_compile_time_args())
    weight_rw_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [k_tiles, n_weight_per_core]  # [90, 12]

    # Runtime args (per-core)
    act_reader_rt_args = ttnn.RuntimeArgs()
    weight_rw_rt_args = ttnn.RuntimeArgs()

    core_idx = 0
    for y in range(GRID_Y):
        for x in range(GRID_X):
            act_reader_rt_args[x][y] = [
                act_tensor.buffer_address(),
                k_tiles,
                0,  # act_start_tile (same activation row for all cores)
            ]

            weight_rw_rt_args[x][y] = [
                shuffled_w_tensor.buffer_address(),
                output_tensor.buffer_address(),
                k_tiles,  # num_k_tiles = 90
                n_weight_per_core,  # n_weight_tiles = 12
                n_weight_tiles,  # weight_n_total = 180
                core_idx * n_weight_per_core,  # core_weight_offset
                core_idx * n_out_per_core,  # out_start_tile
                n_out_per_core,  # n_output_tiles = 6
            ]

            core_idx += 1

    # Kernel Descriptors
    act_reader_kernel = ttnn.KernelDescriptor(
        kernel_source=ACT_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=act_reader_ct_args,
        runtime_args=act_reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    weight_rw_kernel = ttnn.KernelDescriptor(
        kernel_source=WEIGHT_SWIGLU_RW_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=weight_rw_ct_args,
        runtime_args=weight_rw_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=SWIGLU_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        ),
    )

    program_desc = ttnn.ProgramDescriptor(
        kernels=[act_reader_kernel, weight_rw_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb0_desc, cb1_desc, cb2_desc],
    )

    logger.info(
        f"Running gate_up+SwiGLU: [{P}, {D}] × [{D}, {D_FF}] → SwiGLU → [{P}, {D_FF_HALF}] "
        f"on {NUM_CORES} cores (K_tiles={k_tiles}, N_weight/core={n_weight_per_core}, N_out/core={n_out_per_core})"
    )

    output = ttnn.generic_op(io_tensors, program_desc)
    ttnn.synchronize_device(device)

    output_torch = ttnn.to_torch(output).squeeze().float()  # [P, D_FF_HALF]
    logger.info(f"Output shape: {output_torch.shape}")

    # === Reference: dequantized shuffled weights → matmul → per-core SwiGLU ===
    w_dequant = ttnn.to_torch(shuffled_w_tensor).squeeze().float()
    ref_gate_up = act_torch.float() @ w_dequant  # [P, D_FF] in shuffled column order

    # Per-core SwiGLU on shuffled result
    ref_output = torch.empty(P, D_FF_HALF, dtype=torch.float32)
    for c in range(NUM_CORES):
        gate_part = ref_gate_up[:, c * cols_per_core : c * cols_per_core + half_cols]
        up_part = ref_gate_up[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
        ref_output[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(gate_part, up_part)

    # Overall PCC
    pcc = torch.corrcoef(torch.stack([output_torch.flatten(), ref_output.flatten()]))[0, 1].item()
    max_err = (output_torch - ref_output).abs().max().item()
    mean_err = (output_torch - ref_output).abs().mean().item()
    logger.info(f"[gate_up_swiglu] PCC: {pcc:.6f}")
    logger.info(f"[gate_up_swiglu] Max abs error: {max_err:.4f}")
    logger.info(f"[gate_up_swiglu] Mean abs error: {mean_err:.4f}")

    # Per-core PCC
    for c in range(NUM_CORES):
        col_start = c * half_cols
        col_end = col_start + half_cols
        core_out = output_torch[:, col_start:col_end].flatten()
        core_ref = ref_output[:, col_start:col_end].flatten()
        core_pcc = torch.corrcoef(torch.stack([core_out, core_ref]))[0, 1].item()
        logger.info(f"[gate_up_swiglu] Core {c:2d} (cols {col_start:4d}-{col_end:4d}): PCC={core_pcc:.6f}")

    assert pcc >= 0.99, f"[gate_up_swiglu] PCC {pcc:.6f} < 0.99 threshold"
    logger.info("[gate_up_swiglu] PASSED: gate_up matmul + SwiGLU verified")


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_expert_full_pipeline(device):
    """End-to-end single-expert: gate_up → SwiGLU(SFPU) → DRAM → down matmul.

    Step 2.2 of Phase 2: two separate programs (implicit DRAM barrier):
      Program 1: activation × gate_up_weights → SwiGLU → inter_tensor [1,1,P,D_FF/2]
      Program 2: inter_tensor × down_weights → output [1,1,P,D]

    All on 15 cores. PCC target ≥ 0.985 (two BFP4_b matmuls + SwiGLU nonlinearity).
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2  # 2880
    P = 32
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles_gate_up = D // TILE  # 90
    n_weight_tiles_gate_up = D_FF // TILE  # 180
    n_weight_per_core_gate_up = n_weight_tiles_gate_up // NUM_CORES  # 12
    n_out_per_core_gate_up = n_weight_per_core_gate_up // 2  # 6

    k_tiles_down = D_FF_HALF // TILE  # 90
    n_tiles_down = D // TILE  # 90
    n_per_core_down = n_tiles_down // NUM_CORES  # 6

    torch.manual_seed(42)

    # Create test data
    act_torch = torch.randn(P, D, dtype=torch.bfloat16)
    gate_up_w = torch.randn(D, D_FF, dtype=torch.bfloat16)
    down_w = torch.randn(D_FF_HALF, D, dtype=torch.bfloat16)

    # Pre-shuffle gate_up weights
    gate_cols = gate_up_w[:, :D_FF_HALF]
    up_cols = gate_up_w[:, D_FF_HALF:]
    cols_per_core = n_weight_per_core_gate_up * TILE  # 384
    half_cols = n_out_per_core_gate_up * TILE  # 192

    shuffled_w = torch.empty_like(gate_up_w)
    for c in range(NUM_CORES):
        shuffled_w[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[
            :, c * half_cols : (c + 1) * half_cols
        ]
        shuffled_w[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
            :, c * half_cols : (c + 1) * half_cols
        ]

    # Upload all tensors to device
    act_tensor = ttnn.from_torch(
        act_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shuffled_w_tensor = ttnn.from_torch(
        shuffled_w.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_w_tensor = ttnn.from_torch(
        down_w.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inter_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, P, D_FF_HALF]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, P, D]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    # ========== Program 1: gate_up + SwiGLU → inter_tensor ==========
    logger.info("Program 1: gate_up + SwiGLU")

    cb0_p1 = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1_p1 = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_weight_per_core_gate_up,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2_p1 = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_out_per_core_gate_up,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    act_ct_p1 = ttnn.TensorAccessorArgs(act_tensor).get_compile_time_args()
    wrw_ct_p1 = list(ttnn.TensorAccessorArgs(shuffled_w_tensor).get_compile_time_args())
    wrw_ct_p1.extend(ttnn.TensorAccessorArgs(inter_tensor).get_compile_time_args())

    act_rt_p1 = ttnn.RuntimeArgs()
    wrw_rt_p1 = ttnn.RuntimeArgs()
    core_idx = 0
    for y in range(GRID_Y):
        for x in range(GRID_X):
            act_rt_p1[x][y] = [act_tensor.buffer_address(), k_tiles_gate_up, 0]
            wrw_rt_p1[x][y] = [
                shuffled_w_tensor.buffer_address(),
                inter_tensor.buffer_address(),
                k_tiles_gate_up,
                n_weight_per_core_gate_up,
                n_weight_tiles_gate_up,
                core_idx * n_weight_per_core_gate_up,
                core_idx * n_out_per_core_gate_up,
                n_out_per_core_gate_up,
            ]
            core_idx += 1

    prog1 = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=ACT_READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=act_ct_p1,
                runtime_args=act_rt_p1,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WEIGHT_SWIGLU_RW_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=wrw_ct_p1,
                runtime_args=wrw_rt_p1,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=SWIGLU_COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=[k_tiles_gate_up, n_weight_per_core_gate_up],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=True,
                ),
            ),
        ],
        semaphores=[],
        cbs=[cb0_p1, cb1_p1, cb2_p1],
    )

    inter_result = ttnn.generic_op([act_tensor, shuffled_w_tensor, inter_tensor], prog1)
    ttnn.synchronize_device(device)
    logger.info(f"Intermediate shape: {ttnn.to_torch(inter_result).squeeze().shape}")

    # ========== Program 2: inter_tensor × down_weights → output ==========
    logger.info("Program 2: down matmul")

    cb0_p2 = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1_p2 = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_per_core_down,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2_p2 = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_per_core_down,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    act_ct_p2 = ttnn.TensorAccessorArgs(inter_result).get_compile_time_args()
    wrw_ct_p2 = list(ttnn.TensorAccessorArgs(down_w_tensor).get_compile_time_args())
    wrw_ct_p2.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    act_rt_p2 = ttnn.RuntimeArgs()
    wrw_rt_p2 = ttnn.RuntimeArgs()
    core_idx = 0
    for y in range(GRID_Y):
        for x in range(GRID_X):
            act_rt_p2[x][y] = [inter_result.buffer_address(), k_tiles_down, 0]
            wrw_rt_p2[x][y] = [
                down_w_tensor.buffer_address(),
                output_tensor.buffer_address(),
                k_tiles_down,
                n_per_core_down,
                n_tiles_down,
                core_idx * n_per_core_down,
                core_idx * n_per_core_down,
            ]
            core_idx += 1

    prog2 = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=ACT_READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=act_ct_p2,
                runtime_args=act_rt_p2,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WEIGHT_RW_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=wrw_ct_p2,
                runtime_args=wrw_rt_p2,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=GENERIC_COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=[k_tiles_down, n_per_core_down, 6],  # N_BLOCK=6
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=True,
                ),
            ),
        ],
        semaphores=[],
        cbs=[cb0_p2, cb1_p2, cb2_p2],
    )

    final_output = ttnn.generic_op([inter_result, down_w_tensor, output_tensor], prog2)
    ttnn.synchronize_device(device)

    output_torch = ttnn.to_torch(final_output).squeeze().float()  # [P, D]
    logger.info(f"Expert output shape: {output_torch.shape}")

    # ========== Reference: full pipeline with dequantized weights ==========
    w_gate_up_dequant = ttnn.to_torch(shuffled_w_tensor).squeeze().float()
    w_down_dequant = ttnn.to_torch(down_w_tensor).squeeze().float()

    # gate_up matmul
    ref_gate_up = act_torch.float() @ w_gate_up_dequant  # [P, D_FF] shuffled

    # Per-core SwiGLU
    ref_inter = torch.empty(P, D_FF_HALF, dtype=torch.float32)
    for c in range(NUM_CORES):
        gate_part = ref_gate_up[:, c * cols_per_core : c * cols_per_core + half_cols]
        up_part = ref_gate_up[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
        ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(gate_part, up_part)

    # Simulate BF16 DRAM roundtrip
    ref_inter_bf16 = ref_inter.bfloat16().float()

    # down matmul
    ref_output = ref_inter_bf16 @ w_down_dequant

    # Compare
    pcc = torch.corrcoef(torch.stack([output_torch.flatten(), ref_output.flatten()]))[0, 1].item()
    max_err = (output_torch - ref_output).abs().max().item()
    mean_err = (output_torch - ref_output).abs().mean().item()
    logger.info(f"[expert_full_pipeline] PCC: {pcc:.6f}")
    logger.info(f"[expert_full_pipeline] Max abs error: {max_err:.4f}")
    logger.info(f"[expert_full_pipeline] Mean abs error: {mean_err:.4f}")

    # Also check intermediate PCC
    inter_torch = ttnn.to_torch(inter_result).squeeze().float()
    inter_pcc = torch.corrcoef(torch.stack([inter_torch.flatten(), ref_inter.flatten()]))[0, 1].item()
    logger.info(f"[expert_full_pipeline] Intermediate PCC: {inter_pcc:.6f}")

    assert pcc >= 0.985, f"[expert_full_pipeline] PCC {pcc:.6f} < 0.985 threshold"
    logger.info("[expert_full_pipeline] PASSED: full gate_up → SwiGLU → down pipeline verified")


# ===== Helper functions for multi-expert tests =====


def make_gate_up_swiglu_program(
    act_tensor,
    weight_tensor,
    out_tensor,
    core_grid,
    num_cores,
    grid_x,
    grid_y,
    k_tiles,
    n_weight_per_core,
    n_weight_tiles,
    n_out_per_core,
):
    """Create ProgramDescriptor for gate_up matmul + SwiGLU → output."""
    cb0 = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1 = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_weight_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2 = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_out_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    act_ct = ttnn.TensorAccessorArgs(act_tensor).get_compile_time_args()
    wrw_ct = list(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    wrw_ct.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())

    act_rt = ttnn.RuntimeArgs()
    wrw_rt = ttnn.RuntimeArgs()
    idx = 0
    for y in range(grid_y):
        for x in range(grid_x):
            act_rt[x][y] = [act_tensor.buffer_address(), k_tiles, 0]
            wrw_rt[x][y] = [
                weight_tensor.buffer_address(),
                out_tensor.buffer_address(),
                k_tiles,
                n_weight_per_core,
                n_weight_tiles,
                idx * n_weight_per_core,
                idx * n_out_per_core,
                n_out_per_core,
            ]
            idx += 1

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=ACT_READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=act_ct,
                runtime_args=act_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WEIGHT_SWIGLU_RW_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=wrw_ct,
                runtime_args=wrw_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=SWIGLU_COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=[k_tiles, n_weight_per_core],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=True,
                ),
            ),
        ],
        semaphores=[],
        cbs=[cb0, cb1, cb2],
    )


def make_down_matmul_program(
    act_tensor, weight_tensor, out_tensor, core_grid, num_cores, grid_x, grid_y, k_tiles, n_per_core, n_tiles
):
    """Create ProgramDescriptor for down matmul."""
    cb0 = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1 = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2 = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    act_ct = ttnn.TensorAccessorArgs(act_tensor).get_compile_time_args()
    wrw_ct = list(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    wrw_ct.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())

    act_rt = ttnn.RuntimeArgs()
    wrw_rt = ttnn.RuntimeArgs()
    idx = 0
    for y in range(grid_y):
        for x in range(grid_x):
            act_rt[x][y] = [act_tensor.buffer_address(), k_tiles, 0]
            wrw_rt[x][y] = [
                weight_tensor.buffer_address(),
                out_tensor.buffer_address(),
                k_tiles,
                n_per_core,
                n_tiles,
                idx * n_per_core,
                idx * n_per_core,
            ]
            idx += 1

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=ACT_READER_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=act_ct,
                runtime_args=act_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WEIGHT_RW_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=wrw_ct,
                runtime_args=wrw_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=GENERIC_COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=[k_tiles, n_per_core, n_per_core],  # N_BLOCK = n_per_core
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=True,
                ),
            ),
        ],
        semaphores=[],
        cbs=[cb0, cb1, cb2],
    )


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_multi_expert_pipeline(device):
    """Test 4-expert sequential pipeline: each expert runs gate_up+SwiGLU→down.

    Step 2.4 of Phase 2: validates that 4 experts can run sequentially on the same
    15-core grid, each producing correct independent outputs. Uses 2 programs per
    expert (8 total program launches).
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2
    P = 32
    NUM_EXPERTS = 4
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles_gu = D // TILE  # 90
    n_w_tiles_gu = D_FF // TILE  # 180
    n_w_per_core_gu = n_w_tiles_gu // NUM_CORES  # 12
    n_out_per_core_gu = n_w_per_core_gu // 2  # 6

    k_tiles_dn = D_FF_HALF // TILE  # 90
    n_tiles_dn = D // TILE  # 90
    n_per_core_dn = n_tiles_dn // NUM_CORES  # 6

    cols_per_core = n_w_per_core_gu * TILE  # 384
    half_cols = n_out_per_core_gu * TILE  # 192

    torch.manual_seed(42)

    # Create per-expert weights
    act_torch = torch.randn(P, D, dtype=torch.bfloat16)
    gate_up_ws = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]
    down_ws = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]

    # Pre-shuffle gate_up weights per expert
    shuffled_ws = []
    for w in gate_up_ws:
        gate_cols = w[:, :D_FF_HALF]
        up_cols = w[:, D_FF_HALF:]
        s = torch.empty_like(w)
        for c in range(NUM_CORES):
            s[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[:, c * half_cols : (c + 1) * half_cols]
            s[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
                :, c * half_cols : (c + 1) * half_cols
            ]
        shuffled_ws.append(s)

    # Upload shared activation
    act_tensor = ttnn.from_torch(
        act_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Upload per-expert weights
    gu_w_tensors = [
        ttnn.from_torch(
            s.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in shuffled_ws
    ]
    dn_w_tensors = [
        ttnn.from_torch(
            w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in down_ws
    ]

    # Allocate per-expert intermediate and output buffers
    inter_tensors = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D_FF_HALF]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(NUM_EXPERTS)
    ]
    out_tensors = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(NUM_EXPERTS)
    ]

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    # Run 4 experts sequentially
    expert_outputs = []
    for i in range(NUM_EXPERTS):
        logger.info(f"Expert {i}: gate_up + SwiGLU")
        prog1 = make_gate_up_swiglu_program(
            act_tensor,
            gu_w_tensors[i],
            inter_tensors[i],
            core_grid,
            NUM_CORES,
            GRID_X,
            GRID_Y,
            k_tiles_gu,
            n_w_per_core_gu,
            n_w_tiles_gu,
            n_out_per_core_gu,
        )
        inter_result = ttnn.generic_op([act_tensor, gu_w_tensors[i], inter_tensors[i]], prog1)
        ttnn.synchronize_device(device)

        logger.info(f"Expert {i}: down matmul")
        prog2 = make_down_matmul_program(
            inter_result,
            dn_w_tensors[i],
            out_tensors[i],
            core_grid,
            NUM_CORES,
            GRID_X,
            GRID_Y,
            k_tiles_dn,
            n_per_core_dn,
            n_tiles_dn,
        )
        expert_out = ttnn.generic_op([inter_result, dn_w_tensors[i], out_tensors[i]], prog2)
        ttnn.synchronize_device(device)
        expert_outputs.append(expert_out)

    # Compute references and compare
    all_passed = True
    for i in range(NUM_EXPERTS):
        out_torch = ttnn.to_torch(expert_outputs[i]).squeeze().float()
        w_gu_dq = ttnn.to_torch(gu_w_tensors[i]).squeeze().float()
        w_dn_dq = ttnn.to_torch(dn_w_tensors[i]).squeeze().float()

        # gate_up → per-core SwiGLU → bf16 → down
        ref_gu = act_torch.float() @ w_gu_dq
        ref_inter = torch.empty(P, D_FF_HALF, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ w_dn_dq

        pcc = torch.corrcoef(torch.stack([out_torch.flatten(), ref_out.flatten()]))[0, 1].item()
        logger.info(f"[multi_expert] Expert {i}: PCC={pcc:.6f}")

        if pcc < 0.985:
            logger.error(f"[multi_expert] Expert {i} FAILED: PCC {pcc:.6f} < 0.985")
            all_passed = False

    assert all_passed, "One or more experts failed PCC threshold"
    logger.info("[multi_expert] PASSED: all 4 experts verified")


# ===== Phase 3: Single-program expert compute with cross-core barrier =====

EXPERT_COMPUTE_KERNEL = f"{KERNEL_DIR}/compute_expert.cpp"
EXPERT_READER_KERNEL = f"{KERNEL_DIR}/expert_reader.cpp"
EXPERT_WRITER_KERNEL = f"{KERNEL_DIR}/expert_writer.cpp"


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_single_program_expert(device):
    """Test single-program expert: gate_up+SwiGLU → barrier → down in one program.

    Step 3.1 of Phase 3: validates the cross-core L1 semaphore barrier between
    gate_up+SwiGLU (Phase A) and down matmul (Phase B). All 15 cores write their
    SwiGLU shards to DRAM, barrier synchronizes, then all cores read the full
    intermediate for down matmul.

    PCC target ≥ 0.985 (should match Phase 2 two-program pipeline).
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2  # 2880
    P = 32
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles = D // TILE  # 90 (same for both phases)
    n_weight_tiles_gu = D_FF // TILE  # 180
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES  # 12
    n_out_per_core = n_weight_per_core_gu // 2  # 6 (SwiGLU output)
    n_tiles_dn = D // TILE  # 90
    n_per_core_dn = n_tiles_dn // NUM_CORES  # 6

    cols_per_core = n_weight_per_core_gu * TILE  # 384
    half_cols = n_out_per_core * TILE  # 192

    torch.manual_seed(42)

    # Create test data
    act_torch = torch.randn(P, D, dtype=torch.bfloat16)
    gate_up_w = torch.randn(D, D_FF, dtype=torch.bfloat16)
    down_w = torch.randn(D_FF_HALF, D, dtype=torch.bfloat16)

    # Pre-shuffle gate_up weights: each core's 12 tiles = [6 gate, 6 up]
    gate_cols = gate_up_w[:, :D_FF_HALF]
    up_cols = gate_up_w[:, D_FF_HALF:]
    shuffled_w = torch.empty_like(gate_up_w)
    for c in range(NUM_CORES):
        shuffled_w[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[
            :, c * half_cols : (c + 1) * half_cols
        ]
        shuffled_w[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
            :, c * half_cols : (c + 1) * half_cols
        ]

    # Upload to device
    act_tensor = ttnn.from_torch(
        act_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shuffled_w_tensor = ttnn.from_torch(
        shuffled_w.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_w_tensor = ttnn.from_torch(
        down_w.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inter_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, P, D_FF_HALF]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, P, D]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    io_tensors = [act_tensor, shuffled_w_tensor, down_w_tensor, inter_tensor, output_tensor]

    # Core grid
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    # Get physical core coordinates for NOC semaphore addressing
    phys_coords = []  # [(x0,y0), (x1,y1), ...]
    for y in range(GRID_Y):
        for x in range(GRID_X):
            phys = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            phys_coords.append((phys.x, phys.y))

    # Leader is core (0,0) — index 0
    leader_phys_x, leader_phys_y = phys_coords[0]
    logger.info(f"Leader physical coords: ({leader_phys_x}, {leader_phys_y})")
    for i, (px, py) in enumerate(phys_coords):
        logger.info(f"Core {i:2d} logical=({i % GRID_X}, {i // GRID_X}) physical=({px}, {py})")

    # CB Descriptors
    cb0_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1_desc = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_weight_per_core_gu,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_out_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    # Semaphore Descriptors
    sem_barrier = ttnn.SemaphoreDescriptor(
        id=0,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=core_grid,
        initial_value=0,
    )
    sem_go = ttnn.SemaphoreDescriptor(
        id=1,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=core_grid,
        initial_value=0,
    )

    # Compile-time args
    reader_ct_args = list(ttnn.TensorAccessorArgs(act_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(inter_tensor).get_compile_time_args())

    writer_ct_args = list(ttnn.TensorAccessorArgs(shuffled_w_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(inter_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(down_w_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [k_tiles, n_weight_per_core_gu, n_per_core_dn]

    # Runtime args (per-core)
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()

    # Flatten physical coords for runtime args
    flat_phys = []
    for px, py in phys_coords:
        flat_phys.extend([px, py])

    core_idx = 0
    for y in range(GRID_Y):
        for x in range(GRID_X):
            is_leader = 1 if core_idx == 0 else 0

            reader_rt_args[x][y] = [
                act_tensor.buffer_address(),
                k_tiles,
                inter_tensor.buffer_address(),
                is_leader,
                NUM_CORES,
            ] + flat_phys

            writer_rt_args[x][y] = [
                shuffled_w_tensor.buffer_address(),  # [0] gate_up_w_addr
                inter_tensor.buffer_address(),  # [1] inter_write_addr
                down_w_tensor.buffer_address(),  # [2] down_w_addr
                output_tensor.buffer_address(),  # [3] output_addr
                k_tiles,  # [4]
                n_weight_per_core_gu,  # [5] 12
                n_weight_tiles_gu,  # [6] 180
                core_idx * n_weight_per_core_gu,  # [7] core_weight_offset_gu
                core_idx * n_out_per_core,  # [8] core_out_offset_gu
                n_out_per_core,  # [9] 6
                n_per_core_dn,  # [10] 6
                n_tiles_dn,  # [11] 90
                core_idx * n_per_core_dn,  # [12] core_dn_offset
                leader_phys_x,  # [13]
                leader_phys_y,  # [14]
            ]

            core_idx += 1

    # Kernel Descriptors
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        ),
    )

    program_desc = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[sem_barrier, sem_go],
        cbs=[cb0_desc, cb1_desc, cb2_desc],
    )

    logger.info(f"Running single-program expert: gate_up+SwiGLU+barrier+down on {NUM_CORES} cores")

    output = ttnn.generic_op(io_tensors, program_desc)
    ttnn.synchronize_device(device)

    output_torch = ttnn.to_torch(output).squeeze().float()  # [P, D]
    logger.info(f"Expert output shape: {output_torch.shape}")

    # Also read intermediate for debugging
    inter_torch = ttnn.to_torch(inter_tensor).squeeze().float()  # [P, D_FF_HALF]
    logger.info(f"Intermediate shape: {inter_torch.shape}")

    # ========== Reference ==========
    w_gu_dq = ttnn.to_torch(shuffled_w_tensor).squeeze().float()
    w_dn_dq = ttnn.to_torch(down_w_tensor).squeeze().float()

    # gate_up matmul
    ref_gu = act_torch.float() @ w_gu_dq  # [P, D_FF] shuffled

    # Per-core SwiGLU
    ref_inter = torch.empty(P, D_FF_HALF, dtype=torch.float32)
    for c in range(NUM_CORES):
        g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
        u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
        ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)

    # Simulate BF16 DRAM roundtrip
    ref_inter_bf16 = ref_inter.bfloat16().float()

    # down matmul
    ref_output = ref_inter_bf16 @ w_dn_dq

    # Intermediate PCC
    inter_pcc = torch.corrcoef(torch.stack([inter_torch.flatten(), ref_inter.flatten()]))[0, 1].item()
    logger.info(f"[single_program_expert] Intermediate PCC: {inter_pcc:.6f}")

    # Final output PCC
    pcc = torch.corrcoef(torch.stack([output_torch.flatten(), ref_output.flatten()]))[0, 1].item()
    max_err = (output_torch - ref_output).abs().max().item()
    mean_err = (output_torch - ref_output).abs().mean().item()
    logger.info(f"[single_program_expert] PCC: {pcc:.6f}")
    logger.info(f"[single_program_expert] Max abs error: {max_err:.4f}")
    logger.info(f"[single_program_expert] Mean abs error: {mean_err:.4f}")

    assert inter_pcc >= 0.99, f"[single_program_expert] Intermediate PCC {inter_pcc:.6f} < 0.99"
    assert pcc >= 0.985, f"[single_program_expert] PCC {pcc:.6f} < 0.985 threshold"
    logger.info("[single_program_expert] PASSED: single-program gate_up→SwiGLU→barrier→down verified")


def _make_single_program_expert(
    device,
    act_tensor,
    shuffled_w_tensor,
    down_w_tensor,
    inter_tensor,
    output_tensor,
    core_grid,
    phys_coords,
    leader_phys_x,
    leader_phys_y,
    NUM_CORES,
    GRID_X,
    GRID_Y,
    k_tiles,
    n_weight_per_core_gu,
    n_weight_tiles_gu,
    n_out_per_core,
    n_per_core_dn,
    n_tiles_dn,
):
    """Create a ProgramDescriptor for single-program expert (gate_up+SwiGLU+barrier+down)."""
    # Compile-time args
    reader_ct_args = list(ttnn.TensorAccessorArgs(act_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(inter_tensor).get_compile_time_args())

    writer_ct_args = list(ttnn.TensorAccessorArgs(shuffled_w_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(inter_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(down_w_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [k_tiles, n_weight_per_core_gu, n_per_core_dn]

    # Runtime args
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()

    flat_phys = []
    for px, py in phys_coords:
        flat_phys.extend([px, py])

    core_idx = 0
    for y in range(GRID_Y):
        for x in range(GRID_X):
            is_leader = 1 if core_idx == 0 else 0

            reader_rt_args[x][y] = [
                act_tensor.buffer_address(),
                k_tiles,
                inter_tensor.buffer_address(),
                is_leader,
                NUM_CORES,
            ] + flat_phys

            writer_rt_args[x][y] = [
                shuffled_w_tensor.buffer_address(),
                inter_tensor.buffer_address(),
                down_w_tensor.buffer_address(),
                output_tensor.buffer_address(),
                k_tiles,
                n_weight_per_core_gu,
                n_weight_tiles_gu,
                core_idx * n_weight_per_core_gu,
                core_idx * n_out_per_core,
                n_out_per_core,
                n_per_core_dn,
                n_tiles_dn,
                core_idx * n_per_core_dn,
                leader_phys_x,
                leader_phys_y,
            ]

            core_idx += 1

    # Semaphores
    sem_barrier = ttnn.SemaphoreDescriptor(
        id=0,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=core_grid,
        initial_value=0,
    )
    sem_go = ttnn.SemaphoreDescriptor(
        id=1,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=core_grid,
        initial_value=0,
    )

    # CB Descriptors
    cb0_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1_desc = ttnn.CBDescriptor(
        total_size=BFP4_TILE_BYTES * n_weight_per_core_gu,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES * n_out_per_core,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    # Kernel Descriptors
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[sem_barrier, sem_go],
        cbs=[cb0_desc, cb1_desc, cb2_desc],
    )


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_single_program_multi_expert(device):
    """Test 4 experts using single-program approach: 4 program launches (vs 8 in Phase 2).

    Step 3.2 of Phase 3: validates that 4 experts run correctly with the single-program
    kernels (gate_up+SwiGLU → barrier → down). Each expert uses a separate program launch
    but only 1 program per expert (vs 2 in Phase 2).

    PCC target: >= 0.985 per expert (should match Phase 2 results).
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2
    P = 32
    NUM_EXPERTS = 4
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles = D // TILE  # 90
    n_weight_tiles_gu = D_FF // TILE  # 180
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES  # 12
    n_out_per_core = n_weight_per_core_gu // 2  # 6
    n_tiles_dn = D // TILE  # 90
    n_per_core_dn = n_tiles_dn // NUM_CORES  # 6

    cols_per_core = n_weight_per_core_gu * TILE  # 384
    half_cols = n_out_per_core * TILE  # 192

    torch.manual_seed(42)

    # Create per-expert data
    act_torch = torch.randn(P, D, dtype=torch.bfloat16)
    gate_up_ws = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]
    down_ws = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]

    # Pre-shuffle gate_up weights per expert
    shuffled_ws = []
    for w in gate_up_ws:
        gate_cols = w[:, :D_FF_HALF]
        up_cols = w[:, D_FF_HALF:]
        s = torch.empty_like(w)
        for c in range(NUM_CORES):
            s[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[:, c * half_cols : (c + 1) * half_cols]
            s[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
                :, c * half_cols : (c + 1) * half_cols
            ]
        shuffled_ws.append(s)

    # Upload shared activation
    act_tensor = ttnn.from_torch(
        act_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Upload per-expert weights
    gu_w_tensors = [
        ttnn.from_torch(
            s.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in shuffled_ws
    ]
    dn_w_tensors = [
        ttnn.from_torch(
            w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in down_ws
    ]

    # Per-expert intermediate and output buffers
    inter_tensors = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D_FF_HALF]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(NUM_EXPERTS)
    ]
    out_tensors = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(NUM_EXPERTS)
    ]

    # Core grid and physical coords
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    phys_coords = []
    for y in range(GRID_Y):
        for x in range(GRID_X):
            phys = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            phys_coords.append((phys.x, phys.y))

    leader_phys_x, leader_phys_y = phys_coords[0]

    # Run 4 experts sequentially, each with a single program
    expert_outputs = []
    for i in range(NUM_EXPERTS):
        logger.info(f"Expert {i}: single-program gate_up+SwiGLU+barrier+down")
        prog = _make_single_program_expert(
            device,
            act_tensor,
            gu_w_tensors[i],
            dn_w_tensors[i],
            inter_tensors[i],
            out_tensors[i],
            core_grid,
            phys_coords,
            leader_phys_x,
            leader_phys_y,
            NUM_CORES,
            GRID_X,
            GRID_Y,
            k_tiles,
            n_weight_per_core_gu,
            n_weight_tiles_gu,
            n_out_per_core,
            n_per_core_dn,
            n_tiles_dn,
        )
        io_tensors = [act_tensor, gu_w_tensors[i], dn_w_tensors[i], inter_tensors[i], out_tensors[i]]
        expert_out = ttnn.generic_op(io_tensors, prog)
        ttnn.synchronize_device(device)
        expert_outputs.append(expert_out)

    # Verify each expert
    all_passed = True
    for i in range(NUM_EXPERTS):
        out_torch = ttnn.to_torch(expert_outputs[i]).squeeze().float()
        w_gu_dq = ttnn.to_torch(gu_w_tensors[i]).squeeze().float()
        w_dn_dq = ttnn.to_torch(dn_w_tensors[i]).squeeze().float()

        # Reference: gate_up → per-core SwiGLU → bf16 roundtrip → down
        ref_gu = act_torch.float() @ w_gu_dq
        ref_inter = torch.empty(P, D_FF_HALF, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ w_dn_dq

        pcc = torch.corrcoef(torch.stack([out_torch.flatten(), ref_out.flatten()]))[0, 1].item()
        logger.info(f"[single_program_multi_expert] Expert {i}: PCC={pcc:.6f}")

        if pcc < 0.985:
            logger.error(f"[single_program_multi_expert] Expert {i} FAILED: PCC {pcc:.6f} < 0.985")
            all_passed = False

    assert all_passed, "One or more experts failed PCC threshold"
    logger.info(
        f"[single_program_multi_expert] PASSED: all {NUM_EXPERTS} experts verified ({NUM_EXPERTS} programs vs {NUM_EXPERTS * 2} in Phase 2)"
    )


# =====================================================================
# Phase 4 Step 4.1: Host Dispatch + Device Compute + Host Combine
# =====================================================================


def host_dispatch(hidden_states_torch, topk_indices_torch, num_experts, P):
    """Sort tokens by expert, create per-expert pkt_buf tensors (zero-padded to P rows).

    Args:
        hidden_states_torch: [N_tokens, D] bfloat16 tensor
        topk_indices_torch: [N_tokens, K] int64 tensor
        num_experts: number of local experts
        P: packet size (rows per pkt_buf)

    Returns:
        List of dicts per expert, each containing:
            pkt_buf: [P, D] bfloat16 tensor (zero-padded)
            M_e: int (actual token count, <= P)
            token_indices: list of global token indices
            k_indices: list of which top-k slot each token came from
    """
    N_tokens, D = hidden_states_torch.shape
    K = topk_indices_torch.shape[1]

    routing = []
    for e in range(num_experts):
        pkt_buf = torch.zeros(P, D, dtype=hidden_states_torch.dtype)
        token_indices = []
        k_indices = []

        for t in range(N_tokens):
            for k in range(K):
                if topk_indices_torch[t, k].item() == e:
                    token_indices.append(t)
                    k_indices.append(k)

        M_e = min(len(token_indices), P)
        for i in range(M_e):
            pkt_buf[i] = hidden_states_torch[token_indices[i]]

        routing.append(
            {
                "pkt_buf": pkt_buf,
                "M_e": M_e,
                "token_indices": token_indices[:M_e],
                "k_indices": k_indices[:M_e],
            }
        )

    return routing


def host_combine(out_bufs_torch, routing_meta, topk_weights_torch, N_tokens, D):
    """Weighted accumulation of expert outputs back to token positions.

    Args:
        out_bufs_torch: list of [P, D] float32 tensors per expert
        routing_meta: list of dicts from host_dispatch
        topk_weights_torch: [N_tokens, K] float tensor
        N_tokens: total number of tokens
        D: hidden dimension

    Returns:
        output: [N_tokens, D] float32 tensor
    """
    output = torch.zeros(N_tokens, D, dtype=torch.float32)

    for e, meta in enumerate(routing_meta):
        for i in range(meta["M_e"]):
            t = meta["token_indices"][i]
            k = meta["k_indices"][i]
            w = topk_weights_torch[t, k].float().item()
            output[t] += w * out_bufs_torch[e][i].float()

    return output


def moe_reference_dequant(
    hidden_states_torch,
    gu_w_dequant_list,
    dn_w_dequant_list,
    topk_weights_torch,
    routing_meta,
    d_ff,
    num_cores,
    n_weight_per_core_gu,
    n_out_per_core,
):
    """Full MoE reference using dequantized BFP4_b weights (matching device precision).

    Uses per-core SwiGLU on shuffled gate_up layout to match device behavior exactly.
    """
    N_tokens, D = hidden_states_torch.shape
    num_experts = len(gu_w_dequant_list)
    d_ff_half = d_ff // 2
    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE
    P = routing_meta[0]["pkt_buf"].shape[0]

    output = torch.zeros(N_tokens, D, dtype=torch.float32)

    for e in range(num_experts):
        meta = routing_meta[e]
        M_e = meta["M_e"]
        if M_e == 0:
            continue

        # Build the pkt_buf in float32 (same as what device receives after bf16→f32)
        pkt = torch.zeros(P, D, dtype=torch.float32)
        for i in range(M_e):
            pkt[i] = hidden_states_torch[meta["token_indices"][i]].float()

        # gate_up matmul with dequantized shuffled weights
        ref_gu = pkt @ gu_w_dequant_list[e]  # [P, D_FF] in shuffled layout

        # Per-core SwiGLU (matches device behavior)
        ref_inter = torch.empty(P, d_ff_half, dtype=torch.float32)
        for c in range(num_cores):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)

        # BF16 DRAM roundtrip simulation
        ref_inter_bf16 = ref_inter.bfloat16().float()

        # down matmul with dequantized weights
        ref_out = ref_inter_bf16 @ dn_w_dequant_list[e]  # [P, D]

        # Weighted accumulate only M_e actual tokens
        for i in range(M_e):
            t = meta["token_indices"][i]
            k = meta["k_indices"][i]
            w = topk_weights_torch[t, k].float().item()
            output[t] += w * ref_out[i]

    return output


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_moe_host_dispatch_device_compute_host_combine(device):
    """Phase 4 Step 4.1: End-to-end MoE with host dispatch/combine, device compute.

    Validates MoE arithmetic: host sorts tokens by expert (dispatch), device runs
    gate_up+SwiGLU+down per expert (reusing Phase 3 single-program kernels),
    host accumulates weighted outputs (combine).

    PCC target: >= 0.97 (BFP4_b quantization in both matmuls + SwiGLU nonlinearity).
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2
    P = 32
    N_TOKENS = 32
    NUM_EXPERTS = 4
    TOP_K = 2
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles = D // TILE
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    n_tiles_dn = D // TILE
    n_per_core_dn = n_tiles_dn // NUM_CORES

    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE

    torch.manual_seed(123)

    # ---- Generate inputs ----
    hidden_states = torch.randn(N_TOKENS, D, dtype=torch.bfloat16)

    # Random top-k routing: each token picks TOP_K distinct experts from {0..NUM_EXPERTS-1}
    topk_indices = torch.zeros(N_TOKENS, TOP_K, dtype=torch.int64)
    for t in range(N_TOKENS):
        perm = torch.randperm(NUM_EXPERTS)[:TOP_K]
        topk_indices[t] = perm

    # Softmax-like routing weights (normalized per token)
    raw_weights = torch.rand(N_TOKENS, TOP_K, dtype=torch.float32)
    topk_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)

    # Per-expert weights
    gate_up_ws = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]
    down_ws = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]

    # Pre-shuffle gate_up weights (interleave gate/up per core)
    shuffled_ws = []
    for w in gate_up_ws:
        gate_cols = w[:, :D_FF_HALF]
        up_cols = w[:, D_FF_HALF:]
        s = torch.empty_like(w)
        for c in range(NUM_CORES):
            s[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[:, c * half_cols : (c + 1) * half_cols]
            s[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
                :, c * half_cols : (c + 1) * half_cols
            ]
        shuffled_ws.append(s)

    # ---- Host dispatch: sort tokens by expert ----
    routing_meta = host_dispatch(hidden_states, topk_indices, NUM_EXPERTS, P)

    for e in range(NUM_EXPERTS):
        logger.info(
            f"Expert {e}: M_e={routing_meta[e]['M_e']} tokens " f"(indices={routing_meta[e]['token_indices'][:8]})"
        )

    # ---- Upload weights to device ----
    gu_w_tensors = [
        ttnn.from_torch(
            s.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in shuffled_ws
    ]
    dn_w_tensors = [
        ttnn.from_torch(
            w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in down_ws
    ]

    # ---- Core grid and physical coords ----
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    phys_coords = []
    for y in range(GRID_Y):
        for x in range(GRID_X):
            phys = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            phys_coords.append((phys.x, phys.y))
    leader_phys_x, leader_phys_y = phys_coords[0]

    # ---- Device compute per expert ----
    device_out_bufs = []
    for e in range(NUM_EXPERTS):
        M_e = routing_meta[e]["M_e"]
        if M_e == 0:
            logger.info(f"Expert {e}: skipping (0 tokens)")
            device_out_bufs.append(torch.zeros(P, D, dtype=torch.float32))
            continue

        # Upload pkt_buf as activation
        act_tensor = ttnn.from_torch(
            routing_meta[e]["pkt_buf"].unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Allocate intermediate and output buffers
        inter_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D_FF_HALF]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        out_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        logger.info(f"Expert {e}: running single-program compute (M_e={M_e})")
        prog = _make_single_program_expert(
            device,
            act_tensor,
            gu_w_tensors[e],
            dn_w_tensors[e],
            inter_tensor,
            out_tensor,
            core_grid,
            phys_coords,
            leader_phys_x,
            leader_phys_y,
            NUM_CORES,
            GRID_X,
            GRID_Y,
            k_tiles,
            n_weight_per_core_gu,
            n_weight_tiles_gu,
            n_out_per_core,
            n_per_core_dn,
            n_tiles_dn,
        )
        io_tensors = [act_tensor, gu_w_tensors[e], dn_w_tensors[e], inter_tensor, out_tensor]
        expert_out = ttnn.generic_op(io_tensors, prog)
        ttnn.synchronize_device(device)

        out_torch = ttnn.to_torch(expert_out).squeeze().float()
        device_out_bufs.append(out_torch)

    # ---- Host combine: weighted accumulation ----
    combined_output = host_combine(device_out_bufs, routing_meta, topk_weights, N_TOKENS, D)

    # ---- Reference computation with dequantized weights ----
    gu_w_dequant = [ttnn.to_torch(t).squeeze().float() for t in gu_w_tensors]
    dn_w_dequant = [ttnn.to_torch(t).squeeze().float() for t in dn_w_tensors]

    ref_output = moe_reference_dequant(
        hidden_states,
        gu_w_dequant,
        dn_w_dequant,
        topk_weights,
        routing_meta,
        D_FF,
        NUM_CORES,
        n_weight_per_core_gu,
        n_out_per_core,
    )

    # ---- Verify PCC on all tokens ----
    # With TOP_K=2, all tokens are routed to at least one expert
    pcc = torch.corrcoef(torch.stack([combined_output.flatten(), ref_output.flatten()]))[0, 1].item()
    max_err = (combined_output - ref_output).abs().max().item()
    mean_err = (combined_output - ref_output).abs().mean().item()

    logger.info(f"[moe_host_dispatch_device_compute] N_tokens={N_TOKENS}, K={TOP_K}, " f"experts={NUM_EXPERTS}")
    logger.info(f"[moe_host_dispatch_device_compute] PCC={pcc:.6f} max_err={max_err:.6f} " f"mean_err={mean_err:.6f}")

    # Per-expert PCC for diagnostics
    for e in range(NUM_EXPERTS):
        meta = routing_meta[e]
        M_e = meta["M_e"]
        if M_e == 0:
            continue
        # Compare just this expert's raw output (before combine weighting)
        dev_out_e = device_out_bufs[e][:M_e]
        # Compute per-expert reference
        pkt = torch.zeros(P, D, dtype=torch.float32)
        for i in range(M_e):
            pkt[i] = hidden_states[meta["token_indices"][i]].float()
        ref_gu_e = pkt @ gu_w_dequant[e]
        ref_inter_e = torch.empty(P, D_FF_HALF, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu_e[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu_e[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter_e[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out_e = ref_inter_e.bfloat16().float() @ dn_w_dequant[e]
        ref_out_e_active = ref_out_e[:M_e]

        epcc = torch.corrcoef(torch.stack([dev_out_e.flatten(), ref_out_e_active.flatten()]))[0, 1].item()
        logger.info(f"  Expert {e}: M_e={M_e}, per-expert PCC={epcc:.6f}")

    assert pcc >= 0.97, f"MoE combined output PCC {pcc:.6f} < 0.97"
    logger.info("[moe_host_dispatch_device_compute] PASSED")


# Kernel path for dispatch
DISPATCH_WRITER_KERNEL = f"{KERNEL_DIR}/dispatch_writer.cpp"


def _make_dispatch_program(device, hs_tensor, pkt_buf_tensor, num_tiles, dispatch_core):
    """Create a ProgramDescriptor for the dispatch tile-copy kernel on 1 core."""
    ct_args = list(ttnn.TensorAccessorArgs(hs_tensor).get_compile_time_args())
    ct_args.extend(ttnn.TensorAccessorArgs(pkt_buf_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    rt_args[0][0] = [
        hs_tensor.buffer_address(),
        pkt_buf_tensor.buffer_address(),
        num_tiles,
    ]

    cb_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=dispatch_core,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    kernel_desc = ttnn.KernelDescriptor(
        kernel_source=DISPATCH_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=dispatch_core,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[kernel_desc],
        semaphores=[],
        cbs=[cb_desc],
    )


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_moe_device_dispatch_host_combine(device):
    """Phase 4 Step 4.2: Device dispatch + device compute + host combine.

    Replaces host_dispatch with a device kernel that copies activation tiles
    from hidden_states to per-expert pkt_bufs. Since N_tokens = P = 32, all
    tokens share one tile row and dispatch is a tile-level copy.

    The combine step uses original token positions (not gathered positions)
    since pkt_buf contains all tokens at their original rows.

    PCC target: >= 0.97 (should match Step 4.1 exactly).
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2
    P = 32
    N_TOKENS = 32
    NUM_EXPERTS = 4
    TOP_K = 2
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles = D // TILE
    D_tiles = D // TILE  # 90
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    n_tiles_dn = D // TILE
    n_per_core_dn = n_tiles_dn // NUM_CORES

    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE

    torch.manual_seed(123)

    # ---- Generate inputs (same as Step 4.1) ----
    hidden_states = torch.randn(N_TOKENS, D, dtype=torch.bfloat16)

    topk_indices = torch.zeros(N_TOKENS, TOP_K, dtype=torch.int64)
    for t in range(N_TOKENS):
        perm = torch.randperm(NUM_EXPERTS)[:TOP_K]
        topk_indices[t] = perm

    raw_weights = torch.rand(N_TOKENS, TOP_K, dtype=torch.float32)
    topk_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)

    gate_up_ws = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]
    down_ws = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]

    # Pre-shuffle gate_up weights
    shuffled_ws = []
    for w in gate_up_ws:
        gate_cols = w[:, :D_FF_HALF]
        up_cols = w[:, D_FF_HALF:]
        s = torch.empty_like(w)
        for c in range(NUM_CORES):
            s[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[:, c * half_cols : (c + 1) * half_cols]
            s[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
                :, c * half_cols : (c + 1) * half_cols
            ]
        shuffled_ws.append(s)

    # ---- Build routing metadata (host-side, for combine) ----
    routing_meta = host_dispatch(hidden_states, topk_indices, NUM_EXPERTS, P)

    for e in range(NUM_EXPERTS):
        logger.info(
            f"Expert {e}: M_e={routing_meta[e]['M_e']} tokens " f"(indices={routing_meta[e]['token_indices'][:8]})"
        )

    # ---- Upload to device ----
    hs_tensor = ttnn.from_torch(
        hidden_states.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gu_w_tensors = [
        ttnn.from_torch(
            s.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in shuffled_ws
    ]
    dn_w_tensors = [
        ttnn.from_torch(
            w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in down_ws
    ]

    # ---- Core grids ----
    # Dispatch: 1 core at (0, 0)
    dispatch_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    # Compute: 15 cores at (0,0)-(4,2) — same grid, separate programs
    compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    phys_coords = []
    for y in range(GRID_Y):
        for x in range(GRID_X):
            phys = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            phys_coords.append((phys.x, phys.y))
    leader_phys_x, leader_phys_y = phys_coords[0]

    # ---- Per-expert: dispatch → compute ----
    device_out_bufs = []
    for e in range(NUM_EXPERTS):
        # Allocate pkt_buf for this expert
        pkt_buf_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run dispatch: copy hidden_states tiles → pkt_buf
        dispatch_prog = _make_dispatch_program(device, hs_tensor, pkt_buf_tensor, D_tiles, dispatch_core)
        ttnn.generic_op([hs_tensor, pkt_buf_tensor], dispatch_prog)
        ttnn.synchronize_device(device)

        # Verify dispatch: pkt_buf should match hidden_states
        if e == 0:
            pkt_torch = ttnn.to_torch(pkt_buf_tensor).squeeze().float()
            hs_torch = ttnn.to_torch(hs_tensor).squeeze().float()
            dispatch_pcc = torch.corrcoef(torch.stack([pkt_torch.flatten(), hs_torch.flatten()]))[0, 1].item()
            logger.info(f"Dispatch verification: pkt_buf vs hidden_states PCC={dispatch_pcc:.6f}")
            assert dispatch_pcc > 0.999, f"Dispatch copy failed: PCC={dispatch_pcc:.6f}"

        # Allocate intermediate and output
        inter_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D_FF_HALF]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        out_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run compute: pkt_buf + weights → out_buf
        logger.info(f"Expert {e}: dispatch done, running compute")
        compute_prog = _make_single_program_expert(
            device,
            pkt_buf_tensor,
            gu_w_tensors[e],
            dn_w_tensors[e],
            inter_tensor,
            out_tensor,
            compute_grid,
            phys_coords,
            leader_phys_x,
            leader_phys_y,
            NUM_CORES,
            GRID_X,
            GRID_Y,
            k_tiles,
            n_weight_per_core_gu,
            n_weight_tiles_gu,
            n_out_per_core,
            n_per_core_dn,
            n_tiles_dn,
        )
        io_tensors = [pkt_buf_tensor, gu_w_tensors[e], dn_w_tensors[e], inter_tensor, out_tensor]
        expert_out = ttnn.generic_op(io_tensors, compute_prog)
        ttnn.synchronize_device(device)

        out_torch = ttnn.to_torch(expert_out).squeeze().float()
        device_out_bufs.append(out_torch)

    # ---- Host combine using ORIGINAL token positions ----
    # Since pkt_buf = hidden_states (all tokens), out_buf[e] row t = expert e output for token t
    combined_output = torch.zeros(N_TOKENS, D, dtype=torch.float32)
    for e in range(NUM_EXPERTS):
        meta = routing_meta[e]
        for i in range(meta["M_e"]):
            t = meta["token_indices"][i]
            k = meta["k_indices"][i]
            w = topk_weights[t, k].float().item()
            combined_output[t] += w * device_out_bufs[e][t].float()

    # ---- Reference (same as Step 4.1 but using full-token expert outputs) ----
    gu_w_dequant = [ttnn.to_torch(t).squeeze().float() for t in gu_w_tensors]
    dn_w_dequant = [ttnn.to_torch(t).squeeze().float() for t in dn_w_tensors]

    ref_output = torch.zeros(N_TOKENS, D, dtype=torch.float32)
    for e in range(NUM_EXPERTS):
        meta = routing_meta[e]
        if meta["M_e"] == 0:
            continue

        # All-token expert compute (matches device behavior: pkt_buf = hidden_states)
        ref_gu = hidden_states.float() @ gu_w_dequant[e]
        ref_inter = torch.empty(P, D_FF_HALF, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ dn_w_dequant[e]

        for i in range(meta["M_e"]):
            t = meta["token_indices"][i]
            k = meta["k_indices"][i]
            w = topk_weights[t, k].float().item()
            ref_output[t] += w * ref_out[t]

    # ---- Verify PCC ----
    pcc = torch.corrcoef(torch.stack([combined_output.flatten(), ref_output.flatten()]))[0, 1].item()
    max_err = (combined_output - ref_output).abs().max().item()
    mean_err = (combined_output - ref_output).abs().mean().item()

    logger.info(f"[moe_device_dispatch_host_combine] N_tokens={N_TOKENS}, K={TOP_K}, " f"experts={NUM_EXPERTS}")
    logger.info(f"[moe_device_dispatch_host_combine] PCC={pcc:.6f} max_err={max_err:.6f} " f"mean_err={mean_err:.6f}")

    assert pcc >= 0.97, f"MoE output PCC {pcc:.6f} < 0.97"
    logger.info("[moe_device_dispatch_host_combine] PASSED")


# Kernel path for combine
COMBINE_DM_KERNEL = f"{KERNEL_DIR}/combine_dm.cpp"


def _make_combine_program(
    device,
    output_tensor,
    out_buf_tensors,
    routing_meta,
    topk_weights,
    D_tiles,
    combine_core,
):
    """Create a ProgramDescriptor for the combine weighted-accumulate kernel on 1 core.

    The combine kernel reads expert out_buf tiles and the output tile, does scalar
    BF16 weighted accumulate for assigned tokens, and writes the output back.
    """
    ct_args = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Build runtime args: output_addr, D_tiles, num_experts, [per-expert data]
    rt_args_list = [
        output_tensor.buffer_address(),
        D_tiles,
        len(out_buf_tensors),
    ]

    for e, out_buf in enumerate(out_buf_tensors):
        meta = routing_meta[e]
        M_e = meta["M_e"]
        rt_args_list.append(out_buf.buffer_address())
        rt_args_list.append(M_e)

        # Token row indices
        for i in range(M_e):
            rt_args_list.append(meta["token_indices"][i])

        # Weights as BF16 bit patterns (lower 16 bits of uint32)
        for i in range(M_e):
            t = meta["token_indices"][i]
            k = meta["k_indices"][i]
            w = topk_weights[t, k].item()
            # Convert to BF16 bits: torch bf16 → int16 view → uint32
            w_bf16 = int(torch.tensor(w, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
            rt_args_list.append(w_bf16)

    rt_args = ttnn.RuntimeArgs()
    rt_args[0][0] = rt_args_list

    # 2 CBs: output tile and expert tile
    cb_out_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=combine_core,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb_exp_desc = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=combine_core,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    kernel_desc = ttnn.KernelDescriptor(
        kernel_source=COMBINE_DM_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=combine_core,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[kernel_desc],
        semaphores=[],
        cbs=[cb_out_desc, cb_exp_desc],
    )


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_moe_device_dispatch_device_combine(device):
    """Phase 4 Step 4.3: Device dispatch + device compute + device combine.

    Full on-device MoE pipeline. Dispatch copies hidden_states to per-expert
    pkt_bufs, compute runs gate_up+SwiGLU+down per expert, combine does
    weighted accumulation into output tensor using scalar BF16 kernel on RISC-V.

    PCC target: >= 0.96 (additional BF16 truncation in combine kernel).
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2
    P = 32
    N_TOKENS = 32
    NUM_EXPERTS = 4
    TOP_K = 2
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles = D // TILE
    D_tiles = D // TILE  # 90
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    n_tiles_dn = D // TILE
    n_per_core_dn = n_tiles_dn // NUM_CORES

    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE

    torch.manual_seed(123)

    # ---- Generate inputs (same seed as Step 4.1/4.2) ----
    hidden_states = torch.randn(N_TOKENS, D, dtype=torch.bfloat16)

    topk_indices = torch.zeros(N_TOKENS, TOP_K, dtype=torch.int64)
    for t in range(N_TOKENS):
        perm = torch.randperm(NUM_EXPERTS)[:TOP_K]
        topk_indices[t] = perm

    raw_weights = torch.rand(N_TOKENS, TOP_K, dtype=torch.float32)
    topk_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)

    gate_up_ws = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]
    down_ws = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]

    # Pre-shuffle gate_up weights
    shuffled_ws = []
    for w in gate_up_ws:
        gate_cols = w[:, :D_FF_HALF]
        up_cols = w[:, D_FF_HALF:]
        s = torch.empty_like(w)
        for c in range(NUM_CORES):
            s[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[:, c * half_cols : (c + 1) * half_cols]
            s[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
                :, c * half_cols : (c + 1) * half_cols
            ]
        shuffled_ws.append(s)

    # Build routing metadata
    routing_meta = host_dispatch(hidden_states, topk_indices, NUM_EXPERTS, P)

    for e in range(NUM_EXPERTS):
        logger.info(
            f"Expert {e}: M_e={routing_meta[e]['M_e']} tokens " f"(indices={routing_meta[e]['token_indices'][:8]})"
        )

    # ---- Upload to device ----
    hs_tensor = ttnn.from_torch(
        hidden_states.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gu_w_tensors = [
        ttnn.from_torch(
            s.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in shuffled_ws
    ]
    dn_w_tensors = [
        ttnn.from_torch(
            w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in down_ws
    ]

    # ---- Core grids ----
    dispatch_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])
    combine_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    phys_coords = []
    for y in range(GRID_Y):
        for x in range(GRID_X):
            phys = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            phys_coords.append((phys.x, phys.y))
    leader_phys_x, leader_phys_y = phys_coords[0]

    # ---- Phase 1: Per-expert dispatch → compute ----
    out_buf_tensors = []
    for e in range(NUM_EXPERTS):
        # Allocate pkt_buf
        pkt_buf_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Dispatch: copy tiles
        dispatch_prog = _make_dispatch_program(device, hs_tensor, pkt_buf_tensor, D_tiles, dispatch_core)
        ttnn.generic_op([hs_tensor, pkt_buf_tensor], dispatch_prog)
        ttnn.synchronize_device(device)

        # Allocate intermediate and output buffers
        inter_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D_FF_HALF]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        out_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Compute: pkt_buf + weights → out_buf
        logger.info(f"Expert {e}: running compute")
        compute_prog = _make_single_program_expert(
            device,
            pkt_buf_tensor,
            gu_w_tensors[e],
            dn_w_tensors[e],
            inter_tensor,
            out_tensor,
            compute_grid,
            phys_coords,
            leader_phys_x,
            leader_phys_y,
            NUM_CORES,
            GRID_X,
            GRID_Y,
            k_tiles,
            n_weight_per_core_gu,
            n_weight_tiles_gu,
            n_out_per_core,
            n_per_core_dn,
            n_tiles_dn,
        )
        io_tensors = [pkt_buf_tensor, gu_w_tensors[e], dn_w_tensors[e], inter_tensor, out_tensor]
        expert_out = ttnn.generic_op(io_tensors, compute_prog)
        ttnn.synchronize_device(device)
        out_buf_tensors.append(expert_out)

    # ---- Phase 2: Device combine ----
    # Create zero-filled output tensor
    output_tensor = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info("Running device combine kernel")
    combine_prog = _make_combine_program(
        device,
        output_tensor,
        out_buf_tensors,
        routing_meta,
        topk_weights,
        D_tiles,
        combine_core,
    )
    # io_tensors: output + all out_bufs (combine reads from all)
    combine_io = [output_tensor] + out_buf_tensors
    combined_result = ttnn.generic_op(combine_io, combine_prog)
    ttnn.synchronize_device(device)

    # Read output_tensor directly (generic_op returns last io_tensor, not the output)
    combined_output = ttnn.to_torch(output_tensor).squeeze().float()

    # ---- Reference (same as Step 4.2) ----
    gu_w_dequant = [ttnn.to_torch(t).squeeze().float() for t in gu_w_tensors]
    dn_w_dequant = [ttnn.to_torch(t).squeeze().float() for t in dn_w_tensors]

    ref_output = torch.zeros(N_TOKENS, D, dtype=torch.float32)
    for e in range(NUM_EXPERTS):
        meta = routing_meta[e]
        if meta["M_e"] == 0:
            continue

        ref_gu = hidden_states.float() @ gu_w_dequant[e]
        ref_inter = torch.empty(P, D_FF_HALF, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ dn_w_dequant[e]

        for i in range(meta["M_e"]):
            t = meta["token_indices"][i]
            k = meta["k_indices"][i]
            w = topk_weights[t, k].float().item()
            ref_output[t] += w * ref_out[t]

    # ---- Verify PCC ----
    pcc = torch.corrcoef(torch.stack([combined_output.flatten(), ref_output.flatten()]))[0, 1].item()
    max_err = (combined_output - ref_output).abs().max().item()
    mean_err = (combined_output - ref_output).abs().mean().item()

    logger.info(f"[moe_device_dispatch_device_combine] N_tokens={N_TOKENS}, K={TOP_K}, " f"experts={NUM_EXPERTS}")
    logger.info(f"[moe_device_dispatch_device_combine] PCC={pcc:.6f} max_err={max_err:.6f} " f"mean_err={mean_err:.6f}")

    assert pcc >= 0.97, f"MoE output PCC {pcc:.6f} < 0.97"
    logger.info("[moe_device_dispatch_device_combine] PASSED")


# Kernel paths for fused pipeline


# Kernel paths for fused pipeline


# Kernel paths for fused pipeline
DISPATCH_WRITER_FUSED_KERNEL = f"{KERNEL_DIR}/dispatch_writer_fused.cpp"
EXPERT_READER_FUSED_KERNEL = f"{KERNEL_DIR}/expert_reader_fused.cpp"
EXPERT_WRITER_FUSED_KERNEL = f"{KERNEL_DIR}/expert_writer_fused.cpp"
COMBINE_DM_FUSED_KERNEL = f"{KERNEL_DIR}/combine_dm_fused.cpp"


def _make_fused_expert_program(
    device,
    # Dispatch inputs
    hs_tensor,
    pkt_buf_tensor,
    # Compute inputs/outputs
    shuffled_w_tensor,
    down_w_tensor,
    inter_tensor,
    out_tensor,
    # Combine inputs
    output_tensor,
    routing_meta_e,
    topk_weights,
    # Core grids
    compute_grid,
    dispatch_core,
    combine_core,
    # Physical coords
    phys_coords,
    leader_phys_x,
    leader_phys_y,
    dispatch_phys_x,
    dispatch_phys_y,
    combine_phys_x,
    combine_phys_y,
    # Tile counts
    NUM_CORES,
    GRID_X,
    GRID_Y,
    k_tiles,
    D_tiles,
    n_weight_per_core_gu,
    n_weight_tiles_gu,
    n_out_per_core,
    n_per_core_dn,
    n_tiles_dn,
):
    """Create a ProgramDescriptor for a single fused expert: dispatch → compute → combine.

    Uses 4 L1 semaphores to chain dispatch, compute, and combine stages
    without device synchronization between stages:
      SEM_BARRIER (0): Cross-core barrier for Phase A and Phase B within compute
      SEM_GO (1): Leader → all cores go signal within compute
      SEM_PKT_READY (2): Dispatch → compute leader signal
      SEM_EXPERT_DONE (3): Compute leader → combine core signal
    """
    # ---- Compile-time args ----
    # Reader (dm0 on compute cores): TensorAccessorArgs for act (=pkt_buf) and inter
    reader_ct_args = list(ttnn.TensorAccessorArgs(pkt_buf_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(inter_tensor).get_compile_time_args())

    # Writer (dm1 on compute cores): TensorAccessorArgs for gate_up_w, inter, down_w, output (=out_buf)
    writer_ct_args = list(ttnn.TensorAccessorArgs(shuffled_w_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(inter_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(down_w_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(out_tensor).get_compile_time_args())

    # Compute: compile-time args
    compute_ct_args = [k_tiles, n_weight_per_core_gu, n_per_core_dn]

    # Dispatch writer (dm1 on dispatch core): TensorAccessorArgs for hs and pkt_buf
    dispatch_ct_args = list(ttnn.TensorAccessorArgs(hs_tensor).get_compile_time_args())
    dispatch_ct_args.extend(ttnn.TensorAccessorArgs(pkt_buf_tensor).get_compile_time_args())

    # Combine reader (dm0 on combine core): TensorAccessorArgs for output tensor
    combine_ct_args = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---- Runtime args: compute reader (per-core) ----
    reader_rt_args = ttnn.RuntimeArgs()
    for y in range(GRID_Y):
        for x in range(GRID_X):
            core_idx = y * GRID_X + x
            is_leader = 1 if core_idx == 0 else 0
            args = [
                pkt_buf_tensor.buffer_address(),  # act_addr (pkt_buf is the activation)
                k_tiles,
                inter_tensor.buffer_address(),
                is_leader,
                NUM_CORES,
                combine_phys_x,
                combine_phys_y,
            ]
            # Physical coords of all compute cores
            for px, py in phys_coords:
                args.append(px)
                args.append(py)
            reader_rt_args[x][y] = args

    # ---- Runtime args: compute writer (per-core) ----
    writer_rt_args = ttnn.RuntimeArgs()
    for y in range(GRID_Y):
        for x in range(GRID_X):
            core_idx = y * GRID_X + x
            args = [
                shuffled_w_tensor.buffer_address(),
                inter_tensor.buffer_address(),
                down_w_tensor.buffer_address(),
                out_tensor.buffer_address(),
                k_tiles,
                n_weight_per_core_gu,
                n_weight_tiles_gu,
                core_idx * n_weight_per_core_gu,  # core_weight_offset_gu
                core_idx * n_out_per_core,  # core_out_offset_gu
                n_out_per_core,
                n_per_core_dn,
                n_tiles_dn,
                core_idx * n_per_core_dn,  # core_dn_offset
                leader_phys_x,
                leader_phys_y,
            ]
            writer_rt_args[x][y] = args

    # ---- Runtime args: dispatch writer (single core at logical (0, GRID_Y)) ----
    # RuntimeArgs indexed by [x][y] logical coords
    dispatch_rt_args = ttnn.RuntimeArgs()
    dispatch_rt_args[0][GRID_Y] = [
        hs_tensor.buffer_address(),
        pkt_buf_tensor.buffer_address(),
        D_tiles,
        leader_phys_x,
        leader_phys_y,
    ]

    # ---- Runtime args: combine reader (single core) ----
    # Pad to fixed size (5 + 2*MAX_TOKENS) so generic_op program caching works across experts.
    # IMPORTANT: token indices and weights are packed contiguously (M_e tokens then M_e weights),
    # with padding only at the end. The kernel computes weights_base = tokens_base + M_e,
    # so there must be NO padding between the token indices and weight values.
    MAX_TOKENS = 32  # P = max tokens per packet
    FIXED_COMBINE_ARGS = 5 + 2 * MAX_TOKENS  # 69 total args
    meta = routing_meta_e
    M_e = meta["M_e"]
    combine_args_list = [
        output_tensor.buffer_address(),
        D_tiles,
        1,  # num_experts = 1 (one expert per fused program)
    ]
    # Expert 0 (the only one in this program)
    combine_args_list.append(out_tensor.buffer_address())
    combine_args_list.append(M_e)
    # M_e token row indices (NO padding here — weights must immediately follow)
    for i in range(M_e):
        combine_args_list.append(meta["token_indices"][i])
    # M_e weights as BF16 bit patterns (NO padding here)
    for i in range(M_e):
        t = meta["token_indices"][i]
        k = meta["k_indices"][i]
        w = topk_weights[t, k].item()
        w_bf16 = int(torch.tensor(w, dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
        combine_args_list.append(w_bf16)
    # Pad remainder to fixed total size
    while len(combine_args_list) < FIXED_COMBINE_ARGS:
        combine_args_list.append(0)

    # RuntimeArgs indexed by [x][y] logical coords — combine core at (1, GRID_Y)
    combine_rt_args = ttnn.RuntimeArgs()
    combine_rt_args[1][GRID_Y] = combine_args_list

    # ---- Semaphores: all 4 on all cores ----
    all_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1)),  # compute
            ttnn.CoreRange(ttnn.CoreCoord(0, GRID_Y), ttnn.CoreCoord(1, GRID_Y)),  # dispatch + combine
        ]
    )

    sem_barrier = ttnn.SemaphoreDescriptor(
        id=0,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=all_cores,
        initial_value=0,
    )
    sem_go = ttnn.SemaphoreDescriptor(
        id=1,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=all_cores,
        initial_value=0,
    )
    sem_pkt_ready = ttnn.SemaphoreDescriptor(
        id=2,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=all_cores,
        initial_value=0,
    )
    sem_expert_done = ttnn.SemaphoreDescriptor(
        id=3,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=all_cores,
        initial_value=0,
    )

    # ---- Circular Buffers ----
    # Compute cores
    cb0_compute = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=compute_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb1_compute = ttnn.CBDescriptor(
        total_size=n_weight_per_core_gu * BFP4_TILE_BYTES,
        core_ranges=compute_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=1,
                data_format=ttnn.bfloat4_b,
                page_size=BFP4_TILE_BYTES,
            )
        ],
    )
    cb2_compute = ttnn.CBDescriptor(
        total_size=n_out_per_core * BF16_TILE_BYTES,
        core_ranges=compute_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=2,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    # Dispatch core — uses CB3 (unique index to avoid conflict with compute CB0)
    cb3_dispatch = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=dispatch_core,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=3,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    # Combine core — uses CB4, CB5 (unique indices to avoid conflict with compute CB0/CB1)
    cb4_combine = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=combine_core,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=4,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )
    cb5_combine = ttnn.CBDescriptor(
        total_size=BF16_TILE_BYTES,
        core_ranges=combine_core,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=5,
                data_format=ttnn.bfloat16,
                page_size=BF16_TILE_BYTES,
            )
        ],
    )

    # ---- Kernel Descriptors ----
    # Compute reader (dm0 on compute cores)
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_READER_FUSED_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=compute_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Compute writer (dm1 on compute cores)
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_WRITER_FUSED_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=compute_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel (tensix on compute cores) — UNCHANGED from Phase 3
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=EXPERT_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=compute_grid,
        compile_time_args=compute_ct_args,
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        ),
    )

    # Dispatch writer (dm1 on dispatch core)
    dispatch_kernel = ttnn.KernelDescriptor(
        kernel_source=DISPATCH_WRITER_FUSED_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=dispatch_core,
        compile_time_args=dispatch_ct_args,
        runtime_args=dispatch_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Combine reader (dm0 on combine core)
    combine_kernel = ttnn.KernelDescriptor(
        kernel_source=COMBINE_DM_FUSED_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=combine_core,
        compile_time_args=combine_ct_args,
        runtime_args=combine_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel, dispatch_kernel, combine_kernel],
        semaphores=[sem_barrier, sem_go, sem_pkt_ready, sem_expert_done],
        cbs=[
            cb0_compute,
            cb1_compute,
            cb2_compute,
            cb3_dispatch,
            cb4_combine,
            cb5_combine,
        ],
    )


@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_moe_semaphore_chained(device):
    """Phase 4 Step 4.5: Semaphore-chained fused dispatch+compute+combine per expert.

    Each expert runs as a single program with 3 stages chained by L1 semaphores:
      1. Dispatch core copies hidden_states → pkt_buf, signals SEM_PKT_READY
      2. Compute cores wait for pkt_buf, run gate_up+SwiGLU+barrier+down, signal SEM_EXPERT_DONE
      3. Combine core waits for output, runs weighted accumulation into output tensor

    Core layout: compute (0,0)-(4,2), dispatch (0,3), combine (1,3).
    4 program launches (one per expert), no synchronize between stages within each.

    PCC target: >= 0.96 (matches Step 4.3).
    """
    D = 2880
    D_FF = 5760
    D_FF_HALF = D_FF // 2
    P = 32
    N_TOKENS = 32
    NUM_EXPERTS = 4
    TOP_K = 2
    NUM_CORES = 15
    GRID_X, GRID_Y = 5, 3

    k_tiles = D // TILE
    D_tiles = D // TILE  # 90
    n_weight_tiles_gu = D_FF // TILE
    n_weight_per_core_gu = n_weight_tiles_gu // NUM_CORES
    n_out_per_core = n_weight_per_core_gu // 2
    n_tiles_dn = D // TILE
    n_per_core_dn = n_tiles_dn // NUM_CORES

    cols_per_core = n_weight_per_core_gu * TILE
    half_cols = n_out_per_core * TILE

    torch.manual_seed(123)

    # ---- Generate inputs (same seed as Steps 4.1-4.3) ----
    hidden_states = torch.randn(N_TOKENS, D, dtype=torch.bfloat16)

    topk_indices = torch.zeros(N_TOKENS, TOP_K, dtype=torch.int64)
    for t in range(N_TOKENS):
        perm = torch.randperm(NUM_EXPERTS)[:TOP_K]
        topk_indices[t] = perm

    raw_weights = torch.rand(N_TOKENS, TOP_K, dtype=torch.float32)
    topk_weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)

    gate_up_ws = [torch.randn(D, D_FF, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]
    down_ws = [torch.randn(D_FF_HALF, D, dtype=torch.bfloat16) for _ in range(NUM_EXPERTS)]

    # Pre-shuffle gate_up weights
    shuffled_ws = []
    for w in gate_up_ws:
        gate_cols = w[:, :D_FF_HALF]
        up_cols = w[:, D_FF_HALF:]
        s = torch.empty_like(w)
        for c in range(NUM_CORES):
            s[:, c * cols_per_core : c * cols_per_core + half_cols] = gate_cols[:, c * half_cols : (c + 1) * half_cols]
            s[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core] = up_cols[
                :, c * half_cols : (c + 1) * half_cols
            ]
        shuffled_ws.append(s)

    # Build routing metadata
    routing_meta = host_dispatch(hidden_states, topk_indices, NUM_EXPERTS, P)

    for e in range(NUM_EXPERTS):
        logger.info(
            f"Expert {e}: M_e={routing_meta[e]['M_e']} tokens " f"(indices={routing_meta[e]['token_indices'][:8]})"
        )

    # ---- Upload to device ----
    hs_tensor = ttnn.from_torch(
        hidden_states.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    gu_w_tensors = [
        ttnn.from_torch(
            s.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in shuffled_ws
    ]
    dn_w_tensors = [
        ttnn.from_torch(
            w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in down_ws
    ]

    # ---- Core grids ----
    compute_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])
    dispatch_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, GRID_Y), ttnn.CoreCoord(0, GRID_Y))])
    combine_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, GRID_Y), ttnn.CoreCoord(1, GRID_Y))])

    # Physical coords for compute cores
    phys_coords = []
    for y in range(GRID_Y):
        for x in range(GRID_X):
            phys = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            phys_coords.append((phys.x, phys.y))
    leader_phys_x, leader_phys_y = phys_coords[0]

    # Physical coords for dispatch and combine cores
    dispatch_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(0, GRID_Y))
    dispatch_phys_x, dispatch_phys_y = dispatch_phys.x, dispatch_phys.y
    combine_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(1, GRID_Y))
    combine_phys_x, combine_phys_y = combine_phys.x, combine_phys.y

    logger.info(
        f"Core layout: compute (0,0)-({GRID_X-1},{GRID_Y-1}), "
        f"dispatch (0,{GRID_Y}) phys=({dispatch_phys_x},{dispatch_phys_y}), "
        f"combine (1,{GRID_Y}) phys=({combine_phys_x},{combine_phys_y})"
    )

    # ---- Create zero-filled output tensor (persists across expert programs) ----
    output_tensor = ttnn.from_torch(
        torch.zeros(1, 1, P, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ---- Per-expert fused pipeline ----
    for e in range(NUM_EXPERTS):
        meta = routing_meta[e]
        logger.info(f"Expert {e}: running fused dispatch+compute+combine (M_e={meta['M_e']})")

        # Allocate per-expert buffers
        pkt_buf_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        inter_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D_FF_HALF]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        out_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, P, D]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Build fused program
        fused_prog = _make_fused_expert_program(
            device,
            hs_tensor,
            pkt_buf_tensor,
            gu_w_tensors[e],
            dn_w_tensors[e],
            inter_tensor,
            out_tensor,
            output_tensor,
            meta,
            topk_weights,
            compute_grid,
            dispatch_core,
            combine_core,
            phys_coords,
            leader_phys_x,
            leader_phys_y,
            dispatch_phys_x,
            dispatch_phys_y,
            combine_phys_x,
            combine_phys_y,
            NUM_CORES,
            GRID_X,
            GRID_Y,
            k_tiles,
            D_tiles,
            n_weight_per_core_gu,
            n_weight_tiles_gu,
            n_out_per_core,
            n_per_core_dn,
            n_tiles_dn,
        )

        # io_tensors: all tensors whose DRAM addresses are used
        io_tensors = [
            hs_tensor,
            pkt_buf_tensor,
            gu_w_tensors[e],
            dn_w_tensors[e],
            inter_tensor,
            out_tensor,
            output_tensor,
        ]
        ttnn.generic_op(io_tensors, fused_prog)
        ttnn.synchronize_device(device)

    # ---- Read output (from output_tensor directly, NOT generic_op return) ----
    combined_output = ttnn.to_torch(output_tensor).squeeze().float()

    # ---- Reference (same as Step 4.3) ----
    gu_w_dequant = [ttnn.to_torch(t).squeeze().float() for t in gu_w_tensors]
    dn_w_dequant = [ttnn.to_torch(t).squeeze().float() for t in dn_w_tensors]

    ref_output = torch.zeros(N_TOKENS, D, dtype=torch.float32)
    for e in range(NUM_EXPERTS):
        meta = routing_meta[e]
        if meta["M_e"] == 0:
            continue

        ref_gu = hidden_states.float() @ gu_w_dequant[e]
        ref_inter = torch.empty(P, D_FF_HALF, dtype=torch.float32)
        for c in range(NUM_CORES):
            g = ref_gu[:, c * cols_per_core : c * cols_per_core + half_cols]
            u = ref_gu[:, c * cols_per_core + half_cols : (c + 1) * cols_per_core]
            ref_inter[:, c * half_cols : (c + 1) * half_cols] = swiglu_sfpu_reference(g, u)
        ref_out = ref_inter.bfloat16().float() @ dn_w_dequant[e]

        for i in range(meta["M_e"]):
            t = meta["token_indices"][i]
            k = meta["k_indices"][i]
            w = topk_weights[t, k].float().item()
            ref_output[t] += w * ref_out[t]

    # ---- Verify PCC ----
    pcc = torch.corrcoef(torch.stack([combined_output.flatten(), ref_output.flatten()]))[0, 1].item()
    max_err = (combined_output - ref_output).abs().max().item()
    mean_err = (combined_output - ref_output).abs().mean().item()

    logger.info(f"[moe_semaphore_chained] N_tokens={N_TOKENS}, K={TOP_K}, " f"experts={NUM_EXPERTS}")
    logger.info(f"[moe_semaphore_chained] PCC={pcc:.6f} max_err={max_err:.6f} " f"mean_err={mean_err:.6f}")

    assert pcc >= 0.96, f"MoE output PCC {pcc:.6f} < 0.96"
    logger.info("[moe_semaphore_chained] PASSED")
