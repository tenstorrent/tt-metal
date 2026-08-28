# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from models.common.utility_functions import (
    is_wormhole_b0,
    is_blackhole,
    skip_for_wormhole_b0,
    skip_for_blackhole,
)
from models.common.utility_functions import torch2tt_tensor, tt2torch_tensor
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.nightly.unit_tests.operations.matmul.utility_functions import ttnn_matmul, ttnn_linear
from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


def pad_to_dram_banks(num, num_banks):
    lcm = 32 * num_banks
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    padded_number = num + padding_needed
    return padded_number


def run_test_matmul_in1_dram_sharded(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    packer_l1_acc,
    has_bias,
    activation,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
    num_workers_per_dram_bank=1,
    N_padded_override=None,
    atol_factor_override=None,
    rtol_factor_override=None,
    pcc_threshold_override=None,
):
    if is_blackhole():
        num_banks = device.dram_grid_size().x  # need to match harvesting of dram
    else:
        num_banks = 12

    # Explicit multi-reader tests prepare storage for the requested count. Automatic
    # tests retain the established bank-only padding so they exercise selection
    # against the caller's existing layout.
    storage_worker_count = num_workers_per_dram_bank if num_workers_per_dram_bank is not None else 1
    N_padded = (
        pad_to_dram_banks(N, num_banks * storage_worker_count) if N_padded_override is None else N_padded_override
    )

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    in1_shard_shape = [K, N_padded // num_banks]
    bias_shape = [1, 1, N]
    bias_shard_shape = [32, N_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    logger.debug("N_padded " + str(N_padded))
    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    logger.debug("in1_shard_shape " + str(in1_shard_shape))
    logger.debug("in1_shard_grid " + str(in1_shard_grid))

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=in0_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in1_mem_config, tt_dtype=in1_dtype)

    if has_bias:
        bias = torch.randn(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, 32 - bias_padded.size(2)), "constant", 0)
        bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        bias_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec
        )
        bias_t = torch2tt_tensor(bias_padded, device, tt_memory_config=bias_mem_config, tt_dtype=ttnn.bfloat16)

    in0_t = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M, int(in0_block_w * 32)],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    if isinstance(activation, str):
        activation_map = {
            "relu": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "gelu": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
            "silu": ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            "sigmoid": ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
        }
        fused_activation = activation_map.get(activation, None)
    else:
        fused_activation = activation

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=fused_activation,
        num_workers_per_dram_bank=num_workers_per_dram_bank,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )

    if has_bias:
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
    output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    # Apply activation if specified
    if activation is not None:
        if activation == "relu" or (hasattr(activation, "op_type") and activation.op_type == ttnn.UnaryOpType.RELU):
            pt_out = torch.nn.functional.relu(pt_out)
        elif activation == "gelu" or (hasattr(activation, "op_type") and activation.op_type == ttnn.UnaryOpType.GELU):
            pt_out = torch.nn.functional.gelu(pt_out)
        elif activation == "silu" or (hasattr(activation, "op_type") and activation.op_type == ttnn.UnaryOpType.SILU):
            pt_out = torch.nn.functional.silu(pt_out)
        elif activation == "sigmoid" or (
            hasattr(activation, "op_type") and activation.op_type == ttnn.UnaryOpType.SIGMOID
        ):
            pt_out = torch.sigmoid(pt_out)

    tt_out = tt2torch_tensor(output_t)

    # Determine tolerances and PCC threshold based on activation type and fidelity
    if activation == "sigmoid":
        atol_factor = 0.01
        rtol_factor = 3.0
        if fidelity == ttnn.MathFidelity.LoFi:
            pcc_threshold = 0.995
        else:
            pcc_threshold = 0.997
    elif activation in ["hardtanh", "hardsigmoid"]:
        atol_factor = 0.008
        rtol_factor = 2.5
        if fidelity == ttnn.MathFidelity.LoFi:
            pcc_threshold = 0.99
        else:
            pcc_threshold = 0.995
    elif activation is not None:
        # Other activations
        atol_factor = 0.005
        rtol_factor = 2.0
        if fidelity == ttnn.MathFidelity.LoFi:
            pcc_threshold = 0.998
        else:
            pcc_threshold = 0.999
    else:
        atol_factor = 0.002
        rtol_factor = 1.062
        pcc_threshold = 0.999

    atol_factor = atol_factor if atol_factor_override is None else atol_factor_override
    rtol_factor = rtol_factor if rtol_factor_override is None else rtol_factor_override
    pcc_threshold = pcc_threshold if pcc_threshold_override is None else pcc_threshold_override

    assert_numeric_metrics(
        pt_out,
        tt_out,
        atol=atol_factor * K,
        rtol=rtol_factor * K,
        frobenius_threshold=0.001 * K,
        pcc_threshold=pcc_threshold,
        check_ulp=False,
    )


def test_matmul_in1_dram_sharded_worker_count_default_is_auto():
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=1,
        per_core_M=1,
        per_core_N=1,
        fused_activation=None,
    )

    assert program_config.num_workers_per_dram_bank is None
    assert "num_workers_per_dram_bank=std::nullopt" in repr(program_config)


@pytest.mark.parametrize(
    "K,N,grid_size",
    [
        # The natural per-bank shard is even, but this small weight stays on the
        # conservative one-reader path.
        (2048, 2048, (8, 2)),
        # A large compressed weight on 64 input cores exercises the two-reader
        # automatic regime without caller-side padding or repacking.
        (8192, 2048, (8, 8)),
        # 33 logical tiles naturally pad to an odd shard width on p100, p150,
        # and Wormhole. Automatic mode must retain one reader.
        (2048, 1056, (1, 1)),
        # Fifteen logical tiles occupy sixteen storage tiles on an eight-bank
        # Blackhole. The shard is even, but a two-reader split would leave a
        # partial final reader, so automatic mode must retain one reader.
        (512, 480, (1, 1)),
    ],
    ids=[
        "small_even_shard_falls_back",
        "large_64_core_weight_uses_two_readers",
        "natural_odd_shard_falls_back",
        "partial_final_reader_falls_back",
    ],
)
def test_matmul_in1_dram_sharded_auto_worker_count(device, K, N, grid_size, function_level_defaults):
    torch.manual_seed(0)
    run_test_matmul_in1_dram_sharded(
        device=device,
        in0_sharded=True,
        out_sharded=True,
        in1_in_dram=True,
        M=32,
        K=K,
        N=N,
        fidelity=ttnn.MathFidelity.HiFi2,
        packer_l1_acc=True,
        has_bias=False,
        activation=None,
        grid_size=grid_size,
        in0_dtype=ttnn.bfloat16,
        in1_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat16,
        function_level_defaults=function_level_defaults,
        num_workers_per_dram_bank=None,
    )


@pytest.mark.parametrize("num_workers_per_dram_bank", [1, 2, 3], ids=["one_worker", "two_workers", "three_workers"])
@pytest.mark.parametrize(
    "N,fidelity,has_bias,activation,grid_size,in1_dtype",
    [
        (1024, ttnn.MathFidelity.HiFi2, True, None, (8, 2), ttnn.bfloat8_b),
        (1024, ttnn.MathFidelity.HiFi2, False, None, (8, 2), ttnn.bfloat8_b),
        # Use eight output-storage cores and 168 logical output tiles. This is
        # divisible by every reader count on both p100 (seven banks) and p150
        # (eight banks), while the eight-core grid also divides K exactly.
        (5376, ttnn.MathFidelity.LoFi, False, "relu", (8, 1), ttnn.bfloat4_b),
    ],
    ids=["bfp8_hifi2_bias", "bfp8_hifi2_no_bias", "bfp4_lofi_padded_relu"],
)
def test_matmul_in1_dram_sharded_worker_counts(
    device,
    N,
    fidelity,
    has_bias,
    activation,
    grid_size,
    in1_dtype,
    num_workers_per_dram_bank,
    function_level_defaults,
):
    if num_workers_per_dram_bank > 1 and not is_blackhole():
        pytest.skip("Multiple DRAM-sharded matmul workers per bank are currently Blackhole-only")

    torch.manual_seed(0)
    run_test_matmul_in1_dram_sharded(
        device=device,
        in0_sharded=True,
        out_sharded=True,
        in1_in_dram=True,
        M=32,
        K=2048,
        N=N,
        fidelity=fidelity,
        packer_l1_acc=True,
        has_bias=has_bias,
        activation=activation,
        grid_size=grid_size,
        in0_dtype=ttnn.bfloat16,
        in1_dtype=in1_dtype,
        out_dtype=ttnn.bfloat16,
        function_level_defaults=function_level_defaults,
        num_workers_per_dram_bank=num_workers_per_dram_bank,
        # Match established BFP4 matmul coverage. The quantized one-worker
        # reference itself is around 0.99 PCC at LoFi for this shape.
        atol_factor_override=0.04 if in1_dtype == ttnn.bfloat4_b else None,
        rtol_factor_override=68.5 if in1_dtype == ttnn.bfloat4_b else None,
        pcc_threshold_override=0.99 if in1_dtype == ttnn.bfloat4_b else None,
    )


@pytest.mark.parametrize(
    "num_workers_per_dram_bank,shard_width_tiles",
    [(2, 8), (3, 12)],
    ids=["two_workers", "three_workers"],
)
def test_matmul_in1_dram_sharded_multi_workers_rejects_oversized_storage(
    device, function_level_defaults, expect_error, num_workers_per_dram_bank, shard_width_tiles
):
    if not is_blackhole():
        pytest.skip("Multiple DRAM-sharded matmul workers per bank are currently Blackhole-only")

    with expect_error(RuntimeError, "requires weight shard width"):
        run_test_matmul_in1_dram_sharded(
            device=device,
            in0_sharded=True,
            out_sharded=True,
            in1_in_dram=True,
            M=32,
            K=2048,
            N=288,
            fidelity=ttnn.MathFidelity.HiFi2,
            packer_l1_acc=True,
            has_bias=False,
            activation=None,
            grid_size=(1, 1),
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat16,
            function_level_defaults=function_level_defaults,
            num_workers_per_dram_bank=num_workers_per_dram_bank,
            # Keep the intentionally oversized shard tile-aligned for each
            # Blackhole variant. p150 has eight banks, while p100 has seven.
            N_padded_override=device.dram_grid_size().x * 32 * shard_width_tiles,
        )


def test_matmul_in1_dram_sharded_explicit_multi_worker_rejects_nondivisible_shard(
    device, function_level_defaults, expect_error
):
    if not is_blackhole():
        pytest.skip("Multiple DRAM-sharded matmul workers per bank are currently Blackhole-only")

    with expect_error(RuntimeError, "must be divisible by workers_per_bank"):
        run_test_matmul_in1_dram_sharded(
            device=device,
            in0_sharded=True,
            out_sharded=True,
            in1_in_dram=True,
            M=32,
            K=2048,
            N=1056,
            fidelity=ttnn.MathFidelity.HiFi2,
            packer_l1_acc=True,
            has_bias=False,
            activation=None,
            grid_size=(1, 1),
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat16,
            function_level_defaults=function_level_defaults,
            num_workers_per_dram_bank=2,
            N_padded_override=device.dram_grid_size().x * 32 * 5,
        )


@pytest.mark.parametrize(
    "fidelity",
    [
        ttnn.MathFidelity.HiFi2,
        ttnn.MathFidelity.LoFi,
    ],
    ids=["HiFi2", "LoFi"],
)
@pytest.mark.parametrize(
    "packer_l1_acc",
    [
        False,
        True,
    ],
    ids=["no_packer_l1_acc", "packer_l1_acc"],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        False,
        True,
    ],
    ids=["no_bias", "bias"],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, grid_size",
    # "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation, grid_size, in0_dtype, in1_dtype, out_dtype",
    [
        (False, True, True, 32, 8192, 1280, None, (8, 1)),
        (False, True, True, 32, 8192, 4096, None, (8, 2)),
        (False, True, True, 32, 8192, 1024, None, (8, 1)),
        (False, True, True, 32, 32768, 1024, None, (8, 2)),
        (False, True, True, 32, 4096, 1280, "relu", (8, 1)),
        (False, True, True, 32, 4096, 1024, "gelu", (8, 1)),
        (False, True, True, 32, 4096, 2048, "silu", (8, 2)),
        (False, True, True, 32, 4096, 1024, "sigmoid", (8, 1)),
        # (False, True, True, 32, 4096, 6144, None, (8, 2), ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        # (False, True, True, 32, 4096, 14336, None, (8, 2), ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b),
        # (False, True, True, 32, 14336, 4096, None, (8, 2), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b),
        # (False, True, True, 32, 4096, 14336, None, (8, 2), ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b),
    ],
)
def test_matmul_in1_dram_sharded_with_program_cache(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    packer_l1_acc,
    has_bias,
    activation,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
):
    torch.manual_seed(0)
    for _ in range(2):
        run_test_matmul_in1_dram_sharded(
            device,
            in0_sharded,
            out_sharded,
            in1_in_dram,
            M,
            K,
            N,
            fidelity,
            packer_l1_acc,
            has_bias,
            activation,
            grid_size,
            in0_dtype,
            in1_dtype,
            out_dtype,
            function_level_defaults,
        )
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        )
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)
    assert device.num_program_cache_entries() == 3


def run_test_matmul_in1_dram_sharded_mm_chain(
    device,
    in0_sharded,
    out_sharded,
    in1_in_dram,
    M,
    K,
    N,
    fidelity,
    has_bias,
    activation,
    grid_size,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
):
    if is_blackhole():
        num_banks = device.dram_grid_size().x  # need to match harvesting of dram
    else:
        num_banks = 12
    N_padded = pad_to_dram_banks(N, num_banks)

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    in1_shard_shape = [K, N_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    logger.debug("N_padded " + str(N_padded))
    logger.debug("in0 block h w " + str(in0_block_h * 32) + " " + str(in0_block_w * 32))
    logger.debug("in1 block h w " + str(in0_block_w * 32) + " " + str(out_block_w * 32))
    logger.debug("out block h w " + str(out_block_h * 32) + " " + str(out_block_w * 32))
    logger.debug("out subblock h w " + str(out_subblock_h * 32) + " " + str(out_subblock_w * 32))

    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_shard_grid = (grid_size[0] - 1, grid_size[1] - 1)
    in0_shard_shape = [M, int(in0_block_w * 32)]
    in0_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in0_shard_grid)})
    in0_shard_spec = ttnn.ShardSpec(in0_shard_grid, in0_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in0_dtype)

    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in1_mem_config, tt_dtype=in1_dtype)

    # Convert string activation to UnaryWithParam if needed
    if isinstance(activation, str):
        activation_map = {
            "relu": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            "gelu": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
            "silu": ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            "sigmoid": ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
        }
        fused_activation = activation_map.get(activation, None)
    else:
        fused_activation = activation

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=fused_activation,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # 1st mm
    output_t = ttnn_matmul(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=sharded_mem_config,
        dtype=out_dtype,
        compute_kernel_config=compute_kernel_config,
    )

    for _ in range(100):
        output_t = ttnn_matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )

    output_t = output_t.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    pt_out = in0 @ in1

    tt_out = tt2torch_tensor(output_t)

    print(tt_out)
    print(pt_out)

    assert_numeric_metrics(pt_out, tt_out, check_allclose=False, check_frobenius=False, check_ulp=False)


@pytest.mark.parametrize(
    "fidelity",
    [
        ttnn.MathFidelity.HiFi2,
    ],
    ids=[
        "HiFi2",
    ],
)
@pytest.mark.parametrize(
    "has_bias",
    [
        False,
    ],
    ids=["no_bias"],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
    ],
)
def test_matmul_in1_dram_sharded_with_mm_chain(
    device,
    fidelity,
    has_bias,
    in0_dtype,
    in1_dtype,
    out_dtype,
    function_level_defaults,
):
    torch.manual_seed(0)
    M = 32
    K = 4096
    N = 4096
    grid_size = (8, 2)
    run_test_matmul_in1_dram_sharded_mm_chain(
        device,
        True,
        True,
        True,
        M,
        K,
        N,
        fidelity,
        has_bias,
        None,
        grid_size,
        in0_dtype,
        in1_dtype,
        out_dtype,
        function_level_defaults,
    )


@pytest.mark.parametrize("packer_l1_acc", [True, False], ids=["pack_l1", "no_pack_l1"])
@pytest.mark.parametrize(
    "fp32_acc_mode",
    [
        True,
    ],
    ids=["fp32"],
)
@pytest.mark.parametrize(
    "fidelity",
    [
        ttnn.MathFidelity.LoFi,
    ],
    ids=["LoFi"],
)
@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize(
    "M, K, N, activation, in0_sharded, fuse_batch",
    [
        (1024, 1024, 1024, None, True, True),
        (1024, 4096, 2048, None, False, False),
    ],
)
def test_matmul_2d_in1_dram_sharded(
    device,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    M,
    K,
    N,
    activation,
    in0_sharded,
    fuse_batch,
    function_level_defaults,
):
    torch.manual_seed(0)
    if is_blackhole():
        num_banks = device.dram_grid_size().x  # need to match harvesting of dram
    else:
        num_banks = 12
    N_padded = pad_to_dram_banks(N, num_banks)

    if fuse_batch:
        in0_shape = [1, 1, M, K]
    else:
        in0_shape = [1, 2, M, K]
    in1_shape = [1, 1, K, N]
    in1_shard_shape = [K, N_padded // num_banks]
    bias_shape = [1, 1, N]
    bias_shard_shape = [32, N_padded // num_banks]
    grid_size = (4, 4)

    in0_block_h = M // grid_size[1] // 32
    in0_block_w = K // grid_size[0] // 32
    out_block_h = M // grid_size[1] // 32
    out_block_w = N // grid_size[0] // 32

    # full block too large to fit in L1
    if in0_block_h * in0_block_w >= 48 or in0_block_w * out_block_w >= 48:
        in0_block_w = in0_block_w // 2

    if out_block_w < 4:
        out_subblock_w = out_block_w
        out_subblock_h = out_block_h // out_subblock_w
    else:
        out_subblock_w = 4
        out_subblock_h = 1

    logger.debug("in0 block w h " + str(in0_block_w * 32) + " " + str(in0_block_h * 32))
    logger.debug("in1 block w h " + str(out_block_w * 32) + " " + str(in0_block_w * 32))
    logger.debug("out block w h " + str(out_block_w * 32) + " " + str(out_block_h * 32))
    logger.debug("out subblock w h " + str(out_subblock_w * 32) + " " + str(out_subblock_h * 32))

    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    interleaved_mem_config_L1 = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttnn.bfloat16)
    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[1], K // grid_size[0]],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    in1 = torch.randn(in1_shape).bfloat16().float()
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in1_mem_config, tt_dtype=ttnn.bfloat16)

    if has_bias:
        bias = torch.ones(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        bias_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec
        )
        bias_t = torch2tt_tensor(bias_padded, device, tt_memory_config=bias_mem_config, tt_dtype=ttnn.bfloat16)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w // 4,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=activation,
        fuse_batch=fuse_batch,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=fp32_acc_mode,
        packer_l1_acc=packer_l1_acc,
    )
    if has_bias:
        output_t = ttnn_linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=sharded_mem_config if in0_sharded else interleaved_mem_config_DRAM,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        output_t = ttnn_matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config if in0_sharded else interleaved_mem_config_DRAM,
            compute_kernel_config=compute_kernel_config,
        )

    if in0_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config_DRAM)
    tt_out = tt2torch_tensor(output_t)

    pt_out = in0 @ in1
    if has_bias:
        pt_out = pt_out + bias

    # Apply activation if specified
    if activation is not None:
        if activation == "relu" or (hasattr(activation, "op_type") and activation.op_type == ttnn.UnaryOpType.RELU):
            pt_out = torch.nn.functional.relu(pt_out)
        elif activation == "gelu" or (hasattr(activation, "op_type") and activation.op_type == ttnn.UnaryOpType.GELU):
            pt_out = torch.nn.functional.gelu(pt_out)
        elif activation == "silu" or (hasattr(activation, "op_type") and activation.op_type == ttnn.UnaryOpType.SILU):
            pt_out = torch.nn.functional.silu(pt_out)
        elif activation == "sigmoid" or (
            hasattr(activation, "op_type") and activation.op_type == ttnn.UnaryOpType.SIGMOID
        ):
            pt_out = torch.sigmoid(pt_out)

    assert_numeric_metrics(pt_out, tt_out, check_allclose=False, check_frobenius=False, check_ulp=False)
