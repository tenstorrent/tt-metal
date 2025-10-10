import pytest
import torch
import ttnn
import os
from loguru import logger

from ....utils.tensor import bf16_tensor
from ....utils.check import assert_quality

from tracy.process_model_log import (
    post_process_ops_log,
    run_device_profiler,
)


def run_test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w, use_bias):
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_dtype = torch.float32

    # torch_input = torch.full((M, K), 1.0, dtype=torch_dtype)
    # weight_input = torch.full((K, N), 3.0, dtype=torch_dtype)
    torch_input = torch.randn((M, K), dtype=torch_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype)
    if use_bias:
        bias_input = torch.randn((1, N), dtype=torch_dtype)

    # Prepare TT tensors
    tt_input = bf16_tensor(torch_input, device=device)
    tt_weight = bf16_tensor(weight_input, device=device)
    if use_bias:
        tt_bias = bf16_tensor(bias_input, device=device)

    with torch.no_grad():
        torch_output = torch_input @ weight_input
        if use_bias:
            torch_output = torch_output + bias_input

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    core_grid = ttnn.CoreCoord(8, 8)

    # This is the optimal single-core config for 4096x4096x4096
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )
    tt_output = ttnn.experimental.minimal_matmul(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias if use_bias else None,
        compute_kernel_config=compute_config,
        config=matmul_config,
    )
    tt_output = ttnn.to_torch(tt_output)
    check_result = assert_quality(torch_output, tt_output)
    return check_result


@pytest.mark.parametrize(
    "M, K, N",
    [(4096, 4096, 4096)],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w, use_bias",
    [(8, 8, 8, 2, 2, True)],
)
def test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w, use_bias):
    check_result = run_test_linear(
        device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w, use_bias
    )
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize("M", [32, 96, 320, 4096])
@pytest.mark.parametrize("K", [32, 96, 320, 4096])
@pytest.mark.parametrize("N", [32, 96, 320, 4096])
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear_padded_sweep(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    [
        (9472, 5120, 1280),
        (9472, 5120, 3456),
        (9472, 3456, 5120),
    ],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear_padded_wan_shapes(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    check_result = run_test_linear(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "M, K, N",
    [(4096, 4096, 4096)],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
def test_linear_ablate_datamovement(device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w):
    device.disable_and_clear_program_cache()  # since env vars aren't captured by hash
    for skip_in0 in [False, True]:
        for skip_in1 in [False, True]:
            for skip_out in [False, True]:
                if skip_in0:
                    os.environ["TT_MM_SKIP_IN0"] = "1"
                if skip_in1:
                    os.environ["TT_MM_SKIP_IN1"] = "1"
                if skip_out:
                    os.environ["TT_MM_SKIP_OUT"] = "1"

                print(f"skip_in0: {skip_in0}, skip_in1: {skip_in1}, skip_out: {skip_out}")
                check_result = run_test_linear(
                    device, M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w
                )
                os.environ.pop("TT_MM_SKIP_IN0", None)
                os.environ.pop("TT_MM_SKIP_IN1", None)
                os.environ.pop("TT_MM_SKIP_OUT", None)


def test_linear_sweep_subblocks(device):
    M, K, N = 4096, 4096, 4096
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_execution_dtype = torch.float32
    torch_dtype = torch.bfloat16

    torch_input = torch.randn((M, K), dtype=torch_dtype).to(torch_execution_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype).to(torch_execution_dtype)

    # Prepare TT tensors
    tt_input = bf16_tensor(torch_input, device=device)
    tt_weight = bf16_tensor(weight_input, device=device)

    with torch.no_grad():
        torch_output = torch_input @ weight_input

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    core_grid = ttnn.CoreCoord(1, 1)
    subblocks = [(1, 4), (4, 1), (2, 2)]
    block_sizes = (1, 2, 4, 8, 16, 32)

    """
    Skip all data movement since we're just testing subblocks here
    """
    import os

    os.environ["TT_MM_SKIP_IN0"] = "1"
    os.environ["TT_MM_SKIP_IN1"] = "1"
    os.environ["TT_MM_SKIP_OUT"] = "1"

    from itertools import product

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        block_sizes, block_sizes, block_sizes, subblocks
    ):
        if subblock_h > M_block_size or subblock_w > N_block_size:
            continue

        logger.info(
            f"Running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}"
        )

        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )
        try:
            tt_output = ttnn.experimental.minimal_matmul(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
            tt_output = ttnn.to_torch(tt_output)
            check_result = assert_quality(torch_output, tt_output)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}: {e}"
            )


@pytest.mark.parametrize(
    "M, K, N",
    [
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 2048),
        (4096, 4096, 1024),
        (4096, 4096, 512),
        (9472, 5120, 1280),
        (9472, 5120, 3456),
        (9472, 3456, 5120),
    ],
)
def test_linear_sweep_blocks(device, M, K, N):
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_execution_dtype = torch.float32
    torch_dtype = torch.bfloat16

    torch_input = torch.randn((M, K), dtype=torch_dtype).to(torch_execution_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype).to(torch_execution_dtype)

    # Prepare TT tensors
    tt_input = bf16_tensor(torch_input, device=device)
    tt_weight = bf16_tensor(weight_input, device=device)

    with torch.no_grad():
        torch_output = torch_input @ weight_input

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    core_grid = ttnn.CoreCoord(8, 8)
    subblocks = [(2, 2)]

    m_block_sizes = [2, 4, 8, 16]
    n_block_sizes = [2, 4, 8, 16]
    k_block_sizes = [4, 8, 16]

    from itertools import product

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        m_block_sizes, k_block_sizes, n_block_sizes, subblocks
    ):
        if (M_block_size < subblock_h) or (N_block_size < subblock_w):
            continue
        logger.info(
            f"Running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}"
        )

        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )
        try:
            tt_output = ttnn.experimental.minimal_matmul(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
            tt_output = ttnn.to_torch(tt_output)
            check_result = assert_quality(torch_output, tt_output)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}: {e}"
            )


TABLE_CONFIGS = [
    (512, 512, 512),
    (512, 1024, 1024),
    (512, 1024, 2048),
    (1024, 1024, 1024),
    (1024, 1024, 2048),
    (1024, 2048, 2048),
    (2048, 2048, 2048),
    (2048, 2048, 3072),
    (2048, 3072, 3072),
    (3072, 3072, 3072),
    (3072, 3072, 4096),
    (3072, 4096, 4096),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16384, 16384, 16384),
]


@pytest.mark.parametrize(
    "M, K, N",
    TABLE_CONFIGS,
)
@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "fp16_acc"])
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4],
    ids=["LoFi", "HiFi2", "HiFi4"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bf16", "bf8b", "bf4b"])
def test_perf_table_sweep(device, M, K, N, fp32_acc, math_fidelity, dtype):
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_execution_dtype = torch.float32
    torch_dtype = torch.bfloat16

    torch_input = torch.randn((M, K), dtype=torch_dtype).to(torch_execution_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype).to(torch_execution_dtype)

    # Prepare TT tensors
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(weight_input, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_input @ weight_input

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    core_grid = ttnn.CoreCoord(8, 8)
    subblocks = [(2, 2)]

    m_block_sizes = [2, 4, 8, 16]
    n_block_sizes = [2, 4, 8, 16]
    k_block_sizes = [2, 4, 8, 16]

    from itertools import product

    for M_block_size, K_block_size, N_block_size, (subblock_h, subblock_w) in product(
        m_block_sizes, k_block_sizes, n_block_sizes, subblocks
    ):
        logger.info(
            f"Running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}"
        )

        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )
        try:
            tt_output = ttnn.experimental.minimal_matmul(
                input_tensor=tt_input,
                weight_tensor=tt_weight,
                bias_tensor=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
            tt_output = ttnn.to_torch(tt_output)
            check_result = assert_quality(torch_output, tt_output)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running minimal_matmul with M_block_size={M_block_size}, K_block_size={K_block_size}, N_block_size={N_block_size}, subblock_h={subblock_h}, subblock_w={subblock_w}"
            )


def perf_model(M, K, N, core_count, fidelity_div):
    mm_flops = 2 * M * K * N
    core_flop_per_cycle = 2 * 8 * 16 * 16
    core_flop_per_cycle_with_fidelity = core_flop_per_cycle / fidelity_div
    chip_flop_per_cycle = core_flop_per_cycle_with_fidelity * core_count
    ideal_cycles = mm_flops / chip_flop_per_cycle
    return ideal_cycles


@pytest.mark.parametrize(
    "fidelity, dtype, fp32_acc",
    [
        ("HiFi2", "bf16", "fp32_acc"),
    ],
)
def test_create_perf_table(fidelity, dtype, fp32_acc):
    fidelity_div = {
        "HiFi2": 2,
        "HiFi4": 4,
        "LoFi": 1,
    }[fidelity]
    perf_results = []
    expected_results = []
    subdir = "ttnn_linear_performance"
    for M, K, N in TABLE_CONFIGS:
        cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
        command = f'pytest models/experimental/tt_dit/tests/unit/performance/test_minimal_matmul.py::test_perf_table_sweep -k "{M}-{K}-{N} and {fp32_acc} and {fidelity} and {dtype}"'

        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
        r = post_process_ops_log(subdir, cols, op_name="", sum_vals=False, has_signposts=False)

        core_count = int(r["CORE COUNT"][0])
        duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())
        expected_ns = perf_model(M, K, N, core_count, fidelity_div)

        perf_results.append(duration_ns)
        expected_results.append(expected_ns)

    # Pretty summary table
    config_details = f"DTYPE: {dtype}, FP32 ACC: {fp32_acc}, FIDELITY: {fidelity}"
    header = "| M, K, N | math util (%) | measured perf (ms) | expected perf at 60% util (ms) |"
    sep = "|---|---:|---:|---:|"
    print(config_details)
    print(header)
    print(sep)
    for idx in range(len(TABLE_CONFIGS)):
        M, K, N = TABLE_CONFIGS[idx]
        measured_ns = perf_results[idx]
        ideal_ns = expected_results[idx]

        if measured_ns is None or ideal_ns is None or measured_ns == 0:
            measured_ms_str = "-"
            util_str = "-"
            expected60_ms_str = "-"
        else:
            measured_ms = measured_ns / 1e6
            # Assume 1 cycle ≈ 1 ns for ideal estimate already returned from perf_model
            math_util = (ideal_ns / measured_ns) * 100.0
            expected60_ms = (ideal_ns / 0.60) / 1e6

            measured_ms_str = f"{measured_ms:.3f}"
            util_str = f"{math_util:.1f}"
            expected60_ms_str = f"{expected60_ms:.3f}"

        print(f"| ({M}, {K}, {N}) | {util_str} | {measured_ms_str} | {expected60_ms_str} |")
