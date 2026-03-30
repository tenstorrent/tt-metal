# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.matmul.test_matmul_deepseek import _run_matmul_2d_interleaved_in0_sharded_in1


@pytest.mark.parametrize(
    "batch_size, channel_a, channel_b, m_size, k_size, n_size, has_bias",
    [
        (1, 2, 1, 1024, 640, 2560, False),
        (2, 8, 8, 64, 96, 160, False),
        (1, 2, 1, 4096, 320, 1280, False),
        (1, 2, 1, 64, 1280, 5120, False),
        (2, 8, 8, 64, 64, 160, False),
        (1, 2, 1, 1024, 640, 768, False),
        (2, 8, 8, 96, 160, 96, False),
        (2, 8, 8, 1024, 1024, 96, False),
        (1, 2, 1, 96, 768, 1024, False),
        (1, 1, 1, 32, 1280, 1280, True),
        (2, 8, 8, 4096, 96, 64, False),
        (1, 2, 1, 64, 5120, 1280, True),
        (2, 8, 8, 4096, 64, 96, False),
        (1, 2, 1, 1024, 768, 640, True),
        (1, 2, 1, 256, 1280, 1280, True),
        (2, 8, 8, 1024, 96, 96, False),
        (1, 2, 1, 1024, 640, 2304, False),
        (1, 1, 1, 32, 1280, 320, True),
        (1, 2, 1, 96, 768, 2560, False),
        (1, 2, 1, 4096, 1280, 320, True),
        (1, 2, 1, 1024, 2560, 640, True),
        (1, 2, 1, 256, 1280, 3840, False),
        (1, 1, 1, 32, 320, 1280, True),
        (1, 2, 1, 4096, 512, 320, True),
        (1, 2, 1, 64, 1280, 1280, True),
        (1, 2, 1, 256, 5120, 1280, True),
        (1, 2, 1, 256, 1280, 1280, False),
        (2, 8, 8, 256, 160, 96, False),
        (2, 8, 8, 256, 256, 160, False),
        (1, 2, 1, 96, 768, 1536, False),
        (1, 2, 1, 64, 1280, 3840, False),
        (2, 8, 8, 1024, 96, 1024, False),
        (2, 8, 8, 256, 96, 160, False),
        (1, 2, 1, 64, 1280, 1280, False),
        (2, 8, 8, 4096, 64, 4096, False),
        (1, 1, 1, 32, 1280, 640, True),
        (2, 8, 8, 64, 160, 64, False),
        (1, 2, 1, 4096, 320, 1536, False),
        (1, 2, 1, 256, 1280, 5120, False),
        (2, 8, 8, 4096, 4096, 64, False),
        (2, 8, 8, 256, 160, 256, False),
        (1, 2, 1, 4096, 320, 512, False),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
@pytest.mark.timeout(600)
def test_sd_matmul(device, batch_size, channel_a, channel_b, m_size, k_size, n_size, has_bias, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")
    core_grid = ttnn.CoreGrid(x=8, y=8)
    TILE_HEIGHT = 32

    if batch_size == 2:
        if (m_size == 1024 and k_size == 96 and n_size == 1024) or (m_size == 4096 and k_size == 64 and n_size == 4096):
            # NOTE: matmul errors out with OOM otherwise
            core_grid = None

    torch_input_tensor_a = torch.randn((batch_size, channel_a, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((batch_size, channel_b, k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if has_bias:
        torch_input_tensor_c = torch.randn((1, 1, TILE_HEIGHT, n_size), dtype=torch.bfloat16)
        _torch_input_tensor_c = torch.repeat_interleave(
            torch_input_tensor_c, torch_output_tensor.shape[2] // TILE_HEIGHT, dim=2
        )
        torch_output_tensor = torch_output_tensor + _torch_input_tensor_c

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_c = (
        ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype) if has_bias else None
    )
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    if has_bias:
        output_tensor = ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=input_tensor_c,
            core_grid=core_grid,
        )
    else:
        output_tensor = ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            core_grid=core_grid,
        )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


@pytest.mark.parametrize("core_grid", [ttnn.CoreGrid(y=8, x=5)])
@pytest.mark.parametrize(
    "M, K, N, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, has_gelu",
    [
        (1024, 1280, 5120, 4, 1, 8, 4, 32, True),
        (256, 1280, 5120, 4, 1, 8, 1, 32, False),
        (1024, 640, 2560, 4, 1, 8, 4, 16, False),
    ],
)
def test_sdxl_matmul(
    device,
    core_grid,
    M,
    K,
    N,
    in0_block_w,
    out_subblock_h,
    out_subblock_w,
    per_core_M,
    per_core_N,
    has_gelu,
    perf_test_mode=False,
):
    torch.manual_seed(0)

    act_shape = (1, 1, M, K)
    weights_shape = (1, 1, K, N)
    bias_shape = (1, 1, N)

    torch_act = torch.randn(act_shape, dtype=torch.bfloat16)
    torch_weights = torch.randn(weights_shape, dtype=torch.bfloat16)
    torch_bias = torch.randn(bias_shape, dtype=torch.bfloat16)

    tt_act = ttnn.from_torch(torch_act, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    tt_weights = ttnn.from_torch(torch_weights, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)
    tt_bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)

    if not perf_test_mode:
        torch_output_tensor = torch_act @ torch_weights + torch_bias
        if has_gelu:
            torch_output_tensor = torch.nn.functional.gelu(torch_output_tensor)

    sharded_mem_config = ttnn.create_sharded_memory_config(
        act_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_act_block_sharded = ttnn.to_memory_config(tt_act, sharded_mem_config)
    prog_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fuse_batch=True,
        fused_activation=[ttnn.UnaryOpType.GELU, True] if has_gelu else None,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    output_tensor = ttnn.linear(
        tt_act_block_sharded,
        tt_weights,
        bias=tt_bias,
        program_config=prog_config,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.synchronize_device(device)

    if not perf_test_mode:
        output_tensor = ttnn.to_torch(output_tensor)
        assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@pytest.mark.parametrize("batch", [1, 25])
@pytest.mark.parametrize("seq_len", [63, 5000])
@pytest.mark.parametrize("k", [63, 32])
@pytest.mark.parametrize("n", [63, 32])
@pytest.mark.parametrize("num_dram_banks", [8, 9, 11, 12])
@pytest.mark.timeout(120)
def test_matmul_2d_interleaved_sharded_dimension_sweep(device, batch, seq_len, k, n, num_dram_banks):
    """
    Sweep test for 2D matmul with DRAM interleaved in0 and DRAM sharded in1
    across various tensor dimension combinations and DRAM bank counts.
    Exercises both batched (HEIGHT sharded) and unbatched (WIDTH sharded) paths.
    """
    _run_matmul_2d_interleaved_in0_sharded_in1(
        device=device,
        batch=batch,
        seq_len=seq_len,
        k=k,
        n=n,
        in0_dtype=ttnn.bfloat16,
        in1_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat16,
        has_bias=False,
        num_dram_banks=num_dram_banks,
        expected_pcc=0.99,
    )


@pytest.mark.parametrize(
    "batch, seq_len, k, n, num_dram_banks",
    [
        (1, 512, 128, 128, 12),
    ],
)
@pytest.mark.parametrize("in0_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize(
    "has_bias, bias_dtype",
    [
        (False, None),
        (True, ttnn.bfloat16),
        (True, ttnn.bfloat8_b),
        (True, ttnn.float32),
    ],
)
@pytest.mark.timeout(120)
def test_matmul_2d_interleaved_sharded_dtype_bias_sweep(
    device, batch, seq_len, k, n, num_dram_banks, in0_dtype, in1_dtype, out_dtype, has_bias, bias_dtype
):
    """
    Sweep test for 2D matmul with DRAM interleaved in0 and DRAM sharded in1
    across various data type combinations for inputs, outputs, and bias,
    restricted to batch == 1 so that ttnn.Linear() can be used with bias.
    """
    expected_pcc = 0.99

    _run_matmul_2d_interleaved_in0_sharded_in1(
        device=device,
        batch=batch,
        seq_len=seq_len,
        k=k,
        n=n,
        in0_dtype=in0_dtype,
        in1_dtype=in1_dtype,
        out_dtype=out_dtype,
        has_bias=has_bias,
        bias_dtype=bias_dtype if has_bias else None,
        num_dram_banks=num_dram_banks,
        expected_pcc=expected_pcc,
    )


@pytest.mark.parametrize(
    "batch, m, k, n, program_config",
    [
        pytest.param(
            256,
            32,
            128,
            128,
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=1,
                out_subblock_h=2,
                out_subblock_w=2,
                per_core_M=8,
                per_core_N=4,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ),
            id="1d_mcast_in1_fuse_batch",
        ),
        pytest.param(
            8,
            32,
            128,
            128,
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(2, 1),
                in0_block_w=1,
                out_subblock_h=2,
                out_subblock_w=2,
                per_core_M=8,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            ),
            id="1d_mcast_in0_fuse_batch",
        ),
        pytest.param(
            64,
            32,
            128,
            128,
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(2, 8),
                in0_block_w=1,
                out_subblock_h=2,
                out_subblock_w=2,
                per_core_M=8,
                per_core_N=2,
                transpose_mcast=False,
                fuse_batch=True,
            ),
            id="2d_mcast_fuse_batch",
        ),
    ],
)
@pytest.mark.timeout(120)
def test_matmul_transpose_a_fuse_batch(device, batch, m, k, n, program_config):
    """
    Regression test for transpose_a stride calculation with fuse_batch enabled
    in 1D mcast (both mcast_in0 and mcast_in1) and 2D mcast program configs.

    When batches are fused into M and transpose_a is True, the reader kernel
    strides must use M_per_batch (not the fused M) to correctly traverse tiles.
    Program configs are specified explicitly to guard against future changes in
    automatic config selection.
    """
    torch.manual_seed(0)

    torch_a = torch.randn((batch, m, k), dtype=torch.bfloat16)
    torch_b = torch.randn((1, k, n), dtype=torch.bfloat16)
    torch_out = torch.matmul(torch_a, torch_b)

    torch_a_phys = torch_a.transpose(-1, -2)
    torch_b_phys = torch_b.transpose(-1, -2)

    a = ttnn.from_torch(torch_a_phys, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b_phys, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    output = ttnn.matmul(
        a,
        b,
        transpose_a=True,
        transpose_b=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    output = ttnn.to_torch(output)

    assert output.shape == torch_out.shape
    assert_with_pcc(torch_out, output, pcc=0.999)
