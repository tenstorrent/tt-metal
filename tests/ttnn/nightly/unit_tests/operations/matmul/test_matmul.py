# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import math
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


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
@pytest.mark.timeout(120)
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
    ((1024, 1280, 5120, 4, 1, 8, 4, 32, True),),
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
