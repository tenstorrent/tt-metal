# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import pytest
import torch

import ttnn


from models.utility_functions import pad_by_zero, torch2tt_tensor, comp_pcc, is_grayskull


def ref_layernorm(x, gamma, beta, eps):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, eps)


def ref_rmsnorm(x, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma + beta


def run_layernorm_mix_precision_tests(test_id, in_dtype, gamma_dtype, in0_mem_config, out_mem_config, device):
    epsf = 1e-2

    test_dims = ((1, 9, 384, 1024),)
    for test_shape in test_dims:
        in0 = torch.rand(test_shape) * 2 - 0.95
        in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)

        if test_id <= 5:
            in1 = torch.rand(test_shape) * 2 - 0.8
            in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)

        if test_id % 3 == 0:
            gamma = torch.ones(test_shape[3])
            beta = torch.zeros(test_shape[3])
        if test_id % 3 == 1:
            gamma = torch.rand(test_shape[3]) * 2 - 1
            beta = torch.zeros(test_shape[3])
        if test_id % 3 == 2:
            gamma = torch.rand(test_shape[3]) * 2 - 1
            beta = torch.rand(test_shape[3]) * 2.0 - 1.1

        gamma_t = pad_by_zero(gamma, device, in0_mem_config, gamma_dtype)[0]
        beta_t = pad_by_zero(beta, device, in0_mem_config, gamma_dtype)[0]

        if not is_grayskull():
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=True,
                fp32_dest_acc_en=True if in_dtype == ttnn.float32 else False,
            )

        if test_id == 0:
            ttz = ttnn.layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 1:
            ttz = ttnn.layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 2:
            ttz = ttnn.layer_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 3:
            ttz = ttnn.rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 4:
            ttz = ttnn.rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 5:
            ttz = ttnn.rms_norm(
                in0_t,
                residual_input_tensor=in1_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 6:
            ttz = ttnn.layer_norm(
                in0_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 7:
            ttz = ttnn.layer_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 8:
            ttz = ttnn.layer_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 9:
            ttz = ttnn.rms_norm(
                in0_t,
                epsilon=epsf,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 10:
            ttz = ttnn.rms_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )
        if test_id == 11:
            ttz = ttnn.rms_norm(
                in0_t,
                epsilon=epsf,
                weight=gamma_t,
                bias=beta_t,
                memory_config=out_mem_config,
                compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
            )

        tt_got_back = ttz.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        pt_in = in0 + in1 if test_id <= 5 else in0
        if test_id <= 2 or 6 <= test_id <= 8:
            ref_fn = ref_layernorm
        else:
            ref_fn = ref_rmsnorm

        ref_lnorm = ref_fn(pt_in, gamma.flatten(), beta.flatten(), epsf)

        passing, output = comp_pcc(ref_lnorm, tt_got_back)

        assert passing, output


@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_L1",
    ],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_L1",
    ],
)
@pytest.mark.parametrize(
    "gamma_dtype",
    (ttnn.bfloat16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=["FLOAT32", "BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    ids=[
        "add_LN",
        "add_LN_G",
        "add_LN_GB",
        "add_RMSN",
        "add_RMSN_G",
        "add_RMSN_GB",
        "LN",
        "LN_G",
        "LN_GB",
        "RMSN",
        "RMSN_G",
        "RMSN_GB",
    ],
)
def test_layernorm_mix_precision(test_id, in_dtype, gamma_dtype, in0_mem_config, out_mem_config, device):
    if is_grayskull() and in_dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    run_layernorm_mix_precision_tests(test_id, in_dtype, gamma_dtype, in0_mem_config, out_mem_config, device)


@pytest.mark.parametrize("h", [1632, 8192, 16384])
@pytest.mark.parametrize("w", [1280])
@pytest.mark.parametrize("num_chunks", [1, 4])
def test_layer_norm_4D_llama(device, h, w, num_chunks):
    """
    Test specific shapes for LLama and other LLMs
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, num_chunks, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    passing, output = comp_pcc(torch_output_tensor, output_tensor)
    assert passing, output
