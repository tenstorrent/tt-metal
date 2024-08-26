# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from models.utility_functions import tt2torch_tensor, torch2tt_tensor, skip_for_grayskull

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc


def reference(x, n_devices, is_rmsnorm):
    S = x[0].shape[2]
    B = x[0].shape[0]
    sumxs = []
    sumx2s = []
    # Distributed processing
    for chunk in x:
        sumx_local = torch.sum(chunk, dim=-1, keepdim=True)
        sumx2_local = torch.sum(torch.square(chunk), dim=-1, keepdim=True)
        sumxs.append(sumx_local)
        sumx2s.append(sumx2_local)

    # pad with zeros as for tiles
    output = []
    for i in range(n_devices):
        if is_rmsnorm:
            output.append(torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31])], dim=-1))
        else:
            output.append(
                torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31]), sumxs[i], torch.zeros([B, 1, S, 31])], dim=-1)
            )

    return output


def referencefp32(x, n_devices, is_rmsnorm):
    S = x[0].shape[2]
    B = x[0].shape[0]
    sumxs = []
    sumx2s = []
    # Distributed processing
    for chunk in x:
        chunk = chunk.to(torch.float32)
        count_local = chunk.shape[-1]
        sumx_local = torch.sum(chunk, dim=-1, keepdim=True)
        sumx2_local = torch.sum(torch.square(chunk), dim=-1, keepdim=True)

        sumxs.append(sumx_local)
        sumx2s.append(sumx2_local)

    # pad with zeros as for tiles
    output = []
    for i in range(n_devices):
        if is_rmsnorm:
            output.append(torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31])], dim=-1))
        else:
            output.append(
                torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31]), sumxs[i], torch.zeros([B, 1, S, 31])], dim=-1)
            )

    return output


def ln_pre_allgather_op(xs, n_devices, is_rmsnorm, out_dtpe):
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    tt_out = []
    for d in range(n_devices):
        if is_rmsnorm:
            tt_out.append(ttnn.rms_norm_pre_all_gather(xs[d], compute_kernel_config=kernel_config, dtype=out_dtpe))
        else:
            tt_out.append(ttnn.layer_norm_pre_all_gather(xs[d], compute_kernel_config=kernel_config, dtype=out_dtpe))
    return tt_out


def run_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device):
    # Set print options
    torch.set_printoptions(threshold=100)

    torch.manual_seed(42)

    if input_dtype == ttnn.float32:
        canon_inp = torch.randn(inp_shape)
    else:
        canon_inp = torch.randn(inp_shape).bfloat16()

    # Get per-chunk inputs
    inp_chunked = canon_inp.chunk(n_devices, dim=-1)

    # Reference
    out_torch = reference(inp_chunked, n_devices, is_rmsnorm)
    out_torch = torch.concat(out_torch, -1)

    # out_torchfp32 = referencefp32(inp_chunked, n_devices, is_rmsnorm)
    # out_torchfp32 = torch.concat(out_torchfp32, -1)

    dram_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    tt_inp = []
    for d in range(n_devices):
        tt_inp.append(
            torch2tt_tensor(
                inp_chunked[d],
                tt_dtype=input_dtype,
                tt_device=device,
                tt_layout=ttnn.TILE_LAYOUT,
                tt_memory_config=dram_memcfg,
            )
        )

    # LN pre all gather OP
    tt_out = ln_pre_allgather_op(tt_inp, n_devices, is_rmsnorm, output_dtype)
    tt_output_host = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_out], -1)

    all_passing = True

    reduction_width = inp_shape[-1] // n_devices

    for i in range(n_devices):
        device_offset = i * 32 if is_rmsnorm else i * 64
        # Compare sum(xˆ2)

        # print("out_torch sum(x^2):")
        # print(out_torch[0, 0, :64, 0 + device_offset])

        # print("tt_output_host sum(x^2):")
        # print(tt_output_host[0, 0, :64, 0 + device_offset])

        passing, output_str = comp_allclose_and_pcc(
            out_torch[:, :, :, 0 + device_offset],
            tt_output_host[:, :, :, 0 + device_offset],
            rtol=1e-01 * reduction_width,  # Issue 9908: large error in reduce, set new rtol target when fixed!
            atol=0,
            pcc=0.9,
        )
        logger.debug(f"tt vs torch sum(xˆ2) = {output_str}")

        all_passing &= passing

        # Check if zeros are same
        passing, output_str = comp_equal(
            out_torch[:, :, :, 1 + device_offset : 32 + device_offset],
            tt_output_host[:, :, :, 1 + device_offset : 32 + device_offset],
        )
        logger.debug(f"tt vs torch padding 1 = {output_str}")

        all_passing &= passing

        if not is_rmsnorm:
            # Compare sum(x)
            # print("out_torch sum(x):")
            # print(out_torch[0, 0, :64, 32 + device_offset])

            # print("tt_output_host sum(x):")
            # print(tt_output_host[0, 0, :64, 32 + device_offset])

            passing, output_str = comp_allclose_and_pcc(
                out_torch[:, :, :, 32 + device_offset],
                tt_output_host[:, :, :, 32 + device_offset],
                rtol=5e-01
                * reduction_width,  # Issue 9908: large error in reduce, set new rtol and atol target when fixed!
                atol=10,
                pcc=0.97,
            )
            logger.debug(f"tt vs torch sum(x) = {output_str}")

            all_passing &= passing

            # Check if zeros are same
            passing, output_str = comp_equal(
                out_torch[:, :, :, 33 + device_offset : 64 + device_offset],
                tt_output_host[:, :, :, 33 + device_offset : 64 + device_offset],
            )
            logger.debug(f"tt vs torch padding 2 = {output_str}")

            all_passing &= passing

    assert all_passing


@skip_for_grayskull("Requires wormhole")
@pytest.mark.parametrize(
    "input_dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 32, 8192),
        (1, 1, 128, 8192),
        (1, 1, 2048, 8192),
        (1, 1, 8192, 8192),
        (2, 1, 128, 8192),
        (1, 1, 128, 2048),
    ],
)
@pytest.mark.parametrize(
    "n_devices",
    [4, 8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
def test_layernorm_part_1_with_program_cache(
    inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device, use_program_cache
):
    run_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device)


@skip_for_grayskull("Requires wormhole")
@pytest.mark.parametrize(
    "input_dtype",
    [ttnn.bfloat16],
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16],
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 2048, 8192),
    ],
)
@pytest.mark.parametrize(
    "n_devices",
    [8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
def test_layernorm_part_1_with_program_cache2(
    inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device, use_program_cache
):
    dummy_tensors = []

    dram_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    for i in range(2):
        if i > 0:
            dummy_tensors.append(
                torch2tt_tensor(
                    torch.randn(inp_shape),
                    tt_dtype=input_dtype,
                    tt_device=device,
                    tt_layout=ttnn.TILE_LAYOUT,
                    tt_memory_config=dram_memcfg,
                )
            )
        run_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device)

    assert device.num_program_cache_entries() == 1, "Program cache should have only one entry" + str(
        device.num_program_cache_entries()
    )
