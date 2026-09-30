# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf research (not for merge): packer L1 accumulate of BF16-dest tiles into an FP32 (or BF16) tile."""

import os

import pytest
import torch
import ttnn

K = "tests/ttnn/unit_tests/operations/sdpa/kernels/"
DF = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"


@pytest.mark.skipif(os.getenv("TEST_SDPA_PERF_L1ACC") != "1", reason="Opt-in perf research")
@pytest.mark.parametrize("fp32_dest", [False, True])
@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("n", [16, 256])
def test_packer_l1acc_kernel(device, n, out_dtype, fp32_dest):
    torch.manual_seed(0)
    x = torch.rand(1, 1, 32, 32 * n).bfloat16()
    ref = x.double().reshape(32, n, 32).sum(1)
    tin = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), out_dtype, ttnn.TILE_LAYOUT, device)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    in_page = 2048
    out_page = 4096 if out_dtype == ttnn.float32 else 2048
    out_fmt = ttnn.float32 if out_dtype == ttnn.float32 else ttnn.bfloat16
    cbs = [
        ttnn.CBDescriptor(total_size=2 * in_page, core_ranges=core,
                          format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=in_page)]),
        ttnn.CBDescriptor(total_size=out_page, core_ranges=core,
                          format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=out_fmt, page_size=out_page)]),
    ]
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rr[0][0] = [tin.buffer_address(), n, 0]
    wr[0][0] = [tout.buffer_address(), 1, 0]
    cr[0][0] = []
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.fp32_dest_acc_en = fp32_dest
    kernels = [
        ttnn.KernelDescriptor(kernel_source=DF + "reader_unary_interleaved_start_id.cpp", core_ranges=core,
                              compile_time_args=ttnn.TensorAccessorArgs(tin).get_compile_time_args(),
                              runtime_args=rr, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=DF + "writer_unary_interleaved_start_id.cpp", core_ranges=core,
                              compile_time_args=[16] + ttnn.TensorAccessorArgs(tout).get_compile_time_args(),
                              runtime_args=wr, config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=K + "l1acc_probe.cpp", core_ranges=core, compile_time_args=[n],
                              runtime_args=cr, config=cfg),
    ]
    ttnn.generic_op([tin, tout], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    got = ttnn.to_torch(tout).double().reshape(32, 32)
    bf16_running = torch.zeros(32, 32).bfloat16()
    for i in range(n):
        bf16_running = (bf16_running.float() + x.reshape(32, n, 32)[:, i].float()).bfloat16()
    rel = ((got - ref).norm() / ref.norm()).item()
    rel_bf16 = ((bf16_running.double() - ref).norm() / ref.norm()).item()
    print(f"L1ACCK n={n} out={out_dtype} fp32_dest={fp32_dest} rel_l2={rel:.3e} (bf16 running sum {rel_bf16:.3e})")
