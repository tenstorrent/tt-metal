# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Functional tests for the eltwise-chain packer-ReLU PackTile knob (PackRelu::Zero)."""

import pytest
import torch
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/relu/pack_relu.cpp"

MODE_BASIC = 0
MODE_ESCAPE = 1
MODE_EXP_RELU = 2
MODE_EXP_PLAIN = 3
MODE_MIXED = 4


def _alloc_out(shape, dt, device):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)


def test_pack_relu_basic(device):
    """copy -> pack(relu): the packer clamps negatives to zero. out == relu(in0)."""
    n = 8
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1701, scale=1.0, bias=0.0)
    tt_out = _alloc_out(shape, dt, device)
    assert (torch_in.to(torch.float32) < 0).any(), "input must contain negatives to exercise the clamp"

    cbs = [lib.cb_descriptor(0, dt, 2, cg), lib.cb_descriptor(16, dt, 2, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(KERNEL, [n, MODE_BASIC], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    output = ttnn.generic_op([tt_in, tt_out], program)
    golden = torch.clamp(torch_in.to(torch.float32), min=0.0)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, 0.999)
    logger.info(f"pack_relu basic | {msg}")
    assert pcc_ok, msg


def test_pack_relu_escape_reset(device):
    """A relu chain must restore packer pass-through so a following linear pack is NOT clamped."""
    n = 8
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()

    _, tt_in0 = lib.make_input(shape, dt, device, seed=1701, scale=1.0, bias=0.0)
    torch_in1, tt_in1 = lib.make_input(shape, dt, device, seed=2027, scale=1.0, bias=0.0)
    tt_out = _alloc_out(shape, dt, device)
    assert (torch_in1.to(torch.float32) < 0).any(), "in1 must contain negatives that must survive (not be clamped)"

    cbs = [
        lib.cb_descriptor(0, dt, 2, cg),
        # The shared reader fills c_0 and c_1 in lockstep, but the compute drains c_0 (chain 1) fully
        # before touching c_1 (chain 2). c_1 must hold all of in1 meanwhile, else the single-threaded
        # reader blocks on c_1 and starves chain 1. Same reason the relu sink c_2 needs n pages.
        lib.cb_descriptor(1, dt, n, cg),
        lib.cb_descriptor(2, dt, n, cg),  # dead sink for the relu chain (nobody pops it -> needs n pages)
        lib.cb_descriptor(16, dt, 2, cg),
    ]
    reader = lib.build_reader_kernel([tt_in0, tt_in1], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(KERNEL, [n, MODE_ESCAPE], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    output = ttnn.generic_op([tt_in0, tt_in1, tt_out], program)
    golden = torch_in1.to(torch.float32)  # linear pack: in1 unchanged, negatives preserved
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, 0.999)
    logger.info(f"pack_relu escape reset | {msg}")
    assert pcc_ok, msg


def test_pack_relu_exp_ab(device):
    """copy -> exp -> pack, WITH vs WITHOUT packer relu. exp>0 so relu is a no-op: both == exp(in0)."""
    n = 8
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1701, scale=1.0, bias=0.0)
    assert (torch_in.to(torch.float32) < 0).any(), "input has negatives, but exp maps them to positive outputs"
    golden = torch.exp(torch_in.to(torch.float32))

    def run(mode):
        tt_out = _alloc_out(shape, dt, device)
        cbs = [lib.cb_descriptor(0, dt, 2, cg), lib.cb_descriptor(16, dt, 2, cg)]
        reader = lib.build_reader_kernel([tt_in], n, cg)
        writer = lib.build_writer_1out_kernel(tt_out, n, cg)
        compute = lib.build_compute_kernel(KERNEL, [n, mode], cg)
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        return ttnn.to_torch(ttnn.generic_op([tt_in, tt_out], program)).to(torch.float32)

    out_relu = run(MODE_EXP_RELU)
    out_plain = run(MODE_EXP_PLAIN)

    # ReLU on exp output is a no-op: the two variants must be bit-identical to each other.
    assert torch.equal(out_relu, out_plain), "packer ReLU changed exp(x)>0 output — relu is not a pass-through!"

    ok_relu, msg_relu = comp_pcc(golden, out_relu, 0.99)
    ok_plain, msg_plain = comp_pcc(golden, out_plain, 0.99)
    logger.info(f"exp A/B | relu: {msg_relu} | plain: {msg_plain}")
    assert ok_relu, msg_relu
    assert ok_plain, msg_plain


def test_pack_relu_mixed(device):
    """One heterogeneous chain: a relu pack (in0 -> c_16) and a linear pack (in1 -> c_17). The
    per-pack set/restore must clamp ONLY the relu site: out16 == relu(in0), out17 == in1 unchanged."""
    n = 8
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()

    torch_in0, tt_in0 = lib.make_input(shape, dt, device, seed=1701, scale=1.0, bias=0.0)
    torch_in1, tt_in1 = lib.make_input(shape, dt, device, seed=2027, scale=1.0, bias=0.0)
    tt_o0 = _alloc_out(shape, dt, device)
    tt_o1 = _alloc_out(shape, dt, device)
    assert (torch_in0.to(torch.float32) < 0).any() and (torch_in1.to(torch.float32) < 0).any()

    cbs = [
        lib.cb_descriptor(0, dt, 2, cg),
        lib.cb_descriptor(1, dt, 2, cg),
        lib.cb_descriptor(16, dt, 2, cg),
        lib.cb_descriptor(17, dt, 2, cg),
    ]
    reader = lib.build_reader_kernel([tt_in0, tt_in1], n, cg)
    writer = lib.build_writer_2out_kernel([tt_o0, tt_o1], n, cg)
    compute = lib.build_compute_kernel(KERNEL, [n, MODE_MIXED], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    ttnn.generic_op([tt_in0, tt_in1, tt_o0, tt_o1], program)
    out16 = ttnn.to_torch(tt_o0).to(torch.float32)
    out17 = ttnn.to_torch(tt_o1).to(torch.float32)

    ok_relu, msg_relu = comp_pcc(torch.clamp(torch_in0.to(torch.float32), min=0.0), out16, 0.999)
    ok_lin, msg_lin = comp_pcc(torch_in1.to(torch.float32), out17, 0.999)
    logger.info(f"mixed | relu site: {msg_relu} | linear site: {msg_lin}")
    assert ok_relu, f"relu pack site wrong: {msg_relu}"
    assert ok_lin, f"linear pack site clamped (per-pack restore failed): {msg_lin}"
