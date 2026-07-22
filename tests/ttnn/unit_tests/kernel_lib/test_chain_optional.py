# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile-time optional and runtime-conditional chain elements.

  - optional_unary.cpp: gate a Negative. ON -> out = -A, OFF -> out = A (inert marker).
  - optional_pack.cpp:  gate a second PackTile (fan-out). ON -> both cb_out0 and cb_out1 written;
    OFF -> only cb_out0, and the tag-less marker must remain neutral in pack planning and emission.
  - runtime_conditional.cpp: exercise grouped when(...) and ordered
    runtime_if(...).else_if(...).otherwise(...) branches.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

OPT_UNARY = "ttnn/cpp/ttnn/kernel_lib/tests/axes/optional_unary.cpp"
OPT_PACK = "ttnn/cpp/ttnn/kernel_lib/tests/axes/optional_pack.cpp"
RUNTIME_CONDITIONAL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/runtime_conditional.cpp"


@pytest.mark.parametrize("cond,name", [(1, "on"), (0, "off")])
def test_optional_unary_gate(device, cond, name):
    """ON applies Negative (out=-A); OFF leaves an inert marker (out=A) and still compiles + runs."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1401)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(OPT_UNARY, [n, cond], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)

    a = torch_in.to(torch.float32)
    golden = -a if cond else a
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"OptionalChainElement unary gate={name} | {msg}")
    assert pcc_ok, f"gate {name}: {msg}"


def test_optional_pack_on_fanout(device):
    """ON: DEST packed to BOTH outputs (fan-out) — both equal the input."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1402)
    tt_o0 = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    tt_o1 = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [
        lib.cb_descriptor(0, dt, 2, core_grid),
        lib.cb_descriptor(16, dt, 2, core_grid),
        lib.cb_descriptor(17, dt, 2, core_grid),
    ]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_2out_kernel([tt_o0, tt_o1], n, core_grid)
    compute = lib.build_compute_kernel(OPT_PACK, [n, 1], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([tt_in, tt_o0, tt_o1], program)

    golden = torch_in.to(torch.float32)
    for tag, t in (("out0", tt_o0), ("out1", tt_o1)):
        out = ttnn.to_torch(t).to(torch.float32)
        ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
        logger.info(f"OptionalChainElement pack ON {tag} | {msg}")
        assert ok, f"{tag}: {msg}"


def test_optional_pack_off_inert_marker(device):
    """OFF: only cb_out0 is written; the optional PackTile marker stays neutral and emits no work."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1403)
    tt_o0 = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_o0, n, core_grid)
    compute = lib.build_compute_kernel(OPT_PACK, [n, 0], core_grid)  # cond=0 -> optional pack is inert

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_o0], program)

    golden = torch_in.to(torch.float32)
    out = ttnn.to_torch(output).to(torch.float32)
    ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"OptionalChainElement pack OFF (inert marker) | {msg}")
    assert ok, msg


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_runtime_conditional(device, mode):
    """The first matching runtime_if arm runs; when guards its whole two-element sequence."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1404)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel_rt(RUNTIME_CONDITIONAL, [n], [mode], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)

    a = torch_in.to(torch.float32)
    golden = {0: -a, 1: -(a * a), 2: torch.abs(a), 3: torch.abs(-a)}[mode]
    out = ttnn.to_torch(output).to(torch.float32)
    ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"runtime conditional mode={mode} | {msg}")
    assert ok, msg
