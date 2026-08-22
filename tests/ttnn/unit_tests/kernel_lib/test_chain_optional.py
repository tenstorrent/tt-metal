# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile-time optional and runtime-conditional chain elements.

  - compile-time unary gate: ON -> out = -A, OFF -> out = A (inert marker).
  - compile-time pack gate: gate a second PackTile (fan-out). ON -> both cb_out0 and cb_out1 written;
    OFF -> only cb_out0, and the tag-less marker must remain neutral in pack planning and emission.
  - runtime conditional: exercise bare runtime_if(...), runtime_if(...).else_if(...), and explicit
    .otherwise(...) branches.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/optional.cpp"


def _run_optional_unary(device, enabled):
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1401)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n, 0, int(enabled)], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)
    return torch_in.to(torch.float32), ttnn.to_torch(output).to(torch.float32)


def test_optional_unary_gate(device):
    """One paired check proves both the enabled operation and the disabled marker's neutrality."""
    outputs = {}
    for enabled in (False, True):
        a, out = _run_optional_unary(device, enabled)
        golden = -a if enabled else a
        pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
        logger.debug(f"Optional unary gate enabled={enabled} | {msg}")
        assert pcc_ok, f"gate enabled={enabled}: {msg}"
        outputs[enabled] = out
    assert torch.equal(outputs[True], -outputs[False])


def _run_optional_pack(device, enabled):
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1402)
    tt_o0 = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    outputs = [tt_o0]
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    if enabled:
        outputs.append(
            ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        )
        cbs.append(lib.cb_descriptor(17, dt, 2, core_grid))
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = (
        lib.build_writer_2out_kernel(outputs, n, core_grid)
        if enabled
        else lib.build_writer_1out_kernel(tt_o0, n, core_grid)
    )
    compute = lib.build_compute_kernel(KERNEL, [n, 1, int(enabled)], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([tt_in, *outputs], program)
    return torch_in.to(torch.float32), [ttnn.to_torch(t).to(torch.float32) for t in outputs]


def test_optional_pack_gate(device):
    """The same paired test covers disabled-marker neutrality and enabled fan-out."""
    primary = {}
    for enabled in (False, True):
        golden, outputs = _run_optional_pack(device, enabled)
        for index, out in enumerate(outputs):
            assert torch.equal(golden, out), f"optional pack enabled={enabled} output={index} changed data"
        primary[enabled] = outputs[0]
    assert torch.equal(primary[False], primary[True])


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5, 6])
def test_runtime_conditional(device, mode):
    """The first matching runtime_if arm runs; an unmatched conditional is inert."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1404)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel_rt(KERNEL, [n, 2, 0], [mode], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)

    a = torch_in.to(torch.float32)
    golden = {
        0: -a,
        1: -(a * a),
        2: torch.abs(a),
        3: torch.abs(-a),
        4: a * a,
        5: a * a,
        6: -a,
    }[mode]
    out = ttnn.to_torch(output).to(torch.float32)
    ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.debug(f"runtime conditional mode={mode} | {msg}")
    assert ok, msg
