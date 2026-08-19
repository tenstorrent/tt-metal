# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Lifecycle & CB-synchronization (the hang suite). Run under --dev.

A lifecycle is the (WaitPolicy, PopPolicy) pair an input declares — whether the chain or the CALLER
emits cb_wait_front / cb_pop_front. A miscount deadlocks the device.

held_b.cpp computes out[i] = A[i] + B[0]: A streams, B is one held tile reused each iter on a
selectable lifecycle, with the kernel supplying whatever edge the chain doesn't. Each case asserts
BOTH no-hang (--dev timeout trips triage) AND correct values (a miscount reads a stale tile).
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

HELD_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/lifecycle/held_b.cpp"

# Selector -> lifecycle name (must match held_b.cpp).
LIFECYCLES = {
    0: "Bulk",
    1: "HeldBulk",
    2: "HeldStream",
    3: "CallerManaged",
    4: "DeferredPop",
}


@pytest.mark.parametrize("life,name", list(LIFECYCLES.items()), ids=list(LIFECYCLES.values()))
def test_held_b_lifecycle(device, life, name):
    n = 8
    dt = ttnn.bfloat16
    a_shape = [1, 1, 32, 32 * n]
    b_shape = [1, 1, 32, 32]  # single held tile
    core_grid = lib.single_core_grid()

    torch_a, tt_a = lib.make_input(a_shape, dt, device, seed=701)
    torch_b, tt_b = lib.make_input(b_shape, dt, device, seed=702)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(a_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [
        lib.cb_descriptor(0, dt, 2, core_grid),
        lib.cb_descriptor(1, dt, 2, core_grid),
        lib.cb_descriptor(16, dt, 2, core_grid),
    ]
    reader = lib.build_reader_asym_kernel([tt_a, tt_b], [n, 1], core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(HELD_KERNEL, [n, life], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)  # a hang here trips --dev triage

    golden = torch_a.to(torch.float32) + torch_b.to(torch.float32).repeat(1, 1, 1, n)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.debug(f"lifecycle={name} | no-hang + {msg}")
    assert pcc_ok, f"lifecycle {name}: {msg}"


OUTPUT_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/lifecycle/out_lifecycle.cpp"
OUT_LIFECYCLES = {
    0: "Streaming",
    1: "Bulk",
    2: "ReserveAllPushPerTile",
    3: "CallerManaged",
    4: "ReserveNonePushEnd",
}


@pytest.mark.parametrize("life,name", list(OUT_LIFECYCLES.items()), ids=list(OUT_LIFECYCLES.values()))
def test_output_lifecycle(device, life, name):
    """Validate reserve/push ownership independently from input wait/pop."""
    n = 8
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=901)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, n, core_grid)]
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_1out_kernel(tt_out, n, core_grid),
            lib.build_compute_kernel(OUTPUT_KERNEL, [n, life], core_grid),
        ],
        semaphores=[],
        cbs=cbs,
    )
    output = ttnn.generic_op([tt_in, tt_out], program)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(torch_in.to(torch.float32), out, lib.pcc_threshold([dt]))
    logger.debug(f"output lifecycle={name} | no-hang + {msg}")
    assert pcc_ok, f"output lifecycle {name}: {msg}"


INPLACE_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/lifecycle/inplace_chain.cpp"
INPLACE_LIFECYCLES = {
    0: "BulkDrain+Streaming",
    1: "PerBlockSize+PerBlockSize",
    2: "Streaming+Streaming",
}


@pytest.mark.parametrize("life,name", list(INPLACE_LIFECYCLES.items()), ids=list(INPLACE_LIFECYCLES.values()))
def test_inplace_chain_lifecycle(device, life, name):
    """Validate safe input/output lifecycle pairs when one CB is both source and destination."""
    n = 8
    block_size = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1301)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [
        lib.cb_descriptor(0, dt, n, core_grid),
        lib.cb_descriptor(1, dt, n, core_grid),
        lib.cb_descriptor(16, dt, n, core_grid),
    ]
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_1out_kernel(tt_out, n, core_grid),
            lib.build_compute_kernel(INPLACE_KERNEL, [n, life, block_size], core_grid),
        ],
        semaphores=[],
        cbs=cbs,
    )
    output = ttnn.generic_op([tt_in, tt_out], program)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(torch.exp(torch_in.to(torch.float32)), out, lib.pcc_threshold([dt]))
    logger.debug(f"in-place lifecycle={name} | no-hang + {msg}")
    assert pcc_ok, f"in-place lifecycle {name}: {msg}"


OUTER_DIR = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/outer_stream"


def _build_outer_reader(tt_a, tt_b, height_tiles, width_tiles, grid):
    cta = list(ttnn.TensorAccessorArgs(tt_a).get_compile_time_args())
    cta += list(ttnn.TensorAccessorArgs(tt_b).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [tt_a.buffer_address(), tt_b.buffer_address(), height_tiles, width_tiles]
    return ttnn.KernelDescriptor(
        kernel_source=f"{OUTER_DIR}/reader_a_full_b_per_row.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )


@pytest.mark.parametrize("Ht,Wt,fp32_dest_acc_en", [(2, 1, False), (3, 5, True)])
def test_outer_stream_broadcast(device, Ht, Wt, fp32_dest_acc_en):
    """PerTile + Col holds one streamed B tile across each output row."""
    dt = ttnn.bfloat16
    a_shape = [1, 1, 32, 32 * Ht * Wt]
    b_shape = [1, 1, 32, 32 * Ht]
    torch_a, tt_a = lib.make_input(a_shape, dt, device, seed=101)
    torch_b, tt_b = lib.make_input(b_shape, dt, device, seed=202)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(a_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    grid = lib.single_core_grid()
    cbs = [
        lib.cb_descriptor(0, dt, 2, grid),
        lib.cb_descriptor(1, dt, 2, grid),
        lib.cb_descriptor(16, dt, 2, grid),
    ]
    program = ttnn.ProgramDescriptor(
        kernels=[
            _build_outer_reader(tt_a, tt_b, Ht, Wt, grid),
            lib.build_writer_1out_kernel(tt_out, Ht * Wt, grid),
            lib.build_compute_kernel(
                f"{OUTER_DIR}/chain_outer_stream.cpp",
                [Ht, Wt],
                grid,
                fp32_dest_acc_en=fp32_dest_acc_en,
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)
    a_v = torch_a.to(torch.float32).view(1, 1, 32, Ht, Wt, 32)
    b_v = torch_b.to(torch.float32).view(1, 1, 32, Ht, 1, 32)
    golden = (a_v + b_v).reshape(1, 1, 32, 32 * Ht * Wt)
    pcc_ok, msg = comp_pcc(golden, torch_out, 0.999)
    logger.debug(f"StreamedCol | Ht={Ht} Wt={Wt} fp32_dest_acc_en={fp32_dest_acc_en} | {msg}")
    assert pcc_ok, msg
