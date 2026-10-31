# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Two programs per token, handing state over through an L1 arena, through ttnn.experimental.generic_op.

tt-blaze's way past the per-core DFB caps for a full layer: split the fused program at a phase boundary into two
programs that run in alternation, each with its own DFB set. State that crosses the boundary stays in a lockstep L1
arena at a planned offset. Program A computes h = x * w_a and packs h into a DFB borrowed from the arena at OFFSET_H;
its data movement kernel pops h, which leaves the bytes in place. Program B borrows the same arena bytes
(borrowed_from + borrowed_offset) as a DFB, re-announces them with a push that moves no data, and computes
y = h + w_b. The weights w_a and w_b stay resident in L1 tensors and are re-announced the same way on every run,
since a program's DFBs start empty. Three tokens run A, B, A, B, A, B with new x and y each time, and only two
program-cache entries.
"""

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
GRID = ttnn.CoreRangeSet({ttnn.CoreRange(CORE, CORE)})
TILE_BYTES = 2 * 32 * 32
ARENA_TILES = 8
OFFSET_H = 5 * TILE_BYTES  # where h lives between program A and program B
TOKENS = 3

KERNEL_INCLUDES = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
"""

COMPUTE_INCLUDES = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
"""

# Program A, data movement: x from DRAM into `x`, re-announce the resident weight, then retire h. Popping h releases
# the entry but leaves its bytes in the arena for program B.
A_DM_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    Noc noc;
    DataflowBuffer x_dfb(dfb::x);
    DataflowBuffer w(dfb::w);
    DataflowBuffer h(dfb::h);
    const auto x = TensorAccessor(tensor::x);
    x_dfb.reserve_back(1);
    noc.async_read(x, x_dfb, x_dfb.get_entry_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    x_dfb.push_back(1);
    w.reserve_back(1);
    w.push_back(1);
    h.wait_front(1);
    h.pop_front(1);
}
"""
)

# Program B, data movement: re-announce h (left in the arena by program A) and the resident weight, then write y.
B_DM_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    Noc noc;
    DataflowBuffer h(dfb::h);
    DataflowBuffer w(dfb::w);
    DataflowBuffer y_dfb(dfb::y);
    const auto y = TensorAccessor(tensor::y);
    h.reserve_back(1);
    h.push_back(1);
    w.reserve_back(1);
    w.push_back(1);
    y_dfb.wait_front(1);
    noc.async_write(y_dfb, y, y_dfb.get_entry_size(), {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
    y_dfb.pop_front(1);
}
"""
)

# out = in0 OP in1, one tile.
COMPUTE_SOURCE = (
    COMPUTE_INCLUDES
    + """
void kernel_main() {
    DataflowBuffer in0(dfb::IN0);
    DataflowBuffer in1(dfb::w);
    DataflowBuffer out(dfb::OUT);
    compute_kernel_hw_startup(dfb::IN0, dfb::w, dfb::OUT);
    in0.wait_front(1);
    in1.wait_front(1);
    tile_regs_acquire();
    binary_tiles_init<true, EltwiseBinaryType::OP>(dfb::IN0, dfb::w);
    BINARY(dfb::IN0, dfb::w, 0, 0, 0);
    tile_regs_commit();
    out.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, dfb::OUT);
    tile_regs_release();
    out.push_back(1);
    in0.pop_front(1);
    in1.pop_front(1);
}
"""
)


def l1_sharded(device, torch_tiles, num_tiles):
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(GRID, [32 * num_tiles, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch_tiles, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


def dram_tile(device, t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def tile_dfb(unique_id, borrowed_from=None, offset=0):
    kwargs = {}
    if borrowed_from is not None:
        kwargs = dict(borrowed_from=borrowed_from, advanced_options=E.DFBAdvancedOptions(borrowed_offset=offset))
    return E.DataflowBufferSpec(
        unique_id=unique_id, entry_size=TILE_BYTES, num_entries=1, data_format_metadata=ttnn.bfloat16, **kwargs
    )


def compute_source(in0, out, op, binary):
    return (
        COMPUTE_SOURCE.replace("dfb::IN0", f"dfb::{in0}")
        .replace("dfb::OUT", f"dfb::{out}")
        .replace("EltwiseBinaryType::OP", f"EltwiseBinaryType::{op}")
        .replace("BINARY(", f"{binary}(")
    )


def make_program(name, tensors, dm_source, dm_bindings, compute_bindings, compute_code, dfbs, dm_tensors):
    return E.ProgramSpec(
        name=name,
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=dm_source),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=dm_bindings,
                tensor_bindings=[E.TensorBinding(tensor_parameter_name=t, accessor_name=t) for t in dm_tensors],
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=compute_code),
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=compute_bindings,
            ),
        ],
        dataflow_buffers=dfbs,
        tensor_parameters=[E.TensorParameter(unique_id=t, spec=v.spec) for t, v in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=CORE)],
    )


def program_a(tensors):
    return make_program(
        "arena_handoff_a",
        tensors,
        A_DM_SOURCE,
        [E.producer_of("x", "x"), E.producer_of("w", "w"), E.consumer_of("h", "h")],
        [E.consumer_of("x", "x"), E.consumer_of("w", "w"), E.producer_of("h", "h")],
        compute_source("x", "h", "ELWMUL", "mul_tiles"),
        [tile_dfb("x"), tile_dfb("w", borrowed_from="w"), tile_dfb("h", borrowed_from="arena", offset=OFFSET_H)],
        ["x"],
    )


def program_b(tensors):
    return make_program(
        "arena_handoff_b",
        tensors,
        B_DM_SOURCE,
        [E.producer_of("h", "h"), E.producer_of("w", "w"), E.consumer_of("y", "y")],
        [E.consumer_of("h", "h"), E.consumer_of("w", "w"), E.producer_of("y", "y")],
        compute_source("h", "y", "ELWADD", "add_tiles"),
        [tile_dfb("h", borrowed_from="arena", offset=OFFSET_H), tile_dfb("w", borrowed_from="w"), tile_dfb("y")],
        ["y"],
    )


def test_generic_spec_op_arena_handoff(device):
    torch.manual_seed(0)
    torch_w_a = torch.randn(32, 32).bfloat16()
    torch_w_b = torch.randn(32, 32).bfloat16()
    arena = l1_sharded(device, torch.zeros(32 * ARENA_TILES, 32).bfloat16(), ARENA_TILES)
    w_a = l1_sharded(device, torch_w_a, 1)
    w_b = l1_sharded(device, torch_w_b, 1)

    entries = device.num_program_cache_entries()
    for token in range(TOKENS):
        torch_x = torch.randn(32, 32).bfloat16()
        a_tensors = {"x": dram_tile(device, torch_x), "w": w_a, "arena": arena}
        E.generic_op(list(a_tensors.values()), program_a(a_tensors), E.ProgramRunArgs(tensor_args=a_tensors))

        y = dram_tile(device, torch.zeros(32, 32).bfloat16())
        b_tensors = {"y": y, "w": w_b, "arena": arena}
        E.generic_op(list(b_tensors.values()), program_b(b_tensors), E.ProgramRunArgs(tensor_args=b_tensors))

        expected = (torch_x.float() * torch_w_a.float()).bfloat16().float() + torch_w_b.float()
        assert_with_pcc(expected, ttnn.to_torch(y).float(), 0.999)
    assert device.num_program_cache_entries() == entries + 2
