# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase barriers between a compute kernel and data movement kernels through small "token" DFBs.

On Quasar a semaphore cannot be shared by compute and data movement kernels, so a DFB is the only way they
synchronize. A token DFB carries no data: its producer reserves and pushes an entry, and its consumer waits for it and
pops it. On Quasar the compute side must still issue an engine instruction between the two: dummy_pack()
between reserve_back and push_back on PACK, and dummy_unpack() between wait_front and pop_front on UNPACK. Without it
the tile-counter update is not ordered after the wait (TEN-4746), which LLK asserts catch but the functional simulators
do not. Implicit sync is off on every data movement kernel, as in blaze's lowering. Both tests run NUM_PHASES phases
on one core, with the buffers borrowed from an L1 arena tensor at planned offsets.

RAW: compute copies input tile p into Y (which is not the token DFB), then pushes a "done" token from PACK. A data
movement kernel waits for the token and then reads Y's bytes out of the arena tensor, not through Y, so only the token
orders the read after the packer's write.

WAR: S1 and S2 are borrowed at the same arena offset, and phase p uses S1 or S2 alternately, so every phase reuses the
same bytes. Each phase, the reader sends a "go" token; compute unpacks the phase's tile from S, packs it to O and pushes
a "done" token. The reader writes the next phase's tile into the same bytes only after that token.
"""

import pytest
import torch
import ttnn

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
NUM_PHASES = 4
TILE_BYTES = 2 * 32 * 32  # one bfloat16 tile
TOKEN_BYTES = 64
ARENA_BYTES = 16 * 1024
Y_OFFSET = 4 * 1024  # RAW: Y in the arena
S_OFFSET = 8 * 1024  # WAR: S1 and S2 in the arena

DM_INCLUDES = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
"""

COMPUTE_INCLUDES = """
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
"""

READ_TILES_SOURCE = (
    DM_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_phases = get_arg(args::num_phases);
    Noc noc;
    DataflowBuffer in(dfb::in);
    const auto input = TensorAccessor(tensor::input);
    for (uint32_t p = 0; p < num_phases; ++p) {
        in.reserve_back(1);
        noc.async_read(input, in, in.get_entry_size(), {.page_id = p}, {.offset_bytes = 0});
        noc.async_read_barrier();
        in.push_back(1);
    }
}
"""
)

# RAW compute: in -> DST -> Y, then the done token. TOKEN_FIRST (a mutant) pushes the token before packing Y.
RAW_COMPUTE_SOURCE = (
    COMPUTE_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_phases = get_arg(args::num_phases);
    DataflowBuffer in(dfb::in);
    DataflowBuffer y(dfb::y);
    DataflowBuffer done(dfb::done);
    compute_kernel_hw_startup(dfb::in, dfb::y);
    copy_init(dfb::in);
    for (uint32_t p = 0; p < num_phases; ++p) {
        in.wait_front(1);
        tile_regs_acquire();
        copy_tile(dfb::in, 0, 0);
        tile_regs_commit();
        y.reserve_back(1);
#ifdef TOKEN_FIRST
        done.reserve_back(1);
        dummy_pack(dfb::done);
        done.push_back(1);
#endif
        tile_regs_wait();
        pack_tile(0, dfb::y);
        tile_regs_release();
#ifndef TOKEN_FIRST
        done.reserve_back(1);
        dummy_pack(dfb::done);
        done.push_back(1);
#endif
        y.push_back(1);
        in.pop_front(1);
    }
}
"""
)

# RAW reader of Y: waits for the done token, copies Y's bytes out of the arena, then retires Y.
RAW_DRAIN_SOURCE = (
    DM_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_phases = get_arg(args::num_phases);
    constexpr uint32_t y_offset = get_arg(args::y_offset);
    Noc noc;
    DataflowBuffer y(dfb::y);
    DataflowBuffer done(dfb::done);
    const LocalTensorAccessor<uint32_t> arena(tensor::arena);
    const auto output = TensorAccessor(tensor::output);
    const uint32_t tile_bytes = y.get_entry_size();
    for (uint32_t p = 0; p < num_phases; ++p) {
        done.wait_front(1);
        noc.async_write(arena, output, tile_bytes, {.offset_bytes = y_offset}, {.page_id = p});
        noc.async_write_barrier();
        done.pop_front(1);
        y.wait_front(1);
        y.pop_front(1);
    }
}
"""
)

# WAR reader: per phase, fills S1 or S2 (the same bytes) and sends go. It writes phase p only after compute's done
# token for phase p - 1. WRITE_BEFORE_DONE (a mutant) writes first and waits after.
WAR_READ_SOURCE = (
    DM_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_phases = get_arg(args::num_phases);
    Noc noc;
    DataflowBuffer s1(dfb::s1);
    DataflowBuffer s2(dfb::s2);
    DataflowBuffer go(dfb::go);
    DataflowBuffer done(dfb::done);
    const auto input = TensorAccessor(tensor::input);
    auto fill = [&](auto& s, uint32_t p) {
        s.reserve_back(1);
        noc.async_read(input, s, s.get_entry_size(), {.page_id = p}, {.offset_bytes = 0});
        noc.async_read_barrier();
        s.push_back(1);
    };
    for (uint32_t p = 0; p < num_phases; ++p) {
#ifndef WRITE_BEFORE_DONE
        if (p > 0) {
            done.wait_front(1);
            done.pop_front(1);
        }
#endif
        if (p % 2 == 0) {
            fill(s1, p);
        } else {
            fill(s2, p);
        }
#ifdef WRITE_BEFORE_DONE
        if (p > 0) {
            done.wait_front(1);
            done.pop_front(1);
        }
#endif
        go.reserve_back(1);
        go.push_back(1);
    }
    done.wait_front(1);
    done.pop_front(1);
}
"""
)

WAR_COMPUTE_SOURCE = (
    COMPUTE_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_phases = get_arg(args::num_phases);
    DataflowBuffer s1(dfb::s1);
    DataflowBuffer s2(dfb::s2);
    DataflowBuffer o(dfb::o);
    DataflowBuffer go(dfb::go);
    DataflowBuffer done(dfb::done);
    compute_kernel_hw_startup(dfb::s1, dfb::o);
    auto phase = [&](auto& s, auto s_token) {
        copy_init(s_token);
        s.wait_front(1);
        tile_regs_acquire();
        copy_tile(s_token, 0, 0);
        tile_regs_commit();
        o.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::o);
        tile_regs_release();
        o.push_back(1);
        s.pop_front(1);
        done.reserve_back(1);  // After the pack, so after the unpack of s.
        dummy_pack(dfb::done);
        done.push_back(1);
    };
    for (uint32_t p = 0; p < num_phases; ++p) {
        go.wait_front(1);
        dummy_unpack(dfb::go);
        go.pop_front(1);
        if (p % 2 == 0) {
            phase(s1, dfb::s1);
        } else {
            phase(s2, dfb::s2);
        }
    }
}
"""
)

WRITE_TILES_SOURCE = (
    DM_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_phases = get_arg(args::num_phases);
    Noc noc;
    DataflowBuffer o(dfb::o);
    const auto output = TensorAccessor(tensor::output);
    for (uint32_t p = 0; p < num_phases; ++p) {
        o.wait_front(1);
        noc.async_write(o, output, o.get_entry_size(), {.offset_bytes = 0}, {.page_id = p});
        noc.async_write_barrier();
        o.pop_front(1);
    }
}
"""
)


def make_io(device, seed):
    torch.manual_seed(seed)
    torch_input = torch.randn(32, 32 * NUM_PHASES).bfloat16()
    to_dram = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(CORE, CORE)})
    arena_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ARENA_BYTES // 1024, 256], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tensors = {
        "input": ttnn.from_torch(torch_input, **to_dram),
        "arena": ttnn.allocate_tensor_on_device(
            ttnn.Shape([ARENA_BYTES // 1024, 256]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, arena_config
        ),
        "output": ttnn.from_torch(torch.zeros_like(torch_input), **to_dram),
    }
    return torch_input, tensors


def kernel(unique_id, source, hw_config, defines, dfbs, tensors=(), **fields):
    return E.KernelSpec(
        unique_id=unique_id,
        source=E.KernelSpec.SourceCode(code=source),
        hw_config=hw_config,
        compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
        dfb_bindings=dfbs,
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in tensors],
        compile_time_args={"num_phases": NUM_PHASES, **fields.pop("compile_time_args", {})},
        **fields,
    )


def tile_dfb(unique_id, borrowed_offset=None, num_entries=1):
    borrowed = {}
    if borrowed_offset is not None:
        borrowed = dict(borrowed_from="arena", advanced_options=E.DFBAdvancedOptions(borrowed_offset=borrowed_offset))
    return E.DataflowBufferSpec(
        unique_id=unique_id,
        entry_size=TILE_BYTES,
        num_entries=num_entries,
        data_format_metadata=ttnn.bfloat16,
        **borrowed,
    )


def token_dfb(unique_id):
    # A compute kernel binds it, so it needs a data format, but no one reads its bytes.
    return E.DataflowBufferSpec(
        unique_id=unique_id, entry_size=TOKEN_BYTES, num_entries=1, data_format_metadata=ttnn.bfloat16
    )


def program(tensors, kernels, dataflow_buffers):
    return E.ProgramSpec(
        name="token_barrier",
        kernels=kernels,
        dataflow_buffers=dataflow_buffers,
        tensor_parameters=[E.TensorParameter(unique_id=name, spec=tensor.spec) for name, tensor in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=[k.unique_id for k in kernels], target_nodes=CORE)],
    )


def raw_program(tensors, defines):
    reader = E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True)
    writer = E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True)
    return program(
        tensors,
        [
            kernel("read", READ_TILES_SOURCE, reader, defines, [E.producer_of("in", "in")], ["input"]),
            kernel(
                "compute",
                RAW_COMPUTE_SOURCE,
                E.ComputeHardwareConfig(),
                defines,
                [E.consumer_of("in", "in"), E.producer_of("y", "y"), E.producer_of("done", "done")],
            ),
            kernel(
                "drain",
                RAW_DRAIN_SOURCE,
                writer,
                defines,
                [E.consumer_of("y", "y"), E.consumer_of("done", "done")],
                ["arena", "output"],
                compile_time_args={"y_offset": Y_OFFSET},
            ),
        ],
        [tile_dfb("in", num_entries=2), tile_dfb("y", borrowed_offset=Y_OFFSET), token_dfb("done")],
    )


def war_program(tensors, defines):
    reader = E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True)
    writer = E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True)
    return program(
        tensors,
        [
            kernel(
                "read",
                WAR_READ_SOURCE,
                reader,
                defines,
                [E.producer_of(name, name) for name in ("s1", "s2", "go")] + [E.consumer_of("done", "done")],
                ["input"],
            ),
            kernel(
                "compute",
                WAR_COMPUTE_SOURCE,
                E.ComputeHardwareConfig(),
                defines,
                [E.consumer_of(name, name) for name in ("s1", "s2", "go")]
                + [E.producer_of("o", "o"), E.producer_of("done", "done")],
            ),
            kernel("write", WRITE_TILES_SOURCE, writer, defines, [E.consumer_of("o", "o")], ["output"]),
        ],
        [
            tile_dfb("s1", borrowed_offset=S_OFFSET),
            tile_dfb("s2", borrowed_offset=S_OFFSET),
            tile_dfb("o", num_entries=2),
            token_dfb("go"),
            token_dfb("done"),
        ],
    )


def run(device, make_program, seed, defines=None):
    torch_input, tensors = make_io(device, seed)
    spec = make_program(tensors, defines or {})
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))
    return torch_input, ttnn.to_torch(tensors["output"])


@pytest.mark.parametrize("make_program", [raw_program, war_program], ids=["raw", "war"])
def test_generic_spec_op_token_barrier(device, make_program):
    for seed in range(2):
        expected, result = run(device, make_program, seed)
        assert torch.equal(result, expected)
