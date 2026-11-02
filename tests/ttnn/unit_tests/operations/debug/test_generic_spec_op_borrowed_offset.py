# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.generic_op with DataflowBufferSpecs borrowed from one L1 "arena" tensor at byte offsets.

This is how blaze lays out its planned circular buffers in an arena tensor: each buffer is borrowed from the tensor at
DFBAdvancedOptions.borrowed_offset. The tests use the ttnn.experimental names (those of #63251). The kernels move rows of DRAM tensors through the buffers as FIFOs on one core.
The consumer writes each entry out through a LocalTensorAccessor of the arena rather than through the buffer, from
where the spec placed it, so the output is right only if the buffer really sits at its offset.
"""

import pytest
import torch
import ttnn

E = ttnn.experimental

CORE = ttnn.CoreCoord(0, 0)
GRID = ttnn.CoreRangeSet({ttnn.CoreRange(CORE, CORE)})
ROW_WORDS = 256  # one entry: a row of 256 uint32
ENTRY_BYTES = ROW_WORDS * 4
NUM_ROWS = 8  # rows moved through each buffer: four trips around a two-entry ring
ARENA_ROWS = 16  # arena shard: 16 KiB

KERNEL_INCLUDES = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
"""

# Reads row i of tensors a and b into buffers a and b.
READ_TWO_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_rows = get_arg(args::num_rows);
    Noc noc;
    DataflowBuffer dfb_a(dfb::a);
    DataflowBuffer dfb_b(dfb::b);
    const auto in_a = TensorAccessor(tensor::in_a);
    const auto in_b = TensorAccessor(tensor::in_b);
    const uint32_t entry_bytes = dfb_a.get_entry_size();
    for (uint32_t i = 0; i < num_rows; ++i) {
        dfb_a.reserve_back(1);
        dfb_b.reserve_back(1);
        noc.async_read(in_a, dfb_a, entry_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(in_b, dfb_b, entry_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_a.push_back(1);
        dfb_b.push_back(1);
    }
}
"""
)

# Writes entry i of buffers a and b out of the arena, at offset_a or offset_b plus (i % 2) entries: with one producer
# and one consumer, entry i of a two-entry buffer sits there.
WRITE_TWO_FROM_ARENA_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_rows = get_arg(args::num_rows);
    constexpr uint32_t offset_a = get_arg(args::offset_a);
    constexpr uint32_t offset_b = get_arg(args::offset_b);
    Noc noc;
    DataflowBuffer dfb_a(dfb::a);
    DataflowBuffer dfb_b(dfb::b);
    const LocalTensorAccessor<uint32_t> arena(tensor::arena);
    const auto out_a = TensorAccessor(tensor::out_a);
    const auto out_b = TensorAccessor(tensor::out_b);
    const uint32_t entry_bytes = dfb_a.get_entry_size();
    for (uint32_t i = 0; i < num_rows; ++i) {
        const uint32_t slot = (i % 2) * entry_bytes;
        dfb_a.wait_front(1);
        dfb_b.wait_front(1);
        noc.async_write(arena, out_a, entry_bytes, {.offset_bytes = offset_a + slot}, {.page_id = i});
        noc.async_write(arena, out_b, entry_bytes, {.offset_bytes = offset_b + slot}, {.page_id = i});
        noc.async_write_barrier();
        dfb_a.pop_front(1);
        dfb_b.pop_front(1);
    }
}
"""
)

# Phase 1 moves tensor a through buffer a; phase 2 moves tensor b through buffer b, whose bytes overlap a's. The
# producer starts phase 2 only after the consumer signals that it is done with a.
READ_TWO_PHASES_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_rows = get_arg(args::num_rows);
    Noc noc;
    Semaphore a_done(sem::a_done);
    const auto in_a = TensorAccessor(tensor::in_a);
    const auto in_b = TensorAccessor(tensor::in_b);
    {
        DataflowBuffer dfb_a(dfb::a);
        for (uint32_t i = 0; i < num_rows; ++i) {
            dfb_a.reserve_back(1);
            noc.async_read(in_a, dfb_a, dfb_a.get_entry_size(), {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_a.push_back(1);
        }
    }
    a_done.down(1);
    {
        DataflowBuffer dfb_b(dfb::b);
        for (uint32_t i = 0; i < num_rows; ++i) {
            dfb_b.reserve_back(1);
            noc.async_read(in_b, dfb_b, dfb_b.get_entry_size(), {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_b.push_back(1);
        }
    }
}
"""
)

WRITE_TWO_PHASES_FROM_ARENA_SOURCE = (
    KERNEL_INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t num_rows = get_arg(args::num_rows);
    constexpr uint32_t offset_a = get_arg(args::offset_a);
    constexpr uint32_t offset_b = get_arg(args::offset_b);
    Noc noc;
    Semaphore a_done(sem::a_done);
    const LocalTensorAccessor<uint32_t> arena(tensor::arena);
    const auto out_a = TensorAccessor(tensor::out_a);
    const auto out_b = TensorAccessor(tensor::out_b);
    {
        DataflowBuffer dfb_a(dfb::a);
        for (uint32_t i = 0; i < num_rows; ++i) {
            const uint32_t slot = (i % 2) * dfb_a.get_entry_size();
            dfb_a.wait_front(1);
            noc.async_write(arena, out_a, dfb_a.get_entry_size(), {.offset_bytes = offset_a + slot}, {.page_id = i});
            noc.async_write_barrier();
            dfb_a.pop_front(1);
        }
    }
    a_done.up(1);
    {
        DataflowBuffer dfb_b(dfb::b);
        for (uint32_t i = 0; i < num_rows; ++i) {
            const uint32_t slot = (i % 2) * dfb_b.get_entry_size();
            dfb_b.wait_front(1);
            noc.async_write(arena, out_b, dfb_b.get_entry_size(), {.offset_bytes = offset_b + slot}, {.page_id = i});
            noc.async_write_barrier();
            dfb_b.pop_front(1);
        }
    }
}
"""
)


def make_arena(device):
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(GRID, [ARENA_ROWS, ROW_WORDS], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([ARENA_ROWS, ROW_WORDS]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, memory_config
    )


def make_rows(device, torch_rows):
    return ttnn.from_torch(torch_rows, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def make_io(device, seed):
    torch.manual_seed(seed)
    torch_a = torch.randint(0, 1 << 30, (NUM_ROWS, ROW_WORDS), dtype=torch.int32)
    torch_b = torch.randint(0, 1 << 30, (NUM_ROWS, ROW_WORDS), dtype=torch.int32)
    zeros = torch.zeros_like(torch_a)
    tensors = {
        "in_a": make_rows(device, torch_a),
        "in_b": make_rows(device, torch_b),
        "arena": make_arena(device),
        "out_a": make_rows(device, zeros),
        "out_b": make_rows(device, zeros),
    }
    return torch_a, torch_b, tensors


def borrowed_dfb(unique_id, offset, alias_with=()):
    return E.DataflowBufferSpec(
        unique_id=unique_id,
        entry_size=ENTRY_BYTES,
        num_entries=2,
        borrowed_from="arena",
        advanced_options=E.DFBAdvancedOptions(borrowed_offset=offset, alias_with=list(alias_with)),
    )


def tensor_bindings(*names):
    return [E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in names]


def make_program(tensors, *, offset_a, offset_b, phases, dfb_a=None, dfb_b=None):
    semaphore_bindings = []
    semaphores = []
    if phases:
        semaphores = [E.SemaphoreSpec(unique_id="a_done", target_nodes=CORE)]
        semaphore_bindings = [E.SemaphoreBinding(semaphore_spec_name="a_done", accessor_name="a_done")]
    return E.ProgramSpec(
        name="borrowed_offset",
        kernels=[
            E.KernelSpec(
                unique_id="reader",
                source=E.KernelSpec.SourceCode(code=READ_TWO_PHASES_SOURCE if phases else READ_TWO_SOURCE),
                hw_config=E.create_reader_datamovement_config(),
                dfb_bindings=[E.producer_of("a", "a"), E.producer_of("b", "b")],
                semaphore_bindings=semaphore_bindings,
                tensor_bindings=tensor_bindings("in_a", "in_b"),
                compile_time_args={"num_rows": NUM_ROWS},
            ),
            E.KernelSpec(
                unique_id="writer",
                source=E.KernelSpec.SourceCode(
                    code=WRITE_TWO_PHASES_FROM_ARENA_SOURCE if phases else WRITE_TWO_FROM_ARENA_SOURCE
                ),
                hw_config=E.create_writer_datamovement_config(),
                dfb_bindings=[E.consumer_of("a", "a"), E.consumer_of("b", "b")],
                semaphore_bindings=semaphore_bindings,
                tensor_bindings=tensor_bindings("arena", "out_a", "out_b"),
                compile_time_args={"num_rows": NUM_ROWS, "offset_a": offset_a, "offset_b": offset_b},
            ),
        ],
        dataflow_buffers=[dfb_a or borrowed_dfb("a", offset_a), dfb_b or borrowed_dfb("b", offset_b)],
        semaphores=semaphores,
        tensor_parameters=[E.TensorParameter(unique_id=name, spec=tensor.spec) for name, tensor in tensors.items()],
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=GRID)],
    )


def run(device, tensors, spec):
    E.generic_op(list(tensors.values()), spec, E.ProgramRunArgs(tensor_args=tensors))


@pytest.mark.parametrize(
    "offset_a, offset_b, phases",
    [
        (0, 8 * 1024, False),  # two FIFOs side by side, with a gap
        (4 * 1024, 2 * 1024, False),  # b before a, packed against it
        (0, ENTRY_BYTES, True),  # b starts at a's second entry: phase reuse
    ],
    ids=["fifos_gap", "fifos_packed", "phase_reuse"],
)
def test_generic_spec_op_borrowed_offset(device, offset_a, offset_b, phases):
    # The second call hits the program cache with new tensors, so the offsets must apply to the new arena's address.
    entries = device.num_program_cache_entries()
    for seed in range(2):
        torch_a, torch_b, tensors = make_io(device, seed)
        run(device, tensors, make_program(tensors, offset_a=offset_a, offset_b=offset_b, phases=phases))
        assert torch.equal(ttnn.to_torch(tensors["out_a"]).to(torch.int32), torch_a)
        assert torch.equal(ttnn.to_torch(tensors["out_b"]).to(torch.int32), torch_b)
    assert device.num_program_cache_entries() == entries + 1


@pytest.mark.parametrize(
    "offset, message",
    [
        (8, "has borrowed_offset = 8, which is not a multiple of the L1 alignment"),
        (ARENA_ROWS * ENTRY_BYTES - ENTRY_BYTES, "does not fit in the per-bank allocation of its borrowed"),
    ],
    ids=["misaligned", "out_of_range"],
)
def test_generic_spec_op_rejects_bad_borrowed_offset(device, expect_error, offset, message):
    _, _, tensors = make_io(device, 0)
    spec = make_program(tensors, offset_a=0, offset_b=offset, phases=False)
    with expect_error(RuntimeError, message):
        run(device, tensors, spec)


def test_generic_spec_op_rejects_borrowed_offset_without_borrowed_from(device, expect_error):
    _, _, tensors = make_io(device, 0)
    not_borrowed = E.DataflowBufferSpec(
        unique_id="b",
        entry_size=ENTRY_BYTES,
        num_entries=2,
        advanced_options=E.DFBAdvancedOptions(borrowed_offset=1024),
    )
    spec = make_program(tensors, offset_a=0, offset_b=1024, phases=False, dfb_b=not_borrowed)
    with expect_error(RuntimeError, "sets borrowed_offset = 1024 but does not set borrowed_from"):
        run(device, tensors, spec)


def test_generic_spec_op_rejects_aliases_with_different_borrowed_offsets(device, expect_error):
    # Aliases share one address, so a and b may not sit at different offsets of the tensor they both borrow.
    _, _, tensors = make_io(device, 0)
    spec = make_program(
        tensors,
        offset_a=0,
        offset_b=4 * 1024,
        phases=False,
        dfb_a=borrowed_dfb("a", 0, alias_with=["b"]),
        dfb_b=borrowed_dfb("b", 4 * 1024, alias_with=["a"]),
    )
    with expect_error(RuntimeError, "have different borrowed_offset"):
        run(device, tensors, spec)


def float32_dfb(unique_id, offset):
    return E.DataflowBufferSpec(
        unique_id=unique_id,
        entry_size=ENTRY_BYTES,
        num_entries=2,
        data_format_metadata=ttnn.float32,
        borrowed_from="arena",
        advanced_options=E.DFBAdvancedOptions(borrowed_offset=offset),
    )


@pytest.mark.parametrize(
    "variant",
    [
        # a and b overlap at offset 0 in sequential phases, a valid spec with or without declaring them aliases.
        {"dfb_a": borrowed_dfb("a", 0, alias_with=["b"]), "dfb_b": borrowed_dfb("b", 0, alias_with=["a"])},
        # An unset data format must not hash like Float32 (DataFormat value 0).
        {"dfb_a": float32_dfb("a", 0)},
    ],
    ids=["alias_with", "format_unset_vs_float32"],
)
def test_generic_spec_op_program_hash_tells_specs_apart(device, variant):
    # Two specs that differ only in variant's DFBs must get their own cached programs.
    entries = device.num_program_cache_entries()
    for dfbs in ({}, variant):
        torch_a, torch_b, tensors = make_io(device, 0)
        run(device, tensors, make_program(tensors, offset_a=0, offset_b=0, phases=True, **dfbs))
        assert torch.equal(ttnn.to_torch(tensors["out_a"]).to(torch.int32), torch_a)
        assert torch.equal(ttnn.to_torch(tensors["out_b"]).to(torch.int32), torch_b)
    assert device.num_program_cache_entries() == entries + 2


def test_dfb_advanced_options_round_trip():
    dfb = E.DataflowBufferSpec(
        unique_id="d", entry_size=64, num_entries=1, advanced_options=E.DFBAdvancedOptions(borrowed_offset=128)
    )
    assert dfb.advanced_options.borrowed_offset == 128
    assert E.DataflowBufferSpec(unique_id="d", entry_size=64, num_entries=1).advanced_options.borrowed_offset == 0
