# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DFBs borrowed from a per-core allocated tensor, through ttnn.experimental.generic_op (slow dispatch only).

A per-core allocated tensor has its shard at a different L1 address on each core. A DFB borrowed from it sits at
DFBAdvancedOptions.borrowed_offset from each core's own shard, as blaze's planner places per-core weights. On two
cores, a reader moves DRAM rows through the DFB as a FIFO, and a writer copies each entry out of the tensor at the
planned offset (through a LocalTensorAccessor, not through the DFB), so the output is right only if every core's DFB
sits at its own shard. A filler on one core first makes the two cores' shard addresses differ, which the test checks.

The alias variant moves a second set of rows through a second DFB aliased with the first (the same tensor and
offset), after a semaphore says the writer is done with the first.

Per-core allocation needs the hybrid allocator, chosen when the device opens, so the fixture here sets
TT_METAL_ALLOCATOR_MODE_HYBRID before opening a 1x1 mesh. Fast dispatch gives every core of a range the same DFB config,
so it rejects these DFBs.
"""

import os

import pytest
import torch
import ttnn

E = ttnn.experimental

CORES = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
GRID = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in CORES])
ROW_WORDS = 256  # uint32: one entry is one 1 KiB row
ENTRY_BYTES = ROW_WORDS * 4
SHARD_ROWS = 8  # 8 KiB of per-core tensor on each core
OFFSET = 2 * ENTRY_BYTES  # the DFB's place in each core's shard
ROWS_PER_CORE = 4  # rows moved per core per phase: two trips around a two-entry ring

INCLUDES = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
"""

# Phase 1 moves this core's rows of `in` through DFB a; with ALIAS, phase 2 moves its rows of `in` (the second half)
# through DFB b, once the writer has signalled that it is done with a.
READ_SOURCE = (
    INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t rows = get_arg(args::rows);
    const uint32_t first_row = get_arg(args::first_row);
    Noc noc;
    const auto in = TensorAccessor(tensor::in);
    auto fill = [&](auto& dfb, uint32_t first) {
        for (uint32_t i = 0; i < rows; ++i) {
            dfb.reserve_back(1);
            noc.async_read(in, dfb, dfb.get_entry_size(), {.page_id = first + i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb.push_back(1);
        }
    };
    {
        DataflowBuffer a(dfb::a);
        fill(a, first_row);
    }
#ifdef ALIAS
    Semaphore a_done(sem::a_done);
    a_done.down(1);
    {
        DataflowBuffer b(dfb::b);
        fill(b, first_row + get_arg(args::second_phase_rows));
    }
#endif
}
"""
)

# Writes each entry out of the per-core tensor at offset + (i % 2) entries: entry i of a two-entry ring sits there.
WRITE_SOURCE = (
    INCLUDES
    + """
void kernel_main() {
    constexpr uint32_t rows = get_arg(args::rows);
    constexpr uint32_t offset = get_arg(args::offset);
    const uint32_t first_row = get_arg(args::first_row);
    Noc noc;
    const LocalTensorAccessor<uint32_t> weights(tensor::weights);
    const auto out = TensorAccessor(tensor::out);
    auto drain = [&](auto& dfb, uint32_t first) {
        const uint32_t entry_bytes = dfb.get_entry_size();
        for (uint32_t i = 0; i < rows; ++i) {
            dfb.wait_front(1);
            noc.async_write(
                weights, out, entry_bytes, {.offset_bytes = offset + (i % 2) * entry_bytes}, {.page_id = first + i});
            noc.async_write_barrier();
            dfb.pop_front(1);
        }
    };
    {
        DataflowBuffer a(dfb::a);
        drain(a, first_row);
    }
#ifdef ALIAS
    Semaphore a_done(sem::a_done);
    a_done.up(1);
    {
        DataflowBuffer b(dfb::b);
        drain(b, first_row + get_arg(args::second_phase_rows));
    }
#endif
}
"""
)


@pytest.fixture
def hybrid_mesh_device():
    os.environ["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    yield mesh
    ttnn.close_mesh_device(mesh)
    os.environ.pop("TT_METAL_ALLOCATOR_MODE_HYBRID", None)


def per_core_config(cores, shard_rows):
    config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]),
            [shard_rows, ROW_WORDS],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    config.experimental_set_per_core_allocation(True)
    return config


def to_mesh(mesh, torch_tensor, memory_config=None):
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        device=mesh,
    )


def borrowed_dfb(unique_id, alias_with=()):
    return E.DataflowBufferSpec(
        unique_id=unique_id,
        entry_size=ENTRY_BYTES,
        num_entries=2,
        borrowed_from="weights",
        advanced_options=E.DFBAdvancedOptions(borrowed_offset=OFFSET, alias_with=list(alias_with)),
    )


def program(tensors, alias):
    phases = 2 if alias else 1
    defines = {"ALIAS": "1"} if alias else {}
    dfbs = [borrowed_dfb("a", ["b"] if alias else []), *([borrowed_dfb("b", ["a"])] if alias else [])]
    names = [d.unique_id for d in dfbs]
    semaphore_bindings = [E.SemaphoreBinding(semaphore_spec_name="a_done", accessor_name="a_done")] if alias else []
    common = {"rows": ROWS_PER_CORE}
    run_arg_names = ["first_row"] + (["second_phase_rows"] if alias else [])
    spec = E.ProgramSpec(
        name="per_core_borrowed_dfb",
        kernels=[
            E.KernelSpec(
                unique_id="read",
                source=E.KernelSpec.SourceCode(code=READ_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
                dfb_bindings=[E.producer_of(n, n) for n in names],
                semaphore_bindings=semaphore_bindings,
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="in", accessor_name="in")],
                compile_time_args=common,
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=run_arg_names),
            ),
            E.KernelSpec(
                unique_id="write",
                source=E.KernelSpec.SourceCode(code=WRITE_SOURCE),
                hw_config=E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                compiler_options=E.KernelSpec.CompilerOptions(defines=defines),
                dfb_bindings=[E.consumer_of(n, n) for n in names],
                semaphore_bindings=semaphore_bindings,
                tensor_bindings=[
                    E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in ("weights", "out")
                ],
                compile_time_args={**common, "offset": OFFSET},
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=run_arg_names),
            ),
        ],
        dataflow_buffers=dfbs,
        semaphores=[E.SemaphoreSpec(unique_id="a_done", target_nodes=GRID)] if alias else [],
        tensor_parameters=[E.TensorParameter(unique_id=name, spec=t.spec) for name, t in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["read", "write"], target_nodes=GRID)],
    )
    # Core c's phase-1 rows come first, then (alias) every core's phase-2 rows.
    first_row = {core: c * ROWS_PER_CORE for c, core in enumerate(CORES)}
    values = {"first_row": first_row}
    if alias:
        values["second_phase_rows"] = {core: len(CORES) * ROWS_PER_CORE for core in CORES}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.ProgramRunArgs.KernelRunArgs(kernel=k, runtime_arg_values=values) for k in ("read", "write")
        ],
        tensor_args=tensors,
    )
    return spec, run_args


def make_io(mesh, seed, alias):
    torch.manual_seed(seed)
    rows = len(CORES) * ROWS_PER_CORE * (2 if alias else 1)
    torch_in = torch.randint(0, 1 << 30, (rows, ROW_WORDS), dtype=torch.int32)
    tensors = {
        "in": to_mesh(mesh, torch_in),
        "weights": to_mesh(
            mesh,
            torch.zeros(len(CORES) * SHARD_ROWS, ROW_WORDS, dtype=torch.int32),
            per_core_config(CORES, SHARD_ROWS),
        ),
        "out": to_mesh(mesh, torch.zeros_like(torch_in)),
    }
    return torch_in, tensors


def weights_addresses(tensors):
    return [tensors["weights"].experimental_per_core_buffer_address(ttnn.MeshCoordinate(0, 0), c) for c in CORES]


@pytest.mark.parametrize("alias", [False, True], ids=["fifo", "alias"])
def test_generic_spec_op_per_core_borrowed_dfb(hybrid_mesh_device, alias):
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") != "1":
        pytest.skip("Per-core borrowed DFBs need slow dispatch")
    mesh = hybrid_mesh_device
    fillers = []
    entries = mesh.num_program_cache_entries()
    for seed in range(2):
        # Each round, another filler on core 1 only shifts that core's later per-core allocations.
        fillers.append(
            to_mesh(mesh, torch.zeros(SHARD_ROWS, ROW_WORDS, dtype=torch.int32), per_core_config(CORES[1:], SHARD_ROWS))
        )
        expected, tensors = make_io(mesh, seed, alias)
        addresses = weights_addresses(tensors)
        assert addresses[0] != addresses[1], addresses
        spec, run_args = program(tensors, alias)
        E.generic_op(list(tensors.values()), spec, run_args)
        assert torch.equal(ttnn.to_torch(tensors["out"]).to(torch.int32), expected)
    # The second round reuses the cached program with per-core tensors at other addresses.
    assert mesh.num_program_cache_entries() == entries + 1


def test_generic_spec_op_per_core_borrowed_dfb_needs_slow_dispatch(hybrid_mesh_device, expect_error):
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") == "1":
        pytest.skip("Checks the fast dispatch rejection")
    mesh = hybrid_mesh_device
    filler = to_mesh(
        mesh, torch.zeros(SHARD_ROWS, ROW_WORDS, dtype=torch.int32), per_core_config(CORES[1:], SHARD_ROWS)
    )
    _, tensors = make_io(mesh, 0, alias=False)
    spec, run_args = program(tensors, alias=False)
    with expect_error(RuntimeError, "DFB 'a' borrows from the per-core allocated TensorParameter 'weights'"):
        E.generic_op(list(tensors.values()), spec, run_args)
