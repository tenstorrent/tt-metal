# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-core allocated tensors through ttnn.experimental.generic_op.

A per-core allocated tensor has its shard at a different L1 address on each core. Metal 2.0 binds such a
TensorParameter per node (#61952): each node's address goes into that node's runtime args, and the kernel reaches its
own shard through a LocalTensorAccessor. Here a data movement kernel on two cores copies its src shard into its dst
shard. A filler tensor on one core first makes the two cores' addresses differ, which the test checks, so the copy is
right only if each core gets its own address.

Per-core allocation needs the hybrid allocator, which is chosen when the device opens, so the device fixture here sets
TT_METAL_ALLOCATOR_MODE_HYBRID before opening a 1x1 mesh.
"""

import os

import pytest
import torch
import ttnn

E = ttnn.experimental

CORES = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
ROW_WORDS = 256  # uint32
SHARD_ROWS = 8  # 8 KiB per core

COPY_SHARD_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t shard_bytes = get_arg(args::shard_bytes);
    Noc noc;
    const LocalTensorAccessor<uint32_t> src(tensor::src);
    const LocalTensorAccessor<uint32_t> dst(tensor::dst);
    noc.async_write(src, dst, shard_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_write_barrier();
}
"""


@pytest.fixture
def hybrid_mesh_device():
    os.environ["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    yield mesh
    ttnn.close_mesh_device(mesh)
    os.environ.pop("TT_METAL_ALLOCATOR_MODE_HYBRID", None)


def per_core_config(cores, shard_rows):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_rows, ROW_WORDS], ttnn.ShardOrientation.ROW_MAJOR),
    )
    config.experimental_set_per_core_allocation(True)
    return config


def to_mesh(mesh, torch_tensor, config):
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        device=mesh,
    )


def copy_program(src, dst):
    return E.ProgramSpec(
        name="per_node_copy",
        kernels=[
            E.KernelSpec(
                unique_id="copy",
                source=E.KernelSpec.SourceCode(code=COPY_SHARD_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                tensor_bindings=[
                    E.TensorBinding(tensor_parameter_name=name, accessor_name=name) for name in ("src", "dst")
                ],
                compile_time_args={"shard_bytes": SHARD_ROWS * ROW_WORDS * 4},
            )
        ],
        tensor_parameters=[
            E.TensorParameter(unique_id="src", spec=src.spec),
            E.TensorParameter(unique_id="dst", spec=dst.spec),
        ],
        work_units=[
            E.WorkUnitSpec(kernels=["copy"], target_nodes=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in CORES]))
        ],
    )


def per_core_addresses(tensor):
    return [tensor.experimental_per_core_buffer_address(ttnn.MeshCoordinate(0, 0), core) for core in CORES]


def test_generic_spec_op_per_node_tensors(hybrid_mesh_device):
    mesh = hybrid_mesh_device
    shape = (SHARD_ROWS * len(CORES), ROW_WORDS)
    fillers = []
    entries = mesh.num_program_cache_entries()
    for seed in range(2):
        # Each round, another filler on core 1 only shifts that core's next per-core allocations.
        fillers.append(
            to_mesh(mesh, torch.zeros(SHARD_ROWS, ROW_WORDS, dtype=torch.int32), per_core_config(CORES[1:], SHARD_ROWS))
        )
        torch.manual_seed(seed)
        torch_src = torch.randint(0, 1 << 30, shape, dtype=torch.int32)
        src = to_mesh(mesh, torch_src, per_core_config(CORES, SHARD_ROWS))
        dst = to_mesh(mesh, torch.zeros(shape, dtype=torch.int32), per_core_config(CORES, SHARD_ROWS))
        assert src.is_per_core_allocated() and dst.is_per_core_allocated()
        src_addresses, dst_addresses = per_core_addresses(src), per_core_addresses(dst)
        assert src_addresses[0] != src_addresses[1], src_addresses
        assert dst_addresses[0] != dst_addresses[1], dst_addresses

        E.generic_op([src, dst], copy_program(src, dst), E.ProgramRunArgs(tensor_args={"src": src, "dst": dst}))

        assert torch.equal(ttnn.to_torch(dst).to(torch.int32), torch_src)
    # The second round reuses the cached program with tensors at other addresses.
    assert mesh.num_program_cache_entries() == entries + 1
