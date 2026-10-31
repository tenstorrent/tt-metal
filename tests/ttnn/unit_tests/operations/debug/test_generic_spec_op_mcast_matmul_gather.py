# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""tt-blaze's decode Mcast -> Matmul -> Gather as one ProgramSpec on two cores, through ttnn.experimental.generic_op.

out[8, 64] = a[8, 128] @ w[128, 64] for NUM_ITERS tokens with the same weights, on cores (0, 0) and (1, 0):

- a is [8, 128] per token in Tile([8, 32]), on core 0. Each iteration core 0 copies the token's 4 tiles into its in0 and
  multicasts them into core 1's in0 (ready / valid semaphores). in0 is borrowed at IN0_OFFSET from a lockstep L1 arena
  tensor, so it sits at the same address on both cores, which the multicast relies on.
- w is width-sharded, 32 columns per core, in per-core allocated L1 (a filler on core 1 makes the two cores' addresses
  differ). in1 is borrowed from it: pushed once, waited once, popped when the kernels end.
- Each core computes its [8, 32] slice with matmul_block on the tiny tiles into its out DFB.
- Gather: core 1 writes its slice into slot 1 of core 0's gather region (the arena at GATHER_OFFSET, addressed through
  the arena's pages) and bumps valid on core 0. Core 0 copies its own slice into slot 0, waits for valid, writes the
  token's [8, 64] row out, and then gives core 1 a ready credit for the next token.

Implicit sync is off on every data movement kernel. Each core runs two data movement kernels and one compute kernel.
"""

import os

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

CORES = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)]
GRID = ttnn.CoreRangeSet([ttnn.CoreRange(CORES[0], CORES[1])])
M, K, N = 8, 128, 64
KT = K // 32
N_PER_CORE = N // len(CORES)
NUM_ITERS = 3
TINY_TILE = ttnn.Tile([M, 32])
TINY_BYTES = M * 32 * 2  # one bf16 [8, 32] tile
FULL_BYTES = 32 * 32 * 2
ROW_WORDS = 256  # arena rows: 1 KiB of uint32
ARENA_ROWS_PER_CORE = 16
IN0_OFFSET = 0  # KT tiny tiles
GATHER_OFFSET = 4 * 1024  # one slot per core

FEED_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    const uint32_t is_sender = get_arg(args::is_sender);
    const uint32_t sender_x = get_arg(args::sender_x);
    const uint32_t sender_y = get_arg(args::sender_y);
    const uint32_t num_receivers = get_arg(args::num_receivers);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    Noc noc;
    // The weights are resident in this core's per-core shard: announce them once.
    in1.reserve_back(kt);
    in1.push_back(kt);
    // NoC 1 runs the other way, so its multicast rectangle starts at the far corner.
    const bool noc0 = noc.get_noc_id() == 0;
    const uint32_t x_start = noc0 ? get_arg(args::receivers_x_min) : get_arg(args::receivers_x_max);
    const uint32_t y_start = noc0 ? get_arg(args::receivers_y_min) : get_arg(args::receivers_y_max);
    const uint32_t x_end = noc0 ? get_arg(args::receivers_x_max) : get_arg(args::receivers_x_min);
    const uint32_t y_end = noc0 ? get_arg(args::receivers_y_max) : get_arg(args::receivers_y_min);
    Semaphore<> ready(sem::mcast_ready);
    Semaphore<> valid(sem::mcast_valid);
    const uint32_t bytes = kt * in0.get_tile_size();
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        in0.reserve_back(kt);
        if (is_sender) {
            const LocalTensorAccessor<uint32_t> a(tensor::a);
            noc.async_read(a, in0, bytes, {.offset_bytes = iter * bytes}, {.offset_bytes = 0});
            noc.async_read_barrier();
            ready.wait(num_receivers);
            ready.set(0);
            noc.async_write_multicast(
                a,
                in0,
                bytes,
                num_receivers,
                {.offset_bytes = iter * bytes},
                {.noc_x_start = x_start, .noc_y_start = y_start, .noc_x_end = x_end, .noc_y_end = y_end,
                 .offset_bytes = 0});
            noc.async_write_barrier();  // The data lands before the flag that says it is there.
            valid.set(1);
            valid.set_multicast(noc, x_start, y_start, x_end, y_end, num_receivers);
            noc.async_write_barrier();
        } else {
            valid.set(0);
            ready.up(noc, sender_x, sender_y, 1);
            valid.wait(1);
        }
        in0.push_back(kt);
    }
}
"""

MATMUL_SOURCE = """
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer out(dfb::out);
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out);
    matmul_block_init(dfb::in0, dfb::in1, /*transpose=*/0, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/kt);
    in1.wait_front(kt);  // resident for every token
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        in0.wait_front(kt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(dfb::in0, dfb::in1, k, k, 0, /*transpose=*/0, 1, 1, kt);
        }
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();
        out.push_back(1);
        in0.pop_front(kt);
    }
    in1.pop_front(kt);
}
"""

GATHER_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t gather_offset = get_arg(args::gather_offset);
    constexpr uint32_t gather_page = get_arg(args::gather_page);
    const uint32_t is_root = get_arg(args::is_root);
    const uint32_t peer_x = get_arg(args::peer_x);
    const uint32_t peer_y = get_arg(args::peer_y);
    DataflowBuffer out(dfb::out);
    Noc noc;
    Semaphore<> ready(sem::gather_ready);
    Semaphore<> valid(sem::gather_valid);
    const uint32_t slice = out.get_tile_size();
    if (is_root) {
        const LocalTensorAccessor<uint32_t> region(tensor::arena);
        const auto result = TensorAccessor(tensor::result);
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            ready.up(noc, peer_x, peer_y, 1);  // The region is free for this token.
            out.wait_front(1);
            noc.async_write(out, region, slice, {.offset_bytes = 0}, {.offset_bytes = gather_offset});
            noc.async_write_barrier();
            out.pop_front(1);
            valid.down(1);  // The other core's slice is in slot 1.
            for (uint32_t s = 0; s < 2; ++s) {
                noc.async_write(
                    region, result, slice, {.offset_bytes = gather_offset + s * slice}, {.page_id = iter * 2 + s});
            }
            noc.async_write_barrier();
        }
    } else {
        const auto arena = TensorAccessor(tensor::arena);  // page p is row p of the arena, on the core that holds it
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            out.wait_front(1);
            ready.down(1);
            noc.async_write(out, arena, slice, {.offset_bytes = 0}, {.page_id = gather_page, .offset_bytes = slice});
            noc.async_write_barrier();  // The slice lands before the signal that says it is there.
            valid.up(noc, peer_x, peer_y, 1);
            out.pop_front(1);
        }
    }
}
"""


@pytest.fixture
def hybrid_mesh_device():
    os.environ["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    yield mesh
    ttnn.close_mesh_device(mesh)
    os.environ.pop("TT_METAL_ALLOCATOR_MODE_HYBRID", None)


def to_mesh(mesh, t, dtype, layout, memory_config, tile=None):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        tile=tile,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        device=mesh,
    )


def sharded(strategy, cores, shard_shape, per_core=False):
    config = ttnn.MemoryConfig(
        strategy,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]), list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )
    if per_core:
        config.experimental_set_per_core_allocation(True)
    return config


def test_generic_spec_op_mcast_matmul_gather(hybrid_mesh_device):
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") != "1":
        pytest.skip("The per-core borrowed weights DFB needs slow dispatch")
    mesh = hybrid_mesh_device
    torch.manual_seed(0)
    torch_a = torch.randn(NUM_ITERS * M, K).bfloat16()
    torch_w = (torch.randn(K, N) / 8).bfloat16()
    expected = torch_a.float() @ torch_w.float()

    # A filler on core 1 shifts that core's per-core allocations, so the two weight shards sit at different addresses.
    filler = to_mesh(
        mesh,
        torch.zeros(K, 32).bfloat16(),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, CORES[1:], (K, 32), per_core=True),
    )
    tensors = {
        "a": to_mesh(
            mesh,
            torch_a,
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, CORES[:1], (NUM_ITERS * M, K)),
            TINY_TILE,
        ),
        "w": to_mesh(
            mesh,
            torch_w,
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, CORES, (K, N_PER_CORE), per_core=True),
        ),
        "arena": to_mesh(
            mesh,
            torch.zeros(len(CORES) * ARENA_ROWS_PER_CORE, ROW_WORDS, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, CORES, (ARENA_ROWS_PER_CORE, ROW_WORDS)),
        ),
        "result": to_mesh(
            mesh,
            torch.zeros(NUM_ITERS * M, N).bfloat16(),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            TINY_TILE,
        ),
    }
    w_addresses = [tensors["w"].experimental_per_core_buffer_address(ttnn.MeshCoordinate(0, 0), c) for c in CORES]
    assert w_addresses[0] != w_addresses[1], w_addresses

    noc = [mesh.worker_core_from_logical_core(c) for c in CORES]
    sem_names = ("mcast_ready", "mcast_valid", "gather_ready", "gather_valid")
    feed_args = ["is_sender", "sender_x", "sender_y", "num_receivers"] + [
        f"receivers_{a}_{b}" for b in ("min", "max") for a in ("x", "y")
    ]
    common = {"kt": KT, "num_iters": NUM_ITERS}
    spec = E.ProgramSpec(
        name="mcast_matmul_gather",
        kernels=[
            E.KernelSpec(
                unique_id="feed",
                source=E.KernelSpec.SourceCode(code=FEED_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of("in0", "in0"), E.producer_of("in1", "in1")],
                semaphore_bindings=[E.SemaphoreBinding(semaphore_spec_name=n, accessor_name=n) for n in sem_names[:2]],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="a", accessor_name="a")],
                compile_time_args=common,
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=feed_args),
            ),
            E.KernelSpec(
                unique_id="matmul",
                source=E.KernelSpec.SourceCode(code=MATMUL_SOURCE),
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=[E.consumer_of("in0", "in0"), E.consumer_of("in1", "in1"), E.producer_of("out", "out")],
                compile_time_args=common,
            ),
            E.KernelSpec(
                unique_id="gather",
                source=E.KernelSpec.SourceCode(code=GATHER_SOURCE),
                hw_config=E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.consumer_of("out", "out")],
                semaphore_bindings=[E.SemaphoreBinding(semaphore_spec_name=n, accessor_name=n) for n in sem_names[2:]],
                tensor_bindings=[
                    E.TensorBinding(tensor_parameter_name=n, accessor_name=n) for n in ("arena", "result")
                ],
                compile_time_args={
                    "num_iters": NUM_ITERS,
                    "gather_offset": GATHER_OFFSET,
                    "gather_page": GATHER_OFFSET // (ROW_WORDS * 4),
                },
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["is_root", "peer_x", "peer_y"]),
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(
                unique_id="in0",
                entry_size=TINY_BYTES,
                num_entries=KT,
                data_format_metadata=ttnn.bfloat16,
                tile_format_metadata=TINY_TILE,
                borrowed_from="arena",
                advanced_options=E.DFBAdvancedOptions(borrowed_offset=IN0_OFFSET),
            ),
            E.DataflowBufferSpec(
                unique_id="in1",
                entry_size=FULL_BYTES,
                num_entries=KT,
                data_format_metadata=ttnn.bfloat16,
                borrowed_from="w",
            ),
            E.DataflowBufferSpec(
                unique_id="out",
                entry_size=TINY_BYTES,
                num_entries=2,
                data_format_metadata=ttnn.bfloat16,
                tile_format_metadata=TINY_TILE,
            ),
        ],
        semaphores=[E.SemaphoreSpec(unique_id=n, target_nodes=GRID) for n in sem_names],
        tensor_parameters=[E.TensorParameter(unique_id=n, spec=t.spec) for n, t in tensors.items()],
        work_units=[E.WorkUnitSpec(kernels=["feed", "matmul", "gather"], target_nodes=GRID)],
    )
    receivers = noc[1:]
    feed_values = {
        "is_sender": {CORES[0]: 1, CORES[1]: 0},
        "sender_x": {c: noc[0].x for c in CORES},
        "sender_y": {c: noc[0].y for c in CORES},
        "num_receivers": {c: len(receivers) for c in CORES},
        "receivers_x_min": {c: min(r.x for r in receivers) for c in CORES},
        "receivers_y_min": {c: min(r.y for r in receivers) for c in CORES},
        "receivers_x_max": {c: max(r.x for r in receivers) for c in CORES},
        "receivers_y_max": {c: max(r.y for r in receivers) for c in CORES},
    }
    gather_values = {
        "is_root": {CORES[0]: 1, CORES[1]: 0},
        "peer_x": {CORES[0]: noc[1].x, CORES[1]: noc[0].x},
        "peer_y": {CORES[0]: noc[1].y, CORES[1]: noc[0].y},
    }
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.ProgramRunArgs.KernelRunArgs(kernel="feed", runtime_arg_values=feed_values),
            E.ProgramRunArgs.KernelRunArgs(kernel="gather", runtime_arg_values=gather_values),
        ],
        tensor_args=tensors,
    )
    E.generic_op(list(tensors.values()), spec, run_args)

    assert_with_pcc(expected, ttnn.to_torch(tensors["result"]).float(), 0.999)
