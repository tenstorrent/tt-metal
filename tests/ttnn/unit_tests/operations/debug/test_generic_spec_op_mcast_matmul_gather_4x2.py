# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""tt-blaze's decode Mcast -> Matmul -> Gather on a 4x2 grid of cores, through ttnn.experimental.generic_op.

The eight-core version of test_generic_spec_op_mcast_matmul_gather.py: out[8, 256] = a[8, 128] @ w[128, 256] for
NUM_ITERS tokens, 32 output columns per core.

- Core (0, 0) multicasts each token's in0 to the other seven cores. One rectangle cannot leave the sender out, so it
  sends to two: row 0 without itself, and row 1 (on NoC 1 each rectangle starts at its far corner). It waits until all
  seven receivers have reserved their in0 (mcast_ready counts to 7), and sets mcast_valid on both rectangles.
- Each core's 32 weight columns are per-core allocated and borrowed as in1, resident for every token.
- Gather: core c writes its [8, 32] slice into slot c of core 0's gather region (the arena at GATHER_OFFSET, two slots
  per arena page) and increments gather_valid on core 0, which waits for 7. Core 0 then writes the [8, 256] row out
  and multicasts the next token number into gather_ready on the other cores, which each wait until it reaches their
  token, so a slot is not overwritten before core 0 has read it.

Implicit sync is off on every data movement kernel. Each core runs two data movement kernels and one compute kernel.
"""

import os

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

GRID_X, GRID_Y = 4, 2
CORES = [ttnn.CoreCoord(x, y) for y in range(GRID_Y) for x in range(GRID_X)]  # core index c = y * GRID_X + x
NUM_CORES = len(CORES)
GRID = ttnn.CoreRangeSet([ttnn.CoreRange(CORES[0], CORES[-1])])
# The two multicast rectangles that cover every core but the sender (0, 0).
RECTS = [
    (ttnn.CoreCoord(1, 0), ttnn.CoreCoord(GRID_X - 1, 0)),
    (ttnn.CoreCoord(0, 1), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1)),
]
M, K = 8, 128
KT = K // 32
N_PER_CORE = 32
N = N_PER_CORE * NUM_CORES
NUM_ITERS = 3
TINY_TILE = ttnn.Tile([M, 32])
TINY_BYTES = M * 32 * 2
FULL_BYTES = 32 * 32 * 2
ROW_WORDS = 256  # arena rows: 1 KiB of uint32
ARENA_ROWS_PER_CORE = 16
IN0_OFFSET = 0
GATHER_OFFSET = 4 * 1024  # NUM_CORES slots of TINY_BYTES, two per arena page

FEED_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

struct Rect {
    uint32_t x_start, y_start, x_end, y_end, num_cores;
};

void kernel_main() {
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t num_receivers = get_arg(args::num_receivers);
    const uint32_t is_sender = get_arg(args::is_sender);
    DataflowBuffer in0(dfb::in0);
    DataflowBuffer in1(dfb::in1);
    Noc noc;
    in1.reserve_back(kt);  // resident weights
    in1.push_back(kt);
    // NoC 1 runs the other way, so each rectangle starts at its far corner there.
    const bool noc0 = noc.get_noc_id() == 0;
    auto rect = [&](uint32_t x_min, uint32_t y_min, uint32_t x_max, uint32_t y_max, uint32_t n) {
        return noc0 ? Rect{x_min, y_min, x_max, y_max, n} : Rect{x_max, y_max, x_min, y_min, n};
    };
    const Rect rects[2] = {
        rect(get_arg(args::r0_x_min), get_arg(args::r0_y_min), get_arg(args::r0_x_max), get_arg(args::r0_y_max),
             get_arg(args::r0_cores)),
        rect(get_arg(args::r1_x_min), get_arg(args::r1_y_min), get_arg(args::r1_x_max), get_arg(args::r1_y_max),
             get_arg(args::r1_cores)),
    };
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
            for (const Rect& r : rects) {
                noc.async_write_multicast(
                    a,
                    in0,
                    bytes,
                    r.num_cores,
                    {.offset_bytes = iter * bytes},
                    {.noc_x_start = r.x_start, .noc_y_start = r.y_start, .noc_x_end = r.x_end, .noc_y_end = r.y_end,
                     .offset_bytes = 0});
            }
            noc.async_write_barrier();  // The data lands before the flag that says it is there.
            valid.set(1);
            for (const Rect& r : rects) {
                valid.set_multicast(noc, r.x_start, r.y_start, r.x_end, r.y_end, r.num_cores);
            }
            noc.async_write_barrier();
        } else {
            valid.set(0);
            ready.up(noc, get_arg(args::sender_x), get_arg(args::sender_y), 1);
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
    in1.wait_front(kt);
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
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    constexpr uint32_t gather_offset = get_arg(args::gather_offset);
    constexpr uint32_t gather_page = get_arg(args::gather_page);
    constexpr uint32_t slots_per_page = get_arg(args::slots_per_page);
    const uint32_t core_index = get_arg(args::core_index);
    DataflowBuffer out(dfb::out);
    Noc noc;
    Semaphore<> ready(sem::gather_ready);
    Semaphore<> valid(sem::gather_valid);
    const uint32_t slice = out.get_tile_size();
    if (core_index == 0) {
        const bool noc0 = noc.get_noc_id() == 0;
        const uint32_t rects[2][5] = {
            {get_arg(args::r0_x_min), get_arg(args::r0_y_min), get_arg(args::r0_x_max), get_arg(args::r0_y_max),
             get_arg(args::r0_cores)},
            {get_arg(args::r1_x_min), get_arg(args::r1_y_min), get_arg(args::r1_x_max), get_arg(args::r1_y_max),
             get_arg(args::r1_cores)},
        };
        const LocalTensorAccessor<uint32_t> region(tensor::arena);
        const auto result = TensorAccessor(tensor::result);
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            // Token `iter` may use the region: tell every other core.
            ready.set(iter + 1);
            for (const auto& r : rects) {
                ready.set_multicast(
                    noc, noc0 ? r[0] : r[2], noc0 ? r[1] : r[3], noc0 ? r[2] : r[0], noc0 ? r[3] : r[1], r[4]);
            }
            noc.async_write_barrier();
            out.wait_front(1);
            noc.async_write(out, region, slice, {.offset_bytes = 0}, {.offset_bytes = gather_offset});
            noc.async_write_barrier();
            out.pop_front(1);
            valid.down(num_cores - 1);  // every other core's slice is in its slot
            for (uint32_t c = 0; c < num_cores; ++c) {
                noc.async_write(
                    region, result, slice, {.offset_bytes = gather_offset + c * slice}, {.page_id = iter * num_cores + c});
            }
            noc.async_write_barrier();
        }
    } else {
        const auto arena = TensorAccessor(tensor::arena);  // page p is row p of the arena, on the core that holds it
        const uint32_t root_x = get_arg(args::root_x);
        const uint32_t root_y = get_arg(args::root_y);
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            out.wait_front(1);
            ready.wait_min(iter + 1);
            noc.async_write(
                out,
                arena,
                slice,
                {.offset_bytes = 0},
                {.page_id = gather_page + core_index / slots_per_page, .offset_bytes = (core_index % slots_per_page) * slice});
            noc.async_write_barrier();  // The slice lands before the signal that says it is there.
            valid.up(noc, root_x, root_y, 1);
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


def test_generic_spec_op_mcast_matmul_gather_4x2(hybrid_mesh_device):
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") != "1":
        pytest.skip("The per-core borrowed weights DFB needs slow dispatch")
    mesh = hybrid_mesh_device
    grid_size = mesh.compute_with_storage_grid_size()
    if grid_size.x < GRID_X or grid_size.y < GRID_Y:
        pytest.skip(f"Needs a {GRID_X}x{GRID_Y} grid")
    torch.manual_seed(0)
    torch_a = torch.randn(NUM_ITERS * M, K).bfloat16()
    torch_w = (torch.randn(K, N) / 8).bfloat16()
    expected = torch_a.float() @ torch_w.float()

    # Fillers on the odd cores shift their per-core allocations, so the weight shards do not share one address.
    filler = to_mesh(
        mesh,
        torch.zeros(K, 32 * (NUM_CORES // 2)).bfloat16(),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, CORES[1::2], (K, 32), per_core=True),
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
            torch.zeros(NUM_CORES * ARENA_ROWS_PER_CORE, ROW_WORDS, dtype=torch.int32),
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
    w_addresses = {tensors["w"].experimental_per_core_buffer_address(ttnn.MeshCoordinate(0, 0), c) for c in CORES}
    assert len(w_addresses) > 1, w_addresses

    noc = [mesh.worker_core_from_logical_core(c) for c in CORES]
    rect_values = {}
    for i, (lo, hi) in enumerate(RECTS):
        lo_noc, hi_noc = mesh.worker_core_from_logical_core(lo), mesh.worker_core_from_logical_core(hi)
        rect_values.update(
            {
                f"r{i}_x_min": min(lo_noc.x, hi_noc.x),
                f"r{i}_y_min": min(lo_noc.y, hi_noc.y),
                f"r{i}_x_max": max(lo_noc.x, hi_noc.x),
                f"r{i}_y_max": max(lo_noc.y, hi_noc.y),
                f"r{i}_cores": (hi.x - lo.x + 1) * (hi.y - lo.y + 1),
            }
        )
    assert sum(rect_values[f"r{i}_cores"] for i in range(len(RECTS))) == NUM_CORES - 1
    rect_names = list(rect_values)
    sem_names = ("mcast_ready", "mcast_valid", "gather_ready", "gather_valid")
    feed_args = ["is_sender", "sender_x", "sender_y"] + rect_names
    gather_args = ["core_index", "root_x", "root_y"] + rect_names
    common = {"kt": KT, "num_iters": NUM_ITERS}
    spec = E.ProgramSpec(
        name="mcast_matmul_gather_4x2",
        kernels=[
            E.KernelSpec(
                unique_id="feed",
                source=E.KernelSpec.SourceCode(code=FEED_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of("in0", "in0"), E.producer_of("in1", "in1")],
                semaphore_bindings=[E.SemaphoreBinding(semaphore_spec_name=n, accessor_name=n) for n in sem_names[:2]],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="a", accessor_name="a")],
                compile_time_args={**common, "num_receivers": NUM_CORES - 1},
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
                    "num_cores": NUM_CORES,
                    "gather_offset": GATHER_OFFSET,
                    "gather_page": GATHER_OFFSET // (ROW_WORDS * 4),
                    "slots_per_page": ROW_WORDS * 4 // TINY_BYTES,
                },
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=gather_args),
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
    every = lambda value: {c: value for c in CORES}  # noqa: E731
    feed_values = {
        "is_sender": {c: int(i == 0) for i, c in enumerate(CORES)},
        "sender_x": every(noc[0].x),
        "sender_y": every(noc[0].y),
        **{name: every(value) for name, value in rect_values.items()},
    }
    gather_values = {
        "core_index": {c: i for i, c in enumerate(CORES)},
        "root_x": every(noc[0].x),
        "root_y": every(noc[0].y),
        **{name: every(value) for name, value in rect_values.items()},
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
