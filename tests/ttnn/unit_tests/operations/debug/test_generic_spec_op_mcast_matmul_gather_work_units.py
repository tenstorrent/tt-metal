# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""tt-blaze's decode Mcast -> Matmul -> Gather -> Copy in its node classes, one node-disjoint WorkUnitSpec per class,
through ttnn.experimental.generic_op.

result[8, N] = a[8, 128] @ w[128, N] for NUM_ITERS tokens, 32 output columns per weight-grid core.

Variant "sender_is_root", blaze's legacy layout, on a (3, 1) or (8, 4) rectangle of cores:
- Class S, the rectangle's last core: `send` (the Mcast sender), `gather_root` and `copy`. No compute.
- Class G, every other core (the weight grid): `recv`, `matmul` and `gather_send`.

Variant "root_in_grid", on a (2, 1) or (8, 3) weight grid plus a sender core below it:
- Class S, the sender: `send` only.
- Class G minus root: `recv`, `matmul` and `gather_send`.
- Class root, grid core (0, 0): the same `recv` and `matmul` KernelSpecs, then `gather_root` (which also writes the
  root's own slice) and `copy`.

The kernels:
- `send` binds no DFB. It multicasts each token's act from its shard of `a` into act on the G cores, over rectangles
  that leave S out. A DFB is a multicast destination only for a kernel that binds it, and a LocalTensorAccessor never
  is, so the destination is a MulticastEndpoint at act's address: act is borrowed at ACT_OFFSET from the lockstep
  arena, which S also holds a shard of, at the same address. act holds two pushes, so `send` mirrors the receivers'
  ring: token `iter` goes to slot iter % 2.
- `recv` reserves act, ups mcast_ready on S, waits for mcast_valid and pushes act. Weights are resident in in1.
- `matmul` runs matmul_block into mm_out.
- `gather_send` waits until gather_ready reaches its token, writes its slice into the root's region (arena pages
  through TensorAccessor), and ups gather_valid on the root.
- `gather_root`, per token: (1) reserve `gathered`, the root-only DFB borrowed over the region, which blocks until
  `copy` has popped the previous token; (2) set gather_ready to the token and multicast it to the senders;
  (3) gather_valid.down(senders); (4) push `gathered`, with no data movement.
- `copy` (a data movement kernel, like blaze's Copy op) writes `gathered` to DRAM and pops it.

Step (1) before (2) is the sync rule: TOKEN_FIRST (a mutant, not part of the test) releases the token before the
reserve, the old ordering, and COPY_DELAY holds `copy` between its wait and its writes, so the senders' next slices
overwrite the region while `copy` still reads it. On Quasar craq-sim (sender_is_root, 3x1, COPY_DELAY 20000), with the
watcher and LLK asserts on, nothing asserts or hangs: every slice `copy` writes for tokens 0-2 holds the next token's
values, and only the last token is right. The gate-first order with the same delay is right on every token.

A work unit takes at most one compute kernel, so a downstream compute on the root would have to be fused into the
root's matmul kernel. Implicit sync is off on every data movement kernel.
"""

import math
import os

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_quasar

E = ttnn.experimental

# The root WorkUnitSpec holds 3 DM kernels; Quasar allows up to 6 per node, Blackhole 2.
pytestmark = pytest.mark.skipif(not is_quasar(), reason="Needs Quasar's up to 6 DM kernels per node")

M, K = 8, 128
KT = K // 32
N_PER_CORE = 32
NUM_ITERS = 4  # act's two-push ring wraps twice
ACT_PUSHES = 2  # act capacity, in pushes of KT tiles
TINY_TILE = ttnn.Tile([M, 32])
TINY_BYTES = M * 32 * 2
FULL_BYTES = 32 * 32 * 2
ROW_WORDS = 256  # arena rows: 1 KiB of uint32
ROW_BYTES = ROW_WORDS * 4
ACT_OFFSET = 0
GATHER_OFFSET = ACT_OFFSET + ACT_PUSHES * KT * TINY_BYTES  # one TINY_BYTES slot per slice, two per arena page
SLOTS_PER_PAGE = ROW_BYTES // TINY_BYTES

RECTS_CPP = """
struct Rect {
    uint32_t x_start, y_start, x_end, y_end, num_cores;
};

// The rectangles from the r0_* / r1_* args; NoC 1 runs the other way, so each starts at its far corner there.
inline void load_rects(const Noc& noc, Rect (&rects)[2]) {
    const bool noc0 = noc.get_noc_id() == 0;
    const uint32_t r[2][5] = {
        {get_arg(args::r0_x_min), get_arg(args::r0_y_min), get_arg(args::r0_x_max), get_arg(args::r0_y_max),
         get_arg(args::r0_cores)},
        {get_arg(args::r1_x_min), get_arg(args::r1_y_min), get_arg(args::r1_x_max), get_arg(args::r1_y_max),
         get_arg(args::r1_cores)},
    };
    for (uint32_t i = 0; i < 2; ++i) {
        rects[i] = noc0 ? Rect{r[i][0], r[i][1], r[i][2], r[i][3], r[i][4]} : Rect{r[i][2], r[i][3], r[i][0], r[i][1], r[i][4]};
    }
}
"""

SEND_SOURCE = (
    """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"
"""
    + RECTS_CPP
    + """
void kernel_main() {
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t num_receivers = get_arg(args::num_receivers);
    constexpr uint32_t num_rects = get_arg(args::num_rects);
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    constexpr uint32_t act_offset = get_arg(args::act_offset);
    constexpr uint32_t act_pushes = get_arg(args::act_pushes);
    Noc noc;
    Rect rects[2];
    load_rects(noc, rects);
    Semaphore<> ready(sem::mcast_ready);
    Semaphore<> valid(sem::mcast_valid);
    const LocalTensorAccessor<uint32_t> a(tensor::a);
    // act is borrowed at act_offset from the lockstep arena, which has a shard here too, at the same address.
    const LocalTensorAccessor<uint32_t> arena(tensor::arena);
    const uint32_t act_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&arena[act_offset / sizeof(uint32_t)]));
    const MulticastEndpoint receivers;
    const uint32_t bytes = kt * tile_bytes;
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        ready.wait(num_receivers);  // every receiver has reserved this token's act slot
        ready.set(0);
        // The receivers' ring: each push is one token, so token iter is in slot iter % act_pushes.
        const uint32_t slot_addr = act_addr + (iter % act_pushes) * bytes;
        for (uint32_t i = 0; i < num_rects; ++i) {
            const Rect& r = rects[i];
            noc.async_write_multicast(
                a,
                receivers,
                bytes,
                r.num_cores,
                {.offset_bytes = iter * bytes},
                {.noc_x_start = r.x_start, .noc_y_start = r.y_start, .noc_x_end = r.x_end, .noc_y_end = r.y_end,
                 .addr = slot_addr});
        }
        noc.async_write_barrier();  // The data lands before the flag that says it is there.
        valid.set(1);
        for (uint32_t i = 0; i < num_rects; ++i) {
            const Rect& r = rects[i];
            valid.set_multicast(noc, r.x_start, r.y_start, r.x_end, r.y_end, r.num_cores);
        }
        noc.async_write_barrier();
    }
}
"""
)

RECV_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t kt = get_arg(args::kt);
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    DataflowBuffer act(dfb::act);
    DataflowBuffer in1(dfb::in1);
    Noc noc;
    in1.reserve_back(kt);  // resident weights
    in1.push_back(kt);
    Semaphore<> ready(sem::mcast_ready);
    Semaphore<> valid(sem::mcast_valid);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        act.reserve_back(kt);
        valid.set(0);
        ready.up(noc, get_arg(args::sender_x), get_arg(args::sender_y), 1);
        valid.wait(1);
        act.push_back(kt);
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
    DataflowBuffer act(dfb::act);
    DataflowBuffer in1(dfb::in1);
    DataflowBuffer mm_out(dfb::mm_out);
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::act, dfb::in1, dfb::mm_out);
    matmul_block_init(dfb::act, dfb::in1, /*transpose=*/0, /*ct_dim=*/1, /*rt_dim=*/1, /*kt_dim=*/kt);
    in1.wait_front(kt);
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        act.wait_front(kt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(dfb::act, dfb::in1, k, k, 0, /*transpose=*/0, 1, 1, kt);
        }
        tile_regs_commit();
        mm_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::mm_out);
        tile_regs_release();
        mm_out.push_back(1);
        act.pop_front(kt);
    }
    in1.pop_front(kt);
}
"""

GATHER_SEND_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t region_page = get_arg(args::region_page);
    constexpr uint32_t slots_per_page = get_arg(args::slots_per_page);
    const uint32_t slot = get_arg(args::slot);
    DataflowBuffer mm_out(dfb::mm_out);
    Noc noc;
    Semaphore<> ready(sem::gather_ready);
    Semaphore<> valid(sem::gather_valid);
    const uint32_t slice = mm_out.get_tile_size();
    const auto arena = TensorAccessor(tensor::arena);  // page p is row p of the arena, on the core that holds it
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        mm_out.wait_front(1);
        ready.wait_min(iter + 1);
        noc.async_write(
            mm_out,
            arena,
            slice,
            {.offset_bytes = 0},
            {.page_id = region_page + slot / slots_per_page, .offset_bytes = (slot % slots_per_page) * slice});
        noc.async_write_barrier();  // The slice lands before the signal that says it is there.
        valid.up(noc, get_arg(args::root_x), get_arg(args::root_y), 1);
        mm_out.pop_front(1);
    }
}
"""

GATHER_ROOT_SOURCE = (
    """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"
"""
    + RECTS_CPP
    + """
void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t num_slices = get_arg(args::num_slices);
    constexpr uint32_t num_senders = get_arg(args::num_senders);
    constexpr uint32_t num_rects = get_arg(args::num_rects);
    Noc noc;
    Rect rects[2];
    load_rects(noc, rects);
    Semaphore<> ready(sem::gather_ready);
    Semaphore<> valid(sem::gather_valid);
    DataflowBuffer gathered(dfb::gathered);
#ifdef ROOT_SLICE
    constexpr uint32_t gather_offset = get_arg(args::gather_offset);
    DataflowBuffer mm_out(dfb::mm_out);
    const LocalTensorAccessor<uint32_t> region(tensor::arena);
#endif
    auto release_token = [&](uint32_t token) {
        ready.set(token);
        for (uint32_t i = 0; i < num_rects; ++i) {
            const Rect& r = rects[i];
            ready.set_multicast(noc, r.x_start, r.y_start, r.x_end, r.y_end, r.num_cores);
        }
        noc.async_write_barrier();
    };
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
#ifdef TOKEN_FIRST
        release_token(iter + 1);
        gathered.reserve_back(num_slices);
#else
        gathered.reserve_back(num_slices);  // copy has popped the previous token's region
        release_token(iter + 1);            // only now may the senders write this token's slices
#endif
#ifdef ROOT_SLICE
        mm_out.wait_front(1);
        noc.async_write(mm_out, region, mm_out.get_tile_size(), {.offset_bytes = 0}, {.offset_bytes = gather_offset});
        noc.async_write_barrier();
        mm_out.pop_front(1);
#endif
        valid.down(num_senders);  // every sender's slice is in its slot
        gathered.push_back(num_slices);  // the region, re-announced; no data moves
    }
}
"""
)

COPY_SOURCE = """
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t num_slices = get_arg(args::num_slices);
    DataflowBuffer gathered(dfb::gathered);
    Noc noc;
    const auto result = TensorAccessor(tensor::result);
    const uint32_t slice = gathered.get_entry_size();
    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        gathered.wait_front(num_slices);
#ifdef COPY_DELAY
        for (volatile uint32_t spin = 0; spin < COPY_DELAY; spin = spin + 1) {
        }
#endif
        for (uint32_t c = 0; c < num_slices; ++c) {
            noc.async_write(gathered, result, slice, {.offset_bytes = c * slice}, {.page_id = iter * num_slices + c});
        }
        noc.async_write_barrier();
        gathered.pop_front(num_slices);
    }
}
"""

# Layouts tt-metal rejects, with the error each gives.
REJECTED_LAYOUTS = {
    "second_compute_kernel_on_root": "WorkUnitSpec 'root' has more than one compute kernel",
    "root_also_in_grid": "WorkUnitSpecs 'grid' and 'root' overlap in target nodes",
}


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


def node_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def sharded(strategy, cores, shard_shape, per_core=False):
    config = ttnn.MemoryConfig(
        strategy,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(node_set(cores), list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )
    if per_core:
        config.experimental_set_per_core_allocation(True)
    return config


def classes(variant, size_x, size_y):
    """Returns (sender, root, grid): the weight grid's cores in slot order, and the sender and root cores."""
    if variant == "sender_is_root":
        cores = [ttnn.CoreCoord(x, y) for y in range(size_y) for x in range(size_x)]
        return cores[-1], cores[-1], cores[:-1]
    grid = [ttnn.CoreCoord(x, y) for y in range(size_y) for x in range(size_x)]
    return ttnn.CoreCoord(0, size_y), grid[0], grid


def cover(size_x, size_y, excluded):
    """Rectangles (lo, hi) covering a size_x x size_y rectangle of cores without `excluded`, its first or last core."""
    last_x, last_y = size_x - 1, size_y - 1
    if excluded is None:
        return [((0, 0), (last_x, last_y))]
    if (excluded.x, excluded.y) == (0, 0):
        rects = [((1, 0), (last_x, 0))] if size_x > 1 else []
        return rects + ([((0, 1), (last_x, last_y))] if size_y > 1 else [])
    rects = [((0, 0), (last_x, last_y - 1))] if size_y > 1 else []
    return rects + ([((0, last_y), (last_x - 1, last_y))] if size_x > 1 else [])


def rect_args(mesh, rects):
    args = {"num_rects": len(rects)}
    for i in range(2):
        lo, hi = rects[i] if i < len(rects) else ((0, 0), (0, 0))
        n_lo = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*lo))
        n_hi = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*hi))
        args.update(
            {
                f"r{i}_x_min": min(n_lo.x, n_hi.x),
                f"r{i}_y_min": min(n_lo.y, n_hi.y),
                f"r{i}_x_max": max(n_lo.x, n_hi.x),
                f"r{i}_y_max": max(n_lo.y, n_hi.y),
                f"r{i}_cores": (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) if i < len(rects) else 0,
            }
        )
    return args


def build(mesh, variant, size_x, size_y, layout=None, defines=None):
    """Returns (spec, run_args, tensors, expected, keep_alive). `layout` names one of REJECTED_LAYOUTS; `defines` go to
    gather_root and copy (the mutant)."""
    sender, root, grid = classes(variant, size_x, size_y)
    root_has_slice = variant == "root_in_grid"
    senders = [c for c in grid if c != root]  # the cores that gather_send runs on
    num_slices = len(grid)
    num_cols = N_PER_CORE * num_slices
    arena_cores = grid + [sender] if root_has_slice else grid + [root]
    arena_rows_per_core = GATHER_OFFSET // ROW_BYTES + math.ceil(num_slices / SLOTS_PER_PAGE)

    torch.manual_seed(0)
    torch_a = torch.randn(NUM_ITERS * M, K).bfloat16()
    torch_w = (torch.randn(K, num_cols) / 8).bfloat16()
    expected = torch_a.float() @ torch_w.float()

    # Fillers on every other grid core shift their per-core allocations, so the weight shards do not share one address.
    filler = to_mesh(
        mesh,
        torch.zeros(K, 32 * len(grid[1::2])).bfloat16(),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid[1::2], (K, 32), per_core=True),
    )
    tensors = {
        "a": to_mesh(
            mesh,
            torch_a,
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [sender], (NUM_ITERS * M, K)),
            TINY_TILE,
        ),
        "w": to_mesh(
            mesh,
            torch_w,
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid, (K, N_PER_CORE), per_core=True),
        ),
        # Lockstep over the grid and the sender: act and the region are at the same address on every core.
        "arena": to_mesh(
            mesh,
            torch.zeros(len(arena_cores) * arena_rows_per_core, ROW_WORDS, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, arena_cores, (arena_rows_per_core, ROW_WORDS)),
        ),
        "result": to_mesh(
            mesh,
            torch.zeros(NUM_ITERS * M, num_cols).bfloat16(),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            TINY_TILE,
        ),
    }
    w_addresses = {tensors["w"].experimental_per_core_buffer_address(ttnn.MeshCoordinate(0, 0), c) for c in grid}
    assert len(w_addresses) > 1, w_addresses
    shard_order = ttnn.corerange_to_cores(tensors["arena"].memory_config().shard_spec.grid, row_wise=True)
    root_shard = [(c.x, c.y) for c in shard_order].index((root.x, root.y))
    region_page = root_shard * arena_rows_per_core + GATHER_OFFSET // ROW_BYTES

    noc = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
    if variant == "sender_is_root":
        mcast_rects = gather_rects = cover(size_x, size_y, root)
    else:
        mcast_rects, gather_rects = cover(size_x, size_y, None), cover(size_x, size_y, root)
    mcast_args, gather_args = rect_args(mesh, mcast_rects), rect_args(mesh, gather_rects)
    assert sum(mcast_args[f"r{i}_cores"] for i in range(2)) == len(grid)
    assert sum(gather_args[f"r{i}_cores"] for i in range(2)) == len(senders)

    def sems(*names):
        return [E.SemaphoreBinding(semaphore_spec_name=n, accessor_name=n) for n in names]

    def dm(unique_id, source, writer=False, **fields):
        make_config = E.create_writer_datamovement_config if writer else E.create_reader_datamovement_config
        return E.KernelSpec(
            unique_id=unique_id,
            source=E.KernelSpec.SourceCode(code=source),
            hw_config=make_config(disable_dfb_implicit_sync_for_all=True),
            **fields,
        )

    common = {"kt": KT, "num_iters": NUM_ITERS}
    root_defines = {**(defines or {}), **({"ROOT_SLICE": "1"} if root_has_slice else {})}
    gather_root_dfbs = [E.producer_of("gathered", "gathered")]
    if root_has_slice:
        gather_root_dfbs.append(E.consumer_of("mm_out", "mm_out"))
    kernels = [
        dm(
            "send",
            SEND_SOURCE,
            semaphore_bindings=sems("mcast_ready", "mcast_valid"),
            tensor_bindings=[E.TensorBinding(tensor_parameter_name=n, accessor_name=n) for n in ("a", "arena")],
            compile_time_args={
                **common,
                **mcast_args,
                "num_receivers": len(grid),
                "tile_bytes": TINY_BYTES,
                "act_offset": ACT_OFFSET,
                "act_pushes": ACT_PUSHES,
            },
        ),
        dm(
            "recv",
            RECV_SOURCE,
            dfb_bindings=[E.producer_of("act", "act"), E.producer_of("in1", "in1")],
            semaphore_bindings=sems("mcast_ready", "mcast_valid"),
            compile_time_args={**common, "sender_x": noc(sender).x, "sender_y": noc(sender).y},
        ),
        E.KernelSpec(
            unique_id="matmul",
            source=E.KernelSpec.SourceCode(code=MATMUL_SOURCE),
            hw_config=E.ComputeHardwareConfig(),
            dfb_bindings=[E.consumer_of("act", "act"), E.consumer_of("in1", "in1"), E.producer_of("mm_out", "mm_out")],
            compile_time_args=common,
        ),
        dm(
            "gather_send",
            GATHER_SEND_SOURCE,
            writer=True,
            dfb_bindings=[E.consumer_of("mm_out", "mm_out")],
            semaphore_bindings=sems("gather_ready", "gather_valid"),
            tensor_bindings=[E.TensorBinding(tensor_parameter_name="arena", accessor_name="arena")],
            compile_time_args={
                "num_iters": NUM_ITERS,
                "region_page": region_page,
                "slots_per_page": SLOTS_PER_PAGE,
                "root_x": noc(root).x,
                "root_y": noc(root).y,
            },
            runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["slot"]),
        ),
        dm(
            "gather_root",
            GATHER_ROOT_SOURCE,
            writer=True,
            compiler_options=E.KernelSpec.CompilerOptions(defines=root_defines),
            dfb_bindings=gather_root_dfbs,
            semaphore_bindings=sems("gather_ready", "gather_valid"),
            tensor_bindings=[E.TensorBinding(tensor_parameter_name="arena", accessor_name="arena")],
            compile_time_args={
                "num_iters": NUM_ITERS,
                "num_slices": num_slices,
                "num_senders": len(senders),
                "gather_offset": GATHER_OFFSET,
                **gather_args,
            },
        ),
        dm(
            "copy",
            COPY_SOURCE,
            compiler_options=E.KernelSpec.CompilerOptions(defines=defines or {}),
            dfb_bindings=[E.consumer_of("gathered", "gathered")],
            tensor_bindings=[E.TensorBinding(tensor_parameter_name="result", accessor_name="result")],
            compile_time_args={"num_iters": NUM_ITERS, "num_slices": num_slices},
        ),
    ]
    if variant == "sender_is_root":
        work_units = [
            E.WorkUnitSpec(name="root", kernels=["send", "gather_root", "copy"], target_nodes=root),
            E.WorkUnitSpec(name="grid", kernels=["recv", "matmul", "gather_send"], target_nodes=node_set(grid)),
        ]
    else:
        root_kernels = ["recv", "matmul", "gather_root", "copy"]
        grid_nodes = senders
        if layout == "second_compute_kernel_on_root":  # a downstream compute kernel of its own next to the matmul
            kernels.append(
                E.KernelSpec(
                    unique_id="post",
                    source=E.KernelSpec.SourceCode(code="void kernel_main() {}"),
                    hw_config=E.ComputeHardwareConfig(),
                )
            )
            root_kernels.append("post")
        elif layout == "root_also_in_grid":
            grid_nodes = grid
        work_units = [
            E.WorkUnitSpec(name="sender", kernels=["send"], target_nodes=sender),
            E.WorkUnitSpec(name="grid", kernels=["recv", "matmul", "gather_send"], target_nodes=node_set(grid_nodes)),
            E.WorkUnitSpec(name="root", kernels=root_kernels, target_nodes=root),
        ]

    def tiny_dfb(unique_id, num_entries, offset=None):
        borrowed = {}
        if offset is not None:
            borrowed = dict(borrowed_from="arena", advanced_options=E.DFBAdvancedOptions(borrowed_offset=offset))
        return E.DataflowBufferSpec(
            unique_id=unique_id,
            entry_size=TINY_BYTES,
            num_entries=num_entries,
            data_format_metadata=ttnn.bfloat16,
            tile_format_metadata=TINY_TILE,
            **borrowed,
        )

    spec = E.ProgramSpec(
        name=f"mcast_matmul_gather_{variant}_{size_x}x{size_y}",
        kernels=kernels,
        dataflow_buffers=[
            tiny_dfb("act", ACT_PUSHES * KT, ACT_OFFSET),
            E.DataflowBufferSpec(
                unique_id="in1",
                entry_size=FULL_BYTES,
                num_entries=KT,
                data_format_metadata=ttnn.bfloat16,
                borrowed_from="w",
            ),
            tiny_dfb("mm_out", 2),
            tiny_dfb("gathered", num_slices, GATHER_OFFSET),
        ],
        semaphores=[
            E.SemaphoreSpec(unique_id="mcast_ready", target_nodes=sender),
            # The sender and the root each multicast from their own copy, so they hold the semaphore too.
            E.SemaphoreSpec(unique_id="mcast_valid", target_nodes=node_set(grid + [sender])),
            E.SemaphoreSpec(unique_id="gather_ready", target_nodes=node_set(senders + [root])),
            E.SemaphoreSpec(unique_id="gather_valid", target_nodes=root),
        ],
        tensor_parameters=[E.TensorParameter(unique_id=n, spec=t.spec) for n, t in tensors.items()],
        work_units=work_units,
    )
    slot_base = 1 if root_has_slice else 0  # the root's own slice takes slot 0
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.ProgramRunArgs.KernelRunArgs(
                kernel="gather_send", runtime_arg_values={"slot": {c: slot_base + i for i, c in enumerate(senders)}}
            ),
        ],
        tensor_args=tensors,
    )
    return spec, run_args, tensors, expected, filler


def skip_unless_runnable(mesh, variant, size_x, size_y):
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") != "1":
        pytest.skip("The per-core borrowed weights DFB needs slow dispatch")
    rows = size_y + (variant == "root_in_grid")
    grid_size = mesh.compute_with_storage_grid_size()
    if grid_size.x < size_x or grid_size.y < rows:
        pytest.skip(f"Needs {size_x}x{rows} cores")


def run(mesh, variant, size_x, size_y, defines=None):
    """Returns (expected, result), each [NUM_ITERS * M, N] float32."""
    spec, run_args, tensors, expected, _ = build(mesh, variant, size_x, size_y, defines=defines)
    E.generic_op(list(tensors.values()), spec, run_args)
    return expected, ttnn.to_torch(tensors["result"]).float()


@pytest.mark.parametrize(
    "variant, size_x, size_y",
    [
        ("sender_is_root", 3, 1),
        ("sender_is_root", 8, 4),
        ("root_in_grid", 2, 1),
        ("root_in_grid", 8, 3),
    ],
    ids=["sender_is_root-3x1", "sender_is_root-8x4", "root_in_grid-2x1+1", "root_in_grid-8x3+1"],
)
def test_generic_spec_op_mcast_matmul_gather_work_units(hybrid_mesh_device, variant, size_x, size_y):
    skip_unless_runnable(hybrid_mesh_device, variant, size_x, size_y)
    expected, result = run(hybrid_mesh_device, variant, size_x, size_y)
    for i in range(NUM_ITERS):
        rows = slice(i * M, (i + 1) * M)
        assert_with_pcc(expected[rows], result[rows], 0.999)


@pytest.mark.parametrize("layout", list(REJECTED_LAYOUTS))
def test_generic_spec_op_mcast_matmul_gather_work_units_rejects(hybrid_mesh_device, expect_error, layout):
    skip_unless_runnable(hybrid_mesh_device, "root_in_grid", 2, 1)
    spec, run_args, tensors, _, _ = build(hybrid_mesh_device, "root_in_grid", 2, 1, layout)
    with expect_error(RuntimeError, REJECTED_LAYOUTS[layout]):
        E.generic_op(list(tensors.values()), spec, run_args)
