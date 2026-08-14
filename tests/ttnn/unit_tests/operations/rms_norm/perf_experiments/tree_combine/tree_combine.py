# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off of rms_norm's cross-core stat COMBINE.

Everything except the combine is held trivial and constant: each core's `B`
float32 partial-stat tiles are already resident in its L1 shard (the op's
`cb_sq_partials`, which the real op fills from `sum_of_squares`), and the result
is the finalized `1/rms` tile landing in every member's L1 output shard (the op's
`cb_rms_recip`).  No DRAM, no gamma, no scale pass — so the measured delta is the
combine topology alone.

Variants (identical arithmetic, identical precision contract):

  flat_root   — the op's CURRENT approach.  Every core NoC-writes its B stat
                tiles into the ROOT's `cb_gathered_partials` page `r*s + slice`
                plus one semaphore increment; the root waits for `s` arrivals,
                runs ONE `ckl::reduce<SUM, REDUCE_ROW>` over the (B, s) gathered
                block with the fused finalize (*1/W, +eps, rsqrt), and
                `mcast_pipe`-broadcasts the B finalized tiles over the rect.

  tree        — two-level hierarchy.  The group's `s` cores are partitioned into
                `L = s/K` subgroups of `K` consecutive cores (for a rect with
                K = rect_w that is exactly one grid ROW per subgroup; on a 1-D
                line it is a hierarchy ON the line).  Level 1: each core writes
                its B tiles to its subgroup LEADER, which elementwise-sums the K
                contributions per row (partial stat tiles stay in per-column
                form, so no information is lost).  Level 2: the L leaders write
                their B partials to the root, which runs the SAME
                `reduce<SUM, REDUCE_ROW>` + finalize over the (B, L) block.
                Level 3: the same single `mcast_pipe` broadcast over the rect.
                Sum-then-collapse == collapse-then-sum, so the arithmetic is the
                baseline's, with the fan-in cut from `s` to `K` then `L`.

Raw-LLK note (level-1 sum): the level-1 sum must NOT collapse within the tile —
that is `ckl::reduce`'s `ReduceWithinTile::Skip`, which is UNREACHABLE through
`compute_kernel_lib::reduce()`: the "Skip is AccumulateViaAdd-only" static_assert
(reduce_helpers_compute.inl:885-891) sits AFTER the `if constexpr
(AccumulateViaAdd) { ...; return; }` block, so it is not in a discarded statement
and fires for the AccumulateViaAdd instantiation too.  Level 1 is therefore
written as raw pairwise FPU `add_tiles(acc_to_dest=true)` — the datapath
`examples/tensix_all_reduce_compute` measured at 5.92x over the SFPU form.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import ttnn

TILE = 32

CB_STAGE1 = 2  # level-1 landing (K*B fp32 pages) — tree only
CB_STAGE1_OUT = 3  # level-1 result (B fp32 pages) — tree only
CB_GATHERED = 4  # root's landing: s*B (flat) or L*B (tree) fp32 pages
CB_RMS_BCAST = 5  # root's finalized 1/rms, source of the broadcast
CB_SCALER = 7  # reduce scaler (1.0), bf16 — as in the op

SEM_MCAST_READY = 0
SEM_MCAST_CONSUMED = 1
SEM_GATHER = 2  # top-level arrivals at the root
SEM_STAGE1 = 3  # level-1 arrivals at a subgroup leader (tree only)

VARIANTS = ("flat_root", "tree")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """One row-group placement: `num_groups` rects of rect_w x rect_h cores."""

    rect_w: int
    rect_h: int
    num_groups: int
    block_rows: int  # B — stat tiles per core per combine
    fanin: int  # K — level-1 subgroup size (tree only); rect_w by default
    hidden_tiles: int = 5  # S — only sets the magnitude of the stat values / W

    @property
    def slices(self) -> int:
        return self.rect_w * self.rect_h

    @property
    def leaders(self) -> int:
        return self.slices // self.fanin

    @property
    def w(self) -> int:
        """The op's W: 32 * S * s elements across the row-group's hidden axis."""
        return TILE * self.hidden_tiles * self.slices

    @property
    def label(self) -> str:
        return (
            f"{self.rect_w}x{self.rect_h}(s={self.slices}) g={self.num_groups} " f"B={self.block_rows} K={self.fanin}"
        )


@dataclass(frozen=True)
class Layout:
    geometry: Geometry
    groups: tuple[tuple[tuple[int, int], ...], ...]  # per group, cores in slice order
    core_ranges: "ttnn.CoreRangeSet"

    @property
    def active_cores(self):
        return tuple(core for group in self.groups for core in group)


def build_layout(device, geometry: Geometry) -> Layout:
    grid = device.compute_with_storage_grid_size()
    w, h = geometry.rect_w, geometry.rect_h
    if geometry.slices % geometry.fanin:
        raise ValueError(f"fanin {geometry.fanin} must divide s={geometry.slices}")
    across = grid.x // w
    down = grid.y // h
    if geometry.num_groups > across * down:
        raise ValueError(f"cannot place {geometry.num_groups} rects of {w}x{h} on a {grid.x}x{grid.y} grid")
    groups = []
    ranges = []
    for g in range(geometry.num_groups):
        ox = (g % across) * w
        oy = (g // across) * h
        # Slice order == the op's shard order: row-major inside the rect.
        groups.append(tuple((ox + dx, oy + dy) for dy in range(h) for dx in range(w)))
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(ox, oy), ttnn.CoreCoord(ox + w - 1, oy + h - 1)))
    return Layout(geometry, tuple(groups), ttnn.CoreRangeSet(ranges))


def create_stat_memory_config(device, geometry: Geometry):
    """One shard of B float32 stat tiles per core, in slice order."""
    layout = build_layout(device, geometry)
    return ttnn.create_sharded_memory_config(
        shape=(TILE, geometry.block_rows * TILE),
        core_grid=layout.core_ranges,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

_DATAFLOW_PRELUDE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_stage1 = 2;
constexpr uint32_t cb_stage1_out = 3;
constexpr uint32_t cb_gathered = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_scaler = 7;
"""

_COMPUTE_PRELUDE = r"""
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_stage1 = 2;
constexpr uint32_t cb_stage1_out = 3;
constexpr uint32_t cb_gathered = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_scaler = 7;

// The op's own crossover: pairwise add_tiles + one within-tile collapse beats the
// matmul-with-ones reduce_tile datapath once there are >= 4 tiles to amortize it.
constexpr uint32_t COMBINE_ACCUMULATE_MIN_TILES = 4;
"""

# --- flat_root (the op's current approach) ----------------------------------

_FLAT_READER = (
    _DATAFLOW_PRELUDE
    + r"""
void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t CT = mc.next_compile_time_args_offset();
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t NUM_SLICES = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(CT + 3);

    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t out_addr = get_arg_val<uint32_t>(RT + 0);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 1);

    Noc noc;
    // Pipes FIRST: a ReceiverPipe ctor inits its own data_ready flag to INVALID, so it
    // must not run after a broadcast has already landed (mcast_pipe.hpp:27-29).
    auto sender_pipe = mc.sender(noc);
    auto receiver_pipe = mc.receiver(noc);
    Semaphore<> gather_progress(GATHER_SEM_ID);

    calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    if (is_root) {
        cb_reserve_back(cb_gathered, NUM_SLICES * BLOCK_ROWS);
        gather_progress.wait_min(NUM_SLICES);
        cb_push_back(cb_gathered, NUM_SLICES * BLOCK_ROWS);

        cb_wait_front(cb_rms_bcast, BLOCK_ROWS);
        sender_pipe.send(get_read_ptr(cb_rms_bcast), out_addr, BLOCK_ROWS * STAT_TILE_BYTES);
        cb_pop_front(cb_rms_bcast, BLOCK_ROWS);
    } else {
        receiver_pipe.receive();
    }
}
"""
)

_FLAT_WRITER = (
    _DATAFLOW_PRELUDE
    + r"""
void kernel_main() {
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_SLICES = get_compile_time_arg_val(1);
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(2);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(3);

    const uint32_t stat_addr = get_arg_val<uint32_t>(0);
    const uint32_t root_x = get_arg_val<uint32_t>(1);
    const uint32_t root_y = get_arg_val<uint32_t>(2);
    const uint32_t slice_index = get_arg_val<uint32_t>(3);

    Noc noc;
    Semaphore<> gather_progress(GATHER_SEM_ID);
    // Every CB in this program is declared on one common core set, so the root's
    // landing address is derivable from THIS core's own write pointer (the op's
    // rms_norm_writer.cpp does exactly this).
    const uint32_t gather_base = get_write_ptr(cb_gathered);

    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        const uint32_t page = r * NUM_SLICES + slice_index;
        noc_async_write(
            stat_addr + r * STAT_TILE_BYTES,
            get_noc_addr(root_x, root_y, gather_base + page * STAT_TILE_BYTES),
            STAT_TILE_BYTES);
    }
    noc_async_write_barrier();
    gather_progress.up(noc, root_x, root_y, 1);
}
"""
)

_FLAT_COMPUTE = (
    _COMPUTE_PRELUDE
    + r"""
void kernel_main() {
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_SLICES = get_compile_time_arg_val(1);
    constexpr auto ALGORITHM = NUM_SLICES >= COMBINE_ACCUMULATE_MIN_TILES
                                   ? ckl::ReduceAlgorithm::AccumulateViaAdd
                                   : ckl::ReduceAlgorithm::Auto;

    const uint32_t inv_w_bits = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_gathered, cb_scaler, cb_rms_bcast);

    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    cb_wait_front(cb_gathered, NUM_SLICES * BLOCK_ROWS);
    ckl::reduce<
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        cb_gathered,
        cb_scaler,
        cb_rms_bcast,
        ckl::ReduceInputPolicy::BulkWaitBulkPop,
        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        ALGORITHM,
        ckl::NoAccumulation,
        decltype(finalize)>(
        ckl::ReduceInputBlockShape::of(BLOCK_ROWS, NUM_SLICES),
        ckl::ReduceInputMemoryLayout::contiguous(),
        ckl::NoAccumulation{},
        finalize);
}
"""
)

# --- tree (two-level) -------------------------------------------------------

_TREE_READER = (
    _DATAFLOW_PRELUDE
    + r"""
void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t CT = mc.next_compile_time_args_offset();
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t FANIN = get_compile_time_arg_val(CT + 1);        // K
    constexpr uint32_t NUM_LEADERS = get_compile_time_arg_val(CT + 2);  // L
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(CT + 4);
    constexpr uint32_t STAGE1_SEM_ID = get_compile_time_arg_val(CT + 5);

    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t out_addr = get_arg_val<uint32_t>(RT + 0);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 1);
    const uint32_t is_leader = get_arg_val<uint32_t>(RT + 2);

    Noc noc;
    auto sender_pipe = mc.sender(noc);
    auto receiver_pipe = mc.receiver(noc);
    Semaphore<> gather_progress(GATHER_SEM_ID);
    Semaphore<> stage1_progress(STAGE1_SEM_ID);

    calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    // Level 1: a leader publishes its subgroup's K*B landed tiles to compute.
    if (is_leader) {
        cb_reserve_back(cb_stage1, FANIN * BLOCK_ROWS);
        stage1_progress.wait_min(FANIN);
        cb_push_back(cb_stage1, FANIN * BLOCK_ROWS);
    }

    if (is_root) {
        // Level 2: the L leaders' partials.  Fan-in L, not s.
        cb_reserve_back(cb_gathered, NUM_LEADERS * BLOCK_ROWS);
        gather_progress.wait_min(NUM_LEADERS);
        cb_push_back(cb_gathered, NUM_LEADERS * BLOCK_ROWS);

        cb_wait_front(cb_rms_bcast, BLOCK_ROWS);
        sender_pipe.send(get_read_ptr(cb_rms_bcast), out_addr, BLOCK_ROWS * STAT_TILE_BYTES);
        cb_pop_front(cb_rms_bcast, BLOCK_ROWS);
    } else {
        receiver_pipe.receive();
    }
}
"""
)

_TREE_WRITER = (
    _DATAFLOW_PRELUDE
    + r"""
void kernel_main() {
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t FANIN = get_compile_time_arg_val(1);        // K
    constexpr uint32_t NUM_LEADERS = get_compile_time_arg_val(2);  // L
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(4);
    constexpr uint32_t STAGE1_SEM_ID = get_compile_time_arg_val(5);

    const uint32_t stat_addr = get_arg_val<uint32_t>(0);
    const uint32_t leader_x = get_arg_val<uint32_t>(1);
    const uint32_t leader_y = get_arg_val<uint32_t>(2);
    const uint32_t root_x = get_arg_val<uint32_t>(3);
    const uint32_t root_y = get_arg_val<uint32_t>(4);
    const uint32_t pos_in_subgroup = get_arg_val<uint32_t>(5);
    const uint32_t subgroup_index = get_arg_val<uint32_t>(6);
    const uint32_t is_leader = get_arg_val<uint32_t>(7);

    Noc noc;
    Semaphore<> gather_progress(GATHER_SEM_ID);
    Semaphore<> stage1_progress(STAGE1_SEM_ID);
    const uint32_t stage1_base = get_write_ptr(cb_stage1);
    const uint32_t gather_base = get_write_ptr(cb_gathered);

    // Level 1: contribute to this core's subgroup leader (the leader included, by
    // NoC loopback — the same "root is not special-cased" uniformity the op keeps).
    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        const uint32_t page = r * FANIN + pos_in_subgroup;
        noc_async_write(
            stat_addr + r * STAT_TILE_BYTES,
            get_noc_addr(leader_x, leader_y, stage1_base + page * STAT_TILE_BYTES),
            STAT_TILE_BYTES);
    }
    noc_async_write_barrier();
    stage1_progress.up(noc, leader_x, leader_y, 1);

    // Level 2: a leader forwards its subgroup partial to the root.
    if (is_leader) {
        cb_wait_front(cb_stage1_out, BLOCK_ROWS);
        const uint32_t src = get_read_ptr(cb_stage1_out);
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            const uint32_t page = r * NUM_LEADERS + subgroup_index;
            noc_async_write(
                src + r * STAT_TILE_BYTES,
                get_noc_addr(root_x, root_y, gather_base + page * STAT_TILE_BYTES),
                STAT_TILE_BYTES);
        }
        noc_async_write_barrier();
        gather_progress.up(noc, root_x, root_y, 1);
        cb_pop_front(cb_stage1_out, BLOCK_ROWS);
    }
}
"""
)

_TREE_COMPUTE = (
    _COMPUTE_PRELUDE
    + r"""
void kernel_main() {
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t FANIN = get_compile_time_arg_val(1);        // K
    constexpr uint32_t NUM_LEADERS = get_compile_time_arg_val(2);  // L
    constexpr auto ALGORITHM = NUM_LEADERS >= COMBINE_ACCUMULATE_MIN_TILES
                                   ? ckl::ReduceAlgorithm::AccumulateViaAdd
                                   : ckl::ReduceAlgorithm::Auto;

    const uint32_t inv_w_bits = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);

    // Both level-1 add operands come from cb_stage1; the level-2 reduce re-inits
    // its own formats (reduce_helpers_compute.inl:287 reconfigs both operands to
    // the input CB, and reduce_init_short_with_dt brings in the scaler).
    compute_kernel_hw_startup(cb_stage1, cb_stage1, cb_stage1_out);

    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    // ---- level 1: elementwise sum of the subgroup's K stat tiles, per row ----
    // RAW LLK, deliberately: this is the sum WITHOUT the within-tile collapse
    // (ReduceWithinTile::Skip), which ckl::reduce cannot express (see the module
    // docstring).  Pairwise add_tiles with acc_to_dest keeps the running sum in
    // DEST; DEST is seeded explicitly (add with acc_to_dest=false / copy_tile) so
    // nothing depends on DEST being zero at acquire.
    cb_wait_front(cb_stage1, FANIN * BLOCK_ROWS);
    cb_reserve_back(cb_stage1_out, BLOCK_ROWS);
    tile_regs_acquire();
    uint32_t first = 0;
    if constexpr (FANIN & 1u) {
        copy_tile_to_dst_init_short(cb_stage1);
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            copy_tile(cb_stage1, r * FANIN, r);
        }
        first = 1;
    } else {
        add_tiles_init(cb_stage1, cb_stage1, false);
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            add_tiles(cb_stage1, cb_stage1, r * FANIN, r * FANIN + 1, r);
        }
        first = 2;
    }
    if constexpr (FANIN > 2) {
        add_tiles_init(cb_stage1, cb_stage1, true);
        for (uint32_t k = first; k + 1 < FANIN; k += 2) {
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                add_tiles(cb_stage1, cb_stage1, r * FANIN + k, r * FANIN + k + 1, r);
            }
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        pack_tile(r, cb_stage1_out);
    }
    tile_regs_release();
    cb_push_back(cb_stage1_out, BLOCK_ROWS);
    cb_pop_front(cb_stage1, FANIN * BLOCK_ROWS);

    // ---- level 2 (root only): the op's own combine reduce, over L not s ----
    if (is_root) {
        cb_wait_front(cb_gathered, NUM_LEADERS * BLOCK_ROWS);
        ckl::reduce<
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            cb_gathered,
            cb_scaler,
            cb_rms_bcast,
            ckl::ReduceInputPolicy::BulkWaitBulkPop,
            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Fast,
            ALGORITHM,
            ckl::NoAccumulation,
            decltype(finalize)>(
            ckl::ReduceInputBlockShape::of(BLOCK_ROWS, NUM_LEADERS),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            finalize);
    }
}
"""
)


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


def _inline(source, core_ranges, ct_args, rt_args, config):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        config=config,
    )


def _cb(index, pages, page_bytes, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def create_program_descriptor(
    stat_tensor,
    out_tensor,
    *,
    variant: str,
    geometry: Geometry,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    device = stat_tensor.device()
    layout = build_layout(device, geometry)
    B = geometry.block_rows
    s = geometry.slices
    K = geometry.fanin
    L = geometry.leaders
    stat_tile = ttnn.tile_size(ttnn.float32)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    inv_w_bits = _f32_bits(1.0 / float(geometry.w))
    eps_bits = _f32_bits(epsilon)
    stat_addr = stat_tensor.buffer_address()
    out_addr = out_tensor.buffer_address()
    if compute_kernel_config is None:
        compute_kernel_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False
        )

    core_ranges = layout.core_ranges

    # One Mcast2D per row-group rect, origin = the group's root (slice 0).
    cfg = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_0,
        handshake=False,  # single broadcast per kernel, as the op does for num_blocks == 1
        sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
    )
    mcasts = []
    for cores in layout.groups:
        rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*cores[0]), ttnn.CoreCoord(*cores[-1]))])
        mcasts.append(ttnn.Mcast2D(device, rect, ttnn.CoreCoord(*cores[0]), cfg, s - 1))
    mcast_ct = list(mcasts[0].compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    compute_cores = []

    for cores, mc in zip(layout.groups, mcasts):
        root = cores[0]
        root_virt = device.worker_core_from_logical_core(ttnn.CoreCoord(*root))
        for slice_index, (cx, cy) in enumerate(cores):
            is_root = 1 if slice_index == 0 else 0
            mcast_rt = list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))
            if variant == "flat_root":
                reader_rt[cx][cy] = mcast_rt + [out_addr, is_root]
                writer_rt[cx][cy] = [stat_addr, root_virt.x, root_virt.y, slice_index]
                if is_root:
                    compute_cores.append((cx, cy))
                    compute_rt[cx][cy] = [inv_w_bits, eps_bits]
            else:
                subgroup = slice_index // K
                pos = slice_index % K
                is_leader = 1 if pos == 0 else 0
                leader = cores[subgroup * K]
                leader_virt = device.worker_core_from_logical_core(ttnn.CoreCoord(*leader))
                reader_rt[cx][cy] = mcast_rt + [out_addr, is_root, is_leader]
                writer_rt[cx][cy] = [
                    stat_addr,
                    leader_virt.x,
                    leader_virt.y,
                    root_virt.x,
                    root_virt.y,
                    pos,
                    subgroup,
                    is_leader,
                ]
                if is_leader:
                    compute_cores.append((cx, cy))
                    compute_rt[cx][cy] = [inv_w_bits, eps_bits, is_root]

    compute_ranges = _core_range_set(compute_cores)

    # Every CB on the SAME core set, so the L1 map is identical across the rect and a
    # contributor can derive its target's landing address from its own write pointer.
    cbs = [
        _cb(CB_RMS_BCAST, B, stat_tile, ttnn.float32, core_ranges),
        _cb(CB_SCALER, 1, bf16_tile, ttnn.bfloat16, core_ranges),
    ]
    if variant == "flat_root":
        cbs.append(_cb(CB_GATHERED, s * B, stat_tile, ttnn.float32, core_ranges))
    else:
        cbs.append(_cb(CB_STAGE1, K * B, stat_tile, ttnn.float32, core_ranges))
        cbs.append(_cb(CB_STAGE1_OUT, B, stat_tile, ttnn.float32, core_ranges))
        cbs.append(_cb(CB_GATHERED, L * B, stat_tile, ttnn.float32, core_ranges))

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_STAGE1, core_ranges=core_ranges, initial_value=0),
    ]

    if variant == "flat_root":
        scalar_ct = [B, s, stat_tile, SEM_GATHER]
        reader = _inline(_FLAT_READER, core_ranges, mcast_ct + scalar_ct, reader_rt, ttnn.ReaderConfigDescriptor())
        writer = _inline(_FLAT_WRITER, core_ranges, scalar_ct, writer_rt, ttnn.WriterConfigDescriptor())
        compute = _inline(_FLAT_COMPUTE, compute_ranges, [B, s], compute_rt, compute_kernel_config)
    else:
        scalar_ct = [B, K, L, stat_tile, SEM_GATHER, SEM_STAGE1]
        reader = _inline(_TREE_READER, core_ranges, mcast_ct + scalar_ct, reader_rt, ttnn.ReaderConfigDescriptor())
        writer = _inline(_TREE_WRITER, core_ranges, scalar_ct, writer_rt, ttnn.WriterConfigDescriptor())
        compute = _inline(_TREE_COMPUTE, compute_ranges, [B, K, L], compute_rt, compute_kernel_config)

    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)


def combine(stat_tensor, *, variant, geometry: Geometry, epsilon: float = 1e-6):
    """Run one cross-core combine round; returns the per-core finalized 1/rms tiles."""
    out_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(stat_tensor.shape)),
        stat_tensor.dtype,
        stat_tensor.layout,
        stat_tensor.device(),
        stat_tensor.memory_config(),
    )
    descriptor = create_program_descriptor(stat_tensor, out_tensor, variant=variant, geometry=geometry, epsilon=epsilon)
    return ttnn.generic_op([stat_tensor, out_tensor], descriptor)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def reference_rms_recip(torch_stats, layout: Layout, *, epsilon: float = 1e-6):
    """1/rms per (core, block row, row-in-tile), in the reduce's column-0 form.

    torch_stats is the [ncores*32, B*32] float32 stat matrix; shard i belongs to
    `layout.active_cores[i]`.  Returns a [ncores*32, B] tensor of the expected
    column-0 values.
    """
    import torch

    geometry = layout.geometry
    B = geometry.block_rows
    ncores = len(layout.active_cores)
    index = {core: i for i, core in enumerate(layout.active_cores)}
    expected = torch.zeros((ncores * TILE, B), dtype=torch.float32)
    for cores in layout.groups:
        idx = [index[c] for c in cores]
        for r in range(B):
            total = None
            for i in idx:
                tile = torch_stats[i * TILE : (i + 1) * TILE, r * TILE : (r + 1) * TILE]
                rowsum = tile.sum(dim=1)
                total = rowsum if total is None else total + rowsum
            value = torch.rsqrt(total / float(geometry.w) + epsilon)
            for i in idx:
                expected[i * TILE : (i + 1) * TILE, r] = value
    return expected
