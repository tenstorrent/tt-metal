// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// TRISC includes compute kernel API which provides CB and tile operations
// BRISC/NCRISC include dataflow API which provides NOC operations
#ifdef COMPILE_FOR_TRISC
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"
using namespace ckernel;
#endif

// Data movement processors need dataflow API (already included by default)
// Core coordinate accessors (get_absolute_logical_x/y) are already provided by:
// - compute_kernel_api/common.h for TRISC
// - dataflow_api.h for BRISC/NCRISC

// ============================================================================
// Multicore tile distribution helpers
// ============================================================================

struct TileRange {
    uint32_t start;
    uint32_t end;
    uint32_t count() const { return end - start; }
};

// Get this core's tile range for block distribution across a grid
// Usage: auto range = tile_range(total_tiles, grid_x, grid_y);
//        for (uint32_t i = range.start; i < range.end; i++) { ... }
inline TileRange tile_range(uint32_t total_tiles, uint32_t grid_x, uint32_t grid_y) {
    uint32_t core_x = get_absolute_logical_x();
    uint32_t core_y = get_absolute_logical_y();
    uint32_t core_idx = core_y * grid_x + core_x;
    uint32_t num_cores = grid_x * grid_y;

    uint32_t tiles_per_core = total_tiles / num_cores;
    uint32_t remainder = total_tiles % num_cores;

    TileRange range;
    if (core_idx < remainder) {
        range.start = core_idx * (tiles_per_core + 1);
        range.end = range.start + tiles_per_core + 1;
    } else {
        range.start = remainder * (tiles_per_core + 1) + (core_idx - remainder) * tiles_per_core;
        range.end = range.start + tiles_per_core;
    }
    return range;
}

// Simpler: get tile range for single-core (all tiles)
inline TileRange tile_range(uint32_t total_tiles) { return {0, total_tiles}; }

// Use with auto-generated grid_x, grid_y from INIT_ARGUMENTS
// Usage: auto range = tile_range(a_n_tiles, grid_x, grid_y);
//   or:  for (auto i : tile_range(a_n_tiles, grid_x, grid_y)) { ... }
#define MY_TILE_RANGE(total) tile_range(total, grid_x, grid_y)

#ifdef COMPILE_FOR_TRISC
#define KERNEL_MAIN       \
    namespace NAMESPACE { \
    void MAIN;            \
    }                     \
    void NAMESPACE::MAIN
#else
#define KERNEL_MAIN void kernel_main()
#endif

namespace unified_kernel::detail {

#ifdef COMPILE_FOR_TRISC
uint32_t read_pages[TOTAL_NUM_CIRCULAR_BUFFERS];
uint32_t popped_pages[TOTAL_NUM_CIRCULAR_BUFFERS];
bool dst_ready = true;  // tracks if DST register is ready (no pending compute)
#endif
#ifdef COMPILE_FOR_BRISC
uint32_t offset_pages[TOTAL_NUM_CIRCULAR_BUFFERS];
#endif

FORCE_INLINE void release_read_tile(uint32_t cb_id) {
#ifdef COMPILE_FOR_TRISC
    cb_pop_front(cb_id, 1);
    popped_pages[cb_id] += 1;
#endif
#ifdef COMPILE_FOR_BRISC
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
    offset_pages[cb_id] -= 1;
#endif
}

struct ReadTile {
    constexpr ReadTile(uint32_t cb_id, uint32_t id) : cb_id_(cb_id), id_(id) {}
    ReadTile(const ReadTile&) = delete;
    ReadTile& operator=(const ReadTile&) = delete;
    ~ReadTile() { release_read_tile(cb_id_); }
    uint32_t local_id() const {
#ifdef COMPILE_FOR_TRISC
        return id_ - popped_pages[cb_id_];
#else
        return id_;
#endif
    }
    uint32_t cb_id() const { return cb_id_; }

private:
    uint32_t cb_id_ = 0;
    uint32_t id_ = 0;
};
struct ConstantTile {
    constexpr ConstantTile(uint32_t cb_id, uint32_t id) : cb_id_(cb_id), id_(id) {}
    uint32_t local_id() const { return id_; }
    uint32_t cb_id() const { return cb_id_; }

private:
    uint32_t cb_id_ = 0;
    uint32_t id_ = 0;
};

// DstTile represents a tile in the destination register
// This enables FMA patterns like: auto result = mul(a, b); add(result, c);
struct DstTile {
    constexpr explicit DstTile(uint32_t idx) : idx_(idx) {}
    uint32_t idx() const { return idx_; }

private:
    uint32_t idx_ = 0;
};

// Helper to extract dst index from either raw int or DstTile
FORCE_INLINE uint32_t get_dst_idx(uint32_t idx) { return idx; }
FORCE_INLINE uint32_t get_dst_idx(const DstTile& tile) { return tile.idx(); }

template <typename Accessor>
FORCE_INLINE auto read_tile_impl(
    const uint32_t cb_id, const uint32_t id, const Accessor& addrgen, const uint32_t tile_size_bytes) {
#ifdef COMPILE_FOR_TRISC
    cb_wait_front(cb_id, 1);
    read_pages[cb_id] += 1;
    return ReadTile(cb_id, read_pages[cb_id] - 1);
#elif COMPILE_FOR_BRISC
    cb_reserve_back(cb_id, 1);
    noc_async_read_tile(id, addrgen, get_write_ptr(cb_id) + tile_size_bytes * offset_pages[cb_id]);
    offset_pages[cb_id] += 1;
    return ReadTile(cb_id, offset_pages[cb_id] - 1);
#else
    return ConstantTile(0, 0);
#endif
}

FORCE_INLINE void ensure_dst_ready() {
#ifdef COMPILE_FOR_TRISC
    if (!dst_ready) {
        tile_regs_commit();
        tile_regs_wait();
        dst_ready = true;
    }
#endif
}

template <typename Accessor>
FORCE_INLINE void write_tile_impl(
    const uint32_t from_dst_idx,
    const uint32_t cb_id,
    const uint32_t into_page_id,
    const Accessor& addrgen,
    const uint32_t tile_size_bytes) {
#ifdef COMPILE_FOR_TRISC
    ensure_dst_ready();
    cb_reserve_back(cb_id, 1);
    pack_tile(from_dst_idx, cb_id);
    cb_push_back(cb_id, 1);
#elif COMPILE_FOR_NCRISC
    cb_wait_front(cb_id, 1);
    noc_async_write_tile(into_page_id, addrgen, get_read_ptr(cb_id));
    noc_async_write_barrier();
    cb_pop_front(cb_id, 1);
#endif
}

}  // namespace unified_kernel::detail

#define read_tile(tensor, id) unified_kernel::detail::read_tile_impl(tensor##_cb, id, tensor, tensor##_page_size_bytes)
// write_tile: accepts either raw dst_idx (int) or DstTile
// Uses _Generic in C or template helper in C++
#define write_tile(dst_or_tile, tensor, into_page_id) \
    unified_kernel::detail::write_tile_impl(          \
        unified_kernel::detail::get_dst_idx(dst_or_tile), tensor##_cb, into_page_id, tensor, tensor##_page_size_bytes)
using ReadTile = unified_kernel::detail::ReadTile;
using ConstantTile = unified_kernel::detail::ConstantTile;

// ============================================================================
// Unified compute primitives - no-ops on data movement processors
// ============================================================================

// Binary operation initialization - compute only
// These wrap the raw LLK functions to be no-ops on data movement processors
#ifdef COMPILE_FOR_TRISC
// Re-export eltwise binary types for user convenience
using ckernel::EltwiseBinaryReuseDestType;
using ckernel::EltwiseBinaryType;

// binary_op_init_common is already available via compute_kernel_api.h on TRISC
// mul_tiles_init, add_tiles_init are already available via eltwise_binary.h on TRISC

#define INIT_BINARY_ADD(cb_a, cb_b, cb_out)    \
    binary_op_init_common(cb_a, cb_b, cb_out); \
    add_tiles_init(cb_a, cb_b)

#define INIT_BINARY_MUL(cb_a, cb_b, cb_out)    \
    binary_op_init_common(cb_a, cb_b, cb_out); \
    mul_tiles_init(cb_a, cb_b)

// FMA init: A*B + C (initializes mul, add will use binary_dest_reuse)
#define INIT_FMA(cb_a, cb_b, cb_c, cb_out)     \
    binary_op_init_common(cb_a, cb_b, cb_out); \
    mul_tiles_init(cb_a, cb_b);                \
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_c)
#else
// Data movement processors - these are no-ops
// Define stub enums so template syntax compiles even on DM processors
enum class EltwiseBinaryType { ELWADD, ELWSUB, ELWMUL, ELWDIV };
enum class EltwiseBinaryReuseDestType { NONE, DEST_TO_SRCA, DEST_TO_SRCB };

#define binary_op_init_common(...) \
    do {                           \
    } while (0)
#define mul_tiles_init(...) \
    do {                    \
    } while (0)
#define add_tiles_init(...) \
    do {                    \
    } while (0)
template <EltwiseBinaryType A, EltwiseBinaryReuseDestType B>
FORCE_INLINE void binary_dest_reuse_tiles_init(uint32_t) {}

#define INIT_BINARY_ADD(cb_a, cb_b, cb_out) \
    do {                                    \
    } while (0)
#define INIT_BINARY_MUL(cb_a, cb_b, cb_out) \
    do {                                    \
    } while (0)
#define INIT_FMA(cb_a, cb_b, cb_c, cb_out) \
    do {                                   \
    } while (0)
#endif

// Tile register management - compute only
#ifdef COMPILE_FOR_TRISC
#define ACQUIRE_DST()                              \
    do {                                           \
        tile_regs_acquire();                       \
        unified_kernel::detail::dst_ready = false; \
    } while (0)
#define COMMIT_DST() tile_regs_commit()
#define WAIT_DST()                                \
    do {                                          \
        tile_regs_wait();                         \
        unified_kernel::detail::dst_ready = true; \
    } while (0)
#define RELEASE_DST() tile_regs_release()
#else
#define ACQUIRE_DST() \
    do {              \
    } while (0)
#define COMMIT_DST() \
    do {             \
    } while (0)
#define WAIT_DST() \
    do {           \
    } while (0)
#define RELEASE_DST() \
    do {              \
    } while (0)
#endif

// Combined acquire/commit/wait pattern
#ifdef COMPILE_FOR_TRISC
#define DST_ACQUIRE_COMMIT_WAIT()                 \
    do {                                          \
        tile_regs_commit();                       \
        tile_regs_wait();                         \
        unified_kernel::detail::dst_ready = true; \
    } while (0)
#else
#define DST_ACQUIRE_COMMIT_WAIT() \
    do {                          \
    } while (0)
#endif

// ============================================================================
// Tile operations with DstTile support for FMA patterns
// Operations auto-init when CBs change - user just calls binary_op_init_common once
// ============================================================================

using DstTile = unified_kernel::detail::DstTile;

#ifdef COMPILE_FOR_TRISC
namespace unified_kernel::detail {
// Track last init state for auto-reinit on CB change
enum class LastOp { NONE, MUL_CB, ADD_CB, MUL_DST, ADD_DST };
inline LastOp last_op = LastOp::NONE;
inline uint32_t last_cb0 = UINT32_MAX;
inline uint32_t last_cb1 = UINT32_MAX;
}  // namespace unified_kernel::detail
#endif

// add: CB + CB -> DST (auto-reinits if CBs changed)
template <typename TileA, typename TileB>
FORCE_INLINE DstTile add(const TileA& in0, const TileB& in1, uint32_t dst_idx = 0) {
#ifdef COMPILE_FOR_TRISC
    using namespace unified_kernel::detail;
    if (last_op != LastOp::ADD_CB || last_cb0 != in0.cb_id() || last_cb1 != in1.cb_id()) {
        add_tiles_init(in0.cb_id(), in1.cb_id());
        last_op = LastOp::ADD_CB;
        last_cb0 = in0.cb_id();
        last_cb1 = in1.cb_id();
    }
    add_tiles(in0.cb_id(), in1.cb_id(), in0.local_id(), in1.local_id(), dst_idx);
#endif
    return DstTile(dst_idx);
}

// mul: CB * CB -> DST (auto-reinits if CBs changed)
template <typename TileA, typename TileB>
FORCE_INLINE DstTile mul(const TileA& in0, const TileB& in1, uint32_t dst_idx = 0) {
#ifdef COMPILE_FOR_TRISC
    using namespace unified_kernel::detail;
    if (last_op != LastOp::MUL_CB || last_cb0 != in0.cb_id() || last_cb1 != in1.cb_id()) {
        mul_tiles_init(in0.cb_id(), in1.cb_id());
        last_op = LastOp::MUL_CB;
        last_cb0 = in0.cb_id();
        last_cb1 = in1.cb_id();
    }
    mul_tiles(in0.cb_id(), in1.cb_id(), in0.local_id(), in1.local_id(), dst_idx);
#endif
    return DstTile(dst_idx);
}

// add: DST + CB -> DST (for FMA, auto-reinits if CB changed)
template <typename TileCB>
FORCE_INLINE DstTile add(const DstTile& dst_tile, const TileCB& cb_tile, uint32_t out_dst_idx = 0) {
#ifdef COMPILE_FOR_TRISC
    using namespace unified_kernel::detail;
    if (last_op != LastOp::ADD_DST || last_cb0 != cb_tile.cb_id()) {
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_tile.cb_id());
        last_op = LastOp::ADD_DST;
        last_cb0 = cb_tile.cb_id();
    }
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        cb_tile.cb_id(), cb_tile.local_id(), out_dst_idx);
#endif
    return DstTile(out_dst_idx);
}

// mul: DST * CB -> DST (auto-reinits if CB changed)
template <typename TileCB>
FORCE_INLINE DstTile mul(const DstTile& dst_tile, const TileCB& cb_tile, uint32_t out_dst_idx = 0) {
#ifdef COMPILE_FOR_TRISC
    using namespace unified_kernel::detail;
    if (last_op != LastOp::MUL_DST || last_cb0 != cb_tile.cb_id()) {
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_tile.cb_id());
        last_op = LastOp::MUL_DST;
        last_cb0 = cb_tile.cb_id();
    }
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        cb_tile.cb_id(), cb_tile.local_id(), out_dst_idx);
#endif
    return DstTile(out_dst_idx);
}

// Legacy compatibility: add_tiles/mul_tiles that write to dst_idx (void return)
template <typename TileA, typename TileB>
FORCE_INLINE void add_tiles(const TileA& in0, const TileB& in1, uint32_t dst_idx) {
#ifdef COMPILE_FOR_TRISC
    add_tiles(in0.cb_id(), in1.cb_id(), in0.local_id(), in1.local_id(), dst_idx);
#endif
}

template <typename TileA, typename TileB>
FORCE_INLINE void mul_tiles(const TileA& in0, const TileB& in1, uint32_t dst_idx) {
#ifdef COMPILE_FOR_TRISC
    mul_tiles(in0.cb_id(), in1.cb_id(), in0.local_id(), in1.local_id(), dst_idx);
#endif
}

// Unified reduce_tile - takes Tile objects, no-op on data movement
template <typename TileA, typename TileB>
FORCE_INLINE void reduce_tile(const TileA& in0, const TileB& in1, uint32_t dst_idx) {
#ifdef COMPILE_FOR_TRISC
    reduce_tile(in0.cb_id(), in1.cb_id(), in0.local_id(), in1.local_id(), dst_idx);
#endif
}
