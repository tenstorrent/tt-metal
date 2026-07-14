// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// On-device tensor / dataflow layer for the static-state-tracking experiment.
// Compile-time shapes only (see tile_shape.h).
//
// Supports BOTH dataflow substrates the metal stack now ships — the compute
// kernels in tree are mixed, some CB-backed and some DFB-backed:
//   Backend::Cb  — raw circular-buffer sync (llk_wait_tiles / llk_pop_tiles /
//                  llk_wait_for_free_tiles / llk_push_tiles), addressed by the
//                  c_* CB index, L1 pointer read from get_local_cb_interface.
//   Backend::Dfb — the metal2 DataflowBuffer API (reserve_back / push_back /
//                  wait_front / pop_front + get_read_ptr / get_write_ptr),
//                  addressed by the DFB LOGICAL id (the value of a dfb::<name>
//                  accessor from the generated bindings), NOT the c_* index.
//
// The backend is a compile-time TYPE parameter, so Tensor<Shape, Cb> and
// Tensor<Shape, Dfb> are distinct types: dispatch is zero-cost, and a CB index
// can never reach a DFB release path (or vice-versa). The verbs are uniform
// across both backends — wait_front / reserve_back / pop_front / push_back — so
// only the `Backend` in the type changes between a CB kernel and a DFB kernel.
// The ops (copy_tile, untilize_block, …) are backend-agnostic.
//
// `Tensor<Shape, B>` stores the tile count (r_tiles/c_tiles), so pop_front /
// push_back derive num_tiles() from the handle.

#ifndef SST_TENSOR_TENSOR_H
#define SST_TENSOR_TENSOR_H

#include <cstdint>

#include "experiments/static-state-tracking/compute/defs.h"
#include "resolver.h"
#include "tile_shape.h"

#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
#include "internal/circular_buffer_interface.h"  // get_local_cb_interface (Cb address)
#include "api/dataflow/dataflow_buffer.h"        // DataflowBuffer (Dfb backend)
#endif
#ifdef TRISC_UNPACK
#include "llk_io_unpack.h"  // llk_wait_tiles, llk_pop_tiles (Cb backend)
#endif
#ifdef TRISC_PACK
#include "llk_io_pack.h"  // llk_wait_for_free_tiles, llk_push_tiles (Cb backend)
#endif

namespace sst::tensor {

// Which dataflow substrate a Tensor window is bound to.
enum class Backend : uint8_t { Cb, Dfb };
// Call-site aliases: Tensor<Shape, Cb> / Tensor<Shape, Dfb>.
inline constexpr Backend Cb = Backend::Cb;
inline constexpr Backend Dfb = Backend::Dfb;

// Cb-backend L1 pointer snapshot (16-B words): UNPACK reads the read pointer,
// PACK the write pointer, MATH has no L1 address. A given window's address is
// only dereferenced on its owning engine (input on UNPACK, output on PACK), so
// one helper serves both wait_front and reserve_back.
ALWI uint32_t cb_snapshot_addr_16B(uint32_t cb) {
#if defined(TRISC_UNPACK)
    return get_local_cb_interface(cb).fifo_rd_ptr;
#elif defined(TRISC_PACK)
    return get_local_cb_interface(cb).fifo_wr_ptr;
#else
    (void)cb;
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Tensor<Shape, B> — owning dataflow window handle. 12 bytes: the base L1
// pointer (16-B words) for the current window on the owning engine, the window's
// tile extents, and the buffer id (Cb: c_* index; Dfb: dfb::<name> logical id).
// `tile_addr_16B(i)` strides by whole tiles of `Shape`.
// ---------------------------------------------------------------------------
template <typename Shape, Backend B>
struct Tensor {
    using shape_t = Shape;
    static constexpr Backend backend = B;

    uint32_t l1_addr_16B = 0;
    uint16_t r_tiles = 0;
    uint16_t c_tiles = 0;
    uint8_t id = 0;  // Cb: c_* index; Dfb: logical id (value of a dfb::<name>)

    constexpr uint16_t num_tiles() const { return uint16_t(uint32_t(r_tiles) * c_tiles); }
    uint32_t tile_addr_16B(uint32_t tile_index) const { return l1_addr_16B + tile_index * Shape::tile_words(); }

    // Blocking acquire of `r_tiles * c_tiles` at the front (input); snapshot the
    // read pointer for the owning engine.
    static Tensor wait_front(uint32_t id, uint16_t r_tiles, uint16_t c_tiles = 1) {
        const uint32_t n = uint32_t(r_tiles) * c_tiles;
        uint32_t addr = 0;
#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
        if constexpr (B == Backend::Dfb) {
            DataflowBuffer d(static_cast<uint16_t>(id));
            d.wait_front(static_cast<uint16_t>(n));
            addr = d.get_read_ptr();
        } else {
            UNPACK((llk_wait_tiles(id, n)));
            addr = cb_snapshot_addr_16B(id);
        }
#else
        (void)n;
#endif
        return Tensor{addr, r_tiles, c_tiles, static_cast<uint8_t>(id)};
    }
    // Blocking reserve of `r_tiles * c_tiles` free at the back (output); snapshot
    // the write pointer for the owning engine.
    static Tensor reserve_back(uint32_t id, uint16_t r_tiles, uint16_t c_tiles = 1) {
        const uint32_t n = uint32_t(r_tiles) * c_tiles;
        uint32_t addr = 0;
#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
        if constexpr (B == Backend::Dfb) {
            DataflowBuffer d(static_cast<uint16_t>(id));
            d.reserve_back(static_cast<uint16_t>(n));
            addr = d.get_write_ptr();
        } else {
            PACK((llk_wait_for_free_tiles<false, false, false>(id, n)));
            addr = cb_snapshot_addr_16B(id);
        }
#else
        (void)n;
#endif
        return Tensor{addr, r_tiles, c_tiles, static_cast<uint8_t>(id)};
    }
};

// ---------------------------------------------------------------------------
// Release — free pop_front / push_back over the handle. Backend is deduced from
// the Tensor, so the call sites are uniform and can never mismatch the acquire
// (the handle carries its backend in its type). Reads the tile count from the
// handle.
// ---------------------------------------------------------------------------
template <typename Shape, Backend B>
ALWI void pop_front(const Tensor<Shape, B>& t) {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
    if constexpr (B == Backend::Dfb) {
        DataflowBuffer(static_cast<uint16_t>(t.id)).pop_front(static_cast<uint16_t>(t.num_tiles()));
    } else {
        UNPACK((llk_pop_tiles(uint32_t(t.id), uint32_t(t.num_tiles()))));
    }
#else
    (void)t;
#endif
}
template <typename Shape, Backend B>
ALWI void push_back(const Tensor<Shape, B>& t) {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
    if constexpr (B == Backend::Dfb) {
        DataflowBuffer(static_cast<uint16_t>(t.id)).push_back(static_cast<uint16_t>(t.num_tiles()));
    } else {
        PACK((llk_push_tiles<false, false>(uint32_t(t.id), uint32_t(t.num_tiles()))));
    }
#else
    (void)t;
#endif
}

}  // namespace sst::tensor

#endif  // SST_TENSOR_TENSOR_H
