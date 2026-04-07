// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Standalone debug dump utilities for inspecting CB state and L1 memory.
//
// These functions can be called from any kernel at any point — no barrier or
// synchronization required. They use whichever print backend is active
// (DPRINT or DEVICE_PRINT). When no print backend is enabled, they are no-ops.
//
// Usage:
//   #include "api/debug/dump.h"
//
//   debug_dump_cb(0, 8);           // CB0: metadata + 8 hex words from read ptr
//   debug_dump_cb(16);             // CB16: metadata only
//   debug_dump_cb_typed(0, 0);     // CB0 tile 0: typed values via TileSlice
//   debug_dump_l1(0x100000, 16);   // 16 hex words starting at L1 address 0x100000

#pragma once

#if defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)

#include "api/debug/dprint.h"
#include "api/debug/device_print.h"
#include "internal/circular_buffer_interface.h"

// ---------------------------------------------------------------------------
// debug_dump_cb: Print CB metadata and optionally raw hex data.
// Available on BRISC, NCRISC, TRISC0, TRISC2 (not TRISC1/Math).
// ---------------------------------------------------------------------------
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_DM) || \
    (defined(COMPILE_FOR_TRISC) && (COMPILE_FOR_TRISC != 1))

inline void debug_dump_cb(uint32_t cb_id, uint32_t num_words = 0) {
    auto& iface = get_local_cb_interface(cb_id);

#if defined(USE_DEVICE_PRINT)
    DEVICE_PRINT(
        "CB{} sz={} rd={} wr={} ack={} rcv={}",
        cb_id,
        iface.fifo_size,
        iface.fifo_rd_ptr,
        iface.fifo_wr_ptr,
        (uint32_t)iface.tiles_acked,
        (uint32_t)iface.tiles_received);
#else
    DPRINT << "CB" << cb_id << " sz=" << iface.fifo_size << " rd=" << iface.fifo_rd_ptr << " wr=" << iface.fifo_wr_ptr
           << " ack=" << iface.tiles_acked << " rcv=" << iface.tiles_received << ENDL();
#endif

    if (num_words > 0) {
        volatile tt_l1_ptr uint32_t* data_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.fifo_rd_ptr << cb_addr_shift);
        for (uint32_t w = 0; w < num_words; w += 4) {
            uint32_t chunk = (num_words - w > 4) ? 4 : (num_words - w);
#if defined(USE_DEVICE_PRINT)
            for (uint32_t j = 0; j < chunk; j++) {
                DEVICE_PRINT("  [{0}] {1:#010x}", w + j, data_ptr[w + j]);
            }
#else
            DPRINT << "  [" << w << "] ";
            for (uint32_t j = 0; j < chunk; j++) {
                DPRINT << HEX() << data_ptr[w + j] << " ";
            }
            DPRINT << DEC() << ENDL();
#endif
        }
    }
}

#else  // TRISC1 (Math) — no CB access

inline void debug_dump_cb([[maybe_unused]] uint32_t cb_id, [[maybe_unused]] uint32_t num_words = 0) {
#if defined(USE_DEVICE_PRINT)
    DEVICE_PRINT("debug_dump_cb: not available on math thread");
#else
    DPRINT << "debug_dump_cb: not available on math thread" << ENDL();
#endif
}

#endif  // COMPILE_FOR guards

// ---------------------------------------------------------------------------
// debug_dump_cb_typed: Print tile data interpreted by the CB's data format.
// Uses TileSlice for format-aware output. DPRINT only (TileSlice is not
// supported by DEVICE_PRINT). Available on TRISC0 (Unpack) and TRISC2 (Pack)
// for input and output CBs respectively. Also available on BRISC/NCRISC with
// explicit cb_type parameter.
// ---------------------------------------------------------------------------
#if !defined(USE_DEVICE_PRINT)
#include "api/debug/dprint_tile.h"

#if defined(COMPILE_FOR_TRISC) && (COMPILE_FOR_TRISC == 0)
// Unpack thread: CBs are inputs, use read pointer
inline void debug_dump_cb_typed(uint32_t cb_id, uint32_t tile_idx = 0, bool untilize = true) {
    DPRINT << "CB" << cb_id << " tile " << tile_idx << " (typed):" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << TileSlice(
                      cb_id,
                      tile_idx,
                      SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                      true,
                      untilize)
               << ENDL();
    }
}
#elif defined(COMPILE_FOR_TRISC) && (COMPILE_FOR_TRISC == 2)
// Pack thread: CBs are outputs, use write pointer
inline void debug_dump_cb_typed(uint32_t cb_id, uint32_t tile_idx = 0, bool untilize = true) {
    DPRINT << "CB" << cb_id << " tile " << tile_idx << " (typed):" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << TileSlice(
                      cb_id,
                      tile_idx,
                      SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                      true,
                      untilize)
               << ENDL();
    }
}
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// Dataflow threads: need explicit cb_type (input or output)
inline void debug_dump_cb_typed(
    uint32_t cb_id, uint32_t tile_idx = 0, dprint_tslice_cb_t cb_type = TSLICE_INPUT_CB, bool untilize = true) {
    DPRINT << "CB" << cb_id << " tile " << tile_idx << " (typed):" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << TileSlice(
                      cb_id,
                      tile_idx,
                      SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1},
                      cb_type,
                      TSLICE_RD_PTR,
                      true,
                      untilize)
               << ENDL();
    }
}
#else
// Math thread or unsupported — no-op
inline void debug_dump_cb_typed(
    [[maybe_unused]] uint32_t cb_id, [[maybe_unused]] uint32_t tile_idx = 0, [[maybe_unused]] bool untilize = true) {
    DPRINT << "debug_dump_cb_typed: not available on math thread" << ENDL();
}
#endif

#endif  // !USE_DEVICE_PRINT

// ---------------------------------------------------------------------------
// debug_dump_l1: Hex-dump arbitrary L1 memory.
// Available from all RISCs.
// ---------------------------------------------------------------------------
inline void debug_dump_l1(uint32_t addr, uint32_t num_words) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);

#if defined(USE_DEVICE_PRINT)
    DEVICE_PRINT("L1[{:#x}] {} words:", addr, num_words);
#else
    DPRINT << "L1[" << HEX() << addr << DEC() << "] " << num_words << " words:" << ENDL();
#endif

    for (uint32_t w = 0; w < num_words; w += 4) {
        uint32_t chunk = (num_words - w > 4) ? 4 : (num_words - w);
#if defined(USE_DEVICE_PRINT)
        for (uint32_t j = 0; j < chunk; j++) {
            DEVICE_PRINT("  [{0}] {1:#010x}", w + j, ptr[w + j]);
        }
#else
        DPRINT << "  [" << HEX() << (addr + w * 4) << "] ";
        for (uint32_t j = 0; j < chunk; j++) {
            DPRINT << ptr[w + j] << " ";
        }
        DPRINT << DEC() << ENDL();
#endif
    }
}

#else  // !DEBUG_PRINT_ENABLED || FORCE_DPRINT_OFF — no-ops

inline void debug_dump_cb([[maybe_unused]] uint32_t cb_id, [[maybe_unused]] uint32_t num_words = 0) {}
inline void debug_dump_l1([[maybe_unused]] uint32_t addr, [[maybe_unused]] uint32_t num_words) {}

#endif  // DEBUG_PRINT_ENABLED
