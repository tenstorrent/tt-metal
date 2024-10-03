// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/dprint_common.h"
#include "llk_io.h"

struct SliceRange {
    // A slice object encoding semantics of np.slice(h0:h1:hs, w0:w1:ws)
    // This is only used with DPRINT for TileSlice object
    uint16_t h0, h1, hs, w0, w1, ws;
    // [0:32:16, 0:32:16]
    static inline SliceRange hw0_32_16() { return SliceRange{ .h0 = 0, .h1 = 32, .hs = 16, .w0 = 0, .w1 = 32, .ws = 16 }; }
    // [0:32:8, 0:32:8]
    static inline SliceRange hw0_32_8() { return SliceRange{ .h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8 }; }
    // [0:32:4, 0:32:4]
    static inline SliceRange hw0_32_4() { return SliceRange{ .h0 = 0, .h1 = 32, .hs = 4, .w0 = 0, .w1 = 32, .ws = 4 }; }
    // [0, 0:32]
    static inline SliceRange h0_w0_32() { return SliceRange{ .h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    // [0:32, 0]
    static inline SliceRange h0_32_w0() { return SliceRange{ .h0 = 0, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 1, .ws = 1 }; }
    // [0:32:1, 1]
    static inline SliceRange h0_32_w1() { return SliceRange{ .h0 = 0, .h1 = 32, .hs = 1, .w0 = 1, .w1 = 2, .ws = 1 }; }
    // [0:4:1, 0:4:1]
    static inline SliceRange hw041() { return SliceRange{.h0 = 0, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1}; }
};

//
// Slices/samples elements of a tile 'itile' from cb using a given numpy style slice object SliceRange.
// Sampling happens relative to the current CB read or write pointer.
// This means that for printing a tile read from the front of the CB,
// the DPRINT << TSLICE(...) call has to occur after cb_wait_front and before cb_pop_front
// For the case of printing a tile from the back of the CB
// the DPRINT << TSLICE(...) call has to occur after cb_reserve_back and before cb_push_back.
//
// MAXCOUNT is the size of reserved space in the print buffer
// if the total element count produced by the slice spec exceeds MAXCOUNT, it will be truncated
//
typedef bool dprint_tslice_ptr_t;
#define TSLICE_RD_PTR true
#define TSLICE_WR_PTR false

template<int MAXCOUNT=32>
struct TileSlice : TileSliceHostDev<MAXCOUNT> {
    static inline int min_(int a, int b) { return a < b ? a : b; } // to avoid inclusion of <algorithm>
    static inline int tilize_rm_index(int i) {
        // map from rm-index to tiled index
        int w = (i&31), h = (i>>5); // RM i -> RM hw
        int iface = int(w>=16) + 2*int(h>=16);
        w &= 15; h &= 15;
        return (iface<<8) + (h<<4) + w;
    }

    struct Tile { uint16_t vals[1024] __attribute__((packed)); } __attribute__((aligned(2)));
    // samples the tile using python style slice with strides [h0:h1:hs, w0:w1:ws]
    // endl_rows=false skips printing the endl in the end of row, so it's easier to visualize tall columns

    __attribute__((__noinline__)) TileSlice(
        int cb,
        int itile,
        const SliceRange& s,
        // For NCRISC and BRISC, have access to both rd and wr ptr, let user choose w/ arg.
#if defined(COMPILE_FOR_NCRISC)
        dprint_tslice_ptr_t ptr_type = TSLICE_WR_PTR,
#elif defined(COMPILE_FOR_BRISC)
        dprint_tslice_ptr_t ptr_type = TSLICE_RD_PTR,
#endif
        bool endl_rows = true,
        bool print_untilized = true) {
        // The math risc uses a different mechanism for syncing data, and as such doesn't have
        // access to CBs, so TileSlice printing is skipped on this risc.
        this->count_ = 0;
        volatile Tile* t;
        // Pointer value depends on whether we're looking at read or write ptr
#if defined(UCK_CHLKC_PACK)
        this->ptr_ = cb_interface[cb].fifo_wr_ptr << 4; // PACK only has write pointer
#elif defined(UCK_CHLKC_UNPACK)
        this->ptr_ = cb_interface[cb].fifo_rd_ptr << 4; // UNPACK only has read pointer
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        // For BRISC/NCRISC, user chooses which pointer.
        this->ptr_ =
            (ptr_type == TSLICE_WR_PTR) ? cb_interface[cb].fifo_wr_ptr << 4 : cb_interface[cb].fifo_rd_ptr << 4;
#else
        this->ptr_ = 0;
#endif
#if defined(DEBUG_PRINT_ENABLED) && (defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC))
        this->ptr_ += itile * sizeof(Tile);
        if (this->ptr_ < L1_UNRESERVED_BASE || this->ptr_ >= MEM_L1_SIZE) {
            this->w0_ = 0xFFFF;
            return; // bad tile pointer, return
        }
        this->endl_rows_ = endl_rows;
        this->w0_ = s.w0;  this->w1_ = s.w1;  this->ws_ = s.ws;
        this->h0_ = s.h0;  this->h1_ = s.h1;  this->hs_ = s.hs;
        t = reinterpret_cast<volatile Tile*>(this->ptr_);
        bool max_count_exceeded = false;
        for (int h = s.h0; h < s.h1; h += s.hs) {
            for (int w = s.w0; w < s.w1; w += s.ws) {
                // Tile size is 32, so 1D index is w_idx + h_idx * 32
                const int log_tile_height = 5;
                int i = w + (h << log_tile_height);
                if (print_untilized) i = TileSlice::tilize_rm_index(i); // tilize the index
                this->samples_[this->count_] = t->vals[i];
                this->count_ ++;
                // If we've gone over the maximum data points to print, break
                if (this->count_ >= MAXCOUNT) {
                    max_count_exceeded = true;
                    break;
                }
            }

            if (max_count_exceeded)
                break;
        }
#endif
    }
} ATTR_PACK;

using TSLICE8  = TileSlice<8>;
using TSLICE32 = TileSlice<32>;
using TSLICE   = TileSlice<32>;

template<> uint8_t DebugPrintTypeToId<TileSlice<8>>()  { return DPrintTILESLICE; } // TODO(AP): can we use SFINAE here?
template<> uint8_t DebugPrintTypeToId<TileSlice<32>>() { return DPrintTILESLICE; }

template DebugPrinter operator<< <TSLICE8>(DebugPrinter, TSLICE8 val);
template DebugPrinter operator<< <TSLICE32>(DebugPrinter, TSLICE32 val);

// Macros for printing circular buffer internals
#define CB_RD_PTR(id) (cb_interface[id].fifo_rd_ptr<<4) // only valid in unpacker thread
#define CB_RD_LIM(id) ((cb_interface[id].fifo_limit_plus_1-1)<<4)
#define CB_RD_SZ(id) (cb_interface[id].fifo_size<<4)

#define CB_WR_PTR(id) (cb_interface[id].fifo_wr_ptr<<4) // only valid in packer thread
#define CB_WR_TILES(id) (cb_interface[output].fifo_num_pages)
