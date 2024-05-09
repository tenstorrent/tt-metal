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
    static inline SliceRange h1_w0_32() { return SliceRange{ .h0 = 1, .h1 = 2, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h2_w0_32() { return SliceRange{ .h0 = 2, .h1 = 3, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h3_w0_32() { return SliceRange{ .h0 = 3, .h1 = 4, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h4_w0_32() { return SliceRange{ .h0 = 4, .h1 = 5, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h5_w0_32() { return SliceRange{ .h0 = 5, .h1 = 6, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h6_w0_32() { return SliceRange{ .h0 = 6, .h1 = 7, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h7_w0_32() { return SliceRange{ .h0 = 7, .h1 = 8, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h8_w0_32() { return SliceRange{ .h0 = 8, .h1 = 9, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h9_w0_32() { return SliceRange{ .h0 = 9, .h1 = 10, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h10_w0_32() { return SliceRange{ .h0 = 10, .h1 = 11, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h11_w0_32() { return SliceRange{ .h0 = 11, .h1 = 12, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h12_w0_32() { return SliceRange{ .h0 = 12, .h1 = 13, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h13_w0_32() { return SliceRange{ .h0 = 13, .h1 = 14, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h14_w0_32() { return SliceRange{ .h0 = 14, .h1 = 15, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h15_w0_32() { return SliceRange{ .h0 = 15, .h1 = 16, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h16_w0_32() { return SliceRange{ .h0 = 16, .h1 = 17, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h17_w0_32() { return SliceRange{ .h0 = 17, .h1 = 18, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h18_w0_32() { return SliceRange{ .h0 = 18, .h1 = 19, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h19_w0_32() { return SliceRange{ .h0 = 19, .h1 = 20, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h20_w0_32() { return SliceRange{ .h0 = 20, .h1 = 21, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h21_w0_32() { return SliceRange{ .h0 = 21, .h1 = 22, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h22_w0_32() { return SliceRange{ .h0 = 22, .h1 = 23, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h23_w0_32() { return SliceRange{ .h0 = 23, .h1 = 24, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h24_w0_32() { return SliceRange{ .h0 = 24, .h1 = 25, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h25_w0_32() { return SliceRange{ .h0 = 25, .h1 = 26, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h26_w0_32() { return SliceRange{ .h0 = 26, .h1 = 27, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h27_w0_32() { return SliceRange{ .h0 = 27, .h1 = 28, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h28_w0_32() { return SliceRange{ .h0 = 28, .h1 = 29, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h29_w0_32() { return SliceRange{ .h0 = 29, .h1 = 30, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h30_w0_32() { return SliceRange{ .h0 = 30, .h1 = 31, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }
    static inline SliceRange h31_w0_32() { return SliceRange{ .h0 = 31, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }; }

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

    __attribute__((__noinline__))
    TileSlice(int cb, int itile, const SliceRange& s, bool endl_rows = true, bool print_untilized = true) {
        // The math risc uses a different mechanism for syncing data, and as such doesn't have
        // access to CBs, so TileSlice printing is skipped on this risc.
        this->count_ = 0;
        volatile Tile* t;
#if defined(TRISC_PACK) || defined(COMPILE_FOR_NCRISC)
        this->ptr_ = cb_interface[cb].fifo_wr_ptr<<4;
#elif defined(TRISC_UNPACK) || defined(COMPILE_FOR_BRISC)
        this->ptr_ = cb_interface[cb].fifo_rd_ptr<<4;
#else
        this->ptr_ = 0;
#endif
#if defined(TRISC_PACK) || defined(TRISC_UNPACK) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
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
