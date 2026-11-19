// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MLA megakernel reader core — activation mover (NCRISC, NoC1), every reader phase in order (kernels/mla/mla_common.hpp
// for the phase model): wait for the phase's producers (hand-off semaphore), feed the compute its in0 (pulled K
// blocks, the RMSNorm input + weight, or head rows gathered from a [1, T, H, D] tensor), write the output tiles
// (column tiles, or head rows scattered into a [1, T, H, D] tensor) and signal the phase's consumers.
//
// Compile-time args: NPH, CB_IN0, CB_X, CB_NW, CB_GIN, IN0_CAP, GIN_CAP (tiles), CB_STG (face-row staging, >= 256 B),
// then per phase p 21 values at 8 + 21p:
//   SRC, DST, NB, KB, NR, NH, KG, NG, WAIT_SEM, WAIT_COUNT (0 = no wait), SIGNAL_SEM, NCONS, XOFF, YOFF, OUT_CB, HPT
//   (tile rows per token of the [1, T, H, D] rows tensor: ceil(H / 32)), NWOFF (CT offset of the norm weight's
//   accessor args; SRC_NORM), RS (1: the output rows are scaled by a per-token column tile, rc.cpp), LATE_SEM,
//   LATE_COUNT (the hand-off of that tile, waited for only before the first output: the matmul runs meanwhile), RSOFF
//   (CT offset of the row-scale tensor's accessor args), then the TensorAccessorArgs blocks.
//   KX (SRC_NORM: tiles normalised) = KG of that phase.
// Runtime args: per phase p 8 values at 8p: x address, y address, n0 (SKIP = sits the phase out: it still waits and
//   signals), h0, norm weight address, T (rows), cons (runtime-arg index of the phase's NCONS consumer (x, y) pairs),
//   the row-scale tensor's address (RS).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/mla_common.hpp"

// TP mesh (tt/tp_mla.py, Stage 14's tt/fabric_ar.py FabricPhase "cb" mode, as kernels/mk/reader_a.cpp): a phase whose
// bit is set in MK_FAB_MASK (o_proj) hands its PARTIAL output to the in-program fabric all-reduce: instead of writing
// its tiles to y and signalling the phase's consumers, the reader writes the VALID rows of each output tile
// (MK_FAB_BROWS <= 16: face-0 rows then face-1 rows, 32 B per row per face; 32: whole tiles) into the `own` buffer of
// the tile's lane (lane = tile / 64) at (tile % 64) * V, then increments the OWN_READY word of BOTH lanes (every
// reader, also one sitting the phase out); the lanes signal the consumers once the reduced result is ready. Runtime
// args at MK_FAB_RT: lane 0 (x, y), lane 1 (x, y), own address, lane sync address.
#if defined(MK_FAB_MASK)
constexpr uint32_t FAB_CH = MK_FAB_BROWS * 32;
constexpr uint32_t FAB_V = MK_FAB_BROWS >= 32 ? 2048 : 2 * FAB_CH;
constexpr uint32_t FAB_LANE_TILES = 64;
#endif

namespace {

constexpr uint32_t NPH = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_IN0 = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_X = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_NW = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t CB_GIN = get_compile_time_arg_val(PH_CT0 + 4);
constexpr uint32_t IN0_CAP = get_compile_time_arg_val(PH_CT0 + 5);
constexpr uint32_t GIN_CAP = get_compile_time_arg_val(PH_CT0 + 6);
constexpr uint32_t CB_STG = get_compile_time_arg_val(PH_CT0 + 7);
constexpr uint32_t PH0 = 8, PHN = 21, RTN = 8;
constexpr uint32_t CB_RS = 38;  // the row-scale tile (RS phases; tt/mk_mla.py CB_RS)

// 64-B aligned staging slots (128 B each) inside CB_STG (the CB is sized STG_SLOTS * 128 + 64 B of slack): the face
// rows of one gather / scatter batch (one barrier per batch). MLA_STG_SLOTS (tt/mk_mla.py stg_slots(t): 32 up to t = 4,
// 128 above -- at t = 32 a W_uv K block is 512 face rows, i.e. 16 barrier round trips at 32 slots)
#if defined(MLA_STG_SLOTS)
constexpr uint32_t STG_SLOTS = MLA_STG_SLOTS;
#else
constexpr uint32_t STG_SLOTS = 32;
#endif
FORCE_INLINE uint32_t staging() { return (get_write_ptr(CB_STG) + 63) & ~63u; }

// in0 of one K block of sub-matmul j: KB tiles [x0, x0 + KB) of x (page = column tile). MLA_ROW_PULL (T <= 16): only
// the T valid rows of each tile -- face 0 and face 1 rows 0..T-1, two T x 32-B reads at tile offsets 0 / 512 (source
// and destination congruent mod 64) -- instead of the whole 2-KB tile (1/32 of the NoC bytes at T = 1); rows >= T of
// CB_IN0 stay zero (zeroed at kernel start, never written; the output rows they feed are never read)
#if defined(MLA_PULL_ZMASK)
// rev 34 (tt/mk_mla.py MkMla.ot at t < 32): the phases in MLA_PULL_ZMASK read a token-row tensor whose rows >= T are
// never written (o_T: the attention roots write rows < T only) -> after the tile reads land, rows >= T of each pulled
// tile are overwritten with zeros from the core's zero region (local NoC reads, source congruent mod 64 with the
// destination: mla_common.hpp; <= 448 B per read so the shifted source stays inside the 512-B region)
FORCE_INLINE void zero_span(uint32_t dst, uint32_t len) {
    while (len > 0) {
        const uint32_t src = MEM_ZEROS_BASE + (((dst & 63) - (MEM_ZEROS_BASE & 63)) & 63);
        const uint32_t n = len < 448 ? len : 448;
        noc_async_read(get_noc_addr(src), dst, n);
        dst += n;
        len -= n;
    }
}
FORCE_INLINE void zero_rows_ge(uint32_t tile, uint32_t t_rows) {
    if (t_rows < 16) {
        zero_span(tile + t_rows * 32, (16 - t_rows) * 32);        // face 0 rows T..15
        zero_span(tile + 512 + t_rows * 32, (16 - t_rows) * 32);  // face 1 rows T..15
        zero_span(tile + 1024, 1024);                             // faces 2, 3
    } else if (t_rows < 32) {
        zero_span(tile + 1024 + (t_rows - 16) * 32, (32 - t_rows) * 32);
        zero_span(tile + 1536 + (t_rows - 16) * 32, (32 - t_rows) * 32);
    }
}
#endif

template <uint32_t KB, typename XA>
FORCE_INLINE void pull_block(const XA& x, uint32_t x0, uint32_t t_rows, bool zero_tail = false) {
    cb_reserve_back(CB_IN0, KB);
    const uint32_t l1 = get_write_ptr(CB_IN0);
    const uint32_t page = get_tile_size(CB_IN0);
#if defined(MLA_ROW_PULL)
    if (t_rows <= 16) {
        const uint32_t nb = t_rows * 32;
        for (uint32_t k = 0; k < KB; ++k) {
            const uint64_t src = x.get_noc_addr(x0 + k, 0);
            noc_async_read(src, l1 + k * page, nb);
            noc_async_read(src + 512, l1 + k * page + 512, nb);
        }
    } else
#endif
    {
        for (uint32_t k = 0; k < KB; ++k) {
            noc_async_read_page(x0 + k, x, l1 + k * page);
        }
    }
    noc_async_read_barrier();
#if defined(MLA_PULL_ZMASK)
    if (zero_tail && t_rows < 32) {  // after the tile reads landed (NoC reads may complete out of order)
        for (uint32_t k = 0; k < KB; ++k) {
            zero_rows_ge(l1 + k * page, t_rows);
        }
        noc_async_read_barrier();
    }
#else
    (void)zero_tail;
#endif
    cb_push_back(CB_IN0, KB);
}

// in0 of one K block gathered from a [1, T, H, KG * 32] tensor: CB_GIN tile k, row t <- head h's row of token t in
// column tile kk0 + k (two 32-B face rows); rows >= T of CB_GIN stay zero (zeroed at kernel start, never written).
// The source rows are read as 64-B windows from their 64-B aligned-down address into staging slots (source and
// destination congruent mod 64, mla_common.hpp), one barrier per batch, then copied by the RISC.
template <uint32_t KB, uint32_t KG, uint32_t HPT, typename XA>
FORCE_INLINE void gather_block(const XA& x, uint32_t h, uint32_t kk0, uint32_t t_rows) {
    cb_reserve_back(CB_GIN, KB);
    const uint32_t l1 = get_write_ptr(CB_GIN);
    const uint32_t src_off = mla::row_off(h & 31);
    const uint32_t stg = staging();
    // rev 29: the gather was RISC-bound (vtp8 t = 32: W_uv ~175 us at any staging batch): one TensorAccessor address
    // per (token, K tile) serves both face rows (+512 B), and the 64-B window offset is computed ONCE -- every page
    // of the interleaved tensor shares the buffer base's alignment mod 64 (2-KB pages) and the faces are 512 B apart
    const uint32_t off = static_cast<uint32_t>(x.get_noc_addr(((h >> 5)) * KG + kk0, src_off)) & 63;
    ASSERT((off & 31) == 0);          // a 32-B face row: the 64-B window holds it whole only at offset 0 or 32
    const uint32_t np = KB * t_rows;  // (token, K tile) pairs of this block, 2 face rows each
    constexpr uint32_t PAIRS = STG_SLOTS / 2;
    for (uint32_t p0 = 0; p0 < np; p0 += PAIRS) {
        const uint32_t m = np - p0 < PAIRS ? np - p0 : PAIRS;
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t q = p0 + i, t = q % t_rows, k = q / t_rows;
            // the 64-B windows holding the 32-B rows: never past a 64-B aligned buffer end
            const uint64_t src = x.get_noc_addr((t * HPT + (h >> 5)) * KG + kk0 + k, src_off) - off;
            noc_async_read(src, stg + (2 * i) * 128, 64);
            noc_async_read(src + 512, stg + (2 * i + 1) * 128, 64);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t q = p0 + i, t = q % t_rows, k = q / t_rows;
            const uint32_t dst = l1 + k * mla::TILE_BYTES_BF16 + mla::row_off(t);
            mla::copy_row32(stg + (2 * i) * 128 + off, dst);
            mla::copy_row32(stg + (2 * i + 1) * 128 + off, dst + 512);
        }
    }
    invalidate_l1_cache();  // the RISC stores land before the unpacker reads the tiles
    cb_push_back(CB_GIN, KB);
}

// head rows of NR output tiles (rows = tokens) -> a [1, T, H, NG * 32] tensor: output column col = c0 + c belongs to
// head col / NG, tile column col % NG; token t's row goes to row h % 32 of tile row t * HPT + h / 32. Each 32-B face
// row is copied by the RISC into a staging slot congruent (mod 64) with its destination, then written; one barrier
// per batch of STG_SLOTS rows.
template <uint32_t NR, uint32_t NG, uint32_t HPT, typename YA>
FORCE_INLINE void scatter_rows(const YA& y, uint32_t l1, uint32_t opage, uint32_t c0, uint32_t t_rows) {
    const uint32_t stg = staging();
    const uint32_t np = NR * t_rows;  // (token, column tile) pairs, 2 face rows each (rev 29: one address per pair)
    constexpr uint32_t PAIRS = STG_SLOTS / 2;
    invalidate_l1_cache();  // the packer's tiles, not stale cache lines
    for (uint32_t p0 = 0; p0 < np; p0 += PAIRS) {
        const uint32_t m = np - p0 < PAIRS ? np - p0 : PAIRS;
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t q = p0 + i, t = q % t_rows, c = q / t_rows;
            const uint32_t col = c0 + c, h = col / NG, cc = col % NG;
            const uint64_t dst = y.get_noc_addr((t * HPT + (h >> 5)) * NG + cc, mla::row_off(h & 31));
            const uint32_t off = static_cast<uint32_t>(dst) & 63;  // the same for both faces (512 B apart)
            const uint32_t src = l1 + c * opage + mla::row_off(t);
            mla::copy_row32(src, stg + (2 * i) * 128 + off);
            mla::copy_row32(src + 512, stg + (2 * i + 1) * 128 + off);
        }
        invalidate_l1_cache();
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t q = p0 + i, t = q % t_rows, c = q / t_rows;
            const uint32_t col = c0 + c, h = col / NG, cc = col % NG;
            const uint64_t dst = y.get_noc_addr((t * HPT + (h >> 5)) * NG + cc, mla::row_off(h & 31));
            const uint32_t off = static_cast<uint32_t>(dst) & 63;
            noc_async_write(stg + (2 * i) * 128 + off, dst, 32);
            noc_async_write(stg + (2 * i + 1) * 128 + off, dst + 512, 32);
        }
        noc_async_write_barrier();
    }
}

#if defined(MK_FAB_MASK)
// output tile c (global column tile) of a fabric phase -> its lane's own buffer (packed valid rows)
FORCE_INLINE void fab_write(uint32_t src, uint32_t c) {
    const uint32_t lane = c / FAB_LANE_TILES;
    const uint64_t dst = get_noc_addr(
        get_arg_val<uint32_t>(MK_FAB_RT + 2 * lane),
        get_arg_val<uint32_t>(MK_FAB_RT + 2 * lane + 1),
        get_arg_val<uint32_t>(MK_FAB_RT + 4) + (c % FAB_LANE_TILES) * FAB_V);
    if constexpr (MK_FAB_BROWS >= 32) {
        noc_async_write(src, dst, 2048);
    } else {
        noc_async_write(src, dst, FAB_CH);
        noc_async_write(src + 512, dst + FAB_CH, FAB_CH);
    }
}

FORCE_INLINE void fab_ready() {
    const uint32_t lsync = get_arg_val<uint32_t>(MK_FAB_RT + 5);  // OWN_READY = word 0 of the lane's sync words
    for (uint32_t lane = 0; lane < 2; ++lane) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(MK_FAB_RT + 2 * lane), get_arg_val<uint32_t>(MK_FAB_RT + 2 * lane + 1), lsync),
            1);
    }
    noc_async_atomic_barrier();
}
#endif

template <uint32_t P>
FORCE_INLINE void move_phase() {
    constexpr uint32_t B = PH0 + PHN * P;
    constexpr uint32_t SRC = get_compile_time_arg_val(PH_CT0 + B + 0);
    constexpr uint32_t DST = get_compile_time_arg_val(PH_CT0 + B + 1);
    constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + B + 2);
    constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + B + 3);
    constexpr uint32_t NR = get_compile_time_arg_val(PH_CT0 + B + 4);
    constexpr uint32_t NH = get_compile_time_arg_val(PH_CT0 + B + 5);
    constexpr uint32_t KG = get_compile_time_arg_val(PH_CT0 + B + 6);
    constexpr uint32_t NG = get_compile_time_arg_val(PH_CT0 + B + 7);
    constexpr uint32_t WAIT_SEM = get_compile_time_arg_val(PH_CT0 + B + 8);
    constexpr uint32_t WAIT_COUNT = get_compile_time_arg_val(PH_CT0 + B + 9);
    constexpr uint32_t SIGNAL_SEM = get_compile_time_arg_val(PH_CT0 + B + 10);
    constexpr uint32_t NCONS = get_compile_time_arg_val(PH_CT0 + B + 11);
    constexpr uint32_t XOFF = get_compile_time_arg_val(PH_CT0 + B + 12);
    constexpr uint32_t YOFF = get_compile_time_arg_val(PH_CT0 + B + 13);
    constexpr uint32_t OUT_CB = get_compile_time_arg_val(PH_CT0 + B + 14);
    constexpr uint32_t HPT = get_compile_time_arg_val(PH_CT0 + B + 15);
    constexpr uint32_t NWOFF = get_compile_time_arg_val(PH_CT0 + B + 16);
    constexpr uint32_t RS = get_compile_time_arg_val(PH_CT0 + B + 17);
    constexpr uint32_t LATE_SEM = get_compile_time_arg_val(PH_CT0 + B + 18);
    constexpr uint32_t LATE_COUNT = get_compile_time_arg_val(PH_CT0 + B + 19);
    constexpr uint32_t RSOFF = get_compile_time_arg_val(PH_CT0 + B + 20);
    constexpr uint32_t BPH = KG / KB;  // K blocks per sub-matmul
    static_assert(KG % KB == 0 && NB == NH * BPH, "blocks");
#if defined(MK_FAB_MASK)
    constexpr bool FAB = (((MK_FAB_MASK) >> P) & 1) != 0;
    static_assert(!FAB || DST == mla::DST_TENSOR, "a fabric phase writes column tiles");
#else
    constexpr bool FAB = false;
#endif
    constexpr auto x_args = TensorAccessorArgs<XOFF>();
    constexpr auto y_args = TensorAccessorArgs<YOFF>();
    const uint32_t a0 = ph_rt0 + RTN * P;
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(a0 + 0));
    const auto y = TensorAccessor(y_args, get_arg_val<uint32_t>(a0 + 1));
    const uint32_t n0 = get_arg_val<uint32_t>(a0 + 2);
    const uint32_t h0 = get_arg_val<uint32_t>(a0 + 3);
    const uint32_t t_rows = get_arg_val<uint32_t>(a0 + 5);
    const uint32_t cons = get_arg_val<uint32_t>(a0 + 6);

    if constexpr (WAIT_COUNT > 0) {
        mk::wait_handoff(WAIT_SEM, WAIT_COUNT);
    }
    MK_MARK();
    if (n0 != mla::SKIP) {
        if constexpr (SRC == mla::SRC_NORM) {
            // the whole normalised input once (the compute needs every tile for the row sums), then per K block the
            // norm weight tiles (fp32, row 0 = w)
            constexpr auto nw_args = TensorAccessorArgs<NWOFF>();
            const auto nw = TensorAccessor(nw_args, get_arg_val<uint32_t>(a0 + 4));
            cb_reserve_back(CB_X, KG);
            const uint32_t l1 = get_write_ptr(CB_X);
            const uint32_t page = get_tile_size(CB_X);
            for (uint32_t k = 0; k < KG; ++k) {
                noc_async_read_page(k, x, l1 + k * page);
            }
            noc_async_read_barrier();
            cb_push_back(CB_X, KG);
            const uint32_t wpage = get_tile_size(CB_NW);
            for (uint32_t blk = 0; blk < NB; ++blk) {
                cb_reserve_back(CB_NW, KB);
                const uint32_t wl1 = get_write_ptr(CB_NW);
                for (uint32_t k = 0; k < KB; ++k) {
                    noc_async_read_page(blk * KB + k, nw, wl1 + k * wpage);
                }
                noc_async_read_barrier();
                cb_push_back(CB_NW, KB);
            }
        }
        for (uint32_t j = 0; j < NH; ++j) {
            if constexpr (SRC == mla::SRC_TENSOR) {
                for (uint32_t b = 0; b < BPH; ++b) {
#if defined(MLA_PULL_ZMASK)
                    pull_block<KB>(x, (h0 + j) * KG + b * KB, t_rows, (((MLA_PULL_ZMASK) >> P) & 1) != 0);
#else
                    pull_block<KB>(x, (h0 + j) * KG + b * KB, t_rows);
#endif
                }
            } else if constexpr (SRC == mla::SRC_ROWS) {
                for (uint32_t b = 0; b < BPH; ++b) {
                    gather_block<KB, KG, HPT>(x, h0 + j, b * KB, t_rows);
                }
            }
            if constexpr (RS == 1) {  // the row scale before the first output (the compute waits for it there)
                if (j == 0) {
                    if constexpr (LATE_COUNT > 0) {
                        mk::wait_handoff(LATE_SEM, LATE_COUNT);
                    }
                    constexpr auto rs_args = TensorAccessorArgs<RSOFF>();
                    const auto rs = TensorAccessor(rs_args, get_arg_val<uint32_t>(a0 + 7));
                    cb_reserve_back(CB_RS, 1);
                    noc_async_read_page(0, rs, get_write_ptr(CB_RS));
                    noc_async_read_barrier();
                    cb_push_back(CB_RS, 1);
                }
            }
            // sub-matmul j's NR output tiles
            cb_wait_front(OUT_CB, NR);
            const uint32_t l1 = get_read_ptr(OUT_CB);
            const uint32_t opage = get_tile_size(OUT_CB);
            const uint32_t c0 = n0 + j * NG;
            for (uint32_t c = 0; c < NR; ++c) {
                if constexpr (FAB) {
#if defined(MK_FAB_MASK)
                    fab_write(l1 + c * opage, c0 + c);
#endif
                } else if constexpr (DST == mla::DST_TENSOR) {
                    noc_async_write_page(c0 + c, y, l1 + c * opage);
                }
            }
            if constexpr (DST == mla::DST_ROWS) {
                scatter_rows<NR, NG, HPT>(y, l1, opage, c0, t_rows);
            }
            noc_async_write_barrier();
            cb_pop_front(OUT_CB, NR);
            mk::cb_rewind(OUT_CB);  // the next sub-matmul's / phase's NR tiles start at the base again
        }
        // phase end: every in0 page pushed was consumed (the compute packed the last output after it) -> the producer
        // side back to the base for the next phase (kernels/mk/mk_common.hpp: rewind only an empty CB)
        if constexpr (SRC == mla::SRC_NORM) {
            cb_reserve_back(CB_X, KG);
            mk::cb_rewind(CB_X);
            cb_reserve_back(CB_NW, 2 * KB);
            mk::cb_rewind(CB_NW);
        } else if constexpr (SRC == mla::SRC_ROWS) {
            cb_reserve_back(CB_GIN, GIN_CAP);
            mk::cb_rewind(CB_GIN);
        } else {
            cb_reserve_back(CB_IN0, IN0_CAP);
            mk::cb_rewind(CB_IN0);
        }
    }
    if constexpr (FAB) {  // the lanes signal the consumers once the all-reduced rows are ready
#if defined(MK_FAB_MASK)
        fab_ready();
#endif
    } else if constexpr (NCONS > 0) {
        mk::signal_handoff(SIGNAL_SEM, ph_rt0 + cons, NCONS);
    }
    MK_MARK();
}

template <uint32_t P>
FORCE_INLINE void run_a() {
    if constexpr (P < NPH) {
        move_phase<P>();
        run_a<P + 1>();
    }
}

}  // namespace

void kernel_main() {
    MK_MARK();
#if defined(MLA_ROW_PULL)
    // CB_IN0 rows >= T must read as zero (the row pull writes rows < T only): zero the CB's whole region per launch
    // (another program of the step may have used this L1)
    {
        const uint32_t base = get_write_ptr(CB_IN0);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < IN0_CAP * mla::TILE_BYTES_BF16; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
#endif
#if defined(MLA_GIN_ZERO)
    // CB_GIN rows >= T must read as zero (the gather writes rows < T only): zero the CB's whole region once
    {
        const uint32_t base = get_write_ptr(CB_GIN);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < GIN_CAP * mla::TILE_BYTES_BF16; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
#endif
    run_a<0>();
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB) + 64);
#endif
}
