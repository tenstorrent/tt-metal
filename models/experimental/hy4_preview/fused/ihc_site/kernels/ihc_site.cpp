// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused iHC site of Hy4-preview decode for 8 token rows on one chip (fused/ihc_site/op.py builds the programs,
// layout.py documents the pages). One source for the three RISC roles of each of the K site cores; core k owns the
// hidden columns [k C / K, (k + 1) C / K) of every stream.
//
// Stages (compile-time flags do_post / do_pre select them; both = post-mix of one sublayer + pre-mix of the next):
//   post     streams += post_j * F, in place, SFPU fp32; with do_pre the squares of the new streams are summed too
//   ss       (do_pre without do_post) sum of squares of the streams, SFPU fp32
//   project  streams @ fn^T, FPU HiFi4 with fp32 accumulation, one [16, 32] x [32, 16] matmul per pair of faces
//            (19-bit operands: 5e-4 relative); or with sfpu_proj in the SFPU, fp32 throughout, one streams page
//            against its four fn pages at a time
//   exchange every core sends its partial (projection, sum of squares; 512 B) to every core; each core sums the K
//            partials in core order, so all of them hold the same coefficients. With root_exchange (more than 16
//            cores) the partials go to the root core only, which multicasts the coefficient page it derives.
//   param    rsqrt, sigmoid parametrisation, every coefficient broadcast over its own cell of the coefficient page
//            (kept in DEST for the mix and packed to the state tensor for the post-mix that follows the sublayer)
//   mix      y = sum_j pre_j * stream j, SFPU fp32, packed to the rows tensor; y^2 summed
//   stat     every core sends its sum of y^2 (256 B) to the root core, which leaves 1 / RMS(y) per token row
//
// Waits (who produces what each wait needs):
//   compute  tensor-backed input CBs: this core's reader pushes them unconditionally at start;
//            cb_xchg: this core's reader, after K increments of sem_part, one from every core's writer, each sent
//            once that core's compute packed its partial (which needs only that core's own inputs);
//            cb_xchg2 (root): the root's reader after K increments of sem_stat, one from every writer after its
//            core's mix (which needs only cb_xchg);
//            cb_part reserve (1 page): free at start, and freed again by this core's writer after the first send.
//   writer   cb_part twice: this core's compute, after project and after mix.
//   reader   the two semaphores above. No wait depends on a later stage of another core.
//   root_exchange: only the root's reader waits for sem_part, and only the root's compute for cb_xchg; the root's
//            reader then waits for cb_coef_out (the root's compute, after param) and multicasts it; every other
//            reader waits for one increment of sem_coef (that multicast) and presents cb_coef2 to its compute. The
//            multicast is the only one in the program and nothing else is on the NoC while it runs: every partial
//            has arrived, and no core sends its sum of y^2 before it holds the coefficients.

#include <cstdint>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_common.h"
#endif
#include "ihc_sfpu.hpp"
#endif

namespace {
constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_xmm = get_named_compile_time_arg_val("cb_xmm");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t cb_f = get_named_compile_time_arg_val("cb_f");
constexpr uint32_t cb_coef = get_named_compile_time_arg_val("cb_coef");
constexpr uint32_t cb_const = get_named_compile_time_arg_val("cb_const");
constexpr uint32_t cb_xchg = get_named_compile_time_arg_val("cb_xchg");
constexpr uint32_t cb_xchg2 = get_named_compile_time_arg_val("cb_xchg2");
constexpr uint32_t cb_xout = get_named_compile_time_arg_val("cb_xout");
constexpr uint32_t cb_y = get_named_compile_time_arg_val("cb_y");
constexpr uint32_t cb_part = get_named_compile_time_arg_val("cb_part");
constexpr uint32_t cb_coef_out = get_named_compile_time_arg_val("cb_coef_out");
constexpr uint32_t cb_stat = get_named_compile_time_arg_val("cb_stat");
constexpr uint32_t cb_dbg = get_named_compile_time_arg_val("cb_dbg");
constexpr uint32_t cb_coef2 = get_named_compile_time_arg_val("cb_coef2");
constexpr uint32_t K = get_named_compile_time_arg_val("num_cores");
constexpr uint32_t px = get_named_compile_time_arg_val("x_pages");  // streams pages of this core
constexpr uint32_t py = px / 4;                                     // rows pages of this core
constexpr bool do_post = get_named_compile_time_arg_val("do_post") == 1;
constexpr bool do_pre = get_named_compile_time_arg_val("do_pre") == 1;
constexpr bool probe = get_named_compile_time_arg_val("probe") == 1;
constexpr bool f_fp32 = get_named_compile_time_arg_val("f_fp32") == 1;
constexpr uint32_t sem_part = get_named_compile_time_arg_val("sem_part");
constexpr uint32_t sem_stat = get_named_compile_time_arg_val("sem_stat");
constexpr uint32_t sem_coef = get_named_compile_time_arg_val("sem_coef");
// partials to the root core only, coefficients back by one multicast (instead of every core to every core)
constexpr bool root_exchange = get_named_compile_time_arg_val("root_exchange") == 1;
// stop after a stage (latency breakdown only): 0 = run everything, 1 = after post / ss, 2 = after project,
// 3 = after param, 4 = after mix (no statistic)
constexpr uint32_t stop_after = get_named_compile_time_arg_val("stop_after");
// the 4 streams pages of a batch in one unpacker run (16 faces) instead of 4 (latency experiment)
constexpr bool block_unpack = get_named_compile_time_arg_val("block_unpack") == 1;
// fallback, chosen from the probe: unpack pages into DEST slots 0..3 only and move those that belong higher with
// the SFPU (the unpacker's reach beyond slot 3 of an fp32 DEST is what the probe checks)
constexpr bool low_unpack = get_named_compile_time_arg_val("low_unpack") == 1;
// fallback projection: the streams page and the fn^T page as [32, 32] tiles, one standard full-tile matmul per
// page (twice the multiplies; chosen by the tests if the [16, 32] x [32, 16] form is off on the device)
constexpr bool mm32 = get_named_compile_time_arg_val("mm32") == 1;
constexpr uint32_t mm_tiles = mm32 ? px : 2 * px;
// the projection in the SFPU instead of the FPU: cb_w then holds fp32 fn pages, four per streams page
// (layout.py fn_to_row_pages), and cb_xmm is not used
constexpr bool sfpu_proj = get_named_compile_time_arg_val("sfpu_proj") == 1;
constexpr uint32_t w_tiles = sfpu_proj ? 4 * px : mm_tiles;
// fn^T and the constants of the site come from DRAM tensors that hold every site (the reader fetches this core's
// pages of the site given by its runtime args) instead of from L1 tensors of one site
constexpr bool site_dram = get_named_compile_time_arg_val("site_dram") == 1;
// latency breakdown only: every kernel returns at once, which leaves the cost of launching the program on the
// site cores
constexpr bool empty_program = get_named_compile_time_arg_val("empty_program") == 1;
// tests only: leave this core's scratch page (projection, partial sums, mixes, coefficients) in the debug tensor
constexpr bool debug = get_named_compile_time_arg_val("debug") == 1;

constexpr uint32_t xchg_pages = (K + 7) / 8;  // 8 cores' partials per exchange page
constexpr uint32_t PART_BYTES = 512;          // face 1, rows 0..7 of the scratch page
constexpr uint32_t PART_OFFSET = 1024;
constexpr uint32_t STAT_BYTES = 256;  // face 2, rows 4..7 of the scratch page
constexpr uint32_t STAT_OFFSET = 2048 + 256;
constexpr uint32_t stat_pages = (K + 15) / 16;  // 16 cores' sums of y^2 per exchange page
constexpr uint32_t COEF_BYTES = 2048;           // faces 0, 1 of the coefficient page
}  // namespace

#if defined(COMPILE_FOR_NCRISC)

namespace {
FORCE_INLINE void present(uint32_t cb, uint32_t pages) {
    cb_reserve_back(cb, pages);
    cb_push_back(cb, pages);
}
}  // namespace

// The site's constants page and this core's fn^T pages from the DRAM tensors of all sites. A template on the
// position of the tensor accessor arguments, so that it is only compiled into a program that has them.
// Runtime args 5..8: address of the fn^T tensor, of the constants tensor, this core's first fn^T page of the site,
// the site's constants page. The constants first (the compute starts with them); fn^T is not needed before the
// projection.
template <uint32_t Base>
FORCE_INLINE void fetch_site() {
    constexpr uint32_t w_page = sfpu_proj ? 4096 : 2048;
    constexpr uint32_t w_pages = sfpu_proj ? 4 * px : px;
    constexpr auto w_args = TensorAccessorArgs<Base>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto consts = TensorAccessor(c_args, get_arg_val<uint32_t>(6), 4096);
    cb_reserve_back(cb_const, 1);
    noc_async_read(consts.get_noc_addr(get_arg_val<uint32_t>(8)), get_write_ptr(cb_const), 4096);
    noc_async_read_barrier();
    cb_push_back(cb_const, 1);
    const auto weights = TensorAccessor(w_args, get_arg_val<uint32_t>(5), w_page);
    const uint32_t page0 = get_arg_val<uint32_t>(7);
    cb_reserve_back(cb_w, w_tiles);
    const uint32_t dst = get_write_ptr(cb_w);
    for (uint32_t i = 0; i < w_pages; ++i) {
        noc_async_read(weights.get_noc_addr(page0 + i), dst + i * w_page, w_page);
    }
    noc_async_read_barrier();
    cb_push_back(cb_w, w_tiles);
}

void kernel_main() {
    if constexpr (empty_program) {
        return;
    }
    const uint32_t is_root = get_arg_val<uint32_t>(0);
    if constexpr (probe) {
        present(cb_x, 1);
        present(cb_const, 1);
        return;
    }
    if constexpr (do_pre && !site_dram) {
        present(cb_const, 1);
    }
    if constexpr (do_post) {
        present(cb_coef, 1);
        present(cb_f, py);
    }
    present(cb_x, px);
    if constexpr (do_pre) {
        if constexpr (!sfpu_proj) {
            present(cb_xmm, mm_tiles);
        }
        if constexpr (site_dram) {
            fetch_site<0>();
        } else {
            present(cb_w, w_tiles);
        }
        if constexpr (stop_after == 0 || stop_after >= 3) {
            if (!root_exchange || is_root) {
                cb_reserve_back(cb_xchg, xchg_pages);
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part)), K);
                cb_push_back(cb_xchg, xchg_pages);
            }
            if constexpr (root_exchange && K > 1) {
                if (is_root) {
                    // runtime args 1..4: the site cores' rectangle (noc x, y of its first and last core)
                    cb_wait_front(cb_coef_out, 1);
                    const uint32_t page = get_read_ptr(cb_coef_out);  // same address on every site core
                    const uint32_t xs = get_arg_val<uint32_t>(1);
                    const uint32_t ys = get_arg_val<uint32_t>(2);
                    const uint32_t xe = get_arg_val<uint32_t>(3);
                    const uint32_t ye = get_arg_val<uint32_t>(4);
                    noc_async_write_multicast(page, get_noc_multicast_addr(xs, ys, xe, ye, page), COEF_BYTES, K - 1);
                    noc_async_write_barrier();
                    noc_semaphore_inc_multicast(
                        get_noc_multicast_addr(xs, ys, xe, ye, get_semaphore(sem_coef)), 1, K - 1);
                    noc_async_atomic_barrier();
                } else {
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_coef)), 1);
                    present(cb_coef2, 1);
                }
            }
        }
        if constexpr (stop_after == 0) {
            if (is_root) {
                cb_reserve_back(cb_xchg2, stat_pages);
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stat)), K);
                cb_push_back(cb_xchg2, stat_pages);
            }
        }
    }
}

#elif defined(COMPILE_FOR_BRISC)

void kernel_main() {
    if constexpr (do_pre && !probe && !empty_program && (stop_after == 0 || stop_after >= 3)) {
        // runtime args: [index of this core, root noc x, root noc y, (noc x, noc y) of every site core]
        const uint32_t index = get_arg_val<uint32_t>(0);
        {
            cb_wait_front(cb_part, 1);
            const uint32_t src = get_read_ptr(cb_part) + PART_OFFSET;
            const uint32_t dst = get_write_ptr(cb_xchg) + index * PART_BYTES;  // same address on every site core
            const uint32_t sem = get_semaphore(sem_part);
            if constexpr (root_exchange) {
                const uint32_t rx = get_arg_val<uint32_t>(1);
                const uint32_t ry = get_arg_val<uint32_t>(2);
                noc_async_write(src, get_noc_addr(rx, ry, dst), PART_BYTES);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(rx, ry, sem), 1);
            } else {
                for (uint32_t c = 0; c < K; ++c) {
                    noc_async_write(
                        src,
                        get_noc_addr(get_arg_val<uint32_t>(3 + 2 * c), get_arg_val<uint32_t>(4 + 2 * c), dst),
                        PART_BYTES);
                }
                noc_async_write_barrier();
                for (uint32_t c = 0; c < K; ++c) {
                    noc_semaphore_inc(
                        get_noc_addr(get_arg_val<uint32_t>(3 + 2 * c), get_arg_val<uint32_t>(4 + 2 * c), sem), 1);
                }
            }
            noc_async_atomic_barrier();
            cb_pop_front(cb_part, 1);
        }
        if constexpr (stop_after == 0) {
            cb_wait_front(cb_part, 1);
            const uint32_t src = get_read_ptr(cb_part) + STAT_OFFSET;
            const uint32_t dst = get_write_ptr(cb_xchg2) + index * STAT_BYTES;
            const uint32_t rx = get_arg_val<uint32_t>(1);
            const uint32_t ry = get_arg_val<uint32_t>(2);
            noc_async_write(src, get_noc_addr(rx, ry, dst), STAT_BYTES);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(rx, ry, get_semaphore(sem_stat)), 1);
            noc_async_atomic_barrier();
            cb_pop_front(cb_part, 1);
        }
    }
}

#else  // TRISC

// One SFPU sweep addressed from DEST slot 0 (the sweeps address every slot themselves).
#define IHC_SFPU(...) MATH((_llk_math_eltwise_unary_sfpu_params_(__VA_ARGS__, 0, VectorMode::RC_custom)))
// Hand DEST back to the math thread without clearing it: the coefficient and constant pages and the running sums
// live there for the whole program.
#define IHC_RELEASE() PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, DST_ACCUM_MODE, false>()))

#ifdef TRISC_UNPACK
namespace ihc_unpack {
// The unpacker program of an fp32 page going straight to DEST is one UNPACR per face, each stepping the L1 side and
// the DEST side by one face (llk_unpack_A.h, unpack_srca_to_dest). A page is 4 faces; consecutive pages in L1 and
// consecutive DEST slots are consecutive faces on both sides, so `faces` = 16 moves 4 pages in one run.
inline void program(uint32_t faces) {
    static constexpr uint32_t unpack_face =
        TT_OP_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    ckernel::ckernel_template tmp(faces, 1, unpack_face);
    tmp.program();
}

// Run the programmed unpack from page `tile_index` of `operand` (the DEST slot comes from the math thread).
inline void run(uint32_t operand, uint32_t tile_index) {
    const uint32_t id = get_operand_id(operand);
    const uint32_t address =
        get_local_cb_interface(id).fifo_rd_ptr - 1 + get_local_cb_interface(id).fifo_page_size * tile_index;
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
        address, unpack_src_format[id], unpack_dst_format[id]);
}
}  // namespace ihc_unpack
#endif

namespace {
// The 4 streams pages of batch b into DEST slots 0..3.
ALWI void copy_streams(uint32_t b) {
    if constexpr (block_unpack) {
        UNPACK((ihc_unpack::run(cb_x, 4 * b)));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0, cb_x)));
    } else {
        for (uint32_t k = 0; k < 4; ++k) {
            copy_tile(cb_x, 4 * b + k, k);
        }
    }
}

// The unpacker program for copy_streams / for single pages (copy_tile).
ALWI void unpack_pages(uint32_t pages) {
    if constexpr (block_unpack) {
        UNPACK((ihc_unpack::program(4 * pages)));
    }
}

// A page CB as the source of copy_tile (fp32 pages go straight to DEST, BF16 ones through SrcA).
ALWI void copy_from(uint32_t cb) {
    reconfig_full_operand_srca(cb);
    copy_init(cb);
}

ALWI void sfpu_setup() { MATH((_llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>())); }

// One page of `cb` into DEST slot `slot` (4..7): straight there, or through slot `via` (0..3) with low_unpack.
template <int SLOT_TO, int VIA>
ALWI void copy_page_to(uint32_t cb, uint32_t tile) {
    if constexpr (low_unpack) {
        copy_tile(cb, tile, VIA);
        IHC_SFPU(ihc::copy_page<VIA, SLOT_TO>);
    } else {
        copy_tile(cb, tile, SLOT_TO);
    }
}

// An empty DEST section: gives the packer the section the math thread just committed.
ALWI void pack_page(uint32_t slot, uint32_t cb) {
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(slot, cb);
    IHC_RELEASE();
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    if constexpr (empty_program) {
        return;
    }
    const uint32_t is_root = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_x, cb_xout);

    if constexpr (probe) {
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_const, 1);
        tile_regs_acquire();
        copy_from(cb_x);
        copy_tile(cb_x, 0, 0);
        copy_tile(cb_const, 0, 7);
        sfpu_setup();
        IHC_SFPU(ihc::probe_cells);
        MATH((ckernel::sfpu::sqrt_init<false>()));
        IHC_SFPU(ihc::probe_rsqrt);
        MATH((ckernel::sfpu::sfpu_reciprocal_init<false>()));
        IHC_SFPU(ihc::probe_sigmoid);
        tile_regs_commit();
        pack_page(6, cb_coef_out);
        // second section: DEST slot 6 must still hold what the first one left
        tile_regs_acquire();
        IHC_SFPU(ihc::probe_persist);
        tile_regs_commit();
        pack_page(1, cb_stat);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_const, 1);
        return;
    }

    // ---- constants, post coefficients, zeroed running sums
    cb_wait_front(cb_x, px);
    if constexpr (do_pre) {
        cb_wait_front(cb_const, 1);
    }
    if constexpr (do_post) {
        cb_wait_front(cb_coef, 1);
        cb_wait_front(cb_f, py);
    }
    sfpu_setup();
    tile_regs_acquire();
    copy_from(cb_x);
    if constexpr (do_pre) {
        copy_page_to<7, 0>(cb_const, 0);
    }
    if constexpr (do_post) {
        copy_page_to<6, 0>(cb_coef, 0);
    }
    IHC_SFPU(ihc::zero_state);
    tile_regs_commit();
    tile_regs_wait();
    IHC_RELEASE();
    if constexpr (do_post) {
        cb_pop_front(cb_coef, 1);
    }

    // ---- post-mix (with the sum of squares of the new streams), or the sum of squares alone
    if constexpr (do_post) {
        cb_reserve_back(cb_xout, px);
        pack_reconfig_data_format(cb_xout);
        for (uint32_t b = 0; b < py; ++b) {
            tile_regs_acquire();
            if constexpr (!f_fp32) {
                copy_from(cb_f);
            }
            if constexpr (f_fp32) {
                unpack_pages(1);
            }
            copy_page_to<4, 3>(cb_f, b);
            if constexpr (!f_fp32) {
                copy_from(cb_x);
            }
            unpack_pages(4);
            copy_streams(b);
            IHC_SFPU(ihc::post_rows<0, do_pre>);
            IHC_SFPU(ihc::post_rows<1, do_pre>);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < 4; ++k) {
                pack_tile<true>(k, cb_xout, 4 * b + k);
            }
            IHC_RELEASE();
        }
        cb_push_back(cb_xout, px);
        cb_pop_front(cb_f, py);
        // the streams are read again below: wait until the packer has written them back
        cb_wait_front(cb_xout, px);
    } else {
        unpack_pages(4);
        for (uint32_t b = 0; b < py; ++b) {
            tile_regs_acquire();
            copy_streams(b);
            IHC_SFPU(ihc::ss_rows<0>);
            IHC_SFPU(ihc::ss_rows<1>);
            tile_regs_commit();
            tile_regs_wait();
            IHC_RELEASE();
        }
    }

    if constexpr (do_pre && stop_after != 1 && sfpu_proj) {
        // ---- projection of this core's columns in the SFPU, then its partial sums to every core. One DEST section
        // for the whole pass: nothing is packed before its end.
        cb_wait_front(cb_w, w_tiles);
        copy_from(cb_x);
        sfpu_setup();
        tile_regs_acquire();
        for (uint32_t t = 0; t < px; ++t) {
            // the streams page into slot 4, its four fn pages into slots 0..3
            unpack_pages(1);
            copy_tile(cb_x, t, 4);
            if constexpr (block_unpack) {
                unpack_pages(4);
                UNPACK((ihc_unpack::run(cb_w, 4 * t)));
                MATH((llk_math_eltwise_unary_datacopy<
                      DataCopyType::A2D,
                      DST_ACCUM_MODE,
                      BroadcastType::NONE,
                      UnpackToDestEn>(0, cb_w)));
            } else {
                for (uint32_t j = 0; j < 4; ++j) {
                    copy_tile(cb_w, 4 * t + j, j);
                }
            }
            IHC_SFPU(ihc::proj_page<0>);
            IHC_SFPU(ihc::proj_page<1>);
            IHC_SFPU(ihc::proj_page<2>);
            IHC_SFPU(ihc::proj_page<3>);
        }
        IHC_SFPU(ihc::partial_cells_sfpu);
        tile_regs_commit();
        pack_page(5, cb_part);
        cb_pop_front(cb_w, w_tiles);
    }

    if constexpr (do_pre && stop_after != 1 && !sfpu_proj) {
        // ---- projection of this core's columns, then its partial sums to every core
        cb_wait_front(cb_xmm, mm_tiles);
        cb_wait_front(cb_w, mm_tiles);
        reconfig_full_operand<SrcOrder::Reverse>(cb_xmm, cb_w);
        matmul_init(cb_xmm, cb_w);
        tile_regs_acquire();
        if constexpr (mm32) {
            sfpu_setup();
            IHC_SFPU(ihc::zero_slot0);
            MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));  // the multiplies accumulate onto it
            for (uint32_t t = 0; t < mm_tiles; ++t) {
                matmul_tiles(cb_xmm, cb_w, t, t, 0);
            }
            sfpu_setup();
            IHC_SFPU(ihc::partial_cells_mm32);
        } else {
            for (uint32_t t = 0; t < mm_tiles; ++t) {
                matmul_tiles(cb_xmm, cb_w, t, t, 5);
            }
            sfpu_setup();
            IHC_SFPU(ihc::partial_cells);
        }
        tile_regs_commit();
        pack_page(5, cb_part);
        cb_pop_front(cb_xmm, mm_tiles);
        cb_pop_front(cb_w, mm_tiles);
    }

    if constexpr (do_pre && (stop_after == 0 || stop_after >= 3)) {
        // ---- the K partials -> coefficients (DEST slot 6, and the state tensor for the next post-mix)
        if (!root_exchange || is_root) {
            cb_wait_front(cb_xchg, xchg_pages);
            tile_regs_acquire();
            copy_from(cb_xchg);
            for (uint32_t i = 0; i < xchg_pages; ++i) {
                copy_tile(cb_xchg, i, i);
            }
            MATH((ckernel::sfpu::sqrt_init<false>()));
            IHC_SFPU(ihc::param_norm<K>);
            MATH((ckernel::sfpu::sfpu_reciprocal_init<false>()));
            IHC_SFPU(ihc::param_sigmoid);
            IHC_SFPU(ihc::param_broadcast);
            if constexpr (root_exchange) {
                IHC_SFPU(ihc::zero_y2);
            }
            tile_regs_commit();
            pack_page(6, cb_coef_out);
            cb_pop_front(cb_xchg, xchg_pages);
        } else {
            // the coefficient page the root core multicast into this core's state tensor
            cb_wait_front(cb_coef2, 1);
            tile_regs_acquire();
            copy_from(cb_coef2);
            copy_page_to<6, 0>(cb_coef2, 0);
            tile_regs_commit();
            tile_regs_wait();
            IHC_RELEASE();
            cb_pop_front(cb_coef2, 1);
        }
    }

    if constexpr (do_pre && (stop_after == 0 || stop_after >= 4)) {
        // ---- pre-mix: the collapsed sublayer input, and the sum of its squares
        cb_reserve_back(cb_y, py);
        pack_reconfig_data_format(cb_y);
        copy_from(cb_x);
        unpack_pages(4);
        for (uint32_t b = 0; b < py; ++b) {
            tile_regs_acquire();
            copy_streams(b);
            IHC_SFPU(ihc::mix_rows<0>);
            IHC_SFPU(ihc::mix_rows<1>);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(4, cb_y, b);
            IHC_RELEASE();
        }
        cb_push_back(cb_y, py);
    }

    if constexpr (do_pre && stop_after == 0) {
        // ---- this core's sum of y^2 to the root core; the root leaves 1 / RMS(y) per token row
        tile_regs_acquire();
        tile_regs_commit();
        pack_page(5, cb_part);
        if (is_root) {
            cb_wait_front(cb_xchg2, stat_pages);
            tile_regs_acquire();
            copy_from(cb_xchg2);
            for (uint32_t i = 0; i < stat_pages; ++i) {
                copy_tile(cb_xchg2, i, i);
            }
            MATH((ckernel::sfpu::sqrt_init<false>()));
            IHC_SFPU(ihc::stat_cells<K>);
            tile_regs_commit();
            pack_page(root_exchange ? 4 : 1, cb_stat);  // ihc::STAT_SLOT
            cb_pop_front(cb_xchg2, stat_pages);
        }
    }
    if constexpr (do_pre && debug) {
        tile_regs_acquire();
        tile_regs_commit();
        pack_page(5, cb_dbg);
    }
    cb_pop_front(cb_x, px);
    if constexpr (do_pre) {
        cb_pop_front(cb_const, 1);
    }
}

#endif
