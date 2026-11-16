// wq_b compute: per plan segment (wqb_plan.h), MT output tiles accumulate in DST over the segment's K range with one
// matmul_block (rt = MT, ct = 1) per k. in0 is CB 0 as [kk][m]; the weight tiles of chunk c (plan units
// [c * CH, ...)) are in CB 1 or CB 5, whichever RISC claimed the chunk (wqb_wstream.h); the unpacker finds it through
// the claim tables in CB 7. Non-finalized segments are packed to CB 3 (sent away by the writer). A finalized
// segment adds its partials from CB 4 (slot (fi * 3 + s) * MT) into DST before packing to CB 2.
// compile-time: [KH, NT, NCOLS, HROWS, MT, KB, CH]
// common runtime: NCOLS * HROWS + 1 plan offsets (wqb_plan.h)
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "wqb_plan.h"

void kernel_main() {
    constexpr uint32_t KH = get_compile_time_arg_val(0), NT = get_compile_time_arg_val(1);
    constexpr uint32_t NCOLS = get_compile_time_arg_val(2), HROWS = get_compile_time_arg_val(3);
    constexpr uint32_t MT = get_compile_time_arg_val(4), KB = get_compile_time_arg_val(5);
    constexpr uint32_t CH = get_compile_time_arg_val(6);
    constexpr uint32_t CB_IN0 = 0, CB_W0 = 1, CB_OUT = 2, CB_PSEND = 3, CB_PRECV = 4, CB_W1 = 5, CB_TABLE = 7;

    const auto plan = wqb::make_plan<KH, NT, NCOLS, HROWS, 0>();
    const bool top = plan.half == 0;
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_W0, CB_OUT);
    matmul_block_init(CB_IN0, CB_W0, 0, 1, MT, 1);

    uint32_t j = 0, in0_ready = 0;
    // UNPACK thread only: where the current chunk's tiles are (CB, first tile) and how far each claim table was read
    uint32_t cur_cb = CB_W0, cur_base = 0, next_k[2] = {0, 0};
    auto locate_chunk = [&](uint32_t c) {
        UNPACK(({
            volatile uint32_t* table = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(CB_TABLE).fifo_rd_ptr
                                                                             << CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT);
            bool found = false;
            while (!found) {
                for (uint32_t r = 0; r < 2 && !found; ++r) {
                    const uint32_t cb = r ? CB_W1 : CB_W0;
                    const uint16_t avail = (uint16_t)reg_read((uint32_t)get_cb_tiles_received_ptr(cb)) -
                                           get_local_cb_interface(cb).tiles_acked;
                    const uint32_t chunks = (avail + CH - 1) / CH;
                    while (next_k[r] < chunks && table[r * wqb::TABLE_STRIDE + next_k[r]] < c) {
                        ++next_k[r];  // a chunk both RISCs claimed
                    }
                    if (next_k[r] < chunks && table[r * wqb::TABLE_STRIDE + next_k[r]] == c) {
                        cur_cb = cb;
                        cur_base = next_k[r] * CH;
                        ++next_k[r];
                        found = true;
                    }
                }
            }
        }));
    };
    auto mm_segment = [&](const wqb::Seg& sg) {
        for (uint32_t kk = sg.kb; kk < sg.ke; ++kk, ++j) {
            const uint32_t need = (kk / KB + 1) * MT * KB;
            if (need > in0_ready) {
                cb_wait_front(CB_IN0, need);
                in0_ready = need;
            }
            if (j % CH == 0) {
                locate_chunk(j / CH);
            }
            matmul_block(CB_IN0, cur_cb, kk * MT, cur_base + j % CH, 0, 0, 1, MT, 1);
        }
    };
    auto pack_to = [&](uint32_t ocb) {
        cb_reserve_back(ocb, MT);
        for (uint32_t m = 0; m < MT; ++m) {
            pack_tile(m, ocb);
        }
        cb_push_back(ocb, MT);
    };

    uint32_t n_fin = 0;
    for (uint32_t si = 0; si < plan.n; ++si) {
        const auto& sg = plan.s[si];
        tile_regs_acquire();
        mm_segment(sg);
        const bool fin = top && sg.kind != wqb::HEAD;
        if (fin) {
            cb_wait_front(CB_PRECV, 3 * MT * (n_fin + 1));
            reconfig_data_format_srca(CB_W0, CB_PRECV);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_PRECV);
            for (uint32_t p = 0; p < wqb::expected_partials(sg); ++p) {
                for (uint32_t m = 0; m < MT; ++m) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(CB_PRECV, (sg.fi * 3 + p) * MT + m, m);
                }
            }
            reconfig_data_format_srca(CB_PRECV, CB_W0);
            matmul_block_init(CB_IN0, CB_W0, 0, 1, MT, 1);
        }
        n_fin += fin;
        tile_regs_commit();
        tile_regs_wait();
        pack_to(fin ? CB_OUT : CB_PSEND);
        tile_regs_release();
    }
}
