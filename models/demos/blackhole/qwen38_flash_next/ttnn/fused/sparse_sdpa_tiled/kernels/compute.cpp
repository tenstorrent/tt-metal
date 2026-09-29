// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_tiled compute: the streaming flash loop of sparse_sdpa_compute.cpp over a tile of R = TQ * H query
// rows (Sqt = R / 32 tile rows) and the tile's union chunks of Skt key tiles.  Per tile: tilize Q [Sqt, DQt]; per
// chunk: tilize the K/V rows to [Skt, DHt] (V in tile columns V_OFF_T.., K in K_OFF_T..), then per query group of
// qsb tile rows: Phase 1 Q @ K^T into the held cb_qk_im band, the writer's mask band L1-accumulated onto every tile
// row of the group (the band depends on row % TQ only), the running row max, exp((s - max) * scale) in place with
// the partial row sum, Phase 2 probs @ V, and the SALAD correction of the previous chunk's out / sum.  The last
// chunk normalizes and the tile's [Sqt, vDHt] result is untilized to row-major for the writer.
//
// With FP32_STATE the scores, the running max / sum / out, the correction and the reciprocal scratch are fp32 CBs
// (one pack format per chunk); without it they are bf16 as in sparse_sdpa.  The probabilities' exp is the
// approximate SFPU exp on every config (sub_exp_block_bcast_cols); MATH_APPROX selects the correction exp only.
//
// Named compile-time args: TQ H SKT DQT DHT VDHT K_OFF_T V_OFF_T SCALE QSB MATH_APPROX and the CB ids CB_Q_RM CB_Q_IN
// CB_K_RM CB_K_IN CB_SCALE CB_COL_IDENTITY CB_MASK_BAND CB_QK_IM CB_MAX_A CB_MAX_B CB_SUM_A CB_SUM_B CB_OUT_A CB_OUT_B
// CB_CORR CB_OUT_IM CB_OUT_RM CB_RECIP CB_CTRL.  Runtime args: 0 tile_count.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "common.h"
#include "../../kernels/zones.h"
// Study builds (DEBUG_STAGE named arg): 1 = chunk-0 scores after the band, 2 = raw scores, 3 = probabilities leave
// as the tile's output; 0 = the kernel.
constexpr uint32_t SST_DEBUG_STAGE = get_named_compile_time_arg_val("DEBUG_STAGE");
// compute_streaming.hpp reads EXP_APPROX_MODE (the correction exp's mode) and needs compute_common.hpp first.
constexpr bool EXP_APPROX_MODE = get_named_compile_time_arg_val("MATH_APPROX") != 0;
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"
#include "normalize.h"

namespace {

// Make in-place PACK writes to a held CB visible to the next UNPACK read (sparse_sdpa_compute.cpp).
ALWI void pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

ALWI void swap_cb(CircularBuffer& a, CircularBuffer& b) {
    const CircularBuffer t = a;
    a = b;
    b = t;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t TQ = get_named_compile_time_arg_val("TQ");
    constexpr uint32_t H = get_named_compile_time_arg_val("H");
    constexpr uint32_t Skt = get_named_compile_time_arg_val("SKT");
    constexpr uint32_t DQt = get_named_compile_time_arg_val("DQT");
    constexpr uint32_t DHt = get_named_compile_time_arg_val("DHT");
    constexpr uint32_t vDHt = get_named_compile_time_arg_val("VDHT");
    constexpr uint32_t k_off_t = get_named_compile_time_arg_val("K_OFF_T");
    constexpr uint32_t v_off_t = get_named_compile_time_arg_val("V_OFF_T");
    constexpr uint32_t scale_fp32 = get_named_compile_time_arg_val("SCALE");
    constexpr uint32_t qsb = get_named_compile_time_arg_val("QSB");
    constexpr uint32_t cb_q_rm = get_named_compile_time_arg_val("CB_Q_RM");
    constexpr uint32_t cb_q_in = get_named_compile_time_arg_val("CB_Q_IN");
    constexpr uint32_t cb_k_rm = get_named_compile_time_arg_val("CB_K_RM");
    constexpr uint32_t cb_k_in = get_named_compile_time_arg_val("CB_K_IN");
    constexpr uint32_t cb_scale = get_named_compile_time_arg_val("CB_SCALE");
    constexpr uint32_t cb_col_identity = get_named_compile_time_arg_val("CB_COL_IDENTITY");
    constexpr uint32_t cb_mask_band = get_named_compile_time_arg_val("CB_MASK_BAND");
    constexpr uint32_t cb_qk_im = get_named_compile_time_arg_val("CB_QK_IM");
    constexpr uint32_t cb_max_a = get_named_compile_time_arg_val("CB_MAX_A");
    constexpr uint32_t cb_max_b = get_named_compile_time_arg_val("CB_MAX_B");
    constexpr uint32_t cb_sum_a = get_named_compile_time_arg_val("CB_SUM_A");
    constexpr uint32_t cb_sum_b = get_named_compile_time_arg_val("CB_SUM_B");
    constexpr uint32_t cb_out_a = get_named_compile_time_arg_val("CB_OUT_A");
    constexpr uint32_t cb_out_b = get_named_compile_time_arg_val("CB_OUT_B");
    constexpr uint32_t cb_corr = get_named_compile_time_arg_val("CB_CORR");
    constexpr uint32_t cb_out_im = get_named_compile_time_arg_val("CB_OUT_IM");
    constexpr uint32_t cb_out_rm = get_named_compile_time_arg_val("CB_OUT_RM");
    constexpr uint32_t cb_recip_scratch = get_named_compile_time_arg_val("CB_RECIP");
    constexpr uint32_t cb_ctrl = get_named_compile_time_arg_val("CB_CTRL");

    constexpr uint32_t R = TQ * H;
    constexpr uint32_t Sqt = R / 32;
    static_assert(R % 32 == 0, "the tile's rows must be whole tile rows");
    static_assert(Sqt % qsb == 0, "the query groups must be equal");
    constexpr uint32_t q_groups = Sqt / qsb;
    constexpr uint32_t KT_stride = Skt;  // cb_qk_im physical row width
    constexpr uint32_t k_chunk = Skt * 32;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    static_assert(qsb <= dst_size, "a query group must fit DEST");
    // sub_exp packs qsb * sbw tiles into DEST: the full Skt width when one group fits, else one key-tile column.
    constexpr uint32_t exp_sbw = (qsb * Skt <= dst_size) ? Skt : 1;
    static_assert(SST_DEBUG_STAGE == 0 || Skt == vDHt, "the stage dump reuses the output drain: Skt must equal vDHt");

    CircularBuffer q_in_cb(cb_q_in), k_in_cb(cb_k_in), qk_cb(cb_qk_im), scale_cb(cb_scale), ctrl_cb(cb_ctrl);
    CircularBuffer band_cb(cb_mask_band), corr_cb(cb_corr);

    const uint32_t tile_count = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_q_rm, cb_q_in);
    scale_cb.wait_front(1);  // persistent reduce scaler

    for (uint32_t tile = 0; tile < tile_count; ++tile) {
        FUSED_ZONE("fz_ss_c_tile");
        compute_kernel_lib::tilize<DQt, cb_q_rm, cb_q_in>(/*num_blocks=*/Sqt, /*total_input_pages=*/R);

        ctrl_cb.wait_front(1);
        const uint32_t n_chunks = read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/sst::ctrl::N_CHUNKS);
        ctrl_cb.pop_front(1);

        // Flash running state, ping-pong; every buffer starts empty for the tile.
        CircularBuffer max_prev(cb_max_a), max_cur(cb_max_b);
        CircularBuffer sum_prev(cb_sum_a), sum_cur(cb_sum_b);
        CircularBuffer out_prev(cb_out_a), out_cur(cb_out_b);

        for (uint32_t chunk = 0; chunk < n_chunks; ++chunk) {
            FUSED_ZONE("fz_ss_c_chunk");
            // K/V rows -> [Skt, DHt] tiles (the wait absorbs the gather).
            if constexpr (SST_DEBUG_STAGE == 7 || SST_DEBUG_STAGE == 9) {
                // Study: no tilize; the rows are consumed and the tile CB carries stale bytes.
                CircularBuffer k_rm_cb(cb_k_rm);
                k_rm_cb.wait_front(k_chunk);
                k_rm_cb.pop_front(k_chunk);
                k_in_cb.reserve_back(Skt * DHt);
                k_in_cb.push_back(Skt * DHt);
            } else {
                compute_kernel_lib::tilize<DHt, cb_k_rm, cb_k_in>(/*num_blocks=*/Skt, /*total_input_pages=*/k_chunk);
            }
            // The tilize left srcA / the packer in its formats: QK reads K (srcA) and Q (srcB); the downstream
            // packs share cb_qk_im's format (fp32 or bf16) until the final normalize.
            reconfig_full_operand(cb_k_in, cb_q_in);
            pack_reconfig_data_format(cb_qk_im);

            const bool is_first = (chunk == 0);
            const bool is_last = (chunk == n_chunks - 1);

            qk_cb.reserve_back(Sqt * KT_stride);
            sum_cur.reserve_back(Sqt);
            out_cur.reserve_back(Sqt * vDHt);
            k_in_cb.wait_front(Skt * DHt);
            q_in_cb.wait_front(Sqt * DQt);
            band_cb.wait_front(Skt);  // the writer's mask band of this chunk

            if constexpr (SST_DEBUG_STAGE == 4 || SST_DEBUG_STAGE >= 6) {
                // Study: no math; the gather / union / band path alone.  The out / sum tiles keep stale L1 bytes,
                // the last chunk's normalize runs on them (finite or not: the output is not read).
                sum_cur.push_back(Sqt);
                out_cur.push_back(Sqt * vDHt);
                band_cb.pop_front(Skt);
                if (is_last) {
                    max_cur.reserve_back(Sqt);
                    max_cur.push_back(Sqt);
                    sst::normalize_rows<vDHt, dst_size, cb_col_identity, cb_recip_scratch, cb_out_im>(
                        sum_cur.get_cb_id(), out_cur.get_cb_id(), Sqt);
                    max_cur.pop_front(Sqt);
                } else {
                    sum_cur.pop_front(Sqt);
                    out_cur.pop_front(Sqt * vDHt);
                }
                qk_cb.push_back(Sqt * KT_stride);
                qk_cb.pop_front(Sqt * KT_stride);
                k_in_cb.pop_front(Skt * DHt);
                continue;
            }

            for (uint32_t qg = 0; qg < q_groups; ++qg) {
                const uint32_t row_base = qg * qsb;

                exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();

                // ===== Phase 1: Q @ K^T over the K half of the rows -> the held cb_qk_im band =====
                mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, qsb, DQt);
                configure_row_pack_width(cb_qk_im, 1);
                for (uint32_t kt = 0; kt < Skt; ++kt) {
                    blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                        cb_q_in,
                        cb_k_in,
                        cb_qk_im,
                        /*in0_index_start=*/row_base * DQt,
                        /*in1_index_start=*/kt * DHt + k_off_t,
                        /*row_subblock_idx=*/qg,
                        /*out_col_offset=*/kt,
                        /*subblock_w=*/1,
                        /*subblock_h=*/qsb,
                        /*inner_dim=*/DQt,
                        /*matmul_stride=*/DQt,
                        /*skip_pack_configure=*/true);
                }
                cb_push_back_hold_wr_ptr(cb_qk_im, qsb * KT_stride);

                // ===== The membership band: scores += band (0 / MASK_FLOOR) on every tile row of the group =====
                qk_cb.wait_front((qg + 1) * qsb * KT_stride);
                if constexpr (SST_DEBUG_STAGE != 2) {
                    // srcA holds K's format after Phase 1; the band copy reads the band's (the helper's <true>
                    // form reconfigures only when cb_qk_im's and the band's formats differ: right by coincidence)
                    reconfig_data_format_srca(cb_mask_band);
                    begin_mask_l1_accumulate<false>(cb_qk_im, cb_mask_band);
                    apply_provided_mask_streaming<qsb, KT_stride, /*mask_stride=*/0>(cb_mask_band, cb_qk_im, qg, Skt);
                    end_mask_l1_accumulate();
                    pack_to_unpack_sync();
                }
                if constexpr (SST_DEBUG_STAGE == 1 || SST_DEBUG_STAGE == 2) {
                    // Study: the held scores of chunk 0 (after the band at 1, raw at 2) leave as the output.
                    if (is_first && qg == q_groups - 1) {
                        compute_kernel_lib::untilize<Skt, cb_qk_im, cb_out_rm>(/*num_blocks=*/Sqt);
                    }
                    if (is_first) {
                        continue;
                    }
                }

                // ===== running row max (eltwise max against the previous chunk's on chunk > 0) =====
                reconfig_data_format(cb_qk_im, cb_scale);
                max_cur.reserve_back(qsb);
                configure_single_tile_pack(max_cur.get_cb_id());
                reduce_c_row_group<cb_qk_im, cb_scale, KT_stride>(
                    max_cur.get_cb_id(),
                    max_prev.get_cb_id(),
                    /*row_group_index=*/qg,
                    /*do_eltwise_max=*/!is_first,
                    qsb,
                    Skt);
                max_cur.push_back(qsb);

                // ===== exp((s - max) * scale) in place + the partial row sum (L1-accumulated into sum_cur) =====
                reconfig_data_format(cb_qk_im, max_cur.get_cb_id());
                for (uint32_t kc = 0; kc < Skt; kc += exp_sbw) {
                    sub_exp_block_bcast_cols<false, scale_fp32>(
                        cb_qk_im,
                        max_cur.get_cb_id(),
                        sum_cur.get_cb_id(),
                        /*cols_in_row=*/KT_stride,
                        /*q_subblock=*/qg,
                        /*global_col_base=*/kc,
                        /*sbh=*/qsb,
                        /*sbw=*/exp_sbw);
                }
                pack_to_unpack_sync();
                if constexpr (SST_DEBUG_STAGE == 3) {
                    // Study: the probabilities of chunk 0 leave as the output.
                    if (is_first && qg == q_groups - 1) {
                        compute_kernel_lib::untilize<Skt, cb_qk_im, cb_out_rm>(/*num_blocks=*/Sqt);
                    }
                    if (is_first) {
                        continue;
                    }
                }

                // ===== Phase 2: probs @ V (the V tile columns of the same K/V tiles) -> out_cur band =====
                reconfig_data_format(cb_k_in, cb_qk_im);
                mm_no_mop_init_short(cb_qk_im, cb_k_in, /*transpose=*/false, 1, qsb, Skt);
                configure_row_pack_width(out_cur.get_cb_id(), 1);
                for (uint32_t vd = 0; vd < vDHt; ++vd) {
                    blocked_matmul_and_pack<false, /*in1_stride=*/DHt, /*out_num_cols=*/vDHt>(
                        cb_qk_im,
                        cb_k_in,
                        out_cur.get_cb_id(),
                        /*in0_index_start=*/row_base * Skt,
                        /*in1_index_start=*/v_off_t + vd,
                        /*row_subblock_idx=*/qg,
                        /*out_col_offset=*/vd,
                        /*subblock_w=*/1,
                        /*subblock_h=*/qsb,
                        /*inner_dim=*/Skt,
                        /*matmul_stride=*/KT_stride,
                        /*skip_pack_configure=*/true);
                }
                pack_to_unpack_sync();
                reconfig_data_format_srca(cb_qk_im);

                // ===== SALAD: cur += prev * exp((prev_max - cur_max) * scale) =====
                if (!is_first) {
                    exp_packthread_tile_init<EXP_APPROX_MODE>();
                    corr_cb.reserve_back(qsb);
                    sub_exp_first_col_blocks<false, scale_fp32>(
                        max_prev.get_cb_id(), max_cur.get_cb_id(), cb_corr, /*q_subblock=*/qg, qsb);
                    corr_cb.push_back(qsb);
                    PACK((
                        llk_pack_init<ckernel::PackMode::Default, false, false, false>(out_cur.get_cb_id(), dst_size)));
                    pack_reconfig_l1_acc(1);
                    salad_correct_fused<qsb, vDHt, dst_size>(
                        out_prev.get_cb_id(),
                        sum_prev.get_cb_id(),
                        cb_corr,
                        out_cur.get_cb_id(),
                        sum_cur.get_cb_id(),
                        /*ob_q_subblock=*/0,
                        /*sum_q_subblock=*/qg,
                        /*write_q_subblock=*/qg);
                    pack_reconfig_l1_acc(0);
                    corr_cb.pop_front(qsb);
                    out_prev.pop_front(qsb * vDHt);
                }
            }  // query groups

            band_cb.pop_front(Skt);
            if constexpr (SST_DEBUG_STAGE >= 1 && SST_DEBUG_STAGE <= 3) {
                // Study builds: chunk 0 left the group loop early (the dump's untilize popped the held score
                // tiles and filled the output drain).  Release what was reserved; the later chunks run the loop
                // (their state is never drained) and the final untilize is skipped.
                if (is_first) {
                    sum_cur.push_back(Sqt);
                    out_cur.push_back(Sqt * vDHt);
                    k_in_cb.pop_front(Skt * DHt);
                    swap_cb(max_prev, max_cur);
                    swap_cb(sum_prev, sum_cur);
                    swap_cb(out_prev, out_cur);
                    continue;
                }
            }
            if (!is_first) {
                max_prev.pop_front(Sqt);
                sum_prev.pop_front(Sqt);
            }
            sum_cur.push_back(Sqt);
            out_cur.push_back(Sqt * vDHt);

            if (is_last) {
                FUSED_ZONE("fz_ss_c_normalize");
                // sst::normalize_rows = normalize_row_streaming with the unpacker formats set per operand pair (the
                // fp32 running state reads garbage through the shared helper's col-identity format; normalize.h).
                sst::normalize_rows<vDHt, dst_size, cb_col_identity, cb_recip_scratch, cb_out_im>(
                    sum_cur.get_cb_id(), out_cur.get_cb_id(), Sqt);
                max_cur.pop_front(Sqt);
            }

            qk_cb.pop_front(Sqt * KT_stride);
            k_in_cb.pop_front(Skt * DHt);

            swap_cb(max_prev, max_cur);
            swap_cb(sum_prev, sum_cur);
            swap_cb(out_prev, out_cur);
        }

        q_in_cb.pop_front(Sqt * DQt);
        if constexpr (SST_DEBUG_STAGE == 0 || SST_DEBUG_STAGE >= 4) {
            compute_kernel_lib::untilize<vDHt, cb_out_im, cb_out_rm>(/*num_blocks=*/Sqt);
        }
    }
}
