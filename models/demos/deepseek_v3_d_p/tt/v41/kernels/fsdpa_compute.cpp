// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute of the decode step's flash-decode SDPA (tt/v41/flash_sdpa.py; fsdpa_common.hpp has the layout). Per unit
// of this core: sparse_sdpa's single-chunk arithmetic on one query tile (the chip's heads) and KPU keys (the K slab
// tilized as the cache stores it: the scaled-FP8 latent into bfp8 tiles times its fp32 scales, the bf16 RoPE tail
// apart; or the bf16 rows), Q @ K^T as the latent and the RoPE products, the -inf mask past the unit's valid keys,
// the row max, exp((s - max) * scale) in place with the partial row sums L1-accumulated into the sum tile, P @ V:
// the max tile, the sum tile and the VT acc tiles go to the writer unnormalized. Per merge unit of this core
// (row, v): over the row's partials j, M = max_j m_j, w_j = exp((m_j - M) * scale) on the first column (the SALAD
// correction's exp), S = sum_j w_j * s_j, the denominator S @ col_identity + exp((sink - M) * scale) (the sink fold
// of normalize_row_streaming), its reciprocal on the first column, A = sum_j w_j * a_jv, out = A * (1 / l) and the
// tile untilized for the writer. The sums over j accumulate in DST (fp32 when the KV is fp8, as the unit's
// matmuls).
//
// compile_time_args = fsdpa::ct; runtime_args = [core]

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "models/demos/deepseek_v3_d_p/tt/v41/kernels/fsdpa_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"
constexpr bool EXP_APPROX_MODE = get_compile_time_arg_val(fsdpa::ct::MATH_APPROX) != 0;
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/constants.hpp>

using namespace fsdpa;

// In-place scores[tile] += row-broadcast of mask row 0 (sparse_sdpa_compute.cpp).
ALWI void add_bcast_row_mask_tile(uint32_t scores_cb, uint32_t mask_cb, uint32_t score_tile) {
    tile_regs_acquire();
    add_tiles_bcast_rows(scores_cb, mask_cb, score_tile, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, scores_cb, score_tile);
    tile_regs_release();
}

// Make in-place PACK writes to a held CB visible to the next UNPACK read (sparse_sdpa_compute.cpp).
ALWI void pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

// Tilize a 32-row field of a mixed-format packed slab (sparse_sdpa_compute.cpp).
template <
    uint32_t input_cb,
    uint32_t output_cb,
    uint32_t physical_ct_dim,
    uint32_t field_ct_dim,
    uint32_t field_offset_16b,
    bool manage_input_lifecycle = true>
ALWI void tilize_packed_field() {
    CircularBuffer in(input_cb), out(output_cb);
    if constexpr (manage_input_lifecycle) {
        in.wait_front(tt::constants::TILE_HEIGHT);
    }
    out.reserve_back(field_ct_dim);
    reconfig_data_format_srca(input_cb);
    pack_reconfig_data_format(output_cb);
    tilize_init(input_cb, physical_ct_dim, output_cb);
    uint32_t saved_rd_ptr = 0;
    UNPACK({
        auto& iface = get_local_cb_interface(input_cb);
        saved_rd_ptr = iface.fifo_rd_ptr;
        iface.fifo_rd_ptr = saved_rd_ptr + field_offset_16b;
    });
    tilize_block(input_cb, field_ct_dim, output_cb);
    UNPACK(get_local_cb_interface(input_cb).fifo_rd_ptr = saved_rd_ptr;);
    out.push_back(field_ct_dim);
    if constexpr (manage_input_lifecycle) {
        in.pop_front(tt::constants::TILE_HEIGHT);
    }
    tilize_uninit(input_cb, output_cb);
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(ct::H);
    constexpr uint32_t D = get_compile_time_arg_val(ct::D);
    constexpr uint32_t V = get_compile_time_arg_val(ct::V);
    constexpr uint32_t KPU = get_compile_time_arg_val(ct::KPU);
    constexpr uint32_t UMAX = get_compile_time_arg_val(ct::UMAX);
    constexpr uint32_t N = get_compile_time_arg_val(ct::N);
    constexpr bool scaled_kv = get_compile_time_arg_val(ct::SCALED_KV) != 0;
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(ct::PACKED_ROW_BYTES);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(ct::SCALE);
    constexpr uint32_t VT = get_compile_time_arg_val(ct::VT);
    constexpr uint32_t DHt = D / tt::constants::TILE_WIDTH;
    constexpr uint32_t vDHt = V / tt::constants::TILE_WIDTH;
    constexpr uint32_t Skt = KPU / tt::constants::TILE_HEIGHT;
    static_assert(H == tt::constants::TILE_HEIGHT, "one tile of heads");
    static_assert(VT == vDHt, "one merge unit a column tile");
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t exp_sbw = (Skt <= dst_size) ? Skt : 1;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
    constexpr uint16_t negated_scale_bf16 = scale_bf16 ^ 0x8000;
    const uint32_t core = get_arg_val<uint32_t>(0);

    CircularBuffer q_in_cb(cb_q_in), k_in_cb(cb_k_in), qk_cb(cb_qk_im), scale_cb(cb_scale), ctrl(cb_ctrl_c);
    CircularBuffer neginf_cb(cb_neginf), mask_part_cb(cb_mask_part), max_cb(cb_max), sum_cb(cb_sum), out_cb(cb_out);
    CircularBuffer k_rm_cb(cb_k_rm), scale_bcast_cb(cb_k_scale_bcast);
    CircularBuffer latent_tile_cb(cb_k_latent_tile), rope_tile_cb(cb_k_rope_tile);
    CircularBuffer pm(cb_pm), ps(cb_ps), pa(cb_pa), mM(cb_mM), mw(cb_mw), mws(cb_mws), mS(cb_mS);
    CircularBuffer mrecip(cb_mrecip), mA(cb_mA), max_w(cb_max_w), o_rm(cb_o_rm), colid(cb_col_identity),
        sink_cb(cb_sink);

    compute_kernel_hw_startup(cb_q_rm, cb_q_in);
    scale_cb.wait_front(1);

    ctrl.wait_front(1);
    const uint32_t n_live = ckernel::read_tile_value(cb_ctrl_c, 0, PLAN_N_LIVE);
    ctrl.pop_front(1);

    // ---- the units ----
    const uint32_t total_units = n_live * UMAX;
    for (uint32_t u = core; u < total_units; u += N) {
        ctrl.wait_front(1);
        const uint32_t valid = ckernel::read_tile_value(cb_ctrl_c, 0, 0);
        ctrl.pop_front(1);
        if (valid == 0) {
            continue;
        }
        // Q rows -> [1, DHt]
        compute_kernel_lib::tilize<DHt, cb_q_rm, cb_q_in>(/*num_blocks=*/1, /*total_input_pages=*/H);

        if constexpr (scaled_kv) {
            constexpr uint32_t latent_DHt = vDHt;
            constexpr uint32_t rope_DHt = DHt - vDHt;
            constexpr uint32_t latent_width = latent_DHt * tt::constants::TILE_WIDTH;
            constexpr uint32_t tiles_per_scale_block = sparse_sdpa::SCALE_BLOCK_WIDTH / tt::constants::TILE_WIDTH;
            constexpr uint32_t scale_blocks = sparse_sdpa::scale_block_count(latent_width);
            constexpr uint32_t scale_bytes = sparse_sdpa::scaled_kv_scale_bytes(latent_width);
            static_assert(rope_DHt == 0 || sparse_sdpa::scaled_kv_rope_offset_is_aligned(latent_width));
            static_assert(packed_row_bytes % (tt::constants::TILE_WIDTH * sizeof(uint16_t)) == 0);
            constexpr uint32_t latent_physical_ct = packed_row_bytes / tt::constants::TILE_WIDTH;
            constexpr uint32_t rope_physical_ct = packed_row_bytes / (tt::constants::TILE_WIDTH * sizeof(uint16_t));
            constexpr uint32_t rope_offset_16b =
                (latent_width + scale_bytes) / sparse_sdpa::PACKED_FIELD_ADDRESS_UNIT_BYTES;

            k_in_cb.reserve_back(Skt * vDHt);
            for (uint32_t row_tile = 0; row_tile < Skt; ++row_tile) {
                k_rm_cb.wait_front(tt::constants::TILE_HEIGHT);
                tilize_packed_field<cb_k_rm, cb_k_latent_tile, latent_physical_ct, latent_DHt, 0, false>();
                {
                    latent_tile_cb.wait_front(latent_DHt);
                    scale_bcast_cb.wait_front(scale_blocks);
                    reconfig_data_format(cb_k_latent_tile, cb_k_scale_bcast);
                    pack_reconfig_data_format(cb_k_in);
                    mul_bcast_cols_init(cb_k_latent_tile, cb_k_scale_bcast);
                    for (uint32_t block = 0; block < scale_blocks; ++block) {
                        for (uint32_t tile = 0; tile < tiles_per_scale_block; ++tile) {
                            tile_regs_acquire();
                            mul_tiles_bcast_cols(
                                cb_k_latent_tile, cb_k_scale_bcast, block * tiles_per_scale_block + tile, block, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile<true>(0, cb_k_in, row_tile * vDHt + block * tiles_per_scale_block + tile);
                            tile_regs_release();
                        }
                    }
                    latent_tile_cb.pop_front(latent_DHt);
                    scale_bcast_cb.pop_front(scale_blocks);
                }
                if constexpr (rope_DHt > 0) {
                    UNPACK(get_local_cb_interface(cb_k_rope_rm).fifo_rd_ptr =
                               get_local_cb_interface(cb_k_rm).fifo_rd_ptr;);
                    tilize_packed_field<
                        cb_k_rope_rm,
                        cb_k_rope_tile,
                        rope_physical_ct,
                        rope_DHt,
                        rope_offset_16b,
                        false>();
                }
                k_rm_cb.pop_front(tt::constants::TILE_HEIGHT);
            }
            k_in_cb.push_back(Skt * vDHt);
        } else {
            compute_kernel_lib::tilize<DHt, cb_k_rm, cb_k_in>(/*num_blocks=*/Skt, /*total_input_pages=*/KPU);
        }
        reconfig_full_operand(cb_k_in, cb_q_in);
        pack_reconfig_data_format(cb_qk_im);

        qk_cb.reserve_back(Skt);
        sum_cb.reserve_back(1);
        out_cb.reserve_back(vDHt);
        if constexpr (scaled_kv) {
            k_in_cb.wait_front(Skt * vDHt);
            if constexpr (DHt > vDHt) {
                rope_tile_cb.wait_front(Skt * (DHt - vDHt));
            }
        } else {
            k_in_cb.wait_front(Skt * DHt);
        }
        q_in_cb.wait_front(DHt);

        const uint32_t full_start = (valid + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
        const uint32_t part_t = valid / tt::constants::TILE_WIDTH;
        const bool has_part_mask = (valid % tt::constants::TILE_WIDTH) != 0;

        exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
        // ===== Q @ K^T -> cb_qk_im (held), mask, row max =====
        if constexpr (scaled_kv) {
            constexpr uint32_t rope_DHt = DHt - vDHt;
            reconfig_data_format(cb_k_in, cb_q_in);
            mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, 1, DHt);
            configure_row_pack_width(cb_qk_im, 1);
            for (uint32_t kt = 0; kt < Skt; ++kt) {
                blocked_matmul_and_pack<true, 1, Skt>(
                    cb_q_in, cb_k_in, cb_qk_im, 0, kt * vDHt, 0, kt, 1, 1, vDHt, DHt, true);
            }
            if constexpr (rope_DHt > 0) {
                reconfig_data_format(cb_k_rope_tile, cb_q_in);
                mm_no_mop_init_short(cb_q_in, cb_k_rope_tile, /*transpose=*/true, 1, 1, DHt);
                configure_row_pack_width(cb_qk_im, 1);
                pack_reconfig_l1_acc(1);
                for (uint32_t kt = 0; kt < Skt; ++kt) {
                    blocked_matmul_and_pack<true, 1, Skt>(
                        cb_q_in, cb_k_rope_tile, cb_qk_im, vDHt, kt * rope_DHt, 0, kt, 1, 1, rope_DHt, DHt, true);
                }
                pack_reconfig_l1_acc(0);
            }
        } else {
            mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, 1, DHt);
            configure_row_pack_width(cb_qk_im, 1);
            for (uint32_t kt = 0; kt < Skt; ++kt) {
                blocked_matmul_and_pack<true, 1, Skt>(
                    cb_q_in, cb_k_in, cb_qk_im, 0, kt * DHt, 0, kt, 1, 1, DHt, DHt, true);
            }
        }
        cb_push_back_hold_wr_ptr(cb_qk_im, Skt);

        reconfig_data_format(cb_qk_im, cb_scale);
        if (full_start < Skt || has_part_mask) {
            qk_cb.wait_front(Skt);
            if (full_start < Skt) {
                neginf_cb.wait_front(1);
                add_bcast_rows_init(cb_qk_im, cb_neginf);
                configure_single_tile_pack(cb_qk_im);
                for (uint32_t t = full_start; t < Skt; ++t) {
                    add_bcast_row_mask_tile(cb_qk_im, cb_neginf, t);
                }
            }
            if (has_part_mask) {
                mask_part_cb.wait_front(1);
                add_bcast_rows_init(cb_qk_im, cb_mask_part);
                configure_single_tile_pack(cb_qk_im);
                add_bcast_row_mask_tile(cb_qk_im, cb_mask_part, part_t);
            }
            pack_to_unpack_sync();
        }
        // the row max: the compute's copy (sub_exp reads it, popped below) and the writer's (the partial's max)
        max_cb.reserve_back(1);
        max_w.reserve_back(1);
        configure_single_tile_pack(cb_max);
        reduce_c_row_group<cb_qk_im, cb_scale, Skt>(
            cb_max, cb_max, 0, /*do_eltwise_max=*/false, 1, Skt, false, /*mirror_cb=*/cb_max_w);
        max_cb.push_back(1);
        max_w.push_back(1);
        for (uint32_t kc = 0; kc < Skt; kc += exp_sbw) {
            sub_exp_block_bcast_cols<false, scale_fp32>(cb_qk_im, cb_max, cb_sum, Skt, 0, kc, 1, exp_sbw);
        }
        pack_to_unpack_sync();

        // ===== P @ V -> cb_out =====
        {
            qk_cb.wait_front(Skt);
            reconfig_data_format(cb_k_in, cb_qk_im);
            mm_no_mop_init_short(cb_qk_im, cb_k_in, /*transpose=*/false, 1, 1, Skt);
            configure_row_pack_width(cb_out, 1);
            for (uint32_t vd = 0; vd < vDHt; ++vd) {
                blocked_matmul_and_pack<false, scaled_kv ? vDHt : DHt, vDHt>(
                    cb_qk_im, cb_k_in, cb_out, 0, vd, 0, vd, 1, 1, Skt, Skt, true);
            }
            pack_to_unpack_sync();
            reconfig_data_format_srca(cb_qk_im);
        }
        if (has_part_mask) {
            mask_part_cb.pop_front(1);
        }
        sum_cb.push_back(1);
        out_cb.push_back(vDHt);
        max_cb.pop_front(1);
        qk_cb.pop_front(Skt);
        if constexpr (scaled_kv) {
            k_in_cb.pop_front(Skt * vDHt);
            if constexpr (DHt > vDHt) {
                rope_tile_cb.pop_front(Skt * (DHt - vDHt));
            }
        } else {
            k_in_cb.pop_front(Skt * DHt);
        }
        q_in_cb.pop_front(DHt);
        // the writer drains cb_max_w, cb_sum and cb_out
    }

    // ---- the merge units ----
    const uint32_t total_merges = n_live * VT;
    for (uint32_t mu = N - 1 - core; mu < total_merges; mu += N) {
        ctrl.wait_front(1);
        const uint32_t U = ckernel::read_tile_value(cb_ctrl_c, 0, 0);
        ctrl.pop_front(1);
        pm.wait_front(U);
        ps.wait_front(U);
        pa.wait_front(U);
        colid.wait_front(1);
        sink_cb.wait_front(1);

        // M = max_j m_j (the first column): the running max ping-pongs through cb_mM's two pages. No FPU op follows
        // an SFPU op inside one DST acquire anywhere in the merge (the SFPU's reads and the FPU's writes of a DST
        // tile are not ordered against each other): every step is FPU loads, then SFPU, then pack.
        reconfig_data_format(cb_pm, cb_pm);
        pack_reconfig_data_format(cb_mM);
        copy_init(cb_pm);
        binary_max_tile_init();
        mM.reserve_back(1);
        tile_regs_acquire();
        copy_tile(cb_pm, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(cb_mM);
        pack_tile(0, cb_mM);
        tile_regs_release();
        mM.push_back(1);
        for (uint32_t j = 1; j < U; ++j) {
            mM.wait_front(1);
            mM.reserve_back(1);
            tile_regs_acquire();
            copy_tile(cb_pm, j, 0);
            copy_tile(cb_mM, 0, 1);
            binary_max_tile(0, 1, 0, VectorMode::C);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_mM);
            tile_regs_release();
            mM.push_back(1);
            mM.pop_front(1);
        }
        mM.wait_front(1);

        // w_j = exp((m_j - M) * scale), w_sink = exp((sink - M) * scale) (first columns)
        reconfig_data_format(cb_pm, cb_mM);
        pack_reconfig_data_format(cb_mw);
        sub_init(cb_pm, cb_mM);
        mw.reserve_back(U);
        configure_single_tile_pack(cb_mw);
        for (uint32_t j = 0; j < U; ++j) {
            tile_regs_acquire();
            sub_tiles(cb_pm, cb_mM, j, 0, 0);
            // the first-column exp needs the generic unary SFPU state (normalize_row_streaming's sink fold)
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
            MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_mw);
            tile_regs_release();
        }
        mw.push_back(U);
        mws.reserve_back(1);
        reconfig_data_format(cb_mM, cb_sink);
        sub_init(cb_mM, cb_sink);
        tile_regs_acquire();
        sub_tiles(cb_mM, cb_sink, 0, 0, 0);
        MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
        MATH((exp_tile_first_column<EXP_APPROX_MODE, negated_scale_bf16>(0)));
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(cb_mws);
        pack_tile(0, cb_mws);
        tile_regs_release();
        mws.push_back(1);
        mw.wait_front(U);
        mws.wait_front(1);
        mM.pop_front(1);  // cb_mM is the final tile's CB below

        // S = sum_j w_j (x) s_j ; A = sum_j w_j (x) a_jv  (column broadcast; the sums L1-accumulated by the packer,
        // as salad_correct_fused accumulates the running state)
        reconfig_data_format(cb_ps, cb_mw);
        mul_bcast_cols_init(cb_ps, cb_mw);
        mS.reserve_back(1);
        configure_single_tile_pack(cb_mS);
        for (uint32_t j = 0; j < U; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_ps, cb_mw, j, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(j > 0 ? 1 : 0);
            pack_tile<true>(0, cb_mS, 0);
            tile_regs_release();
        }
        pack_reconfig_l1_acc(0);
        mS.push_back(1);

        reconfig_data_format(cb_pa, cb_mw);
        mul_bcast_cols_init(cb_pa, cb_mw);
        mA.reserve_back(1);
        configure_single_tile_pack(cb_mA);
        for (uint32_t j = 0; j < U; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_pa, cb_mw, j, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(j > 0 ? 1 : 0);
            pack_tile<true>(0, cb_mA, 0);
            tile_regs_release();
        }
        pack_reconfig_l1_acc(0);
        mA.push_back(1);
        mS.wait_front(1);
        mA.wait_front(1);

        // 1 / (S @ col_identity + w_sink) on the first column
        mrecip.reserve_back(1);
        reconfig_data_format(cb_mS, cb_col_identity);
        matmul_block_init(cb_mS, cb_col_identity, 0, 1, 1, 1);
        tile_regs_acquire();
        matmul_block(cb_mS, cb_col_identity, 0, 0, 0, 0, 1, 1, 1);
        reconfig_data_format_srca(cb_mws);
        copy_init(cb_mws);
        copy_tile(cb_mws, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        recip_tile_init();
        MATH((recip_tile(0, VectorMode::C)));
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(cb_mrecip);
        pack_tile(0, cb_mrecip);
        tile_regs_release();
        mrecip.push_back(1);
        mrecip.wait_front(1);

        // out = A (x) (1 / l) into cb_mM, untilized for the writer
        mM.reserve_back(1);
        reconfig_data_format(cb_mA, cb_mrecip);
        mul_bcast_cols_init(cb_mA, cb_mrecip);
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_mA, cb_mrecip, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        configure_single_tile_pack(cb_mM);
        pack_tile(0, cb_mM);
        tile_regs_release();
        mM.push_back(1);
        compute_kernel_lib::untilize<1, cb_mM, cb_o_rm>(1);

        pm.pop_front(U);
        ps.pop_front(U);
        pa.pop_front(U);
        mw.pop_front(U);
        mws.pop_front(1);
        mS.pop_front(1);
        mA.pop_front(1);
        mrecip.pop_front(1);
    }
}
