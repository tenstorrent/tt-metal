// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused router tail — writer (NCRISC, NoC1): top-8 of the token row's 288 choices on the RISC (integer keys, no FPU
// needed), the ids row -> idx [1,1,B,8] uint16 (one 16-B write), the bf16 scores row -> scores [1,1,B,288]
// (9 x 2 face-row writes of 32 B). Compile-time args: TensorAccessorArgs of idx, scores. Runtime: addresses, the row
// argument (first row | (rows - 1) << 16), then (megakernel) the signal count + NoC (x, y).
// RTAIL_WEIGHTS (the routed experts' routing weights, computed here once per row instead of on every expert reader):
// + TensorAccessorArgs of weights [1,1,B,8] bf16, SCALING_BITS, EPS_BITS (compile time), the weights address at
// runtime arg RTAIL_W_RT; per row w_j = bf16_rne(s_j * scaling / (sum_j s_j + eps)) over the 8 selected bf16 scores in
// selection order, fp32 on the RISC — fused_experts' compute_expert_ids formula (mkx/xw.cpp before MKX_RW).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/router_tail_common.hpp"

using namespace rtail;

#if defined(RTAIL_WEIGHTS)
namespace {
FORCE_INLINE float bf16_to_f32(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}
FORCE_INLINE uint16_t f32_to_bf16_rne(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return static_cast<uint16_t>(((bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u) >> 16);
}
}  // namespace
#endif

void kernel_main() {
    constexpr auto idx_args = TensorAccessorArgs<PH_CT0 + 0>();
    constexpr auto sc_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    const auto idx = TensorAccessor(idx_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto scores = TensorAccessor(sc_args, get_arg_val<uint32_t>(ph_rt0 + 1));
#if defined(RTAIL_WEIGHTS)
    constexpr auto wt_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr uint32_t SCALING_BITS = get_compile_time_arg_val(wt_args.next_compile_time_args_offset() + 0);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(wt_args.next_compile_time_args_offset() + 1);
    const auto weights = TensorAccessor(wt_args, get_arg_val<uint32_t>(ph_rt0 + RTAIL_W_RT));
#endif
    const uint32_t arg = get_arg_val<uint32_t>(ph_rt0 + 2);
    if (arg == ROW_SKIP) {
        return;  // megakernel: an owner without a token row
    }
    MK_MARK();  // tail probe 0: start
#if defined(RTAIL_RM_OUT)
    // prefill (PMK P12): program B reads ids / weights as ONE ROW_MAJOR page each ([1, C x 8], 16 B per token row): the
    // core's rows are collected past the scratch's first 64 B and stored once per tensor (512 B for 32 rows, aligned)
    const uint32_t blk_ids = get_write_ptr(CB_IDX) + 64, blk_wt = blk_ids + 32 * TOP_K * 2;
#endif
    for (uint32_t t = first_row(arg), end = first_row(arg) + row_count(arg); t < end; ++t) {
        cb_wait_front(CB_CHOICE, 1);
        MK_MARK();              // per row: choice tile ready
        invalidate_l1_cache();  // packer-written data read by this RISC
        {
            volatile tt_l1_ptr uint32_t* choice =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_CHOICE));
            uint32_t keys[TOP_K];
            uint32_t ids[TOP_K];
            uint32_t n = 0;
            for (uint32_t e = 0; e < N_EXPERTS; ++e) {
                const uint32_t k = sort_key(choice[tile_off(e >> 5, e & 31u)]);
                if (n == TOP_K && k <= keys[TOP_K - 1]) {
                    continue;  // not larger than the current 8th (ties keep the lower expert id)
                }
                uint32_t pos = n < TOP_K ? n : TOP_K - 1;
                while (pos > 0 && keys[pos - 1] < k) {  // strictly smaller move down: equal keys stay ahead
                    keys[pos] = keys[pos - 1];
                    ids[pos] = ids[pos - 1];
                    --pos;
                }
                keys[pos] = k;
                ids[pos] = e;
                if (n < TOP_K) {
                    ++n;
                }
            }
            cb_reserve_back(CB_IDX, 1);
#if defined(RTAIL_RM_OUT)
            volatile tt_l1_ptr uint16_t* out =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(blk_ids + (t - first_row(arg)) * TOP_K * 2);
#else
            volatile tt_l1_ptr uint16_t* out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_IDX));
#endif
            for (uint32_t j = 0; j < TOP_K; ++j) {
                out[j] = static_cast<uint16_t>(ids[j]);
            }
#if defined(RTAIL_RM_OUT)
            // (stored once after the rows)
#else
            noc_async_write(get_write_ptr(CB_IDX), idx.get_noc_addr(0, tile_off(t, 0) * 2), TOP_K * 2);
#endif
            cb_wait_front(CB_SCORES, 1);
#if defined(RTAIL_WEIGHTS)
            {
                // the 8 selected scores (bf16, as written to the scores row), their fp32 sum in selection order
                invalidate_l1_cache();  // packer-written scores read by this RISC
                volatile tt_l1_ptr uint16_t* sc =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CB_SCORES));
                float sv[TOP_K];
                float sum = 0.0f;
                for (uint32_t j = 0; j < TOP_K; ++j) {
                    sv[j] = bf16_to_f32(sc[tile_off(ids[j] >> 5, ids[j] & 31u)]);
                    sum += sv[j];
                }
                constexpr float scaling = __builtin_bit_cast(float, SCALING_BITS);
                constexpr float eps = __builtin_bit_cast(float, EPS_BITS);
                const float inv = scaling / (sum + eps);
#if defined(RTAIL_RM_OUT)
                volatile tt_l1_ptr uint16_t* wout =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(blk_wt + (t - first_row(arg)) * TOP_K * 2);
#else
                volatile tt_l1_ptr uint16_t* wout = out + TOP_K;  // CB_IDX's second 16 B
#endif
                for (uint32_t j = 0; j < TOP_K; ++j) {
                    wout[j] = f32_to_bf16_rne(sv[j] * inv);
                }
#if defined(RTAIL_RM_OUT)
                // (stored once after the rows)
#else
                noc_async_write(
                    get_write_ptr(CB_IDX) + TOP_K * 2, weights.get_noc_addr(0, tile_off(t, 0) * 2), TOP_K * 2);
#endif
            }
#endif
        }
#if !defined(RTAIL_RM_OUT)  // prefill (PMK P12): B reads ids + weights only, no scores row
        const uint32_t s_l1 = get_read_ptr(CB_SCORES);
        for (uint32_t r = 0; r < SCORE_TILES; ++r) {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_write(s_l1 + tile_off(r, 16 * h) * 2, scores.get_noc_addr(r, tile_off(t, 16 * h) * 2), 32);
            }
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(CB_CHOICE, 1);
        cb_pop_front(CB_SCORES, 1);
        MK_MARK();  // per row: top-8 + ids / scores written
    }
#if defined(RTAIL_RM_OUT)
    {
        const uint32_t nb = row_count(arg) * TOP_K * 2;  // the core's rows, contiguous in each one-page tensor
        noc_async_write(blk_ids, idx.get_noc_addr(0, first_row(arg) * TOP_K * 2), nb);
#if defined(RTAIL_WEIGHTS)
        noc_async_write(blk_wt, weights.get_noc_addr(0, first_row(arg) * TOP_K * 2), nb);
#endif
        noc_async_write_barrier();
    }
#endif
#if defined(MK_RTAIL_SIGNAL_SEM)
    mk::signal_handoff(MK_RTAIL_SIGNAL_SEM, ph_rt0 + 4, get_arg_val<uint32_t>(ph_rt0 + 3));  // ids / scores row landed
#endif
#if defined(RTAIL_PLAN_SIG_SEM)
    // Stage 14 B2.3 step (2): this core's ids / weights rows landed (the write barrier above) -> +1 on every early-plan
    // core (program B's plan overlapped with P13 on the idle MM-only BRISCs, b_wrap_w.cpp PMK_PLAN_EARLY): the count at
    // ph_rt0 + RTAIL_PLAN_SIG_RT, then (x, y) per plan core
    mk::signal_handoff(
        RTAIL_PLAN_SIG_SEM, ph_rt0 + RTAIL_PLAN_SIG_RT + 1, get_arg_val<uint32_t>(ph_rt0 + RTAIL_PLAN_SIG_RT));
#endif
    MK_MARK();  // signalled
#if defined(RTAIL_PROBE)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
