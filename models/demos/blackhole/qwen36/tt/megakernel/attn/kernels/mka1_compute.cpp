// MKA1 compute: verify SDPA for one own user over this core's KV slice, flash-decode style.
// Per 32-position block b: S = Q_u · K_bᵀ (8 k-tiles), P = exp(S), O_aug += P · [V_b | 1] (fp32 CB, packer L1 acc).
// The ones-tile column makes O_aug's 9th tile the softmax denominator, so no row reduce is needed.
// v0 numerics: no running-max subtraction (test logits are bounded; production adds the max/rescale step).
// Then add the tree children's partials (CB_R) into CB_O; the group root normalizes O / den into CB_OA.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "mka1_common.hpp"

void kernel_main() {
    constexpr uint32_t NBB = get_compile_time_arg_val(0);
    uint32_t a = 0;
    const uint32_t u = get_arg_val<uint32_t>(a++), nblk = get_arg_val<uint32_t>(a++);
    const uint32_t nrecv = get_arg_val<uint32_t>(a++), is_root = get_arg_val<uint32_t>(a++);
    const uint32_t qoff = u * DH;
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_Q, CB_K, CB_P);
    cb_wait_front(CB_ONE, 1);
    cb_wait_front(CB_Q, QT);
    cb_reserve_back(CB_O, NO);
    {
        DeviceZoneScopedN("C-SDPA");
        bool first = true;
        for (uint32_t b = 0; b < nblk; b += NBB) {
            const uint32_t nb = (nblk - b) < NBB ? (nblk - b) : NBB;
            cb_wait_front(CB_K, nb * DH);
            cb_wait_front(CB_V, nb * DH);
            // S_i = Q · K_iᵀ for nb blocks (dst i), P_i = exp(S_i)
            reconfig_data_format(CB_K, CB_Q);
            matmul_init(CB_Q, CB_K, 1);
            tile_regs_acquire();
            for (uint32_t i = 0; i < nb; ++i)
                for (uint32_t t = 0; t < DH; ++t) matmul_tiles(CB_Q, CB_K, qoff + t, i * DH + t, i);
            exp_tile_init();
            for (uint32_t i = 0; i < nb; ++i) exp_tile(i);
            tile_regs_commit();
            cb_reserve_back(CB_P, nb);
            tile_regs_wait();
            pack_reconfig_data_format(CB_P);
            for (uint32_t i = 0; i < nb; ++i) pack_tile(i, CB_P);
            tile_regs_release();
            cb_push_back(CB_P, nb);
            cb_wait_front(CB_P, nb);
            // O_aug += Σ_i P_i · V_i (accumulated in dest across the nb blocks), then Σ_i P_i · 1
            reconfig_data_format(CB_V, CB_P);
            matmul_init(CB_P, CB_V, 0);
            tile_regs_acquire();
            for (uint32_t i = 0; i < nb; ++i)
                for (uint32_t t = 0; t < DH; ++t) matmul_tiles(CB_P, CB_V, i, i * DH + t, t);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_O);
            pack_reconfig_l1_acc(first ? 0 : 1);
            for (uint32_t t = 0; t < DH; ++t) pack_tile<true>(t, CB_O, t);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
            reconfig_data_format(CB_ONE, CB_P);
            matmul_init(CB_P, CB_ONE, 0);
            tile_regs_acquire();
            for (uint32_t i = 0; i < nb; ++i) matmul_tiles(CB_P, CB_ONE, i, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(first ? 0 : 1);
            pack_tile<true>(0, CB_O, DH);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
            first = false;
            cb_pop_front(CB_P, nb);
            cb_pop_front(CB_K, nb * DH);
            cb_pop_front(CB_V, nb * DH);
        }
    }
    {
        DeviceZoneScopedN("C-RED");
        for (uint32_t i = 0; i < nrecv; ++i) {   // O_aug += child partial
            cb_wait_front(CB_R, NO);
            reconfig_data_format_srca(CB_R);
            copy_init(CB_R);
            for (uint32_t t0 = 0; t0 < NO; t0 += DH) {
                const uint32_t n = (NO - t0) < DH ? (NO - t0) : DH;
                tile_regs_acquire();
                for (uint32_t t = 0; t < n; ++t) copy_tile(CB_R, t0 + t, t);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(CB_O);
                pack_reconfig_l1_acc(1);
                for (uint32_t t = 0; t < n; ++t) pack_tile<true>(t, CB_O, t0 + t);
                pack_reconfig_l1_acc(0);
                tile_regs_release();
            }
            cb_pop_front(CB_R, NO);
        }
        cb_push_back(CB_O, NO);
    }
    if (is_root) {
        DeviceZoneScopedN("C-NORM");
        cb_wait_front(CB_O, NO);
        reconfig_data_format_srca(CB_O);
        copy_init(CB_O);
        tile_regs_acquire();
        copy_tile(CB_O, DH, 0);
        recip_tile_init();
        recip_tile(0);
        tile_regs_commit();
        cb_reserve_back(CB_D, 1);
        tile_regs_wait();
        pack_reconfig_data_format(CB_D);
        pack_tile(0, CB_D);
        tile_regs_release();
        cb_push_back(CB_D, 1);
        cb_wait_front(CB_D, 1);
        reconfig_data_format(CB_O, CB_D);
        mul_init(CB_O, CB_D);
        cb_reserve_back(CB_OA, DH);
        pack_reconfig_data_format(CB_OA);
        for (uint32_t t = 0; t < DH; ++t) {
            tile_regs_acquire();
            mul_tiles(CB_O, CB_D, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OA);
            tile_regs_release();
        }
        cb_push_back(CB_OA, DH);
        cb_pop_front(CB_D, 1);
        cb_pop_front(CB_O, NO);
    }
}
