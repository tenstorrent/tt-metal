// Fused MLP v11 compute, persistent over NL layers. Per layer: gate_up cores h = silu(x @ g) * (x @ u) for their
// pair; every core y = h @ Wd[:, cols]. IS_GU is runtime arg 0.
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t CB_X = 0, CB_W = 1, CB_HL = 2, CB_H = 3, CB_WD = 4, CB_W2 = 5, CB_OUT = 16;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1);
    constexpr uint32_t HT = get_compile_time_arg_val(2), NDN = get_compile_time_arg_val(3);
    constexpr uint32_t GRP = get_compile_time_arg_val(4), NL = get_compile_time_arg_val(5);
    const uint32_t IS_GU = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_W, CB_HL);
    for (uint32_t L = 0; L < NL; ++L) {
        if (IS_GU) {
            reconfig_data_format(CB_W, CB_X);
            matmul_block_init(CB_X, CB_W, 0, 2, 1, KB);
            cb_wait_front(CB_X, KT);
            DeviceZoneScopedN("C-GU");
            tile_regs_acquire();
            constexpr uint32_t NB = KT / KB;
            for (uint32_t b0 = 0, g = 0; b0 < NB; b0 += GRP, ++g) {
                const uint32_t nb = (NB - b0) < GRP ? (NB - b0) : GRP;
                const uint32_t cbw = (g & 1) ? CB_W2 : CB_W;  // even groups from NOC1 (reader), odd from NOC0 (writer)
                cb_wait_front(cbw, nb * KB * 2);
                for (uint32_t b = 0; b < nb; ++b)
                    for (uint32_t k = 0; k < KB; ++k)
                        matmul_block(CB_X, cbw, (b0 + b) * KB + k, b * KB * 2 + k * 2, 0, 0, 2, 1, KB);
                cb_pop_front(cbw, nb * KB * 2);
            }
            silu_tile_init();
            silu_tile(0);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            cb_reserve_back(CB_HL, 1);
            tile_regs_wait();
            pack_reconfig_data_format(CB_HL);
            pack_tile(0, CB_HL);
            tile_regs_release();
            cb_push_back(CB_HL, 1);
        } else {
            cb_wait_front(CB_X, KT);
        }
        cb_pop_front(CB_X, KT);
        reconfig_data_format(CB_WD, CB_H);
        matmul_block_init(CB_H, CB_WD, 0, NDN, 1, HT);
        cb_wait_front(CB_WD, HT * NDN);
        cb_wait_front(CB_H, HT);
        DeviceZoneScopedN("C-DN");
        tile_regs_acquire();
        for (uint32_t k = 0; k < HT; ++k) matmul_block(CB_H, CB_WD, k, k * NDN, 0, 0, NDN, 1, HT);
        tile_regs_commit();
        cb_reserve_back(CB_OUT, NDN);
        tile_regs_wait();
        pack_reconfig_data_format(CB_OUT);
        for (uint32_t j = 0; j < NDN; ++j) pack_tile(j, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, NDN);
        cb_pop_front(CB_WD, HT * NDN);
        cb_pop_front(CB_H, HT);
    }
}
