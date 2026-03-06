// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_custom_mm_compressed_common.h"

namespace compressed {

/**
 * @brief Runtime loop reading packed pairs from an L1 tensor.
 *
 * fmt_l1_addr: byte address of uint32 array in L1, one packed word per pair.
 * Each word: [sz1:8 | sz0:8 | fmt1:8 | fmt0:8]
 * No constexpr arrays, no stack usage -- scales to any K dimension.
 */
template <uint32_t KT_DIM, uint32_t CT_DIM>
FORCE_INLINE void custom_mm_compressed_block_runtime(
    uint32_t fmt_l1_addr, uint32_t addr_in0, uint32_t addr_in1, uint32_t in0_face_r_dim, uint32_t dst_index) {
    static_assert(CT_DIM > 0, "CT_DIM must be > 0");
    static_assert(
        (CT_DIM == 1 && (KT_DIM % 2 == 0)) || (CT_DIM > 1 && (CT_DIM % 2 == 0)),
        "ct=1 requires even KT_DIM; ct>1 requires even CT_DIM");

    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();
        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;

        wait_for_next_context(1);
        reset_config_context();

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = addr_in0;
        TT_MOP(0, (KT_DIM / 2) - 1, 0);

        uint32_t address_a = addr_in1;
        // Read packed pairs directly from L1 tensor
        const volatile uint32_t* fmt_ptr = reinterpret_cast<const volatile uint32_t*>(fmt_l1_addr);

        union PairInfo {
            uint32_t packed;
            struct {
                uint8_t fmt0, fmt1, sz0, sz1;
            };
        };

        if constexpr (CT_DIM == 1) {
            constexpr uint32_t num_pairs = KT_DIM / 2;
            for (uint32_t pair = 0; pair < num_pairs; pair++) {
                PairInfo p;
                p.packed = fmt_ptr[pair];  // direct L1 load

                wait_for_next_context(2);
                reconfig_custom_mm_srca_input_only(cfg, p.fmt0, reg0_base);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
                address_a += p.sz0;
                semaphore_post(semaphore::UNPACK_SYNC);

                wait_for_next_context(2);
                reconfig_custom_mm_srca_input_only(cfg, p.fmt1, reg0_base);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
                address_a += p.sz1;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
        } else {
            // ct>1 path: consume packed pairs in row-major (k major, ct pairs minor).
            constexpr uint32_t pairs_per_row = CT_DIM / 2;
            uint32_t pair_idx = 0;
            for (uint32_t k = 0; k < KT_DIM; k++) {
                for (uint32_t ct_pair = 0; ct_pair < pairs_per_row; ct_pair++) {
                    PairInfo p;
                    p.packed = fmt_ptr[pair_idx++];  // direct L1 load

                    wait_for_next_context(2);
                    reconfig_custom_mm_srca_input_only(cfg, p.fmt0, reg0_base);
                    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
                    address_a += p.sz0;
                    semaphore_post(semaphore::UNPACK_SYNC);

                    wait_for_next_context(2);
                    reconfig_custom_mm_srca_input_only(cfg, p.fmt1, reg0_base);
                    cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
                    address_a += p.sz1;
                    semaphore_post(semaphore::UNPACK_SYNC);
                }
            }
        }
    }));
    MATH((_llk_math_custom_mm_<true>(in0_face_r_dim, dst_index, KT_DIM, CT_DIM)));
}

}  // namespace compressed
