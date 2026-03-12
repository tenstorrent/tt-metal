// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_custom_mm_compressed_common.h"

namespace compressed {

/**
 * @brief Runtime loop reading per-pair tile info from an L1 tensor.
 *
 * fmt_l1_addr: byte address of uint32 array in L1, two uint32s per pair.
 *   Each uint32: [addr:24 | fmt:8]
 *     addr = THCON-shifted address (absolute when RELATIVE_ADDR=false,
 *            relative to subblock start when RELATIVE_ADDR=true)
 *     fmt  = DataFormat value for unpacker reconfig
 *   Layout: {info0, info1, info0, info1, ...} (pairs of tiles)
 *
 * When RELATIVE_ADDR=true (DRAM streaming), tile addresses are relative offsets
 * within the subblock. The kernel computes absolute addresses as addr_in1 + offset.
 * Zero tiles use ZERO_TILE_SENTINEL (0xFFFFFF) and resolve to ZEROS_ADDR_SHIFTED.
 *
 * Hot loop per tile: one L1 load (uint32), one mask, one shift, two cfg stores.
 * Zero tiles have abs_addr = ZEROS_ADDR_SHIFTED, precomputed by host.
 * No per-tile arithmetic, no branches (when RELATIVE_ADDR=false).
 */
template <uint32_t KT_DIM, uint32_t CT_DIM, bool FINALIZE = true, bool RELATIVE_ADDR = false>
FORCE_INLINE void custom_mm_compressed_block_runtime(
    uint32_t fmt_l1_addr, uint32_t addr_in0, uint32_t addr_in1, uint32_t in0_face_r_dim, uint32_t dst_index) {
    static_assert(CT_DIM > 0, "CT_DIM must be > 0");
    static_assert(CT_DIM == 1 || CT_DIM % 2 == 0, "CT_DIM must be 1 or even");
    static_assert(
        CT_DIM == 1 || KT_DIM % 2 == 0 || CT_DIM % 2 == 0, "ct=1 requires even KT_DIM; ct>1 requires even CT_DIM");

    // Sentinel for zero tiles in relative-address mode (must match host-side _ZERO_TILE_SENTINEL)
    constexpr uint32_t ZERO_TILE_SENTINEL = 0xFFFFFF;

    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();
        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;

        // Per-pair tile info: two uint32s loaded together.
        // Each uint32: [addr:24 | fmt:8], precomputed by host.
        union TileInfo {
            uint32_t packed;
            struct {
                uint8_t fmt;         // bits [7:0]  — DataFormat value
                uint32_t addr : 24;  // bits [31:8] — THCON-shifted address
            };
        };

        const volatile TileInfo* tile_ptr = reinterpret_cast<const volatile TileInfo*>(fmt_l1_addr);

        wait_for_next_context(1);
        reset_config_context();

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = addr_in0;
        TT_MOP(0, (KT_DIM / 2) - 1, 0);

        if constexpr (CT_DIM == 1) {
            constexpr uint32_t num_pairs = KT_DIM / 2;
            for (uint32_t pair = 0; pair < num_pairs; pair++) {
                TileInfo t0, t1;
                t0.packed = tile_ptr[pair * 2].packed;
                t1.packed = tile_ptr[pair * 2 + 1].packed;

                uint32_t addr0 = t0.addr;
                uint32_t addr1 = t1.addr;
                if constexpr (RELATIVE_ADDR) {
                    addr0 = (addr0 == ZERO_TILE_SENTINEL) ? ZEROS_ADDR_SHIFTED : (addr_in1 + addr0);
                    addr1 = (addr1 == ZERO_TILE_SENTINEL) ? ZEROS_ADDR_SHIFTED : (addr_in1 + addr1);
                }

                wait_for_next_context(2);
                reconfig_custom_mm_srca_input_only(cfg, t0.fmt, reg0_base);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = addr0;
                semaphore_post(semaphore::UNPACK_SYNC);

                wait_for_next_context(2);
                reconfig_custom_mm_srca_input_only(cfg, t1.fmt, reg0_base);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = addr1;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
        } else {
            // ct>1 path: tiles stored in row-major order (k major, ct minor).
            uint32_t tile_idx = 0;
            for (uint32_t k = 0; k < KT_DIM; k++) {
                constexpr uint32_t pairs_per_row = CT_DIM / 2;
                for (uint32_t ct_pair = 0; ct_pair < pairs_per_row; ct_pair++) {
                    TileInfo t0, t1;
                    t0.packed = tile_ptr[tile_idx].packed;
                    t1.packed = tile_ptr[tile_idx + 1].packed;
                    tile_idx += 2;

                    uint32_t addr0 = t0.addr;
                    uint32_t addr1 = t1.addr;
                    if constexpr (RELATIVE_ADDR) {
                        addr0 = (addr0 == ZERO_TILE_SENTINEL) ? ZEROS_ADDR_SHIFTED : (addr_in1 + addr0);
                        addr1 = (addr1 == ZERO_TILE_SENTINEL) ? ZEROS_ADDR_SHIFTED : (addr_in1 + addr1);
                    }

                    wait_for_next_context(2);
                    reconfig_custom_mm_srca_input_only(cfg, t0.fmt, reg0_base);
                    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = addr0;
                    semaphore_post(semaphore::UNPACK_SYNC);

                    wait_for_next_context(2);
                    reconfig_custom_mm_srca_input_only(cfg, t1.fmt, reg0_base);
                    cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = addr1;
                    semaphore_post(semaphore::UNPACK_SYNC);
                }
            }
        }

        // Reset counters so subsequent subblock calls start clean (matches _llk_unpack_AB_custom_mm_run_)
        wait_for_next_context(1);
        reset_config_context();
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
    }));
    MATH((_llk_math_custom_mm_<FINALIZE>(in0_face_r_dim, dst_index, KT_DIM, CT_DIM)));
}

}  // namespace compressed
