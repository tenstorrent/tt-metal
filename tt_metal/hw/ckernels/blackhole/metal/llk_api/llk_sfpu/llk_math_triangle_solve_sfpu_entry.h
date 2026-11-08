// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/tt-1xx/cache.h"
#include "internal/tt-1xx/risc_common.h"
#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sanitizer/api.h"
#include "sfpu/ckernel_sfpu_triangle_solve.h"
#include "llk_triangle_solve_sync.h"

namespace ckernel {

// The state the firmware gives this RISC's L1 data cache at boot (configure_l1_data_cache).
#if defined(ENABLE_L1_DATA_CACHE)
constexpr bool kL1DataCacheOnAtEntry = true;
#else
constexpr bool kL1DataCacheOnAtEntry = false;
#endif

/**
 * @brief Program the SFPU for the triangle solve: ADDR_MOD_7 for its SFPLOAD/SFPSTOREs, the default SFPU config
 * register, and reset RWCs.
 *
 * @note Call before the first @ref llk_math_triangle_solve_sfpu_tile of a kernel section; the solve keeps no other
 * state.
 */
inline void llk_math_triangle_solve_sfpu_init() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_binary_sfpu_init<SfpuType::triangle_solve>();
}

/**
 * @brief Solve L X = DST[idst_in] into DST[idst_out] for one 32x32 tile, L unit lower-triangular and read in place from
 * L1.
 *
 * @tparam L_FORMAT: Format of the L tile in L1, values = <Float32/Float16_b>
 * @tparam L_NEGATED: The tile holds -L below the diagonal
 * @tparam L_CACHED: Read L through this RISC's L1 data cache; false reads it uncached
 * @param l1_base: L1 byte address of the L tile, resident until this function returns.
 * @param idst_in: DEST tile index of the right-hand side.
 * @param idst_out: DEST tile index that receives X; must differ from idst_in.
 * @note Call @ref llk_math_triangle_solve_sfpu_init before this function. The L1 reads of L are issued by this RISC and
 *       complete before the function returns; the SFPU instructions still in flight touch only DEST and the LREGs.
 *       The function releases the L tile to UNPACK through the MATH -> UNPACK mailbox (@ref
 *       llk_math_triangle_solve_release_l); the UNPACK thread must consume that release with
 *       @ref llk_unpack_triangle_solve_wait_l_released before it pops or overwrites the tile.
 * @note The L1 data cache is switched to L_CACHED for the solve when that differs from the state at boot, invalidated
 *       (a fence) before the reads of L, and switched back at the end. The cache is not flushed: it is write-through
 *       and L is only read.
 */
template <DataFormat L_FORMAT, bool L_NEGATED, bool L_CACHED = true>
inline void llk_math_triangle_solve_sfpu_tile(
    const std::uint32_t l1_base, const std::uint32_t idst_in, const std::uint32_t idst_out) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_sfpu_assert_dst_index_<DST_SYNC_MODE>(idst_in, "triangle_solve_tile: idst_in out of range");
    _llk_math_eltwise_sfpu_assert_dst_index_<DST_SYNC_MODE>(idst_out, "triangle_solve_tile: idst_out out of range");
    LLK_ASSERT(idst_in != idst_out, "triangle_solve_tile: idst_out must differ from idst_in");
    if constexpr (L_CACHED != kL1DataCacheOnAtEntry) {
        set_l1_data_cache<L_CACHED>();
    }
    // With the cache on, the RISC reads L through its write-through L1 cache, which may still hold this address from
    // an earlier tile written by the packer or the NoC.
    invalidate_l1_cache();
    // The solve addresses DEST absolutely (tile index * rows per tile), so the DEST base is 0.
    _llk_math_eltwise_sfpu_start_(0 /*dst_index*/);
    sfpu::_triangle_solve_tile_<L_FORMAT, L_NEGATED>(idst_in, idst_out, l1_base);
    if constexpr (L_CACHED != kL1DataCacheOnAtEntry) {
        set_l1_data_cache<kL1DataCacheOnAtEntry>();
        invalidate_l1_cache();
    }
    llk_math_triangle_solve_release_l();  // every L1 load of L has returned: UNPACK may pop the tile
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace ckernel
