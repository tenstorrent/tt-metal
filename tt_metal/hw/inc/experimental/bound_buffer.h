// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"

namespace experimental {

/**
 * @brief Kernel-side wrapper that reads its DRAM buffer descriptor
 * (address, bank_id, num_tiles, per-tile DRAM stride) from a fixed runtime-arg
 * slot populated by BindBufferToKernel on the host.
 *
 * The host-side BindBufferToKernel packs 4 runtime args at `slot..slot+3`:
 *     arg[slot+0] = base address of buffer in DRAM
 *     arg[slot+1] = bank_id (currently 0; reserved for future interleaved support)
 *     arg[slot+2] = num_tiles
 *     arg[slot+3] = per-tile DRAM stride (= buffer.size() / num_tiles)
 *
 * Kernel devs use this instead of computing strides themselves:
 *     experimental::BoundBuffer<AllocatorBankType::DRAM> src(0);
 *     for (uint32_t i = 0; i < src.num_tiles(); ++i) {
 *         noc.async_read(src.bank(), dfb,
 *                        {.bank_id = src.bank_id(), .addr = src.tile_addr(i)}, {});
 *     }
 *
 * The DRAM page-alignment vs. native-tile-size asymmetry that previously
 * forced kernel devs to thread an explicit `dram_page_stride` runtime arg is
 * fully hidden: the host computes the right stride from buffer.size(), and
 * the kernel just walks tiles.
 *
 * @note Experimental; subject to change as the host API redesign lands.
 */
template <AllocatorBankType bank_type>
class BoundBuffer {
public:
    explicit FORCE_INLINE BoundBuffer(uint32_t arg_slot = 0) :
        base_addr_(get_arg_val<uint32_t>(arg_slot + 0)),
        bank_id_(get_arg_val<uint32_t>(arg_slot + 1)),
        num_tiles_(get_arg_val<uint32_t>(arg_slot + 2)),
        per_tile_stride_(get_arg_val<uint32_t>(arg_slot + 3)) {}

    FORCE_INLINE uint32_t num_tiles() const { return num_tiles_; }
    FORCE_INLINE uint32_t bank_id() const { return bank_id_; }
    FORCE_INLINE uint32_t base_addr() const { return base_addr_; }
    FORCE_INLINE uint32_t stride() const { return per_tile_stride_; }
    FORCE_INLINE uint32_t tile_addr(uint32_t tile_index) const { return base_addr_ + tile_index * per_tile_stride_; }
    FORCE_INLINE AllocatorBank<bank_type> bank() const { return AllocatorBank<bank_type>{}; }

    // First runtime-arg index *after* this binding's reserved slots. Use this
    // when chaining multiple BoundBuffers or computing where user extra-args
    // start. For a single binding at slot 0, user extras live at index 4+.
    static constexpr uint32_t kArgCount = 4;

private:
    uint32_t base_addr_;
    uint32_t bank_id_;
    uint32_t num_tiles_;
    uint32_t per_tile_stride_;
};

}  // namespace experimental
