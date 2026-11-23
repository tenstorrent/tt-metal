// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The groups of a Tensor prefetcher tensor that its kernels on a dispatch engine send: every group of a grouped
// weight (an expert stack), or only those whose word in its group selector's mask is non-zero, in ascending
// order. Used by the kernel that plans each tensor (tensor_prefetcher_fifo_device.h), which reads the mask once
// and publishes how many blocks flow, so no later kernel reads it.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"

namespace tensor_prefetcher {

class GroupSelection {
public:
    // `scratch_base` is DRAM-aligned L1 of `scratch_bytes` that holds a mask, and then the selected groups'
    // indices: up to tt::tt_metal::kTensorPrefetcherSelectorScratchBytes.
    FORCE_INLINE GroupSelection(uint32_t scratch_base, uint32_t scratch_bytes) :
        scratch_base_(scratch_base), scratch_bytes_(scratch_bytes) {}

    // Selects the groups of the tensor `entry` names and returns how many there are. With a selector the mask is
    // read here, from DRAM by bank or from a NoC endpoint's L1, and its words become the list of selected groups.
    FORCE_INLINE uint32_t select(volatile tt_l1_ptr TensorPrefetcherEntry* entry) {
        const uint32_t num_groups = entry->num_groups;
        listed_ = entry->selector_mode == tt::tt_metal::TENSOR_PREFETCHER_SELECTOR_NONZERO_MASK;
        if (!listed_) {
            return num_groups;
        }
        const uint32_t read_bytes =
            (num_groups * sizeof(uint16_t) + fifo::kLineBytes - 1) / fifo::kLineBytes * fifo::kLineBytes;
        ASSERT(read_bytes <= scratch_bytes_);
        const uint32_t xy = entry->selector_noc_xy;
        const uint64_t mask_addr = entry->selector_location == tt::tt_metal::TENSOR_PREFETCHER_SELECTOR_IN_DRAM_BANK
                                       ? get_noc_addr_from_bank_id<true>(xy, entry->selector_addr)
                                       : get_noc_addr(xy & 0xFFFFu, xy >> 16, entry->selector_addr);
        noc_async_read(mask_addr, scratch_base_, read_bytes);
        noc_async_read_barrier_with_trid(kTrid);
        // Group g is listed at a position no later than g, so the mask's words become the list in place.
        volatile tt_l1_ptr uint16_t* words = listed_groups();
        uint32_t num_selected = 0;
        for (uint32_t g = 0; g < num_groups; ++g) {
            if (words[g] != 0) {
                words[num_selected++] = g;
            }
        }
        return num_selected;
    }

    // The index of the `k`-th selected group.
    FORCE_INLINE uint32_t group(uint32_t k) const { return listed_ ? listed_groups()[k] : k; }

private:
    FORCE_INLINE volatile tt_l1_ptr uint16_t* listed_groups() const { return region_ptr<uint16_t>(scratch_base_, 0); }

    uint32_t scratch_base_;
    uint32_t scratch_bytes_;
    bool listed_ = false;
};

}  // namespace tensor_prefetcher
