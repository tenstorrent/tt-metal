// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher reader. Fast dispatch runs it on a dispatch engine beside the CQ kernels, paired with
// tensor_prefetcher_pusher.cpp: the reader stages weight blocks from DRAM into this core's L1 and the
// pusher sends staged blocks on to the receivers. Its loop is tensor_prefetcher_reader.h; this file is the
// DRAM block source. Its own state (TENSOR_PREFETCHER_STATE_BASE) holds a group selector's mask. Defines:
// tensor_prefetcher_fifo_device.h.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_reader.h"

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;
using tensor_prefetcher::kernel_state_base;
using tensor_prefetcher::kernel_state_size;
using tensor_prefetcher::LanePage;
using tensor_prefetcher::region_ptr;
using tensor_prefetcher::TensorStream;
using tt::tt_metal::TensorPrefetcherEntry;
using tt::tt_metal::TensorPrefetcherTensorLayout;

namespace {

// Reads each receiver's blocks from its slab in its lane's DRAM bank. Weights are receiver-contiguous, so
// block `b` of a group is `b` blocks past the start of that group in the receiver's slab, and every block is
// ready. A grouped weight's slab stacks its groups, and the reader stages the blocks of every selected group
// in ascending group order.
class DramBlockSource {
public:
    uint32_t begin_tensor(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, const TensorStream& stream) {
        block_bytes_ = stream.block_bytes;
        block_count_ = stream.block_count;
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            const uint32_t lane = stream.lane[i];
            const LanePage page(slot, lane);
            volatile tt_l1_ptr TensorPrefetcherEntry* entry = page.entry(e);
            volatile tt_l1_ptr TensorPrefetcherTensorLayout* layout = page.layout_of(entry);
            bank_[i] = slot->lane_bank_id[lane];
            slab_base_[i] = entry->bank_local_base +
                            (layout->recv_index_base + stream.receiver_in_lane[i]) * layout->recv_stride_bytes;
        }
        // Every lane names the same groups and the same selector.
        const LanePage page(slot, 0);
        volatile tt_l1_ptr TensorPrefetcherEntry* entry = page.entry(e);
        group_stride_ = page.layout_of(entry)->group_stride_bytes;
        return select_groups(entry) * block_count_;
    }
    bool ready(uint32_t /*i*/, uint32_t /*block*/) const { return true; }
    void read(uint32_t i, uint32_t block, uint32_t dst) const {
        const uint32_t selected = block / block_count_;
        const uint32_t group_block = block - selected * block_count_;
        noc_async_read(
            get_noc_addr_from_bank_id<true>(
                bank_[i], slab_base_[i] + group(selected) * group_stride_ + group_block * block_bytes_),
            dst,
            block_bytes_);
    }
    void release(uint32_t /*i*/, uint32_t /*block*/) {}

private:
    // Picks the tensor's groups -- every group, or with a selector those whose mask word is non-zero -- and
    // returns how many. The selector is read once here, so every kernel after the reader sends the same groups
    // even if the mask changes before they do; the reader keeps the selected groups' indices, ascending, in
    // its own state.
    uint32_t select_groups(volatile tt_l1_ptr TensorPrefetcherEntry* entry) {
        const uint32_t num_groups = entry->num_groups;
        listed_ = entry->selector_mode == tt::tt_metal::TENSOR_PREFETCHER_SELECTOR_NONZERO_MASK;
        if (!listed_) {
            return num_groups;
        }
        const uint32_t read_bytes =
            (num_groups * sizeof(uint16_t) + fifo::kLineBytes - 1) / fifo::kLineBytes * fifo::kLineBytes;
        ASSERT(read_bytes <= kernel_state_size);
        const uint32_t xy = entry->selector_noc_xy;
        const uint64_t mask_addr = entry->selector_location == tt::tt_metal::TENSOR_PREFETCHER_SELECTOR_IN_DRAM_BANK
                                       ? get_noc_addr_from_bank_id<true>(xy, entry->selector_addr)
                                       : get_noc_addr(xy & 0xFFFFu, xy >> 16, entry->selector_addr);
        noc_async_read(mask_addr, kernel_state_base, read_bytes);
        noc_async_read_barrier();
        // The mask's words become the list of selected groups in place: group g is listed at a position no
        // later than g.
        volatile tt_l1_ptr uint16_t* words = selected_groups();
        uint32_t num_selected = 0;
        for (uint32_t g = 0; g < num_groups; ++g) {
            if (words[g] != 0) {
                words[num_selected++] = g;
            }
        }
        return num_selected;
    }
    // The index of the `k`-th group the reader stages.
    uint32_t group(uint32_t k) const { return listed_ ? selected_groups()[k] : k; }
    static volatile tt_l1_ptr uint16_t* selected_groups() { return region_ptr<uint16_t>(kernel_state_base, 0); }

    uint32_t block_bytes_ = 0;
    uint32_t block_count_ = 0;
    uint32_t group_stride_ = 0;
    bool listed_ = false;
    uint32_t bank_[fifo::kMaxReceivers];
    uint32_t slab_base_[fifo::kMaxReceivers];
};

}  // namespace

void kernel_main() {
    DramBlockSource source;
    tensor_prefetcher::run_reader(source);
}
