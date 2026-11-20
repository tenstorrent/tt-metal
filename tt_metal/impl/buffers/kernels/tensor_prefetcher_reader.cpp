// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher reader. Fast dispatch runs it on a dispatch engine beside the CQ kernels, paired with
// tensor_prefetcher_pusher.cpp: the reader stages weight blocks from DRAM into this core's L1 and the
// pusher sends staged blocks on to the receivers. Its loop is tensor_prefetcher_reader.h; this file is the
// DRAM block source. As the first kernel it plans each tensor, and its own state (TENSOR_PREFETCHER_STATE_BASE)
// holds a group selector's mask. Defines: tensor_prefetcher_fifo_device.h.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_groups.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_reader.h"

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;
using tensor_prefetcher::GroupSelection;
using tensor_prefetcher::kernel_state_base;
using tensor_prefetcher::kernel_state_size;
using tensor_prefetcher::LanePage;
using tensor_prefetcher::TensorStream;
using tt::tt_metal::TensorPrefetcherEntry;
using tt::tt_metal::TensorPrefetcherTensorLayout;

static_assert(tensor_prefetcher::kPlansBlocks, "The DRAM reader reads the weights' source, so it plans what flows");

namespace {

// Reads each receiver's blocks from its slab in its lane's DRAM bank. Weights are receiver-contiguous, so
// block `b` of a group is `b` blocks past the start of that group in the receiver's slab, every block is
// ready, and a run of a group's blocks is one read. A grouped weight's slab stacks its groups, and the reader stages
// the blocks of every selected group in ascending group order.
class DramBlockSource {
public:
    uint32_t plan_blocks(
        volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, const TensorStream& stream, uint32_t read_trid) {
        // Every lane names the same groups and the same selector.
        return groups_.select(LanePage(slot, 0).entry(e), read_trid) * stream.block_count;
    }
    void begin_tensor(
        volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, const TensorStream& stream, uint32_t /*blocks*/) {
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
            selected_[i] = 0;
            group_block_[i] = 0;
        }
        const LanePage page(slot, 0);
        group_stride_ = page.layout_of(page.entry(e))->group_stride_bytes;
    }
    bool ready(uint32_t /*i*/, uint32_t /*block*/) const { return true; }
    // A group's blocks are back to back in the slab, so a read runs to the end of the block's group.
    uint32_t contiguous_blocks(uint32_t i, uint32_t block, uint32_t max_blocks) const {
        ASSERT(block == selected_[i] * block_count_ + group_block_[i]);
        const uint32_t to_group_end = block_count_ - group_block_[i];
        return to_group_end < max_blocks ? to_group_end : max_blocks;
    }
    void read(uint32_t i, uint32_t block, uint32_t num_blocks, uint32_t dst) {
        ASSERT(block == selected_[i] * block_count_ + group_block_[i]);
        noc_async_read(
            get_noc_addr_from_bank_id<true>(
                bank_[i], slab_base_[i] + groups_.group(selected_[i]) * group_stride_ + group_block_[i] * block_bytes_),
            dst,
            num_blocks * block_bytes_);
        group_block_[i] += num_blocks;
        if (group_block_[i] == block_count_) {
            group_block_[i] = 0;
            ++selected_[i];
        }
    }

private:
    // The reader's own state holds a group selector's mask.
    GroupSelection groups_{kernel_state_base, kernel_state_size};
    uint32_t block_bytes_ = 0;
    uint32_t block_count_ = 0;
    uint32_t group_stride_ = 0;
    uint32_t bank_[fifo::kMaxReceivers];
    uint32_t slab_base_[fifo::kMaxReceivers];
    // Where each receiver's next read starts: which of the selected groups, and which block of it. The reader reads
    // each receiver's blocks in order, so these follow it without dividing by the group's block count.
    uint32_t selected_[fifo::kMaxReceivers];
    uint32_t group_block_[fifo::kMaxReceivers];
};

}  // namespace

void kernel_main() {
    DramBlockSource source;
    tensor_prefetcher::run_reader(source);
}
