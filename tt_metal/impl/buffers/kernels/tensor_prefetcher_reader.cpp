// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher reader. Fast dispatch runs it on a dispatch engine beside the CQ kernels, paired with
// tensor_prefetcher_pusher.cpp: the reader stages weight blocks from DRAM into this core's L1 and the
// pusher sends staged blocks on to the receivers. Its loop is tensor_prefetcher_reader.h; this file is the
// DRAM block source. Defines: tensor_prefetcher_fifo_device.h.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_reader.h"

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;
using tensor_prefetcher::LanePage;
using tensor_prefetcher::TensorStream;
using tt::tt_metal::TensorPrefetcherEntry;
using tt::tt_metal::TensorPrefetcherTensorLayout;

namespace {

// Reads each receiver's blocks from its slab in its lane's DRAM bank. Weights are receiver-contiguous, so
// block `b` of a receiver is `b` blocks past the start of its slab, and every block is ready.
class DramBlockSource {
public:
    void begin_tensor(volatile tt_l1_ptr fifo::SlotHeader* slot, uint32_t e, const TensorStream& stream) {
        block_bytes_ = stream.block_bytes;
        for (uint32_t i = 0; i < stream.num_receivers; ++i) {
            const uint32_t lane = stream.lane[i];
            const LanePage page(slot, lane);
            volatile tt_l1_ptr TensorPrefetcherEntry* entry = page.entry(e);
            volatile tt_l1_ptr TensorPrefetcherTensorLayout* layout = page.layout_of(entry);
            bank_[i] = slot->lane_bank_id[lane];
            slab_base_[i] = entry->bank_local_base +
                            (layout->recv_index_base + stream.receiver_in_lane[i]) * layout->recv_stride_bytes;
        }
    }
    bool ready(uint32_t /*i*/, uint32_t /*block*/) const { return true; }
    void read(uint32_t i, uint32_t block, uint32_t dst) const {
        noc_async_read(
            get_noc_addr_from_bank_id<true>(bank_[i], slab_base_[i] + block * block_bytes_), dst, block_bytes_);
    }
    void release(uint32_t /*i*/, uint32_t /*block*/) {}

private:
    uint32_t block_bytes_ = 0;
    uint32_t bank_[fifo::kMaxReceivers];
    uint32_t slab_base_[fifo::kMaxReceivers];
};

}  // namespace

void kernel_main() {
    DramBlockSource source;
    tensor_prefetcher::run_reader(source);
}
