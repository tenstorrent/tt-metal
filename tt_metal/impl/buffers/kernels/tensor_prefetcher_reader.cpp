// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher reader. Fast dispatch runs it on a dispatch engine beside the CQ kernels, paired with
// tensor_prefetcher_pusher.cpp: the reader stages weight blocks from DRAM into this core's L1 and the
// pusher sends staged blocks on to the receivers. Both retire every request slot the host writes.
//
// Compile-time defines (set by TensorPrefetcherKernel):
//   TENSOR_PREFETCHER_REGION_BASE  L1 base of the region described in tensor_prefetcher_fifo.hpp

#include <stdint.h>

#include "api/debug/assert.h"
#include "tt_metal/impl/buffers/kernels/tensor_prefetcher_fifo_device.h"

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;

constexpr uint32_t region_base = TENSOR_PREFETCHER_REGION_BASE;

void kernel_main() {
    tensor_prefetcher::FifoConsumer requests(region_base, fifo::kReaderRdOffset);
    while (true) {
        volatile tt_l1_ptr fifo::SlotHeader* slot = requests.wait_for_slot();
        if (slot == nullptr) {
            break;
        }
        switch (slot->command) {
            case fifo::kSlotNop: break;
            default: ASSERT(false); break;
        }
        requests.retire_slot();
    }
}
