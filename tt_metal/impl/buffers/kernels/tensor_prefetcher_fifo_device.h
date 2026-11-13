// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device side of the request FIFO shared by the Tensor prefetcher kernels that fast dispatch runs on a
// dispatch engine. Layout: tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp.

#include <stdint.h>

#include "internal/risc_attribs.h"
#include "tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp"

namespace tensor_prefetcher {

namespace fifo = tt::tt_metal::tensor_prefetcher_fifo;

// Every word in the region is shared with the host, the other kernel or the NoC, none of which snoop
// this core's data cache, so it is always accessed through the uncached L1 alias.
template <typename T>
FORCE_INLINE volatile tt_l1_ptr T* region_ptr(uint32_t region_base, uint32_t offset) {
#ifdef ARCH_QUASAR
    return reinterpret_cast<volatile tt_l1_ptr T*>(region_base + offset + MEM_L1_UNCACHED_BASE);
#else
    return reinterpret_cast<volatile tt_l1_ptr T*>(region_base + offset);
#endif
}

// One kernel's view of the FIFO: it retires slots in order and publishes how many it has retired.
class FifoConsumer {
public:
    FORCE_INLINE FifoConsumer(uint32_t region_base, uint32_t rd_offset) :
        region_base_(region_base),
        terminate_(region_ptr<uint32_t>(region_base, fifo::kTerminateOffset)),
        wr_(region_ptr<uint32_t>(region_base, fifo::kFifoWrOffset)),
        rd_(region_ptr<uint32_t>(region_base, rd_offset)),
        rd_count_(*rd_) {}

    FORCE_INLINE bool terminate_requested() const { return *terminate_ != 0; }

    // Waits for the next unretired slot. Returns nullptr if termination was requested first.
    FORCE_INLINE volatile tt_l1_ptr fifo::SlotHeader* wait_for_slot() const {
        while (*wr_ == rd_count_) {
            if (terminate_requested()) {
                return nullptr;
            }
        }
        return region_ptr<fifo::SlotHeader>(region_base_, fifo::slot_offset(rd_count_));
    }

    FORCE_INLINE void retire_slot() { *rd_ = ++rd_count_; }

    uint32_t retired() const { return rd_count_; }

private:
    uint32_t region_base_;
    volatile tt_l1_ptr uint32_t* terminate_;
    volatile tt_l1_ptr uint32_t* wr_;
    volatile tt_l1_ptr uint32_t* rd_;
    uint32_t rd_count_;
};

}  // namespace tensor_prefetcher
