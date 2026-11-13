// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What the GlobalCircularBuffer test kernels share: the page-size check, the front wait and the resize result.

#pragma once

#include <stdint.h>

#include "cce_gcb_receiver_abi.h"
#include "experimental/cce_hart.h"
#include "experimental/cce_remote_cb.h"

namespace cce_gcb {

namespace abi = cce_gcb_receiver_abi;
namespace hart = experimental::cce_hart;
namespace remote_cb = experimental::cce_remote_cb;

enum class FrontWait : uint32_t { Ready, Stopped, Expired };

inline FrontWait wait_front_or_stop(
    const remote_cb::Receiver& receiver, uint32_t num_pages, const volatile uint32_t* command, uint64_t budget_cycles) {
    const hart::WaitDeadline deadline(budget_cycles);
    while (!remote_cb::front_ready(receiver, num_pages)) {
        if (*command == abi::kCommandStop) {
            return FrontWait::Stopped;
        }
        if (deadline.expired()) {
            return FrontWait::Expired;
        }
    }
    return FrontWait::Ready;
}

inline bool gcb_page_bytes_valid(uint32_t page_bytes, uint32_t fifo_size) {
    return page_bytes != 0 && page_bytes % remote_cb::kCreditUnitBytes == 0 && page_bytes <= fifo_size;
}

inline uint32_t code_for(FrontWait wait) {
    return wait == FrontWait::Stopped ? abi::kCodeStopped : abi::kCodeCreditTimeout;
}

// kCodeOk once rd_ptr is at the new page size's boundary; kCodeStopped or kCodeCreditTimeout if the padding wait ended.
inline uint32_t resize_or_stop(
    remote_cb::Receiver& receiver, uint32_t page_bytes, const volatile uint32_t* command, uint64_t budget_cycles) {
    const remote_cb::WaitResult resized = remote_cb::resize_receiver_bounded(
        receiver, page_bytes, budget_cycles, [command]() { return *command == abi::kCommandStop; });
    if (resized == remote_cb::WaitResult::Stopped) {
        return abi::kCodeStopped;
    }
    return resized == remote_cb::WaitResult::Expired ? abi::kCodeCreditTimeout : abi::kCodeOk;
}

}  // namespace cce_gcb
