// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Consumes a GlobalCircularBuffer receiver's pages without reading them, in one of three modes
// (cce_gcb_receiver_abi.h): drain the schedule, drain until the host stops it, or drain the schedule and then stall
// holding credits until the host releases it.

#include "cce_gcb_common.h"
#include "cce_gcb_receiver_abi.h"
#include "cce_test_common.h"
#include "experimental/cce_hart.h"
#include "experimental/cce_remote_cb.h"

namespace abi = cce_gcb_receiver_abi;
namespace hart = experimental::cce_hart;
namespace remote_cb = experimental::cce_remote_cb;
using cce_gcb::code_for;
using cce_gcb::FrontWait;
using cce_gcb::gcb_page_bytes_valid;
using cce_gcb::resize_or_stop;
using cce_gcb::wait_front_or_stop;

struct DiscardProgress {
    uint32_t requests_done = 0;
    uint32_t pages_done = 0;
};

uint32_t drain_count(
    remote_cb::Receiver& receiver,
    const volatile uint32_t* command,
    uint64_t budget_cycles,
    uint32_t num_requests,
    DiscardProgress& progress) {
    for (uint32_t request = 0; request < num_requests; ++request) {
        const uint32_t page_bytes = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestPageBytes));
        const uint32_t num_pages = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestNumPages));
        if (!gcb_page_bytes_valid(page_bytes, receiver.fifo_size)) {
            return abi::kCodeBadSchedule;
        }
        if (const uint32_t code = resize_or_stop(receiver, page_bytes, command, budget_cycles); code != abi::kCodeOk) {
            return code;
        }
        for (uint32_t page = 0; page < num_pages; ++page) {
            const FrontWait wait = wait_front_or_stop(receiver, 1, command, budget_cycles);
            if (wait != FrontWait::Ready) {
                return code_for(wait);
            }
            remote_cb::pop_front(receiver, 1);
            progress.pages_done++;
        }
        progress.requests_done++;
    }
    return abi::kCodeOk;
}

// Pages of the first request's size. The budget is per page, so the kernel exits on its own after a whole budget
// with no page and no stop.
uint32_t drain_until_stopped(
    remote_cb::Receiver& receiver,
    const volatile uint32_t* command,
    uint64_t budget_cycles,
    DiscardProgress& progress) {
    const uint32_t page_bytes = get_arg_val<uint32_t>(abi::schedule_arg(0, abi::kRequestPageBytes));
    if (!gcb_page_bytes_valid(page_bytes, receiver.fifo_size)) {
        return abi::kCodeBadSchedule;
    }
    if (const uint32_t code = resize_or_stop(receiver, page_bytes, command, budget_cycles); code != abi::kCodeOk) {
        return code;
    }
    while (true) {
        const FrontWait wait = wait_front_or_stop(receiver, 1, command, budget_cycles);
        if (wait != FrontWait::Ready) {
            return code_for(wait);
        }
        remote_cb::pop_front(receiver, 1);
        progress.pages_done++;
    }
}

// Waits for `stall_pages` pages at the current page size and holds them until the host writes kCommandStop.
uint32_t stall_until_released(
    const remote_cb::Receiver& receiver,
    const volatile uint32_t* command,
    uint64_t budget_cycles,
    uint32_t stall_pages,
    uint32_t block_addr) {
    const FrontWait wait = wait_front_or_stop(receiver, stall_pages, command, budget_cycles);
    if (wait != FrontWait::Ready) {
        return code_for(wait);
    }
    *hart::own_tl1_words(block_addr + abi::kHoldingLineOffset) = stall_pages;
    const hart::WaitDeadline deadline(budget_cycles);
    while (!deadline.expired()) {
        if (*command == abi::kCommandStop) {
            return abi::kCodeReleased;
        }
    }
    return abi::kCodeReleaseTimeout;
}

void write_status(uint32_t block_addr, uint32_t code, const DiscardProgress& progress) {
    volatile uint32_t* status = hart::own_tl1_words(block_addr + abi::kStatusLineOffset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusRequestsDone] = progress.requests_done;
    status[abi::kStatusPagesDone] = progress.pages_done;
    cce_test::publish_done_magic(status, abi::kDoneMagic);
}

void kernel_main() {
    const uint32_t config_addr = get_arg_val<uint32_t>(abi::kArgGcbAddr);
    const uint32_t block_addr = get_arg_val<uint32_t>(abi::kArgBlockAddr);
    const uint64_t budget_cycles = hart::arg_u64(abi::kArgBudgetLo, abi::kArgBudgetHi);
    const uint32_t mode = get_arg_val<uint32_t>(abi::kArgMode);
    const uint32_t stall_pages = get_arg_val<uint32_t>(abi::kArgStallPages);
    const uint32_t num_requests = get_arg_val<uint32_t>(abi::kArgNumRequests);
    const volatile uint32_t* command = hart::own_tl1_words(block_addr + abi::kCommandLineOffset);

    remote_cb::Receiver receiver = remote_cb::receiver_init(config_addr);
    DiscardProgress progress;
    uint32_t code = abi::kCodeOk;
    if (num_requests == 0 || num_requests > abi::kMaxRequests || mode > abi::kModeStallUntilReleased) {
        code = abi::kCodeBadSchedule;
    } else if (mode == abi::kModeDrainUntilStopped) {
        code = drain_until_stopped(receiver, command, budget_cycles, progress);
    } else {
        code = drain_count(receiver, command, budget_cycles, num_requests, progress);
        if (code == abi::kCodeOk && mode == abi::kModeStallUntilReleased) {
            code = stall_until_released(receiver, command, budget_cycles, stall_pages, block_addr);
        }
    }

    // A released stall leaves the GCB as the host will discard it: credits held, rd_ptr where the host last wrote it.
    if (code != abi::kCodeReleased) {
        remote_cb::store_receiver_rd_ptr(receiver);
    }
    write_status(block_addr, code, progress);
}
