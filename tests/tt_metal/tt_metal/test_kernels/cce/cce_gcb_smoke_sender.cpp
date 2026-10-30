// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes generator pages into a GlobalCircularBuffer's receivers from its sender CCE, without the prefetcher manager:
// one page at a time, per the schedule, then waits until every receiver has acked everything.

#include "cce_gcb_receiver_abi.h"
#include "experimental/cce_hart.h"
#include "experimental/cce_remote_cb.h"

namespace abi = cce_gcb_receiver_abi;
namespace hart = experimental::cce_hart;
namespace remote_cb = experimental::cce_remote_cb;

struct SendProgress {
    uint32_t requests_done = 0;
    uint32_t pages_done = 0;
};

uint32_t send_schedule(remote_cb::Sender& sender, uint32_t seed, uint64_t budget_cycles, SendProgress& progress) {
    const uint32_t num_requests = get_arg_val<uint32_t>(abi::kArgNumRequests);
    if (num_requests > abi::kMaxRequests) {
        return abi::kCodeBadSchedule;
    }
    for (uint32_t request = 0; request < num_requests; ++request) {
        const uint32_t tensor_id = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestTensorId));
        const uint32_t slab = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestSlab));
        const uint32_t page_bytes = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestPageBytes));
        const uint32_t num_pages = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestNumPages));
        if (page_bytes == 0 || page_bytes % remote_cb::kCreditUnitBytes != 0 || page_bytes > sender.fifo_size) {
            return abi::kCodeBadSchedule;
        }
        remote_cb::resize_sender(sender, page_bytes);
        const uint32_t page_words = page_bytes / sizeof(uint32_t);
        for (uint32_t page = 0; page < num_pages; ++page) {
            if (remote_cb::reserve_back_bounded(sender, page_bytes, budget_cycles) == remote_cb::WaitResult::Expired) {
                return abi::kCodeCreditTimeout;
            }
            for (uint32_t receiver = 0; receiver < sender.num_receivers; ++receiver) {
                volatile uint32_t* destination = remote_cb::receiver_back_words(sender, receiver);
                for (uint32_t word = 0; word < page_words; ++word) {
                    destination[word] = abi::gcb_page_word(seed, tensor_id, slab + receiver, page, word);
                }
            }
            remote_cb::finalize(sender, page_bytes);
            progress.pages_done++;
        }
        progress.requests_done++;
    }
    return remote_cb::barrier_bounded(sender, budget_cycles) == remote_cb::WaitResult::Ready ? abi::kCodeOk
                                                                                             : abi::kCodeDrainTimeout;
}

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(abi::kArgGcbAddr);
    const uint32_t block_addr = get_arg_val<uint32_t>(abi::kArgBlockAddr);
    const uint64_t budget_cycles = hart::arg_u64(abi::kArgBudgetLo, abi::kArgBudgetHi);
    const uint32_t seed = get_arg_val<uint32_t>(abi::kArgSeed);

    remote_cb::Sender sender = remote_cb::sender_init(state_addr);
    SendProgress progress;
    const uint32_t code = send_schedule(sender, seed, budget_cycles, progress);
    remote_cb::store_sender_wr_ptr(sender);

    volatile uint32_t* status = hart::own_tl1_words(block_addr + abi::kStatusLineOffset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusRequestsDone] = progress.requests_done;
    status[abi::kStatusPagesDone] = progress.pages_done;
    status[abi::kStatusWaitReceiver] = sender.slowest_receiver;
    status[abi::kStatusWaitPagesSent] = remote_cb::receiver_pages_sent(sender, sender.slowest_receiver);
    status[abi::kStatusWaitPagesAcked] = remote_cb::receiver_pages_acked(sender, sender.slowest_receiver);
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kDoneMagic;
}
