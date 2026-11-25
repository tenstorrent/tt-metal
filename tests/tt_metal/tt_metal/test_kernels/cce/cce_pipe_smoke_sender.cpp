// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes generator pages into a PrefetcherPipe's receivers from its sender CCE, without the prefetcher manager:
// one page at a time, per the schedule, then waits until every receiver has acked everything.

#include "cce_pipe_common.h"
#include "cce_pipe_receiver_abi.h"
#include "cce_test_common.h"
#include "experimental/cce_hart.h"
#include "experimental/cce_prefetcher_pipe.h"

namespace abi = cce_pipe_receiver_abi;
namespace hart = experimental::cce_hart;
namespace prefetcher_pipe = experimental::cce_prefetcher_pipe;
using cce_pipe_set::pipe_page_bytes_valid;

struct SendProgress {
    uint32_t requests_done = 0;
    uint32_t pages_done = 0;
};

uint32_t send_schedule(prefetcher_pipe::Sender& sender, uint32_t seed, uint64_t budget_cycles, SendProgress& progress) {
    const uint32_t num_requests = get_arg_val<uint32_t>(abi::kArgNumRequests);
    if (num_requests > abi::kMaxRequests) {
        return abi::kCodeBadSchedule;
    }
    for (uint32_t request = 0; request < num_requests; ++request) {
        const uint32_t tensor_id = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestTensorId));
        const uint32_t slab = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestSlab));
        const uint32_t page_bytes = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestPageBytes));
        const uint32_t num_pages = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestNumPages));
        if (!pipe_page_bytes_valid(page_bytes, sender.fifo_size)) {
            return abi::kCodeBadSchedule;
        }
        prefetcher_pipe::resize_sender(sender, page_bytes);
        const uint32_t page_words = page_bytes / sizeof(uint32_t);
        for (uint32_t page = 0; page < num_pages; ++page) {
            if (prefetcher_pipe::reserve_back_bounded(sender, page_bytes, budget_cycles) ==
                prefetcher_pipe::WaitResult::Expired) {
                return abi::kCodeCreditTimeout;
            }
            for (uint32_t receiver = 0; receiver < sender.num_receivers; ++receiver) {
                volatile uint32_t* destination = prefetcher_pipe::receiver_back_words(sender, receiver);
                for (uint32_t word = 0; word < page_words; ++word) {
                    destination[word] = abi::pipe_page_word(seed, tensor_id, slab + receiver, page, word);
                }
            }
            prefetcher_pipe::finalize(sender, page_bytes);
            progress.pages_done++;
        }
        progress.requests_done++;
    }
    return prefetcher_pipe::barrier_bounded(sender, budget_cycles) == prefetcher_pipe::WaitResult::Ready
               ? abi::kCodeOk
               : abi::kCodeDrainTimeout;
}

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(abi::kArgPipeAddr);
    const uint32_t block_addr = get_arg_val<uint32_t>(abi::kArgBlockAddr);
    const uint64_t budget_cycles = hart::arg_u64(abi::kArgBudgetLo, abi::kArgBudgetHi);
    const uint32_t seed = get_arg_val<uint32_t>(abi::kArgSeed);

    prefetcher_pipe::Sender sender = prefetcher_pipe::sender_init(state_addr);
    SendProgress progress;
    const uint32_t code = send_schedule(sender, seed, budget_cycles, progress);
    prefetcher_pipe::store_sender_wr_ptr(sender);

    volatile uint32_t* status = hart::own_tl1_words(block_addr + abi::kStatusLineOffset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusRequestsDone] = progress.requests_done;
    status[abi::kStatusPagesDone] = progress.pages_done;
    status[abi::kStatusWaitReceiver] = sender.slowest_receiver;
    status[abi::kStatusWaitPagesSent] = prefetcher_pipe::receiver_pages_sent(sender, sender.slowest_receiver);
    status[abi::kStatusWaitPagesAcked] = prefetcher_pipe::receiver_pages_acked(sender, sender.slowest_receiver);
    cce_test::publish_done_magic(status, abi::kDoneMagic);
}
