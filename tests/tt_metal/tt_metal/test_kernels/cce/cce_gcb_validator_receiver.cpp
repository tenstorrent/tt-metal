// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Consumes a GlobalCircularBuffer receiver's schedule one page at a time and checks every word against the generator.
// It records the first mismatch and keeps acking, so the sender completes. kCommandStop ends a page wait.
// The first mismatched word is read again after a delay: if the re-read is right, the credit overtook the data.

#include "cce_gcb_receiver_abi.h"
#include "experimental/cce_hart.h"
#include "experimental/cce_remote_cb.h"

namespace abi = cce_gcb_receiver_abi;
namespace hart = experimental::cce_hart;
namespace remote_cb = experimental::cce_remote_cb;

constexpr uint64_t kMismatchRecheckDelayCycles = 1u << 16;

struct ValidationProgress {
    uint32_t requests_done = 0;
    uint32_t pages_done = 0;
    uint32_t mismatched_words = 0;
    uint32_t first_mismatch_request = 0;
    uint32_t first_mismatch_page = 0;
    uint32_t first_mismatch_word = 0;
    uint32_t observed = 0;
    uint32_t expected = 0;
    uint32_t rechecked = 0;
};

uint32_t validate_schedule(
    remote_cb::Receiver& receiver,
    const volatile uint32_t* command,
    uint32_t seed,
    uint64_t budget_cycles,
    ValidationProgress& progress) {
    const uint32_t num_requests = get_arg_val<uint32_t>(abi::kArgNumRequests);
    if (num_requests > abi::kMaxRequests) {
        return abi::kCodeBadSchedule;
    }
    for (uint32_t request = 0; request < num_requests; ++request) {
        const uint32_t tensor_id = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestTensorId));
        const uint32_t slab = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestSlab));
        const uint32_t page_bytes = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestPageBytes));
        const uint32_t num_pages = get_arg_val<uint32_t>(abi::schedule_arg(request, abi::kRequestNumPages));
        if (page_bytes == 0 || page_bytes % remote_cb::kCreditUnitBytes != 0 || page_bytes > receiver.fifo_size) {
            return abi::kCodeBadSchedule;
        }
        const remote_cb::WaitResult resized = remote_cb::resize_receiver_bounded(
            receiver, page_bytes, budget_cycles, [command]() { return *command == abi::kCommandStop; });
        if (resized == remote_cb::WaitResult::Stopped) {
            return abi::kCodeStopped;
        }
        if (resized == remote_cb::WaitResult::Expired) {
            return abi::kCodeCreditTimeout;
        }
        const uint32_t page_words = page_bytes / sizeof(uint32_t);
        for (uint32_t page = 0; page < num_pages; ++page) {
            const hart::WaitDeadline deadline(budget_cycles);
            while (!remote_cb::front_ready(receiver, 1)) {
                if (*command == abi::kCommandStop) {
                    return abi::kCodeStopped;
                }
                if (deadline.expired()) {
                    return abi::kCodeCreditTimeout;
                }
            }
            const volatile uint32_t* words = remote_cb::front_words(receiver);
            for (uint32_t word = 0; word < page_words; ++word) {
                const uint32_t expected = abi::gcb_page_word(seed, tensor_id, slab, page, word);
                const uint32_t observed = words[word];
                if (observed == expected) {
                    continue;
                }
                if (progress.mismatched_words == 0) {
                    progress.first_mismatch_request = request;
                    progress.first_mismatch_page = page;
                    progress.first_mismatch_word = word;
                    progress.observed = observed;
                    progress.expected = expected;
                    const hart::WaitDeadline recheck_delay(kMismatchRecheckDelayCycles);
                    while (!recheck_delay.expired()) {
                    }
                    progress.rechecked = words[word];
                }
                progress.mismatched_words++;
            }
            remote_cb::pop_front(receiver, 1);
            progress.pages_done++;
        }
        progress.requests_done++;
    }
    return progress.mismatched_words == 0 ? abi::kCodeOk : abi::kCodeMismatch;
}

void kernel_main() {
    const uint32_t config_addr = get_arg_val<uint32_t>(abi::kArgGcbAddr);
    const uint32_t block_addr = get_arg_val<uint32_t>(abi::kArgBlockAddr);
    const uint64_t budget_cycles = hart::arg_u64(abi::kArgBudgetLo, abi::kArgBudgetHi);
    const uint32_t seed = get_arg_val<uint32_t>(abi::kArgSeed);
    const volatile uint32_t* command = hart::own_tl1_words(block_addr + abi::kCommandLineOffset);

    remote_cb::Receiver receiver = remote_cb::receiver_init(config_addr);
    ValidationProgress progress;
    const uint32_t code = validate_schedule(receiver, command, seed, budget_cycles, progress);
    remote_cb::store_receiver_rd_ptr(receiver);

    volatile uint32_t* status = hart::own_tl1_words(block_addr + abi::kStatusLineOffset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusRequestsDone] = progress.requests_done;
    status[abi::kStatusPagesDone] = progress.pages_done;
    status[abi::kStatusMismatchedWords] = progress.mismatched_words;
    status[abi::kStatusFirstMismatchRequest] = progress.first_mismatch_request;
    status[abi::kStatusFirstMismatchPage] = progress.first_mismatch_page;
    status[abi::kStatusFirstMismatchWord] = progress.first_mismatch_word;
    status[abi::kStatusObserved] = progress.observed;
    status[abi::kStatusExpected] = progress.expected;
    status[abi::kStatusRechecked] = progress.rechecked;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kDoneMagic;
}
