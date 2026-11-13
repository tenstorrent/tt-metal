// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Consumes a GlobalCircularBuffer receiver's schedule one page at a time and checks every word against the generator.
// It records the first mismatch and keeps acking, so the sender completes. kCommandStop ends a page wait.
// The first mismatched word is read again after a delay: if the re-read is right, the credit overtook the data.

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
        if (!gcb_page_bytes_valid(page_bytes, receiver.fifo_size)) {
            return abi::kCodeBadSchedule;
        }
        if (const uint32_t code = resize_or_stop(receiver, page_bytes, command, budget_cycles); code != abi::kCodeOk) {
            return code;
        }
        const uint32_t page_words = page_bytes / sizeof(uint32_t);
        for (uint32_t page = 0; page < num_pages; ++page) {
            const FrontWait wait = wait_front_or_stop(receiver, 1, command, budget_cycles);
            if (wait != FrontWait::Ready) {
                return code_for(wait);
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
    cce_test::publish_done_magic(status, abi::kDoneMagic);
}
