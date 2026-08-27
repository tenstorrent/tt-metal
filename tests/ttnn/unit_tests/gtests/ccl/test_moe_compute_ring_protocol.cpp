// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_ring_common.h"

namespace {

using Ring = std::vector<std::vector<uint32_t>>;
using Address = std::pair<uint32_t, uint32_t>;

struct Transfer {
    Address source;
    Address destination;
};

struct CreditCounters {
    uint32_t signaled = 0;
    uint32_t waited = 0;
};

std::string read_repo_source(const std::filesystem::path& relative_source) {
    std::filesystem::path root = std::filesystem::current_path();
    for (uint32_t depth = 0; depth < 8; ++depth) {
        const auto source = root / relative_source;
        if (std::filesystem::exists(source)) {
            std::ifstream input(source);
            if (!input.good()) {
                ADD_FAILURE() << "could not read " << source;
                return {};
            }
            std::ostringstream contents;
            contents << input.rdbuf();
            return contents.str();
        }
        root = root.parent_path();
    }
    ADD_FAILURE() << "could not locate " << relative_source;
    return {};
}

uint32_t count_occurrences(const std::string& source, const std::string& needle) {
    uint32_t count = 0;
    for (std::size_t pos = 0; (pos = source.find(needle, pos)) != std::string::npos; pos += needle.size()) {
        ++count;
    }
    return count;
}

bool has_source_destination_overlap(const std::vector<Transfer>& transfers) {
    std::set<Address> sources;
    std::set<Address> destinations;
    for (const auto& transfer : transfers) {
        sources.insert(transfer.source);
        destinations.insert(transfer.destination);
    }
    for (const auto& source : sources) {
        if (destinations.contains(source)) {
            return true;
        }
    }
    return false;
}

// Model an adversarial NoC that may overwrite every destination before any
// source read completes. Payloads are not snapshotted until after the address
// sets are proven disjoint and destinations are poisoned.
void apply_adversarial_queued_transfers(Ring& ring, const std::vector<Transfer>& transfers) {
    ASSERT_FALSE(has_source_destination_overlap(transfers));

    constexpr uint32_t poison = UINT32_MAX - 1;
    for (const auto& transfer : transfers) {
        ring[transfer.destination.first][transfer.destination.second] = poison;
    }

    std::vector<uint32_t> payloads;
    payloads.reserve(transfers.size());
    for (const auto& transfer : transfers) {
        payloads.push_back(ring[transfer.source.first][transfer.source.second]);
        ASSERT_NE(payloads.back(), poison);
    }
    for (uint32_t i = 0; i < transfers.size(); ++i) {
        ring[transfers[i].destination.first][transfers[i].destination.second] = payloads[i];
    }
}

std::vector<Transfer> make_hop(uint32_t ring_cores, uint32_t source_slot, uint32_t destination_slot) {
    std::vector<Transfer> transfers;
    transfers.reserve(ring_cores);
    for (uint32_t core = 0; core < ring_cores; ++core) {
        transfers.push_back({{core, source_slot}, {(core + 1) % ring_cores, destination_slot}});
    }
    return transfers;
}

void run_chunk(uint32_t ring_cores, uint32_t passes, CreditCounters& credits) {
    const uint32_t slots = ring_cores - 1;
    Ring ring(ring_cores, std::vector<uint32_t>(slots, UINT32_MAX));
    for (uint32_t core = 0; core < ring_cores; ++core) {
        ring[core][0] = core;
    }

    uint32_t start_slot = 0;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        EXPECT_EQ(start_slot, moe_ring::pass_start_slot(pass, slots));
        const bool rotate = moe_ring::requires_interpass_rotation(pass, passes);
        for (uint32_t step = 0; step < ring_cores; ++step) {
            const uint32_t source_slot = moe_ring::physical_slot(start_slot, step, slots);
            for (uint32_t core = 0; core < ring_cores; ++core) {
                EXPECT_EQ(ring[core][source_slot], (core + ring_cores - step) % ring_cores);
            }

            // These credits are emitted only after compute's CB reservation
            // proves the corresponding destination slot was consumed.
            if (step == 1) {
                ++credits.signaled;
            }
            if (step == 2 && rotate) {
                ++credits.signaled;
            }

            if (step + 1 < ring_cores) {
                const uint32_t destination_slot = moe_ring::physical_slot(start_slot, step + 1, slots);
                if (destination_slot == start_slot) {
                    ASSERT_GT(credits.signaled, credits.waited);
                    ++credits.waited;
                }
                apply_adversarial_queued_transfers(ring, make_hop(ring_cores, source_slot, destination_slot));
            }
        }

        if (rotate) {
            const uint32_t next_start_slot = moe_ring::physical_slot(start_slot, 1, slots);
            ASSERT_GT(credits.signaled, credits.waited);
            ++credits.waited;
            apply_adversarial_queued_transfers(ring, make_hop(ring_cores, start_slot, next_start_slot));
            start_slot = next_start_slot;
            for (uint32_t core = 0; core < ring_cores; ++core) {
                EXPECT_EQ(ring[core][start_slot], core);
            }
        }
    }
    EXPECT_EQ(credits.signaled, credits.waited);
}

TEST(MoEComputeRingProtocol, RotatedStartIsCorrectForMultipleRingAndPassCounts) {
    for (const uint32_t ring_cores : {8u, 12u, 16u}) {
        for (const uint32_t passes : {1u, 2u, 3u, 5u}) {
            CreditCounters credits;
            run_chunk(ring_cores, passes, credits);
        }
    }
}

TEST(MoEComputeRingProtocol, CreditCountersCarryAcrossChunksWithoutDrift) {
    for (const uint32_t ring_cores : {8u, 12u, 16u}) {
        CreditCounters credits;
        for (const uint32_t passes : {1u, 2u, 4u, 3u, 5u}) {
            run_chunk(ring_cores, passes, credits);
        }
        EXPECT_GT(credits.signaled, 0u);
        EXPECT_EQ(credits.signaled, credits.waited);
    }
}

TEST(MoEComputeRingProtocol, InPlaceReturnHasTheUnsafeDmaAddressRace) {
    for (const uint32_t ring_cores : {8u, 12u, 16u}) {
        EXPECT_TRUE(has_source_destination_overlap(make_hop(ring_cores, 0, 0)));
        EXPECT_FALSE(has_source_destination_overlap(make_hop(ring_cores, 0, 1)));
    }
}

TEST(MoEComputeRingProtocol, CompactReaderWrapsInsteadOfAddressingPastTheBuffer) {
    for (const uint32_t ring_cores : {8u, 12u, 16u}) {
        const uint32_t slots = ring_cores - 1;
        for (uint32_t pass = 0; pass < 5; ++pass) {
            const uint32_t start_slot = moe_ring::pass_start_slot(pass, slots);
            for (uint32_t step = 0; step < ring_cores; ++step) {
                EXPECT_LT(moe_ring::physical_slot(start_slot, step, slots), slots);
            }
        }
        EXPECT_FALSE((ring_cores - 1) < slots);  // old linear final-step index is out of bounds
    }
}

TEST(MoEComputeRingProtocol, RingKernelDoesNotMixPersistentInlineAndAtomicSemaphoreState) {
    const std::string source =
        read_repo_source("ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm1.cpp");
    ASSERT_FALSE(source.empty());

    // noc_semaphore_inc uses the atomic command buffer. A persistent inline
    // state on that same buffer is invalid after any intervening credit.
    EXPECT_EQ(source.find("noc_inline_dw_write_set_state"), std::string::npos);
    EXPECT_EQ(source.find("noc_inline_dw_write_with_state"), std::string::npos);
    EXPECT_NE(source.find("noc_semaphore_inc</*posted=*/true>"), std::string::npos);
    EXPECT_NE(source.find("noc_semaphore_inc</*posted=*/false>"), std::string::npos);
}

TEST(MoEComputeRingProtocol, CumulativeCombineBarriersAcceptSemaphoreOvershoot) {
    const std::string source =
        read_repo_source("ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm1.cpp");
    ASSERT_FALSE(source.empty());

    // The combine semaphore is cumulative across expert chunks. Multiple NoC
    // increments may land before a waiter observes the value, so exact waits
    // can strand EPD2 forever after an overshoot. All three barriers (empty
    // expert, first chunk/double-buffer reuse, and final reset) must accept a
    // value at least as large as the cumulative threshold.
    EXPECT_EQ(count_occurrences(source, "combine_sem.wait_min(combine_semaphore_val)"), 3u);
    EXPECT_EQ(count_occurrences(source, "combine_sem.wait(combine_semaphore_val)"), 0u);
}

TEST(MoEComputeRingProtocol, RingPayloadIsAcknowledgedBeforeEveryReadySignal) {
    const std::string source =
        read_repo_source("ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm1.cpp");
    ASSERT_FALSE(source.empty());
    const auto ring_begin = source.find("// Ring synchronization:");
    const auto ring_end = source.find("uint32_t width_tiles_to_send", ring_begin);
    ASSERT_NE(ring_begin, std::string::npos);
    ASSERT_NE(ring_end, std::string::npos);
    const std::string ring = source.substr(ring_begin, ring_end - ring_begin);

    EXPECT_EQ(ring.find("noc_async_write_one_packet_set_state</*posted=*/true>"), std::string::npos);
    EXPECT_EQ(ring.find("noc_async_write_one_packet_with_state</*posted=*/true>"), std::string::npos);
    EXPECT_GE(count_occurrences(ring, "noc_async_write_one_packet_with_state</*posted=*/false>"), 2u);

    const std::string ready = "noc_semaphore_inc</*posted=*/true>";
    std::size_t previous_ready = 0;
    std::size_t ready_pos = 0;
    uint32_t ready_count = 0;
    while ((ready_pos = ring.find(ready, ready_pos)) != std::string::npos) {
        const auto barrier_pos = ring.rfind("noc1_obj.async_write_barrier();", ready_pos);
        ASSERT_NE(barrier_pos, std::string::npos);
        EXPECT_GE(barrier_pos, previous_ready);
        previous_ready = ready_pos;
        ready_pos += ready.size();
        ++ready_count;
    }
    EXPECT_EQ(ready_count, 2u);
}

TEST(MoEComputeRingProtocol, ComputeKernelImplementsRotatedStartAndPhysicalWrap) {
    const std::string source =
        read_repo_source("ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp");
    ASSERT_FALSE(source.empty());

    EXPECT_NE(source.find("moe_ring::pass_start_slot(iter, num_a2a_buffer_slots) * tiles_per_step"), std::string::npos);
    EXPECT_NE(
        source.find("if (in2_offset == num_a2a_buffer_slots * tiles_per_step) {\n"
                    "                                in2_offset = 0;\n"
                    "                            }"),
        std::string::npos);
}

}  // namespace
