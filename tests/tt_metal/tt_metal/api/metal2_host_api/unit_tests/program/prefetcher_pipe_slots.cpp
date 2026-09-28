// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe slot reservation, relay DFB creation and pipe accessor tokens in MakeProgramFromSpec.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/dataflow_buffer/prefetcher_pipe.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/prefetcher_pipe_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeFullPipeSpec;
using test_helpers::ParticipantOn;
using test_helpers::pipe_entry_size;
using test_helpers::pipe_param_name;
using test_helpers::pipe_ring_size;
using test_helpers::pipe_sender_node;
using test_helpers::PrefetcherPipeSpecTestQuasar;
using test_helpers::relay_dfb_name;

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MakeProgramReservesOneSlotPerAccessorGroup) {
    // Sender accessor + receiver accessor -> two slots. No pipe is bound; the slot carries the
    // geometry the kernels compile against.
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec(/*receiver_threads=*/2));
    const auto& impl = program.impl();
    ASSERT_EQ(impl.num_prefetcher_pipe_slots(), 2u);

    const auto* binding = impl.get_prefetcher_pipe_parameter(pipe_param_name.get());
    ASSERT_NE(binding, nullptr);
    EXPECT_EQ(binding->bound_pipe, nullptr);
    EXPECT_EQ(binding->receivers.num_cores(), 3u);
    EXPECT_EQ(binding->ring_size, pipe_ring_size);
    ASSERT_EQ(binding->slots.size(), 2u);

    uint32_t sender_slots = 0;
    uint32_t receiver_slots = 0;
    for (const auto& slot_cores : binding->slots) {
        const auto& slot = impl.get_prefetcher_pipe_slot(slot_cores.prefetcher_pipe_id);
        EXPECT_EQ(slot.ring_size, pipe_ring_size);
        EXPECT_EQ(slot.entry_size, pipe_entry_size);
        if (slot.receiver_cores.num_cores() == 0) {
            ++sender_slots;
            EXPECT_TRUE(slot_cores.sender_role);
            EXPECT_TRUE(slot_cores.cores.contains(pipe_sender_node));
            EXPECT_EQ(slot.cores.num_cores(), 1u);
            EXPECT_EQ(slot.num_credit_lanes, 1u);
            EXPECT_FALSE(slot.relay_dfb_host_id.has_value());
        } else {
            ++receiver_slots;
            EXPECT_EQ(slot.cores.num_cores(), 3u);
            EXPECT_EQ(slot.num_credit_lanes, 2u);  // receiver kernel num_threads
            EXPECT_TRUE(slot.relay_dfb_host_id.has_value());
        }
        for (const CoreCoord& core : corerange_to_cores(slot_cores.cores)) {
            const auto* participant = ParticipantOn(program, core, slot_cores.prefetcher_pipe_id);
            ASSERT_NE(participant, nullptr);
            EXPECT_EQ(participant->pipe, nullptr);
            EXPECT_EQ(participant->entry_size, pipe_entry_size);
        }
    }
    EXPECT_EQ(sender_slots, 1u);
    EXPECT_EQ(receiver_slots, 1u);
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_MakeProgramCreatesRelayDFBWithoutAddress) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec());
    const auto& impl = program.impl();
    const uint32_t relay_id = impl.get_dfb_handle(relay_dfb_name.get());
    auto relay = impl.get_dataflow_buffer(relay_id);
    ASSERT_NE(relay, nullptr);
    EXPECT_TRUE(relay->borrows_memory());
    EXPECT_TRUE(relay->config.is_relay);
    EXPECT_EQ(relay->borrowed_addr_, 0u);  // pointed at the ring when the pipe binds
    EXPECT_TRUE(impl.get_prefetcher_pipe_id_for_relay(relay_id).has_value());
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_KernelsGetOnePipeAccessorTokenPerBinding) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeFullPipeSpec());
    const auto& impl = program.impl();
    std::vector<uint8_t> slot_ids;
    for (const char* name : {"sender", "receiver"}) {
        const auto& handles = impl.get_kernel_by_spec_name(name)->prefetcher_pipe_binding_handles();
        ASSERT_EQ(handles.size(), 1u) << name;
        EXPECT_EQ(handles[0].accessor_name, "weights") << name;
        slot_ids.push_back(handles[0].prefetcher_pipe_id);
    }
    EXPECT_NE(slot_ids[0], slot_ids[1]);  // sender and receiver accessors are different slots
    EXPECT_TRUE(impl.get_kernel_by_spec_name("compute")->prefetcher_pipe_binding_handles().empty());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
