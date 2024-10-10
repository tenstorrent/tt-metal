// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/impl/routing/flow_control/host_flow_control_sender.hpp"
#include "tt_metal/impl/routing/flow_control/host_flow_control_receiver.hpp"

//////////////////////////////
// Flow Control Sender Tests
//////////////////////////////
TEST(FlowControlSenderHost, BasicCreditsSending_NonPowerOf2SizedQueue) {
    size_t sender_local_wrptr = 0;
    size_t sender_local_rdptr_ack = 0;
    size_t sender_local_rdptr_completions = 0;

    size_t sender_remote_wrptr = 0;
    size_t sender_remote_rdptr_ack = 0;
    size_t sender_remote_rdptr_completions = 0;

    static constexpr size_t q_capacity = 9;
    auto fcs = tt_metal::routing::flow_control::PacketizedHostFlowControlSender(
        &sender_remote_wrptr,
        q_capacity,
        RemoteQueuePtrManager(
            &sender_local_wrptr, &sender_local_rdptr_ack, &sender_local_rdptr_completions, q_capacity));

    ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
    ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);
    ASSERT_EQ(fcs.get_num_outstanding_ack_credits(), 0);
    ASSERT_EQ(fcs.local_has_free_credits(q_capacity), true);

    for (size_t i = 0; i < q_capacity; i++) {
        fcs.advance_write_credits(1);
        fcs.send_credits();

        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity - i - 1);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - i - 1);
        ASSERT_EQ(fcs.get_num_outstanding_ack_credits(), i + 1);
        ASSERT_EQ(fcs.local_has_free_credits(q_capacity - i - 1), true);

        ASSERT_EQ(sender_remote_wrptr, i + 1);
    }
    ASSERT_EQ(fcs.get_num_free_local_credits(), 0);
    ASSERT_EQ(fcs.get_num_free_remote_credits(), 0);
}

// Test FCS pushing into an empty queue
TEST(FlowControlSenderHost, BasicCreditsSending_PowerOf2SizedQueue) {
    size_t sender_local_wrptr = 0;
    size_t sender_local_rdptr_ack = 0;
    size_t sender_local_rdptr_completions = 0;

    size_t sender_remote_wrptr = 0;
    size_t sender_remote_rdptr_ack = 0;
    size_t sender_remote_rdptr_completions = 0;

    static constexpr size_t q_capacity = 8;
    auto fcs = tt_metal::routing::flow_control::PacketizedHostFlowControlSender<q_capacity>(
        &sender_remote_wrptr,
        q_capacity,
        RemoteQueuePtrManager<q_capacity>(
            &sender_local_wrptr, &sender_local_rdptr_ack, &sender_local_rdptr_completions));

    ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
    ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);
    ASSERT_EQ(fcs.get_num_outstanding_ack_credits(), 0);
    ASSERT_EQ(fcs.local_has_free_credits(q_capacity), true);

    for (size_t i = 0; i < q_capacity; i++) {
        fcs.advance_write_credits(1);
        fcs.send_credits();

        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity - i - 1);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - i - 1);
        ASSERT_EQ(fcs.get_num_outstanding_ack_credits(), i + 1);
        ASSERT_EQ(fcs.local_has_free_credits(q_capacity - i - 1), true);

        ASSERT_EQ(sender_remote_wrptr, i + 1);
    }
    ASSERT_EQ(fcs.get_num_free_local_credits(), 0);
    ASSERT_EQ(fcs.get_num_free_remote_credits(), 0);
}



//////////////////////////////
// Flow Control Receiver Tests
//////////////////////////////

// Test FCR APIs are correct for when data is "coming in"
TEST(FlowControlReceiverHost, BasicAckAndCompletionCreditsSending_NonPowerOf2SizedQueue) {
    size_t receiver_local_wrptr = 0;
    size_t receiver_local_rdptr_ack = 0;
    size_t receiver_local_rdptr_completions = 0;

    size_t receiver_remote_wrptr = 0;
    size_t receiver_remote_rdptr_ack = 0;
    size_t receiver_remote_rdptr_completions = 0;

    static constexpr size_t q_capacity = 9;
    auto fcr = tt_metal::routing::flow_control::PacketizedHostFlowControlReceiver(
        &receiver_remote_rdptr_ack,
        &receiver_remote_rdptr_completions,
        q_capacity,
        RemoteQueuePtrManager(
            &receiver_local_wrptr, &receiver_local_rdptr_ack, &receiver_local_rdptr_completions, q_capacity));

    ASSERT_EQ(fcr.get_num_unacked_credits(), 0);
    ASSERT_EQ(fcr.get_num_incompleted_credits(), 0);
    ASSERT_EQ(fcr.local_has_unacknowledged_credits(), false);
    ASSERT_EQ(fcr.local_has_incomplete_credits(), false);
    ASSERT_EQ(fcr.local_has_data(), false);
    ASSERT_EQ(fcr.local_has_free_credits(q_capacity), true);

    for (size_t i = 0; i < q_capacity; i++) {
        // "Signal" the flow control receiver that data is available
        receiver_local_wrptr = i + 1;

        ASSERT_EQ(fcr.local_has_data(), true);
        ASSERT_EQ(fcr.local_has_free_credits(q_capacity - i - 1),  true);
        ASSERT_EQ(fcr.local_has_free_credits(q_capacity - i),      false);

        ASSERT_EQ(fcr.get_num_unacked_credits(), i + 1);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), i + 1);
        ASSERT_EQ(fcr.local_has_unacknowledged_credits(), true);
        ASSERT_EQ(fcr.local_has_incomplete_credits(), true);
    }
}

// Test FCR APIs are correct for when data is "coming in"
TEST(FlowControlReceiverHost, BasicAckAndCompletionCreditsSending_PowerOf2SizedQueue) {
    size_t receiver_local_wrptr = 0;
    size_t receiver_local_rdptr_ack = 0;
    size_t receiver_local_rdptr_completions = 0;

    size_t receiver_remote_wrptr = 0;
    size_t receiver_remote_rdptr_ack = 0;
    size_t receiver_remote_rdptr_completions = 0;

    static constexpr size_t q_capacity = 8;
    auto fcr = tt_metal::routing::flow_control::PacketizedHostFlowControlReceiver(
        &receiver_remote_rdptr_ack,
        &receiver_remote_rdptr_completions,
        RemoteQueuePtrManager<q_capacity>(
            &receiver_local_wrptr, &receiver_local_rdptr_ack, &receiver_local_rdptr_completions));

    ASSERT_EQ(fcr.get_num_unacked_credits(), 0);
    ASSERT_EQ(fcr.get_num_incompleted_credits(), 0);
    ASSERT_EQ(fcr.local_has_unacknowledged_credits(), false);
    ASSERT_EQ(fcr.local_has_incomplete_credits(), false);
    ASSERT_EQ(fcr.local_has_data(), false);
    ASSERT_EQ(fcr.local_has_free_credits(q_capacity), true);

    for (size_t i = 0; i < q_capacity; i++) {
        // "Signal" the flow control receiver that data is available
        receiver_local_wrptr = i + 1;

        ASSERT_EQ(fcr.local_has_data(), true);
        ASSERT_EQ(fcr.local_has_free_credits(q_capacity - i - 1),  true);
        ASSERT_EQ(fcr.local_has_free_credits(q_capacity - i),      false);

        ASSERT_EQ(fcr.get_num_unacked_credits(), i + 1);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), i + 1);
        ASSERT_EQ(fcr.local_has_unacknowledged_credits(), true);
        ASSERT_EQ(fcr.local_has_incomplete_credits(), true);
    }
}

// Test FCR ack functionality in isolation
// Test FCR APIs are correct for when data is "coming in"
TEST(FlowControlReceiverHost, BasicDrainQueueAcksCreditsSending_PowerOf2SizedQueue) {
    size_t receiver_local_wrptr = 0;
    size_t receiver_local_rdptr_ack = 0;
    size_t receiver_local_rdptr_completions = 0;

    size_t receiver_remote_wrptr = 0;
    size_t receiver_remote_rdptr_ack = 0;
    size_t receiver_remote_rdptr_completions = 0;

    static constexpr size_t q_capacity = 8;
    auto fcr = tt_metal::routing::flow_control::PacketizedHostFlowControlReceiver(
        &receiver_remote_rdptr_ack,
        &receiver_remote_rdptr_completions,
        q_capacity,
        RemoteQueuePtrManager<q_capacity>(
            &receiver_local_wrptr, &receiver_local_rdptr_ack, &receiver_local_rdptr_completions, q_capacity));

    receiver_local_wrptr = q_capacity;

    // "Drain" the acks
    for (size_t i = 0; i < q_capacity; i++) {
        // "Signal" the flow control receiver that data is available
        ASSERT_EQ(fcr.local_has_data(), true);

        ASSERT_EQ(fcr.get_num_unacked_credits(), q_capacity - i);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), q_capacity);
        ASSERT_EQ(fcr.local_has_unacknowledged_credits(), true);
        ASSERT_EQ(fcr.local_has_incomplete_credits(), true);

        fcr.advance_ack_credits(1);
        fcr.send_ack_credits();
    }
    ASSERT_EQ(fcr.local_has_data(), true);

    // Now "Drain" the completions
    for (size_t i = 0; i < q_capacity; i++) {
        ASSERT_EQ(fcr.local_has_data(), true);

        ASSERT_EQ(fcr.get_num_unacked_credits(), 0);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), q_capacity - i);

        ASSERT_EQ(fcr.local_has_unacknowledged_credits(), false);
        ASSERT_EQ(fcr.local_has_incomplete_credits(), true);

        fcr.advance_completion_credits(1);
        fcr.send_completion_credits();
    }

    ASSERT_EQ(fcr.local_has_data(), false);
}

TEST(FlowControlReceiverHost, BasicDrainQueueAcksCreditsSending_NonPowerOf2SizedQueue_AllQueueStartOffsets) {
    size_t receiver_local_wrptr = 0;
    size_t receiver_local_rdptr_ack = 0;
    size_t receiver_local_rdptr_completions = 0;

    size_t receiver_remote_wrptr = 0;
    size_t receiver_remote_rdptr_ack = 0;
    size_t receiver_remote_rdptr_completions = 0;

    static constexpr size_t q_capacity = 9;
    auto fcr = tt_metal::routing::flow_control::PacketizedHostFlowControlReceiver(
        &receiver_remote_rdptr_ack,
        &receiver_remote_rdptr_completions,
        q_capacity,
        RemoteQueuePtrManager(
            &receiver_local_wrptr, &receiver_local_rdptr_ack, &receiver_local_rdptr_completions, q_capacity));

    for (size_t offset = 0; offset < q_capacity; offset++) {
        auto start_offset = (offset + (offset * q_capacity)) % (2 * q_capacity);
        if (offset > 0) {
            receiver_local_wrptr = (receiver_local_wrptr + 1) % (2 * q_capacity);
            // Start at the next offset into the queue
            fcr.advance_ack_credits(1);
            fcr.send_ack_credits();
            fcr.advance_completion_credits(1);
            fcr.send_completion_credits();
        }

        // fill the queue from this offset
        receiver_local_wrptr = (start_offset + q_capacity) % (2 * q_capacity);

        // "Drain" the acks
        for (size_t i = 0; i < q_capacity; i++) {
            // "Signal" the flow control receiver that data is available
            ASSERT_EQ(fcr.local_has_data(), true);

            ASSERT_EQ(fcr.get_num_unacked_credits(), q_capacity - i);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), q_capacity);
            ASSERT_EQ(fcr.local_has_unacknowledged_credits(), true);
            ASSERT_EQ(fcr.local_has_incomplete_credits(), true);

            fcr.advance_ack_credits(1);
            fcr.send_ack_credits();
        }
        ASSERT_EQ(fcr.local_has_data(), true);

        // Now "Drain" the completions
        for (size_t i = 0; i < q_capacity; i++) {
            ASSERT_EQ(fcr.local_has_data(), true);

            ASSERT_EQ(fcr.get_num_unacked_credits(), 0);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), q_capacity - i);

            ASSERT_EQ(fcr.local_has_unacknowledged_credits(), false);
            ASSERT_EQ(fcr.local_has_incomplete_credits(), true);

            fcr.advance_completion_credits(1);
            fcr.send_completion_credits();
        }

        ASSERT_EQ(fcr.local_has_data(), false);
    }
}



/////////////////////////////////////////
// Flow Control Sender + Receiver Tests
/////////////////////////////////////////

TEST(FlowControlSenderAndReceiverHost, BasicFillDrainSequence_FillOneDrainOne_LoopedOverAllOffsets) {
    size_t receiver_wrptr = 0;
    size_t receiver_rdptr_ack = 0;
    size_t receiver_rdptr_completions = 0;

    size_t sender_wrptr = 0;
    size_t sender_rdptr_ack = 0;
    size_t sender_rdptr_completions = 0;

    static constexpr size_t q_capacity = 9;
    auto fcr = tt_metal::routing::flow_control::PacketizedHostFlowControlReceiver(
        &sender_rdptr_ack,
        &sender_rdptr_completions,
        q_capacity,
        RemoteQueuePtrManager(
            &receiver_wrptr, &receiver_rdptr_ack, &receiver_rdptr_completions, q_capacity));
    auto fcs = tt_metal::routing::flow_control::PacketizedHostFlowControlSender(
        &receiver_wrptr,
        q_capacity,
        RemoteQueuePtrManager(
            &sender_wrptr, &sender_rdptr_ack, &sender_rdptr_completions, q_capacity));

    for (size_t offset = 0; offset < q_capacity; offset++) {
        // Start at the next offset into the queue - checks here for sanity check
        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);
        ASSERT_EQ(fcr.local_has_data(), false);

        fcs.advance_write_credits(1);
        fcs.send_credits();
        ASSERT_EQ(fcr.local_has_data(), true);
        ASSERT_EQ(fcr.get_num_unacked_credits(), 1);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), 1);

        fcr.advance_ack_credits(1);
        fcr.advance_completion_credits(1);
        fcr.send_ack_credits();
        fcr.send_completion_credits();
        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);


        for (size_t i = 0; i < q_capacity; i++) {
            ASSERT_EQ(fcr.local_has_data(), false);
            fcs.advance_write_credits(1);
            fcs.send_credits();
            ASSERT_EQ(fcr.local_has_data(), true);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity - 1);
            ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - 1);
            ASSERT_EQ(fcr.get_num_unacked_credits(), 1);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), 1);

            fcr.advance_ack_credits(1);
            fcr.send_ack_credits();
            ASSERT_EQ(fcr.local_has_data(), true);
            ASSERT_EQ(fcr.get_num_unacked_credits(), 0);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), 1);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
            ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - 1);

            fcr.advance_completion_credits(1);
            fcr.send_completion_credits();
            ASSERT_EQ(fcr.local_has_data(), false);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), 0);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
            ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);
        }
    }
}

TEST(FlowControlSenderAndReceiverHost, BasicFillDrainSequence_FillEntirelyThenAckEntirelyCompleteEntirely_LoopedOverAllOffsets) {
    size_t receiver_wrptr = 0;
    size_t receiver_rdptr_ack = 0;
    size_t receiver_rdptr_completions = 0;

    size_t sender_wrptr = 0;
    size_t sender_rdptr_ack = 0;
    size_t sender_rdptr_completions = 0;

    static constexpr size_t q_capacity = 9;
    auto fcr = tt_metal::routing::flow_control::PacketizedHostFlowControlReceiver(
        &sender_rdptr_ack,
        &sender_rdptr_completions,
        q_capacity,
        RemoteQueuePtrManager(
            &receiver_wrptr, &receiver_rdptr_ack, &receiver_rdptr_completions, q_capacity));
    auto fcs = tt_metal::routing::flow_control::PacketizedHostFlowControlSender(
        &receiver_wrptr,
        q_capacity,
        RemoteQueuePtrManager(
            &sender_wrptr, &sender_rdptr_ack, &sender_rdptr_completions, q_capacity));

    for (size_t offset = 0; offset < q_capacity; offset++) {
        // Start at the next offset into the queue - checks here for sanity check
        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);
        ASSERT_EQ(fcr.local_has_data(), false);

        fcs.advance_write_credits(1);
        fcs.send_credits();
        ASSERT_EQ(fcr.local_has_data(), true);
        ASSERT_EQ(fcr.get_num_unacked_credits(), 1);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), 1);

        fcr.advance_ack_credits(1);
        fcr.advance_completion_credits(1);
        fcr.send_ack_credits();
        fcr.send_completion_credits();
        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);


        ASSERT_EQ(fcr.local_has_data(), false);
        fcs.advance_write_credits(q_capacity);
        fcs.send_credits();

        ASSERT_EQ(fcr.local_has_data(), true);
        ASSERT_EQ(fcs.get_num_free_local_credits(), 0);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), 0);
        ASSERT_EQ(fcr.get_num_unacked_credits(), q_capacity);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), q_capacity);

        fcr.advance_ack_credits(q_capacity);
        fcr.send_ack_credits();
        ASSERT_EQ(fcr.local_has_data(), true);
        ASSERT_EQ(fcr.get_num_unacked_credits(), 0);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), q_capacity);
        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), 0);

        fcr.advance_completion_credits(q_capacity);
        fcr.send_completion_credits();
        ASSERT_EQ(fcr.local_has_data(), false);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), 0);
        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);
    }
}

TEST(FlowControlSenderAndReceiverHost, BasicFillDrainSequence_FillAllPossibleContigSizesThenAckEntirelyCompleteEntirely_LoopedOverAllOffsets) {
    size_t receiver_wrptr = 0;
    size_t receiver_rdptr_ack = 0;
    size_t receiver_rdptr_completions = 0;

    size_t sender_wrptr = 0;
    size_t sender_rdptr_ack = 0;
    size_t sender_rdptr_completions = 0;

    static constexpr size_t q_capacity = 9;
    auto fcr = tt_metal::routing::flow_control::PacketizedHostFlowControlReceiver(
        &sender_rdptr_ack,
        &sender_rdptr_completions,
        q_capacity,
        RemoteQueuePtrManager(
            &receiver_wrptr, &receiver_rdptr_ack, &receiver_rdptr_completions, q_capacity));
    auto fcs = tt_metal::routing::flow_control::PacketizedHostFlowControlSender(
        &receiver_wrptr,
        q_capacity,
        RemoteQueuePtrManager(
            &sender_wrptr, &sender_rdptr_ack, &sender_rdptr_completions, q_capacity));

    size_t sender_total_write_offset = 0;
    size_t receiver_total_write_offset = 0;
    size_t sender_total_ack_offset = 0;
    size_t receiver_total_ack_offset = 0;
    size_t sender_total_completion_offset = 0;
    size_t receiver_total_completion_offset = 0;
    for (size_t offset = 0; offset < q_capacity; offset++) {
        // Start at the next offset into the queue - checks here for sanity check
        ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);
        ASSERT_EQ(fcr.local_has_data(), false);

        for (size_t send_size = 1; send_size < q_capacity; send_size++) {
            ASSERT_EQ(fcr.local_has_data(), false);
            fcs.advance_write_credits(send_size);
            sender_total_write_offset += send_size;
            ASSERT_EQ(sender_total_write_offset % (2 * q_capacity), sender_wrptr);

            fcs.send_credits();
            receiver_total_write_offset += send_size;
            ASSERT_EQ(receiver_total_write_offset % (2 * q_capacity), receiver_wrptr);

            ASSERT_EQ(fcr.local_has_data(), true);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity - send_size);
            ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - send_size);
            ASSERT_EQ(fcr.get_num_unacked_credits(), send_size);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), send_size);

            fcr.advance_ack_credits(send_size);
            receiver_total_ack_offset += send_size;
            ASSERT_EQ(receiver_total_ack_offset % (2 * q_capacity), receiver_rdptr_ack);

            fcr.send_ack_credits();
            sender_total_ack_offset += send_size;
            ASSERT_EQ(sender_total_ack_offset % (2 * q_capacity), sender_rdptr_ack);
            ASSERT_EQ(fcr.local_has_data(), true);
            ASSERT_EQ(fcr.get_num_unacked_credits(), 0);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), send_size);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
            ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - send_size);

            fcr.advance_completion_credits(send_size);
            receiver_total_completion_offset += send_size;
            ASSERT_EQ(receiver_total_completion_offset % (2 * q_capacity), receiver_rdptr_completions);
            fcr.send_completion_credits();
            sender_total_completion_offset += send_size;
            ASSERT_EQ(sender_total_completion_offset % (2 * q_capacity), sender_rdptr_completions);
            ASSERT_EQ(fcr.local_has_data(), false);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), 0);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity);
            ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity);
        }
    }
}

TEST(FlowControlSenderAndReceiverHost, BasicFillDrainSequence_SenderLeadsAcksBy3LeadsCompletionsBy3_LoopedOverAllOffsets) {
    size_t receiver_wrptr = 0;
    size_t receiver_rdptr_ack = 0;
    size_t receiver_rdptr_completions = 0;

    size_t sender_wrptr = 0;
    size_t sender_rdptr_ack = 0;
    size_t sender_rdptr_completions = 0;

    size_t ack_delay = 3;
    size_t completion_delay = 6;

    static constexpr size_t q_capacity = 9;
    auto fcr = tt_metal::routing::flow_control::PacketizedHostFlowControlReceiver(
        &sender_rdptr_ack,
        &sender_rdptr_completions,
        q_capacity,
        RemoteQueuePtrManager(
            &receiver_wrptr, &receiver_rdptr_ack, &receiver_rdptr_completions, q_capacity));
    auto fcs = tt_metal::routing::flow_control::PacketizedHostFlowControlSender(
        &receiver_wrptr,
        q_capacity,
        RemoteQueuePtrManager(
            &sender_wrptr, &sender_rdptr_ack, &sender_rdptr_completions, q_capacity));

    size_t sender_total_write_offset = 0;
    size_t receiver_total_write_offset = 0;
    size_t sender_total_ack_offset = 0;
    size_t receiver_total_ack_offset = 0;
    size_t sender_total_completion_offset = 0;
    size_t receiver_total_completion_offset = 0;

    size_t num_credits_to_send = 1000;
    for (size_t c = 0; c < num_credits_to_send; c++) {
        ASSERT_EQ(fcr.local_has_data(), c > 0);
        fcs.advance_write_credits(1);
        sender_total_write_offset += 1;
        ASSERT_EQ(sender_total_write_offset % (2 * q_capacity), sender_wrptr);

        fcs.send_credits();
        receiver_total_write_offset += 1;
        ASSERT_EQ(receiver_total_write_offset % (2 * q_capacity), receiver_wrptr);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), c >= completion_delay ? q_capacity - completion_delay - 1 : q_capacity - c - 1);
        ASSERT_EQ(fcr.get_num_unacked_credits(), std::min(c+1,ack_delay + 1));

        if (c >= ack_delay) {
            fcr.advance_ack_credits(1);
            receiver_total_ack_offset += 1;
            ASSERT_EQ(receiver_total_ack_offset % (2 * q_capacity), receiver_rdptr_ack);

            ASSERT_EQ(fcs.get_num_free_remote_credits(), c >= completion_delay ? q_capacity - completion_delay - 1 : q_capacity - c - 1);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity - ack_delay - 1);
            fcr.send_ack_credits();
            sender_total_ack_offset += 1;
            ASSERT_EQ(sender_total_ack_offset % (2 * q_capacity), sender_rdptr_ack);
            ASSERT_EQ(fcr.local_has_data(), true);
            ASSERT_EQ(fcr.get_num_unacked_credits(), ack_delay);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), std::min(c + 1, completion_delay + 1));
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity - ack_delay);

            if (c >= completion_delay) {
                fcr.advance_completion_credits(1);
                receiver_total_completion_offset += 1;
                ASSERT_EQ(receiver_total_completion_offset % (2 * q_capacity), receiver_rdptr_completions);
                fcr.send_completion_credits();
                sender_total_completion_offset += 1;
                ASSERT_EQ(sender_total_completion_offset % (2 * q_capacity), sender_rdptr_completions);
                ASSERT_EQ(fcr.local_has_data(), true);
                ASSERT_EQ(fcr.get_num_incompleted_credits(), completion_delay);
                ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - completion_delay);
            }
        }
        ASSERT_EQ(fcr.local_has_data(), true);
    }

    for (size_t c = 0; c < completion_delay; c++) {
        if (c < ack_delay) {
            fcr.advance_ack_credits(1);
            receiver_total_ack_offset += 1;
            ASSERT_EQ(receiver_total_ack_offset % (2 * q_capacity), receiver_rdptr_ack);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity - (ack_delay - c));

            fcr.send_ack_credits();
            sender_total_ack_offset += 1;
            ASSERT_EQ(sender_total_ack_offset % (2 * q_capacity), sender_rdptr_ack);
            ASSERT_EQ(fcr.local_has_data(), true);
            ASSERT_EQ(fcr.get_num_unacked_credits(), ack_delay - c - 1);
            ASSERT_EQ(fcr.get_num_incompleted_credits(), completion_delay - c);
            ASSERT_EQ(fcs.get_num_free_local_credits(), q_capacity - (ack_delay - c - 1));
        }
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - (completion_delay - c));

        fcr.advance_completion_credits(1);
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - (completion_delay - c));
        receiver_total_completion_offset += 1;
        ASSERT_EQ(receiver_total_completion_offset % (2 * q_capacity), receiver_rdptr_completions);

        fcr.send_completion_credits();
        sender_total_completion_offset += 1;
        ASSERT_EQ(fcs.get_num_free_remote_credits(), q_capacity - (completion_delay - c - 1));
        ASSERT_EQ(sender_total_completion_offset % (2 * q_capacity), sender_rdptr_completions);
        ASSERT_EQ(fcr.local_has_data(), c != completion_delay - 1);
        ASSERT_EQ(fcr.get_num_incompleted_credits(), completion_delay - c - 1);

    }
}
