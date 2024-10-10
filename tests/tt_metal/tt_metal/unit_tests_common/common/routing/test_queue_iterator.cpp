// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/impl/routing/flow_control/queue_iterator.hpp"

TEST(QueueIteratorAPI, Construction_NonConstexprQueueSize) {
    size_t wrptr, rdptr_ack, rdptr_completions;
    for (size_t q_size = 1; q_size < 100; q_size++) {
        auto qptrs = RemoteQueuePtrManager(&wrptr, &rdptr_ack, &rdptr_completions, q_size);

        ASSERT_EQ(qptrs.get_local_space_available(), q_size);
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), 0);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);
        ASSERT_EQ(qptrs.get_wrptr(), 0);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        ASSERT_EQ(qptrs.get_queue_size(), q_size);
    }
}

TEST(QueueIteratorAPI, Construction_ConstexprQueueSizes_NonPowerOf2) {
    size_t wrptr, rdptr_ack, rdptr_completions;
    {
        constexpr size_t q_size = 1;
        auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

        ASSERT_EQ(qptrs.get_local_space_available(), q_size);
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), 0);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);
        ASSERT_EQ(qptrs.get_wrptr(), 0);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        ASSERT_EQ(qptrs.get_queue_size(), q_size);
    }
    {
        constexpr size_t q_size = 3;
        auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

        ASSERT_EQ(qptrs.get_local_space_available(), q_size);
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), 0);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);
        ASSERT_EQ(qptrs.get_wrptr(), 0);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        ASSERT_EQ(qptrs.get_queue_size(), q_size);
    }
    {
        constexpr size_t q_size = 5;
        auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

        ASSERT_EQ(qptrs.get_local_space_available(), q_size);
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), 0);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);
        ASSERT_EQ(qptrs.get_wrptr(), 0);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        ASSERT_EQ(qptrs.get_queue_size(), q_size);
    }
    {
        constexpr size_t q_size = 6;
        auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

        ASSERT_EQ(qptrs.get_local_space_available(), q_size);
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), 0);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);
        ASSERT_EQ(qptrs.get_wrptr(), 0);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        ASSERT_EQ(qptrs.get_queue_size(), q_size);
    }
    {
        constexpr size_t q_size = 1023;
        auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

        ASSERT_EQ(qptrs.get_local_space_available(), q_size);
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), 0);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);
        ASSERT_EQ(qptrs.get_wrptr(), 0);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        ASSERT_EQ(qptrs.get_queue_size(), q_size);
    }
}

template <size_t q_size>
void test_qptrs() {
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

    ASSERT_EQ(qptrs.get_local_space_available(), q_size);
    ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
    ASSERT_EQ(qptrs.get_num_credits_incomplete(), 0);
    ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);
    ASSERT_EQ(qptrs.get_wrptr(), 0);
    ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
    ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
    ASSERT_EQ(qptrs.get_queue_size(), q_size);
};
TEST(QueueIteratorAPI, Construction_ConstexprQueueSizes_PowerOf2) {
    test_qptrs<2>();
    test_qptrs<4>();
    test_qptrs<8>();
    test_qptrs<16>();
    test_qptrs<32>();
    test_qptrs<64>();
    test_qptrs<128>();
    test_qptrs<1024>();
}

TEST(QueueIteratorAPI, QueueFill_OneAtATime_PowerOf2) {
    constexpr size_t q_size = 8;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

    for (size_t i = 0; i < q_size; i++) {
        ASSERT_EQ(qptrs.get_wrptr(), i);
        qptrs.advance_write_credits(1);
        ASSERT_EQ(qptrs.get_local_space_available(), q_size - (i + 1));
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size - (i + 1));
        ASSERT_EQ(qptrs.get_num_unacked_credits(), i + 1);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), i + 1);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
    }
    ASSERT_EQ(qptrs.get_queue_size(), q_size);
    // check wrap-around
    ASSERT_EQ(qptrs.get_wrptr(), 0);
    ASSERT_EQ(qptrs.get_local_space_available(), 0);
}

TEST(QueueIteratorAPI, QueueFill_OneAtATime_NonPowerOf2) {
    constexpr size_t q_size = 9;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager(&wrptr, &rdptr_ack, &rdptr_completions, q_size);

    for (size_t i = 0; i < q_size; i++) {
        ASSERT_EQ(qptrs.get_wrptr(), i);
        qptrs.advance_write_credits(1);
        ASSERT_EQ(qptrs.get_local_space_available(), q_size - (i + 1));
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size - (i + 1));
        ASSERT_EQ(qptrs.get_num_unacked_credits(), i + 1);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), i + 1);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
    }
    ASSERT_EQ(qptrs.get_queue_size(), q_size);
    // check wrap-around
    ASSERT_EQ(qptrs.get_wrptr(), 0);
    ASSERT_EQ(qptrs.get_local_space_available(), 0);
}

TEST(QueueIteratorAPI, QueueFill_MultipleAtATime_PowerOf2) {
    constexpr size_t q_size = 8;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

    for (size_t i = 0; i < q_size; i += 3) {
        ASSERT_EQ(qptrs.get_wrptr(), i);
        auto n = std::min<size_t>(3, q_size - i);
        auto end = i + n;
        qptrs.advance_write_credits(n);
        ASSERT_EQ(qptrs.get_local_space_available(), q_size - end);
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size - end);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), end);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), end);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
    }
    ASSERT_EQ(qptrs.get_queue_size(), q_size);
    // check wrap-around
    ASSERT_EQ(qptrs.get_wrptr(), 0);
    ASSERT_EQ(qptrs.get_local_space_available(), 0);
}

TEST(QueueIteratorAPI, QueueFill_MultipleAtATime_NonPowerOf2) {
    constexpr size_t q_size = 9;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager(&wrptr, &rdptr_ack, &rdptr_completions, q_size);

    for (size_t i = 0; i < q_size; i += 3) {
        ASSERT_EQ(qptrs.get_wrptr(), i);
        auto n = std::min<size_t>(3, q_size - i);
        auto end = i + n;
        qptrs.advance_write_credits(n);
        ASSERT_EQ(qptrs.get_local_space_available(), q_size - end);
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size - end);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), end);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), end);
        ASSERT_EQ(qptrs.get_rdptr_acks(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
    }
    ASSERT_EQ(qptrs.get_queue_size(), q_size);
    // check wrap-around
    ASSERT_EQ(qptrs.get_wrptr(), 0);
    ASSERT_EQ(qptrs.get_local_space_available(), 0);
}

TEST(QueueIteratorAPI, QueueDrain_OneAtATime_PowerOf2) {
    constexpr size_t q_size = 8;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

    for (size_t i = 0; i < q_size; i++) {
        qptrs.advance_write_credits(1);
    }

    // "Drain" the acks
    for (size_t i = 0; i < q_size; i++) {
        ASSERT_EQ(qptrs.get_local_space_available(), i);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), q_size - i);
        ASSERT_EQ(qptrs.get_rdptr_acks(), i);
        ASSERT_EQ(qptrs.get_remote_space_available(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        qptrs.advance_read_ack_credits(1);
    }
    ASSERT_EQ(qptrs.get_remote_space_available(), 0);
    ASSERT_EQ(qptrs.get_local_space_available(), q_size);
    ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);

    // "Drain" the completions
    for (size_t i = 0; i < q_size; i++) {
        ASSERT_EQ(qptrs.get_remote_space_available(), i);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), q_size - i);
        ASSERT_EQ(qptrs.get_rdptr_completions(), i);
        ASSERT_EQ(qptrs.get_remote_space_available(), i);
        qptrs.advance_read_completion_credits(1);
    }
    ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
}

TEST(QueueIteratorAPI, QueueDrain_OneAtATime_NonPowerOf2) {
    constexpr size_t q_size = 9;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager(&wrptr, &rdptr_ack, &rdptr_completions, q_size);

    for (size_t i = 0; i < q_size; i++) {
        qptrs.advance_write_credits(1);
    }

    // "Drain" the acks
    for (size_t i = 0; i < q_size; i++) {
        ASSERT_EQ(qptrs.get_local_space_available(), i);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), q_size - i);
        ASSERT_EQ(qptrs.get_rdptr_acks(), i);
        ASSERT_EQ(qptrs.get_remote_space_available(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        qptrs.advance_read_ack_credits(1);
    }
    ASSERT_EQ(qptrs.get_remote_space_available(), 0);
    ASSERT_EQ(qptrs.get_local_space_available(), q_size);
    ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);

    // "Drain" the completions
    for (size_t i = 0; i < q_size; i++) {
        ASSERT_EQ(qptrs.get_remote_space_available(), i);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), q_size - i);
        ASSERT_EQ(qptrs.get_rdptr_completions(), i);
        ASSERT_EQ(qptrs.get_remote_space_available(), i);
        qptrs.advance_read_completion_credits(1);
    }
    ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
}

TEST(QueueIteratorAPI, QueueDrain_MultipleAtATime_PowerOf2) {
    constexpr size_t q_size = 8;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

    for (size_t i = 0; i < q_size; i++) {
        qptrs.advance_write_credits(1);
    }

    constexpr size_t drain_rate = 3;
    // "Drain" the acks
    for (size_t i = 0; i < q_size; i += drain_rate) {
        auto num_to_ack = std::min(drain_rate, q_size - i);
        ASSERT_EQ(qptrs.get_local_space_available(), i);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), q_size - i);
        ASSERT_EQ(qptrs.get_rdptr_acks(), i);
        ASSERT_EQ(qptrs.get_remote_space_available(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        qptrs.advance_read_ack_credits(num_to_ack);
    }
    ASSERT_EQ(qptrs.get_remote_space_available(), 0);
    ASSERT_EQ(qptrs.get_local_space_available(), q_size);
    ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);

    // "Drain" the completions
    for (size_t i = 0; i < q_size; i += drain_rate) {
        auto num_to_complete = std::min(drain_rate, q_size - i);
        ASSERT_EQ(qptrs.get_remote_space_available(), i);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), q_size - i);
        ASSERT_EQ(qptrs.get_rdptr_completions(), i);
        ASSERT_EQ(qptrs.get_remote_space_available(), i);
        qptrs.advance_read_completion_credits(num_to_complete);
    }
    ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
}

TEST(QueueIteratorAPI, QueueDrain_MultipleAtATime_NonPowerOf2) {
    constexpr size_t q_size = 9;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager(&wrptr, &rdptr_ack, &rdptr_completions, q_size);

    for (size_t i = 0; i < q_size; i++) {
        qptrs.advance_write_credits(1);
    }

    constexpr size_t drain_rate = 3;
    // "Drain" the acks
    for (size_t i = 0; i < q_size; i += drain_rate) {
        auto num_to_ack = std::min(drain_rate, q_size - i);
        ASSERT_EQ(qptrs.get_local_space_available(), i);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), q_size - i);
        ASSERT_EQ(qptrs.get_rdptr_acks(), i);
        ASSERT_EQ(qptrs.get_remote_space_available(), 0);
        ASSERT_EQ(qptrs.get_rdptr_completions(), 0);
        qptrs.advance_read_ack_credits(num_to_ack);
    }
    ASSERT_EQ(qptrs.get_remote_space_available(), 0);
    ASSERT_EQ(qptrs.get_local_space_available(), q_size);
    ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);

    // "Drain" the completions
    for (size_t i = 0; i < q_size; i += drain_rate) {
        auto num_to_complete = std::min(drain_rate, q_size - i);
        ASSERT_EQ(qptrs.get_remote_space_available(), i);
        ASSERT_EQ(qptrs.get_num_credits_incomplete(), q_size - i);
        ASSERT_EQ(qptrs.get_rdptr_completions(), i);
        ASSERT_EQ(qptrs.get_remote_space_available(), i);
        qptrs.advance_read_completion_credits(num_to_complete);
    }
    ASSERT_EQ(qptrs.get_remote_space_available(), q_size);
}


TEST(QueueIteratorAPI, QueueDrain_OneAtATime_PowerOf2_CyclingThroughQueue) {
    constexpr size_t q_size = 8;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager<q_size>(&wrptr, &rdptr_ack, &rdptr_completions);

    for (size_t start_wrptr = 0; start_wrptr < 2 * q_size; start_wrptr++) {

        for (size_t i = 0; i < q_size; i++) {
            qptrs.advance_write_credits(1);
        }

        // "Drain" the acks
        for (size_t i = 0; i < q_size; i++) {
            ASSERT_EQ(qptrs.get_local_space_available(), i);
            ASSERT_EQ(qptrs.get_num_unacked_credits(), q_size - i);
            ASSERT_EQ(qptrs.get_rdptr_acks(), (start_wrptr + i) % q_size);
            ASSERT_EQ(qptrs.get_remote_space_available(), 0);
            ASSERT_EQ(qptrs.get_rdptr_completions(), start_wrptr % q_size);
            qptrs.advance_read_ack_credits(1);
        }
        ASSERT_EQ(qptrs.get_remote_space_available(), 0);
        ASSERT_EQ(qptrs.get_local_space_available(), q_size);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);

        // "Drain" the completions
        for (size_t i = 0; i < q_size; i++) {
            ASSERT_EQ(qptrs.get_remote_space_available(), i);
            ASSERT_EQ(qptrs.get_num_credits_incomplete(), q_size - i);
            ASSERT_EQ(qptrs.get_rdptr_completions(), (start_wrptr + i) % q_size);
            ASSERT_EQ(qptrs.get_remote_space_available(), i);
            qptrs.advance_read_completion_credits(1);
        }
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size);

        // Advance the wrptr to the next logical start_wrptr and make sure the queue is empty
        qptrs.advance_write_credits(1);
        qptrs.advance_read_ack_credits(1);
        qptrs.advance_read_completion_credits(1);
    }
}

TEST(QueueIteratorAPI, QueueDrain_OneAtATime_NonPowerOf2_CyclingThroughQueue) {
    constexpr size_t q_size = 9;
    size_t wrptr, rdptr_ack, rdptr_completions;
    auto qptrs = RemoteQueuePtrManager(&wrptr, &rdptr_ack, &rdptr_completions, q_size);
    for (size_t start_wrptr = 0; start_wrptr < 2 * q_size; start_wrptr++) {
        for (size_t i = 0; i < q_size; i++) {
            qptrs.advance_write_credits(1);
        }

        // "Drain" the acks
        for (size_t i = 0; i < q_size; i++) {
            ASSERT_EQ(qptrs.get_local_space_available(), i);
            ASSERT_EQ(qptrs.get_num_unacked_credits(), q_size - i);
            ASSERT_EQ(qptrs.get_rdptr_acks(), (start_wrptr + i) % q_size);
            ASSERT_EQ(qptrs.get_remote_space_available(), 0);
            ASSERT_EQ(qptrs.get_rdptr_completions(), start_wrptr % q_size);
            qptrs.advance_read_ack_credits(1);
        }
        ASSERT_EQ(qptrs.get_remote_space_available(), 0);
        ASSERT_EQ(qptrs.get_local_space_available(), q_size);
        ASSERT_EQ(qptrs.get_num_unacked_credits(), 0);

        // "Drain" the completions
        for (size_t i = 0; i < q_size; i++) {
            ASSERT_EQ(qptrs.get_remote_space_available(), i);
            ASSERT_EQ(qptrs.get_num_credits_incomplete(), q_size - i);
            ASSERT_EQ(qptrs.get_rdptr_completions(), (start_wrptr + i) % q_size);
            ASSERT_EQ(qptrs.get_remote_space_available(), i);
            qptrs.advance_read_completion_credits(1);
        }
        ASSERT_EQ(qptrs.get_remote_space_available(), q_size);

        // Advance the wrptr to the next logical start_wrptr and make sure the queue is empty
        qptrs.advance_write_credits(1);
        qptrs.advance_read_ack_credits(1);
        qptrs.advance_read_completion_credits(1);
    }
}
