// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/distributed_context.hpp>

#include "ttnn/distributed/mpi_socket.hpp"

namespace {

using tt::tt_metal::distributed::multihost::DistributedContext;
using tt::tt_metal::distributed::multihost::Rank;
using ttnn::distributed::detail::MPITensorDescriptor;
using ttnn::distributed::detail::receive_and_validate_mpi_tensor_descriptor;
using ttnn::distributed::detail::send_mpi_tensor_descriptor;

MPITensorDescriptor make_descriptor() {
    return MPITensorDescriptor{
        .dtype = static_cast<uint32_t>(tt::tt_metal::DataType::BFLOAT16),
        .layout = static_cast<uint32_t>(tt::tt_metal::Layout::TILE),
        .logical_shape = {1, 1, 31, 32},
        .padded_shape = {1, 1, 32, 32},
        .segment_sizes = {2048},
    };
}

void reject_then_reuse_context(Rank sender_rank) {
    const auto& context = DistributedContext::get_current_world();
    ASSERT_EQ(*context->size(), 2U);

    const Rank receiver_rank{1U - *sender_rank};
    auto sent = make_descriptor();
    auto mismatched = sent;
    mismatched.logical_shape = {1, 1, 32, 32};

    if (context->rank() == sender_rank) {
        EXPECT_ANY_THROW(send_mpi_tensor_descriptor(context, receiver_rank, sent));
    } else {
        EXPECT_ANY_THROW(receive_and_validate_mpi_tensor_descriptor(context, sender_rank, mismatched));
    }
    context->barrier();

    if (context->rank() == sender_rank) {
        EXPECT_NO_THROW(send_mpi_tensor_descriptor(context, receiver_rank, sent));
    } else {
        EXPECT_NO_THROW(receive_and_validate_mpi_tensor_descriptor(context, sender_rank, sent));
    }
    context->barrier();
}

TEST(MPISocketProtocol, RejectsMismatchAndReusesContextRank0ToRank1) { reject_then_reuse_context(Rank{0}); }

TEST(MPISocketProtocol, RejectsMismatchAndReusesContextRank1ToRank0) { reject_then_reuse_context(Rank{1}); }

}  // namespace
