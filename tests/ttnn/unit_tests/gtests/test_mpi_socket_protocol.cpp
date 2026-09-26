// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/distributed/mpi_socket.hpp"

namespace {

using ttnn::distributed::detail::compare_mpi_tensor_descriptors;
using ttnn::distributed::detail::deserialize_mpi_tensor_descriptor;
using ttnn::distributed::detail::MPITensorDescriptor;
using ttnn::distributed::detail::serialize_mpi_tensor_descriptor;

MPITensorDescriptor make_descriptor() {
    return MPITensorDescriptor{
        .dtype = static_cast<uint32_t>(ttnn::DataType::BFLOAT16),
        .layout = static_cast<uint32_t>(ttnn::Layout::TILE),
        .logical_shape = {1, 1, 31, 32},
        .padded_shape = {1, 1, 32, 32},
        .segment_sizes = {8, 24},
    };
}

TEST(MPISocketProtocol, DescriptorRoundTrips) {
    const auto expected = make_descriptor();
    const auto bytes = serialize_mpi_tensor_descriptor(expected);
    MPITensorDescriptor actual;
    std::string error;
    ASSERT_TRUE(deserialize_mpi_tensor_descriptor(bytes, actual, error)) << error;
    EXPECT_EQ(actual, expected);
    EXPECT_TRUE(compare_mpi_tensor_descriptors(expected, actual).empty());
}

TEST(MPISocketProtocol, RejectsEqualByteSemanticAndSegmentationMismatches) {
    const auto expected = make_descriptor();

    auto actual = expected;
    actual.logical_shape = {1, 1, 32, 32};
    EXPECT_FALSE(compare_mpi_tensor_descriptors(expected, actual).empty());

    actual = expected;
    actual.dtype = static_cast<uint32_t>(ttnn::DataType::UINT16);
    EXPECT_FALSE(compare_mpi_tensor_descriptors(expected, actual).empty());

    actual = expected;
    actual.segment_sizes = {16, 16};
    EXPECT_FALSE(compare_mpi_tensor_descriptors(expected, actual).empty());

    actual = expected;
    actual.segment_sizes = {32};
    EXPECT_FALSE(compare_mpi_tensor_descriptors(expected, actual).empty());
}

TEST(MPISocketProtocol, RejectsMalformedDescriptor) {
    auto bytes = serialize_mpi_tensor_descriptor(make_descriptor());
    bytes.pop_back();
    MPITensorDescriptor descriptor;
    std::string error;
    EXPECT_FALSE(deserialize_mpi_tensor_descriptor(bytes, descriptor, error));
    EXPECT_FALSE(error.empty());
}

}  // namespace
