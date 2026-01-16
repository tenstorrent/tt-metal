// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/device_tensor.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {
namespace {

TEST(ExperimentalTensorTest, VerifyBuildFails) {
    EXPECT_TRUE(false) << "This test intentionally fails to verify build works";
}

}  // namespace
}  // namespace tt::tt_metal
