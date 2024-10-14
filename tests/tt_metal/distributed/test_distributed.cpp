// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/distributed/mesh_device_view.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed::test {

static inline void skip_test_if_not_t3000() {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const auto arch = tt::Cluster::instance().arch();
    const size_t num_devices = tt::Cluster::instance().number_of_devices();

    if (slow_dispatch) {
        GTEST_SKIP() << "Skipping Multi-Device test suite, since it can only be run in Fast Dispatch Mode.";
    }
    if (num_devices < 8 or arch != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping T3K Multi-Device test suite on non T3K machine.";
    }
}
class MeshDevice_T3000 : public ::testing::Test {
   protected:
    void SetUp() override {
        skip_test_if_not_t3000();
        this->mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    }

    void TearDown() override {
        mesh_device_->close_devices();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(MeshDevice_T3000, SimpleMeshDeviceTest) {
    EXPECT_EQ(mesh_device_->num_devices(), 8);
    EXPECT_EQ(mesh_device_->num_rows(), 2);
    EXPECT_EQ(mesh_device_->num_cols(), 4);
}

}  // namespace tt::tt_metal::distributed::test
