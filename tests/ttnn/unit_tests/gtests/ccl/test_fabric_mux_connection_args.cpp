// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program.hpp>
#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::ccl::test {
namespace {

class MeshDeviceFabric1DFixture : public tt::tt_metal::MeshDeviceFixtureBase {
protected:
    MeshDeviceFabric1DFixture() :
        MeshDeviceFixtureBase(Config{.fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D}) {}
};

TEST_F(MeshDeviceFabric1DFixture, FabricMuxClientReusesSuppliedTerminationSemaphore) {
    const tt::tt_fabric::FabricMuxConfig mux_config(
        /*num_full_size_channels=*/1,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/2,
        /*num_buffers_header_only_channel=*/0,
        tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
        mesh_device_->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    const CoreCoord client_logical_core{0, 0};
    const CoreCoord client_virtual_core = mesh_device_->worker_core_from_logical_core(client_logical_core);
    constexpr uint32_t supplied_termination_semaphore_id = 7;
    tt::tt_metal::Program program;
    std::vector<uint32_t> runtime_args;

    fabric_mux_connection_rt_args(
        /*mux_connection_valid=*/true,
        /*is_termination_master=*/false,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        client_virtual_core,
        /*worker_id=*/0,
        client_logical_core,
        mux_config,
        program,
        client_virtual_core,
        runtime_args,
        supplied_termination_semaphore_id);

    ASSERT_EQ(runtime_args.size(), 17u);
    EXPECT_EQ(runtime_args[10], supplied_termination_semaphore_id);
    // The four local semaphores are the only ones created on the core, so they take its first IDs.
    const std::vector<uint32_t> local_semaphore_ids(runtime_args.begin() + 11, runtime_args.begin() + 15);
    EXPECT_EQ(local_semaphore_ids, (std::vector<uint32_t>{0, 1, 2, 3}));
}

}  // namespace
}  // namespace ttnn::ccl::test
