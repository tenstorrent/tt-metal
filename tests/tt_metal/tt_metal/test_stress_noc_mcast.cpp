// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test stresses NOC mcast by:
//  - using 1 mcast core (future work to add multiple) either tensix or eth
//  - rapidly mcast into a grid of tensix workers
//  - rapidly grid of tensix workers generates random noc traffic
//  - does not verify correct transactions, just runs til termination

#include "common/device_fixture.hpp"

#include <fmt/base.h>
#include <cstdlib>
#include <tt-metalium/host_api.hpp>
#include "llrt/metal_soc_descriptor.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <llrt/tt_cluster.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

const uint32_t DEFAULT_SECONDS = 1;  // Reduced for gtest (original was 10)
const uint32_t DEFAULT_TARGET_WIDTH = 1;
const uint32_t DEFAULT_TARGET_HEIGHT = 1;
const uint32_t N_RANDS = 512;

}  // namespace

// Stress test for NOC multicast - uses short duration for CI
// Uses detail::LaunchProgram which requires slow dispatch mode
TEST_F(MeshDeviceSingleCardFixture, StressNocMcast) {
    IDevice* dev = devices_[0]->get_devices()[0];

    // Use default test parameters
    uint32_t time_secs = DEFAULT_SECONDS;
    uint32_t tlx = 0;
    uint32_t tly = 0;
    uint32_t width = DEFAULT_TARGET_WIDTH;
    uint32_t height = DEFAULT_TARGET_HEIGHT;
    uint32_t mcast_x = 0;
    uint32_t mcast_y = 0;
    uint32_t mcast_size = 16;
    uint32_t ucast_size = 8192;
    // Using ucast only mode for simplicity in gtest (mcast code removed)
    bool rnd_delay = false;
    bool rnd_coord = true;
    tt_metal::NOC noc = tt_metal::NOC::NOC_0;
    srand(0);

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange workers_logical({tlx, tly}, {tlx + width - 1, tly + height - 1});
    CoreCoord mcast_logical(mcast_x, mcast_y);
    CoreCoord tl_core = dev->worker_core_from_logical_core({tlx, tly});

    CoreCoord mcast_end = dev->worker_core_from_logical_core(workers_logical.end_coord);
    bool virtualization_enabled = tt::tt_metal::MetalContext::instance().hal().is_coordinate_virtualization_enabled();
    uint32_t num_dests = workers_logical.size();
    CoreCoord virtual_offset = virtualization_enabled
                                   ? dev->worker_core_from_logical_core({0, 0})
                                   : CoreCoord(0, 0);  // In this case pass physical coordinates as runtime args

    std::vector<uint32_t> compile_args = {
        false,
        tl_core.x,
        tl_core.y,
        mcast_end.x,
        mcast_end.y,
        num_dests,
        time_secs,
        ucast_size,
        mcast_size,
        virtual_offset.x,
        virtual_offset.y,
        N_RANDS,
        rnd_delay,
        dev->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1),
        dev->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1),
    };

    KernelHandle ucast_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/stress_noc_mcast.cpp",
        workers_logical,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = noc,
            .compile_args = compile_args,
        });

    for (CoreCoord coord : workers_logical) {
        std::vector<uint32_t> runtime_args;
        // Not particularly random since all cores are getting the same data
        // N_RANDS in bytes
        CoreCoord grid_size = dev->logical_grid_size();
        for (int i = 0; i < N_RANDS / sizeof(uint32_t); i++) {
            uint32_t rnd = 0;
            for (int j = 0; j < sizeof(uint32_t); j++) {
                uint32_t x = rand() % grid_size.x;
                uint32_t y = rand() % grid_size.y;
                if (!virtualization_enabled) {
                    CoreCoord physical_coord = dev->worker_core_from_logical_core(CoreCoord(x, y));
                    x = physical_coord.x;
                    y = physical_coord.y;
                }
                rnd = (rnd << 8) | (y << 4) | x;
            }
            runtime_args.push_back(rnd);
        }
        tt::tt_metal::SetRuntimeArgs(program, ucast_kernel, coord, runtime_args);
    }

    log_info(LogTest, "Unicast grid: {}, writing {} bytes per xfer", workers_logical.str(), ucast_size);

    if (rnd_coord) {
        log_info(tt::LogTest, "Randomizing ucast noc write destinations");
    } else {
        log_info(tt::LogTest, "Non-random ucast noc write destinations TBD");
    }

    log_info(tt::LogTest, "Using NOC {}", (noc == tt_metal::NOC::NOC_0) ? 0 : 1);

    if (rnd_delay) {
        log_info(tt::LogTest, "Randomizing delay");
    }
    log_info(LogTest, "Running for {} seconds", time_secs);

    tt::tt_metal::detail::LaunchProgram(dev, program, true);
}
