// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This test stresses NOC mcast by:
//  - using 1 mcast core (future work to add multiple) either tensix or eth
//  - rapidly mcast into a grid of tensix workers
//  - rapidly grid of tensix workers generates random noc traffic
//  - does not verify correct transactions, just runs til termination

#include <fmt/base.h>
#include <stdlib.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
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
#include "umd/device/types/xy_pair.h"

using namespace tt;

const uint32_t CB_ELEMENTS = 2048;
const uint32_t DEFAULT_SECONDS = 10;
const uint32_t DEFAULT_TARGET_WIDTH = 1;
const uint32_t DEFAULT_TARGET_HEIGHT = 1;
const uint32_t N_RANDS = 512;

uint32_t device_num_g = 0;
uint32_t time_secs_g = DEFAULT_SECONDS;
uint32_t tlx_g = 0;
uint32_t tly_g = 0;
uint32_t width_g = DEFAULT_TARGET_WIDTH;
uint32_t height_g = DEFAULT_TARGET_HEIGHT;
uint32_t mcast_x_g = 0;
uint32_t mcast_y_g = 0;
uint32_t mcast_size_g = 16;
uint32_t ucast_size_g = 8192;
uint32_t mcast_from_n_eth_g;
bool mcast_from_eth_g;
bool ucast_only = false;
bool rnd_delay_g = false;
bool rnd_coord_g = true;
tt_metal::NOC noc_g = tt_metal::NOC::NOC_0;

void init(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") || test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "     -v: device number to run on (default 0) ", DEFAULT_SECONDS);
        log_info(LogTest, "     -t: time in seconds (default {})", DEFAULT_SECONDS);
        log_info(LogTest, "     -x: grid top left x");
        log_info(LogTest, "     -y: grid top left y");
        log_info(LogTest, "-width: unicast grid width (default {})", DEFAULT_TARGET_WIDTH);
        log_info(LogTest, "-height: unicast grid height (default {})", DEFAULT_TARGET_HEIGHT);
        log_info(LogTest, "    -mx: mcast core x");
        log_info(LogTest, "    -my: mcast core y");
        log_info(LogTest, "     -e: mcast from nth idle eth core (ignores -mx,-my)");
        log_info(LogTest, "     -m: mcast packet size");
        log_info(LogTest, "     -u: ucast packet size");
        log_info(LogTest, "     -ucast-only: skip multicasting");
        log_info(LogTest, "-rdelay: insert random delay between noc transactions");
        log_info(LogTest, "     -n: noc to use (default 0)");
        log_info(LogTest, "     -s: seed random number generator");
        exit(0);
    }

    device_num_g = test_args::get_command_option_uint32(input_args, "-v", 0);
    time_secs_g = test_args::get_command_option_uint32(input_args, "-t", DEFAULT_SECONDS);
    tlx_g = test_args::get_command_option_uint32(input_args, "-x", 0);
    tly_g = test_args::get_command_option_uint32(input_args, "-y", 0);
    width_g = test_args::get_command_option_uint32(input_args, "-width", DEFAULT_TARGET_WIDTH);
    height_g = test_args::get_command_option_uint32(input_args, "-height", DEFAULT_TARGET_HEIGHT);
    mcast_x_g = test_args::get_command_option_uint32(input_args, "-mx", 0);
    mcast_y_g = test_args::get_command_option_uint32(input_args, "-my", 0);
    mcast_from_n_eth_g = test_args::get_command_option_uint32(input_args, "-e", 0xffff);
    mcast_size_g = test_args::get_command_option_uint32(input_args, "-m", 16);
    ucast_size_g = test_args::get_command_option_uint32(input_args, "-u", 8192);
    mcast_from_eth_g = (mcast_from_n_eth_g != 0xffff);
    ucast_only = test_args::has_command_option(input_args, "-ucast-only");
    rnd_delay_g = test_args::has_command_option(input_args, "-rdelay");
    uint32_t noc = test_args::get_command_option_uint32(input_args, "-n", 0);
    noc_g = (noc == 0) ? tt_metal::NOC::NOC_0 : tt_metal::NOC::NOC_1;
    uint32_t seed = test_args::get_command_option_uint32(input_args, "-s", 0);
    srand(seed);

    if (mcast_from_eth_g && ucast_only) {
        log_fatal(tt::LogTest, "Cannot request both mcast from eth and ucast only");
    }

    if (!ucast_only && !mcast_from_eth_g && mcast_x_g >= tlx_g && mcast_x_g <= tlx_g + width_g - 1 &&
        mcast_y_g >= tly_g && mcast_y_g <= tly_g + height_g - 1) {
        log_fatal(tt::LogTest, "Mcast core can't be within mcast grid");
        exit(-1);
    }
}

int main(int argc, char** argv) {
    init(argc, argv);

    tt_metal::IDevice* device = tt_metal::CreateDevice(device_num_g);
    tt_metal::Program program = tt_metal::CreateProgram();

    const auto& eth_cores = device->get_inactive_ethernet_cores();

    CoreRange workers_logical({tlx_g, tly_g}, {tlx_g + width_g - 1, tly_g + height_g - 1});
    CoreCoord mcast_logical(mcast_x_g, mcast_y_g);
    CoreCoord tl_core = device->worker_core_from_logical_core({tlx_g, tly_g});

    if (mcast_from_eth_g) {
        CoreCoord eth_logical(0, mcast_from_n_eth_g);
        bool found = false;
        for (const auto& eth_core : eth_cores) {
            if (eth_logical == eth_core) {
                found = true;
                break;
            }
        }
        if (!found) {
            log_fatal(tt::LogTest, "{} not found in the list of idle eth cores", mcast_from_n_eth_g);
            tt_metal::CloseDevice(device);
            exit(-1);
        }
        mcast_logical = eth_logical;
    }

    CoreCoord mcast_end = device->worker_core_from_logical_core(workers_logical.end_coord);
    bool virtualization_enabled = tt::tt_metal::MetalContext::instance().hal().is_coordinate_virtualization_enabled();
    uint32_t num_dests = workers_logical.size();
    CoreCoord virtual_offset = virtualization_enabled
                                   ? device->worker_core_from_logical_core({0, 0})
                                   : CoreCoord(0, 0);  // In this case pass physical coordinates as runtime args

    std::vector<uint32_t> compile_args = {
        false,
        tl_core.x,
        tl_core.y,
        mcast_end.x,
        mcast_end.y,
        num_dests,
        time_secs_g,
        ucast_size_g,
        mcast_size_g,
        virtual_offset.x,
        virtual_offset.y,
        N_RANDS,
        rnd_delay_g,
        device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1),
        mcast_from_eth_g ? tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
                               HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED)
                         : device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1),
    };

    KernelHandle ucast_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/stress_noc_mcast.cpp",
        workers_logical,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = noc_g,
            .compile_args = compile_args,
        });

    for (CoreCoord coord : workers_logical) {
        std::vector<uint32_t> runtime_args;
        // Not particularly random since all cores are getting the same data
        // N_RANDS in bytes
        CoreCoord grid_size = device->logical_grid_size();
        for (int i = 0; i < N_RANDS / sizeof(uint32_t); i++) {
            uint32_t rnd = 0;
            for (int j = 0; j < sizeof(uint32_t); j++) {
                uint32_t x = rand() % grid_size.x;
                uint32_t y = rand() % grid_size.y;
                if (!virtualization_enabled) {
                    CoreCoord physical_coord = device->worker_core_from_logical_core(CoreCoord(x, y));
                    x = physical_coord.x;
                    y = physical_coord.y;
                }
                rnd = (rnd << 8) | (y << 4) | x;
            }
            runtime_args.push_back(rnd);
        }
        tt::tt_metal::SetRuntimeArgs(program, ucast_kernel, coord, runtime_args);
    }

    if (!ucast_only) {
        compile_args[0] = true;
        KernelHandle mcast_kernel;
        if (mcast_from_eth_g) {
            mcast_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/stress_noc_mcast.cpp",
                mcast_logical,
                tt_metal::EthernetConfig{
                    .eth_mode = Eth::IDLE,
                    .noc = noc_g,
                    .compile_args = compile_args,
                });
        } else {
            mcast_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/stress_noc_mcast.cpp",
                mcast_logical,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = noc_g,
                    .compile_args = compile_args,
                });
        }

        std::vector<uint32_t> runtime_args;
        for (int i = 0; i < 128; i++) {
            runtime_args.push_back(rand());
        }
        tt::tt_metal::SetRuntimeArgs(program, mcast_kernel, mcast_logical, runtime_args);

        CoreCoord mcast_virtual;
        CoreCoord mcast_physical;
        if (mcast_from_eth_g) {
            mcast_virtual = device->ethernet_core_from_logical_core(mcast_logical);
            mcast_physical = tt::tt_metal::MetalContext::instance()
                                 .get_cluster()
                                 .get_soc_desc(device_num_g)
                                 .get_physical_ethernet_core_from_logical(mcast_logical);
        } else {
            mcast_virtual = device->worker_core_from_logical_core(mcast_logical);
            mcast_physical = tt::tt_metal::MetalContext::instance()
                                 .get_cluster()
                                 .get_soc_desc(device_num_g)
                                 .get_physical_tensix_core_from_logical(mcast_logical);
        }

        log_info(
            LogTest,
            "MCast {} core: {}, virtual {}, physical {}, writing {} bytes per xfer",
            mcast_from_eth_g ? "ETH" : "TENSIX",
            mcast_logical,
            mcast_virtual,
            mcast_physical,
            mcast_size_g);
    }

    log_info(LogTest, "Unicast grid: {}, writing {} bytes per xfer", workers_logical.str(), ucast_size_g);

    if (rnd_coord_g) {
        log_info(tt::LogTest, "Randomizing ucast noc write destinations");
    } else {
        log_info(tt::LogTest, "Non-random ucast noc write destinations TBD");
    }

    log_info(tt::LogTest, "Using NOC {}", (noc_g == tt_metal::NOC::NOC_0) ? 0 : 1);

    if (rnd_delay_g) {
        log_info(tt::LogTest, "Randomizing delay");
    }
    log_info(LogTest, "Running for {} seconds", time_secs_g);

    tt::tt_metal::detail::LaunchProgram(device, program, true);
    tt_metal::CloseDevice(device);
}
