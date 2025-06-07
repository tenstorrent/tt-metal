// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <magic_enum/magic_enum.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
// Do we really want to expose Hal like this?
// This looks like an API level test
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>
#include "watcher_server.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher NOC sanitization.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

enum watcher_features_t {
    SanitizeAddress,
    SanitizeAlignmentL1Write,
    SanitizeAlignmentL1Read,
    SanitizeZeroL1Write,
    SanitizeMailboxWrite,
    SanitizeInlineWriteDram,
};

tt::tt_metal::HalMemType get_buffer_mem_type_for_test(watcher_features_t feature) {
    return feature == watcher_features_t::SanitizeInlineWriteDram ? tt_metal::HalMemType::DRAM
                                                                  : tt_metal::HalMemType::L1;
}

tt::tt_metal::BufferType get_buffer_type_for_test(watcher_features_t feature) {
    return feature == watcher_features_t::SanitizeInlineWriteDram ? tt_metal::BufferType::DRAM
                                                                  : tt_metal::BufferType::L1;
}

uint32_t get_address_for_test(bool use_eth_core, tt::tt_metal::HalL1MemAddrType type, bool high_address = false) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    if (use_eth_core) {
        const auto active_eth_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, type);
        const auto idle_eth_addr = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, type);
        if (high_address) {
            return std::max(active_eth_addr, idle_eth_addr);
        } else {
            return std::min(active_eth_addr, idle_eth_addr);
        }
    } else {
        return hal.get_dev_addr(HalProgrammableCoreType::TENSIX, type);
    }
}

CoreCoord get_core_coord_for_test(const std::shared_ptr<tt::tt_metal::Buffer>& buffer) {
    if (buffer->is_l1()) {
        return buffer->device()->worker_core_from_logical_core(buffer->allocator()->get_logical_core_from_bank_id(0));
    } else {
        auto logical_dram_core = buffer->device()->logical_core_from_dram_channel(0);
        return buffer->device()->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
    }
}

void RunTestOnCore(WatcherFixture* fixture, IDevice* device, CoreCoord &core, bool is_eth_core, watcher_features_t feature, bool use_ncrisc = false) {
    // It's not simple to check the watcher server status from the finish loop for slow dispatch, so just run these
    // tests in FD.
    if (fixture->IsSlowDispatch()) {
        GTEST_SKIP();
    }

    // Set up program
    Program program = Program();
    CoreCoord virtual_core;
    if (is_eth_core) {
        virtual_core = device->ethernet_core_from_logical_core(core);
    } else {
        virtual_core = device->worker_core_from_logical_core(core);
    }
    log_info(LogTest, "Running test on device {} core {}...", device->id(), virtual_core.str());

    // Set up dram buffers
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 50;
    auto buffer_mem_type = get_buffer_mem_type_for_test(feature);
    uint32_t buffer_size = single_tile_size * num_tiles;
    auto config_buffer_type = get_buffer_type_for_test(feature);
    tt_metal::InterleavedBufferConfig local_buffer_config{
        .device = device, .size = buffer_size, .page_size = buffer_size, .buffer_type = tt::tt_metal::BufferType::L1};
    auto local_scratch_buffer = CreateBuffer(local_buffer_config);
    uint32_t buffer_addr = local_scratch_buffer->address();
    // For ethernet core, need to have smaller buffer and force buffer to be at a different address
    if (is_eth_core) {
        buffer_size = 1024;
        buffer_addr = get_address_for_test(true, HalL1MemAddrType::UNRESERVED, true);
    }

    tt_metal::InterleavedBufferConfig buffer_config{
        .device = device, .size = buffer_size, .page_size = buffer_size, .buffer_type = config_buffer_type};
    auto input_buffer = CreateBuffer(buffer_config);
    uint32_t input_buffer_addr = input_buffer->address();

    auto output_buffer = CreateBuffer(buffer_config);
    uint32_t output_buffer_addr = output_buffer->address();

    auto input_buf_noc_xy = get_core_coord_for_test(input_buffer);
    auto output_buf_noc_xy = get_core_coord_for_test(output_buffer);
    log_info(tt::LogTest, "Input/Output Buffer mem type: {}", magic_enum::enum_name(buffer_mem_type));
    log_info(tt::LogTest, "Input Buffer NOC XY: {}", input_buf_noc_xy);
    log_info(tt::LogTest, "Output Buffer NOC XY: {}", output_buf_noc_xy);
    log_info(tt::LogTest, "Local scratch buffer addr: {:#x}", buffer_addr);

    // A copy kernel, we'll feed it incorrect inputs to test sanitization.
    KernelHandle dram_copy_kernel;
    if (is_eth_core) {
        std::map<string, string> dram_copy_kernel_defines = {
            {"SIGNAL_COMPLETION_TO_DISPATCHER", "1"},
        };
        dram_copy_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_to_noc_coord.cpp",
            core,
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0, .defines = dram_copy_kernel_defines});
    } else {
        std::map<string, string> dram_copy_kernel_defines = {
            {"SIGNAL_COMPLETION_TO_DISPATCHER", "1"},
        };
        dram_copy_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_to_noc_coord.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor =
                    (use_ncrisc) ? tt_metal::DataMovementProcessor::RISCV_1 : tt_metal::DataMovementProcessor::RISCV_0,
                .noc = (use_ncrisc) ? tt_metal::NOC::RISCV_1_default : tt_metal::NOC::RISCV_0_default,
                .defines = dram_copy_kernel_defines});
    }

    // Write to the input buffer
    std::vector<uint32_t> input_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::detail::WriteToBuffer(input_buffer, input_vec);

    // Write runtime args - update to a core that doesn't exist or an improperly aligned address,
    // depending on the flags passed in.
    bool use_inline_dw_write = false;
    switch(feature) {
        case SanitizeAddress:
            output_buf_noc_xy.x = 26;
            output_buf_noc_xy.y = 18;
            break;
        case SanitizeAlignmentL1Write:
            output_buffer_addr++;  // This is illegal because reading DRAM->L1 needs DRAM alignment
                                   // requirements (32 byte aligned).
            buffer_size--;
            break;
        case SanitizeAlignmentL1Read:
            input_buffer_addr++;
            buffer_size--;
            break;
        case SanitizeZeroL1Write: output_buffer_addr = 0; break;
        case SanitizeMailboxWrite:
            // This is illegal because we'd be writing to the mailbox memory
            buffer_addr = get_address_for_test(is_eth_core, HalL1MemAddrType::MAILBOX);
            break;
        case SanitizeInlineWriteDram: use_inline_dw_write = true; break;
        default:
            log_warning(LogTest, "Unrecognized feature to test ({}), skipping...", feature);
            GTEST_SKIP();
            break;
    }

    tt_metal::SetRuntimeArgs(
        program,
        dram_copy_kernel,
        core,
        {buffer_addr,
         input_buffer_addr,
         input_buf_noc_xy.x,
         input_buf_noc_xy.y,
         output_buffer_addr,
         output_buf_noc_xy.x,
         output_buf_noc_xy.y,
         buffer_size,
         use_inline_dw_write});

    // Run the kernel, expect an exception here
    try {
        fixture->RunProgram(device, program);
    } catch (std::runtime_error& e) {
        string expected = "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.\n";
        expected += tt::watcher_get_log_file_name();
        const string error = string(e.what());
        log_info(tt::LogTest, "Caught exception (one is expected in this test)");
        EXPECT_TRUE(error.find(expected) != string::npos);
    }

    // We should be able to find the expected watcher error in the log as well.
    string expected;
    int noc = (use_ncrisc) ? 1 : 0;
    CoreCoord input_core_virtual_coords = device->virtual_noc0_coordinate(noc, input_buf_noc_xy);
    CoreCoord output_core_virtual_coords = device->virtual_noc0_coordinate(noc, output_buf_noc_xy);
    string risc_name = (is_eth_core) ? "erisc" : " brisc";
    if (use_ncrisc) {
        risc_name = "ncrisc";
    }
    switch(feature) {
        case SanitizeAddress:
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc0 tried to unicast write {} "
                "bytes from local L1[{:#08x}] to Unknown core w/ virtual coords {} [addr=0x{:08x}] (NOC target "
                "address did not map to any known Tensix/Ethernet/DRAM/PCIE core).",
                device->id(),
                (is_eth_core) ? "acteth" : "worker",
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                (is_eth_core) ? "erisc" : " brisc",
                buffer_size,
                buffer_addr,
                output_buf_noc_xy.str(),
                output_buffer_addr);
            break;
        case SanitizeAlignmentL1Write: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast write {} "
                "bytes from local L1[{:#08x}] to Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (invalid address "
                "alignment in NOC transaction).",
                device->id(),
                (is_eth_core) ? "acteth" : "worker",
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                input_core_virtual_coords,
                output_buffer_addr);
            break;
        }
        case SanitizeAlignmentL1Read: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast read {} "
                "bytes to local L1[{:#08x}] from Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (invalid address "
                "alignment in NOC transaction).",
                device->id(),
                (is_eth_core) ? "acteth" : "worker",
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                input_core_virtual_coords,
                input_buffer_addr);
        } break;
        case SanitizeZeroL1Write: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc{} tried to unicast write {} "
                "bytes from local L1[{:#08x}] to Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (NOC target "
                "overwrites mailboxes).",
                device->id(),
                (is_eth_core) ? "acteth" : "worker",
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                noc,
                buffer_size,
                buffer_addr,
                input_core_virtual_coords,
                output_buffer_addr);
        } break;
        case SanitizeMailboxWrite: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc0 tried to unicast read {} "
                "bytes to local L1[{:#08x}] from Tensix core w/ virtual coords {} L1[addr=0x{:08x}] (Local L1 "
                "overwrites mailboxes).",
                device->id(),
                (is_eth_core) ? "acteth" : "worker",
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                (is_eth_core) ? "erisc" : " brisc",
                buffer_size,
                buffer_addr,
                input_buf_noc_xy.str(),
                input_buffer_addr);
        } break;
        case SanitizeInlineWriteDram: {
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} using noc0 tried to unicast write 4 bytes "
                "from local L1[{:#08x}] to DRAM core w/ virtual coords {} DRAM[addr=0x{:08x}] (inline dw writes do not "
                "support DRAM destination addresses).",
                device->id(),
                (is_eth_core) ? "acteth" : "worker",
                core.x,
                core.y,
                virtual_core.x,
                virtual_core.y,
                risc_name,
                0,
                output_core_virtual_coords.str(),
                output_buffer_addr);
        } break;
        default:
            log_warning(LogTest, "Unrecognized feature to test ({}), skipping...", feature);
            GTEST_SKIP();
            break;
    }

    log_info(LogTest, "Expected error: {}", expected);
    std::string exception = "";
    do {
        exception = get_watcher_exception_message();
    } while (exception == "");
    log_info(LogTest, "Reported error: {}", exception);
    EXPECT_EQ(get_watcher_exception_message(), expected);
}

void RunTestEth(WatcherFixture* fixture, IDevice* device, watcher_features_t feature) {
    if (fixture->IsSlowDispatch()) {
        GTEST_SKIP();
    }
    // Run on the first ethernet core (if there are any).
    if (device->get_active_ethernet_cores(true).empty()) {
        log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
        GTEST_SKIP();
    }
    CoreCoord core = *(device->get_active_ethernet_cores(true).begin());
    RunTestOnCore(fixture, device, core, true, feature);
}

void RunTestIEth(WatcherFixture* fixture, IDevice* device, watcher_features_t feature) {
    if (fixture->IsSlowDispatch()) {
        GTEST_SKIP();
    }
    // Run on the first ethernet core (if there are any).
    if (device->get_inactive_ethernet_cores().empty()) {
        log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
        GTEST_SKIP();
    }
    CoreCoord core = *(device->get_inactive_ethernet_cores().begin());
    RunTestOnCore(fixture, device, core, true, feature);
}

// Run tests for host-side sanitization (uses functions that are from watcher_server.hpp).
void CheckHostSanitization(IDevice* device) {
    // Try reading from a core that doesn't exist
    constexpr CoreCoord core = {16, 16};
    uint64_t addr = 0;
    uint32_t sz_bytes = 4;
    try {
        llrt::read_hex_vec_from_core(device->id(), core, addr, sz_bytes);
    } catch (std::runtime_error& e) {
        const string expected = fmt::format("Host watcher: bad {} NOC coord {}\n", "read", core.str());
        const string error = string(e.what());
        log_info(tt::LogTest, "Caught exception (one is expected in this test)");
        EXPECT_TRUE(error.find(expected) != string::npos);
    }
}

TEST_F(WatcherFixture, TensixTestWatcherSanitize) {
    CheckHostSanitization(this->devices_[0]);

    // Only run on device 0 because this test takes down the watcher server.
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeAddress);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TensixTestWatcherSanitizeAlignmentL1Write) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeAlignmentL1Write);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TensixTestWatcherSanitizeAlignmentL1Read) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeAlignmentL1Read);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TensixTestWatcherSanitizeAlignmentL1ReadNCrisc) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeAlignmentL1Read, true);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TensixTestWatcherSanitizeZeroL1Write) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeZeroL1Write);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TensixTestWatcherSanitizeMailboxWrite) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeMailboxWrite);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, TensixTestWatcherSanitizeInlineWriteDram) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) {
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeInlineWriteDram);
        },
        this->devices_[0]);
}

TEST_F(WatcherFixture, ActiveEthTestWatcherSanitizeEth) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) { RunTestEth(fixture, device, SanitizeAddress); },
        this->devices_[0]);
}

TEST_F(WatcherFixture, ActiveEthTestWatcherSanitizeMailboxWrite) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) { RunTestEth(fixture, device, SanitizeMailboxWrite); },
        this->devices_[0]);
}

TEST_F(WatcherFixture, ActiveEthTestWatcherSanitizeInlineWriteDram) {
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) { RunTestEth(fixture, device, SanitizeInlineWriteDram); },
        this->devices_[0]);
}

TEST_F(WatcherFixture, IdleEthTestWatcherSanitizeIEth) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) { RunTestIEth(fixture, device, SanitizeAddress); },
        this->devices_[0]);
}

TEST_F(WatcherFixture, IdleEthTestWatcherSanitizeInlineWriteDram) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [](WatcherFixture* fixture, IDevice* device) { RunTestIEth(fixture, device, SanitizeInlineWriteDram); },
        this->devices_[0]);
}
