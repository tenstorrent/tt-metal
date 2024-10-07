// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "watcher_fixture.hpp"
#include "test_utils.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "hostdevcommon/common_runtime_address_map.h"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher NOC sanitization.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

typedef enum sanitization_features {
    SanitizeAddress,
    SanitizeAlignmentL1,
    SanitizeAlignmentDRAM
} watcher_features_t;

void RunTestOnCore(WatcherFixture* fixture, Device* device, CoreCoord &core, bool is_eth_core, watcher_features_t feature, bool use_ncrisc = false) {
    // Set up program
    auto program = CreateScopedProgram();
    CoreCoord phys_core;
    if (is_eth_core)
        phys_core = device->ethernet_core_from_logical_core(core);
    else
        phys_core = device->worker_core_from_logical_core(core);
    log_info(LogTest, "Running test on device {} core {}...", device->id(), phys_core.str());

    // Set up dram buffers
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 50;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t l1_buffer_addr = 400 * 1024;


    tt_metal::InterleavedBufferConfig dram_config{
                            .device=device,
                            .size = dram_buffer_size,
                            .page_size = dram_buffer_size,
                            .buffer_type = tt_metal::BufferType::DRAM
                            };
    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();
    log_info("Input DRAM: {}", input_dram_noc_xy);
    log_info("Output DRAM: {}", output_dram_noc_xy);

    // A DRAM copy kernel, we'll feed it incorrect inputs to test sanitization.
    KernelHandle dram_copy_kernel;
    if (is_eth_core) {
        std::map<string, string> dram_copy_kernel_defines = {
        {"SIGNAL_COMPLETION_TO_DISPATCHER", "1"},
    };
    dram_copy_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_0,
            .defines=dram_copy_kernel_defines
        }
    );
    } else {
    std::map<string, string> dram_copy_kernel_defines = {
        {"SIGNAL_COMPLETION_TO_DISPATCHER", "1"},
    };
    dram_copy_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = (use_ncrisc) ? tt_metal::DataMovementProcessor::RISCV_1 : tt_metal::DataMovementProcessor::RISCV_0,
            .noc = (use_ncrisc) ? tt_metal::NOC::RISCV_1_default : tt_metal::NOC::RISCV_0_default,
            .defines=dram_copy_kernel_defines
        }
    );
    }

    // Write to the input DRAM buffer
    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::detail::WriteToBuffer(input_dram_buffer, input_vec);

    // Write runtime args - update to a core that doesn't exist or an improperly aligned address,
    // depending on the flags passed in.
    switch(feature) {
        case SanitizeAddress:
            output_dram_noc_xy.x = 16;
            output_dram_noc_xy.y = 16;
            break;
        case SanitizeAlignmentL1:
            l1_buffer_addr += 16; // This is illegal because reading DRAM->L1 needs DRAM alignment
                                  // requirements (32 byte aligned).
            break;
        case SanitizeAlignmentDRAM:
            input_dram_buffer_addr++;
            break;
        default:
            log_warning(LogTest, "Unrecognized feature to test ({}), skipping...", feature);
            GTEST_SKIP();
            break;
    }
    tt_metal::SetRuntimeArgs(
        program,
        dram_copy_kernel,
        core,
        {l1_buffer_addr,
        input_dram_buffer_addr,
        (std::uint32_t)input_dram_noc_xy.x,
        (std::uint32_t)input_dram_noc_xy.y,
        output_dram_buffer_addr,
        (std::uint32_t)output_dram_noc_xy.x,
        (std::uint32_t)output_dram_noc_xy.y,
        dram_buffer_size});

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
    switch(feature) {
        case SanitizeAddress:
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) phys(x={:2},y={:2}): {} using noc0 tried to unicast write 102400 bytes from local L1[{:#08x}] to Unknown core w/ physical coords {} [addr=0x{:08x}] (NOC target address did not map to any known Tensix/Ethernet/DRAM/PCIE core).",
                device->id(),
                (is_eth_core) ? "ethnet" : "worker",
                core.x, core.y, phys_core.x, phys_core.y,
                (is_eth_core) ? "erisc" : "brisc", l1_buffer_addr, output_dram_noc_xy.str(),
                output_dram_buffer_addr
            );
            break;
        case SanitizeAlignmentL1:
        case SanitizeAlignmentDRAM:
            {
            // NoC-1 has a different coordinate for the same DRAM
            const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device->id());
            int noc = (use_ncrisc) ? 1 : 0;
            CoreCoord target_phys_core = {
                NOC_0_X(noc, soc_d.grid_size.x, input_dram_noc_xy.x),
                NOC_0_Y(noc, soc_d.grid_size.y, input_dram_noc_xy.y)
            };
            string risc_name = (is_eth_core) ? "erisc" : "brisc";
            if (use_ncrisc)
                risc_name = "ncrisc";
            expected = fmt::format(
                "Device {} {} core(x={:2},y={:2}) phys(x={:2},y={:2}): {} using noc{} tried to unicast read 102400 bytes to local L1[{:#08x}] from DRAM core w/ physical coords {} DRAM[addr=0x{:08x}] (invalid address alignment in NOC transaction).",
                device->id(),
                (is_eth_core) ? "ethnet" : "worker",
                core.x, core.y, phys_core.x, phys_core.y,
                risc_name, noc, l1_buffer_addr, target_phys_core,
                input_dram_buffer_addr
            );
            }
            break;
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
    EXPECT_TRUE(get_watcher_exception_message() == expected);
}

static void RunTestEth(WatcherFixture* fixture, Device* device) {
    // Run on the first ethernet core (if there are any).
    if (device->get_active_ethernet_cores(true).empty()) {
        log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
        GTEST_SKIP();
    }
    CoreCoord core = *(device->get_active_ethernet_cores(true).begin());
    RunTestOnCore(fixture, device, core, true, SanitizeAddress);
}

static void RunTestIEth(WatcherFixture* fixture, Device* device) {
    // Run on the first ethernet core (if there are any).
    if (device->get_inactive_ethernet_cores().empty()) {
        log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
        GTEST_SKIP();
    }
    CoreCoord core = *(device->get_inactive_ethernet_cores().begin());
    RunTestOnCore(fixture, device, core, true, SanitizeAddress);
}

// Run tests for host-side sanitization (uses functions that are from watcher_server.hpp).
void CheckHostSanitization(Device *device) {
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

TEST_F(WatcherFixture, TestWatcherSanitize) {
    // Skip this test for slow dipatch for now. Due to how llrt currently sits below device, it's
    // tricky to check watcher server status from the finish loop for slow dispatch. Once issue #4363
    // is resolved, we should add a check for print server handing in slow dispatch as well.
    if (this->slow_dispatch_)
        GTEST_SKIP();

    CheckHostSanitization(this->devices_[0]);

    // Only run on device 0 because this test takes down the watcher server.
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeAddress);
        },
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherSanitizeAlignmentL1) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeAlignmentL1);
        },
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherSanitizeAlignmentDRAM) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeAlignmentDRAM);
        },
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherSanitizeAlignmentDRAMNCrisc) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(
        [](WatcherFixture *fixture, Device *device){
            CoreCoord core{0, 0};
            RunTestOnCore(fixture, device, core, false, SanitizeAlignmentDRAM, true);
        },
        this->devices_[0]
    );
}

TEST_F(WatcherFixture, TestWatcherSanitizeEth) {
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(RunTestEth, this->devices_[0]);
}

TEST_F(WatcherFixture, TestWatcherSanitizeIEth) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    if (this->slow_dispatch_)
        GTEST_SKIP();
    this->RunTestOnDevice(RunTestIEth, this->devices_[0]);
}
