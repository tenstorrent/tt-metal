// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Watcher on Keraunos and Mimir cores. Run with TT_METAL_WATCHER=1 (and TT_METAL_SLOW_DISPATCH_MODE=1); without the
// watcher these tests skip. The watcher sanitizes every NOC address a core or the host uses, so these cover the
// native CCE1 view (CCE1's SRAM one 4 MiB window above CCE0's), the uncached alias, and Mimir CCE traffic.

#include "keraunos_cce_fixture.hpp"

#include <gtest/gtest.h>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "llrt/sanitize_noc_host.hpp"
#include "dev_mem_map.h"
#include "mimir_contract.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

namespace tt::tt_metal {
namespace {

using namespace keraunos_test;
using namespace mimir_contract;

bool watcher_enabled() { return MetalContext::instance().rtoptions().get_watcher_enabled(); }

class KeraunosWatcher : public KeraunosCceTest {
protected:
    void SetUp() override {
        if (!watcher_enabled()) {
            GTEST_SKIP() << "Set TT_METAL_WATCHER=1 to run the watcher tests.";
        }
        KeraunosCceTest::SetUp();
    }
};

// Opening the device loads firmware onto both CCEs of every tile. The host sanitizer must accept CCE1's native
// addresses, and still reject an address past both CCEs.
TEST_F(KeraunosWatcher, HostSanitizerAcceptsBothCcesAndRejectsBeyond) {
    const auto& hal = MetalContext::instance().hal();
    const uint64_t window = hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::BASE);
    EXPECT_TRUE(debug_valid_eth_addr(hal, 0x8080, 64)) << "CCE0 firmware";
    EXPECT_TRUE(debug_valid_eth_addr(hal, window + 0x8080, 64)) << "CCE1 native firmware";
    EXPECT_FALSE(debug_valid_eth_addr(hal, 2 * window - 32, 64)) << "straddles the end of CCE1";
    // 8 bytes: a 4-byte access can land in the register aperture, which is also valid.
    EXPECT_FALSE(debug_valid_eth_addr(hal, 2 * window + 0x100000, 8)) << "past both CCEs";
}

// A CCE0 kernel runs and stores its own SRAM with the watcher enabled.
TEST_F(KeraunosWatcher, Cce0KernelRuns) {
    const auto cce0 = hsio_core(0, 0);
    ASSERT_TRUE(cce0.has_value());
    constexpr uint32_t kBytes = 256;
    std::vector<uint32_t> data(kBytes / 4);
    for (uint32_t i = 0; i < data.size(); i++) {
        data[i] = 0x57A00000u | i;
    }
    write_words(*cce0, kSrcOffset, data);
    write_words(*cce0, kDstOffset, std::vector<uint32_t>(data.size(), 0));
    write_words(*cce0, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    add_idma_copy(
        program,
        *cce0,
        {.src = global_addr(*cce0, kSrcOffset),
         .dst = global_addr(*cce0, kDstOffset),
         .len = kBytes,
         .hart = DataMovementProcessor::RISCV_0});
    launch(program, *cce0);
    EXPECT_EQ(read_words(*cce0, kStatusOffset, kStatusWords)[4], kAcked);
    EXPECT_EQ(read_words(*cce0, kDstOffset, data.size()), data);
}

// A CCE1 kernel runs and its NOC copy reaches CCE0's SRAM.
TEST_F(KeraunosWatcher, Cce1KernelCopiesFromCce0) {
    const auto cce0 = hsio_core(0, 0);
    const auto cce1 = hsio_core(0, 1);
    ASSERT_TRUE(cce0.has_value() && cce1.has_value());
    constexpr uint32_t kBytes = 256;
    std::vector<uint32_t> data(kBytes / 4);
    for (uint32_t i = 0; i < data.size(); i++) {
        data[i] = 0x57A10000u | i;
    }
    write_words(*cce0, kSrcOffset, data);
    write_words(*cce1, kDstOffset, std::vector<uint32_t>(data.size(), 0));
    write_words(*cce1, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    Program program = CreateProgram();
    add_idma_copy(
        program,
        *cce1,
        {.src = global_addr(*cce0, kSrcOffset),
         .dst = global_addr(*cce1, kDstOffset),
         .len = kBytes,
         .hart = DataMovementProcessor::RISCV_0});
    launch(program, *cce1);
    EXPECT_EQ(read_words(*cce1, kStatusOffset, kStatusWords)[4], kAcked);
    EXPECT_EQ(read_words(*cce1, kDstOffset, data.size()), data);
}

// Both CCEs of a tile run kernels at once.
TEST_F(KeraunosWatcher, BothCcesRunTogether) {
    const auto cce0 = hsio_core(0, 0);
    const auto cce1 = hsio_core(0, 1);
    ASSERT_TRUE(cce0.has_value() && cce1.has_value());
    constexpr uint32_t kBytes = 128;
    std::vector<uint32_t> data0(kBytes / 4), data1(kBytes / 4);
    for (uint32_t i = 0; i < data0.size(); i++) {
        data0[i] = 0x57B00000u | i;
        data1[i] = 0x57B10000u | i;
    }
    for (const auto& [core, data] : {std::pair{*cce0, data0}, std::pair{*cce1, data1}}) {
        write_words(core, kSrcOffset, data);
        write_words(core, kDstOffset, std::vector<uint32_t>(data.size(), 0));
        write_words(core, kStatusOffset, std::vector<uint32_t>(kStatusWords, 0));
    }
    Program program = CreateProgram();
    for (const Hsio& core : {*cce0, *cce1}) {
        add_idma_copy(
            program,
            core,
            {.src = global_addr(core, kSrcOffset),
             .dst = global_addr(core, kDstOffset),
             .len = kBytes,
             .hart = DataMovementProcessor::RISCV_0});
    }
    launch(program, *cce0);
    EXPECT_EQ(read_words(*cce0, kDstOffset, data0.size()), data0);
    EXPECT_EQ(read_words(*cce1, kDstOffset, data1.size()), data1);
}

// A Mimir CCE kernel runs (hart 0 stores through its uncached alias) with the watcher enabled.
TEST(MimirWatcher, CceKernelWritesOwnSram) {
    if (!watcher_enabled()) {
        GTEST_SKIP() << "Set TT_METAL_WATCHER=1 to run the watcher tests.";
    }
    skip_unless_programmable_dram_package();
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    IDevice* device = &slow_dispatch::physical_device_from_unit_mesh(*mesh_device);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);

    const CoreCoord core = logical_mimir_cce_core(soc_desc_of(device), 0);
    const uint32_t magic = 0x57A2CAFE;
    const auto& hal = MetalContext::instance().hal();
    const uint32_t result_dev_addr =
        MEM_L1_UNCACHED_BASE + hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint64_t result_noc_addr = hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dram_write_one_uint32.cpp",
        core,
        DramConfig{.noc = NOC::NOC_0, .compile_args = {result_dev_addr, magic}});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);

    uint32_t result = 0;
    read_mimir_cce_sram(device, core, result_noc_addr, &result, sizeof(result));
    EXPECT_EQ(result, magic);
}

}  // namespace
}  // namespace tt::tt_metal
