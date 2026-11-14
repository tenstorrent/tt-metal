// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared fixture for Keraunos CCE tests on the MMK emulator: one MeshDevice per suite (opening it costs
// about 90 s), helpers for the address views a CCE can use, host staging and readback of the CCE's own
// SRAM, a launch that reports the kernel status block on failure, and the IDMA copy kernel driver.

#pragma once

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "dev_mem_map.h"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "mimir_contract.hpp"

namespace tt::tt_metal::keraunos_test {

using namespace mimir_contract;

constexpr const char* kKernelDir = "tests/tt_metal/tt_metal/test_kernels/misc/keraunos/";

// Global (package SPA) view of HSIO tile 0's TL1: the D2D BAR window 0x12_0400_0000 + tile * stride, plus
// the TL1 offset 0x280_0000. bringup_mmk.py programs identity DMRISC windows for these addresses.
// Phase 2 of keraunos_plan.md moves this into a host-written routes table.
constexpr uint64_t kHsioGlobalSramBase = 0x12'0680'0000ULL;

// Status words the kernels write (see keraunos_cce_idma_copy.cpp).
constexpr uint32_t kStarted = 0x1D3A0001;
constexpr uint32_t kIssued = 0x1D3A0002;
constexpr uint32_t kAcked = 0x1D3A0003;
constexpr uint32_t kStatusWords = 8;

// Layout inside the unreserved SRAM of the core under test.
constexpr uint32_t kStatusOffset = 0x0;
constexpr uint32_t kSrcOffset = 0x1000;
constexpr uint32_t kDstOffset = 0x3000;
constexpr uint32_t kRegionBytes = 0x2000;

struct Hsio {
    CoreCoord logical;
    uint32_t hsio = 0;
    uint32_t eth_cce = 0;
};

class KeraunosCceTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!package_has_ethernet_cores() || !device_transport_configured()) {
            return;
        }
        mesh_ = distributed::MeshDevice::create_unit_mesh(0);
    }

    static void TearDownTestSuite() { mesh_.reset(); }

    void SetUp() override {
        skip_unless_ethernet_package();
        ASSERT_NE(mesh_, nullptr);
        device_ = &slow_dispatch::physical_device_from_unit_mesh(*mesh_);
        const auto& hal = MetalContext::instance().hal();
        unreserved_ = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
        unreserved_noc_ = hal.get_dev_noc_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    }

    // Descriptor core (4, hsio) is HSIO hsio CCE 0, and (5, hsio) is its CCE 1.
    std::optional<Hsio> hsio_core(uint32_t hsio, uint32_t eth_cce = 0) const {
        const auto& chiplet = soc_desc_of(device_).get_arch_descriptor().get_keraunos();
        for (const auto& core : chiplet.cores) {
            if (core.hsio == hsio && core.eth_cce == eth_cce) {
                const CoreCoord logical =
                    soc_desc_of(device_).get_logical_ethernet_core_from_physical({core.coord.x, core.coord.y});
                return Hsio{logical, core.hsio, core.eth_cce};
            }
        }
        return std::nullopt;
    }

    // Address of `offset` (relative to the unreserved base) in the views a CCE can use.
    uint64_t local_addr(uint32_t offset) const { return unreserved_ + offset; }
    uint64_t global_addr(const Hsio& c, uint32_t offset) const {
        const auto& arch = soc_desc_of(device_).get_arch_descriptor();
        return kHsioGlobalSramBase + static_cast<uint64_t>(c.hsio) * *arch.get_keraunos().tile_stride +
               static_cast<uint64_t>(c.eth_cce) * arch.get_eth_l1_size() + unreserved_ + offset;
    }
    uint64_t chiplet_local_addr(const Hsio& c, uint32_t offset) const {
        const auto& arch = soc_desc_of(device_).get_arch_descriptor();
        const auto& chiplet = arch.get_keraunos();
        return *chiplet.local_window + static_cast<uint64_t>(c.hsio) * *chiplet.tile_stride +
               static_cast<uint64_t>(c.eth_cce) * arch.get_eth_l1_size() + unreserved_ + offset;
    }

    void write_words(const Hsio& c, uint32_t offset, const std::vector<uint32_t>& words) {
        write_eth_sram(device_, c.logical, unreserved_noc_ + offset, words.data(), words.size() * sizeof(uint32_t));
    }
    std::vector<uint32_t> read_words(const Hsio& c, uint32_t offset, uint32_t count) {
        std::vector<uint32_t> words(count, 0);
        read_eth_sram(device_, c.logical, unreserved_noc_ + offset, words.data(), count * sizeof(uint32_t));
        return words;
    }

    // Clears the status block and the destination, and fills the source with a pattern tagged by `tag`.
    std::vector<uint32_t> stage(const Hsio& c, uint32_t tag, uint32_t bytes, uint32_t status_offset = kStatusOffset) {
        std::vector<uint32_t> pattern(bytes / sizeof(uint32_t));
        for (uint32_t i = 0; i < pattern.size(); i++) {
            pattern[i] = (tag << 16) | i;
        }
        write_words(c, status_offset, std::vector<uint32_t>(kStatusWords, 0));
        write_words(c, kDstOffset, std::vector<uint32_t>(bytes / sizeof(uint32_t), 0));
        write_words(c, kSrcOffset, pattern);
        return pattern;
    }

    // Launches and, if the launch throws (operation timeout), reports the status block before failing.
    void launch(Program& program, const Hsio& c, uint32_t status_offset = kStatusOffset) {
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()), std::move(program));
        try {
            distributed::EnqueueMeshWorkload(mesh_->mesh_command_queue(), workload, /*blocking=*/true);
        } catch (const std::exception& e) {
            const auto status = read_words(c, status_offset, kStatusWords);
            FAIL() << "launch failed: " << e.what() << "\nstatus: " << fmt::format("{:#010x}", fmt::join(status, " "));
        }
    }

    static std::vector<uint32_t> split(uint64_t v) { return {static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32)}; }

    struct IdmaArgs {
        uint64_t src = 0;
        uint64_t dst = 0;
        uint32_t len = 0;
        bool write_vcs = true;
        bool wait_all = false;
        uint32_t copies = 1;
        bool helper = false;
        uint32_t cmdbuf = 0;
        uint32_t status_offset = kStatusOffset;
        DataMovementProcessor hart = DataMovementProcessor::RISCV_0;
    };

    void add_idma_copy(Program& program, const Hsio& c, const IdmaArgs& a) {
        const KernelHandle k = CreateKernel(
            program,
            std::string(kKernelDir) + "keraunos_cce_idma_copy.cpp",
            c.logical,
            EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_0, .processor = a.hart, .compile_args = {a.cmdbuf}});
        std::vector<uint32_t> args = split(a.src);
        const auto dst = split(a.dst);
        args.insert(args.end(), dst.begin(), dst.end());
        args.insert(
            args.end(),
            {a.len,
             a.write_vcs ? 1u : 0u,
             a.wait_all ? 1u : 0u,
             unreserved_ + a.status_offset,
             a.copies,
             a.helper ? 1u : 0u});
        SetRuntimeArgs(program, k, c.logical, args);
    }

    // Runs one IDMA configuration on HSIO 0 CCE 0 and checks the copied block and the status trail.
    void expect_idma_copy(IdmaArgs a, uint32_t tag) {
        const auto core = hsio_core(0);
        ASSERT_TRUE(core.has_value());
        const uint32_t bytes = a.len * a.copies;
        const auto pattern = stage(*core, tag, bytes);
        Program program = CreateProgram();
        add_idma_copy(program, *core, a);
        launch(program, *core);
        const auto status = read_words(*core, kStatusOffset, kStatusWords);
        EXPECT_EQ(status[0], kStarted);
        EXPECT_EQ(status[2], kIssued);
        EXPECT_EQ(status[4], kAcked) << fmt::format("status: {:#010x}", fmt::join(status, " "));
        EXPECT_EQ(read_words(*core, kDstOffset, bytes / sizeof(uint32_t)), pattern);
    }

    static inline std::shared_ptr<distributed::MeshDevice> mesh_;
    IDevice* device_ = nullptr;
    uint32_t unreserved_ = 0;
    uint64_t unreserved_noc_ = 0;
};


}  // namespace tt::tt_metal::keraunos_test
