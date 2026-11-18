// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Harness routers on devices that share a CCE (SPEC-0012), on one mmk chip with fabric disabled.
// Run with TT_METAL_SLOW_DISPATCH_MODE=1 and TT_METAL_EMU_SOC_DESC pointing at mmk.yaml.

#include <gtest/gtest.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "dev_mem_map.h"
#include "tt_metal/tt_fabric/keraunos/vcluster/descriptors.hpp"
#include "tt_metal/tt_fabric/keraunos/vcluster/harness_run.hpp"

namespace tt::tt_metal {
namespace {

// The logical Ethernet core of CCE0 of HSIO tile @p tile on @p device, or nullopt when the descriptor does not list it.
std::optional<CoreCoord> cce0_of_tile(IDevice* device, uint32_t tile) {
    const auto& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    for (const auto& core : soc_desc.get_arch_descriptor().get_keraunos().cores) {
        if (core.hsio == tile && core.cce == 0) {
            return soc_desc.get_logical_ethernet_core_from_physical({core.coord.x, core.coord.y});
        }
    }
    return std::nullopt;
}

// eth_cce_send.h builds and links in an idle Ethernet kernel on hart 1 (SPEC-0012 § Verification 2, first step).
TEST(VclusterNode, EthCceSendBuildsAsIdleKernelOnHart1) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    IDevice* device = &slow_dispatch::physical_device_from_unit_mesh(*mesh);
    ASSERT_EQ(device->arch(), tt::ARCH::QUASAR);
    const auto core = cce0_of_tile(device, 0);
    ASSERT_TRUE(core.has_value()) << "The descriptor lists no CCE0 of HSIO tile 0.";

    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_fabric/keraunos/vcluster/kernels/vc_include_probe.cpp",
        *core,
        EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::NOC_1, .processor = DataMovementProcessor::RISCV_1});
    program.impl().compile(device);
}

namespace vcluster = tt::tt_fabric::vcluster;

bool is_mock() { return MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock; }

std::filesystem::path root_dir() { return MetalContext::instance().rtoptions().get_root_dir(); }

// The package's facts from mmk.yaml, with the harness's region floor.
vcluster::PackageFacts mmk_facts() {
    auto facts = vcluster::package_facts(root_dir() / "tt_metal/third_party/umd/tests/soc_descs/mmk.yaml");
    facts.unreserved_base = MEM_IERISC_ROUTER_UNRESERVED_BASE;
    return facts;
}

constexpr vcluster::RunShape kShape{.packets_per_flow = 5, .payload_bytes = 256};
constexpr auto kTimeout = std::chrono::minutes(30);

// The harness on a manifest of manifests/: on Mock, checks, plans and builds the node for every device and stops
// before launch, since MockChip reads and writes are no-ops; on the emulator, the whole run and its verdict.
class VclusterHarness : public ::testing::TestWithParam<std::string> {};

TEST_P(VclusterHarness, Run) {
    const auto manifest =
        vcluster::load_manifest(root_dir() / "tests/tt_metal/tt_fabric/keraunos/vcluster/manifests" / GetParam());
    const auto facts = mmk_facts();
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    ASSERT_EQ(slow_dispatch::physical_device_from_unit_mesh(*mesh).arch(), tt::ARCH::QUASAR);
    if (is_mock()) {
        ASSERT_TRUE(vcluster::check_manifest(manifest, facts).empty());
        ASSERT_TRUE(vcluster::check_harness_manifest(manifest, facts, kShape).empty());
        vcluster::build_harness_program(*mesh, vcluster::plan_harness(manifest, kShape), /*nonce=*/1);
        GTEST_SKIP() << "Mock: every node built. MockChip reads and writes are no-ops, so nothing launches.";
    }
    const auto result = vcluster::run_harness(*mesh, manifest, facts, kShape, kTimeout);
    for (std::size_t d = 0; d < result.nodes.size(); d++) {
        const auto& r = result.nodes[d];
        log_info(
            tt::LogTest,
            "device {}: status {:#x} postcode {:#x} detail {} originated {} delivered {} bad {} sent [{}] received [{}]",
            d,
            r.status,
            r.postcode,
            r.detail,
            r.originated,
            r.delivered,
            r.bad,
            fmt::join(r.sent, ", "),
            fmt::join(r.received, ", "));
    }
    EXPECT_FALSE(result.timed_out);
    EXPECT_TRUE(result.failures.empty()) << fmt::format("{}", fmt::join(result.failures, "\n"));
}

INSTANTIATE_TEST_SUITE_P(
    VclusterNode,
    VclusterHarness,
    ::testing::Values("pair_local.yaml", "mesh2x2.yaml"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        return info.param.substr(0, info.param.find('.'));
    });

}  // namespace
}  // namespace tt::tt_metal
