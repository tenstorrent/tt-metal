// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Harness routers on devices that share a CCE (SPEC-0012), on one mmk chip with fabric disabled.
// Run with TT_METAL_SLOW_DISPATCH_MODE=1 and TT_METAL_EMU_SOC_DESC pointing at mmk.yaml.

#include <gtest/gtest.h>

#include <optional>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "impl/program/program_impl.hpp"

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

}  // namespace
}  // namespace tt::tt_metal
