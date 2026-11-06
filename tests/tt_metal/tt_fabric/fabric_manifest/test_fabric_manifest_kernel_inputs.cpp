// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RouterArgs over hand-built kernel inputs. No device.

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_kernel_inputs.hpp"

namespace tt::tt_fabric::manifest {
namespace {

KernelInputs make_erisc_inputs(
    std::unordered_map<std::string, uint32_t> named_compile_args, std::map<std::string, std::string> defines = {}) {
    return {
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_0,
        .compile_args = {},
        .named_compile_args = std::move(named_compile_args),
        .defines = std::move(defines),
        .runtime_args = {}};
}

TEST(ManifestRouterArgs, RouterArgFailsWhenEriscsDisagree) {
    const std::vector<KernelInputs> kernel_inputs_by_erisc = {
        make_erisc_inputs({{"CHANNEL_BUFFER_SIZE", 4096}}), make_erisc_inputs({{"CHANNEL_BUFFER_SIZE", 2048}})};
    const RouterArgs args(kernel_inputs_by_erisc);

    EXPECT_EQ(args.read_erisc_arg(1, "CHANNEL_BUFFER_SIZE"), 2048);
    EXPECT_THROW(args.read_router_arg("CHANNEL_BUFFER_SIZE"), std::exception);
}

TEST(ManifestRouterArgs, ReadFailsWhenTheEriscWasNotGivenIt) {
    const std::vector<KernelInputs> kernel_inputs_by_erisc = {
        make_erisc_inputs({{"CHANNEL_BUFFER_SIZE", 4096}}), make_erisc_inputs({})};
    const RouterArgs args(kernel_inputs_by_erisc);

    EXPECT_EQ(args.read_erisc_arg(0, "CHANNEL_BUFFER_SIZE"), 4096);
    EXPECT_THROW(args.read_erisc_arg(1, "CHANNEL_BUFFER_SIZE"), std::exception);
    EXPECT_EQ(args.find_erisc_arg(0, "CHANNEL_BUFFER_SIZE"), 4096);
    EXPECT_EQ(args.find_erisc_arg(1, "CHANNEL_BUFFER_SIZE"), std::nullopt);
    EXPECT_THROW(args.read_erisc_arg(2, "CHANNEL_BUFFER_SIZE"), std::exception);
}

// MAX_NUM_VCS is fed to the router but listed as unrecorded, so the manifest may not read it.
TEST(ManifestRouterArgs, ReadFailsOnANameNoListNames) {
    const std::vector<KernelInputs> kernel_inputs_by_erisc = {
        make_erisc_inputs({{"MAX_NUM_VCS", 3}}, {{"FABRIC_2D", ""}})};
    const RouterArgs args(kernel_inputs_by_erisc);

    EXPECT_THROW(args.read_router_arg("MAX_NUM_VCS"), std::exception);
    EXPECT_THROW(args.find_erisc_arg(0, "MAX_NUM_VCS"), std::exception);
    EXPECT_THROW(args.has_define("FABRIC_2D"), std::exception);
}

TEST(ManifestRouterArgs, ServicingComesFromTheServicedFlags) {
    const std::vector<KernelInputs> kernel_inputs_by_erisc = {
        make_erisc_inputs({{"IS_SENDER_CHANNEL_3_SERVICED", 1}, {"IS_RECEIVER_CHANNEL_0_SERVICED", 1}}),
        make_erisc_inputs({{"IS_SENDER_CHANNEL_3_SERVICED", 0}, {"IS_RECEIVER_CHANNEL_0_SERVICED", 1}})};
    const RouterArgs args(kernel_inputs_by_erisc);

    EXPECT_EQ(args.find_eriscs_servicing_sender(3), std::vector<uint32_t>{0});
    EXPECT_EQ(args.find_eriscs_servicing_receiver(0), (std::vector<uint32_t>{0, 1}));
}

TEST(ManifestRouterArgs, DefineFailsWhenEriscsDisagree) {
    const std::vector<KernelInputs> both = {
        make_erisc_inputs({}, {{"FABRIC_2D_VC1_SERVICED", ""}}),
        make_erisc_inputs({}, {{"FABRIC_2D_VC1_SERVICED", ""}})};
    EXPECT_TRUE(RouterArgs(both).has_define("FABRIC_2D_VC1_SERVICED"));
    EXPECT_FALSE(RouterArgs(both).has_define("FABRIC_2D_VC2_SERVICED"));

    const std::vector<KernelInputs> one = {
        make_erisc_inputs({}, {{"FABRIC_2D_VC1_SERVICED", ""}}), make_erisc_inputs({})};
    EXPECT_THROW(RouterArgs(one).has_define("FABRIC_2D_VC1_SERVICED"), std::exception);
}

}  // namespace
}  // namespace tt::tt_fabric::manifest
