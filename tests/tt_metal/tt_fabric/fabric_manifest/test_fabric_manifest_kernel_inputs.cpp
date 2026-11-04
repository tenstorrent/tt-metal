// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RouterArgs over hand-built kernel inputs. No device.

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_kernel_inputs.hpp"

namespace tt::tt_fabric::manifest {
namespace {

KernelInputs make_erisc_inputs(std::unordered_map<std::string, uint32_t> named_compile_args) {
    return {
        .kernel = 0,
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_0,
        .compile_args = {},
        .named_compile_args = std::move(named_compile_args),
        .defines = {},
        .runtime_args = {}};
}

TEST(ManifestRouterArgs, RouterArgFailsWhenEriscsDisagree) {
    const std::vector<KernelInputs> kernel_inputs_by_erisc = {
        make_erisc_inputs({{"SENDER_TXQ_ID", 0}}), make_erisc_inputs({{"SENDER_TXQ_ID", 1}})};
    const RouterArgs args(kernel_inputs_by_erisc);

    EXPECT_EQ(args.read_erisc_arg(1, "SENDER_TXQ_ID"), 1);
    EXPECT_THROW(args.read_router_arg("SENDER_TXQ_ID"), std::exception);
}

// ERISC 1 doesn't service sender 3, so its placeholder 0 is ignored.
TEST(ManifestRouterArgs, SenderArgComesFromTheEriscsServicingIt) {
    const std::vector<KernelInputs> kernel_inputs_by_erisc = {
        make_erisc_inputs({{"IS_SENDER_CHANNEL_3_SERVICED", 1}, {"SENDER_CH_3_LIVE_CHECK_SKIP", 1}}),
        make_erisc_inputs({{"IS_SENDER_CHANNEL_3_SERVICED", 0}, {"SENDER_CH_3_LIVE_CHECK_SKIP", 0}})};
    const RouterArgs args(kernel_inputs_by_erisc);

    EXPECT_EQ(args.find_eriscs_servicing_sender(3), std::vector<uint32_t>{0});
    EXPECT_EQ(args.read_sender_arg("SENDER_CH_3_LIVE_CHECK_SKIP", 3), 1);
}

TEST(ManifestRouterArgs, OptionalArgMustReachEveryEriscOrNone) {
    const std::vector<KernelInputs> kernel_inputs_by_erisc = {
        make_erisc_inputs({{"LINK_SYNC_ROLE", 2}}), make_erisc_inputs({})};
    const RouterArgs args(kernel_inputs_by_erisc);

    EXPECT_EQ(args.find_router_arg("LINK_SYNC_CHECK"), std::nullopt);
    EXPECT_THROW(args.find_router_arg("LINK_SYNC_ROLE"), std::exception);
}

}  // namespace
}  // namespace tt::tt_fabric::manifest
