// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// test_packet_injection_erisc_to_x280_header_hw.cpp -- hardware device test
// entry for ERISC -> X280 packet-header ingress: writer-count and batch-size
// axes over one idle-ERISC producer per selected Ethernet tile.

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "harness/slow_dispatch.hpp"
#include "suites/packet_injection/erisc_to_x280_header.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

namespace suite = x280::suites::packet_injection::erisc_to_x280_header;

// Every writer count at B=1; B=4 and B=8 at one writer and at the four-writer
// deployment fan-in. With the default S=16, B=8 is the largest batch that
// keeps the ring double-buffered; B=S would serialize publish and drain.
constexpr suite::SweepPoint kSweepPoints[] = {
    {1, 1, "writers1_b1"},
    {2, 1, "writers2_b1"},
    {4, 1, "writers4_b1"},
    {8, 1, "writers8_b1"},
    {0, 1, "writers_all_b1"},
    {1, 4, "writers1_b4"},
    {1, 8, "writers1_b8"},
    {4, 4, "writers4_b4"},
    {4, 8, "writers4_b8"},
};

class PacketInjectionEriscToX280Header : public x280::harness::Runner<suite::Axis> {
protected:
    // Fast dispatch does not launch IDLE_ETH kernels, and Metal reads the mode
    // when the device opens, so it is checked before that.
    void SetUp() override {
        ASSERT_TRUE(x280::harness::slow_dispatch_linked());
        Runner::SetUp();
    }

    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "X280_PACKET_INJECTION_FW_IMAGE",
                    .image_file = "firmware-packet-injection-throughput.bin",
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                    .lim_layout = x280::LimLayout::PacketInjection,
                },
            .suite_label = "packet_injection_erisc_to_x280_header",
        };
    }
};

}  // namespace

TEST_P(PacketInjectionEriscToX280Header, Run) { X280_HARNESS_DISPATCH(suite); }

INSTANTIATE_TEST_SUITE_P(
    Axes,
    PacketInjectionEriscToX280Header,
    ::testing::ValuesIn(kSweepPoints),
    [](const ::testing::TestParamInfo<suite::SweepPoint>& info) { return std::string(info.param.label); });
