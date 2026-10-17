// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// transport_bandwidth suite -- idle-ERISC -> X280 addressable member.
//
// The whole implementation lives under
// suites/transport_bandwidth/erisc_to_x280.hpp. This file wires the
// fixture, the parameterization axis (transfer size + writers), and the
// RunSpec (no X280 image, kernel dir).

#include "harness/artifacts/artifacts.hpp"
#include "harness/gtest_dispatch.hpp"
#include "harness/runner.hpp"
#include "suites/transport_bandwidth/erisc_to_x280.hpp"

#include <gtest/gtest.h>

namespace {

namespace suite = x280::suites::transport_bandwidth::erisc_to_x280;

class TransportBandwidthEriscToX280 : public x280::harness::Runner<suite::Axis> {
protected:
    x280::harness::RunSpec make_spec() override {
        return x280::harness::RunSpec{
            .open =
                {
                    .image_env_var = "",  // ERISC-only; no X280 image is submitted.
                    .kernel_dir = x280::harness::artifacts::kernel_dir_of(__FILE__),
                },
            .suite_label = "transport_bandwidth_erisc_to_x280",
        };
    }
};

}  // namespace

TEST_P(TransportBandwidthEriscToX280, RepeatedIssueMeasuresInjection) { X280_HARNESS_DISPATCH(suite); }

constexpr suite::SweepPoint kSweep[] = {
    {64, 1, "w1_size64"},       {128, 1, "w1_size128"},     {256, 1, "w1_size256"},     {320, 1, "w1_size320"},
    {384, 1, "w1_size384"},     {512, 1, "w1_size512"},     {1024, 1, "w1_size1024"},   {2048, 1, "w1_size2048"},
    {4096, 1, "w1_size4096"},   {8192, 1, "w1_size8192"},   {16384, 1, "w1_size16384"}, {64, 2, "w2_size64"},
    {128, 2, "w2_size128"},     {256, 2, "w2_size256"},     {320, 2, "w2_size320"},     {384, 2, "w2_size384"},
    {512, 2, "w2_size512"},     {1024, 2, "w2_size1024"},   {2048, 2, "w2_size2048"},   {4096, 2, "w2_size4096"},
    {8192, 2, "w2_size8192"},   {16384, 2, "w2_size16384"}, {64, 4, "w4_size64"},       {128, 4, "w4_size128"},
    {256, 4, "w4_size256"},     {320, 4, "w4_size320"},     {384, 4, "w4_size384"},     {512, 4, "w4_size512"},
    {1024, 4, "w4_size1024"},   {2048, 4, "w4_size2048"},   {4096, 4, "w4_size4096"},   {8192, 4, "w4_size8192"},
    {16384, 4, "w4_size16384"}, {64, 8, "w8_size64"},       {128, 8, "w8_size128"},     {256, 8, "w8_size256"},
    {320, 8, "w8_size320"},     {384, 8, "w8_size384"},     {512, 8, "w8_size512"},     {1024, 8, "w8_size1024"},
    {2048, 8, "w8_size2048"},   {4096, 8, "w8_size4096"},   {8192, 8, "w8_size8192"},   {16384, 8, "w8_size16384"},
};

INSTANTIATE_TEST_SUITE_P(
    Sweep,
    TransportBandwidthEriscToX280,
    ::testing::ValuesIn(kSweep),
    [](const ::testing::TestParamInfo<suite::SweepPoint>& info) { return std::string(info.param.label); });
