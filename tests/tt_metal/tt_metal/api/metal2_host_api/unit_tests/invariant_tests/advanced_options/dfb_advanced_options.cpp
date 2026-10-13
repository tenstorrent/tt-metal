// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of DFBAdvancedOptions (advanced_options.hpp).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/prefetcher_pipe_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeFullPipeSpec;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::pipe_param_name;
using test_helpers::PrefetcherPipeSpecTestQuasar;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_MultiBindingFlagOnGen2Fails) {
    // allow_instance_multi_binding is a Gen1-only escape hatch. Setting it on a Gen2 target is a hard
    // error regardless of whether any instance is actually multi-bound — here the DFB is a plain
    // single-producer/single-consumer buffer that would otherwise be perfectly valid.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dfb.advanced_options.allow_instance_multi_binding = true;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("only supported on Gen1")));
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayListsSamePipeTwiceFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.dataflow_buffers[0].advanced_options.prefetcher_pipe_relays = {pipe_param_name, pipe_param_name};
    EXPECT_SPEC_REJECTED(spec, "lists PrefetcherPipeParameter 'weights' more than once");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_RelayWithBorrowedFromFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    spec.tensor_parameters = {test_helpers::MakeMinimalTensorParameter("t", BufferType::L1)};
    spec.dataflow_buffers[0].borrowed_from = TensorParamName{"t"};
    EXPECT_SPEC_REJECTED(spec, "sets both prefetcher_pipe_relays and borrowed_from");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
