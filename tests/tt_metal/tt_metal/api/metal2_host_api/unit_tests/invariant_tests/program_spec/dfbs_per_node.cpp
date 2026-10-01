// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariant of WorkUnitSpec::kernels (program_spec.hpp): its kernels bind a bounded number of distinct
// DFBs, which bounds the DFBs occupying each of its nodes.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/hal.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_MultipleDFBsSucceeds) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "multi_dfb_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb1 = MakeMinimalDFB("dfb1");
    dfb1.data_format_metadata = tt::DataFormat::Float16_b;
    auto dfb2 = MakeMinimalDFB("dfb2");
    dfb2.data_format_metadata = tt::DataFormat::Int8;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb1"}, "out1"));
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb2"}, "out2"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb1"}, "in1"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb2"}, "in2"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb1, dfb2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_TooManyDFBsFailsValidation) {
    // Device slots are per-core: exceeding the per-node slot count on a single node must fail
    // validation rather than blowing up downstream during JIT / enqueue.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    const uint32_t too_many = static_cast<uint32_t>(::dfb::NUM_DFBS) + 1;
    for (uint32_t i = 0; i < too_many; ++i) {
        std::string name = "dfb_" + std::to_string(i);
        auto dfb = MakeMinimalDFB(name);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(dfb);
        producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{name}, "p_" + std::to_string(i)));
        consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{name}, "c_" + std::to_string(i)));
    }

    spec.kernels = {producer, consumer};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    const std::string expected_substr = "places " + std::to_string(too_many) + " DataflowBufferSpecs on node (0, 0)";
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(expected_substr)));
}

TEST_F(ProgramSpecTestGen1, CPU_TooManyDFBsOnSameNodeFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "too_many_dfbs_one_node";

    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);

    const uint32_t too_many = tt::tt_metal::hal::get_num_dataflow_buffers() + 1;
    for (uint32_t i = 0; i < too_many; ++i) {
        const std::string name = "dfb_" + std::to_string(i);
        auto dfb = MakeMinimalDFB(name);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers.push_back(dfb);
        producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{name}, "p_" + std::to_string(i)));
        consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{name}, "c_" + std::to_string(i)));
    }

    spec.kernels = {producer, consumer};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    const std::string expected_substr = "places " + std::to_string(too_many) + " DataflowBufferSpecs on node (0, 0)";
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(expected_substr)));
}

// Device slots are per-core: a ProgramSpec may declare more DFBs than
// get_num_dataflow_buffers() when each core hosts at most one. This is the Metal 2.0
// path that issue #51409 needs — previously ValidateProgramSpec rejected on total count.
TEST_F(ProgramSpecTestGen1, CPU_DisjointNodeDFBsExceedSlotCountSucceeds) {
    const uint32_t max_slots = tt::tt_metal::hal::get_num_dataflow_buffers();
    const uint32_t num_dfbs = max_slots + 1;
    constexpr uint32_t grid_x = 8;  // WH mock worker grid width
    ASSERT_GE(grid_x * 9u, num_dfbs) << "mock WH grid too small for this packing check";

    ProgramSpec spec;
    spec.name = "disjoint_dfb_slot_reuse";
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const NodeCoord node{i % grid_x, i / grid_x};
        const std::string pname = "prod_" + std::to_string(i);
        const std::string cname = "cons_" + std::to_string(i);
        const std::string dname = "dfb_" + std::to_string(i);

        auto prod = MakeMinimalGen1DMKernel(pname, DataMovementProcessor::RISCV_0);
        auto cons = MakeMinimalGen1DMKernel(cname, DataMovementProcessor::RISCV_1);
        auto dfb = MakeMinimalDFB(dname);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        prod.dfb_bindings.push_back(ProducerOf(DFBSpecName{dname}, "out"));
        cons.dfb_bindings.push_back(ConsumerOf(DFBSpecName{dname}, "in"));

        spec.kernels.push_back(std::move(prod));
        spec.kernels.push_back(std::move(cons));
        spec.dataflow_buffers.push_back(std::move(dfb));
        spec.work_units.push_back(MakeMinimalWorkUnit("wu_" + std::to_string(i), node, {pname, cname}));
    }

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const std::string dname = "dfb_" + std::to_string(i);
        EXPECT_EQ(program.impl().get_dataflow_buffer(program.impl().get_dfb_handle(dname))->device_slot, 0u)
            << dname << " is alone on its node and should reuse device slot 0";
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental
