// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Smallest repro for a Quasar DataflowBuffer credit defect: a producer that fills one slot with two
// NoC reads instead of one releases its consumer before the slot is filled.
//
// One producer, one consumer, one buffer, one DRAM round trip. No compute kernel, no scratchpad, no
// second destination, nothing from ttnn. The buffer is used exactly as documented on both sides.
//
// What the producer does differently from a correct program is a single line: it fills each entry with
// two half-entry reads rather than one full-entry read, so it issues two NoC transactions per slot it
// announces with push_back. Both reads land inside the slot it just reserved. That is the entire
// difference, and it is enough.
//
// The contract being violated: wait_front(1) must not return until the matching push_back(1). Since
// each entry carries distinct values, a consumer released early copies out a slot the producer has not
// written yet, so the DRAM round trip does not return what went in.
//
// Wormhole is the control. The identical program passes there, so what differs between the two
// generations is the buffer's behaviour, not this test.
//
// This file deliberately duplicates one case from test_dfb_gen2_credits_hw.cpp, which explores the
// surrounding space: what happens with a matching or deficient transaction count, whether the
// destination of the extra transfer matters, whether disabling Gen2 implicit sync helps, and why a
// compute consumer is unaffected. Read that file to bound the defect; attach this one to a bug report.
//
// Run it:
//   ./build_Release/test/tt_metal/unit_tests_api \
//     --gtest_filter="Gen2DFBSplitReadReproTest.*"
// Expected: fails on Quasar, passes on Wormhole. Requires TT_METAL_SLOW_DISPATCH_MODE=1, plus the
// Quasar simulator variables for a Quasar run.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal::experimental {
namespace {

class Gen2DFBSplitReadReproTest : public tt::tt_metal::MeshDeviceFixture {};

// A data-movement kernel config for whichever generation is running. The two config types are
// unrelated, and on Gen1 the two kernels must sit on different NOCs or the device hangs for reasons
// that have nothing to do with this defect.
DataMovementHardwareConfig dm_config(bool gen2, DataMovementProcessor proc) {
    if (gen2) {
        return DataMovementGen2Config{};
    }
    return DataMovementGen1Config{
        .processor = proc, .noc = (proc == DataMovementProcessor::RISCV_0) ? NOC::NOC_0 : NOC::NOC_1};
}

TEST_F(Gen2DFBSplitReadReproTest, TwoReadsPerAnnouncedSlotCorruptsTheStream) {
    // Four entries through a two-slot ring: enough that the ring wraps, which is what lets an early
    // release hand the consumer a slot whose previous contents are still there.
    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries = 4;
    constexpr uint32_t ring_depth = 2;

    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    const bool gen2 = device->arch() == tt::ARCH::QUASAR;

    const NodeCoord node{0, 0};
    const uint32_t total_bytes = num_entries * entry_size;

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto in_buffer = CreateBuffer(dram_config);
    auto out_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "dfb_split_read_repro";

    KernelSpec producer{
        .unique_id = KernelSpecName{"producer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_split_read_producer.cpp",
        .num_threads = 1,
        .hw_config = dm_config(gen2, DataMovementProcessor::RISCV_0),
    };
    producer.advanced_options.num_runtime_varargs = 2;

    KernelSpec consumer{
        .unique_id = KernelSpecName{"consumer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_split_read_consumer.cpp",
        .num_threads = 1,
        .hw_config = dm_config(gen2, DataMovementProcessor::RISCV_1),
    };
    consumer.advanced_options.num_runtime_varargs = 2;

    DataflowBufferSpec ring{
        .unique_id = DFBSpecName{"ring"},
        .entry_size = entry_size,
        .num_entries = ring_depth,
    };

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"ring"}, "ring"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"ring"}, "ring"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {ring};
    spec.work_units = std::vector<WorkUnitSpec>{WorkUnitSpec{
        .name = "work_unit_0",
        .kernels = {KernelSpecName{"producer"}, KernelSpecName{"consumer"}},
        .target_nodes = node,
    }};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {static_cast<uint32_t>(in_buffer->address()), num_entries}}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {static_cast<uint32_t>(out_buffer->address()), num_entries}}},
                },
        },
    };
    SetProgramRunArgs(program, run_args);

    // Distinct values per word, offset so that entry 0 does not begin with zero, which a slot that was
    // never written could coincidentally match.
    std::vector<uint32_t> input(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<uint32_t>(i) + 0x1000u;
    }
    // A sentinel in the output tells a slot that was never delivered apart from one delivered wrongly.
    std::vector<uint32_t> sentinel(input.size(), 0xDEADBEEFu);

    detail::WriteToBuffer(in_buffer, input);
    detail::WriteToBuffer(out_buffer, sentinel);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(out_buffer, output);

    ASSERT_EQ(output.size(), input.size());

    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    for (uint32_t entry = 0; entry < num_entries; entry++) {
        const uint32_t expected = input[entry * words_per_entry];
        const uint32_t actual = output[entry * words_per_entry];
        EXPECT_EQ(actual, expected) << "entry " << entry << " of " << num_entries << " came back wrong: expected "
                                    << expected << ", got " << actual
                                    << ". The consumer's wait_front returned before the producer's push_back, so it "
                                    << "copied out a slot the producer had not filled.";
    }
    EXPECT_EQ(output, input) << "the DRAM round trip did not return what went in";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
