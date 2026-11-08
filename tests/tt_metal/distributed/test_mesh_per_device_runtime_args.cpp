// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-device runtime args of a MeshWorkload program (distributed::SetRuntimeArgs on a MeshWorkload): one program on
// the whole mesh, chosen runtime-arg words set per device. The kernel writes its unique and common runtime args to L1,
// read back per device.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <distributed/mesh_device_impl.hpp>
#include "impl/context/metal_context.hpp"

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp";
constexpr uint32_t kCommonSeparationWords = 1024;  // runtime_args_kernel.cpp: common args at results + 1024 words
constexpr uint32_t kNumArgs = 6;
constexpr uint32_t kNumCommonArgs = 4;

// Helpers and test bodies over one MeshDevice, shared by the fast-dispatch fixture and the slow-dispatch test.
class PerDeviceRuntimeArgsChecks {
protected:
    std::shared_ptr<MeshDevice> mesh_device_;

    uint32_t results_addr() const { return mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1); }

    std::vector<uint32_t> read_words(const MeshCoordinate& coord, const CoreCoord& core, uint32_t addr, uint32_t n) {
        std::vector<uint32_t> words;
        auto* device = mesh_device_->impl().get_device(coord);
        tt::tt_metal::detail::ReadFromDeviceL1(device, core, addr, n * sizeof(uint32_t), words);
        return words;
    }

    // A kernel on `cores` writing kNumArgs unique and kNumCommonArgs common runtime args to L1 at `results`.
    KernelHandle create_kernel(
        Program& program,
        const CoreRangeSet& cores,
        uint32_t results,
        DataMovementProcessor processor,
        std::map<std::string, std::string> extra_defines = {}) {
        std::map<std::string, std::string> defines{
            {"RESULTS_ADDR", std::to_string(results)},
            {"NUM_RUNTIME_ARGS", std::to_string(kNumArgs)},
            {"COMMON_RUNTIME_ARGS", "1"},
            {"NUM_COMMON_RUNTIME_ARGS", std::to_string(kNumCommonArgs)}};
        defines.insert(extra_defines.begin(), extra_defines.end());
        return CreateKernel(
            program,
            kKernel,
            cores,
            DataMovementConfig{
                .processor = processor,
                .noc = processor == DataMovementProcessor::RISCV_0 ? NOC::RISCV_0_default : NOC::RISCV_1_default,
                .defines = defines});
    }

    static std::vector<uint32_t> default_args(const CoreCoord& core) {
        std::vector<uint32_t> args(kNumArgs);
        for (uint32_t i = 0; i < kNumArgs; i++) {
            args[i] = 100 * (core.x + 1) + i;
        }
        return args;
    }
    static std::vector<uint32_t> default_common_args() { return {20, 21, 22, 23}; }

    void unique_and_common_fields_eager();
    void batched_setter_and_clear();
    void trace_replays_capture_time_values();
    void field_after_kernel_without_args_on_core();
    void rejects_invalid_fields();
    void program_on_sub_range();
    void trace_node_in_several_partitions();
};

class MeshPerDeviceRuntimeArgsTest : public MeshDeviceFixtureBase, public PerDeviceRuntimeArgsChecks {
protected:
    MeshPerDeviceRuntimeArgsTest() : MeshDeviceFixtureBase(Config{.num_cqs = 1, .trace_region_size = (64 << 20)}) {}
    void SetUp() override {
        MeshDeviceFixtureBase::SetUp();
        PerDeviceRuntimeArgsChecks::mesh_device_ = MeshDeviceFixtureBase::mesh_device_;
    }
};

void PerDeviceRuntimeArgsChecks::unique_and_common_fields_eager() {
    const MeshCoordinateRange range(mesh_device_->shape());
    const CoreCoord plain_core{0, 0}, field_core{1, 0};
    const uint32_t results = results_addr();

    Program program = CreateProgram();
    const auto kernel = create_kernel(
        program, CoreRangeSet(CoreRange(plain_core, field_core)), results, DataMovementProcessor::RISCV_0);
    for (const auto& core : {plain_core, field_core}) {
        SetRuntimeArgs(program, kernel, core, default_args(core));
    }
    SetCommonRuntimeArgs(program, kernel, default_common_args());
    MeshWorkload workload;
    workload.add_program(range, std::move(program));

    // Words [2, 4) of field_core's args and common word 1 per device; the first device keeps the program's own.
    auto unique_value = [](uint32_t idx, uint32_t word, uint32_t round) {
        return 1000 * (round + 1) + 10 * idx + word;
    };
    auto common_value = [](uint32_t idx, uint32_t round) { return 50000 + 1000 * round + idx; };
    auto set_fields = [&](uint32_t round) {
        uint32_t idx = 0;
        for (const auto& coord : range) {
            if (idx != 0) {
                SetRuntimeArgs(
                    workload,
                    range,
                    coord,
                    kernel,
                    field_core,
                    2,
                    std::vector<uint32_t>{unique_value(idx, 2, round), unique_value(idx, 3, round)});
                SetCommonRuntimeArgs(
                    workload, range, coord, kernel, 1, std::vector<uint32_t>{common_value(idx, round)});
            }
            idx++;
        }
    };
    auto check = [&](uint32_t round, uint32_t word0) {
        uint32_t idx = 0;
        for (const auto& coord : range) {
            auto expected = default_args(field_core);
            expected[0] = word0;
            auto expected_common = default_common_args();
            if (idx != 0) {
                expected[2] = unique_value(idx, 2, round);
                expected[3] = unique_value(idx, 3, round);
                expected_common[1] = common_value(idx, round);
            }
            EXPECT_EQ(read_words(coord, field_core, results, kNumArgs), expected) << "device " << coord;
            EXPECT_EQ(read_words(coord, plain_core, results, kNumArgs), default_args(plain_core)) << "device " << coord;
            const auto common_addr = results + kCommonSeparationWords * sizeof(uint32_t);
            EXPECT_EQ(read_words(coord, field_core, common_addr, kNumCommonArgs), expected_common)
                << "device " << coord;
            EXPECT_EQ(read_words(coord, plain_core, common_addr, kNumCommonArgs), expected_common)
                << "device " << coord;
            idx++;
        }
    };

    auto& cq = mesh_device_->mesh_command_queue();
    set_fields(0);
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);
    check(0, default_args(field_core)[0]);

    // A word outside the fields, updated on the program, reaches every device; new field values reach theirs.
    GetRuntimeArgs(workload.get_programs().at(range), kernel, field_core)[0] = 77;
    set_fields(1);
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);
    check(1, 77);
}

void PerDeviceRuntimeArgsChecks::batched_setter_and_clear() {
    const MeshCoordinateRange range(mesh_device_->shape());
    const CoreCoord core{0, 0};
    const uint32_t results = results_addr();

    Program program = CreateProgram();
    const auto kernel = create_kernel(program, CoreRangeSet(CoreRange(core)), results, DataMovementProcessor::RISCV_0);
    SetRuntimeArgs(program, kernel, core, default_args(core));
    SetCommonRuntimeArgs(program, kernel, default_common_args());
    MeshWorkload workload;
    workload.add_program(range, std::move(program));

    const uint32_t num_devices = range.shape().mesh_size();
    std::vector<uint32_t> values(num_devices);
    for (uint32_t i = 0; i < num_devices; i++) {
        values[i] = 7000 + i;
    }
    SetRuntimeArgsPerDevice(workload, range, kernel, core, 5, 1, values);
    const auto cleared = *range.begin();
    ClearRuntimeArgs(workload, range, cleared, kernel, core, 5);

    auto& cq = mesh_device_->mesh_command_queue();
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);
    uint32_t idx = 0;
    for (const auto& coord : range) {
        auto expected = default_args(core);
        if (coord != cleared) {
            expected[5] = 7000 + idx;
        }
        EXPECT_EQ(read_words(coord, core, results, kNumArgs), expected) << "device " << coord;
        idx++;
    }
}

void PerDeviceRuntimeArgsChecks::trace_replays_capture_time_values() {
    const MeshCoordinateRange range(mesh_device_->shape());
    const CoreCoord core{0, 0};
    const uint32_t results = results_addr();

    Program program = CreateProgram();
    const auto kernel = create_kernel(program, CoreRangeSet(CoreRange(core)), results, DataMovementProcessor::RISCV_0);
    SetRuntimeArgs(program, kernel, core, default_args(core));
    SetCommonRuntimeArgs(program, kernel, default_common_args());
    MeshWorkload workload;
    workload.add_program(range, std::move(program));

    auto set_fields = [&](uint32_t base) {
        uint32_t idx = 0;
        for (const auto& coord : range) {
            SetRuntimeArgs(workload, range, coord, kernel, core, 1, std::vector<uint32_t>{base + idx});
            SetCommonRuntimeArgs(workload, range, coord, kernel, 3, std::vector<uint32_t>{base + 500 + idx});
            idx++;
        }
    };
    auto check = [&](uint32_t base) {
        uint32_t idx = 0;
        for (const auto& coord : range) {
            auto expected = default_args(core);
            expected[1] = base + idx;
            auto expected_common = default_common_args();
            expected_common[3] = base + 500 + idx;
            EXPECT_EQ(read_words(coord, core, results, kNumArgs), expected) << "device " << coord;
            EXPECT_EQ(
                read_words(coord, core, results + kCommonSeparationWords * sizeof(uint32_t), kNumCommonArgs),
                expected_common)
                << "device " << coord;
            idx++;
        }
    };

    auto& cq = mesh_device_->mesh_command_queue();
    set_fields(10000);
    EnqueueMeshWorkload(cq, workload, false);  // compiles and loads the program
    Finish(cq);
    check(10000);

    set_fields(20000);
    const auto trace_id = mesh_device_->begin_mesh_trace(cq.id());
    EnqueueMeshWorkload(cq, workload, false);
    mesh_device_->end_mesh_trace(cq.id(), trace_id);

    set_fields(30000);  // after capture: the trace keeps the capture-time values
    for (int replay = 0; replay < 3; replay++) {
        mesh_device_->replay_mesh_trace(cq.id(), trace_id, false);
        Finish(cq);
        check(20000);
    }
    mesh_device_->release_mesh_trace(trace_id);

    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);
    check(30000);
}

// Kernel A has args on core 0 only; kernel B has args on both cores of the same kernel group, with a field on core 1,
// where A has none. B's words on core 1 sit at A's stride, not at A's (empty) size.
void PerDeviceRuntimeArgsChecks::field_after_kernel_without_args_on_core() {
    const MeshCoordinateRange range(mesh_device_->shape());
    const CoreCoord core0{0, 0}, core1{0, 1};
    const uint32_t results_a = results_addr();
    const uint32_t results_b = results_a + 8192;

    Program program = CreateProgram();
    const CoreRangeSet cores(CoreRange(core0, core1));
    // A reads no args on core1 (logical y >= 1).
    const auto kernel_a = create_kernel(
        program,
        cores,
        results_a,
        DataMovementProcessor::RISCV_0,
        {{"NUM_RUNTIME_ARGS_CR1", "0"}, {"CR1_START_Y", "1"}});
    const auto kernel_b = create_kernel(program, cores, results_b, DataMovementProcessor::RISCV_1);
    SetRuntimeArgs(program, kernel_a, core0, default_args(core0));
    SetRuntimeArgs(program, kernel_b, core0, default_args(core0));
    SetRuntimeArgs(program, kernel_b, core1, default_args(core1));
    SetCommonRuntimeArgs(program, kernel_a, default_common_args());
    SetCommonRuntimeArgs(program, kernel_b, default_common_args());
    MeshWorkload workload;
    workload.add_program(range, std::move(program));

    uint32_t idx = 0;
    for (const auto& coord : range) {
        SetRuntimeArgs(workload, range, coord, kernel_b, core1, 0, std::vector<uint32_t>{9000 + idx, 9500 + idx});
        idx++;
    }
    auto& cq = mesh_device_->mesh_command_queue();
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);
    // A cache-hit update of B on core1 through the program's view lands in B's words too.
    GetRuntimeArgs(workload.get_programs().at(range), kernel_b, core1)[5] = 4242;
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    idx = 0;
    for (const auto& coord : range) {
        auto expected_b = default_args(core1);
        expected_b[0] = 9000 + idx;
        expected_b[1] = 9500 + idx;
        expected_b[5] = 4242;
        EXPECT_EQ(read_words(coord, core1, results_b, kNumArgs), expected_b) << "device " << coord;
        EXPECT_EQ(read_words(coord, core0, results_a, kNumArgs), default_args(core0)) << "device " << coord;
        EXPECT_EQ(read_words(coord, core0, results_b, kNumArgs), default_args(core0)) << "device " << coord;
        idx++;
    }
}

void PerDeviceRuntimeArgsChecks::rejects_invalid_fields() {
    const MeshCoordinateRange range(mesh_device_->shape());
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    const auto kernel =
        create_kernel(program, CoreRangeSet(CoreRange(core)), results_addr(), DataMovementProcessor::RISCV_0);
    SetRuntimeArgs(program, kernel, core, default_args(core));
    SetCommonRuntimeArgs(program, kernel, default_common_args());
    MeshWorkload workload;
    workload.add_program(range, std::move(program));
    const auto coord = *range.begin();

    SetRuntimeArgs(workload, range, coord, kernel, core, 2, std::vector<uint32_t>{1, 2});
    // Same first word, another length.
    EXPECT_THAT(
        [&] { SetRuntimeArgs(workload, range, coord, kernel, core, 2, std::vector<uint32_t>{1}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("has 2 words, not 1")));
    // Overlap with words [2, 4).
    EXPECT_THAT(
        [&] { SetRuntimeArgs(workload, range, coord, kernel, core, 3, std::vector<uint32_t>{1, 2}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("overlap")));
    // Beyond the program's args.
    EXPECT_THAT(
        [&] { SetRuntimeArgs(workload, range, coord, kernel, core, kNumArgs - 1, std::vector<uint32_t>{1, 2}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("lie outside")));
    EXPECT_THAT(
        [&] { SetCommonRuntimeArgs(workload, range, coord, kernel, kNumCommonArgs, std::vector<uint32_t>{1}); },
        ThrowsMessage<std::runtime_error>(HasSubstr("lie outside")));
    // A range the workload has no program on.
    const MeshCoordinateRange other(coord, coord);
    if (other != range) {
        EXPECT_THAT(
            [&] { SetRuntimeArgs(workload, other, coord, kernel, core, 0, std::vector<uint32_t>{1}); },
            ThrowsMessage<std::runtime_error>(HasSubstr("has no program on range")));
    }
    // Batched values of the wrong size.
    EXPECT_THAT(
        [&] {
            SetRuntimeArgsPerDevice(
                workload, range, kernel, core, 0, 1, std::vector<uint32_t>(range.shape().mesh_size() + 1, 1));
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("expected")));
}

// A program on a range that does not start at the mesh origin: values are indexed within the program's range.
void PerDeviceRuntimeArgsChecks::program_on_sub_range() {
    const auto shape = mesh_device_->shape();
    if (shape.mesh_size() < 2) {
        GTEST_SKIP() << "Needs at least two devices";
    }
    const MeshCoordinateRange full(shape);
    const MeshCoordinate start = *std::next(full.begin());  // the second device in row-major order
    const MeshCoordinateRange range(start, full.end_coord());
    const CoreCoord core{0, 0};
    const uint32_t results = results_addr();

    Program program = CreateProgram();
    const auto kernel = create_kernel(program, CoreRangeSet(CoreRange(core)), results, DataMovementProcessor::RISCV_0);
    SetRuntimeArgs(program, kernel, core, default_args(core));
    SetCommonRuntimeArgs(program, kernel, default_common_args());
    MeshWorkload workload;
    workload.add_program(range, std::move(program));
    for (const auto& coord : range) {
        SetRuntimeArgs(
            workload, range, coord, kernel, core, 4, std::vector<uint32_t>{6000 + coord[0] * 100 + coord[1]});
    }
    auto& cq = mesh_device_->mesh_command_queue();
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);
    for (const auto& coord : range) {
        auto expected = default_args(core);
        expected[4] = 6000 + coord[0] * 100 + coord[1];
        EXPECT_EQ(read_words(coord, core, results, kNumArgs), expected) << "device " << coord;
    }
}

// A full-mesh program with fields captured in one trace with a program on part of the mesh: the full-mesh node is
// serialized into several partition streams, each patched per device.
void PerDeviceRuntimeArgsChecks::trace_node_in_several_partitions() {
    const auto shape = mesh_device_->shape();
    if (shape.mesh_size() < 2) {
        GTEST_SKIP() << "Needs at least two devices";
    }
    const MeshCoordinateRange full(shape);
    const MeshCoordinateRange first(*full.begin(), *full.begin());
    const CoreCoord core{0, 0}, other_core{1, 0};
    const uint32_t results = results_addr();

    Program program = CreateProgram();
    const auto kernel = create_kernel(program, CoreRangeSet(CoreRange(core)), results, DataMovementProcessor::RISCV_0);
    SetRuntimeArgs(program, kernel, core, default_args(core));
    SetCommonRuntimeArgs(program, kernel, default_common_args());
    MeshWorkload workload;
    workload.add_program(full, std::move(program));
    uint32_t idx = 0;
    for (const auto& coord : full) {
        SetRuntimeArgs(workload, full, coord, kernel, core, 0, std::vector<uint32_t>{8000 + idx});
        SetCommonRuntimeArgs(workload, full, coord, kernel, 0, std::vector<uint32_t>{8500 + idx});
        idx++;
    }
    // A plain program on the first device only, on another core.
    Program other = CreateProgram();
    const auto other_kernel =
        create_kernel(other, CoreRangeSet(CoreRange(other_core)), results, DataMovementProcessor::RISCV_0);
    SetRuntimeArgs(other, other_kernel, other_core, default_args(other_core));
    SetCommonRuntimeArgs(other, other_kernel, default_common_args());
    MeshWorkload other_workload;
    other_workload.add_program(first, std::move(other));

    auto& cq = mesh_device_->mesh_command_queue();
    EnqueueMeshWorkload(cq, workload, false);
    EnqueueMeshWorkload(cq, other_workload, false);
    Finish(cq);
    const auto trace_id = mesh_device_->begin_mesh_trace(cq.id());
    EnqueueMeshWorkload(cq, other_workload, false);
    EnqueueMeshWorkload(cq, workload, false);
    mesh_device_->end_mesh_trace(cq.id(), trace_id);
    mesh_device_->replay_mesh_trace(cq.id(), trace_id, false);
    Finish(cq);
    mesh_device_->release_mesh_trace(trace_id);

    idx = 0;
    for (const auto& coord : full) {
        auto expected = default_args(core);
        expected[0] = 8000 + idx;
        auto expected_common = default_common_args();
        expected_common[0] = 8500 + idx;
        EXPECT_EQ(read_words(coord, core, results, kNumArgs), expected) << "device " << coord;
        EXPECT_EQ(
            read_words(coord, core, results + kCommonSeparationWords * sizeof(uint32_t), kNumCommonArgs),
            expected_common)
            << "device " << coord;
        idx++;
    }
    EXPECT_EQ(read_words(*full.begin(), other_core, results, kNumArgs), default_args(other_core));
}

TEST_F(MeshPerDeviceRuntimeArgsTest, UniqueAndCommonFieldsEager) { unique_and_common_fields_eager(); }
TEST_F(MeshPerDeviceRuntimeArgsTest, ProgramOnSubRange) { program_on_sub_range(); }
TEST_F(MeshPerDeviceRuntimeArgsTest, TraceNodeInSeveralPartitions) { trace_node_in_several_partitions(); }
TEST_F(MeshPerDeviceRuntimeArgsTest, BatchedSetterAndClear) { batched_setter_and_clear(); }
TEST_F(MeshPerDeviceRuntimeArgsTest, TraceReplaysCaptureTimeValues) { trace_replays_capture_time_values(); }
TEST_F(MeshPerDeviceRuntimeArgsTest, FieldAfterKernelWithoutArgsOnCore) { field_after_kernel_without_args_on_core(); }
TEST_F(MeshPerDeviceRuntimeArgsTest, RejectsInvalidFields) { rejects_invalid_fields(); }

// Slow dispatch writes runtime args per device from the kernels' own storage. Mesh fixtures run in fast dispatch only,
// so this opens a 1x2 mesh itself when TT_METAL_SLOW_DISPATCH_MODE is set.
class MeshPerDeviceRuntimeArgsSlowDispatchTest : public ::testing::Test, public PerDeviceRuntimeArgsChecks {
protected:
    void SetUp() override {
        if (getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
            GTEST_SKIP() << "Slow dispatch only (set TT_METAL_SLOW_DISPATCH_MODE)";
        }
        if (tt::tt_metal::MetalContext::instance().get_system_mesh().shape().mesh_size() < 2) {
            GTEST_SKIP() << "Needs at least two devices";
        }
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));
    }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }
};

TEST_F(MeshPerDeviceRuntimeArgsSlowDispatchTest, UniqueAndCommonFields) { unique_and_common_fields_eager(); }
TEST_F(MeshPerDeviceRuntimeArgsSlowDispatchTest, BatchedSetterAndClear) { batched_setter_and_clear(); }
TEST_F(MeshPerDeviceRuntimeArgsSlowDispatchTest, FieldAfterKernelWithoutArgsOnCore) {
    field_after_kernel_without_args_on_core();
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
