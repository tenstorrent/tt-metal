// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="UnitMeshFixture.SharedKernelCb*"
//
// A kernel is JIT-built once for all its cores. Its CB descriptor (unpack_src_format[],
// unpack_tile_*[]) comes from ProgramImpl::set_cb_data_fmt_and_tile, which walks every CB on
// any of the kernel's core ranges in CB creation order, so a later CB on a slot overwrites
// an earlier one.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {
namespace shared_kernel_cb_descriptor {

const CoreCoord kLeft{0, 0};
const CoreCoord kRight{1, 0};
const CoreRangeSet kBothCores{CoreRange(kLeft, kRight)};

struct SlotExpectation {
    uint32_t slot;
    DataFormat format;
    uint32_t tile_r_dim;
    uint32_t num_faces;
    uint32_t tile_size;
};

// Compute and DM kernels whose JIT compile is the assertion: each static_asserts the
// descriptor silicon's build would bake for the given slots.
std::string compute_kernel_asserting(const std::vector<SlotExpectation>& slots) {
    std::string src = "#include \"api/compute/common.h\"\nvoid kernel_main() {\n";
    for (const auto& s : slots) {
        const std::string i = std::to_string(s.slot);
        src += "static_assert(unpack_src_format[" + i + "] == " + std::to_string(static_cast<uint32_t>(s.format)) +
               ", \"unpack_src_format[" + i + "]\");\n";
        src += "static_assert(unpack_tile_r_dim[" + i + "] == " + std::to_string(s.tile_r_dim) +
               ", \"unpack_tile_r_dim[" + i + "]\");\n";
        src += "static_assert(unpack_tile_num_faces[" + i + "] == " + std::to_string(s.num_faces) +
               ", \"unpack_tile_num_faces[" + i + "]\");\n";
        src += "static_assert(unpack_tile_size[" + i + "] == " + std::to_string(s.tile_size) + ", \"unpack_tile_size[" +
               i + "]\");\n";
    }
    return src + "}\n";
}

std::string dm_kernel_asserting(const std::vector<SlotExpectation>& slots) {
    std::string src = "#include \"api/dataflow/dataflow_api.h\"\nvoid kernel_main() {\n";
    for (const auto& s : slots) {
        const std::string i = std::to_string(s.slot);
        src += "static_assert(static_cast<uint32_t>(get_dataformat(" + i +
               ")) == " + std::to_string(static_cast<uint32_t>(s.format)) + ", \"get_dataformat(" + i + ")\");\n";
        src += "static_assert(get_tile_size(" + i + ") == " + std::to_string(s.tile_size) + ", \"get_tile_size(" + i +
               ")\");\n";
    }
    return src + "}\n";
}

void add_asserting_kernels(Program& program, const std::vector<SlotExpectation>& slots) {
    CreateKernelFromString(program, compute_kernel_asserting(slots), kBothCores, ComputeConfig{});
    CreateKernelFromString(
        program,
        dm_kernel_asserting(slots),
        kBothCores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
}

CircularBufferConfig one_tile_cb(uint8_t slot, DataFormat format, const std::optional<Tile>& tile = std::nullopt) {
    const uint32_t page = tile.value_or(Tile()).get_tile_size(format);
    auto config = CircularBufferConfig(page, {{slot, format}}).set_page_size(slot, page);
    if (tile.has_value()) {
        config.set_tile_dims(slot, *tile);
    }
    return config;
}

}  // namespace shared_kernel_cb_descriptor

using namespace shared_kernel_cb_descriptor;

// CB0 exists only on the left core, CB1 (a 16x32 tile) only on the right: the one kernel
// spanning both is built with both descriptors, not just its first core's.
TEST_F(UnitMeshFixture, SharedKernelCbDescriptorCoversEveryCore) {
    const Tile half_tile({16, 32});
    Program program = CreateProgram();
    CreateCircularBuffer(program, kLeft, one_tile_cb(0, DataFormat::Float16_b));
    CreateCircularBuffer(program, kRight, one_tile_cb(1, DataFormat::Float32, half_tile));
    add_asserting_kernels(
        program,
        {{0, DataFormat::Float16_b, 32, 4, Tile().get_tile_size(DataFormat::Float16_b)},
         {1, DataFormat::Float32, 16, 2, half_tile.get_tile_size(DataFormat::Float32)}});
    EXPECT_NO_THROW(LaunchProgram(this->device(), std::move(program)));
}

// Two cores give slot 3 different CBs: the later-created one wins, whichever core it is on.
// A CB with no Tile writes its format and full-tile size but keeps the earlier tile dims.
TEST_F(UnitMeshFixture, SharedKernelCbDescriptorLaterCbWinsSlot) {
    const Tile half_tile({16, 32});
    {
        Program program = CreateProgram();
        CreateCircularBuffer(program, kLeft, one_tile_cb(3, DataFormat::Float16_b));
        CreateCircularBuffer(program, kRight, one_tile_cb(3, DataFormat::Float32));
        add_asserting_kernels(program, {{3, DataFormat::Float32, 32, 4, Tile().get_tile_size(DataFormat::Float32)}});
        EXPECT_NO_THROW(LaunchProgram(this->device(), std::move(program))) << "left then right";
    }
    {
        Program program = CreateProgram();
        CreateCircularBuffer(program, kRight, one_tile_cb(3, DataFormat::Float32, half_tile));
        CreateCircularBuffer(program, kLeft, one_tile_cb(3, DataFormat::Float16_b));
        add_asserting_kernels(
            program, {{3, DataFormat::Float16_b, 16, 2, Tile().get_tile_size(DataFormat::Float16_b)}});
        EXPECT_NO_THROW(LaunchProgram(this->device(), std::move(program))) << "right then left";
    }
}

// End to end: each core copies one FP32 tile through the input CB its runtime arg names
// (CB0 on the left core, CB1 on the right) into the shared output CB2.
TEST_F(UnitMeshFixture, SharedKernelCbPerCoreInputCopy) {
    constexpr uint32_t tile_words = 1024, tile_bytes = tile_words * sizeof(float);
    auto& device = this->device();
    auto& cq = device.mesh_command_queue();
    auto make_buffer = [&] {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = 2 * tile_bytes},
            {.page_size = tile_bytes, .buffer_type = BufferType::DRAM},
            &device);
    };
    auto input = make_buffer();
    auto output = make_buffer();
    std::vector<float> expected(2 * tile_words), actual(2 * tile_words, -42.0f);
    for (uint32_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<float>(i) / 32.0f;
    }
    distributed::EnqueueWriteMeshBuffer(cq, input, expected, true);
    distributed::EnqueueWriteMeshBuffer(cq, output, actual, true);

    Program program = CreateProgram();
    CreateCircularBuffer(program, kLeft, one_tile_cb(0, DataFormat::Float32));
    CreateCircularBuffer(program, kRight, one_tile_cb(1, DataFormat::Float32));
    CreateCircularBuffer(program, kBothCores, one_tile_cb(2, DataFormat::Float32));

    std::vector<uint32_t> reader_args, writer_args{tile_bytes};
    TensorAccessorArgs(*input).append_to(reader_args);
    TensorAccessorArgs(*output).append_to(writer_args);
    auto reader = CreateKernelFromString(
        program,
        R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            const uint32_t address = get_arg_val<uint32_t>(0);
            const uint32_t page = get_arg_val<uint32_t>(1);
            const uint32_t cb = get_arg_val<uint32_t>(2);
            const auto input = TensorAccessor(TensorAccessorArgs<0>(), address);
            cb_reserve_back(cb, 1);
            noc_async_read_page(page, input, get_write_ptr(cb));
            noc_async_read_barrier();
            cb_push_back(cb, 1);
        }
    )",
        kBothCores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_args});
    auto writer = CreateKernelFromString(
        program,
        R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            constexpr uint32_t tile_bytes = get_compile_time_arg_val(0), cb = 2;
            const uint32_t address = get_arg_val<uint32_t>(0);
            const uint32_t page = get_arg_val<uint32_t>(1);
            const auto output = TensorAccessor(TensorAccessorArgs<1>(), address);
            cb_wait_front(cb, 1);
            noc_async_write_page(page, output, get_read_ptr(cb), tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb, 1);
        }
    )",
        kBothCores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_args});
    auto compute = CreateKernelFromString(
        program,
        R"(
        #include "api/compute/common.h"
        #include "api/compute/compute_kernel_hw_startup.h"
        #include "api/compute/tile_move_copy.h"
        #include "api/compute/pack.h"
        #include "api/compute/reg_api.h"
        void kernel_main() {
            const uint32_t input_cb = get_arg_val<uint32_t>(0);
            constexpr uint32_t output_cb = 2;
            compute_kernel_hw_startup(input_cb, output_cb);
            copy_init(input_cb);
            cb_wait_front(input_cb, 1);
            cb_reserve_back(output_cb, 1);
            tile_regs_acquire();
            copy_tile(input_cb, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output_cb);
            tile_regs_release();
            cb_push_back(output_cb, 1);
            cb_pop_front(input_cb, 1);
        }
    )",
        kBothCores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true});
    for (uint32_t x = 0; x < 2; ++x) {
        const CoreCoord core(x, 0);
        SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(input->address()), x, x});
        SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(output->address()), x});
        SetRuntimeArgs(program, compute, core, {x});
    }

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    distributed::EnqueueReadMeshBuffer(cq, actual, output, true);
    EXPECT_EQ(actual, expected);
}

}  // namespace tt::tt_metal
