// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

TEST_F(UnitMeshFixture, Experiment4_IdmaHeadSplit) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar only";
    }

    constexpr std::uint32_t element_bytes = sizeof(std::uint16_t);
    constexpr std::uint32_t num_sections = 3;
    constexpr std::uint32_t heads_per_section = 2;
    constexpr std::uint32_t head_dim = 64;
    constexpr std::uint32_t seq_len = 32;

    constexpr std::uint32_t num_heads = num_sections * heads_per_section;
    constexpr std::uint32_t chunk_bytes = head_dim * element_bytes;
    constexpr std::uint32_t row_bytes = num_heads * chunk_bytes;
    constexpr std::uint32_t total_bytes = seq_len * row_bytes;
    constexpr std::uint32_t total_elements = seq_len * num_heads * head_dim;

    const CoreCoord node{0, 0};

    distributed::DeviceLocalBufferConfig local_config{.page_size = total_bytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config{.size = total_bytes};
    auto in_buffer = distributed::MeshBuffer::create(buffer_config, local_config, &this->device());
    auto out_buffer = distributed::MeshBuffer::create(buffer_config, local_config, &this->device());

    // s.h.d label in each element, so a mismatch shows where it came from.
    constexpr std::uint16_t seq_tag = 1024;
    constexpr std::uint16_t head_tag = 128;
    std::vector<std::uint16_t> input(total_elements);
    for (std::uint32_t s = 0; s < seq_len; ++s) {
        for (std::uint32_t h = 0; h < num_heads; ++h) {
            for (std::uint32_t d = 0; d < head_dim; ++d) {
                input[(s * num_heads + h) * head_dim + d] = s * seq_tag + h * head_tag + d;
            }
        }
    }

    std::vector<std::uint16_t> golden(total_elements);
    for (std::uint32_t h = 0; h < num_heads; ++h) {
        for (std::uint32_t s = 0; s < seq_len; ++s) {
            for (std::uint32_t d = 0; d < head_dim; ++d) {
                golden[(h * seq_len + s) * head_dim + d] = input[(s * num_heads + h) * head_dim + d];
            }
        }
    }

    const m2::KernelSpecName HEAD_SPLIT{"head_split"};
    auto head_split =
        make_dm_kernel(HEAD_SPLIT, "tests/tt_metal/tt_metal/test_kernels/dataflow/test_experiment_4_head_split.cpp", 1);
    head_split.compile_time_args = {
        {"in_addr", static_cast<std::uint32_t>(in_buffer->address())},
        {"out_addr", static_cast<std::uint32_t>(out_buffer->address())},
        {"num_heads", num_heads},
        {"seq_len", seq_len},
        {"head_dim", head_dim},
        {"element_bytes", element_bytes},
    };

    m2::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {HEAD_SPLIT},
        .target_nodes = m2::NodeCoord{0, 0},
    };
    m2::ProgramSpec spec{
        .name = "exp4_head_split",
        .kernels = {head_split},
        .work_units = {wu},
    };
    Program program = m2::MakeProgramFromSpec(this->device(), spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {m2::ProgramRunArgs::KernelRunArgs{.kernel = HEAD_SPLIT}};
    m2::SetProgramRunArgs(program, params);

    slow_dispatch::WriteToL1(
        this->device(),
        node,
        in_buffer->address(),
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(input.data()), total_bytes));
    // L1 keeps data across runs on emu/silicon; a sentinel stops a stale correct result from passing.
    constexpr std::uint16_t sentinel = 0xFFFF;
    const std::vector<std::uint16_t> cleared(total_elements, sentinel);
    slow_dispatch::WriteToL1(
        this->device(),
        node,
        out_buffer->address(),
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(cleared.data()), total_bytes));
    LaunchProgram(this->device(), std::move(program));

    std::vector<std::uint16_t> output(total_elements);
    slow_dispatch::ReadFromL1(
        this->device(),
        node,
        out_buffer->address(),
        std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(output.data()), total_bytes));
    EXPECT_EQ(golden, output);
}

}  // namespace tt::tt_metal
