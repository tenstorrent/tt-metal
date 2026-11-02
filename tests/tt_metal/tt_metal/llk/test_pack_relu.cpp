// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include "llk_device_fixture.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

namespace unit_tests::compute::pack_relu {

// Mirrors ckernel::ReluType; the packed config follows ReluConfig::from_packed (bits [1:0] mode, [31:16] threshold).
enum class ReluMode : std::uint32_t { None = 0, Zero = 1, MinThreshold = 2, MaxThreshold = 3 };

struct PackReluConfig {
    ReluMode mode = ReluMode::Zero;
    float threshold = 0.0f;  // bf16-representable; ignored by None / Zero
    bool dst_full_sync_en = false;
};

static std::uint32_t packed_relu_config(const PackReluConfig& config) {
    const std::uint32_t threshold_bf16 = std::bit_cast<std::uint32_t>(config.threshold) >> 16;
    return static_cast<std::uint32_t>(config.mode) | (threshold_bf16 << 16);
}

// Packer ReLU on one fp32 datum, the tt-llk PackGolden::apply_relu semantics.
static float golden_relu(float x, const PackReluConfig& config) {
    switch (config.mode) {
        case ReluMode::None: return x;
        case ReluMode::Zero: return std::max(x, 0.0f);
        case ReluMode::MinThreshold: return x <= config.threshold ? 0.0f : x;
        case ReluMode::MaxThreshold: return std::clamp(x, 0.0f, config.threshold);
    }
    return x;
}

static tt::tt_metal::TensorSpec make_flat_dram_tensor_spec(std::uint32_t entry_size, std::uint32_t total_entries) {
    const std::uint32_t entry_size_words = entry_size / sizeof(std::uint32_t);
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    return tt::tt_metal::TensorSpec(tt::tt_metal::Shape{total_entries, entry_size_words}, tensor_layout);
}

// One core copies num_tiles Float32 tiles through a 32-bit DEST (unpack-to-dest) with the packer ReLU configured;
// the output must equal golden_relu bit-exact.
void run_single_core_pack_relu(distributed::MeshDevice& mesh_device, const PackReluConfig& config) {
    const experimental::NodeCoord node{0, 0};

    constexpr std::uint32_t num_tiles = 4;
    constexpr std::uint32_t num_buffer_tiles = 8;
    const std::uint32_t tile_size = constants::TILE_HW * sizeof(std::uint32_t);
    const std::uint32_t num_elems = constants::TILE_HW * num_tiles;

    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, make_flat_dram_tensor_spec(tile_size, num_tiles));
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, make_flat_dram_tensor_spec(tile_size, num_tiles));

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};
    const experimental::TensorParamName IN_TENSOR{"in_tensor"};
    const experimental::TensorParamName OUT_TENSOR{"out_tensor"};

    experimental::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = tile_size,
        .num_entries = num_buffer_tiles,
        .data_format_metadata = tt::DataFormat::Float32,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = tile_size,
        .num_entries = num_buffer_tiles,
        .data_format_metadata = tt::DataFormat::Float32,
    };

    const experimental::DataMovementHardwareConfig dm_hw_config{
        .config_2xx =
            experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true,
            },
    };
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out_data")},
        .tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = dm_hw_config,
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = dm_hw_config,
    };
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/pack_relu_fp32.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"NUM_TILES", num_tiles}, {"RELU_CONFIG", packed_relu_config(config)}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .enable_32_bit_dest = true,
                .double_buffer_dest = !config.dst_full_sync_en,
                .unpack_modes = {{INPUT_DFB, tt::tt_metal::UnpackMode::UnpackToDest}},
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };
    experimental::ProgramSpec spec{
        .name = "pack_relu_fp32",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN_TENSOR, experimental::ProgramRunArgs::TensorArgument{in_tensor}},
        {OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{out_tensor}},
    };
    experimental::SetProgramRunArgs(program, params);

    // Seeded fp32 in [-2, 2]; values within 1/64 of a threshold are resampled so the compare cannot hinge on how the
    // packer rounds at the threshold.
    constexpr std::uint32_t kRandomSeed = 0x5e1f;
    std::mt19937 rng(kRandomSeed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const bool has_threshold = config.mode == ReluMode::MinThreshold || config.mode == ReluMode::MaxThreshold;
    std::vector<std::uint32_t> src_vec(num_elems);
    for (auto& word : src_vec) {
        float v = dist(rng);
        while (has_threshold && std::fabs(v - config.threshold) < (1.0f / 64.0f)) {
            v = dist(rng);
        }
        word = std::bit_cast<std::uint32_t>(v);
    }
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), src_vec);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), result_vec);
    ASSERT_EQ(result_vec.size(), src_vec.size());

    std::uint32_t mismatches = 0;
    for (std::uint32_t i = 0; i < num_elems; ++i) {
        const float expected = golden_relu(std::bit_cast<float>(src_vec[i]), config);
        if (result_vec[i] != std::bit_cast<std::uint32_t>(expected)) {
            if (mismatches == 0) {
                log_error(
                    LogTest,
                    "Pack ReLU mismatch at {}: in {:#x} expected {:#x} got {:#x}",
                    i,
                    src_vec[i],
                    std::bit_cast<std::uint32_t>(expected),
                    result_vec[i]);
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

}  // namespace unit_tests::compute::pack_relu

// Zero-mode ReLU on a 32-bit DEST: negative fp32 inputs pack as +0, everything else bit-exact.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarPackReluFp32Dest) {
    for (const bool dst_full_sync_en : {false, true}) {
        SCOPED_TRACE(dst_full_sync_en ? "dst_full_sync_en=true (SyncFull)" : "dst_full_sync_en=false (SyncHalf)");
        unit_tests::compute::pack_relu::run_single_core_pack_relu(
            this->device(),
            {.mode = unit_tests::compute::pack_relu::ReluMode::Zero, .dst_full_sync_en = dst_full_sync_en});
    }
}

// Threshold modes on a 32-bit DEST: the bf16 threshold must land in the high half-word of RELU_THRESHOLD. With the
// 16-bit-DEST pin it sits in the low half-word (~2e-41), so min-threshold degrades to zero-ReLU and max clamps to ~0.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarPackReluThresholdFp32Dest) {
    using unit_tests::compute::pack_relu::ReluMode;
    for (const ReluMode mode : {ReluMode::MinThreshold, ReluMode::MaxThreshold}) {
        for (const bool dst_full_sync_en : {false, true}) {
            SCOPED_TRACE(
                std::string(mode == ReluMode::MinThreshold ? "MinThreshold" : "MaxThreshold") +
                (dst_full_sync_en ? " dst_full_sync_en=true (SyncFull)" : " dst_full_sync_en=false (SyncHalf)"));
            unit_tests::compute::pack_relu::run_single_core_pack_relu(
                this->device(), {.mode = mode, .threshold = 0.5f, .dst_full_sync_en = dst_full_sync_en});
        }
    }
}

}  // namespace tt::tt_metal
