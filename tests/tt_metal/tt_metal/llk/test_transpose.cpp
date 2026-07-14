// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <random>
#include <sys/types.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/df/float32.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::transpose {

enum TransposeType : uint8_t { WH = 0 };

struct TransposeConfig {
    bool short_init;
    bool transpose_dest;
    uint32_t single_tile_size;
    std::vector<uint32_t> shape;
    TransposeType transpose_type;
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    bool dst_full_sync_en = false;
};

// Tiled dimensions derived from a 4-D NCHW tensor shape, with shared validation.
struct TransposeDims {
    uint32_t W;
    uint32_t H;
    uint32_t NC;
    uint32_t Wt;
    uint32_t Ht;
    uint32_t num_tensor_tiles;
};

static TransposeDims compute_and_validate_transpose_dims(const std::vector<uint32_t>& shape) {
    TT_FATAL(shape.size() == 4, "Error");
    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    TT_FATAL(W % 32 == 0 && H % 32 == 0, "Error");
    TT_FATAL(H > 0 && W > 0 && NC > 0, "Error");
    const uint32_t Wt = W / 32;
    // size of DST register, with unary r/w this currently only works if the entire Wt fits into DST for reduce
    TT_FATAL(Wt <= 16, "Error");
    const uint32_t Ht = H / 32;
    return TransposeDims{
        .W = W,
        .H = H,
        .NC = NC,
        .Wt = Wt,
        .Ht = Ht,
        .num_tensor_tiles = NC * H * W / (32 * 32),
    };
}

// Validate a WH-transpose device result against a CPU golden for any tested data format.
// 16-bit (bfloat16) packs two datums per uint32 and goes through format conversions, so it uses a
// tolerant float compare. 32-bit formats (Float32/Int32) are exact, lossless data movement, so they
// use an exact word compare.
void validate_transpose_wh(
    const std::vector<uint32_t>& src_vec,
    const std::vector<uint32_t>& shape,
    const std::vector<uint32_t>& result_vec,
    tt::DataFormat data_format) {
    TT_FATAL(shape.size() == 4, "Error");
    const vector<uint32_t> shapeR{shape[0], shape[1], shape[3], shape[2]};

    bool pass = false;
    int argfail = -1;
    if (tt::datum_size(data_format) == sizeof(uint32_t)) {
        // 32-bit datum: one uint32 per element, exact compare.
        auto src_linear =
            convert_layout<uint32_t>(src_vec, shape, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
        auto gold_lin = ::unit_tests::compute::gold_transpose_wh(src_linear, shape);
        auto gold_tiled =
            convert_layout<uint32_t>(gold_lin, shapeR, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);

        ASSERT_EQ(result_vec.size(), gold_tiled.size());
        const auto res_it = std::mismatch(result_vec.begin(), result_vec.end(), gold_tiled.begin()).first;
        pass = (res_it == result_vec.end());
        if (!pass) {
            argfail = static_cast<int>(std::distance(result_vec.begin(), res_it));
        }
    } else {
        // 16-bit datum (bfloat16): two datums packed per uint32, tolerant float compare.
        auto comparison_function = [](float a, float b) {
            const float rtol = 0.02f;
            const float atol = 1e-3f;
            const float absdiff = fabsf(a - b);
            return (absdiff <= atol) || (absdiff < rtol * fmaxf(fabsf(a), fabsf(b)));
        };
        auto src_linear = convert_layout<uint16_t>(
            u16_from_u32_vector(src_vec), shape, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
        auto gold_lin = ::unit_tests::compute::gold_transpose_wh(src_linear, shape);
        auto gold_tiled = u32_from_u16_vector(convert_layout<uint16_t>(
            gold_lin, shapeR, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));

        pass = packed_uint32_t_vector_comparison(result_vec, gold_tiled, comparison_function, &argfail);
    }

    if (!pass) {
        log_error(LogTest, "Transpose WH mismatch at position {}", argfail);
    }
    EXPECT_TRUE(pass);
}

// Build a TensorSpec describing a flat DRAM-interleaved buffer of `total_entries`
// pages, each `entry_size` bytes. Used to bind src/dst tensors as TensorParameters
// to the reader/writer kernels via the Metal 2.0 named TensorAccessor ctor.
static inline tt::tt_metal::TensorSpec make_flat_dram_tensor_spec(uint32_t entry_size, uint32_t total_entries) {
    const uint32_t entry_size_words = entry_size / sizeof(uint32_t);
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
    return tt::tt_metal::TensorSpec(tt::tt_metal::Shape{total_entries, entry_size_words}, tensor_layout);
}

void run_single_core_transpose(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TransposeConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const TransposeDims dims = compute_and_validate_transpose_dims(test_config.shape);
    const uint32_t NC = dims.NC, Wt = dims.Wt, Ht = dims.Ht, num_tensor_tiles = dims.num_tensor_tiles;

    uint32_t dram_buffer_size = test_config.single_tile_size * num_tensor_tiles;

    auto in_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(test_config.single_tile_size, num_tensor_tiles), TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(test_config.single_tile_size, num_tensor_tiles), TensorTopology{});

    constexpr uint32_t num_buffer_tiles = 32;
    constexpr uint32_t num_output_buffer_tiles = 32;

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};
    const experimental::TensorParamName IN_TENSOR{"in_tensor"};
    const experimental::TensorParamName OUT_TENSOR{"out_tensor"};

    experimental::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = test_config.single_tile_size,
        .num_entries = num_buffer_tiles,
        .data_format_metadata = test_config.data_format,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = test_config.single_tile_size,
        .num_entries = num_output_buffer_tiles,
        .data_format_metadata = test_config.data_format,
    };

    experimental::DataMovementHardwareConfig reader_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        reader_hw_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
    } else {
        reader_hw_config = experimental::DataMovementGen1Config{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default};
    }
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_8bank.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}},
        .runtime_arg_schema = {.runtime_arg_names = {"N", "Ht", "Wt", "HtWt"}},
        .hw_config = reader_hw_config,
    };

    experimental::DataMovementHardwareConfig writer_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        writer_hw_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = true};
    } else {
        writer_hw_config = experimental::DataMovementGen1Config{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default};
    }
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = writer_hw_config,
    };

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
    if (test_config.short_init) {
        compute_defines.emplace("SHORT_INIT", "1");
    }

    const char* compute_kernel_path = test_config.transpose_dest
                                          ? "tests/tt_metal/tt_metal/test_kernels/compute/transpose_wh_dest.cpp"
                                          : "tests/tt_metal/tt_metal/test_kernels/compute/transpose_wh.cpp";

    // Enable 32-bit dest accumulate for the 32-bit formats (Float32/Int32); the compute kernel forwards
    // DST_ACCUM_MODE as the transpose-dest EN_32BIT_DEST template arg, so the two must agree.
    const bool fp32_dest_acc_en =
        (test_config.data_format == tt::DataFormat::Float32 || test_config.data_format == tt::DataFormat::Int32);
    experimental::ComputeHardwareConfig compute_hw_config;
    experimental::ComputeUnpackToDestModes unpack_modes{};
    if (fp32_dest_acc_en) {
        unpack_modes = {{INPUT_DFB, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32}};
    }
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        compute_hw_config = experimental::ComputeGen2Config{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .unpack_to_dest_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_modes,
        };
    } else {
        compute_hw_config = experimental::ComputeGen1Config{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .unpack_to_dest_mode = unpack_modes,
        };
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = compute_kernel_path,
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
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
        .compile_time_args = {{"NHtWt", Ht * Wt * NC}},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "transpose_wh",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = {{node, {{"N", NC}, {"Ht", Ht}, {"Wt", Wt}, {"HtWt", Ht * Wt}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = {{node, {{"num_tiles", num_tensor_tiles}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN_TENSOR, experimental::ProgramRunArgs::TensorArgument{in_tensor}},
        {OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{out_tensor}},
    };
    experimental::SetProgramRunArgs(program_run, params);

    // Fixed seed so each test produces a repeatable input vector across runs.
    constexpr std::uint32_t kRandomSeed = 0x1234;
    vector<uint32_t> src_vec;
    const std::uint32_t n_u32 = dram_buffer_size / sizeof(uint32_t);
    // Fill src_vec with seeded random words: `dist` draws values and `convert` reinterprets
    // each draw into its uint32 word representation for the given data format.
    auto fill_random = [&](auto dist, auto convert) {
        src_vec.resize(n_u32);
        std::mt19937 rng(kRandomSeed);
        std::generate(src_vec.begin(), src_vec.end(), [&]() { return convert(dist(rng)); });
    };
    if (test_config.data_format == tt::DataFormat::Float32) {
        fill_random(
            std::uniform_real_distribution<float>(-100.0f, 100.0f), [](float v) { return std::bit_cast<uint32_t>(v); });
    } else if (test_config.data_format == tt::DataFormat::Int32) {
        fill_random(
            std::uniform_int_distribution<int32_t>(-10000, 10000), [](int32_t v) { return static_cast<uint32_t>(v); });
    } else {
        src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100.0f, kRandomSeed);
    }
    tt_metal::detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), src_vec);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), result_vec);

    const std::uint32_t bytes_per_elem = tt::datum_size(test_config.data_format);
    EXPECT_EQ(result_vec.size(), (dims.NC * dims.H * dims.W * bytes_per_elem) / sizeof(uint32_t));

    validate_transpose_wh(src_vec, test_config.shape, result_vec, test_config.data_format);
}

}  // namespace unit_tests::compute::transpose

TEST_F(LLKMeshDeviceFixture, TensixComputeTransposeWH) {
    unit_tests::compute::transpose::TransposeConfig test_config = {
        .short_init = false,
        .transpose_dest = false,
        .single_tile_size = 2 * 1024,
        .shape = {1, 3, 3 * 32 * 1, 4 * 32 * 1},
        .transpose_type = unit_tests::compute::transpose::TransposeType::WH};
    unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
}

TEST_F(LLKMeshDeviceFixture, TensixComputeTransposeWHShortInit) {
    unit_tests::compute::transpose::TransposeConfig test_config = {
        .short_init = true,
        .transpose_dest = false,
        .single_tile_size = 2 * 1024,
        .shape = {1, 3, 3 * 32 * 1, 4 * 32 * 1},
        .transpose_type = unit_tests::compute::transpose::TransposeType::WH};
    unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
}

TEST_F(LLKMeshDeviceFixture, TensixComputeTransposeWHDest) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "TensixComputeTransposeWHDest not implemented for Quasar yet";
    }
    unit_tests::compute::transpose::TransposeConfig test_config = {
        .short_init = false,
        .transpose_dest = true,
        .single_tile_size = 2 * 1024,
        .shape = {1, 3, 3 * 32 * 1, 4 * 32 * 1},
        .transpose_type = unit_tests::compute::transpose::TransposeType::WH};
    unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarTransposeWHDestFloat32) {
    // Tests SyncHalf and SyncFull
    for (const bool dst_full_sync_en : {false, true}) {
        SCOPED_TRACE(dst_full_sync_en ? "dst_full_sync_en=true (SyncFull)" : "dst_full_sync_en=false (SyncHalf)");
        unit_tests::compute::transpose::TransposeConfig test_config = {
            .short_init = false,
            .transpose_dest = true,
            .single_tile_size = constants::TILE_HW * sizeof(uint32_t),
            .shape = {1, 1, 64, 64},
            .transpose_type = unit_tests::compute::transpose::TransposeType::WH,
            .data_format = tt::DataFormat::Float32,
            .dst_full_sync_en = dst_full_sync_en,
        };
        unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarTransposeWHDestFloat16b) {
    // 16-bit dest (EN_32BIT_DEST=false): exercises the implied-math-format-disabled config path.
    // Tests SyncHalf and SyncFull
    for (const bool dst_full_sync_en : {false, true}) {
        SCOPED_TRACE(dst_full_sync_en ? "dst_full_sync_en=true (SyncFull)" : "dst_full_sync_en=false (SyncHalf)");
        unit_tests::compute::transpose::TransposeConfig test_config = {
            .short_init = false,
            .transpose_dest = true,
            .single_tile_size = constants::TILE_HW * sizeof(uint16_t),
            .shape = {1, 1, 64, 64},
            .transpose_type = unit_tests::compute::transpose::TransposeType::WH,
            .data_format = tt::DataFormat::Float16_b,
            .dst_full_sync_en = dst_full_sync_en,
        };
        unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
    }
}

}  // namespace tt::tt_metal
