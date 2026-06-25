// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/types/arch.hpp>
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
using namespace constants;

namespace unit_tests::compute::reduce {

enum ReduceDim : uint8_t { H = 0, W = 1, HW = 2 };

enum ReduceType : uint8_t { SUM = 0, AVG = 1, MAX = 2 };
struct ReduceConfig {
    tt_metal::Tile tile_shape = tt_metal::Tile({TILE_HEIGHT, TILE_WIDTH});
    std::vector<uint32_t> shape;
    ReduceDim reduce_dim;
    ReduceType reduce_type = ReduceType::SUM;
    float data_gen_rand_max;
    int data_gen_seed;
    float data_gen_offset;
    float atol;
    float rtol;
    std::function<std::vector<uint16_t>(
        const std::vector<uint16_t>&, const std::vector<uint32_t>&, float, uint8_t, bool)>
        golden_function;
    std::vector<uint32_t> result_shape;
    bool math_only_reduce = false;
    // Whether or not we want the result to be stored in DST in FP32:
    bool fp32_dest_acc_en = false;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
};

float get_scaler(const ReduceConfig& test_config) {
    uint32_t H = test_config.shape[2];
    uint32_t W = test_config.shape[3];
    // If PoolType is MAX or SUM, then the operation is determined by PoolType,
    // but the scaler is 1
    if (test_config.reduce_type != ReduceType::AVG) {
        return 1.0f;
    }  // If PoolType is AVG, the scaler depends on PoolDim, but the op is SUM
    switch (test_config.reduce_dim) {
        case ReduceDim::H: return (1.0f / H);
        case ReduceDim::W: return (1.0f / W);
        case ReduceDim::HW: return (1.0f / (H * W));
        default: {
            TT_THROW("Unsupported ReduceDim={}", test_config.reduce_dim);
            break;
        }
    }
}

// Tiled dimensions and buffer sizes derived from a 4-D NCHW tensor shape, with shared validation.
struct ReduceDims {
    uint32_t tile_H;
    uint32_t tile_W;
    uint32_t W;
    uint32_t H;
    uint32_t NC;
    uint32_t N;
    uint32_t Wt;
    uint32_t Ht;
    uint32_t num_tensor_tiles;
    uint32_t single_tile_bytes;
    uint32_t dram_buffer_size;
    uint32_t num_golden_elements;
    uint32_t output_size_bytes;
};

static ReduceDims compute_and_validate_reduce_dims(const ReduceConfig& test_config) {
    ReduceDims dims{};
    dims.tile_H = test_config.tile_shape.get_tile_shape()[0];
    dims.tile_W = test_config.tile_shape.get_tile_shape()[1];
    dims.W = test_config.shape[3];
    dims.H = test_config.shape[2];
    dims.NC = test_config.shape[1] * test_config.shape[0];
    dims.N = test_config.shape[0] * test_config.shape[1];

    TT_FATAL(
        (dims.tile_H == 16 && dims.tile_W == 32) || (dims.tile_H == 32 && dims.tile_W == 32),
        "Error: Invalid tile shape");
    TT_FATAL(
        dims.W % dims.tile_W == 0 && dims.H % dims.tile_H == 0,
        "Error: Tensor height/width must be multiple of tile height/width");
    TT_FATAL(dims.H > 0 && dims.W > 0 && dims.NC > 0, "Error: All tensor dims must be greater than 0");

    dims.Wt = dims.W / dims.tile_W;
    dims.Ht = dims.H / dims.tile_H;
    dims.num_tensor_tiles = dims.NC * dims.H * dims.W / (dims.tile_W * dims.tile_H);

    const uint32_t divisor = test_config.reduce_dim == ReduceDim::W ? dims.Wt : dims.Ht;
    TT_FATAL(dims.num_tensor_tiles % divisor == 0, "Error");

    dims.single_tile_bytes = 2 * (dims.tile_W * dims.tile_H);
    dims.dram_buffer_size = dims.single_tile_bytes * dims.num_tensor_tiles;

    switch (test_config.reduce_dim) {
        case ReduceDim::H:
            dims.num_golden_elements = dims.NC * dims.W * 32 / 2;
            dims.output_size_bytes = dims.dram_buffer_size / dims.Ht;
            break;
        case ReduceDim::W:
            dims.num_golden_elements = dims.NC * dims.H * dims.tile_W / 2;
            dims.output_size_bytes = dims.dram_buffer_size / dims.Wt;
            break;
        case ReduceDim::HW:
            dims.num_golden_elements = dims.NC * 32 * 32 / 2;
            dims.output_size_bytes = dims.dram_buffer_size / (dims.Ht * dims.Wt);
            break;
        default: TT_THROW("Unsupported reduce dim!");
    }

    return dims;
}

void set_math_fid_masks_binary(
    uint16_t& srca_fid_mask, uint16_t& srcb_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2: {
            srcb_fid_mask = 0xFFFE;
            ;
            break;
        }
        case MathFidelity::LoFi: {
            srca_fid_mask = 0xFFF8;
            srcb_fid_mask = 0xFFFE;
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }
}

std::pair<KernelHandle, KernelHandle> add_reader_writer_kernels(
    distributed::MeshWorkload& workload,
    distributed::MeshCoordinateRange& device_range,
    const CoreCoord& logical_core,
    const ReduceConfig& test_config,
    const std::shared_ptr<distributed::MeshBuffer>& src_dram_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& dst_dram_buffer,
    uint32_t dst_buffer_id) {
    uint32_t tile_H = test_config.tile_shape.get_tile_shape()[0], tile_W = test_config.tile_shape.get_tile_shape()[1];
    uint32_t W = test_config.shape[3], H = test_config.shape[2], NC = test_config.shape[1] * test_config.shape[0];
    uint32_t N = test_config.shape[0] * test_config.shape[1];
    uint32_t Wt = W / tile_W;
    uint32_t Ht = H / tile_H;
    uint32_t num_tensor_tiles = NC * H * W / (tile_W * tile_H);
    float scaler = get_scaler(test_config);

    auto& program = workload.get_programs().at(device_range);

    KernelHandle unary_reader_kernel;
    KernelHandle unary_writer_kernel;

    switch (test_config.reduce_dim) {
        case ReduceDim::H: {
            bfloat16 bfloat_scaler_value = bfloat16(scaler);
            uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
            std::vector<uint32_t> reader_compile_args = {};
            tt_metal::TensorAccessorArgs(src_dram_buffer).append_to(reader_compile_args);
            reader_compile_args.push_back(packed_scaler_value);
            std::map<std::string, std::string> reader_defines = {{"REDUCE_SCALER", "1"}};

            unary_reader_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp",
                logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = reader_compile_args,
                    .defines = reader_defines});

            std::vector<uint32_t> writer_compile_args = {dst_buffer_id};
            tt_metal::TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_args);

            unary_writer_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",  // no need to transpose the
                                                                                         // output since output Ht=1
                logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = writer_compile_args});

            tt_metal::SetRuntimeArgs(
                program, unary_reader_kernel, logical_core, {src_dram_buffer->address(), N, Ht, Wt, Ht * Wt});

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel,
                logical_core,
                {
                    dst_dram_buffer->address(),
                    (uint32_t)0,           // dram bank id
                    num_tensor_tiles / Ht  // num tiles
                });

            return {unary_reader_kernel, unary_writer_kernel};
            break;
        }
        case ReduceDim::HW: {
            scaler = std::sqrt(scaler);
        }  // Needed because AVG pool multiplies twice by the scaler
        case ReduceDim::W: {
            std::vector<uint32_t> reader_compile_args = {};
            tt_metal::TensorAccessorArgs(src_dram_buffer).append_to(reader_compile_args);

            unary_reader_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank_reduce.cpp",
                logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = reader_compile_args});

            std::vector<uint32_t> writer_compile_args = {dst_buffer_id};
            tt_metal::TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_args);

            unary_writer_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
                logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .compile_args = writer_compile_args});

            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel,
                logical_core,
                {
                    src_dram_buffer->address(),
                    (uint32_t)0,  // dram bank id
                    (uint32_t)0,  // unused
                    num_tensor_tiles,
                    NC,
                    Ht,
                    Wt,
                    Ht * Wt,
                    *reinterpret_cast<uint32_t*>(&scaler),
                });

            uint32_t num_tiles =
                test_config.reduce_dim == ReduceDim::W ? (num_tensor_tiles / Wt) : (num_tensor_tiles / (Wt * Ht));
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel,
                logical_core,
                {dst_dram_buffer->address(),
                 (uint32_t)0,  // dram bank id
                 num_tiles});

            return {unary_reader_kernel, unary_writer_kernel};
            break;
        }
        default: TT_THROW("Unsupported reduce dim!");
        return {0, 0};
    }
}

std::string get_reduce_dim_define_string(const ReduceDim& reduce_dim) {
    std::string reduce_dim_define_str;
    switch (reduce_dim) {
        case ReduceDim::H: reduce_dim_define_str = "ReduceDim::REDUCE_COL"; break;
        case ReduceDim::W: reduce_dim_define_str = "ReduceDim::REDUCE_ROW"; break;
        case ReduceDim::HW: reduce_dim_define_str = "ReduceDim::REDUCE_SCALAR"; break;
        default: TT_THROW("Unsupported reduce dim!");
    }
    return reduce_dim_define_str;
}

std::string get_compute_kernel_name(const ReduceDim& reduce_dim) {
    std::string compute_kernel_name;
    switch (reduce_dim) {
        case ReduceDim::H: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_h.cpp"; break;
        case ReduceDim::W: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_w.cpp"; break;
        case ReduceDim::HW: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_hw.cpp"; break;
        default: TT_THROW("Unsupported reduce dim!");
    }
    return compute_kernel_name;
}

void validate_reduce_result(
    const std::vector<uint32_t>& result_vec,
    uint32_t num_golden_elements,
    const ReduceConfig& test_config,
    const std::vector<uint32_t>& src_vec,
    float scaler) {
    EXPECT_EQ(result_vec.size(), num_golden_elements);

    int argfail = -1;
    auto comparison_function = [&](float a, float b) {
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        auto result = (absdiff <= test_config.atol) || (absdiff <= test_config.rtol * maxabs);
        return result;
    };

    auto u16_src0_vec = u16_from_u32_vector(src_vec);
    if (test_config.reduce_type == ReduceType::AVG) {
        uint16_t srca_fid_mask = 0xFFFF;
        uint16_t srcb_fid_mask = 0xFFFF;
        set_math_fid_masks_binary(srca_fid_mask, srcb_fid_mask, test_config.math_fidelity);
        uint32_t uint32_scaler = *reinterpret_cast<uint32_t*>(&scaler);
        uint32_scaler &= (0xFFFFFFFF & (srcb_fid_mask << 16));
        scaler = *reinterpret_cast<float*>(&uint32_scaler);
        for (unsigned short& val : u16_src0_vec) {
            val &= srca_fid_mask;
        }
    }
    uint32_t tile_H = test_config.tile_shape.get_tile_shape()[0];
    uint32_t tile_W = test_config.tile_shape.get_tile_shape()[1];
    std::vector<uint16_t> src_linear = convert_layout<uint16_t>(
        u16_src0_vec,
        test_config.shape,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR,
        PhysicalSize{tile_H, tile_W});
    std::vector<uint16_t> gold_reduced =
        test_config.golden_function(src_linear, test_config.shape, scaler, uint8_t(test_config.reduce_type), true);

    auto gold_4f_u32 = u32_from_u16_vector(convert_layout<uint16_t>(
        gold_reduced,
        test_config.result_shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        PhysicalSize{tile_H, tile_W}));

    bool pass = packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    if (!pass) {
        log_error(LogTest, "Failure position={}", argfail);
    }
    EXPECT_TRUE(pass);
}

static experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines build_reduce_defines(
    const ReduceConfig& test_config) {
    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines reduce_defines;
    reduce_defines.emplace_back("REDUCE_DIM", get_reduce_dim_define_string(test_config.reduce_dim));
    switch (test_config.reduce_type) {
        case ReduceType::SUM: reduce_defines.emplace_back("REDUCE_OP", "PoolType::SUM"); break;
        case ReduceType::AVG: reduce_defines.emplace_back("REDUCE_OP", "PoolType::AVG"); break;
        case ReduceType::MAX: reduce_defines.emplace_back("REDUCE_OP", "PoolType::MAX"); break;
    }
    reduce_defines.emplace_back("MATH_ONLY", test_config.math_only_reduce ? "1" : "0");
    reduce_defines.emplace_back("DST_ACCUM_MODE", test_config.fp32_dest_acc_en ? "1" : "0");
    return reduce_defines;
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

void run_single_core_reduce_program_quasar(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ReduceConfig& test_config) {
    const experimental::metal2_host_api::NodeCoord node{0, 0};

    const ReduceDims dims = compute_and_validate_reduce_dims(test_config);

    float scaler = get_scaler(test_config);
    if (test_config.reduce_dim == ReduceDim::HW) {
        scaler = std::sqrt(scaler);
    }

    const uint32_t num_input_pages = dims.dram_buffer_size / dims.single_tile_bytes;
    const uint32_t num_output_pages = dims.output_size_bytes / dims.single_tile_bytes;
    auto in_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(dims.single_tile_bytes, num_input_pages), TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_dram_tensor_spec(dims.single_tile_bytes, num_output_pages), TensorTopology{});

    constexpr uint32_t num_buffer_tiles = 32;
    constexpr uint32_t num_output_buffer_tiles = 32;
    constexpr const char* SRC0_DFB = "src0_dfb";
    constexpr const char* SRC1_DFB = "src1_dfb";
    constexpr const char* DST_DFB = "dst_dfb";
    constexpr const char* READER = "reader";
    constexpr const char* WRITER = "writer";
    constexpr const char* COMPUTE = "compute";
    constexpr const char* IN_TENSOR = "in_tensor";
    constexpr const char* OUT_TENSOR = "out_tensor";

    // Match pre-migration behavior: legacy DataflowBufferConfig set enable_implicit_sync=false on all 3 DFBs.
    experimental::metal2_host_api::DataflowBufferSpec src0_dfb_spec{
        .unique_id = SRC0_DFB,
        .entry_size = dims.single_tile_bytes,
        .num_entries = num_buffer_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .tile_format_metadata = test_config.tile_shape,
        .disable_implicit_sync = true,
    };
    experimental::metal2_host_api::DataflowBufferSpec src1_dfb_spec{
        .unique_id = SRC1_DFB,
        .entry_size = 2 * TILE_WIDTH * TILE_HEIGHT,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .tile_format_metadata = tt_metal::Tile({32, 32}),
        .disable_implicit_sync = true,
    };
    experimental::metal2_host_api::DataflowBufferSpec dst_dfb_spec{
        .unique_id = DST_DFB,
        .entry_size = dims.single_tile_bytes,
        .num_entries = num_output_buffer_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
        .tile_format_metadata = test_config.tile_shape,
        .disable_implicit_sync = true,
    };

    // Build reader spec depending on reduce dim
    std::string reader_kernel_path;
    experimental::metal2_host_api::KernelSpec::CompileTimeArgBindings reader_cta_bindings;
    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines reader_defines;
    std::vector<std::string> reader_named_runtime_args;
    if (test_config.reduce_dim == ReduceDim::H) {
        reader_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp";
        bfloat16 bfloat_scaler_value = bfloat16(scaler);
        uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
        reader_cta_bindings = {{"scaler", packed_scaler_value}};
        reader_defines.emplace_back("REDUCE_SCALER", "1");
        reader_named_runtime_args = {"N", "Ht", "Wt", "HtWt"};
    } else {
        reader_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank_reduce.cpp";
        reader_named_runtime_args = {"num_tiles", "scaler"};
    }

    experimental::metal2_host_api::KernelSpec reader_spec{
        .unique_id = READER,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{reader_kernel_path},
        .num_threads = 1,
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = SRC0_DFB,
                 .local_accessor_name = "out_data",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = SRC1_DFB,
                 .local_accessor_name = "out_scaler",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}},
        .compile_time_arg_bindings = reader_cta_bindings,
        .runtime_arguments_schema = {.named_runtime_args = reader_named_runtime_args},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"},
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = DST_DFB,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}},
        .runtime_arguments_schema = {.named_runtime_args = {"num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =
            experimental::metal2_host_api::KernelSpec::SourceFilePath{get_compute_kernel_name(test_config.reduce_dim)},
        .num_threads = 1,
        .compiler_options = {.defines = build_reduce_defines(test_config)},
        .dfb_bindings =
            {{
                 .dfb_spec_name = SRC0_DFB,
                 .local_accessor_name = "in_data",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = SRC1_DFB,
                 .local_accessor_name = "in_scaler",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = DST_DFB,
                 .local_accessor_name = "out",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .compile_time_arg_bindings = {{"Ht", dims.Ht}, {"Wt", dims.Wt}, {"NC", dims.NC}},
        .config_spec =
            experimental::metal2_host_api::ComputeConfiguration{
                .math_fidelity = test_config.math_fidelity,
                .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
                .dst_full_sync_en = test_config.dst_full_sync_en,
            },
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "single_core_reduce",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, dst_dfb_spec},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    // Reader/writer RTAs depend on reduce_dim
    std::unordered_map<std::string, uint32_t> reader_named_rtas;
    uint32_t writer_num_tiles;
    if (test_config.reduce_dim == ReduceDim::H) {
        reader_named_rtas = {{"N", dims.N}, {"Ht", dims.Ht}, {"Wt", dims.Wt}, {"HtWt", dims.Ht * dims.Wt}};
        writer_num_tiles = dims.num_tensor_tiles / dims.Ht;
    } else {
        reader_named_rtas = {{"num_tiles", dims.num_tensor_tiles}, {"scaler", *reinterpret_cast<uint32_t*>(&scaler)}};
        writer_num_tiles = test_config.reduce_dim == ReduceDim::W ? (dims.num_tensor_tiles / dims.Wt)
                                                                  : (dims.num_tensor_tiles / (dims.Wt * dims.Ht));
    }

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = READER,
            .named_runtime_args = {{.node = node, .args = reader_named_rtas}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = WRITER,
            .named_runtime_args = {{.node = node, .args = {{"num_tiles", writer_num_tiles}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = COMPUTE,
        },
    };
    params.tensor_args = {
        {.tensor_parameter_name = IN_TENSOR, .tensor = in_tensor},
        {.tensor_parameter_name = OUT_TENSOR, .tensor = out_tensor},
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dims.dram_buffer_size, test_config.data_gen_rand_max, test_config.data_gen_seed, test_config.data_gen_offset);
    tt_metal::detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), src_vec);

    auto* dev = mesh_device->get_devices()[0];
    tt_metal::detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), result_vec);

    validate_reduce_result(result_vec, dims.num_golden_elements, test_config, src_vec, get_scaler(test_config));

    log_info(
        LogTest,
        "TileDimH = {}, TileDimW = {}, MathFid = {}, ReduceType = {}, FP32DestAcc = {}, DstSyncFull = {}",
        test_config.tile_shape.get_tile_shape()[0],
        test_config.tile_shape.get_tile_shape()[1],
        test_config.math_fidelity,
        test_config.reduce_type,
        test_config.fp32_dest_acc_en,
        test_config.dst_full_sync_en);
}

void run_single_core_reduce_program_legacy(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ReduceConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    CoreCoord core = {0, 0};

    const ReduceDims dims = compute_and_validate_reduce_dims(test_config);

    distributed::DeviceLocalBufferConfig src_local_config{
        .page_size = dims.single_tile_bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig src_buffer_config{.size = dims.dram_buffer_size};

    distributed::DeviceLocalBufferConfig dst_local_config{
        .page_size = dims.single_tile_bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig dst_buffer_config{.size = dims.output_size_bytes};

    std::shared_ptr<distributed::MeshBuffer> src_dram_buffer =
        distributed::MeshBuffer::create(src_buffer_config, src_local_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer =
        distributed::MeshBuffer::create(dst_buffer_config, dst_local_config, mesh_device.get());

    uint32_t num_buffer_tiles = 32;
    uint32_t num_output_buffer_tiles = 32;

    {
        uint32_t src0_cb_index = 0;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_buffer_tiles * dims.single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, dims.single_tile_bytes)
                .set_tile_dims(src0_cb_index, test_config.tile_shape);
        tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_buffer_tiles * dims.single_tile_bytes, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, dims.single_tile_bytes)
                .set_tile_dims(ouput_cb_index, test_config.tile_shape);
        tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

        tt_metal::CircularBufferConfig cb_temp_reduce_tile_config =
            tt_metal::CircularBufferConfig(
                2 * (2 * TILE_WIDTH * TILE_HEIGHT), {{CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_2, dims.single_tile_bytes)
                .set_tile_dims(CBIndex::c_2, tt_metal::Tile({32, 32}));
        tt_metal::CreateCircularBuffer(program_, core, cb_temp_reduce_tile_config);
    }

    const uint32_t dst_buffer_id = static_cast<uint32_t>(tt::CBIndex::c_16);

    add_reader_writer_kernels(
        workload, device_range, core, test_config, src_dram_buffer, dst_dram_buffer, dst_buffer_id);

    vector<uint32_t> compute_kernel_args = {
        uint(dims.Ht),
        uint(dims.Wt),
        uint(dims.NC),
    };

    std::map<std::string, std::string> reduce_defines = {
        {"REDUCE_DIM", get_reduce_dim_define_string(test_config.reduce_dim)}};
    switch (test_config.reduce_type) {
        case ReduceType::SUM: {
            reduce_defines["REDUCE_OP"] = "PoolType::SUM";
            break;
        }
        case ReduceType::AVG: {
            reduce_defines["REDUCE_OP"] = "PoolType::AVG";
            break;
        }
        case ReduceType::MAX: {
            reduce_defines["REDUCE_OP"] = "PoolType::MAX";
            break;
        }
    }
    reduce_defines["MATH_ONLY"] = test_config.math_only_reduce ? "1" : "0";
    reduce_defines["DST_ACCUM_MODE"] = test_config.fp32_dest_acc_en ? "1" : "0";

    std::string compute_kernel_name = get_compute_kernel_name(test_config.reduce_dim);

    tt_metal::CreateKernel(
        program_,
        compute_kernel_name,
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = test_config.math_fidelity,
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
            .defines = reduce_defines});

    vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dims.dram_buffer_size, test_config.data_gen_rand_max, test_config.data_gen_seed, test_config.data_gen_offset);

    distributed::WriteShard(cq, src_dram_buffer, src_vec, zero_coord);

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, zero_coord);

    validate_reduce_result(result_vec, dims.num_golden_elements, test_config, src_vec, get_scaler(test_config));

    log_info(
        LogTest,
        "TileDimH = {}, TileDimW = {}, MathFid = {}, ReduceType = {}, FP32DestAcc = {}, DstSyncFull = {}",
        test_config.tile_shape.get_tile_shape()[0],
        test_config.tile_shape.get_tile_shape()[1],
        test_config.math_fidelity,
        test_config.reduce_type,
        test_config.fp32_dest_acc_en,
        test_config.dst_full_sync_en);
}

void run_single_core_reduce_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ReduceConfig& test_config) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        run_single_core_reduce_program_quasar(mesh_device, test_config);
    } else {
        run_single_core_reduce_program_legacy(mesh_device, test_config);
    }
}

}  // namespace unit_tests::compute::reduce

using namespace unit_tests::compute::reduce;

TEST_F(LLKMeshDeviceFixture, TensixComputeReduceH) {
    if (this->arch_ != tt::ARCH::BLACKHOLE && this->arch_ != tt::ARCH::QUASAR) {
        // (issue #10181: disabling due to sporadic failures in slow dispatch mode)
        GTEST_SKIP();
    }
    std::vector<uint32_t> shape = {1, 3, 19 * TILE_HEIGHT, 17 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], TILE_HEIGHT, shape[3]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool dst_full_sync_en : {true, false}) {
                    if (this->arch_ == tt::ARCH::QUASAR &&
                        !(!fp32_dest_acc_en && !dst_full_sync_en && reduce_type == ReduceType::AVG &&
                          math_fid == uint8_t(MathFidelity::HiFi4))) {
                        // TODO (#38092): Remove when we can run back to back tests on Quasar
                        continue;
                    }
                    ReduceConfig test_config = {
                        .shape = shape,
                        .reduce_type = ReduceType(reduce_type),
                        .data_gen_rand_max = 10.0f,
                        .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                        .data_gen_offset = -10.0f,
                        .atol = 1e-2f,
                        .rtol = 0.08f,
                        .golden_function = ::unit_tests::compute::gold_reduce_h,
                        .result_shape = result_shape,
                        .fp32_dest_acc_en = fp32_dest_acc_en,
                        .dst_full_sync_en = dst_full_sync_en,
                        .math_fidelity = MathFidelity(math_fid),
                    };
                    run_single_core_reduce_program(this->devices_.at(0), test_config);
                }
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeReduceW) {
    std::vector<uint32_t> shape = {1, 3, 17 * TILE_HEIGHT, 19 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool dst_full_sync_en : {true, false}) {
                    if (this->arch_ == tt::ARCH::QUASAR &&
                        !(!fp32_dest_acc_en && !dst_full_sync_en && reduce_type == ReduceType::AVG &&
                          math_fid == uint8_t(MathFidelity::HiFi4))) {
                        // TODO (#38092): Remove when we can run back to back tests on Quasar
                        continue;
                    }
                    ReduceConfig test_config = {
                        .shape = shape,
                        .reduce_dim = ReduceDim::W,
                        .reduce_type = ReduceType(reduce_type),
                        .data_gen_rand_max = 10.0f,
                        .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                        .data_gen_offset = -10.0f,
                        .atol = 1e-2f,
                        .rtol = 0.08f,
                        .golden_function = ::unit_tests::compute::gold_reduce_w,
                        .result_shape = result_shape,
                        .fp32_dest_acc_en = fp32_dest_acc_en,
                        .dst_full_sync_en = dst_full_sync_en,
                        .math_fidelity = MathFidelity(math_fid),
                    };
                    run_single_core_reduce_program(this->devices_.at(0), test_config);
                }
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeReduceHW) {
    std::vector<uint32_t> shape = {1, 2, 7 * TILE_HEIGHT, 5 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], 32, 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                // Currently fp32 dest unsupported with reduce scalar
                if (fp32_dest_acc_en && this->arch_ != tt::ARCH::QUASAR) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    if (this->arch_ == tt::ARCH::QUASAR &&
                        !(!fp32_dest_acc_en && !dst_full_sync_en && reduce_type == ReduceType::AVG &&
                          math_fid == uint8_t(MathFidelity::HiFi4))) {
                        // TODO (#38092): Remove when we can run back to back tests on Quasar
                        continue;
                    }
                    ReduceConfig test_config = {
                        .shape = shape,
                        .reduce_dim = ReduceDim::HW,
                        .reduce_type = ReduceType(reduce_type),
                        .data_gen_rand_max = 10.0f,
                        .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                        .data_gen_offset = -10.0f,
                        .atol = 1e-2f,
                        .rtol = 0.08f,
                        .golden_function = ::unit_tests::compute::gold_reduce_hw,
                        .result_shape = result_shape,
                        .fp32_dest_acc_en = fp32_dest_acc_en,
                        .dst_full_sync_en = dst_full_sync_en,
                        .math_fidelity = MathFidelity(math_fid)};
                    run_single_core_reduce_program(this->devices_.at(0), test_config);
                }
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeReduceHMathOnly) {
    if (this->arch_ != tt::ARCH::BLACKHOLE && this->arch_ != tt::ARCH::QUASAR) {
        // (issue #10181: disabling due to sporadic failures in slow dispatch mode)
        GTEST_SKIP();
    }
    std::vector<uint32_t> shape = {1, 3, 19 * TILE_HEIGHT, 17 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], TILE_HEIGHT, shape[3]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool dst_full_sync_en : {true, false}) {
                    if (this->arch_ == tt::ARCH::QUASAR &&
                        !(!fp32_dest_acc_en && !dst_full_sync_en && reduce_type == ReduceType::AVG &&
                          math_fid == uint8_t(MathFidelity::HiFi4))) {
                        // TODO (#38092): Remove when we can run back to back tests on Quasar
                        continue;
                    }
                    ReduceConfig test_config = {
                        .shape = shape,
                        .reduce_dim = ReduceDim::H,
                        .reduce_type = ReduceType(reduce_type),
                        .data_gen_rand_max = 10.0f,
                        .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                        .data_gen_offset = -10.0f,
                        .atol = 1e-2f,
                        .rtol = 0.08f,
                        .golden_function = ::unit_tests::compute::gold_reduce_h,
                        .result_shape = result_shape,
                        .math_only_reduce = true,
                        .fp32_dest_acc_en = fp32_dest_acc_en,
                        .dst_full_sync_en = dst_full_sync_en,
                        .math_fidelity = MathFidelity(math_fid)};
                    run_single_core_reduce_program(this->devices_.at(0), test_config);
                }
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeReduceWMathOnly) {
    std::vector<uint32_t> shape = {1, 3, 17 * TILE_HEIGHT, 19 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool dst_full_sync_en : {true, false}) {
                    if (this->arch_ == tt::ARCH::QUASAR &&
                        !(!fp32_dest_acc_en && !dst_full_sync_en && reduce_type == ReduceType::AVG &&
                          math_fid == uint8_t(MathFidelity::HiFi4))) {
                        // TODO (#38092): Remove when we can run back to back tests on Quasar
                        continue;
                    }
                    ReduceConfig test_config = {
                        .shape = shape,
                        .reduce_dim = ReduceDim::W,
                        .reduce_type = ReduceType(reduce_type),
                        .data_gen_rand_max = 10.0f,
                        .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                        .data_gen_offset = -10.0f,
                        .atol = 1e-2f,
                        .rtol = 0.08f,
                        .golden_function = ::unit_tests::compute::gold_reduce_w,
                        .result_shape = result_shape,
                        .math_only_reduce = true,
                        .fp32_dest_acc_en = fp32_dest_acc_en,
                        .dst_full_sync_en = dst_full_sync_en,
                        .math_fidelity = MathFidelity(math_fid)};
                    run_single_core_reduce_program(this->devices_.at(0), test_config);
                }
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeReduceHWMathOnly) {
    std::vector<uint32_t> shape = {1, 2, 7 * TILE_HEIGHT, 5 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], 32, 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                // Currently fp32 dest unsupported with reduce scalar
                if (fp32_dest_acc_en && this->arch_ != tt::ARCH::QUASAR) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    if (this->arch_ == tt::ARCH::QUASAR &&
                        !(!fp32_dest_acc_en && !dst_full_sync_en && reduce_type == ReduceType::AVG &&
                          math_fid == uint8_t(MathFidelity::HiFi4))) {
                        // TODO (#38092): Remove when we can run back to back tests on Quasar
                        continue;
                    }
                    ReduceConfig test_config = {
                        .shape = shape,
                        .reduce_dim = ReduceDim::HW,
                        .reduce_type = ReduceType(reduce_type),
                        .data_gen_rand_max = 10.0f,
                        .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                        .data_gen_offset = -10.0f,
                        .atol = 1e-2f,
                        .rtol = 0.08f,
                        .golden_function = ::unit_tests::compute::gold_reduce_hw,
                        .result_shape = result_shape,
                        .math_only_reduce = true,
                        .fp32_dest_acc_en = fp32_dest_acc_en,
                        .dst_full_sync_en = dst_full_sync_en,
                        .math_fidelity = MathFidelity(math_fid)};
                    run_single_core_reduce_program(this->devices_.at(0), test_config);
                }
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeReduceWTinyTiles) {
    tt_metal::Tile tile_shape = tt_metal::Tile({TILE_HEIGHT / 2, TILE_WIDTH});
    std::vector<uint32_t> shape = {1, 1, 1 * tile_shape.get_tile_shape()[0], 13 * tile_shape.get_tile_shape()[1]};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], tile_shape.get_tile_shape()[1]};
    if (this->arch_ == tt::ARCH::QUASAR) {
        // Tiny tiles not yet supported on Quasar
        GTEST_SKIP();
    }
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool dst_full_sync_en : {true, false}) {
                    ReduceConfig test_config = {
                        .tile_shape = tile_shape,
                        .shape = shape,
                        .reduce_dim = ReduceDim::W,
                        .reduce_type = ReduceType(reduce_type),
                        .data_gen_rand_max = 0.0f,
                        .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                        .data_gen_offset = 1.0f,
                        .atol = 1e-2f,
                        .rtol = 0.08f,
                        .golden_function = ::unit_tests::compute::gold_reduce_w,
                        .result_shape = result_shape,
                        .fp32_dest_acc_en = fp32_dest_acc_en,
                        .dst_full_sync_en = dst_full_sync_en,
                        .math_fidelity = MathFidelity(math_fid),
                    };
                    run_single_core_reduce_program(this->devices_.at(0), test_config);
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
