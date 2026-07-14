// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Test pipeline (one core):
//   Input tensor (DRAM/L1) → reader kernel (producer) → DFB → writer kernel (consumer) → output tensor
//
// WH/BH: 1 DM thread each — Gen1 BRISC (producer) + NCRISC (consumer), STRIDED DFB
// Quasar: 3 DM threads each — Gen2 reader (producer) + Gen2 writer (consumer), STRIDED DFB
//

#include <gtest/gtest.h>
#include <fmt/format.h>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "tests/tt_metal/tt_metal/common/device_fixture.hpp"
#include "tests/tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/api/ttnn/distributed/api.hpp"

namespace {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::tt_metal::experimental::test_helpers;
using namespace ttnn;

// ============================================================================
// Kernel source paths
// ============================================================================

constexpr const char* kReaderKernelPath =
    "tests/ttnn/unit_tests/gtests/accessor/kernels/reader_strided_pages_dfb.cpp";
constexpr const char* kWriterKernelPath =
    "tests/ttnn/unit_tests/gtests/accessor/kernels/writer_strided_pages_dfb.cpp";

// ============================================================================
// Helper: build KERNEL_COMPILE_TIME_ARGS define value from TensorAccessorArgs CTAs
// ============================================================================

// Returns the comma-separated CTA value string for the KERNEL_COMPILE_TIME_ARGS define.
// This enables device kernels that use TensorAccessorArgs<0, 0>() (positional CTA access)
// to compile correctly under the Metal 2.0 API, which does not pass positional CTAs through
// compile_time_args.
std::string build_cta_define(const TensorAccessorArgs& accessor_args) {
    const auto ctas = accessor_args.get_compile_time_args();
    std::ostringstream ss;
    for (size_t i = 0; i < ctas.size(); ++i) {
        if (i > 0) {
            ss << ',';
        }
        ss << ctas[i];
    }
    return ss.str();
}

// ============================================================================
// Helper: map DataType to a DM-safe DataFormat for DFB data_format_metadata
// ============================================================================
tt::DataFormat dtype_to_dm_data_format(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return tt::DataFormat::Float16_b;
        case DataType::FLOAT32:  return tt::DataFormat::Float32;
        case DataType::UINT8:    return tt::DataFormat::RawUInt8;
        case DataType::UINT16:   return tt::DataFormat::RawUInt16;
        case DataType::UINT32:   return tt::DataFormat::RawUInt32;
        case DataType::INT32:    return tt::DataFormat::Int32;
        default:                 return tt::DataFormat::Float16_b;
    }
}

// ============================================================================
// Test parameters
// ============================================================================

struct StridedDFBTestParams {
    tt::tt_metal::Shape tensor_shape;
    Layout layout;
    DataType dtype;
    BufferType buffer_type;
};

// ============================================================================
// Core test helper
// ============================================================================

// Runs a strided_pages copy test via the Metal 2.0 Host API + DFB.
//
// Pipeline: input_tensor → reader (producer) → DFB → writer (consumer) → output_tensor
//
// The same kernel source is used for both architectures. On WH/BH (1 DM thread per RISC),
// strided_pages() degenerates to pages(0, total_pages) — every page, single thread.
// On Quasar (num_dm_threads DM threads), each thread handles every N-th page.
//
// kernel_builder_fn: a callable (KernelSpec& reader, KernelSpec& writer) → void
//    that sets the arch-specific fields (num_threads, hw_config, DFB bindings).
template <typename T, typename KernelBuilderFn>
void run_strided_dfb_copy_test(
    const StridedDFBTestParams& params,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    uint32_t num_dfb_entries,
    KernelBuilderFn kernel_builder_fn) {
    auto* device = mesh_device->get_devices().at(0);

    MemoryConfig mem_config(TensorMemoryLayout::INTERLEAVED, params.buffer_type);
    TensorSpec tensor_spec(params.tensor_shape, TensorLayout(params.dtype, PageConfig(params.layout), mem_config));

    const auto src = tt::test_utils::generate_uniform_random_vector<T>(0, UINT8_MAX, params.tensor_shape.volume());
    auto input_tensor = Tensor::from_vector(src, tensor_spec, mesh_device);
    auto output_tensor =
        Tensor::from_vector(std::vector<T>(params.tensor_shape.volume()), tensor_spec, mesh_device);

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    ASSERT_NE(input_buffer, nullptr);
    ASSERT_NE(output_buffer, nullptr);

    const uint32_t aligned_page_size = static_cast<uint32_t>(input_buffer->aligned_page_size());
    const uint32_t total_pages = static_cast<uint32_t>(input_buffer->num_pages());

    // -----------------------------------------------------------------------
    // Build ProgramSpec
    // -----------------------------------------------------------------------
    const NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "strided_pages_dfb_copy";

    const TensorAccessorArgs input_accessor_args(*input_buffer);
    const TensorAccessorArgs output_accessor_args(*output_buffer);
    const std::string input_cta_str = build_cta_define(input_accessor_args);
    const std::string output_cta_str = build_cta_define(output_accessor_args);

    // Placeholder kernel specs — filled in by the arch-specific builder lambda
    KernelSpec reader = MakeMinimalGen2DMKernel("reader");  // overwritten by builder
    KernelSpec writer = MakeMinimalGen2DMKernel("writer");  // overwritten by builder

    // Let the arch-specific caller configure kernel specs and DFB bindings
    kernel_builder_fn(reader, writer);

    // Inject CTA define so TensorAccessorArgs<0, 0>() resolves at compile time
    reader.source = kReaderKernelPath;
    writer.source = kWriterKernelPath;
    reader.compiler_options.defines.emplace("KERNEL_COMPILE_TIME_ARGS", input_cta_str);
    writer.compiler_options.defines.emplace("KERNEL_COMPILE_TIME_ARGS", output_cta_str);

    // Runtime varargs: [0]=base_addr, [1]=total_pages
    reader.advanced_options.num_runtime_varargs = 2;
    writer.advanced_options.num_runtime_varargs = 2;

    // DFB: one entry per page, pipelined depth = num_dfb_entries.
    // data_format_metadata must be set — set_dfb_tile_dims calls get_tile_size() unconditionally
    // and throws on DataFormat::Invalid. Use Raw variants so all dtypes are valid on Quasar too.
    auto dfb = MakeMinimalDFB("staging_dfb", aligned_page_size, num_dfb_entries);
    dfb.data_format_metadata = dtype_to_dm_data_format(params.dtype);

    spec.kernels = {reader, writer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = {MakeMinimalWorkUnit("work_unit_0", node, {"reader", "writer"})};

    // -----------------------------------------------------------------------
    // Create Program and set runtime args
    // -----------------------------------------------------------------------
    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"reader"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {input_buffer->address(), total_pages}}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{"writer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {output_buffer->address(), total_pages}}},
                },
        },
    };
    SetProgramRunArgs(program, run_params);

    // -----------------------------------------------------------------------
    // Dispatch and verify
    // -----------------------------------------------------------------------
    detail::LaunchProgram(device, program);

    const auto output_cpu = output_tensor.cpu(true);
    const auto output_shard = ttnn::distributed::get_device_tensors(output_cpu).front();
    EXPECT_EQ(output_shard.template to_vector<T>(), src);
}

// ============================================================================
// Test fixture
// ============================================================================

class TensorAccessorStridedDFBTest : public tt::tt_metal::MeshDeviceFixture,
                                     public ::testing::WithParamInterface<StridedDFBTestParams> {};

// ============================================================================
// Gen1 (WH/BH) tests: single DM thread, BRISC producer + NCRISC consumer
// ============================================================================

TEST_P(TensorAccessorStridedDFBTest, Gen1StridedPagesCopy) {
    if (this->IsSkipped()) {
        return;
    }
    const tt::ARCH arch = devices_.at(0)->arch();
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Gen1 strided_pages DFB test requires Wormhole B0 or Blackhole hardware";
    }

    const auto& params = GetParam();

    auto kernel_builder = [&](KernelSpec& reader, KernelSpec& writer) {
        reader = MakeMinimalGen1DMKernel("reader", DataMovementProcessor::RISCV_0);
        writer = MakeMinimalGen1DMKernel("writer", DataMovementProcessor::RISCV_1);
        // STRIDED with 1 thread: stride=1, each thread accesses all DFB entries
        reader.dfb_bindings.push_back(ProducerOf(experimental::DFBSpecName{"staging_dfb"}, "my_dfb"));
        writer.dfb_bindings.push_back(ConsumerOf(experimental::DFBSpecName{"staging_dfb"}, "my_dfb"));
    };

    switch (params.dtype) {
        case DataType::UINT8:
            run_strided_dfb_copy_test<uint8_t>(params, devices_.at(0).get(), /*num_dfb_entries=*/2, kernel_builder);
            break;
        case DataType::UINT16:
            run_strided_dfb_copy_test<uint16_t>(params, devices_.at(0).get(), /*num_dfb_entries=*/2, kernel_builder);
            break;
        case DataType::BFLOAT16:
            run_strided_dfb_copy_test<bfloat16>(params, devices_.at(0).get(), /*num_dfb_entries=*/2, kernel_builder);
            break;
        default: TT_THROW("Unsupported data type");
    }
}

INSTANTIATE_TEST_SUITE_P(
    Gen1StridedPagesCopy,
    TensorAccessorStridedDFBTest,
    testing::ValuesIn({
        StridedDFBTestParams{
            .tensor_shape = tt::tt_metal::Shape{64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        StridedDFBTestParams{
            .tensor_shape = tt::tt_metal::Shape{96, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::L1,
        },
        StridedDFBTestParams{
            .tensor_shape = tt::tt_metal::Shape{128, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::DRAM,
        },
        StridedDFBTestParams{
            .tensor_shape = tt::tt_metal::Shape{4, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
    }));

// ============================================================================
// Quasar tests: 3 DM threads each, Gen2 producer + Gen2 consumer, STRIDED DFB
// ============================================================================
//
// Each of the 3 DM threads calls accessor.strided_pages(total_pages) which uses
// get_my_thread_id() / get_num_threads() directly to select pages i, i+3, i+6, ...
// The STRIDED DFB routes each thread to DFB entries N, N+3, N+6, ... so producer
// and consumer thread partitions align without any explicit coordination.

TEST_P(TensorAccessorStridedDFBTest, QuasarStridedPagesCopy) {
    if (this->IsSkipped()) {
        return;
    }
    const tt::ARCH arch = devices_.at(0)->arch();
    if (arch != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar strided_pages DFB test requires Quasar hardware";
    }

    constexpr uint8_t kNumDMThreads = 3;

    auto kernel_builder = [&](KernelSpec& reader, KernelSpec& writer) {
        reader = MakeMinimalGen2DMKernel("reader", kNumDMThreads);
        writer = MakeMinimalGen2DMKernel("writer", kNumDMThreads);
        // STRIDED with 3 threads: each thread owns every 3rd DFB entry
        reader.dfb_bindings.push_back(ProducerOf(experimental::DFBSpecName{"staging_dfb"}, "my_dfb"));
        writer.dfb_bindings.push_back(ConsumerOf(experimental::DFBSpecName{"staging_dfb"}, "my_dfb"));
    };

    // DFB depth: 2 entries per DM thread (double-buffer) × 3 threads = 6 entries
    constexpr uint32_t kNumDFBEntries = kNumDMThreads * 2;
    const auto& params = GetParam();

    switch (params.dtype) {
        case DataType::UINT8:
            run_strided_dfb_copy_test<uint8_t>(
                params, devices_.at(0).get(), kNumDFBEntries, kernel_builder);
            break;
        case DataType::UINT16:
            run_strided_dfb_copy_test<uint16_t>(
                params, devices_.at(0).get(), kNumDFBEntries, kernel_builder);
            break;
        case DataType::BFLOAT16:
            run_strided_dfb_copy_test<bfloat16>(
                params, devices_.at(0).get(), kNumDFBEntries, kernel_builder);
            break;
        default: TT_THROW("Unsupported data type");
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuasarStridedPagesCopy,
    TensorAccessorStridedDFBTest,
    testing::ValuesIn({
        StridedDFBTestParams{
            .tensor_shape = tt::tt_metal::Shape{64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        StridedDFBTestParams{
            .tensor_shape = tt::tt_metal::Shape{128, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::DRAM,
        },
        StridedDFBTestParams{
            .tensor_shape = tt::tt_metal::Shape{4, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        // {6,64,64} → 6×2×2 = 24 tiles, evenly divisible by 3 (num_dm_threads)
        StridedDFBTestParams{
            .tensor_shape = tt::tt_metal::Shape{6, 64, 64},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::DRAM,
        },
    }));

}  // namespace
