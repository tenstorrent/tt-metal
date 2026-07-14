// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <cstdint>
#include <sys/types.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <bit>
#include <cctype>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/float8.hpp>
#include <tt-metalium/int8.hpp>
#include <tt-metalium/uint8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
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
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
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

namespace unit_tests::compute::tilize {

enum UntilizeType : std::uint8_t { PACK = 1, DST = 2 };

enum TilizeType : std::uint8_t {
    UNPACK_A = 0,
    UNPACK_A_B = 1,
};

// TilizeA_B takes 2 input source vectors instead of one
using GoldenFunc = std::variant<
    std::function<std::vector<std::uint32_t>(
        const std::vector<std::uint32_t>&, const ::unit_tests::compute::GoldenConfig& config)>,
    std::function<std::vector<std::uint32_t>(
        const std::vector<std::uint32_t>&,
        const std::vector<std::uint32_t>&,
        const ::unit_tests::compute::GoldenConfig& config)>>;

struct TestConfig {
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
    // Whether or not we want the result to be stored in DST in FP32 is
    // controlled with this flag:
    bool fp32_dest_acc_en = false;
    bool fast_tilize = false;
    std::uint32_t input_single_tile_size;
    std::uint32_t output_single_tile_size;
    // Block height in tiles:
    std::uint32_t num_tiles_r;
    // Block width in tiles:
    std::uint32_t num_tiles_c;
    std::uint32_t num_faces_per_tile = 4;
    // Face height in datums:
    std::uint32_t face_r_dim = 16;
    std::optional<UntilizeType> untilize_type = std::nullopt;
    std::optional<TilizeType> tilize_type = std::nullopt;
    tt::DataFormat input_fmt = tt::DataFormat::Float16_b;
    tt::DataFormat output_fmt = tt::DataFormat::Float16_b;
    // Pre-generated source data; if empty, create_arange_vector_of_bfloat16 is used.
    std::vector<std::uint32_t> src0_data;
    GoldenFunc golden_function;
};

// Compute golden + validate; shared between the Metal 2.0 helper and the Gen1-only
// UNPACK_A_B helper (whose compute kernel `unpack_tilizeA_B.cpp` has no DFB rewrite
// in scope for this migration; see `run_single_core_unpack_tilizeA_B_program`).
static void validate_result(
    const TestConfig& test_config,
    const std::vector<std::uint32_t>& src0_vec,
    const std::vector<std::uint32_t>& src1_vec,
    const std::vector<std::uint32_t>& result_vec) {
    vector<std::uint32_t> golden;
    ::unit_tests::compute::GoldenConfig config = {
        .num_tiles_r_dim = test_config.num_tiles_r,
        .num_tiles_c_dim = test_config.num_tiles_c,
        .face_r_dim = test_config.face_r_dim,
        .face_c_dim = 16,
        .num_faces = test_config.num_faces_per_tile,
        .datum_bytes = tt::datum_size(test_config.input_fmt),
    };

    std::visit(
        [&](auto&& func) {
            using FuncType = std::decay_t<decltype(func)>;
            if constexpr (std::is_same_v<
                              FuncType,
                              std::function<std::vector<std::uint32_t>(
                                  const std::vector<std::uint32_t>&,
                                  const ::unit_tests::compute::GoldenConfig& config)>>) {
                golden = func(src0_vec, config);
            } else if constexpr (std::is_same_v<
                                     FuncType,
                                     std::function<std::vector<std::uint32_t>(
                                         const std::vector<std::uint32_t>&,
                                         const std::vector<std::uint32_t>&,
                                         const ::unit_tests::compute::GoldenConfig& config)>>) {
                golden = func(src0_vec, src1_vec, config);
            } else {
                log_fatal(tt::LogTest, "Invalid golden function type");
            }
        },
        test_config.golden_function);

    if (test_config.output_fmt == tt::DataFormat::Float32) {
        vector<bfloat16> golden_unpacked = unpack_vector<bfloat16, std::uint32_t>(golden);
        // Increasing the size since from BFP16 two times, since storing is in FP32
        golden.resize(golden.size() * 2);
        for (auto i = 0; i < golden_unpacked.size(); i++) {
            golden[i] = std::bit_cast<std::uint32_t>(static_cast<float>(golden_unpacked[i]));
        }
    }

    bool pass = true;
    if (test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A_B) {
        pass &= (golden.size() == result_vec.size());
        if (test_config.output_fmt == tt::DataFormat::Float32) {
            std::vector<float> golden_f(golden.size());
            std::vector<float> result_f(result_vec.size());
            std::transform(golden.begin(), golden.end(), golden_f.begin(), [](std::uint32_t w) {
                return std::bit_cast<float>(w);
            });
            std::transform(result_vec.begin(), result_vec.end(), result_f.begin(), [](std::uint32_t w) {
                return std::bit_cast<float>(w);
            });
            pass &=
                is_close_vectors<float>(result_f, golden_f, [&](float a, float b) { return is_close(a, b, 0.01f); });
        } else {
            pass &= is_close_packed_vectors<bfloat16, std::uint32_t>(
                result_vec, golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.01f); });
        }
    } else {
        pass &= (golden.size() == result_vec.size());
        pass &= (golden == result_vec);
    }

    if (not pass) {
        std::cout << "GOLDEN " << std::endl;
        print_vector(unpack_vector<bfloat16, std::uint32_t>(golden));
        std::cout << "RESULTS " << std::endl;
        print_vector(unpack_vector<bfloat16, std::uint32_t>(result_vec));
    }
    log_info(
        tt::LogTest,
        "Done running test with: num_tiles_r = {}, num_tiles_c = {}, FP32_DestAcc = {}, DstSyncFull = {}, "
        "FastTilize = {}, pass = {}",
        test_config.num_tiles_r,
        test_config.num_tiles_c,
        test_config.fp32_dest_acc_en,
        test_config.dst_full_sync_en,
        test_config.fast_tilize,
        pass);
    ASSERT_TRUE(pass);
}

// Metal 2.0 single-core helper covering all migrated tilize/untilize kernels:
//   - UntilizeType::PACK / DST (compute kernels: pack_untilize / dst_untilize)
//   - TilizeType::UNPACK_A (compute kernel: tilize.cpp, optionally with FAST_TILIZE)
void run_single_core_tilize_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t num_tiles = test_config.num_tiles_r * test_config.num_tiles_c;
    const std::uint32_t input_dram_buffer_size = test_config.input_single_tile_size * num_tiles;
    const std::uint32_t output_dram_buffer_size = test_config.output_single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig input_dram_config{
        .device = dev,
        .size = input_dram_buffer_size,
        .page_size = input_dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    tt_metal::InterleavedBufferConfig output_dram_config{
        .device = dev,
        .size = output_dram_buffer_size,
        .page_size = output_dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(input_dram_config);
    auto dst_dram_buffer = CreateBuffer(output_dram_config);
    const std::uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    const std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    const tt::DataFormat output_buf_format = test_config.output_fmt;

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = test_config.input_single_tile_size,
        .num_entries = num_tiles,
        .data_format_metadata = test_config.input_fmt,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = test_config.output_single_tile_size,
        .num_entries = num_tiles,
        .data_format_metadata = output_buf_format,
    };
    if (test_config.untilize_type.has_value() && test_config.untilize_type == UntilizeType::DST) {
        // DST untilize reads face geometry from the output CB metadata (no explicit kernel args).
        output_dfb_spec.unpack_face_geometry_metadata =
            tt::tt_metal::FaceGeometry{test_config.face_r_dim, test_config.num_faces_per_tile};
    }

    // Reader kernel: untilize types stream native tiles from DRAM (`reader_unary_2_0`);
    // UNPACK_A tilize uses the push-N variant (`reader_unary_push_n_2_0`) so the reader
    // hands the compute kernel `num_tiles_c` tiles per ublock, mirroring the legacy
    // `reader_unary_push_n.cpp` contract.
    const bool is_unpack_a_tilize =
        test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A;
    const std::string reader_kernel_path =
        is_unpack_a_tilize ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n_2_0.cpp"
                           : "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_2_0.cpp";

    experimental::KernelSpec::RuntimeArgSchema reader_schema;
    if (is_unpack_a_tilize) {
        reader_schema.runtime_arg_names = {
            "src_addr", "src_dram_bank_id", "num_tiles", "ublock_size_tiles", "reader_only"};
    } else {
        reader_schema.runtime_arg_names = {"src_addr", "bank_id", "num_tiles"};
    }

    experimental::DataMovementHardwareConfig reader_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        reader_hw_config = experimental::DataMovementGen2Config{};
    } else {
        reader_hw_config = experimental::DataMovementGen1Config{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default};
    }
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = reader_kernel_path,
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .runtime_arg_schema = reader_schema,
        .hw_config = reader_hw_config,
    };

    experimental::DataMovementHardwareConfig writer_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        writer_hw_config = experimental::DataMovementGen2Config{};
    } else {
        writer_hw_config = experimental::DataMovementGen1Config{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default};
    }
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    std::string compute_kernel;
    experimental::KernelSpec::CompileTimeArgs compute_cta_bindings = {
        {"per_core_block_cnt", test_config.num_tiles_r},
        {"per_core_block_tile_cnt", test_config.num_tiles_c},
    };
    if (test_config.untilize_type.has_value()) {
        // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
        std::string untilize_type = enchantum::to_string(test_config.untilize_type.value()).data();
        std::transform(untilize_type.begin(), untilize_type.end(), untilize_type.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
        compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/" + untilize_type + "_untilize.cpp";
    } else if (is_unpack_a_tilize) {
        compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/tilize.cpp";
    } else {
        log_fatal(tt::LogTest, "run_single_core_tilize_program: unsupported config (UNPACK_A_B uses dedicated helper)");
    }

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
    if (test_config.fp32_dest_acc_en) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    if (test_config.fast_tilize) {
        compute_defines.emplace("FAST_TILIZE", "1");
    }

    experimental::ComputeHardwareConfig compute_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        compute_hw_config = experimental::ComputeGen2Config{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
        };
    } else {
        compute_hw_config = experimental::ComputeGen1Config{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
        };
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = compute_kernel,
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
        .compile_time_args = compute_cta_bindings,
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "tilize_untilize",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    std::vector<std::uint32_t> src0_vec = test_config.src0_data.empty()
                                              ? create_arange_vector_of_bfloat16(input_dram_buffer_size, false)
                                              : test_config.src0_data;
    tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);

    experimental::ProgramRunArgs params;
    if (is_unpack_a_tilize) {
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node,
                  {{"src_addr", dram_buffer_src0_addr},
                   {"src_dram_bank_id", 0u},
                   {"num_tiles", num_tiles},
                   {"ublock_size_tiles", test_config.num_tiles_c},
                   {"reader_only", 0u}}}},
        });
    } else {
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node, {{"src_addr", dram_buffer_src0_addr}, {"bank_id", 0u}, {"num_tiles", num_tiles}}}},
        });
    }
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = WRITER,
        .runtime_arg_values = {{node, {{"dst_addr", dram_buffer_dst_addr}, {"bank_id", 0u}, {"num_tiles", num_tiles}}}},
    });
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE});
    experimental::SetProgramRunArgs(program_, params);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    validate_result(test_config, src0_vec, /*src1_vec=*/{}, result_vec);
}

// Gen1-only single-core helper for the `unpack_tilizeA_B` + eltwise binary add compute kernel.
// On Quasar, llk_unpack_tilizeA_B is only compatible with the math reduce kernel, not eltwise
// binary add, so the caller (`TensixComputeUnpackTilizeA_B`) skips on Quasar.
void run_single_core_unpack_tilizeA_B_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = tt::tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    CoreCoord core = {0, 0};

    std::uint32_t num_tiles = test_config.num_tiles_r * test_config.num_tiles_c;
    std::uint32_t input_dram_buffer_size = test_config.input_single_tile_size * num_tiles;
    std::uint32_t output_dram_buffer_size = test_config.output_single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = input_dram_buffer_size,
        .page_size = input_dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    tt_metal::InterleavedBufferConfig output_dram_config{
        .device = device,
        .size = output_dram_buffer_size,
        .page_size = output_dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(input_dram_config);
    std::uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto src1_dram_buffer = CreateBuffer(input_dram_config);
    std::uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(output_dram_config);
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    std::uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            num_tiles * test_config.input_single_tile_size, {{src0_cb_index, test_config.input_fmt}})
            .set_page_size(src0_cb_index, test_config.input_single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src0_config);

    std::uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(
            num_tiles * test_config.input_single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, test_config.input_single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_src1_config);

    std::uint32_t output_cb_index = tt::CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_tiles * test_config.output_single_tile_size, {{output_cb_index, test_config.output_fmt}})
            .set_page_size(output_cb_index, test_config.output_single_tile_size);
    tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<std::uint32_t> compute_kernel_args = {
        std::uint32_t(test_config.num_tiles_r),
        std::uint32_t(test_config.num_tiles_c),
    };

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unpack_tilizeA_B.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
        });

    std::vector<std::uint32_t> src0_vec = test_config.src0_data.empty()
                                              ? create_arange_vector_of_bfloat16(input_dram_buffer_size, false)
                                              : test_config.src0_data;
    tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);

    std::vector<std::uint32_t> src1_vec = create_constant_vector_of_bfloat16(input_dram_buffer_size, 1.0f);
    tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_vec);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            dram_buffer_src0_addr,
            (std::uint32_t)0,
            dram_buffer_src1_addr,
            (std::uint32_t)0,
            (std::uint32_t)num_tiles,
        });
    tt_metal::SetRuntimeArgs(program_, writer_kernel, core, {dram_buffer_dst_addr, (std::uint32_t)0, num_tiles});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    validate_result(test_config, src0_vec, src1_vec, result_vec);
}

// Metal 2.0 single-core helper for the Quasar `unpack_tilizeA_B` + reduce path.
// Quasar's unpack_tilizeA_B is only compatible with the reduce math kernel (not eltwise binary).
// Uses REDUCE_COL + MAX: tilizes row-major src0 data, reduces each tile independently
// (column-wise max within each tile), producing output tiles with only row 0 populated.
void run_single_core_unpack_tilizeA_B_reduce_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t num_tiles_in = test_config.num_tiles_r * test_config.num_tiles_c;
    const std::uint32_t num_tiles_out = num_tiles_in;  // each tile reduced independently, same count as input
    const std::uint32_t input_dram_buffer_size = test_config.input_single_tile_size * num_tiles_in;

    auto make_flat_tensor_spec = [](std::uint32_t entry_size, std::uint32_t total_entries) {
        const std::uint32_t entry_size_words = entry_size / sizeof(std::uint32_t);
        auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
        auto memory_config =
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
        auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::UINT32, page_config, memory_config);
        return tt::tt_metal::TensorSpec(tt::tt_metal::Shape{total_entries, entry_size_words}, tensor_layout);
    };

    auto in_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_tensor_spec(test_config.input_single_tile_size, num_tiles_in), TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(
        *mesh_device, make_flat_tensor_spec(test_config.output_single_tile_size, num_tiles_out), TensorTopology{});

    const experimental::DFBSpecName INP_DATA_DFB{"inp_data_dfb"};
    const experimental::DFBSpecName INP_SCALER_DFB{"inp_scaler_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};
    const experimental::TensorParamName IN_TENSOR{"in_tensor"};
    const experimental::TensorParamName OUT_TENSOR{"out_tensor"};

    experimental::DataflowBufferSpec inp_data_dfb_spec{
        .unique_id = INP_DATA_DFB,
        .entry_size = test_config.input_single_tile_size,
        .num_entries = std::max(2u, test_config.num_tiles_c),
        .data_format_metadata = test_config.input_fmt,
    };
    const std::uint32_t scaler_tile_size = tt::datum_size(test_config.input_fmt) * 32 * 32;
    experimental::DataflowBufferSpec inp_scaler_dfb_spec{
        .unique_id = INP_SCALER_DFB,
        .entry_size = scaler_tile_size,
        .num_entries = 2,
        .data_format_metadata = test_config.input_fmt,
    };
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = test_config.output_single_tile_size,
        .num_entries = std::max(2u, test_config.num_tiles_c),
        .data_format_metadata = test_config.output_fmt,
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
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = {{"GENERATE_BCAST_SCALER", "1"}, {"BLOCK_SIZE", "1"}}},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP_DATA_DFB,
                 .accessor_name = "out_data",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP_SCALER_DFB,
                 .accessor_name = "out_scaler",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "scaler"}},
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
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = writer_hw_config,
    };

    experimental::KernelSpec::CompilerOptions::Defines compute_defines = {
        {"REDUCE_OP", "PoolType::MAX"},
        {"REDUCE_DIM", "ReduceDim::REDUCE_COL"},
    };
    if (test_config.fp32_dest_acc_en) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }

    experimental::ComputeHardwareConfig compute_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        compute_hw_config = experimental::ComputeGen2Config{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
        };
    } else {
        compute_hw_config = experimental::ComputeGen1Config{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
        };
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/unpack_tilizeA_B_reduce.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP_DATA_DFB,
                 .accessor_name = "in_data",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP_SCALER_DFB,
                 .accessor_name = "in_scaler",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", test_config.num_tiles_r}, {"per_core_block_tile_cnt", test_config.num_tiles_c}},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "unpack_tilizeA_B_reduce",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {inp_data_dfb_spec, inp_scaler_dfb_spec, out_dfb_spec},
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
    auto& program_ = workload.get_programs().at(device_range);

    std::vector<std::uint32_t> src0_vec = create_random_vector_of_bfloat16(input_dram_buffer_size, 100, 42);
    tt_metal::detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), src0_vec);

    float scaler_f = 1.0f;
    std::vector<std::uint32_t> scaler_tile_vec = create_constant_vector_of_bfloat16(scaler_tile_size, scaler_f);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node, {{"num_tiles", num_tiles_in}, {"scaler", *reinterpret_cast<std::uint32_t*>(&scaler_f)}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = {{node, {{"num_tiles", num_tiles_out}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN_TENSOR, experimental::ProgramRunArgs::TensorArgument{in_tensor}},
        {OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{out_tensor}},
    };
    experimental::SetProgramRunArgs(program_, params);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), result_vec);

    validate_result(test_config, src0_vec, scaler_tile_vec, result_vec);
}

}  // namespace unit_tests::compute::tilize

/**************************************
Following tests are for Unpack Tilize
***************************************/

TEST_F(LLKMeshDeviceFixture, TensixComputeUnpackTilize) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {true, false}) {
            for (bool dst_full_sync_en : {true, false}) {
                unit_tests::compute::tilize::TestConfig test_config = {
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .input_single_tile_size = 2 * 1024,
                    .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                    .num_tiles_r = num_tile[0],
                    .num_tiles_c = num_tile[1],
                    .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
                    .output_fmt = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b,
                    .golden_function = ::unit_tests::compute::gold_standard_tilize};
                unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputeUnpackTilizeFp8e4m3) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {true, false}) {
            std::uint32_t num_tiles_total = num_tile[0] * num_tile[1];
            auto src_data = create_random_vector_of_float8_e4m3(
                tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles_total, /*rand_max_float=*/20, /*seed=*/42);
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .fp32_dest_acc_en = true,  // BH: Fp8 requires fp32_dest_acc_en=true (JIT-enforced)
                .input_single_tile_size = tt::tile_size(tt::DataFormat::Fp8_e4m3),
                .output_single_tile_size = tt::tile_size(tt::DataFormat::Fp8_e4m3),
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
                .input_fmt = tt::DataFormat::Fp8_e4m3,
                .output_fmt = tt::DataFormat::Fp8_e4m3,
                .src0_data = src_data,
                .golden_function = ::unit_tests::compute::gold_standard_tilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputeUnpackTilizeInt8) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {false, true}) {
            std::uint32_t num_tiles_total = num_tile[0] * num_tile[1];
            auto src_data =
                create_random_vector_of_int8(tt::tile_size(tt::DataFormat::Int8) * num_tiles_total, /*seed=*/42);
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .fp32_dest_acc_en = true,
                .input_single_tile_size = tt::tile_size(tt::DataFormat::Int8),
                .output_single_tile_size = tt::tile_size(tt::DataFormat::Int8),
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
                .input_fmt = tt::DataFormat::Int8,
                .output_fmt = tt::DataFormat::Int8,
                .src0_data = src_data,
                .golden_function = ::unit_tests::compute::gold_standard_tilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputeUnpackTilizeUInt8) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {false, true}) {
            std::uint32_t num_tiles_total = num_tile[0] * num_tile[1];
            auto src_data =
                create_random_vector_of_uint8(tt::tile_size(tt::DataFormat::UInt8) * num_tiles_total, /*seed=*/42);
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .fp32_dest_acc_en = true,
                .input_single_tile_size = tt::tile_size(tt::DataFormat::UInt8),
                .output_single_tile_size = tt::tile_size(tt::DataFormat::UInt8),
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
                .input_fmt = tt::DataFormat::UInt8,
                .output_fmt = tt::DataFormat::UInt8,
                .src0_data = src_data,
                .golden_function = ::unit_tests::compute::gold_standard_tilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeFastTilize) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {false}) {
            for (bool dst_full_sync_en : {false}) {
                unit_tests::compute::tilize::TestConfig test_config = {
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .fast_tilize = true,
                    .input_single_tile_size = 2 * 1024,
                    .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                    .num_tiles_r = num_tile[0],
                    .num_tiles_c = num_tile[1],
                    .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
                    .output_fmt = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b,
                    .golden_function = ::unit_tests::compute::gold_standard_tilize};
                unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeUnpackTilizeA_B) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "tilizeA_B + eltwise binary is not supported on Quasar";
    }
    for (bool dst_full_sync_en : {true, false}) {
        unit_tests::compute::tilize::TestConfig test_config = {
            .dst_full_sync_en = dst_full_sync_en,
            .input_single_tile_size = 2 * 1024,
            .output_single_tile_size = 2 * 1024,
            .num_tiles_r = 2,
            .num_tiles_c = 8,
            .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A_B,
            .output_fmt = tt::DataFormat::Float16_b,
            .golden_function = ::unit_tests::compute::gold_standard_tilize_w_elwadd};
        unit_tests::compute::tilize::run_single_core_unpack_tilizeA_B_program(this->devices_.at(0), test_config);
    }
}

/******************************
Following tests are for Quasar
*******************************/
enum class QuasarTestMode { TILIZE, UNTILIZE, UNTILIZE_DST };

static void run_quasar_tilize_untilize_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    std::uint32_t num_tiles_r,
    std::uint32_t num_tiles_c,
    QuasarTestMode mode,
    bool dst_full_sync_en,
    bool fp32_dest_acc_en,
    tt::DataFormat input_data_format,
    tt::DataFormat output_data_format) {
    bool is_tilize = (mode == QuasarTestMode::TILIZE);

    IDevice* dev = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    bool is_8bit_integer = (input_data_format == tt::DataFormat::Int8 || input_data_format == tt::DataFormat::UInt8);
    std::uint32_t num_tiles = num_tiles_r * num_tiles_c;
    std::uint32_t input_single_tile_size = tt::tile_size(input_data_format);
    std::uint32_t output_single_tile_size = tt::tile_size(output_data_format);
    std::uint32_t src_dram_buffer_size = input_single_tile_size * num_tiles;
    std::uint32_t dst_dram_buffer_size = output_single_tile_size * num_tiles;

    InterleavedBufferConfig src_config{
        .device = dev,
        .size = src_dram_buffer_size,
        .page_size = src_dram_buffer_size,
        .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig dst_config{
        .device = dev,
        .size = dst_dram_buffer_size,
        .page_size = dst_dram_buffer_size,
        .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(src_config);
    auto dst_dram_buffer = CreateBuffer(dst_config);
    std::uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    std::uint32_t dfb_num_entries = std::max(2u, num_tiles_c);

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = dfb_num_entries,
        .data_format_metadata = input_data_format,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = dfb_num_entries,
        .data_format_metadata = output_data_format,
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    std::string compute_kernel;
    experimental::KernelSpec::CompileTimeArgs compute_cta_bindings;
    switch (mode) {
        case QuasarTestMode::TILIZE:
            compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/tilize.cpp";
            compute_cta_bindings = {
                {"per_core_block_cnt", num_tiles_r},
                {"per_core_block_tile_cnt", num_tiles_c},
            };
            break;
        case QuasarTestMode::UNTILIZE:
            compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize.cpp";
            compute_cta_bindings = {
                {"per_core_block_cnt", num_tiles_r},
                {"per_core_block_tile_cnt", num_tiles_c},
            };
            break;
        case QuasarTestMode::UNTILIZE_DST: {
            compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/dst_untilize.cpp";
            compute_cta_bindings = {
                {"per_core_block_cnt", num_tiles_r},
                {"per_core_block_tile_cnt", num_tiles_c},
            };
            break;
        }
    }

    experimental::ComputeHardwareConfig compute_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        compute_hw_config = experimental::ComputeGen2Config{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
        };
    } else {
        compute_hw_config = experimental::ComputeGen1Config{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
        };
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = compute_kernel,
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
        .compile_time_args = compute_cta_bindings,
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "quasar_tilize_untilize",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& program_run = workload.get_programs().at(device_range);

    std::vector<std::uint32_t> src_vec;
    if (input_data_format == tt::DataFormat::Int8) {
        src_vec = create_random_vector_of_int8(src_dram_buffer_size, /*seed=*/42);
    } else if (input_data_format == tt::DataFormat::UInt8) {
        src_vec = create_random_vector_of_uint8(src_dram_buffer_size, /*seed=*/42);
    } else if (input_data_format == tt::DataFormat::Int16) {
        src_vec.resize(src_dram_buffer_size / sizeof(std::uint32_t));
        for (std::uint32_t i = 0; i < src_vec.size(); i++) {
            src_vec[i] = (static_cast<std::uint32_t>((2 * i) + 1) << 16) | static_cast<std::uint32_t>(2 * i);
        }
    } else if (is_tilize && input_data_format == tt::DataFormat::Float32) {
        src_vec.resize(src_dram_buffer_size / sizeof(std::uint32_t));
        for (std::uint32_t i = 0; i < src_vec.size(); i++) {
            src_vec[i] = std::bit_cast<std::uint32_t>(static_cast<float>(i));
        }
    } else {
        src_vec = create_arange_vector_of_bfloat16(src_dram_buffer_size, false);
    }
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    // This test configures the DRAM buffers as a single whole-buffer page, so
    // aligned_page_size() returns the whole-buffer stride rather than per-tile.
    // Compute the real per-tile DRAM stride directly from the buffer size.
    const std::uint32_t src_tile_stride_bytes = src_dram_buffer_size / num_tiles;
    const std::uint32_t dst_tile_stride_bytes = dst_dram_buffer_size / num_tiles;

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node,
                  {{"src_addr", dram_buffer_src_addr},
                   {"src_bank_id", 0u},
                   {"num_tiles", num_tiles},
                   {"dram_page_stride", src_tile_stride_bytes}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", dram_buffer_dst_addr},
                   {"dst_bank_id", 0u},
                   {"num_tiles", num_tiles},
                   {"dram_page_stride", dst_tile_stride_bytes}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<std::uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    ::unit_tests::compute::GoldenConfig golden_config = {
        .num_tiles_r_dim = static_cast<int>(num_tiles_r),
        .num_tiles_c_dim = static_cast<int>(num_tiles_c),
        .datum_bytes = tt::datum_size(input_data_format)};
    auto golden = is_tilize ? ::unit_tests::compute::gold_standard_tilize(src_vec, golden_config)
                            : ::unit_tests::compute::gold_standard_untilize(src_vec, golden_config);

    if (is_8bit_integer) {
        // Int8/UInt8 in dest is promoted to Int32. Expand each byte to a uint32_t word.
        // Hardware uses sign-magnitude representation for Int8:
        //   bit 31 = sign (MSB of the byte), bits [6:0] = magnitude (lower 7 bits of the byte)
        bool is_signed = (input_data_format == tt::DataFormat::Int8);
        std::vector<std::uint32_t> golden_int32;
        golden_int32.reserve(golden.size() * 4);
        for (auto word : golden) {
            for (int b = 0; b < 4; b++) {
                std::uint8_t byte_val = (word >> (b * 8)) & 0xFF;
                if (is_signed) {
                    std::uint32_t sign = (byte_val >> 7) & 1;
                    std::uint32_t magnitude = byte_val & 0x7F;
                    golden_int32.push_back((sign << 31) | magnitude);
                } else {
                    golden_int32.push_back(static_cast<std::uint32_t>(byte_val));
                }
            }
        }
        golden = std::move(golden_int32);
    } else if (output_data_format == tt::DataFormat::Float32 && input_data_format != tt::DataFormat::Float32) {
        // For 32-bit output (fp32_dest_acc_en) with 16-bit float input: expand golden from bfloat16 to float32
        // For Float32 input: golden is already 32-bit, no expansion needed
        vector<bfloat16> golden_unpacked = unpack_vector<bfloat16, std::uint32_t>(golden);
        golden.resize(golden.size() * 2);
        for (auto i = 0; i < golden_unpacked.size(); i++) {
            golden[i] = std::bit_cast<std::uint32_t>(static_cast<float>(golden_unpacked[i]));
        }
    }

    EXPECT_EQ(golden.size(), result_vec.size());
    EXPECT_EQ(golden, result_vec);
}

// Pack Untilize (via pack_untilize_block)
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputePackUntilize) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {4, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (tt::DataFormat input_data_format : {tt::DataFormat::Float16_b, tt::DataFormat::Int16}) {
                    if (fp32_dest_acc_en && input_data_format == tt::DataFormat::Int16) {
                        continue;  // Int16 + 32-bit dest mode is not supported on Quasar
                    }
                    run_quasar_tilize_untilize_test(
                        this->devices_.at(0),
                        cfg[0],
                        cfg[1],
                        QuasarTestMode::UNTILIZE,
                        dst_full_sync_en,
                        fp32_dest_acc_en,
                        input_data_format,
                        fp32_dest_acc_en ? tt::DataFormat::Float32 : input_data_format);
                }
            }
        }
    }
}

// Pack Untilize Dst (tiles pre-loaded into dest via copy_tile)
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputePackUntilizeDst) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {4, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (tt::DataFormat input_data_format : {tt::DataFormat::Float16_b, tt::DataFormat::Int16}) {
                    if (fp32_dest_acc_en && input_data_format == tt::DataFormat::Int16) {
                        continue;  // Int16 + 32-bit dest mode is not supported on Quasar
                    }
                    run_quasar_tilize_untilize_test(
                        this->devices_.at(0),
                        cfg[0],
                        cfg[1],
                        QuasarTestMode::UNTILIZE_DST,
                        dst_full_sync_en,
                        fp32_dest_acc_en,
                        input_data_format,
                        fp32_dest_acc_en ? tt::DataFormat::Float32 : input_data_format);
                }
            }
        }
    }
}

// Quasar Unpack Tilize
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarComputeUnpackTilize) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (tt::DataFormat input_data_format : {tt::DataFormat::Float16_b, tt::DataFormat::Int16}) {
                    if (fp32_dest_acc_en && input_data_format == tt::DataFormat::Int16) {
                        continue;  // Int16 + 32-bit dest mode is not supported on Quasar
                    }

                    run_quasar_tilize_untilize_test(
                        this->devices_.at(0),
                        cfg[0],
                        cfg[1],
                        QuasarTestMode::TILIZE,
                        dst_full_sync_en,
                        fp32_dest_acc_en,
                        input_data_format,
                        fp32_dest_acc_en ? tt::DataFormat::Float32 : input_data_format);
                }
            }
        }
    }
}

// Quasar Unpack TilizeA_B (tilize + reduce col max)
// Quasar's unpack_tilizeA_B is only compatible with the reduce math kernel.
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarComputeUnpackTilizeA_B) {
    for (bool dst_full_sync_en : {true, false}) {
        for (bool fp32_dest_acc_en : {true, false}) {
            for (tt::DataFormat input_data_format : {tt::DataFormat::Float16_b}) {
                std::uint32_t tile_size = tt::tile_size(input_data_format);
                tt::DataFormat output_data_format =
                    fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
                std::uint32_t output_tile_size = tt::tile_size(output_data_format);
                unit_tests::compute::tilize::TestConfig test_config = {
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .input_single_tile_size = tile_size,
                    .output_single_tile_size = output_tile_size,
                    .num_tiles_r = 2,
                    .num_tiles_c = 10,
                    .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A_B,
                    .input_fmt = input_data_format,
                    .output_fmt = output_data_format,
                    .golden_function = ::unit_tests::compute::gold_standard_tilize_w_reduce_col_max};
                unit_tests::compute::tilize::run_single_core_unpack_tilizeA_B_reduce_program(
                    this->devices_.at(0), test_config);
            }
        }
    }
}

// Pack Untilize Int8 -> Int32 dest -> Int32 (via pack_untilize_block)
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputePackUntilizeInt32) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {4, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            run_quasar_tilize_untilize_test(
                this->devices_.at(0),
                cfg[0],
                cfg[1],
                QuasarTestMode::UNTILIZE,
                dst_full_sync_en,
                /*fp32_dest_acc_en=*/true,
                tt::DataFormat::Int8,
                /*output_data_format=*/tt::DataFormat::Int32);
        }
    }
}

// Pack Untilize Dst Int8 -> Int32 dest -> Int32 (tiles pre-loaded into dest via copy_tile)
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputePackUntilizeDstInt32) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {4, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            run_quasar_tilize_untilize_test(
                this->devices_.at(0),
                cfg[0],
                cfg[1],
                QuasarTestMode::UNTILIZE_DST,
                dst_full_sync_en,
                /*fp32_dest_acc_en=*/true,
                tt::DataFormat::Int8,
                /*output_data_format=*/tt::DataFormat::Int32);
        }
    }
}

/**************************************
Following tests are for pack untilize
***************************************/

TEST_F(LLKMeshDeviceFixture, TensixComputePackUntilize) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}, {10, 10}, {2, 40}};
    for (auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {true, false}) {
            for (bool dst_full_sync_en : {true, false}) {
                unit_tests::compute::tilize::TestConfig test_config = {
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .input_single_tile_size = 2 * 1024,
                    .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                    .num_tiles_r = num_tile[0],
                    .num_tiles_c = num_tile[1],
                    .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
                    .output_fmt = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b,
                    .golden_function = ::unit_tests::compute::gold_standard_untilize};
                unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputePackUntilizeDst) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}, {10, 10}, {2, 40}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {true, false}) {
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .input_single_tile_size = 2 * 1024,
                .output_single_tile_size = 2 * 1024,
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
                .output_fmt = tt::DataFormat::Float16_b,
                .golden_function = ::unit_tests::compute::gold_standard_untilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputePackUntilizeFp8e4m3) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {true, false}) {
            std::uint32_t num_t = num_tile[0] * num_tile[1];
            auto src_data = create_random_vector_of_float8_e4m3(
                tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_t, /*rand_max_float=*/20, /*seed=*/42);
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .fp32_dest_acc_en = true,  // BH: Fp8 requires fp32_dest_acc_en=true (JIT-enforced)
                .input_single_tile_size = tt::tile_size(tt::DataFormat::Fp8_e4m3),
                .output_single_tile_size = tt::tile_size(tt::DataFormat::Fp8_e4m3),
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
                .input_fmt = tt::DataFormat::Fp8_e4m3,
                .output_fmt = tt::DataFormat::Fp8_e4m3,
                .src0_data = src_data,
                .golden_function = ::unit_tests::compute::gold_standard_untilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputePackUntilizeInt8) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {true, false}) {
            std::uint32_t num_t = num_tile[0] * num_tile[1];
            auto src_data = create_random_vector_of_int8(tt::tile_size(tt::DataFormat::Int8) * num_t, /*seed=*/42);
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .fp32_dest_acc_en = true,
                .input_single_tile_size = tt::tile_size(tt::DataFormat::Int8),
                .output_single_tile_size = tt::tile_size(tt::DataFormat::Int8),
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
                .input_fmt = tt::DataFormat::Int8,
                .output_fmt = tt::DataFormat::Int8,
                .src0_data = src_data,
                .golden_function = ::unit_tests::compute::gold_standard_untilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputePackUntilizeUInt8) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {true, false}) {
            std::uint32_t num_t = num_tile[0] * num_tile[1];
            auto src_data = create_random_vector_of_uint8(tt::tile_size(tt::DataFormat::UInt8) * num_t, /*seed=*/42);
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .fp32_dest_acc_en = true,
                .input_single_tile_size = tt::tile_size(tt::DataFormat::UInt8),
                .output_single_tile_size = tt::tile_size(tt::DataFormat::UInt8),
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
                .input_fmt = tt::DataFormat::UInt8,
                .output_fmt = tt::DataFormat::UInt8,
                .src0_data = src_data,
                .golden_function = ::unit_tests::compute::gold_standard_untilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

// Tests pack_untilize with tiny tile dims.
// Row dim 1x32, which is faces = 2, rows = 1
// Row dim 1x16, which is faces = 1, rows = 1
TEST_F(LLKMeshDeviceFixture, TensixComputePackUntilizeDstTinyTile) {
    vector<vector<std::uint32_t>> test_config_values = {{1, 1, 1, 1}, {1, 1, 2, 1}, {1, 2, 2, 1}};
    std::uint32_t face_c_dim = 16;
    for (auto test_config_value : test_config_values) {
        for (bool dst_full_sync_en : {true, false}) {
            std::uint32_t num_faces_per_tile = test_config_value[2];
            std::uint32_t face_r_dim = test_config_value[3];
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .input_single_tile_size = 2 * 1024,
                .output_single_tile_size = 2 * num_faces_per_tile * face_r_dim * face_c_dim,
                .num_tiles_r = test_config_value[0],
                .num_tiles_c = test_config_value[1],
                .num_faces_per_tile = num_faces_per_tile,
                .face_r_dim = face_r_dim,
                .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
                .output_fmt = tt::DataFormat::Float16_b,
                .golden_function = ::unit_tests::compute::gold_standard_untilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

}  // namespace tt::tt_metal
