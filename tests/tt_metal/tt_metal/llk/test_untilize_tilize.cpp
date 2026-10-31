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
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/debug/watcher_server.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/float8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tile.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/int8.hpp"
#include "tt_metal/test_utils/uint8.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include <tt-umd/types/arch.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "single_core_compute_runners.hpp"

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
    // UntilizeType::PACK only: use fast_untilize_* (api/compute/experimental/fast_untilize.h) instead of
    // plain pack_untilize_*.
    bool fast_untilize = false;
    // Exercise explicit geometry while the output metadata describes a full tile.
    bool explicit_untilize_geometry = false;
    bool explicit_untilize_narrow_row = false;
    // Narrow-row width in datums (0 = one face, the kernel default). Only read when explicit_untilize_narrow_row.
    std::uint32_t explicit_untilize_row_num_datums = 0;
    // UNPACK_A tilize only: tilize the whole block with a nonzero input_tile_index per tile-row
    // (via tilize_across_tile_rows.cpp) so the cross-tile-row stride in llk_unpack_tilize_block is
    // exercised, instead of the default tilize.cpp which uses input_tile_index=0 + pop_front.
    bool tilize_cross_tile_rows = false;
    std::uint32_t input_single_tile_size;
    std::uint32_t output_single_tile_size;
    // Block height in tiles:
    std::uint32_t num_tiles_r;
    // Block width in tiles:
    std::uint32_t num_tiles_c;
    // Face grid. Default 2x2 is a full 32x32 tile.
    std::uint32_t tile_shape_in_faces_r = 2;
    std::uint32_t tile_shape_in_faces_c = 2;
    // Face height in datums:
    std::uint32_t face_r_dim = 16;
    // (Face width is always 16 in this test)
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
        .num_faces = static_cast<int>(test_config.tile_shape_in_faces_r * test_config.tile_shape_in_faces_c),
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
        "FastTilize = {}, FastUntilize = {}, pass = {}",
        test_config.num_tiles_r,
        test_config.num_tiles_c,
        test_config.fp32_dest_acc_en,
        test_config.dst_full_sync_en,
        test_config.fast_tilize,
        test_config.fast_untilize,
        pass);
    ASSERT_TRUE(pass);
}

// Metal 2.0 single-core helper covering all migrated tilize/untilize kernels:
//   - UntilizeType::PACK / DST (compute kernels: pack_untilize (optionally with FAST_UNTILIZE) / dst_untilize)
//   - TilizeType::UNPACK_A (compute kernel: tilize.cpp, optionally with FAST_TILIZE)
void run_single_core_tilize_program(distributed::MeshDevice& mesh_device, const TestConfig& test_config) {
    auto& cq = mesh_device.mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t num_tiles = test_config.num_tiles_r * test_config.num_tiles_c;
    const std::uint32_t input_dram_buffer_size = test_config.input_single_tile_size * num_tiles;
    const std::uint32_t output_dram_buffer_size = test_config.output_single_tile_size * num_tiles;

    auto src0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = input_dram_buffer_size},
        {.page_size = input_dram_buffer_size, .buffer_type = BufferType::DRAM},
        &mesh_device);
    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = output_dram_buffer_size},
        {.page_size = output_dram_buffer_size, .buffer_type = BufferType::DRAM},
        &mesh_device);
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
    if (test_config.untilize_type.has_value() && test_config.untilize_type == UntilizeType::DST &&
        !test_config.explicit_untilize_geometry) {
        // DST untilize reads the face layout from the output CB metadata (no explicit kernel args).
        output_dfb_spec.tile_format_metadata = Tile::from_face_grid(
            test_config.tile_shape_in_faces_r,
            test_config.tile_shape_in_faces_c,
            {test_config.face_r_dim, constants::FACE_WIDTH});
    } else if (
        test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A &&
        (test_config.tile_shape_in_faces_r != 2 || test_config.tile_shape_in_faces_c != 2 ||
         test_config.face_r_dim != 16)) {
        // Tiny/shortened-face tilize (e.g. 16x32, or face_r_dim < 16): the unpack/pack LLKs read the
        // tile's face layout from the CB metadata, so tag both the input and output buffers with it.
        // Gate on either a non-2x2 face grid or a shorter face row dim so shortened four-face tiles are caught too.
        const auto tile = Tile::from_face_grid(
            test_config.tile_shape_in_faces_r,
            test_config.tile_shape_in_faces_c,
            {test_config.face_r_dim, constants::FACE_WIDTH});
        input_dfb_spec.tile_format_metadata = tile;
        output_dfb_spec.tile_format_metadata = tile;
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
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        reader_hw_config = experimental::DataMovementHardwareConfig{};
    } else {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                },
        };
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
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        writer_hw_config = experimental::DataMovementHardwareConfig{};
    } else {
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                },
        };
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
        if (test_config.explicit_untilize_geometry) {
            compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/custom_dst_untilize.cpp";
        }
    } else if (is_unpack_a_tilize) {
        compute_kernel = test_config.tilize_cross_tile_rows
                             ? "tests/tt_metal/tt_metal/test_kernels/compute/tilize_across_tile_rows.cpp"
                             : "tests/tt_metal/tt_metal/test_kernels/compute/tilize.cpp";
    } else {
        log_fatal(tt::LogTest, "run_single_core_tilize_program: unsupported config (UNPACK_A_B uses dedicated helper)");
    }

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
    if (test_config.explicit_untilize_geometry) {
        const auto num_faces = test_config.tile_shape_in_faces_r * test_config.tile_shape_in_faces_c;
        compute_defines.emplace("EXPLICIT_FACE_R_DIM", std::to_string(test_config.face_r_dim));
        compute_defines.emplace("EXPLICIT_NUM_FACES", std::to_string(num_faces));
        compute_defines.emplace("EXPLICIT_NARROW_ROW", test_config.explicit_untilize_narrow_row ? "1" : "0");
        if (test_config.explicit_untilize_narrow_row && test_config.explicit_untilize_row_num_datums != 0) {
            compute_defines.emplace(
                "EXPLICIT_ROW_NUM_DATUMS", std::to_string(test_config.explicit_untilize_row_num_datums));
        }
    }
    if (test_config.fast_tilize) {
        compute_defines.emplace("FAST_TILIZE", "1");
    }
    if (test_config.fast_untilize) {
        compute_defines.emplace("FAST_UNTILIZE", "1");
    }

    experimental::ComputeHardwareConfig compute_hw_config;
    compute_hw_config = experimental::ComputeHardwareConfig{
        .enable_32_bit_dest = test_config.fp32_dest_acc_en,
        .double_buffer_dest = !test_config.dst_full_sync_en,
    };
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

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    std::vector<std::uint32_t> src0_vec = test_config.src0_data.empty()
                                              ? create_arange_vector_of_bfloat16(input_dram_buffer_size, false)
                                              : test_config.src0_data;
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/true);

    experimental::ProgramRunArgs params;
    if (is_unpack_a_tilize) {
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", dram_buffer_src0_addr},
                 {"src_dram_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"ublock_size_tiles", test_config.num_tiles_c},
                 {"reader_only", 0u}}),
        });
    } else {
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"src_addr", dram_buffer_src0_addr}, {"bank_id", 0u}, {"num_tiles", num_tiles}}),
        });
    }
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = WRITER,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node, {{"dst_addr", dram_buffer_dst_addr}, {"bank_id", 0u}, {"num_tiles", num_tiles}}),
    });
    params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE});
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    validate_result(test_config, src0_vec, /*src1_vec=*/{}, result_vec);
}

// Gen1-only single-core helper for the `unpack_tilizeA_B` + eltwise binary add compute kernel.
// On Quasar, llk_unpack_tilizeA_B is only compatible with the math reduce kernel, not eltwise
// binary add, so the caller (`TensixComputeUnpackTilizeA_B`) skips on Quasar.
void run_single_core_unpack_tilizeA_B_program(distributed::MeshDevice& mesh_device, const TestConfig& test_config) {
    auto& cq = mesh_device.mesh_command_queue();
    Program program = tt::tt_metal::CreateProgram(mesh_device);

    CoreCoord core = {0, 0};

    std::uint32_t num_tiles = test_config.num_tiles_r * test_config.num_tiles_c;
    std::uint32_t input_dram_buffer_size = test_config.input_single_tile_size * num_tiles;
    std::uint32_t output_dram_buffer_size = test_config.output_single_tile_size * num_tiles;

    distributed::ReplicatedBufferConfig input_buffer_config{.size = input_dram_buffer_size};
    distributed::DeviceLocalBufferConfig input_dram_config{
        .page_size = input_dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src0_dram_buffer = distributed::MeshBuffer::create(input_buffer_config, input_dram_config, &mesh_device);
    std::uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto src1_dram_buffer = distributed::MeshBuffer::create(input_buffer_config, input_dram_config, &mesh_device);
    std::uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = output_dram_buffer_size},
        {.page_size = output_dram_buffer_size, .buffer_type = BufferType::DRAM},
        &mesh_device);
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    std::uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            num_tiles * test_config.input_single_tile_size, {{src0_cb_index, test_config.input_fmt}})
            .set_page_size(src0_cb_index, test_config.input_single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    std::uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(
            num_tiles * test_config.input_single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, test_config.input_single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    std::uint32_t output_cb_index = tt::CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_tiles * test_config.output_single_tile_size, {{output_cb_index, test_config.output_fmt}})
            .set_page_size(output_cb_index, test_config.output_single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<std::uint32_t> compute_kernel_args = {
        std::uint32_t(test_config.num_tiles_r),
        std::uint32_t(test_config.num_tiles_c),
    };

    tt_metal::CreateKernel(
        program,
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
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/true);

    std::vector<std::uint32_t> src1_vec = create_constant_vector_of_bfloat16(input_dram_buffer_size, 1.0f);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, /*blocking=*/true);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            dram_buffer_src0_addr,
            (std::uint32_t)0,
            dram_buffer_src1_addr,
            (std::uint32_t)0,
            (std::uint32_t)num_tiles,
        });
    tt_metal::SetRuntimeArgs(program, writer_kernel, core, {dram_buffer_dst_addr, (std::uint32_t)0, num_tiles});

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    validate_result(test_config, src0_vec, src1_vec, result_vec);
}

// Metal 2.0 single-core helper for the Quasar `unpack_tilizeA_B` + reduce path.
// Quasar's unpack_tilizeA_B is only compatible with the reduce math kernel (not eltwise binary).
// Uses REDUCE_COL + MAX: tilizes row-major src0 data, reduces each tile independently
// (column-wise max within each tile), producing output tiles with only row 0 populated.
void run_single_core_unpack_tilizeA_B_reduce_program(
    distributed::MeshDevice& mesh_device, const TestConfig& test_config) {
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
        mesh_device, make_flat_tensor_spec(test_config.input_single_tile_size, num_tiles_in));
    auto out_tensor = MeshTensor::allocate_on_device(
        mesh_device, make_flat_tensor_spec(test_config.output_single_tile_size, num_tiles_out));

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
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
    } else {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                },
        };
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
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
    } else {
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                },
        };
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

    experimental::ComputeHardwareConfig compute_hw_config;
    compute_hw_config = experimental::ComputeHardwareConfig{
        .enable_32_bit_dest = test_config.fp32_dest_acc_en,
        .double_buffer_dest = !test_config.dst_full_sync_en,
    };
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

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    std::vector<std::uint32_t> src0_vec = create_random_vector_of_bfloat16(input_dram_buffer_size, 100, 42);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), src0_vec);

    float scaler_f = 1.0f;
    std::vector<std::uint32_t> scaler_tile_vec = create_constant_vector_of_bfloat16(scaler_tile_size, scaler_f);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"num_tiles", num_tiles_in}, {"scaler", *reinterpret_cast<std::uint32_t*>(&scaler_f)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles_out}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN_TENSOR, experimental::ProgramRunArgs::TensorArgument{in_tensor}},
        {OUT_TENSOR, experimental::ProgramRunArgs::TensorArgument{out_tensor}},
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), result_vec);

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
                unit_tests::compute::tilize::run_single_core_tilize_program(*this->devices_.at(0), test_config);
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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputeUnpackTilizeInt8) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {false, true}) {
            std::uint32_t num_tiles_total = num_tile[0] * num_tile[1];
            auto src_data = tt::test_utils::create_random_vector_of_int8(
                tt::tile_size(tt::DataFormat::Int8) * num_tiles_total, /*seed=*/42);
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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputeUnpackTilizeUInt8) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {false, true}) {
            std::uint32_t num_tiles_total = num_tile[0] * num_tile[1];
            auto src_data = tt::test_utils::create_random_vector_of_uint8(
                tt::tile_size(tt::DataFormat::UInt8) * num_tiles_total, /*seed=*/42);
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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
        }
    }
}

// Exercises llk_unpack_tilize_block with a 16x32 tiny tile across multiple tile-rows, using a
// nonzero input_tile_index (tilize_across_tile_rows.cpp) so the cross-tile-row stride is hit.
TEST_F(LLKBlackholeSingleCardFixture, TensixComputeUnpackTilizeTinyTile16x32) {
    constexpr std::uint32_t tile_shape_in_faces_r = 1;  // 16x32 = 1x2 faces of 16x16
    constexpr std::uint32_t tile_shape_in_faces_c = 2;
    constexpr std::uint32_t face_r_dim = tt::constants::FACE_HEIGHT;
    constexpr std::uint32_t face_c_dim = tt::constants::FACE_WIDTH;
    constexpr std::uint32_t tile_bytes =
        tile_shape_in_faces_r * tile_shape_in_faces_c * face_r_dim * face_c_dim * sizeof(std::uint16_t);
    // num_tiles_r >= 2 so the cross-tile-row stride in llk_unpack_tilize_block is exercised.
    vector<vector<std::uint32_t>> num_tiles = {{2, 1}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {false, true}) {
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .tilize_cross_tile_rows = true,
                .input_single_tile_size = tile_bytes,
                .output_single_tile_size = tile_bytes,
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .tile_shape_in_faces_r = tile_shape_in_faces_r,
                .tile_shape_in_faces_c = tile_shape_in_faces_c,
                .face_r_dim = face_r_dim,
                .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
                .golden_function = ::unit_tests::compute::gold_standard_tilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
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
                unit_tests::compute::tilize::run_single_core_tilize_program(*this->devices_.at(0), test_config);
            }
        }
    }
}

// tilizeA_B + eltwise add on narrow tiles: 32x16 (two vertically stacked 16x16 faces) and 16x16 (one face).
// A is row-major, B is a constant, so every output datum is the tilized A datum plus that constant. Small
// integer inputs keep the bf16 sum exact.
TEST_F(LLKMeshDeviceFixture, TensixComputeUnpackTilizeA_BNarrowTile) {
    if (MetalContext::instance().get_cluster().arch() != ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "only the Wormhole tilizeA_B has a narrow-tile path";
    }
    constexpr std::uint32_t tile_w = 16;
    constexpr std::uint32_t face_dim = 16;
    constexpr std::uint32_t dram_bank_id = 0;
    constexpr float b_value = 1.0f;

    for (const std::uint32_t tile_h : {32u, 16u}) {
        const Tile tile({tile_h, tile_w});
        const std::uint32_t tile_bytes = tile.get_tile_size(tt::DataFormat::Float16_b);
        for (const auto& [num_tiles_r, num_tiles_c] : {std::pair<std::uint32_t, std::uint32_t>{1, 1}, {2, 4}}) {
            auto& mesh_device = *this->devices_.at(0);
            auto& cq = mesh_device.mesh_command_queue();
            Program program = CreateProgram(*this->devices_[0]);
            const CoreCoord core = {0, 0};
            const std::uint32_t num_tiles = num_tiles_r * num_tiles_c;
            const std::uint32_t buffer_bytes = num_tiles * tile_bytes;

            auto make_buffer = [&]() {
                return distributed::MeshBuffer::create(
                    distributed::ReplicatedBufferConfig{.size = buffer_bytes},
                    {.page_size = buffer_bytes, .buffer_type = BufferType::DRAM},
                    &mesh_device);
            };
            auto src0 = make_buffer();
            auto src1 = make_buffer();
            auto dst = make_buffer();
            for (const auto cb : {tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16}) {
                CircularBufferConfig config =
                    CircularBufferConfig(buffer_bytes, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, tile_bytes);
                config.set_tile_dims(cb, tile);
                CreateCircularBuffer(program, core, config);
            }
            auto reader = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto writer = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/unpack_tilizeA_B.cpp",
                core,
                ComputeConfig{.compile_args = {num_tiles_r, num_tiles_c}});

            // A: row-major [num_tiles_r * tile_h, num_tiles_c * 16], cycling through the integers in [-50, 50]:
            // 37 and 101 are coprime, so (i * 37) % 101 visits every residue once per 101 datums.
            const std::uint32_t cols = num_tiles_c * tile_w;
            const std::uint32_t rows = num_tiles_r * tile_h;
            std::vector<bfloat16> a(rows * cols);
            for (std::uint32_t i = 0; i < a.size(); i++) {
                a[i] = bfloat16(static_cast<float>(static_cast<int>((i * 37) % 101) - 50));
            }
            distributed::EnqueueWriteMeshBuffer(cq, src0, pack_vector<std::uint32_t, bfloat16>(a), /*blocking=*/true);
            distributed::EnqueueWriteMeshBuffer(
                cq, src1, create_constant_vector_of_bfloat16(buffer_bytes, b_value), /*blocking=*/true);
            SetRuntimeArgs(
                program, reader, core, {src0->address(), dram_bank_id, src1->address(), dram_bank_id, num_tiles});
            SetRuntimeArgs(program, writer, core, {dst->address(), dram_bank_id, num_tiles});
            LaunchProgram(mesh_device, std::move(program));

            std::vector<std::uint32_t> result_packed;
            distributed::EnqueueReadMeshBuffer(cq, result_packed, dst, /*blocking=*/true);
            const auto result = unpack_vector<bfloat16, std::uint32_t>(result_packed);

            // Golden in output tile order: per tile, face 0 (rows 0-15) then, for 32-high tiles, face 1
            // (rows 16-31), each row-major.
            std::vector<bfloat16> golden;
            golden.reserve(a.size());
            for (std::uint32_t tr = 0; tr < num_tiles_r; tr++) {
                for (std::uint32_t tc = 0; tc < num_tiles_c; tc++) {
                    for (std::uint32_t face = 0; face < tile_h / face_dim; face++) {
                        for (std::uint32_t r = 0; r < face_dim; r++) {
                            for (std::uint32_t c = 0; c < face_dim; c++) {
                                const std::uint32_t row = tr * tile_h + face * face_dim + r;
                                golden.push_back(
                                    bfloat16(static_cast<float>(a[row * cols + tc * tile_w + c]) + b_value));
                            }
                        }
                    }
                }
            }
            ASSERT_EQ(result.size(), golden.size());
            std::uint32_t mismatches = 0;
            for (std::uint32_t i = 0; i < golden.size(); i++) {
                mismatches += static_cast<float>(result[i]) != static_cast<float>(golden[i]);
            }
            EXPECT_EQ(mismatches, 0u) << tile_h << "x" << tile_w << " tiles, " << num_tiles_r << "x" << num_tiles_c;
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
        unit_tests::compute::tilize::run_single_core_unpack_tilizeA_B_program(*this->devices_.at(0), test_config);
    }
}

/******************************
Following tests are for Quasar
*******************************/
enum class QuasarTestMode { TILIZE, UNTILIZE, UNTILIZE_DST };

static void run_quasar_tilize_untilize_test(
    distributed::MeshDevice& mesh_device,
    std::uint32_t num_tiles_r,
    std::uint32_t num_tiles_c,
    QuasarTestMode mode,
    bool dst_full_sync_en,
    bool fp32_dest_acc_en,
    tt::DataFormat input_data_format,
    tt::DataFormat output_data_format,
    std::array<std::uint32_t, 2> tile_shape_in_faces = {2, 2},
    std::uint32_t face_r_dim = tt::constants::FACE_HEIGHT,
    bool tilize_cross_tile_rows = false,
    bool unpack_to_dest = false) {
    bool is_tilize = (mode == QuasarTestMode::TILIZE);

    auto& cq = mesh_device.mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    constexpr std::uint32_t face_c_dim = tt::constants::FACE_WIDTH;
    const std::uint32_t num_faces = tile_shape_in_faces[0] * tile_shape_in_faces[1];
    const bool tiny_tile = (tile_shape_in_faces != std::array<std::uint32_t, 2>{2, 2} || face_r_dim != 16);

    bool is_8bit_integer = (input_data_format == tt::DataFormat::Int8 || input_data_format == tt::DataFormat::UInt8);
    std::uint32_t num_tiles = num_tiles_r * num_tiles_c;
    std::uint32_t input_single_tile_size = tt::tile_size(input_data_format);
    std::uint32_t output_single_tile_size = tt::tile_size(output_data_format);
    if (tiny_tile) {
        input_single_tile_size = tt::datum_size(input_data_format) * num_faces * face_r_dim * face_c_dim;
        output_single_tile_size = tt::datum_size(output_data_format) * num_faces * face_r_dim * face_c_dim;
    }
    std::uint32_t src_dram_buffer_size = input_single_tile_size * num_tiles;
    std::uint32_t dst_dram_buffer_size = output_single_tile_size * num_tiles;

    auto src_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = src_dram_buffer_size},
        {.page_size = src_dram_buffer_size, .buffer_type = BufferType::DRAM},
        &mesh_device);
    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dst_dram_buffer_size},
        {.page_size = dst_dram_buffer_size, .buffer_type = BufferType::DRAM},
        &mesh_device);
    std::uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    // tilize_across_tile_rows.cpp waits/reserves the whole block at once instead of streaming
    // per tile-row, so the DFBs must hold all tiles.
    std::uint32_t dfb_num_entries = tilize_cross_tile_rows ? num_tiles : std::max(2u, num_tiles_c);

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
    if (tiny_tile) {
        const auto tile =
            Tile::from_face_grid(tile_shape_in_faces[0], tile_shape_in_faces[1], {face_r_dim, constants::FACE_WIDTH});
        input_dfb_spec.tile_format_metadata = tile;
        output_dfb_spec.tile_format_metadata = tile;
    }

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };

    std::string compute_kernel;
    experimental::KernelSpec::CompileTimeArgs compute_cta_bindings;
    switch (mode) {
        case QuasarTestMode::TILIZE:
            compute_kernel = tilize_cross_tile_rows
                                 ? "tests/tt_metal/tt_metal/test_kernels/compute/tilize_across_tile_rows.cpp"
                                 : "tests/tt_metal/tt_metal/test_kernels/compute/tilize.cpp";
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
    compute_hw_config = experimental::ComputeHardwareConfig{
        .enable_32_bit_dest = fp32_dest_acc_en,
        .double_buffer_dest = !dst_full_sync_en,
    };
    if (unpack_to_dest) {
        compute_hw_config.unpack_modes = {{INPUT_DFB, tt::tt_metal::UnpackMode::UnpackToDest}};
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

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    std::vector<std::uint32_t> src_vec;
    if (input_data_format == tt::DataFormat::Int8) {
        src_vec = tt::test_utils::create_random_vector_of_int8(src_dram_buffer_size, /*seed=*/42);
    } else if (input_data_format == tt::DataFormat::UInt8) {
        src_vec = tt::test_utils::create_random_vector_of_uint8(src_dram_buffer_size, /*seed=*/42);
    } else if (input_data_format == tt::DataFormat::Int16) {
        src_vec.resize(src_dram_buffer_size / sizeof(std::uint32_t));
        for (std::uint32_t i = 0; i < src_vec.size(); i++) {
            src_vec[i] = (static_cast<std::uint32_t>((2 * i) + 1) << 16) | static_cast<std::uint32_t>(2 * i);
        }
    } else if (input_data_format == tt::DataFormat::Float32) {
        src_vec.resize(src_dram_buffer_size / sizeof(std::uint32_t));
        for (std::uint32_t i = 0; i < src_vec.size(); i++) {
            src_vec[i] = std::bit_cast<std::uint32_t>(static_cast<float>(i));
        }
    } else if (input_data_format == tt::DataFormat::Int32) {
        src_vec.resize(src_dram_buffer_size / sizeof(std::uint32_t));
        for (std::uint32_t i = 0; i < src_vec.size(); i++) {
            // Unique per datum with the upper halfword set too (bit 31 stays clear), so a DEST path that dropped the
            // high bits would fail the bit-exact compare.
            src_vec[i] = i * 0x00010001u;
        }
    } else {
        src_vec = create_arange_vector_of_bfloat16(src_dram_buffer_size, false);
    }
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, /*blocking=*/true);

    // This test configures the DRAM buffers as a single whole-buffer page, so
    // aligned_page_size() returns the whole-buffer stride rather than per-tile.
    // Compute the real per-tile DRAM stride directly from the buffer size.
    const std::uint32_t src_tile_stride_bytes = src_dram_buffer_size / num_tiles;
    const std::uint32_t dst_tile_stride_bytes = dst_dram_buffer_size / num_tiles;

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", dram_buffer_src_addr},
                 {"src_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"dram_page_stride", src_tile_stride_bytes}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", dram_buffer_dst_addr},
                 {"dst_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"dram_page_stride", dst_tile_stride_bytes}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    ::unit_tests::compute::GoldenConfig golden_config = {
        .num_tiles_r_dim = static_cast<int>(num_tiles_r),
        .num_tiles_c_dim = static_cast<int>(num_tiles_c),
        .face_r_dim = static_cast<int>(face_r_dim),
        .face_c_dim = static_cast<int>(face_c_dim),
        .num_faces = static_cast<int>(num_faces),
        .tiny_tile = tiny_tile,
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

    ASSERT_EQ(golden.size(), result_vec.size());
    for (std::size_t i = 0; i < golden.size(); i++) {
        if (golden[i] != result_vec[i]) {
            const std::size_t lo = i >= 8 ? i - 8 : 0;
            const std::size_t hi = std::min(golden.size(), i + 8);
            std::string window;
            for (std::size_t j = lo; j < hi; j++) {
                window +=
                    fmt::format("{}[{}] {:#010x} vs {:#010x}\n", j == i ? "->" : "  ", j, golden[j], result_vec[j]);
            }
            FAIL() << "first mismatch at word " << i << " of " << golden.size() << " (golden vs result):\n" << window;
        }
    }
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
                        this->device(),
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
                        this->device(),
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

// Pack Untilize tiny tile (1x32, 2x32) via pack_untilize_block (unpack in the loop).
// {faces_r, faces_c, face_r_dim}: 1x32 = {1, 2, 1}, 2x32 = {1, 2, 2}.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputePackUntilizeTinyTile) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {2, 2}};
    std::vector<std::array<std::uint32_t, 3>> geometries = {{1, 2, 1}, {1, 2, 2}};
    for (auto& cfg : test_configs) {
        for (auto& geo : geometries) {
            for (bool dst_full_sync_en : {true, false}) {
                run_quasar_tilize_untilize_test(
                    this->device(),
                    cfg[0],
                    cfg[1],
                    QuasarTestMode::UNTILIZE,
                    dst_full_sync_en,
                    /*fp32_dest_acc_en=*/false,
                    tt::DataFormat::Float16_b,
                    tt::DataFormat::Float16_b,
                    /*tile_shape_in_faces=*/{geo[0], geo[1]},
                    /*face_r_dim=*/geo[2]);
            }
        }
    }
}

// Pack Untilize Dst tiny tile (1x32, 2x32): tiles pre-loaded into dest via copy_tile.
// {faces_r, faces_c, face_r_dim}: 1x32 = {1, 2, 1}, 2x32 = {1, 2, 2}.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputePackUntilizeDstTinyTile) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {2, 2}};
    std::vector<std::array<std::uint32_t, 3>> geometries = {{1, 2, 1}, {1, 2, 2}};
    for (auto& cfg : test_configs) {
        for (auto& geo : geometries) {
            for (bool dst_full_sync_en : {true, false}) {
                run_quasar_tilize_untilize_test(
                    this->device(),
                    cfg[0],
                    cfg[1],
                    QuasarTestMode::UNTILIZE_DST,
                    dst_full_sync_en,
                    /*fp32_dest_acc_en=*/false,
                    tt::DataFormat::Float16_b,
                    tt::DataFormat::Float16_b,
                    /*tile_shape_in_faces=*/{geo[0], geo[1]},
                    /*face_r_dim=*/geo[2]);
            }
        }
    }
}

// Quasar Unpack Tilize
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputeUnpackTilize) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (tt::DataFormat input_data_format : {tt::DataFormat::Float16_b, tt::DataFormat::Int16}) {
                    if (fp32_dest_acc_en && input_data_format == tt::DataFormat::Int16) {
                        continue;  // Int16 + 32-bit dest mode is not supported on Quasar
                    }

                    run_quasar_tilize_untilize_test(
                        this->device(),
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

// Unpack-to-dest tilize: the unpacker tilizes straight into DEST, the route the tilize op factories take for fp32.
// 32-bit inputs into a 32-bit DEST, Float16_b into a 16-bit DEST. Every tile is its own DEST section, so the wider
// blocks run several SyncHalf sections per block and exercise the bank alternation.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputeUnpackTilizeToDest) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            for (tt::DataFormat input_data_format :
                 {tt::DataFormat::Float32, tt::DataFormat::Int32, tt::DataFormat::Float16_b}) {
                const bool fp32_dest_acc_en = input_data_format != tt::DataFormat::Float16_b;
                run_quasar_tilize_untilize_test(
                    this->device(),
                    cfg[0],
                    cfg[1],
                    QuasarTestMode::TILIZE,
                    dst_full_sync_en,
                    fp32_dest_acc_en,
                    input_data_format,
                    input_data_format,
                    /*tile_shape_in_faces=*/{2, 2},
                    /*face_r_dim=*/tt::constants::FACE_HEIGHT,
                    /*tilize_cross_tile_rows=*/false,
                    /*unpack_to_dest=*/true);
            }
        }
    }
}

// Unpack-to-dest tilize with a nonzero input_tile_index per tile row (tilize_across_tile_rows.cpp), so the
// cross-tile-row L1 offset goes through _llk_unpack_tilize_to_dest_set_src_offset_.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputeUnpackTilizeToDestCrossTileRows) {
    for (bool dst_full_sync_en : {true, false}) {
        run_quasar_tilize_untilize_test(
            this->device(),
            2,
            10,
            QuasarTestMode::TILIZE,
            dst_full_sync_en,
            /*fp32_dest_acc_en=*/true,
            tt::DataFormat::Float32,
            tt::DataFormat::Float32,
            /*tile_shape_in_faces=*/{2, 2},
            /*face_r_dim=*/tt::constants::FACE_HEIGHT,
            /*tilize_cross_tile_rows=*/true,
            /*unpack_to_dest=*/true);
    }
}

// Unpack Tilize tiny tile (1x32, 2x32) via tilize_block.
// {faces_r, faces_c, face_r_dim}: 1x32 = {1, 2, 1}, 2x32 = {1, 2, 2}.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputeUnpackTilizeTinyTile) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {2, 2}};
    std::vector<std::array<std::uint32_t, 3>> geometries = {{1, 2, 1}, {1, 2, 2}};
    for (auto& cfg : test_configs) {
        for (auto& geo : geometries) {
            for (bool dst_full_sync_en : {true, false}) {
                run_quasar_tilize_untilize_test(
                    this->device(),
                    cfg[0],
                    cfg[1],
                    QuasarTestMode::TILIZE,
                    dst_full_sync_en,
                    /*fp32_dest_acc_en=*/false,
                    tt::DataFormat::Float16_b,
                    tt::DataFormat::Float16_b,
                    /*tile_shape_in_faces=*/{geo[0], geo[1]},
                    /*face_r_dim=*/geo[2]);
            }
        }
    }
}

// Unpack Tilize tiny tiles (via tilize_block) with a nonzero input_tile_index per tile-row
// (tilize_across_tile_rows.cpp), so the cross-tile-row stride in llk_unpack_tilize_block is
// exercised, like the Blackhole TensixComputeUnpackTilizeTinyTile16x32 test.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputeUnpackTilizeTinyTileCrossTileRows) {
    std::vector<std::array<std::uint32_t, 3>> geometries = {{1, 2, 1}, {1, 2, 2}};
    std::vector<vector<std::uint32_t>> test_configs = {{2, 1}, {2, 2}};
    for (auto& geometry : geometries) {
        for (auto& cfg : test_configs) {
            for (bool dst_full_sync_en : {true, false}) {
                run_quasar_tilize_untilize_test(
                    this->device(),
                    cfg[0],
                    cfg[1],
                    QuasarTestMode::TILIZE,
                    dst_full_sync_en,
                    false,
                    tt::DataFormat::Float16_b,
                    tt::DataFormat::Float16_b,
                    /*tile_shape_in_faces=*/{geometry[0], geometry[1]},
                    /*face_r_dim=*/geometry[2],
                    /*tilize_cross_tile_rows=*/true);
            }
        }
    }
}

// Quasar Unpack TilizeA_B (tilize + reduce col max)
// Quasar's unpack_tilizeA_B is only compatible with the reduce math kernel.
// Disabled due to hang: tenstorrent/tt-metal#59923
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, DISABLED_QuasarComputeUnpackTilizeA_B) {
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
                    this->device(), test_config);
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
                this->device(),
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
                this->device(),
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

// Pack untilize with the input unpacked straight into DEST (UnpackToDestEn): the FLOAT32 / INT32 untilize path on
// Quasar (#62275), plus Float16_b into a 16-bit DEST as in the tilize-to-dest sweep. Each block row is one DEST
// section on all three threads; {2, 10} splits a row into 2-tile blocks.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputePackUntilizeToDest) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {4, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            for (tt::DataFormat input_data_format :
                 {tt::DataFormat::Float32, tt::DataFormat::Int32, tt::DataFormat::Float16_b}) {
                const bool fp32_dest_acc_en = input_data_format != tt::DataFormat::Float16_b;
                run_quasar_tilize_untilize_test(
                    this->device(),
                    cfg[0],
                    cfg[1],
                    QuasarTestMode::UNTILIZE,
                    dst_full_sync_en,
                    fp32_dest_acc_en,
                    input_data_format,
                    input_data_format,
                    /*tile_shape_in_faces=*/{2, 2},
                    /*face_r_dim=*/tt::constants::FACE_HEIGHT,
                    /*tilize_cross_tile_rows=*/false,
                    /*unpack_to_dest=*/true);
            }
        }
    }
}

// Tilize-to-dest then pack-untilize-to-dest on the same device, as the fp32 tilize -> untilize round trip runs them:
// checks the untilize leg right after a tilize program has run on the core (unpacker counter leftovers).
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputeTilizeThenPackUntilizeToDest) {
    for (bool dst_full_sync_en : {false, true}) {
        for (tt::DataFormat input_data_format : {tt::DataFormat::Float32, tt::DataFormat::Int32}) {
            for (QuasarTestMode mode : {QuasarTestMode::TILIZE, QuasarTestMode::UNTILIZE}) {
                run_quasar_tilize_untilize_test(
                    this->device(),
                    /*num_tiles_r=*/2,
                    /*num_tiles_c=*/4,
                    mode,
                    dst_full_sync_en,
                    /*fp32_dest_acc_en=*/true,
                    input_data_format,
                    input_data_format,
                    /*tile_shape_in_faces=*/{2, 2},
                    /*face_r_dim=*/tt::constants::FACE_HEIGHT,
                    /*tilize_cross_tile_rows=*/false,
                    /*unpack_to_dest=*/true);
            }
        }
    }
}

// Regular unpack-to-dest (copy_tile in tile_regs_acquire / commit) then pack_untilize_dest on the tiles it placed,
// for 32-bit inputs (dst_untilize.cpp with UnpackToDestEn).
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarComputePackUntilizeDstToDest) {
    std::vector<vector<std::uint32_t>> test_configs = {{1, 1}, {4, 4}, {5, 3}, {2, 10}};
    for (auto& cfg : test_configs) {
        for (bool dst_full_sync_en : {true, false}) {
            for (tt::DataFormat input_data_format : {tt::DataFormat::Float32, tt::DataFormat::Int32}) {
                run_quasar_tilize_untilize_test(
                    this->device(),
                    cfg[0],
                    cfg[1],
                    QuasarTestMode::UNTILIZE_DST,
                    dst_full_sync_en,
                    /*fp32_dest_acc_en=*/true,
                    input_data_format,
                    input_data_format,
                    /*tile_shape_in_faces=*/{2, 2},
                    /*face_r_dim=*/tt::constants::FACE_HEIGHT,
                    /*tilize_cross_tile_rows=*/false,
                    /*unpack_to_dest=*/true);
            }
        }
    }
}

// Quasar fast untilize: no dedicated fast-untilize LLK on Quasar, so fast_untilize_* (api/compute/experimental/
// fast_untilize.h) forwards to the plain pack_untilize path -- this exercises that forwarding on real Quasar
// single-card CI.
// Disabled due to hang: tenstorrent/tt-metal#59923
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, DISABLED_QuasarComputeFastUntilize) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 4}, {2, 2}};
    for (auto num_tile : num_tiles) {
        unit_tests::compute::tilize::TestConfig test_config = {
            .dst_full_sync_en = false,
            .fp32_dest_acc_en = false,
            .fast_untilize = true,
            .input_single_tile_size = 2 * 1024,
            .output_single_tile_size = 2 * 1024,
            .num_tiles_r = num_tile[0],
            .num_tiles_c = num_tile[1],
            .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
            .output_fmt = tt::DataFormat::Float16_b,
            .golden_function = ::unit_tests::compute::gold_standard_untilize};
        unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
    }
}

// Quasar fast tilize: no dedicated fast-tilize LLK on Quasar, so fast_tilize_* forwards to the plain
// unpack_tilize path (tilize.h) -- this exercises that forwarding on real Quasar single-card CI.
// Disabled due to hang: tenstorrent/tt-metal#59923
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, DISABLED_QuasarComputeFastTilize) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 4}, {2, 2}};
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
                unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
            }
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
                unit_tests::compute::tilize::run_single_core_tilize_program(*this->devices_.at(0), test_config);
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
            unit_tests::compute::tilize::run_single_core_tilize_program(*this->devices_.at(0), test_config);
        }
    }
}

TEST_F(LLKMeshDeviceFixture, TensixComputeFastUntilize) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 4}, {2, 2}};
    for (auto num_tile : num_tiles) {
        unit_tests::compute::tilize::TestConfig test_config = {
            .dst_full_sync_en = false,
            .fp32_dest_acc_en = false,
            .fast_untilize = true,
            .input_single_tile_size = 2 * 1024,
            .output_single_tile_size = 2 * 1024,
            .num_tiles_r = num_tile[0],
            .num_tiles_c = num_tile[1],
            .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
            .output_fmt = tt::DataFormat::Float16_b,
            .golden_function = ::unit_tests::compute::gold_standard_untilize};
        unit_tests::compute::tilize::run_single_core_tilize_program(*this->devices_.at(0), test_config);
    }
}

// disabled due to non-deterministic failures: tenstorrent/tt-metal#63295
TEST_F(LLKBlackholeSingleCardFixture, DISABLED_TensixComputePackUntilizeFp8e4m3) {
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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputePackUntilizeInt8) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {true, false}) {
            std::uint32_t num_t = num_tile[0] * num_tile[1];
            auto src_data =
                tt::test_utils::create_random_vector_of_int8(tt::tile_size(tt::DataFormat::Int8) * num_t, /*seed=*/42);
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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixComputePackUntilizeUInt8) {
    vector<vector<std::uint32_t>> num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for (auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {true, false}) {
            std::uint32_t num_t = num_tile[0] * num_tile[1];
            auto src_data = tt::test_utils::create_random_vector_of_uint8(
                tt::tile_size(tt::DataFormat::UInt8) * num_t, /*seed=*/42);
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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
        }
    }
}

// Tests pack_untilize with tiny tile dims.
// 1x16 = 1x1 faces of height 1; 1x32 = 1x2 faces of height 1.
TEST_F(LLKMeshDeviceFixture, TensixComputePackUntilizeDstTinyTile) {
    if (MetalContext::instance().get_cluster().arch() == ARCH::QUASAR) {
        // Copies a full 32x32 tile into dest; Quasar's tiny-tile untilize expects a tiny tile there.
        GTEST_SKIP() << "Covered on Quasar by QuasarComputePackUntilizeDstTinyTile";
    }
    struct TinyTileCase {
        std::uint32_t num_tiles_r;
        std::uint32_t num_tiles_c;
        std::uint32_t tile_shape_in_faces_r;
        std::uint32_t tile_shape_in_faces_c;
        std::uint32_t face_r_dim;
    };
    const std::vector<TinyTileCase> test_cases = {
        {.num_tiles_r = 1, .num_tiles_c = 1, .tile_shape_in_faces_r = 1, .tile_shape_in_faces_c = 1, .face_r_dim = 1},
        {.num_tiles_r = 1, .num_tiles_c = 1, .tile_shape_in_faces_r = 1, .tile_shape_in_faces_c = 2, .face_r_dim = 1},
        {.num_tiles_r = 1, .num_tiles_c = 2, .tile_shape_in_faces_r = 1, .tile_shape_in_faces_c = 2, .face_r_dim = 1},
    };
    std::uint32_t face_c_dim = 16;
    for (const auto& test_case : test_cases) {
        for (bool dst_full_sync_en : {true, false}) {
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .input_single_tile_size = 2 * 1024,
                .output_single_tile_size = 2 * test_case.tile_shape_in_faces_r * test_case.tile_shape_in_faces_c *
                                           test_case.face_r_dim * face_c_dim,
                .num_tiles_r = test_case.num_tiles_r,
                .num_tiles_c = test_case.num_tiles_c,
                .tile_shape_in_faces_r = test_case.tile_shape_in_faces_r,
                .tile_shape_in_faces_c = test_case.tile_shape_in_faces_c,
                .face_r_dim = test_case.face_r_dim,
                .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
                .output_fmt = tt::DataFormat::Float16_b,
                .golden_function = ::unit_tests::compute::gold_standard_untilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(*this->devices_.at(0), test_config);
        }
    }
}

TEST_F(LLKBlackholeSingleCardFixture, TensixCustomPackUntilizeExplicitGeometry) {
    // Three rows exercise half-sync bank reuse. Width 12 requires several DEST
    // sections, so block column offsets are checked as well as face geometry.
    for (const auto num_faces : {1u, 2u, 4u}) {
        // copy_tile populates full 16x16 face slots. One/two-face tiny tiles
        // select their leading rows; the four-face case uses the full shape.
        // Short four-face SDPA layouts require a different DEST producer.
        // One face is 1x1, two faces are 1x2, four faces are 2x2.
        const std::uint32_t faces_r = num_faces > 2 ? 2 : 1;
        const std::uint32_t faces_c = num_faces == 1 ? 1 : 2;
        const auto face_heights = num_faces == 4 ? vector<std::uint32_t>{16} : vector<std::uint32_t>{1, 8, 16};
        const auto narrow_modes = num_faces == 1 ? vector<bool>{false, true} : vector<bool>{false};
        for (const auto face_rows : face_heights) {
            for (const auto width : {1u, 12u}) {
                for (const bool full_sync : {false, true}) {
                    for (const bool narrow_row : narrow_modes) {
                        SCOPED_TRACE(fmt::format(
                            "faces={}, rows={}, width={}, full_sync={}, narrow_row={}",
                            num_faces,
                            face_rows,
                            width,
                            full_sync,
                            narrow_row));
                        unit_tests::compute::tilize::TestConfig config{
                            .dst_full_sync_en = full_sync,
                            .explicit_untilize_geometry = true,
                            .explicit_untilize_narrow_row = narrow_row,
                            .input_single_tile_size = 2048,
                            .output_single_tile_size = 2 * num_faces * face_rows * 16,
                            .num_tiles_r = 3,
                            .num_tiles_c = width,
                            .tile_shape_in_faces_r = faces_r,
                            .tile_shape_in_faces_c = faces_c,
                            .face_r_dim = face_rows,
                            .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
                            .golden_function = ::unit_tests::compute::gold_standard_untilize};
                        unit_tests::compute::tilize::run_single_core_tilize_program(*devices_.at(0), config);
                    }
                }
            }
        }
    }
}

namespace {

// Golden for a narrow-row untilize of full 32x32 tiles: every untilized row holds, for each tile column in turn,
// the first row_num_datums datums of that tile's row (bf16: two datums per word, so row_num_datums must be even).
// With one tile column that is a prefix of the untilized row; with several it is the per-column prefixes back to
// back, which is how the narrow MOP packs a block of several tiles (one short PACR per tile, closing the row after
// the last).
std::function<std::vector<std::uint32_t>(const std::vector<std::uint32_t>&, const ::unit_tests::compute::GoldenConfig&)>
narrow_row_untilize_golden(std::uint32_t row_num_datums) {
    return [row_num_datums](const std::vector<std::uint32_t>& src, const ::unit_tests::compute::GoldenConfig& config) {
        const auto full = ::unit_tests::compute::gold_standard_untilize(src, config);
        const std::size_t tile_row_words = constants::TILE_WIDTH * config.datum_bytes / sizeof(std::uint32_t);
        const std::size_t full_row_words = static_cast<std::size_t>(config.num_tiles_c_dim) * tile_row_words;
        const std::size_t narrow_row_words = row_num_datums * config.datum_bytes / sizeof(std::uint32_t);
        std::vector<std::uint32_t> narrow;
        for (std::size_t row_start = 0; row_start < full.size(); row_start += full_row_words) {
            for (std::size_t column = 0; column < static_cast<std::size_t>(config.num_tiles_c_dim); ++column) {
                const auto tile_row = full.begin() + row_start + column * tile_row_words;
                narrow.insert(narrow.end(), tile_row, tile_row + narrow_row_words);
            }
        }
        return narrow;
    };
}

// Watcher poll period for the narrow-row assert fixture below. A clean run sleeps kPollsBeforeCleanCheck
// polls after completion before its watcher state is read: one for the poll in flight when the program
// finished, one for a late report.
constexpr int kNarrowRowWatcherIntervalMs = 250;
constexpr int kPollsBeforeCleanCheck = 2;
// Every narrow row in this file is bf16 unless a case says otherwise (the guard kernel's CBs take the case's format).
constexpr tt::DataFormat kNarrowRowFormat = tt::DataFormat::Float16_b;
constexpr std::uint32_t kNarrowRowDatumBytes = tt::datum_size(kNarrowRowFormat);

// Arguments of the one pack_untilize_dest the guard kernel issues when asked to (Wormhole dest-tile guards).
struct NarrowRowGuardExecute {
    std::uint32_t tile_dst_ct_offset = 0;
    std::uint32_t block_rt_dim = 1;
    std::uint32_t tile_dst_rt_offset = 0;
};

// Builds the program for pack_untilize_narrow_row_guard.cpp, the init(-only) kernel, for row_num_datums across
// full_ct_dim tile columns in blocks of block_ct_dim (default: one block) on one core with a CB pair in `format`
// (default bf16); the output CB carries output_tile's face geometry (default: a full 32x32 tile), which is what the
// init reads its face_r_dim and num_faces from. With `execute` the kernel also issues one pack_untilize_dest with
// those arguments. The caller compiles or launches it.
Program make_pack_untilize_narrow_row_guard_program(
    std::uint32_t row_num_datums,
    std::uint32_t full_ct_dim,
    std::optional<std::uint32_t> block_ct_dim = std::nullopt,
    std::optional<Tile> output_tile = std::nullopt,
    tt::DataFormat format = kNarrowRowFormat,
    std::optional<NarrowRowGuardExecute> execute = std::nullopt) {
    const CoreCoord core{0, 0};
    Program program = CreateProgram();
    const std::uint32_t tile_bytes = tt::tile_size(format);
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(tile_bytes, {{tt::CBIndex::c_0, format}}).set_page_size(tt::CBIndex::c_0, tile_bytes));
    const std::uint32_t output_tile_bytes = output_tile.has_value() ? output_tile->get_tile_size(format) : tile_bytes;
    CircularBufferConfig output_config(output_tile_bytes, {{tt::CBIndex::c_16, format}});
    output_config.set_page_size(tt::CBIndex::c_16, output_tile_bytes);
    if (output_tile.has_value()) {
        output_config.set_tile_dims(tt::CBIndex::c_16, *output_tile);
    }
    CreateCircularBuffer(program, core, output_config);
    const NarrowRowGuardExecute exec = execute.value_or(NarrowRowGuardExecute{});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_narrow_row_guard.cpp",
        core,
        ComputeConfig{
            .compile_args = {
                row_num_datums,
                full_ct_dim,
                block_ct_dim.value_or(full_ct_dim),
                exec.tile_dst_ct_offset,
                exec.block_rt_dim,
                exec.tile_dst_rt_offset,
                execute.has_value() ? 1u : 0u}});
    return program;
}

// The guard kernel's geometry: datums per narrow row, tile columns, tiles per block (default: one block), the
// output tile (default: a full 32x32 tile, four 16-row faces), the CB format (default bf16) and, when set, the
// arguments of one pack_untilize_dest the kernel issues after the init.
struct NarrowRowGuardCase {
    std::uint32_t row_num_datums;
    std::uint32_t full_ct_dim = 1;
    std::optional<std::uint32_t> block_ct_dim = std::nullopt;
    std::optional<Tile> output_tile = std::nullopt;
    tt::DataFormat format = kNarrowRowFormat;
    std::optional<NarrowRowGuardExecute> execute = std::nullopt;
};

// Runs the guard program for the case. Returns true when the program completed; false when the wait aborted
// because the watcher reported an error (the only exception tolerated), leaving the watcher state for the caller
// to read.
bool run_pack_untilize_narrow_row_guard_kernel(distributed::MeshDevice& mesh, const NarrowRowGuardCase& guard_case) {
    Program program = make_pack_untilize_narrow_row_guard_program(
        guard_case.row_num_datums,
        guard_case.full_ct_dim,
        guard_case.block_ct_dim,
        guard_case.output_tile,
        guard_case.format,
        guard_case.execute);
    distributed::MeshWorkload workload;
    const distributed::MeshCoordinate zero(0, 0);
    workload.add_program(distributed::MeshCoordinateRange(zero, zero), std::move(program));
    auto& cq = mesh.mesh_command_queue();
    try {
        distributed::EnqueueMeshWorkload(cq, workload, false /*blocking*/);
        distributed::Finish(cq);
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Aborting wait due to watcher error"), std::string::npos) << e.what();
        return false;
    }
    return true;
}

std::string wait_for_narrow_row_watcher_exception() {
    constexpr auto timeout = std::chrono::milliseconds(5000);
    constexpr auto poll_interval = std::chrono::milliseconds(100);
    const auto start = std::chrono::steady_clock::now();
    std::string exception;
    while (std::chrono::steady_clock::now() - start < timeout) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        if (!exception.empty()) {
            break;
        }
        std::this_thread::sleep_for(poll_interval);
    }
    return exception;
}

// The case must trip one of the pack untilize init's narrow-row LLK_ASSERTs on TRISC2; `consequence` says what
// an init that accepted it would pack wrong.
void run_pack_untilize_narrow_row_rejects(
    distributed::MeshDevice& mesh, const NarrowRowGuardCase& guard_case, const std::string& consequence) {
    // The program may complete (no guard), so completion is not checked here; the watcher state is.
    run_pack_untilize_narrow_row_guard_kernel(mesh, guard_case);
    const std::string exception = wait_for_narrow_row_watcher_exception();
    ASSERT_FALSE(exception.empty()) << "a narrow row of " << guard_case.row_num_datums << " datums x "
                                    << guard_case.full_ct_dim << " tile columns in blocks of "
                                    << guard_case.block_ct_dim.value_or(guard_case.full_ct_dim)
                                    << " was accepted by the pack untilize init; " << consequence;
    EXPECT_NE(exception.find("tripped an assert"), std::string::npos) << exception;
    EXPECT_NE(exception.find("pack_untilize_narrow_row_guard"), std::string::npos) << exception;
    EXPECT_NE(exception.find("TRISC2"), std::string::npos) << exception;
}

}  // namespace

// Narrow-row untilize of full 32x32 bf16 tiles at the row widths the sharded row-major transpose op uses
// (H % 32 datums per row, the row's tiles in one block), data-checked against the per-tile-column golden: every
// width up to one face whose row byte width (tile columns x datums x 2) is a multiple of 16 matches it. One tile
// column: 8 and 16 datums (the only such even bf16 widths; wider rows are rejected by the init's static_assert).
// Two tile columns in one block: 4, 8, 12 and 16 datums, so that two 8-byte PACRs are shown to land back to back
// in one 16-byte row and two 24-byte PACRs in one 48-byte row (4 or 12 datums alone are rejected, see the assert
// fixture below); four tile columns of 6 datums (12-byte PACRs, a 48-byte row) likewise. Three tile rows exercise
// half-sync bank reuse.
TEST_F(LLKBlackholeSingleCardFixture, TensixCustomPackUntilizeNarrowRowWidths) {
    struct Case {
        std::uint32_t row_num_datums;
        std::uint32_t num_tiles_c;
    };
    for (const auto [row_num_datums, num_tiles_c] :
         std::initializer_list<Case>{{8, 1}, {16, 1}, {4, 2}, {8, 2}, {12, 2}, {16, 2}, {6, 4}}) {
        for (const bool full_sync : {false, true}) {
            SCOPED_TRACE(
                fmt::format("row_num_datums={}, num_tiles_c={}, full_sync={}", row_num_datums, num_tiles_c, full_sync));
            unit_tests::compute::tilize::TestConfig config{
                .dst_full_sync_en = full_sync,
                .explicit_untilize_geometry = true,
                .explicit_untilize_narrow_row = true,
                .explicit_untilize_row_num_datums = row_num_datums,
                .input_single_tile_size = tt::tile_size(kNarrowRowFormat),
                .output_single_tile_size = kNarrowRowDatumBytes * constants::TILE_HEIGHT * row_num_datums,
                .num_tiles_r = 3,
                .num_tiles_c = num_tiles_c,
                .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
                .golden_function = narrow_row_untilize_golden(row_num_datums)};
            unit_tests::compute::tilize::run_single_core_tilize_program(*devices_.at(0), config);
        }
    }
}

// Compile-time contract on Wormhole and Blackhole. Width: a narrow row wider than one face does not compile, one
// face does; 20 bf16 datums (40 bytes) would also fail Blackhole's 16-byte row-width check, 24 datums (48 bytes)
// would pass it and is still packed wrong on both architectures because the per-row PACR reads row_num_datums
// consecutive datums of one face; 0 datums would wrap the packer X-end (row_num_datums - 1) and pass the 16-byte
// check, so it does not compile either. Blocks: Blackhole packs a row split over several blocks only at a full face
// width (the wrapper's block column offset is in face-column widths), Wormhole packs exactly one tile per call.
// Dest offset: Wormhole's narrow path packs dest tile 0 only, so a nonzero tile_dst_ct_offset does not compile
// there (Blackhole honours it). Each rejected case is pinned to its guard: the JIT build failure carries the
// compiler log, so the static_assert message of the width, block or dest-offset limit must be in the exception
// text. Quasar's compute API rejects narrow_row itself, for every width.
TEST_F(LLKMeshDeviceSingleCardFixture, PackUntilizeNarrowRowRejectsRowsWiderThanAFaceAtCompileTime) {
    const auto arch = detail::detect_arch();
    if (arch == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar has no narrow-row pack_untilize";
    }
    struct Case {
        std::uint32_t row_num_datums;
        std::uint32_t block_ct_dim;
        std::uint32_t full_ct_dim;
        std::uint32_t tile_dst_ct_offset = 0;
    };
    for (const auto [row_num_datums, block_ct_dim, full_ct_dim, tile_dst_ct_offset] : std::initializer_list<Case>{
             {0, 1, 1}, {16, 1, 1}, {20, 1, 1}, {24, 1, 1}, {8, 1, 2}, {12, 1, 2}, {16, 1, 2}, {16, 1, 1, 1}}) {
        SCOPED_TRACE(fmt::format(
            "row_num_datums={}, block_ct_dim={}, full_ct_dim={}, tile_dst_ct_offset={}",
            row_num_datums,
            block_ct_dim,
            full_ct_dim,
            tile_dst_ct_offset));
        const bool one_face_wide = row_num_datums >= 1 && row_num_datums <= constants::FACE_WIDTH;
        const bool blocks_accepted = arch == tt::ARCH::BLACKHOLE
                                         ? (block_ct_dim == full_ct_dim || row_num_datums == constants::FACE_WIDTH)
                                         : (block_ct_dim == 1 && full_ct_dim == 1);
        const bool dest_offset_accepted = arch == tt::ARCH::BLACKHOLE || tile_dst_ct_offset == 0;
        // The init's width limit is checked first, then its block limit; the dest offset is checked in the execute.
        const std::string expected_guard =
            !one_face_wide     ? "row_num_datums must be between 1 and FACE_C_DIM for narrow_row packing"
            : !blocks_accepted ? (arch == tt::ARCH::BLACKHOLE
                                      ? "narrow_row packing with several blocks requires row_num_datums == FACE_C_DIM"
                                      : "narrow_row packing requires block_ct_dim == full_ct_dim == 1")
                               : "narrow_row packing packs dest tile 0: tile_dst_ct_offset must be 0";
        // The dest-offset case compiles the execute as well (never launched); the others stay init-only.
        Program program = make_pack_untilize_narrow_row_guard_program(
            row_num_datums,
            full_ct_dim,
            block_ct_dim,
            std::nullopt,
            kNarrowRowFormat,
            tile_dst_ct_offset != 0 ? std::optional{NarrowRowGuardExecute{.tile_dst_ct_offset = tile_dst_ct_offset}}
                                    : std::nullopt);
        if (one_face_wide && blocks_accepted && dest_offset_accepted) {
            EXPECT_NO_THROW(program.impl().compile(&this->device()));
            continue;
        }
        try {
            program.impl().compile(&this->device());
            ADD_FAILURE() << "compiled although the narrow-row init should reject it (" << expected_guard << ")";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(expected_guard), std::string::npos)
                << "the build failed for another reason than the expected guard: " << e.what();
        }
    }
}

// Opens one device of the given architecture per test with the watcher and LLK asserts enabled, so a device-side
// LLK_ASSERT surfaces as a watcher exception instead of a silent hang; skips on every other architecture. Mirrors
// MeshWatcherFixture (tests/tt_metal/tt_metal/debug_tools/debug_tools_fixture.hpp); the options must be set
// before the device opens, and a tripped assert parks a TRISC, so the device is reopened for every test.
template <tt::ARCH Arch>
class LLKPackUntilizeAssertFixture : public MeshDispatchFixture {
    static_assert(Arch == tt::ARCH::BLACKHOLE || Arch == tt::ARCH::WORMHOLE_B0, "narrow_row exists on these two");

public:
    // No suite-shared device: the options below change the firmware build, so the device must open after
    // they are set, once per test.
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

protected:
    void SetUp() override {
        // Capture before any early exit: TearDown still runs after a skip or a failed assertion here, and
        // must restore the real previous values, not the zero-initialised members.
        auto& rtoptions = MetalContext::instance().rtoptions();
        watcher_previous_enabled_ = rtoptions.get_watcher_enabled();
        watcher_previous_interval_ = rtoptions.get_watcher_interval();
        test_mode_previous_ = rtoptions.get_test_mode_enabled();
        llk_asserts_previous_ = rtoptions.get_llk_asserts();

        if (detail::detect_arch() != Arch) {
            GTEST_SKIP() << "This test can only be run on " << (Arch == tt::ARCH::BLACKHOLE ? "Blackhole" : "Wormhole")
                         << " cards";
        }
        rtoptions.set_watcher_enabled(true);
        rtoptions.set_watcher_interval(kNarrowRowWatcherIntervalMs);
        // In test mode a watcher error stops the server and fails the wait instead of ending the process.
        rtoptions.set_test_mode_enabled(true);
        rtoptions.set_llk_asserts(true);

        const ChipId mmio_device_id = *MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        auto id_to_device = distributed::MeshDevice::create_unit_meshes(
            {mmio_device_id},
            l1_small_size_,
            trace_region_size_,
            1 /*num_command_queues*/,
            MetalContext::instance().resolve_dispatch_core_config());
        devices_ = {id_to_device.at(mmio_device_id)};
        arch_ = Arch;
        MetalContext::instance().watcher_server()->clear_log();
    }

    void TearDown() override {
        for (auto& device : devices_) {
            device->close();
        }
        devices_.clear();
        auto& rtoptions = MetalContext::instance().rtoptions();
        rtoptions.set_llk_asserts(llk_asserts_previous_);
        rtoptions.set_test_mode_enabled(test_mode_previous_);
        rtoptions.set_watcher_interval(watcher_previous_interval_);
        rtoptions.set_watcher_enabled(watcher_previous_enabled_);
    }

private:
    bool watcher_previous_enabled_{};
    int watcher_previous_interval_{};
    bool test_mode_previous_{};
    bool llk_asserts_previous_{};
};

using LLKBlackholePackUntilizeAssertFixture = LLKPackUntilizeAssertFixture<tt::ARCH::BLACKHOLE>;
using LLKWormholePackUntilizeAssertFixture = LLKPackUntilizeAssertFixture<tt::ARCH::WORMHOLE_B0>;

namespace {

// Runs the init-only kernel for an accepted case and checks that the program completed with a clean watcher.
void run_pack_untilize_narrow_row_accepts(distributed::MeshDevice& mesh, const NarrowRowGuardCase& guard_case) {
    ASSERT_TRUE(run_pack_untilize_narrow_row_guard_kernel(mesh, guard_case))
        << "the narrow-row init on " << guard_case.row_num_datums << " bf16 datums x " << guard_case.full_ct_dim
        << " tile columns did not complete: " << MetalContext::instance().watcher_server()->exception_message();
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollsBeforeCleanCheck * kNarrowRowWatcherIntervalMs));
    EXPECT_TRUE(MetalContext::instance().watcher_server()->exception_message().empty())
        << MetalContext::instance().watcher_server()->exception_message();
}

}  // namespace

namespace {

// A row byte width that is not a multiple of 16 overlaps on Blackhole: the per-row L1 advance is in 16-byte units.
const std::string kSixteenByteRowConsequence =
    "the packer advances its L1 destination in 16-byte units per row, so these rows overlap";
// Face geometries for the guard kernel's output CB.
const Tile kOneFaceTile = Tile::from_face_grid(1, 1, {constants::FACE_HEIGHT, constants::FACE_WIDTH});
const Tile kShortOneFaceTile = Tile::from_face_grid(1, 1, {constants::FACE_HEIGHT / 2, constants::FACE_WIDTH});
const Tile kTwoFaceTile = Tile::from_face_grid(1, 2, {constants::FACE_HEIGHT, constants::FACE_WIDTH});
// Two faces stacked: a 32x16 narrow tile, whose packer dest offsets differ from the 16x32 two-face tile's.
const Tile kTallTwoFaceTile = Tile::from_face_grid(2, 1, {constants::FACE_HEIGHT, constants::FACE_WIDTH});
const Tile kShortFourFaceTile = Tile::from_face_grid(2, 2, {constants::FACE_HEIGHT / 2, constants::FACE_WIDTH});

}  // namespace

// Positive control: a 16-byte row (8 bf16 datums) passes the init with a clean watcher.
TEST_F(LLKBlackholePackUntilizeAssertFixture, PackUntilizeNarrowRowAcceptsSixteenByteRows) {
    run_pack_untilize_narrow_row_accepts(*devices_.at(0), {.row_num_datums = 8});
}

// The row is full_ct_dim tiles wide: 4 bf16 datums (8 bytes) per tile across two tile columns, packed as one
// block, is a 16-byte row and passes, although the same 4 datums in one column (below) are rejected. The data
// check for this layout is TensixCustomPackUntilizeNarrowRowWidths.
TEST_F(LLKBlackholePackUntilizeAssertFixture, PackUntilizeNarrowRowAcceptsEightByteTilesInTwoColumns) {
    run_pack_untilize_narrow_row_accepts(*devices_.at(0), {.row_num_datums = 4, .full_ct_dim = 2});
}

// 4 bf16 datums = 8 bytes: the per-row advance floors to 0 and every row lands on the same 16 bytes.
TEST_F(LLKBlackholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsEightByteRows) {
    run_pack_untilize_narrow_row_rejects(*devices_.at(0), {.row_num_datums = 4}, kSixteenByteRowConsequence);
}

// 12 bf16 datums = 24 bytes: the per-row advance floors to 16 and each row overwrites the previous row's tail.
TEST_F(LLKBlackholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsTwentyFourByteRows) {
    run_pack_untilize_narrow_row_rejects(*devices_.at(0), {.row_num_datums = 12}, kSixteenByteRowConsequence);
}

// The row-width check does not depend on the face geometry: 8-byte rows of a short one-face tile (8x16) are
// rejected as well.
TEST_F(LLKBlackholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsEightByteRowsOfShortOneFaceTiles) {
    run_pack_untilize_narrow_row_rejects(
        *devices_.at(0), {.row_num_datums = 4, .output_tile = kShortOneFaceTile}, kSixteenByteRowConsequence);
}

// A full-face-width row split over two blocks of one tile is accepted for one-face tiles: the wrapper's block
// column offset (one face-column width per tile) then equals the narrow block width.
TEST_F(LLKBlackholePackUntilizeAssertFixture, PackUntilizeNarrowRowAcceptsOneFaceTilesInTwoBlocks) {
    run_pack_untilize_narrow_row_accepts(
        *devices_.at(0), {.row_num_datums = 16, .full_ct_dim = 2, .block_ct_dim = 1, .output_tile = kOneFaceTile});
}

// The same split with two-face tiles trips the init's num_faces assert: the block column offset is then two
// face-column widths per tile, so the second block would land one face column too far right.
TEST_F(LLKBlackholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsTwoFaceTilesInTwoBlocks) {
    run_pack_untilize_narrow_row_rejects(
        *devices_.at(0),
        {.row_num_datums = 16, .full_ct_dim = 2, .block_ct_dim = 1, .output_tile = kTwoFaceTile},
        "the wrapper's block column offset is two face-column widths per two-face tile, so the second block lands "
        "one face column too far right");
}

// Wormhole packs a narrow row only for one full-height four-face tile per call (see the Wormhole init's @note):
// the default 32x32 output tile passes with a clean watcher; a two-face tile or a short four-face tile trips the
// init's face assert. The rejected two-face tile is the 32x16 one, which really loses rows (packer 1 starts at
// face 1); the init rejects the 16x32 one as well because num_faces alone cannot tell them apart.
TEST_F(LLKWormholePackUntilizeAssertFixture, PackUntilizeNarrowRowAcceptsOneFourFaceTile) {
    run_pack_untilize_narrow_row_accepts(*devices_.at(0), {.row_num_datums = 16});
}

TEST_F(LLKWormholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsTwoFaceTiles) {
    run_pack_untilize_narrow_row_rejects(
        *devices_.at(0),
        {.row_num_datums = 16, .output_tile = kTallTwoFaceTile},
        "a 32x16 two-face tile is a narrow tile whose second packer starts at face 1, so rows 8-15 of each face are "
        "never packed");
}

TEST_F(LLKWormholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsShortFaces) {
    run_pack_untilize_narrow_row_rejects(
        *devices_.at(0),
        {.row_num_datums = 16, .output_tile = kShortFourFaceTile},
        "the narrow MOP packs TILE_R_DIM / 4 rows per packer, so an 8-row face leaves rows unpacked");
}

// Wormhole places packer k at k times the bytes of the eight rows a packer writes, floored to 16 bytes: 5 Int8
// datums put packer 1 at 32 bytes while packer 0 writes 40, so the init rejects a 1-byte row that is not a
// multiple of 16 bytes over eight rows (bf16 rows always are).
TEST_F(LLKWormholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsOddOneByteRows) {
    run_pack_untilize_narrow_row_rejects(
        *devices_.at(0),
        {.row_num_datums = 5, .format = tt::DataFormat::Int8},
        "the packers' start addresses are 16-byte granular, so packer 1 would start inside packer 0's 40 bytes");
}

// Wormhole's narrow MOP never advances the dest tile, so the execute rejects a second row block (wrapper) and a
// nonzero runtime dest row offset (LLK) before any PACR is issued; both would re-pack dest tile 0.
TEST_F(LLKWormholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsSecondRowBlock) {
    run_pack_untilize_narrow_row_rejects(
        *devices_.at(0),
        {.row_num_datums = 16, .execute = NarrowRowGuardExecute{.block_rt_dim = 2}},
        "the narrow MOP does not advance the dest tile, so the second row block would re-pack dest tile 0");
}

TEST_F(LLKWormholePackUntilizeAssertFixture, PackUntilizeNarrowRowRejectsDestRowOffset) {
    run_pack_untilize_narrow_row_rejects(
        *devices_.at(0),
        {.row_num_datums = 16, .execute = NarrowRowGuardExecute{.tile_dst_rt_offset = 1}},
        "the narrow MOP runs without set_dst_write_addr, so a runtime dest row offset is ignored and tile 0 packed");
}

// ============================================================================
// Id-free (2.0) tilize / pack_untilize / pack_untilize_dest, validated against the gold_standard_tilize /
// gold_standard_untilize host goldens (exact, bitwise), not a legacy kernel. Single Float16_b tile. Runs on
// Blackhole (BH-only API). tilize: row-major c_0 -> tiled c_16; (pack_)untilize: tiled c_0 -> row-major c_16.
// ============================================================================
TEST_F(LLKBlackholeSingleCardFixture, TensixTilizeIdFreeGolden) {
    constexpr std::uint32_t num_tiles = 1;
    auto src = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result = unit_tests::llk::single_core::run_unary(
        this->device(),
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/tilize_2_0.cpp");

    ::unit_tests::compute::GoldenConfig config{
        .num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .face_r_dim = 16, .face_c_dim = 16, .num_faces = 4};
    auto golden = ::unit_tests::compute::gold_standard_tilize(src, config);
    EXPECT_EQ(golden, result);
}

TEST_F(LLKBlackholeSingleCardFixture, TensixPackUntilizeIdFreeGolden) {
    constexpr std::uint32_t num_tiles = 1;
    auto src = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result = unit_tests::llk::single_core::run_unary(
        this->device(),
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_2_0.cpp");

    ::unit_tests::compute::GoldenConfig config{
        .num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .face_r_dim = 16, .face_c_dim = 16, .num_faces = 4};
    auto golden = ::unit_tests::compute::gold_standard_untilize(src, config);
    EXPECT_EQ(golden, result);
}

TEST_F(LLKBlackholeSingleCardFixture, TensixPackUntilizeDestIdFreeGolden) {
    constexpr std::uint32_t num_tiles = 1;
    auto src = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result = unit_tests::llk::single_core::run_unary(
        this->device(),
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_dest_2_0.cpp");

    ::unit_tests::compute::GoldenConfig config{
        .num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .face_r_dim = 16, .face_c_dim = 16, .num_faces = 4};
    auto golden = ::unit_tests::compute::gold_standard_untilize(src, config);
    EXPECT_EQ(golden, result);
}

// Id-free (2.0) pack_untilize with block_ct_dim = 4 (> 1). This executes pack_untilize_block4_2_0.cpp -- the
// only 2.0 kernel exercising the block>1 window: it reads 4 tiled input tiles in ONE pack_untilize_block<4,4>
// call at in.l1_address + c * tile_stride_words(...) and writes one row-major row 4 tiles wide at the
// full_ct_dim * tile_stride_words(...) output stride. Validated against gold_standard_untilize for a 1x4 tile
// block (exact). (The partial block-float per-tile stride is covered separately by the copy_block Bfp8 partial-
// tile test in test_copy_block_matmul_partials.cpp.)
TEST_F(LLKBlackholeSingleCardFixture, TensixPackUntilizeBlock4IdFreeGolden) {
    constexpr std::uint32_t num_tiles = 4;  // one 1x4 pack_untilize_block<4,4> window
    auto src = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result = unit_tests::llk::single_core::run_unary(
        this->device(),
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_block4_2_0.cpp",
        /*cb_depth_tiles=*/num_tiles);

    ::unit_tests::compute::GoldenConfig config{
        .num_tiles_r_dim = 1, .num_tiles_c_dim = 4, .face_r_dim = 16, .face_c_dim = 16, .num_faces = 4};
    auto golden = ::unit_tests::compute::gold_standard_untilize(src, config);
    EXPECT_EQ(golden, result);
}

// NOTE: no distinguishing test for tilize_block's restored input_tile_index / output_tile_index params.
// The restored parameters default to 0, in which case tile_index folds to `t` and the generated code is
// byte-identical to before (covered by TensixTilizeIdFreeGolden above). Observing a NONZERO index cleanly
// would require a block-configured, wide row-major tilize harness: column-tile b of an N-wide row-major
// input sits at a b*TILE_WIDTH offset with an N*TILE_WIDTH per-row stride, which only tilizes correctly when
// the unpack MOP is configured for block=N (as legacy's single llk_unpack_tilize_block(base, block, index)
// call is). The id-free tilize_block uses a per-tile llk_unpack_tilize loop with block=1 init, so an
// input_tile_index-addressed nonzero column tile reads the input as contiguous standalone tiles instead of a
// wide block and cannot match gold_standard_tilize. The restored params are correct-for-parity but not
// cleanly device-observable via this path (same resolution as the D0/D1/Copilot#2 unobservable fixes).

}  // namespace tt::tt_metal
