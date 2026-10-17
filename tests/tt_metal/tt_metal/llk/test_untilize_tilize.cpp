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
#include <algorithm>
#include <array>
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
#include <tt-metalium/constants.hpp>
#include <tt-metalium/float8.hpp>
#include <tt-metalium/int8.hpp>
#include <tt-metalium/uint8.hpp>
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
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include <umd/device/types/arch.hpp>
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
    // Compute kernel to run instead of the one untilize_type / tilize_type selects; the buffer setup and
    // the defines of that type still apply, plus compute_defines.
    std::string compute_kernel;
    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
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
    if (!test_config.compute_kernel.empty()) {
        compute_kernel = test_config.compute_kernel;
    } else if (test_config.untilize_type.has_value()) {
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
    }
    for (const auto& define : test_config.compute_defines) {
        compute_defines.insert(define);
    }
    if (test_config.fp32_dest_acc_en) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
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
    Program program = tt::tt_metal::CreateProgram();

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
    if (test_config.fp32_dest_acc_en) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }

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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
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

namespace {

// Row-major bf16 input of num_tiles_r x num_tiles_c 32x32 tiles where every datum carries its face: within a
// tile, face f (0..3, row-major over the 2x2 face grid) holds 16*f + (row % 16) + 0.25 * ((col % 16) / 4). All
// values are exact in bf16 (<= 63.75, eight significant bits), the four faces occupy disjoint value ranges, and
// a face packed into the wrong slot is identified by its value range alone.
std::vector<std::uint32_t> make_face_tagged_rowmajor_bf16(std::uint32_t num_tiles_r, std::uint32_t num_tiles_c) {
    constexpr std::uint32_t tile_dim = constants::TILE_HEIGHT;
    constexpr std::uint32_t face_dim = constants::FACE_HEIGHT;
    static_assert(constants::TILE_WIDTH == tile_dim && constants::FACE_WIDTH == face_dim, "square tiles and faces");
    const std::uint32_t rows = num_tiles_r * tile_dim;
    const std::uint32_t cols = num_tiles_c * tile_dim;
    std::vector<bfloat16> src(rows * cols);
    for (std::uint32_t row = 0; row < rows; row++) {
        for (std::uint32_t col = 0; col < cols; col++) {
            const std::uint32_t r = row % tile_dim;
            const std::uint32_t c = col % tile_dim;
            const std::uint32_t face = (r / face_dim) * 2 + (c / face_dim);
            const float value =
                static_cast<float>(16 * face + (r % face_dim)) + 0.25f * static_cast<float>((c % face_dim) / 4);
            src[row * cols + col] = bfloat16(value);
        }
    }
    return pack_vector<std::uint32_t, bfloat16>(src);
}

// For the first output tile: the value found at the start of each 16x16 face slot (a face-tagged input makes
// 16*f the expected value of slot f) and the total mismatch count, for the failure message.
std::string describe_tilize_face_slots(
    const std::vector<std::uint32_t>& result, const std::vector<std::uint32_t>& golden) {
    const auto result_bf16 = unpack_vector<bfloat16, std::uint32_t>(result);
    const auto golden_bf16 = unpack_vector<bfloat16, std::uint32_t>(golden);
    std::uint32_t mismatches = 0;
    for (std::size_t i = 0; i < std::min(result_bf16.size(), golden_bf16.size()); i++) {
        mismatches += static_cast<float>(result_bf16[i]) != static_cast<float>(golden_bf16[i]);
    }
    std::string msg = fmt::format("{} mismatching datums; tile 0 face slots (result/golden first datum):", mismatches);
    constexpr std::size_t face_datums = constants::FACE_HW;
    for (std::size_t face = 0; face < 4 && (face + 1) * face_datums <= result_bf16.size(); face++) {
        msg += fmt::format(
            " slot{}={}/{}",
            face,
            static_cast<float>(result_bf16[face * face_datums]),
            static_cast<float>(golden_bf16[face * face_datums]));
    }
    return msg;
}

// Number of result words that mismatch the golden, with the first mismatching word, for the failure message.
std::string describe_word_mismatch(const std::vector<std::uint32_t>& result, const std::vector<std::uint32_t>& golden) {
    std::size_t mismatches = 0;
    std::optional<std::size_t> first;
    for (std::size_t i = 0; i < std::min(result.size(), golden.size()); i++) {
        if (result[i] != golden[i]) {
            mismatches++;
            if (!first.has_value()) {
                first = i;
            }
        }
    }
    std::string msg = fmt::format("{} of {} words mismatch", mismatches, golden.size());
    if (first.has_value()) {
        msg += fmt::format(" (first at word {}: {:#x} vs golden {:#x})", *first, result[*first], golden[*first]);
    }
    return msg;
}

// Words per block row of full 32x32 tiles for the golden config's datum size.
std::size_t block_row_words(const ::unit_tests::compute::GoldenConfig& config) {
    return static_cast<std::size_t>(config.num_tiles_c_dim) * constants::TILE_HW * config.datum_bytes /
           sizeof(std::uint32_t);
}

// Golden for fast_tilize_reconfig_then_pack.cpp / fast_untilize_reconfig_then_pack.cpp: every block row but the
// last transformed by gold_fn (tilize / untilize), the last one passed through the Default pack unchanged.
std::vector<std::uint32_t> gold_rows_then_identity(
    const std::vector<std::uint32_t>& src,
    const ::unit_tests::compute::GoldenConfig& config,
    std::vector<std::uint32_t> (*gold_fn)(
        const std::vector<std::uint32_t>&, const ::unit_tests::compute::GoldenConfig&)) {
    auto head = config;
    head.num_tiles_r_dim = config.num_tiles_r_dim - 1;
    const auto split = src.begin() + static_cast<std::ptrdiff_t>(head.num_tiles_r_dim * block_row_words(config));
    auto golden = gold_fn(std::vector<std::uint32_t>(src.begin(), split), head);
    golden.insert(golden.end(), split, src.end());
    return golden;
}

// The first row_num_datums datums of face 0 of every input tile, laid out as num_tiles_r blocks of face_r_dim rows
// of (num_tiles_c * row_num_datums) datums: the narrow-row pack untilize of 32x32 bf16 input tiles into a one-face
// output (pack_untilize_narrow_row_reconfig.cpp).
std::vector<std::uint32_t> gold_narrow_row_untilize_face0(
    const std::vector<std::uint32_t>& src,
    std::uint32_t num_tiles_r,
    std::uint32_t num_tiles_c,
    std::uint32_t row_num_datums) {
    const auto src_bf16 = unpack_vector<bfloat16, std::uint32_t>(src);
    std::vector<bfloat16> golden;
    for (std::uint32_t block = 0; block < num_tiles_r; block++) {
        for (std::uint32_t row = 0; row < constants::FACE_HEIGHT; row++) {
            for (std::uint32_t tile = 0; tile < num_tiles_c; tile++) {
                const std::size_t face0 = static_cast<std::size_t>(block * num_tiles_c + tile) * constants::TILE_HW;
                for (std::uint32_t col = 0; col < row_num_datums; col++) {
                    golden.push_back(src_bf16[face0 + row * constants::FACE_WIDTH + col]);
                }
            }
        }
    }
    return pack_vector<std::uint32_t, bfloat16>(golden);
}

}  // namespace

// Blackhole: tilize with pack_reconfig_data_format between tilize_init and tilize_block, which the reconfig API
// supports (the reconfig keeps the two-face Z layout tilize_init armed). tilize_init arms the packer for a scratch
// output CB of a different format (c_17, never written), the reconfig switches it to the real bf16 output, and the
// block is tilized into it. The Blackhole tilize pack MOP steps the dest Z counter by two faces per increment, so
// a reconfig that reprograms the strides in Default mode (one face) makes the second half of every output tile
// hold faces 1,2 instead of 2,3. Exact bitwise golden.
TEST_F(LLKBlackholeSingleCardFixture, TensixComputeTilizeAfterPackReconfig) {
    constexpr std::uint32_t dram_bank_id = 0;
    const std::uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t scratch_tile_bytes = tt::tile_size(tt::DataFormat::Bfp8_b);

    for (const auto& [num_tiles_r, num_tiles_c] : {std::pair<std::uint32_t, std::uint32_t>{1, 1}, {1, 2}, {2, 2}}) {
        for (const bool dst_full_sync_en : {false, true}) {
            SCOPED_TRACE(fmt::format("{}x{} tiles, dst_full_sync_en={}", num_tiles_r, num_tiles_c, dst_full_sync_en));
            auto& mesh_device = this->device();
            auto& cq = mesh_device.mesh_command_queue();
            Program program = CreateProgram();
            const CoreCoord core = {0, 0};
            const std::uint32_t num_tiles = num_tiles_r * num_tiles_c;
            const std::uint32_t buffer_bytes = num_tiles * tile_bytes;

            auto make_buffer = [&]() {
                return distributed::MeshBuffer::create(
                    distributed::ReplicatedBufferConfig{.size = buffer_bytes},
                    {.page_size = buffer_bytes, .buffer_type = BufferType::DRAM},
                    &mesh_device);
            };
            auto src = make_buffer();
            auto dst = make_buffer();

            for (const auto cb : {tt::CBIndex::c_0, tt::CBIndex::c_16}) {
                CreateCircularBuffer(
                    program,
                    core,
                    CircularBufferConfig(buffer_bytes, {{cb, tt::DataFormat::Float16_b}})
                        .set_page_size(cb, tile_bytes));
            }
            // Scratch output the packer is initially configured for; a different format and tile size.
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(scratch_tile_bytes, {{tt::CBIndex::c_17, tt::DataFormat::Bfp8_b}})
                    .set_page_size(tt::CBIndex::c_17, scratch_tile_bytes));

            auto reader = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto writer = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/tilize_pack_reconfig.cpp",
                core,
                ComputeConfig{.dst_full_sync_en = dst_full_sync_en, .compile_args = {num_tiles_r, num_tiles_c}});

            const auto src_vec = make_face_tagged_rowmajor_bf16(num_tiles_r, num_tiles_c);
            distributed::EnqueueWriteMeshBuffer(cq, src, src_vec, /*blocking=*/true);
            SetRuntimeArgs(
                program,
                reader,
                core,
                {src->address(),
                 dram_bank_id,
                 num_tiles,
                 static_cast<std::uint32_t>(tt::CBIndex::c_0),
                 num_tiles_c /* ublock_size_tiles */,
                 0u /* reader_only */});
            SetRuntimeArgs(program, writer, core, {dst->address(), dram_bank_id, num_tiles});
            LaunchProgram(mesh_device, std::move(program));

            std::vector<std::uint32_t> result;
            distributed::EnqueueReadMeshBuffer(cq, result, dst, /*blocking=*/true);

            const ::unit_tests::compute::GoldenConfig config{
                .num_tiles_r_dim = static_cast<int>(num_tiles_r), .num_tiles_c_dim = static_cast<int>(num_tiles_c)};
            const auto golden = ::unit_tests::compute::gold_standard_tilize(src_vec, config);
            ASSERT_EQ(result.size(), golden.size());
            EXPECT_TRUE(result == golden) << describe_tilize_face_slots(result, golden);
        }
    }
}

// Id-free (2.0) counterpart: tilize_init for a scratch Bfp8_b output operand, pack_reconfig_data_format to the real
// bf16 output, tilize_block into it (tilize_pack_reconfig_2_0.cpp). Same face-tagged input and exact golden.
TEST_F(LLKBlackholeSingleCardFixture, TensixTilizeAfterPackReconfigIdFreeGolden) {
    constexpr std::uint32_t num_tiles = 1;
    const auto src = make_face_tagged_rowmajor_bf16(num_tiles, num_tiles);
    auto result = unit_tests::llk::single_core::run_unary(
        this->device(),
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/tilize_pack_reconfig_2_0.cpp");

    const ::unit_tests::compute::GoldenConfig config{
        .num_tiles_r_dim = static_cast<int>(num_tiles), .num_tiles_c_dim = static_cast<int>(num_tiles)};
    const auto golden = ::unit_tests::compute::gold_standard_tilize(src, config);
    ASSERT_EQ(result.size(), golden.size());
    EXPECT_TRUE(result == golden) << describe_tilize_face_slots(result, golden);
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

// Blackhole fast tilize with a same-format pack_reconfig_data_format between fast_tilize_init and the blocks, then
// fast_tilize_uninit and a Default pack of the last block row (fast_tilize_reconfig_then_pack.cpp). The fast-tilize
// init programs its own packer strides; a reconfig that re-derives the strides from the pack mode instead of
// leaving them scrambles every fast-tilized tile. fp32 dest accumulation off (with it on, a reconfig between the
// fast-tilize init and its execute is not supported: it re-enables the 32-bit DEST reads the init disables).
// Exact golden.
TEST_F(LLKBlackholeSingleCardFixture, TensixFastTilizeReconfigBetweenInitAndExecuteThenPack) {
    for (const auto& [num_tiles_r, num_tiles_c] : {std::pair<std::uint32_t, std::uint32_t>{2, 2}, {2, 4}, {3, 3}}) {
        SCOPED_TRACE(fmt::format("{}x{} tiles", num_tiles_r, num_tiles_c));
        unit_tests::compute::tilize::TestConfig config{
            .dst_full_sync_en = false,
            .fp32_dest_acc_en = false,
            .input_single_tile_size = 2 * 1024,
            .output_single_tile_size = 2 * 1024,
            .num_tiles_r = num_tiles_r,
            .num_tiles_c = num_tiles_c,
            .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
            .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/fast_tilize_reconfig_then_pack.cpp",
            .golden_function = [](const std::vector<std::uint32_t>& src,
                                  const ::unit_tests::compute::GoldenConfig& config) {
                return gold_rows_then_identity(src, config, ::unit_tests::compute::gold_standard_tilize);
            }};
        unit_tests::compute::tilize::run_single_core_tilize_program(*devices_.at(0), config);
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
            Program program = CreateProgram();
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
    bool tilize_cross_tile_rows = false) {
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

// Blackhole fast untilize with a same-format pack_reconfig_data_format between fast_untilize_init and the blocks,
// then fast_untilize_uninit and a Default pack of the last block row (fast_untilize_reconfig_then_pack.cpp). The
// fast-untilize init programs its own packer strides, including the channel-1 row stride of the strided path
// (width above one 4-tile chunk); a reconfig that re-derives the strides from the pack mode instead of leaving
// them scrambles the output. Widths 4 (contiguous), 6 and 8 (strided). Exact golden.
TEST_F(LLKBlackholeSingleCardFixture, TensixFastUntilizeReconfigBetweenInitAndExecuteThenPack) {
    for (const std::uint32_t num_tiles_c : {4u, 6u, 8u}) {
        SCOPED_TRACE(fmt::format("2x{} tiles", num_tiles_c));
        unit_tests::compute::tilize::TestConfig config{
            .dst_full_sync_en = false,
            .fp32_dest_acc_en = false,
            .input_single_tile_size = 2 * 1024,
            .output_single_tile_size = 2 * 1024,
            .num_tiles_r = 2,
            .num_tiles_c = num_tiles_c,
            .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
            .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/fast_untilize_reconfig_then_pack.cpp",
            .golden_function = [](const std::vector<std::uint32_t>& src,
                                  const ::unit_tests::compute::GoldenConfig& config) {
                return gold_rows_then_identity(src, config, ::unit_tests::compute::gold_standard_untilize);
            }};
        unit_tests::compute::tilize::run_single_core_tilize_program(*devices_.at(0), config);
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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
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
            unit_tests::compute::tilize::run_single_core_tilize_program(this->device(), test_config);
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

// Blackhole: pack_reconfig_data_format between pack_untilize_dest_init and pack_untilize_dest
// (pack_untilize_reconfig_after_tilize_init.cpp). The pack-untilize init programs the dest Z stride that steps
// the untilize MOP from a tile's top face row to its bottom one; the reconfig must leave it. Exact golden on
// 32x32 tiles, with and without a preceding tilize_init that no Default pack init follows: a reconfig that
// resets the strides to the Default layout (one face) fails both variants, one that re-derives them for the
// armed pack mode fails the plain variant only.
TEST_F(LLKBlackholeSingleCardFixture, TensixPackUntilizeReconfigBetweenInitAndExecuteKeepsStride) {
    for (const bool tilize_init_first : {false, true}) {
        for (const bool full_sync : {false, true}) {
            SCOPED_TRACE(fmt::format("tilize_init_first={}, full_sync={}", tilize_init_first, full_sync));
            unit_tests::compute::tilize::TestConfig config{
                .dst_full_sync_en = full_sync,
                .input_single_tile_size = 2048,
                .output_single_tile_size = 2048,
                .num_tiles_r = 2,
                .num_tiles_c = 4,
                .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
                .compute_kernel =
                    "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_reconfig_after_tilize_init.cpp",
                .compute_defines = {{"TILIZE_INIT_FIRST", tilize_init_first ? "1" : "0"}},
                .golden_function = ::unit_tests::compute::gold_standard_untilize};
            unit_tests::compute::tilize::run_single_core_tilize_program(*devices_.at(0), config);
        }
    }
}

// Blackhole: pack-untilize init hand-off without an uninit (pack_untilize_init_handoff.cpp). pack_untilize_init on a
// Float16_b scratch output (c_17, never written), then pack_untilize_dest_init on the real Float32 output, under
// fp32 dest accumulation, so the pack source datum size goes from 2 to 4 bytes between the two inits. The second
// init must program its own Y/W strides for the 4-byte source: an init that programs only Z and leaves Y/W to the
// data-format reconfig in front of it keeps the 2-byte strides (the reconfig leaves the strides alone while a
// pack-untilize init is armed) and untilizes the wrong dest rows. Exact golden: the untilized bf16 input widened
// to fp32.
TEST_F(LLKBlackholeSingleCardFixture, TensixPackUntilizeInitHandOffDifferentDatumSize) {
    constexpr std::uint32_t dram_bank_id = 0;
    const std::uint32_t in_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t out_tile_bytes = tt::tile_size(tt::DataFormat::Float32);

    for (const auto& [num_tiles_r, num_tiles_c] : {std::pair<std::uint32_t, std::uint32_t>{1, 1}, {2, 4}}) {
        for (const bool dst_full_sync_en : {false, true}) {
            SCOPED_TRACE(fmt::format("{}x{} tiles, dst_full_sync_en={}", num_tiles_r, num_tiles_c, dst_full_sync_en));
            auto& mesh_device = this->device();
            auto& cq = mesh_device.mesh_command_queue();
            Program program = CreateProgram();
            const CoreCoord core = {0, 0};
            const std::uint32_t num_tiles = num_tiles_r * num_tiles_c;

            auto src = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = num_tiles * in_tile_bytes},
                {.page_size = num_tiles * in_tile_bytes, .buffer_type = BufferType::DRAM},
                &mesh_device);
            auto dst = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = num_tiles * out_tile_bytes},
                {.page_size = num_tiles * out_tile_bytes, .buffer_type = BufferType::DRAM},
                &mesh_device);

            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(num_tiles * in_tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                    .set_page_size(tt::CBIndex::c_0, in_tile_bytes));
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(num_tiles * out_tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float32}})
                    .set_page_size(tt::CBIndex::c_16, out_tile_bytes));
            // Scratch output the first untilize init targets: 2-byte pack source under fp32 dest accumulation.
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(in_tile_bytes, {{tt::CBIndex::c_17, tt::DataFormat::Float16_b}})
                    .set_page_size(tt::CBIndex::c_17, in_tile_bytes));

            auto reader = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto writer = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_init_handoff.cpp",
                core,
                ComputeConfig{
                    .fp32_dest_acc_en = true,
                    .dst_full_sync_en = dst_full_sync_en,
                    .compile_args = {num_tiles_r, num_tiles_c}});

            const auto src_vec = create_arange_vector_of_bfloat16(num_tiles * in_tile_bytes, false);
            distributed::EnqueueWriteMeshBuffer(cq, src, src_vec, /*blocking=*/true);
            SetRuntimeArgs(program, reader, core, {src->address(), dram_bank_id, num_tiles});
            SetRuntimeArgs(program, writer, core, {dst->address(), dram_bank_id, num_tiles});
            LaunchProgram(mesh_device, std::move(program));

            std::vector<std::uint32_t> result;
            distributed::EnqueueReadMeshBuffer(cq, result, dst, /*blocking=*/true);

            const ::unit_tests::compute::GoldenConfig config{
                .num_tiles_r_dim = static_cast<int>(num_tiles_r), .num_tiles_c_dim = static_cast<int>(num_tiles_c)};
            const auto golden_bf16 =
                unpack_vector<bfloat16, std::uint32_t>(::unit_tests::compute::gold_standard_untilize(src_vec, config));
            std::vector<std::uint32_t> golden(golden_bf16.size());
            std::transform(golden_bf16.begin(), golden_bf16.end(), golden.begin(), [](const bfloat16& value) {
                return std::bit_cast<std::uint32_t>(static_cast<float>(value));
            });
            ASSERT_EQ(result.size(), golden.size());
            EXPECT_TRUE(result == golden) << describe_word_mismatch(result, golden);
        }
    }
}

// Blackhole: pack_reconfig_data_format between a narrow-row pack_untilize_dest_init and pack_untilize_dest
// (pack_untilize_narrow_row_reconfig.cpp). The narrow-row init programs the packer X (datum) counter end to
// row_num_datums - 1 and the execute never re-programs it, so the reconfig must leave the counter alone while
// the init is armed; one that resets it to a full face row packs 16 datums per row into 8-datum row slots. The
// output CB is a one-face 16x16 tile; a block row of num_tiles_c input tiles fills num_tiles_c * 8 / 16 of its
// pages with 16 rows of num_tiles_c * 8 datums. Exact golden.
TEST_F(LLKBlackholeSingleCardFixture, TensixPackUntilizeNarrowRowReconfigBetweenInitAndExecute) {
    constexpr std::uint32_t dram_bank_id = 0;
    constexpr std::uint32_t row_num_datums = 8;
    const Tile out_tile({constants::FACE_HEIGHT, constants::FACE_WIDTH});
    const std::uint32_t in_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t out_tile_bytes = out_tile.get_tile_size(tt::DataFormat::Float16_b);

    for (const std::uint32_t num_tiles_c : {2u, 4u}) {
        for (const bool dst_full_sync_en : {false, true}) {
            SCOPED_TRACE(fmt::format("2x{} tiles, dst_full_sync_en={}", num_tiles_c, dst_full_sync_en));
            constexpr std::uint32_t num_tiles_r = 2;
            auto& mesh_device = this->device();
            auto& cq = mesh_device.mesh_command_queue();
            Program program = CreateProgram();
            const CoreCoord core = {0, 0};
            const std::uint32_t num_tiles = num_tiles_r * num_tiles_c;
            const std::uint32_t num_out_pages = num_tiles_r * num_tiles_c * row_num_datums / constants::FACE_WIDTH;

            auto src = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = num_tiles * in_tile_bytes},
                {.page_size = num_tiles * in_tile_bytes, .buffer_type = BufferType::DRAM},
                &mesh_device);
            auto dst = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = num_out_pages * out_tile_bytes},
                {.page_size = num_out_pages * out_tile_bytes, .buffer_type = BufferType::DRAM},
                &mesh_device);

            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(num_tiles * in_tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                    .set_page_size(tt::CBIndex::c_0, in_tile_bytes));
            // One-face output tile; the page is the tile, so the writer and the CB agree on the page size.
            CircularBufferConfig out_cb_config =
                CircularBufferConfig(num_out_pages * out_tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
                    .set_page_size(tt::CBIndex::c_16, out_tile_bytes);
            out_cb_config.set_tile_dims(tt::CBIndex::c_16, out_tile);
            CreateCircularBuffer(program, core, out_cb_config);

            auto reader = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
            auto writer = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_narrow_row_reconfig.cpp",
                core,
                ComputeConfig{
                    .dst_full_sync_en = dst_full_sync_en, .compile_args = {num_tiles_r, num_tiles_c, row_num_datums}});

            const auto src_vec = create_arange_vector_of_bfloat16(num_tiles * in_tile_bytes, false);
            distributed::EnqueueWriteMeshBuffer(cq, src, src_vec, /*blocking=*/true);
            SetRuntimeArgs(program, reader, core, {src->address(), dram_bank_id, num_tiles});
            SetRuntimeArgs(program, writer, core, {dst->address(), dram_bank_id, num_out_pages});
            LaunchProgram(mesh_device, std::move(program));

            std::vector<std::uint32_t> result;
            distributed::EnqueueReadMeshBuffer(cq, result, dst, /*blocking=*/true);

            const auto golden = gold_narrow_row_untilize_face0(src_vec, num_tiles_r, num_tiles_c, row_num_datums);
            ASSERT_EQ(result.size(), golden.size());
            EXPECT_TRUE(result == golden) << describe_word_mismatch(result, golden);
        }
    }
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
