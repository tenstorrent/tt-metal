// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// compute_kernel_lib::reduce() explicit modes on Quasar: TileSum on each bulk path, a partial mask, scale_bits and
// cross-call accumulation, with a Native case as the harness baseline.

#include "common/device_fixture.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/data_format/bfloat16_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE = 32;
constexpr uint32_t TILE_ELEMENTS = TILE * TILE;
constexpr uint32_t BF16_TILE_BYTES = TILE_ELEMENTS * sizeof(uint16_t);
constexpr uint32_t FP32_TILE_BYTES = TILE_ELEMENTS * sizeof(float);

// ttnn::kernel_lib::ReduceAuxiliaryTileType
constexpr uint32_t AUX_FIRST_ROW = 0;
constexpr uint32_t AUX_ZERO = 3;

enum class Dim { Row, Col, Scalar };

struct ReduceCase {
    std::string name;
    Dim dim;
    uint32_t rows;
    uint32_t cols;
    uint32_t batches = 1;
    bool tile_sum = true;
    bool average = false;
    uint32_t partial = 0;  // REDUCE_ROW only: valid columns of the last tile, 0 = full
    uint32_t output_group = 0;
    bool accumulate = false;  // two calls folded through an accumulator DFB
    uint32_t input_entries = 0;
};

uint32_t float_bits(float f) { return std::bit_cast<uint32_t>(f); }

// Element (r, c) of a 32x32 tile stored as four 16x16 faces.
uint32_t tile_offset(uint32_t r, uint32_t c) { return (r / 16 * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16); }

uint32_t valid_width(const ReduceCase& rc) { return rc.partial ? (rc.cols - 1) * TILE + rc.partial : rc.cols * TILE; }

uint32_t reduced_elements(const ReduceCase& rc) {
    const uint32_t per_call = rc.dim == Dim::Row   ? valid_width(rc)
                              : rc.dim == Dim::Col ? rc.rows * TILE
                                                   : rc.rows * rc.cols * TILE_ELEMENTS;
    return per_call * (rc.accumulate ? 2 : 1);
}

uint32_t output_tiles(const ReduceCase& rc) {
    const uint32_t per_batch = rc.dim == Dim::Row ? rc.rows : rc.dim == Dim::Col ? rc.cols : 1;
    return per_batch * rc.batches;
}

// Expected (tile, offset, value) triples of the reduced outputs; other output elements are not checked.
struct Expected {
    uint32_t tile;
    uint32_t offset;
    float value;
};

std::vector<Expected> golden(const ReduceCase& rc, const std::vector<float>& input) {
    const uint32_t block_tiles = rc.rows * rc.cols * rc.batches;
    const uint32_t calls = rc.accumulate ? 2 : 1;
    const float scale = rc.average ? 1.0f / static_cast<float>(reduced_elements(rc)) : 1.0f;
    // BulkWaitBulkPop REDUCE_COL reads each output_group of columns as one row-major [Ht x group] bulk.
    auto tile_in_batch = [&](uint32_t h, uint32_t w) {
        if (rc.dim != Dim::Col) {
            return h * rc.cols + w;
        }
        const uint32_t first = w / rc.output_group * rc.output_group;
        const uint32_t width = std::min(rc.output_group, rc.cols - first);
        return rc.rows * first + h * width + (w - first);
    };
    auto at = [&](uint32_t call, uint32_t b, uint32_t h, uint32_t w, uint32_t r, uint32_t c) {
        const uint32_t tile = call * block_tiles + b * rc.rows * rc.cols + tile_in_batch(h, w);
        return static_cast<double>(input[tile * TILE_ELEMENTS + tile_offset(r, c)]);
    };
    std::vector<Expected> expected;
    for (uint32_t b = 0; b < rc.batches; ++b) {
        if (rc.dim == Dim::Row) {
            for (uint32_t h = 0; h < rc.rows; ++h) {
                for (uint32_t r = 0; r < TILE; ++r) {
                    double sum = 0;
                    for (uint32_t call = 0; call < calls; ++call) {
                        for (uint32_t x = 0; x < valid_width(rc); ++x) {
                            sum += at(call, b, h, x / TILE, r, x % TILE);
                        }
                    }
                    expected.push_back({b * rc.rows + h, tile_offset(r, 0), static_cast<float>(sum * scale)});
                }
            }
        } else if (rc.dim == Dim::Col) {
            for (uint32_t w = 0; w < rc.cols; ++w) {
                for (uint32_t c = 0; c < TILE; ++c) {
                    double sum = 0;
                    for (uint32_t call = 0; call < calls; ++call) {
                        for (uint32_t y = 0; y < rc.rows * TILE; ++y) {
                            sum += at(call, b, y / TILE, w, y % TILE, c);
                        }
                    }
                    expected.push_back({b * rc.cols + w, tile_offset(0, c), static_cast<float>(sum * scale)});
                }
            }
        } else {
            double sum = 0;
            for (uint32_t call = 0; call < calls; ++call) {
                for (uint32_t h = 0; h < rc.rows; ++h) {
                    for (uint32_t w = 0; w < rc.cols; ++w) {
                        for (uint32_t i = 0; i < TILE_ELEMENTS; ++i) {
                            sum += at(call, b, h, w, i / TILE, i % TILE);
                        }
                    }
                }
            }
            expected.push_back({b, tile_offset(0, 0), static_cast<float>(sum * scale)});
        }
    }
    return expected;
}

void run_reduce_case(distributed::MeshDevice& mesh_device, const ReduceCase& rc) {
    const experimental::NodeCoord node{0, 0};
    const uint32_t calls = rc.accumulate ? 2 : 1;
    const uint32_t input_tiles = rc.rows * rc.cols * rc.batches * calls;
    const uint32_t num_outputs = output_tiles(rc);

    // One page per buffer keeps every tile in bank 0, contiguous, as the reader and writer expect.
    auto make_dram_buffer = [&](uint32_t bytes) {
        distributed::DeviceLocalBufferConfig dram_config{.page_size = bytes, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes}, dram_config, &mesh_device);
    };
    auto src_buffer = make_dram_buffer(input_tiles * BF16_TILE_BYTES);
    auto dst_buffer = make_dram_buffer(num_outputs * FP32_TILE_BYTES);

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName AUX_DFB{"aux_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::DFBSpecName ACC_DFB{"acc_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    const uint32_t auxiliary_tiles = rc.partial ? 2 : 1;
    std::vector<experimental::DataflowBufferSpec> dfbs = {
        {.unique_id = INPUT_DFB,
         .entry_size = BF16_TILE_BYTES,
         .num_entries = rc.input_entries,
         .data_format_metadata = tt::DataFormat::Float16_b},
        {.unique_id = AUX_DFB,
         .entry_size = BF16_TILE_BYTES,
         .num_entries = auxiliary_tiles,
         .data_format_metadata = tt::DataFormat::Float16_b},
        {.unique_id = OUTPUT_DFB,
         .entry_size = FP32_TILE_BYTES,
         .num_entries = 2,
         .data_format_metadata = tt::DataFormat::Float32},
    };
    if (rc.accumulate) {
        dfbs.push_back(
            {.unique_id = ACC_DFB,
             .entry_size = FP32_TILE_BYTES,
             .num_entries = 2,
             .data_format_metadata = tt::DataFormat::Float32});
    }

    // Native reads a FirstRow scaler; TileSum reads [zero], or [mask, zero] with a partial last tile.
    uint32_t aux0_type = AUX_ZERO, aux0_valid = TILE, aux0_value = float_bits(1.0f);
    if (!rc.tile_sum) {
        aux0_type = AUX_FIRST_ROW;
    } else if (rc.partial) {
        aux0_type = AUX_FIRST_ROW;
        aux0_valid = rc.partial;
    }

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reduce_helper_reader_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "in"), experimental::ProducerOf(AUX_DFB, "aux")},
        .compile_time_args =
            {{"auxiliary_tiles", auxiliary_tiles},
             {"aux0_type", aux0_type},
             {"aux0_valid", aux0_valid},
             {"aux0_value", aux0_value},
             {"aux1_type", AUX_ZERO},
             {"aux1_valid", TILE},
             {"aux1_value", float_bits(1.0f)}},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::CreateReaderDataMovementConfig(),
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::CreateWriterDataMovementConfig(),
    };

    const char* dim_define = rc.dim == Dim::Row   ? "ReduceDim::REDUCE_ROW"
                             : rc.dim == Dim::Col ? "ReduceDim::REDUCE_COL"
                                                  : "ReduceDim::REDUCE_SCALAR";
    std::map<std::string, std::string> defines = {
        {"REDUCE_OP", rc.average ? "PoolType::AVG" : "PoolType::SUM"},
        {"REDUCE_DIM", dim_define},
        {"REDUCE_ALGORITHM",
         rc.tile_sum ? "compute_kernel_lib::ReduceAlgorithm::TileSum" : "compute_kernel_lib::ReduceAlgorithm::Native"},
        {"REDUCE_PARTIAL_MODE",
         rc.partial ? "compute_kernel_lib::ReducePartialMode::Mask" : "compute_kernel_lib::ReducePartialMode::None"},
    };
    std::vector<experimental::DFBBinding> compute_bindings = {
        experimental::ConsumerOf(INPUT_DFB, "in"),
        experimental::ConsumerOf(AUX_DFB, "aux"),
        experimental::ProducerOf(OUTPUT_DFB, "out"),
    };
    if (rc.accumulate) {
        defines["REDUCE_ACCUMULATE"] = "1";
        compute_bindings.push_back(experimental::ProducerOf(ACC_DFB, "acc"));
        compute_bindings.push_back(experimental::ConsumerOf(ACC_DFB, "acc"));
    }
    const float scale = rc.average ? 1.0f / static_cast<float>(reduced_elements(rc)) : 1.0f;
    experimental::ComputeHardwareConfig compute_config{.enable_32_bit_dest = true};
    if (rc.accumulate) {
        compute_config.unpack_modes = {{ACC_DFB, UnpackMode::UnpackToSrc}};
    }

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_helper_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = experimental::KernelSpec::CompilerOptions::Defines(defines)},
        .dfb_bindings = compute_bindings,
        .compile_time_args =
            {{"rows", rc.rows},
             {"cols", rc.cols},
             {"batches", rc.batches},
             {"auxiliary_tiles", auxiliary_tiles},
             {"output_group", rc.output_group},
             {"scale_bits", float_bits(scale)}},
        .hw_config = compute_config,
    };

    experimental::ProgramSpec spec{
        .name = "reduce_helper_" + rc.name,
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = dfbs,
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = node,
        }},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<uint16_t> input_bits(input_tiles * TILE_ELEMENTS);
    std::vector<float> input(input_bits.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const bfloat16 value(dist(rng));
        input_bits[i] = std::bit_cast<uint16_t>(value);
        input[i] = static_cast<float>(value);
    }
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src_buffer, u32_from_u16_vector(input_bits), /*blocking=*/true);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", static_cast<uint32_t>(src_buffer->address())},
                 {"src_bank_id", 0u},
                 {"num_tiles", input_tiles},
                 {"dram_page_stride", BF16_TILE_BYTES}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<uint32_t>(dst_buffer->address())},
                 {"dst_bank_id", 0u},
                 {"num_tiles", num_outputs},
                 {"dram_page_stride", FP32_TILE_BYTES}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, dst_buffer, /*blocking=*/true);
    ASSERT_EQ(result.size(), num_outputs * TILE_ELEMENTS);

    uint32_t mismatches = 0;
    for (const auto& e : golden(rc, input)) {
        const float actual = std::bit_cast<float>(result[e.tile * TILE_ELEMENTS + e.offset]);
        if (!(std::fabs(actual - e.value) <= 0.02f * std::fabs(e.value) + 0.05f)) {
            if (mismatches++ < 8) {
                ADD_FAILURE() << rc.name << ": tile " << e.tile << " offset " << e.offset << " expected " << e.value
                              << " got " << actual;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << rc.name;
}

class QuasarReduceHelperFixture : public QuasarMeshDeviceSingleCardFixture,
                                  public ::testing::WithParamInterface<ReduceCase> {};

}  // namespace

TEST_P(QuasarReduceHelperFixture, Reduce) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "Requires the Quasar simulator or emulator.";
    }
    run_reduce_case(this->device(), GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    QuasarReduceHelper,
    QuasarReduceHelperFixture,
    ::testing::Values(
        ReduceCase{.name = "NativeRow", .dim = Dim::Row, .rows = 2, .cols = 2, .tile_sum = false, .input_entries = 4},
        ReduceCase{.name = "TileSumRowBulk", .dim = Dim::Row, .rows = 2, .cols = 3, .input_entries = 6},
        ReduceCase{
            .name = "TileSumColGroupedTail",
            .dim = Dim::Col,
            .rows = 2,
            .cols = 3,
            .batches = 2,
            .output_group = 2,
            .input_entries = 6},
        ReduceCase{
            .name = "TileSumScalarAvg", .dim = Dim::Scalar, .rows = 2, .cols = 2, .average = true, .input_entries = 4},
        ReduceCase{
            .name = "TileSumRowPartialMask", .dim = Dim::Row, .rows = 1, .cols = 2, .partial = 7, .input_entries = 2},
        ReduceCase{
            .name = "TileSumRowAccumulate",
            .dim = Dim::Row,
            .rows = 1,
            .cols = 2,
            .accumulate = true,
            .input_entries = 2}),
    [](const ::testing::TestParamInfo<ReduceCase>& info) { return info.param.name; });
