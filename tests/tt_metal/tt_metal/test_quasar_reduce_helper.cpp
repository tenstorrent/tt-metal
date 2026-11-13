// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// compute_kernel_lib::reduce() explicit modes on Quasar: TileSum on each input policy, strides, within-tile Skip,
// partial masks and scalers, scale_bits, auxiliary offset, cross-call accumulation with both Quasar reload modes and a
// post-reduce op, with Native cases as the harness baseline.

#include "common/device_fixture.hpp"

#include <algorithm>
#include <array>
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
constexpr float POISON = 100.0f;

// ttnn::kernel_lib::ReduceAuxiliaryTileType
constexpr uint32_t AUX_FIRST_ROW = 0;
constexpr uint32_t AUX_FIRST_COLUMN = 1;
constexpr uint32_t AUX_FIRST_ROW_PER_FACE_ROW = 2;
constexpr uint32_t AUX_ZERO = 3;

enum class Dim { Row, Col, Scalar };
enum class Policy { BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop };

struct ReduceCase {
    std::string name;
    Dim dim;
    uint32_t rows;
    uint32_t cols;
    uint32_t batches = 1;
    Policy policy = Policy::BulkWaitBulkPop;
    bool tile_sum = true;
    bool average = false;
    bool skip = false;     // ReduceWithinTile::Skip
    uint32_t partial = 0;  // valid rows/columns of the last reduce-dim tile, 0 = full
    uint32_t output_group = 0;
    uint32_t row_padding = 0;
    uint32_t batch_padding = 0;
    uint32_t auxiliary_offset = 0;
    bool accumulate = false;  // two calls folded through an accumulator DFB
    bool fold_via_add = false;
    float post_mul = 0.0f;       // nonzero: post_reduce_op multiplies each output by this
    uint32_t input_entries = 0;  // 0 = one call's whole input
};

struct AuxTile {
    uint32_t type;
    uint32_t valid;
    float value;
};

uint32_t float_bits(float f) { return std::bit_cast<uint32_t>(f); }

// Element (r, c) of a 32x32 tile stored as four 16x16 faces.
uint32_t tile_offset(uint32_t r, uint32_t c) { return (r / 16 * 2 + c / 16) * 256 + (r % 16) * 16 + (c % 16); }

bool indexed(const ReduceCase& rc) { return rc.policy != Policy::BulkWaitBulkPop; }
uint32_t calls(const ReduceCase& rc) { return rc.accumulate ? 2 : 1; }
uint32_t row_stride(const ReduceCase& rc) { return rc.cols + rc.row_padding; }
uint32_t batch_stride(const ReduceCase& rc) { return rc.rows * row_stride(rc) + rc.batch_padding; }
uint32_t call_tiles(const ReduceCase& rc) {
    return indexed(rc) ? batch_stride(rc) * rc.batches : rc.rows * rc.cols * rc.batches;
}

uint32_t valid_width(const ReduceCase& rc) {
    return rc.dim == Dim::Row && rc.partial ? (rc.cols - 1) * TILE + rc.partial : rc.cols * TILE;
}
uint32_t valid_height(const ReduceCase& rc) {
    return rc.dim == Dim::Col && rc.partial ? (rc.rows - 1) * TILE + rc.partial : rc.rows * TILE;
}

uint32_t average_divisor(const ReduceCase& rc) {
    uint32_t per_call = 0;
    if (rc.skip) {
        per_call = rc.dim == Dim::Row ? rc.cols : rc.dim == Dim::Col ? rc.rows : rc.rows * rc.cols;
    } else {
        per_call = rc.dim == Dim::Row   ? valid_width(rc)
                   : rc.dim == Dim::Col ? valid_height(rc)
                                        : valid_height(rc) * valid_width(rc);
    }
    return per_call * calls(rc);
}

uint32_t output_tiles(const ReduceCase& rc) {
    const uint32_t per_batch = rc.dim == Dim::Row ? rc.rows : rc.dim == Dim::Col ? rc.cols : 1;
    return per_batch * rc.batches;
}

// Leading tiles that a correct auxiliary_tile_offset skips; then Native [scaler(, partial scaler)] or TileSum
// [(mask,) zero].
std::vector<AuxTile> auxiliary_tiles(const ReduceCase& rc) {
    std::vector<AuxTile> tiles(rc.auxiliary_offset, AuxTile{AUX_FIRST_ROW, TILE, 7.0f});
    if (rc.tile_sum) {
        if (rc.partial) {
            tiles.push_back({rc.dim == Dim::Row ? AUX_FIRST_ROW : AUX_FIRST_COLUMN, rc.partial, 1.0f});
        }
        tiles.push_back({AUX_ZERO, TILE, 0.0f});
    } else {
        tiles.push_back({AUX_FIRST_ROW, TILE, 1.0f});
        if (rc.partial) {
            tiles.push_back({rc.dim == Dim::Row ? AUX_FIRST_ROW : AUX_FIRST_ROW_PER_FACE_ROW, rc.partial, 1.0f});
        }
    }
    return tiles;
}

// Index within one call's input of logical tile (b, h, w), in the order the helper consumes it.
uint32_t input_tile(const ReduceCase& rc, uint32_t b, uint32_t h, uint32_t w) {
    if (rc.dim == Dim::Col && !indexed(rc)) {
        // BulkWaitBulkPop REDUCE_COL reads each output_group of columns as one row-major [Ht x group] bulk.
        const uint32_t first = w / rc.output_group * rc.output_group;
        const uint32_t width = std::min(rc.output_group, rc.cols - first);
        return b * rc.rows * rc.cols + rc.rows * first + h * width + (w - first);
    }
    return b * batch_stride(rc) + h * row_stride(rc) + w;
}

// Expected value at (tile, offset) for each reduced output; other output elements are not checked.
struct Expected {
    uint32_t tile;
    uint32_t offset;
    float value;
};

std::vector<Expected> golden(const ReduceCase& rc, const std::vector<float>& input) {
    auto at = [&](uint32_t b, uint32_t h, uint32_t w, uint32_t r, uint32_t c) {
        double sum = 0;
        for (uint32_t call = 0; call < calls(rc); ++call) {
            const uint32_t tile = call * call_tiles(rc) + input_tile(rc, b, h, w);
            sum += input[tile * TILE_ELEMENTS + tile_offset(r, c)];
        }
        return sum;
    };
    const double scale = (rc.average ? 1.0 / average_divisor(rc) : 1.0) * (rc.post_mul != 0.0f ? rc.post_mul : 1.0);
    std::vector<Expected> expected;
    for (uint32_t b = 0; b < rc.batches; ++b) {
        if (rc.dim == Dim::Row) {
            for (uint32_t h = 0; h < rc.rows; ++h) {
                for (uint32_t r = 0; r < TILE; ++r) {
                    double sum = 0;
                    if (rc.skip) {
                        for (uint32_t w = 0; w < rc.cols; ++w) {
                            sum += at(b, h, w, r, 0);
                        }
                    } else {
                        for (uint32_t x = 0; x < valid_width(rc); ++x) {
                            sum += at(b, h, x / TILE, r, x % TILE);
                        }
                    }
                    expected.push_back({b * rc.rows + h, tile_offset(r, 0), static_cast<float>(sum * scale)});
                }
            }
        } else if (rc.dim == Dim::Col) {
            for (uint32_t w = 0; w < rc.cols; ++w) {
                for (uint32_t c = 0; c < TILE; ++c) {
                    double sum = 0;
                    if (rc.skip) {
                        for (uint32_t h = 0; h < rc.rows; ++h) {
                            sum += at(b, h, w, 0, c);
                        }
                    } else {
                        for (uint32_t y = 0; y < valid_height(rc); ++y) {
                            sum += at(b, y / TILE, w, y % TILE, c);
                        }
                    }
                    expected.push_back({b * rc.cols + w, tile_offset(0, c), static_cast<float>(sum * scale)});
                }
            }
        } else {
            double sum = 0;
            for (uint32_t h = 0; h < rc.rows; ++h) {
                for (uint32_t w = 0; w < rc.cols; ++w) {
                    if (rc.skip) {
                        sum += at(b, h, w, 0, 0);
                        continue;
                    }
                    for (uint32_t i = 0; i < TILE_ELEMENTS; ++i) {
                        sum += at(b, h, w, i / TILE, i % TILE);
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
    const uint32_t input_tiles = call_tiles(rc) * calls(rc);
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

    const std::vector<AuxTile> aux = auxiliary_tiles(rc);
    std::array<AuxTile, 3> aux_args{};
    std::copy(aux.begin(), aux.end(), aux_args.begin());
    const auto num_aux = static_cast<uint32_t>(aux.size());

    std::vector<experimental::DataflowBufferSpec> dfbs = {
        {.unique_id = INPUT_DFB,
         .entry_size = BF16_TILE_BYTES,
         .num_entries = rc.input_entries ? rc.input_entries : call_tiles(rc),
         .data_format_metadata = tt::DataFormat::Float16_b},
        {.unique_id = AUX_DFB,
         .entry_size = BF16_TILE_BYTES,
         .num_entries = num_aux,
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
             .num_entries = std::max(2u, output_tiles(rc)),
             .data_format_metadata = tt::DataFormat::Float32});
    }

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reduce_helper_reader_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "in"), experimental::ProducerOf(AUX_DFB, "aux")},
        .compile_time_args =
            {{"auxiliary_tiles", num_aux},
             {"aux0_type", aux_args[0].type},
             {"aux0_valid", aux_args[0].valid},
             {"aux0_value", float_bits(aux_args[0].value)},
             {"aux1_type", aux_args[1].type},
             {"aux1_valid", aux_args[1].valid},
             {"aux1_value", float_bits(aux_args[1].value)},
             {"aux2_type", aux_args[2].type},
             {"aux2_valid", aux_args[2].valid},
             {"aux2_value", float_bits(aux_args[2].value)}},
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
    const char* policy_define = rc.policy == Policy::BulkWaitBulkPop    ? "BulkWaitBulkPop"
                                : rc.policy == Policy::WaitUpfrontNoPop ? "WaitUpfrontNoPop"
                                                                        : "NoWaitNoPop";
    const char* partial_define = !rc.partial   ? "compute_kernel_lib::ReducePartialMode::None"
                                 : rc.tile_sum ? "compute_kernel_lib::ReducePartialMode::Mask"
                                               : "compute_kernel_lib::ReducePartialMode::Scaler";
    std::map<std::string, std::string> defines = {
        {"REDUCE_OP", rc.average ? "PoolType::AVG" : "PoolType::SUM"},
        {"REDUCE_DIM", dim_define},
        {"REDUCE_INPUT_POLICY", policy_define},
        {"REDUCE_ALGORITHM",
         rc.tile_sum ? "compute_kernel_lib::ReduceAlgorithm::TileSum" : "compute_kernel_lib::ReduceAlgorithm::Native"},
        {"REDUCE_WITHIN_TILE", rc.skip ? "Skip" : "Collapse"},
        {"REDUCE_PARTIAL_MODE", partial_define},
        {"REDUCE_RELOAD_MODE", rc.fold_via_add ? "FoldViaAdd" : "CopySeedPairs"},
    };
    std::vector<experimental::DFBBinding> compute_bindings = {
        experimental::ConsumerOf(INPUT_DFB, "in"),
        experimental::ConsumerOf(AUX_DFB, "aux"),
        experimental::ProducerOf(OUTPUT_DFB, "out"),
    };
    experimental::ComputeHardwareConfig compute_config{.enable_32_bit_dest = true};
    if (rc.accumulate) {
        defines["REDUCE_ACCUMULATE"] = "1";
        compute_bindings.push_back(experimental::ProducerOf(ACC_DFB, "acc"));
        compute_bindings.push_back(experimental::ConsumerOf(ACC_DFB, "acc"));
        // FoldViaAdd reads the accumulator through SrcB, which an unpack-to-Dest buffer cannot feed.
        compute_config.unpack_modes = {{ACC_DFB, UnpackMode::UnpackToSrc}};
    }
    if (rc.post_mul != 0.0f) {
        defines["REDUCE_POST_MUL"] = "1";
    }
    const bool strided = rc.row_padding || rc.batch_padding;
    const float scale = rc.average ? 1.0f / static_cast<float>(average_divisor(rc)) : 1.0f;

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
             {"row_stride", rc.row_padding ? row_stride(rc) : 0u},
             {"batch_stride", strided ? batch_stride(rc) : 0u},
             {"input_tiles", call_tiles(rc)},
             {"auxiliary_tiles", num_aux},
             {"auxiliary_offset", rc.auxiliary_offset},
             {"output_group", rc.output_group},
             {"scale_bits", float_bits(scale)},
             {"post_mul_bits", float_bits(rc.post_mul)}},
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

    // Tiles a stride skips hold POISON, so reading one corrupts the result.
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> input(input_tiles * TILE_ELEMENTS, POISON);
    for (uint32_t call = 0; call < calls(rc); ++call) {
        for (uint32_t b = 0; b < rc.batches; ++b) {
            for (uint32_t h = 0; h < rc.rows; ++h) {
                for (uint32_t w = 0; w < rc.cols; ++w) {
                    const uint32_t tile = call * call_tiles(rc) + input_tile(rc, b, h, w);
                    for (uint32_t i = 0; i < TILE_ELEMENTS; ++i) {
                        input[tile * TILE_ELEMENTS + i] = dist(rng);
                    }
                }
            }
        }
    }
    std::vector<uint16_t> input_bits(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const bfloat16 value(input[i]);
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
    // Native ReducePartialMode::Scaler does not work on Quasar: the reduce LLK unpacks only face 0 of the scaler
    // tile and applies it to every face, so the partial scaler also admits the same lanes of the next face along the
    // reduce dim (7 valid columns count 14). TileSum with ReducePartialMode::Mask is the working alternative.
    if (!GetParam().tile_sum && GetParam().partial) {
        GTEST_SKIP() << "Native partial scaler is unsupported on Quasar (the reduce LLK reads only scaler face 0).";
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
            .name = "TileSumColPartialMask",
            .dim = Dim::Col,
            .rows = 2,
            .cols = 2,
            .partial = 5,
            .output_group = 2,
            .input_entries = 4},
        ReduceCase{
            .name = "NativeRowPartialScaler",
            .dim = Dim::Row,
            .rows = 1,
            .cols = 2,
            .tile_sum = false,
            .partial = 7,
            .input_entries = 2},
        ReduceCase{
            .name = "NativeColPartialScaler",
            .dim = Dim::Col,
            .rows = 2,
            .cols = 2,
            .tile_sum = false,
            .partial = 5,
            .output_group = 2,
            .input_entries = 4},
        ReduceCase{
            .name = "TileSumRowWaitUpfront", .dim = Dim::Row, .rows = 2, .cols = 3, .policy = Policy::WaitUpfrontNoPop},
        ReduceCase{
            .name = "TileSumColWaitUpfront", .dim = Dim::Col, .rows = 3, .cols = 5, .policy = Policy::WaitUpfrontNoPop},
        ReduceCase{
            .name = "TileSumRowNoWaitStrides",
            .dim = Dim::Row,
            .rows = 2,
            .cols = 2,
            .batches = 2,
            .policy = Policy::NoWaitNoPop,
            .row_padding = 1,
            .batch_padding = 1},
        ReduceCase{
            .name = "TileSumColNoWaitStrides",
            .dim = Dim::Col,
            .rows = 2,
            .cols = 3,
            .batches = 2,
            .policy = Policy::NoWaitNoPop,
            .row_padding = 1,
            .batch_padding = 1},
        ReduceCase{
            .name = "TileSumScalarWaitUpfrontBatchStrideAvg",
            .dim = Dim::Scalar,
            .rows = 2,
            .cols = 2,
            .batches = 2,
            .policy = Policy::WaitUpfrontNoPop,
            .average = true,
            .batch_padding = 1},
        ReduceCase{.name = "TileSumRowSkip", .dim = Dim::Row, .rows = 2, .cols = 3, .skip = true, .input_entries = 6},
        ReduceCase{
            .name = "TileSumRowAuxOffset",
            .dim = Dim::Row,
            .rows = 2,
            .cols = 2,
            .auxiliary_offset = 1,
            .input_entries = 4},
        ReduceCase{
            .name = "TileSumRowAccumulate",
            .dim = Dim::Row,
            .rows = 1,
            .cols = 2,
            .accumulate = true,
            .input_entries = 2},
        ReduceCase{
            .name = "TileSumRowAccumulateFoldViaAdd",
            .dim = Dim::Row,
            .rows = 1,
            .cols = 3,
            .accumulate = true,
            .fold_via_add = true,
            .input_entries = 3},
        ReduceCase{
            .name = "TileSumRowAccumulatePostMul",
            .dim = Dim::Row,
            .rows = 1,
            .cols = 2,
            .accumulate = true,
            .post_mul = 0.5f,
            .input_entries = 2}),
    [](const ::testing::TestParamInfo<ReduceCase>& info) { return info.param.name; });
