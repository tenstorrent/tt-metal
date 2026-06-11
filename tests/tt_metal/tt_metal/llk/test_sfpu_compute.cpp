// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/int8.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/int8.hpp>

namespace tt::tt_metal {

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::sfpu_util {

const map<std::string, std::map<std::string, std::string>> sfpu_op_to_op_name = {
    // FIXME: #1157
    {"relu", {{"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}}},
    {"exponential", {{"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
    {"reciprocal", {{"SFPU_OP_CHAIN_0", "recip_tile_init(); recip_tile(0);"}}},
    {"gelu", {{"SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);"}}},
    {"sqrt", {{"SFPU_OP_CHAIN_0", "sqrt_tile_init(); sqrt_tile(0);"}}},
    {"sigmoid", {{"SFPU_OP_CHAIN_0", "sigmoid_tile_init(); sigmoid_tile(0);"}}},
    {"silu", {{"SFPU_OP_CHAIN_0", "silu_tile_init(); silu_tile(0);"}}},
    {"log", {{"SFPU_OP_CHAIN_0", "log_tile_init(); log_tile(0);"}}},
    {"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}},
    {"sign", {{"SFPU_OP_CHAIN_0", "sign_tile_init(); sign_tile(0);"}}},
    {"rsqrt", {{"SFPU_OP_CHAIN_0", "rsqrt_tile_init(); rsqrt_tile(0);"}}},
    {"mul_unary", {{"SFPU_OP_CHAIN_0", "binop_with_scalar_tile_init(); mul_unary_tile(0, 0x40000000u);"}}},  // 2.0f
};

// Binary SFPU ops driven by `run_sfpu_binary_two_input_buffer`.
//
// Each entry maps an op name (the test parameter) to the kernel-side macro
// substitutions expanded by the SFPU binary compute kernel
// (`eltwise_sfpu_2_0.cpp`, Metal 2.0 dataflow-buffer based, SFPU_BINARY_OP):
//
//   * SFPU_OP_INIT_0  — runs once before the per-pair loop. Used to set up
//                       SFPU lookup tables / state (e.g. div reciprocal LUT).
//   * SFPU_OP_CHAIN_0 — runs once per (LHS, RHS) pair, inside an
//                       acquire/release section. By convention LHS lives at
//                       DST[0] and RHS at DST[1]; the result is written back
//                       to DST[0] so the packer reads from there.
//
// To add a new binary SFPU op: add an entry here, add a matching arm in
// sfpu_binary_function() or get_binary_int_operation_result() for golden compute,
// and (if its valid input range differs from div) add an arm in generate_packed_sfpu_binary_inputs().
const map<std::string, std::map<std::string, std::string>> sfpu_binary_op_to_op_name = {
    {"div_binary", {{"SFPU_OP_INIT_0", "div_binary_tile_init();"}, {"SFPU_OP_CHAIN_0", "div_binary_tile(0, 1, 0);"}}},
    {"mul_float", {{"SFPU_OP_INIT_0", "mul_binary_tile_init();"}, {"SFPU_OP_CHAIN_0", "mul_binary_tile(0, 1, 0);"}}},
    // add_int: Int8 L1 inputs are promoted to sign-magnitude Int32 in DEST via copy_tile + fp32_dest_acc;
    // add_int_tile<Int32> (sign-mag on Quasar via ARCH_QUASAR) then adds in sign-mag space. Result in DST[0].
    {"add_int",
     {{"SFPU_OP_INIT_0", "add_int_tile_init();"}, {"SFPU_OP_CHAIN_0", "add_int_tile<DataFormat::Int32>(0, 1, 0);"}}},
    {"mul_int",
     {{"SFPU_OP_INIT_0", "mul_int_tile_init<DataFormat::Int32>();"},
      {"SFPU_OP_CHAIN_0", "mul_int_tile<DataFormat::Int32>(0, 1, 0);"}}},
    {"gt_int",
     {{"SFPU_OP_INIT_0", "gt_int_tile_init<DataFormat::Int32>();"},
      {"SFPU_OP_CHAIN_0", "gt_int_tile<DataFormat::Int32>(0, 1, 0);"}}},
    {"binary_max", {{"SFPU_OP_INIT_0", "binary_max_tile_init();"}, {"SFPU_OP_CHAIN_0", "binary_max_tile(0, 1, 0);"}}},
    {"binary_min", {{"SFPU_OP_INIT_0", "binary_min_tile_init();"}, {"SFPU_OP_CHAIN_0", "binary_min_tile(0, 1, 0);"}}},
    {"binary_max_int32",
     {{"SFPU_OP_INIT_0", "binary_max_int32_tile_init();"}, {"SFPU_OP_CHAIN_0", "binary_max_int32_tile(0, 1, 0);"}}},
    {"binary_min_int32",
     {{"SFPU_OP_INIT_0", "binary_min_int32_tile_init();"}, {"SFPU_OP_CHAIN_0", "binary_min_int32_tile(0, 1, 0);"}}},
};

bool is_int8_binary_sfpu_op(const std::string& op_name) {
    return (op_name == "add_int") or (op_name == "mul_int") or (op_name == "gt_int") or
           (op_name == "binary_max_int32") or (op_name == "binary_min_int32");
}

bfloat16 sfpu_function(const std::string& op_name, const bfloat16& input) {
    if (op_name == "relu") {
        return bfloat16(fmaxf(static_cast<float>(input), 0.0f));
    }
    if (op_name == "exponential") {
        return bfloat16(std::exp(static_cast<float>(input)));
    }
    if (op_name == "reciprocal") {
        return bfloat16(1 / static_cast<float>(input));
    }
    if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x = static_cast<float>(input);
        auto x3 = x * x * x;
        float result = x * 0.5 * (1.0 + tanhf(alpha * (x + 0.044715 * x3)));
        return bfloat16(result);
    }
    if (op_name == "sqrt") {
        return bfloat16(sqrtf(static_cast<float>(input)));
    }
    if (op_name == "sigmoid") {
        auto x = static_cast<float>(input);
        float result = 1 / (1 + std::exp(-x));
        return bfloat16(result);
    }
    if (op_name == "silu") {
        auto x = static_cast<float>(input);
        float result = x / (1 + std::exp(-x));
        return bfloat16(result);
    }
    if (op_name == "log") {
        return bfloat16(logf(static_cast<float>(input)));
    }
    if (op_name == "tanh") {
        return bfloat16(std::tanh(static_cast<float>(input)));
    }
    if (op_name == "rsqrt") {
        return bfloat16(1.0f / sqrtf(static_cast<float>(input)));
    }
    if (op_name == "sign") {
        float val = static_cast<float>(input);
        float result = static_cast<float>((val > 0.0f) - (val < 0.0f));
        return bfloat16(result);
    }
    if (op_name == "mul_unary") {
        return bfloat16(static_cast<float>(input) * 2.0f);
    }
    TT_THROW("Unsupported op_name in test");
}

// Reference implementation for float-typed binary SFPU ops.
bfloat16 sfpu_binary_function(const std::string& op_name, const bfloat16& lhs, const bfloat16& rhs) {
    if (op_name == "div_binary") {
        return bfloat16(static_cast<float>(lhs) / static_cast<float>(rhs));
    }
    if (op_name == "mul_float") {
        return bfloat16(static_cast<float>(lhs) * static_cast<float>(rhs));
    }
    if (op_name == "binary_max") {
        return bfloat16(std::max(static_cast<float>(lhs), static_cast<float>(rhs)));
    }
    if (op_name == "binary_min") {
        return bfloat16(std::min(static_cast<float>(lhs), static_cast<float>(rhs)));
    }
    TT_THROW("Unsupported binary op_name in test");
}

// Ternary SFPU ops driven by `run_sfpu_ternary_three_input_buffer`.
//
// Each entry maps an op name to SFPU_OP_CHAIN_0 — the full per-tile body
// (init + compute) run once per (in0, in1, in2) triple inside an
// acquire/release section. Mirrors the unary pattern where init and compute
// are both part of the chain.
const map<std::string, std::map<std::string, std::string>> sfpu_ternary_op_to_op_name = {
    {"where", {{"SFPU_OP_CHAIN_0", "where_tile_init(); where_tile<DataFormat::Float16_b>(0, 1, 2, 3);"}}},
};

bfloat16 sfpu_ternary_function(
    const std::string& op_name, const bfloat16& in0, const bfloat16& in1, const bfloat16& in2) {
    if (op_name == "where") {
        // Condition selects in1 when "true"; treat a near-zero condition as "false".
        constexpr float condition_epsilon = 1e-3f;
        return (std::fabs(static_cast<float>(in0)) < condition_epsilon) ? in2 : in1;
    }
    TT_THROW("Unsupported ternary op_name in test");
}

// Reference for int8 binary SFPU ops after sign-magnitude decode.
int32_t get_binary_int_operation_result(const std::string& op_name, int lhs, int rhs) {
    if (op_name == "add_int") {
        return static_cast<int32_t>(lhs + rhs);
    }
    if (op_name == "mul_int") {
        return static_cast<int32_t>(lhs * rhs);
    }
    if (op_name == "gt_int") {
        return (lhs > rhs) ? 1 : 0;
    }
    if (op_name == "binary_max_int32") {
        return std::max(lhs, rhs);
    }
    if (op_name == "binary_min_int32") {
        return std::min(lhs, rhs);
    }
    TT_THROW("Unsupported int8 binary op_name in test");
}

// Builds packed sign-magnitude Int32 golden for int8 binary SFPU ops.
std::vector<uint32_t> compute_packed_int8_binary_golden(
    const std::vector<uint32_t>& packed_lhs, const std::vector<uint32_t>& packed_rhs, const std::string& op_name) {
    TT_FATAL(packed_lhs.size() == packed_rhs.size(), "lhs/rhs packed size mismatch");
    std::vector<uint32_t> packed_golden(packed_lhs.size() * 4);
    for (size_t w = 0; w < packed_lhs.size(); ++w) {
        for (int b = 0; b < 4; ++b) {
            const auto lhs_byte = static_cast<uint8_t>((packed_lhs[w] >> (b * 8)) & 0xFF);
            const auto rhs_byte = static_cast<uint8_t>((packed_rhs[w] >> (b * 8)) & 0xFF);
            const int lhs = sign_mag_byte_to_int8(lhs_byte);
            const int rhs = sign_mag_byte_to_int8(rhs_byte);
            const int32_t result = get_binary_int_operation_result(op_name, lhs, rhs);
            packed_golden[w * 4 + b] = int32_to_sign_mag_word(result);
        }
    }
    return packed_golden;
}
vector<uint32_t> generate_packed_sfpu_input(const unsigned int numel, const std::string& op_name, const int seed) {
    if ((op_name == "sqrt") or (op_name == "log") or (op_name == "rsqrt")) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(0.0001f, 4.0f, numel, seed);
    }
    if ((op_name == "exponential") or (op_name == "gelu") or (op_name == "reciprocal")) {
        auto possible_values = vector<bfloat16>({-1.0f, -0.5f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_values, numel, seed);
    }
    return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed);
}

// Mirror the LLK `_prepare_div_inputs` helper: uniform in [-4, 4], then snap
// |x| < 0.25 to ±0.25 keeping the sign. Every element ends up in
// [-4, -0.25] ∪ [0.25, 4]. This exercises both halves of the sfpi `setsgn`
// path and avoids sub-normal divisors / spurious 0/0 -> NaN.
static vector<uint32_t> generate_div_operand(const unsigned int numel, const int seed) {
    auto packed = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-4.0f, 4.0f, numel, seed);
    auto unpacked = unpack_vector<bfloat16, uint32_t>(packed);
    for (auto& v : unpacked) {
        float f = static_cast<float>(v);
        float sign = (f >= 0.0f) ? 1.0f : -1.0f;
        float magnitude = std::fabs(f);
        magnitude = std::max(magnitude, 0.25f);
        v = bfloat16(sign * magnitude);
    }
    return pack_vector<uint32_t, bfloat16>(unpacked);
}

// Per-operand stimuli for binary SFPU ops. LHS and RHS use independent seeds so
// their signs and magnitudes vary independently.
std::pair<vector<uint32_t>, vector<uint32_t>> generate_packed_sfpu_binary_inputs(
    const unsigned int numel, const std::string& op_name, const int seed) {
    if (op_name == "div_binary" || op_name == "mul_float") {
        // Reuse the div operand generator: values in [-4,-0.25] ∪ [0.25,4]. For mul this
        // is simply well-conditioned, finite bf16 input with independent lhs/rhs signs.
        auto lhs = generate_div_operand(numel, seed);
        auto rhs = generate_div_operand(numel, seed + 1);
        return {lhs, rhs};
    }
    if (op_name == "binary_max" || op_name == "binary_min") {
        auto lhs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-4.0f, 4.0f, numel, seed);
        auto rhs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-4.0f, 4.0f, numel, seed + 1);
        return {lhs, rhs};
    }
    if (is_int8_binary_sfpu_op(op_name)) {
        auto lhs = create_random_vector_of_int8(numel, seed);
        auto rhs = create_random_vector_of_int8(numel, seed + 1);
        return {lhs, rhs};
    }
    TT_THROW("Unsupported binary op_name in test");
}

// Returns (in0, in1, in2) packed operand vectors for ternary SFPU ops.
std::tuple<vector<uint32_t>, vector<uint32_t>, vector<uint32_t>> generate_packed_sfpu_ternary_inputs(
    const unsigned int numel, const std::string& op_name, const int seed) {
    if (op_name == "where") {
        auto possible_cond = vector<bfloat16>({-1.0f, 0.0f, 1.0f});
        auto packed_cond = generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_cond, numel, seed);
        auto packed_true_val = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed + 1);
        auto packed_false_val = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed + 2);
        return {packed_cond, packed_true_val, packed_false_val};
    }
    TT_THROW("Unsupported ternary op_name in test");
}

bool is_close_packed_sfpu_output(
    const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const std::string& op_name) {
    if (is_int8_binary_sfpu_op(op_name) || op_name == "binary_max" || op_name == "binary_min") {
        return vec_a == vec_b;
    }
    if (op_name == "where") {
        // Matches the LLK pytest's torch.isclose(rtol=0.05, atol=0.05) for
        // Float16 / Float16_b / Float32 (helpers/utils.py:tolerances).
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.05f, 0.05f); });
    }
    if (op_name == "tanh") {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.175f, 0.1f); });
    }
    if ((op_name == "gelu") or (op_name == "relu")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.15f); });
    }
    if ((op_name == "exponential")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.1f, 0.1f); });
    }
    if ((op_name == "log")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.03f, 0.02f); });
    }
    return is_close_packed_vectors<bfloat16, uint32_t>(
        vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.06f, 0.006f); });
}

}  // namespace unit_tests::sfpu_util

namespace unit_tests::compute::sfpu {

struct SfpuConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores;
    std::string sfpu_op;
    bool approx_mode = true;
    bool en_32bit_dest = false;
};

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_all_same_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto& cq = mesh_device->mesh_command_queue();

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = mesh_device->get_devices()[0],
        .size = byte_size,
        .page_size = byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    auto output_dram_buffer = CreateBuffer(dram_config);

    // Host input + golden generation
    std::vector<uint32_t> packed_input = sfpu_util::generate_packed_sfpu_input(
        byte_size / sizeof(bfloat16), test_config.sfpu_op, std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    auto input = unpack_vector<bfloat16, uint32_t>(packed_input);
    std::vector<bfloat16> golden(input.size());
    std::transform(input.begin(), input.end(), golden.begin(), [&](const bfloat16& val) {
        return sfpu_util::sfpu_function(test_config.sfpu_op, val);
    });
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_UNARY_OP"] = "1";
    sfpu_defines["SFPU_OP_EXP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_GELU_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RECIP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_SQRT_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RSQRT_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ELU_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_NEG_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_BINOP_WITH_SCALAR_INCLUDE"] = "1";

    // Every existing parametrization of this test uses a single-core CoreRangeSet of {0, 0};
    // MakeProgramFromSpec models the kernel set per single-core WorkUnit.
    TT_FATAL(
        test_config.cores.ranges().size() == 1,
        "sfpu test expects a single CoreRange (got {})",
        test_config.cores.size());
    const CoreRange& core_range = *test_config.cores.ranges().begin();
    TT_FATAL(core_range.start_coord == core_range.end_coord, "sfpu test expects a single-core CoreRange");
    const CoreCoord core = core_range.start_coord;
    const experimental::NodeCoord node{core.x, core.y};

    const experimental::DFBSpecName IN_DFB{"in_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_input_data_format,
    };
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_output_data_format,
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = IN_DFB,
            .accessor_name = "out",
            .endpoint_type = experimental::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_config =
                    experimental::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
                .gen2_config =
                    experimental::DataMovementHardwareConfig::Gen2Config{.disable_implicit_sync_for = {IN_DFB}}},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "in",
            .endpoint_type = experimental::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_config =
                    experimental::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
                .gen2_config =
                    experimental::DataMovementHardwareConfig::Gen2Config{.disable_implicit_sync_for = {OUT_DFB}}},
    };

    experimental::KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : sfpu_defines) {
        compute_defines.emplace(k, v);
    }

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN_DFB,
                 .accessor_name = "in",
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
            {{"per_core_block_cnt", static_cast<uint32_t>(test_config.num_tiles)}, {"per_core_block_size", 1u}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fp32_dest_acc_en = test_config.en_32bit_dest,
                .math_approx_mode = test_config.approx_mode,
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "sfpu_compute",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
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
            .runtime_arg_values =
                {{node,
                  {{"src_addr", input_dram_buffer->address()},
                   {"bank_id", 0u},
                   {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", output_dram_buffer->address()},
                   {"bank_id", 0u},
                   {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

namespace {

// Validates that cfg describes a single-core CoreRange and returns the Quasar NodeCoord.
experimental::NodeCoord extract_single_core_node(const SfpuConfig& cfg, const char* context) {
    TT_FATAL(cfg.cores.ranges().size() == 1, "{} expects a single CoreRange (got {})", context, cfg.cores.size());
    const CoreRange& cr = *cfg.cores.ranges().begin();
    TT_FATAL(cr.start_coord == cr.end_coord, "{} expects a single-core CoreRange", context);
    return {cr.start_coord.x, cr.start_coord.y};
}

// Builds a DataflowBufferSpec. `entry_size` is derived from `fmt` so that input and output
// DFBs are correctly sized even when their formats differ (e.g. Int8 in → Int32 out).
experimental::DataflowBufferSpec make_dfb_spec(
    const experimental::DFBSpecName& id, const SfpuConfig& cfg, tt::DataFormat fmt) {
    return {
        .unique_id = id,
        .entry_size = static_cast<uint32_t>(tt::tile_size(fmt)),
        .num_entries = static_cast<uint32_t>(cfg.num_tiles),
        .data_format_metadata = fmt,
    };
}

// Converts a string→string defines map to the CompilerOptions::Defines vector form.
experimental::KernelSpec::CompilerOptions::Defines to_kernel_defines(const std::map<std::string, std::string>& m) {
    experimental::KernelSpec::CompilerOptions::Defines defines;
    for (const auto& [k, v] : m) {
        defines.emplace(k, v);
    }
    return defines;
}

// Builds a writer_unary KernelSpec bound to a single output DFB.
experimental::KernelSpec make_writer_unary_quasar_spec(
    const experimental::KernelSpecName& kernel_id, const experimental::DFBSpecName& out_dfb_id) {
    return {
        .unique_id = kernel_id,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = out_dfb_id,
            .accessor_name = "in",
            .endpoint_type = experimental::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen2_config =
                    experimental::DataMovementHardwareConfig::Gen2Config{.disable_implicit_sync_for = {out_dfb_id}}},
    };
}

// Creates a writer_unary kernel on the legacy (non-Quasar) path.
tt_metal::KernelHandle create_legacy_writer_kernel(tt_metal::Program& program, const SfpuConfig& cfg) {
    return tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        cfg.cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
}

// Compiles `spec`, sets `params`, uploads each input buffer, launches on the device, and
// returns the raw output tile data. Shared by the binary and ternary Quasar SFPU runners.
std::vector<uint32_t> sfpu_quasar_run(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const experimental::ProgramSpec& spec,
    const experimental::ProgramRunArgs& params,
    const std::vector<std::pair<std::shared_ptr<tt::tt_metal::Buffer>, const std::vector<uint32_t>*>>& inputs,
    const std::shared_ptr<tt::tt_metal::Buffer>& out_buf) {
    auto *device = mesh_device->get_devices()[0];
    auto program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::SetProgramRunArgs(program, params);
    for (const auto& [buf, data] : inputs) {
        tt_metal::detail::WriteToBuffer(buf, *data);
    }
    tt_metal::detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);
    std::vector<uint32_t> dest;
    tt_metal::detail::ReadFromBuffer(out_buf, dest);
    return dest;
}

}  // namespace

/// High-level flow:
///
///   DRAM(LHS) ─┐
///              ├─> Reader ─> in0/in1 DFB ─> SFPU Compute (eltwise_sfpu_2_0.cpp, SFPU_BINARY_OP) ─> out DFB ─> Writer
///              ─> DRAM(out)
///   DRAM(RHS) ─┘
///
/// @param mesh_device Device under test.
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_binary_two_input_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuConfig& test_config) {
    auto *device = mesh_device->get_devices()[0];
    const size_t per_buffer_byte_size_input = test_config.num_tiles * tt::tile_size(test_config.l1_input_data_format);
    const size_t per_buffer_byte_size_output = test_config.num_tiles * tt::tile_size(test_config.l1_output_data_format);

    tt::tt_metal::InterleavedBufferConfig dram_config_input{
        .device = device,
        .size = per_buffer_byte_size_input,
        .page_size = per_buffer_byte_size_input,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_config_output{
        .device = device,
        .size = per_buffer_byte_size_output,
        .page_size = per_buffer_byte_size_output,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input0_dram_buffer = CreateBuffer(dram_config_input);
    auto input1_dram_buffer = CreateBuffer(dram_config_input);
    auto output_dram_buffer = CreateBuffer(dram_config_output);

    const bool is_int8_op = sfpu_util::is_int8_binary_sfpu_op(test_config.sfpu_op);
    const size_t element_size = is_int8_op ? sizeof(int8_t) : sizeof(bfloat16);
    const uint32_t numel = per_buffer_byte_size_input / element_size;
    const int seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto [packed_lhs, packed_rhs] = sfpu_util::generate_packed_sfpu_binary_inputs(numel, test_config.sfpu_op, seed);

    std::vector<uint32_t> packed_golden;
    if (is_int8_op) {
        packed_golden = sfpu_util::compute_packed_int8_binary_golden(packed_lhs, packed_rhs, test_config.sfpu_op);
    } else {
        auto lhs = unpack_vector<bfloat16, uint32_t>(packed_lhs);
        auto rhs = unpack_vector<bfloat16, uint32_t>(packed_rhs);
        std::vector<bfloat16> golden(lhs.size());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), golden.begin(), [&](const bfloat16& a, const bfloat16& b) {
            return sfpu_util::sfpu_binary_function(test_config.sfpu_op, a, b);
        });
        packed_golden = pack_vector<uint32_t, bfloat16>(golden);
    }

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_binary_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_BINARY_OP"] = "1";
    if (is_int8_op) {
        if (test_config.sfpu_op == "add_int") {
            sfpu_defines["SFPU_OP_BINARY_ADD_INT_INCLUDE"] = "1";
        } else if (test_config.sfpu_op == "mul_int") {
            sfpu_defines["SFPU_OP_BINARY_MUL_INT_INCLUDE"] = "1";
        } else if (test_config.sfpu_op == "gt_int") {
            sfpu_defines["SFPU_OP_BINARY_GT_INT_INCLUDE"] = "1";
        } else {
            sfpu_defines["SFPU_OP_BINARY_MAX_MIN_INCLUDE"] = "1";
        }
    } else if (test_config.sfpu_op == "binary_max" || test_config.sfpu_op == "binary_min") {
        sfpu_defines["SFPU_OP_BINARY_MAX_MIN_INCLUDE"] = "1";
    } else {
        sfpu_defines["SFPU_OP_BINARY_DIV_INCLUDE"] = "1";
    }

    const auto node = extract_single_core_node(test_config, "Metal 2.0 binary SFPU path");
    const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
    const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen2_config =
                    experimental::DataMovementHardwareConfig::Gen2Config{
                        .disable_implicit_sync_for = {IN0_DFB, IN1_DFB}}},
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = to_kernel_defines(sfpu_defines)},
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1_DFB,
                 .accessor_name = "in1",
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
            {{"per_core_block_cnt", 1u}, {"per_core_block_size", static_cast<uint32_t>(test_config.num_tiles)}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fp32_dest_acc_en = is_int8_op,
                .math_approx_mode = test_config.approx_mode,
            },
    };

    experimental::ProgramSpec spec{
        .name = "sfpu_binary_compute",
        .kernels = {reader_spec, make_writer_unary_quasar_spec(WRITER, OUT_DFB), compute_spec},
        .dataflow_buffers =
            {make_dfb_spec(IN0_DFB, test_config, test_config.l1_input_data_format),
             make_dfb_spec(IN1_DFB, test_config, test_config.l1_input_data_format),
             make_dfb_spec(OUT_DFB, test_config, test_config.l1_output_data_format)},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node,
                  {{"src0_addr", input0_dram_buffer->address()},
                   {"src0_bank_id", 0u},
                   {"src1_addr", input1_dram_buffer->address()},
                   {"src1_bank_id", 0u},
                   {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}}},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", output_dram_buffer->address()},
                   {"bank_id", 0u},
                   {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };

    const auto dest = sfpu_quasar_run(
        mesh_device,
        spec,
        params,
        {{input0_dram_buffer, &packed_lhs}, {input1_dram_buffer, &packed_rhs}},
        output_dram_buffer);
    return sfpu_util::is_close_packed_sfpu_output(dest, packed_golden, test_config.sfpu_op);
}

/// High-level flow:
///
///   DRAM(in0) ─┐
///   DRAM(in1) ─┼─> Reader ─> in0/in1/in2 DFB ─> SFPU Compute (eltwise_sfpu_2_0.cpp, SFPU_TERNARY_OP) ─> out DFB ─>
///   Writer ─> DRAM(out) DRAM(in2) ─┘
///
/// @param mesh_device Device under test.
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_ternary_three_input_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuConfig& test_config) {
    const size_t per_buffer_byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto *device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = per_buffer_byte_size,
        .page_size = per_buffer_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input0_dram_buffer = CreateBuffer(dram_config);
    auto input1_dram_buffer = CreateBuffer(dram_config);
    auto input2_dram_buffer = CreateBuffer(dram_config);
    auto output_dram_buffer = CreateBuffer(dram_config);

    const size_t numel = per_buffer_byte_size / sizeof(bfloat16);
    const int seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto [packed_in0, packed_in1, packed_in2] =
        sfpu_util::generate_packed_sfpu_ternary_inputs(numel, test_config.sfpu_op, seed);

    auto in0 = unpack_vector<bfloat16, uint32_t>(packed_in0);
    auto in1 = unpack_vector<bfloat16, uint32_t>(packed_in1);
    auto in2 = unpack_vector<bfloat16, uint32_t>(packed_in2);
    std::vector<bfloat16> golden(in0.size());
    for (size_t i = 0; i < in0.size(); ++i) {
        golden[i] = sfpu_util::sfpu_ternary_function(test_config.sfpu_op, in0[i], in1[i], in2[i]);
    }
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_ternary_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_OP_WHERE_INCLUDE"] = "1";
    sfpu_defines["SFPU_TERNARY_OP"] = "1";

    std::vector<uint32_t> dest_buffer_data;
    if (device->arch() == ARCH::QUASAR) {
        const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
        const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
        const experimental::DFBSpecName IN2_DFB{"in2_dfb"};
        const experimental::DFBSpecName OUT_DFB{"out_dfb"};
        const experimental::KernelSpecName READER{"reader"};
        const experimental::KernelSpecName WRITER{"writer"};
        const experimental::KernelSpecName COMPUTE{"compute"};

        const auto node = extract_single_core_node(test_config, "Metal 2.0 ternary SFPU path");

        experimental::KernelSpec reader_spec{
            .unique_id = READER,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
            .num_threads = 1,
            .compiler_options = {.defines = {{"LOAD_BUF2_DATA", "1"}}},
            .dfb_bindings =
                {{
                     .dfb_spec_name = IN0_DFB,
                     .accessor_name = "in0",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN1_DFB,
                     .accessor_name = "in1",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN2_DFB,
                     .accessor_name = "in2",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 }},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"src0_addr",
                      "src0_bank_id",
                      "src1_addr",
                      "src1_bank_id",
                      "num_tiles",
                      "src2_addr",
                      "src2_bank_id"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .gen2_config =
                        experimental::DataMovementHardwareConfig::Gen2Config{
                            .disable_implicit_sync_for = {IN0_DFB, IN1_DFB, IN2_DFB}}},
        };

        experimental::KernelSpec compute_spec{
            .unique_id = COMPUTE,
            .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
            .num_threads = 1,
            .compiler_options = {.defines = to_kernel_defines(sfpu_defines)},
            .dfb_bindings =
                {{
                     .dfb_spec_name = IN0_DFB,
                     .accessor_name = "in0",
                     .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN1_DFB,
                     .accessor_name = "in1",
                     .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN2_DFB,
                     .accessor_name = "in2",
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
                {{"per_core_block_cnt", 1u}, {"per_core_block_size", static_cast<uint32_t>(test_config.num_tiles)}},
            .hw_config =
                experimental::ComputeHardwareConfig{
                    .math_approx_mode = test_config.approx_mode,
                },
        };

        experimental::ProgramSpec spec{
            .name = "sfpu_ternary_compute",
            .kernels = {reader_spec, make_writer_unary_quasar_spec(WRITER, OUT_DFB), compute_spec},
            .dataflow_buffers =
                {make_dfb_spec(IN0_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(IN1_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(IN2_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(OUT_DFB, test_config, test_config.l1_output_data_format)},
            .work_units = {experimental::WorkUnitSpec{
                .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
        };

        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = READER,
                .runtime_arg_values =
                    {{node,
                      {{"src0_addr", input0_dram_buffer->address()},
                       {"src0_bank_id", 0u},
                       {"src1_addr", input1_dram_buffer->address()},
                       {"src1_bank_id", 0u},
                       {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)},
                       {"src2_addr", input2_dram_buffer->address()},
                       {"src2_bank_id", 0u}}}},
            },
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values =
                    {{node,
                      {{"dst_addr", output_dram_buffer->address()},
                       {"bank_id", 0u},
                       {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
            },
            experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
        };
        dest_buffer_data = sfpu_quasar_run(
            mesh_device,
            spec,
            params,
            {{input0_dram_buffer, &packed_in0}, {input1_dram_buffer, &packed_in1}, {input2_dram_buffer, &packed_in2}},
            output_dram_buffer);
    } else {
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        distributed::MeshWorkload workload;
        tt_metal::Program program = tt_metal::CreateProgram();
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);

        // reader_binary.cpp with LOAD_BUF2_DATA:
        //   {src0_addr, src0_bank, src1_addr, src1_bank, num_tiles, src2_addr, src2_bank}
        vector<uint32_t> reader_rt_args = {
            (uint32_t)input0_dram_buffer->address(),
            0u,
            (uint32_t)input1_dram_buffer->address(),
            0u,
            (uint32_t)test_config.num_tiles,
            (uint32_t)input2_dram_buffer->address(),
            0u,
        };
        vector<uint32_t> writer_rt_args = {
            (uint32_t)output_dram_buffer->address(), 0u, (uint32_t)test_config.num_tiles};
        // eltwise_sfpu.cpp ternary path reads {per_core_block_cnt, per_core_block_size} as compile-time args.
        vector<uint32_t> compute_kernel_args = {1u, (uint32_t)test_config.num_tiles};

        for (const CoreRange& core_range : test_config.cores.ranges()) {
            auto make_input_cb = [&](tt::CBIndex idx) {
                return tt_metal::CircularBufferConfig(per_buffer_byte_size, {{idx, test_config.l1_input_data_format}})
                    .set_page_size(idx, test_config.tile_byte_size);
            };
            tt_metal::CreateCircularBuffer(program_, core_range, make_input_cb(tt::CBIndex::c_0));
            tt_metal::CreateCircularBuffer(program_, core_range, make_input_cb(tt::CBIndex::c_1));
            tt_metal::CreateCircularBuffer(program_, core_range, make_input_cb(tt::CBIndex::c_2));

            tt_metal::CircularBufferConfig l1_output_cb_config =
                tt_metal::CircularBufferConfig(
                    per_buffer_byte_size, {{tt::CBIndex::c_16, test_config.l1_output_data_format}})
                    .set_page_size(tt::CBIndex::c_16, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program_, core_range, l1_output_cb_config);

            auto reader_kernel = tt_metal::CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
                test_config.cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .defines = {{"LOAD_BUF2_DATA", "1"}}});

            auto writer_kernel = create_legacy_writer_kernel(program_, test_config);

            tt_metal::CreateKernel(
                program_,
                "tt_metal/kernels/compute/eltwise_sfpu.cpp",
                test_config.cores,
                tt_metal::ComputeConfig{
                    .dst_full_sync_en = true,
                    .math_approx_mode = test_config.approx_mode,
                    .compile_args = compute_kernel_args,
                    .defines = sfpu_defines});

            for (const CoreCoord& core_coord : core_range) {
                SetRuntimeArgs(program_, reader_kernel, core_coord, reader_rt_args);
                SetRuntimeArgs(program_, writer_kernel, core_coord, writer_rt_args);
            }
        }

        tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_in0);
        tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_in1);
        tt_metal::detail::WriteToBuffer(input2_dram_buffer, packed_in2);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    }

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

}  // namespace unit_tests::compute::sfpu

// Unary SFPU ops with no Quasar compute-API implementation yet: their
// compute_kernel_api.h / eltwise_unary headers are wrapped in #ifndef ARCH_QUASAR,
// so building the kernel would fail with "not declared in this scope". Skip them on
// Quasar so the suite reflects actual coverage instead of a hard kernel-build failure.
inline bool is_unary_sfpu_op_unsupported_on_quasar(const std::string& sfpu_op) {
    return sfpu_op == "gelu" || sfpu_op == "log" || sfpu_op == "tanh" || sfpu_op == "sign";
}

class SingleCoreSingleMeshDeviceSfpuParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};
TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (arch_ == tt::ARCH::QUASAR && is_unary_sfpu_op_unsupported_on_quasar(sfpu_op)) {
        GTEST_SKIP() << "SFPU unary op '" << sfpu_op << "' has no Quasar compute-API implementation";
    }

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(1, "rsqrt"),
        std::make_tuple(1, "mul_unary"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign"),
        std::make_tuple(4, "rsqrt"),
        std::make_tuple(4, "mul_unary")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

class SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (arch_ == tt::ARCH::QUASAR && is_unary_sfpu_op_unsupported_on_quasar(sfpu_op)) {
        GTEST_SKIP() << "SFPU unary op '" << sfpu_op << "' has no Quasar compute-API implementation";
    }
    if (((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "relu")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "exponential")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "log"))) {
        GTEST_SKIP();
    }
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = true};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(1, "rsqrt"),
        std::make_tuple(1, "mul_unary"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign"),
        std::make_tuple(4, "rsqrt"),
        std::make_tuple(4, "mul_unary")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

class SingleCoreSingleMeshDeviceSfpuParameterized32BitDestFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};
TEST_P(SingleCoreSingleMeshDeviceSfpuParameterized32BitDestFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (arch_ == tt::ARCH::QUASAR && is_unary_sfpu_op_unsupported_on_quasar(sfpu_op)) {
        GTEST_SKIP() << "SFPU unary op '" << sfpu_op << "' has no Quasar compute-API implementation";
    }

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false,
        .en_32bit_dest = true};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterized32BitDestFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(1, "rsqrt"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign"),
        std::make_tuple(4, "rsqrt")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

class SingleCoreSingleMeshDeviceSfpuParameterized32BitDestApproxFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuParameterized32BitDestApproxFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (arch_ == tt::ARCH::QUASAR && is_unary_sfpu_op_unsupported_on_quasar(sfpu_op)) {
        GTEST_SKIP() << "SFPU unary op '" << sfpu_op << "' has no Quasar compute-API implementation";
    }
    if (((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "relu")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "exponential")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "log"))) {
        GTEST_SKIP();
    }
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = true,
        .en_32bit_dest = true};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterized32BitDestApproxFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(1, "rsqrt"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign"),
        std::make_tuple(4, "rsqrt")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

// Binary SFPU parameterized test fixture (mirrors the unary fixture above).
//
// Each test instance is identified by (num_tiles, op_name). The op_name picks
// up macro substitutions from sfpu_binary_op_to_op_name and a host-side
// reference from sfpu_binary_function or get_binary_int_operation_result(). The name generator
// suffixes each instance with its op name (e.g. div_binary_1tiles) so a single op can be run standalone
// via --gtest_filter='*div_binary*' / '*add_int*' / '*mul_int*', while still sharing the single
// MeshDevice that LLKMeshDeviceFixture opens once per suite (no per-test device).
class SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture, TensixSfpuBinaryCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 ||
        MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Binary SFPU op test (div_binary / add_int / mul_int / gt_int) not fixed for WH/BH";
    }

    // add_int/mul_int: Int8 L1 inputs promoted to sign-mag Int32 output. div_binary stays bfloat16.
    const bool is_int8_op = unit_tests::sfpu_util::is_int8_binary_sfpu_op(sfpu_op);
    const tt::DataFormat data_format_input = is_int8_op ? tt::DataFormat::Int8 : tt::DataFormat::Float16_b;
    const tt::DataFormat data_format_output = is_int8_op ? tt::DataFormat::Int32 : tt::DataFormat::Float16_b;
    const size_t tile_byte_size = is_int8_op ? tt::tile_size(tt::DataFormat::Int8) : 2 * 32 * 32;

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = tile_byte_size,
        .l1_input_data_format = data_format_input,
        .l1_output_data_format = data_format_output,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing binary SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_binary_two_input_buffer(devices_.at(id), test_config));
    }
}

// TODO: BinarySFPU ops here can only do 1 tile due to the hardcoding in the macros to indicies (0,1,2)
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuBinaryCompute,
    SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "div_binary"),
        std::make_tuple(1, "mul_float"),
        std::make_tuple(1, "add_int"),
        std::make_tuple(1, "mul_int"),
        std::make_tuple(1, "gt_int"),
        std::make_tuple(1, "binary_max"),
        std::make_tuple(1, "binary_min"),
        std::make_tuple(1, "binary_max_int32"),
        std::make_tuple(1, "binary_min_int32")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

class SingleCoreSingleMeshDeviceSfpuTernaryParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuTernaryParameterizedFixture, TensixSfpuTernaryCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 ||
        MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Ternary where op test not fixed for WH/BH";
    }

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing ternary SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_ternary_three_input_buffer(devices_.at(id), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuTernaryCompute,
    SingleCoreSingleMeshDeviceSfpuTernaryParameterizedFixture,
    ::testing::Values(std::make_tuple(1, "where")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

}  // namespace tt::tt_metal
