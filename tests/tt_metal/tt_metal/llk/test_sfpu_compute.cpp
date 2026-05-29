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
};

// Binary SFPU ops driven by `run_sfpu_binary_two_input_buffer`.
//
// Each entry maps an op name (the test parameter) to the kernel-side macro
// substitutions expanded by the SFPU binary compute kernel
// (`eltwise_binary_sfpu.cpp`, Metal 2.0 dataflow-buffer based):
//
//   * SFPU_OP_INIT_0  — runs once before the per-pair loop. Used to set up
//                       SFPU lookup tables / state (e.g. div reciprocal LUT).
//   * SFPU_OP_CHAIN_0 — runs once per (LHS, RHS) pair, inside an
//                       acquire/release section. By convention LHS lives at
//                       DST[0] and RHS at DST[1]; the result is written back
//                       to DST[0] so the packer reads from there.
//
// To add a new binary SFPU op: add an entry here, add a matching arm in
// sfpu_binary_function() for the golden compute, and (if its valid input
// range differs from div) add an arm in generate_packed_sfpu_binary_inputs().
const map<std::string, std::map<std::string, std::string>> sfpu_binary_op_to_op_name = {
    {"div_binary", {{"SFPU_OP_INIT_0", "div_binary_tile_init();"}, {"SFPU_OP_CHAIN_0", "div_binary_tile(0, 1, 0);"}}},
    // add_int: Int8 L1 inputs are promoted to sign-magnitude Int32 in DEST via copy_tile + fp32_dest_acc;
    // add_int_tile<Int32> (sign-mag on Quasar via ARCH_QUASAR) then adds in sign-mag space. Result in DST[0].
    {"add_int",
     {{"SFPU_OP_INIT_0", "add_int_tile_init();"}, {"SFPU_OP_CHAIN_0", "add_int_tile<DataFormat::Int32>(0, 1, 0);"}}},
};

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
    TT_THROW("Unsupported op_name in test");
}

// Reference implementation for float-typed binary SFPU ops. Integer ops
// (e.g. add_int) bypass this helper; golden is computed in run_sfpu_binary_two_input_buffer.
bfloat16 sfpu_binary_function(const std::string& op_name, const bfloat16& lhs, const bfloat16& rhs) {
    if (op_name == "div_binary") {
        return bfloat16(static_cast<float>(lhs) / static_cast<float>(rhs));
    }
    TT_THROW("Unsupported binary op_name in test");
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
    if (op_name == "div_binary") {
        auto lhs = generate_div_operand(numel, seed);
        auto rhs = generate_div_operand(numel, seed + 1);
        return {lhs, rhs};
    }
    else if (op_name == "add_int") {
        auto lhs = create_random_vector_of_int8(numel, seed);
        auto rhs = create_random_vector_of_int8(numel, seed + 1);
        return {lhs, rhs};
    }
    TT_THROW("Unsupported binary op_name in test");
}

bool is_close_packed_sfpu_output(
    const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const std::string& op_name) {
    if (op_name == "add_int") {
        return vec_a == vec_b;
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

    // Every existing parametrization of this test uses a single-core CoreRangeSet of {0, 0};
    // MakeProgramFromSpec models the kernel set per single-core WorkUnit.
    TT_FATAL(
        test_config.cores.ranges().size() == 1,
        "sfpu test expects a single CoreRange (got {})",
        test_config.cores.size());
    const CoreRange& core_range = *test_config.cores.ranges().begin();
    TT_FATAL(core_range.start_coord == core_range.end_coord, "sfpu test expects a single-core CoreRange");
    const CoreCoord core = core_range.start_coord;
    const experimental::metal2_host_api::NodeCoord node{core.x, core.y};

    constexpr const char* IN_DFB = "in_dfb";
    constexpr const char* OUT_DFB = "out_dfb";
    constexpr const char* READER = "reader";
    constexpr const char* WRITER = "writer";
    constexpr const char* COMPUTE = "compute";

    experimental::metal2_host_api::DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_input_data_format,
    };
    experimental::metal2_host_api::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = static_cast<uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_output_data_format,
    };

    experimental::metal2_host_api::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = IN_DFB,
            .local_accessor_name = "out",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"src_addr", "bank_id", "num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen1_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{
                        .disable_implicit_sync_for = {IN_DFB}}},
    };

    experimental::metal2_host_api::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"dst_addr", "bank_id", "num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen1_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{
                        .disable_implicit_sync_for = {OUT_DFB}}},
    };

    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : sfpu_defines) {
        compute_defines.emplace_back(k, v);
    }

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN_DFB,
                 .local_accessor_name = "in",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .local_accessor_name = "out",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .compile_time_arg_bindings =
            {{"per_core_block_cnt", static_cast<uint32_t>(test_config.num_tiles)}, {"per_core_block_dim", 1u}},
        .config_spec =
            experimental::metal2_host_api::ComputeConfiguration{
                .math_approx_mode = test_config.approx_mode,
            },
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "sfpu_compute",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = READER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"src_addr", input_dram_buffer->address()},
                       {"bank_id", 0u},
                       {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = WRITER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"dst_addr", output_dram_buffer->address()},
                       {"bank_id", 0u},
                       {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = COMPUTE,
        },
    };
    experimental::metal2_host_api::SetProgramRunParameters(program_run, params);

    tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

/// High-level flow:
///
///   DRAM(LHS) ─┐
///              ├─> Reader ─> in0/in1 DFB ─> SFPU Compute (eltwise_binary_sfpu.cpp) ─> out DFB ─> Writer ─> DRAM(out)
///   DRAM(RHS) ─┘
///
/// @param mesh_device Device under test.
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_binary_two_input_buffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuConfig& test_config) {
    // Input/output tiles may differ in width (add_int: Int8 in, Int32 out), so size each buffer by its own format.
    const size_t input_tile_byte_size = tt::tile_size(test_config.l1_input_data_format);
    const size_t output_tile_byte_size = tt::tile_size(test_config.l1_output_data_format);
    const size_t per_buffer_byte_size_input = test_config.num_tiles * input_tile_byte_size;
    const size_t per_buffer_byte_size_output = test_config.num_tiles * output_tile_byte_size;
    auto& cq = mesh_device->mesh_command_queue();

    tt::tt_metal::InterleavedBufferConfig dram_config_input{
        .device = mesh_device->get_devices()[0],
        .size = per_buffer_byte_size_input,
        .page_size = per_buffer_byte_size_input,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_config_output{
        .device = mesh_device->get_devices()[0],
        .size = per_buffer_byte_size_output,
        .page_size = per_buffer_byte_size_output,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    auto input0_dram_buffer = CreateBuffer(dram_config_input);
    auto input1_dram_buffer = CreateBuffer(dram_config_input);
    auto output_dram_buffer = CreateBuffer(dram_config_output);

    // add_int uses Int8 L1 inputs (promoted to sign-magnitude Int32 in dest); div_binary stays bfloat16.
    const bool is_int_op = (test_config.sfpu_op == "add_int");
    const size_t element_size = is_int_op ? sizeof(int8_t) : sizeof(bfloat16);
    const uint32_t numel = per_buffer_byte_size_input / element_size;
    const int seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto [packed_lhs, packed_rhs] = sfpu_util::generate_packed_sfpu_binary_inputs(numel, test_config.sfpu_op, seed);

    std::vector<uint32_t> packed_golden;
    if (is_int_op) {
        // HW interprets each Int8 datum as sign-mag on the wire (bit7=sign, bits[6:0]=mag),
        // promotes to sign-mag Int32 in dest via copy_tile + fp32_dest_acc, then SFPU add
        // (sign-mag on Quasar via ARCH_QUASAR) writes sign-mag Int32 to DRAM.
        TT_FATAL(packed_lhs.size() == packed_rhs.size(), "lhs/rhs packed size mismatch");
        const size_t num_words = packed_lhs.size() * 4;
        packed_golden.resize(num_words);
        for (size_t w = 0; w < packed_lhs.size(); ++w) {
            for (int b = 0; b < 4; ++b) {
                const auto lhs_byte = static_cast<uint8_t>((packed_lhs[w] >> (b * 8)) & 0xFF);
                const auto rhs_byte = static_cast<uint8_t>((packed_rhs[w] >> (b * 8)) & 0xFF);
                const int sum = sign_mag_byte_to_int8(lhs_byte) + sign_mag_byte_to_int8(rhs_byte);
                packed_golden[w * 4 + b] = int32_to_sign_mag_word(static_cast<int32_t>(sum));
            }
        }
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
    if (is_int_op) {
        sfpu_defines["SFPU_OP_BINARY_ADD_INT_INCLUDE"] = "1";
    } else {
        sfpu_defines["SFPU_OP_BINARY_DIV_INCLUDE"] = "1";
    }

    TT_FATAL(
        test_config.cores.ranges().size() == 1,
        "sfpu binary test expects a single CoreRange (got {})",
        test_config.cores.size());
    const CoreRange& core_range = *test_config.cores.ranges().begin();
    TT_FATAL(core_range.start_coord == core_range.end_coord, "sfpu binary test expects a single-core CoreRange");
    const CoreCoord core = core_range.start_coord;
    const experimental::metal2_host_api::NodeCoord node{core.x, core.y};

    constexpr const char* IN0_DFB = "in0_dfb";
    constexpr const char* IN1_DFB = "in1_dfb";
    constexpr const char* OUT_DFB = "out_dfb";
    constexpr const char* READER = "reader";
    constexpr const char* WRITER = "writer";
    constexpr const char* COMPUTE = "compute";

    auto make_input_dfb = [&](const char* name) {
        return experimental::metal2_host_api::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = static_cast<uint32_t>(input_tile_byte_size),
            .num_entries = static_cast<uint32_t>(test_config.num_tiles),
            .data_format_metadata = test_config.l1_input_data_format,
        };
    };

    experimental::metal2_host_api::DataflowBufferSpec in0_dfb_spec = make_input_dfb(IN0_DFB);
    experimental::metal2_host_api::DataflowBufferSpec in1_dfb_spec = make_input_dfb(IN1_DFB);
    experimental::metal2_host_api::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = static_cast<uint32_t>(output_tile_byte_size),
        .num_entries = static_cast<uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_output_data_format,
    };

    experimental::metal2_host_api::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .local_accessor_name = "in0",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1_DFB,
                 .local_accessor_name = "in1",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .runtime_arguments_schema =
            {.named_runtime_args = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen1_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{
                        .disable_implicit_sync_for = {IN0_DFB, IN1_DFB}}},
    };

    experimental::metal2_host_api::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "in",
            .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"dst_addr", "bank_id", "num_tiles"}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen1_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{
                        .disable_implicit_sync_for = {OUT_DFB}}},
    };

    experimental::metal2_host_api::KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : sfpu_defines) {
        compute_defines.emplace_back(k, v);
    }

    experimental::metal2_host_api::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary_sfpu.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = std::move(compute_defines)},
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .local_accessor_name = "in0",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1_DFB,
                 .local_accessor_name = "in1",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .local_accessor_name = "out",
                 .endpoint_type = experimental::metal2_host_api::KernelSpec::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::metal2_host_api::DFBAccessPattern::STRIDED,
             }},
        .compile_time_arg_bindings =
            {{"per_core_block_cnt", 1u}, {"per_core_block_size", static_cast<uint32_t>(test_config.num_tiles)}},
        .config_spec =
            experimental::metal2_host_api::ComputeConfiguration{
                .fp32_dest_acc_en = is_int_op,
                .math_approx_mode = test_config.approx_mode,
            },
    };

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "sfpu_binary_compute",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in0_dfb_spec, in1_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& program_run = workload.get_programs().at(device_range);

    const uint32_t num_tiles_u = static_cast<uint32_t>(test_config.num_tiles);
    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = READER,
            .named_runtime_args =
                {{.node = node,
                  .args =
                      {{"src0_addr", input0_dram_buffer->address()},
                       {"src0_bank_id", 0u},
                       {"src1_addr", input1_dram_buffer->address()},
                       {"src1_bank_id", 0u},
                       {"num_tiles", num_tiles_u}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = WRITER,
            .named_runtime_args =
                {{.node = node,
                  .args = {{"dst_addr", output_dram_buffer->address()}, {"bank_id", 0u}, {"num_tiles", num_tiles_u}}}},
        },
        experimental::metal2_host_api::ProgramRunParams::KernelRunParams{
            .kernel_spec_name = COMPUTE,
        },
    };
    experimental::metal2_host_api::SetProgramRunParameters(program_run, params);

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_lhs);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_rhs);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

}  // namespace unit_tests::compute::sfpu
class SingleCoreSingleMeshDeviceSfpuParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};
TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

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
        std::make_tuple(4, "rsqrt")));

class SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

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
        std::make_tuple(4, "rsqrt")));

// Binary SFPU parameterized test fixture (mirrors the unary fixture above).
//
// Each test instance is identified by (num_tiles, op_name). The op_name picks
// up macro substitutions from sfpu_binary_op_to_op_name and a host-side
// reference from sfpu_binary_function. The name generator suffixes each instance
// with its op name (e.g. div_binary_1tiles) so a single op can be run standalone
// via --gtest_filter='*div_binary*' / '*add_int*', while still sharing the single
// MeshDevice that LLKMeshDeviceFixture opens once per suite (no per-test device).
class SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture, TensixSfpuBinaryCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 ||
        MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Binary SFPU op test (div_binary / add_int) not fixed for WH/BH";
    }

    // add_int: Int8 L1 inputs promoted to sign-mag Int32 output. div_binary stays bfloat16.
    const bool is_int_op = (sfpu_op == "add_int");
    const tt::DataFormat data_format_input = is_int_op ? tt::DataFormat::Int8 : tt::DataFormat::Float16_b;
    const tt::DataFormat data_format_output = is_int_op ? tt::DataFormat::Int32 : tt::DataFormat::Float16_b;
    const size_t tile_byte_size = is_int_op ? tt::tile_size(tt::DataFormat::Int8) : 2 * 32 * 32;

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
    ::testing::Values(std::make_tuple(1, "div_binary"), std::make_tuple(1, "add_int")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

}  // namespace tt::tt_metal
