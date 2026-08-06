// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::binary {
const map<std::string, std::string> binary_op_name_to_op_type = {
    {"add", "EltwiseBinaryType::ELWADD"},
    {"sub", "EltwiseBinaryType::ELWSUB"},
    {"mul", "EltwiseBinaryType::ELWMUL"},
    {"add_with_dest_reuse", "EltwiseBinaryType::ELWADD"},
    {"sub_with_dest_reuse", "EltwiseBinaryType::ELWSUB"},
    {"mul_with_dest_reuse", "EltwiseBinaryType::ELWMUL"},
};
const map<std::string, std::string> binary_op_name_to_op_kernel = {
    {"add", "add_tiles"},
    {"sub", "sub_tiles"},
    {"mul", "mul_tiles"},
};

// SrcB broadcast requested from binary_dest_reuse_tiles. Only ROW and COL are reachable through
// that API: SCALAR + dest reuse is rejected by a static_assert in the LLK unpacker
// (tt_llk_blackhole/llk_lib/llk_unpack_A.h), and every broadcast mode is DEST_TO_SRCA-only because
// DEST_TO_SRCB routes the dest face into SrcB, which is the register the FPU broadcasts.
enum class BroadcastDim : std::uint8_t { NONE = 0, ROW = 1, COL = 2 };

const map<BroadcastDim, std::string> broadcast_dim_to_type = {
    {BroadcastDim::ROW, "BroadcastType::ROW"},
    {BroadcastDim::COL, "BroadcastType::COL"},
};

struct SingleCoreBinaryConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    size_t input_dram_byte_address = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreCoord core;
    std::string binary_op;
    // Only meaningful for the *_with_dest_reuse ops; see BroadcastDim.
    BroadcastDim broadcast_dim = BroadcastDim::NONE;
    bool acc_to_dest = false;
    // FP32 dest accumulation. This is *not* a define and not derived from the data formats: it is
    // the ComputeGen1Config::enable_32_bit_dest / ComputeGen2Config::enable_32_bit_dest field on the
    // compute KernelSpec's hw_config. MakeGen1ComputeConfig copies it into
    // ComputeConfig::fp32_dest_acc_en, and jit_build emits it as `constexpr bool DST_ACCUM_MODE`
    // (genfiles.cpp emit_compute_scalar_descriptors) -- a constexpr, not a macro, so the kernel's
    // `#if defined(DST_ACCUM_MODE)` branches are dead on the JIT path and only the LLK template
    // arguments see it. binary_dest_reuse_tiles forwards DST_ACCUM_MODE to llk_math_eltwise_binary
    // as is_fp32_dest_acc_en together with clear_fp32_dst_acc=true, which is what selects ZEROACC's
    // 32-bit addressing mode on the ELWMUL dest-reuse path.
    bool fp32_dest_acc = false;
    bool full_init = true;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    tt::tt_metal::Tile tile = tt::tt_metal::Tile({32, 32});
};

void set_math_fid_masks(
    std::uint16_t& srca_fid_mask, std::uint16_t& srcb_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
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

struct BinaryStimulus {
    std::vector<std::uint32_t> packed_input0;
    std::vector<std::uint32_t> packed_input1;
    std::vector<std::uint32_t> packed_input2;
    std::vector<std::uint32_t> packed_golden;
};

// A tile lives in L1 face-major: four 16x16 row-major faces ordered
// {top-left, top-right, bottom-left, bottom-right}. This test writes and reads raw buffers with no
// tilize/untilize pass, so the operand vectors are already in that layout and the broadcast has to
// be modelled there rather than in logical (row, column) space.
//
// ROW: the unpack MOP reloads the two top faces for the bottom face row (its end op resets the
//      SrcB z counter every outer iteration) and the FPU replicates row 0 of each SrcB face, so
//      every datum in the tile reads logical row 0.
// COL: the MOP pushes only the left face of each face row (one SrcB face per two math faces, with
//      CLR_B deferred to the end of the face row) and the FPU replicates column 0, so every datum
//      reads logical column 0.
// Both traced from tt_llk_blackhole/llk_lib/llk_unpack_A.h and llk_math_eltwise_binary.h.
static size_t srcb_broadcast_source_index(size_t index_in_tile, BroadcastDim dim) {
    const size_t face = index_in_tile / constants::FACE_HW;
    const size_t row_in_face = (index_in_tile % constants::FACE_HW) / constants::FACE_WIDTH;
    const size_t col_in_face = index_in_tile % constants::FACE_WIDTH;
    if (dim == BroadcastDim::ROW) {
        // Row 0 of the top face in this face's column: face 0 for the left half, face 1 for the right.
        const size_t face_col = face % 2;
        return (face_col * constants::FACE_HW) + col_in_face;
    }
    // COL: column 0 of the left face in this face's row: face 0 for the top half, face 2 for the bottom.
    const size_t face_row = face / 2;
    return (face_row * 2 * constants::FACE_HW) + (row_in_face * constants::FACE_WIDTH);
}

// Replace every datum with the one the SrcB broadcast substitutes for it, tile by tile.
static std::vector<float> apply_srcb_broadcast(const std::vector<float>& src, BroadcastDim dim) {
    std::vector<float> broadcasted(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        const size_t tile_base = (i / constants::TILE_HW) * constants::TILE_HW;
        broadcasted[i] = src[tile_base + srcb_broadcast_source_index(i % constants::TILE_HW, dim)];
    }
    return broadcasted;
}

static BinaryStimulus generate_binary_stimulus(const SingleCoreBinaryConfig& test_config, bool is_quasar) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    BinaryStimulus s;
    // Use fixed seeds so test results are deterministic and reproducible.
    // Using wall-clock seeds caused intermittent tolerance failures depending on
    // which random inputs were drawn (see https://github.com/tenstorrent/tt-metal/issues/46284).
    s.packed_input0 =
        generate_packed_uniform_random_vector<std::uint32_t, bfloat16>(-1.0f, 1.0f, byte_size / sizeof(bfloat16), 0);
    s.packed_input1 =
        generate_packed_uniform_random_vector<std::uint32_t, bfloat16>(-1.0f, 1.0f, byte_size / sizeof(bfloat16), 1);
    s.packed_input2 =
        generate_packed_uniform_random_vector<std::uint32_t, bfloat16>(-1.0f, 1.0f, byte_size / sizeof(bfloat16), 2);

    auto input0 = unpack_vector<bfloat16, std::uint32_t>(s.packed_input0);
    auto input1 = unpack_vector<bfloat16, std::uint32_t>(s.packed_input1);
    auto input2 = unpack_vector<bfloat16, std::uint32_t>(s.packed_input2);

    std::vector<float> temp_golden(input0.size());
    std::uint16_t srca_fid_mask = 0xFFFF;
    std::uint16_t srcb_fid_mask = 0xFFFF;
    if (!is_quasar) {
        set_math_fid_masks(srca_fid_mask, srcb_fid_mask, test_config.math_fidelity);
    }

    std::transform(
        input0.begin(),
        input0.end(),
        input1.begin(),
        temp_golden.begin(),
        [&](const bfloat16& lhs, const bfloat16& rhs) {
            if (test_config.binary_op == "add") {
                return (static_cast<float>(lhs) + static_cast<float>(rhs));
            }
            if (test_config.binary_op == "sub") {
                return (static_cast<float>(lhs) - static_cast<float>(rhs));
            }
            if (test_config.binary_op == "mul") {
                return (
                    static_cast<float>(std::bit_cast<bfloat16>(
                        static_cast<std::uint16_t>(std::bit_cast<std::uint16_t>(lhs) & srca_fid_mask))) *
                    static_cast<float>(std::bit_cast<bfloat16>(
                        static_cast<std::uint16_t>(std::bit_cast<std::uint16_t>(rhs) & srcb_fid_mask))));
            }
            if (test_config.binary_op.find("with_dest_reuse") != std::string::npos) {
                return static_cast<float>(lhs);
            }
            TT_THROW("Unsupported binary_op={}", test_config.binary_op);
        });

    // binary_dest_reuse_tiles computes DEST [op] <CB operand>, i.e. input2 [op] input0, so input0 --
    // which temp_golden carries through unchanged for the dest-reuse ops -- is the operand that
    // feeds SrcB and therefore the one the FPU broadcasts.
    //
    // input0 is deliberately left fully random rather than masked down to the broadcast row/column:
    // every datum outside the broadcast source then differs from its broadcast value, so a dropped
    // BroadcastType shows up as a whole-tile mismatch instead of silently still matching.
    //
    // test_config.fp32_dest_acc deliberately does not change the golden. Both of the effects it
    // normally introduces are provably no-ops for this test's operand shapes:
    //
    //  - "DEST holds float32 and is only narrowed by the packer": the golden already accumulates in
    //    float and narrows exactly once, at the pack_vector<uint32_t, bfloat16> below. With a 16-bit
    //    DEST the FPU result is instead rounded to bf16 on the DEST write and the packer is then an
    //    identity conversion (pack_src_format for a Float16_b CB is Float16_b either way, see
    //    get_single_pack_src_format), so it is also one rounding. For ELWMUL the multi-pass fidelity
    //    accumulation is the only place the two differ, and fp32 DEST is strictly the more accurate
    //    of the two against this exact-float golden -- it cannot turn a passing case into a failure.
    //
    //  - "MOVD2A reads the high 16 bits of a 32-bit DEST word": true here (a Float16_b CB keeps
    //    unpack_dst_format == Float16_b even under fp32 dest acc -- get_unpack_dst_formats only
    //    consults unpack_conditional_dst_format for Float32 CBs -- so SrcA stays 16-bit-configured
    //    and move_d2a_fixed_face's TTI_MOVD2A(0, ...) truncates). But the only thing ever written to
    //    DEST before that MOVD2A is copy_tile of the bf16 input2 tile, i.e. a bf16 value widened to
    //    float32 with its low 16 mantissa bits zero. Truncating that back to its high half is
    //    lossless, so the tt-llk _dest_to_src_reg model would be an identity here. It would only
    //    start to matter if this test folded more than one op into the same DEST tile.
    if (test_config.broadcast_dim != BroadcastDim::NONE) {
        TT_FATAL(
            test_config.binary_op.find("_with_dest_reuse") != std::string::npos,
            "Broadcast is only modelled for the dest-reuse ops, got binary_op={}",
            test_config.binary_op);
        const auto tile_shape = test_config.tile.get_tile_shape();
        TT_FATAL(
            tile_shape[0] == constants::TILE_HEIGHT && tile_shape[1] == constants::TILE_WIDTH,
            "Broadcast with dest reuse requires a full 32x32 tile; COL needs all four faces");
        temp_golden = apply_srcb_broadcast(temp_golden, test_config.broadcast_dim);
    }

    std::vector<bfloat16> golden(input0.size());
    std::transform(
        input2.begin(), input2.end(), temp_golden.begin(), golden.begin(), [&](const bfloat16& lhs, const float& rhs) {
            if (test_config.acc_to_dest || test_config.binary_op == "add_with_dest_reuse") {
                return (static_cast<float>(lhs) + rhs);
            }
            if (test_config.binary_op == "sub_with_dest_reuse") {
                return (static_cast<float>(lhs) - rhs);
            }
            if (test_config.binary_op == "mul_with_dest_reuse") {
                return (
                    static_cast<float>(std::bit_cast<bfloat16>(
                        static_cast<std::uint16_t>(std::bit_cast<std::uint16_t>(lhs) & srca_fid_mask))) *
                    static_cast<float>(std::bit_cast<bfloat16>(
                        static_cast<std::uint16_t>(std::bit_cast<std::uint16_t>(bfloat16(rhs)) & srcb_fid_mask))));
            }
            return rhs;
        });
    s.packed_golden = pack_vector<std::uint32_t, bfloat16>(golden);
    return s;
}

// Four DRAM buffers: 3 inputs + 1 output.
struct BinaryBuffers {
    std::shared_ptr<distributed::MeshBuffer> input0;
    std::shared_ptr<distributed::MeshBuffer> input1;
    std::shared_ptr<distributed::MeshBuffer> input2;
    std::shared_ptr<distributed::MeshBuffer> output;
};

static BinaryBuffers create_and_populate_binary_buffers(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    const distributed::MeshCoordinate& zero_coord,
    size_t byte_size,
    BinaryStimulus& stimulus) {
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    BinaryBuffers buffers;
    buffers.input0 = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    buffers.input1 = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    buffers.input2 = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    buffers.output = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    distributed::WriteShard(cq, buffers.input0, stimulus.packed_input0, zero_coord, false);
    distributed::WriteShard(cq, buffers.input1, stimulus.packed_input1, zero_coord, false);
    distributed::WriteShard(cq, buffers.input2, stimulus.packed_input2, zero_coord, false);

    return buffers;
}

static bool read_and_validate_binary_result(
    distributed::MeshCommandQueue& cq,
    const std::shared_ptr<distributed::MeshBuffer>& output_dram_buffer,
    const distributed::MeshCoordinate& zero_coord,
    const BinaryStimulus& stimulus) {
    std::vector<std::uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, output_dram_buffer, zero_coord, false);

    return is_close_packed_vectors<bfloat16, std::uint32_t>(
        dest_buffer_data, stimulus.packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, 0.0155f);
        });
}

static std::map<std::string, std::string> build_binary_defines(const SingleCoreBinaryConfig& test_config) {
    std::map<std::string, std::string> defines = {
        {"ELTWISE_OP_TYPE", binary_op_name_to_op_type.at(test_config.binary_op)}};
    if (test_config.binary_op.find("_with_dest_reuse") != std::string::npos) {
        defines["ELTWISE_DEST_REUSE_TYPE"] = "EltwiseBinaryReuseDestType::DEST_TO_SRCA";
        // Left undefined for the non-broadcast cases so the kernel keeps instantiating the
        // two-template-argument form of binary_dest_reuse_tiles{,_init}.
        if (test_config.broadcast_dim != BroadcastDim::NONE) {
            defines["ELTWISE_BCAST_TYPE"] = broadcast_dim_to_type.at(test_config.broadcast_dim);
        }
    } else {
        defines["ELTWISE_OP"] = binary_op_name_to_op_kernel.at(test_config.binary_op);
        if (test_config.full_init) {
            defines["FULL_INIT"] = "1";
        }
        if (test_config.acc_to_dest) {
            defines["LOAD_BUF2_DATA"] = "1";
            defines["ACC_TO_DEST"] = "1";
        }
        defines["ELTWISE_OP_INIT"] = defines["ELTWISE_OP"] + "_init";
        if (test_config.binary_op == "mul") {
            defines["MUL_TILES_WITH_DST_ACCUM"] = "1";
        }
    }
    return defines;
}

/// @brief Does Dramx2 --> Reader --> DFB --> Binary Compute --> DFB --> Writer --> Dram
/// @param mesh_device - The mesh device on which to run the test
/// @param test_config - Configuration of the test -- see SingleCoreBinaryConfig
/// @return true if the test passed, false otherwise
bool single_core_binary(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SingleCoreBinaryConfig& test_config) {
    const bool is_quasar = MetalContext::instance().get_cluster().arch() == ARCH::QUASAR;
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    const experimental::NodeCoord node{
        static_cast<std::uint32_t>(test_config.core.x), static_cast<std::uint32_t>(test_config.core.y)};

    // Math-fidelity masks model WH/BH LLK behavior; Quasar HW does not apply them.
    auto stimulus = generate_binary_stimulus(test_config, is_quasar);
    auto buffers = create_and_populate_binary_buffers(mesh_device, cq, zero_coord, byte_size, stimulus);
    auto& input0_dram_buffer = buffers.input0;
    auto& input1_dram_buffer = buffers.input1;
    auto& input2_dram_buffer = buffers.input2;
    auto& output_dram_buffer = buffers.output;

    auto defines_map = build_binary_defines(test_config);
    experimental::KernelSpec::CompilerOptions::Defines defines;
    for (auto& kv : defines_map) {
        defines.emplace(kv.first, kv.second);
    }

    const experimental::DFBSpecName INP0_DFB{"inp0_dfb"};
    const experimental::DFBSpecName INP1_DFB{"inp1_dfb"};
    const experimental::DFBSpecName INP2_DFB{"inp2_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_input_dfb = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = static_cast<std::uint32_t>(test_config.tile_byte_size),
            .num_entries = static_cast<std::uint32_t>(test_config.num_tiles),
            .data_format_metadata = test_config.l1_input_data_format,
            .tile_format_metadata = test_config.tile,
        };
    };

    experimental::DataflowBufferSpec inp0_dfb_spec = make_input_dfb(INP0_DFB);
    experimental::DataflowBufferSpec inp1_dfb_spec = make_input_dfb(INP1_DFB);
    experimental::DataflowBufferSpec inp2_dfb_spec = make_input_dfb(INP2_DFB);
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = static_cast<std::uint32_t>(test_config.tile_byte_size),
        .num_entries = static_cast<std::uint32_t>(test_config.num_tiles),
        .data_format_metadata = test_config.l1_output_data_format,
        .tile_format_metadata = test_config.tile,
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

            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP2_DFB,
                 .accessor_name = "in2",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles", "src2_addr", "src2_bank_id"}},
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

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    // enable_32_bit_dest is the only route to fp32 dest accumulation: it becomes
    // ComputeConfig::fp32_dest_acc_en and then the generated `constexpr bool DST_ACCUM_MODE`.
    // No explicit unpack_modes entry is needed here -- program_spec only demands one for a *Float32*
    // DFB under enable_32_bit_dest, and every DFB in this test is Float16_b.
    experimental::ComputeHardwareConfig compute_hw_config;
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        compute_hw_config = experimental::ComputeGen2Config{
            .fpu_math_fidelity = test_config.math_fidelity,
            .enable_32_bit_dest = test_config.fp32_dest_acc,
        };
    } else {
        compute_hw_config = experimental::ComputeGen1Config{
            .fpu_math_fidelity = test_config.math_fidelity,
            .enable_32_bit_dest = test_config.fp32_dest_acc,
        };
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP2_DFB,
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
        .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt", "per_core_block_size", "acc_to_dst"}},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_core_binary",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {inp0_dfb_spec, inp1_dfb_spec, inp2_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    const std::uint32_t num_tiles_u = static_cast<std::uint32_t>(test_config.num_tiles);
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", input0_dram_buffer->address()},
                 {"src0_bank_id", 0u},
                 {"src1_addr", input1_dram_buffer->address()},
                 {"src1_bank_id", 0u},
                 {"num_tiles", num_tiles_u},
                 {"src2_addr", input2_dram_buffer->address()},
                 {"src2_bank_id", 0u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"dst_addr", output_dram_buffer->address()}, {"bank_id", 0u}, {"num_tiles", num_tiles_u}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = COMPUTE,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"per_core_block_cnt", num_tiles_u}, {"per_core_block_size", 1u}, {"acc_to_dst", 0u}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    auto* dev = mesh_device->get_devices()[0];
    tt_metal::detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    return read_and_validate_binary_result(cq, output_dram_buffer, zero_coord, stimulus);
}

// Streamed into GTEST_SKIP() by every dest-reuse + broadcast test; see the block comment above the
// first of them for the per-arch reasoning.
constexpr const char* kDestReuseBroadcastSkipReason =
    "binary_dest_reuse_tiles with a SrcB broadcast is only supported on Blackhole";

/// @brief Shared body for the dest-reuse + SrcB-broadcast cases: run one op / broadcast dimension /
/// dest-accumulation width over the four supported MathFidelity levels (index 1 is unused),
/// mirroring the non-broadcast dest-reuse tests above.
///
/// fp32_dest_acc is a separate axis rather than a second helper because it is orthogonal to the op
/// and the broadcast dimension, and because it is the axis the motivating consumer (the Welford
/// group_norm kernel) actually runs in: it builds its compute kernel with fp32_dest_acc_en = true.
/// It is also the axis that exposed the ZEROACC addressing-mode bug one layer down in tt-llk, where
/// a 16-bit-addressed ZEROACC against a 32-bit DEST corrupted everything past the first 128 datums
/// per block on the ELWMUL dest-reuse path. tt-metal's binary_dest_reuse_tiles passes
/// clear_fp32_dst_acc=true alongside DST_ACCUM_MODE, so these cases are the regression guard for
/// that pairing rather than a reproducer for it.
void run_dest_reuse_broadcast_sweep(
    const std::vector<std::shared_ptr<distributed::MeshDevice>>& devices,
    const std::string& binary_op,
    BroadcastDim broadcast_dim,
    bool fp32_dest_acc) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        SingleCoreBinaryConfig test_config = {
            .num_tiles = 4,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = binary_op,
            .broadcast_dim = broadcast_dim,
            .fp32_dest_acc = fp32_dest_acc,
            .math_fidelity = MathFidelity(i),
        };
        log_info(
            tt::LogTest,
            "binary_op = {}, broadcast = {}, fp32_dest_acc = {}, Math Fidelity = {}",
            binary_op,
            broadcast_dim_to_type.at(broadcast_dim),
            fp32_dest_acc,
            i);
        for (auto& device : devices) {
            ASSERT_TRUE(single_core_binary(device, test_config));
        }
    }
}

}  // namespace unit_tests::compute::binary

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileAdd) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileSub) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileMul) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileAddFullInit) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add",
            .full_init = true,
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileSubFullInit) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub",
            .full_init = true,
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreSingleTileMulFullInit) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul",
            .full_init = true,
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 1;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAddWithDestReuse) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add_with_dest_reuse",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSubWithDestReuse) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub_with_dest_reuse",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMulWithDestReuse) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul_with_dest_reuse",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

// binary_dest_reuse_tiles with a SrcB broadcast (the cases below) is Blackhole-only:
//   - Wormhole's llk_unpack_A broadcast MOP issues fewer SrcA dvalids than
//     move_d2a_fixed_face's STALLWAIT(SRCA_VLD) consumes, so DEST_TO_SRCA hangs rather than
//     producing a mismatch.
//   - Quasar rejects broadcast + dest reuse at compile time.
//   - SCALAR is excluded on every arch by the static_assert in llk_unpack_A.h, and COL requires
//     num_faces == 4, i.e. a full 32x32 tile. So only ROW and COL x DEST_TO_SRCA are covered.
// The fixture stays LLKMeshDeviceFixtureSlowDispatchOnly (not LLKBlackholeSingleCardFixture)
// because this file drives programs with detail::LaunchProgram, which is slow-dispatch only, and
// because the merge-gate slow-dispatch jobs select tests by that exact suite name.
//
// Each op x broadcast dimension is run twice: once with a 16-bit DEST and once with fp32 dest
// accumulation (the *Fp32DestAcc cases). The second half is what the Welford group_norm kernel
// actually asks for, and it is the axis that turned a latent ZEROACC addressing-mode bug into an
// observable failure one layer down in tt-llk -- see run_dest_reuse_broadcast_sweep.
TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAddWithDestReuseBcastRow) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "add_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::ROW, /*fp32_dest_acc=*/false);
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAddWithDestReuseBcastCol) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "add_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::COL, /*fp32_dest_acc=*/false);
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSubWithDestReuseBcastRow) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "sub_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::ROW, /*fp32_dest_acc=*/false);
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSubWithDestReuseBcastCol) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "sub_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::COL, /*fp32_dest_acc=*/false);
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMulWithDestReuseBcastRow) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "mul_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::ROW, /*fp32_dest_acc=*/false);
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMulWithDestReuseBcastCol) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "mul_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::COL, /*fp32_dest_acc=*/false);
}

TEST_F(
    LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAddWithDestReuseBcastRowFp32DestAcc) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "add_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::ROW, /*fp32_dest_acc=*/true);
}

TEST_F(
    LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAddWithDestReuseBcastColFp32DestAcc) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "add_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::COL, /*fp32_dest_acc=*/true);
}

TEST_F(
    LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSubWithDestReuseBcastRowFp32DestAcc) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "sub_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::ROW, /*fp32_dest_acc=*/true);
}

TEST_F(
    LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSubWithDestReuseBcastColFp32DestAcc) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "sub_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::COL, /*fp32_dest_acc=*/true);
}

// The ELWMUL cases are the load-bearing ones for the fp32 axis: ELWMUL is the only dest-reuse op
// that reaches the per-face ZEROACC in eltwise_binary_run_with_dest_reuse (ELWADD/ELWSUB take the
// non-ZEROACC branch in llk_math_eltwise_binary.h), and it accumulates into DEST unconditionally on
// WH/BH, so a DEST face cleared with the wrong addressing width corrupts the result instead of
// being overwritten.
TEST_F(
    LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMulWithDestReuseBcastRowFp32DestAcc) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "mul_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::ROW, /*fp32_dest_acc=*/true);
}

TEST_F(
    LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMulWithDestReuseBcastColFp32DestAcc) {
    if (this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << unit_tests::compute::binary::kDestReuseBroadcastSkipReason;
    }
    unit_tests::compute::binary::run_dest_reuse_broadcast_sweep(
        this->devices_, "mul_with_dest_reuse", unit_tests::compute::binary::BroadcastDim::COL, /*fp32_dest_acc=*/true);
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAdd) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSub) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMul) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul",
            .math_fidelity = MathFidelity(i)};
        test_config.num_tiles = 4;
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileAddDestAcc) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .num_tiles = 4,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "add",
            .acc_to_dest = true,
            .math_fidelity = MathFidelity(i),
        };
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileSubDestAcc) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .num_tiles = 4,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "sub",
            .acc_to_dest = true,
            .math_fidelity = MathFidelity(i),
        };
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, TensixBinaryComputeSingleCoreMultiTileMulDestAcc) {
    for (std::uint8_t i = std::uint8_t(MathFidelity::LoFi); i <= std::uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) {
            continue;
        }
        unit_tests::compute::binary::SingleCoreBinaryConfig test_config = {
            .num_tiles = 4,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = CoreCoord(0, 0),
            .binary_op = "mul",
            .acc_to_dest = true,
            .math_fidelity = MathFidelity(i),
        };
        log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (auto& device : this->devices_) {
            ASSERT_TRUE(unit_tests::compute::binary::single_core_binary(device, test_config));
            // TODO: Remove early return once back-to-back tests are passing on Quasar
            if (this->arch_ == ARCH::QUASAR) {
                return;
            }
        }
    }
}

}  // namespace tt::tt_metal
