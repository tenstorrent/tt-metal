// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/types/arch.hpp>

#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"

// Guards the math-side dest addressing of the 32-bit unpack-to-dest broadcast
// (_llk_math_eltwise_unary_datacopy_, the `unpack_to_dest && is_32bit_input` branch).
//
// That branch's MOVD2B/MOVB2D sequence addresses Dst with immediates (dst_index * 64 + row), and
// the hardware adds DEST_TARGET_REG_CFG_MATH_Offset to every Dst access. SrcRegs-path datacopies
// and SFPU ops program that offset to (dest_bank_base + their_tile_index * 64) and leave it there;
// the unpack-to-dest branch never reprograms it (its set_dst_write_addr<..., DestReg> only
// mailboxes the write address to the UNPACKER). So a broadcast that runs after an op which
// targeted a different DST index is displaced by the stale offset: it reads its source rows from,
// and writes its result over, the WRONG dest tile, while its own dst slot keeps the raw
// unbroadcast tile the unpacker deposited.
//
// Every other in-tree caller broadcasts into dst_index 0 right after an op that also targeted
// tile 0, so the leftover offset happens to be correct and the bug stays latent. This test is the
// minimal sequence that breaks the coincidence: copy_tile to DST[1], then a 32-bit unpack-to-dest
// op into DST[0], one core, one tile pair per acquire (see the compute kernel for the full
// walkthrough). Four modes run: the ROW / COL / SCALAR broadcasts (all three sequences in that
// branch share the same Dst addressing) and NONE, a plain 32-bit unpack-to-dest copy_tile.
//
// Only ROW / COL / SCALAR reproduce the bug. NONE is a smoke check that the plain 32-bit copy still
// lands correctly with a dirty offset, not a regression guard on the fix -- it passes with or
// without it on both architectures. With the broadcast sequence compiled out, the only instruction
// left consuming the offset is Blackhole's budabackend/#2730 ZEROACC zero-flag-clear loop, and that
// takes an absolute block index: the offset reaches it through the bank half-select alone, which no
// tile-granular offset writer can flip (fp32 dest tops out at 192 + 15 against a 512-row bank).
// Each mode repeats the sequence over several acquires so it runs against both dest banks, i.e.
// both values of get_dest_buffer_base().
//
//   c_0 (Float16_b): a[r][c] = r + 1               -> copied to DST[1]
//   c_1 (Float32)  : b[r][c] = 100 + r + 41*c      -> broadcast/copied into DST[0]
//   out (Float32)  : per iteration, tile 0 = DST[0], tile 1 = DST[1]
//
//   expected: out0 = broadcast of b (ROW: b[0][c]; COL: b[r][0]; SCALAR: b[0][0]; NONE: b itself),
//             out1[r][c] = r + 1 (copy untouched)
//   bug:      out0 = raw b tile (never broadcast), out1 = broadcast of the COPIED tile over itself
//
// All stimulus values are integers, exactly representable in Float16_b and Float32 through every
// conversion on the path, so the comparison is bit-exact.
//
// Quasar runs the same sequence through the Metal 2.0 spec (run_unpack_to_dest_bcast_dst_offset_quasar) with the
// Float32 DFB marked UnpackToDest: there the unpacker places the broadcast source at its own DEST index and math
// broadcasts it in place, so the guarded property is the same, DST[0] = broadcast and DST[1] = copy, in both banks.

namespace tt::tt_metal {

namespace unit_tests::compute::unpack_to_dest_bcast {

constexpr uint32_t kTileHW = 32 * 32;
// Acquire/pack iterations per run: enough to visit both dest banks twice, so every mode runs
// against both values of get_dest_buffer_base() and against rows the packer has already drained.
constexpr uint32_t kIters = 4;
constexpr const char* kComputeKernel =
    "tests/tt_metal/tt_metal/test_kernels/compute/unpack_to_dest_row_bcast_after_copy.cpp";

std::vector<uint32_t> reinterpret_fp32_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size());
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(out.data(), in.data(), in.size() * sizeof(float));
    return out;
}

std::vector<uint32_t> pack_bf16_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = pack_two_bfloat16_into_uint32({bfloat16(in[2 * i]), bfloat16(in[2 * i + 1])});
    }
    return out;
}

// Matches BCAST_DIM_VAL in the compute kernel. NONE = plain 32-bit unpack-to-dest copy_tile.
enum class BcastDim : uint32_t { ROW = 0, COL = 1, SCALAR = 2, NONE = 3 };

const char* bcast_dim_name(BcastDim dim) {
    switch (dim) {
        case BcastDim::ROW: return "ROW";
        case BcastDim::COL: return "COL";
        case BcastDim::SCALAR: return "SCALAR";
        case BcastDim::NONE: return "NONE (plain copy)";
    }
    return "?";
}

// Tilized inputs and golden, repeated kIters times (the same tile pair every iteration).
struct Stimulus {
    std::vector<uint32_t> a_all;         // c_0 / copy_in: Float16_b tiles
    std::vector<uint32_t> b_all;         // c_1 / bcast_in: Float32 tiles
    std::vector<uint32_t> golden_tiled;  // (DST[0], DST[1]) per iteration
};

Stimulus make_stimulus(BcastDim dim) {
    // Stimulus: integer values, exact in every format on the path; every element of b is distinct
    // so all three broadcast dimensions produce goldens that differ from the raw tile.
    std::vector<float> a_rm(kTileHW);  // c_0, copied to DST[1]
    std::vector<float> b_rm(kTileHW);  // c_1, broadcast into DST[0]
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            a_rm[r * 32 + c] = static_cast<float>(r + 1);
            b_rm[r * 32 + c] = static_cast<float>(100 + r + 41 * c);
        }
    }
    std::vector<float> golden0_rm(kTileHW);  // b broadcast along `dim` (NONE: b copied verbatim)
    std::vector<float> golden1_rm(kTileHW);  // the copied a tile
    auto bcast_src_index = [dim](uint32_t r, uint32_t c) -> uint32_t {
        switch (dim) {
            case BcastDim::ROW: return 0 * 32 + c;
            case BcastDim::COL: return r * 32 + 0;
            case BcastDim::SCALAR: return 0;
            case BcastDim::NONE: return r * 32 + c;
        }
        return 0;
    };
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            golden0_rm[r * 32 + c] = b_rm[bcast_src_index(r, c)];
            golden1_rm[r * 32 + c] = static_cast<float>(r + 1);
        }
    }

    ::unit_tests::compute::GoldenConfig bf16_cfg{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 2};
    ::unit_tests::compute::GoldenConfig fp32_cfg{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};

    auto a_tiled = ::unit_tests::compute::gold_standard_tilize(pack_bf16_as_u32(a_rm), bf16_cfg);
    auto b_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(b_rm), fp32_cfg);
    auto golden0_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(golden0_rm), fp32_cfg);
    auto golden1_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(golden1_rm), fp32_cfg);

    auto repeat = [&](const std::vector<uint32_t>& v) {
        std::vector<uint32_t> out;
        out.reserve(v.size() * kIters);
        for (uint32_t i = 0; i < kIters; ++i) {
            out.insert(out.end(), v.begin(), v.end());
        }
        return out;
    };
    std::vector<uint32_t> golden_pair = golden0_tiled;
    golden_pair.insert(golden_pair.end(), golden1_tiled.begin(), golden1_tiled.end());
    return Stimulus{.a_all = repeat(a_tiled), .b_all = repeat(b_tiled), .golden_tiled = repeat(golden_pair)};
}

bool check_result(const std::vector<uint32_t>& device_tiled, const std::vector<uint32_t>& golden_tiled) {
    if (device_tiled.size() != golden_tiled.size()) {
        log_error(tt::LogTest, "Size mismatch: device={} golden={}", device_tiled.size(), golden_tiled.size());
        return false;
    }

    size_t num_mismatches = 0;
    constexpr size_t max_report = 8;
    for (size_t i = 0; i < device_tiled.size(); ++i) {
        if (device_tiled[i] != golden_tiled[i]) {
            if (num_mismatches < max_report) {
                const size_t tile = i / kTileHW;
                log_error(
                    tt::LogTest,
                    "Mismatch: iteration {} DST[{}] tiled idx {}: golden={} device={}",
                    tile / 2,
                    tile % 2,
                    i % kTileHW,
                    std::bit_cast<float>(golden_tiled[i]),
                    std::bit_cast<float>(device_tiled[i]));
            }
            ++num_mismatches;
        }
    }
    if (num_mismatches != 0) {
        log_error(
            tt::LogTest,
            "Total mismatches: {}/{} (out tile 0 wrong => the broadcast never reached its dst slot; out tile 1 "
            "wrong => the broadcast was displaced onto the neighboring tile)",
            num_mismatches,
            device_tiled.size());
        return false;
    }
    return true;
}

std::map<std::string, std::string> compute_defines(BcastDim dim) {
    return {{"BCAST_DIM_VAL", std::to_string(static_cast<uint32_t>(dim))}, {"NUM_ITERS_VAL", std::to_string(kIters)}};
}

// Wormhole B0 / Blackhole: legacy circular buffers and ComputeConfig::unpack_to_dest_mode.
bool run_unpack_to_dest_bcast_dst_offset(const std::shared_ptr<distributed::MeshDevice>& mesh_device, BcastDim dim) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    Program program = tt_metal::CreateProgram(*mesh_device);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    CoreCoord core = {0, 0};
    const uint32_t bf16_tile_bytes = kTileHW * 2;
    const uint32_t fp32_tile_bytes = kTileHW * 4;

    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false},
            mesh_device.get());
    };
    auto src_copy_buffer = make_dram(kIters * bf16_tile_bytes);   // c_0: one Float16_b tile per iteration
    auto src_bcast_buffer = make_dram(kIters * fp32_tile_bytes);  // c_1: one Float32 tile per iteration
    auto dst_buffer = make_dram(kIters * 2 * fp32_tile_bytes);    // c_16: (DST[0], DST[1]) per iteration

    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(bf16_tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, bf16_tile_bytes));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(fp32_tile_bytes, {{tt::CBIndex::c_1, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_1, fp32_tile_bytes));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(2 * fp32_tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_16, fp32_tile_bytes));

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

    // Route c_1 onto the unpack-to-dest path: Float32 CB format alone is lowered to Tf32-in-SrcB
    // under fp32_dest_acc_en (get_unpack_dst_formats), which never reaches the branch under test.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[tt::CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;

    tt_metal::CreateKernel(
        program_,
        kComputeKernel,
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .defines = compute_defines(dim)});

    Stimulus stim = make_stimulus(dim);
    distributed::WriteShard(cq, src_copy_buffer, stim.a_all, zero_coord);
    distributed::WriteShard(cq, src_bcast_buffer, stim.b_all, zero_coord);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {static_cast<uint32_t>(src_copy_buffer->address()),
         0u,
         static_cast<uint32_t>(src_bcast_buffer->address()),
         0u,
         kIters});
    tt_metal::SetRuntimeArgs(
        program_, writer_kernel, core, {static_cast<uint32_t>(dst_buffer->address()), 0u, 2 * kIters});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> device_tiled;
    distributed::ReadShard(cq, device_tiled, dst_buffer, zero_coord);
    return check_result(device_tiled, stim.golden_tiled);
}

// Quasar: Metal 2.0 spec, the Float32 operand marked UnpackToDest (kernel-wide UnpackToDestEn, 32-bit DEST).
bool run_unpack_to_dest_bcast_dst_offset_quasar(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, BcastDim dim) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    const experimental::NodeCoord node{0, 0};

    const uint32_t bf16_tile_bytes = kTileHW * 2;
    const uint32_t fp32_tile_bytes = kTileHW * 4;
    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false},
            mesh_device.get());
    };
    auto src_copy_buffer = make_dram(kIters * bf16_tile_bytes);
    auto src_bcast_buffer = make_dram(kIters * fp32_tile_bytes);
    auto dst_buffer = make_dram(kIters * 2 * fp32_tile_bytes);

    const experimental::DFBSpecName COPY_DFB{"copy_dfb"};
    const experimental::DFBSpecName BCAST_DFB{"bcast_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::DataflowBufferSpec copy_dfb_spec{
        .unique_id = COPY_DFB,
        .entry_size = bf16_tile_bytes,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec bcast_dfb_spec{
        .unique_id = BCAST_DFB,
        .entry_size = fp32_tile_bytes,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float32,
    };
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = fp32_tile_bytes,
        .num_entries = 4,
        .data_format_metadata = tt::DataFormat::Float32,
    };

    const experimental::DataMovementHardwareConfig dm_hw_config{
        .config_2xx =
            experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true,
            },
    };
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(COPY_DFB, "in0"), experimental::ProducerOf(BCAST_DFB, "in1")},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .hw_config = dm_hw_config,
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = dm_hw_config,
    };

    experimental::KernelSpec::CompilerOptions::Defines defines;
    for (const auto& [name, value] : compute_defines(dim)) {
        defines.emplace(name, value);
    }
    // Only the Float32 operand routes to DEST; the kernel-wide flag it raises also takes the Float16_b copy
    // through UNP_DEST (a legal Float16_b -> Float16_b pair on that path).
    experimental::ComputeHardwareConfig::ComputeUnpackModes unpack_modes = {
        {BCAST_DFB, tt::tt_metal::UnpackMode::UnpackToDest}};
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = kComputeKernel,
        .num_threads = 1,
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {{
                 .dfb_spec_name = COPY_DFB,
                 .accessor_name = "copy_in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = BCAST_DFB,
                 .accessor_name = "bcast_in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .enable_32_bit_dest = true,
                .unpack_modes = unpack_modes,
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };
    experimental::ProgramSpec spec{
        .name = "unpack_to_dest_bcast_dst_offset_quasar",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {copy_dfb_spec, bcast_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));
    auto& program_run = workload.get_programs().at(device_range);

    Stimulus stim = make_stimulus(dim);
    distributed::WriteShard(cq, src_copy_buffer, stim.a_all, zero_coord);
    distributed::WriteShard(cq, src_bcast_buffer, stim.b_all, zero_coord);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", src_copy_buffer->address()},
                 {"src0_bank_id", 0u},
                 {"src1_addr", src_bcast_buffer->address()},
                 {"src1_bank_id", 0u},
                 {"num_tiles", kIters}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node, {{"dst_addr", dst_buffer->address()}, {"bank_id", 0u}, {"num_tiles", 2 * kIters}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> device_tiled;
    distributed::ReadShard(cq, device_tiled, dst_buffer, zero_coord);
    return check_result(device_tiled, stim.golden_tiled);
}

}  // namespace unit_tests::compute::unpack_to_dest_bcast

TEST_F(LLKMeshDeviceFixture, TensixUnpackToDestRowBcastNonzeroDstOffset) {
    using unit_tests::compute::unpack_to_dest_bcast::BcastDim;
    const bool quasar = this->arch_ == tt::ARCH::QUASAR;
    if (!quasar && this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "32-bit unpack-to-dest broadcast path exists on Wormhole B0, Blackhole and Quasar only";
    }
    for (auto& device : this->devices_) {
        for (BcastDim dim : {BcastDim::ROW, BcastDim::COL, BcastDim::SCALAR, BcastDim::NONE}) {
            log_info(
                tt::LogTest,
                "unpack-to-dest {} into dst 0 after copy_tile into dst 1",
                unit_tests::compute::unpack_to_dest_bcast::bcast_dim_name(dim));
            const bool ok =
                quasar
                    ? unit_tests::compute::unpack_to_dest_bcast::run_unpack_to_dest_bcast_dst_offset_quasar(device, dim)
                    : unit_tests::compute::unpack_to_dest_bcast::run_unpack_to_dest_bcast_dst_offset(device, dim);
            EXPECT_TRUE(ok) << "bcast dim " << unit_tests::compute::unpack_to_dest_bcast::bcast_dim_name(dim);
        }
    }
}

}  // namespace tt::tt_metal
