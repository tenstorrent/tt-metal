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
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/arch.hpp>

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
// minimal sequence that breaks the coincidence: copy_tile to DST[1], then unary_bcast into DST[0],
// one acquire, one core, one tile pair (see the compute kernel for the full walkthrough). All
// three broadcast dimensions (ROW / COL / SCALAR) run, since all three sequences in that branch
// share the same Dst addressing.
//
//   c_0 (Float16_b): a[r][c] = r + 1               -> copied to DST[1]
//   c_1 (Float32)  : b[r][c] = 100 + r + 41*c      -> broadcast into DST[0]
//   out (Float32)  : tile 0 = DST[0], tile 1 = DST[1]
//
//   expected: out0 = broadcast of b (ROW: b[0][c]; COL: b[r][0]; SCALAR: b[0][0]),
//             out1[r][c] = r + 1 (copy untouched)
//   bug:      out0 = raw b tile (never broadcast), out1 = broadcast of the COPIED tile over itself
//
// All stimulus values are integers, exactly representable in Float16_b and Float32 through every
// conversion on the path, so the comparison is bit-exact.

namespace tt::tt_metal {

namespace unit_tests::compute::unpack_to_dest_bcast {

constexpr uint32_t kTileHW = 32 * 32;

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

// Matches BCAST_DIM_VAL in the compute kernel.
enum class BcastDim : uint32_t { ROW = 0, COL = 1, SCALAR = 2 };

const char* bcast_dim_name(BcastDim dim) {
    switch (dim) {
        case BcastDim::ROW: return "ROW";
        case BcastDim::COL: return "COL";
        case BcastDim::SCALAR: return "SCALAR";
    }
    return "?";
}

bool run_unpack_to_dest_bcast_dst_offset(const std::shared_ptr<distributed::MeshDevice>& mesh_device, BcastDim dim) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    Program program = tt_metal::CreateProgram();
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
    auto src_copy_buffer = make_dram(bf16_tile_bytes);   // c_0: one Float16_b tile
    auto src_bcast_buffer = make_dram(fp32_tile_bytes);  // c_1: one Float32 tile
    auto dst_buffer = make_dram(2 * fp32_tile_bytes);    // c_16: two Float32 tiles (DST[0], DST[1])

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

    std::map<std::string, std::string> compute_defines = {
        {"BCAST_DIM_VAL", std::to_string(static_cast<uint32_t>(dim))}};
    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unpack_to_dest_row_bcast_after_copy.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .defines = compute_defines});

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
    std::vector<float> golden0_rm(kTileHW);  // b broadcast along `dim`
    std::vector<float> golden1_rm(kTileHW);  // the copied a tile
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t src = (dim == BcastDim::ROW) ? (0 * 32 + c) : (dim == BcastDim::COL) ? (r * 32 + 0) : 0;
            golden0_rm[r * 32 + c] = b_rm[src];
            golden1_rm[r * 32 + c] = static_cast<float>(r + 1);
        }
    }

    ::unit_tests::compute::GoldenConfig bf16_cfg{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 2};
    ::unit_tests::compute::GoldenConfig fp32_cfg{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};

    auto a_tiled = ::unit_tests::compute::gold_standard_tilize(pack_bf16_as_u32(a_rm), bf16_cfg);
    auto b_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(b_rm), fp32_cfg);
    auto golden0_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(golden0_rm), fp32_cfg);
    auto golden1_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(golden1_rm), fp32_cfg);
    std::vector<uint32_t> golden_tiled = golden0_tiled;
    golden_tiled.insert(golden_tiled.end(), golden1_tiled.begin(), golden1_tiled.end());

    distributed::WriteShard(cq, src_copy_buffer, a_tiled, zero_coord);
    distributed::WriteShard(cq, src_bcast_buffer, b_tiled, zero_coord);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {static_cast<uint32_t>(src_copy_buffer->address()),
         0u,
         static_cast<uint32_t>(src_bcast_buffer->address()),
         0u,
         1u});
    tt_metal::SetRuntimeArgs(program_, writer_kernel, core, {static_cast<uint32_t>(dst_buffer->address()), 0u, 2u});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> device_tiled;
    distributed::ReadShard(cq, device_tiled, dst_buffer, zero_coord);

    if (device_tiled.size() != golden_tiled.size()) {
        log_error(tt::LogTest, "Size mismatch: device={} golden={}", device_tiled.size(), golden_tiled.size());
        return false;
    }

    size_t num_mismatches = 0;
    constexpr size_t max_report = 8;
    for (size_t i = 0; i < device_tiled.size(); ++i) {
        if (device_tiled[i] != golden_tiled[i]) {
            if (num_mismatches < max_report) {
                log_error(
                    tt::LogTest,
                    "Mismatch in out tile {} at tiled idx {}: golden={} device={}",
                    i / kTileHW,
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

}  // namespace unit_tests::compute::unpack_to_dest_bcast

TEST_F(LLKMeshDeviceFixture, TensixUnpackToDestRowBcastNonzeroDstOffset) {
    using unit_tests::compute::unpack_to_dest_bcast::BcastDim;
    if (this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "32-bit unpack-to-dest broadcast path exists on Wormhole B0 and Blackhole only";
    }
    for (auto& device : this->devices_) {
        for (BcastDim dim : {BcastDim::ROW, BcastDim::COL, BcastDim::SCALAR}) {
            log_info(
                tt::LogTest,
                "unpack-to-dest {} broadcast into dst 0 after copy_tile into dst 1",
                unit_tests::compute::unpack_to_dest_bcast::bcast_dim_name(dim));
            EXPECT_TRUE(unit_tests::compute::unpack_to_dest_bcast::run_unpack_to_dest_bcast_dst_offset(device, dim))
                << "bcast dim " << unit_tests::compute::unpack_to_dest_bcast::bcast_dim_name(dim);
        }
    }
}

}  // namespace tt::tt_metal
