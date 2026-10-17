// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <umd/device/types/arch.hpp>

#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"

// The math thread's replay buffer is shared between the SFPU bodies (slots [0, 16)) and the FPU MOP bodies
// (slots [16, 32)). The quant / requant / dequant kernels record their per-iteration bodies into the SFPU
// half at init and replay them per call; the FPU ops (reduce, transpose_dest, matmul) record into the other
// half. A body that straddles the split is overwritten by whichever op records last, and that op's own
// recording is corrupted in turn. This test runs an FPU reduce and an SFPU dequant in ONE kernel with both
// inits issued up front (reduce_init first, dequant_tile_init last) and checks both outputs against exact
// goldens: with a straddling dequant body the reduce MOP replays SFPU instructions in place of MVMULs and
// the row sums come out wrong. The kernel also supports the reverse init order (compile arg 1), where the
// reduce recording overwrites the tail of a straddling dequant body and the dequant then replays MVMULs
// that wait forever for Src valid: that order is deliberately NOT registered as a test case, because a
// regression there manifests as a device hang rather than a data failure.

namespace tt::tt_metal {

namespace unit_tests::compute::quant_replay_slots {

constexpr std::uint32_t kTileDim = 32;
constexpr std::uint32_t kFaceDim = 16;
constexpr std::uint32_t kTileElems = kTileDim * kTileDim;

// Index of (row, col) in a 32x32 tile laid out as four 16x16 faces (f0 top-left, f1 top-right, f2, f3).
std::uint32_t tile_index(std::uint32_t row, std::uint32_t col) {
    const std::uint32_t face = (row / kFaceDim) * 2 + (col / kFaceDim);
    return face * kFaceDim * kFaceDim + (row % kFaceDim) * kFaceDim + (col % kFaceDim);
}

std::uint16_t bf16_bits(float v) { return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16); }

// Two bf16 per 32-bit word, element 2i in the low half.
std::vector<std::uint32_t> pack_bf16(const std::vector<float>& values) {
    std::vector<std::uint32_t> packed(values.size() / 2);
    for (size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<std::uint32_t>(bf16_bits(values[2 * i])) |
                    (static_cast<std::uint32_t>(bf16_bits(values[2 * i + 1])) << 16);
    }
    return packed;
}

struct Stimulus {
    std::vector<float> reduce_data;  // tile order, small integers (exact in bf16 and under any fidelity)
    std::vector<std::int32_t> q_in;  // tile order
    float scale;
    std::int32_t zero_point;
};

Stimulus make_stimulus(std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> data_dist(-15, 15);
    std::uniform_int_distribution<std::int32_t> q_dist(-120, 120);
    Stimulus s;
    s.reduce_data.resize(kTileElems);
    s.q_in.resize(kTileElems);
    for (auto& v : s.reduce_data) {
        v = static_cast<float>(data_dist(gen));
    }
    for (auto& v : s.q_in) {
        v = q_dist(gen);
    }
    s.scale = 0.5f;  // exact in bf16/tf32/fp32, so the SrcA route of the scale tile cannot change it
    s.zero_point = 7;
    return s;
}

// REDUCE_ROW SUM with a unit scaler: each row's sum in column 0, everything else zero (packer mask contract).
std::vector<float> golden_row_sums(const Stimulus& s) {
    std::vector<float> golden(kTileElems, 0.0f);
    for (std::uint32_t r = 0; r < kTileDim; ++r) {
        float acc = 0.0f;
        for (std::uint32_t c = 0; c < kTileDim; ++c) {
            acc += s.reduce_data[tile_index(r, c)];
        }
        golden[tile_index(r, 0)] = acc;
    }
    return golden;
}

// (q - zero_point) * scale; every step is exact for this stimulus.
std::vector<float> golden_dequant(const Stimulus& s) {
    std::vector<float> golden(kTileElems);
    for (std::uint32_t i = 0; i < kTileElems; ++i) {
        golden[i] = (static_cast<float>(s.q_in[i]) - static_cast<float>(s.zero_point)) * s.scale;
    }
    return golden;
}

// The scaler tile the reduce reader kernels generate: the scaler in row 0 of each face, zero elsewhere.
std::vector<float> make_scaler_tile(float scaler) {
    std::vector<float> tile(kTileElems, 0.0f);
    for (std::uint32_t face = 0; face < 4; ++face) {
        for (std::uint32_t j = 0; j < kFaceDim; ++j) {
            tile[face * kFaceDim * kFaceDim + j] = scaler;
        }
    }
    return tile;
}

struct Outputs {
    std::vector<float> row_sums;
    std::vector<float> dequant;
};

Outputs run(distributed::MeshDevice& mesh_device, std::uint32_t init_order, const Stimulus& s) {
    Program program = CreateProgram();
    const CoreCoord core = {0, 0};

    const std::uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t int32_tile_bytes = tt::tile_size(tt::DataFormat::Int32);
    const std::uint32_t fp32_tile_bytes = tt::tile_size(tt::DataFormat::Float32);

    auto make_dram = [&](std::uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{.page_size = bytes, .buffer_type = BufferType::DRAM},
            &mesh_device);
    };
    auto src_data = make_dram(bf16_tile_bytes);
    auto src_scaler = make_dram(bf16_tile_bytes);
    auto src_q = make_dram(int32_tile_bytes);
    auto src_scale = make_dram(fp32_tile_bytes);
    auto dst = make_dram(2 * fp32_tile_bytes);

    auto make_cb = [&](tt::CBIndex idx, tt::DataFormat fmt, std::uint32_t tiles) {
        const std::uint32_t tile_bytes = tt::tile_size(fmt);
        CreateCircularBuffer(
            program, core, CircularBufferConfig(tiles * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes));
    };
    make_cb(tt::CBIndex::c_0, tt::DataFormat::Float16_b, 1);
    make_cb(tt::CBIndex::c_1, tt::DataFormat::Float16_b, 1);
    make_cb(tt::CBIndex::c_2, tt::DataFormat::Int32, 1);
    make_cb(tt::CBIndex::c_3, tt::DataFormat::Float32, 1);
    make_cb(tt::CBIndex::c_16, tt::DataFormat::Float32, 2);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dequant_reduce_replay_slots.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    // fp32 Dest holds the Int32 input and the fp32 dequant result; the reduce accumulates exactly in it.
    auto compute = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/dequant_reduce_replay_slots.cpp",
        core,
        ComputeConfig{.fp32_dest_acc_en = true, .compile_args = {init_order}});

    std::vector<std::uint32_t> q_words(kTileElems);
    for (std::uint32_t i = 0; i < kTileElems; ++i) {
        q_words[i] = std::bit_cast<std::uint32_t>(s.q_in[i]);
    }
    const std::vector<std::uint32_t> scale_words(kTileElems, std::bit_cast<std::uint32_t>(s.scale));
    const float neg_zero_point = -static_cast<float>(s.zero_point);

    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src_data, pack_bf16(s.reduce_data), /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, src_scaler, pack_bf16(make_scaler_tile(1.0f)), /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, src_q, q_words, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, src_scale, scale_words, /*blocking=*/true);

    SetRuntimeArgs(
        program,
        reader,
        core,
        {src_data->address(), 0, src_scaler->address(), 0, src_q->address(), 0, src_scale->address(), 0});
    SetRuntimeArgs(program, writer, core, {dst->address(), 0, 2});
    SetRuntimeArgs(program, compute, core, {std::bit_cast<std::uint32_t>(neg_zero_point)});

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, dst, /*blocking=*/true);
    EXPECT_EQ(result.size(), 2 * kTileElems);

    Outputs out;
    out.row_sums.resize(kTileElems);
    out.dequant.resize(kTileElems);
    for (std::uint32_t i = 0; i < kTileElems; ++i) {
        out.row_sums[i] = std::bit_cast<float>(result[i]);
        out.dequant[i] = std::bit_cast<float>(result[kTileElems + i]);
    }
    return out;
}

void check_tile(const std::vector<float>& got, const std::vector<float>& want, const char* what) {
    std::uint32_t mismatches = 0;
    for (std::uint32_t r = 0; r < kTileDim; ++r) {
        for (std::uint32_t c = 0; c < kTileDim; ++c) {
            const std::uint32_t i = tile_index(r, c);
            if (got[i] != want[i]) {
                if (mismatches < 8) {
                    ADD_FAILURE() << what << " mismatch at row " << r << " col " << c << ": got " << got[i] << ", want "
                                  << want[i];
                }
                ++mismatches;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << what << ": " << mismatches << " of " << kTileElems << " datums differ";
}

void run_and_check(distributed::MeshDevice& mesh_device, std::uint32_t init_order) {
    const Stimulus s = make_stimulus(/*seed=*/0x5107 + init_order);
    const Outputs out = run(mesh_device, init_order, s);
    check_tile(out.row_sums, golden_row_sums(s), "reduce row sum");
    check_tile(out.dequant, golden_dequant(s), "dequant");
}

}  // namespace unit_tests::compute::quant_replay_slots

using namespace unit_tests::compute::quant_replay_slots;

// reduce_init records first, dequant_tile_init last: a dequant body that reaches past the SFPU half
// overwrites the head of the reduce MOP body, so the reduce replays SFPU instructions in place of MVMULs.
TEST_F(LLKMeshDeviceSingleCardFixture, TensixComputeDequantRecordsAfterReduce) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "FPU/SFPU replay split is a Wormhole/Blackhole math-thread convention";
    }
    run_and_check(this->device(), /*init_order=*/0);
}

}  // namespace tt::tt_metal
