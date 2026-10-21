// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared host side of the Quasar 2.0 compute API op tests (datacopy, eltwise binary, matmul): one compute kernel
// fed by one DRAM reader per input DFB and drained by one DRAM writer per output DFB, all 1 producer x 1 consumer,
// Float16_b 32x32 tiles. Outputs are pre-filled with a sentinel so a skipped section shows as a wrong tile.

#pragma once

#include "llk_device_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::quasar_2_0 {

constexpr std::uint32_t TILE_ELEMS = 1024;
constexpr std::uint32_t TILE_BYTES = TILE_ELEMS * 2;  // Float16_b
constexpr std::uint16_t SENTINEL = 0x7F81;            // a NaN no op here produces

using Tiles = std::vector<std::uint16_t>;  // bf16 bits, TILE_ELEMS per tile

inline float bf16_to_float(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline std::uint16_t float_to_bf16(float f) {  // round to nearest even
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// Random bf16 in [-1, 1): every tile differs from every other tile.
inline Tiles random_tiles(std::uint32_t num_tiles, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tiles t(num_tiles * TILE_ELEMS);
    for (auto& v : t) {
        v = float_to_bf16(dist(gen));
    }
    return t;
}

// Element (row, col) of a 32x32 tile stored as four 16x16 row-major faces.
inline std::uint32_t tile_elem(std::uint32_t r, std::uint32_t c) {
    return ((r / 16) * 2 + (c / 16)) * 256 + (r % 16) * 16 + (c % 16);
}

struct ProgramIo {
    std::string compute_kernel;
    std::vector<Tiles> inputs;          // one stream per input DFB (in0, in1, ...), in consumption order
    std::vector<std::uint32_t> out_tiles;  // tiles per output DFB (out0, out1, ...)
    std::vector<std::uint32_t> in_depth;   // DFB entries per input DFB
    std::vector<std::uint32_t> out_depth;  // DFB entries per output DFB
    std::vector<std::pair<std::string, std::uint32_t>> ct_args;
    bool fp32_dest = false;
    bool full_sync = false;
};

// Builds and runs one program, returns one tile stream per output DFB.
inline std::vector<Tiles> run_program(distributed::MeshDevice& mesh_device, const ProgramIo& io) {
    namespace ex = experimental;
    auto& cq = mesh_device.mesh_command_queue();
    const ex::NodeCoord node{0, 0};

    auto make_buffer = [&](std::uint32_t tiles) {
        // The direct reader/writer address one DRAM bank: page_size = whole buffer keeps it in one bank.
        const distributed::DeviceLocalBufferConfig cfg{.page_size = TILE_BYTES * tiles, .buffer_type = BufferType::DRAM};
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = TILE_BYTES * tiles}, cfg, &mesh_device);
    };
    auto pack32 = [](const Tiles& t) {
        std::vector<std::uint32_t> v(t.size() / 2);
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = t[2 * i] | (static_cast<std::uint32_t>(t[2 * i + 1]) << 16);
        }
        return v;
    };

    std::vector<ex::KernelSpec> kernels;
    std::vector<ex::DataflowBufferSpec> dfbs;
    std::vector<ex::KernelSpecName> names;
    ex::KernelSpec compute{
        .unique_id = ex::KernelSpecName{"compute"},
        .source = io.compute_kernel,
        .num_threads = 1,
        .hw_config = ex::ComputeHardwareConfig{.enable_32_bit_dest = io.fp32_dest, .double_buffer_dest = !io.full_sync},
    };
    for (const auto& [k, v] : io.ct_args) {
        compute.compile_time_args[k] = v;
    }
    // The kernel references only the DFB tokens it is bound to.
    compute.compiler_options.defines["NUM_IN"] = std::to_string(io.inputs.size());
    compute.compiler_options.defines["NUM_OUT"] = std::to_string(io.out_tiles.size());
    auto dfb_spec = [](const std::string& name, std::uint32_t depth) {
        return ex::DataflowBufferSpec{
            .unique_id = ex::DFBSpecName{name},
            .entry_size = TILE_BYTES,
            .num_entries = depth,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
    };

    std::vector<std::shared_ptr<distributed::MeshBuffer>> in_bufs, out_bufs;
    ex::ProgramRunArgs params;
    for (std::size_t i = 0; i < io.inputs.size(); ++i) {
        const std::string dfb = "in" + std::to_string(i);
        const std::uint32_t tiles = io.inputs[i].size() / TILE_ELEMS;
        dfbs.push_back(dfb_spec(dfb, io.in_depth[i]));
        compute.dfb_bindings.push_back(ex::ConsumerOf(ex::DFBSpecName{dfb}, dfb));
        const ex::KernelSpecName name{"reader" + std::to_string(i)};
        kernels.push_back(ex::KernelSpec{
            .unique_id = name,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {ex::ProducerOf(ex::DFBSpecName{dfb}, "out")},
            .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
            .hw_config = ex::DataMovementHardwareConfig{},
        });
        names.push_back(name);
        in_bufs.push_back(make_buffer(tiles));
        distributed::EnqueueWriteMeshBuffer(cq, in_bufs.back(), pack32(io.inputs[i]), /*blocking=*/true);
        params.kernel_run_args.push_back(
            {.kernel = name,
             .runtime_arg_values = ex::MakeRuntimeArgsForSingleNode(
                 node,
                 {{"src_addr", in_bufs.back()->address()},
                  {"src_bank_id", 0u},
                  {"num_tiles", tiles},
                  {"dram_page_stride", TILE_BYTES}})});
    }
    for (std::size_t i = 0; i < io.out_tiles.size(); ++i) {
        const std::string dfb = "out" + std::to_string(i);
        dfbs.push_back(dfb_spec(dfb, io.out_depth[i]));
        compute.dfb_bindings.push_back(ex::ProducerOf(ex::DFBSpecName{dfb}, dfb));
        const ex::KernelSpecName name{"writer" + std::to_string(i)};
        kernels.push_back(ex::KernelSpec{
            .unique_id = name,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {ex::ConsumerOf(ex::DFBSpecName{dfb}, "in")},
            .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
            .hw_config = ex::DataMovementHardwareConfig{},
        });
        names.push_back(name);
        out_bufs.push_back(make_buffer(io.out_tiles[i]));
        distributed::EnqueueWriteMeshBuffer(
            cq, out_bufs.back(), pack32(Tiles(io.out_tiles[i] * TILE_ELEMS, SENTINEL)), /*blocking=*/true);
        params.kernel_run_args.push_back(
            {.kernel = name,
             .runtime_arg_values = ex::MakeRuntimeArgsForSingleNode(
                 node,
                 {{"dst_addr", out_bufs.back()->address()},
                  {"dst_bank_id", 0u},
                  {"num_tiles", io.out_tiles[i]},
                  {"dram_page_stride", TILE_BYTES}})});
    }
    kernels.push_back(compute);
    names.push_back(compute.unique_id);

    ex::ProgramSpec spec{
        .name = "quasar_2_0_op",
        .kernels = kernels,
        .dataflow_buffers = dfbs,
        .work_units = {ex::WorkUnitSpec{.name = "main", .kernels = names, .target_nodes = node}},
    };
    Program program = ex::MakeProgramFromSpec(mesh_device, spec);
    ex::SetProgramRunArgs(program, params);
    LaunchProgram(mesh_device, std::move(program));

    std::vector<Tiles> outs;
    for (auto& buf : out_bufs) {
        std::vector<std::uint32_t> raw;
        distributed::EnqueueReadMeshBuffer(cq, raw, buf, /*blocking=*/true);
        Tiles t(raw.size() * 2);
        for (std::size_t i = 0; i < raw.size(); ++i) {
            t[2 * i] = raw[i] & 0xFFFF;
            t[2 * i + 1] = raw[i] >> 16;
        }
        outs.push_back(std::move(t));
    }
    return outs;
}

// Bit-exact compare; reports the first mismatching tile (and whether it still holds the sentinel).
inline ::testing::AssertionResult tiles_equal(const Tiles& got, const Tiles& want) {
    if (got.size() != want.size()) {
        return ::testing::AssertionFailure() << "size " << got.size() << " != " << want.size();
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            std::ostringstream msg;  // AssertionResult drops std::hex, so format here
            msg << "tile " << i / TILE_ELEMS << " elem " << i % TILE_ELEMS << ": got 0x" << std::hex << got[i]
                << " want 0x" << want[i] << (got[i] == SENTINEL ? " (sentinel: never written)" : "");
            return ::testing::AssertionFailure() << msg.str();
        }
    }
    return ::testing::AssertionSuccess();
}

// Per-tile tolerance compare: |got - want| <= atol + rtol * |want| for every element, and no sentinel.
inline ::testing::AssertionResult tiles_close(const Tiles& got, const std::vector<float>& want, float rtol, float atol) {
    if (got.size() != want.size()) {
        return ::testing::AssertionFailure() << "size " << got.size() << " != " << want.size();
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float g = bf16_to_float(got[i]);
        if (got[i] == SENTINEL || !(std::fabs(g - want[i]) <= atol + rtol * std::fabs(want[i]))) {
            return ::testing::AssertionFailure() << "tile " << i / TILE_ELEMS << " elem " << i % TILE_ELEMS << ": got "
                                                 << g << " want " << want[i]
                                                 << (got[i] == SENTINEL ? " (sentinel: never written)" : "");
        }
    }
    return ::testing::AssertionSuccess();
}

// Dest tiles per section on Quasar: 8 (SyncHalf, 16-bit), halved for a 32-bit Dest, doubled for SyncFull.
inline std::uint32_t dest_capacity(bool full_sync, bool fp32_dest) { return (full_sync ? 16 : 8) / (fp32_dest ? 2 : 1); }

}  // namespace tt::tt_metal::quasar_2_0
