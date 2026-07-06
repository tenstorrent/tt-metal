// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the DEBUG_CB_HASH compute API:
//   - hash_cb_trisc(): TRISC scalar FNV-1a-32 over a CB, printed via DPRINT.
//   - hash_cb_sfpu():  SFPU lanewise FNV23 over a CB, left in DEST for the
//                      caller to pack out; the host XOR-folds the result tile.
//
// Each test launches a tiny single-core program that reads a tile from DRAM
// into an input CB, invokes the hash probe, then copies the result through to
// an output CB / DRAM. The tests assert, per variant:
//   1. The probe produced a non-zero fingerprint (scalar: parsed from DPRINT;
//      SFPU: host XOR-fold of the packed result tile).
//   2. Two back-to-back runs on the same input produce the same fingerprint.
//   3. Distinct inputs produce distinct fingerprints (discrimination).

#include <cstdint>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tt_metal/tt_metal/debug_tools/debug_tools_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t INPUT_CB = 0;
constexpr uint32_t OUTPUT_CB = 16;
constexpr size_t NUM_TILES = 1;
constexpr uint32_t LABEL = 0xBu;

// Reader / writer kernels: borrow the checkpoint test kernels. They are
// vanilla DRAM ↔ CB tile shuttles; DEBUG_CHECKPOINT expands to nothing in
// builds that don't enable checkpoint, which is the default for DPRINT tests.
constexpr const char* READER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_checkpoint.cpp";
constexpr const char* WRITER_KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_checkpoint.cpp";

// Read the captured DPRINT file and return the hex hash from the *last* line
// matching "hash[0x<label>] cb=<id> tiles=<n> = 0x<hash>". Returns 0 on miss.
//
// The fixture's dprint file accumulates lines across run_once calls inside
// one TEST_F body — taking the last match makes back-to-back runs see their
// own hash rather than the first run's.
uint32_t extract_hash(const std::string& dprint_file, uint32_t label, uint32_t cb_id, uint32_t tiles) {
    std::stringstream label_hex;
    label_hex << std::hex << label;
    std::regex re(
        "hash\\[0x" + label_hex.str() +
        "\\] cb=" + std::to_string(cb_id) +
        " tiles=" + std::to_string(tiles) + " = 0x([0-9a-fA-F]+)");

    std::ifstream f(dprint_file);
    std::string line;
    uint32_t latest = 0u;
    while (std::getline(f, line)) {
        std::smatch m;
        if (std::regex_search(line, m, re)) {
            latest = static_cast<uint32_t>(std::stoul(m[1].str(), nullptr, 16));
        }
    }
    return latest;
}

struct Setup {
    std::shared_ptr<distributed::MeshBuffer> input_dram;
    std::shared_ptr<distributed::MeshBuffer> output_dram;
    distributed::MeshWorkload workload;
    Program* program;
    distributed::MeshCoordinate zero{0, 0};
    CoreCoord core{0, 0};

    Setup(const std::shared_ptr<distributed::MeshDevice>& mesh_device, tt::DataFormat fmt) {
        size_t tile_size = tt::tile_size(fmt);
        size_t buf_sz = NUM_TILES * tile_size;

        auto device_range = distributed::MeshCoordinateRange(zero, zero);
        Program prog = CreateProgram();
        workload.add_program(device_range, std::move(prog));
        program = &workload.get_programs().at(device_range);

        distributed::DeviceLocalBufferConfig lc{.page_size = buf_sz, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig bc{.size = buf_sz};
        input_dram = distributed::MeshBuffer::create(bc, lc, mesh_device.get());
        output_dram = distributed::MeshBuffer::create(bc, lc, mesh_device.get());

        CircularBufferConfig in_cfg = CircularBufferConfig(buf_sz, {{INPUT_CB, fmt}}).set_page_size(INPUT_CB, tile_size);
        CreateCircularBuffer(*program, core, in_cfg);
        CircularBufferConfig out_cfg =
            CircularBufferConfig(buf_sz, {{OUTPUT_CB, fmt}}).set_page_size(OUTPUT_CB, tile_size);
        CreateCircularBuffer(*program, core, out_cfg);
    }
};

// Launch one run of <compute_kernel> on a fresh program, return the hash
// captured from DPRINT.
uint32_t run_once(
    DevicePrintFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const char* compute_kernel,
    tt::DataFormat fmt,
    const std::vector<uint32_t>& input) {
    Setup s(mesh_device, fmt);
    auto& cq = mesh_device->mesh_command_queue();

    auto reader = CreateKernel(
        *s.program,
        READER_KERNEL,
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {INPUT_CB, 0}});
    auto writer = CreateKernel(
        *s.program,
        WRITER_KERNEL,
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {OUTPUT_CB, 0}});
    CreateKernel(
        *s.program,
        compute_kernel,
        s.core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(NUM_TILES), LABEL}});

    SetRuntimeArgs(
        *s.program,
        reader,
        s.core,
        {static_cast<uint32_t>(s.input_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});
    SetRuntimeArgs(
        *s.program,
        writer,
        s.core,
        {static_cast<uint32_t>(s.output_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});

    distributed::WriteShard(cq, s.input_dram, const_cast<std::vector<uint32_t>&>(input), s.zero);
    fixture->RunProgram(mesh_device, s.workload);

    return extract_hash(fixture->dprint_file_name, LABEL, INPUT_CB, NUM_TILES);
}

// Launch one run of the SFPU hash kernel and return the host XOR-fold of the
// packed result tile. hash_cb_sfpu leaves the 32 per-lane accumulators in DEST
// row 0 (rest zeroed); the kernel packs that tile to OUTPUT_CB, the writer
// shuttles it to DRAM, and folding the whole tile recovers XOR(32 accumulators).
uint32_t run_once_sfpu(
    DevicePrintFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::vector<uint32_t>& input) {
    Setup s(mesh_device, tt::DataFormat::Int32);
    auto& cq = mesh_device->mesh_command_queue();

    auto reader = CreateKernel(
        *s.program,
        READER_KERNEL,
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {INPUT_CB, 0}});
    auto writer = CreateKernel(
        *s.program,
        WRITER_KERNEL,
        s.core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {OUTPUT_CB, 0}});
    CreateKernel(
        *s.program,
        "tests/tt_metal/tt_metal/test_kernels/compute/cb_hash_sfpu.cpp",
        s.core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(NUM_TILES)}});

    SetRuntimeArgs(
        *s.program, reader, s.core, {static_cast<uint32_t>(s.input_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});
    SetRuntimeArgs(
        *s.program, writer, s.core, {static_cast<uint32_t>(s.output_dram->address()), 0u, static_cast<uint32_t>(NUM_TILES)});

    distributed::WriteShard(cq, s.input_dram, const_cast<std::vector<uint32_t>&>(input), s.zero);
    fixture->RunProgram(mesh_device, s.workload);

    std::vector<uint32_t> result(NUM_TILES * 1024, 0u);
    distributed::ReadShard(cq, result, s.output_dram, s.zero);

    uint32_t h = 0u;
    for (uint32_t w : result) {
        h ^= w;
    }
    return h;
}

}  // namespace

class CbHashTest : public DevicePrintFixture {};

// ---------- hash_cb_trisc: scalar FNV-1a-32 ----------

TEST_F(CbHashTest, HashCbTriscEmitsLine) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) {
            auto input = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
                -1.0f, 1.0f, NUM_TILES * 1024);
            uint32_t h = run_once(
                f, d, "tests/tt_metal/tt_metal/test_kernels/compute/cb_hash_trisc.cpp", tt::DataFormat::Float16_b, input);
            EXPECT_NE(h, 0u) << "hash_cb_trisc did not emit a hash line we could parse";
        },
        this->devices_[0]);
}

TEST_F(CbHashTest, HashCbTriscDeterminism) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) {
            auto input = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
                -1.0f, 1.0f, NUM_TILES * 1024);
            uint32_t h1 = run_once(
                f, d, "tests/tt_metal/tt_metal/test_kernels/compute/cb_hash_trisc.cpp", tt::DataFormat::Float16_b, input);
            uint32_t h2 = run_once(
                f, d, "tests/tt_metal/tt_metal/test_kernels/compute/cb_hash_trisc.cpp", tt::DataFormat::Float16_b, input);
            EXPECT_NE(h1, 0u);
            EXPECT_EQ(h1, h2) << "hash_cb_trisc is not deterministic across runs";
        },
        this->devices_[0]);
}

// ---------- hash_cb_sfpu: SFPU FNV23, INT32 ----------

TEST_F(CbHashTest, HashCbSfpuProducesHash) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) {
            // INT32 tile = 32*32*4 = 4096 bytes = 1024 u32s.
            std::vector<uint32_t> input(NUM_TILES * 1024);
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = 0xDEAD0000u | static_cast<uint32_t>(i);
            }
            uint32_t h = run_once_sfpu(f, d, input);
            EXPECT_NE(h, 0u) << "hash_cb_sfpu produced a zero fingerprint";
        },
        this->devices_[0]);
}

TEST_F(CbHashTest, HashCbSfpuDeterminism) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) {
            std::vector<uint32_t> input(NUM_TILES * 1024);
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = 0xC0FFEE00u + static_cast<uint32_t>(i);
            }
            uint32_t h1 = run_once_sfpu(f, d, input);
            uint32_t h2 = run_once_sfpu(f, d, input);
            EXPECT_NE(h1, 0u);
            EXPECT_EQ(h1, h2) << "hash_cb_sfpu is not deterministic across runs";
        },
        this->devices_[0]);
}

// Discrimination: distinct inputs must produce distinct hashes. Catches the
// "always returns the same constant" failure mode that EmitsLine + Determinism
// would otherwise let pass.

TEST_F(CbHashTest, HashCbTriscDiscriminates) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) {
            std::vector<uint32_t> a(NUM_TILES * 512, 0x00010001u);  // tile of bf16 ones
            std::vector<uint32_t> b(NUM_TILES * 512, 0x40004000u);  // tile of bf16 twos
            uint32_t ha = run_once(
                f, d, "tests/tt_metal/tt_metal/test_kernels/compute/cb_hash_trisc.cpp", tt::DataFormat::Float16_b, a);
            uint32_t hb = run_once(
                f, d, "tests/tt_metal/tt_metal/test_kernels/compute/cb_hash_trisc.cpp", tt::DataFormat::Float16_b, b);
            EXPECT_NE(ha, 0u);
            EXPECT_NE(hb, 0u);
            EXPECT_NE(ha, hb) << "hash_cb_trisc returned the same value for distinct inputs: 0x" << std::hex << ha;
        },
        this->devices_[0]);
}

TEST_F(CbHashTest, HashCbSfpuDiscriminates) {
    this->RunTestOnDevice(
        [](DevicePrintFixture* f, const std::shared_ptr<distributed::MeshDevice>& d) {
            // Vary words within each tile: a uniform tile makes all 32 SFPU lane
            // accumulators identical, and XOR-folding an even count of equal values
            // cancels to zero even when the hash ran correctly.
            std::vector<uint32_t> a(NUM_TILES * 1024);
            std::vector<uint32_t> b(NUM_TILES * 1024);
            for (size_t i = 0; i < a.size(); ++i) {
                a[i] = 0xAAAA0000u | static_cast<uint32_t>(i);
                b[i] = 0x12340000u | static_cast<uint32_t>(i);
            }
            uint32_t ha = run_once_sfpu(f, d, a);
            uint32_t hb = run_once_sfpu(f, d, b);
            EXPECT_NE(ha, 0u);
            EXPECT_NE(hb, 0u);
            EXPECT_NE(ha, hb) << "hash_cb_sfpu returned the same value for distinct inputs: 0x" << std::hex << ha;
        },
        this->devices_[0]);
}
