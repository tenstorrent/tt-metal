// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// STANDALONE PROTOTYPE: prove that a "streaming" tilize (per-tile pack +
// push_back(1)) is numerically correct, so a downstream fused kernel can use a
// SMALL double-buffered output CB instead of holding the whole tile-row.
//
// For each width W in {4, 128}:
//   * build a random row-major bf16 input [32, W*32] in DRAM
//   * run the ATOMIC tilize (cb_out sized W tiles)   -> reference on-device
//   * run the STREAMING tilize (cb_out sized 2 tiles) -> the thing under test
//   * compare both against a host tilize golden (bit-exact + PCC)
//   * compare streaming vs atomic (must be bit-identical: same LLK datapath)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

constexpr uint32_t TILE_HW = 32 * 32;
constexpr uint32_t TILE_BYTES = TILE_HW * sizeof(bfloat16);  // 2048 for bf16

// Row-major [32, W*32] -> W face-swizzled 32x32 tiles (standard TT tile layout).
std::vector<bfloat16> host_tilize(const std::vector<bfloat16>& in, uint32_t W) {
    const uint32_t cols = W * 32;
    std::vector<bfloat16> out(static_cast<size_t>(W) * TILE_HW, bfloat16(0.0f));
    for (uint32_t t = 0; t < W; ++t) {
        const uint32_t base = t * TILE_HW;
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t face = (r / 16) * 2 + (c / 16);
                const uint32_t fr = r % 16;
                const uint32_t fc = c % 16;
                const uint32_t idx = base + face * 256 + fr * 16 + fc;
                out[idx] = in[r * cols + (t * 32 + c)];
            }
        }
    }
    return out;
}

struct Compare {
    size_t total = 0;
    size_t exact = 0;
    double pcc = 0.0;
};

Compare compare(const std::vector<bfloat16>& got, const std::vector<bfloat16>& ref) {
    Compare r;
    r.total = ref.size();
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    const double n = static_cast<double>(r.total);
    for (size_t i = 0; i < r.total; ++i) {
        const float a = static_cast<float>(got[i]);
        const float b = static_cast<float>(ref[i]);
        if (a == b) {
            r.exact++;
        }
        sx += a;
        sy += b;
        sxx += static_cast<double>(a) * a;
        syy += static_cast<double>(b) * b;
        sxy += static_cast<double>(a) * b;
    }
    const double cov = sxy - sx * sy / n;
    const double vx = sxx - sx * sx / n;
    const double vy = syy - sy * sy / n;
    r.pcc = (vx <= 0 || vy <= 0) ? 1.0 : cov / std::sqrt(vx * vy);
    return r;
}

// Run one tilize case; returns the tiled output read back from DRAM.
std::vector<bfloat16> run_case(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    distributed::MeshCommandQueue& cq,
    uint32_t W,
    bool streaming,
    const std::vector<bfloat16>& input) {
    const uint32_t row_bytes = W * 32 * sizeof(bfloat16);
    const uint32_t cb_out_tiles = streaming ? 2u : W;

    // ---- DRAM buffers ----
    distributed::DeviceLocalBufferConfig in_dram_cfg{.page_size = row_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig in_buf_cfg{.size = static_cast<uint64_t>(row_bytes) * 32};
    auto in_buf = distributed::MeshBuffer::create(in_buf_cfg, in_dram_cfg, mesh_device.get());

    distributed::DeviceLocalBufferConfig out_dram_cfg{.page_size = TILE_BYTES, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig out_buf_cfg{.size = static_cast<uint64_t>(TILE_BYTES) * W};
    auto out_buf = distributed::MeshBuffer::create(out_buf_cfg, out_dram_cfg, mesh_device.get());

    // ---- Program ----
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr auto cb_in = CBIndex::c_0;
    constexpr auto cb_out = CBIndex::c_16;

    CircularBufferConfig cb_in_cfg =
        CircularBufferConfig(W * TILE_BYTES, {{cb_in, DataFormat::Float16_b}}).set_page_size(cb_in, TILE_BYTES);
    CreateCircularBuffer(program, core, cb_in_cfg);

    CircularBufferConfig cb_out_cfg =
        CircularBufferConfig(cb_out_tiles * TILE_BYTES, {{cb_out, DataFormat::Float16_b}}).set_page_size(cb_out, TILE_BYTES);
    CreateCircularBuffer(program, core, cb_out_cfg);

    KernelHandle reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streaming_tilize/kernels/dataflow/reader_rows.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {static_cast<uint32_t>(cb_in), W, row_bytes}});

    KernelHandle writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streaming_tilize/kernels/dataflow/writer_tiles.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {static_cast<uint32_t>(cb_out), W}});

    KernelHandle compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "streaming_tilize/kernels/compute/streaming_tilize_compute.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {
                static_cast<uint32_t>(cb_in), static_cast<uint32_t>(cb_out), W, streaming ? 1u : 0u}});

    EnqueueWriteMeshBuffer(cq, in_buf, input, /*blocking=*/false);
    SetRuntimeArgs(program, reader, core, {static_cast<uint32_t>(in_buf->address())});
    SetRuntimeArgs(program, writer, core, {static_cast<uint32_t>(out_buf->address())});
    SetRuntimeArgs(program, compute, core, {});

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    std::vector<bfloat16> result;
    distributed::EnqueueReadMeshBuffer(cq, result, out_buf, /*blocking=*/true);
    return result;
}

}  // namespace

int main() {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    bool all_ok = true;
    for (uint32_t W : {4u, 128u}) {
        const uint32_t cols = W * 32;
        std::vector<bfloat16> input(static_cast<size_t>(32) * cols);
        for (auto& v : input) {
            v = bfloat16(dist(rng));
        }

        const auto golden = host_tilize(input, W);
        const auto atomic = run_case(mesh_device, cq, W, /*streaming=*/false, input);
        const auto stream = run_case(mesh_device, cq, W, /*streaming=*/true, input);

        const auto ca = compare(atomic, golden);
        const auto cs = compare(stream, golden);
        const auto cx = compare(stream, atomic);

        const bool ok = (cs.exact == cs.total) && (cx.exact == cx.total) && (ca.exact == ca.total);
        all_ok = all_ok && ok;

        fmt::print("\n================ W = {} ({} cols, cb_out stream=2 tiles) ================\n", W, cols);
        fmt::print("  atomic vs golden : exact {}/{}  PCC {:.8f}\n", ca.exact, ca.total, ca.pcc);
        fmt::print("  stream vs golden : exact {}/{}  PCC {:.8f}\n", cs.exact, cs.total, cs.pcc);
        fmt::print("  stream vs atomic : exact {}/{}  PCC {:.8f}\n", cx.exact, cx.total, cx.pcc);
        fmt::print("  => {}\n", ok ? "PASS (streaming tilize is bit-exact with a 2-tile output CB)" : "FAIL");
    }

    fmt::print("\n=========================================================\n");
    fmt::print("OVERALL: {}\n", all_ok ? "PASS" : "FAIL");
    mesh_device->close();
    return all_ok ? 0 : 1;
}
