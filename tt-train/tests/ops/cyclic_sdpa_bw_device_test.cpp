// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Algorithm 2 of the cyclic backward pass, end to end on device: the
// schedule, the gradient arithmetic, DRAM row traffic and the chip-wide
// barrier, producing dQ, dK and dV for a whole sequence.
//
// This is the first step where the answer is a gradient rather than a
// checksum. The transport variants (relay, endpoint counters) replace the
// DRAM row traffic afterwards and must produce the same numbers.
//
// The reference is a dense causal backward pass in float, with its inputs
// rounded to bfloat16 first, because that is what the device reads. Without
// that rounding the comparison is dominated by input quantisation and the
// tolerance would have to be loose enough to hide real errors.
//
// The block-pair schedule and a dense causal mask agree exactly: a pair with
// block row above block column is entirely unmasked, the diagonal pair takes
// the intra-block mask, and pairs below the diagonal are never visited and
// contribute nothing, which is also what P = 0 gives in the dense form.

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "metal/operations.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"

namespace {

using namespace ttml::metal::ops::cyclic_sdpa_bw;
namespace tt_dist = tt::tt_metal::distributed;

constexpr uint32_t kTile = 32;
constexpr const char* kReaderPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/"
    "cyclic_sdpa_bw_reader.cpp";
constexpr const char* kComputePath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/compute/"
    "cyclic_sdpa_bw_compute.cpp";
constexpr const char* kRelayReaderPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/"
    "cyclic_sdpa_bw_relay_reader.cpp";
constexpr const char* kRelayWriterPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/"
    "cyclic_sdpa_bw_relay_writer.cpp";
constexpr const char* kWriterPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/"
    "cyclic_sdpa_bw_writer.cpp";

// Round to bfloat16, ties to even, so the reference starts from the operands
// the device actually gets.
float to_bf16(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t lower = bits & 0xFFFFu;
    const uint32_t round = (lower > 0x8000u) || (lower == 0x8000u && ((bits >> 16) & 1u));
    bits = ((bits >> 16) + round) << 16;
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float sample(uint32_t seed) {
    const uint32_t h = seed * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int32_t>(h % 2001u) - 1000) / 2000.0F;
}

xt::xarray<float> random_bf16_matrix(uint32_t rows, uint32_t cols, uint32_t salt) {
    xt::xarray<float> out = xt::zeros<float>({rows, cols});
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            out(r, c) = to_bf16(sample(salt * 7919u + r * 131u + c));
        }
    }
    return out;
}

struct Reference {
    uint32_t N = 0;
    uint32_t d = 0;
    xt::xarray<float> Q, K, V, dO;
    xt::xarray<float> O;                 // the forward's output, P V
    xt::xarray<float> lse_tile, u_tile;  // (1,1,N,32), per-row value in column 0
    xt::xarray<float> dQ, dK, dV;
};

// Inputs only, with no host-side gradients. make_reference does an O(N^3)
// dense backward pass to check against, which at N = 7040 -- one schedule on
// the whole 11x10 grid -- is 350 GFLOP and a couple of gigabytes on the host.
// A timing run checks nothing, so it does not need any of that. L and D are
// plausible rather than consistent, which changes what the kernel computes
// not at all: the same tiles move and the same arithmetic runs on them.
Reference make_reference_inputs_only(uint32_t N, uint32_t d) {
    Reference r;
    r.N = N;
    r.d = d;
    r.Q = random_bf16_matrix(N, d, 1);
    r.K = random_bf16_matrix(N, d, 2);
    r.V = random_bf16_matrix(N, d, 3);
    r.dO = random_bf16_matrix(N, d, 4);
    r.lse_tile = xt::zeros<float>({1u, 1u, N, kTile});
    r.u_tile = xt::zeros<float>({1u, 1u, N, kTile});
    for (uint32_t i = 0; i < N; ++i) {
        r.lse_tile(0, 0, i, 0) = 1.0F + 0.001F * static_cast<float>(i % 97u);
        r.u_tile(0, 0, i, 0) = 0.01F * static_cast<float>(i % 13u);
    }
    return r;
}

Reference make_reference(uint32_t N, uint32_t d) {
    Reference r;
    r.N = N;
    r.d = d;
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));
    r.Q = random_bf16_matrix(N, d, 1);
    r.K = random_bf16_matrix(N, d, 2);
    r.V = random_bf16_matrix(N, d, 3);
    r.dO = random_bf16_matrix(N, d, 4);

    const xt::xarray<float> S = xt::linalg::dot(r.Q, xt::transpose(r.K)) * scale;
    xt::xarray<float> P = xt::zeros<float>({N, N});
    std::vector<float> lse(N, 0.0F);
    for (uint32_t i = 0; i < N; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j <= i; ++j) {
            m = std::max(m, S(i, j));
        }
        float sum = 0.0F;
        for (uint32_t j = 0; j <= i; ++j) {
            sum += std::exp(S(i, j) - m);
        }
        lse[i] = m + std::log(sum);
        for (uint32_t j = 0; j <= i; ++j) {
            P(i, j) = std::exp(S(i, j) - lse[i]);
        }
    }

    r.O = xt::linalg::dot(P, r.V);
    const xt::xarray<float>& O = r.O;
    const xt::xarray<float> dP = xt::linalg::dot(r.dO, xt::transpose(r.V));
    std::vector<float> u(N, 0.0F);
    for (uint32_t i = 0; i < N; ++i) {
        float sum = 0.0F;
        for (uint32_t c = 0; c < d; ++c) {
            sum += r.dO(i, c) * O(i, c);
        }
        u[i] = sum;
    }

    xt::xarray<float> dS = xt::zeros<float>({N, N});
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            dS(i, j) = P(i, j) * (dP(i, j) - u[i]) * scale;
        }
    }

    r.dQ = xt::linalg::dot(dS, r.K);
    r.dK = xt::linalg::dot(xt::transpose(dS), r.Q);
    r.dV = xt::linalg::dot(xt::transpose(P), r.dO);

    r.lse_tile = xt::zeros<float>({1u, 1u, N, kTile});
    r.u_tile = xt::zeros<float>({1u, 1u, N, kTile});
    for (uint32_t i = 0; i < N; ++i) {
        r.lse_tile(0, 0, i, 0) = lse[i];
        r.u_tile(0, 0, i, 0) = u[i];
    }
    return r;
}

xt::xarray<float> as_4d(const xt::xarray<float>& m) {
    const auto shape = m.shape();
    xt::xarray<float> out = xt::zeros<float>(
        {1u, 1u, static_cast<uint32_t>(shape[0]), static_cast<uint32_t>(shape[1])});
    for (uint32_t r = 0; r < shape[0]; ++r) {
        for (uint32_t c = 0; c < shape[1]; ++c) {
            out(0, 0, r, c) = m(r, c);
        }
    }
    return out;
}

// Median wall time of a workload already built and warmed up. Crude on
// purpose: it is dispatch-to-finish on the host, with no attribution and no
// profiler, so it says whether a change moves the needle and nothing about
// where the time goes. Tracy is the instrument for that, and it needs the
// llvm-20 binutils this build does not have.
double median_enqueue_seconds(
    tt::tt_metal::distributed::MeshCommandQueue& cq,
    tt::tt_metal::distributed::MeshWorkload& workload,
    uint32_t repeats) {
    using clock = std::chrono::steady_clock;
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);  // warm
    std::vector<double> samples;
    samples.reserve(repeats);
    for (uint32_t k = 0; k < repeats; ++k) {
        const auto start = clock::now();
        tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
        samples.push_back(std::chrono::duration<double>(clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct Gradients {
    xt::xarray<float> dQ, dK, dV;
};

Gradients run_algorithm2(
    uint32_t C,
    const Reference& ref,
    uint32_t grid_w,
    uint32_t grid_h,
    double* seconds = nullptr,
    uint32_t Bt = 1) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();

    const uint32_t qWt = ref.d / kTile;
    // How many output tiles one register acquire covers in the three gradient
    // matmuls. Profiling found the port losing most of its FPU occupancy to
    // acquire/commit/release boundaries with two tile matmuls inside each, so
    // this follows sdpa_bw and takes the largest divisor of qWt up to 4
    // rather than the 1 it used to hardcode. qWt and vWt are both d/32 here,
    // so one value serves both loops.
    const uint32_t block_size = get_block_size(qWt, 4U);
    const uint32_t vWt = ref.d / kTile;
    // A block is Bt tiles tall, so every operand of a block is that many
    // times as many tiles and the score intermediates are Bt * Bt.
    const uint32_t rowT = Bt * qWt;
    const uint32_t valT = Bt * vWt;
    const uint32_t scoreT = Bt * Bt;

    const auto query = ttml::core::from_xtensor(as_4d(ref.Q), device);
    const auto key = ttml::core::from_xtensor(as_4d(ref.K), device);
    const auto value = ttml::core::from_xtensor(as_4d(ref.V), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d(ref.dO), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(ref.lse_tile, device);
    const auto u_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(ref.u_tile, device);

    // All three gradients start at zero and are read back, added to and
    // written every timestep.
    const xt::xarray<float> zeros = xt::zeros<float>({1u, 1u, ref.N, ref.d});
    const auto grad_query = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);
    const auto grad_key = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);
    const auto grad_value = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);

    auto program = CreateProgram();
    const auto region = CoreRange(CoreCoord{0, 0}, CoreCoord{grid_w - 1, grid_h - 1});

    const uint32_t bf16_tile = 2 * kTile * kTile;
    const uint32_t fp32_tile = 4 * kTile * kTile;
    const auto make_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format) {
        const uint32_t page = (format == tt::DataFormat::Float32) ? fp32_tile : bf16_tile;
        CreateCircularBuffer(
            program, region,
            CircularBufferConfig(tiles * page, {{index, format}}).set_page_size(index, page));
    };
    // Operands, per timestep.
    make_cb(tt::CBIndex::c_0, rowT, tt::DataFormat::Float16_b);  // Q_i
    make_cb(tt::CBIndex::c_1, rowT, tt::DataFormat::Float16_b);  // K_j
    make_cb(tt::CBIndex::c_2, valT, tt::DataFormat::Float16_b);  // V_j
    make_cb(tt::CBIndex::c_3, valT, tt::DataFormat::Float16_b);  // dO_i
    make_cb(tt::CBIndex::c_4, Bt, tt::DataFormat::Float32);      // L_i
    make_cb(tt::CBIndex::c_5, Bt, tt::DataFormat::Float32);      // D_i
    make_cb(tt::CBIndex::c_6, 1, tt::DataFormat::Float16_b);     // causal mask
    // Intermediates.
    make_cb(tt::CBIndex::c_10, scoreT, tt::DataFormat::Float32);  // P
    make_cb(tt::CBIndex::c_11, scoreT, tt::DataFormat::Float32);  // dP
    make_cb(tt::CBIndex::c_12, scoreT, tt::DataFormat::Float32);  // dS
    make_cb(tt::CBIndex::c_13, scoreT, tt::DataFormat::Float32);  // dS^T
    make_cb(tt::CBIndex::c_14, scoreT, tt::DataFormat::Float32);  // P^T
    // Each gradient: seed from the reader, accumulator, output to the writer.
    make_cb(tt::CBIndex::c_15, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_16, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_17, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_18, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_19, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_20, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_21, valT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_22, valT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_23, valT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_24, 1, tt::DataFormat::Float32);  // control word

    const uint32_t arrive_sem = CreateSemaphore(program, region, 0);
    const uint32_t release_sem = CreateSemaphore(program, region, 0);

    std::vector<uint32_t> reader_args = {C, qWt, vWt, release_sem, Bt};
    for (const auto* t : {&query, &key, &value, &grad_output, &lse, &u_scalar, &grad_query,
                          &grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(reader_args);
    }
    const auto reader = CreateKernel(
        program, kReaderPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_args});

    std::vector<uint32_t> writer_args = {C, qWt, vWt, arrive_sem, release_sem, Bt};
    for (const auto* t : {&grad_query, &grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(writer_args);
    }
    const auto writer = CreateKernel(
        program, kWriterPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_args});

    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(ref.d)));
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    // sqrt(d): the kernel folds the softmax scale into the exponential, so it
    // needs the reciprocal to divide the statistic it subtracts.
    const uint32_t inv_scaler = std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(ref.d)));
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());
    // Every buffer that is copied into DST rather than fed to a matmul keeps
    // FP32 through the copy: P for the elementwise dS chain, and the gradient
    // seeds and accumulators, which carry running sums.
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_mode[tt::CBIndex::c_10] = UnpackToDestMode::UnpackToDestFp32;
    const auto compute = CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            // Half-sync DST holds four Float32 tiles, and Bt = 2 needs
            // exactly four: two score tiles in even registers with a scratch
            // register beside each. Taller blocks need the whole register
            // file, which costs the pipelining between math and pack that
            // half-sync buys.
            .dst_full_sync_en = Bt > 2,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {C, qWt, vWt, scaler, minus_one, custom_inf, block_size, Bt, inv_scaler}});

    const auto coordinator_logical = placement_of(C, grid_w, 1);
    const auto coordinator = device->worker_core_from_logical_core(
        CoreCoord{coordinator_logical.x, coordinator_logical.y});
    const auto mcast_start = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const auto mcast_end =
        device->worker_core_from_logical_core(CoreCoord{grid_w - 1, grid_h - 1});

    for (uint32_t c = 1; c <= C; ++c) {
        const auto xy = placement_of(C, grid_w, c);
        const auto core = CoreCoord{xy.x, xy.y};
        SetRuntimeArgs(
            program, reader, core,
            {c, query.buffer()->address(), key.buffer()->address(), value.buffer()->address(),
             grad_output.buffer()->address(), lse.buffer()->address(), u_scalar.buffer()->address(),
             grad_query.buffer()->address(), grad_key.buffer()->address(),
             grad_value.buffer()->address()});
        SetRuntimeArgs(
            program, writer, core,
            {c, grad_query.buffer()->address(), grad_key.buffer()->address(),
             grad_value.buffer()->address(), static_cast<uint32_t>(coordinator.x),
             static_cast<uint32_t>(coordinator.y), static_cast<uint32_t>(mcast_start.x),
             static_cast<uint32_t>(mcast_start.y), static_cast<uint32_t>(mcast_end.x),
             static_cast<uint32_t>(mcast_end.y), c == 1u ? 1u : 0u});
        SetRuntimeArgs(program, compute, core, {c});
    }

    auto workload = tt_dist::MeshWorkload();
    workload.add_program(tt_dist::MeshCoordinateRange(device->shape()), std::move(program));
    tt_dist::EnqueueMeshWorkload(device->mesh_command_queue(), workload, /*blocking=*/true);
    if (seconds != nullptr) {
        *seconds = median_enqueue_seconds(device->mesh_command_queue(), workload, 5);
    }

    Gradients out;
    out.dQ = ttml::core::to_xtensor(grad_query);
    out.dK = ttml::core::to_xtensor(grad_key);
    out.dV = ttml::core::to_xtensor(grad_value);
    return out;
}

// Algorithm 3: the row packet stays in L1 across an active streak and is
// forwarded to the next consumer, instead of being reloaded from DRAM every
// timestep. The compute kernel is the same one Algorithm 2 uses -- only where
// its operands come from changes -- and the answer has to be the same
// gradients.
//
// The paper's two receive slots are the compute kernel's input buffers with
// room for two packets, so the circular-buffer protocol carries the credits
// and readiness. A producer writes into the receiver's slot at
// base + (u mod 2) * stride, which is where the receiver's write pointer
// stands after u pushes; every core has the same layout, so a producer can
// use its own base as the receiver's.
// The same matrix in every (batch, head) slice of a 4D tensor. Groups all
// run the same problem here, which tests the partitioning without needing a
// separate reference per group.
xt::xarray<float> as_4d_repeated(const xt::xarray<float>& m, uint32_t groups) {
    const auto shape = m.shape();
    const auto rows = static_cast<uint32_t>(shape[0]);
    const auto cols = static_cast<uint32_t>(shape[1]);
    xt::xarray<float> out = xt::zeros<float>({1u, groups, rows, cols});
    for (uint32_t g = 0; g < groups; ++g) {
        xt::view(out, 0, g, xt::all(), xt::all()) = m;
    }
    return out;
}

xt::xarray<float> repeat_4d(const xt::xarray<float>& t, uint32_t groups) {
    const auto shape = t.shape();
    const auto rows = static_cast<uint32_t>(shape[2]);
    const auto cols = static_cast<uint32_t>(shape[3]);
    xt::xarray<float> out = xt::zeros<float>({1u, groups, rows, cols});
    for (uint32_t g = 0; g < groups; ++g) {
        xt::view(out, 0, g, xt::all(), xt::all()) = xt::view(t, 0, 0, xt::all(), xt::all());
    }
    return out;
}

// `groups` independent schedules on disjoint sub-rectangles of the grid, each
// working a different (batch, head) slice. Nothing crosses between them: the
// snake stays inside a group, and so does the barrier's multicast. This is
// how the port fills a grid when the sequence length is fixed by the model --
// T = 2C blocks means one schedule needs N = 2 C B rows, so a short sequence
// leaves cores idle unless several heads run side by side.
Gradients run_relay(
    uint32_t C,
    const Reference& ref,
    uint32_t grid_w,
    uint32_t grid_h,
    bool endpoint_sync = false,
    double* seconds = nullptr,
    uint32_t Bt = 1,
    uint32_t groups = 1) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();

    const uint32_t qWt = ref.d / kTile;
    // How many output tiles one register acquire covers in the three gradient
    // matmuls. Profiling found the port losing most of its FPU occupancy to
    // acquire/commit/release boundaries with two tile matmuls inside each, so
    // this follows sdpa_bw and takes the largest divisor of qWt up to 4
    // rather than the 1 it used to hardcode. qWt and vWt are both d/32 here,
    // so one value serves both loops.
    const uint32_t block_size = get_block_size(qWt, 4U);
    const uint32_t vWt = ref.d / kTile;
    // A block is Bt tiles tall, so every operand of a block is that many
    // times as many tiles and the score intermediates are Bt * Bt.
    const uint32_t rowT = Bt * qWt;
    const uint32_t valT = Bt * vWt;
    const uint32_t scoreT = Bt * Bt;

    const auto query = ttml::core::from_xtensor(as_4d_repeated(ref.Q, groups), device);
    const auto key = ttml::core::from_xtensor(as_4d_repeated(ref.K, groups), device);
    const auto value = ttml::core::from_xtensor(as_4d_repeated(ref.V, groups), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d_repeated(ref.dO, groups), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(ref.lse_tile, groups), device);
    const auto u_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(ref.u_tile, groups), device);

    const xt::xarray<float> zeros = xt::zeros<float>({1u, groups, ref.N, ref.d});
    const auto grad_query = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);
    const auto grad_key = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);
    const auto grad_value = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);

    auto program = CreateProgram();
    // Groups tile the grid: as many across as fit, then down. The kernels are
    // created once over the union, and each core is told which group it is in
    // through its runtime arguments.
    const auto dev_grid = device->compute_with_storage_grid_size();
    const uint32_t groups_x =
        std::max(1u, static_cast<uint32_t>(dev_grid.x) / grid_w);
    std::vector<CoreCoord> group_origin;
    for (uint32_t g = 0; g < groups; ++g) {
        group_origin.push_back(CoreCoord{(g % groups_x) * grid_w, (g / groups_x) * grid_h});
    }
    // The union of the group rectangles, not their bounding box. A bounding
    // box is wrong whenever the groups do not tile it exactly -- on an 11x10
    // grid, eight groups of 2x2 occupy 8x2 and 4x2 across two rows, and the
    // enclosing 10x4 box holds eight cores belonging to no group. Those cores
    // still get the kernels, never get runtime arguments, and then wait on
    // semaphores that nobody will ever post to: a hang, not an error.
    std::vector<CoreRange> group_range;
    for (const auto& o : group_origin) {
        TT_FATAL(
            o.x + grid_w <= dev_grid.x && o.y + grid_h <= dev_grid.y,
            "group at ({}, {}) of {}x{} runs off a {}x{} grid",
            o.x,
            o.y,
            grid_w,
            grid_h,
            dev_grid.x,
            dev_grid.y);
        group_range.emplace_back(o, CoreCoord{o.x + grid_w - 1, o.y + grid_h - 1});
    }
    const auto region = CoreRangeSet(group_range);

    const uint32_t bf16_tile = 2 * kTile * kTile;
    const uint32_t fp32_tile = 4 * kTile * kTile;
    const auto make_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format) {
        const uint32_t page = (format == tt::DataFormat::Float32) ? fp32_tile : bf16_tile;
        CreateCircularBuffer(
            program, region,
            CircularBufferConfig(tiles * page, {{index, format}}).set_page_size(index, page));
    };
    // The packet buffers hold two slots; everything else holds one.
    make_cb(tt::CBIndex::c_0, 2 * rowT, tt::DataFormat::Float16_b);  // Q_i
    make_cb(tt::CBIndex::c_3, 2 * valT, tt::DataFormat::Float16_b);  // dO_i
    make_cb(tt::CBIndex::c_4, 2 * Bt, tt::DataFormat::Float32);      // L_i
    make_cb(tt::CBIndex::c_5, 2 * Bt, tt::DataFormat::Float32);      // D_i
    make_cb(tt::CBIndex::c_15, 2 * rowT, tt::DataFormat::Float32);   // dQ_i, travels along
    make_cb(tt::CBIndex::c_1, rowT, tt::DataFormat::Float16_b);      // K_j
    make_cb(tt::CBIndex::c_2, valT, tt::DataFormat::Float16_b);      // V_j
    make_cb(tt::CBIndex::c_6, 1, tt::DataFormat::Float16_b);         // causal mask
    make_cb(tt::CBIndex::c_10, scoreT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_11, scoreT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_12, scoreT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_13, scoreT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_14, scoreT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_16, rowT, tt::DataFormat::Float32);  // dQ accumulator
    make_cb(tt::CBIndex::c_17, rowT, tt::DataFormat::Float32);  // dQ to the relay
    make_cb(tt::CBIndex::c_18, rowT, tt::DataFormat::Float32);  // dK seed
    make_cb(tt::CBIndex::c_19, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_20, rowT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_21, valT, tt::DataFormat::Float32);  // dV seed
    make_cb(tt::CBIndex::c_22, valT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_23, valT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_24, 1, tt::DataFormat::Float32);  // readiness word
    make_cb(tt::CBIndex::c_25, 1, tt::DataFormat::Float32);  // release word
    make_cb(tt::CBIndex::c_26, 1, tt::DataFormat::Float32);  // column-gradient progress
    make_cb(tt::CBIndex::c_7, 2, tt::DataFormat::Float32);   // slot-release tokens

    const uint32_t arrive_sem = CreateSemaphore(program, region, 0);
    const uint32_t release_sem = CreateSemaphore(program, region, 0);
    // Two readiness words per slot: the immutable fields and dQ arrive, and
    // are needed, at different times.
    const uint32_t ready_imm0_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_imm1_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_dq0_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready_dq1_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_prev_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_next_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_self_sem = CreateSemaphore(program, region, 0);
    const uint32_t endpoint1_sem = CreateSemaphore(program, region, 0);
    const uint32_t endpoint2_sem = CreateSemaphore(program, region, 0);

    std::map<std::string, std::string> sync_defines;
    if (endpoint_sync) {
        sync_defines["ENDPOINT_SYNC"] = "1";
    }
    // The relay keeps the column state resident, as the paper specifies.
    std::map<std::string, std::string> compute_defines = sync_defines;
    compute_defines["COLUMN_RESIDENT"] = "1";
    compute_defines["RELEASE_TOKEN"] = "1";

    std::vector<uint32_t> reader_args = {
        C, qWt, vWt, release_sem, ready_imm0_sem, ready_imm1_sem, ready_dq0_sem,
        ready_dq1_sem, credit_prev_sem, credit_next_sem, credit_self_sem, endpoint1_sem,
        endpoint2_sem, Bt};
    for (const auto* t : {&query, &key, &value, &grad_output, &lse, &u_scalar, &grad_query,
                          &grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(reader_args);
    }
    const auto reader = CreateKernel(
        program, kRelayReaderPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_args,
            .defines = sync_defines});

    std::vector<uint32_t> writer_args = {C, qWt, vWt, arrive_sem, release_sem, Bt};
    for (const auto* t : {&grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(writer_args);
    }
    const auto writer = CreateKernel(
        program, kRelayWriterPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_args,
            .defines = sync_defines});

    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(ref.d)));
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    // sqrt(d): the kernel folds the softmax scale into the exponential, so it
    // needs the reciprocal to divide the statistic it subtracts.
    const uint32_t inv_scaler = std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(ref.d)));
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_mode[tt::CBIndex::c_10] = UnpackToDestMode::UnpackToDestFp32;
    const auto compute = CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            // Half-sync DST holds four Float32 tiles, and Bt = 2 needs
            // exactly four: two score tiles in even registers with a scratch
            // register beside each. Taller blocks need the whole register
            // file, which costs the pipelining between math and pack that
            // half-sync buys.
            .dst_full_sync_en = Bt > 2,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {C, qWt, vWt, scaler, minus_one, custom_inf, block_size, Bt, inv_scaler},
            .defines = compute_defines});

    // Everything below is per group: the snake, the barrier's coordinator and
    // its multicast rectangle all live inside one sub-rectangle, which is why
    // several schedules can share a grid without knowing about each other.
    for (uint32_t g = 0; g < groups; ++g) {
        const auto origin = group_origin[g];
        const auto logical_of = [&](uint32_t core) {
            const auto xy = placement_of(C, grid_w, core);
            return CoreCoord{origin.x + xy.x, origin.y + xy.y};
        };
        const auto noc_of_core = [&](uint32_t core) {
            return device->worker_core_from_logical_core(logical_of(core));
        };
        const auto coordinator = noc_of_core(1);
        const auto mcast_start = device->worker_core_from_logical_core(origin);
        const auto mcast_end = device->worker_core_from_logical_core(
            CoreCoord{origin.x + grid_w - 1, origin.y + grid_h - 1});

        for (uint32_t c = 1; c <= C; ++c) {
            const auto core = logical_of(c);
            const auto neighbors = snake_neighbors(C, c);
            const auto prev = noc_of_core(neighbors.prev != kNoCore ? neighbors.prev : c);
            const auto next = noc_of_core(neighbors.next != kNoCore ? neighbors.next : c);
            std::vector<uint32_t> relay_reader_args = {
                c, g, query.buffer()->address(), key.buffer()->address(),
                value.buffer()->address(), grad_output.buffer()->address(),
                lse.buffer()->address(), u_scalar.buffer()->address(),
                grad_query.buffer()->address(), grad_key.buffer()->address(),
                grad_value.buffer()->address(), static_cast<uint32_t>(prev.x),
                static_cast<uint32_t>(prev.y), static_cast<uint32_t>(next.x),
                static_cast<uint32_t>(next.y)};
            // Every core in this group, for the endpoint reads.
            for (uint32_t r = 1; r <= C; ++r) {
                const auto rc = noc_of_core(r);
                relay_reader_args.push_back(static_cast<uint32_t>(rc.x));
                relay_reader_args.push_back(static_cast<uint32_t>(rc.y));
            }
            SetRuntimeArgs(program, reader, core, relay_reader_args);
            SetRuntimeArgs(
                program, writer, core,
                {c, g, grad_key.buffer()->address(), grad_value.buffer()->address(),
                 static_cast<uint32_t>(coordinator.x), static_cast<uint32_t>(coordinator.y),
                 static_cast<uint32_t>(mcast_start.x), static_cast<uint32_t>(mcast_start.y),
                 static_cast<uint32_t>(mcast_end.x), static_cast<uint32_t>(mcast_end.y),
                 c == 1u ? 1u : 0u});
            SetRuntimeArgs(program, compute, core, {c});
        }
    }

    auto workload = tt_dist::MeshWorkload();
    workload.add_program(tt_dist::MeshCoordinateRange(device->shape()), std::move(program));
    tt_dist::EnqueueMeshWorkload(device->mesh_command_queue(), workload, /*blocking=*/true);
    if (seconds != nullptr) {
        *seconds = median_enqueue_seconds(device->mesh_command_queue(), workload, 5);
    }

    Gradients out;
    out.dQ = ttml::core::to_xtensor(grad_query);
    out.dK = ttml::core::to_xtensor(grad_key);
    out.dV = ttml::core::to_xtensor(grad_value);
    return out;
}

void expect_close(
    const xt::xarray<float>& got,
    const xt::xarray<float>& want,
    float tolerance,
    const std::string& what,
    uint32_t slice = 0) {
    const uint32_t rows = static_cast<uint32_t>(want.shape()[0]);
    const uint32_t cols = static_cast<uint32_t>(want.shape()[1]);
    float max_abs = 0.0F;
    float max_diff = 0.0F;
    uint32_t worst_r = 0;
    uint32_t worst_c = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            max_abs = std::max(max_abs, std::abs(want(r, c)));
            const float diff = std::abs(got(0, slice, r, c) - want(r, c));
            if (diff > max_diff) {
                max_diff = diff;
                worst_r = r;
                worst_c = c;
            }
        }
    }
    const float relative = max_abs > 0.0F ? max_diff / max_abs : max_diff;
    EXPECT_LT(relative, tolerance)
        << what << ": max |diff| " << max_diff << " against max |ref| " << max_abs
        << " (relative " << relative << "), worst at (" << worst_r << "," << worst_c << ") got "
        << got(0, slice, worst_r, worst_c) << " want " << want(worst_r, worst_c);
}

void check_algorithm2(uint32_t C, uint32_t grid_w, uint32_t grid_h, uint32_t d = 64) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid_w > grid.x || grid_h > grid.y) {
        GTEST_SKIP() << "needs " << grid_w << "x" << grid_h;
    }
    const uint32_t N = 2u * C * kTile;  // T = 2C blocks of B = 32 rows
    const auto ref = make_reference(N, d);
    const auto got = run_algorithm2(C, ref, grid_w, grid_h);
    expect_close(got.dQ, ref.dQ, 0.06F, "dQ");
    expect_close(got.dK, ref.dK, 0.06F, "dK");
    expect_close(got.dV, ref.dV, 0.06F, "dV");
}

// Every group runs the same problem on its own slice, so every slice has to
// come out equal to the reference. A group reading or writing outside its
// slice shows up as a wrong answer in some other group's.
void check_relay_groups(
    uint32_t C, uint32_t grid_w, uint32_t grid_h, uint32_t groups, uint32_t Bt = 1) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    const uint32_t groups_x = std::max(1u, static_cast<uint32_t>(grid.x) / grid_w);
    const uint32_t rows = (groups + groups_x - 1u) / groups_x;
    if (grid_w * std::min(groups, groups_x) > grid.x || grid_h * rows > grid.y) {
        GTEST_SKIP() << groups << " groups of " << grid_w << "x" << grid_h << " do not fit";
    }
    const auto ref = make_reference(2u * C * Bt * kTile, 64);
    const auto got = run_relay(
        C, ref, grid_w, grid_h, /*endpoint_sync=*/true, nullptr, Bt, groups);
    for (uint32_t g = 0; g < groups; ++g) {
        const std::string at = " in group " + std::to_string(g);
        expect_close(got.dQ, ref.dQ, 0.06F, "dQ" + at, g);
        expect_close(got.dK, ref.dK, 0.06F, "dK" + at, g);
        expect_close(got.dV, ref.dV, 0.06F, "dV" + at, g);
    }
}

void check_relay(
    uint32_t C,
    uint32_t grid_w,
    uint32_t grid_h,
    uint32_t d = 64,
    bool endpoint_sync = false,
    uint32_t Bt = 1) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid_w > grid.x || grid_h > grid.y) {
        GTEST_SKIP() << "needs " << grid_w << "x" << grid_h;
    }
    // T = 2C blocks of B = Bt * 32 rows each.
    const uint32_t N = 2u * C * Bt * kTile;
    const auto ref = make_reference(N, d);
    const auto got = run_relay(C, ref, grid_w, grid_h, endpoint_sync, nullptr, Bt);
    expect_close(got.dQ, ref.dQ, 0.06F, "dQ");
    expect_close(got.dK, ref.dK, 0.06F, "dK");
    expect_close(got.dV, ref.dV, 0.06F, "dV");
}

}  // namespace

// One core, three block pairs: (2,1), (2,2), (1,1). The whole causal set for
// T = 2, including a diagonal pair and a revisit of column 1.
TEST(CyclicSdpaBwAlgorithm2Test, OneCore) {
    check_algorithm2(1, 1, 1);
}

// Two cores on a 1x2 region, not 2x1. A release multicast whose destination
// rectangle is one row high trips a device-side assert that the watcher
// catches; one column wide is fine, and so is any taller rectangle. The
// serpentine embedding is a single hop either way, so the shape is free to
// choose -- but it is not arbitrary.
TEST(CyclicSdpaBwAlgorithm2Test, TwoCores) {
    check_algorithm2(2, 1, 2);
}

TEST(CyclicSdpaBwAlgorithm2Test, FourCores) {
    check_algorithm2(4, 2, 2);
}

TEST(CyclicSdpaBwAlgorithm2Test, EightCores) {
    check_algorithm2(8, 4, 2);
}

TEST(CyclicSdpaBwAlgorithm2Test, SixteenCores) {
    check_algorithm2(16, 4, 4);
}

// A wider head dimension, so every matmul runs over four inner tiles.
TEST(CyclicSdpaBwAlgorithm2Test, FourCoresWiderHead) {
    check_algorithm2(4, 2, 2, /*d=*/128);
}

// ----------------------------------------------------- Algorithm 3: relay
// At C = 1 every forward is the self-transition -- core 1 handing the packet
// from one of its own slots to the other -- so this exercises the slot
// lifetimes and the readiness tags with no NoC traffic at all.
TEST(CyclicSdpaBwRelayTest, OneCoreSelfTransitionOnly) {
    check_relay(1, 1, 1);
}

TEST(CyclicSdpaBwRelayTest, TwoCores) {
    check_relay(2, 1, 2);
}

TEST(CyclicSdpaBwRelayTest, FourCores) {
    check_relay(4, 2, 2);
}

TEST(CyclicSdpaBwRelayTest, EightCores) {
    check_relay(8, 4, 2);
}

// Blocks two tiles tall. The schedule is untouched -- a block is still a
// block, and T is still 2C -- but each one covers 64 rows instead of 32, so
// the packet carries twice the tiles, the score stages have four instead of
// one, and the diagonal block needs the triangle handled per tile. What it
// buys is FPU occupancy: measured on a single block pair, four times the
// score-stage arithmetic for 2.12x the cycles.
TEST(CyclicSdpaBwRelayTest, TwoCoresTallBlocks) {
    check_relay(2, 1, 2, 64, /* endpoint_sync */ false, /* Bt */ 2);
}

TEST(CyclicSdpaBwRelayTest, FourCoresTallBlocks) {
    check_relay(4, 2, 2, 64, /* endpoint_sync */ false, /* Bt */ 2);
}

// Four tiles tall, which is the most the register scheme allows: score tiles
// sit in even registers so that the mask and the softmax each have a scratch
// register beside them, and FP32 dest has eight.
TEST(CyclicSdpaBwRelayTest, FourCoresFourTileBlocks) {
    check_relay(4, 2, 2, 64, /* endpoint_sync */ false, /* Bt */ 4);
}

// Independent schedules side by side. T = 2C blocks means one schedule needs
// N = 2 C B rows, so at a sequence length the model fixes, a single schedule
// may not fill the grid -- four heads of a 16-core schedule do, on four
// quadrants of an 8x8. Nothing crosses between them: the snake stays inside a
// group and so does the barrier's multicast, which is why this works at all.
TEST(CyclicSdpaBwGroupTest, TwoGroupsOfFour) {
    check_relay_groups(4, 2, 2, /* groups */ 2);
}

TEST(CyclicSdpaBwGroupTest, FourGroupsOfFour) {
    check_relay_groups(4, 2, 2, /* groups */ 4);
}

TEST(CyclicSdpaBwGroupTest, FourGroupsOfSixteen) {
    check_relay_groups(16, 4, 4, /* groups */ 4);
}

TEST(CyclicSdpaBwGroupTest, EightGroupsOfFour) {
    check_relay_groups(4, 2, 2, /* groups */ 8);
}

TEST(CyclicSdpaBwGroupTest, SixteenGroupsOfFour) {
    check_relay_groups(4, 2, 2, /* groups */ 16);
}

TEST(CyclicSdpaBwGroupTest, FourGroupsWithTallBlocks) {
    check_relay_groups(4, 2, 2, /* groups */ 4, /* Bt */ 2);
}

// ------------------------------------------------------------------ the op
// Everything above builds its own program. This is the op: it derives the
// layout from the tensor shapes and the device's grid, allocates its own
// outputs, and is what a model would call.
void check_op(uint32_t C, uint32_t Bt, uint32_t slices, bool use_barrier, uint32_t d = 64) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t N = 2u * C * Bt * kTile;
    const auto ref = make_reference(N, d);

    const auto query = ttml::core::from_xtensor(as_4d_repeated(ref.Q, slices), device);
    const auto key = ttml::core::from_xtensor(as_4d_repeated(ref.K, slices), device);
    const auto value = ttml::core::from_xtensor(as_4d_repeated(ref.V, slices), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d_repeated(ref.dO, slices), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(ref.lse_tile, slices), device);
    const auto row_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(ref.u_tile, slices), device);

    const auto [grad_query, grad_key, grad_value] =
        ttml::metal::cyclic_sdpa_bw(query, key, value, grad_output, lse, row_scalar, Bt, use_barrier);

    const auto dQ = ttml::core::to_xtensor(grad_query);
    const auto dK = ttml::core::to_xtensor(grad_key);
    const auto dV = ttml::core::to_xtensor(grad_value);
    for (uint32_t g = 0; g < slices; ++g) {
        const std::string at = " slice " + std::to_string(g);
        expect_close(dQ, ref.dQ, 0.06F, "dQ" + at, g);
        expect_close(dK, ref.dK, 0.06F, "dK" + at, g);
        expect_close(dV, ref.dV, 0.06F, "dV" + at, g);
    }
}

TEST(CyclicSdpaBwOpTest, OneSlice) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false);
}

// C is not passed in: it follows from the sequence length and the block
// height, so N = 512 with Bt = 1 asks for eight cores by arithmetic alone.
TEST(CyclicSdpaBwOpTest, CoresFollowFromTheSequenceLength) {
    check_op(/* C */ 8, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwOpTest, TallBlocks) {
    check_op(/* C */ 4, /* Bt */ 2, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwOpTest, WiderHead) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false, /* d */ 128);
}

// Several (batch, head) slices at once, each on its own rectangle.
TEST(CyclicSdpaBwOpTest, FourSlices) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 4, /* use_barrier */ false);
}

// The barrier variant, reachable through the same entry point.
TEST(CyclicSdpaBwOpTest, WithTheBarrier) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ true);
}

// The op is program-cached, so a second call with different tensors must be
// re-pointed at the new buffers rather than rebuilt. That path is
// override_runtime_arguments, and it is where a stale address would show.
TEST(CyclicSdpaBwOpTest, SurvivesAProgramCacheHit) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false);
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false);
}

// The overload that takes what a forward pass hands back, computing
// D = rowsum(dO . O) itself. This is the path a model would use, so what it
// tests is the convention: that a width reduction leaves the row value where
// the kernel reads it, in column 0 of a tile.
TEST(CyclicSdpaBwOpTest, ComputesTheRowScalarFromTheAttentionOutput) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t C = 4;
    const uint32_t d = 64;
    const auto ref = make_reference(2u * C * kTile, d);

    const auto query = ttml::core::from_xtensor(as_4d_repeated(ref.Q, 1), device);
    const auto key = ttml::core::from_xtensor(as_4d_repeated(ref.K, 1), device);
    const auto value = ttml::core::from_xtensor(as_4d_repeated(ref.V, 1), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d_repeated(ref.dO, 1), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(ref.lse_tile, 1), device);
    // O = P V, which is what the forward leaves behind.
    const auto attn_output = ttml::core::from_xtensor(as_4d_repeated(ref.O, 1), device);

    const auto [grad_query, grad_key, grad_value] = ttml::metal::cyclic_sdpa_bw_from_forward(
        query, key, value, grad_output, attn_output, lse, /* Bt */ 1, /* use_barrier */ false);

    expect_close(ttml::core::to_xtensor(grad_query), ref.dQ, 0.06F, "dQ from the forward's output");
    expect_close(ttml::core::to_xtensor(grad_key), ref.dK, 0.06F, "dK from the forward's output");
    expect_close(ttml::core::to_xtensor(grad_value), ref.dV, 0.06F, "dV from the forward's output");
}

// ------------------------------------------ Algorithm 4: no chip-wide barrier
// Within a streak the packet is the ordering token. Across a gap, the two
// endpoint counters order a reload after the preceding streak's spill. The
// column gradients, which only pass through DRAM because this step has not
// restored the paper's column residency, are ordered by a local word between
// this core's own two data-movement RISCs -- no chip-wide anything.
//
// At C = 1 there is nothing to publish: both rows have a single streak, so
// every spill is final. That makes it the configuration to run first, since
// it checks that removing the barrier did not break the parts that do not
// need it.
TEST(CyclicSdpaBwEndpointTest, OneCore) {
    check_relay(1, 1, 1, 64, /*endpoint_sync=*/true);
}

// C = 4 is the first size with later streak starts -- row 5's active
// timesteps are {0}, {3,4}, {7,8}, so it reloads twice -- and therefore the
// first that exercises publications and endpoint waits.
TEST(CyclicSdpaBwEndpointTest, FourCores) {
    check_relay(4, 2, 2, 64, true);
}

TEST(CyclicSdpaBwEndpointTest, EightCores) {
    check_relay(8, 4, 2, 64, true);
}

TEST(CyclicSdpaBwEndpointTest, SixteenCores) {
    check_relay(16, 4, 4, 64, true);
}

// Toward the target configuration: C = 32 is N = 2048 over 65 timesteps, and
// C = 64 is N = 4096 over 129. The CPU reference is O(N^2 d), so these are
// the slow tests in the suite.
TEST(CyclicSdpaBwEndpointTest, ThirtyTwoCores) {
    check_relay(32, 8, 4, 64, true);
}

// The barrier-free variant with tall blocks: the endpoint counters order a
// reload against a spill of a whole block now, not a single tile row.
TEST(CyclicSdpaBwEndpointTest, FourCoresTallBlocks) {
    check_relay(4, 2, 2, 64, /* endpoint_sync */ true, /* Bt */ 2);
}

TEST(CyclicSdpaBwEndpointTest, FourCoresFourTileBlocks) {
    check_relay(4, 2, 2, 64, /* endpoint_sync */ true, /* Bt */ 4);
}

TEST(CyclicSdpaBwEndpointTest, SixtyFourCores) {
    check_relay(64, 8, 8, 64, true);
}

// ------------------------------------------------------- bitwise identity
// The three variants run the same schedule, the same five matmuls and the
// same accumulation order; only the synchronisation and the route the row
// packet takes differ. So they should not merely agree to a tolerance, they
// should produce identical bits -- the property test_precision.py pins in the
// simulator, where the relay's DRAM round-trip costs nothing because it
// happens at the accumulation dtype.
//
// This is a much sharper check than comparing each against the reference: it
// fails on a single flipped bit anywhere in the transport, and it cannot be
// satisfied by a transport that quietly reorders or drops an update.
void expect_identical(
    const xt::xarray<float>& a, const xt::xarray<float>& b, const std::string& what) {
    ASSERT_EQ(a.size(), b.size()) << what;
    const float* pa = a.data();
    const float* pb = b.data();
    for (size_t k = 0; k < a.size(); ++k) {
        if (std::memcmp(&pa[k], &pb[k], sizeof(float)) != 0) {
            FAIL() << what << ": first difference at element " << k << ", " << pa[k] << " against "
                   << pb[k];
        }
    }
}

float max_relative_error(const xt::xarray<float>& got, const xt::xarray<float>& want) {
    const uint32_t rows = static_cast<uint32_t>(want.shape()[0]);
    const uint32_t cols = static_cast<uint32_t>(want.shape()[1]);
    float max_abs = 0.0F;
    float max_diff = 0.0F;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            max_abs = std::max(max_abs, std::abs(want(r, c)));
            max_diff = std::max(max_diff, std::abs(got(0, 0, r, c) - want(r, c)));
        }
    }
    return max_abs > 0.0F ? max_diff / max_abs : max_diff;
}

// Removing the barrier changes nothing at all: the relay and the endpoint
// relay differ only in synchronisation, so they agree to the bit. Algorithm 2
// agrees on dQ too, whose route is the same in all three.
//
// Its column gradients differ in the last bits, and that is not a defect in
// either: Algorithm 2 reloads dK_j and dV_j every timestep, and the copy back
// into the accumulator goes through the Src registers, which do not carry
// Float32. Residency keeps the accumulator in L1 across the whole interval
// and never takes that copy, so it is the more accurate of the two -- which
// the next test checks rather than assumes.
TEST(CyclicSdpaBwIdentityTest, RemovingTheBarrierChangesNothing) {
    const uint32_t C = 4;
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid.x < 2 || grid.y < 2) {
        GTEST_SKIP() << "needs a 2x2 region";
    }
    const auto ref = make_reference(2u * C * kTile, 64);

    const auto algorithm2 = run_algorithm2(C, ref, 2, 2);
    const auto relay = run_relay(C, ref, 2, 2, /*endpoint_sync=*/false);
    const auto endpoint = run_relay(C, ref, 2, 2, /*endpoint_sync=*/true);

    expect_identical(relay.dQ, endpoint.dQ, "dQ, the relay against endpoint counters");
    expect_identical(relay.dK, endpoint.dK, "dK, the relay against endpoint counters");
    expect_identical(relay.dV, endpoint.dV, "dV, the relay against endpoint counters");

    // dQ travels the same route in all three.
    expect_identical(algorithm2.dQ, relay.dQ, "dQ, Algorithm 2 against the relay");
}

// The same claim at every block shape. It is the one the whole comparison
// between the two rests on -- if they are bitwise equal then the endpoint
// counters order exactly what the barrier ordered, and any timing difference
// between them is the cost of the ordering and nothing else. Tall blocks
// change what a spill covers, a whole block rather than one tile row, so the
// endpoint thresholds are worth re-pinning against them.
TEST(CyclicSdpaBwIdentityTest, RemovingTheBarrierChangesNothingWithTallBlocks) {
    const uint32_t C = 4;
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid.x < 2 || grid.y < 2) {
        GTEST_SKIP() << "needs a 2x2 region";
    }
    for (uint32_t Bt : {2u, 4u}) {
        const auto ref = make_reference(2u * C * Bt * kTile, 64);
        const auto relay = run_relay(C, ref, 2, 2, /*endpoint_sync=*/false, nullptr, Bt);
        const auto endpoint = run_relay(C, ref, 2, 2, /*endpoint_sync=*/true, nullptr, Bt);
        const std::string at = " at Bt = " + std::to_string(Bt);
        expect_identical(relay.dQ, endpoint.dQ, "dQ, barrier against endpoint counters" + at);
        expect_identical(relay.dK, endpoint.dK, "dK, barrier against endpoint counters" + at);
        expect_identical(relay.dV, endpoint.dV, "dV, barrier against endpoint counters" + at);
    }
}

// The column gradients are where the two differ, and residency is the more
// accurate side. Worth pinning: it says the per-timestep reload was costing
// precision, not just bandwidth, and it would catch a regression that made
// residency the worse of the two.
TEST(CyclicSdpaBwIdentityTest, ResidencyIsTheMoreAccurateColumnPath) {
    const uint32_t C = 4;
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid.x < 2 || grid.y < 2) {
        GTEST_SKIP() << "needs a 2x2 region";
    }
    const auto ref = make_reference(2u * C * kTile, 64);
    const auto algorithm2 = run_algorithm2(C, ref, 2, 2);
    const auto relay = run_relay(C, ref, 2, 2, /*endpoint_sync=*/false);

    const float reload_dk = max_relative_error(algorithm2.dK, ref.dK);
    const float resident_dk = max_relative_error(relay.dK, ref.dK);
    const float reload_dv = max_relative_error(algorithm2.dV, ref.dV);
    const float resident_dv = max_relative_error(relay.dV, ref.dV);
    std::cout << "  dK relative error: reloaded " << reload_dk << ", resident " << resident_dk
              << "\n  dV relative error: reloaded " << reload_dv << ", resident " << resident_dv
              << "\n";
    EXPECT_LE(resident_dk, reload_dk);
    EXPECT_LE(resident_dv, reload_dv);

    // An absolute bound as well as a relative one. Dropping the compute
    // kernels from the default HiFi4 to HiFi2 -- on the argument that every
    // matmul operand is bfloat16, so there should be no mantissa bits for the
    // extra fidelity phases to resolve -- made dK's error 9.8e-3 against the
    // 1.05e-3 here, nine times worse, and bought 3 to 4% of runtime. Every
    // other test in this file passed with that change in place: the
    // gradient-accuracy tolerances are loose enough to hide it, and the
    // relative comparison above only asks that residency beat the reload,
    // which it still did. This is the assertion that catches it.
    EXPECT_LT(resident_dk, 2.5e-3F);
    EXPECT_LT(resident_dv, 2.5e-3F);
    EXPECT_LT(reload_dk, 2.5e-3F);
    EXPECT_LT(reload_dv, 2.5e-3F);
}

// A first indication of whether any of this is faster, not a verdict. The
// numbers are host dispatch-to-finish with no attribution: the plan's Phase 8
// wants profiler timestamps around the named operations, and the build has no
// Tracy because the clang-20 binutils are missing. Repeated enqueues also
// accumulate into the same gradient tensors, which is harmless for timing and
// meaningless for correctness, so nothing is checked here.
//
// Disabled by default: it is slow and it measures rather than asserts.
void time_one_size(
    uint32_t C, uint32_t grid_w, uint32_t grid_h, uint32_t d = 64, uint32_t Bt = 1) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid_w > grid.x || grid_h > grid.y) {
        GTEST_SKIP() << "needs " << grid_w << "x" << grid_h;
    }
    const auto ref = make_reference(2u * C * Bt * kTile, d);

    double dram_seconds = 0.0;
    double relay_seconds = 0.0;
    double endpoint_seconds = 0.0;
    run_algorithm2(C, ref, grid_w, grid_h, &dram_seconds, Bt);
    run_relay(C, ref, grid_w, grid_h, /*endpoint_sync=*/false, &relay_seconds, Bt);
    run_relay(C, ref, grid_w, grid_h, /*endpoint_sync=*/true, &endpoint_seconds, Bt);

    // What Algorithm 4 adds over the barrier: one endpoint write per
    // inter-streak spill, and one wait -- a handful of remote reads -- per
    // later streak start. Both grow linearly in C, which is why the deficit
    // stopped growing once publication stopped being C writes each.
    const CyclicSchedule sched(C);
    uint32_t inter_streak_spills = 0;
    uint32_t endpoint_waits = 0;
    for (uint32_t i = 1; i <= sched.T(); ++i) {
        for (uint32_t t = 0; t <= sched.T(); ++t) {
            if (sched.is_active(i, t) && sched.streak_at(i, t).end == t &&
                sched.has_later_active(i, t)) {
                ++inter_streak_spills;
            }
            if (sched.is_later_streak_start(i, t)) {
                ++endpoint_waits;
            }
        }
    }

    const double us = 1e6;
    std::cout << "  C=" << C << " N=" << 2u * C * Bt * kTile << " Bt=" << Bt << " d=" << d
              << " on " << grid_w << "x"
              << grid_h << ": DRAM " << dram_seconds * us << " us, relay " << relay_seconds * us
              << " us, endpoint " << endpoint_seconds * us << " us"
              << " | relay speedup " << dram_seconds / relay_seconds << "x"
              << ", endpoint against relay " << endpoint_seconds / relay_seconds << "x"
              << " | " << inter_streak_spills << " endpoint writes, " << endpoint_waits
              << " endpoint waits\n";
}

// Does the endpoint variant's deficit against the barrier variant scale with
// C? It should if the publication traffic is what causes it: a publication is
// C unicast writes, and the number of publications grows with C too, so the
// cost is quadratic in C while the work per core is not.
TEST(CyclicSdpaBwTimingTest, DISABLED_CompareTheThreeVariants) {
    time_one_size(4, 2, 2);
    time_one_size(8, 4, 2);
    time_one_size(16, 4, 4);
    time_one_size(32, 8, 4);
    time_one_size(64, 8, 8);
}

// Does more work per timestep change the picture? The head dimension is the
// only knob that adds arithmetic here -- a block is one sequence tile, so
// every intermediate stays one tile while Q, K, V, dO, dQ, dK, dV grow as
// d/32 tiles. Compute and payload bytes both scale linearly in d, so what
// this separates is the *fixed* per-timestep cost -- semaphore waits,
// barrier, kernel overhead -- from the part that scales. If the relay's
// advantage and the endpoint deficit both shrink as d grows, the fixed costs
// are being amortized and the port is moving toward compute bound.
// One profiled run, then an explicit device close. The device profiler
// writes its CSV from ProfilerInitializer::post_teardown, which only runs on
// a real device close -- the process-exit path deliberately skips it, because
// the dump spawns threads and that is unsafe during termination. So a
// profiling run has to close the device itself.
//
//   TT_METAL_DEVICE_PROFILER=1 ttml_tests \
//     --gtest_filter=CyclicSdpaBwProfileTest.* --gtest_also_run_disabled_tests
//
// then generated/profiler/.logs/profile_log_device.csv holds the zones.
TEST(CyclicSdpaBwProfileTest, DISABLED_ProfileTheRelay) {
    const uint32_t C = 16;
    const uint32_t Bt = 4;
    const auto ref = make_reference(2u * C * Bt * kTile, 64);
    run_relay(C, ref, 4, 4, /*endpoint_sync=*/false, nullptr, Bt);
    ttml::autograd::ctx().close_device();
}

// The same total sequence length, cut into blocks two ways: C cores with
// blocks one tile tall, and half as many cores with blocks two tiles tall.
// Same arithmetic either way -- 2C^2 tile pairs -- so this is purely what
// block shape does to how fast a core gets through it.
// Does the relay still earn its place once the compute is efficient? Tall
// blocks cut the compute per unit of arithmetic by 1.9x without touching the
// dataflow, so the dataflow's share of the total goes up and this is where
// the three algorithms should be compared.
// Against the repository's own backward. ttml::metal::sdpa_bw splits causal
// work as NC * St/2 pairs -- batch times heads times sequence tiles over two,
// pairing an early row with a late one to balance the triangle -- so at
// batch 1, head 1 and S = 4096 it has 64 pairs and fills the same 8x8 grid
// this port uses at C = 64. Same shape, same cores, same arithmetic: a fair
// comparison, and the one that says whether any of this was worth doing.
//
// The caveat is that sdpa_bw computes the forward statistics itself from an
// attention output, while this port is handed L and D. Both are timed over
// the backward call alone.
void compare_with_sdpa_bw(
    uint32_t C,
    uint32_t grid_w,
    uint32_t grid_h,
    uint32_t Bt,
    uint32_t groups = 1,
    uint32_t d = 64) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    if (grid_w > grid.x || grid_h > grid.y) {
        GTEST_SKIP() << "needs " << grid_w << "x" << grid_h;
    }
    const uint32_t N = 2u * C * Bt * kTile;
    // Inputs only: nothing here is checked, and the dense host backward at
    // these sizes costs more than the measurement.
    const auto ref = make_reference_inputs_only(N, d);
    double relay_seconds = 0.0;
    run_relay(C, ref, grid_w, grid_h, /*endpoint_sync=*/true, &relay_seconds, Bt, groups);

    // The repository's, on the same tensors. With `groups` slices this is
    // NC = groups for sdpa_bw too, which is where it has its own parallelism:
    // it splits NC * St/2 pairs across the grid, so a short sequence with
    // several heads fills it as readily as one long sequence does.
    const auto query = ttml::core::from_xtensor(as_4d_repeated(ref.Q, groups), device);
    const auto key = ttml::core::from_xtensor(as_4d_repeated(ref.K, groups), device);
    const auto value = ttml::core::from_xtensor(as_4d_repeated(ref.V, groups), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d_repeated(ref.dO, groups), device);
    const auto forward = ttml::metal::sdpa_fw(
        query, key, value, ttml::metal::AttentionMaskType::Causal, std::nullopt, 0.0F,
        /*return_intermediates=*/true);
    const auto attn_output = forward[0].value();
    const auto intermediates = forward[1].value();

    const auto call = [&]() {
        const auto out = ttml::metal::sdpa_bw(
            grad_output, attn_output, query, key, value, intermediates,
            ttml::metal::AttentionMaskType::Causal, std::nullopt, 0.0F);
        distributed::Finish(device->mesh_command_queue());
    };
    call();  // warm
    std::vector<double> samples;
    for (uint32_t k = 0; k < 5; ++k) {
        const auto start = std::chrono::steady_clock::now();
        call();
        samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    const double sdpa_seconds = samples[samples.size() / 2];

    std::cout << "  N=" << N << " d=" << d << " NC=" << groups << " (" << groups
              << " group(s) of " << C << " cores, Bt=" << Bt << "): sdpa_bw "
              << sdpa_seconds * 1e6 << " us, cyclic " << relay_seconds * 1e6
              << " us | cyclic " << sdpa_seconds / relay_seconds << "x\n";
}

// What running several heads side by side is worth. One 16-core schedule
// leaves 48 cores idle on an 8x8 grid; four of them fill it. If four heads
// take about as long as one, the grid is being used four times over.
TEST(CyclicSdpaBwTimingTest, DISABLED_ScaleTheGroups) {
    for (uint32_t groups : {1u, 2u, 4u}) {
        const uint32_t C = 16;
        const uint32_t Bt = 1;
        const auto ref = make_reference(2u * C * Bt * kTile, 64);
        double seconds = 0.0;
        run_relay(C, ref, 4, 4, /*endpoint_sync=*/true, &seconds, Bt, groups);
        std::cout << "  " << groups << " group(s) of " << C << " cores, N=" << ref.N
                  << " each: " << seconds * 1e6 << " us for " << groups << " head(s)"
                  << ", " << seconds * 1e6 / groups << " us per head\n";
    }
}

// Against the repository's own backward, on a full chip every time.
//
// All 64 cores are busy in every row; what varies is how the chip is cut up.
// One schedule needs N = 2 C Bt 32 rows, so a shorter sequence means a
// smaller schedule, and the chip is filled by running several of them side by
// side -- which is also the shape sdpa_bw fills a grid with, since it splits
// NC * St/2 pairs across the cores. So these are matched on cores, on shape,
// and on total arithmetic.
// Does the relay's traffic advantage ever show in wall clock? Both
// implementations are compute bound at d = 64 -- theirs by a prefetching
// reader, this one by the relay -- so the traffic this port saves should buy
// nothing there. The head dimension is the lever: it multiplies the bytes a
// row operand costs without changing how many score tiles there are, so at
// d = 256 a packet is four times heavier while the score stages are
// identical. If the ratio climbs past what fusion and block height explain,
// the saved traffic has started to pay.
TEST(CyclicSdpaBwTimingTest, DISABLED_CompareAcrossHeadDimensions) {
    for (uint32_t d : {64u, 128u, 256u}) {
        compare_with_sdpa_bw(110, 11, 10, /* Bt */ 1, /* groups */ 1, d);
    }
    for (uint32_t d : {64u, 128u, 256u}) {
        compare_with_sdpa_bw(55, 11, 5, /* Bt */ 2, /* groups */ 2, d);
    }
}

TEST(CyclicSdpaBwTimingTest, DISABLED_CompareWithTheRepositorysBackward) {
    struct Shape {
        uint32_t C, w, h, Bt, groups;
    };
    // The compute grid is 11x10, so a full chip is 110 cores, and these are
    // the ways to tile it exactly: one schedule of 110, or 2, 5 and 10 groups
    // whose rectangles are 11 wide. Everything here keeps all 110 busy.
    for (const auto sh : {
             Shape{11, 11, 1, 1, 10},   // NC=10, N=704
             Shape{22, 11, 2, 1, 5},    // NC=5,  N=1408
             Shape{55, 11, 5, 1, 2},    // NC=2,  N=3520
             Shape{110, 11, 10, 1, 1},  // NC=1,  N=7040
             Shape{22, 11, 2, 2, 5},    // NC=5,  N=2816
             Shape{55, 11, 5, 2, 2},    // NC=2,  N=7040
             Shape{110, 11, 10, 2, 1},  // NC=1,  N=14080
             Shape{55, 11, 5, 4, 2},    // NC=2,  N=14080
             Shape{110, 11, 10, 4, 1},  // NC=1,  N=28160
         }) {
        compare_with_sdpa_bw(sh.C, sh.w, sh.h, sh.Bt, sh.groups);
    }
}

TEST(CyclicSdpaBwTimingTest, DISABLED_CompareTheThreeVariantsWithTallBlocks) {
    time_one_size(16, 4, 4, 64, /* Bt */ 4);
    time_one_size(32, 8, 4, 64, /* Bt */ 4);
    time_one_size(64, 8, 8, 64, /* Bt */ 2);
    time_one_size(64, 8, 8, 64, /* Bt */ 4);
}

TEST(CyclicSdpaBwTimingTest, DISABLED_CompareBlockShapes) {
    struct Shape {
        uint32_t C, w, h, Bt;
    };
    for (const auto s : {Shape{16, 4, 4, 1}, Shape{8, 4, 2, 2},
                         Shape{32, 8, 4, 1}, Shape{16, 4, 4, 2},
                         Shape{64, 8, 8, 1}, Shape{32, 8, 4, 2},
                         Shape{8, 4, 2, 4}, Shape{16, 4, 4, 4},
                         Shape{64, 8, 8, 2}, Shape{64, 8, 8, 4}}) {
        const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
        if (s.w > grid.x || s.h > grid.y) {
            continue;
        }
        const uint32_t N = 2u * s.C * s.Bt * kTile;
        const auto ref = make_reference(N, 64);
        double seconds = 0.0;
        run_relay(s.C, ref, s.w, s.h, /* endpoint_sync */ false, &seconds, s.Bt);
        std::cout << "  N=" << N << " on " << s.C << " cores, Bt=" << s.Bt << ": "
                  << seconds * 1e6 << " us\n";
    }
}

TEST(CyclicSdpaBwTimingTest, DISABLED_ScaleTheHeadDimension) {
    for (uint32_t d : {32u, 64u, 128u, 256u}) {
        time_one_size(16, 4, 4, d);
    }
    for (uint32_t d : {32u, 64u, 128u, 256u}) {
        time_one_size(64, 8, 8, d);
    }
}
