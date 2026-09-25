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
#include <bit>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "metal/operations.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"
#include "ops/distributed/ring_attention_sdpa.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"

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

// The all-positive regime the ring tests draw from: uniform(0, 2) rather than
// uniform(-0.5, 0.5). With d = 64 that makes the scores large and positive, so
// the softmax is nearly one-hot -- a much harder numerical case than the
// zero-mean data the rest of this file uses, and the one where a single
// dominant attention weight decides the answer.
float sample_positive(uint32_t seed) {
    const uint32_t h = seed * 1664525u + 1013904223u;
    return 2.0F * static_cast<float>(h % 2001u) / 2000.0F;
}

xt::xarray<float> random_bf16_matrix_positive(uint32_t rows, uint32_t cols, uint32_t salt) {
    xt::xarray<float> out = xt::zeros<float>({rows, cols});
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            out(r, c) = to_bf16(sample_positive(salt * 7919u + r * 131u + c));
        }
    }
    return out;
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

Reference make_reference(uint32_t N, uint32_t d, bool positive = false) {
    Reference r;
    r.N = N;
    r.d = d;
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));
    const auto draw = [&](uint32_t salt) {
        return positive ? random_bf16_matrix_positive(N, d, salt) : random_bf16_matrix(N, d, salt);
    };
    r.Q = draw(1);
    r.K = draw(2);
    r.V = draw(3);
    r.dO = draw(4);

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

// The same reference with nothing masked: every query row attends to every
// key. This is what a ring-attention step computes when the visiting
// key/value chunk is earlier in the sequence than the local query chunk, and
// it is what MaskMode::Dense has to reproduce.
//
// Q and K are drawn independently here, as they are in the ring: the two
// chunks are different slices of the sequence, so there is no reason for the
// block matrix to be square-symmetric in any way the schedule could exploit.
Reference make_dense_reference(uint32_t N, uint32_t d) {
    Reference r;
    r.N = N;
    r.d = d;
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));
    r.Q = random_bf16_matrix(N, d, 11);
    r.K = random_bf16_matrix(N, d, 12);
    r.V = random_bf16_matrix(N, d, 13);
    r.dO = random_bf16_matrix(N, d, 14);

    const xt::xarray<float> S = xt::linalg::dot(r.Q, xt::transpose(r.K)) * scale;
    xt::xarray<float> P = xt::zeros<float>({N, N});
    std::vector<float> lse(N, 0.0F);
    for (uint32_t i = 0; i < N; ++i) {
        float m = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j < N; ++j) {
            m = std::max(m, S(i, j));
        }
        float sum = 0.0F;
        for (uint32_t j = 0; j < N; ++j) {
            sum += std::exp(S(i, j) - m);
        }
        lse[i] = m + std::log(sum);
        for (uint32_t j = 0; j < N; ++j) {
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
        for (uint32_t j = 0; j < N; ++j) {
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
    make_cb(tt::CBIndex::c_6, 2, tt::DataFormat::Float16_b);     // causal mask: triangle, all -inf
    // Intermediates.
    make_cb(tt::CBIndex::c_10, scoreT, tt::DataFormat::Float32);  // P^T
    make_cb(tt::CBIndex::c_8, 1, tt::DataFormat::Float16_b);      // transpose fence
    make_cb(tt::CBIndex::c_9, 2U * Bt, tt::DataFormat::Float16_b);   // -L remainder, column 0
    make_cb(tt::CBIndex::c_29, 2U * Bt, tt::DataFormat::Float16_b);  // -D remainder, column 0
    make_cb(tt::CBIndex::c_28, 1, tt::DataFormat::Float16_b);        // ones column
    CreateCircularBuffer(
        program, region,
        CircularBufferConfig(2 * 64, {{tt::CBIndex::c_31, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_31, 64));  // stats-ready signal
    make_cb(tt::CBIndex::c_12, scoreT, tt::DataFormat::Float32);  // dS^T
    make_cb(tt::CBIndex::c_13, 2U * Bt, tt::DataFormat::Float32);      // L_i, row layout
    make_cb(tt::CBIndex::c_14, 2U * Bt, tt::DataFormat::Float32);      // D_i, row layout
    // Each gradient: seed from the reader, accumulator, output to the writer.
    // dQ seed in, dQ out: two views of one slot (see the compute kernel's
    // UPDATE-DQ). The release protocol keeps the writer's read of one
    // timestep ahead of the reader's load of the next.
    CreateCircularBuffer(
        program,
        region,
        CircularBufferConfig(
            rowT * fp32_tile, {{tt::CBIndex::c_15, tt::DataFormat::Float32}, {tt::CBIndex::c_17, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_15, fp32_tile)
            .set_page_size(tt::CBIndex::c_17, fp32_tile));
    make_cb(tt::CBIndex::c_16, rowT, tt::DataFormat::Float16_b);       // K_j^T (scaled where exact)
    // dK and dV: the seed from DRAM and the output share one slot each (two
    // views), the accumulator sits between them; see the compute kernel.
    CreateCircularBuffer(
        program,
        region,
        CircularBufferConfig(
            rowT * fp32_tile, {{tt::CBIndex::c_18, tt::DataFormat::Float32}, {tt::CBIndex::c_20, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_18, fp32_tile)
            .set_page_size(tt::CBIndex::c_20, fp32_tile));
    make_cb(tt::CBIndex::c_19, rowT, tt::DataFormat::Float32);
    CreateCircularBuffer(
        program,
        region,
        CircularBufferConfig(
            valT * fp32_tile, {{tt::CBIndex::c_21, tt::DataFormat::Float32}, {tt::CBIndex::c_23, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_21, fp32_tile)
            .set_page_size(tt::CBIndex::c_23, fp32_tile));
    make_cb(tt::CBIndex::c_22, valT, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_24, 1, tt::DataFormat::Float32);  // control word

    const uint32_t arrive_sem = CreateSemaphore(program, region, 0);
    const uint32_t release_sem = CreateSemaphore(program, region, 0);

    // The softmax scale lives in the exponential and in the writer's score
    // seed, as in the op (see the compute kernel).
    std::map<std::string, std::string> fold_defines;
    {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "0x%08Xu", std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(ref.d))));
        fold_defines["L_SEED_SCALE_BITS"] = buf;
    }
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
            .compile_args = writer_args,
            .defines = fold_defines});

    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(ref.d)));
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    // sqrt(d): the kernel folds the softmax scale into the exponential, so it
    // needs the reciprocal to divide the statistic it subtracts.
    const uint32_t inv_scaler = std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(ref.d)));
    // The exponential's bias constant, 127 - log2(sqrt d): ln sqrt(d) folded
    // in, so it computes exp(a x) / sqrt(d) (see the compute kernel).
    const uint32_t exp_bias = std::bit_cast<uint32_t>(127.0F - 0.5F * std::log2(static_cast<float>(ref.d)));
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());
    // The dQ seed -- the packet's running accumulator, and its transposed
    // scratch copy -- is unpacked straight into DST, so it keeps all 32 bits
    // at every hop; through a Src register it would lose 13 of them per
    // timestep, which was the dominant dQ error. Nothing else may take this
    // mode: a buffer in it cannot also be read by a matmul, and these two are
    // read only by copies.
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_mode[tt::CBIndex::c_15] = UnpackToDestMode::UnpackToDestFp32;
    // The column gradients' seeds and accumulators too: read only by the
    // reload and handover copies, so the running sums keep all 32 bits
    // across a handover and a reload rather than the register's 19.
    unpack_mode[tt::CBIndex::c_19] = UnpackToDestMode::UnpackToDestFp32;  // EXACT-ACCUM
    unpack_mode[tt::CBIndex::c_22] = UnpackToDestMode::UnpackToDestFp32;  // EXACT-ACCUM
    const auto compute = CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            // Half-sync: the pass works in groups of at most two key tiles,
            // four Float32 registers, so the math and pack threads alternate
            // halves of the file and overlap. Nothing needs more than four.
            .dst_full_sync_en = false,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {C, qWt, vWt, scaler, minus_one, custom_inf, block_size, Bt, inv_scaler, exp_bias},
            .defines = fold_defines});

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
        // The compute kernel is shared with the relay and takes the number of
        // slices this core runs; one, here.
        SetRuntimeArgs(program, compute, core, {c, 1u});
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
    CreateCircularBuffer(
        program, region,
        CircularBufferConfig(2 * Bt * 512, {{tt::CBIndex::c_4, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_4, 512));  // statistic block
    make_cb(tt::CBIndex::c_5, 4 * Bt, tt::DataFormat::Float32);      // L_i, D_i scratch
    // dQ_i, travels along: two slots, two views (see the compute kernel).
    CreateCircularBuffer(
        program,
        region,
        CircularBufferConfig(
            2 * rowT * fp32_tile,
            {{tt::CBIndex::c_15, tt::DataFormat::Float32}, {tt::CBIndex::c_17, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_15, fp32_tile)
            .set_page_size(tt::CBIndex::c_17, fp32_tile));
    make_cb(tt::CBIndex::c_1, rowT, tt::DataFormat::Float16_b);      // K_j
    make_cb(tt::CBIndex::c_2, valT, tt::DataFormat::Float16_b);      // V_j
    make_cb(tt::CBIndex::c_6, 2, tt::DataFormat::Float16_b);         // causal mask: triangle, all -inf
    make_cb(tt::CBIndex::c_10, scoreT, tt::DataFormat::Float32);  // P^T
    make_cb(tt::CBIndex::c_8, 1, tt::DataFormat::Float16_b);      // transpose fence
    make_cb(tt::CBIndex::c_9, 2U * Bt, tt::DataFormat::Float16_b);   // -L remainder, column 0
    make_cb(tt::CBIndex::c_29, 2U * Bt, tt::DataFormat::Float16_b);  // -D remainder, column 0
    make_cb(tt::CBIndex::c_28, 1, tt::DataFormat::Float16_b);        // ones column
    CreateCircularBuffer(
        program, region,
        CircularBufferConfig(2 * 64, {{tt::CBIndex::c_31, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_31, 64));  // stats-ready signal
    make_cb(tt::CBIndex::c_12, scoreT, tt::DataFormat::Float32);  // dS^T
    make_cb(tt::CBIndex::c_13, 2U * Bt, tt::DataFormat::Float32);      // L_i, row layout
    make_cb(tt::CBIndex::c_14, 2U * Bt, tt::DataFormat::Float32);      // D_i, row layout
    make_cb(tt::CBIndex::c_16, rowT, tt::DataFormat::Float16_b);       // K_j^T (scaled where exact)
    // dK and dV: the seed from DRAM and the output share one slot each (two
    // views), the accumulator sits between them; see the compute kernel.
    CreateCircularBuffer(
        program,
        region,
        CircularBufferConfig(
            rowT * fp32_tile, {{tt::CBIndex::c_18, tt::DataFormat::Float32}, {tt::CBIndex::c_20, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_18, fp32_tile)
            .set_page_size(tt::CBIndex::c_20, fp32_tile));
    make_cb(tt::CBIndex::c_19, rowT, tt::DataFormat::Float32);
    CreateCircularBuffer(
        program,
        region,
        CircularBufferConfig(
            valT * fp32_tile, {{tt::CBIndex::c_21, tt::DataFormat::Float32}, {tt::CBIndex::c_23, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_21, fp32_tile)
            .set_page_size(tt::CBIndex::c_23, fp32_tile));
    make_cb(tt::CBIndex::c_22, valT, tt::DataFormat::Float32);
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
    // The softmax scale lives in the exponential and in the writer's score
    // seed (see the compute kernel).
    {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "0x%08Xu", std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(ref.d))));
        sync_defines["L_SEED_SCALE_BITS"] = buf;
    }

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
    // After the accessor args, where the kernel finds them by offset: the
    // packet-readiness semaphores, which the writer polls to start on a
    // timestep's statistics as soon as they land.
    writer_args.push_back(ready_imm0_sem);
    writer_args.push_back(ready_imm1_sem);
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
    // The exponential's bias constant, 127 - log2(sqrt d): ln sqrt(d) folded
    // in, so it computes exp(a x) / sqrt(d) (see the compute kernel).
    const uint32_t exp_bias = std::bit_cast<uint32_t>(127.0F - 0.5F * std::log2(static_cast<float>(ref.d)));
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());
    // The dQ seed -- the packet's running accumulator, and its transposed
    // scratch copy -- is unpacked straight into DST, so it keeps all 32 bits
    // at every hop; through a Src register it would lose 13 of them per
    // timestep, which was the dominant dQ error. Nothing else may take this
    // mode: a buffer in it cannot also be read by a matmul, and these two are
    // read only by copies.
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_mode[tt::CBIndex::c_15] = UnpackToDestMode::UnpackToDestFp32;
    // The column gradients' seeds and accumulators too: read only by the
    // reload and handover copies, so the running sums keep all 32 bits
    // across a handover and a reload rather than the register's 19.
    unpack_mode[tt::CBIndex::c_19] = UnpackToDestMode::UnpackToDestFp32;  // EXACT-ACCUM
    unpack_mode[tt::CBIndex::c_22] = UnpackToDestMode::UnpackToDestFp32;  // EXACT-ACCUM
    const auto compute = CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            // Half-sync: the pass works in groups of at most two key tiles,
            // four Float32 registers, so the math and pack threads alternate
            // halves of the file and overlap. Nothing needs more than four.
            .dst_full_sync_en = false,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {C, qWt, vWt, scaler, minus_one, custom_inf, block_size, Bt, inv_scaler, exp_bias},
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
            // One slice per group here; the kernels loop over slices and read
            // the count and stride after their other arguments.
            relay_reader_args.push_back(1u);  // slice_count
            relay_reader_args.push_back(1u);  // slice_stride
            // One chunk, one pair (0, 0): the whole sequence against itself;
            // as many heads as groups, one slice each, one query head per key
            // head (kv_slices = q_heads = kv_heads = groups, heads_per_group 1).
            for (const uint32_t v : {1u, 1u, groups, groups, groups, groups, 1u, 0u, 0u}) {
                relay_reader_args.push_back(v);
            }
            SetRuntimeArgs(program, reader, core, relay_reader_args);
            SetRuntimeArgs(
                program, writer, core,
                {c, g, grad_key.buffer()->address(), grad_value.buffer()->address(),
                 static_cast<uint32_t>(coordinator.x), static_cast<uint32_t>(coordinator.y),
                 static_cast<uint32_t>(mcast_start.x), static_cast<uint32_t>(mcast_start.y),
                 static_cast<uint32_t>(mcast_end.x), static_cast<uint32_t>(mcast_end.y),
                 c == 1u ? 1u : 0u, 1u, 1u, /* chunks */ 1u, /* pairs */ 1u, /* heads */ groups,
                 /* kv_slices, q_heads, kv_heads, heads_per_group */ groups, groups, groups, 1u, 0u, 0u});
            SetRuntimeArgs(program, compute, core, {c, 1u});
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
    if (std::getenv("CYCLIC_DUMP_BLOCKS") != nullptr) {
        // Per 32-row block: the largest |got| and |ref| of each gradient.
        for (const auto& [name, g, r] : {std::tuple{"dQ", &got.dQ, &ref.dQ}, std::tuple{"dK", &got.dK, &ref.dK},
                                          std::tuple{"dV", &got.dV, &ref.dV}}) {
            std::cout << "  " << name << " blocks:";
            for (uint32_t b = 0; b < N / kTile; ++b) {
                float mg = 0.0F, mr = 0.0F, md = 0.0F;
                for (uint32_t rr = b * kTile; rr < (b + 1u) * kTile; ++rr) {
                    for (uint32_t c = 0; c < d; ++c) {
                        mg = std::max(mg, std::abs((*g)(rr, c)));
                        mr = std::max(mr, std::abs((*r)(rr, c)));
                        md = std::max(md, std::abs((*g)(rr, c) - (*r)(rr, c)));
                    }
                }
                std::cout << " [" << b + 1u << ": got " << mg << " ref " << mr << " diff " << md << "]";
            }
            std::cout << "\n";
        }
    }
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
//
// This was the test that first showed dV coming out near zero, varying from
// run to run, once the dQ update got faster. The cause was found later, when
// the two-core case failed the same way every run: a dest-register transpose
// (transpose_dest) in the dQ update was overlapping the unpack of the next
// matmul's SrcB operand, and its end-of-transpose clear wiped that operand.
// The dQ update now keeps its transposes behind buffer handshakes; see the
// compute kernel.
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
void check_op(uint32_t C, uint32_t Bt, uint32_t slices, bool use_barrier, uint32_t d = 64, uint32_t max_groups = 0) {
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

    const auto [grad_query, grad_key, grad_value] = ttml::metal::cyclic_sdpa_bw(
        query, key, value, grad_output, lse, row_scalar, Bt, use_barrier, ttml::metal::AttentionMaskType::Causal,
        /* accumulate */ false, std::nullopt, std::nullopt, std::nullopt, max_groups);

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

// The unmasked schedule, through the same op. C follows from N and Bt as
// usual; what changes is that the schedule covers every block pair rather
// than the causal triangle, in 2T timesteps rather than T + 1.
void check_dense_op(uint32_t C, uint32_t Bt, uint32_t slices, bool use_barrier, uint32_t d = 64, uint32_t max_groups = 0) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t N = 2u * C * Bt * kTile;
    const auto ref = make_dense_reference(N, d);

    const auto query = ttml::core::from_xtensor(as_4d_repeated(ref.Q, slices), device);
    const auto key = ttml::core::from_xtensor(as_4d_repeated(ref.K, slices), device);
    const auto value = ttml::core::from_xtensor(as_4d_repeated(ref.V, slices), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d_repeated(ref.dO, slices), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(ref.lse_tile, slices), device);
    const auto row_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(ref.u_tile, slices), device);

    const auto [grad_query, grad_key, grad_value] = ttml::metal::cyclic_sdpa_bw(
        query, key, value, grad_output, lse, row_scalar, Bt, use_barrier, ttml::metal::AttentionMaskType::None,
        /* accumulate */ false, std::nullopt, std::nullopt, std::nullopt, max_groups);

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

// The degenerate core count. A ring step whose chunk is short enough gives
// C = n / (2 Bt 32) = 1: a single core owning both columns, so every packet
// transition is a self-transition and nothing is ever forwarded. It is the
// case a ring reaches first, and the one the multi-core tests never touch.
// Relative error of each gradient, printed rather than asserted. The op
// tests grade at 6% of the largest reference value, which is loose enough to
// hide a several-fold degradation; this says what the error actually is, so a
// regression in one configuration can be seen against its neighbours.
void report_op_error(
    uint32_t C, uint32_t Bt, bool dense, uint32_t d = 64, uint32_t slices = 1, bool positive = false) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t N = 2u * C * Bt * kTile;
    const auto ref = dense ? make_dense_reference(N, d) : make_reference(N, d, positive);

    const auto query = ttml::core::from_xtensor(as_4d_repeated(ref.Q, slices), device);
    const auto key = ttml::core::from_xtensor(as_4d_repeated(ref.K, slices), device);
    const auto value = ttml::core::from_xtensor(as_4d_repeated(ref.V, slices), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d_repeated(ref.dO, slices), device);
    // CYCLIC_TRUNC_STATS=1 truncates L and D to 19 bits (what a Src register
    // keeps of a Float32) before upload: a kernel that only ever sees them
    // through a Src register gives identical results with it set.
    auto lse_tile = ref.lse_tile;
    auto u_tile = ref.u_tile;
    if (std::getenv("CYCLIC_TRUNC_STATS") != nullptr) {
        for (auto* t : {&lse_tile, &u_tile}) {
            for (auto& v : *t) {
                uint32_t bits = std::bit_cast<uint32_t>(v) & 0xFFFFE000u;
                v = std::bit_cast<float>(bits);
            }
        }
    }
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(lse_tile, slices), device);
    const auto row_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        repeat_4d(u_tile, slices), device);

    const auto [gq, gk, gv] = ttml::metal::cyclic_sdpa_bw(
        query, key, value, grad_output, lse, row_scalar, Bt, /* use_barrier */ false,
        dense ? ttml::metal::AttentionMaskType::None : ttml::metal::AttentionMaskType::Causal);

    const auto relative = [&](const xt::xarray<float>& got, const xt::xarray<float>& want) {
        float worst = 0.0F;
        for (uint32_t g = 0; g < slices; ++g) {
            float max_abs = 0.0F;
            float max_diff = 0.0F;
            for (uint32_t r = 0; r < want.shape()[0]; ++r) {
                for (uint32_t c = 0; c < want.shape()[1]; ++c) {
                    max_abs = std::max(max_abs, std::abs(want(r, c)));
                    max_diff = std::max(max_diff, std::abs(got(0, g, r, c) - want(r, c)));
                }
            }
            worst = std::max(worst, max_abs > 0.0F ? max_diff / max_abs : max_diff);
        }
        return worst;
    };
    // RMS of the error over RMS of the reference, slice 0: the max above is
    // one element's story, this is everyone's.
    const auto rms_relative = [&](const xt::xarray<float>& got, const xt::xarray<float>& want) {
        double num = 0.0, den = 0.0;
        for (uint32_t r = 0; r < want.shape()[0]; ++r) {
            for (uint32_t c = 0; c < want.shape()[1]; ++c) {
                const double e = got(0, 0, r, c) - want(r, c);
                num += e * e;
                den += double(want(r, c)) * want(r, c);
            }
        }
        return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    };
    // The least-squares scale of the error on the reference, slice 0: with
    // got = (1 + b) want + noise, this is b -- a systematic inflation (b > 0)
    // or shrinkage (b < 0) of the whole gradient, which the RMS hides.
    const auto scale_bias = [&](const xt::xarray<float>& got, const xt::xarray<float>& want) {
        double num = 0.0, den = 0.0;
        for (uint32_t r = 0; r < want.shape()[0]; ++r) {
            for (uint32_t c = 0; c < want.shape()[1]; ++c) {
                const double e = got(0, 0, r, c) - want(r, c);
                num += e * want(r, c);
                den += double(want(r, c)) * want(r, c);
            }
        }
        return den > 0.0 ? num / den : 0.0;
    };
    const auto gq_h = ttml::core::to_xtensor(gq);
    const auto gk_h = ttml::core::to_xtensor(gk);
    const auto gv_h = ttml::core::to_xtensor(gv);
    std::cout << "  " << (dense ? "dense " : "causal") << (positive ? " uniform(0,2)" : " zero-mean ")
              << " C=" << C << " Bt=" << Bt << " d=" << d
              << " N=" << N << " slices=" << slices << ": dQ " << relative(gq_h, ref.dQ)
              << " dK " << relative(gk_h, ref.dK) << " dV " << relative(gv_h, ref.dV)
              << "   rms: dQ " << rms_relative(gq_h, ref.dQ) << " dK " << rms_relative(gk_h, ref.dK) << " dV "
              << rms_relative(gv_h, ref.dV) << "   bias: dQ " << scale_bias(gq_h, ref.dQ) << " dK "
              << scale_bias(gk_h, ref.dK) << " dV " << scale_bias(gv_h, ref.dV) << "\n";
    if (std::getenv("CYCLIC_DUMP_BLOCKS") != nullptr) {
        for (const auto& [name, g, r] : {std::tuple{"dQ", &gq_h, &ref.dQ}, std::tuple{"dK", &gk_h, &ref.dK},
                                          std::tuple{"dV", &gv_h, &ref.dV}}) {
            std::cout << "    " << name << " per block, max |diff|:";
            for (uint32_t b = 0; b < N / kTile; ++b) {
                float md = 0.0F;
                for (uint32_t rr = b * kTile; rr < (b + 1u) * kTile; ++rr) {
                    for (uint32_t c = 0; c < d; ++c) {
                        md = std::max(md, std::abs((*g)(0, 0, rr, c) - (*r)(rr, c)));
                    }
                }
                std::cout << " " << md;
            }
            std::cout << "\n";
        }
    }
}

TEST(CyclicSdpaBwOpTest, DISABLED_ReportErrorAcrossBlockHeights) {
    for (const bool dense : {false, true}) {
        for (const uint32_t C : {1u, 2u, 4u}) {
            for (const uint32_t Bt : {1u, 2u, 4u}) {
                report_op_error(C, Bt, dense);
            }
        }
    }
    // The wider head, where the scale cannot fold into K and goes through
    // the exponential instead.
    for (const bool dense : {false, true}) {
        report_op_error(/* C */ 2, /* Bt */ 2, dense, /* d */ 128);
        report_op_error(/* C */ 2, /* Bt */ 4, dense, /* d */ 128);
    }
    report_op_error(/* C */ 2, /* Bt */ 4, /* dense */ false, /* d */ 128, /* slices */ 1, /* positive */ true);
    // The all-positive regime, where the softmax is nearly one-hot. This is
    // what the ring tests draw from, and the block height is the variable
    // under suspicion there.
    for (const uint32_t C : {1u, 2u}) {
        for (const uint32_t Bt : {1u, 2u, 4u}) {
            report_op_error(C, Bt, /* dense */ false, /* d */ 64, /* slices */ 1, /* positive */ true);
        }
    }

    // The ring's own shape: several (batch, head) slices, each on its own
    // one-core group.
    for (const bool dense : {false, true}) {
        for (const uint32_t slices : {1u, 2u, 4u}) {
            report_op_error(/* C */ 1, /* Bt */ 2, dense, /* d */ 64, slices);
            report_op_error(/* C */ 2, /* Bt */ 1, dense, /* d */ 64, slices);
        }
    }
}

TEST(CyclicSdpaBwDenseOpTest, SingleCore) {
    check_dense_op(/* C */ 1, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwDenseOpTest, SingleCoreTallBlocks) {
    check_dense_op(/* C */ 1, /* Bt */ 2, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwOpTest, SingleCoreTallBlocks) {
    check_op(/* C */ 1, /* Bt */ 2, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwOpTest, SingleCore) {
    check_op(/* C */ 1, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwDenseOpTest, OneSlice) {
    check_dense_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwDenseOpTest, MoreCores) {
    check_dense_op(/* C */ 8, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwDenseOpTest, TallBlocks) {
    check_dense_op(/* C */ 4, /* Bt */ 2, /* slices */ 1, /* use_barrier */ false);
}

TEST(CyclicSdpaBwDenseOpTest, WiderHead) {
    check_dense_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false, /* d */ 128);
}

TEST(CyclicSdpaBwDenseOpTest, FourSlices) {
    check_dense_op(/* C */ 4, /* Bt */ 1, /* slices */ 4, /* use_barrier */ false);
}

// The barrier variant of the dense schedule. It has to agree with the
// endpoint variant exactly, which is the property that says removing the
// chip-wide barrier changed nothing -- the same invariant the causal
// schedule is held to.
TEST(CyclicSdpaBwDenseOpTest, WithTheBarrier) {
    check_dense_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ true);
}

TEST(CyclicSdpaBwDenseOpTest, BarrierAndEndpointAgreeBitwise) {
    auto* device = &ttml::autograd::ctx().get_device();
    constexpr uint32_t C = 4u;
    constexpr uint32_t Bt = 1u;
    constexpr uint32_t d = 64u;
    const uint32_t N = 2u * C * Bt * kTile;
    const auto ref = make_dense_reference(N, d);

    const auto query = ttml::core::from_xtensor(as_4d(ref.Q), device);
    const auto key = ttml::core::from_xtensor(as_4d(ref.K), device);
    const auto value = ttml::core::from_xtensor(as_4d(ref.V), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d(ref.dO), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(ref.lse_tile, device);
    const auto row_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(ref.u_tile, device);

    const auto run = [&](bool use_barrier) {
        const auto [dq, dk, dv] = ttml::metal::cyclic_sdpa_bw(
            query, key, value, grad_output, lse, row_scalar, Bt, use_barrier,
            ttml::metal::AttentionMaskType::None);
        return std::array<xt::xarray<float>, 3>{
            ttml::core::to_xtensor(dq), ttml::core::to_xtensor(dk), ttml::core::to_xtensor(dv)};
    };
    const auto with_barrier = run(true);
    const auto with_endpoints = run(false);
    const char* names[] = {"dQ", "dK", "dV"};
    for (uint32_t k = 0; k < 3u; ++k) {
        EXPECT_TRUE(with_barrier[k] == with_endpoints[k])
            << names[k] << " differs between the barrier and endpoint variants of the dense schedule";
    }
}

// ---------------------------------------------------------- distinct slices
// Every slice-loop test above repeats one problem across its slices, which
// cannot see a slice reading another slice's rows. This gives each (batch,
// head) slice its own problem and checks each against its own reference.
namespace {

void check_op_distinct_slices(uint32_t batch, uint32_t heads, uint32_t C, uint32_t Bt, bool dense = false) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t N = 2u * C * Bt * kTile;
    const uint32_t d = 64u;
    const uint32_t slices = batch * heads;
    std::vector<Reference> refs;
    for (uint32_t sl = 0; sl < slices; ++sl) {
        // Different data per slice: the generators are seeded by a salt, so
        // vary the problem by reusing the two reference builders alternately
        // and scaling by the slice index.
        Reference r = dense ? make_dense_reference(N, d) : make_reference(N, d, /* positive */ (sl % 2u) == 1u);
        refs.push_back(std::move(r));
    }
    const auto stack = [&](auto field) {
        xt::xarray<float> out = xt::zeros<float>({batch, heads, N, d});
        for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t h = 0; h < heads; ++h) {
                xt::view(out, b, h, xt::all(), xt::all()) = field(refs[b * heads + h]);
            }
        }
        return out;
    };
    const auto stack_stat = [&](auto field) {
        xt::xarray<float> out = xt::zeros<float>({batch, heads, N, kTile});
        for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t h = 0; h < heads; ++h) {
                xt::view(out, b, h, xt::all(), xt::all()) = xt::view(field(refs[b * heads + h]), 0, 0, xt::all(), xt::all());
            }
        }
        return out;
    };
    const auto query = ttml::core::from_xtensor(stack([](const Reference& r) { return r.Q; }), device);
    const auto key = ttml::core::from_xtensor(stack([](const Reference& r) { return r.K; }), device);
    const auto value = ttml::core::from_xtensor(stack([](const Reference& r) { return r.V; }), device);
    const auto grad_output = ttml::core::from_xtensor(stack([](const Reference& r) { return r.dO; }), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        stack_stat([](const Reference& r) { return r.lse_tile; }), device);
    const auto row_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        stack_stat([](const Reference& r) { return r.u_tile; }), device);
    const auto [grad_query, grad_key, grad_value] = ttml::metal::cyclic_sdpa_bw(
        query, key, value, grad_output, lse, row_scalar, Bt, false,
        dense ? ttml::metal::AttentionMaskType::None : ttml::metal::AttentionMaskType::Causal);
    const auto dQ = ttml::core::to_xtensor(grad_query);
    const auto dK = ttml::core::to_xtensor(grad_key);
    const auto dV = ttml::core::to_xtensor(grad_value);
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            const auto& r = refs[b * heads + h];
            const std::string at = " slice (" + std::to_string(b) + ", " + std::to_string(h) + ")";
            const auto got = [&](const xt::xarray<float>& x) {
                return xt::xarray<float>(xt::view(x, b, h, xt::all(), xt::all()));
            };
            const auto close = [&](const xt::xarray<float>& x, const xt::xarray<float>& want, const char* name) {
                const float scale = xt::amax(xt::abs(want))();
                const float err = xt::amax(xt::abs(x - want))();
                EXPECT_LT(err, 0.06F * scale) << name << at << ": max error " << err << " against scale " << scale;
            };
            close(got(dQ), r.dQ, "dQ");
            close(got(dK), r.dK, "dK");
            close(got(dV), r.dV, "dV");
        }
    }
}

}  // namespace

TEST(CyclicSdpaBwOpTest, DistinctSlicesShortBlocks) {
    check_op_distinct_slices(/* batch */ 2, /* heads */ 3, /* C */ 2, /* Bt */ 1);
}

TEST(CyclicSdpaBwOpTest, DistinctSlicesTallBlocks) {
    check_op_distinct_slices(/* batch */ 2, /* heads */ 3, /* C */ 2, /* Bt */ 2);
}

TEST(CyclicSdpaBwOpTest, DistinctSlicesTallBlocksOneBatch) {
    check_op_distinct_slices(/* batch */ 1, /* heads */ 4, /* C */ 2, /* Bt */ 2);
}

TEST(CyclicSdpaBwDenseOpTest, DistinctSlicesTallBlocks) {
    check_op_distinct_slices(/* batch */ 2, /* heads */ 3, /* C */ 2, /* Bt */ 2, /* dense */ true);
}

// Running the op twice into the same accumulators doubles every gradient.
// dQ's seed is non-zero the second time, which nothing else here exercises:
// every other test seeds dQ from zeros.
namespace {
void check_accumulate_twice(uint32_t batch, uint32_t heads, uint32_t C, uint32_t Bt, bool dense = false) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t N = 2u * C * Bt * kTile;
    const uint32_t d = 64u;
    const uint32_t slices = batch * heads;
    const auto ref = dense ? make_dense_reference(N, d) : make_reference(N, d);
    const auto q = ttml::core::from_xtensor(as_4d_repeated(ref.Q, slices), device);
    const auto k = ttml::core::from_xtensor(as_4d_repeated(ref.K, slices), device);
    const auto v = ttml::core::from_xtensor(as_4d_repeated(ref.V, slices), device);
    const auto dO = ttml::core::from_xtensor(as_4d_repeated(ref.dO, slices), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(repeat_4d(ref.lse_tile, slices), device);
    const auto u = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(repeat_4d(ref.u_tile, slices), device);
    auto acc_q = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
    auto acc_k = ttnn::zeros_like(k, ttnn::DataType::FLOAT32);
    auto acc_v = ttnn::zeros_like(v, ttnn::DataType::FLOAT32);
    for (int pass = 0; pass < 2; ++pass) {
        std::tie(acc_q, acc_k, acc_v) = ttml::metal::cyclic_sdpa_bw(
            q, k, v, dO, lse, u, Bt, false,
            dense ? ttml::metal::AttentionMaskType::None : ttml::metal::AttentionMaskType::Causal,
            /* accumulate */ true, acc_q, acc_k, acc_v);
    }
    const auto dQ = ttml::core::to_xtensor(acc_q);
    const auto dK = ttml::core::to_xtensor(acc_k);
    const auto dV = ttml::core::to_xtensor(acc_v);
    for (uint32_t g = 0; g < slices; ++g) {
        const std::string at = " slice " + std::to_string(g);
        expect_close(dQ, 2.0F * ref.dQ, 0.06F, "dQ" + at, g);
        expect_close(dK, 2.0F * ref.dK, 0.06F, "dK" + at, g);
        expect_close(dV, 2.0F * ref.dV, 0.06F, "dV" + at, g);
    }
}
}  // namespace

TEST(CyclicSdpaBwOpTest, AccumulateTwiceDoublesShortBlocks) {
    check_accumulate_twice(/* batch */ 2, /* heads */ 3, /* C */ 2, /* Bt */ 1);
}
TEST(CyclicSdpaBwOpTest, AccumulateTwiceDoublesTallBlocks) {
    check_accumulate_twice(/* batch */ 2, /* heads */ 3, /* C */ 2, /* Bt */ 2);
}
TEST(CyclicSdpaBwOpTest, AccumulateTwiceDoublesTallBlocksOneSlice) {
    check_accumulate_twice(/* batch */ 1, /* heads */ 1, /* C */ 2, /* Bt */ 2);
}
TEST(CyclicSdpaBwOpTest, AccumulateTwiceDoublesTallBlocksFourHeads) {
    check_accumulate_twice(/* batch */ 1, /* heads */ 4, /* C */ 2, /* Bt */ 2);
}
TEST(CyclicSdpaBwDenseOpTest, AccumulateTwiceDoublesTallBlocks) {
    check_accumulate_twice(/* batch */ 2, /* heads */ 3, /* C */ 2, /* Bt */ 2, /* dense */ true);
}
TEST(CyclicSdpaBwDenseOpTest, AccumulateTwiceDoublesTallBlocksFourHeads) {
    check_accumulate_twice(/* batch */ 1, /* heads */ 4, /* C */ 2, /* Bt */ 2, /* dense */ true);
}
TEST(CyclicSdpaBwDenseOpTest, AccumulateTwiceDoublesShortBlocks) {
    check_accumulate_twice(/* batch */ 2, /* heads */ 3, /* C */ 2, /* Bt */ 1, /* dense */ true);
}

// ------------------------------------------------------- grouped-query heads
// Grouped-query attention: kv_heads key/value heads per batch, each shared by
// heads_per_group = q_heads / kv_heads query heads. Every query head is its
// own problem for dQ; the heads of a group add into one dK and one dV. The op
// runs a group's heads in turn on one core group and seeds every column's
// gradients from DRAM, so the sums come out in head order.
namespace {

// The backward of one head on the host from given (bf16-rounded) inputs, with
// the same arithmetic as make_reference -- for inputs shared between heads.
Reference reference_from_inputs(
    const xt::xarray<float>& Q,
    const xt::xarray<float>& K,
    const xt::xarray<float>& V,
    const xt::xarray<float>& dO,
    bool causal) {
    Reference r;
    r.N = static_cast<uint32_t>(Q.shape()[0]);
    r.d = static_cast<uint32_t>(Q.shape()[1]);
    const uint32_t N = r.N;
    const uint32_t d = r.d;
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));
    r.Q = Q;
    r.K = K;
    r.V = V;
    r.dO = dO;

    const xt::xarray<float> S = xt::linalg::dot(r.Q, xt::transpose(r.K)) * scale;
    xt::xarray<float> P = xt::zeros<float>({N, N});
    std::vector<float> lse(N, 0.0F);
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t last = causal ? i : N - 1u;
        float m = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j <= last; ++j) {
            m = std::max(m, S(i, j));
        }
        float sum = 0.0F;
        for (uint32_t j = 0; j <= last; ++j) {
            sum += std::exp(S(i, j) - m);
        }
        lse[i] = m + std::log(sum);
        for (uint32_t j = 0; j <= last; ++j) {
            P(i, j) = std::exp(S(i, j) - lse[i]);
        }
    }
    r.O = xt::linalg::dot(P, r.V);
    const xt::xarray<float> dP = xt::linalg::dot(r.dO, xt::transpose(r.V));
    std::vector<float> u(N, 0.0F);
    for (uint32_t i = 0; i < N; ++i) {
        float sum = 0.0F;
        for (uint32_t c = 0; c < d; ++c) {
            sum += r.dO(i, c) * r.O(i, c);
        }
        u[i] = sum;
    }
    xt::xarray<float> dS = xt::zeros<float>({N, N});
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t last = causal ? i : N - 1u;
        for (uint32_t j = 0; j <= last; ++j) {
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

struct GqaProblem {
    uint32_t batch{}, q_heads{}, kv_heads{}, N{}, d{};
    std::vector<Reference> heads;   // per (batch, query head), at b * q_heads + h
    xt::xarray<float> Q, dO;        // (B, H, N, d)
    xt::xarray<float> K, V;         // (B, G, N, d)
    xt::xarray<float> lse, u;       // (B, H, N, 32)
    xt::xarray<float> dK, dV;       // (B, G, N, d): the group's heads, summed in head order
};

GqaProblem make_gqa_problem(uint32_t batch, uint32_t q_heads, uint32_t kv_heads, uint32_t N, uint32_t d, bool dense) {
    GqaProblem p;
    p.batch = batch;
    p.q_heads = q_heads;
    p.kv_heads = kv_heads;
    p.N = N;
    p.d = d;
    const uint32_t hpg = q_heads / kv_heads;
    p.Q = xt::zeros<float>({batch, q_heads, N, d});
    p.dO = xt::zeros<float>({batch, q_heads, N, d});
    p.K = xt::zeros<float>({batch, kv_heads, N, d});
    p.V = xt::zeros<float>({batch, kv_heads, N, d});
    p.lse = xt::zeros<float>({batch, q_heads, N, kTile});
    p.u = xt::zeros<float>({batch, q_heads, N, kTile});
    p.dK = xt::zeros<float>({batch, kv_heads, N, d});
    p.dV = xt::zeros<float>({batch, kv_heads, N, d});
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t g = 0; g < kv_heads; ++g) {
            const uint32_t bg = b * kv_heads + g;
            const auto K = random_bf16_matrix(N, d, 1000u + 50u * bg);
            const auto V = random_bf16_matrix(N, d, 1001u + 50u * bg);
            xt::view(p.K, b, g, xt::all(), xt::all()) = K;
            xt::view(p.V, b, g, xt::all(), xt::all()) = V;
            xt::xarray<float> dK_sum = xt::zeros<float>({N, d});
            xt::xarray<float> dV_sum = xt::zeros<float>({N, d});
            for (uint32_t j = 0; j < hpg; ++j) {
                const uint32_t h = g * hpg + j;
                const uint32_t bh = b * q_heads + h;
                const auto Q = random_bf16_matrix(N, d, 2000u + 50u * bh);
                const auto dO = random_bf16_matrix(N, d, 2001u + 50u * bh);
                Reference r = reference_from_inputs(Q, K, V, dO, /* causal */ !dense);
                xt::view(p.Q, b, h, xt::all(), xt::all()) = Q;
                xt::view(p.dO, b, h, xt::all(), xt::all()) = dO;
                xt::view(p.lse, b, h, xt::all(), xt::all()) = xt::view(r.lse_tile, 0, 0, xt::all(), xt::all());
                xt::view(p.u, b, h, xt::all(), xt::all()) = xt::view(r.u_tile, 0, 0, xt::all(), xt::all());
                dK_sum = dK_sum + r.dK;  // head order, as the device adds them
                dV_sum = dV_sum + r.dV;
                p.heads.push_back(std::move(r));  // b-major, h-minor: b * q_heads + h
            }
            xt::view(p.dK, b, g, xt::all(), xt::all()) = dK_sum;
            xt::view(p.dV, b, g, xt::all(), xt::all()) = dV_sum;
        }
    }
    return p;
}

struct GqaOutputs {
    xt::xarray<float> dQ, dK, dV;
};

// Upload the problem and run the op; with `twice`, accumulate two launches
// into preallocated sums so every gradient should come out doubled.
GqaOutputs run_gqa_op(const GqaProblem& p, uint32_t Bt, bool dense, bool twice = false) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto q = ttml::core::from_xtensor(p.Q, device);
    const auto k = ttml::core::from_xtensor(p.K, device);
    const auto v = ttml::core::from_xtensor(p.V, device);
    const auto dO = ttml::core::from_xtensor(p.dO, device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(p.lse, device);
    const auto u = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(p.u, device);
    const auto mask = dense ? ttml::metal::AttentionMaskType::None : ttml::metal::AttentionMaskType::Causal;
    if (!twice) {
        const auto [gq, gk, gv] = ttml::metal::cyclic_sdpa_bw(q, k, v, dO, lse, u, Bt, false, mask);
        return {ttml::core::to_xtensor(gq), ttml::core::to_xtensor(gk), ttml::core::to_xtensor(gv)};
    }
    auto acc_q = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
    auto acc_k = ttnn::zeros_like(k, ttnn::DataType::FLOAT32);
    auto acc_v = ttnn::zeros_like(v, ttnn::DataType::FLOAT32);
    for (int pass = 0; pass < 2; ++pass) {
        std::tie(acc_q, acc_k, acc_v) = ttml::metal::cyclic_sdpa_bw(
            q, k, v, dO, lse, u, Bt, false, mask, /* accumulate */ true, acc_q, acc_k, acc_v);
    }
    return {ttml::core::to_xtensor(acc_q), ttml::core::to_xtensor(acc_k), ttml::core::to_xtensor(acc_v)};
}

xt::xarray<float> slice_of(const xt::xarray<float>& x, uint32_t b, uint32_t h) {
    return xt::xarray<float>(xt::view(x, b, h, xt::all(), xt::all()));
}

float relative_rms(const xt::xarray<float>& got, const xt::xarray<float>& want) {
    const float num = xt::sum(xt::square(got - want))();
    const float den = xt::sum(xt::square(want))();
    return std::sqrt(num / den);
}

void expect_slice_close(
    const xt::xarray<float>& got, const xt::xarray<float>& want, uint32_t b, uint32_t h, const char* what,
    float factor = 1.0F) {
    const auto g = slice_of(got, b, h);
    const float scale = xt::amax(xt::abs(want))() * factor;
    const float err = xt::amax(xt::abs(g - factor * want))();
    EXPECT_LT(err, 0.06F * scale) << what << " (" << b << ", " << h << "): max error " << err << " against scale "
                                  << scale;
}

void check_gqa(
    uint32_t batch, uint32_t q_heads, uint32_t kv_heads, uint32_t C, uint32_t Bt, bool dense = false,
    bool twice = false) {
    const uint32_t N = 2u * C * Bt * kTile;
    const auto p = make_gqa_problem(batch, q_heads, kv_heads, N, 64u, dense);
    const auto out = run_gqa_op(p, Bt, dense, twice);
    const float factor = twice ? 2.0F : 1.0F;
    ASSERT_EQ(out.dK.shape()[1], kv_heads) << "dK must have the key's head count";
    ASSERT_EQ(out.dV.shape()[1], kv_heads) << "dV must have the key's head count";
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t h = 0; h < q_heads; ++h) {
            expect_slice_close(out.dQ, p.heads[b * q_heads + h].dQ, b, h, "dQ", factor);
        }
        for (uint32_t g = 0; g < kv_heads; ++g) {
            expect_slice_close(out.dK, slice_of(p.dK, b, g), b, g, "dK", factor);
            expect_slice_close(out.dV, slice_of(p.dV, b, g), b, g, "dV", factor);
        }
    }
}

}  // namespace

TEST(CyclicSdpaBwGqaTest, TwoHeadsPerKeyHead) {
    check_gqa(/* batch */ 2, /* q_heads */ 4, /* kv_heads */ 2, /* C */ 2, /* Bt */ 1);
}
TEST(CyclicSdpaBwGqaTest, TwoHeadsPerKeyHeadTallBlocks) {
    check_gqa(2, 4, 2, 2, 2);
}
TEST(CyclicSdpaBwGqaTest, FourHeadsPerKeyHead) {
    check_gqa(1, 8, 2, 2, 1);
}
// One key head, so one core group runs all eight query heads in turn.
TEST(CyclicSdpaBwGqaTest, EightHeadsOneKeyHead) {
    check_gqa(1, 8, 1, 2, 2);
}
// Six (batch, key head) slices: the groups take them side by side and in turn.
TEST(CyclicSdpaBwGqaTest, ThreeBatches) {
    check_gqa(3, 4, 2, 2, 1);
}
TEST(CyclicSdpaBwGqaTest, FourTileBlocks) {
    check_gqa(1, 4, 2, 2, 4);
}
TEST(CyclicSdpaBwGqaTest, WiderHead) {
    const auto p = make_gqa_problem(1, 4, 2, 2u * 2u * 1u * kTile, 128u, false);
    const auto out = run_gqa_op(p, 1, false);
    for (uint32_t h = 0; h < 4; ++h) {
        expect_slice_close(out.dQ, p.heads[h].dQ, 0, h, "dQ");
    }
    for (uint32_t g = 0; g < 2; ++g) {
        expect_slice_close(out.dK, slice_of(p.dK, 0, g), 0, g, "dK");
        expect_slice_close(out.dV, slice_of(p.dV, 0, g), 0, g, "dV");
    }
}
TEST(CyclicSdpaBwGqaDenseTest, TwoHeadsPerKeyHead) {
    check_gqa(2, 4, 2, 2, 1, /* dense */ true);
}
TEST(CyclicSdpaBwGqaDenseTest, FourHeadsPerKeyHeadTallBlocks) {
    check_gqa(1, 8, 2, 2, 2, /* dense */ true);
}
TEST(CyclicSdpaBwGqaTest, AccumulateTwiceDoubles) {
    check_gqa(2, 4, 2, 2, 2, /* dense */ false, /* twice */ true);
}
TEST(CyclicSdpaBwGqaDenseTest, AccumulateTwiceDoubles) {
    check_gqa(1, 8, 2, 2, 1, /* dense */ true, /* twice */ true);
}

TEST(CyclicSdpaBwGqaTest, RefusesHeadCountsThatDoNotDivide) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t N = 2u * 2u * kTile;
    const auto q = ttml::core::from_xtensor(xt::xarray<float>(xt::zeros<float>({1u, 3u, N, 64u})), device);
    const auto k = ttml::core::from_xtensor(xt::xarray<float>(xt::zeros<float>({1u, 2u, N, 64u})), device);
    const auto stat = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        xt::xarray<float>(xt::zeros<float>({1u, 3u, N, kTile})), device);
    EXPECT_ANY_THROW(ttml::metal::cyclic_sdpa_bw(q, k, k, q, stat, stat, 1));
}

// The grouped result is the per-head result summed. Run the same problem with
// K and V repeated to every query head -- plain multi-head attention -- and
// add the heads of each group on the host in head order, which is the order
// the device adds them in: one Float32 add of the seed at each handover.
// dQ does not depend on the grouping at all and must match bit for bit.
TEST(CyclicSdpaBwGqaTest, MatchesRepeatedKeysSummedInHeadOrder) {
    const uint32_t batch = 2, q_heads = 4, kv_heads = 2, C = 2, Bt = 2, d = 64;
    const uint32_t hpg = q_heads / kv_heads;
    const uint32_t N = 2u * C * Bt * kTile;
    const auto p = make_gqa_problem(batch, q_heads, kv_heads, N, d, false);
    const auto grouped = run_gqa_op(p, Bt, false);

    GqaProblem repeated = p;
    repeated.kv_heads = q_heads;
    repeated.K = xt::zeros<float>({batch, q_heads, N, d});
    repeated.V = xt::zeros<float>({batch, q_heads, N, d});
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t h = 0; h < q_heads; ++h) {
            xt::view(repeated.K, b, h, xt::all(), xt::all()) = xt::view(p.K, b, h / hpg, xt::all(), xt::all());
            xt::view(repeated.V, b, h, xt::all(), xt::all()) = xt::view(p.V, b, h / hpg, xt::all(), xt::all());
        }
    }
    const auto per_head = run_gqa_op(repeated, Bt, false);

    const auto count_mismatches = [](const xt::xarray<float>& a, const xt::xarray<float>& b, float& worst) {
        uint32_t n = 0;
        worst = 0.0F;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a.flat(i) != b.flat(i)) {
                ++n;
                worst = std::max(worst, std::abs(a.flat(i) - b.flat(i)));
            }
        }
        return n;
    };
    float worst = 0.0F;
    EXPECT_EQ(count_mismatches(grouped.dQ, per_head.dQ, worst), 0u) << "dQ differs between grouped and repeated keys, worst " << worst;
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t g = 0; g < kv_heads; ++g) {
            xt::xarray<float> dK_sum = xt::zeros<float>({N, d});
            xt::xarray<float> dV_sum = xt::zeros<float>({N, d});
            for (uint32_t j = 0; j < hpg; ++j) {
                dK_sum = dK_sum + slice_of(per_head.dK, b, g * hpg + j);
                dV_sum = dV_sum + slice_of(per_head.dV, b, g * hpg + j);
            }
            const float scale_k = xt::amax(xt::abs(dK_sum))();
            const float scale_v = xt::amax(xt::abs(dV_sum))();
            const uint32_t nk = count_mismatches(slice_of(grouped.dK, b, g), dK_sum, worst);
            std::printf("  gqa vs repeated (%u, %u): dK %u words differ, worst %.3e of %.3e\n", b, g, nk, worst, scale_k);
            EXPECT_LT(worst, 1e-5F * scale_k) << "dK (" << b << ", " << g << ")";
            const uint32_t nv = count_mismatches(slice_of(grouped.dV, b, g), dV_sum, worst);
            std::printf("  gqa vs repeated (%u, %u): dV %u words differ, worst %.3e of %.3e\n", b, g, nv, worst, scale_v);
            EXPECT_LT(worst, 1e-5F * scale_v) << "dV (" << b << ", " << g << ")";
        }
    }
}

// Through the real forward, whose statistics are the ones a model hands over,
// and side by side with the repository's two-pass backward, which has
// grouped-query attention of its own. Both are graded against the host
// reference; the relative RMS of each is printed so the two can be compared.
TEST(CyclicSdpaBwGqaTest, ConsumesTheRealForwardAndAgreesWithTheTwoPassBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t batch = 1, q_heads = 4, kv_heads = 2, C = 4, Bt = 2, d = 64;
    const uint32_t N = 2u * C * Bt * kTile;
    const auto p = make_gqa_problem(batch, q_heads, kv_heads, N, d, false);
    const auto q = ttml::core::from_xtensor(p.Q, device);
    const auto k = ttml::core::from_xtensor(p.K, device);
    const auto v = ttml::core::from_xtensor(p.V, device);
    const auto dO = ttml::core::from_xtensor(p.dO, device);

    const auto forward = ttml::metal::sdpa_fw(
        q, k, v, ttml::metal::AttentionMaskType::Causal, std::nullopt, 0.0F, /*return_intermediates=*/true);
    const auto attn_output = forward[0].value();
    const auto intermediates = forward[1].value();
    const auto O = ttml::core::to_xtensor(attn_output);
    for (uint32_t h = 0; h < q_heads; ++h) {
        expect_slice_close(O, p.heads[h].O, 0, h, "sdpa_fw output against O = P V");
    }

    const auto [cq, ck, cv] = ttml::metal::cyclic_sdpa_bw_from_forward(q, k, v, dO, attn_output, intermediates, Bt);
    const auto [tq, tk, tv] = ttml::metal::sdpa_bw(
        dO, attn_output, q, k, v, intermediates, ttml::metal::AttentionMaskType::Causal);
    ASSERT_EQ(tk.logical_shape()[1], kv_heads) << "the two-pass backward returns dK with the key's head count";
    const auto cyclic = GqaOutputs{ttml::core::to_xtensor(cq), ttml::core::to_xtensor(ck), ttml::core::to_xtensor(cv)};
    const auto two_pass = GqaOutputs{ttml::core::to_xtensor(tq), ttml::core::to_xtensor(tk), ttml::core::to_xtensor(tv)};

    const auto grade = [&](const char* what, const xt::xarray<float>& c, const xt::xarray<float>& t,
                           const xt::xarray<float>& want, uint32_t h) {
        expect_slice_close(c, want, 0, h, (std::string("cyclic ") + what).c_str());
        expect_slice_close(t, want, 0, h, (std::string("two-pass ") + what).c_str());
        std::printf(
            "  %s head %u: relative RMS cyclic %.2e, two-pass %.2e\n", what, h, relative_rms(slice_of(c, 0, h), want),
            relative_rms(slice_of(t, 0, h), want));
    };
    for (uint32_t h = 0; h < q_heads; ++h) {
        grade("dQ", cyclic.dQ, two_pass.dQ, p.heads[h].dQ, h);
    }
    for (uint32_t g = 0; g < kv_heads; ++g) {
        grade("dK", cyclic.dK, two_pass.dK, slice_of(p.dK, 0, g), g);
        grade("dV", cyclic.dV, two_pass.dV, slice_of(p.dV, 0, g), g);
    }
}

// What a ring step sequence does on one chip: the same Q, dO and statistics
// against one key chunk under the causal schedule, then against another
// chunk under the dense one, dQ accumulating across the two launches. Checked
// against the two launches run separately and summed.
namespace {
void check_mini_ring(uint32_t batch, uint32_t heads, uint32_t C, uint32_t Bt) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t N = 2u * C * Bt * kTile;
    const uint32_t d = 64u;
    const uint32_t slices = batch * heads;
    const auto own = make_reference(N, d);
    const auto visiting = make_dense_reference(N, d);
    const auto q = ttml::core::from_xtensor(as_4d_repeated(own.Q, slices), device);
    const auto dO = ttml::core::from_xtensor(as_4d_repeated(own.dO, slices), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(repeat_4d(own.lse_tile, slices), device);
    const auto u = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(repeat_4d(own.u_tile, slices), device);
    const auto kA = ttml::core::from_xtensor(as_4d_repeated(own.K, slices), device);
    const auto vA = ttml::core::from_xtensor(as_4d_repeated(own.V, slices), device);
    const auto kB = ttml::core::from_xtensor(as_4d_repeated(visiting.K, slices), device);
    const auto vB = ttml::core::from_xtensor(as_4d_repeated(visiting.V, slices), device);
    using Mask = ttml::metal::AttentionMaskType;

    const auto separate = [&](const ttnn::Tensor& k, const ttnn::Tensor& v, Mask mask) {
        const auto [dq, dk, dv] = ttml::metal::cyclic_sdpa_bw(q, k, v, dO, lse, u, Bt, false, mask);
        return ttml::core::to_xtensor(dq);
    };
    const xt::xarray<float> want = separate(kA, vA, Mask::Causal) + separate(kB, vB, Mask::None);

    auto acc_q = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
    auto acc_kA = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
    auto acc_vA = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
    auto acc_kB = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
    auto acc_vB = ttnn::zeros_like(q, ttnn::DataType::FLOAT32);
    std::tie(acc_q, acc_kA, acc_vA) =
        ttml::metal::cyclic_sdpa_bw(q, kA, vA, dO, lse, u, Bt, false, Mask::Causal, true, acc_q, acc_kA, acc_vA);
    std::tie(acc_q, acc_kB, acc_vB) =
        ttml::metal::cyclic_sdpa_bw(q, kB, vB, dO, lse, u, Bt, false, Mask::None, true, acc_q, acc_kB, acc_vB);
    const auto got = ttml::core::to_xtensor(acc_q);
    const float rms = std::sqrt(xt::mean(xt::square(got - want))()) / (std::sqrt(xt::mean(xt::square(want))()) + 1e-12F);
    EXPECT_LT(rms, 2e-3F) << "dQ accumulated over a causal launch and a dense launch, rel RMS " << rms;
}
}  // namespace

TEST(CyclicSdpaBwOpTest, MiniRingTallBlocksFourHeads) {
    check_mini_ring(/* batch */ 1, /* heads */ 4, /* C */ 2, /* Bt */ 2);
}
TEST(CyclicSdpaBwOpTest, MiniRingShortBlocksFourHeads) {
    check_mini_ring(/* batch */ 1, /* heads */ 4, /* C */ 2, /* Bt */ 1);
}
TEST(CyclicSdpaBwOpTest, MiniRingTallBlocksOneSlice) {
    check_mini_ring(/* batch */ 1, /* heads */ 1, /* C */ 2, /* Bt */ 2);
}
TEST(CyclicSdpaBwOpTest, MiniRingTallBlocksOneCore) {
    check_mini_ring(/* batch */ 1, /* heads */ 4, /* C */ 1, /* Bt */ 2);
}
TEST(CyclicSdpaBwOpTest, MiniRingTallBlocksThreeHeads) {
    check_mini_ring(/* batch */ 1, /* heads */ 3, /* C */ 2, /* Bt */ 2);
}

// ---------------------------------------------------------- chunk pairs
// Sub-problems as chunk pairs: the local sequence is two chunks back to back,
// and a launch names which (query chunk, key chunk) pairs to run. This is
// what a zigzag ring step is made of. Pinned bitwise against the op run on
// the chunks as separate tensors, which is the same arithmetic on the same
// tiles with only the addressing different.
namespace {

struct TwoChunkProblem {
    uint32_t n{};  // rows per chunk
    uint32_t d{};
    Reference a, b;                        // chunk 0 and chunk 1
    ttnn::Tensor query, key, value, grad_output, lse, row_scalar;  // [a | b], (1, 1, 2n, d)
};

TwoChunkProblem make_two_chunk_problem(uint32_t C, uint32_t Bt, uint32_t d = 64) {
    auto* device = &ttml::autograd::ctx().get_device();
    TwoChunkProblem p;
    p.n = 2u * C * Bt * kTile;
    p.d = d;
    // Two different draws, so a wrong chunk address cannot pass by accident.
    p.a = make_reference_inputs_only(p.n, d);
    p.b = make_dense_reference(p.n, d);
    const auto cat = [](const xt::xarray<float>& x, const xt::xarray<float>& y) {
        return xt::xarray<float>(xt::concatenate(xt::xtuple(x, y), 0));
    };
    const auto cat4 = [](const xt::xarray<float>& x, const xt::xarray<float>& y) {
        return xt::xarray<float>(xt::concatenate(xt::xtuple(x, y), 2));
    };
    p.query = ttml::core::from_xtensor(as_4d(cat(p.a.Q, p.b.Q)), device);
    p.key = ttml::core::from_xtensor(as_4d(cat(p.a.K, p.b.K)), device);
    p.value = ttml::core::from_xtensor(as_4d(cat(p.a.V, p.b.V)), device);
    p.grad_output = ttml::core::from_xtensor(as_4d(cat(p.a.dO, p.b.dO)), device);
    p.lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(cat4(p.a.lse_tile, p.b.lse_tile), device);
    p.row_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(cat4(p.a.u_tile, p.b.u_tile), device);
    return p;
}

// The op on one chunk pair as separate tensors: the reference for the pair.
Gradients run_on_chunks(
    const TwoChunkProblem& p, uint32_t row_chunk, uint32_t col_chunk, uint32_t Bt, ttml::metal::AttentionMaskType mask) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Reference& rows = row_chunk == 0u ? p.a : p.b;
    const Reference& cols = col_chunk == 0u ? p.a : p.b;
    const auto q = ttml::core::from_xtensor(as_4d(rows.Q), device);
    const auto k = ttml::core::from_xtensor(as_4d(cols.K), device);
    const auto v = ttml::core::from_xtensor(as_4d(cols.V), device);
    const auto dO = ttml::core::from_xtensor(as_4d(rows.dO), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(rows.lse_tile, device);
    const auto u = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(rows.u_tile, device);
    const auto [dq, dk, dv] = ttml::metal::cyclic_sdpa_bw(q, k, v, dO, lse, u, Bt, false, mask);
    return {ttml::core::to_xtensor(dq), ttml::core::to_xtensor(dk), ttml::core::to_xtensor(dv)};
}

xt::xarray<float> chunk_rows(const xt::xarray<float>& x, uint32_t chunk, uint32_t n) {
    return xt::xarray<float>(xt::view(x, xt::all(), xt::all(), xt::range(chunk * n, (chunk + 1u) * n), xt::all()));
}

void check_chunk_pairs(
    uint32_t C,
    uint32_t Bt,
    ttml::metal::AttentionMaskType mask,
    const std::vector<uint32_t>& row_chunks,
    const std::vector<uint32_t>& col_chunks) {
    const auto p = make_two_chunk_problem(C, Bt);
    const auto [dq, dk, dv] = ttml::metal::cyclic_sdpa_bw(
        p.query, p.key, p.value, p.grad_output, p.lse, p.row_scalar, Bt, false, mask,
        /* accumulate */ false, std::nullopt, std::nullopt, std::nullopt, /* max_groups */ 0u,
        /* sequence_chunks */ 2u, row_chunks, col_chunks);
    const auto dQ = ttml::core::to_xtensor(dq);
    const auto dK = ttml::core::to_xtensor(dk);
    const auto dV = ttml::core::to_xtensor(dv);

    // Every named pair matches the op on its chunks, bitwise.
    std::array<bool, 2> row_touched{false, false};
    std::array<bool, 2> col_touched{false, false};
    for (size_t i = 0; i < row_chunks.size(); ++i) {
        const auto ref = run_on_chunks(p, row_chunks[i], col_chunks[i], Bt, mask);
        row_touched[row_chunks[i]] = true;
        col_touched[col_chunks[i]] = true;
        EXPECT_TRUE(chunk_rows(dQ, row_chunks[i], p.n) == ref.dQ) << "dQ of pair " << i;
        EXPECT_TRUE(chunk_rows(dK, col_chunks[i], p.n) == ref.dK) << "dK of pair " << i;
        EXPECT_TRUE(chunk_rows(dV, col_chunks[i], p.n) == ref.dV) << "dV of pair " << i;
    }
    // A chunk no pair names is left as the op allocated it: zero.
    for (uint32_t c = 0; c < 2u; ++c) {
        if (!row_touched[c]) {
            EXPECT_TRUE(xt::all(xt::equal(chunk_rows(dQ, c, p.n), 0.0F))) << "dQ chunk " << c << " should be untouched";
        }
        if (!col_touched[c]) {
            EXPECT_TRUE(xt::all(xt::equal(chunk_rows(dK, c, p.n), 0.0F))) << "dK chunk " << c << " should be untouched";
            EXPECT_TRUE(xt::all(xt::equal(chunk_rows(dV, c, p.n), 0.0F))) << "dV chunk " << c << " should be untouched";
        }
    }
}

}  // namespace

TEST(CyclicSdpaBwOpTest, FourCoresWiderHead) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 1, /* use_barrier */ false, /* d */ 128);
}
TEST(CyclicSdpaBwOpTest, FourCoresWiderHeadTallBlocks) {
    check_op(/* C */ 4, /* Bt */ 2, /* slices */ 1, /* use_barrier */ false, /* d */ 128);
}

TEST(CyclicSdpaBwChunkPairTest, OneCausalPairOnTheSecondChunk) {
    check_chunk_pairs(/* C */ 4, /* Bt */ 1, ttml::metal::AttentionMaskType::Causal, {1}, {1});
}

TEST(CyclicSdpaBwChunkPairTest, BothTrianglesInOneLaunch) {
    // The diagonal step of a zigzag ring, causal half.
    check_chunk_pairs(/* C */ 4, /* Bt */ 1, ttml::metal::AttentionMaskType::Causal, {0, 1}, {0, 1});
}

TEST(CyclicSdpaBwChunkPairTest, TheOffDiagonalBlock) {
    // The diagonal step's full block: later local chunk against the earlier one.
    check_chunk_pairs(/* C */ 4, /* Bt */ 1, ttml::metal::AttentionMaskType::None, {1}, {0});
}

// Two pairs that share a chunk cannot be slices of one launch: they would
// race on it. The op says so instead of computing something.
TEST(CyclicSdpaBwChunkPairTest, RefusesPairsThatShareAChunk) {
    const auto p = make_two_chunk_problem(/* C */ 4, /* Bt */ 1);
    for (const auto& [rows, cols] : std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>{
             {{0, 1}, {0, 0}},  // a shared key chunk
             {{1, 1}, {0, 1}},  // a shared query chunk
         }) {
        EXPECT_THROW(
            (void)ttml::metal::cyclic_sdpa_bw(
                p.query, p.key, p.value, p.grad_output, p.lse, p.row_scalar, 1u, false,
                ttml::metal::AttentionMaskType::None, false, std::nullopt, std::nullopt, std::nullopt, 0u, 2u, rows,
                cols),
            std::exception);
    }
}

TEST(CyclicSdpaBwChunkPairTest, WithTallBlocks) {
    check_chunk_pairs(/* C */ 2, /* Bt */ 2, ttml::metal::AttentionMaskType::None, {1}, {0});
}

// Two launches, each accumulating into the same outputs, add up: the second
// pair's dK, dV for the shared key chunk start from the first's. The seed
// passes through the Src registers on its way in and loses low mantissa bits
// (review item 2), so the sum differs from the summed separate runs at the
// 5e-4 level, not the 1e-7 of FP32; graded by RMS at 2e-3.
TEST(CyclicSdpaBwChunkPairTest, TwoLaunchesAccumulateIntoASharedKeyChunk) {
    constexpr uint32_t C = 4u;
    constexpr uint32_t Bt = 1u;
    const auto p = make_two_chunk_problem(C, Bt);
    auto acc_q = ttnn::zeros_like(p.query, ttnn::DataType::FLOAT32);
    auto acc_k = ttnn::zeros_like(p.key, ttnn::DataType::FLOAT32);
    auto acc_v = ttnn::zeros_like(p.value, ttnn::DataType::FLOAT32);
    for (const uint32_t row_chunk : {0u, 1u}) {
        std::tie(acc_q, acc_k, acc_v) = ttml::metal::cyclic_sdpa_bw(
            p.query, p.key, p.value, p.grad_output, p.lse, p.row_scalar, Bt, false,
            ttml::metal::AttentionMaskType::None,
            /* accumulate */ true, acc_q, acc_k, acc_v, 0u, 2u, {row_chunk}, {0u});
    }
    const auto dQ = ttml::core::to_xtensor(acc_q);
    const auto dK = ttml::core::to_xtensor(acc_k);
    const auto dV = ttml::core::to_xtensor(acc_v);
    const auto r0 = run_on_chunks(p, 0, 0, Bt, ttml::metal::AttentionMaskType::None);
    const auto r1 = run_on_chunks(p, 1, 0, Bt, ttml::metal::AttentionMaskType::None);
    const xt::xarray<float> want_dK = r0.dK + r1.dK;
    const xt::xarray<float> want_dV = r0.dV + r1.dV;
    const auto rms = [](const xt::xarray<float>& x, const xt::xarray<float>& y) {
        return std::sqrt(xt::mean(xt::square(x - y))()) / (std::sqrt(xt::mean(xt::square(y))()) + 1e-12F);
    };
    EXPECT_LT(rms(chunk_rows(dK, 0, p.n), want_dK), 2e-3F);
    EXPECT_LT(rms(chunk_rows(dV, 0, p.n), want_dV), 2e-3F);
    EXPECT_TRUE(chunk_rows(dQ, 0, p.n) == r0.dQ);
    EXPECT_TRUE(chunk_rows(dQ, 1, p.n) == r1.dQ);
    EXPECT_TRUE(xt::all(xt::equal(chunk_rows(dK, 1, p.n), 0.0F)));
}

// ---------------------------------------------------------- the slice loop
// More slices than groups: the groups run their slices in turn inside one
// launch. Forced here with max_groups at a size where everything would fit,
// so the loop itself is what is under test -- slot parity and tags carried
// across a slice boundary by the global timestep, the previous slice's last
// column popped at the next slice's start, visited flags reset per slice.
// Uneven counts (5 slices over 2 groups) put a group on its last slice while
// its neighbour has one more to run.
TEST(CyclicSdpaBwOpTest, SlicesLoopWithinAGroup) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 6, /* use_barrier */ false, 64, /* max_groups */ 2);
}

TEST(CyclicSdpaBwOpTest, SlicesLoopUnevenly) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 5, /* use_barrier */ false, 64, /* max_groups */ 2);
}

TEST(CyclicSdpaBwOpTest, SlicesLoopOnOneGroup) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 3, /* use_barrier */ false, 64, /* max_groups */ 1);
}

TEST(CyclicSdpaBwOpTest, SlicesLoopWithTallBlocks) {
    check_op(/* C */ 4, /* Bt */ 2, /* slices */ 4, /* use_barrier */ false, 64, /* max_groups */ 2);
}

// The barrier variant loops too: its arrival counter and release value are
// global timesteps now, and must keep rising across the boundary.
TEST(CyclicSdpaBwOpTest, SlicesLoopWithTheBarrier) {
    check_op(/* C */ 4, /* Bt */ 1, /* slices */ 4, /* use_barrier */ true, 64, /* max_groups */ 2);
}

TEST(CyclicSdpaBwDenseOpTest, SlicesLoopWithinAGroup) {
    check_dense_op(/* C */ 4, /* Bt */ 1, /* slices */ 6, /* use_barrier */ false, 64, /* max_groups */ 2);
}

TEST(CyclicSdpaBwDenseOpTest, SlicesLoopUnevenly) {
    check_dense_op(/* C */ 4, /* Bt */ 1, /* slices */ 5, /* use_barrier */ false, 64, /* max_groups */ 2);
}

// Looping must give the same bits as running the slices side by side: the
// per-slice arithmetic is identical, only the order in time changes.
TEST(CyclicSdpaBwOpTest, LoopedSlicesMatchSideBySideBitwise) {
    auto* device = &ttml::autograd::ctx().get_device();
    constexpr uint32_t C = 4u, Bt = 1u, d = 64u, slices = 4u;
    const uint32_t N = 2u * C * Bt * kTile;
    const auto ref = make_reference(N, d);
    const auto query = ttml::core::from_xtensor(as_4d_repeated(ref.Q, slices), device);
    const auto key = ttml::core::from_xtensor(as_4d_repeated(ref.K, slices), device);
    const auto value = ttml::core::from_xtensor(as_4d_repeated(ref.V, slices), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d_repeated(ref.dO, slices), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(repeat_4d(ref.lse_tile, slices), device);
    const auto row_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(repeat_4d(ref.u_tile, slices), device);
    const auto run = [&](uint32_t max_groups) {
        const auto [dq, dk, dv] = ttml::metal::cyclic_sdpa_bw(
            query, key, value, grad_output, lse, row_scalar, Bt, false, ttml::metal::AttentionMaskType::Causal,
            false, std::nullopt, std::nullopt, std::nullopt, max_groups);
        return std::array<xt::xarray<float>, 3>{
            ttml::core::to_xtensor(dq), ttml::core::to_xtensor(dk), ttml::core::to_xtensor(dv)};
    };
    const auto side_by_side = run(0);
    const auto looped = run(1);
    const char* names[] = {"dQ", "dK", "dV"};
    for (uint32_t k = 0; k < 3u; ++k) {
        EXPECT_TRUE(side_by_side[k] == looped[k]) << names[k] << " differs when the slices are looped";
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

// End to end against the real forward, which is the one thing every other
// test here does not do: they all build L and O from the host reference, so
// they check arithmetic while assuming a convention.
//
// The convention is not obvious. sdpa_fw keeps its scores and its running
// maximum *unscaled*, folding the softmax scale into its exponential, so its
// statistics could easily have been in different units than this kernel
// wants. It computes lse = scale * max + log(sum_exp), which is exactly
// P = exp(aS - L) -- but that is a claim about someone else's code, and this
// is the test that it holds.
void check_against_the_real_forward(uint32_t C, uint32_t Bt, uint32_t d) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto ref = make_reference(2u * C * Bt * kTile, d);

    const auto query = ttml::core::from_xtensor(as_4d_repeated(ref.Q, 1), device);
    const auto key = ttml::core::from_xtensor(as_4d_repeated(ref.K, 1), device);
    const auto value = ttml::core::from_xtensor(as_4d_repeated(ref.V, 1), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d_repeated(ref.dO, 1), device);

    // The forward, on device, with its own statistics.
    const auto forward = ttml::metal::sdpa_fw(
        query, key, value, ttml::metal::AttentionMaskType::Causal, std::nullopt, 0.0F,
        /*return_intermediates=*/true);
    const auto attn_output = forward[0].value();
    const auto intermediates = forward[1].value();

    // Sanity first: the forward agrees with the reference's own O, or nothing
    // below means anything.
    expect_close(ttml::core::to_xtensor(attn_output), ref.O, 0.06F, "sdpa_fw output against O = P V");

    const auto [grad_query, grad_key, grad_value] = ttml::metal::cyclic_sdpa_bw_from_forward(
        query, key, value, grad_output, attn_output, intermediates, Bt, /* use_barrier */ false);

    const std::string at = " (Bt = " + std::to_string(Bt) + ", d = " + std::to_string(d) + ")";
    expect_close(ttml::core::to_xtensor(grad_query), ref.dQ, 0.06F, "dQ from sdpa_fw" + at);
    expect_close(ttml::core::to_xtensor(grad_key), ref.dK, 0.06F, "dK from sdpa_fw" + at);
    expect_close(ttml::core::to_xtensor(grad_value), ref.dV, 0.06F, "dV from sdpa_fw" + at);
}

TEST(CyclicSdpaBwOpTest, ConsumesTheRealForward) {
    check_against_the_real_forward(/* C */ 4, /* Bt */ 1, /* d */ 64);
}

TEST(CyclicSdpaBwOpTest, ConsumesTheRealForwardWithTallBlocks) {
    check_against_the_real_forward(/* C */ 4, /* Bt */ 2, /* d */ 64);
}

// d = 128 is where 1/sqrt(d) is not a power of two, so the scale stays in the
// exponential instead of folding into K. This runs that fallback against the
// real forward too, which the d = 64 cases cannot.
TEST(CyclicSdpaBwOpTest, ConsumesTheRealForwardWhereTheScaleDoesNotFold) {
    check_against_the_real_forward(/* C */ 4, /* Bt */ 1, /* d */ 128);
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

// The column gradients are where the two once differed: the per-timestep
// reload copied the running sum through the Src registers and cost
// precision. Neither path reloads now -- both start an interval's
// accumulator from zero and add what DRAM holds at the handover, exactly --
// so they agree to the order of the summation (a rounding-level difference
// at most), and this pins that they stay that close.
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
    EXPECT_NEAR(resident_dk, reload_dk, 1e-6F);
    EXPECT_NEAR(resident_dv, reload_dv, 1e-6F);

    // An absolute bound as well as a relative one. Dropping the compute
    // kernels from the default HiFi4 to HiFi2 -- on the argument that every
    // matmul operand is bfloat16, so there should be no mantissa bits for the
    // extra fidelity phases to resolve -- made dK's error 9.8e-3 against the
    // 1.05e-3 here, nine times worse, and bought 3 to 4% of runtime. Every
    // other test in this file passed with that change in place: the
    // gradient-accuracy tolerances are loose enough to hide it, and the
    // comparison above only asks that the two paths agree with each other,
    // which they still did. This is the assertion that catches it.
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
    // CYCLIC_PROFILE_BT and CYCLIC_PROFILE_D pick the shape (defaults 4, 64);
    // CYCLIC_PROFILE_DENSE=1 profiles the dense schedule instead.
    uint32_t Bt = 4;
    uint32_t d = 64;
    if (const char* e = std::getenv("CYCLIC_PROFILE_BT"); e != nullptr && *e != '\0') {
        Bt = static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
    }
    if (const char* e = std::getenv("CYCLIC_PROFILE_D"); e != nullptr && *e != '\0') {
        d = static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
    }
    const uint32_t C = 16;
    const auto ref = make_reference_inputs_only(2u * C * Bt * kTile, d);
    run_relay(C, ref, 4, 4, /*endpoint_sync=*/true, nullptr, Bt);
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
// How close to the machine does this get, and where does the head dimension
// take it? Causal attention backward is five matmuls of N x N x d, halved by
// the triangle, so 2.5 N^2 d multiply-accumulates. A tile matmul at HiFi4 is
// 64 cycles for 32768 of them, which is 512 per cycle per core, so 110 cores
// at 1.35 GHz peak at about 152 TFLOP/s.
//
// That peak counts only matmul work. This algorithm also has O(N^2)
// elementwise work -- the exponential, the dS chain, the transposes -- which
// costs real time and appears in no FLOP count, so the percentage below is a
// share of a roof this kernel cannot reach, not a share of what is available.
// It is still the right number to watch as d grows, because the matmul work
// per score tile grows with d while the elementwise work does not.
void report_efficiency(uint32_t C, uint32_t grid_w, uint32_t grid_h, uint32_t Bt, uint32_t d) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid_w > grid.x || grid_h > grid.y) {
        GTEST_SKIP() << "needs " << grid_w << "x" << grid_h;
    }
    const uint32_t N = 2u * C * Bt * kTile;
    const auto ref = make_reference_inputs_only(N, d);
    double seconds = 0.0;
    run_relay(C, ref, grid_w, grid_h, /*endpoint_sync=*/true, &seconds, Bt, 1U);

    const double flops = 5.0 * static_cast<double>(N) * static_cast<double>(N) * d;
    const double achieved = flops / seconds / 1e12;
    constexpr double kPeakTflops = 152.0;
    std::cout << "  N=" << N << " d=" << d << " Bt=" << Bt << " on " << C << " cores: "
              << seconds * 1e6 << " us, " << achieved << " TFLOP/s = " << 100.0 * achieved / kPeakTflops
              << "% of matmul peak\n";
}

TEST(CyclicSdpaBwTimingTest, DISABLED_PushTheHeadDimension) {
    // d rises while the block stays short, so the matmul share of each score
    // tile grows and the elementwise share does not.
    report_efficiency(110, 11, 10, /* Bt */ 1, /* d */ 64);
    report_efficiency(110, 11, 10, /* Bt */ 1, /* d */ 128);
    report_efficiency(110, 11, 10, /* Bt */ 1, /* d */ 256);
    report_efficiency(110, 11, 10, /* Bt */ 1, /* d */ 512);
    // and then both levers together, as far as L1 allows.
    report_efficiency(110, 11, 10, /* Bt */ 2, /* d */ 128);
    report_efficiency(110, 11, 10, /* Bt */ 2, /* d */ 256);
    report_efficiency(110, 11, 10, /* Bt */ 4, /* d */ 64);
    report_efficiency(110, 11, 10, /* Bt */ 4, /* d */ 128);
}

TEST(CyclicSdpaBwTimingTest, DISABLED_CompareAcrossHeadDimensions) {
    for (uint32_t d : {64u, 128u, 256u}) {
        compare_with_sdpa_bw(110, 11, 10, /* Bt */ 1, /* groups */ 1, d);
    }
    for (uint32_t d : {64u, 128u, 256u}) {
        compare_with_sdpa_bw(55, 11, 5, /* Bt */ 2, /* groups */ 2, d);
    }
}

// ttnn's chunk-blocked forward with the log-sum-exp output added to it, against
// tt-train's sdpa_fw, whose intermediates carry the same statistic (column 0
// of a (B, H, S, 32) Float32 tile: lse = scale * max + ln sum). Both are also
// graded against a Float32 host reference. Grouped heads, causal and dense.
namespace {
void check_ttnn_sdpa_with_lse(uint32_t heads, uint32_t kv_heads, uint32_t N, uint32_t d, bool causal, uint32_t chunk) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> Q = xt::zeros<float>({1u, heads, N, d});
    xt::xarray<float> K = xt::zeros<float>({1u, kv_heads, N, d});
    xt::xarray<float> V = xt::zeros<float>({1u, kv_heads, N, d});
    for (uint32_t h = 0; h < heads; ++h) {
        xt::view(Q, 0, h, xt::all(), xt::all()) = random_bf16_matrix(N, d, 6000u + h);
    }
    for (uint32_t g = 0; g < kv_heads; ++g) {
        xt::view(K, 0, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 7000u + g);
        xt::view(V, 0, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 8000u + g);
    }
    const auto q = ttml::core::from_xtensor(Q, device);
    const auto k = ttml::core::from_xtensor(K, device);
    const auto v = ttml::core::from_xtensor(V, device);
    const auto mask = causal ? ttml::metal::AttentionMaskType::Causal : ttml::metal::AttentionMaskType::None;

    const auto fw = ttml::metal::sdpa_fw(q, k, v, mask, std::nullopt, 0.0F, /*return_intermediates=*/true);
    const auto ours_O = ttml::core::to_xtensor(fw[0].value());
    const auto ours_lse = ttml::core::to_xtensor(fw[1].value());

    ttnn::operations::transformer::SDPAProgramConfig cfg{
        .compute_with_storage_grid_size = device->compute_with_storage_grid_size(),
        .sub_core_grids = std::nullopt,
        .q_chunk_size = chunk,
        .k_chunk_size = chunk,
        .exp_approx_mode = std::nullopt};
    const auto [theirs, theirs_lse_t] = ttnn::transformer::scaled_dot_product_attention_with_lse(
        q, k, v, causal, std::nullopt, std::nullopt, cfg);
    ASSERT_EQ(theirs_lse_t.logical_shape(), ttnn::Shape({1u, heads, N, 32u}));
    ASSERT_EQ(theirs_lse_t.dtype(), ttnn::DataType::FLOAT32);
    const auto theirs_O = ttml::core::to_xtensor(theirs);
    const auto theirs_lse = ttml::core::to_xtensor(theirs_lse_t);

    // The same call again into preallocated outputs, which must give the same tensors.
    auto pre_O = ttnn::empty_like(theirs);
    auto pre_lse = ttnn::empty_like(theirs_lse_t);
    const auto [again, again_lse] = ttnn::transformer::scaled_dot_product_attention_with_lse(
        q, k, v, causal, std::nullopt, std::nullopt, cfg, std::nullopt, pre_O, pre_lse);
    EXPECT_EQ(again.buffer()->address(), pre_O.buffer()->address()) << "the preallocated output was not used";
    EXPECT_EQ(again_lse.buffer()->address(), pre_lse.buffer()->address()) << "the preallocated lse was not used";
    {
        const auto again_O = ttml::core::to_xtensor(again);
        const auto again_l = ttml::core::to_xtensor(again_lse);
        const float dO = xt::amax(xt::abs(again_O - theirs_O))();
        // Column 0 only: the other 31 columns of an lse tile are whatever the row reduce left there.
        const float dl = xt::amax(xt::abs(
            xt::view(again_l, xt::all(), xt::all(), xt::all(), 0) -
            xt::view(theirs_lse, xt::all(), xt::all(), xt::all(), 0)))();
        std::printf("  preallocated call vs fresh call: max |dO| %.3e (scale %.3e), max |dlse| %.3e\n", dO,
                    xt::amax(xt::abs(theirs_O))(), dl);
        EXPECT_EQ(dO, 0.0F) << "preallocated output differs from the fresh call";
        EXPECT_EQ(dl, 0.0F) << "preallocated lse differs from the fresh call";
    }

    const uint32_t hpg = heads / kv_heads;
    const std::string at = " (heads " + std::to_string(heads) + "/" + std::to_string(kv_heads) + ", N " +
                           std::to_string(N) + ", d " + std::to_string(d) + (causal ? ", causal" : ", dense") + ")";
    for (uint32_t h = 0; h < heads; ++h) {
        const auto r = reference_from_inputs(
            xt::xarray<float>(xt::view(Q, 0, h, xt::all(), xt::all())),
            xt::xarray<float>(xt::view(K, 0, h / hpg, xt::all(), xt::all())),
            xt::xarray<float>(xt::view(V, 0, h / hpg, xt::all(), xt::all())),
            xt::xarray<float>(xt::view(Q, 0, h, xt::all(), xt::all())),
            causal);
        // lse per row, column 0 of the tiles.
        xt::xarray<float> ref_lse = xt::view(r.lse_tile, 0, 0, xt::all(), 0);
        xt::xarray<float> ours_l = xt::view(ours_lse, 0, h, xt::all(), 0);
        xt::xarray<float> theirs_l = xt::view(theirs_lse, 0, h, xt::all(), 0);
        const float ours_lse_err = xt::amax(xt::abs(ours_l - ref_lse))();
        const float theirs_lse_err = xt::amax(xt::abs(theirs_l - ref_lse))();
        const float between = xt::amax(xt::abs(theirs_l - ours_l))();
        const float rms_ours = relative_rms(slice_of(ours_O, 0, h), r.O);
        const float rms_theirs = relative_rms(slice_of(theirs_O, 0, h), r.O);
        if (h == 0) {
            std::printf(
                "  head 0%s: lse max |err| vs reference: sdpa_fw %.2e, ttnn %.2e, between them %.2e; O relative RMS: "
                "sdpa_fw %.2e, ttnn %.2e\n",
                at.c_str(), ours_lse_err, theirs_lse_err, between, rms_ours, rms_theirs);
        }
        // The lse enters the backward as exp(scale s - lse): 1e-2 absolute is a 1% per-row
        // factor. sdpa_fw is inside 1e-2; ttnn's kernel measures up to 2.6e-2, because its running
        // max and rescale factors exp(scale (m_prev - m_cur)) are bf16 and the roundings compound
        // over the K chunks (Float32 statistics CBs broke its kernel, so that stays as it is for
        // now). Graded at 5e-2 here; the ring tests grade the gradients this feeds.
        EXPECT_LT(theirs_lse_err, 5e-2F) << "ttnn lse head " << h << at;
        EXPECT_LT(ours_lse_err, 2e-2F) << "sdpa_fw lse head " << h << at;
        EXPECT_LT(rms_theirs, 0.06F) << "ttnn output head " << h << at;
    }
}
}  // namespace

TEST(TtnnSdpaLseTest, CausalMatchesSdpaFw) {
    check_ttnn_sdpa_with_lse(4, 4, 1024, 64, /* causal */ true, /* chunk */ 128);
}
TEST(TtnnSdpaLseTest, DenseMatchesSdpaFw) {
    check_ttnn_sdpa_with_lse(4, 4, 1024, 64, /* causal */ false, /* chunk */ 128);
}
TEST(TtnnSdpaLseTest, GroupedHeadsCausal) {
    check_ttnn_sdpa_with_lse(4, 2, 2048, 64, /* causal */ true, /* chunk */ 256);
}
TEST(TtnnSdpaLseTest, GroupedHeadsDenseWiderHead) {
    check_ttnn_sdpa_with_lse(4, 2, 1024, 128, /* causal */ false, /* chunk */ 128);
}

// The fused online-softmax merge of a ring step's partial into the running
// accumulators, on one chip (a ring of one, step 0, so the chip runs it),
// against the host's arithmetic.
TEST(RingSoftmaxMergeTest, MergesAPartialIntoTheAccumulators) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t B = 1, H = 2, S = 128, d = 64;
    xt::xarray<float> out = xt::zeros<float>({B, H, S, d});
    xt::xarray<float> step = xt::zeros<float>({B, H, S, d});
    xt::xarray<float> lse = xt::zeros<float>({B, H, S, 32u});
    xt::xarray<float> step_lse = xt::zeros<float>({B, H, S, 32u});
    for (uint32_t h = 0; h < H; ++h) {
        xt::view(out, 0, h, xt::all(), xt::all()) = random_bf16_matrix(S, d, 9000u + h);
        xt::view(step, 0, h, xt::all(), xt::all()) = random_bf16_matrix(S, d, 9100u + h);
        for (uint32_t r = 0; r < S; ++r) {
            lse(0, h, r, 0) = 2.0F + 0.05F * static_cast<float>((r * 7 + h) % 40);
            step_lse(0, h, r, 0) = 1.0F + 0.07F * static_cast<float>((r * 3 + 2 * h) % 50);
            // Garbage in the other columns, as the kernels leave it.
            for (uint32_t c = 1; c < 32; ++c) {
                step_lse(0, h, r, c) = -std::numeric_limits<float>::infinity();
            }
        }
    }
    // Row 5 of head 0 has no contribution yet: the merge must take the step's values.
    lse(0, 0, 5, 0) = -std::numeric_limits<float>::infinity();

    auto out_t = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(out, device);
    auto lse_t = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(lse, device);
    const auto step_t = ttml::core::from_xtensor(step, device);  // bf16
    const auto step_lse_t = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(step_lse, device);
    const auto [new_out_t, new_lse_t] = ttml::metal::ring_softmax_merge(
        out_t, lse_t, step_t, step_lse_t, /* ring_size */ 1, /* axis */ 0, /* step */ 0);
    EXPECT_EQ(new_out_t.buffer()->address(), out_t.buffer()->address()) << "the merge is in place";
    const auto got_out = ttml::core::to_xtensor(new_out_t);
    const auto got_lse = ttml::core::to_xtensor(new_lse_t);

    float worst_out = 0.0F, worst_lse = 0.0F;
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t r = 0; r < S; ++r) {
            const float a = lse(0, h, r, 0), b = step_lse(0, h, r, 0);
            const float m = std::max(a, b);
            const float ea = std::exp(a - m), eb = std::exp(b - m), sum = ea + eb;
            const float want_lse = m + std::log(sum);
            worst_lse = std::max(worst_lse, std::abs(got_lse(0, h, r, 0) - want_lse));
            for (uint32_t c = 0; c < d; ++c) {
                const float want = (ea * out(0, h, r, c) + eb * step(0, h, r, c)) / sum;
                worst_out = std::max(worst_out, std::abs(got_out(0, h, r, c) - want));
            }
        }
    }
    std::printf("  merge: max |dO| %.3e, max |dlse| %.3e\n", worst_out, worst_lse);
    if (std::getenv("RING_MERGE_DEBUG") != nullptr) {
        for (uint32_t r = 0; r < 3; ++r) {
            std::printf(
                "    row %u: a %.4f b %.4f | got lse cols 0,1,5,31: %.4f %.4f %.4f %.4f | got O[0] %.4f (O %.4f step %.4f)\n",
                r, lse(0, 0, r, 0), step_lse(0, 0, r, 0), got_lse(0, 0, r, 0), got_lse(0, 0, r, 1), got_lse(0, 0, r, 5),
                got_lse(0, 0, r, 31), got_out(0, 0, r, 0), out(0, 0, r, 0), step(0, 0, r, 0));
        }
    }
    // The SFPU's exp and log in Float32 mode land within ~2e-3 on the lse, as the
    // ttnn elementwise chain this replaces did; O is well inside 1e-3.
    EXPECT_LT(worst_out, 2e-3F);
    EXPECT_LT(worst_lse, 5e-3F);
}

// The two forwards on one chip: tt-train's sdpa_fw (one query tile row per
// core pass, K and V re-read per row; the ring's step forward today) against
// ttnn's chunk-blocked flash-attention forward at several chunk sizes. The
// ring's launch shapes, causal (the diagonal step) and dense (the others).
// Useful FLOPs: two matmuls of 2 N^2 d per head, halved by the triangle.
// Prints the max difference of the outputs too, as a coarse sanity check.
TEST(CyclicSdpaBwTimingTest, DISABLED_CompareForwardsWithTtnn) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    struct Shape {
        uint32_t heads, kv_heads, N, d;
    };
    // TTML_CYCLIC_FW_COMPARE_SHAPES="heads:kv:N:d,..." replaces the shapes (few heads, long sequences...).
    std::vector<Shape> shapes{Shape{4, 4, 4096, 64}, Shape{20, 10, 5632, 64}, Shape{32, 8, 5632, 128}};
    if (const char* env = std::getenv("TTML_CYCLIC_FW_COMPARE_SHAPES"); env != nullptr && *env != '\0') {
        shapes.clear();
        std::string spec(env);
        for (size_t pos = 0; pos < spec.size();) {
            const size_t end = spec.find(',', pos);
            Shape sh{};
            std::sscanf(spec.substr(pos, end - pos).c_str(), "%u:%u:%u:%u", &sh.heads, &sh.kv_heads, &sh.N, &sh.d);
            shapes.push_back(sh);
            pos = end == std::string::npos ? spec.size() : end + 1;
        }
    }
    for (const auto sh : shapes) {
        for (const bool causal : {true, false}) {
            xt::xarray<float> Q = xt::zeros<float>({1u, sh.heads, sh.N, sh.d});
            xt::xarray<float> K = xt::zeros<float>({1u, sh.kv_heads, sh.N, sh.d});
            xt::xarray<float> V = xt::zeros<float>({1u, sh.kv_heads, sh.N, sh.d});
            for (uint32_t h = 0; h < sh.heads; ++h) {
                xt::view(Q, 0, h, xt::all(), xt::all()) = random_bf16_matrix(sh.N, sh.d, 3000u + h);
            }
            for (uint32_t g = 0; g < sh.kv_heads; ++g) {
                xt::view(K, 0, g, xt::all(), xt::all()) = random_bf16_matrix(sh.N, sh.d, 4000u + g);
                xt::view(V, 0, g, xt::all(), xt::all()) = random_bf16_matrix(sh.N, sh.d, 5000u + g);
            }
            const auto q = ttml::core::from_xtensor(Q, device);
            const auto k = ttml::core::from_xtensor(K, device);
            const auto v = ttml::core::from_xtensor(V, device);
            const auto mask = causal ? ttml::metal::AttentionMaskType::Causal : ttml::metal::AttentionMaskType::None;

            const auto time_it = [&](const auto& call) {
                call();  // warm: program cache and kernel build
                std::vector<double> samples;
                for (uint32_t r = 0; r < 5; ++r) {
                    const auto start = std::chrono::steady_clock::now();
                    call();
                    samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                }
                std::sort(samples.begin(), samples.end());
                return samples[samples.size() / 2];
            };
            const double flop = (causal ? 2.0 : 4.0) * static_cast<double>(sh.N) * sh.N * sh.d * sh.heads;
            const auto report = [&](const char* name, double seconds) {
                std::printf(
                    "  %s: %.2f ms, %.1f TFLOP/s (%.1f%% of the 110-core LoFi peak)\n", name, seconds * 1e3,
                    flop / seconds / 1e12, 100.0 * flop / seconds / 1e12 / 594.0);
            };
            std::printf(
                "heads=%u kv_heads=%u N=%u d=%u %s\n", sh.heads, sh.kv_heads, sh.N, sh.d, causal ? "causal" : "dense");

            ttnn::Tensor ours;
            const double ours_s = time_it([&]() {
                const auto fw = ttml::metal::sdpa_fw(q, k, v, mask, std::nullopt, 0.0F, /*return_intermediates=*/true);
                ours = fw[0].value();
                distributed::Finish(device->mesh_command_queue());
            });
            report("tt-train sdpa_fw (per tile row, with lse)", ours_s);
            const auto ours_xt = ttml::core::to_xtensor(ours);
            // A Float32 host reference at the small shape, so the two outputs
            // are graded against the truth and not only against each other.
            std::optional<xt::xarray<float>> ref_O;
            if (sh.N <= 4096u) {
                xt::xarray<float> O = xt::zeros<float>({1u, sh.heads, sh.N, sh.d});
                const uint32_t hpg = sh.heads / sh.kv_heads;
                for (uint32_t h = 0; h < sh.heads; ++h) {
                    const auto r = reference_from_inputs(
                        xt::xarray<float>(xt::view(Q, 0, h, xt::all(), xt::all())),
                        xt::xarray<float>(xt::view(K, 0, h / hpg, xt::all(), xt::all())),
                        xt::xarray<float>(xt::view(V, 0, h / hpg, xt::all(), xt::all())),
                        xt::xarray<float>(xt::view(Q, 0, h, xt::all(), xt::all())),  // dO unused here
                        causal);
                    xt::view(O, 0, h, xt::all(), xt::all()) = r.O;
                }
                ref_O = O;
                std::printf("    tt-train relative RMS vs Float32 reference: %.2e\n", relative_rms(ours_xt, O));
            }

            // The forward on the cyclic schedule, at every block height whose
            // schedule fits the grid (C = N / (2 Bt 32) cores per slice).
            for (const uint32_t Bt : {1u, 2u, 4u}) {
                ttnn::Tensor cyc_O;
                double cyc_s = 0.0;
                try {
                    cyc_s = time_it([&]() {
                        auto [o, l] = ttml::metal::cyclic_sdpa_fw(q, k, v, Bt, mask);
                        cyc_O = o;
                        distributed::Finish(device->mesh_command_queue());
                    });
                } catch (const std::exception& e) {
                    std::printf("  cyclic_sdpa_fw Bt %u: does not fit (%.80s)\n", Bt, e.what());
                    continue;
                }
                const std::string name = "cyclic_sdpa_fw (relay, Bt " + std::to_string(Bt) + ")";
                report(name.c_str(), cyc_s);
                const auto cyc_xt = ttml::core::to_xtensor(cyc_O);
                std::printf("    vs tt-train: max |dO| %.3e (scale %.3e), speed-up %.2fx\n",
                            xt::amax(xt::abs(cyc_xt - ours_xt))(), xt::amax(xt::abs(ours_xt))(), ours_s / cyc_s);
                if (ref_O.has_value()) {
                    std::printf("    cyclic relative RMS vs Float32 reference: %.2e\n", relative_rms(cyc_xt, *ref_O));
                }
            }

            // Two settings of ttnn's kernel: its defaults, and the precise
            // ones -- exact exponential, HiFi4 matmuls, Float32 accumulation
            // -- since the defaults trade accuracy for speed.
            for (const uint32_t chunk : {128u, 256u, 512u}) {
                if (sh.N % chunk != 0u) {
                    continue;
                }
                // The variant the ring would call: Float32 accumulation and the lse output.
                ttnn::operations::transformer::SDPAProgramConfig cfg{
                    .compute_with_storage_grid_size = grid,
                    .sub_core_grids = std::nullopt,
                    .q_chunk_size = chunk,
                    .k_chunk_size = chunk,
                    .exp_approx_mode = std::nullopt};
                ttnn::Tensor with_lse_O;
                double with_lse_s = 0.0;
                try {
                    with_lse_s = time_it([&]() {
                        auto [o, l] = ttnn::transformer::scaled_dot_product_attention_with_lse(
                            q, k, v, causal, std::nullopt, std::nullopt, cfg);
                        with_lse_O = o;
                        distributed::Finish(device->mesh_command_queue());
                    });
                } catch (const std::exception& e) {
                    std::printf("  ttnn sdpa with lse, chunk %u: does not fit (%.80s)\n", chunk, e.what());
                    continue;
                }
                const std::string name = "ttnn sdpa with lse, fp32 acc, chunk " + std::to_string(chunk);
                report(name.c_str(), with_lse_s);
                if (ref_O.has_value()) {
                    std::printf(
                        "    ttnn with lse relative RMS vs Float32 reference: %.2e (%.2fx faster than sdpa_fw)\n",
                        relative_rms(ttml::core::to_xtensor(with_lse_O), *ref_O), ours_s / with_lse_s);
                } else {
                    std::printf("    (%.2fx faster than sdpa_fw)\n", ours_s / with_lse_s);
                }
            }
            for (const bool precise : {false, true}) {
            for (const uint32_t chunk : {128u, 256u, 512u}) {
                if (sh.N % chunk != 0u) {
                    continue;
                }
                ttnn::operations::transformer::SDPAProgramConfig cfg{
                    .compute_with_storage_grid_size = grid,
                    .sub_core_grids = std::nullopt,
                    .q_chunk_size = chunk,
                    .k_chunk_size = chunk,
                    .exp_approx_mode = precise ? std::optional<bool>(false) : std::nullopt};
                const std::optional<ttnn::DeviceComputeKernelConfig> kernel_cfg =
                    precise ? std::optional<ttnn::DeviceComputeKernelConfig>(ttml::core::ComputeKernelConfig::precise())
                            : std::nullopt;
                ttnn::Tensor theirs;
                double theirs_s = 0.0;
                try {
                    theirs_s = time_it([&]() {
                        theirs = ttnn::transformer::scaled_dot_product_attention(
                            q, k, v, std::nullopt, causal, std::nullopt, std::nullopt, std::nullopt, cfg, kernel_cfg);
                        distributed::Finish(device->mesh_command_queue());
                    });
                } catch (const std::exception& e) {
                    std::printf("  ttnn sdpa, chunk %u%s: does not fit (%.80s)\n", chunk, precise ? " precise" : "", e.what());
                    continue;
                }
                const std::string name =
                    "ttnn sdpa, chunk " + std::to_string(chunk) + (precise ? " precise" : " default") + " (no lse)";
                report(name.c_str(), theirs_s);
                const auto theirs_xt = ttml::core::to_xtensor(theirs);
                const float diff = xt::amax(xt::abs(theirs_xt - ours_xt))();
                const float scale = xt::amax(xt::abs(ours_xt))();
                std::printf("    max |ttnn - ours| %.3e on scale %.3e (%.2fx faster)\n", diff, scale, ours_s / theirs_s);
                if (ref_O.has_value()) {
                    std::printf("    ttnn relative RMS vs Float32 reference: %.2e\n", relative_rms(theirs_xt, *ref_O));
                }
            }
            }
        }
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

// The kernel's own clock, inputs only, at the shapes the kernel work is
// judged on: a 4x4 group at each block height (the profile's shape), the
// whole 11x10 grid at each block height (the schedule's cap), two 55-core
// groups, and d = 128. Endpoint variant, median of five enqueues.
// The ring's own launches on one chip, through the op: four heads, a
// 4096-row chunk pair on the zigzag layout (two 2048-row chunks), Bt = 4.
// The diagonal step is the two causal triangles in one launch, the dense
// step the block (1, 0). Median of five enqueues; with
// TT_METAL_DEVICE_PROFILER=1 the zones say where a launch's time goes.
// RING_LAUNCH_ROWS overrides the rows per chip, RING_LAUNCH_HEADS the heads.
TEST(CyclicSdpaBwTimingTest, DISABLED_ProfileRingLaunch) {
    using namespace ttml::metal;
    auto* device = &ttml::autograd::ctx().get_device();
    uint32_t rows = 4096;
    uint32_t heads = 4;
    if (const char* e = std::getenv("RING_LAUNCH_ROWS"); e != nullptr && *e != '\0') {
        rows = static_cast<uint32_t>(std::atoi(e));
    }
    if (const char* e = std::getenv("RING_LAUNCH_HEADS"); e != nullptr && *e != '\0') {
        heads = static_cast<uint32_t>(std::atoi(e));
    }
    uint32_t Bt = 4;
    if (const char* e = std::getenv("RING_LAUNCH_BT"); e != nullptr && *e != '\0') {
        Bt = static_cast<uint32_t>(std::atoi(e));
    }
    // RING_LAUNCH_KIND=diagonal or dense runs only that launch (for a profile).
    // RING_LAUNCH_DQ_TRANSPOSED=1 runs the launch as the ring does inside its
    // step loop: dQ in the kernels' tile-transposed form on both sides.
    const bool dq_transposed = std::getenv("RING_LAUNCH_DQ_TRANSPOSED") != nullptr;
    const char* kind = std::getenv("RING_LAUNCH_KIND");
    const std::string only = (kind != nullptr) ? kind : "";
    const uint32_t d = 64;
    const auto ref = make_reference_inputs_only(rows, d);
    const auto q = ttml::core::from_xtensor(as_4d_repeated(ref.Q, heads), device);
    const auto k = ttml::core::from_xtensor(as_4d_repeated(ref.K, heads), device);
    const auto v = ttml::core::from_xtensor(as_4d_repeated(ref.V, heads), device);
    const auto dO = ttml::core::from_xtensor(as_4d_repeated(ref.dO, heads), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(repeat_4d(ref.lse_tile, heads), device);
    const auto u = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(repeat_4d(ref.u_tile, heads), device);
    const xt::xarray<float> zeros = xt::zeros<float>({1u, heads, rows, d});
    auto dq = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);
    auto dk = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);
    auto dv = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros, device);
    const auto time = [&](const char* name, AttentionMaskType mask, std::vector<uint32_t> rc, std::vector<uint32_t> cc) {
        const auto once = [&]() {
            cyclic_sdpa_bw(
                q, k, v, dO, lse, u, Bt, /* use_barrier */ false, mask, /* accumulate */ true, dq, dk, dv,
                /* max_groups */ 0, /* sequence_chunks */ 2, rc, cc,
                /* grad_query_in_tile_transposed */ dq_transposed, /* grad_query_out_tile_transposed */ dq_transposed);
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        };
        once();  // warm: compile
        std::vector<double> samples;
        for (int i = 0; i < 5; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            once();
            samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6);
        }
        std::sort(samples.begin(), samples.end());
        std::cout << "  " << name << " heads=" << heads << " rows=" << rows << " Bt=" << Bt << ": " << samples[2]
                  << " us (min " << samples[0] << ")\n";
    };
    if (only.empty() || only == "diagonal") {
        time("diagonal step (0,0),(1,1) causal", AttentionMaskType::Causal, {0u, 1u}, {0u, 1u});
    }
    if (only.empty() || only == "dense") {
        time("dense step (1,0)", AttentionMaskType::None, {1u}, {0u});
    }
    if (std::getenv("TT_METAL_DEVICE_PROFILER") != nullptr) {
        // The device profiler flushes its zones on close.
        ttml::autograd::ctx().close_device();
    }
}

TEST(CyclicSdpaBwTimingTest, DISABLED_BenchRelay) {
    struct Shape {
        uint32_t C, w, h, Bt, d, groups;
    };
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    std::cout << "cyclic_sdpa_bw relay, endpoint variant, us (median of 5)\n";
    // The last three are the ring's own launch shape at 4096 rows per chip
    // on the zigzag layout: 2048-row chunks, Bt = 4, C = 8 -- alone, and
    // eight such slices side by side as four heads x two chunk pairs run.
    for (const auto s : {Shape{16, 4, 4, 1, 64, 1}, Shape{16, 4, 4, 2, 64, 1}, Shape{16, 4, 4, 4, 64, 1},
                         Shape{16, 4, 4, 4, 128, 1}, Shape{55, 11, 5, 2, 64, 2}, Shape{55, 11, 5, 4, 64, 2},
                         Shape{110, 11, 10, 1, 64, 1}, Shape{110, 11, 10, 2, 64, 1}, Shape{110, 11, 10, 4, 64, 1},
                         Shape{8, 2, 4, 4, 64, 1}, Shape{8, 2, 4, 4, 64, 8}, Shape{8, 2, 4, 2, 64, 8}}) {
        if (s.w > grid.x || s.h > grid.y) {
            continue;
        }
        const uint32_t N = 2u * s.C * s.Bt * kTile;
        const auto ref = make_reference_inputs_only(N, s.d);
        double seconds = 0.0;
        run_relay(s.C, ref, s.w, s.h, /* endpoint_sync */ true, &seconds, s.Bt, s.groups);
        // Work and traffic, for utilisation: five matmuls of 2 B^2 d flops per
        // causal block pair (T = 2C row blocks of B = 32 Bt rows), per group;
        // DRAM traffic at least the operands once (Q, K, V, dO bf16; L, D
        // fp32) and the three fp32 gradients read and written once -- the
        // relay's spills and reloads at streak ends come on top.
        const double B = 32.0 * s.Bt;
        const double T = 2.0 * s.C;
        const double flops = s.groups * (T * (T + 1.0) / 2.0) * 10.0 * B * B * s.d;
        const double bytes = s.groups * (4.0 * N * s.d * 2.0 + 2.0 * N * 4.0 + 2.0 * 3.0 * N * s.d * 4.0);
        const double cores = static_cast<double>(s.w) * s.h * s.groups;
        std::cout << "  C=" << s.C << " Bt=" << s.Bt << " d=" << s.d << " N=" << N << " x" << s.groups
                  << " groups on " << s.w << "x" << s.h << ": " << seconds * 1e6 << " us"
                  << "  [" << flops / seconds / 1e12 << " TFLOP/s, " << flops / seconds / 1e12 / cores
                  << " per core; >= " << bytes / seconds / 1e9 << " GB/s DRAM]\n";
    }
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

// ------------------------------------------------------------ the forward
// The forward pass on the cyclic schedule (cyclic_sdpa_fw, tt-flash-attn's
// Algorithm 8), against the Float32 host reference and against sdpa_fw, whose
// intermediates carry the same statistic (lse in column 0 of a (B, H, S, 32)
// Float32 tile). Grouped heads, causal and dense, every block height.
namespace {
void check_cyclic_forward(
    uint32_t batch, uint32_t heads, uint32_t kv_heads, uint32_t N, uint32_t d, bool causal, uint32_t Bt,
    uint32_t max_groups = 0) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();
    xt::xarray<float> Q = xt::zeros<float>({batch, heads, N, d});
    xt::xarray<float> K = xt::zeros<float>({batch, kv_heads, N, d});
    xt::xarray<float> V = xt::zeros<float>({batch, kv_heads, N, d});
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            xt::view(Q, b, h, xt::all(), xt::all()) = random_bf16_matrix(N, d, 9000u + 37u * b + h);
        }
        for (uint32_t g = 0; g < kv_heads; ++g) {
            xt::view(K, b, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 9500u + 37u * b + g);
            xt::view(V, b, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 9700u + 37u * b + g);
        }
    }
    const auto q = ttml::core::from_xtensor(Q, device);
    const auto k = ttml::core::from_xtensor(K, device);
    const auto v = ttml::core::from_xtensor(V, device);
    const auto mask = causal ? ttml::metal::AttentionMaskType::Causal : ttml::metal::AttentionMaskType::None;

    // TTML_CYCLIC_FW_FAST=1 checks the fast (bf16-register) variant instead.
    const bool fast = std::getenv("TTML_CYCLIC_FW_FAST") != nullptr;
    const auto [out_t, lse_t] = fast ? ttml::metal::cyclic_sdpa_fw_fast(q, k, v, Bt, mask, std::nullopt, std::nullopt, max_groups)
                                     : ttml::metal::cyclic_sdpa_fw(q, k, v, Bt, mask, std::nullopt, std::nullopt, max_groups);
    ASSERT_EQ(out_t.logical_shape(), ttnn::Shape({batch, heads, N, d}));
    ASSERT_EQ(lse_t.logical_shape(), ttnn::Shape({batch, heads, N, 32u}));
    ASSERT_EQ(out_t.dtype(), ttnn::DataType::BFLOAT16);
    ASSERT_EQ(lse_t.dtype(), ttnn::DataType::FLOAT32);
    const auto ours_O = ttml::core::to_xtensor(out_t);
    const auto ours_lse = ttml::core::to_xtensor(lse_t);

    const auto fw = ttml::metal::sdpa_fw(q, k, v, mask, std::nullopt, 0.0F, /*return_intermediates=*/true);
    const auto base_O = ttml::core::to_xtensor(fw[0].value());
    const auto base_lse = ttml::core::to_xtensor(fw[1].value());

    const uint32_t hpg = heads / kv_heads;
    const std::string at = " (batch " + std::to_string(batch) + ", heads " + std::to_string(heads) + "/" +
                           std::to_string(kv_heads) + ", N " + std::to_string(N) + ", d " + std::to_string(d) +
                           (causal ? ", causal" : ", dense") + ", Bt " + std::to_string(Bt) + ")";
    float worst_rms = 0.0F;
    float worst_lse = 0.0F;
    float worst_rms_base = 0.0F;
    float worst_lse_base = 0.0F;
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t h = 0; h < heads; ++h) {
            const auto r = reference_from_inputs(
                xt::xarray<float>(xt::view(Q, b, h, xt::all(), xt::all())),
                xt::xarray<float>(xt::view(K, b, h / hpg, xt::all(), xt::all())),
                xt::xarray<float>(xt::view(V, b, h / hpg, xt::all(), xt::all())),
                xt::xarray<float>(xt::view(Q, b, h, xt::all(), xt::all())),
                causal);
            xt::xarray<float> ref_lse = xt::view(r.lse_tile, 0, 0, xt::all(), 0);
            xt::xarray<float> ours_l = xt::view(ours_lse, b, h, xt::all(), 0);
            xt::xarray<float> base_l = xt::view(base_lse, b, h, xt::all(), 0);
            const float lse_err = xt::amax(xt::abs(ours_l - ref_lse))();
            const float lse_err_base = xt::amax(xt::abs(base_l - ref_lse))();
            const float rms = relative_rms(slice_of(ours_O, b, h), r.O);
            const float rms_base = relative_rms(slice_of(base_O, b, h), r.O);
            worst_rms = std::max(worst_rms, rms);
            worst_lse = std::max(worst_lse, lse_err);
            worst_rms_base = std::max(worst_rms_base, rms_base);
            worst_lse_base = std::max(worst_lse_base, lse_err_base);
            EXPECT_TRUE(std::isfinite(rms)) << "head " << h << at;
            if (std::getenv("TTML_CYCLIC_FW_DEBUG") != nullptr && b == 0 && h == 0) {
                // Relative RMS of O per 32-row block, to see where an error sits.
                std::printf("    O relative RMS per row block:");
                for (uint32_t r0 = 0; r0 < N; r0 += 32) {
                    const xt::xarray<float> got = xt::view(ours_O, b, h, xt::range(r0, r0 + 32), xt::all());
                    const xt::xarray<float> want = xt::view(r.O, xt::range(r0, r0 + 32), xt::all());
                    std::printf(" %.1e", relative_rms(got, want));
                }
                std::printf("\n");
                for (uint32_t i = 0; i < N; i += (N >= 256 ? N / 8 : 8)) {
                    std::printf(
                        "    row %3u: lse ref %9.4f ours %9.4f base %9.4f | O[0..3] ref %8.4f %8.4f %8.4f %8.4f ours %8.4f "
                        "%8.4f %8.4f %8.4f\n",
                        i, ref_lse(i), ours_l(i), base_l(i), r.O(i, 0), r.O(i, 1), r.O(i, 2), r.O(i, 3),
                        ours_O(b, h, i, 0), ours_O(b, h, i, 1), ours_O(b, h, i, 2), ours_O(b, h, i, 3));
                }
                for (uint32_t i : {0u, 1u, 31u, 32u, 33u, 63u}) {
                    if (i < N) {
                        std::printf("    row %3u: lse ref %9.4f ours %9.4f | O ref %8.4f ours %8.4f\n", i, ref_lse(i),
                                    ours_l(i), r.O(i, 0), ours_O(b, h, i, 0));
                    }
                }
            }
            // The other 31 columns of an lse tile must be exact zeros (the
            // backward reads column 0; the merge broadcasts it).
            const float rest = xt::amax(xt::abs(xt::view(ours_lse, b, h, xt::all(), xt::range(1, 32))))();
            EXPECT_EQ(rest, 0.0F) << "lse tile columns 1..31 are not zero, head " << h << at;
        }
    }
    std::printf(
        "  cyclic_sdpa_fw%s: O relative RMS %.2e (sdpa_fw %.2e), lse max |err| %.2e (sdpa_fw %.2e)\n",
        at.c_str(), worst_rms, worst_rms_base, worst_lse, worst_lse_base);
    // bf16 output of a Float32 accumulator: the rounding of the output alone
    // is 2e-3 RMS; sdpa_fw measures 5e-3 to 1e-2 at these shapes.
    EXPECT_LT(worst_rms, 1.5e-2F) << at;
    // The fast variant's bf16 registers put its lse where ttnn's kernel is (a
    // few 1e-2); the exact kernel's is under 1e-3.
    EXPECT_LT(worst_lse, fast ? 1e-1F : 1e-2F) << at;
}
}  // namespace

// The fast variant at a block height only its 16 registers hold; skipped
// unless TTML_CYCLIC_FW_FAST=1 selects that variant.
TEST(CyclicSdpaFwTest, FastTallBlocks) {
    if (std::getenv("TTML_CYCLIC_FW_FAST") == nullptr) {
        GTEST_SKIP() << "the fast variant only (TTML_CYCLIC_FW_FAST=1)";
    }
    check_cyclic_forward(1, 1, 1, /* N */ 2048, /* d */ 64, /* causal */ true, /* Bt */ 8);
    check_cyclic_forward(1, 2, 1, /* N */ 4096, /* d */ 64, /* causal */ true, /* Bt */ 8);
}

TEST(CyclicSdpaFwTest, OneCoreCausal) {
    check_cyclic_forward(1, 1, 1, /* N */ 64, /* d */ 64, /* causal */ true, /* Bt */ 1);
}

TEST(CyclicSdpaFwTest, TwoCoresCausal) {
    check_cyclic_forward(1, 1, 1, 128, 64, true, 1);
}

TEST(CyclicSdpaFwTest, FourCoresCausalBlockHeights) {
    for (uint32_t Bt : {1u, 2u, 4u}) {
        check_cyclic_forward(1, 1, 1, 8u * Bt * 32u, 64, true, Bt);
    }
}

TEST(CyclicSdpaFwTest, FourCoresDense) {
    for (uint32_t Bt : {1u, 2u}) {
        check_cyclic_forward(1, 1, 1, 8u * Bt * 32u, 64, false, Bt);
    }
}

TEST(CyclicSdpaFwTest, HeadDimension128) {
    check_cyclic_forward(1, 1, 1, 256, 128, true, 1);
    check_cyclic_forward(1, 1, 1, 512, 128, true, 2);
}

TEST(CyclicSdpaFwTest, ManyHeadsAndGroups) {
    // Two batches, four heads: eight slices dealt to the groups, and in turn
    // when capped to three groups.
    check_cyclic_forward(2, 4, 4, 128, 64, true, 1);
    check_cyclic_forward(2, 4, 4, 128, 64, true, 1, /* max_groups */ 3);
}

TEST(CyclicSdpaFwTest, GroupedQueryHeads) {
    check_cyclic_forward(1, 4, 2, 128, 64, true, 1);
    check_cyclic_forward(1, 6, 2, 256, 64, false, 2);
}

TEST(CyclicSdpaFwTest, SixteenCores) {
    check_cyclic_forward(1, 1, 1, 1024, 64, true, 1);
    check_cyclic_forward(1, 1, 1, 2048, 64, true, 2);
}


// The cyclic forward alone, timed at one ring launch shape (20/10 heads, 5632
// rows, d 64, causal, Bt 4 by default): the quick loop for kernel work.
// TTML_CYCLIC_FW_TIME="heads:kv:N:d:Bt:causal" overrides the shape.
TEST(CyclicSdpaFwTimingTest, DISABLED_TimeTheForward) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();
    uint32_t heads = 20, kv_heads = 10, N = 5632, d = 64, Bt = 4, causal_u = 1;
    if (const char* env = std::getenv("TTML_CYCLIC_FW_TIME"); env != nullptr && *env != '\0') {
        std::sscanf(env, "%u:%u:%u:%u:%u:%u", &heads, &kv_heads, &N, &d, &Bt, &causal_u);
    }
    const bool causal = causal_u != 0;
    xt::xarray<float> Q = xt::zeros<float>({1u, heads, N, d});
    xt::xarray<float> K = xt::zeros<float>({1u, kv_heads, N, d});
    xt::xarray<float> V = xt::zeros<float>({1u, kv_heads, N, d});
    for (uint32_t h = 0; h < heads; ++h) {
        xt::view(Q, 0, h, xt::all(), xt::all()) = random_bf16_matrix(N, d, 3000u + h);
    }
    for (uint32_t g = 0; g < kv_heads; ++g) {
        xt::view(K, 0, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 4000u + g);
        xt::view(V, 0, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 5000u + g);
    }
    const auto q = ttml::core::from_xtensor(Q, device);
    const auto k = ttml::core::from_xtensor(K, device);
    const auto v = ttml::core::from_xtensor(V, device);
    const auto mask = causal ? ttml::metal::AttentionMaskType::Causal : ttml::metal::AttentionMaskType::None;
    const auto time_it = [&](const auto& call) {
        call();
        std::vector<double> samples;
        for (uint32_t r = 0; r < 5; ++r) {
            const auto start = std::chrono::steady_clock::now();
            call();
            samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    };
    const double flop = (causal ? 2.0 : 4.0) * static_cast<double>(N) * N * d * heads;
    const bool fast = std::getenv("TTML_CYCLIC_FW_FAST") != nullptr;
    const double cyc_s = time_it([&]() {
        auto [o, l] = fast ? ttml::metal::cyclic_sdpa_fw_fast(q, k, v, Bt, mask) : ttml::metal::cyclic_sdpa_fw(q, k, v, Bt, mask);
        distributed::Finish(device->mesh_command_queue());
    });
    const char* experiment = std::getenv("TTML_CYCLIC_FW_EXPERIMENT");
    std::printf(
        "  cyclic_sdpa_fw heads %u/%u N %u d %u %s Bt %u%s%s: %.2f ms, %.1f TFLOP/s\n", heads, kv_heads, N, d,
        causal ? "causal" : "dense", Bt, experiment ? " experiment " : "", experiment ? experiment : "", cyc_s * 1e3,
        flop / cyc_s / 1e12);
}

// The one-chip backward against tt-train's own, both through their op entry
// points with the same tensors: sdpa_bw given sdpa_fw's output and
// intermediates, cyclic_sdpa_bw_from_forward given the cyclic forward's (so
// the cyclic side pays for forming D = rowsum(dO . O), as it does in
// training). Warm, median of seven, blocking on the queue after each call.
// The block height is the ring planner's for the whole sequence. Before
// timing, the two sides' gradients are compared, so the numbers are of the
// same computation. TTML_BW_COMPARE_SHAPES="heads:kv:N:d:Bt,..." replaces
// the table.
TEST(CyclicSdpaBwTimingTest, DISABLED_CompareBackwardWithTtTrain) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();
    // heads, key heads, rows, d, block height (0: the planner's). One head
    // fills the grid only at the block height whose schedule has 110 cores;
    // the planner prefers the tallest block, which is right when many heads
    // share the grid (the model shapes) and not for a single head.
    std::vector<std::array<uint32_t, 5>> shapes = {
        {1, 1, 7040, 64, 1},
        {1, 1, 14080, 64, 2},
        {1, 1, 28160, 64, 4},
        {20, 10, 5632, 64, 0},
        {32, 8, 5632, 128, 0},
    };
    if (const char* spec = std::getenv("TTML_BW_COMPARE_SHAPES"); spec != nullptr && *spec != '\0') {
        shapes.clear();
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::array<uint32_t, 5> sh{};
            std::sscanf(item.c_str(), "%u:%u:%u:%u:%u", &sh[0], &sh[1], &sh[2], &sh[3], &sh[4]);
            shapes.push_back(sh);
        }
    }
    const auto time_it = [&](const auto& call) {
        call();  // warm: kernel build and program cache
        std::vector<double> samples;
        for (uint32_t r = 0; r < 7; ++r) {
            const auto start = std::chrono::steady_clock::now();
            call();
            samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        std::sort(samples.begin(), samples.end());
        return std::array<double, 3>{samples.front(), samples[samples.size() / 2], samples.back()};
    };
    for (const auto& [heads, kv_heads, N, d, Bt_in] : shapes) {
        xt::xarray<float> Q = xt::zeros<float>({1u, heads, N, d});
        xt::xarray<float> dO = xt::zeros<float>({1u, heads, N, d});
        xt::xarray<float> K = xt::zeros<float>({1u, kv_heads, N, d});
        xt::xarray<float> V = xt::zeros<float>({1u, kv_heads, N, d});
        for (uint32_t h = 0; h < heads; ++h) {
            xt::view(Q, 0, h, xt::all(), xt::all()) = random_bf16_matrix(N, d, 3000u + h);
            xt::view(dO, 0, h, xt::all(), xt::all()) = random_bf16_matrix(N, d, 6000u + h);
        }
        for (uint32_t g = 0; g < kv_heads; ++g) {
            xt::view(K, 0, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 4000u + g);
            xt::view(V, 0, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 5000u + g);
        }
        const auto q = ttml::core::from_xtensor(Q, device);
        const auto k = ttml::core::from_xtensor(K, device);
        const auto v = ttml::core::from_xtensor(V, device);
        const auto grad_output = ttml::core::from_xtensor(dO, device);
        const uint32_t Bt = Bt_in != 0U ? Bt_in
                                        : ttml::ops::distributed::plan_rows_per_block_tiles(
                                              q, ttml::metal::ops::RingLayout::Contiguous);

        const auto theirs_fw = ttml::metal::sdpa_fw(
            q, k, v, ttml::metal::AttentionMaskType::Causal, std::nullopt, 0.0F, /*return_intermediates=*/true);
        const auto theirs_o = theirs_fw[0].value();
        const auto theirs_stats = theirs_fw[1].value();
        const auto [ours_o, ours_lse] = ttml::metal::cyclic_sdpa_fw(q, k, v, Bt);

        const auto theirs = [&]() {
            return ttml::metal::sdpa_bw(
                grad_output, theirs_o, q, k, v, theirs_stats, ttml::metal::AttentionMaskType::Causal, std::nullopt,
                0.0F);
        };
        const auto ours = [&]() { return ttml::metal::cyclic_sdpa_bw_from_forward(q, k, v, grad_output, ours_o, ours_lse, Bt); };

        // Same computation: each gradient of the cyclic side against tt-train's
        // and, for one head with TTML_BW_COMPARE_REFERENCE set, both against a
        // host Float32 backward on the same inputs.
        {
            const auto [tq, tk, tv] = theirs();
            const auto [oq, ok, ov] = ours();
            if (heads == 1U && std::getenv("TTML_BW_COMPARE_REFERENCE") != nullptr) {
                const xt::xarray<float> q2 = xt::view(Q, 0, 0, xt::all(), xt::all());
                const xt::xarray<float> k2 = xt::view(K, 0, 0, xt::all(), xt::all());
                const xt::xarray<float> v2 = xt::view(V, 0, 0, xt::all(), xt::all());
                const xt::xarray<float> do2 = xt::view(dO, 0, 0, xt::all(), xt::all());
                const float scale = 1.0F / std::sqrt(static_cast<float>(d));
                xt::xarray<float> P = xt::linalg::dot(q2, xt::transpose(k2)) * scale;
                for (uint32_t i = 0; i < N; ++i) {
                    float m = -std::numeric_limits<float>::infinity();
                    for (uint32_t j = 0; j <= i; ++j) m = std::max(m, P(i, j));
                    double sum = 0.0;
                    for (uint32_t j = 0; j <= i; ++j) sum += std::exp(static_cast<double>(P(i, j) - m));
                    const float lse = m + static_cast<float>(std::log(sum));
                    for (uint32_t j = 0; j < N; ++j) P(i, j) = j <= i ? std::exp(P(i, j) - lse) : 0.0F;
                }
                const xt::xarray<float> O = xt::linalg::dot(P, v2);
                xt::xarray<float> dS = xt::linalg::dot(do2, xt::transpose(v2));
                for (uint32_t i = 0; i < N; ++i) {
                    double u = 0.0;
                    for (uint32_t c = 0; c < d; ++c) u += static_cast<double>(do2(i, c)) * O(i, c);
                    for (uint32_t j = 0; j < N; ++j) dS(i, j) = P(i, j) * (dS(i, j) - static_cast<float>(u)) * scale;
                }
                const std::array<std::pair<const char*, xt::xarray<float>>, 3> ref = {{
                    {"dQ", xt::linalg::dot(dS, k2)},
                    {"dK", xt::linalg::dot(xt::transpose(dS), q2)},
                    {"dV", xt::linalg::dot(xt::transpose(P), do2)}}};
                const std::array<std::pair<ttnn::Tensor, ttnn::Tensor>, 3> dev = {{{tq, oq}, {tk, ok}, {tv, ov}}};
                for (size_t g = 0; g < 3; ++g) {
                    const auto& r = ref[g].second;
                    const auto rel = [&](const ttnn::Tensor& t) {
                        const xt::xarray<float> a = xt::view(ttml::core::to_xtensor(t), 0, 0, xt::all(), xt::all());
                        return std::sqrt(xt::mean(xt::square(a - r))()) / std::sqrt(xt::mean(xt::square(r))());
                    };
                    std::printf(
                        "    %s against the host Float32 reference: sdpa_bw %.2e, cyclic %.2e\n", ref[g].first,
                        rel(dev[g].first), rel(dev[g].second));
                }
            }
            const std::array<std::pair<const char*, std::pair<ttnn::Tensor, ttnn::Tensor>>, 3> grads = {{
                {"dQ", {tq, oq}}, {"dK", {tk, ok}}, {"dV", {tv, ov}}}};
            for (const auto& [name, pair] : grads) {
                const xt::xarray<float> a = ttml::core::to_xtensor(pair.first);
                const xt::xarray<float> b = ttml::core::to_xtensor(pair.second);
                const double rel = std::sqrt(xt::mean(xt::square(a - b))()) / std::sqrt(xt::mean(xt::square(a))());
                // Not a pass criterion: sdpa_bw's own error grows with N on
                // these inputs (1.7e-2 at 7040 rows, 3.6e-2 at 14080 against
                // the host reference, where the cyclic side stays at 2-4e-4),
                // so the two sides drift apart by tt-train's error.
                std::printf("    %s: relative RMS difference between the two %.2e\n", name, rel);
            }
        }
        const auto t_theirs = time_it([&]() {
            auto out = theirs();
            distributed::Finish(device->mesh_command_queue());
        });
        const auto t_ours = time_it([&]() {
            auto out = ours();
            distributed::Finish(device->mesh_command_queue());
        });
        if (std::getenv("TTML_BW_COMPARE_SPLIT") != nullptr) {
            // Where the cyclic side's time goes: forming D alone, and the op
            // with D given.
            const auto form_d = [&]() {
                return ttml::ttnn_fixed::sum_ttnn(ttnn::multiply(grad_output, ours_o, ttnn::DataType::FLOAT32), 3, true);
            };
            const auto D = form_d();
            const auto t_d = time_it([&]() {
                auto out = form_d();
                distributed::Finish(device->mesh_command_queue());
            });
            const auto t_k = time_it([&]() {
                auto out = ttml::metal::cyclic_sdpa_bw(q, k, v, grad_output, ours_lse, D, Bt);
                distributed::Finish(device->mesh_command_queue());
            });
            std::printf("    cyclic split: forming D %.3f ms, cyclic_sdpa_bw given D %.3f ms\n", t_d[1] * 1e3, t_k[1] * 1e3);
        }
        const double flop = 5.0 * static_cast<double>(N) * N * d * heads;  // five matmuls, causal half
        std::printf(
            "  heads %u/%u N %u d %u, Bt %u: sdpa_bw %.2f ms (%.1f TFLOP/s; min %.2f max %.2f), cyclic %.2f ms "
            "(%.1f TFLOP/s; min %.2f max %.2f), %.2fx\n",
            heads, kv_heads, N, d, Bt, t_theirs[1] * 1e3, flop / t_theirs[1] / 1e12, t_theirs[0] * 1e3,
            t_theirs[2] * 1e3, t_ours[1] * 1e3, flop / t_ours[1] / 1e12, t_ours[0] * 1e3, t_ours[2] * 1e3,
            t_theirs[1] / t_ours[1]);
    }
}

// One profiled launch of the cyclic forward, then an explicit device close
// (the profiler writes its CSV on a real close only; see the backward's
// profile test). Shape from TTML_CYCLIC_FW_TIME as in the timing test
// (default 4 heads, 2048 rows, d 64, Bt 4, causal).
//
//   TT_METAL_DEVICE_PROFILER=1 ttml_tests \
//     --gtest_filter=CyclicSdpaFwProfileTest.* --gtest_also_run_disabled_tests
TEST(CyclicSdpaFwProfileTest, DISABLED_ProfileTheForward) {
    auto* device = &ttml::autograd::ctx().get_device();
    uint32_t heads = 4, kv_heads = 4, N = 2048, d = 64, Bt = 4, causal_u = 1;
    if (const char* env = std::getenv("TTML_CYCLIC_FW_TIME"); env != nullptr && *env != '\0') {
        std::sscanf(env, "%u:%u:%u:%u:%u:%u", &heads, &kv_heads, &N, &d, &Bt, &causal_u);
    }
    xt::xarray<float> Q = xt::zeros<float>({1u, heads, N, d});
    xt::xarray<float> K = xt::zeros<float>({1u, kv_heads, N, d});
    xt::xarray<float> V = xt::zeros<float>({1u, kv_heads, N, d});
    for (uint32_t h = 0; h < heads; ++h) {
        xt::view(Q, 0, h, xt::all(), xt::all()) = random_bf16_matrix(N, d, 3000u + h);
    }
    for (uint32_t g = 0; g < kv_heads; ++g) {
        xt::view(K, 0, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 4000u + g);
        xt::view(V, 0, g, xt::all(), xt::all()) = random_bf16_matrix(N, d, 5000u + g);
    }
    const auto q = ttml::core::from_xtensor(Q, device);
    const auto k = ttml::core::from_xtensor(K, device);
    const auto v = ttml::core::from_xtensor(V, device);
    const auto mask = causal_u != 0 ? ttml::metal::AttentionMaskType::Causal : ttml::metal::AttentionMaskType::None;
    auto [o, l] = ttml::metal::cyclic_sdpa_fw(q, k, v, Bt, mask);
    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttml::autograd::ctx().close_device();
}
