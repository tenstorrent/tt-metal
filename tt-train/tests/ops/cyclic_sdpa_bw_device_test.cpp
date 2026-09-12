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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
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
    xt::xarray<float> lse_tile, u_tile;  // (1,1,N,32), per-row value in column 0
    xt::xarray<float> dQ, dK, dV;
};

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

    const xt::xarray<float> O = xt::linalg::dot(P, r.V);
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

struct Gradients {
    xt::xarray<float> dQ, dK, dV;
};

Gradients run_algorithm2(uint32_t C, const Reference& ref, uint32_t grid_w, uint32_t grid_h) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();

    const uint32_t qWt = ref.d / kTile;
    const uint32_t vWt = ref.d / kTile;

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
    make_cb(tt::CBIndex::c_0, qWt, tt::DataFormat::Float16_b);  // Q_i
    make_cb(tt::CBIndex::c_1, qWt, tt::DataFormat::Float16_b);  // K_j
    make_cb(tt::CBIndex::c_2, vWt, tt::DataFormat::Float16_b);  // V_j
    make_cb(tt::CBIndex::c_3, vWt, tt::DataFormat::Float16_b);  // dO_i
    make_cb(tt::CBIndex::c_4, 1, tt::DataFormat::Float32);      // L_i
    make_cb(tt::CBIndex::c_5, 1, tt::DataFormat::Float32);      // D_i
    make_cb(tt::CBIndex::c_6, 1, tt::DataFormat::Float16_b);    // causal mask
    // Intermediates.
    make_cb(tt::CBIndex::c_10, 1, tt::DataFormat::Float32);  // P
    make_cb(tt::CBIndex::c_11, 1, tt::DataFormat::Float32);  // dP
    make_cb(tt::CBIndex::c_12, 1, tt::DataFormat::Float32);  // dS
    make_cb(tt::CBIndex::c_13, 1, tt::DataFormat::Float32);  // dS^T
    make_cb(tt::CBIndex::c_14, 1, tt::DataFormat::Float32);  // P^T
    // Each gradient: seed from the reader, accumulator, output to the writer.
    make_cb(tt::CBIndex::c_15, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_16, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_17, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_18, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_19, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_20, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_21, vWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_22, vWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_23, vWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_24, 1, tt::DataFormat::Float32);  // control word

    const uint32_t arrive_sem = CreateSemaphore(program, region, 0);
    const uint32_t release_sem = CreateSemaphore(program, region, 0);

    std::vector<uint32_t> reader_args = {C, qWt, vWt, release_sem};
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

    std::vector<uint32_t> writer_args = {C, qWt, vWt, arrive_sem, release_sem};
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
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {C, qWt, vWt, scaler, minus_one, custom_inf, 1u}});

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
Gradients run_relay(
    uint32_t C, const Reference& ref, uint32_t grid_w, uint32_t grid_h, bool endpoint_sync = false) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();

    const uint32_t qWt = ref.d / kTile;
    const uint32_t vWt = ref.d / kTile;

    const auto query = ttml::core::from_xtensor(as_4d(ref.Q), device);
    const auto key = ttml::core::from_xtensor(as_4d(ref.K), device);
    const auto value = ttml::core::from_xtensor(as_4d(ref.V), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d(ref.dO), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(ref.lse_tile, device);
    const auto u_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(ref.u_tile, device);

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
    // The packet buffers hold two slots; everything else holds one.
    make_cb(tt::CBIndex::c_0, 2 * qWt, tt::DataFormat::Float16_b);  // Q_i
    make_cb(tt::CBIndex::c_3, 2 * vWt, tt::DataFormat::Float16_b);  // dO_i
    make_cb(tt::CBIndex::c_4, 2, tt::DataFormat::Float32);          // L_i
    make_cb(tt::CBIndex::c_5, 2, tt::DataFormat::Float32);          // D_i
    make_cb(tt::CBIndex::c_15, 2 * qWt, tt::DataFormat::Float32);   // dQ_i, travels along
    make_cb(tt::CBIndex::c_1, qWt, tt::DataFormat::Float16_b);      // K_j
    make_cb(tt::CBIndex::c_2, vWt, tt::DataFormat::Float16_b);      // V_j
    make_cb(tt::CBIndex::c_6, 1, tt::DataFormat::Float16_b);        // causal mask
    make_cb(tt::CBIndex::c_10, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_11, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_12, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_13, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_14, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_16, qWt, tt::DataFormat::Float32);  // dQ accumulator
    make_cb(tt::CBIndex::c_17, qWt, tt::DataFormat::Float32);  // dQ to the relay
    make_cb(tt::CBIndex::c_18, qWt, tt::DataFormat::Float32);  // dK seed
    make_cb(tt::CBIndex::c_19, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_20, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_21, vWt, tt::DataFormat::Float32);  // dV seed
    make_cb(tt::CBIndex::c_22, vWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_23, vWt, tt::DataFormat::Float32);
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
        endpoint2_sem};
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

    std::vector<uint32_t> writer_args = {C, qWt, vWt, arrive_sem, release_sem};
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
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_mode[tt::CBIndex::c_10] = UnpackToDestMode::UnpackToDestFp32;
    const auto compute = CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {C, qWt, vWt, scaler, minus_one, custom_inf, 1u},
            .defines = compute_defines});

    const auto coordinator_logical = placement_of(C, grid_w, 1);
    const auto coordinator = device->worker_core_from_logical_core(
        CoreCoord{coordinator_logical.x, coordinator_logical.y});
    const auto mcast_start = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const auto mcast_end =
        device->worker_core_from_logical_core(CoreCoord{grid_w - 1, grid_h - 1});

    const auto noc_of_core = [&](uint32_t core) {
        const auto xy = placement_of(C, grid_w, core);
        return device->worker_core_from_logical_core(CoreCoord{xy.x, xy.y});
    };

    for (uint32_t c = 1; c <= C; ++c) {
        const auto xy = placement_of(C, grid_w, c);
        const auto core = CoreCoord{xy.x, xy.y};
        const auto neighbors = snake_neighbors(C, c);
        const auto prev = noc_of_core(neighbors.prev != kNoCore ? neighbors.prev : c);
        const auto next = noc_of_core(neighbors.next != kNoCore ? neighbors.next : c);
        std::vector<uint32_t> relay_reader_args = {
            c, query.buffer()->address(), key.buffer()->address(), value.buffer()->address(),
            grad_output.buffer()->address(), lse.buffer()->address(), u_scalar.buffer()->address(),
            grad_query.buffer()->address(), grad_key.buffer()->address(),
            grad_value.buffer()->address(), static_cast<uint32_t>(prev.x),
            static_cast<uint32_t>(prev.y), static_cast<uint32_t>(next.x),
            static_cast<uint32_t>(next.y)};
        // Every core's coordinates, for endpoint publication by unicast.
        for (uint32_t r = 1; r <= C; ++r) {
            const auto rc = noc_of_core(r);
            relay_reader_args.push_back(static_cast<uint32_t>(rc.x));
            relay_reader_args.push_back(static_cast<uint32_t>(rc.y));
        }
        SetRuntimeArgs(program, reader, core, relay_reader_args);
        SetRuntimeArgs(
            program, writer, core,
            {c, grad_key.buffer()->address(), grad_value.buffer()->address(),
             static_cast<uint32_t>(coordinator.x), static_cast<uint32_t>(coordinator.y),
             static_cast<uint32_t>(mcast_start.x), static_cast<uint32_t>(mcast_start.y),
             static_cast<uint32_t>(mcast_end.x), static_cast<uint32_t>(mcast_end.y),
             c == 1u ? 1u : 0u});
        SetRuntimeArgs(program, compute, core, {c});
    }

    auto workload = tt_dist::MeshWorkload();
    workload.add_program(tt_dist::MeshCoordinateRange(device->shape()), std::move(program));
    tt_dist::EnqueueMeshWorkload(device->mesh_command_queue(), workload, /*blocking=*/true);

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
    const std::string& what) {
    const uint32_t rows = static_cast<uint32_t>(want.shape()[0]);
    const uint32_t cols = static_cast<uint32_t>(want.shape()[1]);
    float max_abs = 0.0F;
    float max_diff = 0.0F;
    uint32_t worst_r = 0;
    uint32_t worst_c = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            max_abs = std::max(max_abs, std::abs(want(r, c)));
            const float diff = std::abs(got(0, 0, r, c) - want(r, c));
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
        << got(0, 0, worst_r, worst_c) << " want " << want(worst_r, worst_c);
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

void check_relay(
    uint32_t C, uint32_t grid_w, uint32_t grid_h, uint32_t d = 64, bool endpoint_sync = false) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid_w > grid.x || grid_h > grid.y) {
        GTEST_SKIP() << "needs " << grid_w << "x" << grid_h;
    }
    const uint32_t N = 2u * C * kTile;
    const auto ref = make_reference(N, d);
    const auto got = run_relay(C, ref, grid_w, grid_h, endpoint_sync);
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
}
