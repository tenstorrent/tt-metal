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
Gradients run_relay(uint32_t C, const Reference& ref, uint32_t grid_w, uint32_t grid_h) {
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

    const uint32_t arrive_sem = CreateSemaphore(program, region, 0);
    const uint32_t release_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready0_sem = CreateSemaphore(program, region, 0);
    const uint32_t ready1_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_prev_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_next_sem = CreateSemaphore(program, region, 0);
    const uint32_t credit_self_sem = CreateSemaphore(program, region, 0);

    std::vector<uint32_t> reader_args = {
        C, qWt, vWt, release_sem, ready0_sem, ready1_sem,
        credit_prev_sem, credit_next_sem, credit_self_sem};
    for (const auto* t : {&query, &key, &value, &grad_output, &lse, &u_scalar, &grad_query,
                          &grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(reader_args);
    }
    const auto reader = CreateKernel(
        program, kRelayReaderPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_args});

    std::vector<uint32_t> writer_args = {C, qWt, vWt, arrive_sem, release_sem};
    for (const auto* t : {&grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(writer_args);
    }
    const auto writer = CreateKernel(
        program, kRelayWriterPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_args});

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
            .compile_args = {C, qWt, vWt, scaler, minus_one, custom_inf, 1u}});

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
        SetRuntimeArgs(
            program, reader, core,
            {c, query.buffer()->address(), key.buffer()->address(), value.buffer()->address(),
             grad_output.buffer()->address(), lse.buffer()->address(), u_scalar.buffer()->address(),
             grad_query.buffer()->address(), grad_key.buffer()->address(),
             grad_value.buffer()->address(), static_cast<uint32_t>(prev.x),
             static_cast<uint32_t>(prev.y), static_cast<uint32_t>(next.x),
             static_cast<uint32_t>(next.y)});
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

void check_relay(uint32_t C, uint32_t grid_w, uint32_t grid_h, uint32_t d = 64) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    if (grid_w > grid.x || grid_h > grid.y) {
        GTEST_SKIP() << "needs " << grid_w << "x" << grid_h;
    }
    const uint32_t N = 2u * C * kTile;
    const auto ref = make_reference(N, d);
    const auto got = run_relay(C, ref, grid_w, grid_h);
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
