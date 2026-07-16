// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Test-only harness for the regime_a_matmul diagnostic ablations. Drives the INTERNAL C++ entry point
// ttnn::prim::regime_a_matmul_diag (NOT the public/nanobind op) with a nonzero RegimeADiag mask so the
// ablation kernels can be profiled. One shape/config/mask per process invocation (params come from env);
// the per-test fixture opens+closes its own device, so the device-profiler CSV flushes on TearDown and an
// external Python driver (regime_a_diag_suite.py) parses per-RISC kernel time. Constant 1.0 inputs => the
// mask-0 output is exactly K, so mask-0 correctness is checked here without a torch reference; diagnostic
// modes require only successful completion (numerical output is intentionally garbage).
//
// Env: RA_M RA_K RA_N RA_NS RA_PK RA_SM RA_KB RA_NSB RA_MASK RA_ITERS  (all optional; defaults below).

#include <gtest/gtest.h>
#include <fmt/base.h>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn_test_fixtures.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_device_operation.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"

namespace ttnn::experimental::regime_a_matmul::diag_test {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::PageConfig;
using tt::tt_metal::Tensor;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::TensorSpec;

namespace {
uint32_t envu(const char* k, uint32_t dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? static_cast<uint32_t>(std::atoi(v)) : dflt;
}

// Pearson correlation between two flat vectors (device output vs CPU golden).
double pcc(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = a.size();
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    return (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : (va == 0 && vb == 0 ? 1.0 : 0.0);
}
}  // namespace

class RegimeADiagFixture : public ttnn::TTNNFixtureWithDevice {};

TEST_F(RegimeADiagFixture, Run) {
    const uint32_t M = envu("RA_M", 256), K = envu("RA_K", 2048), N = envu("RA_N", 1024);
    const uint32_t Ns = envu("RA_NS", 1), Pk = envu("RA_PK", 4), Sm = envu("RA_SM", 2);
    const uint32_t kb = envu("RA_KB", 2), nsb = envu("RA_NSB", 2);
    const uint32_t mask = envu("RA_MASK", 0), iters = envu("RA_ITERS", 8);
    auto* device = device_;

    // Constant 1.0 inputs: for mask 0 the reference output is exactly K (independent of layout/pad).
    std::vector<bfloat16> a(static_cast<size_t>(M) * K, bfloat16(1.0f));
    std::vector<bfloat16> b(static_cast<size_t>(K) * N, bfloat16(1.0f));

    const MemoryConfig il{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout lay(DataType::BFLOAT16, PageConfig(Layout::TILE), il);
    Tensor in0 = Tensor::from_vector(a, TensorSpec(ttnn::Shape({M, K}), lay)).to_device(device, il);

    const MemoryConfig wcfg =
        ttnn::experimental::prim::create_regime_a_weight_memory_config(ttnn::Shape({K, N}), DataType::BFLOAT16, device);
    Tensor in1 = Tensor::from_vector(b, TensorSpec(ttnn::Shape({K, N}), lay)).to_device(device, wcfg);

    const ttnn::experimental::prim::RegimeAMatmulConfig cfg{
        .k_slices = Pk, .n_slices = Ns, .m_slices = Sm, .k_block_tiles = kb, .n_subblock_tiles = nsb};

    fmt::print(
        "DIAGCFG M={} K={} N={} Ns={} Pk={} Sm={} kb={} nsb={} mask={} iters={}\n",
        M,
        K,
        N,
        Ns,
        Pk,
        Sm,
        kb,
        nsb,
        mask,
        iters);

    // Warmup / compile (run 0, dropped by the profiler parser).
    Tensor out = ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);

    // Constant-input sanity for the public path (mask 0) + the correct in0-delivery variants (32=scatter,
    // 64=repl2, 128=repl4, 256=xchg, 512=xchgrr): out must equal K. This only catches gross breakage; the
    // real pairing/permutation/repeat-omit check is the random-operand PCC test below. (max_rel_err, NOT PCC.)
    if (mask == 0 || mask == 32 || mask == 64 || mask == 128 || mask == 256 || mask == 512) {
        const std::vector<float> host = out.to_vector<float>();
        double maxrel = 0.0;
        for (float v : host) {
            maxrel = std::max(maxrel, std::abs(static_cast<double>(v) - K) / K);
        }
        fmt::print("DIAGPCC max_rel_err={:.5f}\n", maxrel);
        EXPECT_LT(maxrel, 0.02) << "constant-input output should equal K (mask=" << mask << ")";
    }

    // Timed steady-state iterations (profiler captures kernel time; parsed after device close).
    for (uint32_t i = 0; i < iters; ++i) {
        out = ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);
    }
    // Force completion before TearDown closes the device (which flushes the profiler CSV).
    (void)out.to_vector<float>();
    fmt::print("DIAGDONE mask={}\n", mask);
}

// Real correctness test for the in0-delivery VARIANTS. Constant-1.0 inputs (the perf harness above) cannot
// catch a wrong in0/in1 pairing, a repeated/omitted K shard, or many shard permutations, because every
// product is 1 and the K-sum is K regardless. Here we use RANDOM bf16 operands and compare each variant to
// a CPU f32 golden (computed from the same bf16-rounded inputs, so only accumulation order/rounding differ)
// via PCC. Each mask is run twice: fresh (compile) then cached-program. A mispairing/repeat/omit shifts the
// per-element sums and collapses PCC well below the 0.99 bar.
TEST_F(RegimeADiagFixture, Correctness) {
    const uint32_t M = 256, K = 2048, N = 1024;
    const uint32_t Ns = 1, Pk = 4, Sm = 2, kb = 2, nsb = 2;  // exercises ring + split-K reduction + M-split
    const uint32_t Mt = M / 32, Kt = K / 32, Nt = N / 32;
    auto* device = device_;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<bfloat16> a(static_cast<size_t>(M) * K), b(static_cast<size_t>(K) * N);
    std::vector<float> af(a.size()), bf(b.size());  // bf16-rounded inputs, for the golden
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = bfloat16(dist(rng));
        af[i] = static_cast<float>(a[i]);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = bfloat16(dist(rng));
        bf[i] = static_cast<float>(b[i]);
    }
    // CPU f32 golden C[M,N] = A[M,K] @ B[K,N] (row-major), i-k-j order.
    std::vector<float> golden(static_cast<size_t>(M) * N, 0.0f);
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t kk = 0; kk < K; ++kk) {
            const float aik = af[static_cast<size_t>(i) * K + kk];
            const float* brow = &bf[static_cast<size_t>(kk) * N];
            float* crow = &golden[static_cast<size_t>(i) * N];
            for (uint32_t j = 0; j < N; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }

    const MemoryConfig il{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout lay(DataType::BFLOAT16, PageConfig(Layout::TILE), il);
    Tensor in0 = Tensor::from_vector(a, TensorSpec(ttnn::Shape({M, K}), lay)).to_device(device, il);
    const MemoryConfig wcfg =
        ttnn::experimental::prim::create_regime_a_weight_memory_config(ttnn::Shape({K, N}), DataType::BFLOAT16, device);
    Tensor in1 = Tensor::from_vector(b, TensorSpec(ttnn::Shape({K, N}), lay)).to_device(device, wcfg);
    const ttnn::experimental::prim::RegimeAMatmulConfig cfg{
        .k_slices = Pk, .n_slices = Ns, .m_slices = Sm, .k_block_tiles = kb, .n_subblock_tiles = nsb};
    (void)Mt;
    (void)Kt;
    (void)Nt;

    // ring(0) + the correct in0-delivery variants; each fresh then cached (2nd invocation = cached program).
    for (uint32_t mask : {0u, 32u, 64u, 128u, 256u, 512u}) {
        for (int pass = 0; pass < 2; ++pass) {  // pass 0 = fresh/compile, pass 1 = cached-program replay
            Tensor out =
                ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);
            const std::vector<float> got = out.to_vector<float>();
            const double p = pcc(got, golden);
            fmt::print("DIAGCORR mask={} pass={} pcc={:.5f}\n", mask, pass, p);
            EXPECT_GT(p, 0.99) << "variant mask=" << mask << " pass=" << pass << " PCC too low (" << p << ")";
        }
    }
}

}  // namespace ttnn::experimental::regime_a_matmul::diag_test
