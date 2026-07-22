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
#include <algorithm>
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

// Elementwise error of `got` vs a reference `ref` (the chain output): absolute max/mean/p99, worst relative
// error (ref magnitude > 1e-6), and the fraction of BF16 elements that DIFFER at all. Reassociating reductions
// (tree / reduce-scatter) are PCC-preserving but NOT bit-identical, so these quantify how far from the chain.
struct ErrMetrics {
    double max_abs, mean_abs, p99_abs, max_rel;
    double frac_diff;
};
ErrMetrics err_metrics(const std::vector<float>& got, const std::vector<float>& ref) {
    const size_t n = std::min(got.size(), ref.size());
    std::vector<double> ad;
    ad.reserve(n);
    double sum = 0, mx = 0, mrel = 0;
    size_t diff = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::abs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        ad.push_back(d);
        sum += d;
        mx = std::max(mx, d);
        if (got[i] != ref[i]) {
            diff++;
        }
        const double denom = std::abs(static_cast<double>(ref[i]));
        if (denom > 1e-6) {
            mrel = std::max(mrel, d / denom);
        }
    }
    std::sort(ad.begin(), ad.end());
    const double p99 = ad.empty() ? 0.0 : ad[std::min(ad.size() - 1, static_cast<size_t>(ad.size() * 0.99))];
    return {mx, n ? sum / n : 0.0, p99, mrel, n ? static_cast<double>(diff) / n : 0.0};
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

    // Constant-input sanity for the public path (mask 0) + the correctness-PRESERVING diagnostics
    // (full-wait 1024, barrier-drain 2048, ring-order 4096/16384/65536/262144, placement 524288/2097152, and
    // the in1 fwd flush-order A/B 1<<22 + coalesced-read A/B 1<<25): out must equal K. Pure ablations produce
    // garbage and are NOT checked. (max_rel_err, NOT PCC — the real pairing check is the random-operand PCC
    // test below.) A fwd-order/source-lifetime bug would deliver stale in1 -> output != K, caught here.
    constexpr uint32_t kIn1Preserve = (1u << 22) | (1u << 25);
    if (mask == 0 || mask == 1024 || mask == 2048 || mask == 4096 || mask == 16384 || mask == 65536 || mask == 262144 ||
        mask == 524288 || mask == 2097152 || mask == 32 /*DIAG_RINGDRAIN: source-lifetime/ordering-preserving*/ ||
        mask == 64 /*DIAG_REDTREE: fan-in-2 tree; reassociates sum (NOT bit-exact) but constant-input == K*/ ||
        mask == 128 /*DIAG_RSCATTER: ring reduce-scatter; reassociates but constant-input sums to K exactly*/ ||
        (mask != 0u && (mask & ~kIn1Preserve) == 0u)) {
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

// Random-operand PCC correctness for the DEFAULT public path (mask 0) and the full-slice-startup-wait A/B
// baseline (DIAG_FULL_IN0_WAIT, 1024). Constant-1.0 inputs (the perf harness above) cannot catch a wrong
// in0/in1 pairing or a repeated/omitted K shard, because every product is 1 and the K-sum is K regardless.
// Here we use RANDOM bf16 operands and compare to a CPU f32 golden (from the same bf16-rounded inputs, so
// only accumulation order/rounding differ) via PCC. Each mask is run twice: fresh (compile) then
// cached-program. A mispairing/repeat/omit shifts the per-element sums and collapses PCC below the 0.99 bar.
TEST_F(RegimeADiagFixture, Correctness) {
    const uint32_t M = 256, K = 2048, N = 1024;
    // nsb=4 => N_bpc=1 (one output block/core), so this config is valid for ALL of chain / full-wait / tree /
    // reduce-scatter (the last requires M_block==Pk && N_bpc==1). Exercises ring + split-K reduce + M-split.
    const uint32_t Ns = 1, Pk = 4, Sm = 2, kb = 2, nsb = 4;
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

    // default progressive path (0) + full-wait A/B (1024) + fan-in-2 reduction tree (DIAG_REDTREE, 64) +
    // ring reduce-scatter (DIAG_RSCATTER, 128). This config is Pk=4/Sm=2, so every reduction group is a real
    // depth-2 tree / 4-core reduce-scatter ring. The tree and reduce-scatter sum the SAME four k-slice partials
    // in a DIFFERENT association order (and round bf16 per hop), so they are NOT bit-identical to the chain —
    // PCC vs the f32 golden must stay >= 0.99, and we ALSO report the elementwise max|out - chain| vs the chain
    // (mask 0) to quantify the reassociation (a mis-linked ring / dropped partial / slot aliasing collapses PCC).
    std::vector<float> chain_ref;  // mask-0 pass-0 output, the bitwise reference
    for (uint32_t mask : {0u, 1024u, 64u, 128u}) {
        for (int pass = 0; pass < 2; ++pass) {  // pass 0 = fresh/compile, pass 1 = cached-program replay
            Tensor out =
                ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);
            const std::vector<float> got = out.to_vector<float>();
            const double p = pcc(got, golden);
            if (mask == 0u && pass == 0) {
                chain_ref = got;
            }
            double maxdiff = 0.0;
            if (!chain_ref.empty() && got.size() == chain_ref.size()) {
                for (size_t k = 0; k < got.size(); ++k) {
                    maxdiff = std::max(maxdiff, std::abs(static_cast<double>(got[k]) - chain_ref[k]));
                }
            }
            fmt::print("DIAGCORR mask={} pass={} pcc={:.5f} maxdiff_vs_chain={:.6f}\n", mask, pass, p, maxdiff);
            EXPECT_GT(p, 0.99) << "variant mask=" << mask << " pass=" << pass << " PCC too low (" << p << ")";
        }
    }
}

// Generalized ring reduce-scatter (DIAG_RSCATTER, mask 128) over arbitrary Pk with tile-partition. Random
// BF16 vs f32 golden, PCC >= 0.99, fresh + cached, for a table spanning Pk=2 (chunk=2), Pk=4 (row-partition,
// chunk==N_block), and Pk=6 (chunk=2, Mt=4). Also reports maxdiff vs the chain (mask 0). A mis-routed chunk /
// dropped partial / bad ring-cycle would collapse PCC. All configs are N_bpc==1 with T=M_block*N_sub % Pk==0.
TEST_F(RegimeADiagFixture, RScatterGeneral) {
    struct Cfg {
        uint32_t M, K, N, Ns, Pk, Sm, kb, nsb;
        const char* tag;
    };
    const std::vector<Cfg> cfgs = {
        {32, 2048, 2048, 2, 2, 1, 4, 4, "pk2_chunk2"},    // Pk=2, T=4, chunk=2
        {128, 15360, 768, 1, 6, 1, 2, 3, "pk6_chunk2"},   // Pk=6, T=12, chunk=2, Mt=4
        {256, 2048, 1024, 1, 4, 2, 2, 4, "pk4_rowpart"},  // Pk=4, T=16, chunk=4==N_block (row-partition)
    };
    auto* device = device_;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const std::vector<uint32_t> seeds = {321u, 777u, 20240722u};  // several random seeds per config
    for (const auto& c : cfgs) {
        const uint32_t M = c.M, K = c.K, N = c.N;
        const MemoryConfig il{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
        const TensorLayout lay(DataType::BFLOAT16, PageConfig(Layout::TILE), il);
        const MemoryConfig wcfg = ttnn::experimental::prim::create_regime_a_weight_memory_config(
            ttnn::Shape({K, N}), DataType::BFLOAT16, device);
        const ttnn::experimental::prim::RegimeAMatmulConfig cfg{
            .k_slices = c.Pk, .n_slices = c.Ns, .m_slices = c.Sm, .k_block_tiles = c.kb, .n_subblock_tiles = c.nsb};
        for (uint32_t seed : seeds) {
            std::mt19937 rng(seed);
            std::vector<bfloat16> a(static_cast<size_t>(M) * K), b(static_cast<size_t>(K) * N);
            std::vector<float> af(a.size()), bf(b.size());
            for (size_t i = 0; i < a.size(); ++i) {
                a[i] = bfloat16(dist(rng));
                af[i] = static_cast<float>(a[i]);
            }
            for (size_t i = 0; i < b.size(); ++i) {
                b[i] = bfloat16(dist(rng));
                bf[i] = static_cast<float>(b[i]);
            }
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
            Tensor in0 = Tensor::from_vector(a, TensorSpec(ttnn::Shape({M, K}), lay)).to_device(device, il);
            Tensor in1 = Tensor::from_vector(b, TensorSpec(ttnn::Shape({K, N}), lay)).to_device(device, wcfg);
            std::vector<float> chain_ref;
            for (uint32_t mask : {0u, 128u}) {
                for (int pass = 0; pass < 2; ++pass) {  // fresh (compile) then cached-program
                    Tensor out =
                        ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);
                    const std::vector<float> got = out.to_vector<float>();
                    const double p = pcc(got, golden);
                    if (mask == 0u && pass == 0) {
                        chain_ref = got;
                    }
                    const ErrMetrics e = err_metrics(got, chain_ref);
                    // Raw per-(config,seed,mask,pass) sample line: PCC vs f32 golden + full error vs chain.
                    fmt::print(
                        "DIAGRSGEN {} seed={} mask={} pass={} pcc={:.5f} maxabs={:.5f} meanabs={:.6f} "
                        "p99abs={:.5f} maxrel={:.5f} fracdiff={:.4f}\n",
                        c.tag,
                        seed,
                        mask,
                        pass,
                        p,
                        e.max_abs,
                        e.mean_abs,
                        e.p99_abs,
                        e.max_rel,
                        e.frac_diff);
                    EXPECT_GT(p, 0.99) << "rscatter-general " << c.tag << " seed=" << seed << " mask=" << mask;
                }
            }
        }
    }
}

// Progressive-cumulative-wait (default, mask 0) vs old full-slice startup barrier (DIAG_FULL_IN0_WAIT,
// mask 1024) A/B. Both paths use IDENTICAL config/tensors/ring-transport/reduction — only the CB0 wait
// placement differs, and the matmul accumulation ORDER is unchanged — so the two outputs must be BIT-
// IDENTICAL. Each variant is also checked against a CPU f32 golden (PCC). The config table deliberately
// spans W=1 (both Mt=8 primary targets), W>1 (Pk=1 => 4 blocks/shard), and N_bpc = 1 and >1, plus split-K
// reduce and Pk=1 no-reduce, so the progressive schedule + resident CB0 reuse is exercised across shapes.
TEST_F(RegimeADiagFixture, ProgressiveVsFullWait) {
    struct Case {
        const char* label;
        uint32_t M, K, N, Ns, Pk, Sm, kb, nsb;
        uint32_t W, N_bpc;  // expected, for the log
    };
    // (label, M,K,N, Ns,Pk,Sm,kb,nsb, W, N_bpc)
    const std::vector<Case> cases = {
        {"primary1_2048x1024_W1_Nbpc2", 256, 2048, 1024, 1, 4, 2, 2, 2, 1, 2},  // target1 factorization
        {"primary2_6144x768_W1_Nbpc3", 256, 6144, 768, 1, 12, 1, 2, 1, 1, 3},   // target2 factorization
        {"wgt1_2048x1024_W4_Nbpc2", 256, 2048, 1024, 1, 1, 1, 2, 2, 4, 2},      // Pk=1 => W=4 (>1), no-reduce
        {"nbpc1_2048x1024_W4_Nbpc1", 256, 2048, 1024, 1, 1, 1, 2, 4, 4, 1},     // N_bpc=1 (single N-subblock)
    };
    auto* device = device_;

    for (const auto& c : cases) {
        const uint32_t M = c.M, K = c.K, N = c.N;
        std::mt19937 rng(777);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> a(static_cast<size_t>(M) * K), b(static_cast<size_t>(K) * N);
        std::vector<float> af(a.size()), bf(b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = bfloat16(dist(rng));
            af[i] = static_cast<float>(a[i]);
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = bfloat16(dist(rng));
            bf[i] = static_cast<float>(b[i]);
        }
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
        const MemoryConfig wcfg = ttnn::experimental::prim::create_regime_a_weight_memory_config(
            ttnn::Shape({K, N}), DataType::BFLOAT16, device);
        Tensor in1 = Tensor::from_vector(b, TensorSpec(ttnn::Shape({K, N}), lay)).to_device(device, wcfg);
        const ttnn::experimental::prim::RegimeAMatmulConfig cfg{
            .k_slices = c.Pk, .n_slices = c.Ns, .m_slices = c.Sm, .k_block_tiles = c.kb, .n_subblock_tiles = c.nsb};

        auto run = [&](uint32_t mask) {
            Tensor out =
                ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);
            return out.to_vector<float>();
        };

        // mask 0 = progressive (default), mask 1024 = old full-slice startup wait.
        const std::vector<float> prog = run(0u);
        const std::vector<float> full = run(1024u);
        const double p_prog = pcc(prog, golden);
        const double p_full = pcc(full, golden);
        double ab_maxdiff = 0.0;
        ASSERT_EQ(prog.size(), full.size());
        for (size_t i = 0; i < prog.size(); ++i) {
            ab_maxdiff = std::max(ab_maxdiff, std::abs(static_cast<double>(prog[i]) - full[i]));
        }
        fmt::print(
            "DIAGAB case={} W={} N_bpc={} pcc_prog={:.5f} pcc_full={:.5f} ab_maxdiff={:.6f}\n",
            c.label,
            c.W,
            c.N_bpc,
            p_prog,
            p_full,
            ab_maxdiff);
        EXPECT_GT(p_prog, 0.999) << "progressive PCC too low: " << c.label;
        EXPECT_GT(p_full, 0.999) << "full-wait PCC too low: " << c.label;
        // Only the wait placement differs; matmul order identical => bit-identical output.
        EXPECT_EQ(ab_maxdiff, 0.0) << "progressive vs full-wait differ (should be bit-identical): " << c.label;
    }
}

// Pipelined phase-2 drain (DIAG_PIPELINED_DRAIN, mask 1<<11=2048) vs the barrier baseline (mask 0). Only the
// writer's per-block write synchronization changes (departure-flush + ordered payload->semaphore + one final
// barrier, vs a per-block completion barrier); compute/ring/reduction are identical, so the output must be
// BIT-IDENTICAL. Random BF16 vs a CPU f32 golden, PCC >= 0.999, fresh AND cached. The table spans Pk=1 direct
// output, Pk=2/4/12 chains (bottom/middle/top roles), N_bpc=1/2/3 + wide-N, Sm=1/>1, and balanced K/N tails.
TEST_F(RegimeADiagFixture, PipelinedDrainCorrectness) {
    struct Case {
        const char* label;
        uint32_t M, K, N, Ns, Pk, Sm, kb, nsb;
    };
    const std::vector<Case> cases = {
        {"pk1_direct_Nbpc2", 256, 2048, 1024, 1, 1, 1, 2, 2},   // Pk=1 direct DRAM output
        {"pk2_ns2", 32, 2048, 2048, 2, 2, 1, 4, 4},             // Pk=2 chain (bottom+top), Ns>1
        {"pk4_sm2_Nbpc2", 256, 2048, 1024, 1, 4, 2, 2, 2},      // Pk=4 chain, Sm>1
        {"pk12_Nbpc3_roles", 256, 6144, 768, 1, 12, 1, 2, 1},   // deep chain: bottom/middle/top; N_bpc=3
        {"pk12_Nbpc1", 256, 6144, 768, 1, 12, 1, 2, 3},         // N_bpc=1 (single output block)
        {"wideN_pk12", 256, 6144, 4608, 1, 12, 1, 2, 1},        // wide-N, N_bpc=18
        {"ktail_ntail_pk12", 256, 6080, 4640, 1, 12, 1, 2, 1},  // balanced K-tail (190) + N-tail (145)
    };
    // mask 0 = pipelined (production default); mask 1<<11 = DIAG_BARRIER_DRAIN (old per-block barrier).
    const std::vector<std::pair<uint32_t, const char*>> masks = {{0u, "pipelined"}, {2048u, "barrier"}};
    auto* device = device_;

    for (const auto& c : cases) {
        const uint32_t M = c.M, K = c.K, N = c.N;
        std::mt19937 rng(4242);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> a(static_cast<size_t>(M) * K), b(static_cast<size_t>(K) * N);
        std::vector<float> af(a.size()), bf(b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = bfloat16(dist(rng));
            af[i] = static_cast<float>(a[i]);
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = bfloat16(dist(rng));
            bf[i] = static_cast<float>(b[i]);
        }
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
        const MemoryConfig wcfg = ttnn::experimental::prim::create_regime_a_weight_memory_config(
            ttnn::Shape({K, N}), DataType::BFLOAT16, device);
        Tensor in1 = Tensor::from_vector(b, TensorSpec(ttnn::Shape({K, N}), lay)).to_device(device, wcfg);
        const ttnn::experimental::prim::RegimeAMatmulConfig cfg{
            .k_slices = c.Pk, .n_slices = c.Ns, .m_slices = c.Sm, .k_block_tiles = c.kb, .n_subblock_tiles = c.nsb};

        std::vector<float> baseline_out;
        for (const auto& [mask, tag] : masks) {
            for (int pass = 0; pass < 2; ++pass) {  // fresh then cached-program
                Tensor out =
                    ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);
                const std::vector<float> got = out.to_vector<float>();
                const double p = pcc(got, golden);
                double ab = -1.0;
                if (mask == 0u && pass == 0) {
                    baseline_out = got;  // capture pipelined default (reference) for the A/B bit-identity check
                } else if (mask != 0u) {
                    ab = 0.0;
                    for (size_t i = 0; i < got.size(); ++i) {
                        ab = std::max(ab, std::abs(static_cast<double>(got[i]) - baseline_out[i]));
                    }
                }
                fmt::print("DIAGPD case={} {} pass={} pcc={:.5f} ab_maxdiff={:.6f}\n", c.label, tag, pass, p, ab);
                EXPECT_GT(p, 0.999) << "pipelined-drain case=" << c.label << " " << tag << " pass=" << pass
                                    << " PCC too low (" << p << ")";
                if (mask != 0u) {
                    // Only write-sync differs -> barrier must match the pipelined default bit-for-bit.
                    EXPECT_EQ(ab, 0.0) << "barrier != pipelined default (case=" << c.label << " pass=" << pass << ")";
                }
            }
        }
    }
}

// Physical ring ordering: bank (default, mask 0) vs greedy (1<<12) vs exhaustive-opt (1<<13). Only the in0
// ring visiting order changes (which core seeds which shard + forward route + in1 rotated read) — the math is
// identical, so output must be BIT-IDENTICAL across orders. Random BF16 vs CPU f32 golden, PCC >= 0.999,
// fresh AND cached. Table spans Pk=1 + split-K (both reader-NoC orientations via multi-slice Pk), Ns>1, Sm>1,
// W=1/W>1, and balanced K/N tails.
TEST_F(RegimeADiagFixture, RingOrderCorrectness) {
    struct Case {
        const char* label;
        uint32_t M, K, N, Ns, Pk, Sm, kb, nsb;
    };
    const std::vector<Case> cases = {
        {"pk1_W4_noc0", 256, 2048, 1024, 1, 1, 1, 2, 2},        // Pk=1 single ring (NOC0-reader), W=4
        {"pk2_ns2_bothnoc", 32, 2048, 2048, 2, 2, 1, 4, 4},     // Pk=2 -> both reader NoCs; Ns>1
        {"pk4_sm2_bothnoc", 256, 2048, 1024, 1, 4, 2, 2, 2},    // Sm=2; Pk=4 -> both NoCs (agg over 2 mm-rings)
        {"sm3_2048x1024", 256, 2048, 1024, 1, 1, 3, 2, 2},      // Sm=3 (balanced M-split, agg over 3 mm-rings)
        {"sm4_2048x1024", 256, 2048, 1024, 1, 1, 4, 2, 2},      // largest feasible Sm here (agg over 4 mm-rings)
        {"pk12_768_W1", 256, 6144, 768, 1, 12, 1, 2, 1},        // deep chain, W=1, both NoCs, N_bpc=3
        {"wideN_pk12", 256, 6144, 4608, 1, 12, 1, 2, 1},        // wide-N Sm=1
        {"ktail_ntail_pk12", 256, 6080, 4640, 1, 12, 1, 2, 1},  // balanced K-tail (190) + N-tail (145)
    };
    // mask 0 = opt (production default); 1<<12 = DIAG_RING_BANK (old bank order); 1<<13 = DIAG_RING_GREEDY.
    // mask 0 = pareto (default); 1<<12 bank; 1<<14 mm0; 1<<16 total; 1<<18 maxedge. All must be bit-identical
    // (only the ring order differs, never the math). (greedy/maxring objectives were removed post-decision.)
    const std::vector<std::pair<uint32_t, const char*>> masks = {
        {4096u, "bank"}, {16384u, "mm0"}, {0u, "pareto"}, {65536u, "total"}, {262144u, "maxedge"}};
    auto* device = device_;

    for (const auto& c : cases) {
        const uint32_t M = c.M, K = c.K, N = c.N;
        std::mt19937 rng(9091);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> a(static_cast<size_t>(M) * K), b(static_cast<size_t>(K) * N);
        std::vector<float> af(a.size()), bf(b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = bfloat16(dist(rng));
            af[i] = static_cast<float>(a[i]);
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = bfloat16(dist(rng));
            bf[i] = static_cast<float>(b[i]);
        }
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
        const MemoryConfig wcfg = ttnn::experimental::prim::create_regime_a_weight_memory_config(
            ttnn::Shape({K, N}), DataType::BFLOAT16, device);
        Tensor in1 = Tensor::from_vector(b, TensorSpec(ttnn::Shape({K, N}), lay)).to_device(device, wcfg);
        const ttnn::experimental::prim::RegimeAMatmulConfig cfg{
            .k_slices = c.Pk, .n_slices = c.Ns, .m_slices = c.Sm, .k_block_tiles = c.kb, .n_subblock_tiles = c.nsb};

        std::vector<float> bank_out;
        for (const auto& [mask, tag] : masks) {
            for (int pass = 0; pass < 2; ++pass) {  // fresh then cached-program
                Tensor out =
                    ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);
                const std::vector<float> got = out.to_vector<float>();
                const double p = pcc(got, golden);
                double ab = -1.0;
                if (bank_out.empty()) {
                    bank_out = got;  // first variant (bank) = reference for the A/B bit-identity check
                } else {
                    ab = 0.0;
                    for (size_t i = 0; i < got.size(); ++i) {
                        ab = std::max(ab, std::abs(static_cast<double>(got[i]) - bank_out[i]));
                    }
                }
                fmt::print("DIAGRING case={} {} pass={} pcc={:.5f} ab_maxdiff={:.6f}\n", c.label, tag, pass, p, ab);
                EXPECT_GT(p, 0.999) << "ring-order case=" << c.label << " " << tag << " pass=" << pass
                                    << " PCC too low (" << p << ")";
                if (ab >= 0.0) {
                    EXPECT_EQ(ab, 0.0) << tag << " != bank order (case=" << c.label << " pass=" << pass << ")";
                }
            }
        }
    }
}

// M-split worker PLACEMENT: current (default) vs readers_first (1<<19) vs in1_near (1<<20). Placement changes
// only worker coordinates (logical indices/ownership/forward pairing unchanged). It recomputes the PARETO
// ring order on the new coords, which for Sm>1 can change the K-shard accumulation order -> a benign bf16
// OUTPUT-ULP rounding difference (PCC still 0.99999, ~one ULP at the output magnitude) — NOT bit-identical in
// general. So: require PCC >= 0.999 for every case; additionally require BIT-IDENTICAL only for the Sm=1
// no-op (placement is a true no-op there — catches accidental changes). Random BF16 vs CPU f32 golden,
// fresh AND cached.
TEST_F(RegimeADiagFixture, PlacementCorrectness) {
    struct Case {
        const char* label;
        uint32_t M, K, N, Ns, Pk, Sm, kb, nsb;
    };
    const std::vector<Case> cases = {
        {"sm2_primary", 256, 2048, 1024, 1, 4, 2, 2, 2},
        {"sm2_wide", 128, 6144, 4608, 1, 6, 2, 2, 1},
        {"sm3", 256, 2048, 1024, 1, 1, 3, 2, 2},
        {"sm4", 256, 2048, 1024, 1, 1, 4, 2, 2},
        {"sm4_wide", 256, 6144, 4608, 1, 3, 4, 2, 1},
        {"sm1_noop", 256, 6144, 768, 1, 12, 1, 2, 1},  // Sm=1: placement is a no-op
    };
    // mask 0 = in1_near (default); 1<<21 = current (baseline); 1<<19 = readers_first. Reference = current.
    const std::vector<std::pair<uint32_t, const char*>> masks = {
        {2097152u, "current"}, {0u, "in1_near"}, {524288u, "readers_first"}};
    auto* device = device_;

    for (const auto& c : cases) {
        const uint32_t M = c.M, K = c.K, N = c.N;
        std::mt19937 rng(5150);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> a(static_cast<size_t>(M) * K), b(static_cast<size_t>(K) * N);
        std::vector<float> af(a.size()), bf(b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = bfloat16(dist(rng));
            af[i] = static_cast<float>(a[i]);
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] = bfloat16(dist(rng));
            bf[i] = static_cast<float>(b[i]);
        }
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
        const MemoryConfig wcfg = ttnn::experimental::prim::create_regime_a_weight_memory_config(
            ttnn::Shape({K, N}), DataType::BFLOAT16, device);
        Tensor in1 = Tensor::from_vector(b, TensorSpec(ttnn::Shape({K, N}), lay)).to_device(device, wcfg);
        const ttnn::experimental::prim::RegimeAMatmulConfig cfg{
            .k_slices = c.Pk, .n_slices = c.Ns, .m_slices = c.Sm, .k_block_tiles = c.kb, .n_subblock_tiles = c.nsb};

        std::vector<float> cur_out;
        for (const auto& [mask, tag] : masks) {
            for (int pass = 0; pass < 2; ++pass) {
                Tensor out =
                    ttnn::prim::regime_a_matmul_diag(in0, in1, cfg, std::nullopt, std::nullopt, std::nullopt, mask);
                const std::vector<float> got = out.to_vector<float>();
                const double p = pcc(got, golden);
                double ab = -1.0;
                if (cur_out.empty()) {
                    cur_out = got;  // first entry = current (reference)
                } else {
                    ab = 0.0;
                    for (size_t i = 0; i < got.size(); ++i) {
                        ab = std::max(ab, std::abs(static_cast<double>(got[i]) - cur_out[i]));
                    }
                }
                fmt::print("DIAGPLACE case={} {} pass={} pcc={:.5f} ab_maxdiff={:.6f}\n", c.label, tag, pass, p, ab);
                EXPECT_GT(p, 0.999) << "placement case=" << c.label << " " << tag << " pass=" << pass;
                // Sm=1: placement is a true no-op -> must be bit-identical (catches accidental coord changes).
                // Sm>1: PARETO ring recompute on new coords can perturb accumulation rounding (bf16 ULP), so
                // only PCC is required (bit-identity checked separately by RingOrderCorrectness on fixed coords).
                if (ab >= 0.0 && c.Sm == 1u) {
                    EXPECT_EQ(ab, 0.0) << tag << " != current (Sm=1 no-op; case=" << c.label << " pass=" << pass << ")";
                }
            }
        }
    }
}

// Picker/planner parity (host-only, no device): the auto-picker must never accept an (Mt,Kt,Nt) that the
// planner's bank-interval constraint rejects, else config=None can select a config that FATALs later in
// build_plan(). Both now share nt_width_shard_feasible(); this guards against future drift. Direction
// tested: bank-infeasible Nt => auto_select_config must throw (no feasible config).
TEST(RegimeAPickerPlannerParity, BankFeasibilityShared) {
    namespace plan = ttnn::operations::experimental::regime_a_matmul::plan;
    // Kt/Mt chosen generously so Nt is the only binding constraint across the sweep.
    const uint32_t Mt = 1u, Kt = 192u;
    int checked_infeasible = 0;
    for (uint32_t Nt = 1u; Nt <= 64u; ++Nt) {
        const bool bank_feasible = plan::nt_width_shard_feasible(Nt);
        bool picker_ok = true;
        try {
            (void)ttnn::experimental::prim::auto_select_config(Mt, Kt, Nt);
        } catch (...) {
            picker_ok = false;
        }
        if (!bank_feasible) {
            ++checked_infeasible;
            EXPECT_FALSE(picker_ok) << "Nt=" << Nt << " is bank-infeasible but the picker accepted it";
        }
    }
    EXPECT_GT(checked_infeasible, 0) << "expected some bank-infeasible Nt in [1,64]";
}

}  // namespace ttnn::experimental::regime_a_matmul::diag_test
