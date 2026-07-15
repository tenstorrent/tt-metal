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

    // Correctness is checked for the public path (mask 0) AND the DIAG_IN0_SCATTER variant (32), both of
    // which must produce the exact result; the pure ablations (garbage output) are not checked.
    if (mask == 0 || mask == 32) {
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

}  // namespace ttnn::experimental::regime_a_matmul::diag_test
