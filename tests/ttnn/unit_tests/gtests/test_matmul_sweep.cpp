// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <array>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

#include "impl/emulation/emulated_run_stats.hpp"

namespace ttnn::operations::matmul::test {

// Shape parameters: M, K, N
struct MatmulShape {
    uint32_t M;
    uint32_t K;
    uint32_t N;
};

std::ostream& operator<<(std::ostream& os, const MatmulShape& s) { return os << s.M << "x" << s.K << "x" << s.N; }

class MatmulSweepFixture : public ttnn::TTNNFixtureWithSuiteDevice<MatmulSweepFixture>,
                           public ::testing::WithParamInterface<MatmulShape> {};

// Compute per-test timeout in seconds based on shape dimensions
static int compute_timeout(uint32_t M, uint32_t K, uint32_t N) {
    uint32_t max_dim = std::max({M, K, N});
    if (max_dim <= 128) {
        return 30;
    }
    if (max_dim <= 512) {
        return 60;
    }
    if (max_dim <= 1024) {
        return 120;
    }
    return 300;
}

TEST_P(MatmulSweepFixture, MatmulSweep) {
    auto [M, K, N] = GetParam();
    auto& device = *device_;

    uint32_t out_tiles = (M / 32) * (N / 32);
    int timeout_sec = compute_timeout(M, K, N);

    // A = ones(1,1,M,K), B = ones(1,1,K,N)
    // C = A * B => every element = K
    std::array<uint32_t, 4> a_dims = {1, 1, M, K};
    std::array<uint32_t, 4> b_dims = {1, 1, K, N};
    std::array<uint32_t, 4> c_dims = {1, 1, M, N};

    const auto a_tensor = ttnn::ones(ttnn::Shape(a_dims), DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    const auto b_tensor = ttnn::ones(ttnn::Shape(b_dims), DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    // Run matmul with timeout
    auto t_start = std::chrono::steady_clock::now();

    auto future =
        std::async(std::launch::async, [&]() { return ttnn::operations::matmul::matmul(a_tensor, b_tensor); });

    auto timeout = std::chrono::seconds(timeout_sec);
    if (future.wait_for(timeout) == std::future_status::timeout) {
        FAIL() << "Matmul " << M << "x" << K << "x" << N << " timed out after " << timeout_sec << "s";
        return;
    }
    auto c_tensor = future.get();

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Get execution metadata
    const auto& stats = tt::tt_metal::emule::get_last_emulated_run_stats();
    std::string kernels_str;
    for (size_t i = 0; i < stats.kernel_paths.size(); ++i) {
        if (i > 0) {
            kernels_str += ", ";
        }
        kernels_str += stats.kernel_paths[i];
    }

    // Verify result: expected = full(c_dims, float(K))
    float expected_val = static_cast<float>(K);
    const auto expected = ttnn::full(ttnn::Shape(c_dims), expected_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);

    // Tolerance scales with K for bfloat16 accumulation error
    float atol = std::max(1.0f, K * 0.01f);
    float rtol = 0.05f;

    bool pass = ttnn::allclose<::bfloat16>(ttnn::from_device(expected), ttnn::from_device(c_tensor), atol, rtol);

    // Print result row
    std::ostringstream row;
    row << "| " << std::setw(5) << M << " | " << std::setw(5) << K << " | " << std::setw(5) << N << " | "
        << std::setw(6) << out_tiles << " | " << std::setw(5) << stats.num_cores << " | " << std::setw(6)
        << (pass ? "PASS" : "FAIL") << " | " << std::setw(8) << std::fixed << std::setprecision(1) << elapsed_ms
        << " | " << kernels_str << " |";
    std::cout << row.str() << std::endl;

    EXPECT_TRUE(pass) << "Matmul " << M << "x" << K << "x" << N << " result mismatch (expected all " << expected_val
                      << ")";
}

// Print table header before first test
static bool print_header() {
    std::cout << "+-------+-------+-------+--------+-------+--------+----------+"
                 "-------------------------------------------+"
              << std::endl;
    std::cout << "|     M |     K |     N |  tiles | cores | status | time(ms) |"
                 " kernels                                   |"
              << std::endl;
    std::cout << "+-------+-------+-------+--------+-------+--------+----------+"
                 "-------------------------------------------+"
              << std::endl;
    return true;
}
static bool header_printed = print_header();

INSTANTIATE_TEST_SUITE_P(
    MatmulSweep,
    MatmulSweepFixture,
    ::testing::Values(
        MatmulShape{32, 32, 32},
        MatmulShape{32, 64, 32},
        MatmulShape{64, 64, 64},
        MatmulShape{128, 128, 128},
        MatmulShape{256, 256, 256},
        MatmulShape{512, 512, 512},
        MatmulShape{1024, 1024, 1024},
        MatmulShape{2048, 2048, 2048},
        MatmulShape{64, 1024, 64},
        MatmulShape{2048, 32, 2048},
        MatmulShape{32, 2048, 32},
        MatmulShape{128, 512, 256},
        MatmulShape{320, 384, 320},
        MatmulShape{512, 1024, 512}),
    [](const ::testing::TestParamInfo<MatmulShape>& info) {
        return std::to_string(info.param.M) + "x" + std::to_string(info.param.K) + "x" + std::to_string(info.param.N);
    });

}  // namespace ttnn::operations::matmul::test
