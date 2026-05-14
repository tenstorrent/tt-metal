// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/common.hpp"
#include "ops/matmul.hpp"
#include "ops/eltwise_add.hpp"
#include "ops/eltwise_mul.hpp"
#include "ops/gelu.hpp"
#include "ops/softmax.hpp"
#include "ops/layernorm.hpp"
#include "ops/column_slice.hpp"
#include "ops/column_write.hpp"
#include "ops/transpose.hpp"
#include "model/vit_model.hpp"
#include "weights/weight_loader.hpp"

#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace vit;

bool test_matmul(MeshContext& ctx, uint32_t M, uint32_t N, uint32_t K) {
    fmt::print("=== Test matmul: M={} N={} K={} ===\n", M, N, K);

    uint32_t Mt = num_tiles(M);
    uint32_t Kt = num_tiles(K);
    uint32_t Nt = num_tiles(N);
    uint32_t M_padded = Mt * TILE_H;
    uint32_t K_padded = Kt * TILE_W;
    uint32_t N_padded = Nt * TILE_W;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<bfloat16> a_data(M_padded * K_padded, bfloat16(0.0f));
    std::vector<bfloat16> b_data(K_padded * N_padded, bfloat16(0.0f));
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < K; j++)
            a_data[i * K_padded + j] = bfloat16(dist(rng));
    for (uint32_t i = 0; i < K; i++)
        for (uint32_t j = 0; j < N; j++)
            b_data[i * N_padded + j] = bfloat16(dist(rng));

    std::vector<bfloat16> golden(M_padded * N_padded, bfloat16(0.0f));
    golden_matmul(a_data, b_data, golden, M_padded, N_padded, K_padded);

    auto a_tiled = tilize_nfaces(a_data, M_padded, K_padded);
    auto b_tiled = tilize_nfaces(b_data, K_padded, N_padded);

    auto src0_buf = create_dram_buffer(ctx, Mt * Kt * SINGLE_TILE_SIZE);
    auto src1_buf = create_dram_buffer(ctx, Kt * Nt * SINGLE_TILE_SIZE);
    auto dst_buf = create_dram_buffer(ctx, Mt * Nt * SINGLE_TILE_SIZE);

    write_to_device(ctx, src0_buf, a_tiled);
    write_to_device(ctx, src1_buf, b_tiled);

    matmul_op(ctx, src0_buf, src1_buf, dst_buf, Mt, Kt, Nt);

    std::vector<bfloat16> result(M_padded * N_padded);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, M_padded, N_padded);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.97f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool test_eltwise_add(MeshContext& ctx, uint32_t M, uint32_t N) {
    fmt::print("=== Test eltwise_add: M={} N={} ===\n", M, N);

    uint32_t Mt = num_tiles(M);
    uint32_t Nt = num_tiles(N);
    uint32_t M_padded = Mt * TILE_H;
    uint32_t N_padded = Nt * TILE_W;
    uint32_t total = M_padded * N_padded;
    uint32_t n_tiles = Mt * Nt;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bfloat16> a_data(total), b_data(total), golden(total);
    for (uint32_t i = 0; i < total; i++) {
        a_data[i] = bfloat16(dist(rng));
        b_data[i] = bfloat16(dist(rng));
        golden[i] = bfloat16(static_cast<float>(a_data[i]) + static_cast<float>(b_data[i]));
    }

    auto a_tiled = tilize_nfaces(a_data, M_padded, N_padded);
    auto b_tiled = tilize_nfaces(b_data, M_padded, N_padded);

    auto src0_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    auto src1_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    auto dst_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);

    write_to_device(ctx, src0_buf, a_tiled);
    write_to_device(ctx, src1_buf, b_tiled);

    eltwise_add_op(ctx, src0_buf, src1_buf, dst_buf, n_tiles);

    std::vector<bfloat16> result(total);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, M_padded, N_padded);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.99f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool test_gelu(MeshContext& ctx, uint32_t M, uint32_t N) {
    fmt::print("=== Test GELU: M={} N={} ===\n", M, N);

    uint32_t Mt = num_tiles(M);
    uint32_t Nt = num_tiles(N);
    uint32_t M_padded = Mt * TILE_H;
    uint32_t N_padded = Nt * TILE_W;
    uint32_t total = M_padded * N_padded;
    uint32_t n_tiles = Mt * Nt;

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<bfloat16> input(total), golden(total);
    for (uint32_t i = 0; i < total; i++) {
        float x = dist(rng);
        input[i] = bfloat16(x);
        golden[i] = bfloat16(0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f))));
    }

    auto input_tiled = tilize_nfaces(input, M_padded, N_padded);

    auto src_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    auto dst_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);

    write_to_device(ctx, src_buf, input_tiled);

    gelu_op(ctx, src_buf, dst_buf, n_tiles);

    std::vector<bfloat16> result(total);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, M_padded, N_padded);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.97f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool test_softmax(MeshContext& ctx, uint32_t M, uint32_t N) {
    fmt::print("=== Test softmax: M={} N={} ===\n", M, N);

    uint32_t Mt = num_tiles(M);
    uint32_t Nt = num_tiles(N);
    uint32_t M_padded = Mt * TILE_H;
    uint32_t N_padded = Nt * TILE_W;
    uint32_t total = M_padded * N_padded;
    uint32_t n_tiles = Mt * Nt;

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bfloat16> input(total);
    std::vector<float> input_f(total);
    for (uint32_t i = 0; i < total; i++) {
        float x = dist(rng);
        input[i] = bfloat16(x);
        input_f[i] = static_cast<float>(input[i]);
    }

    std::vector<bfloat16> golden(total);
    for (uint32_t row = 0; row < M_padded; row++) {
        float max_val = -1e30f;
        for (uint32_t col = 0; col < N_padded; col++)
            max_val = std::max(max_val, input_f[row * N_padded + col]);

        float sum = 0.0f;
        std::vector<float> exps(N_padded);
        for (uint32_t col = 0; col < N_padded; col++) {
            exps[col] = std::exp(input_f[row * N_padded + col] - max_val);
            sum += exps[col];
        }
        for (uint32_t col = 0; col < N_padded; col++)
            golden[row * N_padded + col] = bfloat16(exps[col] / sum);
    }

    auto input_tiled = tilize_nfaces(input, M_padded, N_padded);

    auto src_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    auto dst_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);

    write_to_device(ctx, src_buf, input_tiled);

    softmax_op(ctx, src_buf, dst_buf, Mt, Nt);

    std::vector<bfloat16> result(total);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, M_padded, N_padded);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.95f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool test_layernorm(MeshContext& ctx, uint32_t M, uint32_t N) {
    fmt::print("=== Test layernorm: M={} N={} ===\n", M, N);

    uint32_t Mt = num_tiles(M);
    uint32_t Nt = num_tiles(N);
    uint32_t M_padded = Mt * TILE_H;
    uint32_t N_padded = Nt * TILE_W;
    uint32_t total = M_padded * N_padded;
    uint32_t n_tiles = Mt * Nt;
    float eps = 1e-6f;

    std::mt19937 rng(321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> param_dist(0.5f, 1.5f);

    std::vector<bfloat16> input(total);
    std::vector<float> input_f(total);
    for (uint32_t i = 0; i < total; i++) {
        float x = dist(rng);
        input[i] = bfloat16(x);
        input_f[i] = static_cast<float>(input[i]);
    }

    // gamma and beta: [1, N_padded], broadcast-row format
    // In tilized broadcast-row format, each tile row has the same values
    std::vector<bfloat16> gamma_row(N_padded), beta_row(N_padded);
    std::vector<float> gamma_f(N_padded), beta_f(N_padded);
    for (uint32_t j = 0; j < N_padded; j++) {
        float g = param_dist(rng);
        float b = dist(rng) * 0.1f;
        gamma_row[j] = bfloat16(g);
        beta_row[j] = bfloat16(b);
        gamma_f[j] = static_cast<float>(gamma_row[j]);
        beta_f[j] = static_cast<float>(beta_row[j]);
    }

    // Create broadcast-row tiles: each row of the tile has the same values
    std::vector<bfloat16> gamma_bcast(total), beta_bcast(total);
    for (uint32_t row = 0; row < M_padded; row++) {
        for (uint32_t col = 0; col < N_padded; col++) {
            gamma_bcast[row * N_padded + col] = gamma_row[col];
            beta_bcast[row * N_padded + col] = beta_row[col];
        }
    }

    // Golden LayerNorm computation
    std::vector<bfloat16> golden(total);
    for (uint32_t row = 0; row < M_padded; row++) {
        float mean = 0.0f;
        for (uint32_t col = 0; col < N_padded; col++)
            mean += input_f[row * N_padded + col];
        mean /= N_padded;

        float var = 0.0f;
        for (uint32_t col = 0; col < N_padded; col++) {
            float d = input_f[row * N_padded + col] - mean;
            var += d * d;
        }
        var /= N_padded;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (uint32_t col = 0; col < N_padded; col++) {
            float norm = (input_f[row * N_padded + col] - mean) * inv_std;
            golden[row * N_padded + col] = bfloat16(norm * gamma_f[col] + beta_f[col]);
        }
    }

    auto input_tiled = tilize_nfaces(input, M_padded, N_padded);
    auto gamma_tiled = tilize_nfaces(gamma_bcast, M_padded, N_padded);
    auto beta_tiled = tilize_nfaces(beta_bcast, M_padded, N_padded);

    auto src_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    auto gamma_buf = create_dram_buffer(ctx, Nt * SINGLE_TILE_SIZE);
    auto beta_buf = create_dram_buffer(ctx, Nt * SINGLE_TILE_SIZE);
    auto dst_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);

    write_to_device(ctx, src_buf, input_tiled);
    // For gamma/beta, only upload Nt tiles (one row of tiles)
    std::vector<bfloat16> gamma_row_tiles(Nt * TILE_HW), beta_row_tiles(Nt * TILE_HW);
    // Tilize a single row of tiles: [32, N_padded]
    std::vector<bfloat16> gamma_single(TILE_H * N_padded), beta_single(TILE_H * N_padded);
    for (uint32_t r = 0; r < TILE_H; r++) {
        for (uint32_t c = 0; c < N_padded; c++) {
            gamma_single[r * N_padded + c] = gamma_row[c];
            beta_single[r * N_padded + c] = beta_row[c];
        }
    }
    gamma_row_tiles = tilize_nfaces(gamma_single, TILE_H, N_padded);
    beta_row_tiles = tilize_nfaces(beta_single, TILE_H, N_padded);

    write_to_device(ctx, gamma_buf, gamma_row_tiles);
    write_to_device(ctx, beta_buf, beta_row_tiles);

    layernorm_op(ctx, src_buf, gamma_buf, beta_buf, dst_buf, Mt, Nt, eps);

    std::vector<bfloat16> result(total);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, M_padded, N_padded);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.95f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool test_eltwise_mul(MeshContext& ctx, uint32_t M, uint32_t N) {
    fmt::print("=== Test eltwise_mul: M={} N={} ===\n", M, N);

    uint32_t Mt = num_tiles(M);
    uint32_t Nt = num_tiles(N);
    uint32_t M_padded = Mt * TILE_H;
    uint32_t N_padded = Nt * TILE_W;
    uint32_t total = M_padded * N_padded;
    uint32_t n_tiles = Mt * Nt;

    std::mt19937 rng(555);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bfloat16> a_data(total), b_data(total), golden(total);
    for (uint32_t i = 0; i < total; i++) {
        a_data[i] = bfloat16(dist(rng));
        b_data[i] = bfloat16(dist(rng));
        golden[i] = bfloat16(static_cast<float>(a_data[i]) * static_cast<float>(b_data[i]));
    }

    auto a_tiled = tilize_nfaces(a_data, M_padded, N_padded);
    auto b_tiled = tilize_nfaces(b_data, M_padded, N_padded);

    auto src0_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    auto src1_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);
    auto dst_buf = create_dram_buffer(ctx, n_tiles * SINGLE_TILE_SIZE);

    write_to_device(ctx, src0_buf, a_tiled);
    write_to_device(ctx, src1_buf, b_tiled);

    eltwise_mul_op(ctx, src0_buf, src1_buf, dst_buf, n_tiles);

    std::vector<bfloat16> result(total);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, M_padded, N_padded);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.99f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool test_column_slice(MeshContext& ctx) {
    fmt::print("=== Test column_slice ===\n");

    constexpr uint32_t Mt = 7, total_Wt = 18, start_col = 6, slice_Wt = 6;
    constexpr uint32_t M = Mt * TILE_H, total_W = total_Wt * TILE_W, slice_W = slice_Wt * TILE_W;
    uint32_t total_in = M * total_W;
    uint32_t total_out = M * slice_W;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bfloat16> input(total_in);
    for (auto& v : input) v = bfloat16(dist(rng));

    std::vector<bfloat16> golden(total_out);
    for (uint32_t r = 0; r < M; r++) {
        for (uint32_t c = 0; c < slice_W; c++) {
            golden[r * slice_W + c] = input[r * total_W + start_col * TILE_W + c];
        }
    }

    auto in_tiled = tilize_nfaces(input, M, total_W);
    auto src_buf = create_dram_buffer(ctx, Mt * total_Wt * SINGLE_TILE_SIZE);
    auto dst_buf = create_dram_buffer(ctx, Mt * slice_Wt * SINGLE_TILE_SIZE);
    write_to_device(ctx, src_buf, in_tiled);

    column_slice_op(ctx, src_buf, dst_buf, Mt, total_Wt, start_col, slice_Wt);

    std::vector<bfloat16> result(total_out);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, M, slice_W);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.999f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool test_transpose(MeshContext& ctx) {
    fmt::print("=== Test transpose_2d: [224,64] -> [64,224] ===\n");

    constexpr uint32_t Mt = 7, Nt = 2;
    constexpr uint32_t M = Mt * TILE_H, N = Nt * TILE_W;
    uint32_t total = M * N;

    std::mt19937 rng(888);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<bfloat16> input(total);
    for (auto& v : input) v = bfloat16(dist(rng));

    std::vector<bfloat16> golden(total);
    for (uint32_t r = 0; r < M; r++) {
        for (uint32_t c = 0; c < N; c++) {
            golden[c * M + r] = input[r * N + c];
        }
    }

    auto in_tiled = tilize_nfaces(input, M, N);
    auto src_buf = create_dram_buffer(ctx, Mt * Nt * SINGLE_TILE_SIZE);
    auto dst_buf = create_dram_buffer(ctx, Nt * Mt * SINGLE_TILE_SIZE);
    write_to_device(ctx, src_buf, in_tiled);

    transpose_2d_op(ctx, src_buf, dst_buf, Mt, Nt);

    std::vector<bfloat16> result(total);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, N, M);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.99f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

bool test_column_write(MeshContext& ctx) {
    fmt::print("=== Test column_write ===\n");

    constexpr uint32_t Mt = 7, total_Wt = 6, Ht = 2;
    constexpr uint32_t M = Mt * TILE_H, total_W = total_Wt * TILE_W, slice_W = Ht * TILE_W;

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Write 3 slices to concat buffer (simulating 3 attention heads)
    std::vector<bfloat16> golden(M * total_W, bfloat16(0.0f));
    std::vector<std::vector<bfloat16>> slices(3);

    for (uint32_t h = 0; h < 3; h++) {
        slices[h].resize(M * slice_W);
        for (auto& v : slices[h]) v = bfloat16(dist(rng));
        for (uint32_t r = 0; r < M; r++) {
            for (uint32_t c = 0; c < slice_W; c++) {
                golden[r * total_W + h * slice_W + c] = slices[h][r * slice_W + c];
            }
        }
    }

    auto dst_buf = create_dram_buffer(ctx, Mt * total_Wt * SINGLE_TILE_SIZE);

    for (uint32_t h = 0; h < 3; h++) {
        auto slice_tiled = tilize_nfaces(slices[h], M, slice_W);
        auto slice_buf = create_dram_buffer(ctx, Mt * Ht * SINGLE_TILE_SIZE);
        write_to_device(ctx, slice_buf, slice_tiled);
        column_write_op(ctx, slice_buf, dst_buf, Mt, total_Wt, h * Ht, Ht);
    }

    std::vector<bfloat16> result(M * total_W);
    read_from_device(ctx, result, dst_buf);
    result = untilize_nfaces(result, M, total_W);

    float pcc = compute_pcc(golden, result);
    fmt::print("  PCC = {}\n", pcc);
    bool pass = pcc > 0.999f;
    fmt::print("  {}\n", pass ? "PASS" : "FAIL");
    return pass;
}

void run_op_tests(MeshContext& ctx) {
    bool pass = true;

    fmt::print("\n========== MATMUL TESTS ==========\n");
    pass &= test_matmul(ctx, 32, 32, 32);
    pass &= test_matmul(ctx, 64, 64, 64);
    pass &= test_matmul(ctx, 224, 192, 768);
    pass &= test_matmul(ctx, 224, 768, 192);

    fmt::print("\n========== ELTWISE ADD TESTS ==========\n");
    pass &= test_eltwise_add(ctx, 224, 192);

    fmt::print("\n========== GELU TESTS ==========\n");
    pass &= test_gelu(ctx, 224, 768);

    fmt::print("\n========== SOFTMAX TESTS ==========\n");
    pass &= test_softmax(ctx, 224, 224);

    fmt::print("\n========== LAYERNORM TESTS ==========\n");
    pass &= test_layernorm(ctx, 224, 192);

    fmt::print("\n========== ELTWISE MUL TESTS ==========\n");
    pass &= test_eltwise_mul(ctx, 224, 224);

    fmt::print("\n========== COLUMN SLICE TESTS ==========\n");
    pass &= test_column_slice(ctx);

    fmt::print("\n========== TRANSPOSE TESTS ==========\n");
    pass &= test_transpose(ctx);

    fmt::print("\n========== COLUMN WRITE TESTS ==========\n");
    pass &= test_column_write(ctx);

    if (pass) {
        fmt::print("\nAll op tests PASSED\n");
    } else {
        TT_THROW("Some op tests FAILED");
    }
}

void run_vit_inference(MeshContext& ctx, const std::string& weight_dir, const std::string& image_path) {
    // Load weights
    auto weights = load_vit_weights(ctx, weight_dir);

    // Load test image [3, 224, 224]
    std::vector<float> image(3 * 224 * 224);
    {
        std::ifstream f(image_path, std::ios::binary);
        if (!f.is_open()) {
            TT_THROW("Failed to open image file: {}", image_path);
        }
        f.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(float));
    }

    fmt::print("\nRunning ViT Tiny inference...\n");
    auto logits = vit_forward(ctx, image, weights);

    // Top-5 predictions
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
        [&logits](int a, int b) { return logits[a] > logits[b]; });

    fmt::print("\nTop-5 predictions:\n");
    for (int i = 0; i < 5; i++) {
        fmt::print("  Class {}: {:.4f}\n", indices[i], logits[indices[i]]);
    }

    // Compare with reference if available
    std::string ref_path = weight_dir + "/reference_logits.bin";
    std::ifstream ref_file(ref_path, std::ios::binary);
    if (ref_file.is_open()) {
        std::vector<float> ref_logits(1000);
        ref_file.read(reinterpret_cast<char*>(ref_logits.data()), 1000 * sizeof(float));

        // Compute PCC between logits
        float x_mean = 0, y_mean = 0;
        for (int i = 0; i < 1000; i++) {
            x_mean += logits[i];
            y_mean += ref_logits[i];
        }
        x_mean /= 1000; y_mean /= 1000;
        float cov = 0, xv = 0, yv = 0;
        for (int i = 0; i < 1000; i++) {
            float xd = logits[i] - x_mean, yd = ref_logits[i] - y_mean;
            cov += xd * yd; xv += xd * xd; yv += yd * yd;
        }
        float pcc = cov / (std::sqrt(xv) * std::sqrt(yv));
        fmt::print("\nLogits PCC vs PyTorch reference: {}\n", pcc);

        // Check top-5 match
        std::vector<int> ref_indices(1000);
        std::iota(ref_indices.begin(), ref_indices.end(), 0);
        std::partial_sort(ref_indices.begin(), ref_indices.begin() + 5, ref_indices.end(),
            [&ref_logits](int a, int b) { return ref_logits[a] > ref_logits[b]; });
        fmt::print("Reference top-5: {}, {}, {}, {}, {}\n",
            ref_indices[0], ref_indices[1], ref_indices[2], ref_indices[3], ref_indices[4]);
    }
}

int main(int argc, char** argv) {
    try {
        auto ctx = MeshContext::create(0);

        if (argc >= 3) {
            std::string weight_dir = argv[1];
            std::string image_path = argv[2];
            run_vit_inference(ctx, weight_dir, image_path);
        } else {
            run_op_tests(ctx);
        }

        ctx.mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Failed with exception: {}\n", e.what());
        throw;
    }
    return 0;
}
