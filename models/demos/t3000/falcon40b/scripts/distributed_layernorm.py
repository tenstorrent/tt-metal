# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    get_atol_rtol_pcc,
)


def basic_layernorm(x, gamma, beta, epsilon=1e-5):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True)

    # Normalize the input
    x_hat = (x - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def compute_mean_and_variance(chunk):
    n = chunk.shape[-1]
    mean = torch.mean(chunk, dim=-1, keepdim=True)
    variance = torch.var(chunk, dim=-1, keepdim=True)
    return mean, variance, n


def combine_statistics(mean1, var1, count1, mean2, var2, count2):
    combined_count = count1 + count2
    delta = mean2 - mean1
    combined_mean = (count1 * mean1 + count2 * mean2) / combined_count
    combined_variance = (count1 * var1 + count2 * var2 + delta**2 * count1 * count2 / combined_count) / combined_count
    return combined_mean, combined_variance, combined_count


def chunked_layer_norm_direct(x, gamma, beta, chunk_size=1024, epsilon=1e-5):
    total_mean = 0
    total_variance = 0
    total_count = 0

    # Process each chunk
    for i in range(0, x.shape[-1], chunk_size):
        chunk = x[:, i : i + chunk_size]
        chunk_mean, chunk_variance, chunk_count = compute_mean_and_variance(chunk)

        # Combine statistics from the chunk with the total statistics
        total_mean, total_variance, total_count = combine_statistics(
            total_mean, total_variance, total_count, chunk_mean, chunk_variance, chunk_count
        )

    # Normalize the input
    x_hat = (x - total_mean) / np.sqrt(total_variance + epsilon)

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def layer_norm_welford(x, gamma, beta, epsilon=1e-5):
    # Initialize mean and M2 for Welford's algorithm
    mean = torch.zeros((x.shape[0], 1), dtype=x.dtype)
    M2 = torch.zeros((x.shape[0], 1), dtype=x.dtype)
    count = 0

    # First pass to compute mean and variance using Welford's algorithm
    for i in range(x.shape[-1]):
        value = x[:, i : i + 1]
        count += 1
        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        M2 += delta * delta2

    variance = M2 / count

    # Normalize the input
    x_hat = (x - mean) / (variance + epsilon) ** 0.5

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def combine_statistics_welford(n_a, avg_a, M2_a, n_b, avg_b, M2_b):
    n = n_a + n_b
    delta = avg_b - avg_a
    avg_ab = avg_a + delta * n_b / n
    M2_ab = M2_a + M2_b + delta**2 * n_a * n_b / n
    return n, avg_ab, M2_ab


def layer_norm_welford_chunked(x, gamma, beta, chunk_size=1024, epsilon=1e-5):
    mean = torch.zeros((x.shape[0], 1), dtype=x.dtype)
    M2 = torch.zeros((x.shape[0], 1), dtype=x.dtype)
    count = 0

    # Process each chunk
    for c in range(0, x.shape[-1], chunk_size):
        mean_c = torch.zeros((x.shape[0], 1), dtype=x.dtype)
        M2_c = torch.zeros((x.shape[0], 1), dtype=x.dtype)
        count_c = 0
        chunk = x[:, c : c + chunk_size]
        for i in range(chunk.shape[-1]):
            value = chunk[:, i : i + 1]
            count_c += 1
            delta = value - mean_c
            mean_c += delta / count_c
            delta2 = value - mean_c
            M2_c += delta * delta2

        count, mean, M2 = combine_statistics_welford(count, mean, M2, count_c, mean_c, M2_c)

    variance = M2 / count

    # Normalize the input
    x_hat = (x - mean) / (variance + epsilon) ** 0.5

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def layer_norm_decomp_chunked(x, gamma, beta, chunk_size=1024, epsilon=1e-5):
    meanx = torch.zeros((x.shape[0], 1), dtype=x.dtype)
    meanx2 = torch.zeros((x.shape[0], 1), dtype=x.dtype)
    count = 0

    # Process each chunk
    num_chunks = x.shape[-1] // chunk_size
    for i in range(0, x.shape[-1], chunk_size):
        chunk = x[:, i : i + chunk_size]
        count += chunk.shape[-1]

        meanx += torch.mean(chunk, dim=-1, keepdim=True)
        meanx2 += torch.mean(torch.square(chunk), dim=-1, keepdim=True)

    mean = meanx / num_chunks
    meanx2 = meanx2 / num_chunks
    var = meanx2 - torch.square(mean)

    # Normalize the input
    x_hat = (x - mean) / torch.sqrt(var + epsilon)

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def layer_norm_decomp(x, gamma, beta, epsilon=1e-5):
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = x - mean
    var = torch.mean(torch.square(var))
    x_hat = (x - mean) / torch.sqrt(var + epsilon)
    y = gamma * x_hat + beta
    return y


def distributed_layernorm_poc(x, gamma, beta, chunk_size=1024, epsilon=1e-5):
    # Prepare inputs for distributed processing
    num_chunks = x.shape[-1] // chunk_size
    xs = []
    gammas = []
    betas = []
    for i in range(0, x.shape[-1], chunk_size):
        x_chunk = x[:, i : i + chunk_size]
        xs.append(x_chunk)

        gamma_chunk = gamma[i : i + chunk_size]
        gammas.append(gamma_chunk)

        beta_chunk = beta[i : i + chunk_size]
        betas.append(beta_chunk)

    count = []
    meanx = []
    meanx2 = []
    # Distributed processing
    for chunk in xs:
        count_local = chunk.shape[-1]
        meanx_local = torch.mean(chunk, dim=-1, keepdim=True)
        meanx2_local = torch.mean(torch.square(chunk), dim=-1, keepdim=True)

        count.append(count_local)
        meanx.append(meanx_local)
        meanx2.append(meanx2_local)

    # AllReduce cound, meanx, meanx2
    count = torch.torch.FloatTensor(count).sum(dim=0)
    mean = torch.stack(meanx, dim=0).sum(dim=0) / num_chunks
    meanx2 = torch.stack(meanx2, dim=0).sum(dim=0) / num_chunks
    var = meanx2 - torch.square(mean)

    # Distributed processing
    ys = []
    for i in range(num_chunks):
        # Normalize the input
        x_hat_local = (xs[i] - mean) / torch.sqrt(var + epsilon)

        # Scale and shift
        y_local = gammas[i] * x_hat_local + betas[i]
        ys.append(y_local)

    # Post processing: concat ys
    y = torch.cat(ys, dim=-1)

    return y


def main():
    S = 2048
    H = 8192

    input_shape = (S, H)

    x = torch.randn(input_shape, dtype=torch.float32) * 4.0  # Example input

    gamma = torch.randn(H)  # Scale parameter
    beta = torch.randn(H)  # Shift parameter

    # PyTorch LayerNorm
    layer_norm = torch.nn.LayerNorm(H, elementwise_affine=True)
    layer_norm.eval()
    layer_norm.weight.data = gamma
    layer_norm.bias.data = beta
    normalized_output_torch = layer_norm(x)

    # Custom LayerNorm
    basic_layernorm_output = basic_layernorm(x, gamma, beta)
    normalized_output_custom = chunked_layer_norm_direct(x, gamma, beta)
    welford_output = layer_norm_welford(x, gamma, beta)
    tt_output = layer_norm_decomp(x, gamma, beta)
    decomp_chunked_output = layer_norm_decomp_chunked(x, gamma, beta)
    welford_chunked_output = layer_norm_welford_chunked(x, gamma, beta)
    distributed_layernorm_output = distributed_layernorm_poc(x, gamma, beta)

    # Comparison

    print("\nBasic LayerNorm")
    cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(basic_layernorm_output, normalized_output_torch)
    print(output_str)

    print("\nCustom Chunked LayerNorm")
    cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(normalized_output_custom, normalized_output_torch)
    print(output_str)

    print("\nWelford LayerNorm")
    cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(welford_output, normalized_output_torch)
    print(output_str)

    print("\nTT LayerNorm")
    cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(tt_output, normalized_output_torch)
    print(output_str)

    print("\nDecomposed Chunked LayerNorm")
    cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(decomp_chunked_output, normalized_output_torch)
    print(output_str)

    print("\nWelford Chunked LayerNorm")
    cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(welford_chunked_output, normalized_output_torch)
    print(output_str)

    print("\nDistributed LayerNorm")
    cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(distributed_layernorm_output, normalized_output_torch)
    print(output_str)


if __name__ == "__main__":
    main()
