// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Offline CPU preprocessing, adapted from the frozen quantization experiment.
// TT B-format rounding follows blockfloat_common.cpp, Apache-2.0,
// Copyright 2026 Tenstorrent USA, Inc. Added exponent search and GPTQ loops.
// Compile with -ffp-contract=off; no device kernel is built or changed here.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

static int quantize_group(const float* x, float* out, int bits, const int* deltas, int count) {
    uint32_t raw[16];
    int top = 0;
    for (int j = 0; j < 16; ++j) {
        if (!std::isfinite(x[j])) {
            std::fill(out, out + 16, std::numeric_limits<float>::quiet_NaN());
            return 0;  // Python rejects non-finite compensation; never use uninitialized output.
        }
        std::memcpy(raw + j, x + j, sizeof(float));
        top = std::max(top, int((raw[j] >> 23) & 255));
    }
    double best = std::numeric_limits<double>::infinity();
    int choice = 0;
    for (int c = 0; c < count; ++c) {
        const int shared = c == 0 ? top : std::clamp(top + deltas[c], 1, 254);
        float q[16];
        double error = 0;
        const int shift = 24 - (bits - 1);
        for (int j = 0; j < 16; ++j) {
            const int exponent = (raw[j] >> 23) & 255;
            const int alignment = shared - exponent;
            uint32_t mantissa = exponent ? ((raw[j] & 0x7fffff) | 0x800000) : 0;
            mantissa = alignment >= 0 ? mantissa >> std::min(alignment, 31) : mantissa << std::min(-alignment, 2);
            uint32_t magnitude = mantissa >> shift;
            const uint32_t remainder = mantissa & ((1u << shift) - 1);
            const uint32_t half = 1u << (shift - 1);
            magnitude += remainder > half || (remainder == half && (magnitude & 1));
            magnitude = std::min(magnitude, (1u << (bits - 1)) - 1);
            q[j] = std::ldexp(float(magnitude), shared - 127 - (bits - 2));
            if (raw[j] >> 31) {
                q[j] = -q[j];
            }
            const double difference = double(x[j]) - q[j];
            error += difference * difference;
        }
        if (error < best) {
            best = error;
            choice = c;
            std::memcpy(out, q, 16 * sizeof(float));
        }
    }
    return choice;
}

extern "C" void search_bfp(
    const float* x, float* out, int64_t groups, int bits, const int* deltas, int count, int64_t* counts, int threads) {
#pragma omp parallel num_threads(threads) if (groups > 8192)
    {
        int64_t local[4] = {};
#pragma omp for schedule(static)
        for (int64_t group = 0; group < groups; ++group) {
            ++local[quantize_group(x + 16 * group, out + 16 * group, bits, deltas, count)];
        }
#pragma omp critical
        {
            for (int c = 0; c < count; ++c) {
                counts[c] += local[c];
            }
        }
    }
}

extern "C" void gptq_block(
    float* block,
    const float* upper,
    float* quantized,
    float* errors,
    int64_t rows,
    int width,
    const int* deltas,
    int count,
    int64_t* counts,
    int threads) {
#pragma omp parallel num_threads(threads)
    {
        int64_t local_counts[4] = {};
#pragma omp for schedule(static)
        for (int64_t group = 0; group < rows / 16; ++group) {
            const int64_t first = group * 16;
            for (int column = 0; column < width; ++column) {
                float values[16], rounded[16];
                for (int row = 0; row < 16; ++row) {
                    values[row] = block[(first + row) * width + column];
                }
                ++local_counts[quantize_group(values, rounded, 4, deltas, count)];
                const float diagonal = upper[column * width + column];
                for (int row = 0; row < 16; ++row) {
                    const int64_t base = (first + row) * width;
                    quantized[base + column] = rounded[row];
                    const float error = (values[row] - rounded[row]) / diagonal;
                    errors[base + column] = error;
                    for (int k = column; k < width; ++k) {
                        const float product = error * upper[column * width + k];
                        block[base + k] -= product;
                    }
                }
            }
        }
#pragma omp critical
        {
            for (int c = 0; c < count; ++c) {
                counts[c] += local_counts[c];
            }
        }
    }
}
