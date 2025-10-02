// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Multi-Core Mandelbrot Compute Kernel for TT Mesh Device
 *
 * This kernel runs on MULTIPLE cores per device simultaneously:
 * - Each core handles a subset of tiles assigned to its device
 * - Work is distributed using SPMD (Single Program, Multiple Data)
 * - Cores coordinate through different pixel ranges but same algorithm
 */

#include <cstdint>
#include <cstring>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"
#include "debug/dprint.h"

namespace NAMESPACE {

// Compile-time constants from host
constexpr uint32_t IMAGE_WIDTH = get_compile_time_arg_val(0);
constexpr uint32_t IMAGE_HEIGHT = get_compile_time_arg_val(1);
constexpr uint32_t MAX_ITERATIONS = get_compile_time_arg_val(2);
constexpr uint32_t ELEMENTS_PER_TILE = 32 * 32; // 32x32 elements per tile

void MAIN {
    // Runtime arguments - EXTENDED for multi-core
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Coordinate bounds as floats
    uint32_t x_min_bits = get_arg_val<uint32_t>(1);
    uint32_t x_max_bits = get_arg_val<uint32_t>(2);
    uint32_t y_min_bits = get_arg_val<uint32_t>(3);
    uint32_t y_max_bits = get_arg_val<uint32_t>(4);
    uint32_t device_id = get_arg_val<uint32_t>(5);

    // **NEW**: Core-specific pixel range (for multi-core distribution)
    uint32_t core_start_pixel = get_arg_val<uint32_t>(6);
    uint32_t core_end_pixel = get_arg_val<uint32_t>(7);

    // Get core coordinates for debugging
    uint32_t core_x = get_core_coord_x();
    uint32_t core_y = get_core_coord_y();

    // DEBUG: Print multi-core kernel startup info
    DPRINT << "=== MULTI-CORE MANDELBROT COMPUTE KERNEL STARTED ===" << ENDL();
    DPRINT << "CORE TYPE: TRISC (Multi-Core Compute)" << ENDL();
    DPRINT << "Device ID: " << device_id << ENDL();
    DPRINT << "Core coordinates: (" << core_x << "," << core_y << ")" << ENDL();
    DPRINT << "Num tiles: " << num_tiles << ENDL();
    DPRINT << "Core pixel range: [" << core_start_pixel << ", " << core_end_pixel << ")" << ENDL();
    DPRINT << "Image size: " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << ENDL();
    DPRINT << "Max iterations: " << MAX_ITERATIONS << ENDL();

    // Use memcpy to avoid strict-aliasing issues
    float x_min, x_max, y_min, y_max;
    memcpy(&x_min, &x_min_bits, sizeof(float));
    memcpy(&x_max, &x_max_bits, sizeof(float));
    memcpy(&y_min, &y_min_bits, sizeof(float));
    memcpy(&y_max, &y_max_bits, sizeof(float));

    // Output circular buffer
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // DEBUG: Print coordinate bounds
    DPRINT << "Coordinate bounds: x[" << x_min << ", " << x_max << "] y[" << y_min << ", " << y_max << "]" << ENDL();

    // Calculate coordinate scaling factors
    float dx = (x_max - x_min) / static_cast<float>(IMAGE_WIDTH);
    float dy = (y_max - y_min) / static_cast<float>(IMAGE_HEIGHT);

    DPRINT << "Coordinate deltas: dx=" << dx << " dy=" << dy << ENDL();
    DPRINT << "Multi-core processing: Core (" << core_x << "," << core_y << ") handling " << num_tiles << " tiles" << ENDL();

    // Process tiles assigned to THIS SPECIFIC CORE
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {

        // DEBUG: Print tile processing info for multi-core
        if (tile_idx < 3 || tile_idx % 500 == 0 || tile_idx == num_tiles - 1) {
            DPRINT << "[CORE(" << core_x << "," << core_y << ")] Processing tile " << tile_idx << "/" << num_tiles << ENDL();
        }

        // Reserve space in output circular buffer
        cb_reserve_back(cb_out0, 1);

        // Acquire tile registers
        tile_regs_acquire();

        // **MULTI-CORE PIXEL CALCULATION**
        // Calculate tile start pixel based on core's assigned range
        uint32_t tile_start_pixel = core_start_pixel + (tile_idx * ELEMENTS_PER_TILE);

        // Compute Mandelbrot iterations for pixels in this tile
        for (uint32_t elem = 0; elem < ELEMENTS_PER_TILE; elem++) {
            uint32_t global_pixel = tile_start_pixel + elem;

            // Skip if outside this core's assigned range
            if (global_pixel >= core_end_pixel || global_pixel >= IMAGE_WIDTH * IMAGE_HEIGHT) {
                continue;
            }

            // Convert linear pixel index to x, y coordinates
            uint32_t y = global_pixel / IMAGE_WIDTH;
            uint32_t x = global_pixel % IMAGE_WIDTH;

            // Map pixel coordinates to complex plane
            float cx = x_min + static_cast<float>(x) * dx;
            float cy = y_max - static_cast<float>(y) * dy; // Fixed Y-coordinate mapping

            // Mandelbrot computation
            float zx = 0.0f;
            float zy = 0.0f;
            uint32_t iteration = 0;

            // Full Mandelbrot iteration loop
            for (iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
                float zx_squared = zx * zx;
                float zy_squared = zy * zy;

                // Check if point escapes
                if (zx_squared + zy_squared > 4.0f) {
                    break;
                }

                // z = z^2 + c
                float new_zx = zx_squared - zy_squared + cx;
                float new_zy = 2.0f * zx * zy + cy;

                zx = new_zx;
                zy = new_zy;
            }

            // DEBUG: Show some computed results for first tile of each core
            if (tile_idx == 0 && elem < 4) {
                DPRINT << "Core(" << core_x << "," << core_y << ") Pixel(" << x << "," << y << ") c=(" << cx << "," << cy << ") iter=" << iteration << ENDL();
            }
        }

        // Commit and pack the tile
        tile_regs_commit();
        tile_regs_wait();

        // Pack the computed Mandelbrot data to output circular buffer
        pack_tile(0, cb_out0);

        // Push the tile and release registers
        cb_push_back(cb_out0, 1);
        tile_regs_release();

        // DEBUG: Print completion for reduced set of tiles
        if (tile_idx < 3 || tile_idx % 500 == 0 || tile_idx == num_tiles - 1) {
            DPRINT << "Core(" << core_x << "," << core_y << ") Completed tile " << tile_idx << "/" << num_tiles << ENDL();
        }
    }

    // DEBUG: Print core completion
    DPRINT << "=== MULTI-CORE COMPUTE COMPLETE ===" << ENDL();
    DPRINT << "Core(" << core_x << "," << core_y << ") on Device " << device_id << " processed " << num_tiles << " tiles" << ENDL();
}
}  // namespace NAMESPACE
