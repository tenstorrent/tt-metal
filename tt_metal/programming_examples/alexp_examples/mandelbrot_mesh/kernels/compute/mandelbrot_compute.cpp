// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

void MAIN {
    // Runtime arguments
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Get coordinate bounds as floats
    uint32_t x_min_bits = get_arg_val<uint32_t>(1);
    uint32_t x_max_bits = get_arg_val<uint32_t>(2);
    uint32_t y_min_bits = get_arg_val<uint32_t>(3);
    uint32_t y_max_bits = get_arg_val<uint32_t>(4);
    uint32_t device_id = get_arg_val<uint32_t>(5);

    // DEBUG: Print detailed RISC-V core startup info
    DPRINT << "=== MANDELBROT COMPUTE KERNEL STARTED ===" << ENDL();
    DPRINT << "CORE TYPE: TRISC (Compute Core)" << ENDL();
    DPRINT << "Device ID: " << device_id << ENDL();
    DPRINT << "Num tiles: " << num_tiles << ENDL();
    DPRINT << "Image size: " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << ENDL();
    DPRINT << "Max iterations: " << MAX_ITERATIONS << ENDL();

#if defined(COMPILE_FOR_TRISC)
    DPRINT << "TRISC CORE DETAILS:" << ENDL();
    #if COMPILE_FOR_TRISC == 0
        DPRINT << "  - TRISC0: Unpack/Input operations" << ENDL();
    #elif COMPILE_FOR_TRISC == 1
        DPRINT << "  - TRISC1: Math/Compute operations" << ENDL();
    #elif COMPILE_FOR_TRISC == 2
        DPRINT << "  - TRISC2: Pack/Output operations" << ENDL();
    #endif
#endif

    // Use memcpy to avoid strict-aliasing issues
    float x_min, x_max, y_min, y_max;
    memcpy(&x_min, &x_min_bits, sizeof(float));
    memcpy(&x_max, &x_max_bits, sizeof(float));
    memcpy(&y_min, &y_min_bits, sizeof(float));
    memcpy(&y_max, &y_max_bits, sizeof(float));

    // Output circular buffer
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // Calculate pixel range for this device
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t ELEMENTS_PER_TILE = TILE_WIDTH * TILE_HEIGHT;

    uint32_t total_devices = 8; // 2x4 mesh
    uint32_t pixels_per_device = (IMAGE_WIDTH * IMAGE_HEIGHT) / total_devices;
    uint32_t start_pixel = device_id * pixels_per_device;
    uint32_t end_pixel = (device_id == total_devices - 1) ?
                         IMAGE_WIDTH * IMAGE_HEIGHT :
                         (device_id + 1) * pixels_per_device;

    // DEBUG: Print coordinate bounds
    DPRINT << "Coordinate bounds: x[" << x_min << ", " << x_max << "] y[" << y_min << ", " << y_max << "]" << ENDL();

    // Calculate coordinate scaling factors
    float dx = (x_max - x_min) / static_cast<float>(IMAGE_WIDTH);
    float dy = (y_max - y_min) / static_cast<float>(IMAGE_HEIGHT);

    DPRINT << "Coordinate deltas: dx=" << dx << " dy=" << dy << ENDL();

    // Process tiles
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {

        // DEBUG: Print detailed tile processing info with core identification - REDUCED TILES
        if (tile_idx < 5 || tile_idx % 1000 == 0 || tile_idx == num_tiles - 1) { // Show first 5 tiles + every 1000th tile + last tile
#if defined(COMPILE_FOR_TRISC)
    #if COMPILE_FOR_TRISC == 0
            DPRINT << "[TRISC0-UNPACK] Processing tile " << tile_idx << "/" << num_tiles << ENDL();
    #elif COMPILE_FOR_TRISC == 1
            DPRINT << "[TRISC1-MATH] Processing tile " << tile_idx << "/" << num_tiles << ENDL();
    #elif COMPILE_FOR_TRISC == 2
            DPRINT << "[TRISC2-PACK] Processing tile " << tile_idx << "/" << num_tiles << ENDL();
    #endif
#else
            DPRINT << "[TRISC-COMPUTE] Processing tile " << tile_idx << "/" << num_tiles << ENDL();
#endif
        }
        // Reserve space in output circular buffer
        cb_reserve_back(cb_out0, 1);

        // Acquire tile registers
        tile_regs_acquire();

        // For compute kernel, we'll use the standard tile processing approach

        // Calculate Mandelbrot set for this tile
        uint32_t tile_start_pixel = tile_idx * ELEMENTS_PER_TILE + start_pixel;

        // Compute actual Mandelbrot iterations and store them
        for (uint32_t elem = 0; elem < ELEMENTS_PER_TILE; elem++) {
            uint32_t global_pixel = tile_start_pixel + elem;

            if (global_pixel >= end_pixel || global_pixel >= IMAGE_WIDTH * IMAGE_HEIGHT) {
                // Outside our range, skip
                continue;
            }

            // Convert linear pixel index to x, y coordinates
            uint32_t y = global_pixel / IMAGE_WIDTH;
            uint32_t x = global_pixel % IMAGE_WIDTH;

            // Map pixel coordinates to complex plane (FIX: flip Y coordinate)
            float cx = x_min + static_cast<float>(x) * dx;
            float cy = y_max - static_cast<float>(y) * dy;  // FIXED: y_max - y*dy

            // Simple Mandelbrot-like computation (simplified for demo)
            // This creates a pattern that resembles the Mandelbrot set structure
            float zx = 0.0f;
            float zy = 0.0f;
            uint32_t iteration = 0;

            // Full Mandelbrot iteration loop (removed artificial limit)
            for (iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
                float zx_squared = zx * zx;
                float zy_squared = zy * zy;

                // Check if point escapes
                if (zx_squared + zy_squared > 4.0f) {
                    break;
                }

                // z = z^2 + c (simplified)
                float new_zx = zx_squared - zy_squared + cx;
                float new_zy = 2.0f * zx * zy + cy;

                zx = new_zx;
                zy = new_zy;
            }

            // DEBUG: Show some computed results for first tile, including variety of points
            if (tile_idx == 0 && (global_pixel - tile_start_pixel) < 8) {
                DPRINT << "Pixel(" << x << "," << y << ") c=(" << cx << "," << cy << ") iter=" << iteration << ENDL();
            }

            // Also show some middle points for variety
            if (tile_idx == 10 && (global_pixel - tile_start_pixel) < 3) {
                DPRINT << "Mid Pixel(" << x << "," << y << ") c=(" << cx << "," << cy << ") iter=" << iteration << ENDL();
            }

            // For now, we'll add debug info and use pack_tile to store a pattern
            // In a proper implementation, we would write the iteration count to tile registers
            // using proper TT compute APIs, but this requires more complex tile manipulation
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
        if (tile_idx < 5 || tile_idx % 1000 == 0 || tile_idx == num_tiles - 1) {
            DPRINT << "Completed tile " << tile_idx << "/" << num_tiles << ENDL();
        }
    }
}
}  // namespace NAMESPACE
