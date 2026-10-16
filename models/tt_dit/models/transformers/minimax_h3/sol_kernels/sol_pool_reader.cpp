// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn block-pooling pre-pass, reader (NCRISC).
//
// Work unit = (tensor t in {0,1,2}, head h, output tile row g): the 32 Sol blocks [32g, 32g+32) of one head of one
// of the Q/K/V shards [1, H, R, D] (tile layout). For every tile row k of the group ([64g, 64g+64) clipped to RT) the
// reader streams the DT data tiles into cb_x (batches of 4 rows = 4*DT pages, always full) and one pooling tile P[k]
// into cb_ps (32x32, one nonzero row r = (k/2) mod 32 holding 1/len of the block, zero in the columns past the shard's
// valid end), so that compute accumulates O[n] = sum_k P[k] @ X[k, n] without any geometry of its own. Rows at or past
// the valid end get a zero P tile and zero-filled data pages (never read from DRAM). The shard's valid token count
// comes from a per-device uint32 row (geo tensor, word 0), so the same program runs on every device (SPMD).
//
// Compile-time args: 0 H, 1 RT (tile rows per shard), 2 DT, 3 NBPT (output tile rows per head), 4 R (rows per
// shard), then TensorAccessorArgs of x0, x1, x2, geo.
// Common runtime args: 0..2 buffer addresses of x0..x2, 3 geo address. Per-core runtime args: none (unit range from the
// core's logical coordinates).

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_x = 0, cb_ps = 2, cb_zero = 3, cb_geo = 4;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t BLOCK = 64;
constexpr uint32_t GEO_BYTES = 128;

// bf16(1/len) for len = 0..64 (len 0 unused).
constexpr uint16_t RECIP_BF16[65] = {
    0x0000, 0x3f80, 0x3f00, 0x3eab, 0x3e80, 0x3e4d, 0x3e2b, 0x3e12, 0x3e00, 0x3de4, 0x3dcd, 0x3dba, 0x3dab,
    0x3d9e, 0x3d92, 0x3d89, 0x3d80, 0x3d71, 0x3d64, 0x3d58, 0x3d4d, 0x3d43, 0x3d3a, 0x3d32, 0x3d2b, 0x3d24,
    0x3d1e, 0x3d18, 0x3d12, 0x3d0d, 0x3d09, 0x3d04, 0x3d00, 0x3cf8, 0x3cf1, 0x3cea, 0x3ce4, 0x3cdd, 0x3cd8,
    0x3cd2, 0x3ccd, 0x3cc8, 0x3cc3, 0x3cbf, 0x3cba, 0x3cb6, 0x3cb2, 0x3cae, 0x3cab, 0x3ca7, 0x3ca4, 0x3ca1,
    0x3c9e, 0x3c9b, 0x3c98, 0x3c95, 0x3c92, 0x3c90, 0x3c8d, 0x3c8b, 0x3c89, 0x3c86, 0x3c84, 0x3c82, 0x3c80};

inline void zero_tile(uint32_t addr) {
    CoreLocalMem<volatile uint32_t> w(addr);
    for (uint32_t i = 0; i < TILE_BYTES / 4; ++i) {
        w[i] = 0;
    }
}

// Row r of a 32x32 bf16 tile (4 faces of 16x16, faces ordered (r<16,c<16),(r<16,c>=16),(r>=16,c<16),(r>=16,c>=16)).
inline void write_row(uint32_t addr, uint32_t r, uint32_t ncols, uint16_t v) {
    CoreLocalMem<volatile uint16_t> e(addr);
    for (uint32_t c = 0; c < ncols; ++c) {
        const uint32_t face = ((r >> 4) << 1) | (c >> 4);
        e[face * 256 + (r & 15) * 16 + (c & 15)] = v;
    }
}

inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }
}  // namespace

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t DT = get_compile_time_arg_val(2);
    constexpr uint32_t NBPT = get_compile_time_arg_val(3);
    constexpr auto x0_args = TensorAccessorArgs<8>();
    constexpr auto x1_args = TensorAccessorArgs<x0_args.next_compile_time_args_offset()>();
    constexpr auto x2_args = TensorAccessorArgs<x1_args.next_compile_time_args_offset()>();
    constexpr auto geo_args = TensorAccessorArgs<x2_args.next_compile_time_args_offset()>();

    constexpr uint32_t NX = get_compile_time_arg_val(5);
    constexpr uint32_t NY = get_compile_time_arg_val(6);
    constexpr uint32_t UNITS = get_compile_time_arg_val(7);
    // Unit range of this core from its logical coordinates (no per-core runtime args): units [ci*U/N, (ci+1)*U/N).
    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto x0 = TensorAccessor(x0_args, get_common_arg_val<uint32_t>(0));
    const auto x1 = TensorAccessor(x1_args, get_common_arg_val<uint32_t>(1));
    const auto x2 = TensorAccessor(x2_args, get_common_arg_val<uint32_t>(2));
    const auto geo = TensorAccessor(geo_args, get_common_arg_val<uint32_t>(3));

    Noc noc;
    CircularBuffer cbx(cb_x), cbps(cb_ps), cbz(cb_zero), cbg(cb_geo);

    // This shard's valid token count (geo row word 0).
    cbg.reserve_back(1);
    noc.async_read(geo, cbg, GEO_BYTES, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t valid_end = CoreLocalMem<volatile uint32_t>(cbg.get_write_ptr())[0];

    cbz.reserve_back(1);
    zero_tile(cbz.get_write_ptr());
    cbz.push_back(1);

    constexpr uint32_t units_per_tensor = H * NBPT;
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t t = u / units_per_tensor;
        const uint32_t h = (u % units_per_tensor) / NBPT;
        const uint32_t g = u % NBPT;
        const uint32_t k_begin = g * BLOCK;
        const uint32_t k_end = min_u(k_begin + BLOCK, RT);

        auto read_batch = [&](auto& acc, uint32_t k0, uint32_t nk) {
            constexpr uint32_t nt = 4 * DT;  // always a full batch: a reservation must never straddle the CB end
            cbx.reserve_back(nt);
            const uint32_t base = cbx.get_write_ptr();
            uint32_t off = 0;
            bool zeros = false;
            for (uint32_t k = k0; k < k0 + nk; ++k) {
                for (uint32_t n = 0; n < DT; ++n) {
                    if (32 * k < valid_end) {
                        noc.async_read(acc, cbx, TILE_BYTES, {.page_id = (h * RT + k) * DT + n}, {.offset_bytes = off});
                    } else {
                        noc.async_write_zeros(CoreLocalMem<uint32_t>(base + off), TILE_BYTES);
                        zeros = true;
                    }
                    off += TILE_BYTES;
                }
            }
            noc.async_read_barrier();
            if (zeros) {
                noc.write_zeros_l1_barrier();
            }
            cbx.push_back(nt);
        };

        for (uint32_t k0 = k_begin; k0 < k_end; k0 += 4) {
            const uint32_t nk = min_u(4, k_end - k0);
            // Pooling tiles of this batch, one per tile row (zero past the valid end).
            for (uint32_t k = k0; k < k0 + nk; ++k) {
                const uint32_t b = k >> 1;
                const uint32_t len = 32 * k < valid_end ? min_u(BLOCK, valid_end - b * BLOCK) : 0;
                cbps.reserve_back(1);
                const uint32_t addr = cbps.get_write_ptr();
                zero_tile(addr);
                if (len != 0) {
                    write_row(addr, b & 31, min_u(32, valid_end - 32 * k), RECIP_BF16[len]);
                }
                cbps.push_back(1);
            }
            if (t == 0) {
                read_batch(x0, k0, nk);
            } else if (t == 1) {
                read_batch(x1, k0, nk);
            } else {
                read_batch(x2, k0, nk);
            }
        }
    }
}
