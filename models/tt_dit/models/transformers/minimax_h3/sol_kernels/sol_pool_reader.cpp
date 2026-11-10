// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "sol_pool_units.hpp"

namespace {
constexpr uint32_t cb_x = 0, cb_ps = 2, cb_zero = 3, cb_geo = 4, cb_ptab = 5;
constexpr uint32_t TILE_BYTES = 2048;
#ifdef SOL_POOL_X_BFP8
constexpr uint32_t X_TILE_BYTES = 1088;
#else
constexpr uint32_t X_TILE_BYTES = TILE_BYTES;
#endif
constexpr uint32_t BLOCK = sol_pool::BLOCK;
constexpr uint32_t GEO_BYTES = 128;
constexpr uint32_t ROWS_PER_BATCH = 4;
constexpr uint32_t BATCHES_IN_CB = 3;

constexpr uint16_t RECIP_BF16[65] = {
    0x0000, 0x3f80, 0x3f00, 0x3eab, 0x3e80, 0x3e4d, 0x3e2b, 0x3e12, 0x3e00, 0x3de4, 0x3dcd, 0x3dba, 0x3dab,
    0x3d9e, 0x3d92, 0x3d89, 0x3d80, 0x3d71, 0x3d64, 0x3d58, 0x3d4d, 0x3d43, 0x3d3a, 0x3d32, 0x3d2b, 0x3d24,
    0x3d1e, 0x3d18, 0x3d12, 0x3d0d, 0x3d09, 0x3d04, 0x3d00, 0x3cf8, 0x3cf1, 0x3cea, 0x3ce4, 0x3cdd, 0x3cd8,
    0x3cd2, 0x3ccd, 0x3cc8, 0x3cc3, 0x3cbf, 0x3cba, 0x3cb6, 0x3cb2, 0x3cae, 0x3cab, 0x3ca7, 0x3ca4, 0x3ca1,
    0x3c9e, 0x3c9b, 0x3c98, 0x3c95, 0x3c92, 0x3c90, 0x3c8d, 0x3c8b, 0x3c89, 0x3c86, 0x3c84, 0x3c82, 0x3c80};

inline uint32_t tile_elem(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >> 4) << 1) | (c >> 4);
    return face * 256 + (r & 15) * 16 + (c & 15);
}

inline void write_row(uint32_t addr, uint32_t r, uint32_t ncols, uint16_t v) {
    CoreLocalMem<volatile uint32_t> w(addr);
    const uint32_t vv = (static_cast<uint32_t>(v) << 16) | v;
    uint32_t c = 0;
    for (; c + 1 < ncols; c += 2) {
        w[tile_elem(r, c) >> 1] = vv;
    }
    if (c < ncols) {
        CoreLocalMem<volatile uint16_t> e(addr);
        e[tile_elem(r, c)] = v;
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
    constexpr uint32_t NCORES = NX * NY;
    using Layout = sol_pool::Layout<H, RT, NBPT>;
    static_assert(Layout::UNITS == UNITS, "host unit count must match the kernel layout");
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    uint32_t unit_start, unit_count;
    Layout::range(ci, NCORES, unit_start, unit_count);
    if (unit_count == 0) {
        return;
    }
    const auto x0 = TensorAccessor(x0_args, get_common_arg_val<uint32_t>(0));
    const auto x1 = TensorAccessor(x1_args, get_common_arg_val<uint32_t>(1));
    const auto x2 = TensorAccessor(x2_args, get_common_arg_val<uint32_t>(2));
    const auto geo = TensorAccessor(geo_args, get_common_arg_val<uint32_t>(3));

    Noc noc;
    CircularBuffer cbx(cb_x), cbps(cb_ps), cbz(cb_zero), cbg(cb_geo), cbt(cb_ptab);
    const PrecomposedUnicastEndpoint local;

    cbg.reserve_back(1);
    noc.async_read(geo, cbg, GEO_BYTES, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t valid_end = CoreLocalMem<volatile uint32_t>(cbg.get_write_ptr())[0];

    cbz.reserve_back(1);
    const uint32_t zero_addr = cbz.get_write_ptr();
    cbt.reserve_back(32);
    const uint32_t tab = cbt.get_write_ptr();
    bool table_built = false;
    auto build_table = [&]() {
        noc.async_write_zeros(CoreLocalMem<uint32_t>(zero_addr), TILE_BYTES);
        for (uint32_t r = 0; r < 32; ++r) {
            noc.async_write_zeros(CoreLocalMem<uint32_t>(tab + r * TILE_BYTES), TILE_BYTES);
        }
        noc.write_zeros_l1_barrier();
        for (uint32_t r = 0; r < 32; ++r) {
            write_row(tab + r * TILE_BYTES, r, 32, RECIP_BF16[BLOCK]);
        }
        cbz.push_back(1);
        cbt.push_back(32);
        table_built = true;
    };

    constexpr uint32_t NT = ROWS_PER_BATCH * DT;
    constexpr uint32_t X_BYTES = BATCHES_IN_CB * NT * X_TILE_BYTES;
    constexpr uint32_t P_BYTES = BATCHES_IN_CB * ROWS_PER_BATCH * TILE_BYTES;
    const uint32_t x_base = cbx.get_write_ptr();
    const uint32_t p_base = cbps.get_write_ptr();

    bool pending = false;
    uint32_t pend_trid = 2;
    auto finish_pending = [&]() {
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = pend_trid});
        cbps.push_back(ROWS_PER_BATCH);
        cbx.push_back(NT);
        pending = false;
    };
    auto copy_ptile = [&](uint32_t src, uint32_t dst, uint32_t trid) {
        noc.async_read<NocOptions::TXN_ID>(
            local, CoreLocalMem<uint32_t>(dst), TILE_BYTES, {.noc_addr = get_noc_addr(src)}, {}, {.trid = trid});
    };

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const sol_pool::Unit un = Layout::decode(u);
        const uint32_t t = un.t, h = un.h, g = un.g;
        const uint32_t k_begin = g * BLOCK;
        const uint32_t k_end = min_u(k_begin + BLOCK, RT);

        auto read_page = [&](uint32_t page, uint32_t dst, uint32_t trid) {
            if (t == 0) {
                noc.async_read<NocOptions::TXN_ID>(
                    x0, CoreLocalMem<uint32_t>(dst), X_TILE_BYTES, {.page_id = page}, {}, {.trid = trid});
            } else if (t == 1) {
                noc.async_read<NocOptions::TXN_ID>(
                    x1, CoreLocalMem<uint32_t>(dst), X_TILE_BYTES, {.page_id = page}, {}, {.trid = trid});
            } else {
                noc.async_read<NocOptions::TXN_ID>(
                    x2, CoreLocalMem<uint32_t>(dst), X_TILE_BYTES, {.page_id = page}, {}, {.trid = trid});
            }
        };

        for (uint32_t k0 = k_begin; k0 < k_end; k0 += ROWS_PER_BATCH) {
            const uint32_t nk = min_u(ROWS_PER_BATCH, k_end - k0);
            cbx.reserve_back(2 * NT);
            cbps.reserve_back(2 * ROWS_PER_BATCH);
            uint32_t xa = cbx.get_write_ptr(), pa = cbps.get_write_ptr();
            if (pending) {
                xa += NT * X_TILE_BYTES;
                if (xa >= x_base + X_BYTES) {
                    xa -= X_BYTES;
                }
                pa += ROWS_PER_BATCH * TILE_BYTES;
                if (pa >= p_base + P_BYTES) {
                    pa -= P_BYTES;
                }
            }
            const uint32_t trid = pending ? 3 - pend_trid : 1;

            uint32_t tail_rows = 0;
            bool zeros = false;
            auto issue_ptiles = [&]() {
                for (uint32_t kk = 0; kk < nk; ++kk) {
                    const uint32_t k = k0 + kk;
                    const uint32_t prow = pa + kk * TILE_BYTES;
                    if (32 * k >= valid_end) {
                        copy_ptile(zero_addr, prow, trid);
                    } else {
                        const uint32_t b = k >> 1;
                        const uint32_t len = min_u(BLOCK, valid_end - b * BLOCK);
                        const uint32_t ncols = min_u(32, valid_end - 32 * k);
                        if (len == BLOCK && ncols == 32) {
                            copy_ptile(tab + (b & 31) * TILE_BYTES, prow, trid);
                        } else {
                            noc.async_write_zeros(CoreLocalMem<uint32_t>(prow), TILE_BYTES);
                            zeros = true;
                            tail_rows |= 1u << kk;
                        }
                    }
                }
            };
            auto issue_pages = [&]() {
                uint32_t off = 0;
                for (uint32_t kk = 0; kk < nk; ++kk) {
                    const uint32_t k = k0 + kk;
                    for (uint32_t n = 0; n < DT; ++n) {
                        if (32 * k < valid_end) {
                            read_page((h * RT + k) * DT + n, xa + off, trid);
                        } else {
                            noc.async_write_zeros(CoreLocalMem<uint32_t>(xa + off), X_TILE_BYTES);
                            zeros = true;
                        }
                        off += X_TILE_BYTES;
                    }
                }
            };
            if (table_built) {
                issue_ptiles();
                issue_pages();
            } else {
                issue_pages();
                build_table();
                issue_ptiles();
            }
            if (zeros) {
                noc.write_zeros_l1_barrier();
                for (uint32_t kk = 0; kk < nk; ++kk) {
                    if (tail_rows & (1u << kk)) {
                        const uint32_t k = k0 + kk;
                        const uint32_t b = k >> 1;
                        const uint32_t len = min_u(BLOCK, valid_end - b * BLOCK);
                        write_row(pa + kk * TILE_BYTES, b & 31, min_u(32, valid_end - 32 * k), RECIP_BF16[len]);
                    }
                }
            }
            if (pending) {
                finish_pending();
            }
            pending = true;
            pend_trid = trid;
        }
    }
    if (pending) {
        finish_pending();
    }
    noc_async_read_set_trid(0, noc.get_noc_id());
}
