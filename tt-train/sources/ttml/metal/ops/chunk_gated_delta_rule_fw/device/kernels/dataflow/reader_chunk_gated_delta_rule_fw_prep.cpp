// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Prep reader.  Builds the constant tiles (eye, tril, ones [C,C] and the three WY-inverse quadrant
// masks) in L1 once, then for each (value head, chunk) work-item of this core reads
//   q, k  [Ct, Kt] of key head hv / G   out of the token-major [B,1,T,HK*K] grids
//   v     [Ct, Vt] of value head hv     out of the token-major [B,1,T,HV*V] grid
//   g, beta: column hv of [B,1,T,HV], as Ct fp32 column tiles (value in column 0, rest zero).

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tt-train/sources/ttml/metal/ops/chunk_gated_delta_rule_fw/device/kernels/chunk_gated_delta_rule_fw_common.hpp"

using namespace chunk_gated_delta_rule_fw;
using namespace chunk_gated_delta_rule_fw::prep;

constexpr uint32_t Ct = get_compile_time_arg_val(0);
constexpr uint32_t Kt = get_compile_time_arg_val(1);
constexpr uint32_t Vt = get_compile_time_arg_val(2);
constexpr uint32_t NC = get_compile_time_arg_val(3);
constexpr uint32_t HV = get_compile_time_arg_val(4);
constexpr uint32_t HK = get_compile_time_arg_val(5);
constexpr uint32_t GATE_ESZ = get_compile_time_arg_val(6);  // element bytes of g/beta: 2 bf16, 4 fp32

constexpr uint32_t G = HV / HK;
constexpr uint32_t Tt = NC * Ct;
constexpr uint32_t kTileElems = 32U * 32U;
constexpr uint32_t kOne = 0x3F800000U;  // 1.0f

namespace {

void zero_tile(uint32_t addr) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < kTileElems; ++i) {
        p[i] = 0U;
    }
}

void set_one(uint32_t tile_addr, uint32_t r, uint32_t c) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr)[tile_elem_off(r, c)] = kOne;
}

// [C, C] fp32 constant from a predicate on the global (row, col); tiles in row-major order.
template <typename Pred>
void build_cc(uint32_t cb_id, Pred one_at) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(Ct * Ct);
    const uint32_t tile_bytes = cb.get_tile_size();
    for (uint32_t ti = 0; ti < Ct; ++ti) {
        for (uint32_t si = 0; si < Ct; ++si) {
            const uint32_t addr = cb.get_write_ptr() + (ti * Ct + si) * tile_bytes;
            zero_tile(addr);
            for (uint32_t r = 0; r < 32U; ++r) {
                for (uint32_t c = 0; c < 32U; ++c) {
                    if (one_at(ti * 32U + r, si * 32U + c)) {
                        set_one(addr, r, c);
                    }
                }
            }
        }
    }
    cb.push_back(Ct * Ct);
}

void build_masks() {
    CircularBuffer cb(cb_mask);
    cb.reserve_back(kNumMasks);
    const uint32_t tile_bytes = cb.get_tile_size();
    for (uint32_t m = 0; m < kNumMasks; ++m) {
        const uint32_t addr = cb.get_write_ptr() + m * tile_bytes;
        zero_tile(addr);
        for (uint32_t r = 0; r < 32U; ++r) {
            for (uint32_t c = 0; c < 32U; ++c) {
                const bool lo_r = r < 16U;
                const bool lo_c = c < 16U;
                const bool one = (m == 0U) ? (lo_r && lo_c) : (m == 1U) ? (!lo_r && !lo_c) : (!lo_r && lo_c);
                if (one) {
                    set_one(addr, r, c);
                }
            }
        }
    }
    cb.push_back(kNumMasks);
}

}  // namespace

void kernel_main() {
    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t wi_start = get_arg_val<uint32_t>(0);
    const uint32_t wi_count = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t v_addr = get_arg_val<uint32_t>(4);
    const uint32_t g_addr = get_arg_val<uint32_t>(5);
    const uint32_t b_addr = get_arg_val<uint32_t>(6);

    const uint32_t tb_io = get_tile_size(cb_q);
    const uint32_t tb_gate = GATE_ESZ * kTileElems;
    const auto q_acc = TensorAccessor(q_args, q_addr, tb_io);
    const auto k_acc = TensorAccessor(k_args, k_addr, tb_io);
    const auto v_acc = TensorAccessor(v_args, v_addr, tb_io);
    const auto g_acc = TensorAccessor(g_args, g_addr, tb_gate);
    const auto b_acc = TensorAccessor(b_args, b_addr, tb_gate);

    build_cc(cb_eye, [](uint32_t r, uint32_t c) { return r == c; });
    build_cc(cb_tril, [](uint32_t r, uint32_t c) { return c <= r; });
    build_cc(cb_ones, [](uint32_t, uint32_t) { return true; });
    build_masks();

    Noc noc;

    // [Ct, nt] tiles of head `hx` at chunk c from a token-major grid with `row_tiles` tiles per
    // token row, packed row-major into cb_id.
    auto read_flat =
        [&](const auto& acc, uint32_t cb_id, uint32_t b, uint32_t c, uint32_t row_tiles, uint32_t hx, uint32_t nt) {
            CircularBuffer cb(cb_id);
            cb.reserve_back(Ct * nt);
            for (uint32_t rt = 0; rt < Ct; ++rt) {
                const uint32_t row_page = (b * Tt + c * Ct + rt) * row_tiles + hx * nt;
                for (uint32_t j = 0; j < nt; ++j) {
                    noc.async_read(acc, cb, tb_io, {.page_id = row_page + j}, {.offset_bytes = (rt * nt + j) * tb_io});
                }
            }
            noc.async_read_barrier();
            cb.push_back(Ct * nt);
        };

    // Column h of the [B,1,T,HV] gate grid for chunk c, as Ct fp32 column tiles.  Each raw tile lands
    // in its own destination slot; the 32 values are lifted out before the slot is rewritten.
    auto read_gate = [&](const auto& acc, uint32_t cb_id, uint32_t b, uint32_t c, uint32_t h) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(Ct);
        const uint32_t tile_bytes = cb.get_tile_size();
        for (uint32_t ct = 0; ct < Ct; ++ct) {
            const uint32_t addr = cb.get_write_ptr() + ct * tile_bytes;
            noc.async_read(acc, cb, tb_gate, {.page_id = b * Tt + c * Ct + ct}, {.offset_bytes = ct * tile_bytes});
            noc.async_read_barrier();
            uint32_t vals[32];
            for (uint32_t r = 0; r < 32U; ++r) {
                if constexpr (GATE_ESZ == 4U) {
                    vals[r] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr)[tile_elem_off(r, h)];
                } else {
                    vals[r] =
                        static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr)[tile_elem_off(r, h)])
                        << 16;
                }
            }
            zero_tile(addr);
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
            for (uint32_t r = 0; r < 32U; ++r) {
                d[tile_elem_off(r, 0)] = vals[r];
            }
        }
        cb.push_back(Ct);
    };

    for (uint32_t i = 0; i < wi_count; ++i) {
        const uint32_t wi = wi_start + i;  // (value head, chunk) = (bh, c) with wi = bh * NC + c
        const uint32_t bh = wi / NC;
        const uint32_t c = wi % NC;
        const uint32_t b = bh / HV;
        const uint32_t hv = bh % HV;
        read_flat(q_acc, cb_q, b, c, HK * Kt, hv / G, Kt);
        read_flat(k_acc, cb_k, b, c, HK * Kt, hv / G, Kt);
        read_flat(v_acc, cb_v, b, c, HV * Vt, hv, Vt);
        read_gate(g_acc, cb_g, b, c, hv);
        read_gate(b_acc, cb_beta, b, c, hv);
    }
}
