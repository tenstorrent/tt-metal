// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_act_rm = tt::CBIndex::c_0;
constexpr uint32_t cb_weights = tt::CBIndex::c_2;
constexpr uint32_t cb_grad = tt::CBIndex::c_5;

constexpr uint32_t block_ct = get_compile_time_arg_val(0);
constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
constexpr uint32_t Mt = get_compile_time_arg_val(2);
constexpr uint32_t Ct = get_compile_time_arg_val(3);
constexpr bool anti_causal = get_compile_time_arg_val(4) == 1U;

constexpr uint32_t tap_count = 4U;
constexpr uint32_t tile_height = 32U;
constexpr uint32_t tile_width = 32U;
constexpr uint32_t seq_len = Mt * tile_height;
// One block row is block_ct tiles wide; 32 such rows fill block_ct tile-sized CB pages.
constexpr uint32_t block_row_bytes = block_ct * tile_width * sizeof(uint16_t);

template <typename Accessor>
FORCE_INLINE void read_tap_tiles(const Accessor& tap, uint32_t l1_addr, uint32_t ct_start, uint32_t tile_bytes) {
    for (uint32_t ct = 0; ct < block_ct; ++ct) {
        noc_async_read_page(ct_start + ct, tap, l1_addr + ct * tile_bytes);
    }
}

template <typename A0, typename A1, typename A2, typename A3>
FORCE_INLINE void load_weights(
    const A0& tap0, const A1& tap1, const A2& tap2, const A3& tap3, uint32_t ct_start, uint32_t tile_bytes) {
    // The weights CB is laid out [tap][channel tile].
    cb_reserve_back(cb_weights, tap_count * block_ct);
    const uint32_t base = get_write_ptr(cb_weights);
    read_tap_tiles(tap0, base + 0 * block_ct * tile_bytes, ct_start, tile_bytes);
    read_tap_tiles(tap1, base + 1 * block_ct * tile_bytes, ct_start, tile_bytes);
    read_tap_tiles(tap2, base + 2 * block_ct * tile_bytes, ct_start, tile_bytes);
    read_tap_tiles(tap3, base + 3 * block_ct * tile_bytes, ct_start, tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_weights, tap_count * block_ct);
}

FORCE_INLINE void zero_row(uint32_t l1_addr) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < block_row_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = 0U;
    }
}

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(arg++);
    [[maybe_unused]] const uint32_t grad_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t work_start = get_arg_val<uint32_t>(arg++);
    const uint32_t work_count = get_arg_val<uint32_t>(arg++);

    constexpr auto input_args = TensorAccessorArgs<5>();
    constexpr auto tap0_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, input_addr);
    const auto tap0 = TensorAccessor(tap0_args, tap0_addr);
    const auto tap1 = TensorAccessor(tap1_args, tap1_addr);
    const auto tap2 = TensorAccessor(tap2_args, tap2_addr);
    const auto tap3 = TensorAccessor(tap3_args, tap3_addr);
#ifdef SILU_GRAD
    constexpr auto grad_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    const auto grad = TensorAccessor(grad_args, grad_addr);
#endif

    const uint32_t tile_bytes = get_tile_size(cb_weights);
    if constexpr (num_blocks == 1) {
        load_weights(tap0, tap1, tap2, tap3, 0, tile_bytes);
    }

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work = work_start + item;
        const uint32_t mt = work / num_blocks;
        const uint32_t ct_start = (work % num_blocks) * block_ct;
        const uint32_t col_offset_bytes = ct_start * tile_width * sizeof(uint16_t);

        if constexpr (num_blocks > 1) {
            load_weights(tap0, tap1, tap2, tap3, ct_start, tile_bytes);
        }

        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            const int32_t shift = anti_causal ? static_cast<int32_t>(3U - tap) : static_cast<int32_t>(tap) - 3;
            cb_reserve_back(cb_act_rm, block_ct);
            const uint32_t base = get_write_ptr(cb_act_rm);
            for (uint32_t row = 0; row < tile_height; ++row) {
                const int32_t src_row = static_cast<int32_t>(mt * tile_height + row) + shift;
                const uint32_t dst = base + row * block_row_bytes;
                if (src_row < 0 || src_row >= static_cast<int32_t>(seq_len)) {
                    zero_row(dst);
                } else {
                    noc_async_read(
                        input.get_noc_addr(static_cast<uint32_t>(src_row), col_offset_bytes), dst, block_row_bytes);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_act_rm, block_ct);
        }

#ifdef SILU_GRAD
        cb_reserve_back(cb_grad, block_ct);
        const uint32_t grad_base = get_write_ptr(cb_grad);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            noc_async_read_page(mt * Ct + ct_start + ct, grad, grad_base + ct * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_grad, block_ct);
#endif
    }
}
