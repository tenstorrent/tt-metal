#pragma once

#include "transform_view.h"

#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace views {

template <size_t... Is>
constexpr auto init_sfpu() {
    return views::transform<(Is + 1)...>(ckernel::init_sfpu);
}

template <size_t... Is, typename Compute>
constexpr auto with_cb_tiles(Compute compute) {
    return views::transform<0, (Is + 1)...>([=](auto num_tiles_per_cycle, auto... cb_ids) {
        constexpr uint32_t array[]{cb_ids...};
        constexpr auto last = sizeof...(cb_ids) - 1;

        for (uint32_t i = 0; i < last; ++i) {
            ckernel::cb_wait_front(array[i], num_tiles_per_cycle);
        }

        ckernel::cb_reserve_back(array[last], num_tiles_per_cycle);

        ckernel::tile_regs_acquire();

        compute(num_tiles_per_cycle, cb_ids...);

        ckernel::tile_regs_release();

        ckernel::cb_push_back(array[last], num_tiles_per_cycle);

        for (uint32_t i = last; i > 0; --i) {
            ckernel::cb_pop_front(array[i - 1], num_tiles_per_cycle);
        }
    });
}

template <size_t I>
constexpr auto copy_tile(uint32_t start_in_tile_index, uint32_t start_dst_tile_index) {
    return views::transform<0, I + 1>([=](auto num_tiles_per_cycle, uint32_t in_cb_id) {
        ckernel::copy_tile_init(in_cb_id);

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            ckernel::copy_tile(in_cb_id, start_in_tile_index + i, start_dst_tile_index + i);
        }
    });
}

template <size_t A, size_t B>
constexpr auto add_tiles(uint32_t start_a_tile_index, uint32_t start_b_tile_index, uint32_t start_dst_tile_index) {
    return views::transform<0, A + 1, B + 1>([=](auto num_tiles_per_cycle, uint32_t a_cb_id, uint32_t b_cb_id) {
        ckernel::add_tiles_init(a_cb_id, b_cb_id);

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            ckernel::add_tiles(
                a_cb_id, b_cb_id, start_a_tile_index + i, start_b_tile_index + i, start_dst_tile_index + i);
        }
    });
}

template <size_t A, size_t B>
constexpr auto mul_tiles(uint32_t start_a_tile_index, uint32_t start_b_tile_index, uint32_t start_dst_tile_index) {
    return views::transform<0, A + 1, B + 1>([=](auto num_tiles_per_cycle, uint32_t a_cb_id, uint32_t b_cb_id) {
        ckernel::mul_tiles_init(a_cb_id, b_cb_id);

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            ckernel::mul_tiles(
                a_cb_id, b_cb_id, start_a_tile_index + i, start_b_tile_index + i, start_dst_tile_index + i);
        }
    });
}

constexpr auto mul_unary_tile(uint32_t start_dst_tile_index, uint32_t param) {
    return views::transform<0>([=](auto num_tiles_per_cycle) {
        ckernel::binop_with_scalar_tile_init();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            ckernel::mul_unary_tile(start_dst_tile_index + i, param);
        }
    });
}

template <size_t I>
constexpr auto pack_tile(uint32_t start_index_from_dst) {
    return views::transform<0, I + 1>([=](auto num_tiles_per_cycle, uint32_t out_cb_id) {
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            ckernel::pack_tile(start_index_from_dst + i, out_cb_id);
        }
    });
};

}  // namespace views
