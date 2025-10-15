// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transform_view.h"

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/rpow.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace views {

template <size_t In, size_t Out>
constexpr auto init_sfpu(void) {
    constexpr auto compute = [](uint32_t in_cb_id, uint32_t out_cb_id) -> void {
        return ckernel::init_sfpu(in_cb_id, out_cb_id);
    };
    return views::transform<In, Out>(compute);
}

template <size_t... Is, typename Compute>
constexpr auto with_cb_ids(Compute compute) {
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

template <typename NumTilesPerCycle>
void _copy(NumTilesPerCycle num_tiles_per_cycle, uint32_t in_cb_id, uint32_t start_dst_tile_index) {
    ckernel::copy_tile_init(in_cb_id);

    for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
        ckernel::copy_tile(in_cb_id, i, start_dst_tile_index + i);
    }
}

template <typename NumTilesPerCycle>
inline void _pack(NumTilesPerCycle num_tiles_per_cycle, uint32_t out_cb_id) {
    ckernel::tile_regs_commit();
    ckernel::tile_regs_wait();
    ckernel::pack_tile_block(0, out_cb_id, num_tiles_per_cycle);
}

template <void (*Init)(uint32_t, uint32_t), void (*Compute)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)>
constexpr auto _binary(void) {
    return [](auto num_tiles_per_cycle, uint32_t a_cb_id, uint32_t b_cb_id, uint32_t out_cb_id) -> void {
        Init(a_cb_id, b_cb_id);

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Compute(a_cb_id, b_cb_id, i, i, i);
        }

        views::_pack(num_tiles_per_cycle, out_cb_id);
    };
}

template <void (*Init)(void), void (*Compute)(uint32_t, uint32_t, uint32_t)>
constexpr auto _binary_sfpu(void) {
    return [](auto num_tiles_per_cycle, uint32_t a_cb_id, uint32_t b_cb_id, uint32_t out_cb_id) -> void {
        views::_copy(num_tiles_per_cycle, a_cb_id, 0);
        views::_copy(num_tiles_per_cycle, b_cb_id, num_tiles_per_cycle);

        Init();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Compute(i, i + num_tiles_per_cycle, i);
        }

        views::_pack(num_tiles_per_cycle, out_cb_id);
    };
}

constexpr auto add(void) {
    constexpr auto add_tiles_init = [](uint32_t a_cb_id, uint32_t b_cb_id) -> void {
        return ckernel::add_tiles_init(a_cb_id, b_cb_id);
    };
    return _binary<add_tiles_init, &ckernel::add_tiles>();
}

constexpr auto sub(void) {
    constexpr auto sub_tiles_init = [](uint32_t a_cb_id, uint32_t b_cb_id) {
        return ckernel::sub_tiles_init(a_cb_id, b_cb_id);
    };
    return _binary<sub_tiles_init, &ckernel::sub_tiles>();
}

constexpr auto mul(void) { return _binary<&ckernel::mul_tiles_init, &ckernel::mul_tiles>(); }

constexpr auto div_binary(void) { return _binary_sfpu<&ckernel::div_binary_tile_init, &ckernel::div_binary_tile>(); }

constexpr auto power_binary(void) {
    return _binary_sfpu<&ckernel::power_binary_tile_init, &ckernel::power_binary_tile>();
}

template <void (*Init)(void), void (*Compute)(uint32_t, uint32_t)>
constexpr auto _unary_sfpu_with_param(uint32_t param) {
    return [=](auto num_tiles_per_cycle, uint32_t in_cb_id, uint32_t out_cb_id) -> void {
        views::_copy(num_tiles_per_cycle, in_cb_id, 0);

        Init();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Compute(i, param);
        }

        views::_pack(num_tiles_per_cycle, out_cb_id);
    };
}

constexpr auto add_unary(uint32_t param) {
    return _unary_sfpu_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::add_unary_tile>(param);
}

constexpr auto add_unary_int32(uint32_t param) {
    return _unary_sfpu_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::add_unary_tile_int32>(param);
}

constexpr auto sub_unary(uint32_t param) {
    return _unary_sfpu_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::sub_unary_tile>(param);
}

constexpr auto rsub_unary(uint32_t param) {
    return _unary_sfpu_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::rsub_unary_tile>(param);
}

constexpr auto mul_unary(uint32_t param) {
    return _unary_sfpu_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::mul_unary_tile>(param);
}

constexpr auto div_unary(uint32_t param) {
    return _unary_sfpu_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::div_unary_tile>(param);
}

template <void (*Init)(void), void (*Compute)(uint32_t)>
constexpr auto _unary_sfpu(void) {
    return [](auto num_tiles_per_cycle, uint32_t in_cb_id, uint32_t out_cb_id) -> void {
        views::_copy(num_tiles_per_cycle, in_cb_id, 0);

        Init();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Compute(i);
        }

        views::_pack(num_tiles_per_cycle, out_cb_id);
    };
}

constexpr auto recip(void) {
    constexpr auto recip_tile = [](uint32_t dst_tile_index) { return ckernel::recip_tile(dst_tile_index); };
    return _unary_sfpu<&ckernel::recip_tile_init, recip_tile>();
}

constexpr auto negative(void) { return _unary_sfpu<&ckernel::negative_tile_init, &ckernel::negative_tile>(); }

constexpr auto negative_int32(void) {
    return _unary_sfpu<&ckernel::negative_tile_init, &ckernel::negative_tile_int32>();
}

constexpr auto exp(void) {
    constexpr auto exp_tile = [](uint32_t dst_tile_index) { return ckernel::exp_tile(dst_tile_index); };
    return _unary_sfpu<&ckernel::exp_tile_init, exp_tile>();
}

constexpr auto power(uint32_t param) {
    return _unary_sfpu_with_param<&ckernel::power_tile_init, &ckernel::power_tile>(param);
}

constexpr auto rpow(uint32_t param) {
    constexpr auto rpow_tile = [](uint32_t dst_tile_index, uint32_t base_val) {
        return ckernel::rpow_tile(dst_tile_index, base_val);
    };
    return _unary_sfpu_with_param<&ckernel::rpow_tile_init, rpow_tile>(param);
}

constexpr auto eqz(void) { return _unary_sfpu<&ckernel::eqz_tile_init, &ckernel::eqz_tile>(); }

constexpr auto eqz_int32(void) { return _unary_sfpu<&ckernel::eqz_tile_init, &ckernel::eqz_tile_int32>(); }

constexpr auto eqz_uint16(void) { return _unary_sfpu<&ckernel::eqz_tile_init, &ckernel::eqz_tile_uint16>(); }

constexpr auto eqz_uint32(void) { return _unary_sfpu<&ckernel::eqz_tile_init, &ckernel::eqz_tile_uint32>(); }

constexpr auto gez(void) { return _unary_sfpu<&ckernel::gez_tile_init, &ckernel::gez_tile>(); }

constexpr auto gez_int32(void) { return _unary_sfpu<&ckernel::gez_tile_init, &ckernel::gez_tile_int32>(); }

constexpr auto gtz(void) { return _unary_sfpu<&ckernel::gtz_tile_init, &ckernel::gtz_tile>(); }

constexpr auto gtz_int32(void) { return _unary_sfpu<&ckernel::gtz_tile_init, &ckernel::gtz_tile_int32>(); }

constexpr auto lez(void) { return _unary_sfpu<&ckernel::lez_tile_init, &ckernel::lez_tile>(); }

constexpr auto lez_int32(void) { return _unary_sfpu<&ckernel::lez_tile_init, &ckernel::lez_tile_int32>(); }

constexpr auto ltz(void) { return _unary_sfpu<&ckernel::ltz_tile_init, &ckernel::ltz_tile>(); }

constexpr auto ltz_int32(void) { return _unary_sfpu<&ckernel::ltz_tile_init, &ckernel::ltz_tile_int32>(); }

constexpr auto nez(void) { return _unary_sfpu<&ckernel::nez_tile_init, &ckernel::nez_tile>(); }

constexpr auto nez_int32(void) { return _unary_sfpu<&ckernel::nez_tile_init, &ckernel::nez_tile_int32>(); }

constexpr auto nez_uint16(void) { return _unary_sfpu<&ckernel::nez_tile_init, &ckernel::nez_tile_uint16>(); }

constexpr auto nez_uint32(void) { return _unary_sfpu<&ckernel::nez_tile_init, &ckernel::nez_tile_uint32>(); }

constexpr auto logical_not(void) {
    return _unary_sfpu<&ckernel::logical_not_unary_tile_init, &ckernel::logical_not_unary_tile>();
}

constexpr auto logical_not_int32(void) {
    return _unary_sfpu<&ckernel::logical_not_unary_tile_init, &ckernel::logical_not_unary_tile_int32>();
}

constexpr auto logical_not_uint16(void) {
    return _unary_sfpu<&ckernel::logical_not_unary_tile_init, &ckernel::logical_not_unary_tile_uint16>();
}

constexpr auto logical_not_uint32(void) {
    return _unary_sfpu<&ckernel::logical_not_unary_tile_init, &ckernel::logical_not_unary_tile_uint32>();
}

constexpr auto atan(void) { return _unary_sfpu<&ckernel::atan_tile_init, &ckernel::atan_tile>(); }

template <void (*Init)(void), void (*Compute)(uint32_t, uint32_t, uint32_t, uint32_t)>
constexpr auto _ternary_sfpu(void) {
    return
        [](auto num_tiles_per_cycle, uint32_t a_cb_id, uint32_t b_cb_id, uint32_t c_cb_id, uint32_t out_cb_id) -> void {
            views::_copy(num_tiles_per_cycle, a_cb_id, 0);
            views::_copy(num_tiles_per_cycle, b_cb_id, num_tiles_per_cycle);
            views::_copy(num_tiles_per_cycle, c_cb_id, num_tiles_per_cycle * 2);

            Init();

            for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
                Compute(i, i + num_tiles_per_cycle, i + num_tiles_per_cycle * 2, i);
            }

            views::_pack(num_tiles_per_cycle, out_cb_id);
        };
}

constexpr auto where(void) { return _ternary_sfpu<&ckernel::where_tile_init, &ckernel::where_tile>(); }

constexpr auto where_fp32(void) { return _ternary_sfpu<&ckernel::where_tile_init, &ckernel::where_fp32_tile>(); }

constexpr auto where_int32(void) { return _ternary_sfpu<&ckernel::where_tile_init, &ckernel::where_int32_tile>(); }

}  // namespace views
