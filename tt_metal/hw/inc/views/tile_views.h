#pragma once

#include "transform_view.h"

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace views {

template <size_t... Is>
constexpr auto init_sfpu() {
    return views::transform<(Is + 1)...>(ckernel::init_sfpu);
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

template <size_t I>
constexpr auto copy(uint32_t start_in_tile_index, uint32_t start_dst_tile_index) {
    return views::transform<0, I + 1>([=](auto num_tiles_per_cycle, uint32_t in_cb_id) {
        ckernel::copy_tile_init(in_cb_id);

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            ckernel::copy_tile(in_cb_id, start_in_tile_index + i, start_dst_tile_index + i);
        }
    });
}

template <size_t A, size_t B, auto Init, auto Compute>
constexpr auto _binary(uint32_t start_a_tile_index, uint32_t start_b_tile_index, uint32_t start_dst_tile_index) {
    return views::transform<0, A + 1, B + 1>([=](auto num_tiles_per_cycle, uint32_t a_cb_id, uint32_t b_cb_id) {
        Init(a_cb_id, b_cb_id);

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Compute(a_cb_id, b_cb_id, start_a_tile_index + i, start_b_tile_index + i, start_dst_tile_index + i);
        }
    });
}

template <size_t A, size_t B>
constexpr auto add(uint32_t start_a_tile_index, uint32_t start_b_tile_index, uint32_t start_dst_tile_index) {
    constexpr auto add_tiles_init = [](uint32_t a_cb_id, uint32_t b_cb_id) {
        return ckernel::add_tiles_init(a_cb_id, b_cb_id);
    };
    return _binary<A, B, +add_tiles_init, &ckernel::add_tiles>(
        start_a_tile_index, start_b_tile_index, start_dst_tile_index);
}

template <size_t A, size_t B>
constexpr auto sub(uint32_t start_a_tile_index, uint32_t start_b_tile_index, uint32_t start_dst_tile_index) {
    constexpr auto sub_tiles_init = [](uint32_t a_cb_id, uint32_t b_cb_id) {
        return ckernel::sub_tiles_init(a_cb_id, b_cb_id);
    };
    return _binary<A, B, +sub_tiles_init, &ckernel::sub_tiles>(
        start_a_tile_index, start_b_tile_index, start_dst_tile_index);
}

template <size_t A, size_t B>
constexpr auto mul(uint32_t start_a_tile_index, uint32_t start_b_tile_index, uint32_t start_dst_tile_index) {
    return _binary<A, B, &ckernel::mul_tiles_init, &ckernel::mul_tiles>(
        start_a_tile_index, start_b_tile_index, start_dst_tile_index);
}

template <auto Init, auto Compute>
constexpr auto _unary_with_param(uint32_t start_dst_tile_index, uint32_t param) {
    return views::transform<0>([=](auto num_tiles_per_cycle) {
        Init();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Compute(start_dst_tile_index + i, param);
        }
    });
}

constexpr auto add_unary(uint32_t start_dst_tile_index, uint32_t param) {
    return _unary_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::add_unary_tile>(
        start_dst_tile_index, param);
}

constexpr auto add_unary_int32(uint32_t start_dst_tile_index, uint32_t param) {
    return _unary_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::add_unary_tile_int32>(
        start_dst_tile_index, param);
}

constexpr auto sub_unary(uint32_t start_dst_tile_index, uint32_t param) {
    return _unary_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::sub_unary_tile>(
        start_dst_tile_index, param);
}

constexpr auto rsub_unary(uint32_t start_dst_tile_index, uint32_t param) {
    return _unary_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::rsub_unary_tile>(
        start_dst_tile_index, param);
}

constexpr auto mul_unary(uint32_t start_dst_tile_index, uint32_t param) {
    return _unary_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::mul_unary_tile>(
        start_dst_tile_index, param);
}

constexpr auto div_unary(uint32_t start_dst_tile_index, uint32_t param) {
    return _unary_with_param<&ckernel::binop_with_scalar_tile_init, &ckernel::div_unary_tile>(
        start_dst_tile_index, param);
}

template <auto Init, auto Compute>
constexpr auto _unary(uint32_t start_dst_tile_index) {
    return views::transform<0>([=](auto num_tiles_per_cycle) {
        Init();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Compute(start_dst_tile_index + i);
        }
    });
}

constexpr auto negative(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::negative_tile_init, &ckernel::negative_tile>(start_dst_tile_index);
}

constexpr auto negative_int32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::negative_tile_init, &ckernel::negative_tile_int32>(start_dst_tile_index);
}

constexpr auto exp(uint32_t start_dst_tile_index) {
    constexpr auto exp_tile = [](uint32_t dst_tile_index) { return ckernel::exp_tile(dst_tile_index); };
    return _unary<&ckernel::exp_tile_init, +exp_tile>(start_dst_tile_index);
}

constexpr auto power(uint32_t start_dst_tile_index, uint32_t param) {
    return _unary_with_param<&ckernel::power_tile_init, &ckernel::power_tile>(start_dst_tile_index, param);
}

constexpr auto eqz(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::eqz_tile_init, &ckernel::eqz_tile>(start_dst_tile_index);
}

constexpr auto eqz_int32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::eqz_tile_init, &ckernel::eqz_tile_int32>(start_dst_tile_index);
}

constexpr auto eqz_uint16(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::eqz_tile_init, &ckernel::eqz_tile_uint16>(start_dst_tile_index);
}

constexpr auto eqz_uint32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::eqz_tile_init, &ckernel::eqz_tile_uint32>(start_dst_tile_index);
}

constexpr auto gez(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::gez_tile_init, &ckernel::gez_tile>(start_dst_tile_index);
}

constexpr auto gez_int32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::gez_tile_init, &ckernel::gez_tile_int32>(start_dst_tile_index);
}

constexpr auto gtz(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::gtz_tile_init, &ckernel::gtz_tile>(start_dst_tile_index);
}

constexpr auto gtz_int32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::gtz_tile_init, &ckernel::gtz_tile_int32>(start_dst_tile_index);
}

constexpr auto lez(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::lez_tile_init, &ckernel::lez_tile>(start_dst_tile_index);
}

constexpr auto lez_int32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::lez_tile_init, &ckernel::lez_tile_int32>(start_dst_tile_index);
}

constexpr auto ltz(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::ltz_tile_init, &ckernel::ltz_tile>(start_dst_tile_index);
}

constexpr auto ltz_int32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::ltz_tile_init, &ckernel::ltz_tile_int32>(start_dst_tile_index);
}

constexpr auto nez(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::nez_tile_init, &ckernel::nez_tile>(start_dst_tile_index);
}

constexpr auto nez_int32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::nez_tile_init, &ckernel::nez_tile_int32>(start_dst_tile_index);
}

constexpr auto nez_uint16(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::nez_tile_init, &ckernel::nez_tile_uint16>(start_dst_tile_index);
}

constexpr auto nez_uint32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::nez_tile_init, &ckernel::nez_tile_uint32>(start_dst_tile_index);
}

constexpr auto logical_not_unary(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::logical_not_unary_tile_init, &ckernel::logical_not_unary_tile>(start_dst_tile_index);
}

constexpr auto logical_not_unary_int32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::logical_not_unary_tile_init, &ckernel::logical_not_unary_tile_int32>(start_dst_tile_index);
}

constexpr auto logical_not_unary_uint16(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::logical_not_unary_tile_init, &ckernel::logical_not_unary_tile_uint16>(start_dst_tile_index);
}

constexpr auto logical_not_unary_uint32(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::logical_not_unary_tile_init, &ckernel::logical_not_unary_tile_uint32>(start_dst_tile_index);
}

template <size_t I>
constexpr auto pack(uint32_t start_index_from_dst) {
    return views::transform<0, I + 1>([=](auto num_tiles_per_cycle, uint32_t out_cb_id) {
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ckernel::pack_tile_block(start_index_from_dst, out_cb_id, num_tiles_per_cycle);
    });
};

}  // namespace views
