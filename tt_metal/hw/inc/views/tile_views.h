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

template <
    size_t A,
    size_t B,
    void (*Init)(uint32_t, uint32_t),
    void (*Tile)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)>
constexpr auto _binary(uint32_t start_a_tile_index, uint32_t start_b_tile_index, uint32_t start_dst_tile_index) {
    return views::transform<0, A + 1, B + 1>([=](auto num_tiles_per_cycle, uint32_t a_cb_id, uint32_t b_cb_id) {
        Init(a_cb_id, b_cb_id);

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Tile(a_cb_id, b_cb_id, start_a_tile_index + i, start_b_tile_index + i, start_dst_tile_index + i);
        }
    });
}

template <size_t A, size_t B>
inline constexpr auto add = &_binary<A, B, &ckernel::add_tiles_init, &ckernel::add_tiles>;

template <size_t A, size_t B>
inline constexpr auto sub = &_binary<A, B, &ckernel::sub_tiles_init, &ckernel::sub_tiles>;

template <size_t A, size_t B>
inline constexpr auto mul = &_binary<A, B, &ckernel::mul_tiles_init, &ckernel::mul_tiles>;

template <void (*Init)(), void (*Tile)(uint32_t, uint32_t)>
constexpr auto _unary_with_param(uint32_t start_dst_tile_index, uint32_t param) {
    return views::transform<0>([=](auto num_tiles_per_cycle) {
        Init();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Tile(start_dst_tile_index + i, param);
        }
    });
}

template <void (*Tile)(uint32_t, uint32_t)>
constexpr auto _binop_with_scalar(uint32_t start_dst_tile_index, uint32_t param) {
    return _unary_with_param<&ckernel::binop_with_scalar_tile_init, Tile>(start_dst_tile_index, param);
}

inline constexpr auto add_unary = &_binop_with_scalar<&ckernel::add_unary_tile>;
inline constexpr auto add_unary_int32 = &_binop_with_scalar<&ckernel::add_unary_tile_int32>;
inline constexpr auto sub_unary = &_binop_with_scalar<&ckernel::sub_unary_tile>;
inline constexpr auto rsub_unary = &_binop_with_scalar<&ckernel::rsub_unary_tile>;
inline constexpr auto mul_unary = &_binop_with_scalar<&ckernel::mul_unary_tile>;
inline constexpr auto div_unary = &_binop_with_scalar<&ckernel::div_unary_tile>;

template <void (*Init)(), void (*Tile)(uint32_t)>
constexpr auto _unary(uint32_t start_dst_tile_index) {
    return views::transform<0>([=](auto num_tiles_per_cycle) {
        Init();

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            Tile(start_dst_tile_index + i);
        }
    });
}

inline constexpr auto negative = &_unary<&ckernel::negative_tile_init, &ckernel::negative_tile>;
inline constexpr auto negative_int32 = &_unary<&ckernel::negative_tile_init, &ckernel::negative_tile_int32>;

inline constexpr auto exp = &_unary<&ckernel::exp_tile_init, &ckernel::exp_tile>;

inline constexpr auto power = &_unary_with_param<&ckernel::power_tile_init, &ckernel::power_tile>;

template <void (*Tile)(uint32_t)>
constexpr auto _eqz(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::eqz_tile_init, Tile>(start_dst_tile_index);
}

inline constexpr auto eqz = &_eqz<&ckernel::eqz_tile>;
inline constexpr auto eqz_int32 = &_eqz<&ckernel::eqz_tile_int32>;
inline constexpr auto eqz_uint16 = &_eqz<&ckernel::eqz_tile_uint16>;
inline constexpr auto eqz_uint32 = &_eqz<&ckernel::eqz_tile_uint32>;

template <void (*Tile)(uint32_t)>
constexpr auto _gez(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::gez_tile_init, Tile>(start_dst_tile_index);
}

inline constexpr auto gez = &_gez<&ckernel::gez_tile>;
inline constexpr auto gez_int32 = &_gez<&ckernel::gez_tile_int32>;

template <void (*Tile)(uint32_t)>
constexpr auto _gtz(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::gtz_tile_init, Tile>(start_dst_tile_index);
}

inline constexpr auto gtz = &_gtz<&ckernel::gtz_tile>;
inline constexpr auto gtz_int32 = &_gtz<&ckernel::gtz_tile_int32>;

template <void (*Tile)(uint32_t)>
constexpr auto _lez(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::lez_tile_init, Tile>(start_dst_tile_index);
}

inline constexpr auto lez = &_lez<&ckernel::lez_tile>;
inline constexpr auto lez_int32 = &_lez<&ckernel::lez_tile_int32>;

template <void (*Tile)(uint32_t)>
constexpr auto _ltz(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::ltz_tile_init, Tile>(start_dst_tile_index);
}

inline constexpr auto ltz = &_ltz<&ckernel::ltz_tile>;
inline constexpr auto ltz_int32 = &_ltz<&ckernel::ltz_tile_int32>;

template <void (*Tile)(uint32_t)>
constexpr auto _nez(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::nez_tile_init, Tile>(start_dst_tile_index);
}

inline constexpr auto nez = &_nez<&ckernel::nez_tile>;
inline constexpr auto nez_int32 = &_nez<&ckernel::nez_tile_int32>;
inline constexpr auto nez_uint16 = &_nez<&ckernel::nez_tile_uint16>;
inline constexpr auto nez_uint32 = &_nez<&ckernel::nez_tile_uint32>;

template <void (*Tile)(uint32_t)>
constexpr auto _logical_not_unary(uint32_t start_dst_tile_index) {
    return _unary<&ckernel::logical_not_unary_tile_init, Tile>(start_dst_tile_index);
}

inline constexpr auto logical_not_unary = &_logical_not_unary<&ckernel::logical_not_unary_tile>;
inline constexpr auto logical_not_unary_int32 = &_logical_not_unary<&ckernel::logical_not_unary_tile_int32>;
inline constexpr auto logical_not_unary_uint16 = &_logical_not_unary<&ckernel::logical_not_unary_tile_uint16>;
inline constexpr auto logical_not_unary_uint32 = &_logical_not_unary<&ckernel::logical_not_unary_tile_uint32>;

template <size_t I>
constexpr auto pack(uint32_t start_index_from_dst) {
    return views::transform<0, I + 1>([=](auto num_tiles_per_cycle, uint32_t out_cb_id) {
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ckernel::pack_tile_block(start_index_from_dst, out_cb_id, num_tiles_per_cycle);
    });
};

}  // namespace views
