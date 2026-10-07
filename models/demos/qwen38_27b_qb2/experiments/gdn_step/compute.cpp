// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"

namespace {
// All input copies unpack directly into FP32 DEST. No FPU arithmetic or
// DEST-to-Src reuse is used for either the state or the cancellation in delta.
void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_full_operand<SrcOrder::Regular>(cb, cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
template <BroadcastType B>
void broadcast(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_full_operand<SrcOrder::Regular>(cb, cb);
    unary_bcast_init<B>(cb);
    unary_bcast<B>(cb, tile, dst);
    unary_bcast_uninit<B>(cb);
}
void multiply(uint32_t a, uint32_t b, uint32_t dst) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, dst);
}
void add(uint32_t a, uint32_t b, uint32_t dst) {
    add_binary_tile_init();
    add_binary_tile(a, b, dst);
}
// The caller owns a reserved output window and an acquired DEST bank.
void emit(uint32_t dst, uint32_t cb, uint32_t tile) {
    pack_reconfig_data_format<true>(cb);
    pack_init(cb);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(dst, cb, tile);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 2, 8);
    for (uint32_t item = 0; item < count; ++item) {
        for (uint32_t cb = 0; cb < 3; ++cb) {
            cb_wait_front(cb, 4);
        }
        cb_wait_front(3, 1);
        cb_wait_front(4, 1);
        cb_wait_front(5, 16);
        cb_reserve_back(6, 4);
        // delta = beta * (v - k^T (decay * S)). Four partial rows
        // accumulate in DEST before the FP32 column reduction over K.
        for (uint32_t vc = 0; vc < 4; ++vc) {
            tile_regs_acquire();
            for (uint32_t kr = 0; kr < 4; ++kr) {
                const uint32_t product = kr == 0 ? 3 : 0;
                load(5, kr * 4 + vc, product);
                broadcast<BroadcastType::SCALAR>(3, 0, 1);
                multiply(product, 1, product);
                broadcast<BroadcastType::COL>(1, kr, 1);
                multiply(product, 1, product);
                if (kr != 0) {
                    add(3, 0, 3);
                }
            }
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(3);
            load(2, vc, 1);
            sub_binary_tile_init();
            sub_binary_tile(1, 3, 0);
            broadcast<BroadcastType::SCALAR>(4, 0, 1);
            multiply(0, 1, 0);
            emit(0, 6, vc);
        }
        cb_push_back(6, 4);
        cb_wait_front(6, 4);
        cb_reserve_back(7, 16);
        // S_new = decay * S + k outer delta, preserving FP32 throughout.
        for (uint32_t kr = 0; kr < 4; ++kr) {
            for (uint32_t vc = 0; vc < 4; ++vc) {
                tile_regs_acquire();
                load(5, kr * 4 + vc, 0);
                broadcast<BroadcastType::SCALAR>(3, 0, 1);
                multiply(0, 1, 0);
                broadcast<BroadcastType::COL>(1, kr, 1);
                broadcast<BroadcastType::ROW>(6, vc, 2);
                multiply(1, 2, 1);
                add(0, 1, 0);
                emit(0, 7, kr * 4 + vc);
            }
        }
        cb_push_back(7, 16);
        cb_wait_front(7, 16);
        cb_reserve_back(8, 4);
        // Output = q^T S_new. The writer waits for CB8 BEFORE consuming
        // CB7, so these reads cannot race CB7 reclamation by the writer.
        for (uint32_t vc = 0; vc < 4; ++vc) {
            tile_regs_acquire();
            for (uint32_t kr = 0; kr < 4; ++kr) {
                const uint32_t product = kr == 0 ? 3 : 0;
                load(7, kr * 4 + vc, product);
                broadcast<BroadcastType::COL>(0, kr, 1);
                multiply(product, 1, product);
                if (kr != 0) {
                    add(3, 0, 3);
                }
            }
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(3);
            emit(3, 8, vc);
        }
        cb_push_back(8, 4);
        for (uint32_t cb = 0; cb < 3; ++cb) {
            cb_pop_front(cb, 4);
        }
        cb_pop_front(3, 1);
        cb_pop_front(4, 1);
        cb_pop_front(5, 16);
        cb_pop_front(6, 4);
        // CB7 and CB8 have exactly one consumer: the writer.
    }
}
