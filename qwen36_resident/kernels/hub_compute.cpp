// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident hub compute: per round, x = the sum of the chips' partial slots (in pairs, fp32 in DEST, then
// bf16), on 32x32 views; the hub's BRISC hands in the slots once every chip's have arrived and multicasts
// x to the streamers.
//
// Compile-time args: 0 num_chips, 1 32x32 views per slot, 2 rounds. Row groups (named arg groups > 1): every round
// is one sub-round per active group; the hub's BRISC hands in their count (CB 3) at the start.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"

namespace {
constexpr uint32_t num_chips = get_compile_time_arg_val(0);
constexpr uint32_t V = get_compile_time_arg_val(1);
constexpr uint32_t rounds = get_compile_time_arg_val(2);
constexpr uint32_t groups = get_named_compile_time_arg_val("groups");
constexpr uint32_t cb_slots = 0, cb_x = 2, cb_count = 3;
constexpr uint32_t kDst = 8;
static_assert(num_chips == 1 || num_chips % 2 == 0, "slots are summed in pairs");
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_slots, cb_slots, cb_x);
    uint32_t sub_rounds = rounds;
    if constexpr (groups > 1) {
        cb_wait_front(cb_count, 1);
        sub_rounds = rounds * read_tile_value(cb_count, 0, 0);
        cb_pop_front(cb_count, 1);
    }
    for (uint32_t r = 0; r < sub_rounds; r++) {
        cb_wait_front(cb_slots, num_chips * V);
        cb_reserve_back(cb_x, V);
        for (uint32_t v0 = 0; v0 < V; v0 += kDst) {
            const uint32_t n = v0 + kDst <= V ? kDst : V - v0;
            tile_regs_acquire();
            if constexpr (num_chips == 1) {
                copy_init(cb_slots);
                for (uint32_t d = 0; d < n; d++) {
                    copy_tile(cb_slots, v0 + d, d);
                }
            } else {
                for (uint32_t c = 0; c < num_chips; c += 2) {
                    add_init(cb_slots, cb_slots, c > 0 /* acc_to_dest */);
                    for (uint32_t d = 0; d < n; d++) {
                        add_tiles(cb_slots, cb_slots, c * V + v0 + d, (c + 1) * V + v0 + d, d);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < n; d++) {
                pack_tile(d, cb_x);
            }
            tile_regs_release();
        }
        cb_push_back(cb_x, V);
        cb_pop_front(cb_slots, num_chips * V);
    }
}
