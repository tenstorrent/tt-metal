// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void prefetch_projection_weights() {
    DeviceZoneScopedN("PROJECTION-PREFETCH");
    constexpr uint32_t k_block = PREFETCH_ROLE == 0 ? 8 : 7;
    constexpr uint32_t width = PREFETCH_ROLE == 0 ? 28 : 16;
    constexpr uint32_t tile_bytes = PREFETCH_ROLE == 0 ? 576 : 1088;
    constexpr uint32_t bytes = PREFETCH_BLOCKS * k_block * width * tile_bytes;
    constexpr auto weight_args = TensorAccessorArgs<PREFETCH_CT_OFFSET>();
    auto* state = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(LOOP_RT_OFFSET));
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const auto weight = TensorAccessor(weight_args, state[448 + PREFETCH_ROLE], tile_bytes);
    const uint32_t storage = get_write_ptr(32);
    noc_async_read<bytes>(weight.get_noc_addr(bank * width), storage, bytes);
    noc_async_read_barrier();
    // Static CB addresses are determined by dispatch, so publish the helper's
    // address in loop-state storage allocated identically on every worker.
    // Match source/destination low address bits for a four-byte NoC write.
    state[300 + PREFETCH_ROLE] = storage;
    const uint32_t destination = get_arg_val<uint32_t>(LOOP_RT_OFFSET) + (300 + PREFETCH_ROLE) * 4;
    noc_async_write(reinterpret_cast<uint32_t>(state + 300 + PREFETCH_ROLE),
        get_noc_addr(get_arg_val<uint32_t>(PREFETCH_RT_OFFSET + 2 * bank),
                     get_arg_val<uint32_t>(PREFETCH_RT_OFFSET + 2 * bank + 1), destination), 4);
    noc_async_write_barrier();
}
