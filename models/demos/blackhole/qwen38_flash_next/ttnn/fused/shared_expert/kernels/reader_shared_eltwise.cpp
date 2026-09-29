// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared expert eltwise program, reader: this core's gate and up tiles of the concatenated [gate | up | scalar]
// linear's L1 width shard (tile pages c and gate_tiles + c), and on the scalar core the scalar tile
// (page 2 * gate_tiles).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t gus_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile = get_arg_val<uint32_t>(1);
    const uint32_t has_scalar = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr uint32_t cb_up = get_named_compile_time_arg_val("cb_up");
    constexpr uint32_t cb_scalar = get_named_compile_time_arg_val("cb_scalar");
    constexpr uint32_t gate_tiles = get_named_compile_time_arg_val("gate_tiles");
    constexpr uint32_t tile_bytes = 2048;

    constexpr auto gus_args = TensorAccessorArgs<0, 0>();
    const auto gus = TensorAccessor(gus_args, gus_addr);

    Noc noc;
    DataflowBuffer gate(cb_gate);
    DataflowBuffer up(cb_up);
    DataflowBuffer scalar(cb_scalar);

    {
        FUSED_ZONE("fz_se_r_main");
        gate.reserve_back(1);
        up.reserve_back(1);
        noc.async_read(gus, CoreLocalMem<uint32_t>(gate.get_write_ptr()), tile_bytes, {.page_id = tile}, {});
        noc.async_read(gus, CoreLocalMem<uint32_t>(up.get_write_ptr()), tile_bytes, {.page_id = gate_tiles + tile}, {});
        if (has_scalar) {
            scalar.reserve_back(1);
            noc.async_read(
                gus, CoreLocalMem<uint32_t>(scalar.get_write_ptr()), tile_bytes, {.page_id = 2 * gate_tiles}, {});
        }
        noc.async_read_barrier();
        gate.push_back(1);
        up.push_back(1);
        if (has_scalar) {
            scalar.push_back(1);
        }
    }
}
