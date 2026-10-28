// SPDX-License-Identifier: Apache-2.0
// B1 convolution row0, BF16 flat features -> exact FP32 tiled head rows.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto q_a = TensorAccessorArgs<0>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto qo_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto ko_a = TensorAccessorArgs<qo_a.next_compile_time_args_offset()>();
    constexpr auto vo_a = TensorAccessorArgs<ko_a.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(q_a, get_arg_val<uint32_t>(0), 2048);
    const auto k = TensorAccessor(k_a, get_arg_val<uint32_t>(1), 2048);
    const auto v = TensorAccessor(v_a, get_arg_val<uint32_t>(2), 2048);
    const auto qo = TensorAccessor(qo_a, get_arg_val<uint32_t>(3), 4096);
    const auto ko = TensorAccessor(ko_a, get_arg_val<uint32_t>(4), 4096);
    const auto vo = TensorAccessor(vo_a, get_arg_val<uint32_t>(5), 4096);
    const uint32_t worker = get_arg_val<uint32_t>(6);
    const uint32_t role = worker / 4;
    const uint32_t tile = worker % 4;
    CircularBuffer incoming(0), outgoing(1);
    Noc noc;
    // One core owns one complete output tile: no races, including padding.
    noc.async_write_zeros(outgoing, 4096, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    auto repack = [&](const auto& src, const auto& dst, uint32_t heads) {
        for (uint32_t h = 0; h < heads; ++h) {
            // Top two BF16 faces contain row0. One aligned1K read per head
            // avoids eight tiny reads while retaining the exact source bits.
            noc.async_read(src, incoming, 1024, {.page_id = h * 4 + tile}, {.offset_bytes = h * 1024});
        }
        noc.async_read_barrier();
        const CoreLocalMem<volatile uint16_t> in(incoming.get_write_ptr());
        const CoreLocalMem<volatile uint32_t> out(outgoing.get_write_ptr());
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t face = 0; face < 2; ++face) {
                for (uint32_t col = 0; col < 16; ++col) {
                    // BF16 -> FP32 is an exact16-bit left shift, with no
                    // arithmetic, precision change, or host conversion.
                    out[face * 256 + h * 16 + col] = static_cast<uint32_t>(in[h * 512 + face * 256 + col]) << 16;
                }
            }
        }
        auto packed = CoreLocalMem<volatile uint32_t>(outgoing.get_write_ptr());
        noc.async_write(packed, dst, 4096, {.offset_bytes = 0}, {.page_id = tile});
        noc.async_write_barrier();
    };
    if (role == 0) {
        repack(q, qo, 4);
    } else if (role == 1) {
        repack(k, ko, 4);
    } else {
        repack(v, vo, 12);
    }
}
