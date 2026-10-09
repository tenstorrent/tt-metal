// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto voa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto goa = TensorAccessorArgs<voa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0), 2048);
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1), 2048);
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2), 2048);
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3), 4096);
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4), 2048);
    const auto values = TensorAccessor(voa, get_arg_val<uint32_t>(5), 512);
    const auto gates = TensorAccessor(goa, get_arg_val<uint32_t>(6), 32);
    const uint32_t first = get_arg_val<uint32_t>(7);
    const uint32_t stride = get_arg_val<uint32_t>(8);
    const uint32_t count = get_arg_val<uint32_t>(9);
    // CB9 is reader-private staging, independent of writer scratch CB10.
    const uint32_t scratch = get_write_ptr(9);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t head = first + item * stride;
        const uint32_t batch = head / 4;
        cb_reserve_back(0, 4);
        cb_reserve_back(1, 4);
        for (uint32_t tile = 0; tile < 4; ++tile) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t offset = tile * 64 + face * 32;
                // Only token zero. Each 32-byte read is one BF16 face row.
                noc_async_read(q.get_noc_addr(head * 4 + tile, face * 512), scratch + offset, 32);
                noc_async_read(k.get_noc_addr(head * 4 + tile, face * 512), scratch + 256 + offset, 32);
            }
        }
        // The twelve gates fit in face zero of one padded tile per user.
        noc_async_read(decay.get_noc_addr(batch), scratch + 1280, 64);
        noc_async_read(beta.get_noc_addr(batch), scratch + 1344, 32);
        noc_async_read_barrier();
        const auto* raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
        auto* qc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
        auto* kc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
        for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t lane = i % 32;
            const uint32_t column = (i / 32) * 1024 + (lane / 16) * 512 + (lane % 16) * 16;
            // Exact BF16 -> FP32 expansion; no new arithmetic rounding.
            qc[column] = static_cast<uint32_t>(raw[i]) << 16;
            kc[column] = static_cast<uint32_t>(raw[128 + i]) << 16;
        }
        cb_push_back(0, 4);
        cb_push_back(1, 4);
        // Each shared Q/K head serves three disjoint value heads. Their output
        // rows are written once, with a barrier before staging storage is reused.
        for (uint32_t repeat = 0; repeat < 3; ++repeat) {
            const uint32_t value_head = head * 3 + repeat;
            const uint32_t local_head = value_head % 12;
            for (uint32_t tile = 0; tile < 4; ++tile) {
                for (uint32_t face = 0; face < 2; ++face) {
                    noc_async_read(
                        v.get_noc_addr(value_head * 4 + tile, face * 512), scratch + 512 + tile * 64 + face * 32, 32);
                }
            }
            noc_async_read_barrier();
            const auto* raw_v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 512);
            auto* output_v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 768);
            for (uint32_t i = 0; i < 128; ++i) {
                output_v[i] = static_cast<uint32_t>(raw_v[i]) << 16;
            }
            const auto* raw_decay = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 1280);
            const auto* raw_beta = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 1344);
            auto* output_g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 1376);
            output_g[0] = raw_decay[local_head];
            output_g[1] = static_cast<uint32_t>(raw_beta[local_head]) << 16;
            for (uint32_t i = 2; i < 8; ++i) {
                output_g[i] = 0;
            }
            noc_async_write_page(value_head, values, scratch + 768);
            noc_async_write_page(value_head, gates, scratch + 1376);
            noc_async_write_barrier();
        }
    }
}
