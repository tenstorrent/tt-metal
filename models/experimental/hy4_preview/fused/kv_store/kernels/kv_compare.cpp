// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Frame compare (data movement, one processor, on every copier core): the packets kv_copy would move for a plan
// are read from BOTH tensors and compared in L1; nothing is written to either. For checking a pool against a
// contiguous slot on all chips without moving a byte to the host.
//
// The plan (kv_copy.hpp) lists the frames of ONE conversation in order: pair j = (frame in tensor A, frame in
// tensor B) of its positions [8192 j, 8192 j + 8192). Plan word [1] is the number of positions to compare, a
// multiple of 32: of this chip's stripe only the rows of positions below it are compared (whole 32-row shards:
// shard (j, bank) holds stripe rows [256 j + 32 bank, + 32)). Of every row `row_bytes` of `row_stride` are compared
// (a packed latent row is 656 bytes in a 704-byte page).
//
// Result: page (bank * lanes + lane) of the result tensor, uint32:
//   [0] 0x4B564350  [1] packets compared  [2] rows compared  [3] rows that differ
//   [4] pair, [5] kind, [6] source offset of the first row that differs (when [3] != 0)
//
// runtime arguments: [bank, lane, lanes, chip]   common: [plan, A kind 0, A kind 1, B kind 0, B kind 1, result]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "kv_copy.hpp"

namespace {
constexpr uint32_t NKINDS = get_named_compile_time_arg_val("kinds");
constexpr uint32_t cb_plan = get_named_compile_time_arg_val("cb_plan_reader");
constexpr uint32_t cb_a = get_named_compile_time_arg_val("cb_data0");
constexpr uint32_t cb_b = get_named_compile_time_arg_val("cb_data1");
constexpr uint32_t ROW_STRIDE[kv_copy::MAX_KINDS] = {
    get_named_compile_time_arg_val("k0_row_stride"), get_named_compile_time_arg_val("k1_row_stride")};
constexpr uint32_t ROW_BYTES[kv_copy::MAX_KINDS] = {
    get_named_compile_time_arg_val("k0_row_bytes"), get_named_compile_time_arg_val("k1_row_bytes")};
constexpr kv_copy::Kind KINDS[kv_copy::MAX_KINDS] = {
    {get_named_compile_time_arg_val("k0_shard_bytes"),
     get_named_compile_time_arg_val("k0_packets"),
     get_named_compile_time_arg_val("k0_slots"),
     get_named_compile_time_arg_val("k0_src_frame"),
     get_named_compile_time_arg_val("k0_src_slot"),
     get_named_compile_time_arg_val("k0_dst_frame"),
     get_named_compile_time_arg_val("k0_dst_slot")},
    {get_named_compile_time_arg_val("k1_shard_bytes"),
     get_named_compile_time_arg_val("k1_packets"),
     get_named_compile_time_arg_val("k1_slots"),
     get_named_compile_time_arg_val("k1_src_frame"),
     get_named_compile_time_arg_val("k1_src_slot"),
     get_named_compile_time_arg_val("k1_dst_frame"),
     get_named_compile_time_arg_val("k1_dst_slot")}};
}  // namespace

void kernel_main() {
    constexpr auto plan_args = TensorAccessorArgs<0>();
    constexpr auto result_args = TensorAccessorArgs<plan_args.next_compile_time_args_offset()>();
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t lane = get_arg_val<uint32_t>(1);
    const uint32_t lanes = get_arg_val<uint32_t>(2);
    const uint32_t chip = get_arg_val<uint32_t>(3);
    const uint32_t plan_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t result_addr = get_common_arg_val<uint32_t>(1 + 2 * kv_copy::MAX_KINDS);

    cb_reserve_back(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));
    const auto plan_tensor = TensorAccessor(plan_args, plan_addr);
    noc_async_read(plan_tensor.get_noc_addr(0), reinterpret_cast<uint32_t>(plan), kv_copy::PLAN_BYTES);
    noc_async_read_barrier();
    const uint32_t count = plan[0] < kv_copy::MAX_PAIRS ? plan[0] : kv_copy::MAX_PAIRS;
    const uint32_t positions = plan[1];
    volatile tt_l1_ptr uint32_t* pairs = plan + kv_copy::PLAN_HEADER;

    // stripe rows of this chip that hold a position below `positions` (frames.py visible_rows)
    const uint32_t in_period = positions & 4095;
    const uint32_t mine = in_period > chip * 128 ? in_period - chip * 128 : 0;
    const uint32_t visible = ((positions >> 12) << 7) + (mine < 128 ? mine : 128);

    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);
    const uint32_t a = get_write_ptr(cb_a);
    const uint32_t b = get_write_ptr(cb_b);
    uint32_t packets = 0, rows = 0, bad = 0, bad_pair = 0, bad_kind = 0, bad_offset = 0;

    for (uint32_t k = 0; k < NKINDS; ++k) {
        const kv_copy::Kind& kd = KINDS[k];
        const uint32_t pkt = kd.packet_bytes();
        const uint32_t a_addr = get_common_arg_val<uint32_t>(1 + k);
        const uint32_t b_addr = get_common_arg_val<uint32_t>(1 + kv_copy::MAX_KINDS + k);
        const uint32_t words = ROW_BYTES[k] / 4;
        uint32_t turn = 0;
        for (uint32_t j = 0; j < count; ++j) {
            const bool valid = j * 256 + bank * 32 < visible;
            uint32_t s = pairs[2 * j] * kd.src_frame;
            uint32_t d = pairs[2 * j + 1] * kd.dst_frame;
            for (uint32_t l = 0; l < kd.slots; ++l) {
                for (uint32_t h = 0; h < kd.packets; ++h) {
                    if (turn == lane && valid) {
                        noc_async_read_one_packet(get_noc_addr_from_bank_id<true>(bank, a_addr + s + h * pkt), a, pkt);
                        noc_async_read_one_packet(get_noc_addr_from_bank_id<true>(bank, b_addr + d + h * pkt), b, pkt);
                        noc_async_read_barrier();
                        for (uint32_t r = 0; r < pkt; r += ROW_STRIDE[k]) {
                            volatile tt_l1_ptr uint32_t* x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a + r);
                            volatile tt_l1_ptr uint32_t* y = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b + r);
                            uint32_t diff = 0;
                            for (uint32_t i = 0; i < words; ++i) {
                                diff |= x[i] ^ y[i];
                            }
                            if (diff) {
                                if (bad == 0) {
                                    bad_pair = j;
                                    bad_kind = k;
                                    bad_offset = s + h * pkt + r;
                                }
                                ++bad;
                            }
                            ++rows;
                        }
                        ++packets;
                    }
                    turn = turn + 1 == lanes ? 0 : turn + 1;
                }
                s += kd.src_slot;
                d += kd.dst_slot;
            }
        }
    }

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a);
    out[0] = 0x4B564350;
    out[1] = packets;
    out[2] = rows;
    out[3] = bad;
    out[4] = bad_pair;
    out[5] = bad_kind;
    out[6] = bad_offset;
    out[7] = visible;
    const auto result = TensorAccessor(result_args, result_addr);
    noc_async_write(a, result.get_noc_addr(bank * lanes + lane), 64);
    noc_async_write_barrier();
}
