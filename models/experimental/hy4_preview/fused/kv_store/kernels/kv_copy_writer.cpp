// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Frame copy, write side (data movement, NOC1, on every copier core): this lane's packets of its DRAM bank,
// circular buffer -> destination tensor, the same bank. kv_copy.hpp states the job; kv_copy_reader.cpp fills the
// buffer. Pages are given back to the reader `write_batch` at a time, once their writes have left the core.
//
// runtime arguments: [bank, lane, lanes, -]
// common runtime arguments: [plan address, source address of kind 0, kind 1, destination address of kind 0, kind 1]
//
// Waits: pages from the reader of the SAME core (it pushes exactly the packets this kernel enumerates: both read
// the same plan page with the same function) and DRAM writes (always acknowledged).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "kv_copy.hpp"

namespace {
constexpr uint32_t NKINDS = get_named_compile_time_arg_val("kinds");
constexpr uint32_t CAPACITY = get_named_compile_time_arg_val("capacity");
constexpr uint32_t WRITE_BATCH = get_named_compile_time_arg_val("write_batch");
constexpr uint32_t MODE = get_named_compile_time_arg_val("mode");
constexpr uint32_t cb_plan = get_named_compile_time_arg_val("cb_plan_writer");
constexpr uint32_t cb_data[kv_copy::MAX_KINDS] = {
    get_named_compile_time_arg_val("cb_data0"), get_named_compile_time_arg_val("cb_data1")};
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
static_assert(WRITE_BATCH >= 1 && WRITE_BATCH < CAPACITY);
}  // namespace

void kernel_main() {
    if constexpr ((MODE & kv_copy::MODE_RETURN) != 0) {
        return;
    }
    constexpr auto plan_args = TensorAccessorArgs<0>();
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t lane = get_arg_val<uint32_t>(1);
    const uint32_t lanes = get_arg_val<uint32_t>(2);
    const uint32_t plan_addr = get_common_arg_val<uint32_t>(0);

    cb_reserve_back(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));
    const auto plan_tensor = TensorAccessor(plan_args, plan_addr);
    noc_async_read(plan_tensor.get_noc_addr(0), reinterpret_cast<uint32_t>(plan), kv_copy::PLAN_BYTES);
    noc_async_read_barrier();
    const uint32_t count = plan[0] < kv_copy::MAX_PAIRS ? plan[0] : kv_copy::MAX_PAIRS;
    volatile tt_l1_ptr uint32_t* pairs = plan + kv_copy::PLAN_HEADER;

    for (uint32_t k = 0; k < NKINDS; ++k) {
        const kv_copy::Kind& kd = KINDS[k];
        const uint32_t pkt = kd.packet_bytes();
        const uint32_t cb = cb_data[k];
        const uint32_t dst_addr = get_common_arg_val<uint32_t>(1 + kv_copy::MAX_KINDS + k);
        const uint32_t l1_base = get_read_ptr(cb);  // nothing of this kind was popped yet: the buffer's start
        const uint32_t l1_end = l1_base + CAPACITY * pkt;
        uint32_t l1 = l1_base;
        uint32_t pending = 0;
        auto release = [&]() {
            if constexpr ((MODE & kv_copy::MODE_NO_WRITE) == 0) {
                noc_async_writes_flushed();
            }
            cb_pop_front(cb, pending);
            pending = 0;
        };
        kv_copy::for_each_packet(kd, pairs, count, lane, lanes, [&](uint32_t, uint32_t d) {
            cb_wait_front(cb, pending + 1);
            if constexpr ((MODE & kv_copy::MODE_NO_WRITE) == 0) {
                noc_async_write(l1, get_noc_addr_from_bank_id<true>(bank, dst_addr + d), pkt);
            }
            l1 += pkt;
            if (l1 == l1_end) {
                l1 = l1_base;
            }
            if (++pending == WRITE_BATCH) {
                release();
            }
        });
        if (pending) {
            release();
        }
    }
    noc_async_write_barrier();
}
