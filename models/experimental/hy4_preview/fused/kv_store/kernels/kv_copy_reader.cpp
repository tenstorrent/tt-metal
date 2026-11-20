// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Frame copy, read side (data movement, NOC0, on every copier core): this lane's packets of its DRAM bank, source
// tensor -> circular buffer, `depth` packets in flight on their own NoC transaction ids (the stream pattern of
// fused/indexer_scan/kernels/scan_reader.cpp). kv_copy.hpp states the job; kv_copy_writer.cpp empties the buffer.
//
// runtime arguments: [bank, lane, lanes, response virtual channel]
// common runtime arguments: [plan address, source address of kind 0, kind 1, destination address of kind 0, kind 1]
//
// Waits: DRAM reads (always answered) and room in the buffer, which the writer of the SAME core frees on its own;
// the buffer holds depth + write_batch pages, so the two never wait for each other in a cycle.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "kv_copy.hpp"

namespace {
constexpr uint32_t NKINDS = get_named_compile_time_arg_val("kinds");
constexpr uint32_t DEPTH = get_named_compile_time_arg_val("depth");
constexpr uint32_t CAPACITY = get_named_compile_time_arg_val("capacity");  // pages of each data buffer
constexpr uint32_t MODE = get_named_compile_time_arg_val("mode");
constexpr uint32_t cb_plan = get_named_compile_time_arg_val("cb_plan_reader");
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
static_assert(NKINDS >= 1 && NKINDS <= kv_copy::MAX_KINDS && DEPTH >= 1 && DEPTH <= 8 && CAPACITY > DEPTH);
}  // namespace

void kernel_main() {
    if constexpr ((MODE & kv_copy::MODE_RETURN) != 0) {
        return;
    }
    constexpr auto plan_args = TensorAccessorArgs<0>();
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t lane = get_arg_val<uint32_t>(1);
    const uint32_t lanes = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t plan_addr = get_common_arg_val<uint32_t>(0);

    cb_reserve_back(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));
    const auto plan_tensor = TensorAccessor(plan_args, plan_addr);
    noc_async_read(plan_tensor.get_noc_addr(0), reinterpret_cast<uint32_t>(plan), kv_copy::PLAN_BYTES);
    noc_async_read_barrier();
    const uint32_t count = plan[0] < kv_copy::MAX_PAIRS ? plan[0] : kv_copy::MAX_PAIRS;
    volatile tt_l1_ptr uint32_t* pairs = plan + kv_copy::PLAN_HEADER;

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t k = 0; k < NKINDS; ++k) {
        const kv_copy::Kind& kd = KINDS[k];
        const uint32_t pkt = kd.packet_bytes();
        const uint32_t cb = cb_data[k];
        [[maybe_unused]] const uint64_t base =
            get_noc_addr_from_bank_id<true>(bank, get_common_arg_val<uint32_t>(1 + k));
        const uint32_t l1_base = get_write_ptr(cb);  // nothing of this kind was pushed yet: the buffer's start
        const uint32_t l1_end = l1_base + CAPACITY * pkt;
        uint32_t l1 = l1_base;
        uint32_t issued = 0, landed = 0;
        auto land = [&]() {
            if constexpr ((MODE & kv_copy::MODE_NO_READ) == 0) {
                noc_async_read_barrier_with_trid(landed % DEPTH + 1);
            }
            cb_push_back(cb, 1);
            ++landed;
        };
        if constexpr ((MODE & kv_copy::MODE_NO_READ) == 0) {
            noc_async_read_one_packet_set_state<true>(base, pkt, vc);
        }
        kv_copy::for_each_packet(kd, pairs, count, lane, lanes, [&](uint32_t s, uint32_t) {
            if (issued - landed == DEPTH) {
                land();
            }
            cb_reserve_back(cb, issued - landed + 1);
            if constexpr ((MODE & kv_copy::MODE_NO_READ) == 0) {
                const uint32_t trid = issued % DEPTH + 1;
                noc_async_read_set_trid(trid);
                noc_async_read_one_packet_with_state_with_trid(static_cast<uint32_t>(base), s, l1, trid);
            }
            l1 += pkt;
            if (l1 == l1_end) {
                l1 = l1_base;
            }
            ++issued;
        });
        while (landed < issued) {
            land();
        }
    }
    noc_async_read_set_trid(0);
}
