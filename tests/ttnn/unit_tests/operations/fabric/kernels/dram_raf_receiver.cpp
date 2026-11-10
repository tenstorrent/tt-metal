// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Receiver of one probe stream (see test_fabric_dram_read_after_flag.py). Runs on NCRISC; the host
// picks its NoC (NOC0 by default, the configuration that showed the problem).
// For k = 0..n-1 (seq = base + k):
//   1. wait until the flag in this core's L1 is >= 1. The remote router increments it with a
//      flushed atomic after its non-posted DRAM write of the page was acknowledged;
//   2. noc_async_read the page from DRAM through this kernel's NoC (address resolved for that NoC),
//      read barrier, invalidate_l1_cache;
//   3. EXPECTED: the page carries seq (flag seen => data visible). If it does not, record the read
//      and re-read (with a short delay) until it does, counting re-reads and cycles;
//   4. flag -= 1 (atomic), and return a credit to the sender over the fabric.
// With P = 2 at most one page is in flight, so the flag is exactly 1 when the wait ends; a larger
// value would mean an extra increment (counted as flag_over).
//
// Result row (uint32, accumulated over launches) in this core's L1:
//   [0] txns lo [1] txns hi [2] mismatches [3] head-bad [4] tail-only-bad (torn) [5] unresolved
//   [6] max re-reads [7] max cycles [8] timeouts [9] flag_over [10] final flag != 0 [11] logged
//   [16 + 8 i] mismatch i (< 8): seq, w0, w1, wtail, flag at wait, re-reads, cycles, seq - w0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

using namespace tt::tt_fabric;
static_assert(std::is_same_v<PACKET_HEADER_TYPE, HybridMeshPacketHeader>, "needs a 2D fabric");

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t P = get_compile_time_arg_val(1);
    constexpr uint32_t read_cb = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr uint32_t words = page_bytes / 4;
    constexpr uint32_t kMaxRereads = 100000, kDelay = 64, kLog = 8;

    size_t a = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page_base = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t send_x = get_arg_val<uint32_t>(a++);
    const uint32_t send_y = get_arg_val<uint32_t>(a++);
    const uint32_t flag_addr = get_arg_val<uint32_t>(a++);
    const uint32_t credit_addr = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* state = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    const ccl_routing_utils::line_unicast_route_info_t route{
        .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++)),
        .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++))};
    const uint32_t timeout = get_arg_val<uint32_t>(a++);
    const uint32_t num_stop = get_arg_val<uint32_t>(a++);  // hammer cores to release: (x, y)
    const uint32_t stop_w = get_arg_val<uint32_t>(a++);    // their writer's and reader's stop words
    const uint32_t stop_r = get_arg_val<uint32_t>(a++);
    const uint32_t stop_args = a;
    a += 2 * num_stop;
    auto connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    const uint32_t buf = get_write_ptr(read_cb);
    volatile tt_l1_ptr uint32_t* rw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    const uint32_t base = *state;

    auto* header = PacketHeaderPool::allocate_header();
    ccl_routing_utils::fabric_set_line_unicast_route(header, route);
    header->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{get_noc_addr(send_x, send_y, credit_addr), 1, true});
    connection.open();

    auto read_page = [&](uint64_t addr) {
        noc_async_read(addr, buf, page_bytes);
        noc_async_read_barrier();
        invalidate_l1_cache();
    };
    auto head_ok = [&](uint32_t seq) { return rw[0] == seq && rw[1] == ~seq; };
    auto tail_ok = [&](uint32_t seq) { return rw[words - 1] == (seq ^ 0xA5A5A5A5u); };
    auto max_to = [&](uint32_t i, uint32_t v) { res[i] = v > res[i] ? v : res[i]; };

    uint32_t txn = 0, timed_out = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t seq = base + k;
        uint32_t fv, prev = get_timestamp_32b(), elapsed = 0;  // low clock word only: wrap-safe
        while ((invalidate_l1_cache(), fv = *flag) == 0) {
            const uint32_t now = get_timestamp_32b();
            elapsed += now - prev;
            prev = now;
            if (elapsed > timeout) {
                timed_out = 1;
                break;
            }
        }
        if (timed_out) {
            break;
        }
        res[9] += fv > P - 1 ? 1 : 0;

        const uint64_t addr = src.get_noc_addr(page_base + seq % P);  // for this kernel's NoC
        read_page(addr);
        if (!(head_ok(seq) && tail_ok(seq))) {  // flag seen, data not visible
            const uint32_t w0 = rw[0], w1 = rw[1], wt = rw[words - 1];
            const bool head_bad = !head_ok(seq);
            const uint32_t t0 = get_timestamp_32b();
            uint32_t rr = 0;
            while (rr < kMaxRereads && !(head_ok(seq) && tail_ok(seq))) {
                for (uint32_t s = get_timestamp_32b(); get_timestamp_32b() - s < kDelay;) {
                }
                read_page(addr);
                ++rr;
            }
            const uint32_t cyc = get_timestamp_32b() - t0;
            res[2] += 1;
            res[head_bad ? 3 : 4] += 1;
            res[5] += rr == kMaxRereads ? 1 : 0;
            max_to(6, rr);
            max_to(7, cyc);
            if (res[11] < kLog) {
                volatile tt_l1_ptr uint32_t* r = res + 16 + 8 * res[11];
                r[0] = seq, r[1] = w0, r[2] = w1, r[3] = wt, r[4] = fv, r[5] = rr, r[6] = cyc, r[7] = seq - w0;
                res[11] += 1;
            }
        }
        ++txn;
        noc_semaphore_inc(get_noc_addr(flag_addr), 0xFFFFFFFFu);  // flag -= 1
        noc_async_atomic_barrier();
        connection.wait_for_empty_write_slot();  // credit back to the sender
        connection.send_payload_flush_blocking_from_address(
            reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
    }
    invalidate_l1_cache();
    if (!timed_out) {
        if (*flag != 0) {
            res[10] += 1;
            *flag = 0;
        }
        *state = base + n;
    }
    for (uint32_t h = 0; h < num_stop; ++h) {  // release this chip's hammer cores
        const uint32_t x = get_arg_val<uint32_t>(stop_args + 2 * h), y = get_arg_val<uint32_t>(stop_args + 2 * h + 1);
        noc_semaphore_inc(get_noc_addr(x, y, stop_w), 1);
        noc_semaphore_inc(get_noc_addr(x, y, stop_r), 1);
    }
    noc_async_atomic_barrier();
    connection.close();
    const uint64_t t = (static_cast<uint64_t>(res[1]) << 32 | res[0]) + txn;
    res[0] = static_cast<uint32_t>(t);
    res[1] = static_cast<uint32_t>(t >> 32);
    res[8] += timed_out;
}
