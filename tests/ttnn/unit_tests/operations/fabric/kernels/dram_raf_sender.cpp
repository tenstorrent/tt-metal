// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sender of one probe stream (see test_fabric_dram_read_after_flag.py).
// For k = 0..n-1 (seq = base + k): write one page into page (seq % P) of the neighbour chip's
// interleaved DRAM buffer with ONE fabric packet: a fused unicast write + atomic increment
// (flush=true) of the receiver's flag. The destination address is resolved for the router's NoC
// (linear::addrgen_detail::get_noc_address). The page carries seq in its first and last words.
// At most P-1 pages are unacknowledged: the receiver returns a credit (fabric atomic inc of
// `credit`) after reading each page, so a page is never overwritten before it was read.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"

using namespace tt::tt_fabric;
static_assert(std::is_same_v<PACKET_HEADER_TYPE, HybridMeshPacketHeader>, "needs a 2D fabric");

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t P = get_compile_time_arg_val(1);
    constexpr uint32_t payload_cb = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();
    constexpr uint32_t words = page_bytes / 4;

    size_t a = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(a++);
    const uint32_t page_base = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t recv_x = get_arg_val<uint32_t>(a++);
    const uint32_t recv_y = get_arg_val<uint32_t>(a++);
    const uint32_t flag_addr = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    volatile tt_l1_ptr uint32_t* state = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    volatile tt_l1_ptr uint32_t* result =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));  // L1 row
    const ccl_routing_utils::line_unicast_route_info_t route{
        .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++)),
        .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++))};
    const uint32_t timeout = get_arg_val<uint32_t>(a++);
    auto connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    const auto dst = TensorAccessor(dst_args, dst_addr, page_bytes);
    const uint32_t payload = get_write_ptr(payload_cb);
    volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload);
    const uint32_t base = *state;  // seq continues across launches

    auto* header = PacketHeaderPool::allocate_header();
    ccl_routing_utils::fabric_set_line_unicast_route(header, route);
    const uint64_t flag_noc = get_noc_addr(recv_x, recv_y, flag_addr);
    connection.open();

    // Wait for credit >= need. Elapsed time from the low wall-clock word only: an (L, H) read can
    // tear at the low-word wrap and fake a timeout.
    auto wait_credit = [&](uint32_t need) {
        uint32_t prev = get_timestamp_32b(), elapsed = 0;
        while (true) {
            invalidate_l1_cache();
            if (*credit >= need) {
                return true;
            }
            const uint32_t now = get_timestamp_32b();
            elapsed += now - prev;
            prev = now;
            if (elapsed > timeout) {
                return false;
            }
        }
    };

    uint32_t sent = 0;
    bool ok = true;
    for (uint32_t k = 0; k < n; ++k) {
        if (k + 1 >= P && !wait_credit(k + 2 - P)) {  // send k once k-(P-1) was read: <= P-1 outstanding
            ok = false;
            break;
        }
        const uint32_t seq = base + k;
        pw[0] = seq;  // previous send was flush-blocking: the payload slot is free
        pw[1] = ~seq;
        pw[words - 1] = seq ^ 0xA5A5A5A5u;
        const uint64_t page_noc = tt::tt_fabric::linear::addrgen_detail::get_noc_address(dst, page_base + seq % P);
        header->to_noc_fused_unicast_write_atomic_inc(
            NocUnicastAtomicIncFusedCommandHeader{page_noc, flag_noc, 1, true}, page_bytes);
        connection.wait_for_empty_write_slot();
        connection.send_payload_without_header_non_blocking_from_address(payload, page_bytes);
        connection.send_payload_flush_blocking_from_address(
            reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
        ++sent;
    }
    ok = ok && wait_credit(n);  // every page read
    if (ok) {
        *credit = 0;
        *state = base + n;
    }
    connection.close();
    result[0] += sent;  // [sent, timeouts], accumulated over launches
    result[1] += ok ? 0 : 1;
}
