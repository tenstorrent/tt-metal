// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K3.1 EP all-to-all prototype — writer (BRISC).
//
// ROLE_EAST/ROLE_WEST: owns one tt-fabric connection (1D line, one link, one direction).
//   dispatch: each needed token row is multicast once over hops [1, far] to page (MY*T + t) of recv_x
//             on every chip in range (chips in range that the token did not hit just get an unused slot).
//             Link 0 also multicasts this device's T x 64 B metadata block to recv_meta page MY.
//   combine : each needed partial row (s, t) is unicast |s - MY| hops to page (MY*T + t) of comb on s.
//   Then ONE multicast atomic-inc (flush) over the whole direction = per-batch completion flag.
//   Packets are posted non-blocking (EDM credit flow control only); one NOC write flush per row.
// ROLE_SELF: local copies, then waits for (N-1)*L completion flags and subtracts them back
//   (atomic add of -EXPECT, so early flags of the next invocation are never lost; no init barrier).
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ep_common.hpp"
#ifndef EP_FLUSH_EVERY
#define EP_FLUSH_EVERY 1
#endif

using namespace k31ep;

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t MY = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t EPD = get_compile_time_arg_val(5);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(6);
    constexpr uint32_t L = get_compile_time_arg_val(7);
    constexpr uint32_t LINK = get_compile_time_arg_val(8);
    constexpr uint32_t ROLE = get_compile_time_arg_val(9) & 0xFF;
    constexpr bool RING = (get_compile_time_arg_val(9) >> 8) & 1;
    constexpr uint32_t meta_cb = get_compile_time_arg_val(10);
    constexpr uint32_t data_cb = get_compile_time_arg_val(11);
    constexpr uint32_t hdr_cb = get_compile_time_arg_val(12);
    constexpr uint32_t PAYLOAD = get_compile_time_arg_val(13);
    constexpr uint32_t EXPECT = get_compile_time_arg_val(14);
    constexpr uint32_t HDR_BYTES = get_compile_time_arg_val(15);
    static_assert(sizeof(PACKET_HEADER_TYPE) == HDR_BYTES, "kernel packet header size != host fabric header size");
    constexpr auto out_args = TensorAccessorArgs<16>();        // recv_x (dispatch) / comb (combine)
    constexpr auto outm_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();  // recv_meta

    size_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t outm_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t sem_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sem_noc_y = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t META_BLOCK = T * META_BYTES;
    const auto out = TensorAccessor(out_args, out_addr, ROW_BYTES);
    const auto outm = TensorAccessor(outm_args, outm_addr, META_BLOCK);

    if constexpr (ROLE == ROLE_SELF) {
        cb_wait_front(meta_cb, 1);
        const uint32_t meta_l1 = get_read_ptr(meta_cb);
        if constexpr (MODE == MODE_DISPATCH) {
            for (uint32_t t = 0; t < T; ++t) {
                const auto* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + t * META_BYTES);
                uint32_t far;
                if (dispatch_need<ROLE, MY, L, LINK, N, RING>(t, hit_mask<K, EPD>(ids), far)) {
                    cb_wait_front(data_cb, 1);
                    noc_async_write(get_read_ptr(data_cb), out.get_noc_addr(MY * T + t), ROW_BYTES);
                    noc_async_writes_flushed();
                    cb_pop_front(data_cb, 1);
                }
            }
            noc_async_write(meta_l1, outm.get_noc_addr(MY), META_BLOCK);
        } else {
            for (uint32_t t = 0; t < T; ++t) {
                const auto* ids =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + (MY * T + t) * META_BYTES);
                uint32_t hops;
                if (combine_need<ROLE, MY, L, LINK, N, RING>(MY, t, hit_mask<K, EPD>(ids), hops)) {
                    cb_wait_front(data_cb, 1);
                    noc_async_write(get_read_ptr(data_cb), out.get_noc_addr(MY * T + t), ROW_BYTES);
                    noc_async_writes_flushed();
                    cb_pop_front(data_cb, 1);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(meta_cb, 1);
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
        noc_semaphore_wait_min(sem, EXPECT);
        noc_semaphore_inc(get_noc_addr(sem_addr), (uint32_t)(0u - EXPECT));
        noc_async_atomic_barrier();
        return;
    } else {
        // line: hops to the end of the line; ring: N/2 both ways (antipode gets one flag per direction)
        constexpr uint32_t FULL = RING ? N / 2 : ((ROLE == ROLE_EAST) ? (N - 1 - MY) : MY);
        constexpr uint32_t META_RANGE = RING ? ((ROLE == ROLE_EAST) ? N / 2 : N / 2 - 1) : FULL;
        // (host only creates a sender core when FULL > 0; no static_assert: this branch is not a template)
        constexpr uint32_t FLUSH_EVERY = EP_FLUSH_EVERY;  // rows; must divide CB depth and be <= NHDR / packets-per-row
        constexpr uint32_t NHDR = 16;
        uint32_t pending = 0;

        auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
        sender.open_start();

        cb_reserve_back(hdr_cb, NHDR);  // 16 header slots
        const uint32_t hdr_base = get_write_ptr(hdr_cb);
        cb_push_back(hdr_cb, NHDR);
        uint32_t h = 0;
        auto next_hdr = [&]() {
            auto* p = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(hdr_base + h * sizeof(PACKET_HEADER_TYPE));
            h = (h + 1) % NHDR;
            return p;
        };

        cb_wait_front(meta_cb, 1);
        const uint32_t meta_l1 = get_read_ptr(meta_cb);
        sender.open_finish();

        // Send one row (all its packets). mcast: multicast to hops [1, dist]; else unicast dist hops.
        auto send_row = [&](uint32_t src, uint32_t dst_page, uint32_t dist, bool mcast) {
            for (uint32_t off = 0; off < ROW_BYTES; off += PAYLOAD) {
                const uint32_t sz = (ROW_BYTES - off) < PAYLOAD ? (ROW_BYTES - off) : PAYLOAD;
                auto* hdr = next_hdr();
                if (mcast) {
                    hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, (uint8_t)dist});
                } else {
                    hdr->to_chip_unicast((uint8_t)dist);
                }
                hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{out.get_noc_addr(dst_page, off)}, sz);
                sender.wait_for_empty_write_slot();
#ifdef EP_BLOCKING_SEND
                sender.send_payload_without_header_non_blocking_from_address(src + off, sz);
                noc_async_write_barrier();  // payload ACKed in EDM L1 before header + credit post (sim ordering test)
                sender.send_payload_flush_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
#else
                sender.send_payload_without_header_non_blocking_from_address(src + off, sz);
                sender.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
#endif
            }
        };

        if constexpr (MODE == MODE_DISPATCH) {
            for (uint32_t t = 0; t < T; ++t) {
                const auto* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + t * META_BYTES);
                uint32_t far;
                if (dispatch_need<ROLE, MY, L, LINK, N, RING>(t, hit_mask<K, EPD>(ids), far)) {
                    cb_wait_front(data_cb, 1);
                    send_row(get_read_ptr(data_cb), MY * T + t, far, true);
                    noc_async_writes_flushed();
                    cb_pop_front(data_cb, 1);
                }
            }
            if constexpr (LINK == 0) {
                // metadata block (expert ids + weights for all T tokens) to every chip in this direction
                auto* hdr = next_hdr();
                hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, (uint8_t)META_RANGE});
                hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{outm.get_noc_addr(MY)}, META_BLOCK);
                sender.wait_for_empty_write_slot();
                sender.send_payload_without_header_non_blocking_from_address(meta_l1, META_BLOCK);
#ifdef EP_BLOCKING_SEND
                noc_async_write_barrier();
#endif
                sender.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
            }
        } else {
            for (uint32_t s = 0; s < N; ++s) {
                for (uint32_t t = 0; t < T; ++t) {
                    const auto* ids =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(meta_l1 + (s * T + t) * META_BYTES);
                    uint32_t hops;
                    if (combine_need<ROLE, MY, L, LINK, N, RING>(s, t, hit_mask<K, EPD>(ids), hops)) {
                        cb_wait_front(data_cb, pending + 1);
                        send_row(get_read_ptr(data_cb) + pending * ROW_BYTES, MY * T + t, hops, false);
                        if (++pending == FLUSH_EVERY) {
                            noc_async_writes_flushed();
                            cb_pop_front(data_cb, pending);
                            pending = 0;
                        }
                    }
                }
            }
        }
        noc_async_writes_flushed();
        if (pending) {
            cb_pop_front(data_cb, pending);
        }

        // One completion flag per (source, link) per batch; flush=true makes each receiving router flush its
        // prior NOC writes before the increment, so the flag is ordered after this connection's data.
        auto* hdr = next_hdr();
        hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, (uint8_t)FULL});
        hdr->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{get_noc_addr(sem_noc_x, sem_noc_y, sem_addr), 1, true});
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
        sender.close();
        cb_pop_front(meta_cb, 1);
    }
}
