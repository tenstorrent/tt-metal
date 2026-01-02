// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <chrono>
#include <array>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"

using namespace tt;
using namespace tt::tt_fabric;

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr auto CTA_BASE = dst_args.next_compile_time_args_offset();
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;
    constexpr uint32_t PERF_CB_ID = tt::CBIndex::c_1;

    size_t idx = 0;
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t device_perf_base_addr = get_arg_val<uint32_t>(idx++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // Allocate headers as group pages
    volatile tt_l1_ptr PACKET_HEADER_TYPE* pkt_hdr = PacketHeaderPool::allocate_header();
    DPRINT << "Size of PACKET_HEADER_TYPE is " << sizeof(PACKET_HEADER_TYPE) << " bytes\n";

    (void)fabric_set_unicast_route(pkt_hdr, /*dst_dev_id=*/dst_dev_id, /*dst_mesh_id=*/dst_mesh_id);

    sender.open<true>();

    const auto dst_acc = TensorAccessor(dst_args, /*bank_base=*/dst_base_addr, /*page_size=*/PAGE_SIZE);

    uint32_t bytes_to_send = 4;
    // send bytes until it reaches PAGE_SIZE
    cb_wait_front(CB_ID, 1);
    const uint32_t src_l1_addr = get_read_ptr(CB_ID);
    uint64_t dest_noc_addr = dst_acc.get_noc_addr(/*page_id=*/0, /*offset=*/0, /*noc=*/0);
    constexpr uint32_t SAMPLE_COUNT = 10;
    std::array<uint32_t, SAMPLE_COUNT> dur_sec_samples;

    cb_reserve_back(PERF_CB_ID, 1);
    auto perf_l1_write_addr = get_write_ptr(PERF_CB_ID);
    auto device_perf_buf_ptr = reinterpret_cast<uint32_t*>(perf_l1_write_addr);

    uint32_t slot = 0;
    while (slot < 1024) {
        // Sample 10 times of blocking write in a row.
        sender.wait_for_empty_write_slot();
        pkt_hdr->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, bytes_to_send);

        uint64_t st = get_timestamp();
        sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, bytes_to_send);
        sender.send_payload_flush_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
        uint64_t et = get_timestamp();
        device_perf_buf_ptr[slot] = static_cast<uint32_t>(et - st);

        bytes_to_send += 4;
        slot++;
    }
    cb_push_back(PERF_CB_ID, 1);

    cb_pop_front(CB_ID, 1);

    // Final signal: bump receiver semaphore so the receiver kernel exits.
    // In this benchmark we always have a completion semaphore.
    ASSERT(sem_l1_addr != 0);

    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    (void)fabric_set_unicast_route(pkt_hdr, /*dst_dev_id=*/dst_dev_id, /*dst_mesh_id=*/dst_mesh_id);
    pkt_hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1));

    sender.wait_for_empty_write_slot();
    sender.send_payload_flush_non_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));

    sender.close();

    // Write all performance data back to device_perf_buf
    // DPRINT << "PAGE_SIZE is " << PAGE_SIZE << " bytes\n";
    cb_wait_front(PERF_CB_ID, 1);
    // for (int32_t r = 0; r < 32; ++r) {
    //     SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
    //     DPRINT_DATA1({ DPRINT << (uint)r << " --READ--cin1-- " << TileSlice(PERF_CB_ID, 0, sr, TSLICE_OUTPUT_CB,
    //     TSLICE_WR_PTR, true, false) << ENDL(); });
    // }
    auto perf_l1_read_addr = get_read_ptr(PERF_CB_ID);
    auto perf_dram_noc_addr = get_dram_noc_addr(0, PAGE_SIZE, device_perf_base_addr);
    noc_async_write(perf_l1_read_addr, perf_dram_noc_addr, PAGE_SIZE);
    noc_async_write_barrier();
    cb_pop_front(PERF_CB_ID, 1);
}
