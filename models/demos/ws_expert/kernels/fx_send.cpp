// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric link microbench, sender: one worker core with its own fabric connection (no mux) sends NPKT packets of PB
// bytes from one L1 source buffer into a fixed L1 slot of the same core on the destination chip. Headers are routed
// once. The first and the last packet are fused write + atomic inc of the receiver's counter (the receiver times
// between them); with MODE bit 2 every packet is. MODE bit 0: the connection's stateful send path (command buffers
// programmed once); bit 1: posted NoC writes into the router's slots.
// CT: 0 NPKT, 1 PB, 2 source (L1), 3 destination slot (L1), 4 destination counter (L1), 5 MODE
// RT: 0 mesh id, 1 destination chip id, 2 destination core x | y << 8, 3 stamps (L1), then the fabric connection args.
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t NPKT = get_compile_time_arg_val(0);
    constexpr uint32_t PB = get_compile_time_arg_val(1);
    constexpr uint32_t src = get_compile_time_arg_val(2);
    constexpr uint32_t dslot = get_compile_time_arg_val(3);
    constexpr uint32_t dcnt = get_compile_time_arg_val(4);
    constexpr uint32_t MODE = get_compile_time_arg_val(5);
    constexpr bool STATEFUL = MODE & 1, POSTED = MODE & 2, INC_ALL = MODE & 4;

    const uint16_t mesh = get_arg_val<uint32_t>(0);
    const uint8_t dchip = get_arg_val<uint32_t>(1);
    const uint32_t dxy = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    size_t a = 4;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    conn.open();
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hw = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hi = PacketHeaderPool::allocate_header();
    const uint64_t slot = get_noc_addr(dxy & 0xFF, dxy >> 8, dslot);
    const uint64_t cnt = get_noc_addr(dxy & 0xFF, dxy >> 8, dcnt);
    (void)fabric_set_unicast_route(hw, dchip, mesh);
    (void)fabric_set_unicast_route(hi, dchip, mesh);
    hw->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{slot}, PB);
    hi->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{slot, cnt, 1, false}, PB);
    if constexpr (STATEFUL) {
        conn.setup_stateful_send_cmd_bufs<POSTED>();
    }
    const uint32_t t0 = wall();
    for (uint32_t i = 0; i < NPKT; ++i) {
        const bool inc = INC_ALL || i == 0 || i == NPKT - 1;
        const uint32_t h = reinterpret_cast<uint32_t>(inc ? hi : hw);
        conn.wait_for_empty_write_slot();
        if constexpr (STATEFUL) {
            conn.send_current_slot_stateful_non_blocking<POSTED>(src, PB, h);
        } else {
            conn.send_current_slot_non_blocking<POSTED>(src, PB, h);
        }
    }
    const uint32_t t1 = wall();
    noc_async_full_barrier();
    conn.close();
    st[0] = t0;
    st[1] = t1;
}
