// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric mux throughput bench, client (one of up to 20 cores sharing one mux): sends NPKT packets of PB bytes to the
// same core of the next chip through the mux, each a fused write + atomic inc of the receiver's counter, alternating
// between two destination slots; then disconnects, and the termination master stops the mux once every client is done.
// CT: 0 NPKT, 1 PB, 2 destination slots addr (2 x PB), 3 destination counter addr, 4 flush (write ack before the inc),
//     then the mux client CT args (ccl::fabric_mux_connection_ct_args): number of buffers, buffer bytes, mux status
//     address, mux termination signal address, number of clients
// RT: 0 mesh id, 1 destination chip id, 2 destination core xy (x | y << 8), 3 source addr, 4 stamps addr, then the mux
//     client RT args (ccl::fabric_mux_connection_rt_args, 17 words)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;

void kernel_main() {
    constexpr uint32_t NPKT = get_compile_time_arg_val(0), PB = get_compile_time_arg_val(1);
    constexpr uint32_t dslot = get_compile_time_arg_val(2), dcnt = get_compile_time_arg_val(3);
    constexpr bool FLUSH = get_compile_time_arg_val(4) != 0;
    constexpr uint8_t nbuf = get_compile_time_arg_val(5);
    constexpr size_t bufsz = get_compile_time_arg_val(6);
    constexpr size_t status_addr = get_compile_time_arg_val(7);
    constexpr size_t term_addr = get_compile_time_arg_val(8);
    constexpr uint32_t nclients = get_compile_time_arg_val(9);

    size_t a = 0;
    const uint16_t mesh = get_arg_val<uint32_t>(a++);
    const uint8_t dchip = get_arg_val<uint32_t>(a++);
    const uint32_t dxy = get_arg_val<uint32_t>(a++);
    const uint32_t src = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    a++;  // connection valid
    const bool master = get_arg_val<uint32_t>(a++) != 0;
    const uint8_t mx = get_arg_val<uint32_t>(a++), my = get_arg_val<uint32_t>(a++);
    const size_t ch_base = get_arg_val<uint32_t>(a++), conn_info = get_arg_val<uint32_t>(a++);
    const size_t handshake = get_arg_val<uint32_t>(a++), flow = get_arg_val<uint32_t>(a++);
    const size_t bufidx = get_arg_val<uint32_t>(a++);
    const uint8_t chan = get_arg_val<uint32_t>(a++);
    const uint32_t term_sync = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t l_status = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t l_flow = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t l_teardown = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t l_bufidx = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t tmx = get_arg_val<uint32_t>(a++), tmy = get_arg_val<uint32_t>(a++);

    auto conn = tt::tt_fabric::build_connection_to_fabric_endpoint<nbuf>(
        mx, my, chan, nbuf, bufsz, ch_base, conn_info, handshake, flow, bufidx, l_flow, l_teardown, l_bufidx);
    tt::tt_fabric::wait_for_fabric_endpoint_ready(mx, my, status_addr, l_status);
    tt::tt_fabric::fabric_client_connect(conn);
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    const uint64_t cnt = get_noc_addr(dxy & 0xFF, dxy >> 8, dcnt);
    const uint32_t t0 = wall();
    for (uint32_t i = 0; i < NPKT; ++i) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &conn,
            hdr,
            dchip,
            mesh,
            src,
            PB,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                get_noc_addr(dxy & 0xFF, dxy >> 8, dslot + (i & 1) * PB), cnt, 1, FLUSH});
        noc_async_writes_flushed();  // (payload and header out of L1 before the header is reused)
    }
    const uint32_t t1 = wall();
    noc_async_write_barrier();
    tt::tt_fabric::fabric_client_disconnect(conn);
    if (master) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(term_sync), nclients - 1);
        tt::tt_fabric::fabric_endpoint_terminate(mx, my, term_addr);
    } else {
        noc_semaphore_inc(get_noc_addr(tmx, tmy, term_sync), 1);
        noc_async_atomic_barrier();
    }
    st[0] = t0;
    st[1] = t1;
}
