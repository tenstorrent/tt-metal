// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Lane fabric bench, sender (a group core's NCRISC, NOC1, as wl_reader.cpp): the lane's link pattern without its
// schedule - NPKT packets of PB bytes to the same core of the next chip in each active direction, through that
// direction's mux, round robin over the directions, never blocking on a full mux (the lane's QF send: space check,
// flush of the previous send, fused write + atomic inc of the receiver's direction counter). No credits, no slots, no
// cut-through: every packet is sent from this core's own source buffer.
// CT: 0 NPKT, 1 PB, 2 destination slots (2 directions x 2 x PB), 3 destination counters (16 B a direction), 4 flush
//     (router write ack before the inc), 5 direction mask, 6 source addr, 7 INC (packet n carries the atomic inc only
//     when (n + 1) % INC == 0, the others are plain writes), then the mux client CT args (buffers, buffer bytes,
//     status address, termination signal address, clients)
// RT: 0 mesh id, 1 next chip direction 0, 2 next chip direction 1, 3 this core's xy (x | y << 8), 4 stamps, then the
//     mux client RT args (17 words) of direction 0 and of direction 1
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
    constexpr uint32_t dmask = get_compile_time_arg_val(5), src = get_compile_time_arg_val(6);
    constexpr uint32_t INC = get_compile_time_arg_val(7);
    constexpr uint8_t nbuf = get_compile_time_arg_val(8);
    constexpr size_t bufsz = get_compile_time_arg_val(9);
    constexpr size_t status_addr = get_compile_time_arg_val(10);
    constexpr size_t term_addr = get_compile_time_arg_val(11);
    constexpr uint32_t nclients = get_compile_time_arg_val(12);

    const uint16_t mesh = get_arg_val<uint32_t>(0);
    const uint8_t nchip[2] = {(uint8_t)get_arg_val<uint32_t>(1), (uint8_t)get_arg_val<uint32_t>(2)};
    const uint32_t me = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    const uint8_t mex = me & 0xFF, mey = me >> 8;

    tt::tt_fabric::WorkerToFabricMuxSender<nbuf> mux[2];
    bool master[2];
    uint32_t term_sync[2], tmx[2], tmy[2];
    uint8_t muxx[2], muxy[2];
    for (uint32_t d = 0; d < 2; ++d) {
        size_t a = 5 + 17 * d;
        a++;  // connection valid
        master[d] = get_arg_val<uint32_t>(a++) != 0;
        muxx[d] = get_arg_val<uint32_t>(a++);
        muxy[d] = get_arg_val<uint32_t>(a++);
        const size_t ch_base = get_arg_val<uint32_t>(a++), conn_info = get_arg_val<uint32_t>(a++);
        const size_t handshake = get_arg_val<uint32_t>(a++), flow = get_arg_val<uint32_t>(a++);
        const size_t bufidx = get_arg_val<uint32_t>(a++);
        const uint8_t chan = get_arg_val<uint32_t>(a++);
        term_sync[d] = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint32_t l_status = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint32_t l_flow = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint32_t l_teardown = get_semaphore(get_arg_val<uint32_t>(a++));
        const uint32_t l_bufidx = get_semaphore(get_arg_val<uint32_t>(a++));
        tmx[d] = get_arg_val<uint32_t>(a++);
        tmy[d] = get_arg_val<uint32_t>(a++);
        mux[d] = tt::tt_fabric::build_connection_to_fabric_endpoint<nbuf>(
            muxx[d],
            muxy[d],
            chan,
            nbuf,
            bufsz,
            ch_base,
            conn_info,
            handshake,
            flow,
            bufidx,
            l_flow,
            l_teardown,
            l_bufidx);
        tt::tt_fabric::wait_for_fabric_endpoint_ready(muxx[d], muxy[d], status_addr, l_status);
        tt::tt_fabric::fabric_client_connect(mux[d]);
    }
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[2] = {
        PacketHeaderPool::allocate_header(), PacketHeaderPool::allocate_header()};

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t n[2] = {(dmask & 1) ? 0u : NPKT, (dmask & 2) ? 0u : NPKT};
    uint32_t full = 0, d = 0;
    const uint32_t t0 = wall();
    while (n[0] < NPKT || n[1] < NPKT) {
        d ^= 1;
        if (n[d] >= NPKT) {
            continue;
        }
        if (!mux[d].edm_has_space_for_packet()) {
            ++full;
            continue;
        }
        noc_async_writes_flushed();  // (the previous send's header and payload out of L1)
        const uint64_t dst = get_noc_addr(mex, mey, dslot + (2 * d + (n[d] & 1)) * PB);
        if ((n[d] + 1) % INC == 0) {
            fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &mux[d],
                hdr[d],
                nchip[d],
                mesh,
                src,
                PB,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    dst, get_noc_addr(mex, mey, dcnt + d * 16), 1, FLUSH});
        } else {
            fabric_unicast_noc_unicast_write(
                &mux[d], hdr[d], nchip[d], mesh, src, PB, tt::tt_fabric::NocUnicastCommandHeader{dst});
        }
        ++n[d];
    }
    const uint32_t t1 = wall();
    noc_async_write_barrier();
    for (uint32_t dd = 0; dd < 2; ++dd) {
        tt::tt_fabric::fabric_client_disconnect(mux[dd]);
        if (master[dd]) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(term_sync[dd]), nclients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(muxx[dd], muxy[dd], term_addr);
        } else {
            noc_semaphore_inc(get_noc_addr(tmx[dd], tmy[dd], term_sync[dd]), 1);
            noc_async_atomic_barrier();
        }
    }
    st[0] = t0;
    st[1] = t1;
    st[8] = full;
}
