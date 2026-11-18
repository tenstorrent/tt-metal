// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EP-4 combine (ep4_layer.build_combine), port (one fabric connection: a direction of the mesh row and a link). The
// adders write partial rows into their slots of this core's ring (NSL a adder) and raise their flag word (rows
// written); the port sends each row, in each adder's order, round robin over the adders, as a fabric write into P page
// t S5 + CS of the home chip (the descriptor after the row: t | home << 16), up to PIPE packets in flight (a header
// each); once they have left L1 (flushed together), the adders' credits (rows sent, into their credit word for this
// port). Done once every adder's end word (rows + 1) is in and every row sent; then, per home it sent n rows to, one
// atomic increment by n of the home's counter for this chip (the gate core's word CS: the home pass waits for it),
// flushed: the last router waits for every write it delivered to be acknowledged first, and every row of this
// connection to that home took the same path ahead of it, so the counter counts landed rows. Start: flags and ends
// zeroed, then + 1 on every adder's semaphore 0 (the adders wait for every port's). CT: 0 NA, 1 NSL, 2 SLOTB, 3 ring
// (L1), 4 flags (L1), 5 ends (L1), 6 row bytes, 7 S5, 8 counters (L1 on the gate
//     core), 9 adder credits (L1), then TensorAccessorArgs(P).
// RT: 0 P addr, 1 this port's index, 2 CS, 3 gate core NoC x | y << 8, 4 dst mesh id, 5 + h home h's chip id (0xFF
//     none), 9 stamps (0 start, 63 end, 10 sent rows, 11 cycles waiting for router space, 12 loop cycles), 10 + k adder
//     k's NoC x | y
//     << 8, then the fabric connection args.
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t NA = get_compile_time_arg_val(0);
    constexpr uint32_t NSL = get_compile_time_arg_val(1);
    constexpr uint32_t SLOTB = get_compile_time_arg_val(2);
    constexpr uint32_t ring = get_compile_time_arg_val(3);
    constexpr uint32_t flags = get_compile_time_arg_val(4);
    constexpr uint32_t ends = get_compile_time_arg_val(5);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t S5 = get_compile_time_arg_val(7);
    constexpr uint32_t cnt = get_compile_time_arg_val(8);
    constexpr uint32_t acred = get_compile_time_arg_val(9);
    constexpr auto pargs = TensorAccessorArgs<10>();

    const auto ps = TensorAccessor(pargs, get_arg_val<uint32_t>(0), row_bytes);
    const uint32_t me_port = get_arg_val<uint32_t>(1), CS = get_arg_val<uint32_t>(2);
    const uint32_t gxy = get_arg_val<uint32_t>(3);
    const uint16_t mesh = get_arg_val<uint32_t>(4);
    uint8_t chip[4];
    for (uint32_t h = 0; h < 4; ++h) {
        chip[h] = get_arg_val<uint32_t>(5 + h);
    }
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(9));
    stamps[0] = wall();
    auto adder_at = [&](uint32_t k, uint32_t addr) {
        const uint32_t v = get_arg_val<uint32_t>(10 + k);
        return get_noc_addr(v & 0xFF, v >> 8, addr);
    };
    size_t arg_idx = 10 + NA;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    conn.open();
    PacketHeaderPool::reset();
    // (headers routed once: one a (home, packet in flight); the pool holds 12 a RISC)
    constexpr uint32_t PIPE = 3;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hh[4][PIPE] = {};
    for (uint32_t h = 0; h < 4; ++h) {
        if (chip[h] == 0xFF) {
            continue;
        }
        for (uint32_t i = 0; i < PIPE; ++i) {
            hh[h][i] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hh[h][i], chip[h], mesh);
        }
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* chdr = PacketHeaderPool::allocate_header();
    uint32_t pend[PIPE], npend = 0;  // (adders whose sent row may still be in L1)
    auto w = [](uint32_t addr) -> volatile tt_l1_ptr uint32_t& {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    };
    for (uint32_t k = 0; k < NA; ++k) {
        w(flags + k * 16) = 0;
        w(ends + k * 16) = 0;
    }
    const uint32_t go = get_semaphore(0);
    for (uint32_t k = 0; k < NA; ++k) {
        noc_semaphore_inc(adder_at(k, go), 1);
    }
    noc_async_atomic_barrier();
    const uint64_t cnt_noc = get_noc_addr(gxy & 0xFF, gxy >> 8, cnt + CS * 16);
    uint32_t sent[NA];
    for (uint32_t k = 0; k < NA; ++k) {
        sent[k] = 0;
    }
    uint32_t nsent = 0, t_space = 0, nh[4] = {0, 0, 0, 0};
    const uint32_t t0 = wall();
    auto retire = [&]() {  // (the in-flight rows have left L1: credit their adders)
        if (npend) {
            noc_async_writes_flushed();
            for (uint32_t i = 0; i < npend; ++i) {
                noc_inline_dw_write(adder_at(pend[i], acred + me_port * 16), sent[pend[i]]);
            }
            npend = 0;
        }
    };
    uint32_t done = 0;
    while (done < NA) {
        done = 0;
        invalidate_l1_cache();
        bool any = false;
        for (uint32_t k = 0; k < NA; ++k) {
            const uint32_t f = w(flags + k * 16), e = w(ends + k * 16);
            if (e != 0 && sent[k] + 1 == e) {
                ++done;
                continue;
            }
            if (f <= sent[k] || (npend && pend[npend - 1] == k)) {
                continue;  // (one row an adder in flight: its slot is reused only after its credit)
            }
            const uint32_t slot = ring + (k * NSL + sent[k] % NSL) * SLOTB;
            const uint32_t d = w(slot + row_bytes), t = d & 0xFFFF, home = d >> 16;
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hh[home][npend];
            hdr->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{
                    tt::tt_fabric::addrgen_detail::get_noc_address(ps, t * S5 + CS, 0)},
                row_bytes);
            const uint32_t s0 = wall();
            conn.wait_for_empty_write_slot();
            t_space += wall() - s0;
            conn.send_current_slot_non_blocking(slot, row_bytes, reinterpret_cast<uint32_t>(hdr));
            ++nh[home];
            ++sent[k];
            ++nsent;
            pend[npend++] = k;
            any = true;
            if (npend == PIPE) {
                retire();
            }
        }
        if (!any) {
            retire();
        }
    }
    retire();
    for (uint32_t h = 0; h < 4; ++h) {
        if (nh[h] == 0) {
            continue;
        }
        while (!conn.edm_has_space_for_packet()) {
        }
        fabric_unicast_noc_unicast_atomic_inc(
            &conn, chdr, chip[h], mesh, tt::tt_fabric::NocUnicastAtomicIncCommandHeader(cnt_noc, nh[h], true));
        noc_async_writes_flushed();
    }
    conn.close();
    noc_async_full_barrier();
    stamps[10] = nsent;
    stamps[11] = t_space;
    stamps[12] = wall() - t0;
    stamps[63] = wall();
}
