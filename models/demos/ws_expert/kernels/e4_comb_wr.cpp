// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EP-4 pre-sum (ep4_layer.build_combine), adder writer (NCRISC, NOC1: L1 transfers and posted DRAM writes). Per run of
// the core's range with count > 0, in order (group token g = home TX + t), its packed partial (the first row bytes of
// the 4 tiles in CB 16):
//   - home == CS (this chip's column): to P page t S5 + CS of this chip's P (posted);
//   - else: to this adder's slot of the port for (direction of home, link g & 1): the row plus a descriptor word
//     (t | home << 16) after it, in one write, then the port's flag word for this adder (rows written, same command
//     buffer: it lands after the row); a slot is reused once the port's credit (rows it sent) allows (NSL slots).
// Start: this core's credit words zeroed, then the ports' go (semaphore 0 reaches NPORT: each port zeroed its words).
// End: each port's end word for this adder = rows sent to it + 1; once the compute's last page (CB 17) is in, TAB_FLAG
// = 0 for the next launch; the posted writes' fence (an acknowledged 4-B write into every DRAM bank).
// CT: 0 TX, 1 row bytes, 2 tab (L1), 3 tab flag, 4 NPORT, 5 NSL, 6 SLOTB, 7 port ring (L1), 8 port flags (L1, 16 B an
//     adder), 9 port ends (L1, 16 B an adder), 10 credits (L1 here, 16 B a port), 11 S5, 12 scratch (L1, 16 B a port
//     + 16), then TensorAccessorArgs(P), (fence).
// RT: 0 me, 1 a, 2 nt, 3 CS, 4 P addr, 5 fence addr, 6 stamps (7 credit wait cycles, 8 row wait cycles), 7 + 2 h + l
//     the port index for home h on link l (0xFF: none), 15 + p port p's NoC x | y << 8.
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t TX = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tab = get_compile_time_arg_val(2);
    constexpr uint32_t tab_flag = get_compile_time_arg_val(3);
    constexpr uint32_t NPORT = get_compile_time_arg_val(4);
    constexpr uint32_t NSL = get_compile_time_arg_val(5);
    constexpr uint32_t SLOTB = get_compile_time_arg_val(6);
    constexpr uint32_t pring = get_compile_time_arg_val(7);
    constexpr uint32_t pflags = get_compile_time_arg_val(8);
    constexpr uint32_t pends = get_compile_time_arg_val(9);
    constexpr uint32_t cred = get_compile_time_arg_val(10);
    constexpr uint32_t S5 = get_compile_time_arg_val(11);
    constexpr uint32_t scr = get_compile_time_arg_val(12);
    constexpr auto pargs = TensorAccessorArgs<13>();
    constexpr auto fargs = TensorAccessorArgs<pargs.next_compile_time_args_offset()>();
    constexpr uint32_t out_cb = tt::CBIndex::c_16, done_cb = tt::CBIndex::c_17, TPR = 4;
    static_assert(NPORT <= 8 && SLOTB >= row_bytes + 16);

    const uint32_t me = get_arg_val<uint32_t>(0), a = get_arg_val<uint32_t>(1), nt = get_arg_val<uint32_t>(2);
    const uint32_t CS = get_arg_val<uint32_t>(3);
    const auto ps = TensorAccessor(pargs, get_arg_val<uint32_t>(4), row_bytes);
    const auto fs = TensorAccessor(fargs, get_arg_val<uint32_t>(5), 64);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
    auto w = [](uint32_t addr) -> volatile tt_l1_ptr uint32_t& {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    };
    auto port_at = [&](uint32_t pi, uint32_t addr) {
        const uint32_t v = get_arg_val<uint32_t>(15 + pi);
        return get_noc_addr(v & 0xFF, v >> 8, addr);
    };
    for (uint32_t pi = 0; pi < NPORT; ++pi) {
        w(cred + pi * 16) = 0;
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), NPORT);
    while (w(tab_flag) == 0) {
        invalidate_l1_cache();
    }
    uint32_t seq[8] = {};
    uint32_t t_cred = 0, t_row = 0;
    for (uint32_t i = 0; i < nt; ++i) {
        if (w(tab + i * 128 + 4) == 0) {
            continue;
        }
        const uint32_t g = a + i, home = g / TX, t = g % TX;
        uint32_t s0 = wall();
        cb_wait_front(out_cb, TPR);
        t_row += wall() - s0;
        const uint32_t src = get_read_ptr(out_cb);
        if (home == CS) {
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(src, ps.get_noc_addr(t * S5 + CS), row_bytes);
            noc_async_posted_writes_flushed();
        } else {
            const uint32_t pi = get_arg_val<uint32_t>(7 + 2 * home + (g & 1));
            const uint32_t n = seq[pi];
            s0 = wall();
            while (n >= NSL && w(cred + pi * 16) + NSL <= n) {
                invalidate_l1_cache();
            }
            t_cred += wall() - s0;
            w(src + row_bytes) = t | home << 16;
            noc_async_write(src, port_at(pi, pring + (me * NSL + n % NSL) * SLOTB), row_bytes + 16);
            w(scr + pi * 16) = n + 1;
            noc_async_write(scr + pi * 16, port_at(pi, pflags + me * 16), 4);
            noc_async_writes_flushed();
            seq[pi] = n + 1;
        }
        cb_pop_front(out_cb, TPR);
    }
    for (uint32_t pi = 0; pi < NPORT; ++pi) {
        w(scr + pi * 16) = seq[pi] + 1;
        noc_async_write(scr + pi * 16, port_at(pi, pends + me * 16), 4);
    }
    noc_async_writes_flushed();
    cb_wait_front(done_cb, 1);
    cb_pop_front(done_cb, 1);
    w(tab_flag) = 0;
    noc_async_posted_writes_flushed();
    for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
        noc_async_write(scr + NPORT * 16, fs.get_noc_addr(b), 4);
    }
    noc_async_write_barrier();
    stamps[7] = t_cred;
    stamps[8] = t_row;
}
