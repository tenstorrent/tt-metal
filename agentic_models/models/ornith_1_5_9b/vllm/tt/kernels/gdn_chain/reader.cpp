// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Chained multi-token verify (one request, T candidates): core hv walks heads hv, hv+8, ... = candidates 0..T-1
// of value head hv; derived from flat_ordered_conv_gates/reader.cpp (only the state source differs).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
void process_head(uint32_t head, uint32_t live_head, uint32_t wait) {
    Noc noc;
    constexpr auto ca = TensorAccessorArgs<0>();
    constexpr auto ga = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto za = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<za.next_compile_time_args_offset()>();
    const uint32_t batch = head / 8, hv = head % 8, hk = hv / 2;
    const auto conv = TensorAccessor(ca, get_arg_val<uint32_t>(1), 2048);
    const auto g = TensorAccessor(ga, get_arg_val<uint32_t>(2), 4096);
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(3), 2048);
    // Chained verify: candidate `batch` of one request starts from the live slot state (batch 0) or from the state
    // this core's writer stored after candidate batch-1 (chain row batch-1); wait until that write has landed.
    // live_head: the request's live pool head (first candidate) or ~0 (later candidate: chain row head - 8, which
    // this core's writer stored as its `wait`-th head; wait until it has landed).
    const bool first_token = live_head != 0xFFFFFFFFu;
    if (!first_token) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), wait);
    }
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(first_token ? 4 : 9), 4096);
    const uint32_t state_head = first_token ? live_head : head - 8;
    CircularBuffer cs(5);
    cs.reserve_back(16);
    const uint32_t slab = get_write_ptr(5);
    for (uint32_t t = 0; t < 16; ++t) {
        noc_async_read(state.get_noc_addr(state_head * 16 + t), slab + t * 4096, 4096);
    }
    noc_async_read_barrier();
    cs.push_back(16);
    // FP32 copy of the same state for the exact SFPU decay (CB33 is unpacked straight to DST; CB5 feeds the FPU).
    // Local L1 -> L1 copy of the slab just read (no second DRAM read); compute only reads CB5 until it pops it.
    CircularBuffer cs32(33);
    cs32.reserve_back(16);
    noc_async_read(get_noc_addr(slab), get_write_ptr(33), 16 * 4096);
    noc_async_read_barrier();
    cs32.push_back(16);
    CircularBuffer scratch(28);
    scratch.reserve_back(1);
    const uint32_t stage = get_write_ptr(28);
    const uint32_t row_offset = (batch / 16) * 1024 + (batch % 16) * 32;
    for (uint32_t which = 0; which < 3; ++which) {
        const uint32_t cb = 20 + which;
        CircularBuffer c(cb);
        c.reserve_back(4);
        const uint32_t base = get_write_ptr(cb);
        volatile tt_l1_ptr uint32_t* zero = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
        noc.async_write_zeros(c, 8192);
        noc.write_zeros_l1_barrier();
        const uint32_t col = which == 0 ? hk * 4 : (which == 1 ? 16 + hk * 4 : 32 + hv * 4);
        for (uint32_t t = 0; t < 4; ++t) {
            const uint64_t addr = conv.get_noc_addr(col + t);
            if constexpr (ca.is_dram) {
                // DRAM transfers require matching64-byte address alignment.
                // Read aligned neighboring row pairs, then extract our row.
                const uint32_t aligned = row_offset & ~63u, phase = row_offset & 63u;
                noc_async_read(addr + aligned, stage, 64);
                noc_async_read(addr + aligned + 512, stage + 64, 64);
                noc_async_read_barrier();
                for (uint32_t word = 0; word < 8; ++word) {
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * 2048)[word] =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + phase)[word];
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * 2048 + 512)[word] =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64 + phase)[word];
                }
            } else {
                noc_async_read(addr + row_offset, base + t * 2048, 32);
                noc_async_read(addr + row_offset + 512, base + t * 2048 + 512, 32);
            }
        }
        noc_async_read_barrier();
        c.push_back(4);
    }
    // Scalar head gates: only [0,0] is active; all other rows/columns stay zero.
    CircularBuffer cg(23), cb(24);
    cg.reserve_back(1);
    cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* gp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(23));
    volatile tt_l1_ptr uint32_t* bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(24));
    noc.async_write_zeros(cg, 4096);
    noc.async_write_zeros(cb, 2048);
    noc.write_zeros_l1_barrier();
    noc_async_read(g.get_noc_addr(0), stage, 4096);
    noc_async_read_barrier();
    // g in column 0 of EVERY row: the scan decays the FP32 state by an SFPU column broadcast of exp(g), which must
    // be the same scalar for all 32 state rows. Every other consumer multiplies rows that are zero beyond row 0.
    const uint32_t gbits =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[(batch / 16) * 512 + (batch % 16) * 16 + hv];
    for (uint32_t r = 0; r < 32; ++r) {
        gp[(r / 16) * 512 + (r % 16) * 16] = gbits;
    }
    noc_async_read(beta.get_noc_addr(0), stage, 2048);
    noc_async_read_barrier();
    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bp)[0] =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage)[(batch / 16) * 512 + (batch % 16) * 16 + hv];
    asm volatile("fence");
    cg.push_back(1);
    cb.push_back(1);
    CircularBuffer unit(27);
    unit.reserve_back(1);
    volatile tt_l1_ptr uint32_t* u = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(27));
    noc.async_write_zeros(unit, 2048);
    noc.write_zeros_l1_barrier();
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t j = 0; j < 8; ++j) {
            u[f * 128 + j] = 0x3F803F80u;
        }
    }
    asm volatile("fence");
    unit.push_back(1);
    // Setup-only identity tile for the one-token WY inverse and scan operands.
    CircularBuffer eye(14);
    eye.reserve_back(1);
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(14));
    noc.async_write_zeros(eye, 4096);
    noc.write_zeros_l1_barrier();
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t tile_index = (r / 16) * 768 + (r % 16) * 17;
        data[tile_index] = 0x3f800000u;
    }
    asm volatile("fence");
    eye.push_back(1);

    // Keep NCRISC as the sole producer of CB20/22. These second reservations
    // wait for preparation to consume Q/V; unit and identity are already ready.
    const auto gate = TensorAccessor(za, get_arg_val<uint32_t>(5), 2048);
    const auto weight = TensorAccessor(wa, get_arg_val<uint32_t>(6), 2048);
    CircularBuffer gamma(20), z(22);
    gamma.reserve_back(4);
    z.reserve_back(4);
    const uint32_t zbase = get_write_ptr(22);
    volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zbase);
    noc.async_write_zeros(z, 8192);
    noc.write_zeros_l1_barrier();
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(weight.get_noc_addr(t), get_write_ptr(20) + t * 2048, 2048);
        const auto addr = gate.get_noc_addr(64 + hv * 4 + t);
        if constexpr (za.is_dram) {
            const uint32_t aligned = row_offset & ~63u, phase = row_offset & 63u;
            noc_async_read(addr + aligned, stage, 64);
            noc_async_read(addr + aligned + 512, stage + 64, 64);
            noc_async_read_barrier();
            for (uint32_t w = 0; w < 8; ++w) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zbase + t * 2048)[w] =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + phase)[w];
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zbase + t * 2048 + 512)[w] =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + 64 + phase)[w];
            }
        } else {
            noc_async_read(addr + row_offset, zbase + t * 2048, 32);
            noc_async_read(addr + row_offset + 512, zbase + t * 2048 + 512, 32);
        }
    }
    noc_async_read_barrier();
    gamma.push_back(4);
    z.push_back(4);
}

void kernel_main() {
    // items = 8 x users (user u, value head hv); each item walks its T candidates (verify rows u*T .. u*T+T-1).
    const uint32_t first = get_arg_val<uint32_t>(0), stride = get_arg_val<uint32_t>(7),
                   items = get_arg_val<uint32_t>(8), T = get_arg_val<uint32_t>(10);
    uint32_t done = 0;
    for (uint32_t item = first; item < items; item += stride) {
        const uint32_t u = item / 8, hv = item % 8;
        for (uint32_t t = 0; t < T; ++t, ++done) {
            process_head((u * T + t) * 8 + hv, t == 0 ? u * 8 + hv : 0xFFFFFFFFu, done);
        }
    }
}
