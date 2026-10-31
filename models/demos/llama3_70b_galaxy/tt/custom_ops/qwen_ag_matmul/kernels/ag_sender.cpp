// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qwen_ag_matmul fabric sender: one direction of one link of the ring all-gather along M (dim 2) over the mesh
// rows. This device's slot (x_local, slot_rows tile rows x K_tiles tiles) goes to the same pages of every other
// device's gathered buffer by fabric multicast (the routers write and forward), 4 tiles per packet (scatter write).
//
// Link l carries K-tile columns [col0, col0 + link_cols) of every row: half A (the first link_cols / 2 columns)
// goes range_a hops, half B range_b hops (fwd 4 / 3, bwd 3 / 4 on a ring of 8, so every peer gets each half from
// exactly one direction). The rows go in waves (RT list); after each wave-half the sender multicasts an atomic inc
// of that wave's arrival word on the hub core (same range as the data, flushed behind it), so a peer's word reaches
// 2 * links = 8 once the whole wave of this slot has landed in its gathered buffer.
//
// Before sending, the sender waits for its go semaphore (the hub sets it once every peer has launched this call),
// so no peer writes into a buffer a slower device may still be reading. With copy_local, the fwd sender also writes
// its half A pages, the bwd sender its half B pages, to this device's own slot of the gathered buffer.
//
// CT: 0 page_size, 1 range_a, 2 range_b, 3 is_fwd, 4 send_barrier, 5 barrier_range, 6 batch, 7 hdr_stride,
//     8 K_tiles, 9 col0, 10 link_cols, 11 copy_local, 12 num_waves, 13 go_sem_id, 14 cb_stage, 15 cb_hdr,
//     then accessors: x_local (in), gathered (out)
// RT: 0 in_addr, 1 out_addr, 2 slot_row0, 3 hub_x, 4 hub_y, 5 barrier_sem, then num_waves x (row0, rows, arrival
//     word address), then the fabric connection args
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

constexpr uint32_t TPP = 4;  // tiles per packet

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t range_a = get_compile_time_arg_val(1);
    constexpr uint32_t range_b = get_compile_time_arg_val(2);
    constexpr bool is_fwd = get_compile_time_arg_val(3) != 0;
    constexpr bool send_barrier = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t barrier_range = get_compile_time_arg_val(5);
    constexpr uint32_t BATCH = get_compile_time_arg_val(6);  // packets per batch; one header each
    constexpr uint32_t hdr_stride = get_compile_time_arg_val(7);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t col0 = get_compile_time_arg_val(9);
    constexpr uint32_t link_cols = get_compile_time_arg_val(10);
    constexpr bool copy_local = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t num_waves = get_compile_time_arg_val(12);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(13);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(14);
    constexpr uint32_t cb_hdr = get_compile_time_arg_val(15);
    constexpr auto in_args = TensorAccessorArgs<16>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t pkt_bytes = TPP * page_size;
    constexpr uint32_t half_cols = link_cols / 2;
    constexpr uint32_t pkts_per_row_half = half_cols / TPP;
    static_assert(half_cols % TPP == 0, "a link half must be whole packets");

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slot_row0 = get_arg_val<uint32_t>(a++);
    const uint32_t hub_x = get_arg_val<uint32_t>(a++);
    const uint32_t hub_y = get_arg_val<uint32_t>(a++);
    const uint32_t barrier_sem = get_arg_val<uint32_t>(a++);
    const uint32_t waves_idx = a;
    a += 3 * num_waves;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);

    const auto in = TensorAccessor(in_args, in_addr, page_size);
    const auto out = TensorAccessor(out_args, out_addr, page_size);
    const uint64_t barrier_sem_noc = safe_get_noc_addr(hub_x, hub_y, barrier_sem, 0);

    // packet q of the call, in send order: wave j, half h, row r of the wave, packet p of the row-half
    uint32_t total_pkts = 0;
    for (uint32_t j = 0; j < num_waves; j++) {
        total_pkts += 2 * get_arg_val<uint32_t>(waves_idx + 3 * j + 1) * pkts_per_row_half;
    }
    struct Pkt {
        uint32_t row;   // tile row within the slot
        uint32_t col;   // first K tile
        uint32_t half;  // 0 = A, 1 = B
        uint32_t wave;
        bool half_end;
    };
    auto decode = [&](uint32_t q) -> Pkt {
        uint32_t j = 0;
        uint32_t rows = get_arg_val<uint32_t>(waves_idx + 1);
        while (q >= 2 * rows * pkts_per_row_half) {
            q -= 2 * rows * pkts_per_row_half;
            j++;
            rows = get_arg_val<uint32_t>(waves_idx + 3 * j + 1);
        }
        const uint32_t per_half = rows * pkts_per_row_half;
        const uint32_t h = q / per_half;
        const uint32_t w = q - h * per_half;
        const uint32_t r = w / pkts_per_row_half;
        const uint32_t p = w - r * pkts_per_row_half;
        return Pkt{
            get_arg_val<uint32_t>(waves_idx + 3 * j) + r, col0 + h * half_cols + p * TPP, h, j, w + 1 == per_half};
    };

    conn.open();
    // headers live in cb_hdr: BATCH + 1 slots of hdr_stride bytes (the pool holds 8 per RISC)
    const uint32_t hdr_base = get_write_ptr(cb_hdr);
    for (uint32_t i = 0; i < (BATCH + 1) * hdr_stride / 4; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_base)[i] = 0;
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[BATCH];
    for (uint32_t i = 0; i < BATCH; ++i) {
        hdrs[i] = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_base + i * hdr_stride);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* inc_hdr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_base + BATCH * hdr_stride);

    if constexpr (send_barrier) {
        fabric_multicast_noc_unicast_atomic_inc(
            &conn,
            inc_hdr,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{barrier_sem_noc, 1, true},
            1,
            static_cast<uint8_t>(barrier_range));
        noc_async_writes_flushed();
    }

    // L1 staging: two halves of BATCH packets (2 * BATCH * pkt_bytes)
    const uint32_t l1_base = get_write_ptr(cb_stage);
    auto issue_reads = [&](uint32_t first_pkt, uint32_t buf) {
        const uint32_t last = first_pkt + BATCH < total_pkts ? first_pkt + BATCH : total_pkts;
        uint32_t l1 = l1_base + buf * BATCH * pkt_bytes;
        for (uint32_t k = first_pkt; k < last; ++k) {
            const Pkt pk = decode(k);
            for (uint32_t t = 0; t < TPP; ++t) {
                noc_async_read(in.get_noc_addr(pk.row * K_tiles + pk.col + t), l1, page_size);
                l1 += page_size;
            }
        }
    };

    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id));
    issue_reads(0, 0);
    noc_async_read_barrier();
    noc_semaphore_wait_min(go, 1);

    uint32_t buf = 0;
    for (uint32_t b = 0; b < total_pkts; b += BATCH) {
        noc_async_writes_flushed();  // the previous batch's sends (payloads in the other half, headers) have left L1
        if (b + BATCH < total_pkts) {
            issue_reads(b + BATCH, buf ^ 1);
        }
        const uint32_t last = b + BATCH < total_pkts ? b + BATCH : total_pkts;
        uint32_t l1 = l1_base + buf * BATCH * pkt_bytes;
        for (uint32_t k = b; k < last; ++k) {
            const Pkt pk = decode(k);
            const uint32_t pg = (slot_row0 + pk.row) * K_tiles + pk.col;
            const uint64_t d0 = out.get_noc_addr(pg, 0, 0);
            const uint64_t d1 = out.get_noc_addr(pg + 1, 0, 0);
            const uint64_t d2 = out.get_noc_addr(pg + 2, 0, 0);
            const uint64_t d3 = out.get_noc_addr(pg + 3, 0, 0);
            const uint8_t rng = static_cast<uint8_t>(pk.half == 0 ? range_a : range_b);
            // hdrs[k % BATCH] was last used one batch ago; the flush before this batch's reads covers it
            fabric_multicast_noc_scatter_write(
                &conn,
                hdrs[k % BATCH],
                l1,
                pkt_bytes,
                tt::tt_fabric::NocUnicastScatterCommandHeader(
                    {d0, d1, d2, d3},
                    {static_cast<uint16_t>(page_size),
                     static_cast<uint16_t>(page_size),
                     static_cast<uint16_t>(page_size)}),
                1,
                rng);
            if constexpr (copy_local) {
                if ((pk.half == 0) == is_fwd) {
                    for (uint32_t t = 0; t < TPP; ++t) {
                        noc_async_write(l1 + t * page_size, out.get_noc_addr(pg + t), page_size);
                    }
                }
            }
            l1 += pkt_bytes;
            if (pk.half_end) {
                const uint32_t word = get_arg_val<uint32_t>(waves_idx + 3 * pk.wave + 2);
                noc_async_writes_flushed();
                fabric_multicast_noc_unicast_atomic_inc(
                    &conn,
                    inc_hdr,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(hub_x, hub_y, word, 0), 1, true},
                    1,
                    rng);
            }
        }
        noc_async_read_barrier();
        buf ^= 1;
    }
    noc_async_writes_flushed();
    conn.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
