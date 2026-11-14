// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// qwen_ag_matmul v2 in0 (activation) data movement on the full 7x10 grid, with the fabric senders of the row
// all-gather folded into eight of the in0 receiver cores.
//
// Matmul part: as mm_in0.cpp (stock minimal_matmul dm_in0_sender.cpp, transposed grid, explicit per-column block
// lists in common runtime args). The injector (grid row 0) waits, before a remote block of wave j, until its wave
// word j reached arrivals_expected (2 halves x 4 links x 7 remote slots: wave j of every remote slot has landed).
//
// Sender part (SENDER): one direction of one link of the all-gather, as ag_sender.cpp, but run as a pump that sends
// at most one packet per call and never blocks; every wait of the in0 receiver loop calls it. Link l carries K-tile
// columns [col0, col0 + link_cols); half A goes range_a hops, half B range_b; after each wave-half the sender
// increments wave word j on all GX injectors of every receiver. Before sending, the sender waits for its barrier
// word (one inc from each peer's link-0 senders: every peer has launched this call). With copy_local the fwd sender
// also writes its half A pages, the bwd sender its half B pages, to the own slot of the gathered buffer.
//
// CT: 0 K_tiles, 1 padded_K_tiles, 2 M_block_tiles, 3 K_block_tiles, 4 in0_tile_size, 5 in0_sender_sem,
//     6 in0_receiver_sem, 7 in0_valid_sem, 8 is_injector_core, 9 arrivals_expected, 10 slot_rows, 11 my_slot,
//     12 M_tiles, 13 GX, 14 GY, 15 num_waves, then accessors: gathered (in0), x_local;
//     SENDER only, after the accessors: page_size, range_a, range_b, is_fwd, send_barrier, barrier_range, batch,
//     hdr_stride, col0, link_cols, copy_local, cb_stage, cb_hdr, num_sender_cores
// Common RT: 0 in0_addr, 1 x_local_addr, 2 words_addr (wave word j at words_addr + 16 * (1 + j), barrier word at
//     words_addr), then GX NOC x of the grid columns, GY NOC y of the grid rows, GX list offsets, the lists (per
//     grid column: num_blocks, num_blocks x (m_start_tile, rows, wave, or 0xFFFF for an own-slot block)),
//     then num_waves x (row0, rows), then num_sender_cores x (noc_x, noc_y)
// RT (SENDER): fabric connection args
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#ifdef SENDER
#include "tt_metal/fabric/hw/inc/linear/api.h"
using namespace tt::tt_fabric::linear::experimental;
#endif

constexpr uint32_t WORD_STRIDE = 16;
constexpr uint32_t OWN_SLOT = 0xFFFF;
constexpr uint32_t ROWS_RING_PEERS = 7;  // barrier incs expected: one from each peer's link-0 senders

void kernel_main() {
    Noc noc;
    constexpr uint32_t K_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(4);
    Semaphore<> in0_sender_semaphore(get_compile_time_arg_val(5));
    Semaphore<> in0_receiver_semaphore(get_compile_time_arg_val(6));
    Semaphore<> in0_valid_semaphore(get_compile_time_arg_val(7));
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(8);
    constexpr uint32_t arrivals_expected = get_compile_time_arg_val(9);
    constexpr uint32_t slot_rows = get_compile_time_arg_val(10);
    constexpr uint32_t my_slot = get_compile_time_arg_val(11);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t GX = get_compile_time_arg_val(13);
    constexpr uint32_t GY = get_compile_time_arg_val(14);
    constexpr uint32_t num_waves = get_compile_time_arg_val(15);

    const uint32_t cx = get_absolute_logical_x();
    const uint32_t cy = get_absolute_logical_y();
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t xl_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t words_addr = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t PX = 3, PY = 3 + GX, PL = 3 + GX + GY;
    const uint32_t is_sink_core = cy == GY - 1;
    const uint32_t in0_dest_noc_x = get_common_arg_val<uint32_t>(PX + cx);
    const uint32_t in0_dest_noc_y = get_common_arg_val<uint32_t>(PY + (cy + 1 < GY ? cy + 1 : GY - 1));
    const uint32_t in0_sender_noc_x = in0_dest_noc_x;
    const uint32_t in0_sender_noc_y = get_common_arg_val<uint32_t>(PY + (cy > 0 ? cy - 1 : 0));
    const uint32_t list_at = get_common_arg_val<uint32_t>(PL + cx);
    const uint32_t num_blocks = get_common_arg_val<uint32_t>(list_at);
    const uint32_t blocks_idx = list_at + 1;
    // the waves and sender-core table follow the last list
    const uint32_t last_list = get_common_arg_val<uint32_t>(PL + GX - 1);
    const uint32_t waves_idx = last_list + 1 + 3 * get_common_arg_val<uint32_t>(last_list);

    constexpr auto in0_args = TensorAccessorArgs<16>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);
    constexpr auto xl_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto xl_reader = TensorAccessor(xl_args, xl_addr);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    CircularBuffer cb_in0(cb_in0_id);

    // ------------------------------------------------------------------------------------------------------------
    // Sender pump
    // ------------------------------------------------------------------------------------------------------------
#ifdef SENDER
    constexpr uint32_t S0 = xl_args.next_compile_time_args_offset();
    constexpr uint32_t page_size = get_compile_time_arg_val(S0 + 0);
    constexpr uint32_t range_a = get_compile_time_arg_val(S0 + 1);
    constexpr uint32_t range_b = get_compile_time_arg_val(S0 + 2);
    constexpr bool is_fwd = get_compile_time_arg_val(S0 + 3) != 0;
    constexpr bool send_barrier = get_compile_time_arg_val(S0 + 4) != 0;
    constexpr uint32_t barrier_range = get_compile_time_arg_val(S0 + 5);
    constexpr uint32_t BATCH = get_compile_time_arg_val(S0 + 6);
    constexpr uint32_t hdr_stride = get_compile_time_arg_val(S0 + 7);
    constexpr uint32_t col0 = get_compile_time_arg_val(S0 + 8);
    constexpr uint32_t link_cols = get_compile_time_arg_val(S0 + 9);
    constexpr bool copy_local = get_compile_time_arg_val(S0 + 10) != 0;
    constexpr uint32_t cb_stage = get_compile_time_arg_val(S0 + 11);
    constexpr uint32_t cb_hdr = get_compile_time_arg_val(S0 + 12);
    constexpr uint32_t num_sender_cores = get_compile_time_arg_val(S0 + 13);
    constexpr uint32_t TPP = 4;
    constexpr uint32_t pkt_bytes = TPP * page_size;
    constexpr uint32_t half_cols = link_cols / 2;
    constexpr uint32_t pkts_per_row_half = half_cols / TPP;
    static_assert(half_cols % TPP == 0, "a link half must be whole packets");
    const uint32_t senders_idx = waves_idx + 2 * num_waves;

    size_t a = 0;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    const auto snd_in = TensorAccessor(xl_args, xl_addr, page_size);
    const auto snd_out = TensorAccessor(in0_args, in0_addr, page_size);

    uint32_t total_pkts = 0;
    for (uint32_t j = 0; j < num_waves; j++) {
        total_pkts += 2 * get_common_arg_val<uint32_t>(waves_idx + 2 * j + 1) * pkts_per_row_half;
    }
    struct Pkt {
        uint32_t row, col, half, wave;
        bool half_end;
    };
    auto decode = [&](uint32_t q) -> Pkt {
        uint32_t j = 0;
        uint32_t rows = get_common_arg_val<uint32_t>(waves_idx + 1);
        while (q >= 2 * rows * pkts_per_row_half) {
            q -= 2 * rows * pkts_per_row_half;
            j++;
            rows = get_common_arg_val<uint32_t>(waves_idx + 2 * j + 1);
        }
        const uint32_t per_half = rows * pkts_per_row_half;
        const uint32_t h = q / per_half;
        const uint32_t w = q - h * per_half;
        const uint32_t r = w / pkts_per_row_half;
        const uint32_t p = w - r * pkts_per_row_half;
        return Pkt{
            get_common_arg_val<uint32_t>(waves_idx + 2 * j) + r,
            col0 + h * half_cols + p * TPP,
            h,
            j,
            w + 1 == per_half};
    };

    conn.open();
    // headers live in cb_hdr: BATCH data headers, then GX inc headers (one per injector, so an inc round needs no
    // flush between its packets)
    const uint32_t hdr_base = get_write_ptr(cb_hdr);
    for (uint32_t i = 0; i < (BATCH + GX) * hdr_stride / 4; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hdr_base)[i] = 0;
    }
    auto hdr = [&](uint32_t i) {
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_base + i * hdr_stride);
    };
    if constexpr (send_barrier) {
        for (uint32_t i = 0; i < num_sender_cores; i++) {
            noc_async_writes_flushed();
            fabric_multicast_noc_unicast_atomic_inc(
                &conn,
                hdr(BATCH),
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    safe_get_noc_addr(
                        get_common_arg_val<uint32_t>(senders_idx + 2 * i),
                        get_common_arg_val<uint32_t>(senders_idx + 2 * i + 1),
                        words_addr,
                        0),
                    1,
                    true},
                1,
                static_cast<uint8_t>(barrier_range));
        }
        noc_async_writes_flushed();
    }

    const uint32_t l1_base = get_write_ptr(cb_stage);
    auto issue_reads = [&](uint32_t first_pkt, uint32_t half) {
        const uint32_t last = first_pkt + BATCH < total_pkts ? first_pkt + BATCH : total_pkts;
        uint32_t l1 = l1_base + half * BATCH * pkt_bytes;
        for (uint32_t k = first_pkt; k < last; ++k) {
            const Pkt pk = decode(k);
            for (uint32_t t = 0; t < TPP; ++t) {
                noc_async_read(snd_in.get_noc_addr(pk.row * K_tiles + pk.col + t), l1, page_size);
                l1 += page_size;
            }
        }
    };
    volatile tt_l1_ptr uint32_t* barrier_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words_addr);
    bool go = false;
    uint32_t k_pkt = 0;
    bool batch_started = false;
    uint32_t inc_pending = 0;  // incs of the current round still to send (GX per wave-half)
    uint32_t inc_word = 0;
    uint8_t inc_rng = 0;
    issue_reads(0, 0);
    // One step of the sender (one packet or one inc at most); returns at once whenever it would wait.
    auto pump = [&]() {
        if (inc_pending > 0) {
            if (inc_pending == GX && !ncrisc_noc_nonposted_writes_sent(noc_index)) {
                return;  // the previous round's headers and the data ahead of this round have left L1
            }
            if (!conn.edm_has_space_for_packet()) {
                return;
            }
            const uint32_t x = GX - inc_pending;
            fabric_multicast_noc_unicast_atomic_inc(
                &conn,
                hdr(BATCH + x),
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    safe_get_noc_addr(
                        get_common_arg_val<uint32_t>(PX + x), get_common_arg_val<uint32_t>(PY), inc_word, 0),
                    1,
                    true},
                1,
                inc_rng);
            inc_pending--;
            return;
        }
        if (k_pkt >= total_pkts) {
            return;
        }
        if (!go) {
            invalidate_l1_cache();
            if (*barrier_word < ROWS_RING_PEERS) {
                return;
            }
            noc_semaphore_inc(get_noc_addr(words_addr), static_cast<uint32_t>(-static_cast<int32_t>(ROWS_RING_PEERS)));
            go = true;
        }
        const uint32_t half = (k_pkt / BATCH) & 1;
        if (!batch_started) {
            // this batch's reads are done and the other half (sent one batch ago) has left L1: prefetch the next
            if (!ncrisc_noc_reads_flushed(noc_index) || !ncrisc_noc_nonposted_writes_sent(noc_index)) {
                return;
            }
            invalidate_l1_cache();
            const uint32_t next = (k_pkt / BATCH + 1) * BATCH;
            if (next < total_pkts) {
                issue_reads(next, half ^ 1);
            }
            batch_started = true;
        }
        if (!conn.edm_has_space_for_packet()) {
            return;
        }
        const Pkt pk = decode(k_pkt);
        const uint32_t l1 = l1_base + (half * BATCH + k_pkt % BATCH) * pkt_bytes;
        const uint32_t pg = (my_slot * slot_rows + pk.row) * K_tiles + pk.col;
        const uint8_t rng = static_cast<uint8_t>(pk.half == 0 ? range_a : range_b);
        fabric_multicast_noc_scatter_write(
            &conn,
            hdr(k_pkt % BATCH),
            l1,
            pkt_bytes,
            tt::tt_fabric::NocUnicastScatterCommandHeader(
                {snd_out.get_noc_addr(pg, 0, 0),
                 snd_out.get_noc_addr(pg + 1, 0, 0),
                 snd_out.get_noc_addr(pg + 2, 0, 0),
                 snd_out.get_noc_addr(pg + 3, 0, 0)},
                {static_cast<uint16_t>(page_size), static_cast<uint16_t>(page_size), static_cast<uint16_t>(page_size)}),
            1,
            rng);
        if constexpr (copy_local) {
            if ((pk.half == 0) == is_fwd) {
                for (uint32_t t = 0; t < TPP; ++t) {
                    noc_async_write(l1 + t * page_size, snd_out.get_noc_addr(pg + t), page_size);
                }
            }
        }
        if (pk.half_end) {
            inc_pending = GX;
            inc_word = words_addr + WORD_STRIDE * (1 + pk.wave);
            inc_rng = rng;
        }
        k_pkt++;
        if (k_pkt % BATCH == 0) {
            batch_started = false;
        }
    };
#define PUMP() pump()
#else
#define PUMP() \
    do {       \
    } while (0)
#endif

    volatile tt_l1_ptr uint32_t* recv_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_compile_time_arg_val(6)));
    volatile tt_l1_ptr uint32_t* send_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_compile_time_arg_val(5)));

    in0_valid_semaphore.set(VALID);

    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t m_tile = get_common_arg_val<uint32_t>(blocks_idx + 3 * b);
        const uint32_t rows = get_common_arg_val<uint32_t>(blocks_idx + 3 * b + 1);
        const uint32_t wave = get_common_arg_val<uint32_t>(blocks_idx + 3 * b + 2);
        const uint32_t m_tile_end = m_tile + rows;
        const uint32_t current_block_bytes = rows * K_block_tiles * in0_tile_size;
        const bool local = wave == OWN_SLOT;
        uint32_t row_base = m_tile;
        if constexpr (is_injector_core) {
            if (local) {
                row_base = m_tile - my_slot * slot_rows;
            } else {
#ifndef PROBE_NOWAIT
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words_addr + WORD_STRIDE * (1 + wave)),
                    arrivals_expected);
#endif
            }
        }

        for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
            while (!cb_in0.pages_reservable_at_back(in0_block_num_tiles)) {
                PUMP();
            }
            cb_in0.reserve_back(in0_block_num_tiles);

            uint32_t in0_start_address = cb_in0.get_write_ptr();
            if constexpr (is_injector_core) {
                uint32_t write_ptr = in0_start_address;
                const uint32_t d1_start = k_block * K_block_tiles;
                const uint32_t d1_end = d1_start + K_block_tiles;
                for (uint32_t i = 0; i < rows; i++) {
                    if (m_tile + i >= M_tiles) {
                        break;
                    }
                    for (uint32_t j = d1_start; j < d1_end; j++) {
                        const uint32_t tile_id = (row_base + i) * K_tiles + j;
                        if (local) {
                            noc.async_read(
                                xl_reader, CoreLocalMem<uint32_t>(write_ptr), in0_tile_size, {.page_id = tile_id}, {});
                        } else {
                            noc.async_read(
                                in0_reader, CoreLocalMem<uint32_t>(write_ptr), in0_tile_size, {.page_id = tile_id}, {});
                        }
                        write_ptr += in0_tile_size;
                    }
                }
                noc.async_read_barrier();
            } else {
                // Get from previous device
                in0_receiver_semaphore.set(INVALID);
                in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                while (true) {
                    invalidate_l1_cache();
                    if (*recv_sem == VALID) {
                        break;
                    }
                    PUMP();
                }
            }

            cb_in0.push_back(in0_block_num_tiles);

            if (!is_sink_core) {
                while (true) {
                    invalidate_l1_cache();
                    if (*send_sem >= 1) {
                        break;
                    }
                    PUMP();
                }
                in0_sender_semaphore.set(0);

                if (current_block_bytes > 0) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(in0_start_address),
                        UnicastEndpoint{},
                        current_block_bytes,
                        {},
                        {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});
                }

                in0_valid_semaphore.relay_unicast(noc, in0_receiver_semaphore, in0_dest_noc_x, in0_dest_noc_y);
            }
        }
    }
#ifdef SENDER
    while (k_pkt < total_pkts || inc_pending > 0) {
        pump();
    }
    noc_async_writes_flushed();
    conn.close();
#endif
    if constexpr (is_injector_core) {
        // Every injector gets every wave's incs. Once the last wave is complete, every earlier one is too (each sender
        // stream delivers in order), so all words can go back down (peers of the next call increment only after this
        // device launched it).
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words_addr + WORD_STRIDE * num_waves), arrivals_expected);
        for (uint32_t j = 0; j < num_waves; j++) {
            noc_semaphore_inc(
                get_noc_addr(words_addr + WORD_STRIDE * (1 + j)),
                static_cast<uint32_t>(-static_cast<int32_t>(arrivals_expected)));
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
