// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Routed-expert program, writer (NOC 1, one per core).  Replaces moe_compute's 24-hop all-to-all ring by one exchange:
// a core with a gate/up unit sends its in2 tile(s) (columns g_col, g_col + 1 of the 20) to the in2 buffer of each of
// the 16 down cores (double buffered by the expert's parity) and raises their `sem_in2`.  Only bytes [0, 576) of a tile
// travel (face 0 = rows 0..15 of columns 0..15 and rows 0..1 of face 1): the rows are independent in a matmul tile and
// at most 2 are valid, so one 576-byte packet per tile and destination replaces the 2 KB tile.  A down core waits for
// the 12 senders of expert i (sem_in2 >= 12 (i + 1) per parity), hands compute a token, and when compute has packed the
// unit's output rows acknowledges the 12 senders (`sem_ack`, 16 per expert: a sender may refill a parity buffer only
// after all 16 cores have read it) and writes the rows of the tokens that selected the expert to the local combine
// buffer [10, rows, 2560]: page k_slot * rows + row at the unit's column offset (moe_compute's dm1.cpp local-output
// write). Per expert i: receive(i) (the token of W(i)), send(i + 1), out(i); send(0) first.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#include "moe_experts_common.h"
#include "../../kernels/zones.h"

using namespace moe_experts;

void kernel_main() {
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t cb_gout = get_named_compile_time_arg_val("cb_gout");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_tok = get_named_compile_time_arg_val("cb_tok");
    constexpr uint32_t cb_in2 = get_named_compile_time_arg_val("cb_in2");
    constexpr uint32_t cb_mdw = get_named_compile_time_arg_val("cb_mdw");
    // per-parity counters: arrivals / acknowledgements of the experts of one parity are ordered (a sender refills a
    // parity buffer only after all 24 cores acknowledged the previous expert of that parity), those of both parities
    // are not
    constexpr uint32_t sem_in2_ids[2] = {
        get_named_compile_time_arg_val("sem_in2_0"), get_named_compile_time_arg_val("sem_in2_1")};
    constexpr uint32_t sem_ack_ids[2] = {
        get_named_compile_time_arg_val("sem_ack_0"), get_named_compile_time_arg_val("sem_ack_1")};
    constexpr uint32_t n_g = get_named_compile_time_arg_val("n_g");  // gate/up units = senders
    constexpr uint32_t n_w = get_named_compile_time_arg_val("n_w");  // down units = receivers (all cores)
    constexpr uint32_t stage = get_named_compile_time_arg_val("stage");
    constexpr uint32_t dump = get_named_compile_time_arg_val("dump");  // also write the in2 tiles to a DRAM tensor
    constexpr uint32_t tile_bytes = TILE_BYTES;
    constexpr uint32_t row_tile_bytes = 64;  // one tile column of a row: 32 bf16
    constexpr uint32_t sent_bytes = 576;     // face 0 + rows 0 and 1 of face 1 of an in2 tile

    constexpr auto out_args = TensorAccessorArgs<0, 0>();
    constexpr auto dump_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dump_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t vc = get_arg_val<uint32_t>(arg++);
    const uint32_t g_kind = get_arg_val<uint32_t>(arg++);
    const uint32_t w_units = get_arg_val<uint32_t>(arg++);
    uint32_t w_kind[MAX_W_UNITS], col_tile_start[MAX_W_UNITS];
    for (uint32_t u = 0; u < MAX_W_UNITS; ++u) {
        w_kind[u] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t g_col = get_arg_val<uint32_t>(arg++);
    for (uint32_t u = 0; u < MAX_W_UNITS; ++u) {
        col_tile_start[u] = get_arg_val<uint32_t>(arg++);
    }
    uint32_t dest_x[n_w], dest_y[n_w], gcore_x[n_g], gcore_y[n_g];
    for (uint32_t d = 0; d < n_w; ++d) {
        dest_x[d] = get_arg_val<uint32_t>(arg++);
        dest_y[d] = get_arg_val<uint32_t>(arg++);
    }
    for (uint32_t d = 0; d < n_g; ++d) {
        gcore_x[d] = get_arg_val<uint32_t>(arg++);
        gcore_y[d] = get_arg_val<uint32_t>(arg++);
    }

    const auto out_tensor = TensorAccessor(out_args, out_addr);
    const auto dump_tensor = TensorAccessor(dump_args, dump_addr);

    Noc noc1(1);
    CircularBuffer gout(cb_gout);
    CircularBuffer outcb(cb_out);
    CircularBuffer tok(cb_tok);
    CircularBuffer in2(cb_in2);
    CircularBuffer mdw(cb_mdw);
    Semaphore<> sem_in2[2] = {Semaphore<>(sem_in2_ids[0]), Semaphore<>(sem_in2_ids[1])};
    Semaphore<> sem_ack[2] = {Semaphore<>(sem_ack_ids[0]), Semaphore<>(sem_ack_ids[1])};
    const uint32_t sem_in2_addr[2] = {get_semaphore(sem_in2_ids[0]), get_semaphore(sem_in2_ids[1])};
    const uint32_t sem_ack_addr[2] = {get_semaphore(sem_ack_ids[0]), get_semaphore(sem_ack_ids[1])};

    mdw.wait_front(1);
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mdw.get_read_ptr());
    const uint32_t n = page[0];
    const uint32_t in2_base = in2.get_write_ptr();
    const uint32_t g_tiles = g_kind == 1 ? 2 : 1;

    auto send = [&](uint32_t j) {
        FUSED_ZONE("fz_me_w_send");
        gout.wait_front(2);
        const uint32_t src = gout.get_read_ptr();
        if constexpr (stage >= 3) {
            if (j >= 2) {
                sem_ack[j & 1].wait_min(n_w * (j >> 1));  // every core has read the parity buffer expert j - 2 filled
            }
            const uint32_t slot = in2_base + ((j & 1) * NT + g_col) * tile_bytes;
            for (uint32_t d = 0; d < n_w; ++d) {
                noc_async_write_one_packet_set_state</*posted=*/true>(
                    get_noc_addr(dest_x[d], dest_y[d], slot), sent_bytes, /*noc=*/1, vc);
                noc_async_write_one_packet_with_state</*posted=*/true>(src, slot);
                if (g_tiles == 2) {
                    noc_async_write_one_packet_with_state</*posted=*/true>(src + tile_bytes, slot + tile_bytes);
                }
                noc_semaphore_inc</*posted=*/true>(
                    get_noc_addr(dest_x[d], dest_y[d], sem_in2_addr[j & 1]), 1, /*noc_id=*/1, vc);
            }
        }
        if constexpr (dump) {
            for (uint32_t t = 0; t < g_tiles; ++t) {
                noc1.async_write(
                    CoreLocalMem<uint32_t>(src + t * tile_bytes),
                    dump_tensor,
                    tile_bytes,
                    {},
                    {.page_id = j * NT + g_col + t});
            }
        }
        noc1.async_writes_flushed<NocOptions::POSTED>();
        noc1.async_writes_flushed();
        gout.pop_front(2);
    };

    auto receive = [&](uint32_t i) {
        FUSED_ZONE("fz_me_w_in2");
        sem_in2[i & 1].wait_min(n_g * ((i >> 1) + 1));
        tok.reserve_back(1);
        tok.push_back(1);
    };

    auto output = [&](uint32_t i) {
        FUSED_ZONE("fz_me_w_out");
        const uint32_t mask = page[2 + MD_ENTRY_WORDS * i];
        for (uint32_t u = 0; u < w_units; ++u) {
            const uint32_t out_row_bytes = (w_kind[u] == 1 ? 4 : 2) * row_tile_bytes;  // the valid tiles of a row ...
            constexpr uint32_t out_tiles = 4;                                          // ... in rows of 4 tiles
            outcb.wait_front(out_tiles);
            if (u + 1 == w_units) {
                // compute has read the in2 tiles of parity i & 1 (the last unit is packed): the senders may refill them
                for (uint32_t d = 0; d < n_g; ++d) {
                    noc_semaphore_inc</*posted=*/true>(
                        get_noc_addr(gcore_x[d], gcore_y[d], sem_ack_addr[i & 1]), 1, /*noc_id=*/1, vc);
                }
            }
            const uint32_t src = outcb.get_read_ptr();
            for (uint32_t r = 0; r < rows; ++r) {
                if (mask & (1u << r)) {
                    const uint32_t k_slot = page[3 + MD_ENTRY_WORDS * i + r];
                    noc1.async_write(
                        CoreLocalMem<uint8_t>(src + r * out_tiles * row_tile_bytes),
                        out_tensor,
                        out_row_bytes,
                        {},
                        {.page_id = k_slot * rows + r, .offset_bytes = col_tile_start[u] * row_tile_bytes});
                }
            }
            noc1.async_writes_flushed();
            noc1.async_writes_flushed<NocOptions::POSTED>();
            outcb.pop_front(out_tiles);
        }
    };

    if (g_kind != 0 && n > 0) {
        send(0);
    }
    for (uint32_t i = 0; i < n; ++i) {
        // the token of W(i) first: its in2 tiles arrived while G(i) / W(i - 1) ran, and compute runs W(i) right after
        // G(i + 1), which must not wait for this core's own send(i + 1) (up to 24 destinations, ~3 us of this RISC)
        if constexpr (stage >= 3) {
            if (w_units != 0) {
                receive(i);
            }
        }
        if (g_kind != 0 && i + 1 < n) {
            send(i + 1);
        }
        if constexpr (stage >= 3) {
            if (w_units != 0) {
                output(i);
            }
        }
    }
    if constexpr (stage >= 3) {
        for (uint32_t p = 0; p < 2; ++p) {
            const uint32_t count = (n + 1 - p) >> 1;  // experts of parity p
            if (count > 0) {
                if (w_units != 0) {
                    sem_in2[p].set(0);
                }
                if (g_kind != 0) {
                    sem_ack[p].wait_min(n_w * count);  // the acknowledgements of the last experts
                    sem_ack[p].set(0);
                }
            }
        }
    }
    noc1.async_full_barrier();
}
