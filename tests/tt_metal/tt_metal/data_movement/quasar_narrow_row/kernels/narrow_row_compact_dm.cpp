// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// STAGE 2 of the Quasar narrow-row pack-untilize (test id 918): squeeze the tile-aligned
// padding out of stage 1's output, in place of the narrow row the packer cannot produce.
//
// WHAT IT PRODUCES
// Stage 1 leaves 32 untilized rows in the `pad` DFB, each `pad_row_bytes` = ct_dim*32 datums
// wide. The caller wants a DENSE matrix `matrix_w` datums wide, where
//
//     matrix_w = (ct_dim - 1) * 32 + last_tile_w
//
// i.e. every row keeps its leading `out_row_bytes` bytes and drops the tail. That is exactly
// the layout the RV_PACR per-face-row path builds datum by datum; here it falls out of a
// single strided gather, because stage 1 already did the face de-interleave in hardware.
//
//     pad (stride pad_row_bytes)          out (stride out_row_bytes)
//     [ keep | junk ]                     [ keep ]
//     [ keep | junk ]        ==>          [ keep ]
//     ... 32 rows ...                     ... 32 rows ...
//
// NOTE what is NOT here, compared with RV_PACR: no face de-interleave (`remap()`), no
// two-pass narrow-tile-first ordering, no 16-datum spill to overwrite. RV_PACR needs all
// three because its output address is 16-BYTE granular, so it can only ever place a whole
// 16-datum face-row. iDMA addresses L1 by the byte, so each row is written at exactly
// `r * out_row_bytes` and nothing spills. That also means widths RV_PACR cannot reach at all
// (anything below 8 datums for a 16-bit format, or below 16 for an 8-bit one) are just
// another `out_row_bytes` here.
//
// THREE ENGINES, ONE OUTPUT
// All three produce byte-identical output; the host verifies each. They differ only in what
// drives the 32 row moves:
//
//   0 IDMA_SCATTER  ONE iDMA transaction. The hardware walks a 32-entry address list in L1
//                   (8 B/entry, low 32 bits = offset from SCATTER_BASE_ADDR) and the
//                   destination auto-increments, so the RISC issues once for the whole
//                   matrix. This is the proposal.
//   1 IDMA_PER_ROW  32 iDMA transactions, addresses from the address generator, one drain at
//                   the end. Separates "iDMA is fast" from "one issue is fast": its cost is
//                   dominated by per-issue RISC overhead, not by the data path.
//   2 NOC_PER_ROW   32 stateful NOC reads from this core's own L1. This is the CURRENT
//                   workaround -- a read per row so the consumer never sees the junk -- and
//                   therefore the number stage 2 has to beat. Stateful means only the
//                   addresses change per call, so it is the cheapest NOC read available, not
//                   a straw man.
//
// COMMAND BUFFER DISCIPLINE
// The iDMA paths borrow cmdbuf 0, which is also OVERLAY_WR_CMD_BUF -- the buffer the device
// profiler uses to push its timestamp buffer out at kernel end. Leaving it in iDMA mode would
// corrupt that push, so it is restored with init_wr_cmd_buf() after the final ack (reset only
// AFTER the ack: a CMDBUF_RESET with an iDMA ack outstanding can disturb it). The NOC path
// uses the read cmd buf via the Noc API and never touches cmdbuf 0.
//
// MEASUREMENT: the zone repeats the whole compaction `num_iterations` times. Unlike stage 1,
// repeating is safe here -- an L1->L1 gather is idempotent, so the data left behind after the
// last iteration is still the correct answer and the run is a correctness test and a
// steady-state timing at once.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"                     // DeviceZoneScopedN / DeviceTimestampedData
#include "api/dataflow/dataflow_buffer.h"                  // DataflowBuffer
#include "api/dataflow/endpoints.h"                        // Noc, UnicastEndpoint
#include "experimental/kernel_args.h"                      // get_arg(args::name)
#include "internal/tt-2xx/quasar/noc_nonblocking_api.h"    // init_wr_cmd_buf, noc_local_xy
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"  // addrgen src/dest loops
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"  // overlay:: cmdbuf API

using namespace overlay;

namespace {

constexpr std::uint32_t ENGINE_IDMA_SCATTER = 0;
constexpr std::uint32_t ENGINE_IDMA_PER_ROW = 1;
constexpr std::uint32_t ENGINE_NOC_PER_ROW = 2;

// The whole of stage 2's addressing rests on this, so: on a non-TRISC core
// `cb_addr_shift == 0` (circular_buffer_interface.h), so DataflowBuffer::get_read_ptr() is a
// BYTE address here -- not the 16-byte units a TRISC gets, which is why the compute-side
// callers shift by 4 and this one must not. What it does add is MEM_L1_UNCACHED_BASE
// (dataflow_buffer.h, L1_UNCACHED_OFFSET, Quasar DM only). iDMA addresses physical L1 exactly
// as the NOC does, so map that alias back the same way Noc::l1_cached_view() does before it
// reaches a cmdbuf register. The NOC API applies that mapping itself; the overlay cmdbuf API
// does not, which is what makes this explicit call necessary.
inline std::uint32_t l1_phys(std::uint32_t addr) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    return (addr >= MEM_L1_UNCACHED_BASE && addr < MEM_L1_UNCACHED_BASE + MEM_L1_SIZE) ? addr - MEM_L1_UNCACHED_BASE
                                                                                       : addr;
#else
    return addr;
#endif
}

// ENGINE 0 -- one transaction, hardware walks the row list.
// Both registers have to be re-armed every iteration: SCATTER_INDEX advances as the engine
// consumes entries, and the destination pointer has walked to the end of the output.
inline void compact_scatter_list(
    std::uint32_t list_addr, std::uint32_t src_base, std::uint32_t dst_base, std::uint32_t num_rows) {
    set_scatter_list_cmdbuf_0(list_addr, src_base, /*index=*/0, /*times=*/num_rows);
    set_dest_cmdbuf_0(dst_base);
    issue_cmdbuf_0();
    while (!idma_acked_cmdbuf_0()) {
    }
}

// ENGINE 1 -- one transaction per row, addresses supplied by the address generator so the
// RISC does not recompute them. Issue all rows, then a single drain: that measures throughput
// rather than per-transfer latency, which is the favourable reading for this engine.
inline void compact_idma_per_row(
    std::uint32_t src_base,
    std::uint32_t dst_base,
    std::uint32_t num_rows,
    std::uint32_t src_stride,
    std::uint32_t dst_stride) {
    // Counters must restart each iteration or the addrgen walks past the buffers.
    reset_counters_addrgen_0();
    setup_src_base_start_addrgen_0(src_base);
    setup_src_inner_loop_addrgen_0(src_stride, (std::uint64_t)num_rows * src_stride);
    setup_dest_base_start_addrgen_0(dst_base);
    setup_dest_inner_loop_addrgen_0(dst_stride, (std::uint64_t)num_rows * dst_stride);

    for (std::uint32_t r = 0; r < num_rows; r++) {
        push_both_addrgen_0();
        issue_cmdbuf_0();
    }
    while (!idma_acked_cmdbuf_0()) {
    }
}

// ENGINE 2 -- the current workaround: one stateful NOC read per row, local L1 loopback.
inline void compact_noc_per_row(
    const Noc& noc,
    std::uint32_t noc_x,
    std::uint32_t noc_y,
    std::uint32_t src_base,
    std::uint32_t dst_base,
    std::uint32_t row_bytes,
    std::uint32_t num_rows,
    std::uint32_t src_stride,
    std::uint32_t dst_stride) {
    UnicastEndpoint ep;
    for (std::uint32_t r = 0; r < num_rows; r++) {
        noc.async_read_with_state(
            ep,
            ep,
            row_bytes,
            {.noc_x = noc_x, .noc_y = noc_y, .addr = src_base + r * src_stride},
            {.addr = dst_base + r * dst_stride});
    }
    noc.async_read_barrier();
}

}  // namespace

void kernel_main() {
    const std::uint32_t dst_addr = get_arg(args::dst_addr);            // dense output, L1
    const std::uint32_t list_addr = get_arg(args::list_addr);          // scatter list, L1
    const std::uint32_t pad_row_bytes = get_arg(args::pad_row_bytes);  // ct_dim * 32 * datum_bytes
    const std::uint32_t out_row_bytes = get_arg(args::out_row_bytes);  // matrix_w * datum_bytes
    const std::uint32_t num_rows = get_arg(args::num_rows);            // 32 for a 32-row tile-row
    const std::uint32_t engine_mode = get_arg(args::engine_mode);
    const std::uint32_t num_iterations = get_arg(args::num_iterations);
    const std::uint32_t dest_coords = get_arg(args::dest_coords);  // packed (x << 16) | y
    const std::uint32_t test_id = get_arg(args::test_id);
    // Packet split. `out_row_bytes` gives one packet per row, which DISABLES fan-out (the
    // round-robin distributes PACKETS) -- correct while rows are short enough to be issue-
    // bound, wrong once a row is long enough to be data-bound. Host policy, not a constant
    // here. It must be programmed explicitly whatever the value: CMDBUF_RESET zeroes
    // MAX_BYTES_IN_PACKET to its rdl default of 0, which means "never split", and every
    // channel knob then goes silently inert.
    const std::uint32_t max_packet_bytes = get_arg(args::max_packet_bytes);

    std::uint32_t num_channels = get_arg(args::num_channels);
    if (num_channels < 1) {
        num_channels = 1;
    } else if (num_channels > CMDBUF_NUM_IDMA_VCS) {
        num_channels = CMDBUF_NUM_IDMA_VCS;
    }
    const bool fan_out = num_channels > 1;
    const bool use_idma = engine_mode != ENGINE_NOC_PER_ROW;

    const std::uint32_t noc_x = dest_coords >> 16;
    const std::uint32_t noc_y = dest_coords & 0xFFFF;

    Noc noc(noc_index);

    // Wait for stage 1's whole 32-row block, then take its base. The DFB was pushed exactly
    // once and never wraps, so the 32 entries are contiguous from the read pointer.
    DataflowBuffer pad(dfb::pad);
    pad.wait_front(num_rows);
    const std::uint32_t src_base = l1_phys(pad.get_read_ptr());

    // ---- setup, outside the timed region ------------------------------------------------
    if (use_idma) {
        reset_cmdbuf_0();
        if (engine_mode == ENGINE_IDMA_SCATTER) {
            // The list drives ONE side and carries neither size nor xy, so every entry moves
            // set_len bytes. Source is the strided side (skip each row's padding) and the
            // destination walks contiguously, which is precisely a compacting gather:
            //   apply_scatter_to_dest = false -> entries are SOURCE addresses
            //   dest_addr_inc_en      = true  -> dense output, row after row
            idma_setup_as_scatter_list_cmdbuf_0(
                /*apply_scatter_to_dest=*/false,
                /*scatter_list_contains_size=*/false,
                /*scatter_list_contains_xy=*/false,
                /*wrapping_en=*/false);
            setup_ongoing_cmdbuf_0(
                /*src_addr_inc_en=*/false,  // source addresses come from the list
                /*dest_addr_inc_en=*/true,
                /*trid_inc_en=*/false,
                /*req_vc_inc_en=*/fan_out,
                /*resp_vc_inc_en=*/false);
        } else {
            idma_setup_as_copy_cmdbuf_0(/*wrapping_en=*/false);
            setup_ongoing_cmdbuf_0(
                /*src_addr_inc_en=*/false,  // addrgen supplies both addresses
                /*dest_addr_inc_en=*/false,
                /*trid_inc_en=*/false,
                /*req_vc_inc_en=*/fan_out,
                /*resp_vc_inc_en=*/false);
        }
        setup_wrapping_vcs_cmdbuf_0(
            /*wr=*/true,
            /*req_start_vc=*/CMDBUF_FIRST_IDMA_VC,
            /*req_end_vc=*/CMDBUF_FIRST_IDMA_VC + (fan_out ? num_channels - 1 : 0));
        setup_max_bytes_in_packet_cmdbuf_0(max_packet_bytes);
        setup_trids_cmdbuf_0(CMDBUF_DEF_TRID);
        set_len_cmdbuf_0(out_row_bytes);
    } else {
        // Set the NOC read state once, outside the timed region, so the loop pays only for
        // the per-row issue -- matching the noc_api_latency stateful-read kernel.
        UnicastEndpoint ep;
        noc.set_async_read_state(ep, out_row_bytes, {.noc_x = noc_x, .noc_y = noc_y, .addr = src_base});
    }

    // ---- the measurement ------------------------------------------------------------------
    {
        DeviceZoneScopedN("COMPACT");
        for (std::uint32_t iter = 0; iter < num_iterations; iter++) {
            switch (engine_mode) {
                case ENGINE_IDMA_SCATTER: compact_scatter_list(list_addr, src_base, dst_addr, num_rows); break;
                case ENGINE_IDMA_PER_ROW:
                    compact_idma_per_row(src_base, dst_addr, num_rows, pad_row_bytes, out_row_bytes);
                    break;
                default:
                    compact_noc_per_row(
                        noc, noc_x, noc_y, src_base, dst_addr, out_row_bytes, num_rows, pad_row_bytes, out_row_bytes);
                    break;
            }
        }
    }

    if (use_idma) {
        // Every ack is in: safe to hand cmdbuf 0 back to the NoC write path (profiler push).
        init_wr_cmd_buf(noc_local_xy());
    }

    pad.pop_front(num_rows);

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_iterations);
    DeviceTimestampedData("Transaction size in bytes", out_row_bytes);
    DeviceTimestampedData("Number of rows", num_rows);
    DeviceTimestampedData("Packet size in bytes", max_packet_bytes);
    DeviceTimestampedData("Engine mode", engine_mode);
    // Stamped last: the CSV reconstruction uses this to flush a run.
    DeviceTimestampedData("Number of channels", num_channels);
}
