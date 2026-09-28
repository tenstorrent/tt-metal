// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// STAGE 2 of the Quasar narrow-row pack-untilize (test id 918): squeeze the tile-aligned
// padding out of stage 1's output, producing the narrow row the packer cannot.
//
// Stage 1 leaves `num_rows` untilized rows in the `pad` DFB, each `pad_row_bytes` wide with
// junk at the end. This gathers their leading `out_row_bytes` into a dense buffer:
//
//     pad (stride pad_row_bytes)          out (stride out_row_bytes)
//     [ keep | junk ]                     [ keep ]
//     [ keep | junk ]        ==>          [ keep ]
//     ... num_rows ...                    ... num_rows ...
//
// iDMA addresses L1 by the BYTE, so each row lands at exactly `r * out_row_bytes`. That is
// what makes an arbitrary narrow width possible: nothing spills into the next row, and there
// is no minimum width (RV_PACR's output address is 16-byte granular, so 8 datums is its
// floor for a 16-bit format).
//
// TWO ENGINES, ONE OUTPUT. They produce byte-identical results and the host verifies both:
//
//   0 IDMA_PER_ROW  one iDMA transaction per row, addresses from the address generator,
//                   fanned out over `num_channels` backend VCs, ONE drain at the end.
//                   Issue-bound at ~8.7 cyc/row. This is the engine being proposed.
//   1 NOC_PER_ROW   one stateful NOC read per row from this core's own L1. The CURRENT
//                   workaround -- a read per row so the consumer never sees the junk -- and
//                   therefore the bar to beat. Stateful means only the addresses change per
//                   call, so it is the cheapest NOC read available, not a straw man.
//
// COMMAND BUFFER DISCIPLINE. The iDMA path borrows cmdbuf 0, which is also OVERLAY_WR_CMD_BUF
// -- the buffer the device profiler uses to push its timestamps at kernel end. It is restored
// with init_wr_cmd_buf() after the final ack; resetting it with an ack still outstanding can
// disturb the transfer. The NOC path uses the read cmd buf and never touches cmdbuf 0.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/noc_nonblocking_api.h"    // init_wr_cmd_buf, noc_local_xy
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"  // addrgen src/dest loops
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"  // overlay:: cmdbuf API

using namespace overlay;

namespace {

constexpr std::uint32_t ENGINE_IDMA_PER_ROW = 0;
constexpr std::uint32_t ENGINE_NOC_PER_ROW = 1;

// The whole of this kernel's addressing rests on this. On a non-TRISC core
// `cb_addr_shift == 0` (circular_buffer_interface.h), so DataflowBuffer::get_read_ptr() is a
// BYTE address here -- not the 16-byte units a TRISC gets, which is why compute-side callers
// shift by 4 and this one must not. What it does add is MEM_L1_UNCACHED_BASE
// (dataflow_buffer.h, L1_UNCACHED_OFFSET, Quasar DM only). iDMA addresses physical L1 exactly
// as the NOC does, so map that alias back the same way Noc::l1_cached_view() does before it
// reaches a cmdbuf register: the NOC API applies that mapping itself, the cmdbuf API does not.
inline std::uint32_t l1_phys(std::uint32_t addr) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    return (addr >= MEM_L1_UNCACHED_BASE && addr < MEM_L1_UNCACHED_BASE + MEM_L1_SIZE) ? addr - MEM_L1_UNCACHED_BASE
                                                                                       : addr;
#else
    return addr;
#endif
}

// Issue every row, then drain once. Draining inside the loop would serialise the transfers and
// turn a throughput path into a latency one; issuing first also means the RISC has registered
// all `num_rows` outstanding before it polls, so the ack counter cannot read zero early.
inline void compact_idma_per_row(
    std::uint32_t src_base,
    std::uint32_t dst_base,
    std::uint32_t num_rows,
    std::uint32_t src_stride,
    std::uint32_t dst_stride) {
    reset_counters_addrgen_0();
    setup_src_base_start_addrgen_0(src_base);
    setup_src_inner_loop_addrgen_0(src_stride, (std::uint64_t)num_rows * src_stride);
    setup_dest_base_start_addrgen_0(dst_base);
    setup_dest_inner_loop_addrgen_0(dst_stride, (std::uint64_t)num_rows * dst_stride);

    for (std::uint32_t r = 0; r < num_rows; r++) {
        push_both_addrgen_0();  // hand the next src/dst pair to the cmdbuf
        issue_cmdbuf_0();
    }
    while (!idma_acked_cmdbuf_0()) {
    }
}

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
    const std::uint32_t pad_row_bytes = get_arg(args::pad_row_bytes);  // ct_dim * 32 * datum_bytes
    const std::uint32_t out_row_bytes = get_arg(args::out_row_bytes);  // matrix_w * datum_bytes
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const std::uint32_t engine_mode = get_arg(args::engine_mode);
    const std::uint32_t dest_coords = get_arg(args::dest_coords);  // packed (x << 16) | y

    // Fan-out round-robins PACKETS, and one packet per row is the right granularity here: both
    // sides already emit num_rows packets, so 32 rows always outnumber the 8 channels and
    // sub-splitting only multiplies issue cost. MAX_BYTES_IN_PACKET must be programmed
    // explicitly whatever the value -- CMDBUF_RESET zeroes it to its rdl default of 0, which
    // means "never split", and every channel knob then goes silently inert.
    std::uint32_t num_channels = get_arg(args::num_channels);
    if (num_channels < 1) {
        num_channels = 1;
    } else if (num_channels > CMDBUF_NUM_IDMA_VCS) {
        num_channels = CMDBUF_NUM_IDMA_VCS;
    }
    const bool fan_out = num_channels > 1;
    const bool use_idma = engine_mode == ENGINE_IDMA_PER_ROW;

    const std::uint32_t noc_x = dest_coords >> 16;
    const std::uint32_t noc_y = dest_coords & 0xFFFF;

    Noc noc(noc_index);

    // Wait for stage 1's whole block, then take its base. The DFB is pushed exactly once and
    // never wraps, so the rows are contiguous from the read pointer.
    DataflowBuffer pad(dfb::pad);
    pad.wait_front(num_rows);
    const std::uint32_t src_base = l1_phys(pad.get_read_ptr());

    if (use_idma) {
        reset_cmdbuf_0();
        idma_setup_as_copy_cmdbuf_0(/*wrapping_en=*/false);
        setup_ongoing_cmdbuf_0(
            /*src_addr_inc_en=*/false,  // the addrgen supplies both addresses
            /*dest_addr_inc_en=*/false,
            /*trid_inc_en=*/false,
            /*req_vc_inc_en=*/fan_out,  // round-robin the per-row packets across the VCs
            /*resp_vc_inc_en=*/false);
        setup_wrapping_vcs_cmdbuf_0(
            /*wr=*/true,
            /*req_start_vc=*/CMDBUF_FIRST_IDMA_VC,
            /*req_end_vc=*/CMDBUF_FIRST_IDMA_VC + (fan_out ? num_channels - 1 : 0));
        setup_max_bytes_in_packet_cmdbuf_0(out_row_bytes);
        setup_trids_cmdbuf_0(CMDBUF_DEF_TRID);
        set_len_cmdbuf_0(out_row_bytes);

        compact_idma_per_row(src_base, dst_addr, num_rows, pad_row_bytes, out_row_bytes);

        // Every ack is in: safe to hand cmdbuf 0 back to the NoC write path.
        init_wr_cmd_buf(noc_local_xy());
    } else {
        // Set the read state once so the loop pays only for the per-row issue -- the cheapest
        // form of the workaround, which is what makes it a fair baseline.
        UnicastEndpoint ep;
        noc.set_async_read_state(ep, out_row_bytes, {.noc_x = noc_x, .noc_y = noc_y, .addr = src_base});
        compact_noc_per_row(
            noc, noc_x, noc_y, src_base, dst_addr, out_row_bytes, num_rows, pad_row_bytes, out_row_bytes);
    }

    pad.pop_front(num_rows);
}
