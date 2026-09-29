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
//   0 IdmaPerRow    one iDMA transaction per row, addresses from the address generator,
//                   fanned out over `num_channels` backend VCs, ONE drain at the end.
//                   Issue-bound at ~8.7 cyc/row. This is the engine being proposed.
//   1 NocPerRow     one stateful NOC read per row from this core's own L1. The CURRENT
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
#include "narrow_row_engine_mode.hpp"                      // EngineMode, shared with the host

using namespace overlay;

using narrow_row::CHANNELS_ALL;
using narrow_row::EngineMode;

// This is what keeps the host's CHANNELS_MAX honest: it is the only place both it and the
// real constant are visible at once.
static_assert(
    narrow_row::CHANNELS_MAX == CMDBUF_NUM_IDMA_VCS,
    "CHANNELS_MAX no longer mirrors CMDBUF_NUM_IDMA_VCS; update the shared header");

namespace {

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
    // FULL reset, not reset_counters_: that variant keeps "base addresses, sizes and strides
    // intact", which includes the outer-loop, face-size and banking registers this kernel
    // never programs. The addrgen and im2col tests in this same binary DO program them on
    // addrgen_0 of this same logical core, so a counters-only reset inherits their state and
    // generates wrong addresses -- invisibly, because it only shows up when the whole suite
    // runs rather than under a --gtest_filter. Every other addrgen user in the tree resets
    // fully.
    reset_addrgen_0();
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

// Length and NoC coordinates are STICKY in the read state set by the caller: the one-packet
// path writes only the low address word. So this takes neither -- passing them per call would
// look like it configured something it does not, and an edit to those arguments alone would
// silently have no effect. The coordinate fields below are required by the endpoint struct
// and are ignored by this path.
inline void compact_noc_per_row(
    const Noc& noc,
    std::uint32_t src_base,
    std::uint32_t dst_base,
    std::uint32_t num_rows,
    std::uint32_t src_stride,
    std::uint32_t dst_stride) {
    UnicastEndpoint ep;
    for (std::uint32_t r = 0; r < num_rows; r++) {
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            ep, ep, 0, {.addr = src_base + r * src_stride}, {.addr = dst_base + r * dst_stride});
    }
    noc.async_read_barrier();
}

}  // namespace

void kernel_main() {
    const std::uint32_t dst_addr = get_arg(args::dst_addr);            // dense output, L1
    const std::uint32_t pad_row_bytes = get_arg(args::pad_row_bytes);  // ct_dim * 32 * datum_bytes
    const std::uint32_t out_row_bytes = get_arg(args::out_row_bytes);  // matrix_w * datum_bytes
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const auto engine_mode = static_cast<EngineMode>(get_arg(args::engine_mode));
    const std::uint32_t dest_coords = get_arg(args::dest_coords);  // packed (x << 16) | y

    // Fan-out round-robins PACKETS, and one packet per row is the right granularity here: both
    // sides already emit num_rows packets, so 32 rows always outnumber the 8 channels and
    // sub-splitting only multiplies issue cost. MAX_BYTES_IN_PACKET must be programmed
    // explicitly whatever the value -- CMDBUF_RESET zeroes it to its rdl default of 0, which
    // means "never split", and every channel knob then goes silently inert.
    // CHANNELS_ALL resolves here rather than on the host, because CMDBUF_NUM_IDMA_VCS is
    // visible only on this side. Anything above the VC count clamps down to it as well, so a
    // stale caller cannot ask for channels that do not exist.
    std::uint32_t num_channels = get_arg(args::num_channels);
    if (num_channels == CHANNELS_ALL || num_channels > CMDBUF_NUM_IDMA_VCS) {
        num_channels = CMDBUF_NUM_IDMA_VCS;
    }
    const bool use_idma = engine_mode == EngineMode::IdmaPerRow;
    // Anything else would have fallen through to the NOC branch and reported as a passing NOC
    // run, which would hide a runtime-arg plumbing mistake rather than surface it.
    ASSERT(use_idma || engine_mode == EngineMode::NocPerRow);

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
            /*req_vc_inc_en=*/true,  // round-robin the per-row packets across the VC window
            /*resp_vc_inc_en=*/false);
        // The response VC must be passed explicitly. setup_wrapping_vcs_ never reads its `wr`
        // argument -- it is not the selector it looks like -- and its defaulted resp_start_vc
        // and resp_end_vc would force RESP_VC 0, which sits INSIDE the request window this
        // same call programs (CMDBUF_FIRST_IDMA_VC is 0). Sharing a VC between requests and
        // their responses is the classic NoC deadlock configuration. CMDBUF_WR_RESP_VC is what
        // setup_vcs_cmdbuf_0(/*wr=*/true) would have selected.
        setup_wrapping_vcs_cmdbuf_0(
            /*wr=*/true,
            /*req_start_vc=*/CMDBUF_FIRST_IDMA_VC,
            /*req_end_vc=*/CMDBUF_FIRST_IDMA_VC + num_channels - 1,
            /*req_vc_offset=*/0,
            /*resp_start_vc=*/CMDBUF_WR_RESP_VC,
            /*resp_end_vc=*/CMDBUF_WR_RESP_VC);
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
        // <NOC_MAX_BURST_SIZE> selects the ONE-PACKET path. The default max_page_size is
        // NOC_MAX_BURST_SIZE + 1, which falls through to noc_async_read_with_state -- the
        // any-len path, which additionally writes the length register and computes a packet
        // count for the barrier. Neither path chunks in software (the overlay packetizes via
        // MAX_BYTES_IN_PACKET), so the gap is small: switching paths moved the measured
        // per-block engine delta by ~4 cyc out of ~470, which is inside the noise. Rows here
        // are at most 512 B against a 65536 B burst limit, so one-packet is legal, and it
        // makes this baseline the cheapest NOC read rather than merely a cheap one.
        Noc noc;  // defaults to noc_index
        noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            ep, out_row_bytes, {.noc_x = dest_coords >> 16, .noc_y = dest_coords & 0xFFFF, .addr = src_base});
        compact_noc_per_row(noc, src_base, dst_addr, num_rows, pad_row_bytes, out_row_bytes);
    }

    pad.pop_front(num_rows);
}
