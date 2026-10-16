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

// Owns cmdbuf 0 and addrgen 0 for the whole gather: configures both, issues every row, drains
// once, and hands the command buffer back. `num_channels` is the raw runtime arg -- it is
// resolved here because CMDBUF_NUM_IDMA_VCS is visible only on this side.
//
// The transfer length is `dst_stride`: the output is dense, so the bytes kept from each row and
// the destination stride are the same number. It is the SOURCE stride that is larger, and that
// difference is the compaction.
inline void compact_idma_per_row(
    std::uint32_t src_base,
    std::uint32_t dst_base,
    std::uint32_t num_rows,
    std::uint32_t src_stride,
    std::uint32_t dst_stride,
    std::uint32_t num_channels) {
    // CHANNELS_ALL and anything above the VC count both mean "every VC", so a stale caller
    // cannot ask for channels that do not exist.
    if (num_channels == CHANNELS_ALL || num_channels > CMDBUF_NUM_IDMA_VCS) {
        num_channels = CMDBUF_NUM_IDMA_VCS;
    }

    // ---- command buffer: what ONE transfer is -------------------------------------------
    reset_cmdbuf_0();
    idma_setup_as_copy_cmdbuf_0(/*wrapping_en=*/false);
    // Everything AutoIncConfig does not name defaults to false; the addrgen supplies both
    // addresses, so only the request VC advances -- round-robining the per-row packets across
    // the VC window below.
    setup_ongoing_cmdbuf_0({.req_vc = true});
    // `resp` must be passed explicitly. Defaulted it is {0, 0, 0}, which forces RESP_VC 0 --
    // and that sits INSIDE the request window this same call programs, because
    // CMDBUF_FIRST_IDMA_VC is 0. Sharing a VC between requests and their responses is the
    // classic NoC deadlock configuration.
    setup_wrapping_vcs_cmdbuf_0(
        /*req=*/{.start = CMDBUF_FIRST_IDMA_VC, .end = CMDBUF_FIRST_IDMA_VC + num_channels - 1},
        /*resp=*/{.start = CMDBUF_WR_RESP_VC, .end = CMDBUF_WR_RESP_VC});
    // Fan-out round-robins PACKETS, and one packet per row is the right granularity: 32 rows
    // already outnumber the 8 channels, and splitting further only multiplies per-packet cost
    // (measured: ~5 cyc per extra packet, up to 12x slower at 16 B packets). This must be
    // programmed explicitly whatever the value -- CMDBUF_RESET zeroes it to its rdl default of
    // 0, which means "never split", and every channel knob then goes silently inert.
    setup_max_bytes_in_packet_cmdbuf_0(dst_stride);
    setup_trids_cmdbuf_0(CMDBUF_DEF_TRID);
    set_len_cmdbuf_0(dst_stride);

    // ---- address generator: WHERE each transfer goes ------------------------------------
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

    // ---- issue every row, then drain ONCE ------------------------------------------------
    // Draining inside the loop would serialise the transfers and turn a throughput path into a
    // latency one; issuing first also means the RISC has registered all `num_rows` outstanding
    // before it polls, so the ack counter cannot read zero early.
    for (std::uint64_t r = 0; r < num_rows; r++) {
        push_both_addrgen_0();  // hand the next src/dst pair to the cmdbuf
        issue_cmdbuf_0();
    }
    while (!idma_acked_cmdbuf_0()) {
    }

    // Every ack is in: safe to hand cmdbuf 0 back to the NoC write path.
    init_wr_cmd_buf(noc_local_xy());
}

// Owns the read state for the whole gather, mirroring compact_idma_per_row: configure, loop,
// barrier. As there, the transfer length is `dst_stride` -- the dense output row.
inline void compact_noc_per_row(
    std::uint32_t src_base,
    std::uint32_t dst_base,
    std::uint32_t num_rows,
    std::uint32_t src_stride,
    std::uint32_t dst_stride,
    std::uint32_t dest_coords) {
    UnicastEndpoint ep;
    Noc noc;  // defaults to noc_index

    // Set the read state ONCE so the loop pays only for the per-row issue -- the cheapest form
    // of the workaround, which is what makes it a fair baseline. Length and NoC coordinates are
    // sticky in that state, so the loop below writes only the addresses.
    //
    // <NOC_MAX_BURST_SIZE> selects the ONE-PACKET path. The default max_page_size is
    // NOC_MAX_BURST_SIZE + 1, which falls through to noc_async_read_with_state -- the any-len
    // path, which additionally writes the length register and computes a packet count for the
    // barrier. Neither path chunks in software (the overlay packetizes via
    // MAX_BYTES_IN_PACKET), so the gap is small: switching paths moved the measured per-block
    // engine delta by ~4 cyc out of ~470, which is inside the noise. Rows here are at most
    // 512 B against a 65536 B burst limit, so one-packet is legal, and it makes this baseline
    // the cheapest NOC read rather than merely a cheap one.
    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        ep, dst_stride, {.noc_x = dest_coords >> 16, .noc_y = dest_coords & 0xFFFF, .addr = src_base});

    for (std::uint64_t r = 0; r < num_rows; r++) {
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            ep,
            ep,
            0,
            // The addresses are 32-bit L1 offsets; r is 64-bit, so the sums must be narrowed
            // back explicitly or -Werror=narrowing rejects the braced initializer.
            {.addr = static_cast<std::uint32_t>(src_base + r * src_stride)},
            {.addr = static_cast<std::uint32_t>(dst_base + r * dst_stride)});
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

    const std::uint32_t num_channels = get_arg(args::num_channels);  // resolved by the engine
    const bool use_idma = engine_mode == EngineMode::IdmaPerRow;
    // Anything else would have fallen through to the NOC branch and reported as a passing NOC
    // run, which would hide a runtime-arg plumbing mistake rather than surface it.
    ASSERT(use_idma || engine_mode == EngineMode::NocPerRow);

    // Wait for stage 1's whole block, then take its base. The DFB is pushed exactly once and
    // never wraps, so the rows are contiguous from the read pointer.
    DataflowBuffer pad(dfb::pad);
    pad.wait_front(num_rows);
    const std::uint32_t src_base = l1_phys(pad.get_read_ptr());

    // Each engine owns its hardware end to end -- command buffer or read state, the per-row
    // loop, and the drain -- so this is only the choice between them.
    if (use_idma) {
        compact_idma_per_row(src_base, dst_addr, num_rows, pad_row_bytes, out_row_bytes, num_channels);
    } else {
        compact_noc_per_row(src_base, dst_addr, num_rows, pad_row_bytes, out_row_bytes, dest_coords);
    }

    pad.pop_front(num_rows);
}
