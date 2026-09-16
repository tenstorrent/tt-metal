// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The send leg: a Tensix core pushes a payload into its own arena in pinned host RAM, then
// arms its own control word to tell the host the bytes are there.
//
// THE WHOLE PROTOCOL IN FOUR WRITES
//   1. payload  -> my TX arena in host RAM  (chunked posted PCIe writes)
//   2. operands -> my data registers        (one 8 B posted write each)
//   3. fence
//   4. control  -> my TX control register   (ONE indivisible 8 B posted write)
//
// Step 4 is the commit: opcode, which registers hold the operands, how many, and the sequence
// that says it is new all travel in that single 8-byte store, so the host can never see a
// trigger pointing at operands that have not landed.
//
// THAT ORDERING IS SOUND, NOT HOPEFUL. All four are posted PCIe writes from the same source
// (this core's NOC port) to the same endpoint (the PCIe tile), and PCIe producer-consumer
// ordering completes them in order -- so a host that sees the control word is guaranteed the
// payload behind it is in memory. The barrier before step 4 is what stops the NOC reordering
// them before they reach the tile; the PCIe guarantee starts there, not here. Without it the
// host reads a page the write has not finished filling, which looks like torn data.
//
// A CORE IS NEVER HANDED ITS OWN INDEX -- it computes it from get_absolute_logical_x()/y() and
// the grid width, so no caller argument can make this core write into another core's bank.

#include <stdint.h>

#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"

// Spelled from the repo root, not relatively: a relative path compiles only in the layout it
// was written for, and the failure lands in the JIT build at run time -- after the device is
// open and the transport connected -- rather than in the host build where it is cheap to see.
#include <tt-metalium/experimental/sockets/internal/host_uva.hpp>
#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>

namespace {

// The Tensix wall clock. Two 32-bit debug registers; read LOW first -- reading
// RISCV_DEBUG_REG_WALL_CLOCK_L latches the high half for readback, so the other order can
// pair a new low with a stale high and produce a timestamp that jumps backwards across a
// 32-bit rollover.
inline uint64_t wall_clock() {
    volatile uint32_t tt_reg_ptr* lo = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint32_t tt_reg_ptr* hi = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint32_t l = lo[0];
    const uint32_t h = hi[0];
    return static_cast<uint64_t>(l) | (static_cast<uint64_t>(h) << 32);
}

// A single NOC transaction has a length limit and EXCEEDING IT DOES NOT FAIL -- it writes
// nothing. Measured: pages up to 16384 push correctly, 32768 hangs the host forever waiting for
// bytes never sent, with no error on either side. Worst shape a limit can have, so the chunk
// loop is unconditional rather than a cap someone must remember to respect.
constexpr uint32_t kMaxNocWrite = 8192;

// Push `bytes` from L1 `src` to host-region offset `dst_off`. Posted and unfenced: the
// caller fences once at the end rather than per chunk, because a barrier per 8 KiB would
// turn a streaming push into a round trip per chunk.
inline void push_to_host(
    uint32_t src, uint64_t io_base, uint32_t pcie_xy_enc, uint64_t dst_off, uint32_t bytes) {
    uint64_t dst = io_base + dst_off;
    uint32_t remaining = bytes;
    while (remaining > 0) {
        const uint32_t chunk = remaining < kMaxNocWrite ? remaining : kMaxNocWrite;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            noc_index, src, pcie_xy_enc, dst, chunk, 1);
        src += chunk;
        dst += chunk;
        remaining -= chunk;
    }
}

// EVERY WRITE NEEDS ITS OWN STAGING SLOT. These are async posted writes -- the NOC reads the
// source after the call returns -- so sharing one address means later stores clobber the word
// before the earlier transfer has read it.
//
// SLOTS ARE 16 B APART, not 8: the NOC requires source and destination to agree in bits [3:0]
// for a narrow transfer or the data silently lands at the wrong offset (blackhole/noc/noc.h).
// The destination is a 64 B-aligned register, so the source must match.
constexpr uint32_t kStageSlotBytes = 16;

// THE LANDING PROBE. noc_async_write_barrier() is acknowledged by the PCIe TILE, so it means
// "accepted for transmission", not "in host memory" -- at small payloads the push fits in the
// tile's credits and the fence returns before a byte crosses the link (16 KiB on 110 cores
// measured 3.191 us, implying 565 GB/s on a ~15 GB/s link). PCIe forbids a read completion
// passing prior posted writes, so a read that comes back proves they landed. 16 bytes: the size
// is irrelevant, and 16 keeps source and destination agreeing in bits [3:0].
constexpr uint32_t kLandingProbeBytes = 16;

// Reads `size` from host offset `src_pcie` into L1. Same shape as the pull kernel's
// noc_read_page_chunked (test_kernel_pull.cpp:27) -- the 4-argument noc_read_with_state is the
// only read form that takes a 64-bit host offset, as pinned_memory.hpp's NocAddr says.
inline void read_from_host(uint32_t pcie_xy_enc, uint64_t src_pcie, uint32_t dst_l1, uint32_t size) {
    noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
        NOC_INDEX, pcie_xy_enc, src_pcie, dst_l1, size);
}

inline void write_reg64(
    uint32_t stage_addr, uint32_t slot, uint64_t value, uint64_t io_base, uint32_t pcie_xy_enc,
    uint64_t reg_off) {
    const uint32_t src = stage_addr + slot * kStageSlotBytes;
    *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(src) = value;
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        noc_index, src, pcie_xy_enc, io_base + reg_off, sizeof(uint64_t), 1);
}

}  // namespace

void kernel_main() {
    // --- The region, as the device sees it ---------------------------------
    constexpr uint32_t pcie_xy_enc = get_compile_time_arg_val(0);
    constexpr uint32_t io_base_lo = get_compile_time_arg_val(1);
    constexpr uint32_t io_base_hi = get_compile_time_arg_val(2);
    constexpr uint32_t grid_width = get_compile_time_arg_val(3);

    // --- L1 staging --------------------------------------------------------
    constexpr uint32_t payload_addr = get_compile_time_arg_val(4);  // the bytes to send
    constexpr uint32_t stage_addr = get_compile_time_arg_val(5);    // 8 B scratch for register writes
    constexpr uint32_t signal_addr = get_compile_time_arg_val(6);   // rdma_signal: bytes arrived for me

    // --- What to send ------------------------------------------------------
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t iterations = get_compile_time_arg_val(8);
    constexpr uint32_t opcode = get_compile_time_arg_val(9);
    constexpr uint32_t flags = get_compile_time_arg_val(10);
    constexpr uint32_t await_completion = get_compile_time_arg_val(11);
    constexpr uint32_t completion_addr = get_compile_time_arg_val(12);  // rdma_completion: my request retired

    // Landing verification, OFF BY DEFAULT: the probe puts a full PCIe round trip on every
    // message's critical path, so a run with it on measures a slower pipeline. Stage 1 is
    // honest with it, but that run's throughput must not be quoted.
    constexpr uint32_t verify_landing = get_compile_time_arg_val(13);
    constexpr uint32_t landing_addr = get_compile_time_arg_val(14);  // 16 B scratch for the probe

    // The destination UVA's selector. Runtime, not compile-time: it is DATA -- which core
    // on which host the bytes are for -- and unlike this core's own identity it is
    // legitimately supplied from outside. Splitting it this way is the line between "who
    // am I" (derived, unforgeable) and "where is this going" (given).
    const uint32_t dest_selector = get_arg_val<uint32_t>(0);
    const uint32_t dest_offset = get_arg_val<uint32_t>(1);

    const uint64_t io_base = (static_cast<uint64_t>(io_base_hi) << 32) | static_cast<uint64_t>(io_base_lo);

    // MY index, MY bank, MY arena -- all derived, none supplied.
    const uint32_t me = tt::tt_metal::experimental::core_index(get_absolute_logical_x(), get_absolute_logical_y(), grid_width);
    const uint64_t my_tx_arena = tt::tt_metal::experimental::tx_arena_offset(me);
    const uint64_t my_ctrl = tt::tt_metal::experimental::reg_offset(me, tt::tt_metal::experimental::kCtrlTx);
    const uint64_t my_reg0 = tt::tt_metal::experimental::reg_offset(me, 0);
    const uint64_t my_reg1 = tt::tt_metal::experimental::reg_offset(me, 1);
    const uint64_t my_reg2 = tt::tt_metal::experimental::reg_offset(me, 2);
    const uint64_t my_reg3 = tt::tt_metal::experimental::reg_offset(me, 3);

    // The destination as one forwarded 64-bit word, carried unmodified to the far host, so its
    // meaning must not depend on who holds it. Computed once: the destination is fixed for the
    // run, so this measures the transport and not the addressing.
    const uint64_t dest_uva =
        tt::tt_metal::experimental::uva_encode(tt::tt_metal::experimental::kRegionT6, dest_selector, 0, dest_offset);

    // TWO DOORBELLS. `completion`: the local host consumed this core's control word, so the
    // register may re-arm. `signal`: bytes from somebody else landed in L1. Pacing on `signal`
    // deadlocks -- it makes this core's next request depend on a remote peer sending to it.
    volatile tt_l1_ptr uint32_t* completion = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_addr);
    volatile tt_l1_ptr uint32_t* signal = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_addr);
    volatile tt_l1_ptr uint32_t* payload_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_addr);

    noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);

    *completion = 0;
    *signal = 0;

    // A real payload pattern (0x40 + core), because with one shared L1 buffer the receiving
    // side is the witness and does check it. Written before the loop, not per iteration: it is
    // a constant, and re-filling 1 MiB each time would put a memset inside the timed region.
    if (iterations != 0) {
        const uint32_t b = 0x40u + (me & 0x1Fu);
        const uint32_t w = b | (b << 8) | (b << 16) | (b << 24);
        // From word 1: word 0 is the per-iteration stamp, which must keep varying or a stale
        // arena cannot be told from a fresh push.
        for (uint32_t k = 1; k < payload_bytes / sizeof(uint32_t); ++k) {
            payload_word[k] = w;
        }
    }

    for (uint32_t i = 0; i < iterations; ++i) {
        // STAGE 1 STARTS HERE, measured in this core's own cycles and published with the
        // message. Nothing subtracts a host timestamp from a device one; the host needs only a
        // cycles->ns rate.
        const uint64_t t_push0 = wall_clock();
        // Stamp the iteration: without it every message is byte-identical and a host re-reading
        // a STALE arena cannot be told from one that got a fresh push.
        payload_word[0] = i;
        // The selector this message names, so the receiver can check it landed where the address
        // said -- the payload pattern identifies the SENDER and cannot answer that. Written
        // before push_to_host, which is the call that moves these bytes.
        payload_word[tt::tt_metal::experimental::kPayloadDestOffset / sizeof(uint32_t)] = dest_selector;

        push_to_host(payload_addr, io_base, pcie_xy_enc, my_tx_arena, payload_bytes);

        // Operands. Register 0 is the destination UVA, 1 the length, 2 the elapsed
        // accumulator, 3 this core's index so a reply knows where to come back to. Length
        // is an OPERAND rather than a control-word field because with n-argument messages
        // there is no honest place for it in 32 bits -- see host_uva_layout.hpp.
        write_reg64(stage_addr, 0, dest_uva, io_base, pcie_xy_enc, my_reg0);
        // THE LENGTH REGISTER IS SKIPPED IN THE IMMEDIATE FORM, and that is the whole
        // saving: kOpRdmaWriteImm carries the byte count in the control word itself, so an
        // 8-byte store is one operand write and a trigger rather than two and a trigger.
        // The register is not written AND not read -- see host_uva_layout.hpp on why the
        // opcode fixing the operand layout is the point of having two encodings.
        if constexpr (opcode != tt::tt_metal::experimental::kOpRdmaWriteImm) {
            write_reg64(stage_addr, 1, static_cast<uint64_t>(payload_bytes), io_base, pcie_xy_enc, my_reg1);
        }
        write_reg64(stage_addr, 3, static_cast<uint64_t>(me), io_base, pcie_xy_enc, my_reg3);

        // THE FENCE THAT MAKES STEP 4 A COMMIT. Everything above must be at the PCIe tile
        // before the control word joins the queue behind it.
        noc_async_write_barrier();

        // THE FENCE IS NOT LANDING. The barrier above is acknowledged by the PCIe tile, so it
        // proves the writes were accepted, not that they crossed. The probe below closes that
        // gap, measured separately because it is the instrument's cost, not the transfer's.
        const uint64_t t_fenced = wall_clock();
        uint64_t visibility = 0;
        if constexpr (verify_landing != 0) {
            // The TAIL of what we just pushed. The last bytes are the last to be accepted, so a
            // read that returns them has flushed everything ahead of it on this path.
            read_from_host(pcie_xy_enc, io_base + my_tx_arena + payload_bytes - kLandingProbeBytes,
                           landing_addr, kLandingProbeBytes);
            noc_async_read_barrier();
            visibility = wall_clock() - t_fenced;
        }

        // After the fence, before the trigger: the trigger is what makes this visible to the
        // host. BOTH HALVES IN ONE REGISTER per kFlagElapsedSplit -- total low, the probe's own
        // cost high -- reported as t6->host and diag:d2h-visibility so neither hides the other.
        // With the probe off the high half is zero.
        write_reg64(stage_addr, 2,
                    tt::tt_metal::experimental::elapsed_pack((t_fenced - t_push0) + visibility, visibility),
                    io_base, pcie_xy_enc, my_reg2);
        noc_async_write_barrier();

        // THE COMMIT. Three things ride in this one word:
        //
        //   sequence  -- distinguishes a re-armed word from the one already serviced. Wraps at
        //               4096, which is safe unless the host falls 4095 messages behind, by
        //               which point the arena has been overwritten anyway.
        //   count = 4 -- dest UVA, length, elapsed, origin core.
        //   kFlagCycles -- the elapsed field is in Tensix CYCLES and needs the host's rate
        //               applied. A cycle count read as nanoseconds is wrong by roughly the
        //               clock rate and still looks like a plausible duration, so it is flagged
        //               rather than inferred.
        //
        // TWO ENCODINGS, CHOSEN AT COMPILE TIME. The immediate form puts the byte count in bits
        // [17:8] where base/count live for every other opcode, so one call cannot build both --
        // a `base` passed there would be read as part of a length. Its operand layout is fixed
        // BY THE OPCODE instead: register 0 is the destination UVA, 2 the elapsed accumulator,
        // 3 this core's index, and register 1 is not written. See ctrl_encode_imm().
        const uint64_t ctrl =
            (opcode == tt::tt_metal::experimental::kOpRdmaWriteImm)
                ? tt::tt_metal::experimental::ctrl_encode_imm(
                      payload_bytes, flags | tt::tt_metal::experimental::kFlagCycles, i % tt::tt_metal::experimental::kCtrlSeqModulus)
                : tt::tt_metal::experimental::ctrl_encode(
                      opcode, /*base=*/0, /*count=*/4, flags | tt::tt_metal::experimental::kFlagCycles,
                      i % tt::tt_metal::experimental::kCtrlSeqModulus);
        write_reg64(stage_addr, 4, ctrl, io_base, pcie_xy_enc, my_ctrl);
        noc_async_write_barrier();

        if (await_completion) {
            // rdma_completion, not rdma_signal: only the former says MY slot is reusable.
            // Spinning on an L1 word the host writes, not a host-memory word this core reads --
            // a device read of host RAM is a non-posted round trip inside the measured loop.
            while (*completion != (i + 1)) {
                invalidate_l1_cache();
            }
        }
    }
}
