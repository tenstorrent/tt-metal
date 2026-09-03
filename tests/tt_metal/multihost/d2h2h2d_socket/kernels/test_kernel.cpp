// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The send leg: a Tensix core pushing a payload into its own arena in pinned host RAM,
// then arming its own control word to tell the host the bytes are there.
//
// THE WHOLE PROTOCOL IN FOUR WRITES
//
//   1. payload  -> my TX arena in host RAM   (chunked posted PCIe writes)
//   2. operands -> my data registers          (one 8 B posted write each)
//   3. fence
//   4. control  -> my TX control register     (ONE indivisible 8 B posted write)
//
// Step 4 is the commit. Everything it refers to -- the opcode, which registers hold the
// operands, how many there are, and the sequence number that says it is new -- travels in
// that single 8-byte store, so the host can never see a trigger that points at operands
// which have not landed.
//
// WHY THAT ORDERING IS SOUND AND NOT JUST HOPEFUL. All four are posted PCIe writes from
// the SAME source (this core's NOC port) to the SAME endpoint (the PCIe tile). PCIe
// producer-consumer ordering says posted writes on that path complete in order, so a
// host that observes the control word is guaranteed the payload and operands behind it
// are already in memory. The explicit barrier before step 4 is what stops the NOC from
// reordering them before they reach the PCIe tile -- the PCIe guarantee starts there,
// not here. d2h_push_bench learned the same lesson: without the fence the host reads a
// page the write has not finished filling, and it looks like torn data rather than a
// missing barrier.
//
// A CORE IS NEVER HANDED ITS OWN INDEX. It computes it from get_absolute_logical_x()/y()
// and the grid width, so there is no argument a caller could get wrong that would make
// this core write into another core's bank or arena. That is the same structural
// property rdma_reg_layout.hpp relies on, and it is why none of the offset helpers take
// a "which core" parameter from the wire.

#include <stdint.h>

#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"

// Spelled from the repo root, as the host side spells it, NOT relatively. Out of tree the kernel
// sat at finalized/test/kernels/, so "../../" was finalized/ where these two headers lived; in
// tree they are in tt_metal/distributed/ and the kernel is three directories away under tests/.
// A relative path here compiles only in the layout it was written for, and the failure lands in
// the JIT build at run time -- after the device is open and the transport connected -- rather
// than in the host build where it would be cheap to see.
#include "tt_metal/distributed/host_uva.hpp"
#include "tt_metal/distributed/host_uva_layout.hpp"

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

// A single NOC transaction has a length limit, and EXCEEDING IT DOES NOT FAIL -- it
// writes nothing. Measured in d2h_push_bench on this tree: page sizes up to 16384 push
// correctly and 32768 hangs the host forever waiting for bytes that were never sent,
// with no error on either side. That is the worst shape a limit can have, so the chunk
// loop is unconditional rather than a cap someone has to remember to respect.
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

// One 64-bit register. Staged through L1 because the NOC moves memory to memory; there is
// no store-immediate-to-host path.
//
// EVERY WRITE NEEDS ITS OWN STAGING SLOT. These are ASYNC POSTED writes: the NOC reads the
// source some time after the call returns. Staging four operands through one address means
// the second store clobbers the word before the first transfer has read it, and all four
// registers arrive holding whichever value landed last. Measured on the first real device
// run: the host rejected the message with "TX message length out of range" because
// register 1 held the destination UVA instead of the length.
//
// SLOTS ARE 16 B APART, not 8. The NOC requires source and destination to agree in bits
// [3:0] for a narrow transfer, and "if they disagree the one from the destination address
// is assumed" -- the data silently lands at the wrong offset (blackhole/noc/noc.h). The
// destination is a 64 B-aligned register, so bits [3:0] are zero and the source must match.
constexpr uint32_t kStageSlotBytes = 16;

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

    // The destination UVA's selector. Runtime, not compile-time: it is DATA -- which core
    // on which host the bytes are for -- and unlike this core's own identity it is
    // legitimately supplied from outside. Splitting it this way is the line between "who
    // am I" (derived, unforgeable) and "where is this going" (given).
    const uint32_t dest_selector = get_arg_val<uint32_t>(0);
    const uint32_t dest_offset = get_arg_val<uint32_t>(1);

    // RANDOM DESTINATIONS -- the GUPS shape. Zero host_num keeps the fixed destination
    // above, so a driver that does not set these behaves exactly as it always did.
    //
    // WHY THE KERNEL IS ALLOWED THE TOPOLOGY HERE. Everywhere else this file derives "who
    // am I" and is GIVEN "where is this going", and that split is deliberate. It survives:
    // these three come from the HOST, so the kernel and its host cannot disagree about the
    // stride the way two HOSTS could -- which is the trap host_uva.hpp warns about. What
    // the kernel gains is the ability to CHOOSE among destinations, not to invent one.
    //
    // AND THIS IS THE FIRST TEST WHERE THE ADDRESS ACTUALLY VARIES. The fixed-destination
    // modes compute dest_uva once, outside the loop, so they measure the transport and
    // never the addressing -- identical numbers would come from a hardcoded target. Here a
    // fresh (host, chip, core) is resolved per message, which is the claim the UVA design
    // rests on.
    const uint32_t rnd_host_num = get_arg_val<uint32_t>(2);
    const uint32_t rnd_chips_per_host = get_arg_val<uint32_t>(3);
    const uint32_t rnd_cores = get_arg_val<uint32_t>(4);
    const uint32_t rnd_seed = get_arg_val<uint32_t>(5);
    const uint32_t my_host = get_arg_val<uint32_t>(6);
    const uint32_t my_chip = get_arg_val<uint32_t>(7);
    const bool random_dest = (rnd_host_num != 0);

    // xorshift32. Small, no multiply, and its period is far beyond any run here. Seeded per
    // CORE by the host so two cores do not walk the same address stream -- identical streams
    // would make every core hammer the same destination in lockstep, which measures
    // contention rather than random access.
    uint32_t rnd_state = rnd_seed ? rnd_seed : 0x9E3779B9u;

    const uint64_t io_base = (static_cast<uint64_t>(io_base_hi) << 32) | static_cast<uint64_t>(io_base_lo);

    // MY index, MY bank, MY arena -- all derived, none supplied.
    const uint32_t me = tt::tt_metal::experimental::core_index(get_absolute_logical_x(), get_absolute_logical_y(), grid_width);
    const uint64_t my_tx_arena = tt::tt_metal::experimental::tx_arena_offset(me);
    const uint64_t my_ctrl = tt::tt_metal::experimental::reg_offset(me, tt::tt_metal::experimental::kCtrlTx);
    const uint64_t my_reg0 = tt::tt_metal::experimental::reg_offset(me, 0);
    const uint64_t my_reg1 = tt::tt_metal::experimental::reg_offset(me, 1);
    const uint64_t my_reg2 = tt::tt_metal::experimental::reg_offset(me, 2);
    const uint64_t my_reg3 = tt::tt_metal::experimental::reg_offset(me, 3);

    // The destination, as one forwarded 64-bit word. Region kRegionT6 with a positional
    // (host, chip, core) selector: this word is carried unmodified all the way to the
    // far host, so its meaning must not depend on who is holding it.
    // The FIXED destination, still computed once: it is what a non-random run uses, and it
    // is the fallback if a random pick somehow lands on this core.
    const uint64_t fixed_dest_uva =
        tt::tt_metal::experimental::uva_encode(tt::tt_metal::experimental::kRegionT6, dest_selector, 0, dest_offset);

    // TWO DOORBELLS. `completion` says the LOCAL host has consumed this core's control
    // word, so the register is free to re-arm. `signal` says bytes from somebody else
    // landed in L1. Pacing on the wrong one is a deadlock: waiting for `signal` makes this
    // core's next request depend on a REMOTE peer sending to it, which combined with the
    // sender-side credit forms two interlocking depth-1 ladders. Measured stalling at 144
    // of 160 on a two-host run with nothing lost -- just nobody able to proceed.
    volatile tt_l1_ptr uint32_t* completion = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_addr);
    volatile tt_l1_ptr uint32_t* signal = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_addr);
    volatile tt_l1_ptr uint32_t* payload_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_addr);

    noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);

    *completion = 0;
    *signal = 0;

    // Until now this kernel stamped only word 0 (the iteration) and left the rest of the
    // payload as whatever L1 happened to hold. That was invisible because verify_delivery()
    // never ran on the device path: every two-host run has routed_remote > 0 and skips
    // verification as unwitnessable, and the constant-byte probes it uses (0x40 + core at
    // offsets 4, bytes/2 and bytes-1) were only ever produced by the SELF-TEST's stand-in
    // producer. So the bytes a Tensix actually pushed were never checked -- only counted.
    //
    // With one shared L1 buffer (--symmetric) the receiving side is the witness and does run
    // that check, so the pattern has to be real. It is written before the loop rather than per
    // iteration: it is a constant, and re-filling 1 MiB every iteration would put a memset
    // inside the thing being timed.
    {
        const uint32_t b = 0x40u + (me & 0x1Fu);
        const uint32_t w = b | (b << 8) | (b << 16) | (b << 24);
        // From word 1: word 0 is the per-iteration stamp, which must keep varying or a stale
        // arena cannot be told from a fresh push.
        for (uint32_t k = 1; k < payload_bytes / sizeof(uint32_t); ++k) {
            payload_word[k] = w;
        }
    }

    for (uint32_t i = 0; i < iterations; ++i) {
        // STAGE 1 STARTS HERE. This core measures its OWN push -- payload, operands, the
        // fence and the trigger -- in its own cycles, and publishes the duration with the
        // message. Nothing subtracts a host timestamp from a device one anywhere in the
        // chain; the host only needs a cycles->ns RATE, which is one measured scalar
        // rather than a continuously drifting epoch offset.
        const uint64_t t_push0 = wall_clock();
        // STAMP THE ITERATION INTO THE PAYLOAD. Without it every message is
        // byte-identical, and a host that re-reads a STALE arena cannot be told from one
        // that received a fresh push -- the comparison passes either way. That is not
        // hypothetical here: the host is released by the control word, so a payload write
        // that has not landed would still let it proceed. A varying stamp is what makes
        // the ordering claim testable rather than assumed.
        // WHERE THIS MESSAGE IS AIMED, decided before the payload is stamped because the
        // stamp carries it and before push_to_host because that is what moves the bytes.
        uint32_t dest_selector_now = dest_selector;
        uint64_t dest_uva = fixed_dest_uva;
        if (random_dest) {
            // ROTATING HOST, RANDOM CORE -- and the host is a rotation rather than a draw
            // for a reason that is structural, not stylistic.
            //
            // A UNIFORM DRAW DEADLOCKS AT THREE HOSTS OR MORE. A destination core has ONE RX
            // control word and ONE arena slot, and the credit that paces re-arming is
            // accounted per ORIGIN. With two hosts that is safe by arithmetic: only one host
            // can target any given destination, so it has a single writer. At N >= 3 two
            // source hosts can pick the same (host, core) and overwrite each other's notice
            // AND payload, with neither sender aware. Measured exactly there: 2 hosts x 4
            // cores passes, 3 hosts x 1 core hangs with nothing delivered.
            //
            // An atomic does not rescue it -- CmpSwap could arbitrate the 8-byte control word,
            // but the collision is on a 16 KiB..512 KiB payload, and the winner would publish
            // a notice pointing at bytes the loser had already overwritten.
            //
            // (S + 1 + (i mod (N-1))) mod N is a PERMUTATION at every i: each host targets a
            // different host, so every destination has exactly one source. Collision-free by
            // construction, at any N, with no wire change. The destination CORE stays random
            // -- proven safe, since a core's writers are still a single host.
            //
            // NOT UNIFORMLY RANDOM, and the results must say so: it is an all-to-all rotation
            // that visits every (host, core) pair over a run rather than a GUPS draw. What it
            // tests is that the ADDRESS varies and resolves, which no fixed-destination mode
            // does at all.
            // BOTH FIELDS ROTATE, AND BOTH MUST. The host rotation alone is not enough:
            // with a RANDOM destination core, two of this host's own cores draw the same
            // core on the same peer and collide on its single RX slot and arena. That is
            // the identical failure the host draw had, one level down -- and it is why the
            // earlier 2-host run passed, because the destination core was then the SOURCE
            // core, an identity map that is collision-free by accident.
            //
            //   source (S, c)  ->  dest (S + 1 + i mod (N-1),  (c + i) mod C)
            //
            // Injective at every i: two sources share a destination only if they share both
            // S and c. So every destination has exactly ONE source at every step, which is
            // what the single RX control word and single arena slot require.
            //
            // Both fields still VARY across the run -- the host walks every peer, the core
            // walks every core -- which is the point. It is an all-to-all rotation, not a
            // uniform draw, and the results must say so.
            const uint32_t step = 1u + (i % (rnd_host_num - 1u));
            const uint32_t host = (my_host + step) % rnd_host_num;
            const uint32_t core = (me + i) % rnd_cores;
            const uint32_t sel =
                tt::tt_metal::experimental::t6_global_selector(host, my_chip, core, rnd_chips_per_host);
            dest_selector_now = sel;
            dest_uva = tt::tt_metal::experimental::uva_encode(tt::tt_metal::experimental::kRegionT6, sel, 0, dest_offset);
        }

        payload_word[0] = i;
        // THE SELECTOR THIS MESSAGE NAMES, stamped into the payload so the receiver can check
        // it landed where the address said. The memset pattern identifies the SENDER, so it
        // cannot answer that -- and while every mode sent core c to core c the difference was
        // invisible. See kPayloadDestOffset.
        //
        // Written BEFORE push_to_host, because that is the call that moves these bytes.
        payload_word[tt::tt_metal::experimental::kPayloadDestOffset / sizeof(uint32_t)] = dest_selector_now;

        push_to_host(payload_addr, io_base, pcie_xy_enc, my_tx_arena, payload_bytes);

        // Operands. Register 0 is the destination UVA, 1 the length, 2 the elapsed
        // accumulator, 3 this core's index so a reply knows where to come back to. Length
        // is an OPERAND rather than a control-word field because with n-argument messages
        // there is no honest place for it in 32 bits -- see host_uva_layout.hpp.
        // A FRESH DESTINATION PER MESSAGE when random_dest is armed. This is the only line
        // that makes the address vary, and the wire needed no change for it: register 0 was
        // already written every iteration -- it simply held a constant.
        //
        // REMOTE ONLY, by rerolling the host. A random draw legitimately hits this host 1/N
        // of the time and real GUPS counts those, but on this path a local destination is
        // not currently deliverable: under --symmetric the L1 buffer has one slot and a core
        // that is both source and destination gives it two writers (bug 9), and with H2D
        // ring aliasing armed the local memcpy is refused outright because it would bypass
        // the socket's producer bookkeeping. So the locality of this benchmark is REMOTE
        // ONLY and the results must say so rather than imply GUPS conformance.
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

        // The push is complete and fenced, so this is the honest end of stage 1. Written
        // AFTER the fence and before the trigger: the elapsed value must be visible to the
        // host, and the trigger is what makes it so.
        write_reg64(stage_addr, 2, wall_clock() - t_push0, io_base, pcie_xy_enc, my_reg2);
        noc_async_write_barrier();

        // The sequence number is what distinguishes a re-armed word from the one the host
        // already serviced. It wraps at 4096; the host compares against the last sequence
        // it saw for this bank, so a wrap is fine as long as the host is not more than
        // 4095 messages behind -- and if it were, the arena would have been overwritten
        // long before the counter mattered.
        // count = 4: dest UVA, length, elapsed, origin core. kFlagCycles tells the host
        // the elapsed field is in Tensix cycles and needs its rate applied -- a cycle
        // count read as nanoseconds is wrong by roughly the clock rate and still looks
        // like a plausible duration, so it is flagged rather than inferred.
        // TWO ENCODINGS, CHOSEN AT COMPILE TIME. The immediate form puts the byte count in
        // bits [17:8] where base/count live for every other opcode, so the two words cannot
        // be built by one call taking both -- a `base` passed here would be read as part of
        // a length. See ctrl_encode_imm() in host_uva_layout.hpp.
        //
        // The operand layout under the immediate form is fixed BY THE OPCODE rather than
        // described by base/count: register 0 is the destination UVA, 2 the elapsed
        // accumulator, 3 this core's index. Register 1 is not written.
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
            // Wait for rdma_completion -- MY request retired, this control register is
            // free. Not rdma_signal, which is about traffic somebody else sent me and has
            // no bearing on whether my own slot is reusable.
            //
            // Spinning on an L1 word the host writes over PCIe, not on a host-memory word
            // this core would read back: a device read of host RAM is a non-posted round
            // trip and would put a PCIe latency inside the loop being measured.
            while (*completion != (i + 1)) {
                invalidate_l1_cache();
            }
        }
    }
}
