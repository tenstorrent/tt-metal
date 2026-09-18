// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wire contract for the T6 -> HOST RDMA register file. host is the data mover -- the
// register file lives in pinned host memory and the host sweeps the banks, rather than the
// file living in one core's SRAM with that core sweeping. Three consequences, and they are why
// this is a separate program rather than a flag on the old one:
//
//   1. The bank sweep is not serial -- many pinned host threads walk it at once.
//   2. The T6 writes are POSTED PCIe writes, not NOC writes into on-chip SRAM.
//   3. Register file and payload share ONE pinned region, exposed in the MPI RMA window once.
//      Bytes a Tensix core pushed over PCIe are already in a registered MR when the host sends
//      them: no bounce buffer, no per-message registration, no intermediate copy.
//
// That last point is what a per-socket pinned FIFO cannot give: tt-metal's D2HSocket allocates
// its own and memcpys out on read(), which at ~110 cores is ~110 pinned allocations, ~110
// sender TLBs and a copy per record. PinnedMemory::Create() does not allocate -- it pins memory
// the CALLER owns, via PCIDevice::map_for_dma -> TENSTORRENT_IOCTL_PIN_PAGES on our own virtual
// address. The pages do not move and the VA stays ours, so the identical range also goes to
// MPI_Win_create(): two independent refcounted pins of the same pages, one for the TT device
// and one for the NIC.
//
#pragma once

#include <stdint.h>

namespace tt::tt_metal::experimental {

// ---------------------------------------------------------------------------
// The register bank
//
// 32 registers per core, one per 64 B cache line, split 30 data + 2 control.
// ---------------------------------------------------------------------------
constexpr uint32_t kRegisterBytes = 64;
constexpr uint32_t kDataRegisters = 30;
constexpr uint32_t kControlRegisters = 2;
constexpr uint32_t kRegistersPerBank = kDataRegisters + kControlRegisters;  // 32
constexpr uint32_t kBankBytes = kRegistersPerBank * kRegisterBytes;         // 2 KiB

constexpr uint32_t kCtrlTx = kDataRegisters + 0;  // 30: T6 -> host, the request
constexpr uint32_t kCtrlRx = kDataRegisters + 1;  // 31: host -> T6, RX SLOT 0 (see below)

// ---------------------------------------------------------------------------
// RECEIVE SLOT POOL: equal-sized slots per destination core, claimed by any sender.
// DEPTH IS kNumAliasRingSlots, which is 1 today -- see there for why, and for what a deeper
// ring would need first. The register file caps it at kRxNoticeSlots notice words per bank.
//
// Why a pool at all: a destination core had a single RX control word and a single arena, and
// the credit that frees them is accounted per SOURCE. Those coincide only while a destination
// has exactly ONE lifetime source. Vary the destination and the slot changes hands with no
// handshake -- the new sender's credit describes its OWN previous message, not whether this
// slot is free. Measured as a silent stall with no deadline to fire, at two hosts as soon as
// the destination CORE varied, long before three hosts made two source HOSTS collide.
//
// Equal slots rather than variable-size claims, so there is no allocator, no padding and
// nothing straddling the wrap.
//
constexpr uint32_t kPayloadStampOffset = 0;  // uint32 iteration
constexpr uint32_t kPayloadDestOffset = 4;   // uint32 destination selector
constexpr uint32_t kPayloadHeaderBytes = 8;

constexpr uint32_t kRxNoticeSlots = 8;

constexpr uint32_t kRxSlotBase = 8;  // registers 8..14 hold slots 1..7 (slot 0 is kCtrlRx)

constexpr uint32_t rx_slot_reg(uint32_t slot) { return slot == 0 ? kCtrlRx : (kRxSlotBase + slot - 1); }

// ---------------------------------------------------------------------------
// The control word
//
// [63:48] magic      16   0x484F -- rejects an unprovisioned/zeroed bank and rejects
//                              a legacy v1/v2 status word, which used magic 0x57A7
// [47:44] version     4   protocol version; a reader REFUSES a version it does not know
// [43:32] sequence   12   distinguishes a re-armed word from the one already serviced, and
//                              wraps at 4096. Each Tensix kernel counts its own TX bank, so TX
//                              is monotonic per bank; the host's RX notices come from a GLOBAL
//                              counter. The dedupe needs only that a re-arm changes the value
// [31:18] flags      14   unknown bits REJECT -- see below
// [17:13] count       5   number of data registers this message occupies, 1..30
// [12:8]  base        5   first data register index, 0..29
// [7:0]   opcode      8
//
// ---------------------------------------------------------------------------
constexpr uint64_t kCtrlMagic = 0x484Full;  // 'H','O' -- host-owned register file
constexpr uint32_t kCtrlMagicShift = 48;
constexpr uint64_t kCtrlMagicMask = 0xFFFFull;

// the gate ON the credit word's encoding. Nothing in that word says who wrote it, so this is
// what protects it: ctrl_validate() is exact-match, so builds at different versions service no
// control word in either direction and never exchange a credit to misread.
constexpr uint64_t kCtrlVersion = 4ull;
constexpr uint32_t kCtrlVersionShift = 44;
constexpr uint64_t kCtrlVersionMask = 0xFull;

constexpr uint32_t kCtrlSeqShift = 32;
constexpr uint64_t kCtrlSeqMask = 0xFFFull;  // 12 bits: wraps at 4096
constexpr uint32_t kCtrlSeqModulus = 4096;

constexpr uint32_t kCtrlFlagsShift = 18;
constexpr uint64_t kCtrlFlagsMask = 0x3FFFull;

constexpr uint32_t kCtrlCountShift = 13;
constexpr uint64_t kCtrlCountMask = 0x1Full;

constexpr uint32_t kCtrlBaseShift = 8;
constexpr uint64_t kCtrlBaseMask = 0x1Full;

constexpr uint32_t kCtrlOpcodeShift = 0;
constexpr uint64_t kCtrlOpcodeMask = 0xFFull;

enum CtrlOpcode : uint32_t {
    kOpNop = 0x00,      // armed but nothing to do; used to time the notice path alone
    kOpSendUva = 0x01,  // move bytes named by a UVA operand to the UVA's owner

    // 2 encodings of 1 operation, which is the sb/sh/sw/sd vs block-move split:
    //
    //   kOpRdmaWrite      length is an OPERAND REGISTER  -- reg[base+1], 64 bits, any size
    //   kOpRdmaWriteImm   length is an IMMEDIATE in the instruction -- see ctrl_imm()
    //
    // The immediate form spends no operand register and puts no length word on the wire,
    // which is the whole point of an 8-byte store: the opcode IS the width.
    kOpRdmaWrite = 0x03,
    kOpRdmaWriteImm = 0x04,
};

// Does this opcode name a store whose UVA offset must be honoured?
constexpr bool ctrl_op_is_store(uint32_t op) { return op == kOpRdmaWrite || op == kOpRdmaWriteImm; }
// Does this opcode carry its length as an immediate rather than in a register?
constexpr bool ctrl_op_has_imm(uint32_t op) { return op == kOpRdmaWriteImm; }

constexpr uint32_t kCtrlImmShift = kCtrlBaseShift;                        // 8
constexpr uint64_t kCtrlImmMask = (kCtrlCountMask << 5) | kCtrlBaseMask;  // 10 bits
constexpr uint32_t kCtrlImmMax = static_cast<uint32_t>(kCtrlImmMask);     // 1023

constexpr uint32_t ctrl_imm(uint64_t w) { return static_cast<uint32_t>((w >> kCtrlImmShift) & kCtrlImmMask); }

constexpr uint64_t ctrl_encode_imm(uint32_t length, uint64_t flags, uint32_t sequence);

constexpr uint64_t kFlagPullDelivery = 1ull << 0;  // far side: T6 pulls, rather than host pushing
constexpr uint64_t kFlagStamped = 1ull << 1;       // payload carries a per-message stamp to defeat stale-slot reads
constexpr uint64_t kFlagCycles = 1ull << 2;

// reserved, not live: post_notice lost its `reply` parameter and nothing
// sets this bit. Left in the definition AND in kFlagKnownMask below on purpose -- removing it
// from the mask would make a word carrying bit 3 decode as an UNKNOWN-flag protocol error
// rather than as a flag this build simply ignores, which is a wire-contract change dressed up
// as dead-code removal.
constexpr uint64_t kFlagReply = 1ull << 3;

constexpr uint64_t kFlagRemoteNotice = 1ull << 4;

constexpr uint64_t kFlagElapsedSplit = 1ull << 5;

constexpr uint64_t kFlagKnownMask =
    kFlagPullDelivery | kFlagStamped | kFlagCycles | kFlagReply | kFlagRemoteNotice | kFlagElapsedSplit;

// The two halves of register 2 under kFlagElapsedSplit. Shared by the kernel that writes them
// and the host that reads them, so the pair cannot drift the way "operand[2] >> 32" repeated in
// two files would.
constexpr uint64_t kElapsedFieldMask = 0xFFFFFFFFull;
constexpr uint64_t elapsed_pack(uint64_t total, uint64_t visibility) {
    return ((visibility > kElapsedFieldMask ? kElapsedFieldMask : visibility) << 32) |
           (total > kElapsedFieldMask ? kElapsedFieldMask : total);
}
constexpr uint64_t elapsed_total_of(uint64_t w) { return w & kElapsedFieldMask; }
constexpr uint64_t elapsed_visibility_of(uint64_t w) { return (w >> 32) & kElapsedFieldMask; }

// The credit register
//
// Register 4 of a core's bank. A receiver that has delivered a remote message RMAs an
// incrementing count here, in the SENDER's bank, for the sending core. The sender will not
// re-arm a destination until the credit shows the previous message was consumed.
//
// This is the only backward-flowing state in the protocol and it is what makes the RX
// control word safe to reuse: a single-slot mailbox needs the writer to know the slot is
// free, and nothing else in the design tells it.
constexpr uint32_t kArgCreditReg = 4;

// HOW MANY TX-QUEUE SLOTS ONE MESSAGE COSTS, and how many are held back.
//
constexpr uint64_t kTxDepthPerMessage = 3;
constexpr uint64_t kTxDepthReserve = 8;

constexpr uint32_t kNoticeCtrlOffset = 0;
constexpr uint32_t kNoticeLengthOffset = 8;
constexpr uint32_t kNoticeElapsedOffset = 16;
constexpr uint32_t kNoticeOriginOffset = 24;
// The store forms only. Word 4 of the line; absent from a kOpSendUva notice.
constexpr uint32_t kNoticeUvaOffset = 32;

// receive status control register, in the receiving core's L1.
//
// Written by the host over PCIe, polled by the core locally. It lives in L1 rather than in
// the host register file because the alternative is a Tensix issuing a non-posted PCIe read
// per poll.
//
// opcode impleied by register. There is exactly one thing a receive register
// means, so no opcode field is spent saying it.
//
//   [63:48] magic    -- kCtrlMagic, so an UNINITIALISED L1 word is not a live instruction.
//                       L1 is not zeroed between sweep points (one process per point), and
//                       the earlier design paid 144 phantom transfers for exactly this omission.
//   [47:24] length   -- bytes, 24 bits (16 MiB), against a 1.5 MiB arena
//   [23:0]  offset   -- into the arena AND into L1; they are the same number, see above
//
// no sequence field, because the register is zeroed by consumer. Non-zero means armed,
// zero means idle -- the same rule the host now applies to ctrl_tx/ctrl_rx. Freshness is
// therefore a property of the word rather than of remembered state, which is what removes
// the duplicate filter's job on this path.
//
constexpr uint32_t kRxScrMagicShift = 48;
constexpr uint32_t kRxScrLengthShift = 24;
constexpr uint32_t kRxScrOffsetShift = 0;
constexpr uint64_t kRxScrLengthMask = 0xFFFFFFull;  // 24 bits
constexpr uint64_t kRxScrOffsetMask = 0xFFFFFFull;  // 24 bits

constexpr uint32_t rx_scr_magic(uint64_t w) {
    return static_cast<uint32_t>((w >> kRxScrMagicShift) & kCtrlMagicMask);
}

constexpr uint32_t rx_scr_length(uint64_t w) {
    return static_cast<uint32_t>((w >> kRxScrLengthShift) & kRxScrLengthMask);
}

constexpr uint32_t rx_scr_offset(uint64_t w) {
    return static_cast<uint32_t>((w >> kRxScrOffsetShift) & kRxScrOffsetMask);
}

constexpr bool rx_scr_armed(uint64_t w) { return rx_scr_magic(w) == kCtrlMagic && rx_scr_length(w) != 0; }

constexpr uint32_t kNoticeBytes = 32;

// The store forms carry the effective address, so their notice is the full register line.
constexpr uint32_t kNoticeStoreBytes = 40;

// ---------------------------------------------------------------------------
// Control word encode / decode
// ---------------------------------------------------------------------------
constexpr uint64_t ctrl_encode(uint32_t opcode, uint32_t base, uint32_t count, uint64_t flags, uint32_t sequence) {
    return ((kCtrlMagic & kCtrlMagicMask) << kCtrlMagicShift) |
           ((kCtrlVersion & kCtrlVersionMask) << kCtrlVersionShift) |
           ((static_cast<uint64_t>(sequence) & kCtrlSeqMask) << kCtrlSeqShift) |
           ((flags & kCtrlFlagsMask) << kCtrlFlagsShift) |
           ((static_cast<uint64_t>(count) & kCtrlCountMask) << kCtrlCountShift) |
           ((static_cast<uint64_t>(base) & kCtrlBaseMask) << kCtrlBaseShift) |
           ((static_cast<uint64_t>(opcode) & kCtrlOpcodeMask) << kCtrlOpcodeShift);
}

// The immediate form. base/count do not exist here -- bits [17:8] are the length -- so this
// builds the word directly rather than routing through ctrl_encode() with a fake base.
constexpr uint64_t ctrl_encode_imm(uint32_t length, uint64_t flags, uint32_t sequence) {
    return ((kCtrlMagic & kCtrlMagicMask) << kCtrlMagicShift) |
           ((kCtrlVersion & kCtrlVersionMask) << kCtrlVersionShift) |
           ((static_cast<uint64_t>(sequence) & kCtrlSeqMask) << kCtrlSeqShift) |
           ((flags & kCtrlFlagsMask) << kCtrlFlagsShift) |
           ((static_cast<uint64_t>(length) & kCtrlImmMask) << kCtrlImmShift) |
           ((static_cast<uint64_t>(kOpRdmaWriteImm) & kCtrlOpcodeMask) << kCtrlOpcodeShift);
}

constexpr uint32_t ctrl_magic(uint64_t w) { return static_cast<uint32_t>((w >> kCtrlMagicShift) & kCtrlMagicMask); }
constexpr uint32_t ctrl_version(uint64_t w) {
    return static_cast<uint32_t>((w >> kCtrlVersionShift) & kCtrlVersionMask);
}
constexpr uint32_t ctrl_sequence(uint64_t w) { return static_cast<uint32_t>((w >> kCtrlSeqShift) & kCtrlSeqMask); }
constexpr uint64_t ctrl_flags(uint64_t w) { return (w >> kCtrlFlagsShift) & kCtrlFlagsMask; }
constexpr uint32_t ctrl_count(uint64_t w) { return static_cast<uint32_t>((w >> kCtrlCountShift) & kCtrlCountMask); }
constexpr uint32_t ctrl_base(uint64_t w) { return static_cast<uint32_t>((w >> kCtrlBaseShift) & kCtrlBaseMask); }
constexpr uint32_t ctrl_opcode(uint64_t w) { return static_cast<uint32_t>((w >> kCtrlOpcodeShift) & kCtrlOpcodeMask); }

enum CtrlVerdict : uint32_t {
    kCtrlOk = 0,
    kCtrlIdle,           // magic absent: unprovisioned or not yet armed. Not an error.
    kCtrlBadVersion,     // magic present, version is not ours. Version-locked mismatch.
    kCtrlBadRange,       // base/count run off the end of the data registers
    kCtrlUnknownFlag,    // a flag bit this build does not define
    kCtrlUnknownOpcode,  // opcode this build does not implement
};

constexpr bool ctrl_opcode_known(uint32_t op) {
    return op == kOpNop || op == kOpSendUva || op == kOpRdmaWrite || op == kOpRdmaWriteImm;
}

constexpr CtrlVerdict ctrl_validate(uint64_t w) {
    // Order matters. Magic first, because everything after it is meaningless without it;
    // version second, because a foreign version's field geometry may differ and parsing
    // base/count out of it would produce a confident wrong answer.
    if (ctrl_magic(w) != kCtrlMagic) {
        return kCtrlIdle;
    }
    if (ctrl_version(w) != kCtrlVersion) {
        return kCtrlBadVersion;
    }
    if ((ctrl_flags(w) & ~kFlagKnownMask) != 0) {
        return kCtrlUnknownFlag;
    }
    if (!ctrl_opcode_known(ctrl_opcode(w))) {
        return kCtrlUnknownOpcode;
    }
    // THE IMMEDIATE FORM HAS NO base/count TO RANGE-CHECK. Bits [17:8] are a length, so
    // running the operand-descriptor check over them would reject perfectly good lengths
    // (any imm whose low 5 bits land outside the register file) and accept nothing useful.
    // The opcode fixes the operand layout instead: register 0 is the destination UVA, and
    // that is checked here rather than left to the executor.
    if (ctrl_op_has_imm(ctrl_opcode(w))) {
        // A zero-length store is a bug the same way count == 0 is: it names an address and
        // moves nothing, which is always a caller mistake rather than a legal no-op.
        // kOpNop is the legal no-op
        if (ctrl_imm(w) == 0) {
            return kCtrlBadRange;
        }
        return kCtrlOk;
    }
    // kOpNop CARRIES NO OPERANDS, and that is the whole point of it: it arms a bank so the
    // notice path can be timed without a payload behind it. Every other opcode is rejected
    // for count == 0 below, so this exemption has to be explicit -- without it the only
    // well-formed nop was illegal, while ctrl_opcode_known() still advertised the opcode as
    // implemented.
    //
    // The count == 0 requirement is also the ENFORCEMENT, not just a permission. service_tx()
    // dispatches on operand shape rather than on opcode, so a nop arriving with two operands
    // and a plausible length would be serviced as a real send. Rejecting that here makes the
    // mis-send unreachable by construction instead of by convention.
    if (ctrl_opcode(w) == kOpNop) {
        return ctrl_count(w) == 0 ? kCtrlOk : kCtrlBadRange;
    }
    // count == 0 is rejected: a message that names no operands is always a bug, and
    // allowing it would make "base + count" pass trivially for any base.
    const uint32_t base = ctrl_base(w);
    const uint32_t count = ctrl_count(w);
    if (count == 0 || base >= kDataRegisters || base + count > kDataRegisters) {
        return kCtrlBadRange;
    }
    // kOpRdmaWrite reads reg[base+0] as the destination UVA and reg[base+1] as the length,
    // so a two-operand minimum is part of the instruction, not a convention the executor
    // hopes for.
    if (ctrl_opcode(w) == kOpRdmaWrite && count < 2) {
        return kCtrlBadRange;
    }
    return kCtrlOk;
}

inline const char* ctrl_verdict_name(uint32_t v) {
    switch (v) {
        case kCtrlOk: return "ok";
        case kCtrlIdle: return "idle";
        case kCtrlBadVersion: return "bad-version";
        case kCtrlBadRange: return "bad-range";
        case kCtrlUnknownFlag: return "unknown-flag";
        case kCtrlUnknownOpcode: return "unknown-opcode";
        default: return "?";
    }
}

// ---------------------------------------------------------------------------
// The region
//
//   +--------------------------------------+ base, 2 MiB aligned
//   | RegionHeader                         | one 4 KiB page
//   +--------------------------------------+
//   | bank[0] .. bank[kProvisionedCores-1] | 2 KiB each
//   +--------------------------------------+ 2 MiB aligned
//   | core 0: TX arena (1.5 MiB)           |
//   |         RX arena (1.5 MiB)           |
//   | core 1: TX arena, RX arena           |
//   | ...                                  |
//   +--------------------------------------+
//
// ARENAS ARE INTERLEAVED PER CORE
//
// ALL BANKS ARE ALWAYS PINNED even when only N cores run. Banks are 2 KiB, the whole
// array is 256 KiB at kProvisionedCores, and a poller that can address every bank
// unconditionally is worth more than the quarter megabyte.
//
// ---------------------------------------------------------------------------
constexpr uint64_t kArenaBytes = 1536ull * 1024ull;  // 1.5 MiB, one Tensix L1
						     //
// still 2 arenas. The receive pool is carved OUT of the existing RX arena rather than added
// alongside it, so this design costs no memory at all -- see rx_slot_offset().
constexpr uint64_t kArenasPerCore = 2;                           // TX, RX
constexpr uint64_t kArenaStride = kArenaBytes * kArenasPerCore;  // 3 MiB per core

// Provisioned core count. 128 rather than the 110 a Blackhole 11x10 grid actually has:
// it is a power of two, so core -> offset is a shift rather than a multiply on the
// kernel side where that arithmetic sits in the hot path, and it leaves room for a
// larger grid without a wire-contract change. The cost is 18 unused banks -- 36 KiB.
// It does NOT cost 18 unused arenas, because arenas are pinned as a prefix.
constexpr uint32_t kProvisionedCores = 128;

constexpr uint64_t kHeaderBytes = 4096;

constexpr uint64_t kNoticeStageOffset = 2048;
constexpr uint32_t kNoticeStageSlots = 32;
constexpr uint64_t kNoticeStageSlotBytes = 64;
constexpr uint64_t notice_stage_offset(uint32_t slot) {
    return kNoticeStageOffset + static_cast<uint64_t>(slot % kNoticeStageSlots) * kNoticeStageSlotBytes;
}


constexpr uint64_t kBankArrayBytes = static_cast<uint64_t>(kProvisionedCores) * kBankBytes;  // 256 KiB

//   - map_for_dma() requires the PINNED RANGE to be page-aligned with a page-multiple
//     length (it throws otherwise). That is a property of the region base and the pinned
//     length, not of anything inside it.
//   - The region BASE is 2 MiB-aligned so the allocation can be backed by hugepages.
//   - Individual arenas need only PCIe alignment for the device's posted writes, and at
//     a 1.5 MiB stride every arena is 4 KiB-aligned, which clears any PCIe alignment
//     Blackhole asks for by three orders of magnitude.
//
// The arena array still STARTS at 2 MiB so the header+bank block below it can be sized
// or moved without shifting every arena offset in a live region.
constexpr uint64_t kAlign2M = 2ull * 1024ull * 1024ull;
constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

constexpr uint64_t kArenaArrayOffset = align_up(kHeaderBytes + kBankArrayBytes, kAlign2M);

constexpr uint32_t core_index(uint32_t logical_x, uint32_t logical_y, uint32_t grid_width) {
    return logical_y * grid_width + logical_x;
}

// Offsets. Every one of these is a pure function of the core index -- a core computes
// its own from its own firmware coordinates and is handed no index, so it structurally
// cannot address another core's bank or arena. Same property as rdma_reg_layout.hpp,
// and it is the reason none of these take a "which core" parameter from the wire.
constexpr uint64_t bank_offset(uint32_t core) { return kHeaderBytes + static_cast<uint64_t>(core) * kBankBytes; }
constexpr uint64_t reg_offset(uint32_t core, uint32_t reg) {
    return bank_offset(core) + static_cast<uint64_t>(reg) * kRegisterBytes;
}
constexpr uint64_t tx_arena_offset(uint32_t core) {
    return kArenaArrayOffset + static_cast<uint64_t>(core) * kArenaStride;
}
// ONE ARENA PER RECEIVE SLOT. `rx_arena_offset(core)` is slot 0, so every existing caller
// keeps working and means "the first slot" -- which is what a single-slot protocol always
// meant. A sender writes the slot its ticket named; the receiver drains whichever slot's
// notice is armed and delivers to the core the UVA names, NOT to the core whose arena it
// happened to land in. That decoupling is what lets the pool be shared: where bytes land and
// where they are going stopped being the same fact.
constexpr uint64_t rx_arena_offset(uint32_t core) { return tx_arena_offset(core) + kArenaBytes; }

// WHERE ONE SLOT'S BYTES LIVE, inside the single RX arena. Slot 0 is the arena's start, so
// every existing caller keeps meaning what it always meant -- a single-slot protocol IS this
// one at slot 0.
//
// The receiver delivers to the core the UVA NAMES, not to the core whose arena the bytes
// landed in. That decoupling is the whole trick: where bytes land and where they are going
// stopped being the same fact, which is what lets one pool serve every sender.
constexpr uint64_t rx_slot_offset(uint32_t core, uint32_t slot, uint64_t payload_bytes) {
    return rx_arena_offset(core) + static_cast<uint64_t>(slot) * payload_bytes;
}

// 1 PX slot per core. Every consumer must use this same number or a run corrupts silently,
// so four places derive from it:
//   * the test, sizing the aliased ring  -- scfg.fifo_size = kNumAliasRingSlots * payload
//   * D2H2H2DSocket, as the rx_slot it puts on the wire
//   * D2H2H2DSocket, as the structural in-flight ceiling
//   * BankScanner, as how many slots to sweep -- ScanConfig::rx_slots
//
// A Tensix core has a single TX control word (kCtrlTx), so the kernel waits for the host to
// consume message i before arming i+1 -- the kernel await_completion. One message is
// outstanding per core at most, so a deeper would sit empty. Cross-core concurrency is a
// separate axis and is unaffected by this.
//
constexpr uint32_t kNumAliasRingSlots = 1;

// The arena bound is the other half and cannot live here: it depends on the payload, which is
// a per-run value. It is enforced per message instead, on both sides -- see the slot-aware
// length checks in D2H2H2DSocket::deliver_to_l1() and send_try_start().

// The pinned prefix for a run using `cores` cores. Everything the device or the NIC can
// touch must be inside this, and the program asserts that before it arms anything.
constexpr uint64_t pinned_bytes_for(uint32_t cores) {
    return kArenaArrayOffset + static_cast<uint64_t>(cores) * kArenaStride;
}

constexpr uint64_t kRegionBytesMax = pinned_bytes_for(kProvisionedCores);  // 386 MiB

// one credit word per peer, inside the line register 4 already owns.
//
// The register is 8 bytes of a 64 B line, so seven words sit idle behind it -- eight in
// total, one per host id. That is the whole cost of this: no new register, no layout growth.
//
// A credit is an absolute running count written by the receiver, not an increment. for several
// peers, each writes ITS OWN total into the same word and the register reports whichever peer
// wrote last -- never the sum. The sender compares it against notice_sent, which counts messages
// to ALL peers, so the gate stops opening the moment a core switches destination:
//
//   msg 1 -> host A delivers, writes 1.   notice_sent=1 credit=1  ok
//   msg 2 -> host B delivers, writes 1.   notice_sent=2 credit=1  BLOCKED, permanently
//
// Measured exactly that way: a 3-rank random-destination run stalled with the sender started,
// zero messages moved and no counter advancing anywhere.
//
// So each peer writes the word at ITS OWN host id and the sender SUMS them. The sum is
// "messages of mine that have been consumed, anywhere", which is what the gate always meant.
constexpr uint32_t kMaxCreditPeers = kRegisterBytes / sizeof(uint64_t);  // 8

constexpr uint64_t credit_word_offset(uint32_t core, uint32_t peer_host) {
    return reg_offset(core, kArgCreditReg) + static_cast<uint64_t>(peer_host) * sizeof(uint64_t);
}

// The host ceiling is the credit line, not the UVA selector (which fits 16). Host id
// kMaxCreditPeers would write register 5, which credit_total() never sums: the RMA succeeds
// and that core's sender gate never reopens. Checked in D2H2H2DSocket::open().
constexpr uint32_t kMaxHosts = kMaxCreditPeers;

// the credit word: [63] refused | [62:32] turnaround ns on the receiver's clock | [31:0] count
// consumed. one indivisible write, not two registers -- separate one-sided puts have no MPI
// ordering, so a sender seeing count n could pair it with a flag from n-1. 31 turnaround bits,
// not 32, so a real 2.15 s delivery cannot saturate into the flag.
constexpr uint64_t kCreditCountMask = 0xFFFFFFFFull;
constexpr uint64_t kCreditTurnaroundMask = 0x7FFFFFFFull;
constexpr uint64_t kCreditRefusedBit = 1ull << 63;

constexpr uint64_t credit_pack(uint64_t count, uint64_t turnaround_ns, bool refused = false) {
    return ((turnaround_ns > kCreditTurnaroundMask ? kCreditTurnaroundMask : turnaround_ns) << 32) |
           (count & kCreditCountMask) | (refused ? kCreditRefusedBit : 0ull);
}
constexpr uint64_t credit_count_of(uint64_t w) { return w & kCreditCountMask; }
constexpr uint64_t credit_turnaround_of(uint64_t w) { return (w >> 32) & kCreditTurnaroundMask; }
constexpr bool credit_refused_of(uint64_t w) { return (w & kCreditRefusedBit) != 0ull; }

// ---------------------------------------------------------------------------
// The region header
//
// Lets a second process -- the peer in a single-host two-process run, or a tool inspecting a
// live region -- confirm it agrees about the geometry before reading any register. Parties
// disagreeing on kProvisionedCores or kArenaBytes compute different offsets for the same core
// and nothing downstream notices: the reader looks at the wrong 64 bytes and finds them idle.
// So the constants are published here and verify_header() refuses on mismatch.
// ---------------------------------------------------------------------------
constexpr uint64_t kRegionMagic = 0x543648'4F535456ull;  // "T6HOSTV" -- distinctive in a hex dump

struct RegionHeader {
    uint64_t magic;               // kRegionMagic
    uint32_t version;             // kCtrlVersion -- header and control word version together
    uint32_t provisioned_cores;   // kProvisionedCores
    uint64_t arena_bytes;         // kArenaBytes
    uint64_t arena_stride;        // kArenaStride
    uint64_t bank_bytes;          // kBankBytes
    uint64_t arena_array_offset;  // kArenaArrayOffset
    uint32_t cores_in_use;        // the prefix actually pinned and armed
    uint32_t host_id;             // this host's identifier in the UVA selector's host field
    uint32_t chips_per_host;      // the UVA selector stride -- see host_uva.hpp
    uint32_t chip;                // which chip on this host this region serves
    // The two fields a peer must agree with or it addresses the wrong core entirely.
    // grid_width is here for the same reason chips_per_host is: a mismatch does not
    // corrupt an offset, it silently names a different core, and both sides then read a
    // bank that is legitimately idle.
    uint32_t grid_width;
    uint32_t grid_height;
    uint64_t pinned_bytes;    // pinned_bytes_for(cores_in_use)
    uint64_t device_io_base;  // PinnedMemory::get_noc_addr().addr -- what the T6 writes to
    uint32_t pcie_xy_enc;     // PinnedMemory::get_noc_addr().pcie_xy_enc
    uint32_t reserved;
};

}  // namespace tt::tt_metal::experimental
