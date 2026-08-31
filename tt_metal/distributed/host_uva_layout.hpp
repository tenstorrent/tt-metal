// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wire contract for the T6 -> HOST RDMA register file. The host is the data mover.
//
// Where an on-chip coprocessor moves the data, the register file lives in that core's
// local SRAM, that one core sweeps the banks looking for armed status words, and it is
// the data mover. Here the
// register file lives in PINNED HOST MEMORY, the host sweeps the banks, and the host is
// the data mover. Three consequences fall out of that and they are the reason this is a
// separate program rather than a flag on the old one:
//
//   1. The bank sweep is no longer serial. A host walks them from many pinned threads at once
//
//   2. The T6 writes are POSTED PCIe writes, not NOC writes into on-chip SRAM.
//
//   3. The register file and the payload live in the SAME pinned region, and that region
//      is registered with libfabric exactly once. So the bytes a Tensix core pushed over
//      PCIe are already sitting in a registered MR when the host decides to send them --
//      no bounce buffer, no per-message registration, no copy between the device landing
//      them and the NIC picking them up.
//
// tt-metal's D2HSocket allocates its own pinned FIFO per socket and memcpys out of it on
// read(). At ~110 Tensix cores that is ~110 pinned allocations, ~110 sender TLBs and a
// copy per record. PinnedMemory::Create() does not allocate -- it pins memory the CALLER
// owns (`PinnedMemory only supports mapping existing host memory`), via
// PCIDevice::map_for_dma -> TENSTORRENT_IOCTL_PIN_PAGES on our own virtual address. The
// pages do not move and the VA stays ours, so the identical range can also be handed to
// fi_mr_reg(): two independent, refcounted pins of the same pages, one for the TT device
// and one for the NIC. That is the whole reason a single MR over the whole region works.
//
// Included by the host program AND by the RV32 Tensix kernels, so it must stay free of
// anything host-only -- <stdint.h> and constexpr arithmetic only. Kernels reach it as
// "../host_uva_layout.hpp"; a quote-include resolves against the including file's own
// directory, so no kernel include path is needed.
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

static_assert(kBankBytes == 2048, "a bank is 32 lines of 64 B");

// Register indices. Data registers are 0..29; the two control registers follow them, so
// a data index is always its own register number and no arithmetic separates the two
// spaces. Putting control LAST rather than first matters: a bank is zero-filled at
// provisioning, and a control word at offset 0 is the first thing a wild write from a
// miscomputed base lands on.
constexpr uint32_t kCtrlTx = kDataRegisters + 0;  // 30: T6 -> host, the request
constexpr uint32_t kCtrlRx = kDataRegisters + 1;  // 31: host -> T6, RX SLOT 0 (see below)

// ---------------------------------------------------------------------------
// THE RECEIVE SLOT POOL: K slots per destination core, claimed by any sender.
//
// A destination core had a single RX control word and a single
// arena, and the credit that frees them is accounted per SOURCE. Those are the same thing only
// while a destination has exactly ONE lifetime source -- which the fixed ring gives and
// anything else does not. Vary the destination and the slot changes hands with no handshake:
// the new sender's credit describes its OWN previous message, not whether this slot is free.
// Measured as a silent stall with no deadline to fire, at two hosts as soon as the destination
// CORE varied, long before three hosts made two source HOSTS collide.
//
// One slot per source does not scale: memory O(hosts) per host and
// O(hosts^2) arenas across the job -- 448 at 8 hosts, ~32,000 at 64. A design whose cost grows
// with the number of possible senders cannot make a galaxy-scale claim.
//
// The pool is shared and the sender claims. A sender takes a ticket with a one-sided
// fi_fetch_atomic(FI_SUM) on the destination's slot_head, and slot = ticket % kRxSlots. There
// is NO per-peer state on the receiver at all, so 2 hosts and 200 hosts cost the same. verbs
// on this fabric advertises FI_ATOMIC (checked: caps on the FI_EP_MSG endpoint this build
// negotiates), and RoCE RC does FetchAdd natively on 8 bytes.
//
// Overrun is the sender's job. slot_tail is published by the RECEIVER as it drains, and a
// sender must not lap it: ticket - tail < kRxSlots. Pushed lazily rather than per message --
// per message would be the credit again, and O(peers) pushes is the cost this design exists to
// avoid.
//
// Slot count is runtime - costs no memory.
//
// The obvious build gives each slot its own arena, which triples the region and caps nothing.
// The obvious fix -- fixed subdivision -- caps the payload instead: slots x max_payload is the
// arena, so 3 slots means 512 KiB messages and a 1 MiB point has to be refused.
//
// Neither trade is necessary, because the payload is constant (`--bytes` is fixed
// per point). A ring of variable-size claims therefore degenerates to a ring of equal slots
// with no allocator, no padding and no straddling the wrap:
//
//     slots = kArenaBytes / payload_bytes
//
// 96 slots at 16 KiB, 3 at 512 KiB, 1 at 1.5 MiB. Depth is size-dependent, which is the right
// shape -- small messages, where the protocol overhead dominates, get the most concurrency --
// and the payload ceiling stays a whole arena.
constexpr uint32_t kPayloadStampOffset = 0;   // uint32 iteration
constexpr uint32_t kPayloadDestOffset = 4;    // uint32 destination selector
constexpr uint32_t kPayloadHeaderBytes = 8;
static_assert(kPayloadDestOffset + 4 == kPayloadHeaderBytes, "the header is the stamp and the selector");

constexpr uint32_t kRxNoticeSlots = 8;


constexpr uint32_t kRxSlotBase = 8;  // registers 8..14 hold slots 1..7 (slot 0 is kCtrlRx)
static_assert(kRxSlotBase + kRxNoticeSlots - 1 < kDataRegisters,
              "the RX notice block must stay inside the data registers");

constexpr uint32_t rx_slot_reg(uint32_t slot) {
    return slot == 0 ? kCtrlRx : (kRxSlotBase + slot - 1);
}
static_assert(rx_slot_reg(0) == kCtrlRx, "slot 0 keeps its historical home");
static_assert(rx_slot_reg(1) == kRxSlotBase, "slot 1 starts the relocated block");

static_assert(kCtrlTx < kRegistersPerBank && kCtrlRx < kRegistersPerBank, "control regs live in the bank");

// ---------------------------------------------------------------------------
// The control word
//
// [63:48] magic      16   0x484F -- rejects an unprovisioned/zeroed bank and rejects
//                              a legacy v1/v2 status word, which used magic 0x57A7
// [47:44] version     4   protocol version; a reader REFUSES a version it does not know
// [43:32] sequence   12   monotonic per (bank, direction); distinguishes a re-armed
//                              word from the one already serviced
// [31:18] flags      14   unknown bits REJECT -- see below
// [17:13] count       5   number of data registers this message occupies, 1..30
// [12:8]  base        5   first data register index, 0..29
// [7:0]   opcode      8
// ---------------------------------------------------------------------------
constexpr uint64_t kCtrlMagic = 0x484Full;  // 'H','O' -- host-owned register file
constexpr uint32_t kCtrlMagicShift = 48;
constexpr uint64_t kCtrlMagicMask = 0xFFFFull;

constexpr uint64_t kCtrlVersion = 3ull;
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

static_assert(kCtrlMagicShift == kCtrlVersionShift + 4, "magic must abut version");
static_assert(kCtrlVersionShift == kCtrlSeqShift + 12, "version must abut sequence");
static_assert(kCtrlSeqShift == kCtrlFlagsShift + 14, "sequence must abut flags");
static_assert(kCtrlFlagsShift == kCtrlCountShift + 5, "flags must abut count");
static_assert(kCtrlCountShift == kCtrlBaseShift + 5, "count must abut base");
static_assert(kCtrlBaseShift == kCtrlOpcodeShift + 8, "base must abut opcode");
static_assert(kDataRegisters <= kCtrlCountMask + 1, "count field must express every data register");
static_assert(kDataRegisters <= kCtrlBaseMask + 1, "base field must index every data register");

static_assert(kCtrlMagic != 0, "a zeroed bank must not decode as armed");
static_assert(kCtrlMagic != 0x57A7ull, "must not collide with the legacy status magic");

enum CtrlOpcode : uint32_t {
    kOpNop = 0x00,       // armed but nothing to do; used to time the notice path alone
    kOpSendUva = 0x01,   // move bytes named by a UVA operand to the UVA's owner
    kOpEchoUva = 0x02,   // kOpSendUva, and the far side sends it back -- round trip

    // TWO ENCODINGS OF ONE OPERATION, which is the sb/sh/sw/sd vs block-move split:
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

constexpr uint32_t kCtrlImmShift = kCtrlBaseShift;                       // 8
constexpr uint64_t kCtrlImmMask = (kCtrlCountMask << 5) | kCtrlBaseMask; // 10 bits
constexpr uint32_t kCtrlImmMax = static_cast<uint32_t>(kCtrlImmMask);    // 1023
static_assert(kCtrlImmShift == kCtrlBaseShift, "the immediate overlays base:count exactly");
static_assert(kCtrlCountShift == kCtrlBaseShift + 5, "base and count must abut for the overlay to be contiguous");

constexpr uint32_t ctrl_imm(uint64_t w) { return static_cast<uint32_t>((w >> kCtrlImmShift) & kCtrlImmMask); }

constexpr uint64_t ctrl_encode_imm(uint32_t length, uint64_t flags, uint32_t sequence);

constexpr uint64_t kFlagPullDelivery = 1ull << 0;  // far side: T6 pulls, rather than host pushing
constexpr uint64_t kFlagStamped = 1ull << 1;       // payload carries a per-message stamp to defeat stale-slot reads
constexpr uint64_t kFlagCycles = 1ull << 2;

constexpr uint64_t kFlagReply = 1ull << 3;

constexpr uint64_t kFlagRemoteNotice = 1ull << 4;

constexpr uint64_t kFlagKnownMask =
    kFlagPullDelivery | kFlagStamped | kFlagCycles | kFlagReply | kFlagRemoteNotice;

// THE CREDIT REGISTER.
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

// ---------------------------------------------------------------------------
// THE ARENA IS AN L1 MIRROR, AND THAT IS WHAT MAKES A STORE COST TWO FIELDS
// INSTEAD OF THREE.
//
// A store names three things: source, destination, length. The source is the sender's own
// arena; the destination is an address in the FAR core's L1. Carrying both offsets to the
// far side would need three fields on the wire and a translation at the end of it.
//
// It needs neither, because an arena is EXACTLY ONE L1:
//
//     static_assert(kArenaBytes == 0x180000)   // and l1_size_per_core() == 0x180000
//
// So the sending host places the payload in the remote arena at the offset the bytes will
// occupy in L1, and from then on ONE NUMBER means both "where in the arena to read" and
// "where in L1 to write". The receiving core does a straight contiguous pull with no
// address arithmetic at all.
//
// Two consequences worth stating, because neither is obvious:
//
//   * The low part of every arena is never used -- it mirrors the L1 below the allocator
//     base, which belongs to tt-metal. That costs nothing: the region is a static array and
//     untouched pages are not resident.
//   * Messages aimed at different L1 offsets no longer collide IN THE ARENA either, so the
//     one-message-per-core limit becomes purely a property of the single control word rather
//     than of the buffer. Giving a core depth is now a control-word question.
//
// INCOMPATIBLE WITH H2D RING ALIASING, and the two must never both be on. Aliasing requires
// fifo_size == payload so the ring's write pointer returns to 0 for a FIXED target offset
// (h2d_socket.cpp:663-667 wraps only on the exact-fill case). An L1-mirror arena is written
// at VARYING offsets, which walks the write pointer somewhere the device is not reading.
// Refused by name where both are requested.
// ---------------------------------------------------------------------------

// THE RECEIVE STATUS CONTROL REGISTER, in the receiving core's L1.
//
// Written by the host over PCIe, polled by the core locally. It lives in L1 rather than in
// the host register file because the alternative is a Tensix issuing a non-posted PCIe read
// per poll -- the 40 MB/s path mmio_bench measures in the other direction.
//
// THE OPCODE IS IMPLIED BY THE REGISTER. There is exactly one thing a receive register
// means, so no opcode field is spent saying it.
//
//   [63:48] magic    -- kCtrlMagic, so an UNINITIALISED L1 word is not a live instruction.
//                       L1 is not zeroed between sweep points (one process per point), and
//                       the earlier design paid 144 phantom transfers for exactly this omission.
//   [47:24] length   -- bytes, 24 bits (16 MiB), against a 1.5 MiB arena
//   [23:0]  offset   -- into the arena AND into L1; they are the same number, see above
//
// NO SEQUENCE FIELD, because the register is ZEROED BY THE CONSUMER. Non-zero means armed,
// zero means idle -- the same rule the host now applies to ctrl_tx/ctrl_rx. Freshness is
// therefore a property of the word rather than of remembered state, which is what removes
// the duplicate filter's job on this path.
constexpr uint32_t kRxScrMagicShift = 48;
constexpr uint32_t kRxScrLengthShift = 24;
constexpr uint32_t kRxScrOffsetShift = 0;
constexpr uint64_t kRxScrLengthMask = 0xFFFFFFull;  // 24 bits
constexpr uint64_t kRxScrOffsetMask = 0xFFFFFFull;  // 24 bits

constexpr uint64_t rx_scr_encode(uint32_t offset, uint32_t length) {
    return ((kCtrlMagic & kCtrlMagicMask) << kRxScrMagicShift) |
           ((static_cast<uint64_t>(length) & kRxScrLengthMask) << kRxScrLengthShift) |
           ((static_cast<uint64_t>(offset) & kRxScrOffsetMask) << kRxScrOffsetShift);
}
constexpr uint32_t rx_scr_magic(uint64_t w) {
    return static_cast<uint32_t>((w >> kRxScrMagicShift) & kCtrlMagicMask);
}
constexpr uint32_t rx_scr_length(uint64_t w) {
    return static_cast<uint32_t>((w >> kRxScrLengthShift) & kRxScrLengthMask);
}
constexpr uint32_t rx_scr_offset(uint64_t w) {
    return static_cast<uint32_t>((w >> kRxScrOffsetShift) & kRxScrOffsetMask);
}
// Armed AND self-consistent. A length of 0 is refused for the same reason ctrl_validate
// refuses count == 0: it names an address and moves nothing, which is always a caller bug.
constexpr bool rx_scr_armed(uint64_t w) {
    return rx_scr_magic(w) == kCtrlMagic && rx_scr_length(w) != 0;
}

static_assert(rx_scr_offset(rx_scr_encode(0x123456, 0xABCDEF)) == 0x123456, "offset round-trips");
static_assert(rx_scr_length(rx_scr_encode(0x123456, 0xABCDEF)) == 0xABCDEF, "length round-trips");
static_assert(!rx_scr_armed(0), "a zeroed L1 word must not decode as an armed receive");
static_assert(!rx_scr_armed(rx_scr_encode(64, 0)), "a zero-length receive is refused");
static_assert(rx_scr_armed(rx_scr_encode(64, 4096)), "a well-formed receive is armed");
constexpr uint32_t kNoticeBytes = 32;
// The store forms carry the effective address, so their notice is the full register line.
constexpr uint32_t kNoticeStoreBytes = 40;

static_assert(kNoticeBytes <= kRegisterBytes, "the notice must fit inside one register line");
static_assert(kNoticeStoreBytes <= kRegisterBytes, "a store notice must still fit one register line");
static_assert(kNoticeUvaOffset + 8 == kNoticeStoreBytes, "the UVA is the last word of a store notice");
// kCtrlRx is the LAST register of the bank, so a notice may fill its line and not one byte
// more -- past that is the next core's data registers, and banks are contiguous.
static_assert(kCtrlRx * kRegisterBytes + kNoticeStoreBytes <= kBankBytes,
              "a store notice must not run past the end of the bank");

// How many bytes this opcode's notice occupies. One function, so the writers and the
// reader cannot disagree about whether word 4 is present.
constexpr uint32_t notice_bytes_for(uint32_t opcode) {
    return ctrl_op_is_store(opcode) ? kNoticeStoreBytes : kNoticeBytes;
}

// ---------------------------------------------------------------------------
// The register file is the PROTOCOL and it does not change here. What changes is who
// carries it: instead of a Tensix posting operands into host-resident registers and arming
// a control word for the scanner to find, the same control word and the same operands
// travel as a fixed header at the front of a D2HSocket page, with the payload behind them.
// A host worker that reads the page has the whole request -- there is no second place to
// look and no ordering to establish, for exactly the reason the RX notice already gives:
// one transfer cannot be observed half-arrived.
//
// ONE PAGE PER MESSAGE, as on the H2D side. page_size = kD2HHeaderBytes + payload, so a
// message is never split across the ring's end and the sender's push_bytes() and the
// host's read() cannot disagree about tail slack. It also means the socket's page size is
// fixed at construction and a run's payload size with it -- the same constraint the H2D
// direction already carries, and for the same reason.
//
constexpr uint32_t kD2HHeaderCtrlOffset = 0;
constexpr uint32_t kD2HHeaderOperandOffset = 8;   // operands run consecutively from here
constexpr uint32_t kD2HHeaderMaxOperands = 7;
constexpr uint32_t kD2HHeaderBytes = 64;

static_assert(kD2HHeaderOperandOffset + kD2HHeaderMaxOperands * 8 == kD2HHeaderBytes,
              "the header must be exactly one PCIe alignment unit, fully used");
static_assert(kD2HHeaderMaxOperands <= kDataRegisters,
              "cannot carry more operands than the register file has");

// The socket page for a run of `payload_bytes`. PCIe-aligned by construction whenever the
// payload is, because the header is itself a multiple of the alignment.
constexpr uint32_t d2h_page_bytes(uint32_t payload_bytes) {
    return kD2HHeaderBytes + payload_bytes;
}

// ---------------------------------------------------------------------------
// Control word encode / decode
// ---------------------------------------------------------------------------
constexpr uint64_t ctrl_encode(
    uint32_t opcode, uint32_t base, uint32_t count, uint64_t flags, uint32_t sequence) {
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
    return op == kOpNop || op == kOpSendUva || op == kOpEchoUva || op == kOpRdmaWrite || op == kOpRdmaWriteImm;
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
        // kOpNop exists for "armed but nothing to do".
        if (ctrl_imm(w) == 0) {
            return kCtrlBadRange;
        }
        return kCtrlOk;
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

// Round-trip proofs, evaluated at compile time so a field that stops tiling the word is
// a build failure rather than a wrong transfer.
static_assert(ctrl_validate(0) == kCtrlIdle, "a zero word is idle, not armed");
static_assert(ctrl_validate(~0ull) != kCtrlOk, "an all-ones word must not validate");
static_assert(ctrl_validate(ctrl_encode(kOpSendUva, 0, 2, 0, 0)) == kCtrlOk, "minimal 2-arg message");
static_assert(ctrl_validate(ctrl_encode(kOpSendUva, 28, 2, 0, 0)) == kCtrlOk, "2-arg message at the top of the bank");
static_assert(ctrl_validate(ctrl_encode(kOpSendUva, 29, 2, 0, 0)) == kCtrlBadRange, "one past the end is rejected");
static_assert(ctrl_validate(ctrl_encode(kOpSendUva, 0, 30, 0, 0)) == kCtrlOk, "a 30-argument message is legal");
static_assert(ctrl_validate(ctrl_encode(kOpSendUva, 0, 0, 0, 0)) == kCtrlBadRange, "zero operands is a bug");
static_assert(ctrl_validate(ctrl_encode(0xFF, 0, 2, 0, 0)) == kCtrlUnknownOpcode, "unknown opcode is refused");
static_assert(
    ctrl_validate(ctrl_encode(kOpSendUva, 0, 2, 1ull << 13, 0)) == kCtrlUnknownFlag, "unknown flag is refused");
static_assert(ctrl_opcode(ctrl_encode(kOpEchoUva, 7, 3, kFlagPullDelivery, 4095)) == kOpEchoUva, "opcode round-trips");
static_assert(ctrl_base(ctrl_encode(kOpEchoUva, 7, 3, kFlagPullDelivery, 4095)) == 7, "base round-trips");
static_assert(ctrl_count(ctrl_encode(kOpEchoUva, 7, 3, kFlagPullDelivery, 4095)) == 3, "count round-trips");
static_assert(
    ctrl_flags(ctrl_encode(kOpEchoUva, 7, 3, kFlagPullDelivery, 4095)) == kFlagPullDelivery, "flags round-trip");
static_assert(ctrl_sequence(ctrl_encode(kOpEchoUva, 7, 3, kFlagPullDelivery, 4095)) == 4095, "sequence round-trips");
// The cross-protocol rejection, as a proof rather than a comment: a legacy status word
// carries magic 0x57A7 in [63:48], and this validator must call it idle, never parse it.
static_assert(ctrl_validate(0x57A7ull << 48) == kCtrlIdle, "a legacy status word does not decode here");

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
// STILL TWO ARENAS. The receive pool is carved OUT of the existing RX arena rather than added
// alongside it, so this design costs no memory at all -- see rx_slot_offset().
constexpr uint64_t kArenasPerCore = 2;               // TX, RX
constexpr uint64_t kArenaStride = kArenaBytes * kArenasPerCore;  // 3 MiB per core

static_assert(kArenaBytes == 0x180000ull, "an arena is exactly one Blackhole Tensix L1");

// How many messages of this size actually fit, given both bounds. Runtime, and HOST-ONLY --
// no kernel computes an RX offset, which is what allows any of this to be runtime at all.
constexpr uint32_t rx_slots_capacity(uint64_t payload_bytes) {
    const uint64_t fit = payload_bytes ? (kArenaBytes / payload_bytes) : 1;
    const uint64_t capped = fit < kRxNoticeSlots ? fit : kRxNoticeSlots;
    return static_cast<uint32_t>(capped < 1 ? 1 : capped);
}
static_assert(rx_slots_capacity(kArenaBytes) == 1, "a whole-arena payload leaves exactly one slot");
static_assert(rx_slots_capacity(kArenaBytes / 2) == 2, "half an arena leaves two");
static_assert(rx_slots_capacity(16384) == kRxNoticeSlots, "small payloads are bounded by the notice lines");



// Deferred to here because they need kArenaBytes, which is declared below the receive-SCR
// contract. Both are the same claim from two directions: one number addresses the arena and
// the L1, so the field carrying it has to span the larger of the two -- and they are equal.
static_assert(kArenaBytes <= (kRxScrOffsetMask + 1), "an arena offset must fit the receive SCR's offset field");
static_assert(kArenaBytes <= (kRxScrLengthMask + 1), "a whole-arena length must fit the receive SCR");

// Provisioned core count. 128 rather than the 110 a Blackhole 11x10 grid actually has:
// it is a power of two, so core -> offset is a shift rather than a multiply on the
// kernel side where that arithmetic sits in the hot path, and it leaves room for a
// larger grid without a wire-contract change. The cost is 18 unused banks -- 36 KiB.
// It does NOT cost 18 unused arenas, because arenas are pinned as a prefix.
constexpr uint32_t kProvisionedCores = 128;

static_assert((kProvisionedCores & (kProvisionedCores - 1)) == 0, "core count must be a power of two");

constexpr uint64_t kHeaderBytes = 4096;

// NOTICE STAGING. To RMA an inbound notice into a PEER's bank, the source bytes must sit
// in our own registered region -- fi_write cannot send from the stack. These slots are
// that source. They live in the unused second half of the header page, so they cost
// nothing extra and stay inside the always-pinned prefix.
//
// One slot per worker, indexed by worker id, so two workers never stage into the same
// bytes. 32 slots because that is the CPU count where mmio_bench found the scaling knee;
// a pool larger than that is already in a known-bad regime.
constexpr uint64_t kNoticeStageOffset = 2048;
constexpr uint32_t kNoticeStageSlots = 32;
constexpr uint64_t kNoticeStageSlotBytes = 64;
constexpr uint64_t notice_stage_offset(uint32_t slot) {
    return kNoticeStageOffset + static_cast<uint64_t>(slot % kNoticeStageSlots) * kNoticeStageSlotBytes;
}
static_assert(kNoticeStageOffset + kNoticeStageSlots * kNoticeStageSlotBytes <= kHeaderBytes,
              "notice staging must fit the header page");
static_assert(kNoticeStageOffset >= sizeof(uint64_t) * 24, "staging must not overlap the RegionHeader fields");

// THE SMALL-WRITE COMPLETION PROBE, and why a scratch line in the header earns its place.
//
// A provider may complete a small RMA write, or may carry it inline and report nothing --
// MEASURED on a layered verbs provider, which delivered a 32 B write and never produced a
// completion for it, with inject_size 192. Which of the two it does changes how this
// transport must send a payload, and it cannot be inferred: inject_size says a write is small
// enough to be injected, not that the provider will suppress its completion.
//
// So it is measured instead, once, at connect: write 8 bytes into the PEER's probe line and
// see whether the completion arrives. The line is in the header's unused middle -- past the
// RegionHeader fields, before the notice staging slots -- and NOTHING reads it. Its only
// purpose is to be a legal RMA destination that cannot disturb a bank, an arena or a staging
// slot, which is exactly what a probe needs and what no other offset in the region offers.
constexpr uint64_t kSuppressProbeOffset = 1024;
constexpr uint64_t kSuppressProbeBytes = 64;
static_assert(kSuppressProbeOffset + kSuppressProbeBytes <= kNoticeStageOffset,
              "the probe line must not overlap notice staging");

constexpr uint64_t kBankArrayBytes = static_cast<uint64_t>(kProvisionedCores) * kBankBytes;  // 256 KiB

// ALIGNMENT, AND WHAT ACTUALLY NEEDS IT.
//
// A 1.5 MiB arena cannot be 2 MiB-aligned -- the stride is 3 MiB, so every second arena
// starts at an odd 1.5 MiB boundary. That is fine, and it is worth writing down why
// rather than padding the arenas out to 2 MiB and burning 25% of 386 MB on an alignment
// nothing asks for:
//
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

static_assert(kArenaArrayOffset == kAlign2M, "header + 128 banks fits in the first 2 MiB");
static_assert(kArenaStride % kPageBytes == 0, "arena stride keeps every arena page-aligned");

// ---------------------------------------------------------------------------
// CORE INDEXING, AND A DELIBERATE DIVERGENCE FROM rdma_reg_layout.hpp.
//
// The older tree numbers cores `logical_y * 16 + logical_x` -- a FIXED row stride of 16,
// independent of the real grid width. It has a good reason: its register file lives in a
// fixed on-chip SRAM window, and pinning the row width means every bank keeps its address when
// the grid changes shape. The header even records the arithmetic that forced the grid
// height down to 15 to fit 528 KiB of on-chip SRAM.
//
// NONE OF THAT APPLIES HERE, and copying it would cost real memory. On an 11x10
// Blackhole grid a fixed stride of 16 makes the indices SPARSE: (10,9) is 154, so 110
// live cores span 155 slots. At 3 MiB of arena per slot that is 465 MiB to hold 330 MiB
// of arenas, and it would overflow kProvisionedCores = 128 outright.
//
// So this tree numbers cores by the ACTUAL grid width, which makes the indices
// contiguous 0..N-1. Two things follow, and both are handled rather than assumed:
//
//   - The prefix pin becomes meaningful. `header + banks + N * 3 MiB` covers exactly the
//     cores in use only because index N-1 is the last one used. Under a fixed stride it
//     would cover a third of them.
//   - grid_width becomes part of the wire contract. Two parties with different widths
//     compute different indices for the same physical core, which is the same failure
//     shape as a chips_per_host mismatch: each reads the wrong bank and finds it idle.
//     So it is published in RegionHeader and compared on attach, exactly like
//     chips_per_host.
//
// A UVA from this tree and a UVA from that older tree therefore mean different cores for
// the same selector value. That is safe because they are different programs addressing
// different regions and no word crosses between them -- but it is the reason
// host_uva_drift.cpp checks the SELECTOR FORMULA (which is shared) and not core_index
// (which is not).
// ---------------------------------------------------------------------------
constexpr uint32_t core_index(uint32_t logical_x, uint32_t logical_y, uint32_t grid_width) {
    return logical_y * grid_width + logical_x;
}

static_assert(core_index(0, 0, 11) == 0, "Tensix (0,0) is core 0");
static_assert(core_index(10, 0, 11) == 10, "the first row is contiguous");
static_assert(core_index(0, 1, 11) == 11, "the second row follows it with no gap");
static_assert(core_index(10, 9, 11) == 109, "an 11x10 grid ends at 109, not 154");
static_assert(core_index(10, 9, 11) < kProvisionedCores, "a full Blackhole grid fits the provisioned banks");

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
static_assert(rx_slot_offset(0, 0, 16384) == rx_arena_offset(0), "slot 0 is the arena start");
static_assert(rx_slot_offset(0, 7, 16384) + 16384 <= rx_arena_offset(0) + kArenaBytes,
              "the last notice slot's bytes must stay inside the arena at the smallest size");

// THE SLOT WINDOW, in the line register 4 owns.
//
// It REPLACES the per-peer credit words, which replaced the single credit -- and this is the
// version that does not grow with the job. `head` is incremented by SENDERS with a one-sided
// atomic; `tail` is published by the RECEIVER as it drains. A sender may use ticket n only
// while n - tail < kRxSlots.
//
// Both live in one 64 B line with room to spare, and there is exactly one pair per core rather
// than one per (core, peer): the receiver keeps no per-sender state at all.
constexpr uint32_t kSlotWindowReg = kArgCreditReg;  // register 4, reusing the credit's line
constexpr uint64_t slot_head_offset(uint32_t core) { return reg_offset(core, kSlotWindowReg); }
constexpr uint64_t slot_tail_offset(uint32_t core) { return reg_offset(core, kSlotWindowReg) + 8; }
static_assert(slot_tail_offset(0) - slot_head_offset(0) == 8, "head and tail are adjacent words");
static_assert(slot_tail_offset(0) + 8 <= reg_offset(0, kSlotWindowReg) + kRegisterBytes,
              "the slot window must stay inside its own register line");
// The pinned prefix for a run using `cores` cores. Everything the device or the NIC can
// touch must be inside this, and the program asserts that before it arms anything.
constexpr uint64_t pinned_bytes_for(uint32_t cores) {
    return kArenaArrayOffset + static_cast<uint64_t>(cores) * kArenaStride;
}

constexpr uint64_t kRegionBytesMax = pinned_bytes_for(kProvisionedCores);  // 386 MiB

static_assert(bank_offset(0) == kHeaderBytes, "bank 0 follows the header");
static_assert(reg_offset(0, kCtrlTx) == kHeaderBytes + 30 * 64, "the TX control word is register 30");
static_assert(reg_offset(0, kCtrlRx) == kHeaderBytes + 31 * 64, "the RX control word is register 31");

// ONE CREDIT WORD PER PEER, inside the line register 4 already owns.
//
// The register is 8 bytes of a 64 B line, so seven words sit idle behind it -- eight in
// total, one per host id. That is the whole cost of this: no new register, no layout growth.
//
// WHY IT HAD TO CHANGE. A credit is an ABSOLUTE running count written by the receiver, not an
// increment. With one peer that is unambiguous. With several, each writes ITS OWN total into
// the same word and the register reports whichever peer wrote last -- never the sum. The
// sender compares it against notice_sent, which counts messages to ALL peers, so the gate
// stops opening the moment a core switches destination:
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
static_assert(kMaxCreditPeers == 8, "eight credit words fit the line register 4 owns");

constexpr uint64_t credit_word_offset(uint32_t core, uint32_t peer_host) {
    return reg_offset(core, kArgCreditReg) + static_cast<uint64_t>(peer_host) * sizeof(uint64_t);
}
static_assert(credit_word_offset(0, 0) == reg_offset(0, kArgCreditReg), "peer 0 IS the register");
static_assert(credit_word_offset(0, kMaxCreditPeers - 1) + 8 == reg_offset(0, kArgCreditReg) + kRegisterBytes,
              "the last credit word must not leave register 4's line");
static_assert(bank_offset(1) - bank_offset(0) == kBankBytes, "banks are contiguous");
static_assert(tx_arena_offset(0) == kAlign2M, "arena array starts at 2 MiB");
static_assert(rx_arena_offset(0) == kAlign2M + kArenaBytes, "RX follows TX within a core");
// The next core follows ALL of this one's arenas -- TX plus every RX slot. Was written as
// "rx_arena_offset(0) + kArenaBytes" when there were exactly two; that spelling silently
// encoded kArenasPerCore == 2 and is the assertion that caught the change.
static_assert(tx_arena_offset(1) == tx_arena_offset(0) + kArenaStride, "the next core follows every arena");
static_assert(kArenaStride == kArenaBytes * 2, "TX and RX, unchanged -- the pool is carved out of RX");
static_assert(tx_arena_offset(0) % kPageBytes == 0, "every TX arena is page aligned");
static_assert(rx_arena_offset(0) % kPageBytes == 0, "every RX arena is page aligned");
static_assert(tx_arena_offset(127) % kPageBytes == 0, "... including the last one");
static_assert(rx_arena_offset(127) % kPageBytes == 0, "... including the last one");
// The pinned length is what map_for_dma actually checks, so prove it for every prefix
// shape the program can ask for rather than only the full one.
static_assert(pinned_bytes_for(1) % kPageBytes == 0, "a one-core pin is a page multiple");
static_assert(pinned_bytes_for(110) % kPageBytes == 0, "a full-grid pin is a page multiple");
static_assert(kRegionBytesMax % kPageBytes == 0, "a fully provisioned pin is a page multiple");
static_assert(pinned_bytes_for(0) == kArenaArrayOffset, "zero cores still pins header + banks");
static_assert(kRegionBytesMax == kAlign2M + 128ull * kArenaStride, "386 MiB at full provisioning");
// No bank overlaps the arena array, and no arena overlaps the next core's.
static_assert(bank_offset(kProvisionedCores - 1) + kBankBytes <= kArenaArrayOffset, "banks stay clear of arenas");

// ---------------------------------------------------------------------------
// The region header
//
// It exists so a second process attaching to this region -- the peer in a single-host
// two-process run, or a tool inspecting a live region -- can verify it agrees about the
// geometry BEFORE it reads a single register. Two parties disagreeing about
// kProvisionedCores or kArenaBytes compute different offsets for the same core, and
// nothing downstream would report it: the reader simply looks at the wrong 64 bytes and
// finds them idle. This is the same failure mode CLAUDE.md flags for chips_per_host in
// the UVA selector, so it gets the same treatment -- publish the constants, compare
// them, refuse on mismatch.
// ---------------------------------------------------------------------------
constexpr uint64_t kRegionMagic = 0x543648'4F535456ull;  // "T6HOSTV" -- distinctive in a hex dump

struct RegionHeader {
    uint64_t magic;             // kRegionMagic
    uint32_t version;           // kCtrlVersion -- header and control word version together
    uint32_t provisioned_cores; // kProvisionedCores
    uint64_t arena_bytes;       // kArenaBytes
    uint64_t arena_stride;      // kArenaStride
    uint64_t bank_bytes;        // kBankBytes
    uint64_t arena_array_offset;// kArenaArrayOffset
    uint32_t cores_in_use;      // the prefix actually pinned and armed
    uint32_t host_id;           // this host's identifier in the UVA selector's host field
    uint32_t chips_per_host;    // the UVA selector stride -- see host_uva.hpp
    uint32_t chip;              // which chip on this host this region serves
    // The two fields a peer must agree with or it addresses the wrong core entirely.
    // grid_width is here for the same reason chips_per_host is: a mismatch does not
    // corrupt an offset, it silently names a different core, and both sides then read a
    // bank that is legitimately idle.
    uint32_t grid_width;
    uint32_t grid_height;
    uint64_t pinned_bytes;      // pinned_bytes_for(cores_in_use)
    uint64_t device_io_base;    // PinnedMemory::get_noc_addr().addr -- what the T6 writes to
    uint32_t pcie_xy_enc;       // PinnedMemory::get_noc_addr().pcie_xy_enc
    uint32_t reserved;
};

static_assert(sizeof(RegionHeader) <= kHeaderBytes, "the header must fit its page");
// Placed here rather than beside kSuppressProbeOffset because RegionHeader is defined below
// it. The probe line is written by a PEER over RMA, so overlapping a header field would let a
// remote write corrupt the geometry every consistency check in this file depends on.
static_assert(kSuppressProbeOffset >= sizeof(RegionHeader), "the probe line must not overlap the header fields");

}  // namespace tt::tt_metal::experimental
