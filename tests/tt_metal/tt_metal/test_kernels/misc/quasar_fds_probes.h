// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/debug/dprint.h"
#include "overlay/rocc_instructions.hpp"

// Diagnostics shared by the Quasar FDS bring-up kernels (quasar_dispatch_engine_signal.cpp,
// quasar_fds_worker_signal.cpp).
//
// A plain write-then-read of an FDS register cannot tell three very different situations apart:
// a real hardware register, a block of read/write storage with no signalling logic behind it, and
// a custom instruction the processor quietly ignores. The last one is the treacherous case.
// FDS_INTF_READ names an uninitialised register as the instruction's destination, so an
// instruction that does nothing leaves that register holding whatever the compiler last put
// there, which is very often the value that was just written. The result looks exactly like a
// successful readback.
//
// Each probe below is shaped so that the three situations produce three different answers, and
// each restores what it disturbed. The probes report the register address rather than a name
// because the format string must be a compile-time literal.
namespace quasar_fds_probe {

// A real register drops the bits that lie outside its field. Writing a value wider than the field
// therefore separates a real register from storage and from a read that never happened.
inline void field_truncation(uint32_t addr, uint32_t field_mask) {
    const uint32_t original = static_cast<uint32_t>(FDS_INTF_READ(addr));
    FDS_INTF_WRITE(addr, 0xFFFFFFFF);
    const uint32_t observed = static_cast<uint32_t>(FDS_INTF_READ(addr));
    FDS_INTF_WRITE(addr, original);
    DPRINT(
        "[FDS probe] truncation at {:#x}: wrote 0xffffffff, read back {:#x}, field is {:#x}\n",
        addr,
        observed,
        field_mask);
    if (observed == field_mask) {
        DPRINT("[FDS probe]   truncated to the field width - a real register is answering\n");
    } else if (observed == 0xFFFFFFFF) {
        DPRINT("[FDS probe]   every bit survived - storage, or a read that never happened\n");
    } else {
        DPRINT("[FDS probe]   neither the field nor the whole value - semantics unknown\n");
    }
}

// Write two different values to two different registers, then read the first one back. A real
// read returns the first value. A read that is being ignored returns whatever the destination
// register still holds, which is the second value or something unrelated.
inline void cross_address(uint32_t first_addr, uint32_t second_addr) {
    constexpr uint32_t first_value = 0x11;
    constexpr uint32_t second_value = 0x22;
    FDS_INTF_WRITE(first_addr, first_value);
    FDS_INTF_WRITE(second_addr, second_value);
    const uint32_t observed = static_cast<uint32_t>(FDS_INTF_READ(first_addr));
    FDS_INTF_WRITE(first_addr, 0);
    FDS_INTF_WRITE(second_addr, 0);
    DPRINT(
        "[FDS probe] cross address: wrote {:#x} to {:#x} and {:#x} to {:#x}, read back {:#x}\n",
        first_value,
        first_addr,
        second_value,
        second_addr,
        observed);
    if (observed == first_value) {
        DPRINT("[FDS probe]   returned the first register - reads are addressing correctly\n");
    } else if (observed == second_value) {
        DPRINT("[FDS probe]   returned the second value - no read is happening, only a stale register\n");
    } else {
        DPRINT("[FDS probe]   returned neither value - reads are not landing where expected\n");
    }
}

// A group id that neither the handshake nor the other probes touch, used to settle whether the
// register block is one instance per processor or one shared by the whole tile. Every processor
// stamps its own index into this group's count threshold early and reads the register back at the
// very end, by which time every processor has certainly written. Private blocks return each
// processor its own stamp; a shared block returns whichever processor wrote last to all of them.
//
// This is the question the placement sweeps could not answer. Identical results from every
// processor mean the same thing whether the block is shared or merely wired identically, so those
// runs never distinguished the two.
constexpr uint32_t kSharednessGroup = 13;

// Base for that stamp. The count threshold is an eight-bit field on both sides, so the stamp is
// the base plus the processor index, which is distinctive enough not to be confused with any other
// value this test writes.
constexpr uint32_t kStampBase = 0xA0;

// Deglitcher settings for each side to try in turn on its own receive path. Zero is the register's
// reset value and most likely means no filtering at all; every run so far has overwritten it with
// 1 on both sides, making it the one setting on the receive path that has never been varied. A
// deglitcher that rejects an assertion is indistinguishable from an idle lane, so this has to be
// cleared before the register maps can be called exhausted.
//
// Both the go and the done are held rather than pulsed, so a receiver can walk this list and retry
// each value against a signal that is still being driven, with no need for the far side to
// re-send. Phase 0 is the reset value, so the most likely setting is in force from the start.
constexpr uint32_t kFilterSweep[] = {0, 1, 2, 8, 64};
constexpr uint32_t kNumFilterPhases = sizeof(kFilterSweep) / sizeof(kFilterSweep[0]);

// Addresses whose reported widths tell the two register maps apart, and tell a decoded address
// from an aliased one. In the NEO map, group status and group enable are three bits wide where
// the dispatch map has thirty-two at the corresponding places, so a three-bit answer at 0x14 or
// 0x54 identifies the NEO map and a thirty-two-bit answer at 0x288 or 0x2C8 identifies the
// dispatch map. A processor that answers for both hosts both. The 0x400 and 0x800 entries show
// how far the address decode reaches, and the 0x1000 and 0x2000 entries test the per-hart stride
// that the CORE_OFFSET macro in fds_functions.hpp hints at: if the file repeats there, a
// processor can reach a bank other than its own.
//
// Only group 0 registers appear here, so the sweep never touches the group the handshake uses.
constexpr uint32_t kMapSweepAddresses[] = {
    // NEO map: inbox 0, outbox, filter, group enable, group count threshold.
    0x000,
    0x00C,
    0x010,
    0x054,
    0x094,
    // Dispatch map: outbox, inbox 0, filter, group status, group enable, group count threshold.
    0x200,
    0x204,
    0x284,
    0x288,
    0x2C8,
    0x308,
    // Reach of the address decode.
    0x400,
    0x454,
    0x800,
    // Per-hart stride.
    0x1054,
    0x12C8,
    0x2054,
};

// Report the width every address in the sweep truncates a full-width write to. Width is what
// identifies a register, so the pattern across these addresses says which map or maps this
// processor hosts, how far addresses are decoded, and whether the file repeats per hart. Every
// address is left holding what it held.
inline void address_map() {
    for (uint32_t addr : kMapSweepAddresses) {
        const uint32_t original = static_cast<uint32_t>(FDS_INTF_READ(addr));
        FDS_INTF_WRITE(addr, 0xFFFFFFFF);
        const uint32_t observed = static_cast<uint32_t>(FDS_INTF_READ(addr));
        FDS_INTF_WRITE(addr, original);
        DPRINT("[FDS map] {:#x}: held {:#x}, full write reads back {:#x}\n", addr, original, observed);
    }
}

}  // namespace quasar_fds_probe
