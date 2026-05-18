// SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FIX S8 (Boot Fence) + FIX S9 (Idempotent Init / Session ID Tagging)
// Shared constants between host (device.cpp / fabric_init.cpp) and firmware (active_erisc.cc).
//
// These live in the AERISC_FABRIC_SCRATCH region (28 bytes, 7 words), which is reserved
// but unused by fabric router firmware.  Both WH and BH define this region identically.
//
// Memory layout within AERISC_FABRIC_SCRATCH (offset from MEM_AERISC_FABRIC_SCRATCH_BASE):
//   +0x00: boot_fence   (S8)   — host writes BOOT_FENCE_READY after all L1 writes
//   +0x04: session_id   (S9)   — monotonic session counter, 0 = invalid/stale
//   +0x08: fw_ready     (SA-A) — ERISC writes FW_READY_VALUE when init complete
//   +0x0C..+0x1B: reserved for future use

#pragma once

#include <cstdint>

// =============================================================================
// S8: Boot Fence
// =============================================================================
// The host writes BOOT_FENCE_READY to BOOT_FENCE_OFFSET (relative to
// MEM_AERISC_FABRIC_SCRATCH_BASE) after ALL L1 writes are complete — firmware
// binary, launch_msg, go_msg, handshake_bypass — so the ERISC base firmware
// (active_erisc.cc) can poll for this token before entering the go_messages
// dispatch loop.  This replaces the fragile FIX DW (50ms sleep) + FIX DU
// (ROM postcode poll) with an explicit host→firmware synchronization.
//
// L1 clear (addresses_to_clear) zeros the scratch region at session start,
// so boot_fence starts at 0.  Firmware spins until it sees BOOT_FENCE_READY.

#define BOOT_FENCE_OFFSET       0u    // offset within AERISC_FABRIC_SCRATCH
#define BOOT_FENCE_READY_VALUE  0xB00F0001u

// =============================================================================
// S9: Session ID (Idempotent Init)
// =============================================================================
// The host writes a monotonic session_id (>= 1) to SESSION_ID_OFFSET before
// deassert.  Firmware reads this after boot fence and stores it as the
// expected session_id for the remainder of the session.  go_msg and launch_msg
// from previous sessions (session_id == 0 or != expected) are rejected.
//
// Session ID 0 is reserved as "invalid/stale" — any L1 region zeroed by
// addresses_to_clear will have session_id == 0, which firmware treats as
// "no valid session yet".

#define SESSION_ID_OFFSET       4u    // offset within AERISC_FABRIC_SCRATCH
#define SESSION_ID_INVALID      0u

// =============================================================================
// SA-A: Firmware-Side Ready Gate
// =============================================================================
// Two-way handshake between host and ERISC.  After ERISC completes its init
// (flag_disable, go_messages setup in active_erisc.cc), it writes FW_READY_VALUE
// to FW_READY_OFFSET.  The host polls for this token before writing the boot
// fence token — ensuring ERISC has actually exited ROM and initialized before
// the host releases the boot fence.  Without this, the host writes boot_fence
// immediately after deassert (before ERISC even exits ROM), creating a timing
// dependency on ROM link training speed.
//
// If ERISC never writes FW_READY_VALUE (crash, stuck in ROM, link training >10s),
// the host detects this via FW_READY_TIMEOUT_MS and marks the channel dead.

#define FW_READY_OFFSET         8u                // offset within AERISC_FABRIC_SCRATCH (after boot_fence + session_id)
#define FW_READY_VALUE          0xFEED1AB5u       // ERISC writes this when init complete
#define FW_READY_TIMEOUT_MS     10000u            // host waits up to 10s for ERISC init

// =============================================================================
// Convenience: absolute L1 address computation (requires dev_mem_map.h)
// =============================================================================
// Usage from firmware:
//   #include "dev_mem_map.h"
//   #include "hostdev/fabric_boot_fence.h"
//   volatile uint32_t* boot_fence = (volatile uint32_t*)(MEM_AERISC_FABRIC_SCRATCH_BASE + BOOT_FENCE_OFFSET);
//
// Usage from host:
//   auto scratch_base = eth_l1_mem::address_map::AERISC_FABRIC_SCRATCH_BASE;
//   uint32_t boot_fence_addr = scratch_base + BOOT_FENCE_OFFSET;
