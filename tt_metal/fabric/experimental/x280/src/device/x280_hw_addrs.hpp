// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side X280 hardware map: the addresses a host needs to program an L2CPU
// that the firmware itself never sees, so they do not belong in the shared
// include/x280_address_map.h ABI header.
//
// Two address spaces, and mixing them up silently targets the wrong thing:
//
//   ARC core     reset unit (0x8003_xxxx) and PLL4 (0x8002_xxxx)
//   L2CPU tile   L3 cache CSRs (0x0201_xxxx), Zero Device (0x0A00_0000),
//                LIM (0x0800_0000), and the per-hart register file
//
// Private to the component: not installed, not visible to callers.

#pragma once

#include <cstdint>

namespace x280 {

// --- ARC tile -------------------------------------------------------------

// Reset unit. Bit (4 + N) is hardware L2CPU N; set means out of reset.
// Releasing is one-way: harts can leave reset only once per ASIC-reset
// lifetime (tt-isa-documentation BlackholeA0/L2CPUTile/README.md).
constexpr uint64_t kResetUnitBase = 0x80030000ull;
constexpr uint64_t kL2cpuResetReg = kResetUnitBase + 0x14;

// PLL4 drives all four L2CPU tiles. Direct MMIO programming through the ARC
// tile is the bring-up method, not a workaround. CMFW also owns this
// PLL via its power-setting message, so no such message may be issued while a
// slew is in flight.
constexpr uint64_t kPll4Base = 0x80020500ull;
constexpr uint64_t kPll4Cntl1 = kPll4Base + 0x04;  // refdiv:8 | postdiv:8 | fbdiv:16
constexpr uint64_t kPll4Cntl5 = kPll4Base + 0x14;  // postdiv0..3, one byte each

// --- L2CPU tile -----------------------------------------------------------

constexpr uint64_t kL3CacheBase = 0x02010000ull;
// 4 bits, increase-only. Writing N gives N+1 cache ways and leaves
// (15-N) * 128 KiB of LIM. Reverting needs a full ASIC reset.
constexpr uint64_t kL3CacheWayEnable = kL3CacheBase + 0x008;
constexpr uint64_t kL3CacheWayMaskBase = kL3CacheBase + 0x800;
constexpr uint32_t kL3CacheNumMasters = 38;
constexpr uint32_t kL3WayMaskPinWay15 = 0x8000;
constexpr uint64_t kL3WayMaskStride = 8;

// Cached alias used to walk the pinned way while priming ECC.
constexpr uint64_t kZeroDeviceBase = 0x0A000000ull;

// Peripheral port: the tile's own outbound NOC TLB config block, 224 entries of
// 16 bytes. Same value as NOC_TLB_2M_CONFIG_BASE in x280/include/noc.h, which
// firmware reaches as a local load/store; the host reaches it over the NOC.
//
// Well outside the 2 MiB static TLB window at LIM base, so this goes through
// UMD's dynamic-TLB path, as the reset vectors do.
constexpr uint64_t kNocTlb2mConfigBase = 0x2FF00000ull;

// Per-hart reset vectors, two 32-bit halves per hart. This sits far outside the
// 2 MiB static TLB anchored at LIM base, so it reaches the tile through UMD's
// dynamic-TLB path rather than the static window.
constexpr uint64_t kL2cpuRegBase = 0xFFFFF7FEFFF10000ull;
constexpr uint64_t kResetVectorStride = 8;  // per hart: {lo, hi}

// 64 B general-purpose scratch, x280 physical 0x2001_0100, at its high alias
// (tt-isa-documentation BlackholeA0/L2CPUTile/MemoryMap.md, "External
// peripherals"). Survives Metal and runtime opens; tt-smi -r clears it
// (observed; the documentation gives no reset value).
constexpr uint64_t kL2cpuScratchBase = kL2cpuRegBase + 0x100;
constexpr uint64_t kL2cpuScratchBytes = 64;

// --- Priming --------------------------------------------------------------

// Prime the full pinned 128 KiB L3 piece. Instruction fetch from an unprimed
// line raises an uncorrectable ECC fault even when host NoC write/readback
// succeeds. Priming more requires a documented piece-to-LIM mapping.
constexpr uint32_t kLimEccPrimeBytes = 0x20000;
constexpr uint32_t kLimEccPrimeLineSize = 64;

// Extent assumed for a marker that carries no extent word: 16 KiB. Nothing in
// the record says what extent left a bare marker, so this can admit more than
// was primed, or refuse what was. It bounds images on such a tile in place of
// kLimEccPrimeBytes.
constexpr uint32_t kLimEccUnrecordedPrimeBytes = 0x4000;

}  // namespace x280
