// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Canonical X280 addresses shared by host code, firmware, and preprocessed
 * assembly. Keep this header limited to integer constant expressions.
 */
#ifndef X280_ADDRESS_MAP_H
#define X280_ADDRESS_MAP_H

#ifdef __ASSEMBLER__
#define X280_ABI_U32(value) value
#define X280_ABI_U64(value) value
#else
#define X280_ABI_U32(value) value##U
#define X280_ABI_U64(value) value##ULL
#endif

/* L3 local instruction memory. */
#define X280_LIM_BASE X280_ABI_U64(0x08000000)
#define X280_LIM_SIZE X280_ABI_U64(0x001E0000)
#define X280_LIM_END (X280_LIM_BASE + X280_LIM_SIZE)

/* Resident-idle and active-firmware regions. */
#define X280_IDLE_FW_LOAD_ADDR X280_LIM_BASE
#define X280_IDLE_FW_REGION_SIZE X280_ABI_U64(0x00001000)
#define X280_ACTIVE_FW_LOAD_ADDR (X280_LIM_BASE + X280_IDLE_FW_REGION_SIZE)
#define X280_ACTIVE_FW_REGION_END X280_ABI_U64(0x08120000)

/* Packet-injection LIM layout constants. */
#define X280_LIM_PACKET_INJECTION_ARENA_BASE X280_ABI_U64(0x08020000)
#define X280_LIM_PACKET_INJECTION_ARENA_END X280_ABI_U64(0x080FFF80)
#define X280_LIM_PACKET_INJECTION_ARENA_BYTES X280_ABI_U64(917376)
#define X280_LIM_PACKET_INJECTION_IMAGE_END X280_ABI_U64(0x08020000)
#define X280_LIM_PACKET_INJECTION_IMAGE_MAX_BYTES X280_ABI_U64(0x0001F000)

#if defined(__cplusplus)
static_assert(
    X280_ACTIVE_FW_LOAD_ADDR + X280_LIM_PACKET_INJECTION_IMAGE_MAX_BYTES <= X280_LIM_PACKET_INJECTION_IMAGE_END,
    "packet active image maximum must stay inside its reservation");
static_assert(
    X280_LIM_PACKET_INJECTION_IMAGE_END <= X280_LIM_PACKET_INJECTION_ARENA_BASE,
    "packet active image must end before the packet LIM arena");
static_assert(
    X280_LIM_PACKET_INJECTION_ARENA_BASE + X280_LIM_PACKET_INJECTION_ARENA_BYTES <= X280_LIM_PACKET_INJECTION_ARENA_END,
    "packet LIM arena extent must stay inside its reservation");
#elif !defined(__ASSEMBLER__)
_Static_assert(
    X280_ACTIVE_FW_LOAD_ADDR + X280_LIM_PACKET_INJECTION_IMAGE_MAX_BYTES <= X280_LIM_PACKET_INJECTION_IMAGE_END,
    "packet active image maximum must stay inside its reservation");
_Static_assert(
    X280_LIM_PACKET_INJECTION_IMAGE_END <= X280_LIM_PACKET_INJECTION_ARENA_BASE,
    "packet active image must end before the packet LIM arena");
_Static_assert(
    X280_LIM_PACKET_INJECTION_ARENA_BASE + X280_LIM_PACKET_INJECTION_ARENA_BYTES <= X280_LIM_PACKET_INJECTION_ARENA_END,
    "packet LIM arena extent must stay inside its reservation");
#endif

/* Standalone-image trap diagnostics. */
#define X280_TRAP_DIAG_STRIDE X280_ABI_U64(0x20)
#define X280_IDLE_TRAP_DIAG_ADDR X280_ABI_U64(0x08000F80)
#define X280_ACTIVE_TRAP_DIAG_ADDR X280_ABI_U64(0x0811FF80)
#define X280_TRAP_DIAG_ADDR(base, hart) ((base) + (hart) * X280_TRAP_DIAG_STRIDE)

/* Host-owned marker seeded while priming LIM ECC. */
#define X280_ECC_PRIME_MARKER_OFFSET X280_ABI_U64(0x00000800)
#define X280_ECC_PRIME_MARKER_ADDR (X280_LIM_BASE + X280_ECC_PRIME_MARKER_OFFSET)
#define X280_ECC_PRIME_MARKER_MAGIC X280_ABI_U64(0xECC17E57C0DED00D)

/* Beside the marker: bytes from X280_LIM_BASE the prime walked, or zero for a
 * marker seeded without an extent. */
#define X280_ECC_PRIME_EXTENT_OFFSET (X280_ECC_PRIME_MARKER_OFFSET + 8)
#define X280_ECC_PRIME_EXTENT_ADDR (X280_LIM_BASE + X280_ECC_PRIME_EXTENT_OFFSET)

/* Memory Port aperture onto the DRAM tile attached to this X280, and the
 * core-local interrupt controller. Naming the tile is deliberate: the aperture
 * covers that one tile, and it is not the route to any other -- those are NOC
 * targets, not further offsets from this base. */
#define X280_LOCAL_DDR_TILE_BASE X280_ABI_U64(0x400030000000)
#define X280_CLINT_BASE X280_ABI_U64(0x02000000)
#define X280_MSIP_BASE X280_CLINT_BASE

#endif /* X280_ADDRESS_MAP_H */
