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

/* Active images placed in gddr (ld/x280-gddr.ld; x280_add_image PLACEMENT) run from
 * the low 8 MiB of the top 16 MiB of the local GDDR tile, through the cached
 * alias. Metal's DRAM allocator never hands out a bank address at or above
 * MEM_DRAM_SIZE (0xFF000000, Blackhole dev_mem_map.h). A NoC access to the
 * DRAM tile at or above NOC_REG_SPACE_START_ADDR (0xFF000000, Blackhole
 * noc/noc_parameters.h) reaches the tile's NoC registers instead, so the host
 * writes these images through the L2CPU tile, whose own port reaches the DRAM.
 * L2CPU tiles 2 and 3 share one GDDR tile (tt-isa-documentation
 * BlackholeA0/L2CPUTile/README.md), so only tiles 0 and 1 may hold such an image. Each hart's stack
 * tops at REGION_END - hart * STACK_STRIDE, and the image ends at or below
 * IMAGE_END, under all four. The stride is one L3 way plus 4 KiB, so, taking
 * an address's L3 index as the address mod 128 KiB, the top 4 KiB of each
 * hart's stack takes its own index range, and the load address skips the
 * indices the pinned input occupies. */
#define X280_ACTIVE_GDDR_REGION_BASE (X280_LOCAL_DDR_TILE_BASE + X280_ABI_U64(0xFF000000))
#define X280_ACTIVE_GDDR_REGION_SIZE X280_ABI_U64(0x00800000)
#define X280_ACTIVE_GDDR_REGION_END (X280_ACTIVE_GDDR_REGION_BASE + X280_ACTIVE_GDDR_REGION_SIZE)
#define X280_ACTIVE_GDDR_LOAD_ADDR (X280_ACTIVE_GDDR_REGION_BASE + X280_ABI_U64(0x2000))
#define X280_ACTIVE_GDDR_STACK_STRIDE X280_ABI_U64(0x21000)
#define X280_ACTIVE_GDDR_IMAGE_END (X280_ACTIVE_GDDR_REGION_END - 4 * X280_ACTIVE_GDDR_STACK_STRIDE)

/* The last 16 bytes of every active image: this magic, then the image's load
 * address (boot/entry.S, section .x280_image_tag). */
#define X280_ACTIVE_IMAGE_TAG_MAGIC X280_ABI_U64(0x58323830494D4721)

/* Per-hart L2 prefetcher controls, 32-bit registers at X280_L2PF_BASE +
 * (hart << X280_L2PF_HART_SHIFT) (tt-isa-documentation BlackholeA0/L2CPUTile/Caches.md
 * § L2 Cache). */
#define X280_L2PF_BASE X280_ABI_U64(0x02030000)
#define X280_L2PF_HART_SHIFT X280_ABI_U32(13)
#define X280_L2PF_BASIC_CONTROL_OFFSET X280_ABI_U32(0x0)
#define X280_L2PF_USER_CONTROL_OFFSET X280_ABI_U32(0x4)

#endif /* X280_ADDRESS_MAP_H */
