// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains the memory map for the tensix device
//
// It is included on the device, on the host and in linker scripts to
// serve as a single source of truth for memory layout for both hw design and
// sw convention.  The requirement of including this in linker scripts
// requires that everything within be handled by the C pre-processor, hence
// the use of #define
//
// Before adding a define here, read the following:
// 1) Any "truly global" address must be specified explicitly here.  Truly
// global addresses are addresses that are referenced on both the host and
// device or between processors
// 2) Memory section sizes must be specified here, these are used in the
// linker scripts
// 3) static/global variables generally should NOT be listed here.  If
// they are global to a processor, declare them in the that processor's source
// code, they will get placed in local memory
// 4) L1 data sections are no longer supported as addressing them with XIP
// binaries requires runtime address patching.  Instead of using named
// variables in the L1 data section use a mailbox (or address in the mailbox
// range and initialize explicitly)
//

/////////////
// RISC-V Address map definition (hardware)
#define MEM_L1_BASE 0x0
#define MEM_L1_SIZE (4 * 1024 * 1024)
// Note: RISC-V LR/SC and AMO atomics don't work with uncached L1 (they hang).
// Both require the cache coherence system. Only plain loads/stores with fences work uncached.
#define MEM_L1_UNCACHED_BASE (MEM_L1_BASE + MEM_L1_SIZE)  // upper 4MBs bypass cache

#define MEM_ETH_BASE 0x0
// Top 64K is reserved for syseng but host reads/writes from that region
#define MEM_ETH_SIZE (512 * 1024)

#define MEM_DRAM_SIZE (4177920 * 1024U)
#define MEM_LOCAL_BASE 0x0802000

// Memory for (dram/l1)_bank_to_noc_xy arrays, size needs to be at least 2 * NUM_NOCS * (NUM_DRAM_BANKS + NUM_L1_BANKS)
#define MEM_BANK_TO_NOC_XY_SIZE 1024
// Memory for bank_to_dram_offset and bank_to_l1_offset arrays, size needs to be at least 4 * (NUM_DRAM_BANKS +
// NUM_L1_BANKS)
#define MEM_BANK_OFFSET_SIZE 1024

/////////////
// Firmware/kernel code holes
#define MEM_DM_FIRMWARE_SIZE (1024 * 12)
#define MEM_TRISC_FIRMWARE_SIZE (1024 * 5)
#define MEM_DM_KERNEL_SIZE (1024 * 48)
#define MEM_TRISC_KERNEL_SIZE (1024 * 24)
#define MEM_DM_GLOBAL_SIZE (1024 * 2)
#define MEM_TRISC_GLOBAL_SIZE (1024 * 2)
#define MEM_DM_LOCAL_SIZE (1024 * 8)
#define MEM_TRISC_LOCAL_SIZE (1024 * 4)
#define MEM_TRISC_LOCAL_OFFSET (0x2000)

#define NUM_DM_CORES 8
#define NUM_TRISC_CORES 4

#define MEM_TRISC0_KERNEL_SIZE (24 * 1024)
#define MEM_TRISC1_KERNEL_SIZE (24 * 1024)
#define MEM_TRISC2_KERNEL_SIZE (24 * 1024)
#define MEM_TRISC3_KERNEL_SIZE (24 * 1024)

#define MEM_LLK_DEBUG_SIZE 1024

#define MEM_BOOT_CODE_BASE 0
#define MEM_NOC_ATOMIC_RET_VAL_ADDR 4
#define MEM_L1_BARRIER 12

// Used by ARC FW and LLKs to store power throttling state
#define MEM_L1_ARC_FW_SCRATCH 16
#define MEM_L1_ARC_FW_SCRATCH_SIZE 16

// On Blackhole issuing inline writes and atomics requires all 4 memory ports to accept the transaction at the same
// time. If one port on the recipient has no back-pressure then the transaction will hang because there is no mechanism
// to allow one memory port to move ahead of another. To workaround this hang, we emulate inline writes on Blackhole by
// writing the value to be written to local L1 first and then issue a noc async write.
// Each noc has 16B to store value written out by inline writes.
// Base address for each noc to store the value to be written will be `MEM_{E,B,NC}RISC_L1_INLINE_BASE + (noc_index *
// 16)`
#define MEM_L1_INLINE_SIZE_PER_NOC 16
#define MEM_L1_INLINE_BASE 32  // MEM_L1_ARC_FW_SCRATCH + MEM_L1_ARC_FW_SCRATCH_SIZE

// Hardcode below due to compiler bug that cannot statically resolve the expression see GH issue #19265
#define MEM_MAILBOX_BASE 16
#define UNCACHED_MEM_MAILBOX_BASE (0x400010)  // workaround for https://github.com/tenstorrent/tt-metal/issues/19265
// Magic size must be big enough to hold dev_msgs_t.  static_asserts will fire if this is too small
#define MEM_MAILBOX_SIZE 58480
#define MEM_MAILBOX_END (MEM_MAILBOX_BASE + MEM_MAILBOX_SIZE)

#define MEM_LLK_DEBUG_BASE ((MEM_MAILBOX_END + 31) & ~31)

#define MEM_DM0_LOCAL_BASE (MEM_DM_FIRMWARE_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_DM1_LOCAL_BASE (MEM_DM0_LOCAL_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_DM2_LOCAL_BASE (MEM_DM1_LOCAL_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_DM3_LOCAL_BASE (MEM_DM2_LOCAL_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_DM4_LOCAL_BASE (MEM_DM3_LOCAL_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_DM5_LOCAL_BASE (MEM_DM4_LOCAL_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_DM6_LOCAL_BASE (MEM_DM5_LOCAL_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_DM7_LOCAL_BASE (MEM_DM6_LOCAL_BASE + MEM_DM_FIRMWARE_SIZE)

#define MEM_INTERRUPT_TABLE_BASE ((MEM_LLK_DEBUG_BASE + MEM_LLK_DEBUG_SIZE + 255) & ~255)
#define MEM_INTERRUPT_TABLE_SIZE (64 * 4)

#define MEM_DM_FIRMWARE_BASE (MEM_INTERRUPT_TABLE_BASE + MEM_INTERRUPT_TABLE_SIZE)
#define MEM_TRISC0_FIRMWARE_BASE (MEM_DM_FIRMWARE_BASE + MEM_DM_FIRMWARE_SIZE)
#define MEM_TRISC1_FIRMWARE_BASE (MEM_TRISC0_FIRMWARE_BASE + MEM_TRISC_FIRMWARE_SIZE)
#define MEM_TRISC2_FIRMWARE_BASE (MEM_TRISC1_FIRMWARE_BASE + MEM_TRISC_FIRMWARE_SIZE)
#define MEM_TRISC3_FIRMWARE_BASE (MEM_TRISC2_FIRMWARE_BASE + MEM_TRISC_FIRMWARE_SIZE)
#define MEM_DM_GLOBAL_BASE (MEM_TRISC3_FIRMWARE_BASE + MEM_TRISC_FIRMWARE_SIZE)
#define MEM_TRISC0_GLOBAL_BASE (MEM_DM_GLOBAL_BASE + MEM_DM_GLOBAL_SIZE * NUM_DM_CORES + MEM_DM_GLOBAL_SIZE)
#define MEM_TRISC1_GLOBAL_BASE (MEM_TRISC0_GLOBAL_BASE + MEM_TRISC_GLOBAL_SIZE*2)
#define MEM_TRISC2_GLOBAL_BASE (MEM_TRISC1_GLOBAL_BASE + MEM_TRISC_GLOBAL_SIZE*2)
#define MEM_TRISC3_GLOBAL_BASE (MEM_TRISC2_GLOBAL_BASE + MEM_TRISC_GLOBAL_SIZE*2)
#define MEM_DM_LOCAL_BASE (MEM_TRISC3_GLOBAL_BASE + MEM_TRISC_GLOBAL_SIZE*2)
// kernels are loaded as part of kernel_config at MEM_MAP_END or after
// linker needs an address that doesn't overlap any of the FW and data sections
// so just give an address outside of physical memory
#define MEM_KERNEL_BASE (4 * 1024 * 1024)

#define MEM_TRISC_LOCAL_BASE (MEM_LOCAL_BASE)

#define MEM_NOC_COUNTER_SIZE 4
#define MEM_NOC_COUNTER_L1_SIZE (5 * 2 * 2 * MEM_NOC_COUNTER_SIZE)
#define MEM_NOC_COUNTER_BASE TT_L1_MOVABLE_BASE(__l1_noc_counter, L1Region::NOC_COUNTER)

// Fabric transaction counters (similar to NoC counters)
// 3 barrier types × 8 DMs × 4 bytes = 96 bytes; +16 bytes padding = 112 bytes total (16-byte aligned, Quasar has 8 DMs)
#define MEM_FABRIC_COUNTER_SIZE 4
#define MEM_FABRIC_COUNTER_L1_SIZE (3 * 8 * MEM_FABRIC_COUNTER_SIZE + 16)
#define MEM_FABRIC_COUNTER_BASE TT_L1_MOVABLE_BASE(__l1_fabric_counter, L1Region::FABRIC_COUNTER)

// Fabric connection sync region for synchronization across RISCs (BRISC/NCRISC)
// Contains: lock (4) + initialized (4) + connection object (128) + padding (8) = 144 bytes (16-byte aligned)
#define MEM_FABRIC_CONNECTION_LOCK_SIZE 144
#define MEM_FABRIC_CONNECTION_LOCK_BASE \
    TT_L1_MOVABLE_BASE(__l1_fabric_connection_lock, L1Region::FABRIC_CONNECTION_LOCK)

// Tensix routing table for fabric networking
#define MEM_TENSIX_ROUTING_TABLE_BASE TT_L1_MOVABLE_BASE(__l1_routing_table, L1Region::ROUTING_TABLE)
#define MEM_ROUTING_TABLE_SIZE 2576  // struct layout: base(516) + union(1024) + exit(1024) + pad(12)
#define MEM_OFFSET_OF_ROUTING_PATHS 516
#define MEM_ROUTING_TABLE_PADDING 12

#define ROUTING_PATH_SIZE_1D 1024  // Was 256 (64 chips × 16 bytes)
// 2D uncompressed size is too large to fit in L1 memory
#define COMPRESSED_ROUTING_PATH_SIZE_1D 0     // sizeof(intra_mesh_routing_path_t<1, true>)
#define COMPRESSED_ROUTING_PATH_SIZE_2D 1024  // sizeof(intra_mesh_routing_path_t<2, true>)
// Union: 1D and 2D routing tables share the same offset
#define MEM_TENSIX_ROUTING_PATH_BASE TT_L1_MOVABLE_BASE(__l1_routing_path, L1Region::ROUTING_PATH)
#define MEM_TENSIX_ROUTING_PATH_BASE_1D MEM_TENSIX_ROUTING_PATH_BASE  // 516
#define MEM_TENSIX_ROUTING_PATH_BASE_2D MEM_TENSIX_ROUTING_PATH_BASE  // 516
#define MEM_TENSIX_ROUTING_PATH_SIZE 1024                             // max(1024, 1024)

#define MEM_TENSIX_EXIT_NODE_TABLE_BASE TT_L1_MOVABLE_BASE(__l1_exit_node_table, L1Region::EXIT_NODE_TABLE)
#define MEM_EXIT_NODE_TABLE_SIZE 1024  // sizeof(exit_node_table_t)

// Tensix fabric connection metadata for workers
#define MEM_TENSIX_FABRIC_CONNECTIONS_BASE TT_L1_MOVABLE_BASE(__l1_fabric_connections, L1Region::FABRIC_CONNECTIONS)
#define MEM_TENSIX_FABRIC_CONNECTIONS_SIZE 656        // sizeof(tensix_fabric_connections_l1_info_t)
#define MEM_TENSIX_FABRIC_OFFSET_OF_ALIGNED_INFO 400  // offsetof(tensix_fabric_connections_l1_info_t, read_write)

// Packet header pool sizing constants
#define PACKET_HEADER_MAX_SIZE 144  // sizeof(UDMHybridMeshPacketHeader)
#define NUM_PACKET_HEADERS \
    (6 * 2 * MaxDMProcessorsPerCoreType)  // (EAST, WEST, NORTH, SOUTH, UP, DOWN) * convention * (DM0, DM1)

// Packet header pool for fabric networking
// Size: 144 * 6 * 2 * 2 = 3456
#define MEM_PACKET_HEADER_POOL_BASE TT_L1_MOVABLE_BASE(__l1_packet_header_pool, L1Region::PACKET_HEADER_POOL)
#define MEM_PACKET_HEADER_POOL_SIZE (PACKET_HEADER_MAX_SIZE * NUM_PACKET_HEADERS)
// The 16-byte-alignment invariant is now a constexpr static_assert on kIdentityL1Layout in the
// single-definition machinery below: under that scheme the base is no longer a preprocessor constant,
// so the old preprocessor #error could not evaluate it.

// Read-only reserved memory boundary for watcher checks
#define MEM_MAP_READ_ONLY_END (MEM_TENSIX_FABRIC_CONNECTIONS_BASE + MEM_TENSIX_FABRIC_OFFSET_OF_ALIGNED_INFO)
// Read-write reserved memory boundary for watcher checks
#define MEM_MAP_END TT_L1_MOVABLE_BASE(__l1_map_end, L1Region::MAP_END)

// Every address after MEM_MAP_END is a "scratch" address
// These can be used by FW during init, but aren't usable once FW reaches "ready"

/////////////
// Initialization relocation L1 memory
// Host downloads to these addresses, fw copies to destination
// Note: using xmov to copy ncrisc to addresses above 1M hangs the chip
// TODO review need
#define MEM_DM0_INIT_LOCAL_L1_BASE_SCRATCH MEM_MAP_END
#define MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH (MEM_DM0_INIT_LOCAL_L1_BASE_SCRATCH + MEM_DM_LOCAL_SIZE)
#define MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH (MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH + MEM_TRISC_LOCAL_SIZE)
#define MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH (MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH + MEM_TRISC_LOCAL_SIZE)
#define MEM_TRISC3_INIT_LOCAL_L1_BASE_SCRATCH (MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH + MEM_TRISC_LOCAL_SIZE)

#define MEM_BANK_TO_NOC_SCRATCH TT_L1_MOVABLE_BASE(__l1_bank_to_noc_scratch, L1Region::BANK_TO_NOC_SCRATCH)
#define MEM_BANK_TO_NOC_SIZE (MEM_BANK_TO_NOC_XY_SIZE + MEM_BANK_OFFSET_SIZE)

// Scratch area for logical to virtual coordinate mapping.
// This size must match the firmware noc_size_x & noc_size Y. Size is largest chip (X + Y) * sizeof uint8_t.
// Chip sizes must round up to nearest multiple of 4 to deal with uint32_t alignment for L1 to local copies.
#define MEM_LOGICAL_TO_VIRTUAL_SCRATCH \
    TT_L1_MOVABLE_BASE(__l1_logical_to_virtual_scratch, L1Region::LOGICAL_TO_VIRTUAL_SCRATCH)
#define MEM_LOGICAL_TO_VIRTUAL_SIZE ((20 + 12) * sizeof(uint8_t))

// ---------------------------------------------------------------------------------------------
// Dynamic L1 region bases (plan B2, defsym variant / D) -- SINGLE SOURCE.
//
// tt_l1_compute_layout() below owns the movable-region order and sizing. Applied to the fixed
// reservations (kL1Reservations) it yields kIdentityL1Layout, which drives every consumer of these
// bases: (a) the host macro values here, (b) the host HAL mem-map seed (qa_hal_tensix
// create_tensix_mem_map), (c) the link-time defsym fallback (qa_hal linker_flags), and -- with
// measured firmware sizes, via the HAL packing hook -- the packed layout. There is no relational
// #define chain and no #undef: each movable base is declared ONCE via TT_L1_MOVABLE_BASE. On device
// it is the ADDRESS of an absolute linker symbol (__l1_*) that the linker fills from the packed
// layout (D1) -- a lui/addi immediate, no runtime load; on host/non-device it is the constexpr
// identity value. The bases are independent (no cascade off one another). No linker script includes
// this header or expands these macros (audited), so the non-device branch may use C++.
#if defined(FW_BUILD) || defined(KERNEL_BUILD)
#include <cstdint>
extern "C" {
extern char __l1_noc_counter[];
extern char __l1_fabric_counter[];
extern char __l1_fabric_connection_lock[];
extern char __l1_routing_table[];
extern char __l1_routing_path[];
extern char __l1_exit_node_table[];
extern char __l1_fabric_connections[];
extern char __l1_packet_header_pool[];
extern char __l1_bank_to_noc_scratch[];
extern char __l1_logical_to_virtual_scratch[];
extern char __l1_map_end[];
}
// Device: the base is the symbol's ADDRESS (the array name decays to it) -- NEVER its contents. The
// `region` argument is unused, so its tokens are never expanded: L1Region need not exist on device.
#define TT_L1_MOVABLE_BASE(sym, region) ((uint32_t)(uintptr_t)sym)
#else
#include <array>
#include <cstddef>
#include <cstdint>
#include "core_config.h"  // MaxDMProcessorsPerCoreType (via NUM_PACKET_HEADERS -> MEM_PACKET_HEADER_POOL_SIZE)
#include "hostdev/l1_layout.h"
static_assert(NUM_L1_FW_IMAGES == 1u + NUM_TRISC_CORES, "NUM_L1_FW_IMAGES out of sync with NUM_TRISC_CORES");
// The single layout routine: fixed region order + sizing, parameterized by measured firmware section
// sizes in image order {DM, TRISC0..3}. constexpr -- pure arithmetic over the MEM_*_SIZE macros above,
// no logging/throw. The HAL packing hook calls this with measured sizes; kIdentityL1Layout applies it
// to the static reservations, reproducing today's map exactly. MARGIN=0: reserve exactly the measured
// size, 16-aligned; overflow is fail-loud at relink (TrimSegments).
constexpr PackedL1Layout tt_l1_compute_layout(const std::array<FirmwareSectionSizes, NUM_L1_FW_IMAGES>& fw) {
    PackedL1Layout L;
    constexpr uint32_t kMargin = 0;
    auto au = [](uint32_t v) { return (v + kMargin + 15u) & ~15u; };
    // firmware TEXT (XIP): DM at the fixed base; each TRISC packs against the prior image.
    uint64_t p = (uint64_t)MEM_DM_FIRMWARE_BASE;
    L.fw[0].fw_text = p;
    L.fw[0].text_size = au(fw[0].text);
    p += au(fw[0].text);
    for (uint32_t i = 0; i < NUM_TRISC_CORES; ++i) {
        L.fw[1u + i].fw_text = p;
        L.fw[1u + i].text_size = au(fw[1u + i].text);
        p += au(fw[1u + i].text);
    }
    // firmware GLOBALS (.data, L1): measured fw slot + per-core kernel-data BUDGET slots (worst-case).
    constexpr L1Region trisc_global[NUM_TRISC_CORES] = {
        L1Region::TRISC0_GLOBAL, L1Region::TRISC1_GLOBAL, L1Region::TRISC2_GLOBAL, L1Region::TRISC3_GLOBAL};
    L.region_base[(std::size_t)L1Region::DM_GLOBAL] = p;
    L.fw[0].fw_data = p;
    L.fw[0].data_size = au(fw[0].data);
    L.fw[0].kn_data = p + au(fw[0].data);
    p += au(fw[0].data) + (uint64_t)NUM_DM_CORES * MEM_DM_GLOBAL_SIZE;
    for (uint32_t i = 0; i < NUM_TRISC_CORES; ++i) {
        L.region_base[(std::size_t)trisc_global[i]] = p;
        L.fw[1u + i].fw_data = p;
        L.fw[1u + i].data_size = au(fw[1u + i].data);
        L.fw[1u + i].kn_data = p + au(fw[1u + i].data);
        p += au(fw[1u + i].data) + MEM_TRISC_GLOBAL_SIZE;
    }
    // DM LOCAL (.tdata, L1): full per-core reservation. TRISC tls lives in local RAM (0x802000).
    L.region_base[(std::size_t)L1Region::DM_LOCAL] = p;
    L.fw[0].fw_tls = p;
    p += (uint64_t)MEM_DM_LOCAL_SIZE * NUM_DM_CORES;
    for (uint32_t i = 0; i < NUM_TRISC_CORES; ++i) {
        L.fw[1u + i].fw_tls = (uint64_t)MEM_TRISC_LOCAL_BASE + (uint64_t)MEM_TRISC_LOCAL_OFFSET * i;
    }
    // fixed-size system structures (bases move with the packed tail; sizes unchanged)
    L.region_base[(std::size_t)L1Region::NOC_COUNTER] = p;
    p += MEM_NOC_COUNTER_L1_SIZE;
    L.region_base[(std::size_t)L1Region::FABRIC_COUNTER] = p;
    p += MEM_FABRIC_COUNTER_L1_SIZE;
    L.region_base[(std::size_t)L1Region::FABRIC_CONNECTION_LOCK] = p;
    p += MEM_FABRIC_CONNECTION_LOCK_SIZE;
    const uint64_t routing_table = p;
    L.region_base[(std::size_t)L1Region::ROUTING_TABLE] = routing_table;
    L.region_base[(std::size_t)L1Region::ROUTING_PATH] = routing_table + MEM_OFFSET_OF_ROUTING_PATHS;
    const uint64_t exit_node = routing_table + MEM_OFFSET_OF_ROUTING_PATHS + MEM_TENSIX_ROUTING_PATH_SIZE;
    L.region_base[(std::size_t)L1Region::EXIT_NODE_TABLE] = exit_node;
    const uint64_t fabric_conn = exit_node + MEM_EXIT_NODE_TABLE_SIZE + MEM_ROUTING_TABLE_PADDING;
    L.region_base[(std::size_t)L1Region::FABRIC_CONNECTIONS] = fabric_conn;
    const uint64_t packet_pool = fabric_conn + MEM_TENSIX_FABRIC_CONNECTIONS_SIZE;
    L.region_base[(std::size_t)L1Region::PACKET_HEADER_POOL] = packet_pool;
    p = packet_pool + MEM_PACKET_HEADER_POOL_SIZE;
    // worker-L1 boundary, then the transient init scratch chain (above MEM_MAP_END)
    L.region_base[(std::size_t)L1Region::MAP_END] = p;
    L.map_end = p;
    L.region_base[(std::size_t)L1Region::INIT_LOCAL_SCRATCH] = p;
    const uint64_t bank_to_noc = p + (uint64_t)MEM_DM_LOCAL_SIZE + (uint64_t)NUM_TRISC_CORES * MEM_TRISC_LOCAL_SIZE;
    L.region_base[(std::size_t)L1Region::BANK_TO_NOC_SCRATCH] = bank_to_noc;
    L.region_base[(std::size_t)L1Region::LOGICAL_TO_VIRTUAL_SCRATCH] = bank_to_noc + MEM_BANK_TO_NOC_SIZE;
    // per-TRISC firmware-text bases (DM firmware reads these from the layout for the reset-PC)
    L.region_base[(std::size_t)L1Region::TRISC0_FW] = L.fw[1].fw_text;
    L.region_base[(std::size_t)L1Region::TRISC1_FW] = L.fw[2].fw_text;
    L.region_base[(std::size_t)L1Region::TRISC2_FW] = L.fw[3].fw_text;
    L.region_base[(std::size_t)L1Region::TRISC3_FW] = L.fw[4].fw_text;
    L.valid = true;
    return L;
}
inline constexpr std::array<FirmwareSectionSizes, NUM_L1_FW_IMAGES> kL1Reservations = {{
    {MEM_DM_FIRMWARE_SIZE, MEM_DM_GLOBAL_SIZE, MEM_DM_LOCAL_SIZE},
    {MEM_TRISC_FIRMWARE_SIZE, MEM_TRISC_GLOBAL_SIZE, MEM_TRISC_LOCAL_SIZE},
    {MEM_TRISC_FIRMWARE_SIZE, MEM_TRISC_GLOBAL_SIZE, MEM_TRISC_LOCAL_SIZE},
    {MEM_TRISC_FIRMWARE_SIZE, MEM_TRISC_GLOBAL_SIZE, MEM_TRISC_LOCAL_SIZE},
    {MEM_TRISC_FIRMWARE_SIZE, MEM_TRISC_GLOBAL_SIZE, MEM_TRISC_LOCAL_SIZE},
}};
inline constexpr PackedL1Layout kIdentityL1Layout = tt_l1_compute_layout(kL1Reservations);
inline constexpr uint32_t tt_l1_identity_base(L1Region r) {
    return (uint32_t)kIdentityL1Layout.region_base[(std::size_t)r];
}
// Alignment invariant -- formerly the preprocessor #error at the packet-header-pool base definition;
// now a constexpr static_assert since the base is no longer a preprocessor constant under the
// single-definition scheme.
static_assert(
    kIdentityL1Layout.region_base[(std::size_t)L1Region::PACKET_HEADER_POOL] % 16 == 0 &&
        MEM_PACKET_HEADER_POOL_SIZE % 16 == 0,
    "Packet header pool base and size must be 16-byte aligned");
// Host/non-device: the base is the constexpr identity value. `sym` is unused here.
#define TT_L1_MOVABLE_BASE(sym, region) (tt_l1_identity_base(region))
#endif

/////////////
// Stack info
// Stack and globals share the same piece of memory, one grows at the
// expense of the other.
#define MEM_DM_STACK_MIN_SIZE (1024 / 4)
#define MEM_TRISC_STACK_MIN_SIZE (1024 / 4)
#define MEM_NCRISC_STACK_MIN_SIZE 256
#define MEM_TRISC0_STACK_MIN_SIZE 192
#define MEM_TRISC1_STACK_MIN_SIZE 192
#define MEM_TRISC2_STACK_MIN_SIZE 256
#define MEM_ERISC_STACK_MIN_SIZE 192

#define MEM_SYSENG_RESERVED_SIZE (64 * 1024)
// This Barrier is not the same as the metal L1 barrier (MEM_L1_BARRIER)
#define MEM_ERISC_L1_BARRIER_SIZE 64
#define MEM_ERISC_APP_ROUTING_INFO_SIZE 48
#define MEM_MAX_NUM_CONCURRENT_TRANSACTIONS 8
#define MEM_ERISC_SYNC_INFO_SIZE (160 + 16 * MEM_MAX_NUM_CONCURRENT_TRANSACTIONS)
#define MEM_ERISC_FABRIC_ROUTER_RESERVED_SIZE 3088
#define MEM_AERISC_FABRIC_TELEMETRY_SIZE 128
// Routing path sizes (union = same memory, consolidated from intermediate aliases)
#define MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_1D ROUTING_PATH_SIZE_1D
#define MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_2D COMPRESSED_ROUTING_PATH_SIZE_2D
#define MEM_ERISC_FABRIC_ROUTING_PATH_SIZE MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_2D  // Union size
#define MEM_ERISC_MAILBOX_SIZE 12768
#define MEM_ERISC_KERNEL_CONFIG_SIZE (25 * 1024)
#define MEM_ERISC_BASE 0

// From the top of L1. Common.
// Start from the top and work downward
#define MEM_SYSENG_RESERVED_BASE (MEM_ETH_SIZE - MEM_SYSENG_RESERVED_SIZE)
#define MEM_ERISC_BARRIER_BASE (MEM_SYSENG_RESERVED_BASE - MEM_ERISC_L1_BARRIER_SIZE)
#define MEM_ERISC_APP_ROUTING_INFO_BASE (MEM_ERISC_BARRIER_BASE - MEM_ERISC_APP_ROUTING_INFO_SIZE)

#define MEM_ERISC_APP_SYNC_INFO_BASE (MEM_ERISC_APP_ROUTING_INFO_BASE - MEM_ERISC_SYNC_INFO_SIZE)
#define MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE (MEM_ERISC_APP_SYNC_INFO_BASE - MEM_ERISC_FABRIC_ROUTER_RESERVED_SIZE)

#define MEM_AERISC_FABRIC_TELEMETRY_BASE (MEM_ERISC_APP_SYNC_INFO_BASE - MEM_AERISC_FABRIC_TELEMETRY_SIZE)
#define MEM_AERISC_ROUTING_TABLE_SIZE MEM_ROUTING_TABLE_SIZE
#define MEM_AERISC_ROUTING_TABLE_BASE (MEM_AERISC_FABRIC_TELEMETRY_BASE - MEM_AERISC_ROUTING_TABLE_SIZE)
// AERISC fabric routing paths (union = same offset for 1D and 2D)
#define MEM_AERISC_FABRIC_ROUTING_PATH_BASE (MEM_AERISC_ROUTING_TABLE_BASE + MEM_OFFSET_OF_ROUTING_PATHS)
#define MEM_AERISC_FABRIC_ROUTING_PATH_BASE_1D MEM_AERISC_FABRIC_ROUTING_PATH_BASE
#define MEM_AERISC_FABRIC_ROUTING_PATH_BASE_2D MEM_AERISC_FABRIC_ROUTING_PATH_BASE
#define MEM_AERISC_EXIT_NODE_TABLE_BASE (MEM_AERISC_FABRIC_ROUTING_PATH_BASE + MEM_ERISC_FABRIC_ROUTING_PATH_SIZE)
#define MEM_AERISC_FABRIC_ROUTER_STATE_BASE MEM_AERISC_ROUTING_TABLE_BASE
#define MEM_AERISC_FABRIC_ROUTER_COMMAND_BASE (MEM_AERISC_ROUTING_TABLE_BASE + 16)

// This is now the maximum size available for your application
#define MEM_ERISC_MAX_SIZE MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE

// For your static assert, use the base of the lowest reserved section
#define MEM_ERISC_TOP MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE
// Common Misc
#define MEM_RETRAIN_COUNT_ADDR 0x7CC10
#define MEM_RETRAIN_FORCE_ADDR 0x1EFC

#define MEM_SYSENG_ETH_RESULTS_BASE_ADDR 0x7CC00
#define MEM_SYSENG_ETH_MAILBOX_BASE_ADDR 0x7D000

#define MEM_ERISC_LOCAL_SIZE (8 * 1024)
#define MEM_SUBORDINATE_ERISC_LOCAL_SIZE (8 * 1024)
#define MEM_ERISC_FIRMWARE_SIZE (24 * 1024)
#define MEM_SUBORDINATE_ERISC_FIRMWARE_SIZE (24 * 1024)
#define MEM_ERISC_KERNEL_SIZE (24 * 1024)
#define MEM_ERISC_RESERVED1 0
#define MEM_ERISC_RESERVED1_SIZE 1024

/////////////
// Idle ERISC memory map
// TODO: reduce this when mailbox sizes are core type aware for some members (eg watcher/dprint)
// TODO: also, move into gap above in the reserved area
#define MEM_IERISC_LOCAL_SIZE MEM_ERISC_LOCAL_SIZE
#define MEM_SUBORDINATE_IERISC_LOCAL_SIZE MEM_ERISC_LOCAL_SIZE
#define MEM_IERISC_MAILBOX_BASE (MEM_ERISC_RESERVED1 + MEM_ERISC_RESERVED1_SIZE)
#define MEM_IERISC_MAILBOX_SIZE MEM_ERISC_MAILBOX_SIZE
#define MEM_IERISC_MAILBOX_END (MEM_IERISC_MAILBOX_BASE + MEM_IERISC_MAILBOX_SIZE)
#define MEM_IERISC_L1_INLINE_BASE MEM_IERISC_MAILBOX_END
#define MEM_IERISC_L1_INLINE_END MEM_IERISC_L1_INLINE_BASE
#define MEM_IERISC_FIRMWARE_BASE MEM_IERISC_L1_INLINE_END
#define MEM_IERISC_FIRMWARE_SIZE MEM_ERISC_FIRMWARE_SIZE
#define MEM_SUBORDINATE_IERISC_FIRMWARE_BASE (MEM_IERISC_FIRMWARE_BASE + MEM_IERISC_FIRMWARE_SIZE)
#define MEM_SUBORDINATE_IERISC_FIRMWARE_SIZE MEM_ERISC_FIRMWARE_SIZE
#define MEM_SUBORDINATE_IERISC_FIRMWARE_END \
    (MEM_SUBORDINATE_IERISC_FIRMWARE_BASE + MEM_SUBORDINATE_IERISC_FIRMWARE_SIZE)

// IDLE ETH fabric routing paths - placed in dedicated reserved area after firmware
#define MEM_IERISC_ROUTING_TABLE_BASE ((MEM_SUBORDINATE_IERISC_FIRMWARE_END + 31) & ~31)
// IERISC fabric routing paths (union = same offset for 1D and 2D)
#define MEM_IERISC_FABRIC_ROUTING_PATH_BASE (MEM_IERISC_ROUTING_TABLE_BASE + MEM_OFFSET_OF_ROUTING_PATHS)
#define MEM_IERISC_FABRIC_ROUTING_PATH_BASE_1D MEM_IERISC_FABRIC_ROUTING_PATH_BASE
#define MEM_IERISC_FABRIC_ROUTING_PATH_BASE_2D MEM_IERISC_FABRIC_ROUTING_PATH_BASE
#define MEM_IERISC_FABRIC_ROUTING_PATH_END (MEM_IERISC_FABRIC_ROUTING_PATH_BASE + MEM_ERISC_FABRIC_ROUTING_PATH_SIZE)

#define MEM_IERISC_EXIT_NODE_TABLE_BASE MEM_IERISC_FABRIC_ROUTING_PATH_END
#define MEM_IERISC_EXIT_NODE_TABLE_END (MEM_IERISC_EXIT_NODE_TABLE_BASE + MEM_EXIT_NODE_TABLE_SIZE)

#define MEM_IERISC_MAP_END (MEM_IERISC_EXIT_NODE_TABLE_END + MEM_ROUTING_TABLE_PADDING)
#define MEM_IERISC_KERNEL_SIZE MEM_ERISC_KERNEL_SIZE
#define MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH MEM_IERISC_MAP_END
#define MEM_SUBORDINATE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH (MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_ERISC_LOCAL_SIZE)
#define MEM_IERISC_STACK_MIN_SIZE MEM_ERISC_STACK_MIN_SIZE
#define MEM_SUBORDINATE_IERISC_STACK_MIN_SIZE MEM_ERISC_STACK_MIN_SIZE
#define MEM_IERISC_BANK_TO_NOC_SCRATCH \
    (MEM_SUBORDINATE_IERISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_SUBORDINATE_ERISC_LOCAL_SIZE)
#define MEM_IERISC_BANK_TO_NOC_SIZE (MEM_BANK_TO_NOC_XY_SIZE + MEM_BANK_OFFSET_SIZE)

#define IERISC_RESET_PC (MEM_LOCAL_BASE | 0x14000)
#define SUBORDINATE_IERISC_RESET_PC (MEM_LOCAL_BASE | 0x14008)

/////////////
// Active ERISC memory map
// TODO: These are added here to enable aerisc compilation but are replicated in eth_l1_address_map
// eth_l1_address_map should be removed in favour of this file
#define MEM_AERISC_LOCAL_SIZE MEM_ERISC_LOCAL_SIZE
#define MEM_SUBORDINATE_AERISC_LOCAL_SIZE MEM_ERISC_LOCAL_SIZE
#define MEM_AERISC_MAILBOX_BASE (MEM_ERISC_RESERVED1 + MEM_ERISC_RESERVED1_SIZE)
#define MEM_AERISC_MAILBOX_SIZE MEM_ERISC_MAILBOX_SIZE
#define MEM_AERISC_MAILBOX_END (MEM_AERISC_MAILBOX_BASE + MEM_AERISC_MAILBOX_SIZE)
#define MEM_AERISC_L1_INLINE_BASE MEM_AERISC_MAILBOX_END
#define MEM_AERISC_L1_INLINE_END MEM_AERISC_L1_INLINE_BASE
#define MEM_AERISC_FIRMWARE_BASE MEM_AERISC_L1_INLINE_END
#define MEM_AERISC_FIRMWARE_SIZE MEM_ERISC_FIRMWARE_SIZE
#define MEM_SUBORDINATE_AERISC_FIRMWARE_BASE (MEM_AERISC_FIRMWARE_BASE + MEM_AERISC_FIRMWARE_SIZE)
#define MEM_SUBORDINATE_AERISC_FIRMWARE_SIZE MEM_ERISC_FIRMWARE_SIZE
#define MEM_AERISC_MAP_END (MEM_SUBORDINATE_AERISC_FIRMWARE_BASE + MEM_SUBORDINATE_AERISC_FIRMWARE_SIZE)
#define MEM_AERISC_KERNEL_SIZE MEM_ERISC_KERNEL_SIZE

#define MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH MEM_AERISC_MAP_END
#define MEM_SUBORDINATE_AERISC_INIT_LOCAL_L1_BASE_SCRATCH (MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_ERISC_LOCAL_SIZE)

#define MEM_AERISC_STACK_MIN_SIZE MEM_ERISC_STACK_MIN_SIZE
#define MEM_SUBORDINATE_AERISC_STACK_MIN_SIZE MEM_ERISC_STACK_MIN_SIZE

#define MEM_AERISC_BANK_TO_NOC_SCRATCH \
    (MEM_SUBORDINATE_AERISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_SUBORDINATE_ERISC_LOCAL_SIZE)
#define MEM_AERISC_BANK_TO_NOC_SIZE (MEM_BANK_TO_NOC_XY_SIZE + MEM_BANK_OFFSET_SIZE)

#define AERISC_RESET_PC (MEM_LOCAL_BASE | 0x14000)
#define SUBORDINATE_AERISC_RESET_PC (MEM_LOCAL_BASE | 0x14008)

/////////////
// Padding/alignment restriction needed in linker scripts for erisc
#define MEM_IERISC_KERNEL_PAD 32
