// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#define MEM_L1_SIZE (1536 * 1024)

#define MEM_ETH_BASE 0x0
// Top 64K is reserved for syseng but host reads/writes from that region
#define MEM_ETH_SIZE (512 * 1024)

#define MEM_DRAM_SIZE (4177920 * 1024U)

// Local memory for processors WITHOUT base FW
#define MEM_LOCAL_BASE 0xFFB00000
#define MEM_BRISC_LOCAL_SIZE (8 * 1024)
#define MEM_NCRISC_LOCAL_SIZE (8 * 1024)
#define MEM_TRISC_LOCAL_SIZE (4 * 1024)

// Taken from base fw executable. It places the stack here.
#define MEM_ERISC_LOCAL_SIZE (8 * 1024)
#define MEM_ERISC_BASE_FW_LOCAL_SIZE 0x700
#define MEM_ERISC_KERNEL_SIZE (24 * 1024)
#define MEM_ERISC_RESERVED1 0
#define MEM_ERISC_RESERVED1_SIZE 256

// Offset by BASE FW
#define MEM_AERISC_LOCAL_BASE (MEM_LOCAL_BASE + MEM_ERISC_BASE_FW_LOCAL_SIZE)
// No BASE FW
#define MEM_SUBORDINATE_AERISC_LOCAL_BASE MEM_LOCAL_BASE

// Memory for (dram/l1)_bank_to_noc_xy arrays, size needs to be atleast 2 * NUM_NOCS * (NUM_DRAM_BANKS + NUM_L1_BANKS)
#define MEM_BANK_TO_NOC_XY_SIZE 1024
// Memory for bank_to_dram_offset and bank_to_l1_offset arrays, size needs to be atleast 4 * (NUM_DRAM_BANKS +
// NUM_L1_BANKS)
#define MEM_BANK_OFFSET_SIZE 1024

/////////////
// Firmware/kernel code holes
#define MEM_BRISC_FIRMWARE_SIZE (6 * 1024 + 1024)
// TODO: perhaps put NCRISC FW in the scratch area and free 1.5K after init (GS/WH)
#define MEM_NCRISC_FIRMWARE_SIZE 1536
#define MEM_TRISC0_FIRMWARE_SIZE 1536
#define MEM_TRISC1_FIRMWARE_SIZE 1536
#define MEM_TRISC2_FIRMWARE_SIZE 1536

// Blackhole Architecture - No IRAM constraints
// Per-kernel limits set to maximum available L1
// These are enforced by the ELF loader in tt_elffile.cpp
// The real constraint is the kernel_config_buffer aggregate limit, configurable via
// worker_l1_size in MeshDevice::create_unit_meshes
// 1503 KB = 1536 KB (L1 total) - 33 KB (MEM_MAP_END system reserved)
#define MEM_MAX_KERNEL_SIZE (1503 * 1024)

#define MEM_BRISC_KERNEL_SIZE MEM_MAX_KERNEL_SIZE
#define MEM_NCRISC_KERNEL_SIZE MEM_MAX_KERNEL_SIZE
#define MEM_TRISC0_KERNEL_SIZE MEM_MAX_KERNEL_SIZE
#define MEM_TRISC1_KERNEL_SIZE MEM_MAX_KERNEL_SIZE
#define MEM_TRISC2_KERNEL_SIZE MEM_MAX_KERNEL_SIZE

#define MEM_ZEROS_SIZE 512

#define MEM_LLK_DEBUG_SIZE 1024

#define MEM_BOOT_CODE_BASE 0
#define MEM_NOC_ATOMIC_RET_VAL_ADDR 4
#define MEM_L1_BARRIER 12

// Used by ARC FW and LLKs to store power throttling state
#define MEM_L1_ARC_FW_SCRATCH 16
#define MEM_L1_ARC_FW_SCRATCH_SIZE 16

// On Blackhole issuing inline writes and atomics requires all 4 memory ports to accept the transaction at the same
// time. If one port on the receipient has no back-pressure then the transaction will hang because there is no mechanism
// to allow one memory port to move ahead of another. To workaround this hang, we emulate inline writes on Blackhole by
// writing the value to be written to local L1 first and then issue a noc async write.
// Each noc has 16B to store value written out by inline writes.
// Base address for each noc to store the value to be written will be `MEM_{E,B,NC}RISC_L1_INLINE_BASE + (noc_index *
// 16)`
#define MEM_L1_INLINE_SIZE_PER_NOC 16
#define MEM_L1_INLINE_BASE 32  // MEM_L1_ARC_FW_SCRATCH + MEM_L1_ARC_FW_SCRATCH_SIZE

// Hardcode below due to compiler bug that cannot statically resolve the expression see GH issue #19265
#define MEM_MAILBOX_BASE 96  // (MEM_NCRISC_L1_INLINE_BASE + (MEM_L1_INLINE_SIZE_PER_NOC * 2) * 2)  // 2 nocs * 2 (B,NC)
// Magic size must be big enough to hold dev_msgs_t.  static_asserts will fire if this is too small
#define MEM_MAILBOX_SIZE 12768
#define MEM_MAILBOX_END (MEM_MAILBOX_BASE + MEM_MAILBOX_SIZE)
#define MEM_ZEROS_BASE ((MEM_MAILBOX_END + 31) & ~31)

#define MEM_LLK_DEBUG_BASE (MEM_ZEROS_BASE + MEM_ZEROS_SIZE)

#define MEM_BRISC_FIRMWARE_BASE (MEM_LLK_DEBUG_BASE + MEM_LLK_DEBUG_SIZE)
#define MEM_NCRISC_FIRMWARE_BASE (MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_SIZE)
#define MEM_TRISC0_FIRMWARE_BASE (MEM_NCRISC_FIRMWARE_BASE + MEM_NCRISC_FIRMWARE_SIZE)
#define MEM_TRISC1_FIRMWARE_BASE (MEM_TRISC0_FIRMWARE_BASE + MEM_TRISC0_FIRMWARE_SIZE)
#define MEM_TRISC2_FIRMWARE_BASE (MEM_TRISC1_FIRMWARE_BASE + MEM_TRISC1_FIRMWARE_SIZE)

#define MEM_NOC_COUNTER_SIZE 4
#define MEM_NOC_COUNTER_L1_SIZE (5 * 2 * 2 * MEM_NOC_COUNTER_SIZE)
#define MEM_NOC_COUNTER_BASE (MEM_TRISC2_FIRMWARE_BASE + MEM_TRISC2_FIRMWARE_SIZE)

// Fabric transaction counters (similar to NoC counters)
// 3 barrier types × 2 DMs × 4 bytes = 24 bytes + 8 bytes padding for 16-byte alignment
#define MEM_FABRIC_COUNTER_SIZE 4
#define MEM_FABRIC_COUNTER_L1_SIZE (3 * 2 * MEM_FABRIC_COUNTER_SIZE + 8)
#define MEM_FABRIC_COUNTER_BASE (MEM_NOC_COUNTER_BASE + MEM_NOC_COUNTER_L1_SIZE)

// Fabric connection sync region for synchronization across RISCs (BRISC/NCRISC)
// Contains: lock (4) + initialized (4) + connection object (128) + padding (8) = 144 bytes (16-byte aligned)
#define MEM_FABRIC_CONNECTION_LOCK_SIZE 144
#define MEM_FABRIC_CONNECTION_LOCK_BASE (MEM_FABRIC_COUNTER_BASE + MEM_FABRIC_COUNTER_L1_SIZE)

// Tensix routing table for fabric networking
#define MEM_TENSIX_ROUTING_TABLE_BASE (MEM_FABRIC_CONNECTION_LOCK_BASE + MEM_FABRIC_CONNECTION_LOCK_SIZE)
#define MEM_ROUTING_TABLE_SIZE 2544  // struct layout: base(484) + union(1024) + exit(1024) + pad(12)
#define MEM_OFFSET_OF_ROUTING_PATHS 484
#define MEM_ROUTING_TABLE_PADDING 12

#define ROUTING_PATH_SIZE_1D 1024  // 64 chips × 16 bytes
// 2D uncompressed size is too large to fit in L1 memory
#define COMPRESSED_ROUTING_PATH_SIZE_1D 0     // sizeof(intra_mesh_routing_path_t<1, true>)
#define COMPRESSED_ROUTING_PATH_SIZE_2D 1024  // sizeof(intra_mesh_routing_path_t<2, true>)
// Union: 1D and 2D routing tables share the same offset
#define MEM_TENSIX_ROUTING_PATH_BASE (MEM_TENSIX_ROUTING_TABLE_BASE + MEM_OFFSET_OF_ROUTING_PATHS)
#define MEM_TENSIX_ROUTING_PATH_BASE_1D MEM_TENSIX_ROUTING_PATH_BASE  // 484
#define MEM_TENSIX_ROUTING_PATH_BASE_2D MEM_TENSIX_ROUTING_PATH_BASE  // 484
#define MEM_TENSIX_ROUTING_PATH_SIZE 1024                             // max(1024, 1024)

#define MEM_TENSIX_EXIT_NODE_TABLE_BASE (MEM_TENSIX_ROUTING_PATH_BASE + MEM_TENSIX_ROUTING_PATH_SIZE)
#define MEM_EXIT_NODE_TABLE_SIZE 1024  // sizeof(exit_node_table_t)

// Tensix fabric connection metadata for workers
#define MEM_TENSIX_FABRIC_CONNECTIONS_BASE \
    (MEM_TENSIX_EXIT_NODE_TABLE_BASE + MEM_EXIT_NODE_TABLE_SIZE + MEM_ROUTING_TABLE_PADDING)
#define MEM_TENSIX_FABRIC_CONNECTIONS_SIZE 656        // sizeof(tensix_fabric_connections_l1_info_t)
#define MEM_TENSIX_FABRIC_OFFSET_OF_ALIGNED_INFO 400  // offsetof(tensix_fabric_connections_l1_info_t, read_write)

// Packet header pool sizing constants
#define PACKET_HEADER_MAX_SIZE 144  // sizeof(UDMHybridMeshPacketHeader)
#define NUM_PACKET_HEADERS \
    (6 * 2 * MaxDMProcessorsPerCoreType)  // (EAST, WEST, NORTH, SOUTH, UP, DOWN) * convention * (DM0, DM1)

// Packet header pool for fabric networking
// Size: 144 * 6 * 2 * 2 = 3456
#define MEM_PACKET_HEADER_POOL_BASE (MEM_TENSIX_FABRIC_CONNECTIONS_BASE + MEM_TENSIX_FABRIC_CONNECTIONS_SIZE)
#define MEM_PACKET_HEADER_POOL_SIZE (PACKET_HEADER_MAX_SIZE * NUM_PACKET_HEADERS)
#if (MEM_PACKET_HEADER_POOL_BASE % 16 != 0) || (MEM_PACKET_HEADER_POOL_SIZE % 16 != 0)
#error "Packet header pool base and size must be 16-byte aligned"
#endif

// Read-only reserved memory boundary for watcher checks
#define MEM_MAP_READ_ONLY_END (MEM_TENSIX_FABRIC_CONNECTIONS_BASE + MEM_TENSIX_FABRIC_OFFSET_OF_ALIGNED_INFO)
// Read-write reserved memory boundary for watcher checks
#define MEM_MAP_END (MEM_PACKET_HEADER_POOL_BASE + MEM_PACKET_HEADER_POOL_SIZE)

// Every address after MEM_MAP_END is a "scratch" address
// These can be used by FW during init, but aren't usable once FW reaches "ready"

/////////////
// Initialization relocation L1 memory
// Host downloads to these addresses, fw copies to destination
// Note: using xmov to copy ncrisc to addresses above 1M hangs the chip
#define MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH MEM_MAP_END
#define MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH (MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_BRISC_LOCAL_SIZE)
#define MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH (MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_NCRISC_LOCAL_SIZE)
#define MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH (MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH + MEM_TRISC_LOCAL_SIZE)
#define MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH (MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH + MEM_TRISC_LOCAL_SIZE)

#define MEM_NCRISC_INIT_IRAM_L1_BASE_SCRATCH (MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH + MEM_TRISC_LOCAL_SIZE)

#define MEM_BANK_TO_NOC_SCRATCH (MEM_NCRISC_INIT_IRAM_L1_BASE_SCRATCH + MEM_NCRISC_LOCAL_SIZE)
#define MEM_BANK_TO_NOC_SIZE (MEM_BANK_TO_NOC_XY_SIZE + MEM_BANK_OFFSET_SIZE)

// Scratch area for logical to virtual coordinate mapping.
// This size must match the firmware noc_size_x & noc_size Y. Size is largest chip (X + Y) * sizeof uint8_t.
// Chip sizes must round up to nearest multiple of 4 to deal with uint32_t alignment for L1 to local copies.
#define MEM_LOGICAL_TO_VIRTUAL_SCRATCH (MEM_BANK_TO_NOC_SCRATCH + MEM_BANK_TO_NOC_SIZE)
#define MEM_LOGICAL_TO_VIRTUAL_SIZE ((20 + 12) * sizeof(uint8_t))

/////////////
// Stack info
// Stack and globals share the same piece of memory, one grows at the
// expense of the other.
#define MEM_BRISC_STACK_MIN_SIZE 256
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
// Should match with sizeof(FabricTelemetry) - includes postcode (4 bytes) and scratch (28 bytes)
#define MEM_AERISC_FABRIC_TELEMETRY_SIZE 160
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

#define MEM_AERISC_FABRIC_POSTCODES_SIZE 4
#define MEM_AERISC_FABRIC_SCRATCH_SIZE 28
#define MEM_AERISC_FABRIC_POSTCODES_BASE                                                                      \
    (MEM_AERISC_FABRIC_TELEMETRY_BASE + MEM_AERISC_FABRIC_TELEMETRY_SIZE - MEM_AERISC_FABRIC_POSTCODES_SIZE - \
     MEM_AERISC_FABRIC_SCRATCH_SIZE)
#define MEM_AERISC_FABRIC_SCRATCH_BASE \
    (MEM_AERISC_FABRIC_TELEMETRY_BASE + MEM_AERISC_FABRIC_TELEMETRY_SIZE - MEM_AERISC_FABRIC_POSTCODES_SIZE)

#define MEM_AERISC_ROUTING_TABLE_SIZE MEM_ROUTING_TABLE_SIZE
#define MEM_AERISC_ROUTING_TABLE_BASE (MEM_AERISC_FABRIC_TELEMETRY_BASE - MEM_AERISC_ROUTING_TABLE_SIZE)
// AERISC fabric routing paths (union = same offset for 1D and 2D)
#define MEM_AERISC_FABRIC_ROUTING_PATH_BASE (MEM_AERISC_ROUTING_TABLE_BASE + MEM_OFFSET_OF_ROUTING_PATHS)
#define MEM_AERISC_FABRIC_ROUTING_PATH_BASE_1D MEM_AERISC_FABRIC_ROUTING_PATH_BASE
#define MEM_AERISC_FABRIC_ROUTING_PATH_BASE_2D MEM_AERISC_FABRIC_ROUTING_PATH_BASE
#define MEM_AERISC_EXIT_NODE_TABLE_BASE (MEM_AERISC_FABRIC_ROUTING_PATH_BASE + MEM_ERISC_FABRIC_ROUTING_PATH_SIZE)

// This is now the maximum size available for your application
#define MEM_ERISC_MAX_SIZE MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE

// Common Misc
#define MEM_RETRAIN_COUNT_ADDR 0x7CE00
#define MEM_CORR_CW_ADDR 0x7CE90
#define MEM_UNCORR_CW_ADDR 0x7CE98
#define MEM_RETRAIN_FORCE_ADDR 0x1EFC

#define MEM_SYSENG_ETH_RESULTS_BASE_ADDR 0x7CC00
#define MEM_SYSENG_ETH_MAILBOX_BASE_ADDR 0x7D000

#define MEM_SUBORDINATE_ERISC_LOCAL_SIZE (8 * 1024)
#define MEM_ERISC_FIRMWARE_SIZE (24 * 1024)
#define MEM_SUBORDINATE_ERISC_FIRMWARE_SIZE (24 * 1024)

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
#define MEM_IERISC_L1_INLINE_END (MEM_IERISC_L1_INLINE_BASE + (MEM_L1_INLINE_SIZE_PER_NOC * 2) * 2)
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
////////////

// These Dynamic NOC counter buffers come from base FW
#define MEM_AERISC_NOC_COUNTER_SIZE 4
#define MEM_AERISC_NOC_COUNTER_BASE (0x7D040)

#define MEM_AERISC_LOCAL_SIZE (MEM_ERISC_LOCAL_SIZE - MEM_ERISC_BASE_FW_LOCAL_SIZE)  // Minus the base FW local usage
#define MEM_SUBORDINATE_AERISC_LOCAL_SIZE MEM_ERISC_LOCAL_SIZE
#define MEM_AERISC_MAILBOX_BASE (MEM_ERISC_RESERVED1 + MEM_ERISC_RESERVED1_SIZE)
#define MEM_AERISC_MAILBOX_SIZE MEM_ERISC_MAILBOX_SIZE
#define MEM_AERISC_MAILBOX_END (MEM_AERISC_MAILBOX_BASE + MEM_AERISC_MAILBOX_SIZE)
#define MEM_AERISC_L1_INLINE_BASE MEM_AERISC_MAILBOX_END
#define MEM_AERISC_L1_INLINE_END (MEM_AERISC_L1_INLINE_BASE + (MEM_L1_INLINE_SIZE_PER_NOC * 2) * 2)

// This is a dummy write slot. It is not used because we launch active erisc firmware differently
// compared to Wormhole
#define MEM_AERISC_VOID_LAUNCH_FLAG MEM_AERISC_L1_INLINE_END
#define MEM_AERISC_VOID_LAUNCH_FLAG_SIZE 16

#define MEM_AERISC_FIRMWARE_BASE (MEM_AERISC_VOID_LAUNCH_FLAG + MEM_AERISC_VOID_LAUNCH_FLAG_SIZE)
#define MEM_AERISC_FIRMWARE_SIZE MEM_ERISC_FIRMWARE_SIZE
#define MEM_SUBORDINATE_AERISC_FIRMWARE_BASE (MEM_AERISC_FIRMWARE_BASE + MEM_AERISC_FIRMWARE_SIZE)
#define MEM_SUBORDINATE_AERISC_FIRMWARE_SIZE MEM_ERISC_FIRMWARE_SIZE
#define MEM_AERISC_MAP_END (MEM_SUBORDINATE_AERISC_FIRMWARE_BASE + MEM_SUBORDINATE_AERISC_FIRMWARE_SIZE)
#define MEM_AERISC_KERNEL_SIZE MEM_ERISC_KERNEL_SIZE

#define MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH MEM_AERISC_MAP_END
#define MEM_SUBORDINATE_AERISC_INIT_LOCAL_L1_BASE_SCRATCH \
    (MEM_AERISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_AERISC_LOCAL_SIZE)

#define MEM_SUBORDINATE_AERISC_STACK_MIN_SIZE MEM_ERISC_STACK_MIN_SIZE

#define MEM_AERISC_BANK_TO_NOC_SCRATCH \
    (MEM_SUBORDINATE_AERISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_SUBORDINATE_AERISC_LOCAL_SIZE)
#define MEM_AERISC_BANK_TO_NOC_SIZE (MEM_BANK_TO_NOC_XY_SIZE + MEM_BANK_OFFSET_SIZE)

#define AERISC_RESET_PC (MEM_LOCAL_BASE | 0x14000)
#define SUBORDINATE_AERISC_RESET_PC (MEM_LOCAL_BASE | 0x14008)

/////////////
// Padding/alignment restriction needed in linker scripts for erisc
#define MEM_IERISC_KERNEL_PAD 32
