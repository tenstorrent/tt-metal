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
#define MEM_L1_SIZE (1464 * 1024)

#define MEM_ETH_BASE 0x0
// -32 for ETH barrier, see comment in eth_l1_address_map
#define MEM_ETH_SIZE (256 * 1024 - 32)

#define MEM_DRAM_SIZE (1048576 * 1024)

#define MEM_LOCAL_BASE 0xFFB00000
#define MEM_BRISC_LOCAL_SIZE (4 * 1024)
#define MEM_NCRISC_LOCAL_SIZE (4 * 1024)
#define MEM_TRISC_LOCAL_SIZE (2 * 1024)

// Memory for (dram/l1)_bank_to_noc_xy arrays, size needs to be atleast 2 * NUM_NOCS * (NUM_DRAM_BANKS + NUM_L1_BANKS)
#define MEM_BANK_TO_NOC_XY_SIZE 1024
// Memory for bank_to_dram_offset and bank_to_l1_offset arrays, size needs to be atleast 4 * (NUM_DRAM_BANKS +
// NUM_L1_BANKS)
#define MEM_BANK_OFFSET_SIZE 1024

#define NCRISC_HAS_IRAM 1
#define MEM_NCRISC_IRAM_BASE 0xFFC00000
#define MEM_NCRISC_IRAM_SIZE (16 * 1024)

/////////////
// Firmware/kernel code holes
#define MEM_BRISC_FIRMWARE_SIZE (6 * 1024)
// TODO: perhaps put NCRISC FW in the scratch area and free 1.5K after init (GS/WH)
#define MEM_NCRISC_FIRMWARE_SIZE 2048
#define MEM_TRISC0_FIRMWARE_SIZE 1536
#define MEM_TRISC1_FIRMWARE_SIZE 1536
#define MEM_TRISC2_FIRMWARE_SIZE 1536

// Wormhole Architecture - BRISC, TRISC0/1/2 have no IRAM constraints
// Only NCRISC is IRAM constrained
// Per-kernel limits set to maximum available L1
// These are enforced by the ELF loader in tt_elffile.cpp
// The real constraint is the kernel_config_buffer aggregate limit, configurable via
// worker_l1_size in MeshDevice::create_unit_meshes
// 1432 KB = 1464 KB (L1 total) - 32 KB (MEM_MAP_END system reserved)
#define MEM_MAX_KERNEL_SIZE (1432 * 1024)

#define MEM_BRISC_KERNEL_SIZE MEM_MAX_KERNEL_SIZE
#define MEM_NCRISC_KERNEL_SIZE MEM_NCRISC_IRAM_SIZE
#define MEM_TRISC0_KERNEL_SIZE MEM_MAX_KERNEL_SIZE
#define MEM_TRISC1_KERNEL_SIZE MEM_MAX_KERNEL_SIZE
#define MEM_TRISC2_KERNEL_SIZE MEM_MAX_KERNEL_SIZE

#define MEM_ZEROS_SIZE 512

#define MEM_LLK_DEBUG_SIZE 1024

#define MEM_BOOT_CODE_BASE 0
#define MEM_NOC_ATOMIC_RET_VAL_ADDR 4
#define MEM_L1_BARRIER 12
#define MEM_MAILBOX_BASE 16
// Magic size must be big enough to hold dev_msgs_t.  static_asserts will fire if this is too small
#define MEM_MAILBOX_SIZE 12768
// These are used in ncrisc-halt.S, asserted in ncrisc.cc to be valid
#define MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS (MEM_MAILBOX_BASE + 4)
#define MEM_SUBORDINATE_RUN_MAILBOX_ADDRESS (MEM_MAILBOX_BASE + 8)
#define MEM_MAILBOX_END (MEM_MAILBOX_BASE + MEM_MAILBOX_SIZE)
#define MEM_ZEROS_BASE ((MEM_MAILBOX_END + 31) & ~31)

#define MEM_LLK_DEBUG_BASE (MEM_ZEROS_BASE + MEM_ZEROS_SIZE)

#define MEM_BRISC_FIRMWARE_BASE (MEM_LLK_DEBUG_BASE + MEM_LLK_DEBUG_SIZE)
#define MEM_NCRISC_FIRMWARE_BASE (MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_SIZE)
#define MEM_TRISC0_FIRMWARE_BASE (MEM_NCRISC_FIRMWARE_BASE + MEM_NCRISC_FIRMWARE_SIZE)
#define MEM_TRISC1_FIRMWARE_BASE (MEM_TRISC0_FIRMWARE_BASE + MEM_TRISC0_FIRMWARE_SIZE)
#define MEM_TRISC2_FIRMWARE_BASE (MEM_TRISC1_FIRMWARE_BASE + MEM_TRISC1_FIRMWARE_SIZE)

/* Kernel is in IRAM.  */
#define MEM_NCRISC_KERNEL_BASE MEM_NCRISC_IRAM_BASE

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
#define MEM_ROUTING_TABLE_SIZE \
    2288  // struct layout: base(484) + 1d_path(256) + 2d_path(512) + exit_table(1024) + padding(12)
#define MEM_OFFSET_OF_ROUTING_PATHS 484
#define MEM_ROUTING_TABLE_PADDING 12

#define ROUTING_PATH_SIZE_1D 256
// 2D uncompressed size is too large to fit in L1 memory
#define COMPRESSED_ROUTING_PATH_SIZE_1D 0    // sizeof(intra_mesh_routing_path_t<1, true>)
#define COMPRESSED_ROUTING_PATH_SIZE_2D 512  // sizeof(intra_mesh_routing_path_t<2, true>)
#define MEM_TENSIX_ROUTING_PATH_BASE (MEM_TENSIX_ROUTING_TABLE_BASE + MEM_OFFSET_OF_ROUTING_PATHS)
#define MEM_TENSIX_ROUTING_PATH_BASE_1D MEM_TENSIX_ROUTING_PATH_BASE
#define MEM_TENSIX_ROUTING_PATH_BASE_2D (MEM_TENSIX_ROUTING_PATH_BASE + ROUTING_PATH_SIZE_1D)
#define MEM_TENSIX_ROUTING_PATH_SIZE (ROUTING_PATH_SIZE_1D + COMPRESSED_ROUTING_PATH_SIZE_2D)

#define MEM_TENSIX_EXIT_NODE_TABLE_BASE (MEM_TENSIX_ROUTING_PATH_BASE + MEM_TENSIX_ROUTING_PATH_SIZE)
#define MEM_EXIT_NODE_TABLE_SIZE 1024  // sizeof(exit_node_table_t)

// Tensix fabric connection metadata for workers
#define MEM_TENSIX_FABRIC_CONNECTIONS_BASE \
    (MEM_TENSIX_EXIT_NODE_TABLE_BASE + MEM_EXIT_NODE_TABLE_SIZE + MEM_ROUTING_TABLE_PADDING)
#define MEM_TENSIX_FABRIC_CONNECTIONS_SIZE 656        // sizeof(tensix_fabric_connections_l1_info_t)
#define MEM_TENSIX_FABRIC_OFFSET_OF_ALIGNED_INFO 400  // offsetof(tensix_fabric_connections_l1_info_t, read_write)

// Packet header pool sizing constants
#define PACKET_HEADER_MAX_SIZE 112                               // sizeof(UDMHybridMeshPacketHeader)
#define NUM_PACKET_HEADERS (4 * 2 * MaxDMProcessorsPerCoreType)  // (EAST, WEST, NORTH, SOUTH) * convention * (DM0, DM1)

// Packet header pool for fabric networking
// Size: 64 * 4 * 2 * 2 = 1024
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
#define MEM_LOGICAL_TO_VIRTUAL_SIZE ((12 + 12) * sizeof(uint8_t))

/////////////
// Stack info
// Stack and globals share the same piece of memory, one grows at the
// expense of the other.
#define MEM_BRISC_STACK_MIN_SIZE 256
#define MEM_NCRISC_STACK_MIN_SIZE 256
#define MEM_TRISC0_STACK_MIN_SIZE 192
#define MEM_TRISC1_STACK_MIN_SIZE 192
#define MEM_TRISC2_STACK_MIN_SIZE 256
#define MEM_IERISC_STACK_MIN_SIZE 128
#define MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_1D ROUTING_PATH_SIZE_1D
#define MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_2D COMPRESSED_ROUTING_PATH_SIZE_2D

/////////////
// IERISC memory map
#define MEM_IERISC_LOCAL_SIZE (4 * 1024)
#define MEM_IERISC_FIRMWARE_SIZE (16 * 1024)
#define MEM_IERISC_RESERVED1 0
#define MEM_IERISC_RESERVED1_SIZE 1024
#define MEM_IERISC_RESERVED2 4128
#define MEM_IERISC_RESERVED2_SIZE 4064
// TODO: reduce this when mailbox sizes are core type aware for some members (eg watcher/dprint)
// TODO: also, move into gap above in the reserved area
#define MEM_IERISC_MAILBOX_BASE (MEM_IERISC_RESERVED2 + MEM_IERISC_RESERVED2_SIZE)
#define MEM_IERISC_MAILBOX_SIZE 5072
#define MEM_IERISC_MAILBOX_END (MEM_IERISC_MAILBOX_BASE + MEM_IERISC_MAILBOX_SIZE)
#define MEM_IERISC_FIRMWARE_BASE MEM_IERISC_MAILBOX_END
#define MEM_IERISC_FIRMWARE_END (MEM_IERISC_FIRMWARE_BASE + MEM_IERISC_FIRMWARE_SIZE)

// IDLE ETH fabric routing paths - placed in dedicated reserved area after firmware
#define MEM_IERISC_ROUTING_TABLE_BASE ((MEM_IERISC_FIRMWARE_END + 31) & ~31)

#define MEM_IERISC_FABRIC_ROUTING_PATH_BASE_1D (MEM_IERISC_ROUTING_TABLE_BASE + MEM_OFFSET_OF_ROUTING_PATHS)
#define MEM_IERISC_FABRIC_ROUTING_PATH_BASE_2D \
    (MEM_IERISC_FABRIC_ROUTING_PATH_BASE_1D + MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_1D)
#define MEM_IERISC_FABRIC_ROUTING_PATH_END \
    (MEM_IERISC_FABRIC_ROUTING_PATH_BASE_2D + MEM_ERISC_FABRIC_ROUTING_PATH_SIZE_2D)

#define MEM_IERISC_EXIT_NODE_TABLE_BASE MEM_IERISC_FABRIC_ROUTING_PATH_END
#define MEM_IERISC_EXIT_NODE_TABLE_END (MEM_IERISC_EXIT_NODE_TABLE_BASE + MEM_EXIT_NODE_TABLE_SIZE)

#define MEM_IERISC_MAP_END (MEM_IERISC_EXIT_NODE_TABLE_END + MEM_ROUTING_TABLE_PADDING)
#define MEM_IERISC_KERNEL_SIZE (24 * 1024)
#define MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH MEM_IERISC_MAP_END

#define MEM_IERISC_BANK_TO_NOC_SCRATCH (MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_IERISC_LOCAL_SIZE)
#define MEM_IERISC_BANK_TO_NOC_SIZE (MEM_BANK_TO_NOC_XY_SIZE + MEM_BANK_OFFSET_SIZE)

/////////////
// Active ERISC (AERISC) memory map
// These definitions mirror the layout used by eth_l1_address_map for Wormhole
#define MEM_MAX_NUM_CONCURRENT_TRANSACTIONS 8
#define MEM_ERISC_APP_ROUTING_INFO_SIZE 48
#define MEM_ERISC_SYNC_INFO_SIZE (160 + 16 * MEM_MAX_NUM_CONCURRENT_TRANSACTIONS)
#define MEM_ERISC_L1_KERNEL_CONFIG_SIZE ((96 * 4) + (16 * 16))

// Start of ERISC config space in L1 (end of loadable region)
#define MEM_ERISC_MAX_L1_LOADING_ADDR 0x3DC00
#define MEM_ERISC_APP_ROUTING_INFO_BASE MEM_ERISC_MAX_L1_LOADING_ADDR
#define MEM_ERISC_APP_SYNC_INFO_BASE (MEM_ERISC_APP_ROUTING_INFO_BASE + MEM_ERISC_APP_ROUTING_INFO_SIZE)

// AERISC mailbox region
#define MEM_AERISC_MAILBOX_BASE (MEM_ERISC_APP_SYNC_INFO_BASE + MEM_ERISC_SYNC_INFO_SIZE)
#define MEM_AERISC_MAILBOX_SIZE 5072
#define MEM_AERISC_MAILBOX_END (MEM_AERISC_MAILBOX_BASE + MEM_AERISC_MAILBOX_SIZE)

// Kernel config region
#define MEM_ERISC_L1_KERNEL_CONFIG_BASE MEM_AERISC_MAILBOX_END

// Fabric router reserved/config regions
#define MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE \
    ((MEM_ERISC_L1_KERNEL_CONFIG_BASE + MEM_ERISC_L1_KERNEL_CONFIG_SIZE + 31) & ~31)
#define MEM_ERISC_FABRIC_ROUTER_RESERVED_SIZE 3088

#define MEM_AERISC_FABRIC_TELEMETRY_BASE (MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE)
#define MEM_AERISC_FABRIC_TELEMETRY_SIZE 160

#define MEM_AERISC_FABRIC_POSTCODES_SIZE 4
#define MEM_AERISC_FABRIC_SCRATCH_SIZE 28
#define MEM_AERISC_FABRIC_POSTCODES_BASE                                                                      \
    (MEM_AERISC_FABRIC_TELEMETRY_BASE + MEM_AERISC_FABRIC_TELEMETRY_SIZE - MEM_AERISC_FABRIC_POSTCODES_SIZE - \
     MEM_AERISC_FABRIC_SCRATCH_SIZE)
#define MEM_AERISC_FABRIC_SCRATCH_BASE \
    (MEM_AERISC_FABRIC_TELEMETRY_BASE + MEM_AERISC_FABRIC_TELEMETRY_SIZE - MEM_AERISC_FABRIC_POSTCODES_SIZE)

#define MEM_AERISC_ROUTING_TABLE_BASE (MEM_AERISC_FABRIC_TELEMETRY_BASE + MEM_AERISC_FABRIC_TELEMETRY_SIZE)
#define MEM_AERISC_ROUTING_TABLE_SIZE MEM_ROUTING_TABLE_SIZE
#define MEM_AERISC_ROUTING_TABLE_END (MEM_AERISC_ROUTING_TABLE_BASE + MEM_AERISC_ROUTING_TABLE_SIZE)

/////////////
// Padding/alignment restriction needed in linker scripts for erisc
#define MEM_IERISC_KERNEL_PAD 32
