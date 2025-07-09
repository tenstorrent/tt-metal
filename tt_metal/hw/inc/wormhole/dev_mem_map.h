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
// Memory for bank_to_dram_offset and bank_to_l1_offset arrays, size needs to be atleast 4 * (NUM_DRAM_BANKS + NUM_L1_BANKS)
#define MEM_BANK_OFFSET_SIZE 1024

#define NCRISC_HAS_IRAM 1
#define MEM_NCRISC_IRAM_BASE 0xFFC00000
#define MEM_NCRISC_IRAM_SIZE (16 * 1024)

/////////////
// Firmware/kernel code holes
#define MEM_BRISC_FIRMWARE_SIZE (5 * 1024 + 512)
// TODO: perhaps put NCRISC FW in the scratch area and free 1.5K after init (GS/WH)
#define MEM_NCRISC_FIRMWARE_SIZE 2048
#define MEM_TRISC0_FIRMWARE_SIZE 1536
#define MEM_TRISC1_FIRMWARE_SIZE 1536
#define MEM_TRISC2_FIRMWARE_SIZE 1536

#define MEM_BRISC_KERNEL_SIZE (48 * 1024)
#define MEM_NCRISC_KERNEL_SIZE MEM_NCRISC_IRAM_SIZE
#define MEM_TRISC0_KERNEL_SIZE (24 * 1024)
#define MEM_TRISC1_KERNEL_SIZE (24 * 1024)
#define MEM_TRISC2_KERNEL_SIZE (24 * 1024)

#define MEM_ZEROS_SIZE 512

#define MEM_LLK_DEBUG_SIZE 1024

#define MEM_BOOT_CODE_BASE 0
#define MEM_NOC_ATOMIC_RET_VAL_ADDR 4
#define MEM_L1_BARRIER 12
#define MEM_MAILBOX_BASE 16
// Magic size must be big enough to hold dev_msgs_t.  static_asserts will fire if this is too small
#define MEM_MAILBOX_SIZE 12640
// These are used in ncrisc-halt.S, asserted in ncrisc.cc to be valid
#define MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS MEM_MAILBOX_BASE + 4
#define MEM_SUBORDINATE_RUN_MAILBOX_ADDRESS MEM_MAILBOX_BASE + 8
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
#define MEM_NOC_COUNTER_L1_SIZE 5 * 2 * 2 * MEM_NOC_COUNTER_SIZE
#define MEM_NOC_COUNTER_BASE (MEM_TRISC2_FIRMWARE_BASE + MEM_TRISC2_FIRMWARE_SIZE)

// Tensix routing table for fabric networking
#define MEM_TENSIX_ROUTING_TABLE_BASE (MEM_NOC_COUNTER_BASE + MEM_NOC_COUNTER_L1_SIZE)
#define MEM_TENSIX_ROUTING_TABLE_SIZE 2064
#if (MEM_TENSIX_ROUTING_TABLE_BASE % 16 != 0) || (MEM_TENSIX_ROUTING_TABLE_SIZE % 16 != 0)
#error "Tensix routing table base and size must be 16-byte aligned"
#endif

// Tensix fabric connection metadata for workers
#define MEM_TENSIX_FABRIC_CONNECTIONS_BASE (MEM_TENSIX_ROUTING_TABLE_BASE + MEM_TENSIX_ROUTING_TABLE_SIZE)
#define MEM_TENSIX_FABRIC_CONNECTIONS_SIZE 592  // sizeof(tensix_fabric_connections_l1_info_t)
#if (MEM_TENSIX_FABRIC_CONNECTIONS_BASE % 16 != 0) || (MEM_TENSIX_FABRIC_CONNECTIONS_SIZE % 16 != 0)
#error "Tensix fabric connections base and size must be 16-byte aligned"
#endif

#define MEM_MAP_END (MEM_TENSIX_FABRIC_CONNECTIONS_BASE + MEM_TENSIX_FABRIC_CONNECTIONS_SIZE)

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
#define MEM_IERISC_MAP_END (MEM_IERISC_FIRMWARE_BASE + MEM_IERISC_FIRMWARE_SIZE)
#define MEM_IERISC_KERNEL_SIZE (24 * 1024)
#define MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH MEM_IERISC_MAP_END

#define MEM_IERISC_BANK_TO_NOC_SCRATCH (MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH + MEM_IERISC_LOCAL_SIZE)
#define MEM_IERISC_BANK_TO_NOC_SIZE (MEM_BANK_TO_NOC_XY_SIZE + MEM_BANK_OFFSET_SIZE)

/////////////
// Padding/alignment restriction needed in linker scripts for erisc
#define MEM_IERISC_KERNEL_PAD 32
