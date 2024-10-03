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
// device
// 2) Memory section sizes must be specified here, these are used in the
// linker scripts
// 3) Device static/global variables generally should NOT be listed here.  If
// they are global to a core, declare them in the that core's source code and
// tag them if needed with a section (e.g., "l1_data")
//

/////////////
// RISC-V Address map definition (hardware)
#define MEM_L1_BASE 0x0
#define MEM_L1_SIZE (1464 * 1024)

#define MEM_ETH_BASE 0x0
// -32 for ETH barrier, see comment in eth_l1_address_map
#define MEM_ETH_SIZE (512 * 1024 - 32)

#define MEM_LOCAL_BASE 0xFFB00000
#define MEM_BRISC_LOCAL_SIZE (8 * 1024)
#define MEM_NCRISC_LOCAL_SIZE (8 * 1024)
#define MEM_TRISC_LOCAL_SIZE (4 * 1024)

/////////////
// Firmware/kernel code holes
#define MEM_BOOT_CODE_SIZE 4
#define MEM_BRISC_FIRMWARE_SIZE (10 * 1024)
#define MEM_NCRISC_FIRMWARE_SIZE (16 * 1024)
#define MEM_TRISC0_FIRMWARE_SIZE (16 * 1024)
#define MEM_TRISC1_FIRMWARE_SIZE (16 * 1024)
#define MEM_TRISC2_FIRMWARE_SIZE (16 * 1024)
#define MEM_ZEROS_SIZE 512

#define MEM_BOOT_CODE_BASE 0
#define MEM_L1_BARRIER 12
#define MEM_MAILBOX_BASE 16
// Magic size must be big enough to hold dev_msgs_t.  static_asserts will fire if this is too small
#define MEM_MAILBOX_SIZE (5 * 4 * 512 + 4 * 32 + 1600 + 160)
#define MEM_MAILBOX_END (MEM_MAILBOX_BASE + MEM_MAILBOX_SIZE)
#define MEM_ZEROS_BASE ((MEM_MAILBOX_END + 31) & ~31)

#define MEM_BRISC_FIRMWARE_BASE (MEM_ZEROS_BASE + MEM_ZEROS_SIZE)
#define MEM_NCRISC_FIRMWARE_BASE (MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_SIZE)
#define MEM_TRISC0_FIRMWARE_BASE (MEM_NCRISC_FIRMWARE_BASE + MEM_NCRISC_FIRMWARE_SIZE)
#define MEM_TRISC1_FIRMWARE_BASE (MEM_TRISC0_FIRMWARE_BASE + MEM_TRISC0_FIRMWARE_SIZE)
#define MEM_TRISC2_FIRMWARE_BASE (MEM_TRISC1_FIRMWARE_BASE + MEM_TRISC1_FIRMWARE_SIZE)

#define MEM_MAP_END (MEM_TRISC2_FIRMWARE_BASE + MEM_TRISC2_FIRMWARE_SIZE)

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

/////////////
// Stack info
// Increasing the stack size comes at the expense of less local memory for globals
#define MEM_BRISC_STACK_SIZE 768
#define MEM_NCRISC_STACK_SIZE 1040
#define MEM_TRISC0_STACK_SIZE 320
#define MEM_TRISC1_STACK_SIZE 256
#define MEM_TRISC2_STACK_SIZE 768

#define MEM_BRISC_STACK_BASE (MEM_LOCAL_BASE + MEM_BRISC_LOCAL_SIZE - MEM_BRISC_STACK_SIZE)
#define MEM_NCRISC_STACK_BASE (MEM_LOCAL_BASE + MEM_NCRISC_LOCAL_SIZE - MEM_NCRISC_STACK_SIZE)
#define MEM_TRISC0_STACK_BASE (MEM_LOCAL_BASE + MEM_TRISC_LOCAL_SIZE - MEM_TRISC0_STACK_SIZE)
#define MEM_TRISC1_STACK_BASE (MEM_LOCAL_BASE + MEM_TRISC_LOCAL_SIZE - MEM_TRISC1_STACK_SIZE)
#define MEM_TRISC2_STACK_BASE (MEM_LOCAL_BASE + MEM_TRISC_LOCAL_SIZE - MEM_TRISC2_STACK_SIZE)


/////////////
// IERISC memory map
#define MEM_IERISC_LOCAL_SIZE (8 * 1024)
#define MEM_IERISC_FIRMWARE_SIZE (24 * 1024)
#define MEM_IERISC_RESERVED1 0
#define MEM_IERISC_RESERVED1_SIZE 1024
#define MEM_IERISC_MAILBOX_BASE (MEM_IERISC_RESERVED1 + MEM_IERISC_RESERVED1_SIZE)
// TODO: reduce this when mailbox sizes are core type aware for some members (eg watcher/dprint)
#define MEM_IERISC_MAILBOX_SIZE 3104
#define MEM_IERISC_MAILBOX_END (MEM_IERISC_MAILBOX_BASE + MEM_IERISC_MAILBOX_SIZE)
#define MEM_IERISC_RESERVED2 4128
#define MEM_IERISC_RESERVED2_SIZE 4064
#define MEM_IERISC_FIRMWARE_BASE (MEM_IERISC_RESERVED2 + MEM_IERISC_RESERVED2_SIZE)
#define MEM_IERISC_MAP_END (MEM_IERISC_FIRMWARE_BASE + MEM_IERISC_FIRMWARE_SIZE)
#define MEM_IERISC_INIT_LOCAL_L1_BASE_SCRATCH MEM_IERISC_MAP_END
#define MEM_IERISC_STACK_SIZE 1024
#define MEM_IERISC_STACK_BASE (MEM_LOCAL_BASE + MEM_IERISC_LOCAL_SIZE - MEM_IERISC_STACK_SIZE)

/////////////
// Padding/alignment restriction needed in linker scripts for erisc
#define MEM_IERISC_KERNEL_PAD 32
