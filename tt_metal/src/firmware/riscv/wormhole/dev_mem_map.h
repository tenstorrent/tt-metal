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
// tag them if needed with a section (e.g., "l1_data").  If the globals are
// shared across all firmware/kernels, put them in brisc and reference them
// with "extern" in the other cores' source
//


/////////////
// RISC-V Address map definition (hardware)
#define MEM_L1_BASE           0x0
#define MEM_L1_SIZE           (1536 * 1024)

#define MEM_LOCAL_BASE        0xFFB00000
#define MEM_LOCAL_SIZE        (4 * 1024)
#define MEM_TRISC_LOCAL_SIZE  (2 *1024)

#define MEM_L0_BASE           0xFFC00000
#define MEM_NCRISC_IRAM_BASE  0xFFC00000
#define MEM_NCRISC_IRAM_SIZE  (16 * 1024)

/////////////
// Mailboxes
#define MEM_TEST_MAILBOX_ADDRESS       4
#define MEM_ENABLE_CORE_MAILBOX        32
#define MEM_WALL_CLOCK_MAILBOX_ADDRESS 96
#define MEM_DEBUG_MAILBOX_ADDRESS      112
#define MEM_DEBUG_MAILBOX_SIZE         64
#define MEM_CQ_MAILBOX_ADDRESS         368

/////////////
// Firmware/kernel code holes
#define MEM_BRISC_FIRMWARE_SIZE        (20 * 1024)
#define MEM_BRISC_FIRMWARE_CODE_SIZE   ( 7 * 1024 + 512)
#define MEM_NCRISC_FIRMWARE_SIZE       (32 * 1024)
#define MEM_TRISC0_SIZE                (20 * 1024)
#define MEM_TRISC1_SIZE                (16 * 1024)
#define MEM_TRISC2_SIZE                (20 * 1024)

#define MEM_BRISC_FIRMWARE_BASE        0
#define MEM_NCRISC_FIRMWARE_BASE       (MEM_BRISC_FIRMWARE_BASE + MEM_BRISC_FIRMWARE_SIZE)
#define MEM_TRISC0_BASE                (MEM_NCRISC_FIRMWARE_BASE + MEM_NCRISC_FIRMWARE_SIZE)
#define MEM_TRISC1_BASE                (MEM_TRISC0_BASE + MEM_TRISC0_SIZE)
#define MEM_TRISC2_BASE                (MEM_TRISC1_BASE + MEM_TRISC1_SIZE)
