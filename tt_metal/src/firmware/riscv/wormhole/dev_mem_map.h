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
#define L1_MEM_BASE           0x0         // 0x00000000 - 0xFFBFFFFF
#define LOCAL_MEM_BASE        0xFFB00000
#define L0_MEM_BASE           0xFFC00000  // 0xFFC00000 - 0xFFDFFFFF
#define NCRISC_IRAM_MEM_BASE  0xFFC00000

#define LOCAL_MEM_SIZE        4096

/////////////
// Mailboxes
#define TEST_MAILBOX_ADDRESS       4
#define ENABLE_CORE_MAILBOX        32
#define WALL_CLOCK_MAILBOX_ADDRESS 96
#define DEBUG_MAILBOX_ADDRESS      112
#define DEBUG_MAILBOX_SIZE         64
#define CQ_MAILBOX_ADDRESS         368
