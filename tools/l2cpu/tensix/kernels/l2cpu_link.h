// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

// Layout of a Tensix <-> L2CPU link channel: a region of L2CPU memory (local GDDR of the L2CPU tile, reached
// coherently at NoC tile (8,3) through the Memory Port alias). Offsets are from the channel base. Every word a
// producer and a consumer exchange lives on its own 64-byte line. Shared by the Tensix kernels (riscv32, C++),
// the x280 responder (rv64, C) and the host (Python mirror in ops.py).
#pragma once
#include "../../include/l2cpu_hw.h"

#define L2CPU_LINK_LINE 64u

#define L2CPU_LINK_OFF_REQ_SEQ 0x000u      // producer (notify): request number, only increases
#define L2CPU_LINK_OFF_DONE_SEQ 0x040u     // responder: = req_seq once the request is served
#define L2CPU_LINK_OFF_WAIT_STATUS 0x080u  // wait kernel: 0, or 0xDEAD0000 | (req & 0xFFFF) after a timeout
#define L2CPU_LINK_OFF_LANDED 0x0C0u       // streamed push: (req & 0xFFFF) << 16 | rows complete
#define L2CPU_LINK_OFF_DIAG 0x100u         // 2 lines of u32 diagnostics (device timestamps, stress results)
#define L2CPU_LINK_OFF_PUSH_SRC 0x180u     // push source table (host-written): see l2cpu_push.cpp
#define L2CPU_LINK_OFF_REPLY 0x300u        // responder reply line: 16 x u32
#define L2CPU_LINK_OFF_STOP 0x340u         // host: non-zero stops the test responder
#define L2CPU_LINK_OFF_RESP_STATUS 0x380u  // responder: magic when ready, then served count (u64 at +8)
#define L2CPU_LINK_SIZE 0x1000u            // data zones (e.g. pushed rows) start at or after this offset

#define L2CPU_LINK_RESP_MAGIC 0x4C4E4B31u  // "1KNL": the test responder is serving
#define L2CPU_LINK_REPLY_WORDS 16u
// Test responder reply for request r: reply[i] = r * 64 + i.
#define L2CPU_LINK_IMAGE_OFFSET 0x10000u  // test responder image: channel base + this offset
