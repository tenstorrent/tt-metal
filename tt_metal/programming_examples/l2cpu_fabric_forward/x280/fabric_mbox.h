// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shared mailbox / connection-parameter contract between the x280 fabric-worker
// firmware (fw_fabric.c, C) and the host orchestrator (l2cpu_fabric_forward.cpp,
// C++). Both include this so the two sides cannot drift on offsets. Plain #defines
// only — safe to include from freestanding RV64 firmware and from host C++.
//
// The mailbox lives in the L2CPU tile's UNCACHED GDDR alias so firmware stores are
// instantly NOC-visible and host/Tensix reads see them without a cache flush
// (same alias as the sibling l2cpu_noc_transfer example: MBOX 0x3010_0000).

#ifndef L2CPU_FABRIC_FORWARD_MBOX_H
#define L2CPU_FABRIC_FORWARD_MBOX_H

#define FF_MBOX 0x30100000u  // uncached GDDR alias, L2CPU tile

// --- Line 0: firmware-owned status (u64 fields) -----------------------------
#define FF_MBOX_HEARTBEAT (FF_MBOX + 0x000)    // u64, ++ every fw loop
#define FF_MBOX_FW_STATE (FF_MBOX + 0x008)     // u64, FF_STATE_*
#define FF_MBOX_HARTID (FF_MBOX + 0x010)       // u64
#define FF_MBOX_TRAP_COUNT (FF_MBOX + 0x018)   // u64, bumped by start.S trap handler
#define FF_MBOX_FAULT_CODE (FF_MBOX + 0x020)   // u64, FF_FAULT_* (0 = no fault)
#define FF_MBOX_BOOT_MARKER (FF_MBOX + 0x030)  // u64, stamped by start.S (0xB0071E55)

// --- Request block (Tensix producer -> x280), u32 fields --------------------
#define FF_MBOX_REQ (FF_MBOX + 0x080)
#define FF_REQ_SEQ 0x00          // u32, unique + nonzero per request (0 = none)
#define FF_REQ_PAYLOAD_LIM 0x04  // u32, x280 physical addr of payload in LIM/GDDR
#define FF_REQ_SIZE 0x08         // u32, payload size in bytes (<= buffer_size_bytes)
#define FF_REQ_DEST_NOC_X 0x0c   // u32, chip-B receiver NOC0 x
#define FF_REQ_DEST_NOC_Y 0x10   // u32, chip-B receiver NOC0 y
#define FF_REQ_DEST_L1 0x14      // u32, chip-B receiver L1 dest address

// --- Connection params (host -> x280), u32 fields ---------------------------
// These mirror WorkerToFabricEdmSender::build_from_args (VC2 runtime-arg path):
// tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp:133-147.
#define FF_MBOX_CONN (FF_MBOX + 0x100)
#define FF_CONN_EDM_NOC_X 0x00          // u32, EDM (eth core A) NOC x
#define FF_CONN_EDM_NOC_Y 0x04          // u32, EDM (eth core A) NOC y
#define FF_CONN_EDM_BUFFER_BASE 0x08    // u32, edm_buffer_base_addr (eth L1)
#define FF_CONN_NUM_BUFFERS 0x0c        // u32, num_buffers_per_channel
#define FF_CONN_BUFFER_SIZE 0x10        // u32, buffer_size_bytes (per slot)
#define FF_CONN_HANDSHAKE_ADDR 0x14     // u32, edm_connection_handshake_l1_addr
#define FF_CONN_WORKER_LOC_INFO 0x18    // u32, edm_worker_location_info_addr
#define FF_CONN_WR_COUNTER_ADDR 0x1c    // u32, edm_copy_of_wr_counter_addr
#define FF_CONN_CREDITS_STREAM_ID 0x20  // u32, sender_channel_credits_stream_id
#define FF_CONN_WORKER_FREESLOTS_L1 \
    0x24                       // u32, LIM addr the EDM pushes free-slot
                               //      credits into (x280 polls it)
#define FF_CONN_NUM_HOPS 0x28  // u32, hops to chip B (1 for adjacent)

// --- Status block (x280 -> host), u32 fields --------------------------------
#define FF_MBOX_STATUS (FF_MBOX + 0x180)
#define FF_STATUS_STATE 0x00           // u32, FF_STATE_* mirror for the worker FSM
#define FF_STATUS_SLOTS_SEEN 0x04      // u32, free slots observed before send
#define FF_STATUS_CREDIT_WRITES 0x08   // u32, stream-reg credit writes issued
#define FF_STATUS_LAST_FREESLOTS 0x0c  // u32, last free-slot count read

// fw_state values (line-0 FW_STATE)
#define FF_STATE_BOOT 0x0
#define FF_STATE_ALIVE 0xA11FE        // reached main loop (matches sibling fw)
#define FF_STATE_PARAMS_READY 0x0C04  // conn params consumed
#define FF_STATE_OPENED 0x0EDA        // EDM connection opened
#define FF_STATE_SENT 0x5E27          // packet pushed to EDM
#define FF_STATE_CLOSED 0xC105E       // connection torn down

// fault codes (line-0 FAULT_CODE); 0 = healthy
#define FF_FAULT_NONE 0
#define FF_FAULT_OPEN_TIMEOUT 1   // EDM never acked the open handshake
#define FF_FAULT_SLOT_TIMEOUT 2   // no free EDM buffer slot within the deadline
#define FF_FAULT_BAD_PARAMS 3     // conn params failed a sanity check
#define FF_FAULT_CLOSE_TIMEOUT 4  // EDM never acked teardown

#endif  // L2CPU_FABRIC_FORWARD_MBOX_H
