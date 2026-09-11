// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shared mailbox contract between the x280 fabric-worker firmware (fw_fabric.c),
// the Tensix helper kernels (kernels/*.cpp) and the host (l2cpu_fabric_forward.cpp).
// Plain #defines only — included from freestanding RV64 C, JIT'd Tensix C++ and host C++.
//
// Everything lives in the L2CPU tile's UNCACHED GDDR alias (x280 physical
// 0x3010_0000 + off). A Tensix/host NOC write to the L2CPU tile at that address and an
// x280 load/store of the same address see the same bytes with no cache maintenance.
// The x280_boot tool zeroes the first 64 KiB of this region before releasing the hart.

#ifndef L2CPU_FABRIC_FORWARD_MBOX_H
#define L2CPU_FABRIC_FORWARD_MBOX_H

#define FF_MBOX 0x30100000u

// --- Line 0 (0x00..0x3f): firmware-owned status, u64 fields ------------------
// 0x18/0x20/0x30/0x38 are written by start.S (trap count, mcause, boot marker, scratch).
#define FF_MBOX_HEARTBEAT (FF_MBOX + 0x00)    // u64, ++ every fw loop
#define FF_MBOX_FW_STATE (FF_MBOX + 0x08)     // u64, FF_STATE_*
#define FF_MBOX_HARTID (FF_MBOX + 0x10)       // u64
#define FF_MBOX_TRAP_COUNT (FF_MBOX + 0x18)   // u64, start.S trap handler
#define FF_MBOX_MCAUSE (FF_MBOX + 0x20)       // u64, start.S trap handler
#define FF_MBOX_FAULT_CODE (FF_MBOX + 0x28)   // u64, FF_FAULT_* (0 = healthy)
#define FF_MBOX_BOOT_MARKER (FF_MBOX + 0x30)  // u64, start.S (0xB0071E55)

// --- Request block (host/Tensix -> x280) @ 0x80, u32 fields, 64 B ------------
// Writer fills every field, then writes SEQ last (nonzero, unique per request).
#define FF_MBOX_REQ (FF_MBOX + 0x80)
#define FF_REQ_SEQ 0x00        // u32, 0 = no request
#define FF_REQ_SRC_ADDR 0x04   // u32, x280 physical addr of payload (uncached alias)
#define FF_REQ_SIZE 0x08       // u32, payload bytes (chunked into packets by fw)
#define FF_REQ_DST_NOC_X 0x0c  // u32, destination NOC x on the remote chip (translated coords)
#define FF_REQ_DST_NOC_Y 0x10  // u32
#define FF_REQ_DST_ADDR 0x14   // u32, destination address on that core/tile
#define FF_REQ_NUM_HOPS 0x18   // u32, fabric hops (1 = adjacent chip)
#define FF_REQ_FLAG_ADDR 0x1c  // u32, if nonzero: after the payload, send a 16 B inbox header here
#define FF_REQ_MODE 0x20       // u32, FF_MODE_*
#define FF_MODE_SEND 0
#define FF_MODE_CLOSE 1
#define FF_MODE_REOPEN 2

// --- Response block (x280 -> host) @ 0xc0, u32 fields, 64 B -------------------
#define FF_MBOX_RESP (FF_MBOX + 0xc0)
#define FF_RESP_SEQ 0x00          // u32, echoes FF_REQ_SEQ when done
#define FF_RESP_STATUS 0x04       // u32, FF_RSTATUS_*
#define FF_RESP_PACKETS 0x08      // u32, packets pushed for this request
#define FF_RESP_FREE_BEFORE 0x0c  // u32, free slots seen before the first packet
#define FF_RESP_SREG_BEFORE 0x10  // u32, EDM free-slots stream reg read before the first credit
#define FF_RESP_SREG_AFTER 0x14   // u32, ... and after the last credit
#define FF_RESP_CYCLES_LO 0x18    // u32, mcycle delta for the request
#define FF_RESP_CYCLES_HI 0x1c
#define FF_RESP_FREE_AFTER 0x20  // u32, free slots after the request drained (bounded wait)
#define FF_RSTATUS_OK 1
#define FF_RSTATUS_SLOT_TIMEOUT 2
#define FF_RSTATUS_BAD_REQ 3
#define FF_RSTATUS_CLOSED 4
#define FF_RSTATUS_CLOSE_TIMEOUT 5
#define FF_RSTATUS_NOT_OPEN 6

// --- Connection block (Tensix setup kernel -> x280) @ 0x100, u32, 128 B --------
// Mirrors the fields WorkerToFabricEdmSender::build_from_args<TENSIX>() resolves from
// the device-init L1 connection table, plus what the x280 needs that a Tensix gets
// for free (its own NOC coords, the header size, local sink addresses).
#define FF_MBOX_CONN (FF_MBOX + 0x100)
#define FF_CONN_EDM_NOC_X 0x00        // u32, EDM eth core x as in the conn table (translated coords)
#define FF_CONN_EDM_NOC_Y 0x04        // u32
#define FF_CONN_EDM_NOC0_X 0x08       // u32, same core in physical NOC0 coords (window candidate)
#define FF_CONN_EDM_NOC0_Y 0x0c       // u32  (0xFFFFFFFF = not supplied)
#define FF_CONN_BUFFER_BASE 0x10      // u32, edm_buffer_base_addr (eth L1)
#define FF_CONN_NUM_BUFFERS 0x14      // u32, num_buffers_per_channel
#define FF_CONN_BUFFER_SIZE 0x18      // u32, buffer_size_bytes (slot, incl. header)
#define FF_CONN_HANDSHAKE 0x1c        // u32, edm_connection_handshake_l1_addr
#define FF_CONN_WORKER_LOC_INFO 0x20  // u32, edm_worker_location_info_addr
#define FF_CONN_WR_COUNTER 0x24       // u32, edm_copy_of_wr_counter_addr (SenderChannelProducerCursor)
#define FF_CONN_SREG_WRITE 0x28       // u32, stream reg addr the worker decrements (credit)
#define FF_CONN_SREG_READ 0x2c        // u32, stream reg addr to read the router's free-slot count
#define FF_CONN_SELF_NOC_X 0x30       // u32, this L2CPU tile's NOC x (the EDM writes credits here)
#define FF_CONN_SELF_NOC_Y 0x34       // u32
#define FF_CONN_HDR_SIZE 0x38         // u32, sizeof(PACKET_HEADER_TYPE) on this fabric (48 or 64)
#define FF_CONN_FREESLOTS_SINK 0x3c   // u32, local uncached addr the EDM pushes its read counter to
#define FF_CONN_TEARDOWN_WORD 0x40    // u32, local uncached addr the EDM acks teardown to
#define FF_CONN_CURSOR_MAGIC 0x44     // u32, value the setup kernel wrote to cursor pad0 (coord probe)
#define FF_CONN_PEER_NOC_X 0x48       // u32, peer L2CPU tile x (for firmware-initiated replies)
#define FF_CONN_PEER_NOC_Y 0x4c       // u32
#define FF_CONN_PEER_INBOX 0x50       // u32, peer inbox base (FF_MBOX_INBOX on the peer)
#define FF_CONN_PEER_HOPS 0x54        // u32
#define FF_CONN_FLAGS 0x58            // u32, FF_CFLAG_*
#define FF_CONN_VALID \
    0x7c                                 // u32, written LAST; nonzero and unique per setup. The firmware
                                         //      (re)loads the block, re-probes and re-opens whenever it changes,
                                         //      so a new host run can reconfigure a running firmware.
#define FF_CONN_VALID_MAGIC 0xC0FFEE01u  // default/base value
#define FF_CFLAG_AUTO_ECHO 0x1           // firmware echoes inbox messages back to the peer inbox

// --- Diagnostics block (x280 -> host) @ 0x180, u32, 64 B ----------------------
#define FF_MBOX_DIAG (FF_MBOX + 0x180)
#define FF_DIAG_PROBE_RESULT 0x00  // u32, FF_PROBE_*
#define FF_DIAG_PROBE_TRANS 0x04   // u32, cursor pad0 read via translated coords
#define FF_DIAG_PROBE_NOC0 0x08    // u32, cursor pad0 read via NOC0 coords
#define FF_DIAG_OPEN_CTR 0x0c      // u32, cursor write_counter adopted at open
#define FF_DIAG_OPEN_IDX 0x10      // u32, cursor write_index adopted at open
#define FF_DIAG_OPEN_RDCTR 0x14    // u32, edm_read_counter read at open
#define FF_DIAG_SREG_OPEN 0x18     // u32, free-slots stream reg at open
#define FF_DIAG_FREE_NOW 0x1c      // u32, free slots at last loop
#define FF_DIAG_INBOX_SEEN 0x20    // u32, inbox messages observed
#define FF_DIAG_ECHOES 0x24        // u32, echoes sent
#define FF_DIAG_HANDSHAKE_RB 0x28  // u32, handshake word read back after open
#define FF_DIAG_WINDOW_X 0x2c      // u32, window coords in use for the EDM
#define FF_DIAG_WINDOW_Y 0x30
#define FF_DIAG_CONFIG_GEN 0x34    // u32, FF_CONN_VALID value the current connection was configured from
#define FF_DIAG_TEARDOWN_ACK 0x38  // u32, 1 if the router's noc_semaphore_inc teardown ack landed in TEARDOWN_WORD
#define FF_PROBE_NONE 0
#define FF_PROBE_TRANSLATED 1
#define FF_PROBE_NOC0 2
#define FF_PROBE_NEITHER 3

// --- Router-written words (this tile) @ 0x1c0 / 0x1d0 ---------------------------
// The EDM pushes its read counter (credits) to the first and the teardown ack to the
// second. Handed to the router at open via EDMChannelWorkerLocationInfo.
#define FF_MBOX_FREESLOTS_SINK (FF_MBOX + 0x1c0)
#define FF_MBOX_TEARDOWN_WORD (FF_MBOX + 0x1d0)
// Firmware-owned uncached scratch (e.g. the 16 B inbox header it sends) @ 0x1e0..0x1ff.
#define FF_MBOX_FW_SCRATCH (FF_MBOX + 0x1e0)

// --- Inbox (peer -> this tile) @ 0x1000 ----------------------------------------
// Message = data packets into FF_MBOX_INBOX_DATA, then one 16 B header packet into
// FF_MBOX_INBOX (same connection => in order). Receiver polls the header SEQ.
#define FF_MBOX_INBOX (FF_MBOX + 0x1000)
#define FF_INBOX_LEN 0x00  // u32, data bytes
#define FF_INBOX_SEQ 0x04  // u32, nonzero, written by the sender last
#define FF_INBOX_TAG 0x08  // u32, FF_TAG_*
#define FF_INBOX_RSVD 0x0c
#define FF_MBOX_INBOX_DATA (FF_MBOX + 0x1040)
#define FF_INBOX_DATA_MAX 0x3FC0  // bytes
#define FF_TAG_ORIGINAL 0x0816u
#define FF_TAG_ECHO 0xEC40u

// --- Outbox (host-staged payload for requests) @ 0x8000, 32 KiB ------------------
#define FF_MBOX_OUTBOX (FF_MBOX + 0x8000)
#define FF_OUTBOX_MAX 0x8000

// fw_state values
#define FF_STATE_BOOT 0x0
#define FF_STATE_ALIVE 0xA11FE
#define FF_STATE_PARAMS_READY 0x0C04
#define FF_STATE_PROBED 0x9E0B
#define FF_STATE_OPENED 0x0EDA
#define FF_STATE_SENT 0x5E27
#define FF_STATE_CLOSED 0xC105E

// fault codes (line-0 FAULT_CODE); 0 = healthy. Faults never park the hart: the
// firmware keeps heartbeating and serving requests where it can.
#define FF_FAULT_NONE 0
#define FF_FAULT_OPEN_TIMEOUT 1
#define FF_FAULT_SLOT_TIMEOUT 2
#define FF_FAULT_BAD_PARAMS 3
#define FF_FAULT_CLOSE_TIMEOUT 4
#define FF_FAULT_PROBE_FAILED 5

#endif  // L2CPU_FABRIC_FORWARD_MBOX_H
