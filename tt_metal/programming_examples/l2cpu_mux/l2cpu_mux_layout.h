// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Memory layout of the L2CPU-resident fabric mux, shared by the x280 firmware
// (x280/fw_mux.c) and the host (l2cpu_mux.cpp). Plain #defines only.
//
// Everything lives in the L2CPU tile's UNCACHED GDDR alias, separate from the
// fabric-worker mailbox (0x3010_0000, fabric_mbox.h) which the same firmware uses for
// its own router connection. A worker kernel is given the per-channel addresses below
// exactly as a V1 FabricMuxConfig hands out its getters; the kernel-side client
// (kernels/tt_fabric_l2cpu_mux_interface.hpp) speaks the V1 protocol against them,
// with one substitution: the commit is a plain write of the producer's write counter
// into LM_WRITE_COUNTER(ch) instead of a stream-register decrement (the L2CPU tile has
// no stream registers).

#ifndef L2CPU_MUX_LAYOUT_H
#define L2CPU_MUX_LAYOUT_H

#define LM_BASE 0x30200000u

// --- control words (u32, 16 B strided) --------------------------------------
#define LM_STATUS (LM_BASE + 0x00)        // EDMStatus: STARTED / READY_FOR_TRAFFIC / TERMINATED
#define LM_TERMINATION (LM_BASE + 0x10)   // TerminationSignal: 0 keep running, 1 graceful, 2 immediate
#define LM_HEARTBEAT (LM_BASE + 0x20)     // u32, ++ per service loop
#define LM_NUM_CHANNELS (LM_BASE + 0x30)  // u32, channels the firmware serves (written by firmware)
#define LM_SLOT_BYTES (LM_BASE + 0x40)    // u32, slot size the firmware uses (= router slot size)
#define LM_CONFIG_GEN \
    (LM_BASE + 0x50)  // u32, FF_CONN_VALID nonce the current router connection was built from;
                      //      clients that deliver the connection block themselves wait for it

// --- per-channel stats (x280 -> host), 16 B per channel ------------------------
#define LM_STATS(ch) (LM_BASE + 0x100 + (ch) * 16)
#define LM_STAT_FORWARDED 0x0   // packets forwarded to the router
#define LM_STAT_CONNECTS 0x4    // open handshakes seen
#define LM_STAT_TEARDOWNS 0x8   // close handshakes seen
#define LM_STAT_LAST_BYTES 0xc  // header+payload bytes of the last forwarded packet

// --- per-channel protocol blocks (same roles as FabricMuxConfig getters) --------
#define LM_MAX_CHANNELS 8
#define LM_NUM_BUFFERS 8                                     // slots per channel
#define LM_CONN_INFO(ch) (LM_BASE + 0x1000 + (ch) * 64)      // EDMChannelWorkerLocationInfo (64 B)
#define LM_HANDSHAKE(ch) (LM_BASE + 0x1400 + (ch) * 16)      // get_connection_handshake_address
#define LM_WRITE_COUNTER(ch) (LM_BASE + 0x1500 + (ch) * 16)  // get_flow_control_address: producer write counter
#define LM_CURSOR(ch) (LM_BASE + 0x1800 + (ch) * 64)         // get_buffer_index_address: SenderChannelProducerCursor
#define LM_CHANNELS_BASE (LM_BASE + 0x2000)                  // slot rings start here
// NOC transfers between Tensix L1 and L2CPU memory must use the same offset within a
// 64 B line on both sides (measured: a 16 B read into an L1 address at +0x20 mod 64
// returned the bytes 0x20 past the requested source). Slots are therefore strided to a
// 64 B multiple so every slot starts at offset 0, the header lands at 0 and the payload
// at hdr_size mod 64 — and clients keep their local header/payload buffers at the same
// offsets (see kernels/tt_fabric_l2cpu_mux_interface.hpp).
#define LM_SLOT_STRIDE(slot_bytes) ((((slot_bytes) + 63u) / 64u) * 64u)
#define LM_CHANNEL_BASE(ch, slot_bytes) (LM_CHANNELS_BASE + (ch) * LM_NUM_BUFFERS * LM_SLOT_STRIDE(slot_bytes))

// EDMStatus / TerminationSignal values (tt_metal/fabric/fabric_edm_packet_header.hpp)
#define LM_STATUS_STARTED 0xA0B0C0D0u
#define LM_STATUS_READY_FOR_TRAFFIC 0xA3B3C3D3u
#define LM_STATUS_TERMINATED 0xA4B4C4D4u
#define LM_TERM_KEEP_RUNNING 0u
#define LM_TERM_GRACEFUL 1u
#define LM_TERM_IMMEDIATE 2u

#endif  // L2CPU_MUX_LAYOUT_H
