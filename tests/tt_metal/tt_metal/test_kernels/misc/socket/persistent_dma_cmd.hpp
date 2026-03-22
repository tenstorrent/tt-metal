// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared command protocol for the persistent DMA engine.
// Included by both device kernels (BRISC/NCRISC) and host C++ code.
// Uses only standard integer types so it compiles in both environments.
//
// ── L1 layout per DMA core (absolute byte addresses) ─────────────────────
//
//   kDmaD2HCmdBase       [0x80000]  D2H command ring        (4 × 48 B = 192 B)
//   kDmaH2DCmdBase       [0x800C0]  H2D command ring        (4 × 48 B = 192 B)
//   kDmaD2HReadyFlag     [0x80180]  D2H kernel-ready flag   (4 B)
//   kDmaH2DReadyFlag     [0x80184]  H2D kernel-ready flag   (4 B)
//   kDmaD2HProgress      [0x80188]  D2H progress counter    (4 B)
//   kDmaH2DProgress      [0x8018C]  H2D progress counter    (4 B)
//   kDmaD2HCompletionSeq [0x80190]  D2H completion seq      (4 B)  ← Phase 1
//   kDmaH2DCompletionSeq [0x80194]  H2D completion seq      (4 B)  ← Phase 1
//                  ... gap to 1 KB boundary ...
//   kDmaD2HPingBase      [0x80400]  D2H ping buffer         (16 KB)
//   kDmaD2HPongBase      [0x84400]  D2H pong buffer         (16 KB)
//   kDmaH2DPingBase      [0x88400]  H2D ping buffer         (16 KB)
//   kDmaH2DPongBase      [0x8C400]  H2D pong buffer         (16 KB)
//
// ── Command encoding ──────────────────────────────────────────────────────
//
//   D2H (BRISC) command:
//     src_lo           = DRAM bank_base_address
//     src_hi           = dram_page_start   (first interleaved page index)
//     dst_lo / dst_hi  = host physical address (64-bit)
//     done_flag_lo/hi  = host done-flag phys addr (64-bit)
//
//   H2D (NCRISC) command:
//     src_lo / src_hi  = host physical address (64-bit)
//     dst_lo           = DRAM bank_base_address
//     dst_hi           = dram_page_start   (first interleaved page index)
//
//   D2H completion: kernel writes last_seq to host done-flag via PCIe posted
//   write AFTER the data write barrier.  PCIe strong ordering (spec §2.4.1)
//   guarantees data is committed to host memory before the done flag.
//   Host polls done flags via CPU load (not BAR read).
//
//   H2D completion: still via kDmaH2DCompletionSeq BAR reads (Phase 1).
//   H2D writes go from host→device, so there is no cross-direction ordering
//   issue — the kernel can safely signal via L1.
//
// ── DMA core layout ───────────────────────────────────────────────────────
//
//   Up to 72 cores: logical (col, row) with col in [0,7], row in [0,8].
//   Core index c = row * kDmaCoreCols + col.

#pragma once
#include <cstdint>

// ── Opcodes ───────────────────────────────────────────────────────────────
static constexpr uint32_t DMA_OP_IDLE = 0;         // slot not yet written
static constexpr uint32_t DMA_OP_TRANSFER = 1;     // execute transfer (DRAM ↔ host)
static constexpr uint32_t DMA_OP_EXIT = 2;         // terminate kernel loop
static constexpr uint32_t DMA_OP_L1_ONLY = 3;      // L1-direct (skip DRAM, pure PCIe BW)
static constexpr uint32_t DMA_OP_H2D_PUSH = 4;     // host-push H2D (BAR writes to L1, kernel drains to DRAM)
static constexpr uint32_t DMA_OP_H2D_PUSH_L1 = 5;  // host-push H2D, L1-only (skip DRAM drain, pure PCIe BW)

// ── Command slot — 48 bytes (12 × uint32_t) ──────────────────────────────
struct DmaCmd {
    uint32_t seq;           //  0: host increments by 1 to signal a new command
    uint32_t opcode;        //  1: DMA_OP_*
    uint32_t src_lo;        //  2
    uint32_t src_hi;        //  3
    uint32_t dst_lo;        //  4
    uint32_t dst_hi;        //  5
    uint32_t num_chunks;    //  6: transfer size in units of kDmaChunkSize
    uint32_t done_flag_lo;  //  7: host done-flag phys addr [31:0] (D2H)
    uint32_t done_flag_hi;  //  8: host done-flag phys addr [63:32]
    uint32_t _pad[3];       //  9-11: pad to 48 bytes
};
// Verify at host-compile time only (kernel compiler may not support this)
#ifndef TENSIX_FIRMWARE
static_assert(sizeof(DmaCmd) == 48, "DmaCmd must be 48 bytes");
#endif

// ── L1 address constants ──────────────────────────────────────────────────
static constexpr uint32_t kDmaUserBase = 512u * 1024u;  // 0x80000
#ifdef KDMA_RING_DEPTH
static constexpr uint32_t kDmaRingDepth = KDMA_RING_DEPTH;
#else
static constexpr uint32_t kDmaRingDepth = 4u;
#endif
static constexpr uint32_t kDmaSlotBytes = 48u;                            // sizeof(DmaCmd)
static constexpr uint32_t kDmaRingBytes = kDmaRingDepth * kDmaSlotBytes;  // 192 B default
static constexpr uint32_t kDmaD2HCmdBase = kDmaUserBase;                  // 0x80000
static constexpr uint32_t kDmaH2DCmdBase = kDmaUserBase + kDmaRingBytes;  // 0x800C0 default
// Ping-pong buffers start at a 1 KB boundary past the rings
static constexpr uint32_t kDmaBufBase = kDmaUserBase + 1024u;  // 0x80400
// kDmaChunkSize and kDmaH2DNumBufs can be overridden at compile time by the
// host passing defines via DataMovementConfig::defines, e.g.:
//   {"KDMA_CHUNK_SIZE", "32768"}, {"KDMA_H2D_NUM_BUFS", "4"}
// This lets DmaEngine be constructed with different configs without modifying
// device kernel source files.
#ifdef KDMA_CHUNK_SIZE
static constexpr uint32_t kDmaChunkSize = KDMA_CHUNK_SIZE;
#else
static constexpr uint32_t kDmaChunkSize = 16u * 1024u;  // 16 KB (NOC_MAX_BURST on BH)
#endif
static constexpr uint32_t kDmaD2HPingBase = kDmaBufBase;                      // 0x80400
static constexpr uint32_t kDmaD2HPongBase = kDmaD2HPingBase + kDmaChunkSize;  // 0x84400

// H2D uses 8 buffers for deep read pipelining.  PCIe reads are non-posted
// (request→completion round-trip), so we need many in-flight reads to
// saturate the link.  noc_async_read_barrier() is global ("wait for ALL
// outstanding reads"), so we issue a batch of 8 reads, barrier, then drain
// all 8 to DRAM.  8 × 16 KB = 128 KB per core in the PCIe pipeline.
#ifdef KDMA_H2D_NUM_BUFS
static constexpr uint32_t kDmaH2DNumBufs = KDMA_H2D_NUM_BUFS;
#else
static constexpr uint32_t kDmaH2DNumBufs = 8u;
#endif
static constexpr uint32_t kDmaH2DBufStart = kDmaD2HPongBase + kDmaChunkSize;  // 0x88400
// buf[i] at kDmaH2DBufStart + i * kDmaChunkSize
// buf[7] ends at 0x88400 + 8*0x4000 = 0xA8400  (168 KB into L1 user region)

// Legacy aliases (still used by some test code)
static constexpr uint32_t kDmaH2DPingBase = kDmaH2DBufStart;
static constexpr uint32_t kDmaH2DPongBase = kDmaH2DBufStart + kDmaChunkSize;

// Ready flags: written by each processor after ring init, before spin loop.
// Host polls these after EnqueueMeshWorkload to know all kernels are live.
// Placed just before the ping-pong buffer region.
static constexpr uint32_t kDmaD2HReadyFlag = kDmaUserBase + 0x180u;  // 0x80180
static constexpr uint32_t kDmaH2DReadyFlag = kDmaUserBase + 0x184u;  // 0x80184

// Per-transfer progress counters — written by kernel at each major checkpoint.
// Host reads these on timeout to diagnose where the kernel is stuck.
// Values: 0=idle, 1=cmd_received, 2=prime_done, 3=in_loop, 4=drain+fence_done, 5=completion_seq_written
static constexpr uint32_t kDmaD2HProgress = kDmaUserBase + 0x188u;  // 0x80188
static constexpr uint32_t kDmaH2DProgress = kDmaUserBase + 0x18Cu;  // 0x8018C

// D2H completion: kernel writes last_seq to HOST done-flag via PCIe posted
// write (strong ordering after data).  L1 copy kept for debug.
// H2D completion: kernel writes last_seq to L1 (host polls via BAR read).
static constexpr uint32_t kDmaD2HCompletionSeq = kDmaUserBase + 0x190u;  // 0x80190 (L1, debug only)
static constexpr uint32_t kDmaH2DCompletionSeq = kDmaUserBase + 0x194u;  // 0x80194

// Host-push H2D handshake counters (monotonically increasing).
// push_seq: host increments after writing 8 chunks into H2D bufs via BAR.
// push_ack: kernel increments after draining those 8 chunks to DRAM.
// Host waits for push_ack >= batch_num before overwriting buffers.
static constexpr uint32_t kDmaH2DPushSeq = kDmaUserBase + 0x198u;  // 0x80198 (host writes)
static constexpr uint32_t kDmaH2DPushAck = kDmaUserBase + 0x19Cu;  // 0x8019C (kernel writes)

// Per-command diagnostic area — kernel writes after each done-flag write.
// Host reads on timeout to diagnose PCIe write failures.
//   [0] done_flag_lo      [1] done_flag_hi
//   [2] last_seq           [3] cmd_count (total cmds processed)
//   [4] sw_acked (noc_nonposted_writes_acked before done-flag barrier)
//   [5] hw_acked (NIU_MST_WR_ACK_RECEIVED after done-flag barrier)
//   [6] sw_issued (noc_nonposted_writes_num_issued after done-flag barrier)
//   [7] hw_sent (NIU_MST_NONPOSTED_WR_REQ_SENT after done-flag barrier)
static constexpr uint32_t kDmaD2HDiagBase = kDmaUserBase + 0x1A0u;  // 0x801A0
static constexpr uint32_t kDmaD2HDiagWords = 8u;

// H2D per-transfer timing diagnostic — kernel writes after each H2D command.
// Host reads via BAR when --verbose is passed to show where cycles go.
// Wall clock on WH runs at 1 GHz, so cycles ≈ nanoseconds.
//   [0] total_read_issue_cycles   — time calling noc_read_with_state
//   [1] total_read_barrier_cycles — time stalled in noc_async_read_barrier
//   [2] total_write_issue_cycles  — time calling noc_async_write (DRAM drain)
//   [3] total_write_barrier_cycles— time stalled in noc_async_write_barrier
//   [4] num_read_batches          — how many read barrier waits
//   [5] num_chunks                — total chunks transferred
//   [6] total_transfer_cycles     — wall clock from cmd start to completion
//   [7] noc_rd_resp_received      — HW counter: NIU_MST_RD_RESP_RECEIVED
static constexpr uint32_t kDmaH2DDiagBase = kDmaD2HDiagBase + kDmaD2HDiagWords * 4u;  // 0x801C0
static constexpr uint32_t kDmaH2DDiagWords = 8u;

// ── Engine topology ───────────────────────────────────────────────────────
static constexpr uint32_t kDmaCoreCols = 8u;                           // columns 0-7
static constexpr uint32_t kDmaCoreRows = 9u;                           // rows 0-8
static constexpr uint32_t kDmaNumCores = kDmaCoreCols * kDmaCoreRows;  // 72
