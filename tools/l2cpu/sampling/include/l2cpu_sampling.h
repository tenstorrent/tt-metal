// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * l2cpu_sampling.h: the region layout of the sampling application (fw/app.c) on top of the generic runtime
 * (l2cpu_boot.h) and the Tensix link block (l2cpu_link.h).
 *
 * Shared by the firmware (rv64, C), Tensix kernels (riscv32 JIT, C++) and the host (generated Python mirror
 * host/l2cpu/sampling/layout.py, tools/gen_boot_py.py). Rules as in l2cpu_boot.h: fixed-width types, explicit
 * offsets tied to the structs by static asserts, every synchronisation word on its own 64-byte line, producer
 * writes data -> fence -> sequence word, consumer reads the sequence word -> fence -> data. Change the layout =>
 * bump L2S_LAYOUT_VERSION (written into ident.app_layout_version).
 *
 * Region map (offsets from the region base):
 *   0x0001000  link block (l2cpu_link.h, L2CPU_LINK_SIZE): req_seq, done_seq, wait_status, landed, diag, push source
 *              table, ... The Tensix kernels' channel base is region + L2S_OFF_LINK.
 *   0x0002000  sampling control (l2s_ctrl_t): batch, vocab, flags, stream config, descriptors, per-user params,
 *              next_tokens
 *   0x0008000  timing ring (one record per published request)
 *   0x0020000  output ring (tokens of every published request)
 *   0x1000000  coherent logits zone (rows pushed through the Memory Port alias, read cached)
 *   0x2000000  uncached logits zone (rows pushed through the System Port alias, read uncached only, <= 2 readers)
 *   0x4000000  end (L2S_REGION_SIZE)
 */
#ifndef L2CPU_SAMPLING_H
#define L2CPU_SAMPLING_H

#include "l2cpu_boot.h"
#include "l2cpu_link.h"

/* ---- identity and geometry ------------------------------------------------------------------------------- */
#define L2S_APP_ID 2u         /* ident.app_id (examples/heartbeat is 1) */
#define L2S_LAYOUT_VERSION 2u /* 2: ctrl.user_base, users per hart = ceil(batch / 4) */
#define L2S_NHARTS 4u
#define L2S_MAX_USERS 32u
#define L2S_USERS_PER_HART 8u /* at most; a request gives ceil(batch / 4) users to each hart */
#define L2S_MAX_BANKS 24u
#define L2S_RING_SLOTS 2048u
#define L2S_TIMING_SLOTS 1024u

#define L2S_OFF_LINK 0x1000u /* = L2CPU_OFF_APP_CTRL */
#define L2S_OFF_CTRL 0x2000u /* = L2S_OFF_LINK + 0x1000 */
#define L2S_CTRL_SIZE 0x2000u
#define L2S_OFF_TIMING 0x8000u /* = L2CPU_OFF_APP_LOW */
#define L2S_OFF_RING 0x20000u
#define L2S_OFF_LOGITS 0x1000000u
#define L2S_OFF_LOGITS_UC 0x2000000u
#define L2S_REGION_SIZE 0x4000000u         /* 64 MiB: the minimum region of the sampling image */
#define L2S_LOGITS_ROW_STRIDE_QWEN 303872u /* 151936 bf16 = 4748 x 64 B */

/* ---- link words (absolute offsets; the link block is tensix/kernels/l2cpu_link.h) -------------------------------- */
#define L2S_OFF_REQ_SEQ (L2S_OFF_LINK + L2CPU_LINK_OFF_REQ_SEQ)         /* producer: request number */
#define L2S_OFF_DONE_SEQ (L2S_OFF_LINK + L2CPU_LINK_OFF_DONE_SEQ)       /* firmware: = req_seq when published */
#define L2S_OFF_WAIT_STATUS (L2S_OFF_LINK + L2CPU_LINK_OFF_WAIT_STATUS) /* Tensix wait kernel only */
#define L2S_OFF_LANDED (L2S_OFF_LINK + L2CPU_LINK_OFF_LANDED)           /* stream producer: L2S_LANDED(seq, n) */
#define L2S_OFF_DIAG (L2S_OFF_LINK + L2CPU_LINK_OFF_DIAG)
#define L2S_OFF_PUSH_SRC (L2S_OFF_LINK + L2CPU_LINK_OFF_PUSH_SRC)

/* ---- errors (header.error.code / hart_state.error; context = req_seq) ------------------------------------ */
#define L2S_ERR_BAD_DESC (L2CPU_ERR_APP + 0u)       /* descriptor / config validation failed; arg = L2S_BAD_* */
#define L2S_ERR_SEQ (L2CPU_ERR_APP + 1u)            /* req_seq moved backwards, or a worker got a foreign seq */
#define L2S_ERR_SAMPLE (L2CPU_ERR_APP + 2u)         /* sampling library returned an error code (arg) */
#define L2S_ERR_STREAM_TIMEOUT (L2CPU_ERR_APP + 3u) /* landed stopped advancing for the stream timeout; arg = need */
#define L2S_ERR_GATE_TIMEOUT (L2CPU_ERR_APP + 4u)   /* the uncached-reader gate stayed full for L2S_GATE_TIMEOUT_US */

#define L2S_BAD_BATCH 1u
#define L2S_BAD_VOCAB 2u
#define L2S_BAD_LOGITS_DTYPE 3u
#define L2S_BAD_LAYOUT 4u
#define L2S_BAD_SHAPE 5u
#define L2S_BAD_BANKS 6u
#define L2S_BAD_PAGE 7u
#define L2S_BAD_TOKENS_DTYPE 8u
#define L2S_BAD_UNMAPPABLE 9u
#define L2S_BAD_WORKSPACE 10u
#define L2S_BAD_LOCATION 11u /* unknown location, or a region-local tensor outside its zone */
#define L2S_BAD_STREAM 12u   /* FLAG_STREAMED with an unknown stream mode */

/* ---- application build flags (ident.build_flags >> L2CPU_BUILD_APP_SHIFT) -------------------------------- */
#define L2S_BUILD_STREAM 0x1u /* stream protocol implemented */
#define L2S_BUILD_RVV 0x2u    /* sampling library built with RVV */
#define L2S_BUILD_FAST 0x4u   /* sampling library bf16 fast paths (SAMPLING_FAST=1) */

/* ---- application counters (generic counters[h].app[i], byte offsets from L2CPU_OFF_COUNTERS + 64*h) ------ */
#define L2S_CNT_REQUESTS (L2CPU_CNT_APP + 0x00u)  /* requests this hart worked on */
#define L2S_CNT_USERS (L2CPU_CNT_APP + 0x08u)     /* rows sampled */
#define L2S_CNT_NAN (L2CPU_CNT_APP + 0x10u)       /* NaN logits seen (treated as -inf) */
#define L2S_CNT_KMAX_CAPS (L2CPU_CNT_APP + 0x18u) /* rows where the K_MAX cap removed candidates */

/* ---- tensors --------------------------------------------------------------------------------------------- */
#define L2S_DTYPE_FP32 0u /* = X280S_DTYPE_F32 */
#define L2S_DTYPE_BF16 1u /* = X280S_DTYPE_BF16 */
#define L2S_DTYPE_UINT32 2u
#define L2S_DTYPE_INT32 3u
#define L2S_LAYOUT_ROW_MAJOR 0u
#define L2S_LAYOUT_TILE 1u /* 32x32 tiles of four 16x16 faces, row-major faces, tile = page */
#define L2S_TILE_DIM 32u
#define L2S_FACE_DIM 16u
#define L2S_LOC_NOC 0u       /* interleaved ttnn DRAM tensor, reached through TLB windows */
#define L2S_LOC_REGION 1u    /* rows in the coherent zone: region + local_off + row * row_stride (ROW_MAJOR) */
#define L2S_LOC_REGION_UC 2u /* rows in the uncached zone (same addressing); read uncached, <= 2 harts at a time */

/* ---- request flags (ctrl.flags). Tokens always go to next_tokens[u]. -------------------------------------- */
#define L2S_FLAG_TOKENS_REMOTE 1u /* also write the tokens descriptor's tensor through an uncached TLB window */
#define L2S_FLAG_NO_RING 2u       /* do not write the output ring */
#define L2S_FLAG_STREAMED 4u      /* rows land after req_seq: wait on landed (stream protocol below) */

/*
 * Stream protocol. The producer (the Tensix push kernel in streamed mode) publishes req_seq + doorbell at the START of
 * the push, then after each increment of data has fully landed (write barrier) writes landed = L2S_LANDED(req_seq,
 * count) as one aligned 32-bit store; count only grows within a request.
 *   ROWS: count = rows complete, in the order l2s_stream_row(0.., uph) restricted to rows < batch, uph =
 *         l2s_users_per_hart(batch) (= the push kernel's streamed order with groups = 4, group_rows = uph).
 *   stream.timeout_us: a consumer parks with L2S_ERR_STREAM_TIMEOUT if count stops advancing that long (0 -> 100 ms).
 */
#define L2S_STREAM_ROWS 1u
#define L2S_LANDED(seq, count) ((((uint32_t)(seq) & 0xFFFFu) << 16) | ((uint32_t)(count) & 0xFFFFu))

/* ---- wait bounds (README "Waits and their bounds") ------------------------------------------------------- */
#define L2S_STREAM_TIMEOUT_US 100000u /* default of stream.timeout_us */
#define L2S_GATE_TIMEOUT_US 100000u   /* uncached-reader gate: a holder copies one row (< 0.1 ms) */
#define L2S_WORK_TIMEOUT_US \
    1000000u              /* default of ctrl.work_timeout_us: hart 0 waits for its workers (+ stream timeout) */
#define L2S_UC_READERS 2u /* harts reading the uncached zone at once (2 scale, 3+ collapse) */

#if !defined(__ASSEMBLER__)
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t x, y, noc, _pad0;
    uint32_t _pad1;
    uint64_t addr; /* NoC address of page-round 0 in this bank */
} l2s_bank_t;
L2CPU_STATIC_ASSERT(sizeof(l2s_bank_t) == 16, "bank");

/*
 * Tensor descriptor. location REGION / REGION_UC: rows inside the region (row r at region + local_off +
 * r * row_stride), ROW_MAJOR only, bank fields ignored. location NOC: a 2-D view [rows, cols] of an interleaved DRAM
 * tensor; pages round-robin over num_banks banks from first_bank (l2s_page_location()).
 */
typedef struct {
    uint32_t dtype, layout, rows, cols;
    uint32_t page_size;   /* bytes of data per page */
    uint32_t page_stride; /* bytes between consecutive pages in one bank */
    uint32_t num_banks, first_bank;
    uint32_t location;   /* L2S_LOC_* */
    uint32_t row_stride; /* REGION*: bytes between rows */
    uint32_t local_off;  /* REGION*: region offset of row 0 */
    uint32_t _pad[5];
    l2s_bank_t bank[L2S_MAX_BANKS];
    uint8_t _reserved[0x200 - 64 - 16 * L2S_MAX_BANKS];
} l2s_tensor_desc_t;
L2CPU_STATIC_ASSERT(sizeof(l2s_tensor_desc_t) == 0x200, "tensor desc");

typedef struct {
    float temperature; /* <= 0: greedy */
    uint32_t top_k;    /* 0 = none (capped at K_MAX) */
    float top_p;
    uint32_t _pad0;
    uint64_t seed;
    uint64_t _pad[5];
} l2s_user_params_t; /* the first 24 bytes equal x280s_params_t */
L2CPU_STATIC_ASSERT(sizeof(l2s_user_params_t) == 64, "params");

typedef struct {
    uint32_t mode;       /* L2S_STREAM_ROWS */
    uint32_t timeout_us; /* 0 -> L2S_STREAM_TIMEOUT_US */
    uint32_t _reserved0;
    uint32_t _pad[13];
} l2s_stream_cfg_t;
L2CPU_STATIC_ASSERT(sizeof(l2s_stream_cfg_t) == 64, "stream cfg");

typedef struct { /* written by the firmware at boot: constants of the build */
    uint32_t ring_offset, ring_slots, slot_size, tokens_offset, max_users, timing_offset, timing_slots;
    uint32_t _pad[9];
} l2s_ring_desc_t;
L2CPU_STATIC_ASSERT(sizeof(l2s_ring_desc_t) == 64, "ring desc");

typedef struct {
    l2cpu_line32_t batch;           /* 0x000 users in this request (1..32) */
    l2cpu_line32_t vocab;           /* 0x040 real vocab V (columns >= V ignored) */
    l2cpu_line32_t vocab_padded;    /* 0x080 padded vocab (<= logits.cols) */
    l2cpu_line32_t step_seq_base;   /* 0x0C0 sampling step = req_seq - step_seq_base */
    l2cpu_line32_t flags;           /* 0x100 L2S_FLAG_* */
    l2cpu_line32_t ring_wr;         /* 0x140 firmware: output ring slots written */
    l2cpu_line32_t timing_count;    /* 0x180 firmware: timing records written */
    l2cpu_line32_t seq_skips;       /* 0x1C0 firmware: requests skipped because req_seq advanced by more than 1 */
    l2s_stream_cfg_t stream;        /* 0x200 host, before streamed requests */
    l2s_ring_desc_t ring;           /* 0x240 firmware */
    l2cpu_line32_t work_timeout_us; /* 0x280 host: hart 0's bound on its workers (0 -> L2S_WORK_TIMEOUT_US) */
    l2cpu_line32_t user_base;       /* 0x2C0 host: global index of user 0 of this request (several tiles split a
                                       batch): the draw uses user_base + u, a remote token goes to element
                                       user_base + u; params, next_tokens and the ring stay tile-local (index u) */
    uint8_t _pad0[0x400 - 0x300];
    l2s_tensor_desc_t logits;                  /* 0x400 */
    l2s_tensor_desc_t tokens;                  /* 0x600 next-input token tensor (uint32/int32), FLAG_TOKENS_REMOTE */
    l2s_user_params_t params[L2S_MAX_USERS];   /* 0x800 */
    l2cpu_line32_t next_tokens[L2S_MAX_USERS]; /* 0x1000 token of user u for the next step (always written) */
    uint8_t _reserved[0x2000 - 0x1800];
} l2s_ctrl_t;
L2CPU_STATIC_ASSERT(sizeof(l2s_ctrl_t) == L2S_CTRL_SIZE, "ctrl");

/* One record per published request, slot (count - 1) % L2S_TIMING_SLOTS, written before done_seq. mtime_* in the
 * CLINT timebase (ident.mtime_hz); cyc_* are mcycle sums (hart 0 unless named). */
typedef struct {
    uint32_t req_seq, batch;
    uint64_t mtime_wake;    /* hart 0 saw the request */
    uint64_t mtime_publish; /* just before done_seq */
    uint32_t cyc_read;      /* hart 0: logits rows -> workspace (or the stream wait) */
    uint32_t cyc_sample;    /* hart 0: sampling library */
    uint32_t cyc_write;     /* hart 0: token write-back + barrier */
    uint32_t cyc_wait;      /* hart 0: waiting for its workers */
    uint32_t cyc_total;     /* hart 0: wake -> publish */
    uint32_t cyc_worker[3]; /* harts 1..3: their share (0 if unused) */
    uint32_t _pad[2];
} l2s_timing_t;
L2CPU_STATIC_ASSERT(sizeof(l2s_timing_t) == 64, "timing");

typedef struct { /* slot (ring_wr - 1) % L2S_RING_SLOTS holds the request published last */
    uint32_t req_seq, batch, step;
    uint32_t _pad[13];
    uint32_t tok[L2S_MAX_USERS];
    uint32_t _pad2[16];
} l2s_ring_slot_t;
L2CPU_STATIC_ASSERT(sizeof(l2s_ring_slot_t) == 256, "ring slot");
#endif /* !__ASSEMBLER__ */

/* ---- explicit offsets (absolute from the region base; the Python mirror reads these) ---------------------- */
#define L2S_OFF_BATCH (L2S_OFF_CTRL + 0x000u)
#define L2S_OFF_VOCAB (L2S_OFF_CTRL + 0x040u)
#define L2S_OFF_VOCAB_PADDED (L2S_OFF_CTRL + 0x080u)
#define L2S_OFF_STEP_SEQ_BASE (L2S_OFF_CTRL + 0x0C0u)
#define L2S_OFF_FLAGS (L2S_OFF_CTRL + 0x100u)
#define L2S_OFF_RING_WR (L2S_OFF_CTRL + 0x140u)
#define L2S_OFF_TIMING_COUNT (L2S_OFF_CTRL + 0x180u)
#define L2S_OFF_SEQ_SKIPS (L2S_OFF_CTRL + 0x1C0u)
#define L2S_OFF_STREAM_MODE (L2S_OFF_CTRL + 0x200u)
#define L2S_OFF_STREAM_TIMEOUT_US (L2S_OFF_CTRL + 0x204u)
#define L2S_OFF_RING_DESC (L2S_OFF_CTRL + 0x240u)
#define L2S_OFF_WORK_TIMEOUT_US (L2S_OFF_CTRL + 0x280u)
#define L2S_OFF_USER_BASE (L2S_OFF_CTRL + 0x2C0u)
#define L2S_OFF_LOGITS_DESC (L2S_OFF_CTRL + 0x400u)
#define L2S_OFF_TOKENS_DESC (L2S_OFF_CTRL + 0x600u)
#define L2S_OFF_PARAMS (L2S_OFF_CTRL + 0x800u)       /* + 64*u */
#define L2S_OFF_NEXT_TOKENS (L2S_OFF_CTRL + 0x1000u) /* + 64*u: u32 token id */

#define L2S_DESC_DTYPE 0x00u
#define L2S_DESC_LAYOUT 0x04u
#define L2S_DESC_ROWS 0x08u
#define L2S_DESC_COLS 0x0Cu
#define L2S_DESC_PAGE_SIZE 0x10u
#define L2S_DESC_PAGE_STRIDE 0x14u
#define L2S_DESC_NUM_BANKS 0x18u
#define L2S_DESC_FIRST_BANK 0x1Cu
#define L2S_DESC_LOCATION 0x20u
#define L2S_DESC_ROW_STRIDE 0x24u
#define L2S_DESC_LOCAL_OFF 0x28u
#define L2S_DESC_BANK 0x40u /* + 16*i: u8 x, u8 y, u8 noc, pad, u64 addr at +8 */
#define L2S_BANK_SIZE 16u
#define L2S_DESC_SIZE 0x200u

#define L2S_UP_TEMPERATURE 0x00u
#define L2S_UP_TOP_K 0x04u
#define L2S_UP_TOP_P 0x08u
#define L2S_UP_SEED 0x10u
#define L2S_UP_SIZE 64u

#define L2S_TIMING_SIZE 64u
#define L2S_TM_REQ_SEQ 0x00u
#define L2S_TM_BATCH 0x04u
#define L2S_TM_MTIME_WAKE 0x08u
#define L2S_TM_MTIME_PUBLISH 0x10u
#define L2S_TM_CYC_READ 0x18u
#define L2S_TM_CYC_SAMPLE 0x1Cu
#define L2S_TM_CYC_WRITE 0x20u
#define L2S_TM_CYC_WAIT 0x24u
#define L2S_TM_CYC_TOTAL 0x28u
#define L2S_TM_CYC_WORKER 0x2Cu /* + 4*(h-1) */

#define L2S_RING_SLOT_SIZE 256u
#define L2S_RS_REQ_SEQ 0x00u
#define L2S_RS_BATCH 0x04u
#define L2S_RS_STEP 0x08u
#define L2S_RS_TOK 0x40u /* + 4*u */

#if !defined(__ASSEMBLER__)
L2CPU_STATIC_ASSERT(L2S_OFF_LINK == L2CPU_OFF_APP_CTRL && L2CPU_LINK_SIZE <= L2S_OFF_CTRL - L2S_OFF_LINK, "link block");
L2CPU_STATIC_ASSERT(L2S_OFF_CTRL + L2S_CTRL_SIZE <= L2CPU_OFF_APP_CTRL + L2CPU_APP_CTRL_SIZE, "ctrl in APP_CTRL");
L2CPU_STATIC_ASSERT(
    L2S_OFF_TIMING >= L2CPU_OFF_APP_LOW && L2S_OFF_TIMING + L2S_TIMING_SLOTS * L2S_TIMING_SIZE <= L2S_OFF_RING,
    "timing in APP_LOW");
L2CPU_STATIC_ASSERT(
    L2S_OFF_RING + L2S_RING_SLOTS * L2S_RING_SLOT_SIZE <= L2CPU_OFF_APP_LOW + L2CPU_APP_LOW_SIZE, "ring in APP_LOW");
L2CPU_STATIC_ASSERT(
    L2S_OFF_LOGITS >= L2CPU_OFF_APP_HIGH &&
        L2S_OFF_LOGITS + L2S_MAX_USERS * L2S_LOGITS_ROW_STRIDE_QWEN <= L2S_OFF_LOGITS_UC,
    "coherent zone");
L2CPU_STATIC_ASSERT(L2S_OFF_LOGITS_UC + L2S_MAX_USERS * L2S_LOGITS_ROW_STRIDE_QWEN <= L2S_REGION_SIZE, "uncached zone");
L2CPU_STATIC_ASSERT(L2S_NHARTS == L2CPU_NHARTS && L2S_NHARTS * L2S_USERS_PER_HART == L2S_MAX_USERS, "users split");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, batch) + L2S_OFF_CTRL == L2S_OFF_BATCH, "batch");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, vocab) + L2S_OFF_CTRL == L2S_OFF_VOCAB, "vocab");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, vocab_padded) + L2S_OFF_CTRL == L2S_OFF_VOCAB_PADDED, "vpad");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, step_seq_base) + L2S_OFF_CTRL == L2S_OFF_STEP_SEQ_BASE, "ssb");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, flags) + L2S_OFF_CTRL == L2S_OFF_FLAGS, "flags");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, ring_wr) + L2S_OFF_CTRL == L2S_OFF_RING_WR, "ring_wr");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, timing_count) + L2S_OFF_CTRL == L2S_OFF_TIMING_COUNT, "timing_count");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, seq_skips) + L2S_OFF_CTRL == L2S_OFF_SEQ_SKIPS, "seq_skips");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, stream.mode) + L2S_OFF_CTRL == L2S_OFF_STREAM_MODE, "stream mode");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, stream.timeout_us) + L2S_OFF_CTRL == L2S_OFF_STREAM_TIMEOUT_US, "stream to");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, ring) + L2S_OFF_CTRL == L2S_OFF_RING_DESC, "ring desc");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, work_timeout_us) + L2S_OFF_CTRL == L2S_OFF_WORK_TIMEOUT_US, "work to");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, user_base) + L2S_OFF_CTRL == L2S_OFF_USER_BASE, "user base");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, logits) + L2S_OFF_CTRL == L2S_OFF_LOGITS_DESC, "logits");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, tokens) + L2S_OFF_CTRL == L2S_OFF_TOKENS_DESC, "tokens");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, params) + L2S_OFF_CTRL == L2S_OFF_PARAMS, "params");
L2CPU_STATIC_ASSERT(offsetof(l2s_ctrl_t, next_tokens) + L2S_OFF_CTRL == L2S_OFF_NEXT_TOKENS, "next_tokens");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, dtype) == L2S_DESC_DTYPE, "d.dtype");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, layout) == L2S_DESC_LAYOUT, "d.layout");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, rows) == L2S_DESC_ROWS, "d.rows");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, cols) == L2S_DESC_COLS, "d.cols");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, page_size) == L2S_DESC_PAGE_SIZE, "d.ps");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, page_stride) == L2S_DESC_PAGE_STRIDE, "d.pst");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, num_banks) == L2S_DESC_NUM_BANKS, "d.nb");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, first_bank) == L2S_DESC_FIRST_BANK, "d.fb");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, location) == L2S_DESC_LOCATION, "d.loc");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, row_stride) == L2S_DESC_ROW_STRIDE, "d.rs");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, local_off) == L2S_DESC_LOCAL_OFF, "d.lo");
L2CPU_STATIC_ASSERT(offsetof(l2s_tensor_desc_t, bank) == L2S_DESC_BANK, "d.bank");
L2CPU_STATIC_ASSERT(offsetof(l2s_bank_t, addr) == 8, "b.addr");
L2CPU_STATIC_ASSERT(offsetof(l2s_user_params_t, temperature) == L2S_UP_TEMPERATURE, "up.t");
L2CPU_STATIC_ASSERT(offsetof(l2s_user_params_t, top_k) == L2S_UP_TOP_K, "up.k");
L2CPU_STATIC_ASSERT(offsetof(l2s_user_params_t, top_p) == L2S_UP_TOP_P, "up.p");
L2CPU_STATIC_ASSERT(offsetof(l2s_user_params_t, seed) == L2S_UP_SEED, "up.seed");
L2CPU_STATIC_ASSERT(offsetof(l2s_timing_t, mtime_wake) == L2S_TM_MTIME_WAKE, "tm.wake");
L2CPU_STATIC_ASSERT(offsetof(l2s_timing_t, mtime_publish) == L2S_TM_MTIME_PUBLISH, "tm.pub");
L2CPU_STATIC_ASSERT(offsetof(l2s_timing_t, cyc_read) == L2S_TM_CYC_READ, "tm.read");
L2CPU_STATIC_ASSERT(offsetof(l2s_timing_t, cyc_sample) == L2S_TM_CYC_SAMPLE, "tm.sample");
L2CPU_STATIC_ASSERT(offsetof(l2s_timing_t, cyc_write) == L2S_TM_CYC_WRITE, "tm.write");
L2CPU_STATIC_ASSERT(offsetof(l2s_timing_t, cyc_wait) == L2S_TM_CYC_WAIT, "tm.wait");
L2CPU_STATIC_ASSERT(offsetof(l2s_timing_t, cyc_total) == L2S_TM_CYC_TOTAL, "tm.total");
L2CPU_STATIC_ASSERT(offsetof(l2s_timing_t, cyc_worker) == L2S_TM_CYC_WORKER, "tm.worker");
L2CPU_STATIC_ASSERT(offsetof(l2s_ring_slot_t, step) == L2S_RS_STEP, "rs.step");
L2CPU_STATIC_ASSERT(offsetof(l2s_ring_slot_t, tok) == L2S_RS_TOK, "rs.tok");

/* ---- page and element location: the ONLY place that encodes the interleaving rule ------------------------- */
static inline uint32_t l2s_dtype_size(uint32_t dtype) {
    return dtype == L2S_DTYPE_BF16 ? 2u : (dtype <= L2S_DTYPE_INT32 ? 4u : 0u);
}

/* Page p -> bank and NoC address (tt-metal InterleavedAddrGen with a start bank): k = first_bank + p,
 * bank = k % num_banks, addr = bank[bank].addr + (k / num_banks) * page_stride. */
static inline void l2s_page_location(
    const l2s_tensor_desc_t* d, uint32_t page, uint32_t* bank_out, uint64_t* addr_out) {
    uint32_t k = d->first_bank + page;
    uint32_t b = k % d->num_banks;
    *bank_out = b;
    *addr_out = d->bank[b].addr + (uint64_t)(k / d->num_banks) * d->page_stride;
}

/* Element (row, col) -> page and byte offset inside it. Returns 0 on success. */
static inline int l2s_elem_location(
    const l2s_tensor_desc_t* d, uint32_t row, uint32_t col, uint32_t* page_out, uint32_t* off_out) {
    uint32_t es = l2s_dtype_size(d->dtype);
    if (es == 0 || row >= d->rows || col >= d->cols || d->page_size == 0) {
        return -1;
    }
    if (d->layout == L2S_LAYOUT_ROW_MAJOR) {
        uint32_t row_bytes = d->cols * es;
        uint32_t ppr = (row_bytes + d->page_size - 1) / d->page_size;
        uint32_t b = col * es;
        *page_out = row * ppr + b / d->page_size;
        *off_out = b % d->page_size;
        return 0;
    }
    if (d->layout == L2S_LAYOUT_TILE) {
        uint32_t tpr = d->cols / L2S_TILE_DIM;
        uint32_t rr = row % L2S_TILE_DIM, cc = col % L2S_TILE_DIM;
        uint32_t face = (rr / L2S_FACE_DIM) * 2u + (cc / L2S_FACE_DIM);
        uint32_t idx = face * L2S_FACE_DIM * L2S_FACE_DIM + (rr % L2S_FACE_DIM) * L2S_FACE_DIM + (cc % L2S_FACE_DIM);
        *page_out = (row / L2S_TILE_DIM) * tpr + col / L2S_TILE_DIM;
        *off_out = idx * es;
        return 0;
    }
    return -1;
}

/* Consecutive elements from (row, col) that are contiguous in one page. */
static inline uint32_t l2s_run_length(const l2s_tensor_desc_t* d, uint32_t row, uint32_t col) {
    uint32_t es = l2s_dtype_size(d->dtype);
    (void)row;
    if (d->layout == L2S_LAYOUT_TILE) {
        return L2S_FACE_DIM - (col % L2S_FACE_DIM);
    }
    uint32_t off = (col * es) % d->page_size;
    uint32_t left_page = (d->page_size - off) / es;
    uint32_t left_row = d->cols - col;
    return left_page < left_row ? left_page : left_row;
}

/* Users per hart of a request: hart h samples users h * uph .. h * uph + uph - 1 (< batch). Batch 32 -> 8, batch 8 ->
 * 2 (a tile that serves a quarter of a batch-32 step still uses all four harts), batch 1 -> 1. */
static inline uint32_t l2s_users_per_hart(uint32_t batch) { return (batch + L2S_NHARTS - 1u) / L2S_NHARTS; }

/* ROWS stream order: the p-th row completed is (p % 4) * uph + p / 4 (every hart's first row early); the inverse
 * gives row u's position among the rows < batch. */
static inline uint32_t l2s_stream_row(uint32_t p, uint32_t uph) { return (p % L2S_NHARTS) * uph + p / L2S_NHARTS; }
static inline uint32_t l2s_stream_pos(uint32_t u, uint32_t batch) {
    uint32_t pos = 0, uph = l2s_users_per_hart(batch);
    for (uint32_t p = 0; p < L2S_NHARTS * uph; p++) {
        uint32_t r = l2s_stream_row(p, uph);
        if (r == u) {
            return pos;
        }
        if (r < batch) {
            pos++;
        }
    }
    return pos;
}
#endif /* !__ASSEMBLER__ */

#endif /* L2CPU_SAMPLING_H */
