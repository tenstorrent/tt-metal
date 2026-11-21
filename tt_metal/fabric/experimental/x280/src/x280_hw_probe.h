// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * x280_hw_probe.h -- the argument block shared by the doorbell and
 * coherent-alias test images (src/doorbell.c, src/coherent_alias.c), their
 * device tests and the Tensix kernel those tests run. C, C++ and the Tensix
 * kernel compiler all include it, so it holds only integer constants and one
 * pattern function.
 *
 * The block lives in the tile's available LIM, which the X280 reaches uncached.
 * Every line is 64 bytes. The host writes the whole block before enqueue, so
 * every line the image later stores into has initialised ECC, and afterwards
 * writes only whole lines (write_available_lim):
 *
 *   line 0  header    host: run value, DDR offset, region lines
 *   line 1  command   host: op, first line, line count, seed
 *   line 2  doorbell  host: command sequence, written after line 1
 *   line 3  response  image: completed sequence, status
 *   line 4  record    image: the doorbell image's wake record
 *   line 5+ data      image: lines a READ copied out
 */
#ifndef X280_HW_PROBE_H
#define X280_HW_PROBE_H

#include <stdint.h>

#define X280_PROBE_LINE_BYTES 64u
#define X280_PROBE_DATA_LINES 32u
#define X280_PROBE_BLOCK_BYTES ((5u + X280_PROBE_DATA_LINES) * X280_PROBE_LINE_BYTES)

/* Byte offsets of the u64 words, from the start of the block. */
#define X280_PROBE_RUN_VALUE 0x00u
#define X280_PROBE_DDR_OFFSET 0x08u
#define X280_PROBE_REGION_LINES 0x10u

#define X280_PROBE_CMD_OP 0x40u
#define X280_PROBE_CMD_FIRST 0x48u
#define X280_PROBE_CMD_COUNT 0x50u
#define X280_PROBE_CMD_SEED 0x58u
#define X280_PROBE_CMD_SEQ 0x80u

#define X280_PROBE_RESP_SEQ 0xC0u
#define X280_PROBE_RESP_STATUS 0xC8u

/* Doorbell record. ARMED holds the run value once hart 0 has enabled the
 * doorbell and is about to enter WFI. */
#define X280_PROBE_DB_ARMED 0x100u
#define X280_PROBE_DB_WAKES 0x108u       /* doorbell wakes: MEIP set and source 6 claimed */
#define X280_PROBE_DB_TIMER_WAKES 0x110u /* wakes by the STOP-poll timer alone */
#define X280_PROBE_DB_OTHER_WAKES 0x118u /* wakes with neither MEIP nor MTIP, or a claim of another source */
#define X280_PROBE_DB_SOURCE 0x120u      /* source claimed on the last doorbell wake */
#define X280_PROBE_DB_VALUE 0x128u       /* last value popped from the MSI catcher */
#define X280_PROBE_DB_POPS 0x130u        /* values popped on the last doorbell wake */
#define X280_PROBE_DB_ENABLE 0x138u      /* hart 0 M-context enable word 0, read back after arming */

#define X280_PROBE_DATA 0x140u

/* Coherent-alias commands. */
#define X280_PROBE_OP_READ 1u  /* cached loads of the lines, copied to data */
#define X280_PROBE_OP_WRITE 2u /* cached stores of the seed's pattern, no flush */
#define X280_PROBE_OP_FLUSH 3u /* CCACHE0_FLUSH64 of each line */
#define X280_PROBE_OP_EXIT 4u  /* return from main */

#define X280_PROBE_STATUS_OK 1u
#define X280_PROBE_STATUS_BAD_COMMAND 2u

/* Published as hart 0's result when the block is absent, short or stale. */
#define X280_PROBE_RESULT_BAD_ARGS 0xBAD0A7650000000ull

/* Word `word` (0..15) of line `line` of the pattern for `seed`. Never 0, so a
 * zeroed line never matches. */
static inline uint32_t x280_probe_pattern(uint32_t seed, uint32_t line, uint32_t word) {
    uint32_t v = seed ^ (line * 0x9E3779B1u) ^ (word * 0x85EBCA77u);
    v ^= v >> 15;
    v *= 0x2C1B3C6Du;
    v ^= v >> 12;
    return v | 1u;
}

#endif /* X280_HW_PROBE_H */
