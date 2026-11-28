// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * coherent_alias.c - test image: cached loads and stores of the tile's local
 * DRAM on the host's command.
 *
 * Hart 0 serves commands from the block (x280_hw_probe.h) on a region of
 * header.lines 64-byte lines at X280_LOCAL_DDR_TILE_BASE + header.ddr_offset,
 * through its cached Memory Port: READ copies lines into the block, WRITE
 * stores a pattern with no flush, FLUSH writes each line to CCACHE0_FLUSH64,
 * EXIT returns. Between commands the host has a Tensix kernel read or write the
 * same bytes over the NOC, so the test sees what each NOC path observes of the
 * X280's caches. It polls STOP while it waits (obligation 1). Harts 1-3 return
 * at once.
 *
 * Used by X280CoherentAliasHw (tests/device/test_x280_coherent_alias_hw.cpp).
 */
#include "x280_hart_protocol.h"
#include "x280_hw_probe.h"

#include "rt/x280_idle.h"
#include "rt/x280_pinned_input.h"

static inline volatile uint64_t* word_at(uint64_t block, uint32_t offset) {
    return (volatile uint64_t*)(uintptr_t)(block + offset);
}

static uint64_t serve(uint64_t block, uint64_t region, uint64_t lines) {
    const uint64_t op = *word_at(block, X280_PROBE_CMD_OP);
    const uint64_t first = *word_at(block, X280_PROBE_CMD_FIRST);
    const uint64_t count = *word_at(block, X280_PROBE_CMD_COUNT);
    const uint32_t seed = (uint32_t)*word_at(block, X280_PROBE_CMD_SEED);
    if (first > lines || count > lines - first || (op == X280_PROBE_OP_READ && count > X280_PROBE_DATA_LINES)) {
        return X280_PROBE_STATUS_BAD_COMMAND;
    }

    for (uint64_t i = 0; i < count; i++) {
        const uint64_t line = region + (first + i) * X280_PROBE_LINE_BYTES;
        if (op == X280_PROBE_OP_READ) {
            const volatile uint64_t* src = (volatile const uint64_t*)(uintptr_t)line;
            volatile uint64_t* dst = word_at(block, X280_PROBE_DATA + (uint32_t)i * X280_PROBE_LINE_BYTES);
            for (uint32_t w = 0; w < X280_PROBE_LINE_BYTES / 8u; w++) {
                dst[w] = src[w];
            }
        } else if (op == X280_PROBE_OP_WRITE) {
            volatile uint32_t* dst = (volatile uint32_t*)(uintptr_t)line;
            for (uint32_t w = 0; w < X280_PROBE_LINE_BYTES / 4u; w++) {
                dst[w] = x280_probe_pattern(seed, (uint32_t)(first + i), w);
            }
        } else if (op == X280_PROBE_OP_FLUSH) {
            x280_fence_iorw();
            *(volatile uint64_t*)(uintptr_t)X280_CCACHE0_FLUSH64_ADDR = line;
        } else {
            return X280_PROBE_STATUS_BAD_COMMAND;
        }
    }
    x280_fence_iorw();
    return X280_PROBE_STATUS_OK;
}

int main(uint64_t hartid) {
    if (hartid != 0) {
        return 0;
    }
    const uint64_t run_value = x280_hart_control(0)->run_value;
    const uint64_t block = x280_hart_args(0);
    if (block == 0 || x280_hart_args_bytes() < X280_PROBE_BLOCK_BYTES ||
        *word_at(block, X280_PROBE_RUN_VALUE) != run_value) {
        x280_hart_set_result(0, X280_PROBE_RESULT_BAD_ARGS);
        return 0;
    }
    const uint64_t region = X280_LOCAL_DDR_TILE_BASE + *word_at(block, X280_PROBE_DDR_OFFSET);
    const uint64_t lines = *word_at(block, X280_PROBE_REGION_LINES);

    uint64_t served = 0;
    uint64_t last = *word_at(block, X280_PROBE_CMD_SEQ);
    for (;;) {
        x280_fence_ir();
        const uint64_t seq = *word_at(block, X280_PROBE_CMD_SEQ);
        if (seq == last) {
            if (x280_hart_stop_requested(0, run_value)) {
                break;
            }
            x280_pause();
            continue;
        }
        x280_fence_ir();
        const int exit = *word_at(block, X280_PROBE_CMD_OP) == X280_PROBE_OP_EXIT;
        *word_at(block, X280_PROBE_RESP_STATUS) = exit ? X280_PROBE_STATUS_OK : serve(block, region, lines);
        x280_fence_ow();
        *word_at(block, X280_PROBE_RESP_SEQ) = seq;
        x280_fence_ow();
        last = seq;
        served++;
        if (exit) {
            break;
        }
    }
    x280_hart_set_result(0, served);
    return 0;
}
