// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry.c - experimental X280 telemetry active image.
 *
 * Linked behind the lifecycle harness (src/x280_active_start.c), which calls
 * main on all four harts and returns them to resident idle afterwards.
 * For now each hart just reports its hart id as the result.
 */
#include "rt/x280_idle.h"

#include "common/telemetry_command.h"
#include "common/telemetry_table.h"

static volatile TelemetryTable* const g_table = (volatile TelemetryTable*)TABLE_ADDR;
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

void clear_memory(volatile void* ptr, size_t size) {
    volatile uint8_t* curr = (volatile uint8_t*)ptr;
    for (size_t i = 0; i < size; ++i) {
        *curr = 0;
        ++curr;
    }
}

void copy_memory(volatile void* dest_ptr, volatile const void* src_ptr, size_t num_bytes_to_copy) {
    const volatile uint8_t* src = (volatile const uint8_t*)src_ptr;
    volatile uint8_t* dest = (volatile uint8_t*)dest_ptr;
    for (size_t i = 0; i < num_bytes_to_copy; ++i) {
        *dest = *src;
        ++dest;
        ++src;
    }
}

size_t str_len(const char* str) {
    if (str == NULL) {
        return 0;
    }

    size_t len = 0;
    while (*str) {
        ++len;
        ++str;
    }

    return len;
}

const char* TT_FIRMWARE_VERSION_STR = "tt_firmware_version";
const char* FIRMWARE_VERSION_STR = "0.0.1-alpha";
const uint64_t TABLE_VERSION = 0x20261026;  // October 26, 2026

void init_table() {
    clear_memory(&(g_table->header), sizeof(TableHeader));

    // This is the version metric.
    TableRow row;
    clear_memory(&row, sizeof(TableRow));
    copy_memory(row.metric_name, TT_FIRMWARE_VERSION_STR, str_len(TT_FIRMWARE_VERSION_STR));
    row.metric_type = metric_type_string;
    copy_memory(row.str_data, FIRMWARE_VERSION_STR, str_len(FIRMWARE_VERSION_STR));

    copy_memory(g_table->rows, &row, sizeof(TableRow));

    g_table->header.table_version = TABLE_VERSION;
    g_table->header.sequence_number = 0;
    g_table->header.num_rows = 1;
}

void do_work(uint64_t run_value) {
    for (;;) {
        if (x280_hart_stop_requested(0, run_value)) {
            break;
        }
    }
}

int main(uint64_t hartid) {
    uint64_t run_value = x280_hart_control(hartid)->run_value;
    if (hartid == 0) {
        init_table();
        do_work(run_value);
        x280_hart_set_result(hartid, hartid);
    } else {
        x280_hart_set_result(hartid, hartid);
        return 0;
    }
}
