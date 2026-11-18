// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Standalone skeleton: booted straight from reset, no resident idle. Hart 0 publishes a
// "I'm alive" word and a counter in LIM; harts 1-3 park. It never returns.
// The heartbeat address and magic are shared with the host in x280_address_map.h.

#include <stdint.h>

#include "rt/x280_hw.h"
#include "noc.h"
#include "rt/x280_idle.h"

#include "arc_info.h"
#include "common/telemetry_table_v2.h"
#include "telemetry_helpers.h"

static volatile MetricsTelemetryTable* const g_table = (volatile MetricsTelemetryTable*)TABLE_ADDR;
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

static ArcInfo g_arc_info = {.do_not_put_into_bss_pal = 1867, .version = 0, .length = 0};

typedef struct {
    uint64_t magic;    // X280_STANDALONE_HEARTBEAT_MAGIC once the firmware is up; the host checks this first.
    uint64_t counter;  // Incremented about once a millisecond, so a changing value means it's running.
} Heartbeat;

static volatile Heartbeat* g_heartbeat = (volatile Heartbeat*)X280_STANDALONE_HEARTBEAT_ADDR;


const uint64_t g_ticks_per_ms = X280_MTIME_HZ / 1000;
const uint64_t g_ticks_per_2s = g_ticks_per_ms * 2000;
uint64_t g_next_heartbeat = 1867;

const char* TT_FIRMWARE_VERSION_STR = "tt_firmware_version";
const char* FIRMWARE_VERSION_STR = "0.0.1-alpha";
const char* TT_FIRMWARE_AUTHOR_STR = "tt_firmware_athor";
const char* FIRMWARE_AUTHOR_STR = "yzabuairTT";
const char* TT_FIRMWARE_COPYRIGHT_STR = "tt_firmware_copyright";
const char* FIRMWARE_COPYRIGHT_STR = "(c) 2026 TensTorrent";
const uint64_t TABLE_VERSION = 0x20261026;  // October 26, 2026

const char* TT_ARC_VERSION_STR = "tt_arc_version";
const char* TT_ARC_LENGTH_STR = "tt_arc_length";
const char* TT_ARC_TAG_NAME_STR = "tt_arc_tag_name";
const char* TT_ARC_UNKNOWN_TAG_STR = "TT_ARC_UNKNOWN";
const char* TT_ARC_INVALID_VALUE_STR = "TT_ARC_INVALID_VALUE";


void heart_beat() {
    if (x280_mtime() >= g_next_heartbeat) {
        g_next_heartbeat += g_ticks_per_ms;
        g_heartbeat->counter = g_heartbeat->counter + 1;
        if (g_heartbeat->counter >= UINT32_MAX-1) {
            g_heartbeat->counter = 1;
        }
    }
}


bool read_arc() {
    bool result = true;

    x280_noc_coord_t coords = {.x = 8, .y = 0};
    volatile uint32_t* arc_table_ptr = NULL;
    noc_window_status_t win_status = noc_open_window_2m(
        16, /* ARC can overwrite 0-15. */
        coords,
        0x80030434, /* SCRATCH_RAM_13 on ARC. */
        NOC_ORDER_DEFAULT,
        (volatile void**)&arc_table_ptr);
    if (win_status != NOC_WINDOW_OPEN) {
        result = false;
        goto cleanup;
    }

    // Just keep looping untl we see the value goto non-zero.
    // TODO: We should give up at some point?
    while (*arc_table_ptr == 0) {
        heart_beat();
    }

    // We now have the address of the table, map that in.
    volatile uint32_t* arc_table = NULL;
    win_status = noc_open_window_2m(
        16, /* ARC can overwrite 0-15. */
        coords,
        *arc_table_ptr, /* value in SCRATCH_RAM_13 on ARC. */
        NOC_ORDER_DEFAULT,
        (volatile void**)&arc_table);
    if (win_status != NOC_WINDOW_OPEN) {
        result = false;
        goto cleanup;
    }

    g_arc_info.version = arc_table[0];
    g_arc_info.length = arc_table[1] > 256 ? 256 : arc_table[1];
    copy_u32_memory(g_arc_info.data, arc_table + 2, g_arc_info.length);

cleanup:
    noc_clear_tlb_2m(16);

    return result;
}

uint32_t get_arc_value(uint16_t value_index) {
    uint32_t value = 0;

    x280_noc_coord_t coords = {.x = 8, .y = 0};
    volatile uint32_t* arc_value_table_ptr = NULL;
    noc_window_status_t win_status = noc_open_window_2m(
        16, /* ARC can overwrite 0-15. */
        coords,
        0x80030430, /* SCRATCH_RAM_12 on ARC. */
        NOC_ORDER_DEFAULT,
        (volatile void**)&arc_value_table_ptr);
    if (win_status != NOC_WINDOW_OPEN) {
        goto cleanup;
    }

    // We now have the address of the value table, map that in.
    volatile uint32_t* arc_value_table = NULL;
    win_status = noc_open_window_2m(
        16, /* ARC can overwrite 0-15. */
        coords,
        *arc_value_table_ptr, /* value in SCRATCH_RAM_12 on ARC. */
        NOC_ORDER_DEFAULT,
        (volatile void**)&arc_value_table);
    if (win_status != NOC_WINDOW_OPEN) {
        goto cleanup;
    }

    value = arc_value_table[value_index];

cleanup:
    noc_clear_tlb_2m(16);

    return value;
}

size_t init_arc_data_in_table(size_t start_index) {
    size_t curr_idx = start_index;
    size_t str_idx = start_index;
    size_t num_idx = 0;
    for (int i = 0; i < g_arc_info.length && curr_idx < 128; ++i) {
        uint32_t curr_word = g_arc_info.data[i];
        uint16_t tag_index = curr_word & 0xFFFF;
        uint16_t value_offset = curr_word >> 16;

        if (tag_index >= NUMBER_OF_TAGS) {
            copy_memory(g_table->labels[curr_idx].metric_name, TT_ARC_TAG_NAME_STR, str_len(TT_ARC_TAG_NAME_STR));
            copy_memory(
                g_table->str_values[str_idx].str_value, TT_ARC_UNKNOWN_TAG_STR, str_len(TT_ARC_UNKNOWN_TAG_STR));
            g_table->labels[curr_idx].metric_type = 1;
            g_table->labels[curr_idx].value_index = str_idx;
            g_table->labels[curr_idx].arc_index = 999;
            ++curr_idx;
            ++str_idx;
        } else {
            copy_memory(
                g_table->labels[curr_idx].metric_name, g_arc_tag_names[tag_index], str_len(g_arc_tag_names[tag_index]));
            g_table->labels[curr_idx].arc_index = value_offset;
            if (value_offset >= g_arc_info.length) {
                copy_memory(
                    g_table->str_values[str_idx].str_value,
                    TT_ARC_INVALID_VALUE_STR,
                    str_len(TT_ARC_INVALID_VALUE_STR));
                g_table->labels[curr_idx].metric_type = 1;
                g_table->labels[curr_idx].value_index = str_idx;
                g_table->labels[curr_idx].arc_index = 999;
                ++str_idx;
            } else {
                g_table->num_values[num_idx].num_value = get_arc_value(value_offset);
                g_table->labels[curr_idx].metric_type = 0;
                g_table->labels[curr_idx].value_index = num_idx;
                ++num_idx;
            }
            ++curr_idx;
        }
    }

    return curr_idx;
}


bool init_table() {
    clear_memory(g_table, sizeof(MetricsTelemetryTable));

    copy_memory(g_table->labels[0].metric_name, TT_FIRMWARE_VERSION_STR, str_len(TT_FIRMWARE_VERSION_STR));
    copy_memory(g_table->str_values[0].str_value, FIRMWARE_VERSION_STR, str_len(FIRMWARE_VERSION_STR));
    g_table->labels[0].metric_type = 1;
    g_table->labels[0].value_index = 0;
    g_table->labels[0].arc_index = 999;

    copy_memory(g_table->labels[1].metric_name, TT_FIRMWARE_AUTHOR_STR, str_len(TT_FIRMWARE_AUTHOR_STR));
    copy_memory(g_table->str_values[1].str_value, FIRMWARE_AUTHOR_STR, str_len(FIRMWARE_AUTHOR_STR));
    g_table->labels[1].metric_type = 1;
    g_table->labels[1].value_index = 1;
    g_table->labels[1].arc_index = 999;

    copy_memory(g_table->labels[2].metric_name, TT_FIRMWARE_COPYRIGHT_STR, str_len(TT_FIRMWARE_COPYRIGHT_STR));
    copy_memory(g_table->str_values[2].str_value, FIRMWARE_COPYRIGHT_STR, str_len(FIRMWARE_COPYRIGHT_STR));
    g_table->labels[2].metric_type = 1;
    g_table->labels[2].value_index = 2;
    g_table->labels[2].arc_index = 999;

    size_t total_metrics = init_arc_data_in_table(3);

    g_table->num_metrics = total_metrics;

    x280_fence_iorw();
    g_table->table_version = TABLE_VERSION;

    return true;
}


void update_arc_data_in_table() {
     for(int i = 0; i < g_table->num_metrics; ++i) {
        if (g_table->labels[i].arc_index == 999) {
            continue;
        } 

        g_table->num_values[g_table->labels[i].value_index].num_value = get_arc_value(g_table->labels[i].arc_index);
     }
}

int main(uint64_t hartid) {
    if (hartid == 0) {
        //volatile Heartbeat* heartbeat = (volatile Heartbeat*)X280_STANDALONE_HEARTBEAT_ADDR;

        g_heartbeat->counter = 0;
        x280_fence_ow();
        g_heartbeat->magic = X280_STANDALONE_HEARTBEAT_MAGIC;  // Last, so the host never sees the magic with a stale counter.
        g_next_heartbeat = x280_mtime() + g_ticks_per_ms;

        auto result = read_arc();
        if (!result) {
            // TODO: How do I indicate failure.
        }

        result = init_table();
        if (!result) {
            // TODO: How do I indicate failure.
        }


        uint64_t next_work = x280_mtime() + g_ticks_per_2s;
        for (;;) {
            heart_beat();

            if (x280_mtime() >= next_work) {
                next_work += g_ticks_per_2s;
                result = read_arc();
                if (result) {
                    update_arc_data_in_table();
                }       
            }

            x280_pause();
        }
    }

    for (;;) {
        x280_pause();
    }
}
