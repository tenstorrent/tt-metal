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
#include "noc.h"

#include "common/telemetry_command.h"
#include "common/telemetry_table.h"

typedef struct {
    uint32_t do_not_put_into_bss_pal;
    uint32_t version;
    uint32_t length;
    uint32_t data[256];
} ArcInfo;

static ArcInfo g_arc_info = {.do_not_put_into_bss_pal = 1867, .version = 0, .length = 0};

enum TelemetryTag {
    BOARD_ID_HIGH = 1,
    BOARD_ID_LOW = 2,
    ASIC_ID = 3,
    HARVESTING_STATE = 4,
    UPDATE_TELEM_SPEED = 5,
    VCORE = 6,
    TDP = 7,
    TDC = 8,
    VDD_LIMITS = 9,
    THM_LIMIT_SHUTDOWN = 10,
    ASIC_TEMPERATURE = 11,
    VREG_TEMPERATURE = 12,
    BOARD_TEMPERATURE = 13,
    AICLK = 14,
    AXICLK = 15,
    ARCCLK = 16,
    L2CPUCLK0 = 17,
    L2CPUCLK1 = 18,
    L2CPUCLK2 = 19,
    L2CPUCLK3 = 20,
    ETH_LIVE_STATUS = 21,
    GDDR_STATUS = 22,
    GDDR_SPEED = 23,
    ETH_FW_VERSION = 24,
    GDDR_FW_VERSION = 25,
    DM_APP_FW_VERSION = 26,
    DM_BL_FW_VERSION = 27,
    FLASH_BUNDLE_VERSION = 28,
    CM_FW_VERSION = 29,
    L2CPU_FW_VERSION = 30,
    FAN_SPEED = 31,
    TIMER_HEARTBEAT = 32,
    TELEMETRY_ENUM_COUNT = 33,
    ENABLED_TENSIX_COL = 34,
    ENABLED_ETH = 35,
    ENABLED_GDDR = 36,
    ENABLED_L2CPU = 37,
    PCIE_USAGE = 38,
    NOC_TRANSLATION = 40,
    FAN_RPM = 41,
    GDDR_0_1_TEMP = 42,
    GDDR_2_3_TEMP = 43,
    GDDR_4_5_TEMP = 44,
    GDDR_6_7_TEMP = 45,
    GDDR_0_1_CORR_ERRS = 46,
    GDDR_2_3_CORR_ERRS = 47,
    GDDR_4_5_CORR_ERRS = 48,
    GDDR_6_7_CORR_ERRS = 49,
    GDDR_UNCORR_ERRS = 50,
    MAX_GDDR_TEMP = 51,
    ASIC_LOCATION = 52,
    BOARD_POWER_LIMIT = 53,
    INPUT_POWER = 54,
    TDC_LIMIT_MAX = 55,
    THM_LIMIT_THROTTLE = 56,
    TT_FLASH_VERSION = 58,
    THERM_TRIP_COUNT = 60,
    ASIC_ID_HIGH = 61,
    ASIC_ID_LOW = 62,
    AICLK_LIMIT_MAX = 63,
    TDP_LIMIT_MAX = 64,
    AICLK_ARB_MIN = 65,
    AICLK_ARB_MAX = 66,
    ENABLED_MIN_ARB = 67,
    ENABLED_MAX_ARB = 68,
    NUMBER_OF_TAGS = 69,
};

const char* g_arc_tag_names[] = {
    "TT_ARC_UNKNOWN",
    "TT_ARC_BOARD_ID_HIGH",
    "TT_ARC_BOARD_ID_LOW",
    "TT_ARC_ASIC_ID",
    "TT_ARC_HARVESTING_STATE",
    "TT_ARC_UPDATE_TELEM_SPEED",
    "TT_ARC_VCORE",
    "TT_ARC_TDP",
    "TT_ARC_TDC",
    "TT_ARC_VDD_LIMITS",
    "TT_ARC_THM_LIMIT_SHUTDOWN",
    "TT_ARC_ASIC_TEMPERATURE",
    "TT_ARC_VREG_TEMPERATURE",
    "TT_ARC_BOARD_TEMPERATURE",
    "TT_ARC_AICLK",
    "TT_ARC_AXICLK",
    "TT_ARC_ARCCLK",
    "TT_ARC_L2CPUCLK0",
    "TT_ARC_L2CPUCLK1",
    "TT_ARC_L2CPUCLK2",
    "TT_ARC_L2CPUCLK3",
    "TT_ARC_ETH_LIVE_STATUS",
    "TT_ARC_GDDR_STATUS",
    "TT_ARC_GDDR_SPEED",
    "TT_ARC_ETH_FW_VERSION",
    "TT_ARC_GDDR_FW_VERSION",
    "TT_ARC_DM_APP_FW_VERSION",
    "TT_ARC_DM_BL_FW_VERSION",
    "TT_ARC_FLASH_BUNDLE_VERSION",
    "TT_ARC_CM_FW_VERSION",
    "TT_ARC_L2CPU_FW_VERSION",
    "TT_ARC_FAN_SPEED",
    "TT_ARC_TIMER_HEARTBEAT",
    "TT_ARC_TELEMETRY_ENUM_COUNT",
    "TT_ARC_ENABLED_TENSIX_COL",
    "TT_ARC_ENABLED_ETH",
    "TT_ARC_ENABLED_GDDR",
    "TT_ARC_ENABLED_L2CPU",
    "TT_ARC_PCIE_USAGE",
    "TT_ARC_UNKNOWN",
    "TT_ARC_NOC_TRANSLATION",
    "TT_ARC_FAN_RPM",
    "TT_ARC_GDDR_0_1_TEMP",
    "TT_ARC_GDDR_2_3_TEMP",
    "TT_ARC_GDDR_4_5_TEMP",
    "TT_ARC_GDDR_6_7_TEMP",
    "TT_ARC_GDDR_0_1_CORR_ERRS",
    "TT_ARC_GDDR_2_3_CORR_ERRS",
    "TT_ARC_GDDR_4_5_CORR_ERRS",
    "TT_ARC_GDDR_6_7_CORR_ERRS",
    "TT_ARC_GDDR_UNCORR_ERRS",
    "TT_ARC_MAX_GDDR_TEMP",
    "TT_ARC_ASIC_LOCATION",
    "TT_ARC_BOARD_POWER_LIMIT",
    "TT_ARC_INPUT_POWER",
    "TT_ARC_TDC_LIMIT_MAX",
    "TT_ARC_THM_LIMIT_THROTTLE",
    "TT_ARC_UNKNOWN",
    "TT_ARC_TT_FLASH_VERSION",
    "TT_ARC_UNKNOWN",
    "TT_ARC_THERM_TRIP_COUNT",
    "TT_ARC_ASIC_ID_HIGH",
    "TT_ARC_ASIC_ID_LOW",
    "TT_ARC_AICLK_LIMIT_MAX",
    "TT_ARC_TDP_LIMIT_MAX",
    "TT_ARC_AICLK_ARB_MIN",
    "TT_ARC_AICLK_ARB_MAX",
    "TT_ARC_ENABLED_MIN_ARB",
    "TT_ARC_ENABLED_MAX_ARB",
};

static volatile TelemetryTable* const g_table = (volatile TelemetryTable*)TABLE_ADDR;
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

// TODO: Move these functions to a more appropriate location.
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

// If we're going over NOC, copy word at a time.
void copy_u32_memory(volatile uint32_t* dest_ptr, volatile const uint32_t* src_ptr, size_t num_words_to_copy) {
    for (size_t i = 0; i < num_words_to_copy; ++i) {
        *dest_ptr = *src_ptr;
        ++dest_ptr;
        ++src_ptr;
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
const char* TT_FIRMWARE_HART_ID_STR = "tt_firmware_hartid";
const char* TT_FIRMWARE_AUTHOR_STR = "tt_firmware_author";
const char* FIRMWARE_AUTHOR_STR = "yzabuairTT";
const uint64_t TABLE_VERSION = 0x20261026;  // October 26, 2026

const char* TT_ARC_VERSION_STR = "tt_arc_version";
const char* TT_ARC_LENGTH_STR = "tt_arc_length";
const char* TT_ARC_TAG_NAME_STR = "tt_arc_tag_name";
const char* TT_ARC_UNKOWN_TAG_STR = "TT_ARC_UNKNOWN";
const char* TT_ARC_INVALID_VALUE_STR = "TT_ARC_INVALID_VALUE";

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
    for (int i = 0; i < g_arc_info.length; ++i) {
        uint32_t curr_word = g_arc_info.data[i];
        uint16_t tag_index = curr_word & 0xFFFF;
        uint16_t value_offset = curr_word >> 16;

        TableRow row;
        clear_memory(&row, sizeof(TableRow));

        if (tag_index >= NUMBER_OF_TAGS) {
            copy_memory(row.metric_name, TT_ARC_TAG_NAME_STR, str_len(TT_ARC_TAG_NAME_STR));
            row.metric_type = metric_type_string;
            copy_memory(row.str_data, TT_ARC_UNKOWN_TAG_STR, str_len(TT_ARC_UNKOWN_TAG_STR));
        } else {
            copy_memory(row.metric_name, g_arc_tag_names[tag_index], str_len(g_arc_tag_names[tag_index]));
            if (value_offset >= g_arc_info.length) {
                row.metric_type = metric_type_string;
                copy_memory(row.str_data, TT_ARC_INVALID_VALUE_STR, str_len(TT_ARC_INVALID_VALUE_STR));
            } else {
                row.metric_type = metric_type_number;
                uint32_t arc_value = get_arc_value(value_offset);
                copy_memory(&(row.num_data), &arc_value, sizeof(arc_value));
            }
        }

        copy_memory(g_table->rows + start_index++, &row, sizeof(row));
    }

    return start_index;
}

bool init_table() {
    clear_memory(&(g_table->header), sizeof(TableHeader));

    TableRow row;

    clear_memory(&row, sizeof(TableRow));
    copy_memory(row.metric_name, TT_FIRMWARE_VERSION_STR, str_len(TT_FIRMWARE_VERSION_STR));
    row.metric_type = metric_type_string;
    copy_memory(row.str_data, FIRMWARE_VERSION_STR, str_len(FIRMWARE_VERSION_STR));

    copy_memory(g_table->rows, &row, sizeof(TableRow));

    clear_memory(&row, sizeof(TableRow));
    copy_memory(row.metric_name, TT_FIRMWARE_HART_ID_STR, str_len(TT_FIRMWARE_HART_ID_STR));
    row.metric_type = metric_type_number;
    uint64_t hartid = 0;
    copy_memory(&(row.num_data), &hartid, sizeof(uint64_t));

    copy_memory(g_table->rows + 1, &row, sizeof(TableRow));

    clear_memory(&row, sizeof(TableRow));
    copy_memory(row.metric_name, TT_FIRMWARE_AUTHOR_STR, str_len(TT_FIRMWARE_AUTHOR_STR));
    row.metric_type = metric_type_string;
    copy_memory(row.str_data, FIRMWARE_AUTHOR_STR, str_len(FIRMWARE_AUTHOR_STR));

    copy_memory(g_table->rows + 2, &row, sizeof(TableRow));

    // Fill in arc table data.
    clear_memory(&row, sizeof(TableRow));
    copy_memory(row.metric_name, TT_ARC_VERSION_STR, str_len(TT_ARC_VERSION_STR));
    row.metric_type = metric_type_number;
    copy_memory(&(row.num_data), &g_arc_info.version, sizeof(uint32_t));

    copy_memory(g_table->rows + 3, &row, sizeof(TableRow));

    clear_memory(&row, sizeof(TableRow));
    copy_memory(row.metric_name, TT_ARC_LENGTH_STR, str_len(TT_ARC_LENGTH_STR));
    row.metric_type = metric_type_number;
    copy_memory(&(row.num_data), &g_arc_info.length, sizeof(uint32_t));

    copy_memory(g_table->rows + 4, &row, sizeof(TableRow));

    size_t new_count = init_arc_data_in_table(5);

    g_table->header.table_version = TABLE_VERSION;
    g_table->header.sequence_number = 0;
    g_table->header.num_rows = new_count;

    return true;
}

bool read_arc(uint64_t run_value) {
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
    while (*arc_table_ptr == 0) {
        if (x280_hart_stop_requested(0, run_value)) {
            result = false;
            goto cleanup;
        }
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

void run(uint64_t run_value) {
    for (;;) {
        if (x280_hart_stop_requested(0, run_value)) {
            break;
        }
    }
}

int main(uint64_t hartid) {
    uint64_t run_value = x280_hart_control(hartid)->run_value;
    if (hartid == 0) {
        bool result = read_arc(run_value);
        if (result) {
            result = init_table();
        }

        if (result) {
            run(run_value);
        }

        x280_hart_set_result(hartid, hartid);
        return 0;
    } else {
        x280_hart_set_result(hartid, hartid);
        return 0;
    }
}
