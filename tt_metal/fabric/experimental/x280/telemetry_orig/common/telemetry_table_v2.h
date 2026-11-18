// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <x280_hart_protocol.h>

#include <stdint.h>

#define TABLE_ADDR 0x08132000

typedef struct {
    uint32_t control_block;
    char metric_name[32];
    uint32_t metric_type;  // 0 = number, 1 = string.
    uint32_t value_index;
    uint32_t arc_index;
} MetricLabel;

typedef struct {
    uint32_t control_block;
    uint32_t num_value;
} MetricNumValue;

typedef struct {
    uint32_t control_block;
    char str_value[64];
} MetricStrValue;

typedef struct {
    uint32_t table_version;
    uint32_t num_metrics;
    MetricLabel labels[128];
    MetricNumValue num_values[128];
    MetricStrValue str_values[128];
} MetricsTelemetryTable;
