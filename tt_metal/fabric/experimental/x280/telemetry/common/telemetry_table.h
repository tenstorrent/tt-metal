// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <x280_hart_protocol.h>

#include <stdint.h>

#define TABLE_ADDR 0x08132000

typedef struct __attribute__((aligned(X280_HART_RECORD_STRIDE))) {
    uint64_t table_version;
    uint64_t sequence_number;
    uint64_t num_rows;
} TableHeader;

enum MetricType { metric_type_number = 0x00, metric_type_string = 0x01 };

typedef struct __attribute__((aligned(X280_HART_RECORD_STRIDE))) {
    char metric_name[32];
    uint64_t metric_type;
    // TODO: Eventually move this into a union?  if it makes sense.
    uint64_t num_data;
    char str_data[64];
} TableRow;

typedef struct __attribute__((aligned(X280_HART_RECORD_STRIDE))) {
    TableHeader header;
    TableRow rows[128];
} TelemetryTable;
