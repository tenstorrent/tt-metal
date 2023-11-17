// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define NOC_OVERLAY_START_ADDR     0xFFB40000
#define NOC_STREAM_REG_SPACE_SIZE  0x1000

#define STREAM_REG_ADDR(stream_id, reg_id) ((NOC_OVERLAY_START_ADDR) + (((uint32_t)(stream_id))*(NOC_STREAM_REG_SPACE_SIZE)) + (((uint32_t)(reg_id)) << 2))

constexpr int NUM_STREAMS = 64;
constexpr int REGISTERS[5] = {0, 4, 8, 12, 24};
