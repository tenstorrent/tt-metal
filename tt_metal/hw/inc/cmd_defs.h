// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**

This is deprecated CMD code generation for old test_firmware and other older custom firmwares

*/
#pragma once

#define CMD_DECODE_NAME(a, b) a,
#define CMD_DECODE_SIZE(a, b) (b),

#define CMD_DECODE_DEFINE(a, b) CMD_DECODE_NAME(a, b)
enum cmd_code : std::uint32_t {
    CMD_UNUSED_FOR_ENUM_START = 4,
#include "cmds.def"
    CMD_UNUSED_FOR_ENUM_END,
};
#undef CMD_DECODE_DEFINE

#define CMD_DECODE_DEFINE(a, b) CMD_DECODE_SIZE(a, b)
static uint cmd_sizes[] = {
    0,  // CMD_UNUSED_FOR_ENUM_START
#include "cmds.def"
    0,  // CMD_UNUSED_FOR_ENUM_END
};
#undef CMD_DECODE_DEFINE

static constexpr std::uint32_t NUM_CMDS = CMD_UNUSED_FOR_ENUM_END - CMD_UNUSED_FOR_ENUM_START + 1;

static inline std::uint32_t getCmdSize(cmd_code cmd) { return cmd_sizes[cmd - CMD_UNUSED_FOR_ENUM_START]; }
static inline std::uint32_t getCmdSize(std::uint32_t first_cmd_word) {
    return getCmdSize((cmd_code)((first_cmd_word >> 24) & 0xFF));
}
