// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
namespace tt::tt_metal {

inline namespace v0 {
class Program;
}  // namespace v0

struct ProgramHandle {
    uint32_t key = 0;

    bool is_valid() const { return key != 0; }
};

}  // namespace tt::tt_metal
