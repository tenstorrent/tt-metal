// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::tt_metal::inspector {

// Get Python callstack if available, returns empty string if not in Python context
std::string get_python_callstack(int max_frames = 20);

// Get C++ callstack using backtrace
std::string get_cpp_callstack(int max_frames = 20);

// Get callstack with Python preference, fallback to C++
std::string get_callstack();

}  // namespace tt::tt_metal::inspector
