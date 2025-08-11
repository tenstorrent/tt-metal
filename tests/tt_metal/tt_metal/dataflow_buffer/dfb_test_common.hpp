// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <sstream>

namespace tt::tt_metal {

extern thread_local uint8_t current_thread_id;
extern thread_local bool is_overlay_thread;

thread_local std::unique_ptr<std::ofstream> thread_output_file;

std::ostream& get_thread_output_stream();

// Convenient macros for easy logging
#define THREAD_LOG get_thread_output_stream()
#define THREAD_DEBUG(msg) THREAD_LOG << "[Thread " << (uint32_t)current_thread_id << "] " << msg << std::endl

}  // namespace tt::tt_metal