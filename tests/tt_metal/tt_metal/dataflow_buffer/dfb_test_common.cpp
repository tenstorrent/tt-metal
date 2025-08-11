// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dfb_test_common.hpp"

namespace tt::tt_metal {

thread_local uint8_t current_thread_id = 0;
thread_local bool is_overlay_thread = true;

std::ostream& get_thread_output_stream() {
    if (!thread_output_file) {
        std::stringstream filename;
        filename << "thread_" << (uint32_t)current_thread_id << "_output" << (is_overlay_thread ? "_overlay" : "_compute") << ".log";
        thread_output_file = std::make_unique<std::ofstream>(filename.str());
    }
    return *thread_output_file;
}

}  // namespace tt::tt_metal