// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

using namespace tt;

namespace tt::tt_metal {

// Log current kernel compile-time arguments
void log_kernel_defines_and_args (const std::string &out_dir, const std::string &full_kernel_name, const std::string &defines_and_args_str);
// Dump all kernel compile-time arguments to a file
void dump_kernel_defines_and_args(const std::string &out_kernel_root_path);

} // namespace tt::tt_metal
