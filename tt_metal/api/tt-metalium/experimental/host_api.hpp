// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tt-metalium/kernel_types.hpp>

namespace tt::tt_metal {
class Program;

namespace experimental {
KernelHandle CreateKernel(Program& program, const std::string& file_name, const DataMovementConfig& config);
}
}  // namespace tt::tt_metal
