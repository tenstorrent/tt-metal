// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <optional>
#include <vector>

#include "context.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
class Program;
namespace tools {
namespace mem_bench {
struct Context;
}  // namespace mem_bench
}  // namespace tools
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::tools::mem_bench {

std::vector<uint32_t> read_cores(tt::tt_metal::IDevice* device, const CoreRange& cores, uint32_t addr);

std::optional<CoreRange> configure_kernels(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const Context& context,
    uint32_t start_y,
    uint32_t num_kernels,
    bool is_writer,
    uint32_t pcie_size,
    uint32_t pcie_offset = 0);

}  // namespace tt::tt_metal::tools::mem_bench
