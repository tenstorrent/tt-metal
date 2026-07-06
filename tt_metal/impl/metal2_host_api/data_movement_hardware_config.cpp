// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

namespace tt::tt_metal::experimental {

// Build the conventional Gen1 placement for a reader / writer kernel, mirroring the legacy
// Reader/WriterDataMovementConfig convention (see kernel_types.cpp):
//   reader -> NCRISC (RISCV_1) on NOC_0;  writer -> BRISC (RISCV_0) on NOC_1
// NOC mode is always DM_DEDICATED_NOC; DM_DYNAMIC_NOC is a power-user knob reached only by
// constructing a DataMovementGen1Config directly.
DataMovementGen1Config create_reader_gen1_datamovement_config() {
    return DataMovementGen1Config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(tt::ARCH::WORMHOLE_B0),
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
}

DataMovementGen1Config create_writer_gen1_datamovement_config() {
    return DataMovementGen1Config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(tt::ARCH::WORMHOLE_B0),
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
}

}  // namespace tt::tt_metal::experimental
