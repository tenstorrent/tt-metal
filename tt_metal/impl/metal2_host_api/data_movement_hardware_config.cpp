// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::experimental {

// Build the conventional Gen1 placement for a READER/WRITER kernel, mirroring the legacy
// Reader/WriterDataMovementConfig convention (see kernel_types.cpp):
//   READER -> NCRISC (RISCV_1) on NOC_0;  WRITER -> BRISC (RISCV_0) on NOC_1
// NOC mode is always DM_DEDICATED_NOC; DM_DYNAMIC_NOC is a power-user knob reached only by
// constructing a DataMovementGen1Config directly.
DataMovementGen1Config create_from_role(DataMovementRoleHint role) {
    switch (role) {
        case DataMovementRoleHint::READER:
            return DataMovementGen1Config{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_0,
                .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
        case DataMovementRoleHint::WRITER:
            return DataMovementGen1Config{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_1,
                .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC};
    }
    TT_THROW("Unhandled DataMovementRoleHint in create_from_role");
}

}  // namespace tt::tt_metal::experimental
