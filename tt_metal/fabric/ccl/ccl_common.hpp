// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>

namespace tt::tt_fabric {

tt::tt_metal::KernelHandle generate_edm_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::IDevice* device,
    const tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core,
    tt::tt_metal::DataMovementProcessor risc_id,
    tt::tt_metal::NOC noc_id);

}  // namespace tt::tt_fabric
