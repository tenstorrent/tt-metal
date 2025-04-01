// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "core_coord.hpp"
#include "data_types.hpp"
#include "fd_kernel.hpp"
#include "system_memory_manager.hpp"

namespace tt::tt_metal {

struct fabric_router_depedent_config {};

class FabricRouterVC : public FDKernel {
public:
    FabricRouterVC(int node_id, chip_id_t device_id, chip_id_t servicing_device_id, uint8_t cq_id) :
        FDKernel(
            node_id,
            device_id,
            servicing_device_id,
            cq_id,
            {RISCV_0_default, RISCV_0_default, RISCV_0_default} /*Arbitrary*/) {}

    void CreateKernel() override;
    void GenerateStaticConfigs() override;
    void GenerateDependentConfigs() override;
    void ConfigureCore() override;
};

}  // namespace tt::tt_metal
