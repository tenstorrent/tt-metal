// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core_coord.hpp"
#include "fd_kernel.hpp"

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

private:
    std::pair<tt_fabric::routing_plane_id_t, CoreCoord> get_closest_router(
        const IDevice* source, const IDevice* destination);
};

}  // namespace tt::tt_metal
