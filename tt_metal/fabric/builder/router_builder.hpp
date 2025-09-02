// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric/fabric_tensix_builder.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"

namespace tt::tt_metal {
class IDevice;
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_fabric::builder {

// Wrapper struct for some variables needed to define how to do init handshakes
struct InitConfig {
    chan_id_t master_router_chan = 0;
    size_t num_local_fabric_routers = 0;
    uint32_t router_channels_mask = 0;
};

struct RouterBuilder {
    tt::tt_fabric::FabricEriscDatamoverBuilder erisc_kernel_builder;
    std::optional<tt::tt_fabric::FabricTensixDatamoverBuilder> tensix_kernel_builder;

    RouterBuilder(const tt::tt_fabric::FabricEriscDatamoverBuilder&);
    RouterBuilder(
        const tt::tt_fabric::FabricEriscDatamoverBuilder&, const tt::tt_fabric::FabricTensixDatamoverBuilder&);

    void create_kernels(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const InitConfig& init_config,
        chan_id_t eth_chan);

private:
    void create_erisc_kernel(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const InitConfig& init_config,
        chan_id_t eth_chan);
    void create_tensix_kernel(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program);
};

}  // namespace tt::tt_fabric::builder
