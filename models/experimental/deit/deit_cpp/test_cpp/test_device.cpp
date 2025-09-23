// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/types.hpp>
#include <ttnn/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

int main() {
    auto device = ttnn::MeshDevice::create_unit_mesh(0,
        /*l1_small_size=*/24576,
        /*trace_region_size=*/6434816,
        /*num_command_queues=*/2,
        /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    std::cout << "Device created" << std::endl;
    return 0;
}
