// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>
#include <iostream>

int main() {
    const size_t tensor_width = 32;
    const size_t tensor_height = 32;

    std::srand(0);
    auto num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    std::cout << "num_devices:" << num_devices_ << std::endl;
    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    std::cout << "Device created" << std::endl;
    device.reset();
    return 0;
}
