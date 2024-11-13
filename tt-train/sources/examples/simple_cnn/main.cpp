// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>
#include <iostream>

int main() {
    const size_t tensor_width = 32;
    const size_t tensor_height = 32;

    std::srand(0);
    auto arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    std::cout << "Arch:" << tt::test_utils::get_env_arch_name() << std::endl;
    std::cout << "num_devices:" << num_devices_ << std::endl;
    auto device = tt::tt_metal::CreateDevice(0);
    std::cout << "Device created" << std::endl;
    tt::tt_metal::CloseDevice(device);
    return 0;
}
