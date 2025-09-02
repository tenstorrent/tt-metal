// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Device setup
    IDevice* device = CreateDevice(0);

    // Device command queue and program setup
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    CloseDevice(device);
}
