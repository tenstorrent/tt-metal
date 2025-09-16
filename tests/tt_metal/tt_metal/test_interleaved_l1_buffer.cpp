// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <errno.h>
#include <fmt/base.h>
#include <stdint.h>
#include <stdlib.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstring>
#include <exception>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

bool test_interleaved_l1_buffer(
    std::shared_ptr<distributed::MeshDevice> mesh_device, int num_pages_one, int num_pages_two, uint32_t page_size) {
    bool pass = true;
    auto& cq = mesh_device->mesh_command_queue();

    uint32_t buffer_size = num_pages_one * page_size;

    distributed::DeviceLocalBufferConfig local_config = {
        .page_size = page_size, .buffer_type = tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config = {.size = buffer_size};
    auto interleaved_buffer = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

    std::vector<uint32_t> host_buffer =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    distributed::WriteShard(cq, interleaved_buffer, host_buffer, distributed::MeshCoordinate(0, 0));

    std::vector<uint32_t> readback_buffer;
    distributed::ReadShard(cq, readback_buffer, interleaved_buffer, distributed::MeshCoordinate(0, 0));

    pass &= (host_buffer == readback_buffer);

    uint32_t second_buffer_size = num_pages_two * page_size;

    distributed::ReplicatedBufferConfig second_buffer_config = {.size = second_buffer_size};
    auto second_interleaved_buffer =
        distributed::MeshBuffer::create(second_buffer_config, local_config, mesh_device.get());

    std::vector<uint32_t> second_host_buffer = create_random_vector_of_bfloat16(
        second_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    distributed::WriteShard(cq, second_interleaved_buffer, second_host_buffer, distributed::MeshCoordinate(0, 0));

    std::vector<uint32_t> second_readback_buffer;
    distributed::ReadShard(cq, second_readback_buffer, second_interleaved_buffer, distributed::MeshCoordinate(0, 0));

    pass &= (second_host_buffer == second_readback_buffer);

    return pass;
}

int main(int argc, char** argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        uint32_t page_size = 2 * 1024;

        int num_bank_pages_one = 258;
        int num_bank_pages_two = 378;

        pass &= test_interleaved_l1_buffer(mesh_device, num_bank_pages_one, num_bank_pages_two, page_size);

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
