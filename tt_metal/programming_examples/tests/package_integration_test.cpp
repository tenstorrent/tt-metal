// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#include <google/protobuf/empty.pb.h>

#include <iostream>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // This example exists to demonstrate how to integrate other third party libraries into a Metalium program.
    // It exists also to test dependency conflicts between some libraries that might interfere with the build of
    // tt-metal. https://github.com/tenstorrent/tt-metal/issues/27465
    // https://github.com/tenstorrent/tt-metal/issues/29442

    // Test Protobuf
    google::protobuf::Empty empty_msg;
    std::string serialized;
    empty_msg.SerializeToString(&serialized);
    std::cout << "Protobuf: Serialized an empty message of size " << serialized.size() << std::endl;

    // Test TT-Metal Mesh Device
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    std::cout << "TT-Metal: Mesh device created" << std::endl;

    // Close the device
    mesh_device->close();
    std::cout << "TT-Metal: Mesh device closed" << std::endl;

    return 0;
}
