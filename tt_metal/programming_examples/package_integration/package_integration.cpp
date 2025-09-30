// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#include <opencv2/opencv.hpp>

#include <google/protobuf/empty.pb.h>

#include <iostream>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Test OpenCV
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
    std::cout << "OpenCV: Created a blue image of size 100x100" << std::endl;

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
