// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Debug test program to investigate mesh shape issue in all_gather
// This mimics the Python test in test_1x32_mesh_device.py

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/distributed/api.hpp"
#include <tt-logger/tt-logger.hpp>
#include <iostream>
#include <cassert>
#include <functional>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using namespace ttnn;

int main() {
    try {
        // Set fabric config (matching Python code)
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);

        // Open mesh device with shape (1, 32) - matching Python test
        MeshShape requested_mesh_shape(1, 32);
        std::cout << "Requested mesh shape: [" << requested_mesh_shape[0] << ", " << requested_mesh_shape[1] << "]"
                  << std::endl;

        auto mesh_device_config = MeshDeviceConfig(requested_mesh_shape);
        std::shared_ptr<MeshDevice> mesh_device = MeshDevice::create(mesh_device_config);

        // DEBUG POINT 1: Check mesh device shape right after creation
        const MeshShape& device_shape = mesh_device->shape();
        std::cout << "DEBUG POINT 1 - mesh_device->shape(): [" << device_shape[0] << ", " << device_shape[1] << "]"
                  << std::endl;

        const MeshDeviceView& device_view = mesh_device->get_view();
        const MeshShape& view_shape = device_view.shape();
        std::cout << "DEBUG POINT 1 - mesh_device->get_view().shape(): [" << view_shape[0] << ", " << view_shape[1]
                  << "]" << std::endl;

        assert(device_shape[0] == 1 && device_shape[1] == 32 && "Mesh device shape should be (1, 32)");
        assert(view_shape[0] == 1 && view_shape[1] == 32 && "Mesh device view shape should be (1, 32)");

        // Create a tensor of ones on device (matching Python: shape=[1, 1, 256, 256])
        ttnn::Shape tensor_shape({1, 1, 256, 256});
        MeshDevice& device_ref = *mesh_device;  // Get reference like the tests do
        auto ones_tensor = ttnn::ones(
            tensor_shape,
            ttnn::DataType::FLOAT32,
            ttnn::TILE_LAYOUT,
            std::optional<std::reference_wrapper<MeshDevice>>(std::ref(device_ref)));

        // DEBUG POINT 2: Check tensor device shape
        const MeshShape& tensor_device_shape = ones_tensor.device()->shape();
        std::cout << "DEBUG POINT 2 - tensor.device()->shape(): [" << tensor_device_shape[0] << ", "
                  << tensor_device_shape[1] << "]" << std::endl;

        const MeshShape& tensor_view_shape = ones_tensor.device()->get_view().shape();
        std::cout << "DEBUG POINT 2 - tensor.device()->get_view().shape(): [" << tensor_view_shape[0] << ", "
                  << tensor_view_shape[1] << "]" << std::endl;

        assert(tensor_device_shape[0] == 1 && tensor_device_shape[1] == 32 && "Tensor device shape should be (1, 32)");
        assert(tensor_view_shape[0] == 1 && tensor_view_shape[1] == 32 && "Tensor device view shape should be (1, 32)");

        // DEBUG POINT 3: Before calling all_gather, check what shape will be used
        std::cout << "DEBUG POINT 3 - About to call all_gather" << std::endl;
        std::cout << "  Input tensor device shape: [" << ones_tensor.device()->shape()[0] << ", "
                  << ones_tensor.device()->shape()[1] << "]" << std::endl;
        std::cout << "  Input tensor device view shape: [" << ones_tensor.device()->get_view().shape()[0] << ", "
                  << ones_tensor.device()->get_view().shape()[1] << "]" << std::endl;

        // Call all_gather (matching Python: dim=2, cluster_axis=1, num_links=1)
        // This is where the issue occurs - all_gather internally checks mesh_shape
        auto gathered = ttnn::all_gather(
            ones_tensor,
            /*dim=*/2,
            /*cluster_axis=*/1,
            /*subdevice_id=*/std::nullopt,
            /*memory_config=*/std::nullopt,
            /*optional_output_tensor=*/std::nullopt,
            /*num_links=*/1,
            /*topology=*/std::nullopt);

        std::cout << "DEBUG POINT 4 - After all_gather call" << std::endl;
        std::cout << "  Output tensor device shape: [" << gathered.device()->shape()[0] << ", "
                  << gathered.device()->shape()[1] << "]" << std::endl;

        // Close the device
        mesh_device->close();

        std::cout << "Test completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
