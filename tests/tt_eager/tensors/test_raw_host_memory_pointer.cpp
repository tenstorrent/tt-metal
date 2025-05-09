// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

/*

01  import torch
02: import numpy as np
03:
04: device = torch.device("cuda:0")
05:
06: // define tensors on the CPU
07: a_cpu = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.bfloat16)
08:
09: // define tensors on the device with CPU tensors
10: a_dev = torch.from_numpy(a_cpu).to_device(device)
11:
12: c_dev = torch.sqrt(a_dev)
13:
14: print(c_dev[1][0])
15:
16: d_cpu = np.array([[11,12,13,14],[15,16,17,18]])
17: d_dev = d_cpu.to_device(device)
18:
19: e_dev = c_dev + d_dev
20: print(e_dev)

*/

template <typename DataType>
struct NDArray {
    ttnn::Shape shape;
    void* data;

    NDArray(const ttnn::Shape& shape) : shape(shape), data(malloc(shape.volume() * sizeof(DataType))) {}
    ~NDArray() { free(data); }

    std::size_t size() const { return shape.volume(); }
};

void test_raw_host_memory_pointer() {
    using tt::tt_metal::DataType;
    using tt::tt_metal::HostBuffer;
    using tt::tt_metal::HostStorage;
    using tt::tt_metal::Layout;
    using tt::tt_metal::Tensor;

    int device_id = 0;
    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

    ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});

    // Host tensor to print the output
    Tensor tensor_for_printing = Tensor(
        HostStorage{tt::tt_metal::host_buffer::create<bfloat16>(std::vector<bfloat16>(shape.volume()))},
        shape,
        DataType::BFLOAT16,
        Layout::TILE);

    /* Borrow Data from Numpy Start */
    // Create some
    auto a_np_array = NDArray<bfloat16>(shape);
    void* a_np_array_data = a_np_array.data;
    HostBuffer a_cpu_buffer(
        tt::stl::Span<bfloat16>(static_cast<bfloat16*>(a_np_array_data), a_np_array.size()),
        tt::tt_metal::MemoryPin([]() {}, []() {}));
    Tensor a_cpu = Tensor(HostStorage{std::move(a_cpu_buffer)}, shape, DataType::BFLOAT16, Layout::TILE);
    /* Borrow Data from Numpy End */

    /* Sanity Check Start */

    // Test that borrowing numpy array's data works correctly

    // Set every value of tt Tensor to the same non-zero number
    bfloat16 a_value = 4.0f;

    for (auto& element : tt::tt_metal::host_buffer::get_as<bfloat16>(a_cpu)) {
        element = a_value;
    }

    // Check that numpy array's data is now set to that value
    for (auto index = 0; index < a_np_array.size(); index++) {
        auto a_np_array_element = static_cast<bfloat16*>(a_np_array_data)[index];
        TT_ASSERT(a_np_array_element == a_value);
    }
    /* Sanity Check End */

    /*  Run and Print Start   */
    Tensor a_dev = a_cpu.to_device(device.get());

    Tensor c_dev = ttnn::sqrt(a_dev);

    tt::tt_metal::memcpy(tensor_for_printing, c_dev);

    // Check that cpu tensor has correct data
    bfloat16 output_value = 1.99219f;  // Not exactly 2.0f because of rounding errors
    for (auto& element : tt::tt_metal::host_buffer::get_as<bfloat16>(tensor_for_printing)) {
        TT_ASSERT(element == output_value);
    }

    tensor_for_printing.print();
    /*  Run and Print End   */

    /* Alternative Way to Print Start */
    // Alternatively, we could allocate memory manually and create Tensors with borrowed storage on the fly to print the
    // data
    void* storage_of_alternative_tensor_for_printing = malloc(shape.volume() * sizeof(bfloat16));
    tt::tt_metal::memcpy(storage_of_alternative_tensor_for_printing, c_dev);

    HostBuffer alternative_tensor_for_printing_buffer(
        tt::stl::Span<bfloat16>(static_cast<bfloat16*>(storage_of_alternative_tensor_for_printing), shape.volume()),
        tt::tt_metal::MemoryPin([]() {}, []() {}));

    Tensor alternative_tensor_for_printing =
        Tensor(HostStorage{std::move(alternative_tensor_for_printing_buffer)}, shape, DataType::BFLOAT16, Layout::TILE);
    alternative_tensor_for_printing.print();

    for (auto& element : tt::tt_metal::host_buffer::get_as<bfloat16>(alternative_tensor_for_printing)) {
        TT_ASSERT(element == output_value);
    }

    free(storage_of_alternative_tensor_for_printing);
    /* Alternative Way to Print End */

    auto d_np_array = NDArray<bfloat16>(shape);
    void* d_np_array_data = d_np_array.data;
    HostBuffer d_data_buffer(
        tt::stl::Span<bfloat16>(static_cast<bfloat16*>(d_np_array_data), d_np_array.size()),
        tt::tt_metal::MemoryPin([]() {}, []() {}));
    Tensor d_cpu = Tensor(HostStorage{std::move(d_data_buffer)}, shape, DataType::BFLOAT16, Layout::TILE);

    bfloat16 d_value = 8.0f;
    for (auto& element : tt::tt_metal::host_buffer::get_as<bfloat16>(d_cpu)) {
        element = d_value;
    }

    Tensor d_dev = a_dev;
    memcpy(d_dev, d_cpu);

    Tensor e_dev = ttnn::add(c_dev, d_dev);

    tt::tt_metal::memcpy(tensor_for_printing, e_dev);

    for (auto& element : tt::tt_metal::host_buffer::get_as<bfloat16>(tensor_for_printing)) {
        TT_ASSERT(element == bfloat16(10.0f));
    }
}

int main() {
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        test_raw_host_memory_pointer();
    }
    return 0;
}
