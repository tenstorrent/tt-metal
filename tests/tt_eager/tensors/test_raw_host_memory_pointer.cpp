// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/numpy/functions.hpp"

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
10: a_dev = torch.from_numpy(a_cpu).to(device)
11:
12: c_dev = torch.sqrt(a_dev)
13:
14: print(c_dev[1][0])
15:
16: d_cpu = np.array([[11,12,13,14],[15,16,17,18]])
17: d_dev = d_cpu.to(device)
18:
19: e_dev = c_dev + d_dev
20: print(e_dev)

*/

namespace numpy {

template <typename DataType>
struct ndarray {
    ttnn::Shape shape;
    void* data;

    ndarray(ttnn::Shape shape) : shape(shape), data(malloc(tt::tt_metal::compute_volume(shape) * sizeof(DataType))) {}
    ~ndarray() { free(data); }

    std::size_t size() const { return tt::tt_metal::compute_volume(shape); }
};

}  // namespace numpy

void test_raw_host_memory_pointer() {
    using tt::tt_metal::BorrowedStorage;
    using tt::tt_metal::DataType;
    using tt::tt_metal::OwnedStorage;
    using tt::tt_metal::Tensor;
    using namespace tt::tt_metal::borrowed_buffer;
    using namespace tt::tt_metal::owned_buffer;

    int device_id = 0;
    tt::tt_metal::Device* device = tt::tt_metal::CreateDevice(device_id);

    ttnn::Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};

    // Host tensor to print the output
    Tensor tensor_for_printing = Tensor(
        OwnedStorage{owned_buffer::create<bfloat16>(tt::tt_metal::compute_volume(shape))},
        shape,
        DataType::BFLOAT16,
        Layout::TILE);

    /* Borrow Data from Numpy Start */
    // Create some
    auto a_np_array = numpy::ndarray<bfloat16>(shape);
    void* a_np_array_data = a_np_array.data;
    auto on_creation_callback = [] {};
    auto on_destruction_callback = [] {};
    Tensor a_cpu = Tensor(
        BorrowedStorage{
            borrowed_buffer::Buffer(static_cast<bfloat16*>(a_np_array_data), a_np_array.size()),
            on_creation_callback,
            on_destruction_callback},
        shape,
        DataType::BFLOAT16,
        Layout::TILE);
    /* Borrow Data from Numpy End */

    /* Sanity Check Start */

    // Test that borrowing numpy array's data works correctly

    // Set every value of tt Tensor to the same non-zero number
    bfloat16 a_value = 4.0f;

    for (auto& element : borrowed_buffer::get_as<bfloat16>(a_cpu)) {
        element = a_value;
    }

    // Check that numpy array's data is now set to that value
    for (auto index = 0; index < a_np_array.size(); index++) {
        auto a_np_array_element = static_cast<bfloat16*>(a_np_array_data)[index];
        TT_ASSERT(a_np_array_element == a_value);
    }
    /* Sanity Check End */

    /*  Run and Print Start   */
    Tensor a_dev = a_cpu.to(device);

    Tensor c_dev = ttnn::sqrt(a_dev);

    tt::tt_metal::memcpy(tensor_for_printing, c_dev);

    // Check that cpu tensor has correct data
    bfloat16 output_value = 1.99219f;  // Not exactly 2.0f because of rounding errors
    for (auto& element : owned_buffer::get_as<bfloat16>(tensor_for_printing)) {
        TT_ASSERT(element == output_value);
    }

    tensor_for_printing.print();
    /*  Run and Print End   */

    /* Alternative Way to Print Start */
    // Alternatively, we could allocate memory manually and create Tensors with BorrowedStorage on the fly to print the
    // data
    void* storage_of_alternative_tensor_for_printing = malloc(tt::tt_metal::compute_volume(shape) * sizeof(bfloat16));
    tt::tt_metal::memcpy(storage_of_alternative_tensor_for_printing, c_dev);

    Tensor alternative_tensor_for_printing = Tensor(
        BorrowedStorage{
            borrowed_buffer::Buffer(
                static_cast<bfloat16*>(storage_of_alternative_tensor_for_printing),
                tt::tt_metal::compute_volume(shape)),
            on_creation_callback,
            on_destruction_callback},
        shape,
        DataType::BFLOAT16,
        Layout::TILE);
    alternative_tensor_for_printing.print();

    for (auto& element : borrowed_buffer::get_as<bfloat16>(alternative_tensor_for_printing)) {
        TT_ASSERT(element == output_value);
    }

    free(storage_of_alternative_tensor_for_printing);
    /* Alternative Way to Print End */

    auto d_np_array = numpy::ndarray<bfloat16>(shape);
    void* d_np_array_data = d_np_array.data;
    Tensor d_cpu = Tensor(
        BorrowedStorage{
            borrowed_buffer::Buffer(static_cast<bfloat16*>(d_np_array_data), d_np_array.size()),
            on_creation_callback,
            on_destruction_callback},
        shape,
        DataType::BFLOAT16,
        Layout::TILE);

    bfloat16 d_value = 8.0f;
    for (auto& element : borrowed_buffer::get_as<bfloat16>(d_cpu)) {
        element = d_value;
    }

    Tensor d_dev = a_dev;
    memcpy(d_dev, d_cpu);

    Tensor e_dev = ttnn::add(c_dev, d_dev);

    tt::tt_metal::memcpy(tensor_for_printing, e_dev);

    for (auto& element : owned_buffer::get_as<bfloat16>(tensor_for_printing)) {
        TT_ASSERT(element == bfloat16(10.0f));
    }

    TT_FATAL(tt::tt_metal::CloseDevice(device), "Error");
}

int main() {
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        test_raw_host_memory_pointer();
    }
    return 0;
}
