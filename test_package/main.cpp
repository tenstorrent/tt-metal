// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/small_vector.hpp>

void test_ttnn_add() {
    using namespace tt::tt_metal;
    if (tt::tt_metal::GetNumAvailableDevices() == 0) {
        fmt::print("No devices found\n");
        return;
    }
    fmt::print("Devices found: {}\n", tt::tt_metal::GetNumAvailableDevices());
    auto device_id = 0;
    auto device = ttnn::device::open_mesh_device(device_id);

    auto a = ttnn::arange(32, DataType::BFLOAT16, *device);
    auto b = ttnn::arange(32, DataType::BFLOAT16, *device);

    a.print();
    b.print();

    auto c = ttnn::add(a, b);
    c.print();
}

void test_ttnn() {
    // no device is required for this test
    using namespace tt::tt_metal;
    std::vector<float> data(32 * 32);
    for (int i = 0; i < 32 * 32; i++) {
        data[i] = float(i);
    }
    auto tensor1 = Tensor::from_vector(
        data,
        TensorSpec(Shape{32, 32}, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{})));
    tensor1.print();
    auto tensor2 = tensor1.to_layout(Layout::TILE);
    tensor2.print();
}

void test_tt_stl() {
    ttsl::SmallVector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    for (auto& v : vec) {
        std::cout << v << std::endl;
    }
}

int main() {
    char* env_var = std::getenv("TT_METAL_HOME");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_HOME to "
            "the path of the Metalium installation.\n");
    }

    fmt::print("TT_METAL_HOME: {}\n", env_var);
    test_tt_stl();
    test_ttnn();
    test_ttnn_add();
    return 0;
}
