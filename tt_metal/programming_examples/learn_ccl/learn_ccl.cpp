// SPDX-FileCopyrightText: © 2023-2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <cstdint>
#include <random>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/mesh_device.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include <ttnn/tensor/tensor.hpp>

int main() {
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_meshes({0, 1});
    log_info(tt::LogAlways, "Created mesh device: {}", mesh_device);
    std::vector<float> input_data(64 * 64, 0.0f);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i / 64);
    }
    std::vector<float> output_data(64 * 64, 0.0f);
    ttnn::Shape shape = ttnn::Shape({64, 64});
    auto input_host_buffer = tt::tt_metal::HostBuffer(std::move(input_data));
    auto output_host_buffer = tt::tt_metal::HostBuffer(std::move(output_data));

    auto input =
        ttnn::Tensor(input_host_buffer, shape, shape, tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::ROW_MAJOR);
    auto output = ttnn::Tensor(
        output_host_buffer, shape, shape, tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::ROW_MAJOR);
    return 0;
}
