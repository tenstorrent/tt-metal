// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Triage hang app that dispatches a TTNN-level operation which intentionally hangs.
// Exercises the dispatcher op-id -> Inspector operationName/parameters decode path
// in tools/triage/dump_running_operations.py.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>

#include "add_integers_hang_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/types.hpp"

int main() {
    auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);

    constexpr uint32_t M = tt::constants::TILE_HEIGHT;
    constexpr uint32_t N = tt::constants::TILE_WIDTH;
    std::vector<bfloat16> a_data(M * N, bfloat16(1.0f));
    std::vector<bfloat16> b_data(M * N, bfloat16(2.0f));

    tt::tt_metal::TensorLayout tile_layout(
        ttnn::DataType::BFLOAT16,
        tt::tt_metal::PageConfig(ttnn::Layout::TILE),
        ttnn::MemoryConfig(tt::tt_metal::BufferType::DRAM));
    tt::tt_metal::TensorSpec spec(ttnn::Shape({M, N}), tile_layout);

    ttnn::Tensor a = ttnn::Tensor::from_vector<bfloat16>(a_data, spec, mesh_device.get());
    ttnn::Tensor b = ttnn::Tensor::from_vector<bfloat16>(b_data, spec, mesh_device.get());

    try {
        ttnn::Tensor result = triage_hang_apps::add_integers_hang(a, b);
        // Force the dispatch to actually complete (which it won't — the kernel hangs).
        // Reading back will block until the op finishes or times out.
        std::cout << "Number of elements: " << result.to_vector<bfloat16>().size() << std::endl;
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        if (error_msg.find("device timeout") != std::string::npos || error_msg.find("Timeout (") != std::string::npos) {
            printf("Device timeout detected as expected.\n");
            std::_Exit(0);
        }
        throw;
    }

    mesh_device->close();
    return 0;
}
