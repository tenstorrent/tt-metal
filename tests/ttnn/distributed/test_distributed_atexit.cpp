// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <ttnn/core.hpp>
#include <ttnn/distributed/api.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::distributed::test {

// Simplified test without fixture, and mesh variable moved inside test
TEST(DistributedTestStandalone, TestSystemMeshTearDownWithoutClose) {
    static std::shared_ptr<ttnn::MeshDevice> mesh;
    auto& sys = tt::tt_metal::distributed::SystemMesh::instance();
    mesh = ttnn::distributed::open_mesh_device(
        {2, 4}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);

    auto [rows, cols] = sys.get_shape();
    EXPECT_GT(rows, 0);
    EXPECT_GT(cols, 0);
}

}  // namespace ttnn::distributed::test
