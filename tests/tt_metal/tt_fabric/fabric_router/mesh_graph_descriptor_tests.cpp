// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/mesh_graph_descriptor.hpp>

namespace tt::tt_fabric {

namespace fabric_router_tests {

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoString) {
    const std::string text_proto = R"proto(
mesh_descriptors: {
  id: 0
  arch: "WORMHOLE_B0"
  device_topology: { dims: { dim: 1 } }
  channels: { count: 1 }
  host_topology: { dims: 1 }
}
top_level_instance: { mesh: { mesh_descriptor: "m0" id: 0 } }
)proto";

    tt::tt_fabric::MeshGraphDescriptor desc(text_proto);
}

}  // namespace fabric_router_tests

}  // namespace tt::tt_fabric

