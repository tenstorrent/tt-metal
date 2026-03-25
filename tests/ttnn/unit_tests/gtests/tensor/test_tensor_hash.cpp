// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>

#include <ttnn/distributed/tensor_topology.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::Tensor;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::TensorTopology;
using tt::tt_metal::distributed::MeshShape;

TEST(TensorToHashTest, NullTensorAttributes_HashIsStable) {
    Tensor empty;
    EXPECT_EQ(empty.to_hash(), empty.to_hash());
}

TEST(TensorToHashTest, HostTensorsWithIdenticalMetadataShareHash) {
    const TensorSpec spec(ttnn::Shape{1, 1, 1, 4}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    const std::vector<float> data = {1.F, 2.F, 3.F, 4.F};
    Tensor a = Tensor::from_vector(data, spec);
    Tensor b = Tensor::from_vector(data, spec);
    EXPECT_EQ(a.to_hash(), b.to_hash());
}

TEST(TensorToHashTest, WithTensorTopologyChangesHash) {
    const TensorSpec spec(ttnn::Shape{1, 1, 1, 4}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{}));
    const std::vector<float> data = {1.F, 2.F, 3.F, 4.F};
    Tensor base = Tensor::from_vector(data, spec);
    TensorTopology sharded = TensorTopology::create_sharded_tensor_topology(MeshShape(1, 2), 0);
    Tensor with_topo = base.with_tensor_topology(sharded);
    EXPECT_NE(base.to_hash(), with_topo.to_hash());
}
