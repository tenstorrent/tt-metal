// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstddef>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/to_string.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace ttnn::distributed {
namespace {

using ::testing::SizeIs;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::HostBuffer;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ttnn::Tensor;

using BigMeshDualRankTest2x4 = tt::tt_metal::MeshDevice2x4Fixture;

// One shard block as recovered from the printed representation.
struct PrintedShard {
    int device_id = -1;
    uint32_t row = 0;
    uint32_t col = 0;
    float first_value = std::numeric_limits<float>::quiet_NaN();
};

// Recovers the per-shard blocks that `to_string` emits for a device tensor. Each block is a
// "device_id: <id>, MeshCoordinate([<row>, <col>])" header followed by
// "ttnn.Tensor(<data>, shape=...)", so the header tells us which device the printer *claims* the
// data belongs to and the first datum tells us which device it *actually* came from.
std::vector<PrintedShard> parse_printed_shards(const std::string& printed) {
    static const std::regex header_pattern(R"(device_id: (\d+), MeshCoordinate\(\[(\d+), (\d+)\]\))");
    static const std::regex number_pattern(R"(-?\d+(?:\.\d+)?)");
    constexpr std::string_view kBodyPrefix = "ttnn.Tensor(";
    constexpr std::string_view kBodySuffix = ", shape=";

    std::vector<PrintedShard> shards;
    for (auto it = std::sregex_iterator(printed.begin(), printed.end(), header_pattern); it != std::sregex_iterator();
         ++it) {
        const std::smatch& match = *it;
        PrintedShard shard;
        shard.device_id = std::stoi(match[1].str());
        shard.row = static_cast<uint32_t>(std::stoul(match[2].str()));
        shard.col = static_cast<uint32_t>(std::stoul(match[3].str()));

        const auto header_end = static_cast<size_t>(match.position() + match.length());
        const size_t body_start = printed.find(kBodyPrefix, header_end);
        if (body_start != std::string::npos) {
            const size_t data_start = body_start + kBodyPrefix.size();
            const size_t data_end = printed.find(kBodySuffix, data_start);
            if (data_end != std::string::npos) {
                const std::string data = printed.substr(data_start, data_end - data_start);
                std::smatch number_match;
                if (std::regex_search(data, number_match, number_pattern)) {
                    shard.first_value = std::stof(number_match[0].str());
                }
            }
        }
        shards.push_back(shard);
    }
    return shards;
}

// Regression test for GitHub issue #48267.
//
// `to_string_impl` used to pair the buffers returned by `DistributedHostBuffer::apply()` (local,
// populated shards only) positionally against `DeviceStorage::get_coords()` (every coordinate of the
// mesh). On a multi-host mesh the two differ in length, so the printer consumed a *prefix* of the
// coordinate list: it dropped locally-owned shards whose paired coordinate happened to be remote,
// and attributed the shards it did print to the wrong device.
//
// Each shard carries a value unique to its device, so a mislabeling is detectable: the block printed
// under a coordinate must contain the data that actually lives at that coordinate.
TEST_F(BigMeshDualRankTest2x4, ToStringPairsEveryLocalShardWithItsOwnCoordinate) {
    constexpr int kNumDevices = 8;
    constexpr int kElementsPerShard = 4;
    ASSERT_EQ(mesh_device_->num_devices(), kNumDevices);

    // Shard i is filled with the single value 100 + i, so shards are mutually distinguishable.
    std::vector<float> test_data;
    for (int device = 0; device < kNumDevices; device++) {
        test_data.insert(test_data.end(), static_cast<size_t>(kElementsPerShard), 100.F + static_cast<float>(device));
    }

    const Tensor input_tensor = Tensor::from_vector(
        test_data,
        tt::tt_metal::TensorSpec(
            ttnn::Shape{1, kNumDevices, 1, kElementsPerShard},
            TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{})));

    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    const Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper, *mesh_device_);
    ASSERT_EQ(sharded_tensor.storage_type(), ttnn::StorageType::DEVICE);

    // Ground truth, addressed by coordinate rather than by position: read the tensor back and ask
    // the host buffer what it holds at each coordinate. This is independent of how the mapper
    // ordered the shards and of the pairing logic under test.
    const Tensor host_tensor = sharded_tensor.cpu();
    const auto& host_buffer = host_tensor.host_storage().buffer();

    std::vector<MeshCoordinate> expected_coords;
    std::vector<float> expected_values;
    for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
        if (!mesh_device_->is_local(coord)) {
            continue;
        }
        const std::optional<HostBuffer> shard = host_buffer.get_shard(coord);
        ASSERT_TRUE(shard.has_value()) << "no local shard materialized at " << coord;
        expected_coords.push_back(coord);
        expected_values.push_back(shard->view_as<float>()[0]);
    }
    // Each rank of the 2x4 dual-rank binding owns a 2x2 block, i.e. half of the mesh.
    ASSERT_THAT(expected_coords, SizeIs(kNumDevices / 2));

    const std::string printed = ttnn::to_string(sharded_tensor);
    const std::vector<PrintedShard> printed_shards = parse_printed_shards(printed);

    // Every locally-owned shard must appear exactly once.
    ASSERT_THAT(printed_shards, SizeIs(expected_coords.size()))
        << "expected one printed block per local shard, got " << printed_shards.size() << " for "
        << expected_coords.size() << " local shards. Printed:\n"
        << printed;

    for (size_t i = 0; i < expected_coords.size(); i++) {
        const MeshCoordinate& coord = expected_coords[i];
        const PrintedShard& shard = printed_shards[i];

        EXPECT_EQ(MeshCoordinate(shard.row, shard.col), coord) << "block " << i << " printed the wrong coordinate";
        EXPECT_EQ(shard.device_id, static_cast<int>(mesh_device_->get_device(coord)->id()))
            << "block " << i << " printed the wrong device_id for " << coord;
        // The payload must belong to the coordinate in the header, not to some other device.
        EXPECT_FLOAT_EQ(shard.first_value, expected_values[i])
            << "block " << i << " labelled " << coord << " carries the data of another device. Printed:\n"
            << printed;
    }
}

}  // namespace
}  // namespace ttnn::distributed
