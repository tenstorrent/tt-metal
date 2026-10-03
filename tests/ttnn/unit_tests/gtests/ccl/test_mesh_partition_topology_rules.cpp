// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests of detail::compute_mesh_partition_topology, the rule table behind
// MeshPartitionDeviceOperation::compute_output_topologies (the table is in the op header). No device: the rules are a
// pure function of the input label, the device mesh shape, dim, cluster_axis and the tensor rank.

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/operations/ccl/mesh_partition/device/mesh_partition_device_operation.hpp"

namespace {

using tt::tt_metal::TensorTopology;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshShape;
using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
using ttnn::operations::ccl::detail::compute_mesh_partition_topology;

constexpr uint32_t kRank = 4;
constexpr uint32_t kDim = 3;
const std::optional<uint32_t> kWholeMesh = std::nullopt;

std::vector<MeshCoordinate> row_major_coords(const MeshShape& mesh) {
    std::vector<MeshCoordinate> coords;
    for (const auto& coord : MeshCoordinateRange(mesh)) {
        coords.push_back(coord);
    }
    return coords;
}

// One placement per mesh axis, as ShardTensor2dMesh produces.
TensorTopology nd_label(const MeshShape& mesh, const std::vector<Placement>& placements) {
    return TensorTopology(
        mesh, ttsl::SmallVector<Placement>(placements.begin(), placements.end()), row_major_coords(mesh));
}

// {N},[placement] over the mesh in row-major order, as ShardTensorToMesh / ReplicateTensorToMesh produce.
TensorTopology collapsed_label(const MeshShape& mesh, const Placement& placement) {
    return TensorTopology(MeshShape(static_cast<uint32_t>(mesh.mesh_size())), {placement}, row_major_coords(mesh));
}

void expect_placements(const TensorTopology& topology, const std::vector<Placement>& expected) {
    ASSERT_EQ(topology.placements().size(), expected.size()) << topology;
    for (size_t axis = 0; axis < expected.size(); ++axis) {
        EXPECT_EQ(topology.placements()[axis], expected[axis]) << topology << " axis " << axis;
    }
}

}  // namespace

// Rule 1: Shard{dim} on the partitioned axis, the input's placement elsewhere.
TEST(MeshPartitionTopologyRules, NdLabelTakesShardDimOnThePartitionedAxis) {
    const MeshShape mesh(2, 4);
    const auto input = nd_label(mesh, {Shard{2}, Replicate{}});
    for (uint32_t axis : {0U, 1U}) {
        const auto result = compute_mesh_partition_topology(input, mesh, kDim, axis, kRank);
        ASSERT_TRUE(result.topology.has_value()) << "axis " << axis;
        EXPECT_EQ(result.topology->distribution_shape(), mesh);
        EXPECT_EQ(result.topology->mesh_coords(), input.mesh_coords());
        expect_placements(
            *result.topology,
            axis == 0 ? std::vector<Placement>{Shard{kDim}, Replicate{}}
                      : std::vector<Placement>{Shard{2}, Shard{kDim}});
    }
}

// Rule 1(i): one chunk along a size-1 axis is the whole extent.
TEST(MeshPartitionTopologyRules, SameDimOnASizeOneAxisBecomesReplicate) {
    const MeshShape mesh(1, 8);
    const auto result =
        compute_mesh_partition_topology(nd_label(mesh, {Shard{kDim}, Replicate{}}), mesh, kDim, 1, kRank);
    ASSERT_TRUE(result.topology.has_value());
    expect_placements(*result.topology, {Replicate{}, Shard{kDim}});
}

// Rule 1(ii): an OUTER axis already holding Shard{dim} composes with the partitioned axis into row-major
// hierarchical sharding, which the collapsed label states exactly when the partitioned axis held Replicate; when it
// already held Shard{dim} (a label the mappers refuse to build) the collapsed label is rule 2's output-bytes stance.
TEST(MeshPartitionTopologyRules, SameDimOnAnOuterAxisCollapsesRowMajor) {
    const MeshShape mesh(2, 4);
    for (const Placement& partitioned : {Placement{Replicate{}}, Placement{Shard{kDim}}}) {
        const auto input = nd_label(mesh, {Shard{kDim}, partitioned});
        const auto result = compute_mesh_partition_topology(input, mesh, kDim, 1, kRank);
        ASSERT_TRUE(result.topology.has_value()) << input;
        EXPECT_EQ(result.topology->distribution_shape(), MeshShape(8));
        expect_placements(*result.topology, {Shard{kDim}});
        EXPECT_EQ(result.topology->mesh_coords(), input.mesh_coords());
    }
    // Shard dims in the label are compared normalised: -1 is dim 3 at rank 4.
    const auto negative =
        compute_mesh_partition_topology(nd_label(mesh, {Shard{-1}, Replicate{}}), mesh, kDim, 1, kRank);
    ASSERT_TRUE(negative.topology.has_value());
    EXPECT_EQ(negative.topology->distribution_shape(), MeshShape(8));
    // An out-of-range Shard dim (left behind by a rank-changing op) does not count as sharding dim.
    const auto stale = compute_mesh_partition_topology(nd_label(mesh, {Shard{7}, Replicate{}}), mesh, kDim, 1, kRank);
    ASSERT_TRUE(stale.topology.has_value());
    expect_placements(*stale.topology, {Shard{7}, Shard{kDim}});
}

// Rule 1(iii): everything else that would shard dim on two axes has no label.
TEST(MeshPartitionTopologyRules, SameDimOnAnInnerAxisOrAnotherShardOnThePartitionedAxisFallsBack) {
    const MeshShape mesh(2, 4);
    // Columns hold fine slices of dim; partitioning across rows cuts them again: column-major, not expressible.
    const auto inner =
        compute_mesh_partition_topology(nd_label(mesh, {Replicate{}, Shard{kDim}}), mesh, kDim, 0, kRank);
    EXPECT_FALSE(inner.topology.has_value());
    EXPECT_NE(inner.fallback_reason, nullptr);
    // The outer axis shards dim but the partitioned axis held a different Shard.
    const auto other = compute_mesh_partition_topology(nd_label(mesh, {Shard{kDim}, Shard{2}}), mesh, kDim, 1, kRank);
    EXPECT_FALSE(other.topology.has_value());
    EXPECT_NE(other.fallback_reason, nullptr);
    // A third non-trivial axis would make the collapsed label over-claim N distinct slices.
    const MeshShape mesh3(2, 2, 2);
    const auto third = compute_mesh_partition_topology(
        nd_label(mesh3, {Shard{kDim}, Replicate{}, Replicate{}}), mesh3, kDim, 1, kRank);
    EXPECT_FALSE(third.topology.has_value());
    EXPECT_NE(third.fallback_reason, nullptr);
}

// Rule 2: a whole-mesh partition of a replicated input is {N},[Shard{dim}] in row-major device order, whether the
// input label is N-D or collapsed.
TEST(MeshPartitionTopologyRules, WholeMeshCollapsesAReplicatedInput) {
    for (const MeshShape& mesh : {MeshShape(1, 8), MeshShape(2, 4), MeshShape(8, 4)}) {
        const auto nd =
            compute_mesh_partition_topology(nd_label(mesh, {Replicate{}, Replicate{}}), mesh, kDim, kWholeMesh, kRank);
        ASSERT_TRUE(nd.topology.has_value()) << mesh;
        EXPECT_EQ(nd.topology->distribution_shape(), MeshShape(static_cast<uint32_t>(mesh.mesh_size())));
        expect_placements(*nd.topology, {Shard{kDim}});
        EXPECT_EQ(nd.topology->mesh_coords(), row_major_coords(mesh));
        const auto collapsed =
            compute_mesh_partition_topology(collapsed_label(mesh, Replicate{}), mesh, kDim, kWholeMesh, kRank);
        ASSERT_TRUE(collapsed.topology.has_value()) << mesh;
        EXPECT_EQ(*collapsed.topology, *nd.topology) << mesh;
    }
}

// Rule 2 on an input already sharded: Shard{dim} on a non-trivial axis is overwritten (the label then describes the
// output bytes: device k holds chunk k of ITS OWN input shard, not a re-slicing of the input's global view); a Shard
// of another dim on a non-trivial axis has no label; a Shard on a size-1 axis is the whole extent and does not block.
TEST(MeshPartitionTopologyRules, WholeMeshOverwritesShardDimButNotAnotherDim) {
    const MeshShape mesh(2, 4);
    const auto same_dim =
        compute_mesh_partition_topology(nd_label(mesh, {Shard{kDim}, Replicate{}}), mesh, kDim, kWholeMesh, kRank);
    ASSERT_TRUE(same_dim.topology.has_value());
    EXPECT_EQ(same_dim.topology->distribution_shape(), MeshShape(8));
    expect_placements(*same_dim.topology, {Shard{kDim}});

    const auto other_dim =
        compute_mesh_partition_topology(nd_label(mesh, {Shard{2}, Replicate{}}), mesh, kDim, kWholeMesh, kRank);
    EXPECT_FALSE(other_dim.topology.has_value());
    EXPECT_NE(other_dim.fallback_reason, nullptr);

    const MeshShape line(1, 8);
    const auto trivial =
        compute_mesh_partition_topology(nd_label(line, {Shard{2}, Replicate{}}), line, kDim, kWholeMesh, kRank);
    ASSERT_TRUE(trivial.topology.has_value());
    expect_placements(*trivial.topology, {Shard{kDim}});
}

// Rule 3: on a line the collapsed label's only axis IS the partitioned axis; whatever it held is overwritten.
TEST(MeshPartitionTopologyRules, CollapsedLabelOnALineTakesShardDimWhateverItHeld) {
    const MeshShape mesh(1, 8);
    for (const Placement& held : {Placement{Replicate{}}, Placement{Shard{1}}, Placement{Shard{kDim}}}) {
        const auto input = collapsed_label(mesh, held);
        const auto result = compute_mesh_partition_topology(input, mesh, kDim, 1, kRank);
        ASSERT_TRUE(result.topology.has_value()) << input;
        EXPECT_EQ(result.topology->distribution_shape(), MeshShape(8));
        expect_placements(*result.topology, {Shard{kDim}});
        EXPECT_EQ(result.topology->mesh_coords(), input.mesh_coords());
    }
}

// Rule 4: a collapsed Replicate over a multi-axis mesh is spelled out per axis so that one axis can be Shard{dim}.
TEST(MeshPartitionTopologyRules, CollapsedReplicateOnAMultiAxisMeshUncollapses) {
    const MeshShape mesh(2, 4);
    const auto input = collapsed_label(mesh, Replicate{});
    for (uint32_t axis : {0U, 1U}) {
        const auto result = compute_mesh_partition_topology(input, mesh, kDim, axis, kRank);
        ASSERT_TRUE(result.topology.has_value()) << "axis " << axis;
        EXPECT_EQ(result.topology->distribution_shape(), mesh);
        expect_placements(
            *result.topology,
            axis == 0 ? std::vector<Placement>{Shard{kDim}, Replicate{}}
                      : std::vector<Placement>{Replicate{}, Shard{kDim}});
        EXPECT_EQ(result.topology->mesh_coords(), input.mesh_coords());
    }
}

// Rule 5: a collapsed Shard over a multi-axis mesh partitioned along one axis falls back, with a reason like every
// other fallback (neither form can hold both the old Shard and the new one).
TEST(MeshPartitionTopologyRules, CollapsedShardOnAMultiAxisMeshFallsBack) {
    const MeshShape mesh(2, 4);
    for (uint32_t axis : {0U, 1U}) {
        const auto result = compute_mesh_partition_topology(collapsed_label(mesh, Shard{2}), mesh, kDim, axis, kRank);
        EXPECT_FALSE(result.topology.has_value()) << "axis " << axis;
        EXPECT_NE(result.fallback_reason, nullptr) << "axis " << axis;
    }
}

// Rule 0, collapsed labels: no axes, so a label over fewer devices than the mesh says nothing about where its devices
// sit. The fewer-shards label whose N equals the column count is the rule-3 look-alike; the whole-mesh case is the
// regression Copilot flagged on the first version.
TEST(MeshPartitionTopologyRules, CollapsedLabelOverFewerDevicesThanTheMeshFallsBackForEveryClusterAxis) {
    const MeshShape mesh(2, 4);
    auto coords = row_major_coords(mesh);
    coords.erase(coords.begin() + 4, coords.end());
    const TensorTopology fewer_shards(MeshShape(4), {Shard{2}}, coords);
    for (const auto& axis : {std::optional<uint32_t>{0}, std::optional<uint32_t>{1}, kWholeMesh}) {
        const auto result = compute_mesh_partition_topology(fewer_shards, mesh, kDim, axis, kRank);
        EXPECT_FALSE(result.topology.has_value())
            << "cluster_axis " << (axis.has_value() ? static_cast<int>(*axis) : -1);
        EXPECT_NE(result.fallback_reason, nullptr);
    }
}

// Rule 0, N-D sub-mesh labels (mesh_shape_override that fits per axis; coordinates are the block's own): the label is
// kept, and rule 1 applied within the block, only when the cluster axis spans the mesh along that axis, so that every
// partition group lies inside the block. Otherwise a group reaches a device the label does not describe.
TEST(MeshPartitionTopologyRules, SubMeshLabelIsPartitionedWithinItsBlockOnlyAlongAFullAxis) {
    const MeshShape mesh(2, 4);
    auto row0 = row_major_coords(mesh);
    row0.erase(row0.begin() + 4, row0.end());
    const TensorTopology row_block(MeshShape(1, 4), {Replicate{}, Replicate{}}, row0);

    const auto along_row = compute_mesh_partition_topology(row_block, mesh, kDim, 1, kRank);  // 4 == 4
    ASSERT_TRUE(along_row.topology.has_value());
    EXPECT_EQ(along_row.topology->distribution_shape(), MeshShape(1, 4));
    expect_placements(*along_row.topology, {Replicate{}, Shard{kDim}});
    EXPECT_EQ(along_row.topology->mesh_coords(), row0);

    const auto across_rows = compute_mesh_partition_topology(row_block, mesh, kDim, 0, kRank);  // 1 < 2
    EXPECT_FALSE(across_rows.topology.has_value());
    EXPECT_NE(across_rows.fallback_reason, nullptr);

    const auto whole = compute_mesh_partition_topology(row_block, mesh, kDim, kWholeMesh, kRank);
    EXPECT_FALSE(whole.topology.has_value());
    EXPECT_NE(whole.fallback_reason, nullptr);

    // A {2,2} block at an offset (mesh_offset_override): axis 0 spans the mesh, axis 1 does not.
    const std::vector<MeshCoordinate> block{
        MeshCoordinate(0, 2), MeshCoordinate(0, 3), MeshCoordinate(1, 2), MeshCoordinate(1, 3)};
    const TensorTopology square(MeshShape(2, 2), {Replicate{}, Shard{2}}, block);
    const auto down = compute_mesh_partition_topology(square, mesh, kDim, 0, kRank);
    ASSERT_TRUE(down.topology.has_value());
    expect_placements(*down.topology, {Shard{kDim}, Shard{2}});
    EXPECT_EQ(down.topology->mesh_coords(), block);
    EXPECT_FALSE(compute_mesh_partition_topology(square, mesh, kDim, 1, kRank).topology.has_value());
}

// Rule 0, row-major reshapes of the whole mesh ({1,8} or {4,2} over 2x4; mesh_shape_override that does not fit per
// axis): the coordinates enumerate the mesh row-major, so a whole-mesh partition has its exact collapsed label, but
// the label's axes are not the mesh axes the op partitions along.
TEST(MeshPartitionTopologyRules, RowMajorReshapeOfTheMeshPartitionsAsAWholeOnly) {
    const MeshShape mesh(2, 4);
    const TensorTopology reshaped(MeshShape(1, 8), {Replicate{}, Replicate{}}, row_major_coords(mesh));
    const auto whole = compute_mesh_partition_topology(reshaped, mesh, kDim, kWholeMesh, kRank);
    ASSERT_TRUE(whole.topology.has_value());
    EXPECT_EQ(whole.topology->distribution_shape(), MeshShape(8));
    expect_placements(*whole.topology, {Shard{kDim}});
    EXPECT_EQ(whole.topology->mesh_coords(), row_major_coords(mesh));
    for (uint32_t axis : {0U, 1U}) {
        const auto result = compute_mesh_partition_topology(reshaped, mesh, kDim, axis, kRank);
        EXPECT_FALSE(result.topology.has_value()) << "axis " << axis;
        EXPECT_NE(result.fallback_reason, nullptr) << "axis " << axis;
    }
    const auto transposed = compute_mesh_partition_topology(
        TensorTopology(MeshShape(4, 2), {Replicate{}, Replicate{}}, row_major_coords(mesh)), mesh, kDim, 1, kRank);
    EXPECT_FALSE(transposed.topology.has_value());
    EXPECT_NE(transposed.fallback_reason, nullptr);
}

// Every emitted label depends on the coordinates agreeing with the partition: the factory picks chunks from the
// device's own coordinate, not from the label. The mappers always write agreeing coordinates (SUBMESH: an axis-aligned
// block; ROW_MAJOR: the mesh's row-major enumeration); a label put together by hand with update_tensor_topology need
// not, and must fall back rather than state chunks the devices do not hold. The check is per axis: a disagreement
// along one mesh axis leaves partitions along the other axis labelled.
TEST(MeshPartitionTopologyRules, LabelWhoseCoordinatesDisagreeWithThePartitionFallsBack) {
    const MeshShape mesh(2, 4);
    // A block SHAPE over a ROW of devices passes rule 0's per-axis guard (2 == 2 on axis 0), yet every device has
    // global row 0 and takes chunk 0; rule 1 would have labelled tensor row 1 as holding chunk 1.
    const std::vector<MeshCoordinate> row{
        MeshCoordinate(0, 0), MeshCoordinate(0, 1), MeshCoordinate(0, 2), MeshCoordinate(0, 3)};
    const auto block_shape_over_row = compute_mesh_partition_topology(
        TensorTopology(MeshShape(2, 2), {Replicate{}, Replicate{}}, row), mesh, kDim, 0, kRank);
    EXPECT_FALSE(block_shape_over_row.topology.has_value());
    EXPECT_NE(block_shape_over_row.fallback_reason, nullptr);

    // A full-mesh N-D label with two coordinates swapped across rows: rule 0 does not look at it (the shape is the
    // mesh's) and rule 1 would carry the coordinates through. Both swapped devices keep column 1, so partitions
    // along axis 1 are still labelled.
    auto swapped = row_major_coords(mesh);
    std::swap(swapped[1], swapped[5]);  // (0,1) <-> (1,1)
    const TensorTopology permuted_nd(mesh, {Replicate{}, Replicate{}}, swapped);
    const auto nd_rows = compute_mesh_partition_topology(permuted_nd, mesh, kDim, 0, kRank);
    EXPECT_FALSE(nd_rows.topology.has_value());
    EXPECT_NE(nd_rows.fallback_reason, nullptr);
    const auto nd_whole = compute_mesh_partition_topology(permuted_nd, mesh, kDim, kWholeMesh, kRank);
    EXPECT_FALSE(nd_whole.topology.has_value());
    EXPECT_NE(nd_whole.fallback_reason, nullptr);
    EXPECT_TRUE(compute_mesh_partition_topology(permuted_nd, mesh, kDim, 1, kRank).topology.has_value());

    // Collapsed labels: rule 3 on a line with reversed coordinates, rule 4 on the 2x4 with the swap above.
    const MeshShape line(1, 8);
    auto reversed = row_major_coords(line);
    std::reverse(reversed.begin(), reversed.end());
    const auto rule3 =
        compute_mesh_partition_topology(TensorTopology(MeshShape(8), {Replicate{}}, reversed), line, kDim, 1, kRank);
    EXPECT_FALSE(rule3.topology.has_value());
    EXPECT_NE(rule3.fallback_reason, nullptr);
    const TensorTopology permuted_collapsed(MeshShape(8), {Replicate{}}, swapped);
    const auto rule4 = compute_mesh_partition_topology(permuted_collapsed, mesh, kDim, 0, kRank);
    EXPECT_FALSE(rule4.topology.has_value());
    EXPECT_NE(rule4.fallback_reason, nullptr);
    EXPECT_TRUE(compute_mesh_partition_topology(permuted_collapsed, mesh, kDim, 1, kRank).topology.has_value());
}

// Inputs validation rejects right after the hook, or that carry no placements, leave the union default in place
// without a warning (the hook runs before validation and must not crash or shout first).
TEST(MeshPartitionTopologyRules, OutOfRangeClusterAxisOrEmptyLabelLeavesTheUnionDefault) {
    const MeshShape mesh(2, 4);
    const auto bad_axis =
        compute_mesh_partition_topology(nd_label(mesh, {Replicate{}, Replicate{}}), mesh, kDim, 2, kRank);
    EXPECT_FALSE(bad_axis.topology.has_value());
    EXPECT_EQ(bad_axis.fallback_reason, nullptr);

    const TensorTopology no_placements(mesh, {}, row_major_coords(mesh));
    const auto empty = compute_mesh_partition_topology(no_placements, mesh, kDim, 1, kRank);
    EXPECT_FALSE(empty.topology.has_value());
    EXPECT_EQ(empty.fallback_reason, nullptr);
}
