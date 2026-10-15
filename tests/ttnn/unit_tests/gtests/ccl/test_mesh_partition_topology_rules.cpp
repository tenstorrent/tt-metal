// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests of the mesh_partition label rules, which live in the CCL topology helper (ccl_topology_utils.hpp:
// mesh_partition_output_topology, the reduce_scatter rule where nothing is summed) since the inline rule table #58915
// landed with moved there. The cases keep that table's numbering (rules 0-5). No device: the rules are a pure
// function of the input label, the device mesh shape, dim, cluster_axis and the tensor rank. Every test runs
// warn-only, so a refusal is a nullopt here (a TT_FATAL under ttnn.CONFIG.strict_ccl_topology); the op leaves the
// input's label on the output then.

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include "ttnn/config.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"

namespace {

using tt::tt_metal::TensorTopology;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshShape;
using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
namespace topo = ttnn::operations::ccl::common;

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

// Runs the test warn-only (refusals are nullopt, not TT_FATAL) and restores the setting after.
class WarnOnlyScope {
public:
    WarnOnlyScope() : previous_(ttnn::CONFIG.get<"strict_ccl_topology">()) {
        ttnn::CONFIG.set<"strict_ccl_topology">(false);
    }
    ~WarnOnlyScope() { ttnn::CONFIG.set<"strict_ccl_topology">(previous_); }

private:
    bool previous_;
};

std::optional<TensorTopology> partition(
    const TensorTopology& input, const MeshShape& mesh, std::optional<uint32_t> cluster_axis) {
    return topo::mesh_partition_output_topology(input, cluster_axis, mesh, kRank, static_cast<int32_t>(kDim));
}

}  // namespace

// Rule 1: Shard{dim} on the partitioned axis, the input's placement elsewhere -- unless the partitioned axis itself
// held a Shard of another dim: each device would then keep its chunk of that dim next to its new piece of dim, two
// dims on one axis, which no label states, so that case is refused (the inline table overwrote the Shard{2}).
TEST(MeshPartitionTopologyRules, NdLabelTakesShardDimOnThePartitionedAxisUnlessItHeldAnotherShard) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    const auto input = nd_label(mesh, {Shard{2}, Replicate{}});
    const auto along_columns = partition(input, mesh, 1);
    ASSERT_TRUE(along_columns.has_value());
    EXPECT_EQ(along_columns->distribution_shape(), mesh);
    EXPECT_EQ(along_columns->mesh_coords(), input.mesh_coords());
    expect_placements(*along_columns, {Shard{2}, Shard{kDim}});
    EXPECT_FALSE(partition(input, mesh, 0).has_value());  // axis 0 holds Shard{2}
}

// Rule 1(i): one chunk along a size-1 axis is the whole extent.
TEST(MeshPartitionTopologyRules, SameDimOnASizeOneAxisBecomesReplicate) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(1, 8);
    const auto result = partition(nd_label(mesh, {Shard{kDim}, Replicate{}}), mesh, 1);
    ASSERT_TRUE(result.has_value());
    expect_placements(*result, {Replicate{}, Shard{kDim}});
}

// Rule 1(ii): an OUTER axis already holding Shard{dim} composes with the partitioned axis into row-major
// hierarchical sharding, which the collapsed label states exactly when the partitioned axis held Replicate; when it
// already held Shard{dim} (a label the mappers refuse to build) the collapsed label is rule 2's output-bytes stance.
TEST(MeshPartitionTopologyRules, SameDimOnAnOuterAxisCollapsesRowMajor) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    for (const Placement& partitioned : {Placement{Replicate{}}, Placement{Shard{kDim}}}) {
        const auto input = nd_label(mesh, {Shard{kDim}, partitioned});
        const auto result = partition(input, mesh, 1);
        ASSERT_TRUE(result.has_value()) << input;
        EXPECT_EQ(result->distribution_shape(), MeshShape(8));
        expect_placements(*result, {Shard{kDim}});
        EXPECT_EQ(result->mesh_coords(), input.mesh_coords());
    }
    // Shard dims in the label are compared normalised: -1 is dim 3 at rank 4.
    const auto negative = partition(nd_label(mesh, {Shard{-1}, Replicate{}}), mesh, 1);
    ASSERT_TRUE(negative.has_value());
    EXPECT_EQ(negative->distribution_shape(), MeshShape(8));
    // An out-of-range Shard dim (left behind by a rank-changing op) does not count as sharding dim.
    const auto stale = partition(nd_label(mesh, {Shard{7}, Replicate{}}), mesh, 1);
    ASSERT_TRUE(stale.has_value());
    expect_placements(*stale, {Shard{7}, Shard{kDim}});
}

// Rule 1(iii): everything else that would shard dim on two axes has no label.
TEST(MeshPartitionTopologyRules, SameDimOnAnInnerAxisOrAnotherShardOnThePartitionedAxisFallsBack) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    // Columns hold fine slices of dim; partitioning across rows cuts them again: column-major, not expressible.
    EXPECT_FALSE(partition(nd_label(mesh, {Replicate{}, Shard{kDim}}), mesh, 0).has_value());
    // The outer axis shards dim but the partitioned axis held a different Shard.
    EXPECT_FALSE(partition(nd_label(mesh, {Shard{kDim}, Shard{2}}), mesh, 1).has_value());
    // A third non-trivial axis would make the collapsed label over-claim N distinct slices.
    const MeshShape mesh3(2, 2, 2);
    EXPECT_FALSE(partition(nd_label(mesh3, {Shard{kDim}, Replicate{}, Replicate{}}), mesh3, 1).has_value());
}

// Rule 2: a whole-mesh partition of a replicated input is {N},[Shard{dim}] in row-major device order, whether the
// input label is N-D or collapsed.
TEST(MeshPartitionTopologyRules, WholeMeshCollapsesAReplicatedInput) {
    const WarnOnlyScope warn_only;
    for (const MeshShape& mesh : {MeshShape(1, 8), MeshShape(2, 4), MeshShape(8, 4)}) {
        const auto nd = partition(nd_label(mesh, {Replicate{}, Replicate{}}), mesh, kWholeMesh);
        ASSERT_TRUE(nd.has_value()) << mesh;
        EXPECT_EQ(nd->distribution_shape(), MeshShape(static_cast<uint32_t>(mesh.mesh_size())));
        expect_placements(*nd, {Shard{kDim}});
        EXPECT_EQ(nd->mesh_coords(), row_major_coords(mesh));
        const auto collapsed = partition(collapsed_label(mesh, Replicate{}), mesh, kWholeMesh);
        ASSERT_TRUE(collapsed.has_value()) << mesh;
        EXPECT_EQ(*collapsed, *nd) << mesh;
    }
}

// Rule 2 on an input already sharded: Shard{dim} on a non-trivial axis is overwritten (the label then describes the
// output bytes: device k holds chunk k of ITS OWN input shard, not a re-slicing of the input's global view); a Shard
// of another dim on a non-trivial axis has no label, N-D or collapsed; a Shard on a size-1 axis is the whole extent
// and does not block.
TEST(MeshPartitionTopologyRules, WholeMeshOverwritesShardDimButNotAnotherDim) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    const auto same_dim = partition(nd_label(mesh, {Shard{kDim}, Replicate{}}), mesh, kWholeMesh);
    ASSERT_TRUE(same_dim.has_value());
    EXPECT_EQ(same_dim->distribution_shape(), MeshShape(8));
    expect_placements(*same_dim, {Shard{kDim}});

    EXPECT_FALSE(partition(nd_label(mesh, {Shard{2}, Replicate{}}), mesh, kWholeMesh).has_value());
    EXPECT_FALSE(partition(collapsed_label(mesh, Shard{2}), mesh, kWholeMesh).has_value());

    const MeshShape line(1, 8);
    const auto trivial = partition(nd_label(line, {Shard{2}, Replicate{}}), line, kWholeMesh);
    ASSERT_TRUE(trivial.has_value());
    expect_placements(*trivial, {Shard{kDim}});
}

// Rule 3: on a line the collapsed label's only axis IS the partitioned axis; whatever it held is overwritten.
TEST(MeshPartitionTopologyRules, CollapsedLabelOnALineTakesShardDimWhateverItHeld) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(1, 8);
    for (const Placement& held : {Placement{Replicate{}}, Placement{Shard{1}}, Placement{Shard{kDim}}}) {
        const auto input = collapsed_label(mesh, held);
        const auto result = partition(input, mesh, 1);
        ASSERT_TRUE(result.has_value()) << input;
        EXPECT_EQ(result->distribution_shape(), MeshShape(8));
        expect_placements(*result, {Shard{kDim}});
        EXPECT_EQ(result->mesh_coords(), input.mesh_coords());
    }
}

// Rule 4: a collapsed Replicate over a multi-axis mesh is spelled out per axis so that one axis can be Shard{dim}.
TEST(MeshPartitionTopologyRules, CollapsedReplicateOnAMultiAxisMeshUncollapses) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    const auto input = collapsed_label(mesh, Replicate{});
    for (uint32_t axis : {0U, 1U}) {
        const auto result = partition(input, mesh, axis);
        ASSERT_TRUE(result.has_value()) << "axis " << axis;
        EXPECT_EQ(result->distribution_shape(), mesh);
        expect_placements(
            *result,
            axis == 0 ? std::vector<Placement>{Shard{kDim}, Replicate{}}
                      : std::vector<Placement>{Replicate{}, Shard{kDim}});
        EXPECT_EQ(result->mesh_coords(), input.mesh_coords());
    }
}

// Rule 5, read per axis: a collapsed Shard over a multi-axis mesh is row-major hierarchical sharding of its dim. A
// Shard of another dim is refused along either axis (the partitioned axis holds a piece of two dims). A Shard of dim
// itself collapses again along the inner axis (each device keeps a sub-piece of its piece: N distinct, ordered
// slices, the same spelling as the input) and is refused along the outer axis (column-major). The inline table fell
// back for all of these.
TEST(MeshPartitionTopologyRules, CollapsedShardOnAMultiAxisMeshIsReadPerAxis) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    for (uint32_t axis : {0U, 1U}) {
        EXPECT_FALSE(partition(collapsed_label(mesh, Shard{2}), mesh, axis).has_value()) << "axis " << axis;
    }
    EXPECT_EQ(partition(collapsed_label(mesh, Shard{kDim}), mesh, 1), collapsed_label(mesh, Shard{kDim}));
    EXPECT_FALSE(partition(collapsed_label(mesh, Shard{kDim}), mesh, 0).has_value());
}

// Rule 0, collapsed labels: no axes, so a label over fewer devices than the mesh says nothing about where its devices
// sit. The fewer-shards label whose N equals the column count is the rule-3 look-alike; the whole-mesh case is the
// regression Copilot flagged on the first version.
TEST(MeshPartitionTopologyRules, CollapsedLabelOverFewerDevicesThanTheMeshFallsBackForEveryClusterAxis) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    auto coords = row_major_coords(mesh);
    coords.erase(coords.begin() + 4, coords.end());
    const TensorTopology fewer_shards(MeshShape(4), {Shard{2}}, coords);
    for (const auto& axis : {std::optional<uint32_t>{0}, std::optional<uint32_t>{1}, kWholeMesh}) {
        EXPECT_FALSE(partition(fewer_shards, mesh, axis).has_value())
            << "cluster_axis " << (axis.has_value() ? static_cast<int>(*axis) : -1);
    }
}

// Rule 0, N-D sub-mesh labels (mesh_shape_override that fits per axis; coordinates are the block's own): the label is
// kept, and rule 1 applied within the block, only when the cluster axis spans the mesh along that axis, so that every
// partition group lies inside the block. Otherwise a group reaches a device the label does not describe.
TEST(MeshPartitionTopologyRules, SubMeshLabelIsPartitionedWithinItsBlockOnlyAlongAFullAxis) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    auto row0 = row_major_coords(mesh);
    row0.erase(row0.begin() + 4, row0.end());
    const TensorTopology row_block(MeshShape(1, 4), {Replicate{}, Replicate{}}, row0);

    const auto along_row = partition(row_block, mesh, 1);  // 4 == 4
    ASSERT_TRUE(along_row.has_value());
    EXPECT_EQ(along_row->distribution_shape(), MeshShape(1, 4));
    expect_placements(*along_row, {Replicate{}, Shard{kDim}});
    EXPECT_EQ(along_row->mesh_coords(), row0);

    EXPECT_FALSE(partition(row_block, mesh, 0).has_value());           // 1 < 2
    EXPECT_FALSE(partition(row_block, mesh, kWholeMesh).has_value());  // 4 < 8

    // A {2,2} block at an offset (mesh_offset_override): axis 0 spans the mesh, axis 1 does not.
    const std::vector<MeshCoordinate> block{
        MeshCoordinate(0, 2), MeshCoordinate(0, 3), MeshCoordinate(1, 2), MeshCoordinate(1, 3)};
    const TensorTopology square(MeshShape(2, 2), {Replicate{}, Shard{2}}, block);
    const auto down = partition(square, mesh, 0);
    ASSERT_TRUE(down.has_value());
    expect_placements(*down, {Shard{kDim}, Shard{2}});
    EXPECT_EQ(down->mesh_coords(), block);
    EXPECT_FALSE(partition(square, mesh, 1).has_value());
}

// Rule 0, row-major reshapes of the whole mesh ({1,8} or {4,2} over 2x4; mesh_shape_override that does not fit per
// axis): the coordinates enumerate the mesh row-major, so a whole-mesh partition has its exact collapsed label, but
// the label's axes are not the mesh axes the op partitions along.
TEST(MeshPartitionTopologyRules, RowMajorReshapeOfTheMeshPartitionsAsAWholeOnly) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    const TensorTopology reshaped(MeshShape(1, 8), {Replicate{}, Replicate{}}, row_major_coords(mesh));
    const auto whole = partition(reshaped, mesh, kWholeMesh);
    ASSERT_TRUE(whole.has_value());
    EXPECT_EQ(whole->distribution_shape(), MeshShape(8));
    expect_placements(*whole, {Shard{kDim}});
    EXPECT_EQ(whole->mesh_coords(), row_major_coords(mesh));
    for (uint32_t axis : {0U, 1U}) {
        EXPECT_FALSE(partition(reshaped, mesh, axis).has_value()) << "axis " << axis;
    }
    const TensorTopology transposed(MeshShape(4, 2), {Replicate{}, Replicate{}}, row_major_coords(mesh));
    EXPECT_FALSE(partition(transposed, mesh, 1).has_value());
}

// The coordinates must agree with the partition, which the factory computes from the device's own coordinate, not
// from the label. The mappers always write agreeing coordinates (SUBMESH: an axis-aligned block; ROW_MAJOR: the
// mesh's row-major enumeration); a label put together by hand with update_tensor_topology need not. Addressing an
// axis requires the whole label to be a row-major block of the mesh (the inline table checked only the axis
// addressed); a whole-mesh partition takes the label's coordinates as its order -- the Tensor overload respells
// that label over the storage's row-major coordinates, the order the factory partitions in.
TEST(MeshPartitionTopologyRules, LabelWhoseCoordinatesDisagreeWithThePartitionFallsBack) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    // A block SHAPE over a ROW of devices: every device has global row 0 and takes chunk 0; rule 1 would have
    // labelled tensor row 1 as holding chunk 1.
    const std::vector<MeshCoordinate> row{
        MeshCoordinate(0, 0), MeshCoordinate(0, 1), MeshCoordinate(0, 2), MeshCoordinate(0, 3)};
    EXPECT_FALSE(partition(TensorTopology(MeshShape(2, 2), {Replicate{}, Replicate{}}, row), mesh, 0).has_value());

    // A full-mesh N-D label with two coordinates swapped across rows is not a row-major block: refused along either
    // axis; over the whole mesh the pure rule keeps the label's order (see above).
    auto swapped = row_major_coords(mesh);
    std::swap(swapped[1], swapped[5]);  // (0,1) <-> (1,1)
    const TensorTopology permuted_nd(mesh, {Replicate{}, Replicate{}}, swapped);
    EXPECT_FALSE(partition(permuted_nd, mesh, 0).has_value());
    EXPECT_FALSE(partition(permuted_nd, mesh, 1).has_value());
    EXPECT_EQ(partition(permuted_nd, mesh, kWholeMesh), TensorTopology(MeshShape(8), {Shard{kDim}}, swapped));

    // Collapsed labels: a line with reversed coordinates, the 2x4 with the swap above.
    const MeshShape line(1, 8);
    auto reversed = row_major_coords(line);
    std::reverse(reversed.begin(), reversed.end());
    EXPECT_FALSE(partition(TensorTopology(MeshShape(8), {Replicate{}}, reversed), line, 1).has_value());
    const TensorTopology permuted_collapsed(MeshShape(8), {Replicate{}}, swapped);
    EXPECT_FALSE(partition(permuted_collapsed, mesh, 0).has_value());
    EXPECT_FALSE(partition(permuted_collapsed, mesh, 1).has_value());
}

// Inputs validation rejects right after the hook, or that carry no placements, leave the input's label in place (the
// hook runs before validation and must not crash first).
TEST(MeshPartitionTopologyRules, OutOfRangeClusterAxisOrEmptyLabelLeavesTheInputLabel) {
    const WarnOnlyScope warn_only;
    const MeshShape mesh(2, 4);
    EXPECT_FALSE(partition(nd_label(mesh, {Replicate{}, Replicate{}}), mesh, 2).has_value());
    const TensorTopology no_placements(mesh, {}, row_major_coords(mesh));
    EXPECT_FALSE(partition(no_placements, mesh, 1).has_value());
    EXPECT_FALSE(partition(no_placements, mesh, kWholeMesh).has_value());
}
