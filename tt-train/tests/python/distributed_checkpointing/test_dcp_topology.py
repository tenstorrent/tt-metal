# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""From a topology label to chunk boxes, replica groups and owners, for N-D and collapsed labels alike."""


from ttml.distributed_checkpointing._boxes import Box
from ttml.distributed_checkpointing._errors import TopologyError
from ttml.distributed_checkpointing._topology import (
    Label,
    Placement,
    chunk_boxes,
    dist_coords,
    global_shape,
    local_shape,
    owner,
    replica_groups,
)

R = Placement.replicate
S = Placement.shard


def _label(placements, dist_shape, mesh_shape=None):
    return Label(tuple(placements), tuple(dist_shape), tuple(dist_coords(mesh_shape or dist_shape)))


def test_replicate_and_shard_on_a_2d_mesh():
    label = _label([R(), S(2)], (2, 2))
    local = (1, 1, 32, 64)
    assert global_shape(label, local) == (1, 1, 64, 64)
    assert local_shape(label, (1, 1, 64, 64)) == local
    chunks = chunk_boxes(label, local)
    assert [c.mesh_coord for c in chunks] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [c.box.offsets for c in chunks] == [(0, 0, 0, 0), (0, 0, 32, 0), (0, 0, 0, 0), (0, 0, 32, 0)]
    groups = replica_groups(chunks)
    assert list(groups) == [(0,), (1,)]
    assert [c.mesh_coord for c in groups[(1,)]] == [(0, 1), (1, 1)]


def test_collapsed_label_shards_in_row_major_device_order():
    # ShardTensorToMesh(dim=3) on a 2x4 mesh: one axis of 8 walking the mesh row-major.
    label = _label([S(3)], (8,), mesh_shape=(2, 4))
    chunks = chunk_boxes(label, (1, 1, 32, 16))
    assert [c.box.offsets[3] for c in chunks] == [0, 16, 32, 48, 64, 80, 96, 112]
    assert [c.mesh_coord for c in chunks] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    assert len(replica_groups(chunks)) == 8
    assert global_shape(label, (1, 1, 32, 16)) == (1, 1, 32, 128)


def test_two_axes_sharding_one_dim_is_row_major_hierarchical():
    label = _label([S(2), S(2)], (2, 2))
    chunks = chunk_boxes(label, (1, 1, 8, 4))
    assert [c.box.offsets[2] for c in chunks] == [0, 8, 16, 24]
    assert global_shape(label, (1, 1, 8, 4)) == (1, 1, 32, 4)
    assert label.shard_count(2) == 4 and label.shard_count(3) == 1


def test_fsdp_times_tp_has_one_replica_per_box():
    label = _label([S(3), S(2)], (4, 8))
    chunks = chunk_boxes(label, (1, 1, 512, 1024))
    assert global_shape(label, (1, 1, 512, 1024)) == (1, 1, 4096, 4096)
    groups = replica_groups(chunks)
    assert len(groups) == 32 and all(len(g) == 1 for g in groups.values())
    assert chunks[9].box == Box((0, 0, 512, 1024), (1, 1, 512, 1024))  # coord (1, 1): row 1 of fsdp, column 1 of tp


def test_fully_replicated_label_is_one_group():
    label = _label([R(), R()], (2, 4))
    chunks = chunk_boxes(label, (1, 1, 1, 64))
    assert len(replica_groups(chunks)) == 1
    assert label.is_fully_replicated()
    assert global_shape(label, (1, 1, 1, 64)) == (1, 1, 1, 64)


def test_owner_is_deterministic_and_spread():
    label = _label([R(), S(2)], (4, 8))
    groups = replica_groups(chunk_boxes(label, (1, 1, 32, 64)))
    group = groups[(3,)]
    assert owner("model/x", group) == owner("model/x", list(reversed(group)))
    counts = {}
    for i in range(400):
        c = owner(f"model/blocks/{i}/weight", group).mesh_coord
        counts[c] = counts.get(c, 0) + 1
    assert set(counts) == {(0, 3), (1, 3), (2, 3), (3, 3)}
    assert min(counts.values()) >= 60, counts


def test_negative_dims_are_normalized_and_bad_labels_rejected(expect_error):
    label = Label.normalized([R(), S(-1)], (1, 2), [(0, 0), (0, 1)], rank=4)
    assert label.placements[1] == S(3)
    with expect_error(TopologyError, "out of range"):
        Label.normalized([S(4)], (2,), [(0,), (1,)], rank=4)
    with expect_error(TopologyError, "placements"):
        Label([R()], (1, 2), [(0, 0), (0, 1)])
    with expect_error(TopologyError, "mesh coordinates"):
        Label([R(), S(2)], (1, 2), [(0, 0)])
    with expect_error(TopologyError, "twice"):
        Label([R(), S(2)], (1, 2), [(0, 0), (0, 0)])
    with expect_error(TopologyError, "split evenly"):
        local_shape(_label([S(2)], (3,)), (1, 1, 64, 64))


def test_label_round_trips_through_json():
    label = _label([R(), S(2)], (2, 4))
    assert Label.from_json(label.to_json()) == label
