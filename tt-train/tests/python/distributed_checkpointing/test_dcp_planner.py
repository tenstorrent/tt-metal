# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The resharding math, device-free: save on one world with numpy, load onto another, compare with slicing.

Every case saves global device-layout arrays through the save planner into in-memory files, loads them through the
load planner for a destination label and transform, and compares each destination shard with what plain numpy
slicing of the canonical array says it should hold. The cases reshard between TP, DP and FSDP worlds, collapsed and
N-D labels, fused and padded weights, each in both directions, and check what the planner refuses.
"""

import zlib
from dataclasses import dataclass

import numpy as np
import pytest
from dcp_oracle import (
    R,
    S,
    FakeCheckpoint,
    arange,
    assert_shards_equal,
    device_global,
    expected_shards,
    info,
    one_process_owner,
    per_shard_fused,
    tp_split_of,
    world,
)

from ttml.distributed_checkpointing._boxes import Box
from ttml.distributed_checkpointing._errors import (
    CheckpointError,
    CoverageError,
    DtypeMismatchError,
    RoleMismatchError,
    ShapeMismatchError,
    TopologyError,
)
from ttml.distributed_checkpointing._index import ChunkRecord, data_file_name
from ttml.distributed_checkpointing._layouts import Compose, Identity, Pad, fused_segments
from ttml.distributed_checkpointing._planner import TensorInfo, _ChunkLookup, create_load_plan, create_save_plan
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords

TP8_DP4 = world([R(), S(2)], (4, 8))
TP4_DP8 = world([R(), S(2)], (8, 4))
FSDP32_ROWS = world([S(2), R()], (32, 1))
FSDP32_COLS = world([R(), S(2)], (1, 32))
SINGLE = world([R()], (1,))
FSDP4_TP8 = world([S(3), S(2)], (4, 8))
TP8 = world([R(), S(2)], (1, 8))
COLLAPSED_TP8_ON_2X4 = world([S(2)], (8,), mesh_shape=(2, 4))

QKV = (128, 64, 64)  # a GQA fused projection: 128 Q rows, 64 K rows, 64 V rows


def _round_trip(src_label, src_transform, dst_label, dst_transform, global_shape, canonical_shape=None, pad_value=0.0):
    """Save a canonical arange under the source layout, load under the destination layout, return (plan, shards,
    expected)."""
    canonical = arange(canonical_shape or global_shape)
    src_global_shape = tuple(global_shape)
    src = device_global(canonical, src_transform, src_global_shape, pad_value)
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (src, src_label, src_transform)})
    dst_global_shape = _dest_global_shape(dst_transform, canonical.shape, src_global_shape)
    plan, shards = ckpt.load({"w": (dst_label, dst_transform, dst_global_shape)})
    same_layout = src_transform == dst_transform and src_global_shape == dst_global_shape
    expected_global = src if same_layout else device_global(canonical, dst_transform, dst_global_shape, 0.0)
    return plan, shards["w"], expected_shards(expected_global, dst_label)


def _dest_global_shape(dst_transform, canonical_shape, src_global_shape):
    if isinstance(dst_transform, Pad):
        shape = list(canonical_shape)
        shape[dst_transform.dim] = dst_transform.padded
        return tuple(shape)
    return tuple(src_global_shape) if not isinstance(dst_transform, Identity) else tuple(canonical_shape)


@pytest.mark.parametrize(
    "dst_label",
    [FSDP32_ROWS, FSDP32_COLS, TP4_DP8, SINGLE, TP8_DP4, FSDP4_TP8],
    ids=["fsdp32_rows", "fsdp32_cols", "tp4_dp8", "single", "same", "fsdp4_tp8"],
)
def test_identity_weight_from_tp8_dp4(dst_label):
    plan, shards, expected = _round_trip(TP8_DP4, Identity(), dst_label, Identity(), (1, 1, 256, 128))
    assert_shards_equal(shards, expected)
    assert plan.fast_path_keys == (["w"] if dst_label is TP8_DP4 else [])


@pytest.mark.parametrize(
    "src_label", [FSDP32_ROWS, TP4_DP8, SINGLE, FSDP4_TP8], ids=["fsdp32", "tp4_dp8", "single", "fsdp4_tp8"]
)
def test_identity_weight_back_to_tp8_dp4(src_label):
    _, shards, expected = _round_trip(src_label, Identity(), TP8_DP4, Identity(), (1, 1, 256, 128))
    assert_shards_equal(shards, expected)


@pytest.mark.parametrize(
    "src_label, src_tp, dst_label, dst_tp",
    [
        (TP8_DP4, 8, FSDP32_ROWS, 1),
        (TP8_DP4, 8, TP4_DP8, 4),
        (TP8_DP4, 8, SINGLE, 1),
        (TP4_DP8, 4, TP8_DP4, 8),
        (FSDP32_ROWS, 1, TP8_DP4, 8),
        (TP8_DP4, 8, TP8_DP4, 8),
    ],
    ids=["tp8->fsdp32", "tp8->tp4", "tp8->single", "tp4->tp8", "fsdp32->tp8", "tp8->tp8"],
)
def test_fused_qkv_reorders_segments_across_tp_sizes(src_label, src_tp, dst_label, dst_tp):
    # The device arrays are built by slicing (per_shard_fused), independently of fused_segments, so a transform that
    # put the segments in the wrong order would load rows the expected arrays do not hold.
    canonical = arange((1, 1, 256, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (per_shard_fused(canonical, QKV, src_tp, dim=2), src_label, fused_segments(2, QKV, src_tp))})
    plan, shards = ckpt.load({"w": (dst_label, fused_segments(2, QKV, dst_tp), (1, 1, 256, 32))})
    assert_shards_equal(shards["w"], expected_shards(per_shard_fused(canonical, QKV, dst_tp, dim=2), dst_label))
    if src_tp == dst_tp and src_label is dst_label:
        assert plan.fast_path_keys == ["w"]


def test_fused_qkv_device_order_differs_per_tp_so_a_naive_load_would_be_wrong():
    canonical = arange((1, 1, 256, 4))
    tp8_rows = per_shard_fused(canonical, QKV, 8, dim=2)
    assert not np.array_equal(tp8_rows, canonical), "the test would prove nothing if the layouts agreed"
    # Saved without the declaration, the TP8 rows load onto one device in their TP8 order: Q, K and V interleaved.
    naive = FakeCheckpoint()
    naive.save({"w": (tp8_rows, TP8, Identity())})
    _, shards = naive.load({"w": (SINGLE, Identity(), (1, 1, 256, 4))})
    assert not np.array_equal(shards["w"][()], canonical)
    # With it, the same rows load in canonical order.
    declared = FakeCheckpoint()
    declared.save({"w": (tp8_rows, TP8, fused_segments(2, QKV, 8))})
    _, shards = declared.load({"w": (SINGLE, Identity(), (1, 1, 256, 4))})
    np.testing.assert_array_equal(shards["w"][()], canonical)


def test_destination_boxes_follow_the_label_literally():
    # Written out by hand rather than derived from chunk_boxes: a collapsed label numbers its shards in row-major
    # device order, and two axes sharding one dim nest outermost first.
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (arange((1, 1, 256, 8)), SINGLE, Identity())})
    collapsed = world([S(2)], (8,), mesh_shape=(2, 4))
    plan, _ = ckpt.load({"w": (collapsed, Identity(), (1, 1, 256, 8))})
    assert plan.keys["w"].boxes == {(j,): Box((0, 0, 32 * j, 0), (1, 1, 32, 8)) for j in range(8)}
    nested = world([S(2), S(2)], (2, 4))
    plan, _ = ckpt.load({"w": (nested, Identity(), (1, 1, 256, 8))})
    assert plan.keys["w"].boxes == {
        (a, b): Box((0, 0, 32 * (4 * a + b), 0), (1, 1, 32, 8)) for a in range(2) for b in range(4)
    }


def test_padded_vocab_across_tp_sizes_fills_new_padding_and_keeps_old_padding_on_the_same_layout():
    # vocab 130 pads to 192 at tp=2 and to 160 at tp=1; saved padding rows hold -1 so a copy is distinguishable
    # from a fill.
    src_label, dst_label = world([R(), S(2)], (1, 2)), world([R()], (1,))
    plan, shards, expected = _round_trip(
        src_label,
        Pad(2, 130, 192),
        dst_label,
        Pad(2, 130, 160),
        (1, 1, 192, 16),
        canonical_shape=(1, 1, 130, 16),
        pad_value=-1.0,
    )
    assert_shards_equal(shards, expected)
    (shard,) = shards.values()
    assert (shard[:, :, 130:, :] == 0).all(), "new padding rows are filled, not copied"
    assert plan.keys["w"].fills and not plan.fast_path_keys

    # Same padded extent on a different DP size: padding rows copy 1:1 and the whole-chunk fast path applies.
    plan, shards, expected = _round_trip(
        src_label,
        Pad(2, 130, 192),
        world([R(), S(2)], (2, 2)),
        Pad(2, 130, 192),
        (1, 1, 192, 16),
        canonical_shape=(1, 1, 130, 16),
        pad_value=-1.0,
    )
    assert_shards_equal(shards, expected)
    assert all((s[:, :, 130:, :] == -1).all() for s in shards.values())
    assert plan.fast_path_keys == ["w"] and not plan.keys["w"].fills


def test_collapsed_source_label_loads_onto_an_nd_destination_and_back():
    _, shards, expected = _round_trip(
        COLLAPSED_TP8_ON_2X4, Identity(), world([R(), S(2)], (2, 4)), Identity(), (1, 1, 256, 64)
    )
    assert_shards_equal(shards, expected)
    _, shards, expected = _round_trip(
        world([R(), S(2)], (2, 4)), Identity(), COLLAPSED_TP8_ON_2X4, Identity(), (1, 1, 256, 64)
    )
    assert_shards_equal(shards, expected)


def test_fsdp4_tp8_to_tp8_places_four_chunks_side_by_side_and_back():
    _, shards, expected = _round_trip(FSDP4_TP8, Identity(), TP8, Identity(), (1, 1, 128, 256))
    assert_shards_equal(shards, expected)
    _, shards, expected = _round_trip(TP8, Identity(), FSDP4_TP8, Identity(), (1, 1, 128, 256))
    assert_shards_equal(shards, expected)


def test_same_world_reads_every_shard_whole():
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, TP8_DP4, Identity())})
    plan, _ = ckpt.load({"w": (TP8_DP4, Identity(), canonical.shape)})
    kplan = plan.keys["w"]
    assert kplan.fast_path and len(kplan.reads) == 8
    for read in kplan.reads:
        chunk = kplan.entry.chunks[read.chunk]
        assert read.src.offsets == (0, 0, 0, 0) and read.src.sizes == chunk.box.sizes


def test_save_plan_spreads_owners_over_every_file_and_fixes_offsets():
    infos = [
        TensorInfo(
            f"model/blocks/{i}/w",
            "param",
            "BF16",
            "TILE",
            TP8_DP4,
            (1, 1, 32, 64),
            Identity(),
            (f"model/blocks/{i}/w",),
            tp_split=tp_split_of(TP8_DP4, 4),
        )
        for i in range(64)
    ]
    plan = create_save_plan(infos, lambda c: data_file_name(0, 0, c), one_process_owner)
    assert len(plan.files) == 32 and all(f.items for f in plan.files), "every device file gets some chunks"
    assert plan.bytes_written == 64 * 8 * 32 * 64 * 2
    for fplan in plan.files:
        assert (len(fplan.header) % 4096) == 0
        offsets = [it.data_offset for it in fplan.items]
        assert offsets == sorted(offsets) and offsets[0] == 0
        assert sum(it.nbytes for it in fplan.items) == fplan.data_bytes
    entries = plan.entries({(item.key, f.index): 0 for f in plan.files for item in f.items})
    assert len(entries["model/blocks/3/w"].chunks) == 8
    assert {plan.files[c.file].coord[1] for c in entries["model/blocks/3/w"].chunks} == set(range(8))


def test_a_save_plan_writes_only_the_chunks_owner_of_gives_this_process():
    """Where another process writes a replica group, owner_of returns None for it: the group gets no write here, a key
    this process writes nothing of gets no entry in its part of the index, and only the coordinates that write get a
    file."""
    infos = [info(key, TP8_DP4, (1, 1, 64, 32), Identity()) for key in ("w", "v")]

    def owner_of(key, group):
        own = one_process_owner(key, group)
        return own if key == "w" and own.mesh_coord[1] < 4 else None

    plan = create_save_plan(infos, lambda c: data_file_name(0, 0, c), owner_of)
    written = [item for fplan in plan.files for item in fplan.items]
    assert {item.key for item in written} == {"w"} and len(written) == 4
    assert sorted(f.coord for f in plan.files) == sorted(item.chunk.mesh_coord for item in written)
    entries = plan.entries({(item.key, item.file): 0 for item in written})
    assert list(entries) == ["w"] and len(entries["w"].chunks) == 4


def test_two_live_tensors_under_one_key_are_refused(expect_error):
    # A dict by key would keep one of them and silently not save or not load the other.
    info = TensorInfo(
        "w", "param", "F32", "TILE", SINGLE, (1, 1, 64, 32), Identity(), ("w",), tp_split=tp_split_of(SINGLE, 4)
    )
    with expect_error(CheckpointError, "w: two live tensors have this key; only one of them would be saved"):
        create_save_plan([info, info], lambda c: data_file_name(0, 0, c), one_process_owner)
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (arange((1, 1, 64, 32)), SINGLE, Identity())})
    with expect_error(CheckpointError, "only one of them would be loaded"):
        create_load_plan(ckpt.index, [info, info])


def test_mismatches_are_refused_before_any_read(expect_error):
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, TP8_DP4, Identity())})
    with expect_error(ShapeMismatchError, "canonical shape"):
        ckpt.load({"w": (SINGLE, Identity(), (1, 1, 64, 64))})
    with expect_error(DtypeMismatchError, "checkpoint holds"):
        dests = [
            TensorInfo(
                "w",
                "param",
                "BF16",
                "TILE",
                SINGLE,
                (1, 1, 64, 32),
                Identity(),
                ("w",),
                tp_split=tp_split_of(SINGLE, 4),
            )
        ]
        from ttml.distributed_checkpointing._planner import create_load_plan

        create_load_plan(ckpt.index, dests)
    # The plan lists what is missing and unexpected; load judges them under strict.
    plan, _ = ckpt.load({"other": (SINGLE, Identity(), (1, 1, 64, 32))})
    assert plan.missing == ["other"] and plan.unexpected == ["w"]
    plan, _ = ckpt.load({"other": (SINGLE, Identity(), (1, 1, 64, 32)), "w": (SINGLE, Identity(), (1, 1, 64, 32))})
    assert plan.missing == ["other"] and plan.unexpected == []
    ckpt.save({"w": (canonical, TP8_DP4, Identity()), "extra": (canonical, TP8_DP4, Identity())})
    plan, _ = ckpt.load({"w": (SINGLE, Identity(), (1, 1, 64, 32))})
    assert plan.missing == [] and plan.unexpected == ["extra"]


def test_a_rank_plans_only_the_shards_of_the_coordinates_it_holds():
    """With local_coords (a multi-rank load) only the replica groups with a coordinate this rank holds are planned;
    the ranks' parts together are the plan of one process."""
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, TP8_DP4, Identity())})
    dest = info("w", FSDP32_ROWS, canonical.shape, Identity())
    whole = create_load_plan(ckpt.index, [dest]).keys["w"]
    halves = [set(FSDP32_ROWS.mesh_coords[:16]), set(FSDP32_ROWS.mesh_coords[16:])]
    parts = [create_load_plan(ckpt.index, [dest], local_coords=coords).keys["w"] for coords in halves]
    assert [len(part.boxes) for part in parts] == [16, 16] and {**parts[0].boxes, **parts[1].boxes} == whole.boxes

    def by_group(reads):
        return sorted(reads, key=lambda r: (r.group, r.chunk))

    assert by_group(parts[0].reads + parts[1].reads) == by_group(whole.reads)
    # A replicated group is planned by every rank holding one of its coordinates: DP row 0 holds one of each.
    row0 = {c for c in TP8_DP4.mesh_coords if c[0] == 0}
    replicated = create_load_plan(ckpt.index, [info("w", TP8_DP4, canonical.shape, Identity())], local_coords=row0)
    assert len(replicated.keys["w"].boxes) == 8


def test_a_missing_chunk_is_a_coverage_error(expect_error):
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, TP8_DP4, Identity())})
    del ckpt.index.entries["w"].chunks[3]
    with expect_error(CoverageError, "w: the checkpoint's 7 chunks do not tile its global shape"):
        ckpt.load({"w": (SINGLE, Identity(), (1, 1, 64, 32))})


@dataclass(frozen=True)
class _DroppingIdentity(Identity):
    """A broken transform that maps only the first half of every box's rows: what the per-shard check guards against,
    since the transforms the planner ships all partition the boxes they map."""

    def to_canonical(self, box):
        half = box.with_dim(2, box.offsets[2], box.sizes[2] // 2)
        return [(half, half)]


def test_a_destination_shard_the_reads_leave_short_is_a_coverage_error(expect_error):
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (arange((1, 1, 64, 32)), SINGLE, Identity())})
    with expect_error(CoverageError, r"w: the saved chunks do not tile destination shard .*1024 of 2048 elements"):
        ckpt.load({"w": (SINGLE, _DroppingIdentity(), (1, 1, 64, 32))})


def test_chunks_that_are_not_one_grid_are_found_by_scanning():
    # Two writers that split the rows unevenly (3 + 7) are not cells of one grid, so the planner scans the chunks.
    canonical = arange((1, 1, 10, 4))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, SINGLE, Identity())})
    row = 4 * canonical.itemsize
    raw = canonical.tobytes()
    ckpt.files = {0: bytearray(raw[: 3 * row]), 1: bytearray(raw[3 * row :])}
    entry = ckpt.index.entries["w"]
    entry.chunks = [
        ChunkRecord(Box((0, 0, 0, 0), (1, 1, 3, 4)), 0, 0, 3 * row, zlib.crc32(raw[: 3 * row])),
        ChunkRecord(Box((0, 0, 3, 0), (1, 1, 7, 4)), 1, 0, 7 * row, zlib.crc32(raw[3 * row :])),
    ]
    assert _ChunkLookup(entry).by_cell is None, "the uneven chunks take the scan path"
    tp2 = world([R(), S(2)], (1, 2))
    _, shards = ckpt.load({"w": (tp2, Identity(), (1, 1, 10, 4))})
    assert_shards_equal(shards["w"], expected_shards(canonical, tp2))


def test_roles_scope_the_unexpected_keys_of_a_load():
    ckpt = FakeCheckpoint()
    ckpt.save(
        {"w": (arange((1, 1, 64, 32)), SINGLE, Identity()), "opt/w": (arange((1, 1, 64, 32)), SINGLE, Identity())}
    )
    ckpt.index.entries["opt/w"].role = "optimizer_state"
    model_only = [
        TensorInfo(
            "w", "param", "F32", "TILE", SINGLE, (1, 1, 64, 32), Identity(), ("w",), tp_split=tp_split_of(SINGLE, 4)
        )
    ]
    assert create_load_plan(ckpt.index, model_only).unexpected == ["opt/w"]
    plan = create_load_plan(ckpt.index, model_only, roles=["param"])
    assert plan.unexpected == [] and list(plan.keys) == ["w"], "optimizer state is out of a model-only load's scope"


def test_a_renamed_tensor_loads_through_its_previous_live_name():
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"model/old": (canonical, SINGLE, Identity())})
    renamed = TensorInfo(
        "model/new",
        "param",
        "F32",
        "TILE",
        SINGLE,
        (1, 1, 64, 32),
        Identity(),
        ("model/new", "model/old"),
        tp_split=tp_split_of(SINGLE, 4),
    )
    plan = create_load_plan(ckpt.index, [renamed])
    assert plan.keys["model/new"].entry.key == "model/old" and plan.missing == []


def test_padding_through_a_compose_is_filled_with_zeros():
    # Fusion on dim 2 and padding on dim 3 compose; a load that grows the padded extent fills the new padding.
    tp2 = world([R(), S(2)], (1, 2))
    src_t = Compose((Pad(3, 24, 32), fused_segments(2, (8, 8), 2)))
    dst_t = Compose((Pad(3, 24, 40), fused_segments(2, (8, 8), 2)))
    canonical = arange((1, 1, 16, 24))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (device_global(canonical, src_t, (1, 1, 16, 32), pad_value=-1.0), tp2, src_t)})
    plan, shards = ckpt.load({"w": (tp2, dst_t, (1, 1, 16, 40))})
    assert plan.keys["w"].fills and all(f.fill == "zeros" for f in plan.keys["w"].fills)
    assert_shards_equal(shards["w"], expected_shards(device_global(canonical, dst_t, (1, 1, 16, 40), 0.0), tp2))


def test_live_keys_resolve_tied_names():
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"model/Llama/fc/weight": (canonical, SINGLE, Identity())})
    ckpt.index.entries["model/Llama/fc/weight"].live_keys = ("model/Llama/fc/weight", "model/Llama/tok_emb/weight")
    plan, shards = ckpt.load({"model/Llama/tok_emb/weight": (SINGLE, Identity(), canonical.shape)})
    assert "model/Llama/tok_emb/weight" in plan.keys and plan.unexpected == []
    np.testing.assert_array_equal(shards["model/Llama/tok_emb/weight"][()], canonical)


def test_a_label_sharding_a_dim_the_tensor_lacks_is_refused_at_save(expect_error):
    # A shard dim outside the tensor's rank gives every replica group the same box: without the check the save
    # writes duplicate chunks and every later load fails on coverage.
    for dim in (-1, 4):
        label = Label((Placement.replicate(), Placement.shard(dim)), (1, 2), tuple(dist_coords((1, 2))))
        info = TensorInfo(
            "model/w",
            "param",
            "BF16",
            "TILE",
            label,
            (1, 1, 4, 4),
            Identity(),
            ("model/w",),
            tp_split=tp_split_of(label, 4),
        )
        with expect_error(TopologyError, "model/w: distribution axis 1 shards dim"):
            create_save_plan([info], lambda c: data_file_name(0, 0, c), one_process_owner)


def test_saved_chunks_that_overlap_or_leave_a_gap_are_refused_at_load(expect_error):
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, TP8_DP4, Identity())})
    chunks = ckpt.index.entries["w"].chunks
    chunks[3] = chunks[2]  # one box twice, one missing: the volumes still add up
    for dest in (SINGLE, TP8_DP4):
        with expect_error(CoverageError, "w: the checkpoint's 8 chunks do not tile"):
            ckpt.load({"w": (dest, Identity(), canonical.shape)})


def test_a_live_tensor_matching_an_entry_of_another_role_is_refused(expect_error):
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, SINGLE, Identity())})
    state = TensorInfo(
        "w",
        "optimizer_state",
        "F32",
        "TILE",
        SINGLE,
        (1, 1, 64, 32),
        Identity(),
        ("w",),
        tp_split=tp_split_of(SINGLE, 4),
    )
    with expect_error(RoleMismatchError, "w: resolves to checkpoint entry 'w', a 'param'"):
        create_load_plan(ckpt.index, [state])


def test_a_reshard_reads_only_the_chunks_it_needs():
    # Rows to columns: every destination shard needs a piece of every saved chunk; columns to columns: one each.
    canonical = arange((1, 1, 64, 64))
    for dst_label, reads_per_shard in ((world([R(), S(3)], (1, 8)), 8), (world([R(), S(2)], (1, 8)), 1)):
        ckpt = FakeCheckpoint()
        ckpt.save({"w": (canonical, world([R(), S(2)], (1, 8)), Identity())})
        plan, shards = ckpt.load({"w": (dst_label, Identity(), canonical.shape)})
        assert len(plan.keys["w"].reads) == 8 * reads_per_shard
        assert_shards_equal(shards["w"], expected_shards(canonical, dst_label))


def test_save_planning_is_cached_per_label_and_shape(expect_error):
    from ttml.distributed_checkpointing._planner import _replica_layout

    infos = [
        TensorInfo(
            f"model/blocks/{i}/w",
            "param",
            "BF16",
            "TILE",
            TP8_DP4,
            (1, 1, 32, 64),
            Identity(),
            (),
            tp_split=tp_split_of(TP8_DP4, 4),
        )
        for i in range(16)
    ]
    _replica_layout.cache_clear()
    first = create_save_plan(infos, lambda c: data_file_name(0, 0, c), one_process_owner)
    second = create_save_plan(infos, lambda c: data_file_name(0, 0, c), one_process_owner)
    stats = _replica_layout.cache_info()
    assert stats.misses == 1 and stats.hits == 31, "one grouping for 16 keys of one label, reused by the next save"
    assert [f.items[0].key for f in first.files] == [f.items[0].key for f in second.files]
    groups, problem = _replica_layout(TP8_DP4, (1, 1, 32, 64))
    assert problem is None
    with expect_error(TypeError, "does not support item assignment"):
        groups[(0,)] = ()  # the cached groups are shared by every key of the label, so they are read-only
    # A cached failure still names the key that hit it.
    bad = Label((Placement.replicate(), Placement.shard(5)), (1, 2), tuple(dist_coords((1, 2))))
    before = _replica_layout.cache_info()
    for key in ("model/a", "model/b"):
        info = TensorInfo(
            key, "param", "BF16", "TILE", bad, (1, 1, 4, 4), Identity(), (key,), tp_split=tp_split_of(bad, 4)
        )
        with expect_error(TopologyError, f"{key}: distribution axis 1 shards dim 5"):
            create_save_plan([info], lambda c: data_file_name(0, 0, c), one_process_owner)
    after = _replica_layout.cache_info()
    assert (
        after.misses - before.misses == 1 and after.hits - before.hits == 1
    ), "model/b reuses model/a's cached failure"
