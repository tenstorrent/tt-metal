# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The resharding math, device-free: save on one world with numpy, load onto another, compare with slicing.

Every case saves global device-layout arrays through the save planner into in-memory files, loads them through the
load planner for a destination label and transform, and compares each destination shard with what plain numpy
slicing of the canonical array says it should hold. The cases are the design's worked examples, scaled down, in
both directions.
"""

import numpy as np
import pytest
from dcp_oracle import (
    R,
    S,
    FakeCheckpoint,
    arange,
    assert_shards_equal,
    canonical_of,
    device_global,
    expected_shards,
    world,
)

from ttml.distributed_checkpointing._errors import (
    CoverageError,
    DtypeMismatch,
    MissingKeyError,
    ShapeMismatch,
    UnexpectedKeyError,
)
from ttml.distributed_checkpointing._index import data_file_name
from ttml.distributed_checkpointing._layouts import Identity, Pad, fused_segments
from ttml.distributed_checkpointing._planner import TensorInfo, create_save_plan

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
    src_t = fused_segments(2, QKV, src_tp)
    dst_t = fused_segments(2, QKV, dst_tp)
    plan, shards, expected = _round_trip(src_label, src_t, dst_label, dst_t, (1, 1, 256, 32))
    assert_shards_equal(shards, expected)
    if src_tp == dst_tp and src_label is dst_label:
        assert plan.fast_path_keys == ["w"]


def test_fused_qkv_device_order_differs_per_tp_so_a_naive_load_would_be_wrong():
    canonical = arange((1, 1, 256, 4))
    tp8 = device_global(canonical, fused_segments(2, QKV, 8), (1, 1, 256, 4))
    tp1 = device_global(canonical, Identity(), (1, 1, 256, 4))
    assert not np.array_equal(tp8, tp1), "the test would prove nothing if the layouts agreed"
    np.testing.assert_array_equal(canonical_of(tp8, fused_segments(2, QKV, 8), canonical.shape), canonical)


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
        )
        for i in range(64)
    ]
    plan = create_save_plan(infos, lambda c: data_file_name(0, 0, c))
    assert len(plan.files) == 32 and all(f.items for f in plan.files), "every device file gets some chunks"
    assert plan.bytes_written == 64 * 8 * 32 * 64 * 2
    for fplan in plan.files:
        assert (len(fplan.header) % 4096) == 0
        offsets = [it.data_offset for it in fplan.items]
        assert offsets == sorted(offsets) and offsets[0] == 0
        assert sum(it.nbytes for it in fplan.items) == fplan.data_bytes
    entries = plan.entries()
    assert len(entries["model/blocks/3/w"].chunks) == 8
    assert {c.writer_coord[1] for c in entries["model/blocks/3/w"].chunks} == set(range(8))


def test_mismatches_are_refused_before_any_read(expect_error):
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, TP8_DP4, Identity())})
    with expect_error(ShapeMismatch, "canonical shape"):
        ckpt.load({"w": (SINGLE, Identity(), (1, 1, 64, 64))})
    with expect_error(DtypeMismatch, "checkpoint holds"):
        dests = [TensorInfo("w", "param", "BF16", "TILE", SINGLE, (1, 1, 64, 32), Identity(), ("w",))]
        from ttml.distributed_checkpointing._planner import create_load_plan

        create_load_plan(ckpt.index, dests)
    with expect_error(MissingKeyError, "no entry"):
        ckpt.load({"other": (SINGLE, Identity(), (1, 1, 64, 32))})
    plan, _ = ckpt.load(
        {"other": (SINGLE, Identity(), (1, 1, 64, 32)), "w": (SINGLE, Identity(), (1, 1, 64, 32))}, strict=False
    )
    assert plan.missing == ["other"] and plan.unexpected == []
    with expect_error(UnexpectedKeyError, "no live tensor"):
        ckpt.save({"w": (canonical, TP8_DP4, Identity()), "extra": (canonical, TP8_DP4, Identity())})
        ckpt.load({"w": (SINGLE, Identity(), (1, 1, 64, 32))})


def test_a_missing_chunk_is_a_coverage_error(expect_error):
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (canonical, TP8_DP4, Identity())})
    del ckpt.index.entries["w"].chunks[3]
    with expect_error(CoverageError, "do not tile"):
        ckpt.load({"w": (SINGLE, Identity(), (1, 1, 64, 32))})


def test_live_keys_resolve_tied_names():
    canonical = arange((1, 1, 64, 32))
    ckpt = FakeCheckpoint()
    ckpt.save({"model/Llama/fc/weight": (canonical, SINGLE, Identity())})
    ckpt.index.entries["model/Llama/fc/weight"].live_keys = ("model/Llama/fc/weight", "model/Llama/tok_emb/weight")
    plan, shards = ckpt.load({"model/Llama/tok_emb/weight": (SINGLE, Identity(), canonical.shape)})
    assert "model/Llama/tok_emb/weight" in plan.keys and plan.unexpected == []
    np.testing.assert_array_equal(shards["model/Llama/tok_emb/weight"][()], canonical)
