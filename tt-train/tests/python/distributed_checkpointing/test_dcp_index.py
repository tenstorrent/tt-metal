# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""index.json round trip, directory naming, completeness and the staging directory."""

import json
import os


from ttml.distributed_checkpointing._boxes import Box
from ttml.distributed_checkpointing._errors import CheckpointError, IncompleteCheckpoint, VersionError
from ttml.distributed_checkpointing._index import (
    ChunkRecord,
    Entry,
    FileRecord,
    Index,
    begin_staging,
    data_file_name,
    finalize_staging,
    find_latest,
    is_complete,
    parse_dir_name,
    read_header,
    read_index,
    step_dir_name,
    write_header,
    write_index,
)
from ttml.distributed_checkpointing._layouts import Pad, fused_segments
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords


def _index() -> Index:
    label = Label((Placement.replicate(), Placement.shard(2)), (2, 2), tuple(dist_coords((2, 2))))
    chunks = [
        ChunkRecord(
            Box((0, 0, 0, 0), (1, 1, 96, 16)), file=0, data_offset=0, nbytes=3072, writer_coord=(0, 0), crc32=0x3F0A9C11
        ),
        ChunkRecord(
            Box((0, 0, 96, 0), (1, 1, 96, 16)),
            file=3,
            data_offset=4096,
            nbytes=3072,
            writer_coord=(1, 1),
            crc32=0x77C2E0D9,
        ),
    ]
    entries = {
        "model/Llama/fc/weight": Entry(
            "model/Llama/fc/weight",
            "param",
            "BF16",
            "TILE",
            (1, 1, 192, 16),
            (1, 1, 130, 16),
            Pad(2, 130, 192),
            label,
            ("model/Llama/fc/weight", "model/Llama/tok_emb/weight"),
            chunks,
        ),
        "optimizer/exp_avg/model/Llama/fc/weight": Entry(
            "optimizer/exp_avg/model/Llama/fc/weight",
            "optimizer_state",
            "BF16",
            "TILE",
            (1, 1, 192, 16),
            (1, 1, 130, 16),
            Pad(2, 130, 192),
            label,
            ("optimizer/exp_avg/model/Llama/fc/weight",),
            [chunks[0]],
            param_ref="model/Llama/fc/weight",
            state_path=("exp_avg",),
        ),
        "model/Llama/blocks/0/attention/qkv_linear/weight": Entry(
            "model/Llama/blocks/0/attention/qkv_linear/weight",
            "param",
            "F32",
            "TILE",
            (1, 1, 64, 16),
            (1, 1, 64, 16),
            fused_segments(2, (32, 16, 16), 2),
            label,
            ("model/Llama/blocks/0/attention/qkv_linear/weight",),
            [
                ChunkRecord(Box((0, 0, 0, 0), (1, 1, 32, 16)), 0, 3072, 2048, (0, 0)),
                ChunkRecord(Box((0, 0, 32, 0), (1, 1, 32, 16)), 1, 0, 2048, (0, 1)),
            ],
        ),
    }
    files = [FileRecord(data_file_name(0, 0, c), 8192, 0, 0, c) for c in dist_coords((2, 2))]
    world = {"group_size": 1, "ranks": [{"world_rank": 0, "mesh_id": 0, "shape": [2, 2], "axis_names": ["dp", "tp"]}]}
    return Index(
        "step_0000010",
        world,
        files,
        entries,
        scalars={"optimizer/steps": 10, "optimizer/lr": 3e-4, "optimizer/amsgrad": False, "note": "x"},
    )


def test_index_round_trips_through_json_with_columnar_chunks():
    index = _index()
    encoded = index.to_json()
    table = encoded["state"]["model/Llama/fc/weight"]["chunks"]
    assert (
        table["n"] == 2 and table["sizes"] == [[1, 1, 96, 16]] and table["nbytes"] == [3072]
    ), "equal columns broadcast"
    assert table["crc32"] == ["3f0a9c11", "77c2e0d9"]
    assert "crc32" not in encoded["state"]["model/Llama/blocks/0/attention/qkv_linear/weight"]["chunks"]

    decoded = Index.from_json(json.loads(json.dumps(encoded)))

    assert decoded.entries == index.entries
    assert decoded.files == index.files
    assert decoded.scalars == index.scalars and type(decoded.scalars["optimizer/amsgrad"]) is bool
    assert decoded.live_key_map()["model/Llama/tok_emb/weight"] == "model/Llama/fc/weight"


def test_index_rejects_newer_versions_and_foreign_formats(expect_error):
    encoded = _index().to_json()
    encoded["version"] = 2
    with expect_error(VersionError, "newer"):
        Index.from_json(encoded)
    encoded["format"] = "something-else"
    with expect_error(VersionError, "not a"):
        Index.from_json(encoded)


def test_naming():
    assert step_dir_name(1000) == "step_0001000"
    assert step_dir_name(7, prefix="llama8b") == "llama8b_step_0000007"
    assert parse_dir_name("step_0001000") == ("", 1000)
    assert parse_dir_name("llama8b_step_12") == ("llama8b", 12)
    assert parse_dir_name("/runs/x/final") is None
    assert data_file_name(0, 0, (1, 2)) == "data/w000_m0_d1-2.safetensors"
    assert data_file_name(3, 1, (0,)) == "data/w003_m1_d0.safetensors"


def test_find_latest_orders_by_parsed_step_and_skips_partials(tmp_path):
    for name in [
        "step_0000005",
        "step_0000030",
        "step_0000100.partial-123",
        "other_step_0000200",
        "step_0000050.replaced-9",
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "index.json").write_text("{}")
    (tmp_path / "step_0000040").mkdir()  # no index: incomplete
    assert find_latest(str(tmp_path)) == str(tmp_path / "step_0000030")
    assert find_latest(str(tmp_path), include_replaced=True) == str(tmp_path / "step_0000050.replaced-9")
    assert find_latest(str(tmp_path), prefix="other") == str(tmp_path / "other_step_0000200")
    assert find_latest(str(tmp_path / "missing")) is None
    assert is_complete(str(tmp_path / "step_0000030"))
    assert not is_complete(str(tmp_path / "step_0000100.partial-123"))
    assert not is_complete(str(tmp_path / "step_0000040"))


def test_staging_becomes_complete_only_at_finalize(tmp_path, expect_error):
    final = str(tmp_path / "step_0000010")
    staging, nonce = begin_staging(final)
    assert ".partial-" in staging and os.path.isdir(os.path.join(staging, "data"))
    write_header(staging, {"step": 10})
    write_index(staging, _index())
    assert not is_complete(staging) and find_latest(str(tmp_path)) is None

    finalize_staging(staging, final)

    assert is_complete(final) and find_latest(str(tmp_path)) == final
    assert read_header(final) == {"step": 10}
    assert read_index(final).checkpoint_id == "step_0000010"
    assert read_index(final).created, "written at finalize"
    with expect_error(IncompleteCheckpoint, "no index.json"):
        read_index(str(tmp_path / "nowhere"))

    staging2, _ = begin_staging(final)
    write_index(staging2, _index())
    with expect_error(CheckpointError, "already exists"):
        finalize_staging(staging2, final)
    finalize_staging(staging2, final, overwrite=True)
    assert is_complete(final) and not [n for n in os.listdir(tmp_path) if ".replaced-" in n]
