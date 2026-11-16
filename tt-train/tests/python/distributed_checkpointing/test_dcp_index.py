# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""index.json round trip, directory naming, completeness and the staging directory."""

import json
import os
import shutil

import pytest


from ttml.distributed_checkpointing import _index as index_module
from ttml.distributed_checkpointing._boxes import Box
from ttml.distributed_checkpointing._errors import (
    CheckpointError,
    IncompleteCheckpointError,
    TopologyError,
    VersionError,
)
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
    read_checked_header,
    read_header_json,
    read_index,
    read_user_header,
    step_dir_name,
    write_header,
    write_index,
)
from ttml.distributed_checkpointing._layouts import Pad, fused_segments
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords


def _index() -> Index:
    label = Label((Placement.replicate(), Placement.shard(2)), (2, 2), tuple(dist_coords((2, 2))))
    chunks = [
        ChunkRecord(Box((0, 0, 0, 0), (1, 1, 96, 16)), file=0, data_offset=0, nbytes=3072, crc32=0x3F0A9C11),
        ChunkRecord(
            Box((0, 0, 96, 0), (1, 1, 96, 16)),
            file=3,
            data_offset=4096,
            nbytes=3072,
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
            tp_split=(1, 1, 2, 1),
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
            tp_split=(1, 1, 2, 1),
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
                ChunkRecord(Box((0, 0, 0, 0), (1, 1, 32, 16)), 0, 3072, 2048, 0x0BADC0DE),
                ChunkRecord(Box((0, 0, 32, 0), (1, 1, 32, 16)), 1, 0, 2048, 0x00000001),
            ],
            tp_split=(1, 1, 2, 1),
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
    qkv_table = encoded["state"]["model/Llama/blocks/0/attention/qkv_linear/weight"]["chunks"]
    assert qkv_table["crc32"] == ["0badc0de", "00000001"], "eight hex digits, leading zeros kept"

    decoded = Index.from_json(json.loads(json.dumps(encoded)))

    assert decoded.entries == index.entries
    assert decoded.files == index.files
    assert decoded.scalars == index.scalars and type(decoded.scalars["optimizer/amsgrad"]) is bool
    assert decoded.live_key_map()["model/Llama/tok_emb/weight"] == "model/Llama/fc/weight"
    assert all(entry["tp_split"] == [1, 1, 2, 1] for entry in encoded["state"].values()), "every entry records it"


def test_a_malformed_tp_split_is_refused(expect_error):
    encoded = json.loads(json.dumps(_index().to_json()))
    encoded["state"]["model/Llama/blocks/0/attention/qkv_linear/weight"]["tp_split"] = [1, 2, 1]
    with expect_error(CheckpointError, r"qkv_linear/weight': malformed tp_split \[1, 2, 1\]"):
        Index.from_json(encoded)


def test_index_rejects_newer_versions_and_foreign_formats(expect_error):
    encoded = _index().to_json()
    encoded["version"] = 2
    with expect_error(VersionError, "newer"):
        Index.from_json(encoded)
    encoded["format"] = "something-else"
    with expect_error(VersionError, "not a"):
        Index.from_json(encoded)


def test_a_bad_index_names_its_file(tmp_path, expect_error):
    def written(change) -> str:
        ckpt = str(tmp_path / f"step_{len(os.listdir(tmp_path)):07d}")
        os.makedirs(ckpt)
        encoded = _index().to_json()
        change(encoded)
        with open(os.path.join(ckpt, "index.json"), "w", encoding="utf-8") as f:
            json.dump(encoded, f)
        return ckpt

    def placement(value):
        return lambda encoded: encoded["state"]["model/Llama/fc/weight"]["label"]["placements"].__setitem__(1, value)

    newer = written(lambda encoded: encoded.update(version=2))
    with expect_error(VersionError, f"{newer}/index.json: index version 2 is newer"):
        read_index(newer)
    # A placement kind this reader does not know (a future partial-sum label) is refused, not read as a shard.
    partial = written(placement(["partial", 2]))
    with expect_error(TopologyError, f"{partial}/index.json: unknown placement kind 'partial'"):
        read_index(partial)
    with expect_error(TopologyError, "placement shard must have a dim"):
        read_index(written(placement(["shard"])))


def test_naming(expect_error):
    assert step_dir_name(1000) == "step_0001000"
    assert step_dir_name(7, prefix="llama8b") == "llama8b_step_0000007"
    assert parse_dir_name("step_0001000") == ("", 1000)
    assert parse_dir_name("llama8b_step_12") == ("llama8b", 12)
    assert parse_dir_name("/runs/x/final") is None
    assert data_file_name(0, 0, (1, 2)) == "data/w000_m0_d1-2.safetensors"
    assert data_file_name(3, 1, (0,)) == "data/w003_m1_d0.safetensors"
    with expect_error(ValueError, "step must not be negative"):
        step_dir_name(-1)  # "step_-000001" would never parse back, so find_latest would not find it


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
    (tmp_path / "step_0000050").mkdir()
    (tmp_path / "step_0000050" / "index.json").write_text("{}")
    assert find_latest(str(tmp_path), include_replaced=True) == str(
        tmp_path / "step_0000050"
    ), "the final name wins a step tie whatever the listing order"
    assert find_latest(str(tmp_path), prefix="other") == str(tmp_path / "other_step_0000200")
    assert find_latest(str(tmp_path / "missing")) is None
    assert is_complete(str(tmp_path / "step_0000030"))
    assert not is_complete(str(tmp_path / "step_0000100.partial-123"))
    assert not is_complete(str(tmp_path / "step_0000050.replaced-9")), "renamed back to its final name first"
    assert not is_complete(str(tmp_path / "step_0000040"))


def test_read_user_header_returns_only_the_dict_passed_to_save(tmp_path):
    ckpt = str(tmp_path / "step_0000001")
    os.makedirs(ckpt)
    write_header(ckpt, {"format": "ttml-dcp", "version": 1, "user": {"step": 1, "run": "a"}, "train_state": {}})
    assert read_user_header(ckpt) == {"step": 1, "run": "a"}


def test_a_header_save_would_not_write_is_refused(tmp_path, expect_error):
    ckpt = str(tmp_path / "step_0000001")
    os.makedirs(ckpt)
    good = {"format": "ttml-dcp", "version": 1, "user": {"step": 1}}
    write_header(ckpt, good)
    assert read_checked_header(ckpt) == good

    cases = [
        (["not", "an", "object"], IncompleteCheckpointError, r"not a readable header \(TypeError"),
        (
            {"format": "ttml-dcp", "version": 1},
            IncompleteCheckpointError,
            r"not a readable header \(KeyError: 'user'\)",
        ),
        (
            {"format": "ttml-dcp", "user": {}},
            IncompleteCheckpointError,
            r"not a readable header \(KeyError: 'version'\)",
        ),
        ({**good, "version": "1"}, IncompleteCheckpointError, "is not an integer"),
        ({**good, "user": ["step", 1]}, IncompleteCheckpointError, "the user header is a list"),
        ({**good, "format": "other"}, VersionError, "not a ttml-dcp header: format='other'"),
        ({"version": 1, "user": {}}, VersionError, "not a ttml-dcp header: format=None"),
        ({**good, "version": 2}, VersionError, "header version 2 is newer than the supported 1"),
    ]
    for value, error, message in cases:
        write_header(ckpt, value)
        with expect_error(error, message):
            read_checked_header(ckpt)
        with expect_error(error, message):
            read_user_header(ckpt)


def test_staging_becomes_complete_only_at_finalize(tmp_path, expect_error):
    final = str(tmp_path / "step_0000010")
    staging, nonce = begin_staging(final)
    assert ".partial-" in staging and os.path.isdir(os.path.join(staging, "data"))
    write_header(staging, {"step": 10})
    write_index(staging, _index())
    assert not is_complete(staging) and find_latest(str(tmp_path)) is None
    with expect_error(IncompleteCheckpointError, "is a staging directory"):
        read_index(staging)
    with expect_error(IncompleteCheckpointError, "is a staging directory"):
        read_user_header(staging)

    finalize_staging(staging, final)

    assert is_complete(final) and find_latest(str(tmp_path)) == final
    assert read_header_json(final) == {"step": 10}
    assert read_index(final).checkpoint_id == "step_0000010"
    assert read_index(final).created, "written at finalize"
    with expect_error(IncompleteCheckpointError, "no index.json"):
        read_index(str(tmp_path / "nowhere"))

    staging2, _ = begin_staging(final)
    write_index(staging2, _index())
    with expect_error(CheckpointError, "already exists"):
        finalize_staging(staging2, final)
    finalize_staging(staging2, final, overwrite=True)
    assert is_complete(final) and not [n for n in os.listdir(tmp_path) if ".replaced-" in n]


def test_the_crc_column_is_validated(expect_error):
    encoded = _index().to_json()
    short = json.loads(json.dumps(encoded))
    short["state"]["model/Llama/fc/weight"]["chunks"]["crc32"] = ["3f0a9c11"]
    with expect_error(CheckpointError, "columns disagree"):
        Index.from_json(short)
    bad = json.loads(json.dumps(encoded))
    bad["state"]["model/Llama/fc/weight"]["chunks"]["crc32"] = ["3f0a9c11", "not-hex"]
    with expect_error(CheckpointError, "malformed crc32"):
        Index.from_json(bad)
    unchecked = json.loads(json.dumps(encoded))
    unchecked["state"]["model/Llama/fc/weight"]["chunks"]["crc32"] = ["3f0a9c11", None]
    with expect_error(CheckpointError, "malformed crc32"):
        Index.from_json(unchecked)  # every chunk has a checksum


def test_an_unreadable_index_or_header_is_an_incomplete_checkpoint(tmp_path, expect_error):
    final = str(tmp_path / "step_0000010")
    staging, _ = begin_staging(final)
    write_index(staging, _index())
    write_header(staging, {"step": 10})
    finalize_staging(staging, final)
    index_path = os.path.join(final, "index.json")
    text = open(index_path).read()

    with open(index_path, "w") as f:
        f.write(text[: len(text) // 2])  # truncated
    with expect_error(IncompleteCheckpointError, "not a readable index"):
        read_index(final)
    value = json.loads(text)
    del value["state"]["model/Llama/fc/weight"]["chunks"]["file"]
    with open(index_path, "w") as f:
        json.dump(value, f)  # a required column missing
    with expect_error(IncompleteCheckpointError, "not a readable index"):
        read_index(final)
    value = json.loads(text)
    del value["state"]["model/Llama/fc/weight"]["layout"]
    with open(index_path, "w") as f:
        json.dump(value, f)  # a field every save writes, missing: no default stands in for it
    with expect_error(IncompleteCheckpointError, r"not a readable index \(KeyError: 'layout'\)"):
        read_index(final)
    value = json.loads(text)
    del value["state"]["model/Llama/fc/weight"]["chunks"]["crc32"]
    with open(index_path, "w") as f:
        json.dump(value, f)  # the checksum column too: no chunk is read unchecked
    with expect_error(IncompleteCheckpointError, r"not a readable index \(KeyError: 'crc32'\)"):
        read_index(final)
    with open(os.path.join(final, "header.json"), "w") as f:
        f.write("{")
    with expect_error(IncompleteCheckpointError, "not readable JSON"):
        read_header_json(final)
    os.remove(os.path.join(final, "header.json"))
    with expect_error(IncompleteCheckpointError, "has no header.json"):
        read_header_json(final)


def test_only_bool_int_float_and_str_scalars_are_stored(expect_error):
    index = _index()
    index.scalars["optimizer/betas"] = (0.9, 0.999)
    with expect_error(CheckpointError, "scalar of type tuple cannot be stored in the index"):
        index.to_json()


def test_without_overwrite_a_checkpoint_committed_during_finalize_is_refused_not_replaced(
    tmp_path, monkeypatch, expect_error
):
    final = str(tmp_path / "step_0000010")
    staging, _ = begin_staging(final)
    write_index(staging, _index())
    other, _ = begin_staging(final)
    other_index = _index()
    other_index.checkpoint_id = "the other save"
    write_index(other, other_index)
    real_fsync_dir = index_module._fsync_dir

    def fsync_while_another_save_commits(path):  # the other save commits between the first check and the rename
        real_fsync_dir(path)
        if path == staging:
            finalize_staging(other, final)

    monkeypatch.setattr(index_module, "_fsync_dir", fsync_while_another_save_commits)
    with expect_error(CheckpointError, "already exists; pass overwrite=True"):
        finalize_staging(staging, final)
    monkeypatch.setattr(index_module, "_fsync_dir", real_fsync_dir)

    assert read_index(final).checkpoint_id == "the other save", "the committed checkpoint is kept"
    assert not [n for n in os.listdir(tmp_path) if ".replaced-" in n], "nothing was moved aside"


def test_a_failed_overwrite_puts_the_previous_checkpoint_back(tmp_path, monkeypatch, expect_error):
    final = str(tmp_path / "step_0000010")
    staging, _ = begin_staging(final)
    write_index(staging, _index())
    finalize_staging(staging, final)
    previous = read_index(final).checkpoint_id

    staging2, _ = begin_staging(final)
    replacement = _index()
    replacement.checkpoint_id = "replacement"
    write_index(staging2, replacement)
    real_rename = os.rename

    def failing_rename(src, dst):
        if src == staging2:
            raise OSError("simulated NFS failure")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", failing_rename)
    with expect_error(OSError, "simulated NFS failure"):
        finalize_staging(staging2, final, overwrite=True)
    monkeypatch.setattr(os, "rename", real_rename)

    assert is_complete(final) and read_index(final).checkpoint_id == previous
    assert not [n for n in os.listdir(tmp_path) if ".replaced-" in n], "nothing left aside"
    assert os.path.isdir(staging2), "the new checkpoint is untouched for the caller to retry or delete"


def test_a_failed_overwrite_that_cannot_roll_back_says_where_the_previous_checkpoint_is(
    tmp_path, monkeypatch, expect_error
):
    final = str(tmp_path / "step_0000010")
    staging, _ = begin_staging(final)
    write_index(staging, _index())
    finalize_staging(staging, final)
    staging2, _ = begin_staging(final)
    write_index(staging2, _index())
    real_rename = os.rename

    def failing_rename(src, dst):
        if src == final:
            real_rename(src, dst)  # moving the previous checkpoint aside works; nothing after it does
        else:
            raise OSError(f"simulated NFS failure renaming {os.path.basename(src)}")

    monkeypatch.setattr(os, "rename", failing_rename)
    with expect_error(
        CheckpointError, r"nor move the previous checkpoint back from .*\.replaced-.*rename it back by hand"
    ):
        finalize_staging(staging2, final, overwrite=True)
    monkeypatch.setattr(os, "rename", real_rename)
    assert [n for n in os.listdir(tmp_path) if ".replaced-" in n], "the previous checkpoint is still aside"


def test_an_overwrite_warns_when_the_previous_checkpoint_cannot_be_removed(tmp_path, monkeypatch):
    final = str(tmp_path / "step_0000010")
    staging, _ = begin_staging(final)
    write_index(staging, _index())
    finalize_staging(staging, final)
    staging2, _ = begin_staging(final)
    write_index(staging2, _index())

    def busy_rmtree(path, onerror):  # an NFS server keeping an open file as .nfsXXXX
        onerror(os.unlink, os.path.join(path, ".nfs0001"), (OSError, OSError("Device or resource busy"), None))

    monkeypatch.setattr(shutil, "rmtree", busy_rmtree)
    with pytest.warns(UserWarning, match="could not be removed"):
        finalize_staging(staging2, final, overwrite=True)
    assert is_complete(final), "the new checkpoint is in place"
