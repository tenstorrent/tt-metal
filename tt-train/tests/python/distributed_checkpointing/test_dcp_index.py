# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""index.json round trip, directory naming, completeness and the staging directory."""

import gc
import json
import os
import shutil
import warnings

import numpy as np
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
    check_json,
    data_file_name,
    finalize_staging,
    find_latest,
    is_complete,
    parse_dir_name,
    read_checked_header,
    read_header_json,
    read_index,
    read_user_header,
    staging_dir,
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
    # The three entries share one label: the index holds it once and the entries point at it.
    assert len(encoded["labels"]) == 1 and all(entry["label"] == 0 for entry in encoded["state"].values())
    assert len({id(entry.label) for entry in decoded.entries.values()}) == 1, "decoded once, shared"


def test_entries_point_at_the_label_table_by_position(expect_error):
    index = _index()
    other = Label((Placement.shard(2),), (4,), tuple(dist_coords((2, 2))))
    index.entries["optimizer/exp_avg/model/Llama/fc/weight"].label = other
    encoded = json.loads(json.dumps(index.to_json()))
    assert encoded["labels"] == [index.entries["model/Llama/fc/weight"].label.to_json(), other.to_json()]
    assert [entry["label"] for entry in encoded["state"].values()] == [0, 1, 0], "in order of first use"
    assert Index.from_json(encoded).entries == index.entries
    for bad in (2, -1, "0", None):
        encoded["state"]["model/Llama/fc/weight"]["label"] = bad
        with expect_error(CheckpointError, f"'model/Llama/fc/weight': label {bad!r} is not a position in the index's"):
            Index.from_json(encoded)


def test_index_is_written_compact(tmp_path):
    """index.json is compact JSON, and writing or reading it pauses the GC only while it runs: a GC the caller
    paused stays paused."""
    directory = str(tmp_path)
    write_index(directory, _index())
    text = (tmp_path / "index.json").read_text()
    assert "\n" not in text and '": ' not in text, "no indentation and no spaces after separators"
    assert read_index(directory).entries == _index().entries
    assert gc.isenabled(), "writing and reading pause the GC only while they run"
    gc.disable()
    try:
        write_index(directory, _index())
        read_index(directory)
        assert not gc.isenabled(), "a GC the caller paused stays paused"
    finally:
        gc.enable()


def test_a_malformed_tp_split_is_refused(expect_error):
    encoded = json.loads(json.dumps(_index().to_json()))
    encoded["state"]["model/Llama/blocks/0/attention/qkv_linear/weight"]["tp_split"] = [1, 2, 1]
    with expect_error(CheckpointError, r"qkv_linear/weight': malformed tp_split \[1, 2, 1\]"):
        Index.from_json(encoded)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("offsets", [[0, 0, 0, 0], [0, 0, -96, 0]], r"malformed chunk offsets \[0, 0, -96, 0\]"),
        ("offsets", [[0, 0, 0, 0], [0, 0, 96]], r"malformed chunk offsets \[0, 0, 96\]"),
        ("offsets", [[0, 0, 0, 0], [0, 0, 96.0, 0]], r"malformed chunk offsets \[0, 0, 96.0, 0\]"),
        ("sizes", [[1, 1, 96, "16"]], r"malformed chunk sizes \[1, 1, 96, '16'\]"),
        ("global_shape", [1, 1, 192, -16], r"malformed global_shape \[1, 1, 192, -16\]"),
        ("canonical_shape", [1, 130, 16], r"malformed canonical_shape \[1, 130, 16\]"),
    ],
)
def test_malformed_shapes_and_chunk_boxes_are_refused(field, value, message, expect_error):
    # Boxes are built unchecked, so decoding the index is where a bad shape or box must be caught.
    encoded = json.loads(json.dumps(_index().to_json()))
    entry = encoded["state"]["model/Llama/fc/weight"]
    (entry["chunks"] if field in ("offsets", "sizes") else entry)[field] = value
    with expect_error(CheckpointError, "index entry 'model/Llama/fc/weight': " + message):
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
        def change(encoded):
            encoded["labels"][encoded["state"]["model/Llama/fc/weight"]["label"]]["placements"][1] = value

        return change

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


@pytest.mark.filterwarnings("ignore:.*left by saves that did not finish")
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


def test_find_latest_warns_about_leftovers(tmp_path):
    """find_latest warns about the .partial- and .replaced- directories of its prefix, naming each, and removes
    none of them."""
    names = ["step_0000005", "step_0000010.partial-7", "step_0000005.replaced-3", "other_step_0000020.partial-1"]
    for name in names:
        (tmp_path / name).mkdir()
    (tmp_path / "step_0000005" / "index.json").write_text("{}")
    leftovers = r"2 directories left by saves that did not finish: step_0000005\.replaced-3, step_0000010\.partial-7\. "
    with pytest.warns(UserWarning, match=leftovers):
        assert find_latest(str(tmp_path)) == str(tmp_path / "step_0000005")
    with pytest.warns(
        UserWarning, match=r"1 directory left by saves that did not finish: other_step_0000020\.partial-1"
    ):
        assert find_latest(str(tmp_path), prefix="other") is None
    assert sorted(os.listdir(tmp_path)) == sorted(names), "nothing is removed: a .partial- save may still be running"
    clean = tmp_path / "clean"
    (clean / "step_0000001").mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert find_latest(str(clean)) is None, "no leftovers, no warning"


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


def test_staging_directory_naming(tmp_path, expect_error):
    """A staging directory is named by its nonce, and only a shared one (a multi-rank save's) may already exist."""
    final = str(tmp_path / "step_0000010")
    staging = begin_staging(final, nonce=42, shared=False)
    assert staging == staging_dir(final, 42) == final + ".partial-42" and os.path.isdir(os.path.join(staging, "data"))
    # Every rank of a multi-rank save creates the directory its ranks agreed on; a lone save's must be new.
    assert begin_staging(final, nonce=42, shared=True) == staging
    with expect_error(FileExistsError, "File exists"):
        begin_staging(final, nonce=42, shared=False)


def test_staging_becomes_complete_only_at_finalize(tmp_path, expect_error):
    final = str(tmp_path / "step_0000010")
    staging = begin_staging(final, nonce=100, shared=False)
    assert ".partial-" in staging and os.path.isdir(os.path.join(staging, "data"))
    write_header(staging, {"step": 10})
    write_index(staging, _index())
    assert not is_complete(staging)
    # A save still running looks like one that died: find_latest names its staging directory and finds nothing.
    with pytest.warns(UserWarning, match=r"left by saves that did not finish: step_0000010\.partial-100\. "):
        assert find_latest(str(tmp_path)) is None
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

    staging2 = begin_staging(final, nonce=1, shared=False)
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


def test_unreadable_index_or_header_is_incomplete(tmp_path, expect_error):
    """An index or header that is missing, truncated or malformed is an IncompleteCheckpointError."""
    final = str(tmp_path / "step_0000010")
    staging = begin_staging(final, nonce=2, shared=False)
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


def test_finalize_keeps_a_checkpoint_committed_meanwhile(tmp_path, monkeypatch, expect_error):
    """Without overwrite, a checkpoint another save commits while this one finalizes is kept, and this save is
    refused rather than moving it aside."""
    final = str(tmp_path / "step_0000010")
    staging = begin_staging(final, nonce=9, shared=False)
    write_index(staging, _index())
    other = begin_staging(final, nonce=10, shared=False)
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
    staging = begin_staging(final, nonce=3, shared=False)
    write_index(staging, _index())
    finalize_staging(staging, final)
    previous = read_index(final).checkpoint_id

    staging2 = begin_staging(final, nonce=4, shared=False)
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


def test_failed_rollback_names_the_previous_checkpoint(tmp_path, monkeypatch, expect_error):
    """An overwrite that fails and cannot roll back says where the previous checkpoint is."""
    final = str(tmp_path / "step_0000010")
    staging = begin_staging(final, nonce=5, shared=False)
    write_index(staging, _index())
    finalize_staging(staging, final)
    staging2 = begin_staging(final, nonce=6, shared=False)
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


def test_overwrite_warns_if_old_checkpoint_stays(tmp_path, monkeypatch):
    """An overwrite warns when the previous checkpoint, moved aside, cannot be removed."""
    final = str(tmp_path / "step_0000010")
    staging = begin_staging(final, nonce=7, shared=False)
    write_index(staging, _index())
    finalize_staging(staging, final)
    staging2 = begin_staging(final, nonce=8, shared=False)
    write_index(staging2, _index())

    def busy_rmtree(path, onerror):  # an NFS server keeping an open file as .nfsXXXX
        onerror(os.unlink, os.path.join(path, ".nfs0001"), (OSError, OSError("Device or resource busy"), None))

    monkeypatch.setattr(shutil, "rmtree", busy_rmtree)
    with pytest.warns(UserWarning, match="could not be removed"):
        finalize_staging(staging2, final, overwrite=True)
    assert is_complete(final), "the new checkpoint is in place"


def test_check_json(expect_error):
    """check_json returns plain JSON types and names the first value that would not read back as written."""
    assert check_json("header", {"step": 3, "lr": 1e-3, "shape": (1, 2), "tags": ["a"], "none": None}) == {
        "step": 3,
        "lr": 1e-3,
        "shape": [1, 2],
        "tags": ["a"],
        "none": None,
    }
    with expect_error(CheckpointError, "dataloader at /per_shard: key 0 is a int"):
        check_json("dataloader", {"per_shard": {0: 7}})
    with expect_error(CheckpointError, r"scheduler at /eta_min: inf is not finite"):
        check_json("scheduler", {"eta_min": float("inf")})
    with expect_error(CheckpointError, r"header at /steps\[1\]: a int64 has no JSON form"):
        check_json("header", {"steps": [1, np.int64(2)]})
