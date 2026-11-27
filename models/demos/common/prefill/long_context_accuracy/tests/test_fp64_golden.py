# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The golden registry contract on tiny goldens in a temporary root. CPU only, no device."""

import json
import os
import re
import stat
from pathlib import Path

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    GOLDEN_DIR_MODE,
    GOLDEN_FILE_MODE,
    MANIFEST,
    ROOT_ENV,
    TOKEN_IDS,
    Fp64Golden,
    GoldenError,
    GoldenInvalid,
    GoldenMismatch,
    GoldenNotFound,
    MappedFile,
    golden_root,
    make_golden_dir,
    open_golden,
    product_schema,
    read_slices,
    write_manifest,
    write_token_ids,
)

TOKENS = [5, 1, 4, 1, 5, 9, 2, 6, 5, 3]  # 10 rows in segments [0, 4), [4, 8), [8, 10)


def make_golden(root: Path, overrides: dict | None = None) -> Path:
    golden_id = "tiny-l0-kda-fp64-10"
    path = root / golden_id
    path.mkdir(parents=True)
    manifest = {
        "golden_id": golden_id,
        "producer": "tests.tiny",
        "producer_version": 1,
        "identity": {"model": "tiny", "token_source": "hand-written", "weights_sha256": "ab" * 32},
        "rows": 10,
        "token_ids_sha256": write_token_ids(path, TOKENS),
        "segment": 4,
        "segments": [{"start": 0, "end": 4}, {"start": 4, "end": 8}, {"start": 8, "end": 10}],
        "products": {"output": product_schema("row", torch.zeros(4, 3, dtype=torch.float64))},
    }
    manifest.update(overrides or {})
    write_manifest(path, {k: v for k, v in manifest.items() if v is not None})
    return path


def test_open_golden_serves_a_complete_golden(tmp_path: Path) -> None:
    make_golden(tmp_path)
    golden = open_golden("tiny-l0-kda-fp64-10", root=tmp_path)
    assert (golden.golden_id, golden.rows, golden.identity["model"]) == ("tiny-l0-kda-fp64-10", 10, "tiny")
    assert golden.manifest["products"]["output"] == {"kind": "row", "dtype": "float64", "shape": [3]}
    assert golden.token_ids().tolist() == TOKENS
    golden.require(producer="tests.tiny", producer_version=1, identity={"model": "tiny"}, products=["output"])


def test_writers_make_a_golden_readable_by_other_users_whatever_the_umask(tmp_path: Path) -> None:
    """CI reads goldens as users other than their producer, so a producer under umask 077 must still publish
    world-readable files and a traversable golden directory, root and every directory it creates above them (here
    ``new``); an existing directory (``tmp_path``) keeps its mode."""
    existing = stat.S_IMODE(tmp_path.stat().st_mode)
    previous = os.umask(0o077)
    try:
        golden = tmp_path / "new" / "root" / "tiny"
        make_golden_dir(golden)
        write_token_ids(golden, TOKENS)
        write_manifest(golden, {"golden_id": "tiny"})
        write_manifest(golden, {"golden_id": "tiny", "rewritten": True})  # replacing keeps the mode
    finally:
        os.umask(previous)
    modes = {
        path.name: stat.S_IMODE(path.stat().st_mode)
        for path in (golden.parent.parent, golden.parent, golden, *golden.iterdir())
    }
    assert modes == {
        "new": GOLDEN_DIR_MODE,
        "root": GOLDEN_DIR_MODE,
        "tiny": GOLDEN_DIR_MODE,
        MANIFEST: GOLDEN_FILE_MODE,
        TOKEN_IDS: GOLDEN_FILE_MODE,
    }  # no temporary file is left behind either
    assert stat.S_IMODE(tmp_path.stat().st_mode) == existing
    assert (GOLDEN_FILE_MODE, GOLDEN_DIR_MODE) == (0o664, 0o2775)


def test_registry_root_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_golden(tmp_path)
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    assert golden_root() == tmp_path
    assert open_golden("tiny-l0-kda-fp64-10").path == tmp_path / "tiny-l0-kda-fp64-10"


def test_a_missing_golden_raises_rather_than_skipping(tmp_path: Path, expect_error) -> None:
    with expect_error(GoldenNotFound, "no golden 'absent'"):
        open_golden("absent", root=tmp_path)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"golden_id": "other"}, "is not the directory name"),
        ({"producer_version": "1"}, "producer_version must be an integer"),
        ({"identity": {"model": "tiny", "weights_sha256": "x"}}, "identity lacks token_source"),
        ({"segments": [{"start": 0, "end": 4}, {"start": 4, "end": 8}]}, "incomplete"),
        ({"segments": [{"start": 0, "end": 4}, {"start": 5, "end": 8}, {"start": 8, "end": 10}]}, "segment 1 covers"),
        ({"rows": 12}, "segment 2 covers \\(8, 10\\), expected \\[8, 12\\)"),
        ({"products": {"output": {"kind": "rows", "dtype": "float64", "shape": [3]}}}, "product 'output' needs kind"),
        ({"products": None}, "products must be a non-empty object"),
    ],
    ids=["renamed", "version-type", "identity", "incomplete", "gap", "rows", "product-kind", "no-products"],
)
def test_open_golden_refuses_a_manifest_that_breaks_the_contract(
    tmp_path: Path, overrides: dict, message: str, expect_error
) -> None:
    make_golden(tmp_path, overrides)
    with expect_error(GoldenInvalid, message):
        open_golden("tiny-l0-kda-fp64-10", root=tmp_path)


def test_require_names_every_mismatch(tmp_path: Path, expect_error) -> None:
    golden = open_golden(make_golden(tmp_path).name, root=tmp_path)
    with expect_error(GoldenMismatch, "producer is") as raised:
        golden.require(producer="tests.other", producer_version=2, identity={"model": "big"}, products=["state"])
    assert str(raised.value).split(": ", 1)[1] == (
        "producer is 'tests.tiny', expected 'tests.other'; producer_version is 1, expected 2; "
        "identity.model is 'tiny', expected 'big'; no product 'state'"
    )


def test_require_checks_the_weights_a_consumer_loaded_against_the_golden_s(tmp_path: Path, expect_error) -> None:
    """A consumer loads some of the weights a golden was built from; each digest it passes must be the golden's
    (a subset: unnamed layers pass), and a different or unknown one is named."""
    digests = {"embed": "aa" * 32, "layers.0": "bb" * 32, "layers.1": "cc" * 32}
    identity = {"model": "tiny", "token_source": "hand-written", "weights_sha256": digests}
    golden = open_golden(make_golden(tmp_path, {"identity": identity}).name, root=tmp_path)
    golden.require(producer="tests.tiny", producer_version=1, weights_sha256={"layers.0": "bb" * 32})
    with expect_error(GoldenMismatch, "weights layers.0 are dddddddddddd..., the golden's bbbbbbbbbbbb...") as raised:
        golden.require(
            producer="tests.tiny", producer_version=1, weights_sha256={"layers.0": "dd" * 32, "x": "ee" * 32}
        )
    assert "weights x are eeeeeeeeeeee..., the golden's None..." in str(raised.value)


def test_token_ids_must_match_the_manifest_digest(tmp_path: Path, expect_error) -> None:
    path = make_golden(tmp_path)
    golden = open_golden(path.name, root=tmp_path)
    torch.save(torch.tensor(TOKENS[::-1]), path / TOKEN_IDS)
    with expect_error(GoldenInvalid, "does not match"):
        golden.token_ids()
    torch.save(torch.tensor(TOKENS[:9]), path / TOKEN_IDS)
    with expect_error(GoldenInvalid, "must hold 10 int64 token ids"):
        golden.token_ids()
    (path / TOKEN_IDS).unlink()
    with expect_error(GoldenInvalid, "has no token_ids.pt"):
        golden.token_ids()


def test_manifest_writes_are_whole(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"a": 1})
    assert json.loads((tmp_path / MANIFEST).read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == [MANIFEST]  # no temporary left behind


class _TinyReader(Fp64Golden):
    """A reader of ``make_golden``'s layout: row r's ``output`` is ``[r, r, r]`` (no files needed)."""

    producer = "tests.tiny"
    producer_version = 1

    def _read_rows(self, names, start, end):
        return {"output": torch.arange(start, end, dtype=torch.float64).unsqueeze(1).expand(-1, 3)}


def test_read_rows_validates_then_delegates_to_the_reader(tmp_path: Path, expect_error) -> None:
    """``read_rows`` refuses unknown products and rows outside the golden before its reader runs; the base class
    reads no products; a reader that names its producer is checked on open, and ``require`` defaults to it."""
    make_golden(tmp_path)
    golden = open_golden("tiny-l0-kda-fp64-10", root=tmp_path, reader=_TinyReader)
    assert golden.read_rows(["output"], 8, 10)["output"][:, 0].tolist() == [8.0, 9.0]
    golden.require(identity={"model": "tiny"})
    with expect_error(ValueError, "has no products"):
        golden.read_rows(["state"], 0, 4)
    with expect_error(ValueError, "not inside"):
        golden.read_rows(["output"], 4, 11)
    with expect_error(GoldenError, "no reader"):
        open_golden("tiny-l0-kda-fp64-10", root=tmp_path).read_rows(["output"], 0, 4)
    with expect_error(ValueError, "names no producer"):
        open_golden("tiny-l0-kda-fp64-10", root=tmp_path).require()

    class OtherProducer(_TinyReader):
        producer = "tests.other"

    with expect_error(GoldenMismatch, "expected 'tests.other'"):
        open_golden("tiny-l0-kda-fp64-10", root=tmp_path, reader=OtherProducer)


def test_read_slices_copies_mapped_views_as_torch_cat_would(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``read_slices`` returns, per name, exactly ``torch.cat`` of the views: row slices across two files, several
    dtypes, a column view (strided, like K3.1's ``kpe``), a nested state and a whole tensor; the bytes are split
    across streams inside and across views (3 streams over these sizes)."""
    monkeypatch.setattr("models.demos.common.prefill.long_context_accuracy.fp64_golden.READ_STREAMS", 3)
    files = []
    for i in range(2):
        rows = torch.arange(6 * 5, dtype=torch.float64).reshape(6, 5) + 100 * i
        payload = {
            "start": 6 * i,
            "rows": rows,
            "ids": (rows[:, :2] * 7).to(torch.int32),
            "margin": rows[:, 0].float().neg(),
            "state": {"keys": rows[:, 1:].clone(), "recurrent": torch.full((1, 2, 3), float(i), dtype=torch.float64)},
        }
        torch.save(payload, tmp_path / f"segment_{i}.pt")
        files.append(MappedFile(tmp_path / f"segment_{i}.pt"))
    a, b = (f.contents for f in files)
    parts = {
        "rows": [(files[0], a["rows"][4:]), (files[1], b["rows"][:3])],
        "ids": [(files[0], a["ids"][5:]), (files[1], b["ids"][:1])],
        "margin": [(files[0], a["margin"][2:]), (files[1], b["margin"])],
        "columns": [(files[0], a["state"]["keys"][3:, 2:]), (files[1], b["state"]["keys"][:2, 2:])],
        "recurrent": [(files[1], b["state"]["recurrent"])],
    }
    got = read_slices(parts)
    assert list(got) == list(parts)
    for name, views in parts.items():
        want = torch.cat([v for _, v in views])
        assert got[name].dtype == want.dtype and got[name].is_contiguous() and torch.equal(got[name], want), name
    assert got["rows"][:, 0].tolist() == [20.0, 25.0, 100.0, 105.0, 110.0]  # (hand-checked: row r starts at 5 r)


def test_read_slices_refuses_a_view_of_another_file(tmp_path: Path, expect_error) -> None:
    torch.save({"x": torch.zeros(4, 2)}, tmp_path / "a.pt")
    torch.save({"x": torch.ones(64, 64)}, tmp_path / "b.pt")
    a, b = MappedFile(tmp_path / "a.pt"), MappedFile(tmp_path / "b.pt")
    with expect_error(ValueError, "not inside"):
        read_slices({"x": [(a, b.contents["x"][:10])]})


def test_read_slices_refuses_views_that_torch_cat_would_not_join_as_they_are(tmp_path: Path, expect_error) -> None:
    """Views of one name must share dtype and trailing shape (a malformed golden fails, not reinterprets bytes)."""
    torch.save({"x": torch.zeros(4, 3, dtype=torch.float64)}, tmp_path / "a.pt")
    torch.save({"x": torch.zeros(4, 5, dtype=torch.float64), "y": torch.zeros(4, 3)}, tmp_path / "b.pt")
    a, b = MappedFile(tmp_path / "a.pt"), MappedFile(tmp_path / "b.pt")
    with expect_error(GoldenInvalid, re.escape("not torch.float64[*, [3]]")):
        read_slices({"x": [(a, a.contents["x"]), (b, b.contents["x"])]})
    with expect_error(GoldenInvalid, re.escape("torch.float32[4, 3], not torch.float64")):
        read_slices({"x": [(a, a.contents["x"]), (b, b.contents["y"])]})


def test_read_slices_fails_on_a_file_that_ends_before_its_bytes(tmp_path: Path, expect_error) -> None:
    torch.save({"x": torch.arange(64 * 64, dtype=torch.float64).reshape(64, 64)}, tmp_path / "a.pt")
    file = MappedFile(tmp_path / "a.pt")
    view = file.contents["x"][32:]
    os.truncate(tmp_path / "a.pt", file.span(view)[0] + 100)  # (the mapping keeps its size; the reads see the end)
    with expect_error(GoldenInvalid, "ends before byte"):
        read_slices({"x": [(file, view)]})


def test_mapped_file_refuses_a_mapping_whose_storages_do_not_match_its_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expect_error
) -> None:
    """An extra storage outside the file (here a fresh tensor next to the mapped one) cannot be located: refused."""
    torch.save({"x": torch.zeros(4, 3)}, tmp_path / "a.pt")
    load = torch.load
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {**load(*args, **kwargs), "extra": torch.ones(2)})
    with expect_error(GoldenInvalid, "cannot locate its 1 tensor records"):
        MappedFile(tmp_path / "a.pt")
