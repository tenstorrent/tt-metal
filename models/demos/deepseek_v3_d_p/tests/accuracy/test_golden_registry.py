# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The golden registry contract on tiny goldens in a temporary root. CPU only, no device."""

import json
import os
import stat
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.accuracy.golden_registry import (
    GOLDEN_DIR_MODE,
    GOLDEN_FILE_MODE,
    MANIFEST,
    ROOT_ENV,
    TOKEN_IDS,
    GoldenInvalid,
    GoldenMismatch,
    GoldenNotFound,
    golden_root,
    make_golden_dir,
    open_golden,
    product_schema,
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
    world-readable files and a traversable root and golden directory."""
    previous = os.umask(0o077)
    try:
        golden = tmp_path / "root" / "tiny"
        make_golden_dir(golden)
        write_token_ids(golden, TOKENS)
        write_manifest(golden, {"golden_id": "tiny"})
        write_manifest(golden, {"golden_id": "tiny", "rewritten": True})  # replacing keeps the mode
    finally:
        os.umask(previous)
    modes = {path.name: stat.S_IMODE(path.stat().st_mode) for path in (golden.parent, golden, *golden.iterdir())}
    assert modes == {
        "root": GOLDEN_DIR_MODE,
        "tiny": GOLDEN_DIR_MODE,
        MANIFEST: GOLDEN_FILE_MODE,
        TOKEN_IDS: GOLDEN_FILE_MODE,
    }  # no temporary file is left behind either
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
