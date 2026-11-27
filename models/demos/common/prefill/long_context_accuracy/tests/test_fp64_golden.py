# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The golden registry on tiny goldens in a temporary root, where a run would not show an error: the identity check
(``Fp64Golden.require``: a golden from another producer, version, model or weights is refused) and the reader's open
files (a leak passes on a host with a high open-file limit). CPU only, no device."""

import os
import resource
from pathlib import Path

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    GoldenMismatch,
    MappedFile,
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


def test_read_slices_closes_every_file_it_opens(tmp_path: Path) -> None:
    """Many reads of several views of one file (as a KDA chunk's state, four tensors of one segment file) leave no
    file open: under an open-file limit a little above what is open now they all pass (a reader that opened a file
    once per view and kept one descriptor per file leaked the rest and hit Errno 24 within a few reads)."""
    torch.save({f"s{i}": torch.full((2, 3), float(i)) for i in range(4)}, tmp_path / "segment.pt")
    open_now = len(os.listdir("/proc/self/fd"))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_now + 32, hard))
    try:
        for _ in range(200):
            file = MappedFile(tmp_path / "segment.pt")
            got = read_slices({name: [(file, t)] for name, t in file.contents.items()})
        assert [got[f"s{i}"][0, 0].item() for i in range(4)] == [0.0, 1.0, 2.0, 3.0]
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert len(os.listdir("/proc/self/fd")) <= open_now
