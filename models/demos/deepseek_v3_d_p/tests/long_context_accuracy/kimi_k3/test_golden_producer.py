# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 KDA golden producer (``golden_producer.py``) on a tiny configuration: resumed runs are bit-identical,
products are computed in FP64 and stored FP32, the token floor covers the whole golden, and the FP32 conversion
equals a direct run. CPU only."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import MANIFEST, open_golden, write_manifest
from models.demos.common.prefill.long_context_accuracy.windowed_errors import golden_floor
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config, random_weights
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3 import golden_producer
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden import KimiK3KdaGolden
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden_producer import (
    convert_to_fp32_storage,
    resume_state_path,
    run_golden,
)

TOKENS = list(range(10))
SEGMENT = 4  # segments [0, 4), [4, 8), [8, 10): the last one is partial


class Interrupted(Exception):
    pass


def make_run(run_dir: Path, *, fail_after: int | None = None, identity: str = "tiny", max_segments: int | None = None):
    config = make_config()
    weights = random_weights(config)
    table = torch.randn(len(TOKENS), config.hidden_size, generator=torch.Generator().manual_seed(7)).bfloat16()
    calls = []

    def layer_input(ids):
        if fail_after is not None and len(calls) == fail_after:
            raise Interrupted("simulated crash")
        calls.append(ids)
        return table[list(ids)].unsqueeze(0)

    def run():
        return run_golden(
            run_dir=run_dir,
            token_ids=TOKENS,
            layer_input=layer_input,
            weights=weights,
            config=config,
            segment=SEGMENT,
            dtype=torch.float64,
            identity={"model": identity, "token_source": "a random table", "weights_sha256": "random"},
            max_segments=max_segments,
        )

    return run, table, weights, config


def test_resumed_run_is_bit_identical_to_an_uninterrupted_one(tmp_path: Path, expect_error) -> None:
    run_full, table, weights, config = make_run(tmp_path / "full")
    run_full()

    run_interrupted, *_ = make_run(tmp_path / "resumed", fail_after=2)
    with expect_error(Interrupted, "simulated crash"):
        run_interrupted()
    run_resumed, *_ = make_run(tmp_path / "resumed")
    manifest = run_resumed()

    assert [(s["start"], s["end"]) for s in manifest["segments"]] == [(0, 4), (4, 8), (8, 10)]
    assert [session["first_segment"] for session in manifest["sessions"]] == [0, 2]
    for index in range(3):
        full = torch.load(tmp_path / "full" / f"segment_{index:06d}.pt", weights_only=True)
        resumed = torch.load(tmp_path / "resumed" / f"segment_{index:06d}.pt", weights_only=True)
        assert full["output"].dtype == torch.float32 and torch.equal(full["output"], resumed["output"]), index
        for name, tensor in full["state"].items():
            assert tensor.dtype == torch.float32 and torch.equal(tensor, resumed["state"][name]), (index, name)
    # A complete golden keeps no resume state, and a rerun on it computes nothing and needs none.
    for run in ("full", "resumed"):
        assert not list((tmp_path / run).glob("resume_state_*.pt")), run
    assert run_resumed()["segments"] == manifest["segments"]

    # Segmenting changes nothing but rounding: the whole sequence in one call agrees to FP32 storage precision.
    whole_output, whole_state = kda_forward_reference(table.unsqueeze(0), weights, config, dtype=torch.float64)
    final = torch.load(tmp_path / "full" / "segment_000002.pt", weights_only=True)["state"]["recurrent"]
    torch.testing.assert_close(final, whole_state.recurrent.float())


@pytest.mark.parametrize(
    "segment, written, lists",
    [(1, False, [(0, 4)]), (1, True, [(0, 4), (4, 8)]), (2, True, [(0, 4), (4, 8), (8, 10)])],
    ids=["before-1", "after-1", "after-last"],
)
def test_a_crash_around_a_segment_manifest_write_resumes_bit_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expect_error, segment: int, written: bool, lists: list
) -> None:
    """A crash just before or just after the manifest lists a segment (its files already published) leaves the FP64
    state of the manifest's last segment in place; the rerun equals an uninterrupted run bit for bit and leaves no
    resume state (after the last segment it only cleans up)."""
    run_full, *_ = make_run(tmp_path / "full")
    run_full()

    def crash_at_segment_1(run_dir, manifest):
        if len(manifest["segments"]) == segment + 1 and "token_floor" not in manifest:
            if written:
                write_manifest(run_dir, manifest)
            raise Interrupted("simulated crash")
        write_manifest(run_dir, manifest)

    with monkeypatch.context() as patch:
        patch.setattr(golden_producer, "write_manifest", crash_at_segment_1)
        run_crashing, *_ = make_run(tmp_path / "resumed")
        with expect_error(Interrupted, "simulated crash"):
            run_crashing()
    manifest = json.loads((tmp_path / "resumed" / MANIFEST).read_text())
    assert [(s["start"], s["end"]) for s in manifest["segments"]] == lists
    assert resume_state_path(tmp_path / "resumed", len(lists) - 1).exists()

    run_resumed, *_ = make_run(tmp_path / "resumed")
    resumed = run_resumed()
    if len(lists) < 3:
        assert [session["first_segment"] for session in resumed["sessions"]] == [0, len(lists)]
    for index in range(3):
        full = torch.load(tmp_path / "full" / f"segment_{index:06d}.pt", weights_only=True)
        got = torch.load(tmp_path / "resumed" / f"segment_{index:06d}.pt", weights_only=True)
        assert torch.equal(full["output"], got["output"]), index
        assert all(torch.equal(t, got["state"][n]) for n, t in full["state"].items()), index
    assert not list((tmp_path / "resumed").glob("resume_state_*.pt"))


def test_products_are_computed_in_fp64_and_stored_in_fp32(tmp_path: Path) -> None:
    """Each stored tensor is the FP64 computation rounded once to FP32, bit for bit: the first segment against the
    reference run in FP64 on its rows, the second against the FP64 run continued from the FP64 state."""
    run, table, weights, config = make_run(tmp_path)
    manifest = run()
    assert manifest["dtype"] == "torch.float64" and manifest["producer_version"] == 2
    assert {schema["dtype"] for schema in manifest["products"].values()} == {"float32"}
    state = None
    for index in range(2):
        rows = table[index * SEGMENT : (index + 1) * SEGMENT].unsqueeze(0)
        output, state = kda_forward_reference(rows, weights, config, state, dtype=torch.float64)
        stored = torch.load(tmp_path / f"segment_{index:06d}.pt", weights_only=True)
        assert torch.equal(stored["output"].view(torch.int32), output[0].float().view(torch.int32)), index
        for name, tensor in vars(state).items():
            assert torch.equal(stored["state"][name].view(torch.int32), tensor.float().view(torch.int32)), name


def test_a_complete_golden_records_the_whole_golden_token_floor(tmp_path: Path, expect_error) -> None:
    """The KDA scorer's per-row floor: ``golden_floor`` over every output row, written only once the golden is
    complete."""
    partial, *_ = make_run(tmp_path, fail_after=2)
    with expect_error(Interrupted, "simulated crash"):
        partial()
    assert "token_floor" not in json.loads((tmp_path / MANIFEST).read_text())

    run, *_ = make_run(tmp_path)
    manifest = run()
    golden = open_golden(tmp_path.name, root=tmp_path.parent, reader=KimiK3KdaGolden)
    output = golden.read_rows(["output"], 0, golden.rows)["output"]
    assert manifest["token_floor"] == golden_floor(output.double().norm(dim=-1))


def _version_1_golden(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A golden of producer version 1 (products stored FP64): this producer with FP64 storage."""
    with monkeypatch.context() as patch:
        patch.setattr(golden_producer, "STORAGE_DTYPE", torch.float64)
        patch.setattr(golden_producer, "PRODUCER_VERSION", 1)
        run, *_ = make_run(run_dir)
        manifest = run()
    return manifest


def test_conversion_writes_the_golden_the_producer_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expect_error
) -> None:
    """A version-1 golden converted to FP32 storage equals a version-2 run of the same inputs bit for bit: segment
    files, products and token floor; the manifest records the source and its digest."""
    source = _version_1_golden(tmp_path / "fp64", monkeypatch)
    run, *_ = make_run(tmp_path / "direct")
    direct = run()
    converted = convert_to_fp32_storage(tmp_path / "fp64", tmp_path / "fp32")

    assert converted == json.loads((tmp_path / "fp32" / MANIFEST).read_text())
    for key in ("producer", "producer_version", "dtype", "products", "token_floor", "rows"):
        assert converted[key] == direct[key], key
    assert converted["golden_id"] == "fp32" and converted["identity"] == source["identity"]
    conversion = converted["conversion"]
    assert conversion["source_golden_id"] == "fp64" and conversion["source_token_floor"] == source["token_floor"]
    assert (
        conversion["source_manifest_sha256"] == hashlib.sha256((tmp_path / "fp64" / MANIFEST).read_bytes()).hexdigest()
    )
    for index, entry in enumerate(converted["segments"]):
        path = tmp_path / "fp32" / f"segment_{index:06d}.pt"
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        want = torch.load(tmp_path / "direct" / path.name, weights_only=True)
        got = torch.load(path, weights_only=True)
        assert torch.equal(want["output"], got["output"]) and want["output"].dtype == got["output"].dtype
        assert all(
            torch.equal(t, got["state"][n]) and t.dtype == got["state"][n].dtype for n, t in want["state"].items()
        )
    assert not list((tmp_path / "fp32").glob("resume_state_*.pt"))
    golden = open_golden("fp32", root=tmp_path, reader=KimiK3KdaGolden)
    assert golden.token_ids().tolist() == TOKENS
    rows = golden.read_rows(["output", "state.recurrent"], 0, 8)
    assert rows["output"].dtype == torch.float32 and rows["output"].shape[0] == 8
    # The source is unchanged and still a version-1 golden; a destination holding a golden is refused.
    assert json.loads((tmp_path / "fp64" / MANIFEST).read_text()) == source
    with expect_error(ValueError, "already holds a golden"):
        convert_to_fp32_storage(tmp_path / "fp64", tmp_path / "fp32")
    with expect_error(ValueError, "version-1 golden"):
        convert_to_fp32_storage(tmp_path / "direct", tmp_path / "again")
