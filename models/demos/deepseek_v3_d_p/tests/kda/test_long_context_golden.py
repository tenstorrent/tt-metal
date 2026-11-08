# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The resumable long golden on a tiny configuration. CPU only."""

import json
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config, random_weights
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, load_window, run_golden
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import golden_floor

TOKENS = list(range(10))
SEGMENT = 4  # segments [0, 4), [4, 8), [8, 10): the last one is partial


class Interrupted(Exception):
    pass


def make_run(run_dir: Path, *, fail_after: int | None = None, identity: str = "tiny"):
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
            identity={"name": identity},
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
        assert torch.equal(full["output"], resumed["output"]), index
        for name, tensor in full["state"].items():
            assert tensor.dtype == torch.float64 and torch.equal(tensor, resumed["state"][name]), (index, name)

    # Segmenting changes nothing but rounding: the whole sequence in one call agrees to FP64 precision.
    whole_output, whole_state = kda_forward_reference(table.unsqueeze(0), weights, config, dtype=torch.float64)
    output, _ = load_window(tmp_path / "full", 0, 10)
    torch.testing.assert_close(output, whole_output[0], rtol=0, atol=1e-12)
    last = torch.load(tmp_path / "full" / "segment_000002.pt", weights_only=True)["state"]
    torch.testing.assert_close(last["recurrent"], whole_state.recurrent, rtol=0, atol=1e-12)


def test_load_window_returns_rows_and_the_state_before_them(tmp_path: Path, expect_error) -> None:
    run, *_ = make_run(tmp_path)
    run()
    rows, state = load_window(tmp_path, 4, 9)
    full, _ = load_window(tmp_path, 0, 10)
    assert torch.equal(rows, full[4:9])
    first_segment = torch.load(tmp_path / "segment_000000.pt", weights_only=True)["state"]
    assert torch.equal(state.recurrent, first_segment["recurrent"])
    assert load_window(tmp_path, 0, 3)[1] is None
    with expect_error(ValueError, "not a multiple"):
        load_window(tmp_path, 2, 6)
    with expect_error(ValueError, "not inside"):
        load_window(tmp_path, 8, 11)


def test_resume_refuses_a_directory_from_a_different_run(tmp_path: Path, expect_error) -> None:
    run, *_ = make_run(tmp_path, identity="first")
    run()
    other, *_ = make_run(tmp_path, identity="second")
    with expect_error(ValueError, "mismatched fields: \\['identity'\\]"):
        other()


def test_manifest_records_provenance_and_the_whole_golden_token_floor(tmp_path: Path, expect_error) -> None:
    partial, *_ = make_run(tmp_path, fail_after=2)
    with expect_error(Interrupted, "simulated crash"):
        partial()
    assert "token_floor" not in json.loads((tmp_path / MANIFEST).read_text())  # written only when complete

    run, *_ = make_run(tmp_path)
    manifest = run()

    assert manifest["token_floor"] == golden_floor(load_window(tmp_path, 0, 10)[0].norm(dim=-1))
    assert manifest == json.loads((tmp_path / MANIFEST).read_text())
    assert {"code_revision", "torch", "cpu_capability", "machine"} <= set(manifest["sessions"][0])
