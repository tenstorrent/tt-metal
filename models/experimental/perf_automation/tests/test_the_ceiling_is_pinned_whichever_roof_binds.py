# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A ceiling is pinned or it is not, and which roof binds is not the question.

Four inputs build the two roofs. They were anchored one at a time, each after it moved something:

    memory_ms  = bytes / BW      bytes  KIND_STAGE_BYTES / KIND_ACTIVE_BYTES   BW is a constant
    compute_ms = FLOPs / peak    peak   KIND_PEAK_FLOPS
                                 FLOPs  blocks[root].matmul_params             <- was loose

Each gap was found the same way and fixed the same way, and the last one survived because nothing
had happened to expose it yet. matmul_params lives in the arch mirror, written `{**prev, **keep}`,
last-write-wins. The mirror argues it is safe to cache without expiry because "a dtype or grid knob
... cannot change how many towers the model has" -- true of the towers, and NOT true of this figure:
matmul_params subtracts the gathers the profile OBSERVED, so a run observing a different gather set
recomputes it and the compute roof moves under a measurement that did not.

The rule this file exists to hold: EVERY input to EVERY roof is write-once, so no stage can be graded
against a target that moved, whether it binds on memory or on compute. A stage that is compute-bound
today and memory-bound after a dtype win must be scored against the same two roofs both times.
"""
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))


@pytest.fixture()
def led(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    import cc_optimize.measurements as M

    monkeypatch.setattr(
        M, "ledger_path", lambda model="", task="": tmp_path / ("%s_%s.jsonl" % (model or "m", task or "main"))
    )
    return M


def test_every_roof_input_has_an_anchor_kind(led):
    """The registry of what must not move. A new ceiling input added without one is the next bug."""
    for kind in ("KIND_ACTIVE_BYTES", "KIND_STAGE_BYTES", "KIND_PEAK_FLOPS", "KIND_MATMUL_PARAMS", "KIND_FLOOR"):
        assert hasattr(led, kind), kind


def test_matmul_params_is_write_once(led):
    """The compute numerator, pinned. A second, different count must not replace it."""
    assert led.anchor(led.KIND_MATMUL_PARAMS, 3611483136.0, depth="language_model", model="m") == 3611483136.0
    assert led.anchor(led.KIND_MATMUL_PARAMS, 4014000000.0, depth="language_model", model="m") == 3611483136.0
    assert led.anchor_value(led.KIND_MATMUL_PARAMS, depth="language_model", model="m", task="main") == 3611483136.0


def test_it_is_keyed_per_section_so_shared_subtrees_agree(led):
    """prefill and decode run the same subtree; two pins for it would let them disagree about how
    many parameters it multiplies."""
    led.anchor(led.KIND_MATMUL_PARAMS, 3611483136.0, depth="language_model", model="m")
    led.anchor(led.KIND_MATMUL_PARAMS, 637000000.0, depth="audio_tower", model="m")
    assert led.anchor_value(led.KIND_MATMUL_PARAMS, depth="language_model", model="m", task="main") == 3611483136.0
    assert led.anchor_value(led.KIND_MATMUL_PARAMS, depth="audio_tower", model="m", task="main") == 637000000.0


def test_the_producer_prefers_the_pin_over_a_recomputation(tmp_path, monkeypatch):
    """A later gather observation recomputes matmul_params; the pin must win."""
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.delenv("PERF_MCP_LEDGER", raising=False)
    import cc_optimize.measurements as M
    import cc_optimize.run as R

    monkeypatch.setattr(
        M, "ledger_path", lambda model="", task="": tmp_path / ("%s_%s.jsonl" % (model or "m", task or "main"))
    )
    root = tmp_path / "m"
    root.mkdir()
    first = R._ledger_anchor_matmul_params(root, "language_model", 3_611_483_136)
    second = R._ledger_anchor_matmul_params(root, "language_model", 4_014_000_000)
    assert first == 3_611_483_136
    assert second == 3_611_483_136, "a recomputation moved the pinned compute numerator"


def test_an_unreachable_ledger_leaves_the_computed_value_alone(tmp_path):
    """It must never cost a ceiling: no pin means the freshly computed number stands, which is the
    behaviour from before the pin existed."""
    import cc_optimize.run as R

    assert R._ledger_anchor_matmul_params("", "language_model", 123) in (0, 123)


def test_a_stage_that_changes_which_roof_binds_is_graded_against_the_same_two(led):
    """THE POINT. A dtype win can take a stage from memory-bound to compute-bound; if only the roof
    that happened to bind were pinned, the stage would be scored against a target that moved exactly
    when it mattered."""
    led.anchor(led.KIND_STAGE_BYTES, 4786.0, depth="decode", mode="bytes_mb", model="m")
    led.anchor(led.KIND_PEAK_FLOPS, 175.5e12, depth="token", model="m")
    led.anchor(led.KIND_MATMUL_PARAMS, 3611483136.0, depth="language_model", model="m")
    # every later, "better" number is refused
    led.anchor(led.KIND_STAGE_BYTES, 2393.0, depth="decode", mode="bytes_mb", model="m")
    led.anchor(led.KIND_PEAK_FLOPS, 702.0e12, depth="token", model="m")
    led.anchor(led.KIND_MATMUL_PARAMS, 1000.0, depth="language_model", model="m")
    assert led.anchor_value(led.KIND_STAGE_BYTES, depth="decode", model="m", task="main") == 4786.0
    assert led.anchor_value(led.KIND_PEAK_FLOPS, depth="token", model="m", task="main") == 175.5e12
    assert led.anchor_value(led.KIND_MATMUL_PARAMS, depth="language_model", model="m", task="main") == 3611483136.0
