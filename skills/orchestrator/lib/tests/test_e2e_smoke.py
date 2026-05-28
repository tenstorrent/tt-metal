# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration smoke for the orchestrator lib layer.

This is NOT a unit test and NOT a device test. It exercises every lib piece
(``state`` + ``dag`` + ``device`` + ``guard``) together against synthetic
fixtures built in ``tmp_path``. It does not dispatch any real worker
subagent and does not touch ``models/demos/qwen3_tts/`` or any device.
The first real device-touching run will be a future user-driven
``/bringup`` invocation; this file just proves the wiring holds.
"""

from __future__ import annotations

import pytest

from skills.orchestrator.lib import device as device_mod
from skills.orchestrator.lib.dag import eligible_blocks
from skills.orchestrator.lib.guard import verify_block
from skills.orchestrator.lib.state import (
    bootstrap,
    load_state,
    redo,
    render_log,
    save_state,
    skip,
)


# ---------------------------------------------------------------------------
# Test 1: bootstrap -> save -> load -> eligible_blocks reports architecture
# ---------------------------------------------------------------------------


def test_smoke_bootstrap_to_eligible(tmp_path):
    """Fresh bootstrap round-trips through disk and surfaces 'architecture'."""
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    path = tmp_path / state["model_slug"] / ".bringup_state.json"
    save_state(path, state)

    loaded = load_state(path)
    assert loaded["model_id"] == "Acme/Foo-1B"
    assert loaded["components"] == []
    assert eligible_blocks(loaded) == {"phase": "architecture"}


# ---------------------------------------------------------------------------
# Test 2: full pipeline of decisions through every dag rule
# ---------------------------------------------------------------------------


def _three_components():
    """Build a realistic three-component fixture: RMSNorm, MLP, Attention(deps RMSNorm)."""
    return [
        {
            "name": "RMSNorm",
            "kind": "norm",
            "depends_on": [],
            "reference_impl": "models/common/rmsnorm.py",
            "reference": {"status": "pending"},
            "ttnn": {"status": "pending"},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
        {
            "name": "MLP",
            "kind": "mlp",
            "depends_on": [],
            "reference_impl": "models/demos/llama3_70b_galaxy/tt/llama_mlp.py",
            "reference": {"status": "pending"},
            "ttnn": {"status": "pending"},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
        {
            "name": "Attention",
            "kind": "attention",
            "depends_on": ["RMSNorm"],
            "reference_impl": "models/demos/llama3_70b_galaxy/tt/llama_attention.py",
            "reference": {"status": "pending"},
            "ttnn": {"status": "pending"},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
    ]


def test_smoke_full_pipeline_decisions():
    """Walk the whole tick decision tree by direct state mutation."""
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")

    # Step 1: empty components -> architecture
    assert eligible_blocks(state) == {"phase": "architecture"}

    # Step 2: three components, all pending -> reference fan-out of all 3
    state["components"] = _three_components()
    result = eligible_blocks(state)
    assert result["phase"] == "reference"
    assert set(result["blocks"]) == {"RMSNorm", "MLP", "Attention"}

    # Step 3: mark all reference done -> device ttnn for first eligible (RMSNorm)
    for c in state["components"]:
        c["reference"] = {"status": "done", "pcc": 0.999}
    assert eligible_blocks(state) == {
        "phase": "device",
        "block": "RMSNorm",
        "worker": "ttnn",
    }

    # Step 4: RMSNorm.ttnn=done -> next pending ttnn is MLP
    state["components"][0]["ttnn"] = {"status": "done", "pcc": 0.998}
    assert eligible_blocks(state) == {
        "phase": "device",
        "block": "MLP",
        "worker": "ttnn",
    }

    # Step 5: MLP.ttnn=done -> Attention (its dep RMSNorm.ttnn is done)
    state["components"][1]["ttnn"] = {"status": "done", "pcc": 0.997}
    assert eligible_blocks(state) == {
        "phase": "device",
        "block": "Attention",
        "worker": "ttnn",
    }

    # Step 6: Attention.ttnn=failing -> routes to debug worker
    state["components"][2]["ttnn"] = {"status": "failing", "pcc": 0.5}
    assert eligible_blocks(state) == {
        "phase": "device",
        "block": "Attention",
        "worker": "debug",
    }

    # Step 7: Attention.ttnn=done -> optimization fan-out (RMSNorm first)
    state["components"][2]["ttnn"] = {"status": "done", "pcc": 0.996}
    assert eligible_blocks(state) == {
        "phase": "device",
        "block": "RMSNorm",
        "worker": "optimization",
    }

    # Step 8: all optimization=done -> pipeline done
    for c in state["components"]:
        c["optimization"] = {"status": "done"}
    assert eligible_blocks(state) == {"phase": "done"}


# ---------------------------------------------------------------------------
# Test 3: render_log surfaces components + ticks
# ---------------------------------------------------------------------------


def test_smoke_render_log_grows_with_state():
    """render_log includes block names, tick numbers, and the Recent Ticks section."""
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state["components"] = _three_components()[:2]  # RMSNorm + MLP
    state["tick_log"] = [
        {"tick": 1, "ts": "2026-05-27T14:00:00Z", "action": "reference[RMSNorm:reference]", "result": "ok"},
        {"tick": 2, "ts": "2026-05-27T14:05:00Z", "action": "reference[MLP:reference]", "result": "ok"},
    ]

    rendered = render_log(state)
    assert "RMSNorm" in rendered
    assert "MLP" in rendered
    assert "tick 1" in rendered
    assert "tick 2" in rendered
    assert "## Recent Ticks" in rendered


# ---------------------------------------------------------------------------
# Test 4: redo / skip round-trip + host-resident surfaces in log
# ---------------------------------------------------------------------------


def test_smoke_redo_skip_round_trip():
    """redo resets a phase to pending; skip flips host_resident.allowed=True and surfaces in render_log."""
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state["components"] = [
        {
            "name": "Attention",
            "kind": "attention",
            "depends_on": [],
            "reference": {"status": "done"},
            "ttnn": {"status": "failing", "pcc": 0.4, "attempts": 3},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
    ]

    redo(state, "Attention", "ttnn")
    assert state["components"][0]["ttnn"] == {"status": "pending", "attempts": 0}

    # Unknown block must raise KeyError before any mutation.
    with pytest.raises(KeyError):
        skip(state, "AnotherBlock", "ttnn", "test justification", "models/foo/ref.py")

    # Add the block, then skip it.
    state["components"].append(
        {
            "name": "AnotherBlock",
            "kind": "norm",
            "depends_on": [],
            "reference": {"status": "pending"},
            "ttnn": {"status": "pending"},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        }
    )
    skip(state, "AnotherBlock", "ttnn", "test justification", "models/foo/ref.py")
    another = state["components"][1]
    assert another["host_resident"]["allowed"] is True
    assert another["host_resident"]["justification"] == "test justification"

    rendered = render_log(state)
    assert "## Host-Resident Exceptions" in rendered
    assert "AnotherBlock" in rendered.split("## Host-Resident Exceptions", 1)[1]


# ---------------------------------------------------------------------------
# Test 5: guard.verify_block clean and failing verdicts
# ---------------------------------------------------------------------------


def test_smoke_guard_block_verdict_clean_and_failing(tmp_path):
    """verify_block returns ok on a clean block and ok=False with lint hits on a dirty one."""
    ref = tmp_path / "ref.py"
    ref.write_text("# empty reference\n")

    clean = tmp_path / "block_clean.py"
    clean.write_text("import ttnn\n" "class Block:\n" "    def forward(self, x):\n" "        return ttnn.rms_norm(x)\n")
    verdict = verify_block(clean, ["ttnn.rms_norm"], "norm", ref)
    assert verdict.ok is True

    dirty = tmp_path / "block_dirty.py"
    dirty.write_text("class Block:\n" "    def forward(self, x):\n" "        return x.cpu()\n")
    verdict2 = verify_block(dirty, ["ttnn.rms_norm"], "norm", ref)
    assert verdict2.ok is False
    assert len(verdict2.lint) >= 1


# ---------------------------------------------------------------------------
# Test 6: device.extract_defaults + device.device_info wiring
# ---------------------------------------------------------------------------


def test_smoke_device_registry_extract_defaults(tmp_path):
    """extract_defaults greps a tmp reference for known constants; device_info exposes mesh_shape."""
    ref_file = tmp_path / "model_args.py"
    ref_file.write_text("l1_small_size = 16384\n" "trace_region_size = 50_000_000\n" "mesh_shape = (1, 8)\n")

    defaults = device_mod.extract_defaults(tmp_path)
    assert defaults["l1_small_size"] == 16384
    assert defaults["trace_region_size"] == 50_000_000
    assert defaults["mesh_shape"] == (1, 8)

    assert device_mod.device_info("t3k")["mesh_shape"] == (1, 8)


# ---------------------------------------------------------------------------
# Test 7: deadlock detection surfaces blocked block + downstream
# ---------------------------------------------------------------------------


def test_smoke_deadlock_detection():
    """A blocked ttnn pins the queue; eligible_blocks reports the blocking block and its downstream."""
    state = bootstrap("Acme/Foo-1B", "n150", "wormhole_b0")
    state["components"] = [
        {
            "name": "RMSNorm",
            "kind": "norm",
            "depends_on": [],
            "reference": {"status": "done"},
            "ttnn": {"status": "blocked", "last_error": "stuck"},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
        {
            "name": "Attention",
            "kind": "attention",
            "depends_on": ["RMSNorm"],
            "reference": {"status": "done"},
            "ttnn": {"status": "pending"},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
    ]

    result = eligible_blocks(state)
    assert result["phase"] == "deadlock"
    blocking = {entry["name"]: entry["blocks_downstream"] for entry in result["blocking"]}
    assert "RMSNorm" in blocking
    assert "Attention" in blocking["RMSNorm"]
