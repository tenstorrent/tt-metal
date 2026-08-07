# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Task 5: multi-stack perf test skeleton prompt generation.

Three invariants:
  1. Single-stack  (stacks=None or len==1) -> only TT_PERF_LAYERS appears in prompt,
     TT_PERF_STACK0_LAYERS / TT_PERF_STACK1_LAYERS do NOT appear (backward compat).
  2. Multi-stack   (len(stacks)==2)         -> TT_PERF_STACK0_LAYERS and
     TT_PERF_STACK1_LAYERS appear in prompt, TT_PERF_LAYERS does NOT.
  3. Wiring instruction includes stack paths and counts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_AGENT = Path(__file__).resolve().parents[1] / "agent"
sys.path.insert(0, str(_AGENT.parent.parent.parent))  # repo root
sys.path.insert(0, str(_AGENT.parent))  # perf_automation


def _load_perf_test_gen():
    spec = importlib.util.spec_from_file_location("perf_test_gen", str(_AGENT / "perf_test_gen.py"))
    mod = importlib.util.module_from_spec(spec)
    # Stub heavy dependencies before exec
    for _fake in ("ttnn", "torch"):
        if _fake not in sys.modules:
            sys.modules[_fake] = MagicMock()
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Minimal StackInfo stand-in (mirrors cc_optimize._op_sig_probe.StackInfo)
# ---------------------------------------------------------------------------


def _make_stack(path, count, stack_idx):
    s = SimpleNamespace()
    s.path = path
    s.count = count
    s.stack_idx = stack_idx
    return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _captured_prompt(mod, *, stacks, tmp_path):
    """Call generate_perf_test() with a fake runner that captures the prompt."""
    captured = []

    def _runner(p):
        captured.append(p)
        # Return a minimal valid perf test so the generator doesn't reject it
        return (
            "import ttnn\n" "def test_fake_perf(device):\n" "    out = ttnn.zeros([1])\n" "    assert out is not None\n"
        )

    # Build a minimal demo file so the generator can read it
    demo_rel = "demo/demo_fake.py"
    demo_path = tmp_path / demo_rel
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text("import ttnn\n" "def run():\n" "    return ttnn.zeros([1])\n")

    # Patch helpers that try to do I/O or real imports
    with (
        patch.object(mod, "_self_tracing_fns", return_value=set()),
        patch.object(mod, "_inline_inprocess_sources", return_value=""),
        patch.object(mod, "_pipeline_api_hint", return_value=""),
        patch.object(mod, "skeleton_for", return_value="<skeleton>"),
        patch.object(mod, "_sibling_component_perf_ref", return_value="", create=True),
        patch.object(mod, "_shard_slice_directive", return_value="", create=True),
        patch.object(mod, "validate_generated_perf_test", return_value=("ok_1cq", None), create=True),
    ):
        mod.generate_perf_test(
            tmp_path,
            "fake",
            demo_rel,
            runner=_runner,
            validate=False,
            stacks=stacks,
        )

    return captured[0] if captured else ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_stack_uses_perf_layers(tmp_path):
    """Single-stack (stacks=None) -> TT_PERF_LAYERS in prompt, no stack-indexed vars."""
    mod = _load_perf_test_gen()
    prompt = _captured_prompt(mod, stacks=None, tmp_path=tmp_path)

    assert "TT_PERF_LAYERS" in prompt, "Expected TT_PERF_LAYERS in single-stack prompt"
    assert "TT_PERF_STACK0_LAYERS" not in prompt, "TT_PERF_STACK0_LAYERS must not appear for single-stack"
    assert "TT_PERF_STACK1_LAYERS" not in prompt, "TT_PERF_STACK1_LAYERS must not appear for single-stack"


def test_single_stack_explicit_list_uses_perf_layers(tmp_path):
    """stacks=[one_stack] (len==1) -> still single-stack behavior."""
    mod = _load_perf_test_gen()
    single = [_make_stack("model.layers", 32, 0)]
    prompt = _captured_prompt(mod, stacks=single, tmp_path=tmp_path)

    assert "TT_PERF_LAYERS" in prompt
    assert "TT_PERF_STACK0_LAYERS" not in prompt


def test_multi_stack_uses_per_stack_vars(tmp_path):
    """Two stacks -> TT_PERF_STACK0_LAYERS and TT_PERF_STACK1_LAYERS in prompt."""
    mod = _load_perf_test_gen()
    stacks = [
        _make_stack("audio_tower.layers", 32, 0),
        _make_stack("language_model.layers", 30, 1),
    ]
    prompt = _captured_prompt(mod, stacks=stacks, tmp_path=tmp_path)

    assert "TT_PERF_STACK0_LAYERS" in prompt, "Expected TT_PERF_STACK0_LAYERS in multi-stack prompt"
    assert "TT_PERF_STACK1_LAYERS" in prompt, "Expected TT_PERF_STACK1_LAYERS in multi-stack prompt"


def test_multi_stack_does_not_emit_legacy_perf_layers(tmp_path):
    """Multi-stack prompt must NOT tell the LLM to use the legacy TT_PERF_LAYERS."""
    mod = _load_perf_test_gen()
    stacks = [
        _make_stack("audio_tower.layers", 32, 0),
        _make_stack("language_model.layers", 30, 1),
    ]
    prompt = _captured_prompt(mod, stacks=stacks, tmp_path=tmp_path)

    # The MULTI-STACK DEPTH OVERRIDE section must say "Do NOT emit PERF_LAYERS"
    assert "Do NOT emit `PERF_LAYERS`" in prompt, "Expected explicit instruction to drop PERF_LAYERS for multi-stack"


def test_wiring_instruction_includes_stack_paths_and_counts(tmp_path):
    """Wiring instruction contains stack paths and layer counts."""
    mod = _load_perf_test_gen()
    stacks = [
        _make_stack("audio_tower.layers", 32, 0),
        _make_stack("language_model.layers", 30, 1),
    ]
    prompt = _captured_prompt(mod, stacks=stacks, tmp_path=tmp_path)

    assert "audio_tower.layers" in prompt
    assert "language_model.layers" in prompt
    assert "32" in prompt
    assert "30" in prompt
