# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Claude-Code bring-up driver for `auto-up`/`promote --engine cc`.

Drives per-component bring-up through the shared cc harness against the `bringup_mcp` deterministic
gate (which reuses the same per-component PCC runner, cap rule, and graduation snapshot contract as
the fsm loop). Graduation writes the `.py.last_good_native` snapshot, so `auto-up → promote →
emit-e2e` keep chaining exactly as before. `promote` = the same driver with `only`/resume semantics
handled by the gate (already-graduated components are skipped because their snapshot exists).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _bringup_cc_prompt(model_id: str, demo_dir: Path, pcc: float) -> str:
    return (
        f"You are bringing up the TTNN model {model_id} in {demo_dir}. Every NEW/ADAPT component has a "
        f"per-component PCC test that must PASS at PCC >= {pcc} with a NATIVE ttnn implementation.\n"
        "LOOP every iteration: call mcp__bringup-mcp__termination_check FIRST. It is the SOLE authority "
        "on whether you are done (can_stop=true) and names next_target = {unit: component, rung}. Work "
        "EXACTLY next_target.unit at EXACTLY next_target.rung — do NOT self-select another component or "
        "rung.\n"
        "Handle next_target.rung:\n"
        " - emit / repair: (1) run_component(component) to see the current status/failure; (2) if the "
        "test could NOT even run (import/collection/torch-reference error in the summary), the blocker "
        "is in the TEST HARNESS — edit tests/pcc/conftest.py (add the shim/fixture it needs), NOT the "
        "stub; otherwise edit _stubs/<component>.py into a correct NATIVE ttnn forward (never delegate "
        "to the torch reference / _get_torch_submodule); (3) run_component again to verify; (4) "
        "record_result(component, ok, pcc, failure_class) — ok=True graduates it (writes "
        ".last_good_native). If your new PCC is LOWER than a previous attempt, call "
        "restore_best(component) before trying a different fix so you don't lose ground.\n"
        " - fix_harness: the PCC test SKIPPED because the harness can't build valid inputs — the bug "
        "is in the TEST, not the stub. Fix tests/pcc/test_<component>.py (the submodule path "
        "_CANDIDATE_SUBMODULE_PATHS — e.g. add [0] for a ModuleList — and/or the synthetic-input "
        "builder _make_arg_for / sample kwargs to match the module's real forward; check _captured/"
        "<component>/ for real shapes) or tests/pcc/conftest.py; then run_component again. NEVER edit "
        "the stub to satisfy a skipping test.\n"
        " - decompose: call decompose_component(component) to split a stuck composite into children, "
        "then bring up each child. If it reports primitive/leaf, go back to repairing the component.\n"
        " - fallback: call fall_back_to_cpu(component) to retire a cap-exhausted component to CPU "
        "(mixed execution) so the pipeline still runs.\n"
        " - mark_manual: call mark_harness_skipped(component, 'manual') only when a harness-skip is "
        "truly unfixable and can't be decomposed — so the run completes honestly without counting the "
        "skip as graduated.\n"
        "NEVER weaken, skip, or edit the assertion in the PCC test. Re-run termination_check after each "
        "action. STOP only when can_stop=true."
    )


def run_bringup_cc(
    *,
    model_id: str,
    demo_dir: Path,
    max_attempts: int = 2,
    pcc: float = 0.99,
    timeout_s: int = 1800,
    max_rounds: int = 1000,
    agent_bin: str = "claude",
) -> int:
    """Run bring-up via the cc engine. Returns 0 iff the gate reports can_stop (all material components
    graduated or capped-to-fallback). Graduation is persisted via the shared .last_good_native snapshot
    contract, so promote/emit-e2e see it."""
    from .. import cc_harness

    repo_root = Path(__file__).resolve().parents[3]
    thp_dir = repo_root / "scripts" / "tt_hw_planner"
    server_path = thp_dir / "bringup_mcp.py"
    pybin = str(repo_root / "python_env" / "bin" / "python")
    if not Path(pybin).is_file():
        import sys as _sys

        pybin = _sys.executable
    state_path = Path(demo_dir) / ".bringup_cc_state.json"
    mcp_env = {
        "BRINGUP_MCP_DEMO_DIR": str(demo_dir),
        "BRINGUP_MCP_MODEL_ID": model_id,
        "BRINGUP_MCP_STATE": str(state_path),
        "BRINGUP_MCP_MAX_ATTEMPTS": str(max_attempts),
        "BRINGUP_MCP_PCC": str(pcc),
        "BRINGUP_MCP_TIMEOUT": str(timeout_s),
        "TT_METAL_HOME": str(repo_root),
        "PYTHONPATH": str(repo_root),
        "PATH": f"{repo_root / 'python_env' / 'bin'}{os.pathsep}/usr/bin:/bin",
    }
    cfg = cc_harness.build_mcp_config(pybin, server_path, mcp_env, "bringup-mcp")
    import re as _re

    cfg_path = thp_dir / f".bringup_mcp_config_{_re.sub(r'[^A-Za-z0-9._-]', '_', model_id)}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    env = dict(os.environ)
    env["TT_METAL_HOME"] = str(repo_root)
    env["PYTHONPATH"] = str(repo_root)

    def gate_fn():
        return cc_harness.gate_status(pybin, thp_dir, "bringup_mcp", mcp_env, repo_root)

    allowed = [
        "mcp__bringup-mcp__termination_check",
        "mcp__bringup-mcp__list_components",
        "mcp__bringup-mcp__run_component",
        "mcp__bringup-mcp__record_result",
        "mcp__bringup-mcp__restore_best",
        "mcp__bringup-mcp__decompose_component",
        "mcp__bringup-mcp__fall_back_to_cpu",
        "mcp__bringup-mcp__mark_harness_skipped",
        "Read",
        "Edit",
        "Write",
        "Bash",
        "Grep",
        "Glob",
    ]
    prompt = _bringup_cc_prompt(model_id, Path(demo_dir), pcc)
    print("\n  ===== BRING-UP (cc engine): harness loop on the per-component gate =====\n")
    res = cc_harness.run_cc_loop(
        prompt=prompt,
        mcp_config_path=cfg_path,
        allowed_tools=allowed,
        cwd=repo_root,
        env=env,
        gate_fn=gate_fn,
        max_rounds=max_rounds,
        claude_bin=agent_bin,
    )
    final = gate_fn()
    sep = "=" * 78
    print("\n" + sep)
    print(f"  bring-up cc: rounds={res['rounds']} can_stop={final.get('can_stop')} graduated={final.get('graduated')}")
    print(sep)
    return 0 if final.get("can_stop") else 1
