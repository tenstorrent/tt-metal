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
        " - resolve_loader: the PCC test couldn't build its torch reference because the model won't "
        "load via AutoModel.from_pretrained (non-transformers checkpoint). Call "
        "resolve_reference_loader(component) to write a shared tests/pcc/_reference_loader.py, then "
        "run_component again — do NOT hand-edit each test's loader.\n"
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
        " - shard: the component is single-device graduated and the tool wants it tensor-parallel. This "
        "is a REASONING task, not a lookup — the tool gives you principles + references, you derive the "
        "scheme. (1) call get_shard_plan(component) for the general TP principles + the tt_transformers "
        "reference implementations to study; (2) READ the nearest reference and REASON OUT this "
        "component's scheme — which weights split on which axis, which collective, expert-parallel for "
        "MoE, shard-heads-not-scan for Mamba (there is NO prescribed per-weight plan); (3) edit "
        "_stubs/<component>.py so build() shards weights with ttnn.ShardTensorToMesh(mesh_device, "
        "dim=...) (norms/biases/embeddings stay replicated) and forward() places the collective "
        "(all_gather after column-parallel, all_reduce after row-parallel); the MATH MUST NOT CHANGE — "
        "gathered output must still match the golden. (4) run_component(component, mode='shard') to "
        "validate on the mesh; if PCC fails, revise the axis/collective and retry. (5) "
        "record_result(component, ok, pcc, mode='shard') — ok writes .last_good_sharded. NEVER edit the "
        "single-device .last_good_native, and NEVER weaken the PCC assertion.\n"
        "NEVER weaken, skip, or edit the assertion in the PCC test. Re-run termination_check after each "
        "action. STOP only when can_stop=true."
    )


def _mesh_chips(mesh) -> int:
    if not mesh:
        return 1
    try:
        prod = 1
        for tok in str(mesh).lower().replace(",", "x").split("x"):
            if tok.strip():
                prod *= int(tok.strip())
        return max(prod, 1)
    except Exception:
        return 1


def _derive_shard_tp(model_id: str, mesh) -> int:
    """TP degree the mesh implies for this model, via select_parallelism (kernel viability). 1 = no
    sharding needed (single chip, or the selector fell to TP=1) → Phase 2 stays off."""
    chips = _mesh_chips(mesh)
    if chips <= 1:
        return 1
    try:
        from ..cli import evaluate_kernels, probe_model
        from ..parallelism import select_parallelism

        probe = probe_model(model_id)
        if not getattr(probe, "raw_config", None):
            return 1
        kr = evaluate_kernels(probe.raw_config, tp_grid=None)
        return max(int(select_parallelism(chips, kr).tp), 1)
    except Exception:
        return 1


def run_bringup_cc(
    *,
    model_id: str,
    demo_dir: Path,
    max_attempts: int = 2,
    pcc: float = 0.99,
    timeout_s: int = 1800,
    max_rounds: int = 1000,
    agent_bin: str = "claude",
    mesh=None,
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
    import shutil as _shutil

    _agent_abs = _shutil.which(agent_bin) or agent_bin
    _agent_dir = str(Path(_agent_abs).parent) if os.path.sep in _agent_abs else str(Path.home() / ".local" / "bin")
    mcp_env = {
        "BRINGUP_MCP_DEMO_DIR": str(demo_dir),
        "BRINGUP_MCP_MODEL_ID": model_id,
        "BRINGUP_MCP_STATE": str(state_path),
        "BRINGUP_MCP_MAX_ATTEMPTS": str(max_attempts),
        "BRINGUP_MCP_PCC": str(pcc),
        "BRINGUP_MCP_TIMEOUT": str(timeout_s),
        "BRINGUP_MCP_AGENT_BIN": _agent_abs,
        "TT_HW_PLANNER_LOADER_RESOLVER": os.environ.get("TT_HW_PLANNER_LOADER_RESOLVER", ""),
        "TT_METAL_HOME": str(repo_root),
        "PYTHONPATH": str(repo_root),
        "PATH": f"{repo_root / 'python_env' / 'bin'}{os.pathsep}{_agent_dir}{os.pathsep}/usr/bin:/bin",
    }
    _shard_flag = os.environ.get("TT_HW_PLANNER_SHARD", "")
    _shard_tp = os.environ.get("TT_HW_PLANNER_SHARD_TP", "")
    if not _shard_flag:
        _tp = _derive_shard_tp(model_id, mesh)
        if _tp > 1:
            _shard_flag, _shard_tp = "1", str(_tp)
            print(f"  [shard] mesh implies TP={_tp} → Phase 2 (shard-aware bring-up) enabled at TP={_tp}")
    if _shard_flag:
        mcp_env["TT_HW_PLANNER_SHARD"] = _shard_flag
        if _shard_tp:
            mcp_env["TT_HW_PLANNER_SHARD_TP"] = _shard_tp
            from ..parallelism import mesh_graph_descriptor_path

            _mgd = mesh_graph_descriptor_path(int(_shard_tp), repo_root)
            if _mgd:
                print(
                    f"  [shard] TP={_shard_tp} → {Path(_mgd).name} "
                    f"(applied only to shard-mode component runs, not single-device tests)"
                )
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
        "mcp__bringup-mcp__resolve_reference_loader",
        "mcp__bringup-mcp__get_shard_plan",
        "Read",
        "Edit",
        "Write",
        "Bash",
        "Grep",
        "Glob",
    ]
    try:
        from .. import reference_loader_resolver as _rlr

        if _rlr.is_enabled() and not _rlr.has_loader(Path(demo_dir)):
            _files = _rlr._repo_files(model_id)
            _hf_native = any(
                f
                in (
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "pytorch_model.bin",
                    "pytorch_model.bin.index.json",
                )
                for f in _files
            )
            if _files and not _hf_native:
                print(
                    "\n  [loader-resolver] pre-flight: non-transformers checkpoint — resolving reference loader once before the agent loop ..."
                )
                _pf = _rlr.resolve(
                    model_id=model_id,
                    demo_dir=Path(demo_dir),
                    failure_text="pre-flight: repo ships no HF-format weights (non-transformers checkpoint)",
                    agent_bin=_agent_abs,
                    cwd=repo_root,
                )
                print(f"  [loader-resolver] pre-flight result: {_pf.get('resolved')} ({_pf.get('reason')})")
    except Exception as _pf_exc:
        print(f"  [loader-resolver] pre-flight skipped: {type(_pf_exc).__name__}: {_pf_exc}")
    prompt = _bringup_cc_prompt(model_id, Path(demo_dir), pcc)
    sep = "=" * 78
    _seen = {"grad": set(), "shard": set()}

    def _banner(title):
        print()
        print(sep)
        print(f"  {title}")
        print(sep)

    def _announce_graduations(st):
        cur_g = set(st.get("graduated") or [])
        cur_s = set(st.get("shard_graduated") or [])
        for c in sorted(cur_g - _seen["grad"]):
            print(f"  ✓ `{c}` GRADUATED to native TTNN (PCC-verified)")
        for c in sorted(cur_s - _seen["shard"]):
            print(f"  ✓ `{c}` SHARD-GRADUATED on the mesh (gathered-PCC)")
        _seen["grad"], _seen["shard"] = cur_g, cur_s

    def _pre_round(round_no, st):
        _announce_graduations(st)
        cur_g = st.get("graduated") or []
        cur_s = st.get("shard_graduated") or []
        _extra = f", sharded {len(cur_s)}" if cur_s else ""
        _banner(
            f"BRING-UP (cc) round {round_no} for {model_id}: target=`{st.get('next_op') or '?'}` "
            f"rung={st.get('next_rung') or '?'} (graduated {len(cur_g)}{_extra}) → invoke claude → gate"
        )

    _banner(f"Step 6/6  Bring-up (cc engine) — harness loop on the per-component gate for {model_id}")
    res = cc_harness.run_cc_loop(
        prompt=prompt,
        mcp_config_path=cfg_path,
        allowed_tools=allowed,
        cwd=repo_root,
        env=env,
        gate_fn=gate_fn,
        max_rounds=max_rounds,
        claude_bin=agent_bin,
        pre_round=_pre_round,
    )
    final = gate_fn()
    _announce_graduations(final)
    _grad = sorted(final.get("graduated") or [])
    _shard = sorted(final.get("shard_graduated") or [])
    _banner(f"BRING-UP (cc) {'DONE' if final.get('can_stop') else 'INCOMPLETE'} for {model_id}")
    print(f"  rounds={res['rounds']}  graduated={len(_grad)}: {', '.join(_grad) or '-'}")
    if _shard:
        print(f"  shard-graduated={len(_shard)}: {', '.join(_shard)}")
    try:
        from ..cli import _format_compute_split, _format_op_split

        for ln in _format_compute_split(model_id, label="compute split (TT device vs CPU)"):
            print(ln)
        for ln in _format_op_split(model_id, label="operations"):
            print(ln)
    except Exception:
        pass
    print(sep)
    return 0 if final.get("can_stop") else 1
