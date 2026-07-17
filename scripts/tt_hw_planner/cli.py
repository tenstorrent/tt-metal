from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .architecture import DTYPE_BYTES
from .bringup import (
    BringupError,
    REPO_ROOT,
    prepare_bringup,
    render_json as render_bringup_json,
    render_script as render_bringup_script,
    render_text as render_bringup_text,
)
from .discovery import BRINGUP_ROOT, safe_relative_to_root
from .bringup_loop import (
    autofill_stubs,
    emit_runnable_demo,
    find_demo_dir,
    next_task,
    render_json as render_bringup_loop_json,
    render_next,
    render_text as render_bringup_loop_text,
    run_bringup_loop,
    _safe_id,
    _stub_has_graduated_from_autofill,
)
from .op_classifier import (
    classify_ops_in_component,
    format_op_plan,
    summarize_ops,
)
from .op_emitter import emit_partial_stub
from .llm_synth import (
    LLMError,
    apply_all_responses,
    apply_response,
    build_handoff_master,
    emit_prompts,
    list_synth_targets,
    render_synth_json,
    render_synth_results,
    render_synth_targets,
    resolve_llm_config,
    synthesize_all_new,
    synthesize_component,
)
from .compatibility import check_compatibility
from .hardware import HARDWARE, find_box
from .kernel_constraints import evaluate_kernels
from .probe import probe_model
from .report import (
    render_compat_json,
    render_compat_table,
    render_json,
    render_markdown,
    render_table,
)
from .verdict import evaluate_all


def _dtypes_for(category: str, user: List[str], source_dtype: str = "") -> List[str]:
    if user:
        return user
    if category in ("LLM", "VLM"):
        base = ["bf16", "bfp8_b"]
        if source_dtype in ("fp8", "f8_e8m0"):
            return ["fp8", "bfp8_b", "bf16"]
        return base
    return ["bf16"]


def _parse_mesh(s: str) -> Tuple[int, int]:
    parts = s.replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"mesh must be 'rows,cols' or 'rowsxcols' (got '{s}')")
    return int(parts[0]), int(parts[1])


def _download_model_snapshot(model_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for --download-first.") from exc
    snapshot_download(repo_id=model_id, repo_type="model", resume_download=True)


from .commands.plan import cmd_plan  # noqa: F401


def _filter_verdict_by_divisibility(verdict, kernel_report, all_mesh_verdict=None):
    from .bringup import mesh_device_for
    from .kernel_constraints import Severity
    from .verdict import FitVerdict, pick_best

    def row_passes(row) -> bool:
        tp = max(1, int(row.mesh_shape[1]))
        for f in kernel_report.findings_by_tp.get(tp, []):
            if not f.passes and f.severity == Severity.BLOCKER:
                return False
        label, _ = mesh_device_for(row.box.arch, row.mesh_shape)
        if label is None:
            return False
        return True

    pool = all_mesh_verdict.rows if all_mesh_verdict is not None else verdict.rows
    feasible = [r for r in pool if row_passes(r)]
    if not feasible:
        return verdict
    new_best = pick_best(feasible)
    notes = list(verdict.notes)
    if (
        verdict.best is not None
        and new_best is not None
        and (verdict.best.mesh_shape != new_best.mesh_shape or verdict.best.box.name != new_best.box.name)
    ):
        notes.append(
            f"recommendation bumped to {new_best.box.name} mesh "
            f"[{new_best.mesh_shape[0]},{new_best.mesh_shape[1]}]: original best "
            f"[{verdict.best.mesh_shape[0]},{verdict.best.mesh_shape[1]}] fails "
            f"kernel divisibility at TP={verdict.best.mesh_shape[1]}."
        )
    return FitVerdict(rows=verdict.rows, best=new_best, notes=notes)


def _render_weights_only(probe, boxes, dtypes, args) -> int:
    from .verdict import FitRow, FitVerdict, Tightness, pick_best
    from .parallelism import ParallelConfig, ShardedMemory

    rows: List[FitRow] = []
    for box in boxes:
        if args.all_meshes:
            meshes = list(box.mesh_shapes)
        else:
            max_tp = max(r * c for r, c in box.mesh_shapes)
            meshes = [next(s for s in box.mesh_shapes if s[0] * s[1] == max_tp)]
        for dtype in dtypes:
            per_param = DTYPE_BYTES.get(dtype, 2.0)
            on_disk_per_param = probe.bytes_per_param_on_disk or 2.0
            scale = per_param / on_disk_per_param if on_disk_per_param else 1.0
            scaled_weights = int(probe.weight_bytes_total * scale)
            for shape in meshes:
                tp = shape[0] * shape[1]
                pcfg = ParallelConfig(tp=tp)
                per_chip_w = scaled_weights // tp
                per_chip_total = per_chip_w + 1_000_000_000
                usable_gb = box.usable_per_chip_gb(tp)
                per_chip_gb = per_chip_total / 1e9
                headroom_gb = usable_gb - per_chip_gb
                rows.append(
                    FitRow(
                        box=box,
                        dtype=dtype,
                        mesh_shape=shape,
                        parallel=pcfg,
                        sharded=ShardedMemory(
                            weights_bytes=per_chip_w, kv_cache_bytes=0, activation_bytes=1_000_000_000
                        ),
                        usable_per_chip_gb=usable_gb,
                        per_chip_gb=per_chip_gb,
                        headroom_gb=headroom_gb,
                        tightness=Tightness.classify(headroom_gb, usable_gb),
                    )
                )

    verdict = FitVerdict(rows=rows, best=pick_best(rows), notes=["weights-only estimate; no transformer memory model"])

    if args.format == "json":
        print(render_json(probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes))
    elif args.format == "markdown":
        print(render_markdown(probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes))
    else:
        print(
            render_table(
                probe, verdict, args.batch, args.seq, args.kv_dtype, dtypes, show_overhead=not args.no_overhead_detail
            )
        )
    return 0


from .commands.compat import cmd_compat  # noqa: F401


from .commands.scaffold import cmd_scaffold  # noqa: F401


from .commands.bringup import cmd_bringup  # noqa: F401


def _can_load_with_transformers(model_id: str) -> Tuple[bool, str]:
    try:
        import transformers
    except ImportError:
        return False, "transformers is not importable in this env"
    try:
        transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return True, ""
    except Exception as exc:
        installed = getattr(transformers, "__version__", "unknown")
        return False, f"transformers (v{installed}) can't load `{model_id}`: {exc}"


def _purge_transformers_modules() -> None:
    for mod in list(sys.modules):
        if mod == "transformers" or mod.startswith("transformers."):
            del sys.modules[mod]


_TRANSFORMERS_PIN = "transformers==5.8.1"


def _upgrade_transformers_from_upstream() -> Tuple[bool, str]:
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        _TRANSFORMERS_PIN,
    ]
    print(f"  Running: {' '.join(cmd)}")
    print("  (this typically takes 20-40s — PyPI wheel, no source build)")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "pip install timed out after 300s"
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-20:]
        return False, "\n".join(tail)
    return True, ""


_ENV_FIX_ATTEMPTED_FLAG = "TT_HW_PLANNER_ENV_FIX_ATTEMPTED"


def _classify_load_error(msg: str) -> str:
    """2026-05-23: classify why `transformers` can't load a model.

    The pre-flight previously assumed every load failure was a
    `transformers` version mismatch and would attempt
    `pip install -U transformers`. For gated-repo / auth / missing-
    dep failures this is wasted time and a misleading diagnosis.

    Returns one of:
      * "gated"   - HF returned 403 / "Access to model X is restricted"
                    / "gated repo". Needs HF_TOKEN with access, or
                    the repo's access-request page.
      * "auth"    - 401 / token invalid / not logged in.
      * "missing" - 404 / model id doesn't exist on the Hub.
      * "dep"     - ImportError / missing optional dep
                    (Pillow, sentencepiece, torchvision, etc.)
      * "version" - actual API mismatch (TypeError / AttributeError on
                    config fields / unknown kwarg / unknown model_type).
                    This is the only class where upgrading transformers
                    might actually help.
      * "unknown" - couldn't tell.
    """
    m = msg.lower()
    if "403" in m or "gated repo" in m or "is restricted" in m or "not in the authorized list" in m:
        return "gated"
    if (
        "401" in m
        or "invalid token" in m
        or "not logged in" in m
        or "huggingface_hub.errors.localtokennotfounderror" in m
    ):
        return "auth"
    if "404" in m or "repository not found" in m:
        return "missing"
    if "no module named" in m or "importerror" in m or "modulenotfounderror" in m:
        return "dep"
    if (
        "unrecognized configuration class" in m
        or "does not recognize this architecture" in m
        or "unexpected keyword" in m
        or "got an unexpected keyword argument" in m
        or "unsupported operand" in m
        or "object has no attribute" in m
    ):
        return "version"
    return "unknown"


def _print_gated_or_auth_guidance(model_id: str, kind: str) -> None:
    """Actionable guidance for gated/auth/missing-model failures."""
    sep = "-" * 72
    print(f"  {sep}")
    if kind == "gated":
        print(f"  GATED MODEL: `{model_id}` requires manual access approval.")
        print()
        print("  To unblock:")
        print(f"    1. Open  https://huggingface.co/{model_id}  in a browser,")
        print("       sign in, and click 'Request access'. Approval can be")
        print("       instant (click-through) or take hours (manual review).")
        print("    2. Once approved, run `huggingface-cli login` with a token")
        print("       that has read access (https://huggingface.co/settings/tokens).")
        print("    3. Re-run this command. Or set HF_TOKEN=hf_... in the env.")
    elif kind == "auth":
        print(f"  AUTH FAILURE for `{model_id}`.")
        print()
        print("  To unblock:")
        print("    1. Run `huggingface-cli login` with a valid token, OR")
        print("    2. Set HF_TOKEN=hf_... in your shell, then re-run.")
        print("    Get a token at https://huggingface.co/settings/tokens .")
    else:
        print(f"  MODEL NOT FOUND: `{model_id}` does not exist on the Hub")
        print("  (or the org/repo is private and you have no access).")
        print("  Double-check the spelling on https://huggingface.co .")
    print(f"  {sep}")


def _loader_resolver_available(model_id: str) -> bool:
    try:
        from . import reference_loader_resolver as _rlr
    except Exception:
        return False
    try:
        if _rlr.is_enabled():
            return True
        dd = _find_demo_dir_safe(model_id)
        return bool(dd and _rlr.has_loader(dd))
    except Exception:
        return False


def _auto_enable_loader_resolver() -> None:
    os.environ["TT_HW_PLANNER_LOADER_RESOLVER"] = "1"


def _preflight_load_with_autofix(model_id: str, *, allow_fix: bool) -> bool:
    ok, msg = _can_load_with_transformers(model_id)
    if ok:
        print(f"  transformers can load `{model_id}` locally   [ok]")
        return True

    print("  transformers can NOT load this model in this env:")
    for line in msg.splitlines():
        print(f"    {line}")

    kind = _classify_load_error(msg)
    if kind in {"gated", "auth", "missing"}:
        _print_gated_or_auth_guidance(model_id, kind)
        return False
    if kind == "dep":
        print(
            "  This is a MISSING-DEP issue, not a transformers version\n"
            "  issue. Read the ImportError above and pip-install the\n"
            "  named package (e.g. `pip install pillow sentencepiece\n"
            "  torchvision`). Upgrading transformers will not help."
        )
        return False
    _non_hf_native = False
    try:
        from . import reference_loader_resolver as _rlr

        _rfiles = _rlr._repo_files(model_id)
        _non_hf_native = bool(_rfiles) and not any(
            f
            in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
            for f in _rfiles
        )
    except Exception:
        _non_hf_native = False
    if _non_hf_native or kind in {"unknown", "version"} or _loader_resolver_available(model_id):
        _auto_enable_loader_resolver()
        print(
            "  transformers cannot load this architecture here (no model_type/auto_map, an\n"
            "  unrecognized architecture, or a non-transformers checkpoint). Auto-enabling the\n"
            "  reference-loader path: discovery/scaffold/capture build the module tree via the\n"
            "  model's own package or trust_remote_code code plus a synthesized\n"
            "  tests/pcc/_reference_loader.py, and every per-component PCC test imports it.\n"
            "  No flag needed. Correctness stays PCC-gated."
        )
        return True

    if not allow_fix:
        print("  --no-env-fix: skipping the automatic upgrade.\n" f'  Manual fix: pip install -U "{_TRANSFORMERS_PIN}"')
        return False

    if os.environ.get(_ENV_FIX_ATTEMPTED_FLAG):
        print(
            "  Already attempted an automatic upgrade in this invocation and "
            "the model still can't load — this is no longer a transformers "
            "version issue. Likely a missing dep (e.g. Pillow / sentencepiece "
            "/ torchvision) or a private/gated repo. Bring-up will continue; "
            "address the dep and re-run."
        )
        return False

    print("  Attempting automatic fix (upgrading transformers from upstream)...")
    fixed, log_tail = _upgrade_transformers_from_upstream()
    if not fixed:
        print("  Automatic fix failed. Pip output tail:")
        for line in log_tail.splitlines():
            print(f"      {line}")
        print(
            "  Bring-up will continue, but CPU-fallback stubs will error at " "runtime until you fix the env by hand."
        )
        return False

    print(
        "  Upgrade complete. Re-executing this command with the upgraded "
        "transformers so the fresh version is picked up cleanly...\n"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    new_env = dict(os.environ)
    new_env[_ENV_FIX_ATTEMPTED_FLAG] = "1"
    os.execvpe(sys.executable, [sys.executable, "-m", "scripts.tt_hw_planner", *sys.argv[1:]], new_env)
    return False


_AGENT_BIN_CANDIDATES = {
    "cursor": (
        "/home/ttuser/.local/bin/agent",
        os.path.expanduser("~/.local/bin/agent"),
        "agent",
    ),
    "claude": (
        "/home/ttuser/.local/bin/claude",
        os.path.expanduser("~/.local/bin/claude"),
        "claude",
    ),
}


def _resolve_agent_bin(provider: str) -> Optional[str]:
    import shutil

    for cand in _AGENT_BIN_CANDIDATES.get(provider, ()):
        path = shutil.which(cand) if "/" not in cand else (cand if os.access(cand, os.X_OK) else None)
        if path:
            return path
    return None


def _check_agent_ready(provider: str) -> Tuple[bool, str]:
    bin_path = _resolve_agent_bin(provider)
    if provider == "cursor":
        if not bin_path:
            return False, (
                "Cursor `agent` CLI not installed. Run once:\n"
                "    curl https://cursor.com/install -fsS | bash\n"
                "Then authenticate:\n"
                "    ~/.local/bin/agent login    (set NO_OPEN_BROWSER=1 over SSH)\n"
                "Then re-run this command with --auto."
            )
        try:
            import subprocess

            out = subprocess.run([bin_path, "models"], capture_output=True, text=True, timeout=15)
            if out.returncode != 0 or "No models available" in (out.stdout or ""):
                return False, (
                    f"Cursor `agent` CLI installed at {bin_path} but not authenticated.\n"
                    f"Run:\n    {bin_path} login\n"
                    "(over SSH/no browser: NO_OPEN_BROWSER=1 agent login)\n"
                    "Or use --auto-agent claude (requires ANTHROPIC_API_KEY)."
                )
        except Exception as exc:
            return False, f"Cursor `agent` CLI check failed: {exc}"
        return True, bin_path
    if provider == "claude":
        if not bin_path:
            return False, (
                "Claude Code CLI not installed. Run once:\n"
                "    curl -fsSL https://claude.ai/install.sh | bash\n"
                "Then either:\n"
                "    export ANTHROPIC_API_KEY=<your-key>     (recommended for headless)\n"
                "  or run `claude` once interactively to log in via OAuth.\n"
                "Then re-run this command with --auto --auto-agent claude."
            )

        try:
            import subprocess

            probe = subprocess.run(
                [
                    bin_path,
                    "-p",
                    "--dangerously-skip-permissions",
                    "--add-dir",
                    str(REPO_ROOT),
                    "--model",
                    "sonnet",
                    "--output-format",
                    "text",
                    "Reply with exactly: OK",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                stdin=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode != 0:
                detail = (probe.stdout or "") + ("\n" + probe.stderr if probe.stderr else "")
                hint = detail.strip()[:400] or "(no output)"
                return False, (
                    f"Claude CLI installed at {bin_path} but non-interactive auth check failed.\n"
                    f"CLI output: {hint}\n"
                    "Run `claude` once and complete `/login`, or set ANTHROPIC_API_KEY."
                )
        except Exception as exc:
            return False, f"Claude CLI readiness check failed: {exc}"
        return True, bin_path
    return False, f"unknown --auto-agent provider: {provider!r}"


_API_KEY_ENV_VAR = {
    "cursor": "CURSOR_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}

_PROVIDER_LABEL = {
    "cursor": "Cursor",
    "claude": "Anthropic (Claude)",
}


def _summarize_bringup_status(model_id: str) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    from .bringup_loop import find_demo_dir

    demo_dir = find_demo_dir(model_id)
    # ADAPT restored 2026-06-01 with iterate-loop integration.
    counts = {"REUSE": 0, "ADAPT": 0, "NEW": 0}
    rows: List[Tuple[str, str]] = []
    if demo_dir is None:
        return counts, rows
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return counts, rows
    try:
        data = json.loads(status_path.read_text())
    except Exception:
        return counts, rows
    for comp in data.get("components", []):
        s = comp.get("status", "?")
        counts[s] = counts.get(s, 0) + 1
        rows.append((comp.get("name", "?"), s))
    return counts, rows


def _list_component_pcc_tests(demo_dir: Path, *, only: Optional[List[str]] = None) -> List[str]:
    """Return the canonical per-component PCC test files for this demo,
    scoped to the components listed in `bringup_status.json` (status
    NEW or ADAPT — both need per-component PCC validation).

    NEW = LLM writes from scratch; ADAPT = canonical wrapper, may refine.
    REUSE components are trusted as-is (no per-component test).

    This ignores stale/leftover `test_*.py` from earlier template-based
    scaffolds (e.g. `test_sam2_hiera_tiny_for_image_classification.py` that
    references HF classes which don't exist for the current model). Those
    would fail at pytest collection and poison any "verify-end-to-end" sweep.
    """
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return []
    try:
        data = json.loads(status_path.read_text())
    except Exception:
        return []
    components: List[str] = []
    for comp in data.get("components", []):
        if comp.get("status") not in ("NEW", "ADAPT"):
            continue
        name = str(comp.get("name", "")).strip()
        if name and (only is None or name in only):
            components.append(name)
    pcc_dir = demo_dir / "tests" / "pcc"
    out: List[str] = []
    for c in sorted(set(components)):
        test_path = pcc_dir / f"test_{_safe_id(c)}.py"
        if test_path.is_file():
            abs_path = test_path.resolve()
            try:
                out.append(str(safe_relative_to_root(abs_path)))
            except ValueError:
                out.append(str(abs_path))
    return out


def _auto_iteration_blockers(model_id: str) -> Tuple[List[str], List[str]]:
    from .bringup_loop import find_demo_dir

    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        return ["(demo-not-found)"], []
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return ["(bringup-status-missing)"], []
    try:
        data = json.loads(status_path.read_text())
    except Exception:
        return ["(bringup-status-unreadable)"], []
    ungraduated: List[str] = []
    smoke_tests: List[str] = []
    for comp in data.get("components", []):
        # ADAPT components also need per-component validation; they get
        # canonical-wrapper stubs and per-component PCC tests just like NEW.
        if comp.get("status") not in ("NEW", "ADAPT"):
            continue
        name = str(comp.get("name", "")).strip()
        if not name:
            continue
        safe = _safe_id(name)
        stub_path = demo_dir / "_stubs" / f"{safe}.py"
        if not _stub_has_graduated_from_autofill(stub_path):
            ungraduated.append(name)
        test_path = demo_dir / "tests" / "pcc" / f"test_{safe}.py"
        if test_path.is_file():
            head = test_path.read_text(errors="ignore")[:1200]
            if "Phase-1 SMOKE test" in head:
                smoke_tests.append(name)
    return sorted(set(ungraduated)), sorted(set(smoke_tests))


def _classify_components_from_compat(model_id: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "reuse": [],
        "adapt": [],
        "new_native": [],
        "new_fallback": [],
    }
    try:
        from .compatibility import check_compatibility, Status as _Status
        from .probe import probe_model

        probe = probe_model(model_id)
        if probe is None or not probe.raw_config:
            return out
        report = check_compatibility(model_id, probe.raw_config)
    except Exception:
        return out
    for r in report.results or []:
        if not r.needed:
            continue
        name = r.block.name
        if r.status == _Status.SUPPORTED:
            out["reuse"].append(name)
        elif r.status == _Status.PARTIAL:
            out["adapt"].append(name)
        elif r.status == _Status.MISSING:
            out["new_fallback"].append(name)
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


def _classify_components(model_id: str) -> Dict[str, List[str]]:
    from .bringup_loop import find_demo_dir

    demo_dir = find_demo_dir(model_id)
    # ADAPT removed 2026-05-31. Keep "adapt" key as empty list so any
    # consumer that still reads it gets [] rather than KeyError.
    out: Dict[str, List[str]] = {
        "reuse": [],
        "adapt": [],
        "new_native": [],
        "new_fallback": [],
    }
    if demo_dir is None:
        return _classify_components_from_compat(model_id)
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return _classify_components_from_compat(model_id)
    try:
        data = json.loads(status_path.read_text())
    except Exception:
        return out
    for comp in data.get("components", []):
        name = str(comp.get("name", "")).strip()
        status = str(comp.get("status", "?")).strip()
        if not name:
            continue
        if status == "REUSE":
            out["reuse"].append(name)
        elif status == "ADAPT":
            out["adapt"].append(name)
        elif status == "NEW":
            safe = _safe_id(name)
            stub_path = demo_dir / "_stubs" / f"{safe}.py"
            if _stub_has_graduated_from_autofill(stub_path):
                out["new_native"].append(name)
            else:
                out["new_fallback"].append(name)
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


def _runtime_fallback_paths(model_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (`jsonl_path`, `persisted_path`) inside the demo dir or (None, None)."""
    from .bringup_loop import find_demo_dir

    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        return None, None
    return (
        demo_dir / "_runtime_fallbacks.jsonl",
        demo_dir / "_runtime_fallbacks.json",
    )


def _truncate_runtime_fallback_log(model_id: str) -> None:
    """Wipe the per-iteration JSONL just before invoking pytest so the drain
    only sees events from THIS run, not stale events from a previous run."""
    jsonl, _ = _runtime_fallback_paths(model_id)
    if jsonl is None:
        return
    try:
        if jsonl.is_file():
            jsonl.unlink()
    except Exception:
        pass


_FALLBACK_MARKER_RE = re.compile(
    r"\[(CONV2D|ACTIVATION|MATMUL|LINEAR|LAYERNORM|RMSNORM|EMBEDDING)_CPU_FALLBACK\]\s+" r"(?P<helper>_apply_\S+?)[:\s]"
)


def _parse_fallback_markers_from_text(text: str) -> List[Tuple[str, str]]:
    """Extract `(kind, helper)` tuples from any captured pytest output.

    This is a robust complement to the JSONL writer for legacy stubs that
    only printed the human-readable marker. Returned tuples are unique and
    in first-seen order."""
    out: List[Tuple[str, str]] = []
    seen = set()
    for m in _FALLBACK_MARKER_RE.finditer(text or ""):
        kind = m.group(1).lower()
        helper = m.group("helper").rstrip(":")
        key = (kind, helper)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _drain_runtime_fallback_log(model_id: str, *, stdout_text: str = "") -> Dict[str, Dict[str, List[str]]]:
    """Read the JSONL (and optionally a captured pytest stdout dump) and
    return `{component: {kind: [helpers...], "helpers": [...]}}`.

    Components are inferred by:
      1. `event['component']` field on each JSONL row (preferred).
      2. Mapping the helper name back to a component via the on-disk stub
         that declares it (for legacy stubs that didn't include the field).
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    jsonl, _ = _runtime_fallback_paths(model_id)

    def _ensure(comp: str) -> Dict[str, List[str]]:
        if comp not in out:
            out[comp] = {"helpers": [], "kinds": []}
        return out[comp]

    def _push(comp: str, kind: str, helper: str) -> None:
        bucket = _ensure(comp)
        if helper not in bucket["helpers"]:
            bucket["helpers"].append(helper)
        if kind not in bucket["kinds"]:
            bucket["kinds"].append(kind)
        bucket.setdefault(kind, [])
        if helper not in bucket[kind]:
            bucket[kind].append(helper)

    if jsonl is not None and jsonl.is_file():
        try:
            for line in jsonl.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                comp = str(ev.get("component", "")).strip()
                helper = str(ev.get("helper", "")).strip()
                kind = str(ev.get("kind", "")).strip().lower()
                if not comp or not helper:
                    continue
                _push(comp, kind or "unknown", helper)
        except Exception:
            pass

    if stdout_text:
        markers = _parse_fallback_markers_from_text(stdout_text)
        if markers:
            from .bringup_loop import find_demo_dir

            demo_dir = find_demo_dir(model_id)
            stub_helper_index: Dict[str, str] = {}
            if demo_dir is not None:
                stubs_dir = demo_dir / "_stubs"
                if stubs_dir.is_dir():
                    for stub_path in stubs_dir.glob("*.py"):
                        try:
                            txt = stub_path.read_text(errors="ignore")
                        except Exception:
                            continue
                        for h in re.findall(r"def\s+(_apply_[A-Za-z0-9_]+)", txt):
                            stub_helper_index.setdefault(h, stub_path.stem)
            for kind, helper in markers:
                comp = stub_helper_index.get(helper, "")
                if not comp:
                    continue
                _push(comp, kind, helper)

    return out


def _persist_runtime_fallbacks(
    model_id: str, drained: Dict[str, Dict[str, List[str]]], tested_components: Optional[List[str]] = None
) -> None:
    """Merge the drained events into `_runtime_fallbacks.json`.

    The persisted file is the source of truth consumed by
    `_compute_op_split`. We OVERWRITE the entries for the components that
    fired in this drain so the report reflects the most recent run; other
    components are preserved (they might be on-device still, no events
    means they didn't fire a fallback this run).

    `tested_components` (optional): the list of component names whose
    pytest files just ran. If supplied, any tested component NOT present
    in `drained` is treated as "ran cleanly this iteration" — its entry
    in the persisted file is cleared. Without this, a successful
    LLM rewrite that removes the CPU fallback would leave the prior
    stale entry on disk forever, and the auto-iterate loop would keep
    targeting the component as still partial-CPU. Components NOT in
    `tested_components` are preserved (focused reruns must not zero out
    untested components' state).
    """
    _, persisted_path = _runtime_fallback_paths(model_id)
    if persisted_path is None:
        return
    cur: Dict[str, Dict[str, List[str]]] = {}
    if persisted_path.is_file():
        try:
            cur = json.loads(persisted_path.read_text())
        except Exception:
            cur = {}

    if tested_components:
        drained_keys = set(drained.keys())
        drained_safe_keys = {_safe_id(k) for k in drained_keys}
        for tc in tested_components:
            tc_safe = _safe_id(tc)
            if tc not in drained_keys and tc_safe not in drained_safe_keys:
                cur.pop(tc, None)
                cur.pop(tc_safe, None)

    for comp, info in drained.items():
        cur[comp] = info
    try:
        persisted_path.write_text(json.dumps(cur, indent=2, sort_keys=True))
    except Exception:
        pass


def _load_persisted_runtime_fallbacks(model_id: str) -> Dict[str, Dict[str, List[str]]]:
    _, persisted_path = _runtime_fallback_paths(model_id)
    if persisted_path is None or not persisted_path.is_file():
        return {}
    try:
        return json.loads(persisted_path.read_text()) or {}
    except Exception:
        return {}


def _runtime_fallback_helper_count(model_id: str, component: str) -> int:
    """How many distinct `_apply_*` helpers in `component` ran on CPU at
    least once across recent pytest runs? Used to downgrade the on-device
    op count for the compute split."""
    info = _load_persisted_runtime_fallbacks(model_id).get(component, {})
    helpers = info.get("helpers", []) if isinstance(info, dict) else []
    return len(helpers)


def _runtime_fallback_details(model_id: str, component: str) -> Dict[str, List[str]]:
    """Return a dict describing WHICH helpers in `component` are running
    on CPU at runtime, and what kind of op each one wraps. The result
    looks like::

        {
            "helpers": ["_apply_backbone_patch_embed_projection"],
            "kinds":   ["conv2d"],
        }

    Used by the iteration loop to (a) decide a component is partial-CPU
    and worth another iteration, and (b) inject a directive into the LLM
    prompt naming the specific helpers it must convert to native ttnn.
    Empty dict for a clean component."""
    safe = _safe_id(component)
    info = _load_persisted_runtime_fallbacks(model_id).get(safe, {})
    if not isinstance(info, dict):
        return {}
    helpers_raw = info.get("helpers", [])
    kinds_raw = info.get("kinds", [])
    helpers = list(helpers_raw) if isinstance(helpers_raw, list) else []
    kinds = list(kinds_raw) if isinstance(kinds_raw, list) else []
    if not helpers:
        return {}
    return {"helpers": helpers, "kinds": kinds}


def _partial_cpu_components(model_id: str) -> List[str]:
    """Return the list of components whose TTNN stub passes PCC but
    still has at least one `_apply_*` helper that fell back to CPU at
    runtime. The auto-iterate loop adds these to its candidate pool so
    the LLM gets another shot at making them fully on-device, instead
    of declaring "graduated" the moment PCC passes."""
    split = _compute_split(model_id)
    names = split.get("new_native_partial_cpu_names") or []
    if not isinstance(names, list):
        return []
    return [str(n) for n in names if n]


def _compute_split(model_id: str) -> Dict[str, int]:
    cats = _classify_components(model_id)

    from .bringup_loop import find_demo_dir
    from .final_categorization import reuse_adapt_on_device

    _demo = find_demo_dir(model_id)
    verified_reuse: set = set()
    if _demo is not None:
        _status_path = _demo / "bringup_status.json"
        if _status_path.is_file():
            try:
                _clist = json.loads(_status_path.read_text()).get("components", []) or []
                verified_reuse = reuse_adapt_on_device(_demo, _clist)
            except Exception:
                verified_reuse = set()

    reuse_dev = [n for n in cats["reuse"] if n in verified_reuse]
    reuse_cpu = [n for n in cats["reuse"] if n not in verified_reuse]
    adapt_dev = [n for n in cats["adapt"] if n in verified_reuse]
    adapt_cpu = [n for n in cats["adapt"] if n not in verified_reuse]

    rt_partial: List[str] = []
    clean_native: List[str] = []
    for n in cats["new_native"]:
        if _runtime_fallback_helper_count(model_id, _safe_id(n)) > 0:
            rt_partial.append(n)
        else:
            clean_native.append(n)
    on_device = len(reuse_dev) + len(adapt_dev) + len(clean_native) + len(rt_partial)
    on_cpu = len(cats["new_fallback"]) + len(reuse_cpu) + len(adapt_cpu)
    total = on_device + on_cpu

    graduated = len(cats["new_native"])
    if _demo is not None:
        for _n in cats["adapt"]:
            if _stub_has_graduated_from_autofill(_demo / "_stubs" / f"{_safe_id(_n)}.py"):
                graduated += 1
    return {
        "reuse": len(reuse_dev),
        "adapt": len(adapt_dev),
        "reuse_cpu": len(reuse_cpu) + len(adapt_cpu),
        "reuse_cpu_names": sorted(reuse_cpu + adapt_cpu),
        "new_native": len(clean_native),
        "new_native_partial_cpu": len(rt_partial),
        "new_native_partial_cpu_names": rt_partial,
        "new_fallback": len(cats["new_fallback"]),
        "on_device": on_device,
        "on_cpu": on_cpu,
        "graduated": graduated,
        "total": total,
    }


def _format_compute_split(model_id: str, *, label: str = "compute split", indent: str = "  ") -> List[str]:
    s = _compute_split(model_id)
    total = s["total"]
    if total == 0:
        return [f"{indent}{label}: (no tracked components)"]

    def pct(n: int) -> str:
        return f"{(n * 100) // total}%"

    lines = [
        f"{indent}{label}: {s['on_device']}/{total} on device ({pct(s['on_device'])}), "
        f"{s['on_cpu']}/{total} on CPU ({pct(s['on_cpu'])})",
        f"{indent}  Graduated (ON_DEVICE) : {s.get('graduated', 0)}/{total} "
        f"({pct(s.get('graduated', 0))}) actually graduated (native stub, PCC-verified)",
        f"{indent}  on device : REUSE-wired={s['reuse']}  ADAPT-wired={s['adapt']}  "
        f"NEW-native={s['new_native']}  NEW-partial-CPU={s['new_native_partial_cpu']}",
        f"{indent}  on CPU    : NEW-fallback={s['new_fallback']}  REUSE/ADAPT-not-wired={s.get('reuse_cpu', 0)}",
    ]
    if s.get("new_native_partial_cpu", 0) > 0:
        names = ", ".join(s.get("new_native_partial_cpu_names", []))
        lines.append(
            f"{indent}  partial-CPU components (TTNN path exists but >=1 helper ran on CPU " f"at runtime): {names}"
        )
    if s.get("reuse_cpu", 0) > 0:
        names = ", ".join(s.get("reuse_cpu_names", []))
        lines.append(
            f"{indent}  REUSE/ADAPT tagged but NOT wired to a ttnn module in this demo "
            f"(runs on CPU via eager runner): {names}"
        )
    return lines


def _compute_op_split(model_id: str) -> Dict[str, object]:
    """Op-level compute split using the per-component op-synth manifests.

    For each NEW component we look at `_stubs/<safe>.opplan.json` (written
    by `autofill_stubs(..., op_synth=True)`). The manifest tells us how
    many op-REUSE / op-ADAPT / op-NEW leaves the component has. We then
    decide on-device vs on-CPU based on whether the stub's `__call__` has
    graduated from the torch fallback:

      * graduated `__call__` -> all leaves run on device (the LLM rewrote
        the forward path to use the pre-bound `_apply_*` helpers).
      * still on fallback   -> all leaves run on CPU (the `_apply_*`
        helpers exist but are unused; forward delegates to torch).

    REUSE / ADAPT components don't carry per-op manifests today (they
    point at pre-existing tt-demo ports), so we treat them as a single
    op apiece — on device ONLY when the reuse target is actually wired
    into this demo (shared verified classifier), otherwise on CPU (an
    unwired tag runs on the eager runner). NEW components without a
    manifest are also reported as a single op on CPU (plain torch wrapper).

    Returns a dict with op counts plus a per-component breakdown so
    callers can render either a one-line summary or a full table.
    """
    from .bringup_loop import find_demo_dir

    demo_dir = find_demo_dir(model_id)
    out: Dict[str, object] = {
        "on_device": 0,
        "on_cpu": 0,
        "total": 0,
        "components": [],
        "have_manifests": False,
    }
    if demo_dir is None:
        return out
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return out
    try:
        data = json.loads(status_path.read_text())
    except Exception:
        return out

    from .final_categorization import reuse_adapt_on_device

    verified_reuse = reuse_adapt_on_device(demo_dir, data.get("components", []) or [])

    rows: List[Dict[str, object]] = []
    on_device = 0
    on_cpu = 0
    any_manifest = False

    for comp in data.get("components", []):
        name = str(comp.get("name", "")).strip()
        status = str(comp.get("status", "?")).strip()
        if not name:
            continue
        safe = _safe_id(name)
        stub_path = demo_dir / "_stubs" / f"{safe}.py"
        manifest_path = demo_dir / "_stubs" / f"{safe}.opplan.json"

        if status in ("REUSE", "ADAPT"):
            wired = name in verified_reuse
            rows.append(
                {
                    "name": name,
                    "status": status if wired else f"{status} (not wired — CPU)",
                    "where": "device" if wired else "cpu",
                    "on_device": 1 if wired else 0,
                    "on_cpu": 0 if wired else 1,
                    "total": 1,
                    "has_manifest": False,
                }
            )
            if wired:
                on_device += 1
            else:
                on_cpu += 1
            continue

        if status != "NEW":
            continue

        manifest: Dict[str, object] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                any_manifest = True
            except Exception:
                manifest = {}

        if manifest:
            counts = manifest.get("counts", {}) or {}
            n_reuse = int(counts.get("op-REUSE", 0))
            n_adapt = int(counts.get("op-ADAPT", 0))
            n_new = int(counts.get("op-NEW", 0))
            comp_total = n_reuse + n_adapt + n_new
            graduated = _stub_has_graduated_from_autofill(stub_path)
            if graduated:
                runtime_cpu = min(_runtime_fallback_helper_count(model_id, safe), comp_total)
                row = {
                    "name": name,
                    "status": "NEW-native",
                    "where": "device",
                    "on_device": comp_total - runtime_cpu,
                    "on_cpu": runtime_cpu,
                    "total": comp_total,
                    "has_manifest": True,
                    "op_reuse": n_reuse,
                    "op_adapt": n_adapt,
                    "op_new": n_new,
                    "runtime_cpu_fallbacks": runtime_cpu,
                }
                if runtime_cpu == comp_total and runtime_cpu > 0:
                    row["where"] = "cpu"
                    row["status"] = "NEW-native (all helpers fell back at runtime)"
                elif runtime_cpu > 0:
                    row["where"] = "mixed"
                    row["status"] = "NEW-native (partial runtime CPU fallback)"
                on_device += comp_total - runtime_cpu
                on_cpu += runtime_cpu
                rows.append(row)
            else:
                on_cpu += comp_total
                rows.append(
                    {
                        "name": name,
                        "status": "NEW-fallback",
                        "where": "cpu",
                        "on_device": 0,
                        "on_cpu": comp_total,
                        "total": comp_total,
                        "has_manifest": True,
                        "op_reuse": n_reuse,
                        "op_adapt": n_adapt,
                        "op_new": n_new,
                        "runtime_cpu_fallbacks": 0,
                    }
                )
        else:
            graduated = _stub_has_graduated_from_autofill(stub_path)
            if graduated:
                runtime_cpu = _runtime_fallback_helper_count(model_id, safe)
                if runtime_cpu > 0:
                    on_cpu += 1
                    rows.append(
                        {
                            "name": name,
                            "status": "NEW-native (runtime CPU fallback)",
                            "where": "cpu",
                            "on_device": 0,
                            "on_cpu": 1,
                            "total": 1,
                            "has_manifest": False,
                            "runtime_cpu_fallbacks": runtime_cpu,
                        }
                    )
                else:
                    on_device += 1
                    rows.append(
                        {
                            "name": name,
                            "status": "NEW-native",
                            "where": "device",
                            "on_device": 1,
                            "on_cpu": 0,
                            "total": 1,
                            "has_manifest": False,
                            "runtime_cpu_fallbacks": 0,
                        }
                    )
            else:
                on_cpu += 1
                rows.append(
                    {
                        "name": name,
                        "status": "NEW-fallback",
                        "where": "cpu",
                        "on_device": 0,
                        "on_cpu": 1,
                        "total": 1,
                        "has_manifest": False,
                        "runtime_cpu_fallbacks": 0,
                    }
                )

    out["on_device"] = on_device
    out["on_cpu"] = on_cpu
    out["total"] = on_device + on_cpu
    out["components"] = rows
    out["have_manifests"] = any_manifest
    out["total_runtime_cpu_fallbacks"] = sum(int(r.get("runtime_cpu_fallbacks", 0) or 0) for r in rows)
    return out


def _format_op_split(
    model_id: str, *, label: str = "op-level split", indent: str = "  ", show_per_component: bool = False
) -> List[str]:
    """Pretty-print the op-level on-device vs on-CPU split."""
    s = _compute_op_split(model_id)
    total = int(s.get("total", 0) or 0)
    on_device = int(s.get("on_device", 0) or 0)
    on_cpu = int(s.get("on_cpu", 0) or 0)
    if total == 0:
        return [f"{indent}{label}: (no tracked components)"]
    have_manifests = bool(s.get("have_manifests", False))
    detail_suffix = (
        "" if have_manifests else "  (component-level estimate; run with --op-synth for op-level granularity)"
    )
    total_rt_fb = int(s.get("total_runtime_cpu_fallbacks", 0) or 0)
    if total_rt_fb > 0:
        detail_suffix += f"  [includes {total_rt_fb} runtime CPU fallback(s)]"

    def pct(n: int) -> str:
        return f"{(n * 100) // total}%"

    lines = [
        f"{indent}{label}: {on_device}/{total} on device ({pct(on_device)}), "
        f"{on_cpu}/{total} on CPU ({pct(on_cpu)}){detail_suffix}"
    ]
    if show_per_component:
        rows = s.get("components", []) or []
        if rows:
            name_w = max(len(r["name"]) for r in rows)
            for r in rows:
                if r.get("has_manifest"):
                    breakdown = (
                        f"  ops={int(r['total']):3d}"
                        f" (op-REUSE={int(r.get('op_reuse', 0))},"
                        f" op-ADAPT={int(r.get('op_adapt', 0))},"
                        f" op-NEW={int(r.get('op_new', 0))})"
                    )
                else:
                    breakdown = f"  ops=  ?  (no op-synth manifest)"
                rt_fb = int(r.get("runtime_cpu_fallbacks", 0) or 0)
                if rt_fb > 0:
                    breakdown += f"  [CPU-FB={rt_fb}]"
                where = str(r.get("where", "?"))
                if where == "device":
                    badge = "DEVICE"
                elif where == "cpu":
                    badge = "CPU   "
                elif where == "mixed":
                    badge = "MIXED "
                else:
                    badge = "?     "
                lines.append(f"{indent}  [{badge}] {r['name']:<{name_w}s}  " f"({str(r['status'])}){breakdown}")
    return lines


def _check_memory_fit_before_llm(
    model_id: str,
    *,
    box_name: str,
    mesh_str: Optional[str],
    dtype_override: Optional[str],
    sep: str = "=" * 72,
) -> Tuple[str, str]:
    """Pre-LLM memory-fit gate (2026-05-23 user-flagged: 'don't burn LLM
    tokens on a model the planner already said won't fit').

    Returns one of:
      ("fit",     "<plan verdict>")  — the requested (box, mesh, dtype)
                                       has headroom; proceed.
      ("no-fit",  "<diagnostic>")    — the model exceeds the box/mesh's
                                       per-chip budget. Caller MUST abort
                                       before scaffolding (and definitely
                                       before LLM iteration) — no amount
                                       of LLM rewriting can shrink a
                                       fundamentally-too-large model.
      ("unknown", "<reason>")        — the probe has no memory model
                                       (typical for vision / multi-modal
                                       models like sam2-hiera-*; their
                                       budgets are dominated by per-op
                                       L1 scratch, not weights, and the
                                       compat gate has already passed).
                                       Caller SHOULD proceed.

    The gate has no opt-out: if the planner says a model doesn't fit
    its (box, mesh, dtype) tuple, the bring-up aborts. Memory budgets
    are hardware-determined; an LLM cannot rewrite that away."""
    from .verdict import Tightness

    try:
        probe = probe_model(model_id)
    except Exception as exc:
        return ("unknown", f"probe failed ({type(exc).__name__}: {exc})")
    if probe.memory_model is None:
        return (
            "unknown",
            "no LLM-style memory model produced — typically a vision / "
            "multi-modal model whose memory budget is dominated by per-op "
            "scratch, not weights. Compat gate already covered "
            "architectural support.",
        )
    try:
        box = find_box(box_name)
    except Exception:
        return ("unknown", f"unknown box `{box_name}`; deferring to compat gate")
    dtypes = _dtypes_for(
        probe.category,
        [dtype_override] if dtype_override else [],
        probe.saved_dtype,
    )
    verdict = evaluate_all(
        model=probe.memory_model,
        boxes=[box],
        dtypes=dtypes,
        batch=1,
        seq=8192,
        kv_dtype_bytes=2.0,
        all_meshes=(mesh_str is not None),
        explore_pp=False,
    )

    if mesh_str is not None:
        try:
            target_mesh = _parse_mesh(mesh_str)
        except ValueError:
            return ("unknown", f"mesh `{mesh_str}` is not parseable; deferring to compat gate")
        target_rows = [r for r in verdict.rows if r.mesh_shape == target_mesh]
        if not target_rows:
            return (
                "unknown",
                f"mesh {mesh_str} is not canonical for {box_name}; "
                f"deferring to compat gate which lists valid shapes.",
            )
        fitting = [r for r in target_rows if r.fits]
        if not fitting:
            worst = max(target_rows, key=lambda r: r.per_chip_gb)
            return (
                "no-fit",
                (
                    f"requested mesh `{mesh_str}` on `{box_name}` does NOT "
                    f"fit `{model_id}`: needs {worst.per_chip_gb:.2f} GB/chip "
                    f"but only {worst.usable_per_chip_gb:.2f} GB usable "
                    f"(headroom {worst.headroom_gb:.2f} GB, dtype={worst.dtype})."
                ),
            )
        chosen = min(fitting, key=lambda r: -r.headroom_gb)
        return (
            "fit",
            (
                f"mesh `{mesh_str}` on `{box_name}` -> "
                f"{chosen.tightness.value} (needs {chosen.per_chip_gb:.2f} "
                f"GB/chip, {chosen.headroom_gb:.2f} GB headroom, "
                f"dtype={chosen.dtype})"
            ),
        )

    if verdict.best is None:
        if not verdict.rows:
            return ("unknown", "verdict produced no rows; deferring to compat gate")
        worst = max(verdict.rows, key=lambda r: r.per_chip_gb)
        return (
            "no-fit",
            (
                f"NO mesh on `{box_name}` fits `{model_id}`: tightest tried "
                f"({worst.mesh_shape}, {worst.dtype}) still needs "
                f"{worst.per_chip_gb:.2f} GB/chip vs "
                f"{worst.usable_per_chip_gb:.2f} GB usable "
                f"(over by {-worst.headroom_gb:.2f} GB)."
            ),
        )
    return (
        "fit",
        (
            f"best fit on `{box_name}`: mesh {verdict.best.mesh_shape} -> "
            f"{verdict.best.tightness.value} "
            f"({verdict.best.per_chip_gb:.2f} GB/chip, "
            f"{verdict.best.headroom_gb:.2f} GB headroom, "
            f"dtype={verdict.best.dtype})"
        ),
    )


def _enforce_memory_fit_or_abort(
    model_id: str,
    *,
    box_name: str,
    mesh_str: Optional[str],
    dtype_override: Optional[str],
    sep: str = "=" * 72,
) -> Optional[int]:
    """Apply `_check_memory_fit_before_llm` and return an exit code if the
    caller MUST abort. Returns None to mean "proceed". A short status line
    is printed in every case so the convergence log always shows what the
    fit gate decided."""
    status, msg = _check_memory_fit_before_llm(
        model_id,
        box_name=box_name,
        mesh_str=mesh_str,
        dtype_override=dtype_override,
        sep=sep,
    )
    if status == "fit":
        print(f"  Memory fit gate PASSED: {msg}")
        return None
    if status == "unknown":
        print(f"  Memory fit gate SKIPPED: {msg}")
        return None

    print()
    print(sep)
    print(
        f"  Memory fit gate FAILED — aborting before scaffold / autofill / LLM:\n"
        f"  {msg}\n"
        f"\n"
        f"  No amount of LLM iteration can shrink a model that doesn't fit\n"
        f"  the requested mesh's per-chip budget; iterating would just burn\n"
        f"  LLM tokens on an unsolvable problem.\n"
        f"\n"
        f"  Options:\n"
        f"    1. Pick a larger mesh: `--mesh 1x4` or `--mesh 2x4` etc.\n"
        f"    2. Pick a larger box: `--box T3K` or `--box Galaxy`.\n"
        f"    3. Use a smaller dtype: `--dtype bfp8_b` (8-bit weights) or\n"
        f"       `--dtype bfp4_b` (4-bit) if the model supports it."
    )
    print(sep)
    return 2


def _check_demo_environment_compat(
    *, demo_module_path: str = "models.tt_transformers.demo.simple_text_demo"
) -> Tuple[bool, List[str]]:
    """2026-05-23 environment pre-check: validate that the demo path
    the tool is about to invoke can actually run with the installed
    `transformers` version.

    Why this exists: `up --auto <supported-LLM>` ends up invoking
    `tt_transformers/simple_text_demo.py` via pytest. That demo was
    written against `transformers==4.x`. When `transformers==5.x` is
    installed (e.g. 5.8.1 on this box), the demo fails with errors
    like:
      - `ImportError: cannot import name 'AutoModelForVision2Seq'`
      - `TypeError: pow(NoneType, Tensor)` (rope_theta missing)
      - `RuntimeError: Could not infer dtype of tokenizers.Encoding`

    Each individual symptom is opaque. The user has to dig through
    repo code to figure out why their model crashed. The TOOL's
    contract is to catch this kind of thing upfront and abort with
    a clear message.

    Returns (ok, problems_list). On `not ok`, the caller is expected
    to print the problems and abort. Returns `(True, [])` when the
    environment looks clean and the demo should be able to run.

    Strategy: fast static checks only. No subprocess imports (those
    would take 5-10s and pull in ttnn). We grep the demo's import
    chain for known-broken symbols and version-check transformers."""
    problems: List[str] = []

    try:
        import transformers as _tf

        tf_version = getattr(_tf, "__version__", "(unknown)")
    except Exception as exc:
        problems.append(f"transformers is not importable ({type(exc).__name__}: {exc})")
        return False, problems

    try:
        major = int(str(tf_version).split(".")[0])
    except (ValueError, IndexError):
        major = 0

    if major >= 5:
        repo_files_on_demo_path = [
            BRINGUP_ROOT() / "models" / "common" / "llama_models.py",
            BRINGUP_ROOT() / "models" / "tt_transformers" / "tt" / "model_config.py",
            BRINGUP_ROOT() / "models" / "tt_transformers" / "tt" / "common.py",
        ]

        for p in repo_files_on_demo_path:
            if not p.is_file():
                continue
            try:
                src = p.read_text()
            except Exception:
                continue

            for i, line in enumerate(src.splitlines()):
                if "AutoModelForVision2Seq" not in line:
                    continue
                if "from transformers import" not in line:
                    continue
                window = "\n".join(src.splitlines()[max(0, i - 3) : i + 4])
                if "AutoModelForImageTextToText" not in window:
                    problems.append(
                        f"{safe_relative_to_root(p)}:{i + 1} "
                        f"imports `AutoModelForVision2Seq` from "
                        f"transformers (removed in 5.x; renamed "
                        f"to `AutoModelForImageTextToText`). The "
                        f"demo will ImportError before the model "
                        f"even loads."
                    )

        mc_path = BRINGUP_ROOT() / "models" / "tt_transformers" / "tt" / "model_config.py"
        if mc_path.is_file():
            try:
                mc_src = mc_path.read_text()
            except Exception:
                mc_src = ""
            if 'text_config.get("rope_theta")' in mc_src and "rope_parameters" not in mc_src:
                problems.append(
                    "models/tt_transformers/tt/model_config.py reads "
                    "`rope_theta` only; transformers 5.x migrates this "
                    "field to `rope_parameters['rope_theta']` for some "
                    "models (e.g. Phi-3.5). Runtime will crash with "
                    "`TypeError: pow(NoneType, Tensor)` at rope.py."
                )

        common_path = BRINGUP_ROOT() / "models" / "tt_transformers" / "tt" / "common.py"
        if common_path.is_file():
            try:
                common_src = common_path.read_text()
            except Exception:
                common_src = ""
            if "apply_chat_template" in common_src and (
                "_normalize_token_result_to_list" not in common_src
                and "_chat_template_ids" not in common_src
                and 'hasattr(result, "input_ids")' not in common_src
                and 'hasattr(encoded, "input_ids")' not in common_src
                and 'hasattr(encoded, "ids")' not in common_src
            ):
                problems.append(
                    "models/tt_transformers/tt/common.py calls "
                    "`apply_chat_template(tokenize=True)` but does "
                    "not normalize the return type. In transformers "
                    "5.x this returns `BatchEncoding` (or "
                    "`tokenizers.Encoding`), not `List[int]`. "
                    "Runtime will crash with `RuntimeError: Could "
                    "not infer dtype of tokenizers.Encoding`."
                )

    if problems:
        problems.insert(
            0,
            f"transformers=={tf_version} is installed; the "
            f"`{demo_module_path}` codepath assumes 4.x APIs in "
            f"several places. Detected issues:",
        )

    return (not problems), problems


def _run_advisory_meta_plan(
    model_id: str,
    *,
    box: str,
    mesh: Optional[str],
    agent_bin: str = "claude",
    agent_model: str = "sonnet",
) -> None:
    """2026-05-23 Improvement 2: run ONE pre-loop LLM call to evaluate
    the bring-up plan as a whole. ADVISORY ONLY: prints the verdict
    banner and returns. Never raises; never gates the bring-up.

    The goal here is to give the autonomous loop the same
    "should-we-even-do-this?" meta-reasoning that an interactive
    Claude session would naturally have. The output is purely
    informational -- if the meta-planner says "LOW feasibility, this
    needs custom kernels", the iterate loop still proceeds. That way
    a falsely-cautious meta-plan can't lock the user out of a model
    that would have converged anyway."""
    from .meta_plan import run_meta_plan, format_verdict_banner
    from .family_backends import pick_backend_with_quality

    try:
        probe = probe_model(model_id)
    except Exception as exc:
        print(f"  [meta-plan] SKIPPED (probe failed: " f"{type(exc).__name__}: {exc})")
        return
    model_type = ""
    try:
        model_type = str(probe.raw_config.get("model_type") or "")
    except Exception:
        pass
    backend, quality = pick_backend_with_quality(
        category=probe.category,
        model_type=model_type,
        pipeline_tag=getattr(probe, "pipeline_tag", None),
    )

    components: List[Dict[str, Any]] = []
    try:
        from .module_tree import discover_components_from_hf_id

        discovered = discover_components_from_hf_id(model_id)
        for d in discovered[:25]:
            components.append(
                {
                    "name": d.name,
                    "class_name": d.class_name,
                    "submodule_path": d.submodule_path,
                    "occurrences": d.occurrences,
                    "leaf_op_count": d.leaf_op_count,
                }
            )
    except Exception:
        pass

    try:
        verdict = run_meta_plan(
            model_id=model_id,
            category=probe.category,
            model_type=model_type or None,
            backend_name=(backend.name if backend else "(none)"),
            match_quality=quality,
            box=box,
            mesh=mesh,
            components=components,
            agent_bin=agent_bin,
            agent_model=agent_model,
        )
    except Exception as exc:
        print(f"  [meta-plan] SKIPPED (unexpected error: " f"{type(exc).__name__}: {exc})")
        return
    print()
    print(format_verdict_banner(verdict))


def _enforce_backend_match_quality_or_abort(
    model_id: str,
    *,
    accept_closest: bool = False,
    sep: str = "=" * 72,
) -> Optional[int]:
    """Loud-fallback gate (2026-05-23 audit defect 1).

    Re-probes the model and asks `pick_backend_with_quality` HOW the
    backend was selected:
      - "exact"            : model_type matched a backend; proceed silently.
      - "pipeline"         : pipeline_tag matched; proceed with an INFO log.
      - "category-default" : ABORT unless `accept_closest=True`. This is
                             the silent-wrong-template branch that has
                             historically wasted LLM tokens iterating
                             against a backend that has nothing to do
                             with the actual architecture.
      - "none"             : no candidate backend at all; defer to the
                             scaffold step which already aborts loudly
                             for `backend is None`.

    Returns ``None`` to mean "proceed". Returns an exit code (2) to mean
    "caller must abort"."""
    from .family_backends import pick_backend_with_quality

    try:
        probe = probe_model(model_id)
    except Exception as exc:
        print(f"  Backend-match gate SKIPPED (probe failed: " f"{type(exc).__name__}: {exc}); deferring to scaffold.")
        return None
    model_type = ""
    pipeline_tag = None
    try:
        model_type = str(probe.raw_config.get("model_type") or "")
        pipeline_tag = getattr(probe, "pipeline_tag", None)
    except Exception:
        pass
    backend, quality = pick_backend_with_quality(
        category=probe.category,
        model_type=model_type,
        pipeline_tag=pipeline_tag,
    )
    if backend is None:
        auto_picked = _try_auto_onboard_inline(
            model_id=model_id,
            category=probe.category,
            model_type=model_type,
            pipeline_tag=pipeline_tag,
            closest_backend=None,
        )
        if auto_picked is not None:
            new_backend, new_quality = auto_picked
            print(
                f"  Backend match: {new_quality.upper()}  ({new_backend.name})  "
                f"(via auto-onboard; LLM drafted a new entry because "
                f"no backend was registered for category={probe.category!r})"
            )
            return None
        print(
            f"  (no backend registered for category={probe.category!r} "
            f"and auto-onboard could not draft one; deferring to "
            f"scaffold's cold-start path)"
        )
        return None
    if quality == "exact":
        print(f"  Backend match: EXACT  ({backend.name})  " f"via model_type={model_type!r}")
        return None
    if quality == "pipeline":
        print(
            f"  Backend match: PIPELINE-TAG  ({backend.name})  via "
            f"pipeline_tag={pipeline_tag!r} (model_type={model_type!r} "
            f"did not match any registered keys; this is usually OK for "
            f"HF-pipeline-style arches)"
        )
        return None

    if getattr(backend, "routing_mode", "") == "generic":
        if probe.category != "Unknown":
            print(
                f"  Backend match: GENERIC  ({backend.name})  "
                f"(category={probe.category!r}, model_type={model_type!r}; "
                f"this backend is intentionally catch-all for its category, "
                f"not a silent wrong-template fallback)"
            )
            return None
        print(
            f"  Backend match: GENERIC catch-all ({backend.name}) "
            f"but category={probe.category!r} — no real classification. "
            f"Trying inline auto-onboard to draft a real backend."
        )

    if accept_closest:
        print(
            f"  Backend match: CATEGORY-DEFAULT  ({backend.name})  "
            f"(closest-by-category; --accept-closest-backend was set, "
            f"proceeding at user request)"
        )
        return None
    auto_picked = _try_auto_onboard_inline(
        model_id=model_id,
        category=probe.category,
        model_type=model_type,
        pipeline_tag=pipeline_tag,
        closest_backend=backend,
    )
    if auto_picked is not None:
        new_backend, new_quality = auto_picked
        print(
            f"  Backend match: {new_quality.upper()}  ({new_backend.name})  "
            f"(via auto-onboard; LLM drafted + spliced into "
            f"family_backends.py)"
        )
        return None

    print()
    print(sep)
    print(
        f"  CLOSEST-TEMPLATE FALLBACK -- using `{backend.name}` for "
        f"{model_id!r}\n"
        f"\n"
        f"  No registered backend exactly matches:\n"
        f"    model_type   = {model_type!r}\n"
        f"    pipeline_tag = {pipeline_tag!r}\n"
        f"    category     = {probe.category!r}\n"
        f"\n"
        f"  Auto-onboard could not draft a tailored backend entry, so\n"
        f"  the tool will iterate against `{backend.name}` "
        f"(template path: {backend.demo_path}) and hope it's structurally\n"
        f"  close enough. This may produce nonsense if the architectures\n"
        f"  diverge significantly. Watch the per-component PCC pass-rate\n"
        f"  in Step 5 -- if many components fail consecutively, the\n"
        f"  template is likely wrong; hand-add a `FamilyBackend(...)` to\n"
        f"  family_backends.py or rerun `auto-onboard` with a longer\n"
        f"  --timeout-s."
    )
    print(sep)
    return None


_pytest_capture_sink: Optional[str] = None


def _git_worktree_diff_hash() -> str:
    """Cheap, repo-agnostic "did anything change" probe.

    Runs ``git diff --stat`` (working tree vs HEAD) and SHA-1's the
    output. The hash is stable across repeated reads as long as the
    working tree is untouched. Returns the empty string if git is
    not available or this is not a git checkout -- callers should
    treat that as "always different" by comparing against a unique
    value, not by equality.

    Used by the PCC-repair loop to detect "agent made zero edits".
    """
    import hashlib
    import subprocess as _sp

    try:
        out = _sp.run(
            ["git", "-c", "color.ui=never", "diff", "--stat"],
            cwd=BRINGUP_ROOT(),
            capture_output=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return ""
        return hashlib.sha1(out.stdout).hexdigest()
    except (FileNotFoundError, _sp.TimeoutExpired, OSError):
        return ""


_CACHE_AFFECTING_FILE_GLOBS: Tuple[str, ...] = (
    "**/load_checkpoints*.py",
    "**/state_dict*.py",
    "**/tt_transformers/tt/common.py",
)


def _edits_touch_cache_affecting_files(edited_files: Sequence[str]) -> List[str]:
    """Return the subset of ``edited_files`` that match any of
    :data:`_CACHE_AFFECTING_FILE_GLOBS`.

    Used by the PCC-repair loop to decide whether to invalidate the
    TT-native weight cache eagerly (before the next demo run) instead
    of waiting for the demo to stagnate. The matcher uses
    :func:`fnmatch.fnmatch` against the path-as-string so it works on
    both relative paths (from ``git diff --name-only``) and absolute
    paths.
    """
    import fnmatch

    hits: List[str] = []
    for p in edited_files:
        if not p:
            continue

        norm = p.replace("\\", "/")
        for pattern in _CACHE_AFFECTING_FILE_GLOBS:
            if fnmatch.fnmatch(norm, pattern):
                hits.append(p)
                break

    return hits


def _hash_files(files: Sequence[str], repo_root: Path) -> Dict[str, str]:
    """Return ``{rel_path: sha1_hex}`` for each existing file in
    ``files`` (paths are normalised to be repo-root relative).

    Missing files map to the empty string (so a delete-then-recreate
    edit registers as a hash change). Used by the edit-took-effect
    verifier to detect "the agent edited the file but the .pyc was
    cached so the running demo didn't see it" failure mode.
    """
    import hashlib

    out: Dict[str, str] = {}
    for p in files or ():
        if not p:
            continue
        abs_p = (repo_root / p).resolve() if not Path(p).is_absolute() else Path(p)
        try:
            if abs_p.is_file():
                out[p] = hashlib.sha1(abs_p.read_bytes()).hexdigest()
            else:
                out[p] = ""
        except Exception:
            out[p] = ""
    return out


def _purge_pycache_for_edited_files(edited_files: Sequence[str], repo_root: Path) -> int:
    """Delete the ``__pycache__/<name>.cpython-*.pyc`` cache files
    that correspond to each edited ``.py`` file.

    Python uses mtime+size to invalidate .pyc, which is *usually*
    enough -- but the agent's edits sometimes preserve mtime (because
    of git operations) and the demo subprocess can pick up the stale
    .pyc. We purge defensively so every edit reliably takes effect on
    the next demo run.

    Returns the number of .pyc files deleted. Category-agnostic --
    runs on every LLM-edit iteration regardless of model category.
    """
    deleted = 0
    for p in edited_files or ():
        if not p or not p.endswith(".py"):
            continue
        try:
            abs_p = (repo_root / p).resolve() if not Path(p).is_absolute() else Path(p)
            parent = abs_p.parent
            stem = abs_p.stem
            pycache_dir = parent / "__pycache__"
            if not pycache_dir.is_dir():
                continue

            for pyc in pycache_dir.glob(f"{stem}.cpython-*.pyc"):
                try:
                    pyc.unlink()
                    deleted += 1
                except Exception:
                    pass
        except Exception:
            continue
    return deleted


def _verify_edit_took_effect(
    *,
    edited_files: Sequence[str],
    repo_root: Path,
    pre_hashes: Dict[str, str],
    prev_verdict_signature: Optional[str],
    new_verdict_signature: Optional[str],
) -> Tuple[bool, str]:
    """After a demo re-run, check whether the agent's edits actually
    influenced the verdict.

    Returns ``(took_effect, diagnostic_message)``:

      * ``took_effect=True`` is the optimistic default — we only flag
        false if we have STRONG evidence the edit was a no-op.

      * ``False`` is returned when ALL of the following hold:
          1. The agent edited at least one file (``edited_files`` non-empty).
          2. The file's on-disk hash actually changed (so we can rule
             out "no edit happened" -- this is the cli's own check,
             reproduced here for symmetry).
          3. The new verdict signature equals the previous one
             EXACTLY (byte-for-byte same mismatch% / repeat% / reason).

    When these all hold, the most likely causes are:
      * stale .pyc shadowed the edit (mitigated by
        :func:`_purge_pycache_for_edited_files` running BEFORE the
        demo re-run -- this verifier runs AFTER, as a safety net)
      * the edited file is loaded into a long-running parent process
        that didn't re-import (rare; the demo is a fresh subprocess)
      * TT-native weight cache shadowed a weight-conversion edit
        (separate handler -- :func:`_invalidate_tt_weight_cache`)
      * the edit was syntactically valid but didn't change any
        runtime behavior (e.g. comment-only, dead-code path)

    The returned message is suitable for both stdout printing and
    inclusion in the LLM's next prompt as a "previous attempt didn't
    take effect" hint.
    """
    if not edited_files:
        return True, ""
    if not prev_verdict_signature or not new_verdict_signature:
        return True, ""

    if prev_verdict_signature != new_verdict_signature:
        return True, ""

    post_hashes = _hash_files(edited_files, repo_root)
    files_with_real_changes: List[str] = []
    for p in edited_files:
        if not p:
            continue
        before = pre_hashes.get(p, "")
        after = post_hashes.get(p, "")
        if before != after:
            files_with_real_changes.append(p)
    if not files_with_real_changes:
        return True, ""

    short = ", ".join(files_with_real_changes[:3])
    if len(files_with_real_changes) > 3:
        short += f", ... (+{len(files_with_real_changes) - 3} more)"
    msg = (
        f"EDIT MAY NOT HAVE TAKEN EFFECT: the agent modified "
        f"{len(files_with_real_changes)} file(s) ({short}) this "
        f"iteration AND the gate verdict is byte-identical to "
        f"the previous iteration. Likely causes: (1) a stale "
        f"__pycache__/*.pyc shadowed the edit (we already purged "
        f"caches before the demo re-ran, but the demo subprocess "
        f"may have re-cached against a different Python; "
        f"verify the edit is reachable from the demo's import "
        f"graph), (2) the TT-native weight cache is shadowing a "
        f"weight-conversion edit (try --invalidate-tt-cache), "
        f"(3) the edit was syntactically valid but doesn't affect "
        f"any executed code path (e.g. it edited a branch the "
        f"demo doesn't take), (4) the demo loaded its target "
        f"function before the edit (long-running parent process "
        f"-- not applicable here, the demo is a fresh subprocess)."
    )
    return False, msg


def _make_verdict_signature(result: Any) -> str:
    """Compact ``str`` summary of a verdict, used to detect
    byte-identical iterations. Includes ``mismatch_ratio``,
    ``max_repeat_ratio``, ``compared_tokens``, and ``reason``. Two
    iterations producing the same signature is the canonical
    "verdict didn't move" signal.
    """
    if result is None:
        return ""
    try:
        return (
            f"mr={float(getattr(result, 'mismatch_ratio', 0.0)):.4f}|"
            f"rr={float(getattr(result, 'max_repeat_ratio', 0.0)):.4f}|"
            f"nr={float(getattr(result, 'non_ascii_ratio', 0.0)):.4f}|"
            f"ct={int(getattr(result, 'compared_tokens', 0))}|"
            f"r={str(getattr(result, 'reason', ''))[:200]}"
        )
    except Exception:
        return repr(result)[:200]
    return hits


def _build_forced_edit_preamble(iter_idx: int) -> str:
    """Return the high-impact preamble shown at the very TOP of the
    next iter's prompt when the previous iter made zero edits.

    Audit 2026-05-24 (P8 / Layer 1): the medgemma run hit this case
    three times in a row (iters 2, 3, 4 each made zero edits despite
    Claude doing tool_use=28/33/52 worth of Read/Grep/Bash). The
    legacy prompt said "Do NOT exit without an Edit" once, in
    section 7 of 8, buried below the suspect ranking and history.
    The agent ignored it.

    This preamble sits above EVERYTHING (above the model id, above
    suspects, above evidence). It is short, MUST-be-followed
    protocol, and explicitly bounds Read time so the agent is
    forced into action.

    Keep ASCII so claude/codex terminals render it cleanly; keep it
    short so the agent reads to the end."""
    return (
        f"==============================================================\n"
        f"  MANDATORY PROTOCOL  (READ THIS FIRST -- iter {iter_idx})\n"
        f"==============================================================\n"
        f"  Your previous iteration made ZERO edits in 25 minutes.\n"
        f"  This iteration MUST end with at least one Edit call.\n"
        f"\n"
        f"  Required protocol:\n"
        f"    1. State your hypothesis in ONE sentence (no fluff).\n"
        f"    2. Read AT MOST 3 files to verify it.\n"
        f"    3. Make EXACTLY ONE Edit. If you are <100% confident,\n"
        f"       edit anyway -- a committed wrong guess generates\n"
        f"       verdict evidence; a refusal generates nothing.\n"
        f"\n"
        f"  You are NOT allowed to exit without an Edit. The loop\n"
        f"  will detect a no-edit exit and TERMINATE the run with\n"
        f"  FAIL after one more no-edit iteration.\n"
        f"=============================================================="
    )


def _build_investigative_mode_preamble(iter_idx: int, component: str, pcc_history: List[float]) -> str:
    """Preamble for PCC PLATEAU cases — overrides the default
    "DO NOT iterate" instruction in the task_block, telling the LLM to
    investigate the bug freely with its Read/Edit/Bash/Write tools.

    Triggered when a component has been at near-but-not-converged PCC
    for multiple iters with byte-identical or near-identical code each
    time. The default one-shot prompt isn't enough for this class of
    bug — the LLM needs to TRACE the math op-by-op, optionally write
    small test scripts, and converge to the precise op that's wrong.

    Pattern mirrors _build_forced_edit_preamble: prepended at the very
    top of the next iter's prompt so the override is read before any
    later "DO NOT iterate" language.

    Returns a fully-formed preamble string. Caller decides when to use it.
    """
    history_str = " -> ".join(f"{p:.4f}" for p in (pcc_history or [])[-4:])
    return (
        f"==============================================================\n"
        f"  INVESTIGATIVE MODE  (READ THIS FIRST -- iter {iter_idx})\n"
        f"==============================================================\n"
        f"  Component `{component}` has PCC-plateaued: {history_str}\n"
        f"  Code runs end-to-end but ONE math op produces a different\n"
        f"  value than torch reference. Generic full-file rewrites are\n"
        f"  NOT working (the LLM keeps producing nearly the same code).\n"
        f"\n"
        f"  THIS ITERATION RUNS IN INVESTIGATIVE MODE. Override any\n"
        f"  later 'DO NOT iterate' / 'write the COMPLETE file' language\n"
        f"  in this prompt. Instead:\n"
        f"\n"
        f"    1. STATE your hypothesis in ONE sentence (which op do you\n"
        f"       suspect is producing the wrong value? bf16 accumulation?\n"
        f"       wrong reduction order? layout mismatch?).\n"
        f"\n"
        f"    2. INVESTIGATE with your tools:\n"
        f"       - Read the stub + torch reference + ttnn op source.\n"
        f"       - Write a small probe script under /tmp that runs JUST\n"
        f"         the suspect op on a captured input tensor and prints\n"
        f"         the output. Compare to torch on the same input.\n"
        f"         Run it with Bash. Repeat for 1-3 candidate ops if\n"
        f"         budget allows.\n"
        f"       - Read tt-metal's `tt_metal/` C++ or `ttnn/` Python\n"
        f"         source for the suspect op to understand its bf16\n"
        f"         accumulator semantics + tile-alignment requirements.\n"
        f"\n"
        f"    3. MAKE TARGETED EDITS:\n"
        f"       - Use the Edit tool to change ONLY the suspect op (and\n"
        f"         its supporting setup, e.g. kernel config). Do NOT\n"
        f"         rewrite the entire stub from scratch.\n"
        f"       - If multiple edits are needed, make them one at a\n"
        f"         time and verify each with your probe script.\n"
        f"\n"
        f"    4. EXIT when:\n"
        f"       - Your probe script shows the suspect op now matches\n"
        f"         torch (within bf16 noise), OR\n"
        f"       - You have exhausted your investigation budget and want\n"
        f"         the outer pytest to verify your best hypothesis.\n"
        f"\n"
        f"  Common precision knobs to try (when the suspect is a\n"
        f"  reduction/matmul/norm):\n"
        f"    - WormholeComputeKernelConfig(fp32_dest_acc_en=True)\n"
        f"    - MathFidelity.HiFi4 (up from default LoFi)\n"
        f"    - For sub-tile reductions: replace ttnn.mean / ttnn.var\n"
        f"      with ttnn.sum + manual divide by REAL (not padded) size\n"
        f"      (ttnn.mean divides by padded tile size, NOT original).\n"
        f"\n"
        f"  Tool freedom: Read / Edit / Write / Grep / Bash all enabled.\n"
        f"  Treat this iter as a single sustained debugging session;\n"
        f"  the outer pytest validates the final state when you exit.\n"
        f"=============================================================="
    )


def _invalidate_tt_weight_cache(model_id: str) -> Optional[str]:
    """Best-effort: delete the TT-native weight cache for ``model_id``
    so the next ``pytest`` re-run rebuilds it from the HF safetensors
    using the *current* source code.

    Why this exists. Discovered post-medgemma 2026-05-23: when the
    PCC-repair loop's verdict stays *byte-identical* across iterations
    despite the agent making real edits, the most common cause is the
    TT-native cache shadowing the edits. The demo's load path is
    roughly:

        if model_cache/<model_id>/<arch>/*.bin exists:
            torch.load(...)  # bypasses load_checkpoints entirely
        else:
            convert_state_dict_via_load_checkpoints(...)
            save_to_cache(...)

    so any edit the agent makes to ``load_checkpoints.py`` / state-
    dict-renaming / weight-conversion code is invisible until the
    cache is blown away.

    Returns the absolute path that was deleted, or ``None`` if the
    cache directory didn't exist (or deletion failed). The helper is
    intentionally conservative: it ONLY deletes the per-model subtree,
    never the parent ``model_cache/`` directory.
    """
    import shutil

    cache_root = Path(REPO_ROOT) / "model_cache"
    if not cache_root.is_dir():
        return None
    target = cache_root
    for part in model_id.split("/"):
        target = target / part
    if not target.exists():
        return None
    try:
        size_bytes = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    except OSError:
        size_bytes = 0
    try:
        shutil.rmtree(target)
    except OSError as exc:
        print(f"  warning: failed to invalidate TT weight cache at " f"{target}: {exc}. Edits may still be shadowed.")
        return None
    size_gb = size_bytes / (1024**3)
    print(
        f"  INVALIDATED TT weight cache at {target} "
        f"({size_gb:.1f} GB). Next pytest run will re-convert "
        f"weights using the agent's edited code."
    )
    return str(target)


def _git_changed_files() -> List[str]:
    """Return the list of files (paths relative to repo root) that
    differ from HEAD in the working tree. Includes both modified-
    in-tracked and newly untracked files.

    Returns ``[]`` on git failure -- callers should not assume the
    list is exhaustive in that case, only that it's a best-effort
    snapshot for human-readable display.
    """
    import subprocess as _sp

    try:
        out = _sp.run(
            ["git", "status", "--porcelain"],
            cwd=BRINGUP_ROOT(),
            capture_output=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return []
        files: List[str] = []
        for line in out.stdout.decode("utf-8", errors="replace").splitlines():
            if len(line) < 4:
                continue

            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1].strip()
            if path:
                files.append(path)
        return files
    except (FileNotFoundError, _sp.TimeoutExpired, OSError):
        return []


def _run_prepare_capture(
    prepare_argv: argparse.Namespace,
    *,
    capture_dir: Optional[str] = None,
) -> Tuple[int, str]:
    """Run :func:`cmd_prepare` with its pytest output captured.

    Sets the module-level :data:`_pytest_capture_sink` to a temp file,
    invokes ``cmd_prepare(prepare_argv)``, then reads the captured
    output and returns ``(rc, captured_text)``. The capture file is
    deleted on success; on failure it's kept under
    ``generated/tt_hw_planner/`` for post-mortem.

    Used by :func:`_runtime_repair_loop`. Safe to call when no repair
    loop is active -- the global is restored to ``None`` after the
    call regardless of exception path."""
    global _pytest_capture_sink

    import tempfile

    if capture_dir is None:
        capture_dir = str(REPO_ROOT / "generated" / "tt_hw_planner")
    os.makedirs(capture_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(
        suffix=".pytest.log",
        prefix="repair_iter_",
        dir=capture_dir,
    )
    os.close(fd)

    prev_sink = _pytest_capture_sink
    _pytest_capture_sink = path
    try:
        try:
            rc = cmd_prepare(prepare_argv)
        except SystemExit as exc:
            rc = int(exc.code) if exc.code is not None else 2
        except Exception as exc:
            print(
                f"  cmd_prepare crashed during repair-loop capture: " f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            rc = 2
    finally:
        _pytest_capture_sink = prev_sink

    try:
        with open(path, "r") as f:
            captured = f.read()
    except Exception:
        captured = ""
    if rc == 0:
        try:
            os.unlink(path)
        except Exception:
            pass
    else:
        print(f"  (repair-loop captured pytest output -> {path})")
    return rc, captured


def _weights_in_default_cache(model_id: str) -> bool:
    """Best-effort check whether ``model_id``'s weights are already in
    the default HuggingFace cache (or wherever ``HF_HOME`` points).

    Used purely for the startup info banner — *not* to auto-set offline
    mode. The HuggingFace library already checks the cache before any
    network call, so the cache hit is already used automatically; we
    just print a hint so the user knows the run will be fast (and
    won't be silently re-downloading on a flaky network).

    Returns ``False`` on any import / scan failure (treats as
    "couldn't tell"). Never raises.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return False
    try:
        cache = scan_cache_dir()
    except Exception:
        return False
    for repo in getattr(cache, "repos", []):
        if getattr(repo, "repo_id", "") == model_id and getattr(repo, "size_on_disk", 0) > 0:
            return True
    return False


def _resolve_local_weights_env(args: argparse.Namespace) -> Dict[str, str]:
    """Translate the user's --local-dir / --offline-hf flags into the
    env-var overrides that should be applied for subprocess pytest
    runs.

    Pure decision logic — no side effects, returns a dict. Caller is
    responsible for actually setting the env. Kept pure so unit tests
    can pin the contract without mutating ``os.environ``.

    Rules:
      * ``--local-dir <path>``  →  HF_HOME=<path>, HF_HUB_OFFLINE=1
        (the user explicitly pointed at a directory; force HF to use
        it and never touch the network)
      * ``--offline-hf``        →  HF_HUB_OFFLINE=1
        (default cache, but no network — useful for sealed CI)
      * neither                 →  {}
        (HF library's default behavior: cache-first then network)

    NOTE: We do NOT auto-enable HF_HUB_OFFLINE just because the cache
    has the model. A partial / corrupted cache + offline mode would
    raise LocalEntryNotFoundError instead of completing the missing
    download. Offline mode is opt-in.
    """
    overrides: Dict[str, str] = {}
    local_dir = getattr(args, "local_dir", None)
    if local_dir:
        from pathlib import Path as _PathForExpand

        resolved = str(_PathForExpand(local_dir).expanduser().resolve())
        overrides["HF_HOME"] = resolved
        overrides["HF_HUB_OFFLINE"] = "1"
    elif getattr(args, "offline_hf", False):
        overrides["HF_HUB_OFFLINE"] = "1"
    return overrides


def _apply_local_weights_env(args: argparse.Namespace, model_id: str) -> None:
    """Apply ``_resolve_local_weights_env`` overrides to ``os.environ``
    so all subsequent subprocess pytest runs inherit them, and print
    an info banner so the user knows what was applied.

    Also surfaces an "auto-detected cached weights" line when
    ``_weights_in_default_cache`` returns True and no explicit local-
    weights flag was passed — that's informational, not a behavior
    change (HF lib was going to use the cache anyway).
    """
    overrides = _resolve_local_weights_env(args)
    if overrides:
        print()
        print("=" * 72)
        print("  [local-weights] applying user-specified weights resolution:")
        for k, v in overrides.items():
            os.environ[k] = v
            print(f"    {k}={v}")
        print("=" * 72)
        return
    # No explicit flag — print an info line if the cache already has
    # the model. Helps the user understand that the rerun will be fast.
    if _weights_in_default_cache(model_id):
        print(
            f"  [local-weights] found cached weights for {model_id} in the "
            f"default HF cache; HuggingFace will load from there automatically "
            f"(no network attempt unless a file is missing). To force "
            f"offline-only, re-run with --offline-hf."
        )


def _auto_enable_tt_probe(model_id: str) -> str:
    """Set ``TT_PLANNER_PROBE_OUTPUT`` so the TT-side activation probe
    (``agentic.tt_probe.install_probe``) runs and persists per-module
    records every demo subprocess.

    Without this the probe stays opt-in (its docstring expects the
    user to set the env var manually), and the chain-divergence
    diagnostic that runs on e2e PCC fail has no TT records to
    compare against. Setting a deterministic per-model path here
    makes the diagnostic available automatically. Probe overhead is
    per-layer summary stats — small enough that always-on is
    acceptable.

    Returns the path that subprocess pytest runs will write to.
    Idempotent across calls within the same cmd_up invocation.
    """
    from .module_tree import safe_identifier

    path = f"/tmp/tt_hw_planner_probe_{safe_identifier(model_id)}.json"
    os.environ["TT_PLANNER_PROBE_OUTPUT"] = path
    return path


def _run_chain_divergence_diagnostic(
    model_id: str,
    *,
    demo_dir: Optional[Path] = None,
    probe_output_path: Optional[str] = None,
    threshold: float = 0.05,
    hf_timeout_s: float = 300.0,
) -> Optional[Any]:
    """Best-effort chain-divergence diagnostic.

    Loads TT-side probe records, runs HF probe live on CPU, calls
    :func:`agentic.probe.compare_hf_tt_probes` and returns the first
    module where the chain diverged.

    Called from the e2e PCC failure path AFTER the strict gate stamps
    ``ok=False``, BEFORE ``_maybe_escalate_pcc_fail`` fires. Purely
    informational — does NOT mutate rc or routing. The escalation
    behavior is unchanged; this just produces a diagnostic the user
    (and downstream LLM repair prompts) can use to localize the
    failure to a specific module rather than guessing.

    Returns the ``ChainDivergenceResult`` or ``None`` on any setup
    failure (TT records missing, HF probe couldn't load, prompt
    unavailable, etc.). Never raises — failure modes log and degrade
    so the caller's escalation path is never blocked.
    """
    import json

    from .agentic.probe import compare_hf_tt_probes, probe_hf_modules
    from .output_validation import load_demo_first_prompt

    if probe_output_path is None:
        probe_output_path = os.environ.get("TT_PLANNER_PROBE_OUTPUT")
    if not probe_output_path:
        print(
            "  [chain-divergence] skipped: TT_PLANNER_PROBE_OUTPUT not set "
            "(no TT-side activation records to compare against)",
            file=sys.stderr,
        )
        return None

    probe_path = Path(probe_output_path)
    if not probe_path.is_file():
        print(
            f"  [chain-divergence] skipped: TT probe records file "
            f"{probe_path} does not exist (probe may not have installed "
            f"in the demo subprocess)",
            file=sys.stderr,
        )
        return None

    try:
        with open(probe_path, "r", encoding="utf-8") as f:
            tt_blob = json.load(f)
    except Exception as exc:
        print(
            f"  [chain-divergence] skipped: could not read TT probe " f"records ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return None

    tt_records = tt_blob.get("records") if isinstance(tt_blob, dict) else None
    if not isinstance(tt_records, list) or not tt_records:
        print(
            f"  [chain-divergence] skipped: TT probe records empty or " f"malformed at {probe_path}",
            file=sys.stderr,
        )
        return None

    prompt_text: str = ""
    if demo_dir is not None:
        try:
            prompt_text = load_demo_first_prompt(demo_dir) or ""
        except Exception:
            prompt_text = ""
    if not prompt_text:
        prompt_text = "Hello"

    try:
        hf_result = probe_hf_modules(
            model_id=model_id,
            prompt_text=prompt_text,
            max_total_steps=4,
            timeout_s=hf_timeout_s,
            verbose=False,
        )
    except Exception as exc:
        print(
            f"  [chain-divergence] HF probe raised " f"{type(exc).__name__}: {exc}; skipping diagnostic",
            file=sys.stderr,
        )
        return None
    if hf_result is None:
        print(
            "  [chain-divergence] HF probe returned None (transformers "
            "missing, model_id invalid, or other setup error); skipping",
            file=sys.stderr,
        )
        return None

    return compare_hf_tt_probes(hf_result, tt_records, threshold=threshold)


def _log_chain_divergence(result: Any, *, model_id: str) -> None:
    """Render the diagnostic as a short banner. Pure presentation."""
    if result is None:
        return
    sep = "=" * 72
    print()
    print(sep)
    print(f"  CHAIN-DIVERGENCE DIAGNOSTIC for {model_id}")
    print(sep)
    print(f"  Paired modules : {result.paired_modules}")
    print(f"  Unpaired (HF)  : {len(result.unpaired_hf_modules)}")
    print(f"  Unpaired (TT)  : {len(result.unpaired_tt_modules)}")
    print(f"  Threshold      : {result.threshold:.4f}")
    print(f"  Note           : {result.note}")
    if result.first_divergence is not None:
        d = result.first_divergence
        print()
        print(f"  FIRST DIVERGENCE (in HF-trace order):")
        print(f"    module     : {d.qualified_name}  ({d.class_name})")
        print(f"    step       : {d.step}")
        print(f"    max drift  : {d.max_drift:.4f}")
        print(f"    per-stat   :")
        for stat, drift in sorted(d.relative_drift.items(), key=lambda kv: -kv[1]):
            print(f"      {stat:10s}  drift={drift:.4f}  hf={d.hf_stats.get(stat)}  tt={d.tt_stats.get(stat)}")
    print(sep)


def _persist_chain_divergence(result: Any, *, demo_dir: Optional[Path]) -> Optional[Path]:
    """Write the diagnostic to ``<demo_dir>/chain_divergence.json``.
    Best-effort; persistence failure never blocks escalation."""
    if result is None or demo_dir is None:
        return None
    import json
    from dataclasses import asdict

    out_path = demo_dir / "chain_divergence.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        return out_path
    except Exception as exc:
        print(
            f"  [chain-divergence] could not persist diagnostic to " f"{out_path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _run_and_log_chain_divergence(model_id: str, *, demo_dir: Optional[Path] = None) -> None:
    """Convenience: run diagnostic + log + persist in one call. Used
    at each e2e PCC fail site so the wiring is a one-liner."""
    result = _run_chain_divergence_diagnostic(model_id, demo_dir=demo_dir)
    _log_chain_divergence(result, model_id=model_id)
    _persist_chain_divergence(result, demo_dir=demo_dir)


def _resolve_model_type(model_id: str) -> str:
    """Best-effort: extract HF ``model_type`` for the family registry
    key. Returns empty string on any failure (caller treats missing
    family-key as "skip template lookup").

    Reads :func:`probe.probe_model` cached config to avoid extra HF
    network calls; falls back to ``AutoConfig.from_pretrained`` if the
    probe never ran for this model.
    """
    try:
        from .probe import probe_model

        probe = probe_model(model_id)
        mt = getattr(probe, "model_type", None) or ""
        if isinstance(mt, str) and mt:
            return mt
    except Exception:
        pass
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return str(getattr(cfg, "model_type", "")) or ""
    except Exception:
        return ""


def _list_graduated_components_for_orchestrator(demo_dir: Path) -> List[Dict[str, Any]]:
    """Read bringup_status.json and render a list of graduated
    component specs the orchestrator's synthesis prompt can consume.

    Returns [] on any error (missing manifest, malformed JSON). The
    orchestrator handles empty list gracefully.
    """
    if demo_dir is None:
        return []
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return []
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    components = data.get("components", [])
    if not isinstance(components, list):
        return []
    out: List[Dict[str, Any]] = []
    for c in components:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name:
            continue
        out.append(
            {
                "name": name,
                "stub_path": str(c.get("stub_path") or f"_stubs/{name}.py"),
                "hf_reference": str(c.get("hf_reference") or ""),
                "class_name": str(c.get("class_name") or ""),
            }
        )
    return out


def _maybe_run_e2e_orchestrator(
    *,
    model_id: str,
    demo_dir: Optional[Path],
    chain_divergence_summary: str = "",
) -> Optional[Any]:
    """Conditional entry point for the Step 1->2->3 orchestrator.

    Gated behind ``TT_HW_PLANNER_USE_E2E_ORCHESTRATOR=1`` so the
    existing cli flow is unchanged by default. When the env var is
    set, fires :func:`run_e2e_bringup` and returns its
    ``E2EBringupResult`` for the caller to act on. When unset,
    returns ``None`` (caller falls through to legacy logic).

    Best-effort: every failure mode (couldn't resolve model_type,
    couldn't read manifest, orchestrator raised) yields ``None`` so
    Path A's existing logic runs as a safety net.

    Returns the result so caller can map status → outcome label.
    """
    if os.environ.get("TT_HW_PLANNER_USE_E2E_ORCHESTRATOR") != "1":
        return None
    if not model_id or demo_dir is None:
        return None
    try:
        from ._cli_helpers.e2e_orchestrator import run_e2e_bringup

        model_type = _resolve_model_type(model_id)
        components = _list_graduated_components_for_orchestrator(demo_dir)
        return run_e2e_bringup(
            model_id=model_id,
            model_type=model_type,
            demo_dir=demo_dir,
            graduated_components=components,
            chain_divergence_summary=chain_divergence_summary,
        )
    except Exception as exc:
        print(
            f"  [e2e-orchestrator] gated path failed: " f"{type(exc).__name__}: {exc} — falling through to legacy flow",
            file=sys.stderr,
        )
        return None


def cmd_template_list(args) -> int:
    """Print every chained-template registry entry. Default: skip
    demoted (set ``--all`` to include them). Used by operators to
    inspect what families have promoted vs pending templates."""
    from ._cli_helpers.family_template_registry import (
        confirmation_count,
        list_all_templates,
    )

    entries = list_all_templates()
    if not entries:
        print("(no chained-template entries registered yet)")
        return 0
    include_demoted = getattr(args, "all", False)
    shown = 0
    for entry in entries:
        if entry.demoted and not include_demoted:
            continue
        shown += 1
        status_tag = "DEMOTED" if entry.demoted else ("PROMOTED" if entry.promoted else "REGISTERED")
        print()
        print(f"  family={entry.family_key:24}  status={status_tag}")
        print(f"    source_model      : {entry.source_model_id}")
        print(f"    template_demo_src : {entry.template_demo_source}")
        print(f"    confirmed_models  : {confirmation_count(entry)}  ({', '.join(entry.confirmed_models) or '(none)'})")
        if entry.final_pcc is not None:
            print(f"    final_pcc         : {entry.final_pcc:.4f}")
        if entry.demoted:
            print(f"    demoted_reason    : {entry.demoted_reason}")
        if entry.notes:
            print(f"    notes             : {entry.notes}")
    if shown == 0:
        print("(no non-demoted entries — pass --all to include demoted)")
    return 0


def cmd_template_promote(args) -> int:
    """Force-promote a registered chained template by family_key,
    bypassing the multi-model gate threshold.

    Useful when an operator has manually verified a template works
    for additional siblings outside the tool's auto-tracking, or
    wants to opt-in to template reuse before the second sibling
    has been brought up.
    """
    from ._cli_helpers.template_promotion import mark_promoted

    family_key = getattr(args, "family_key", None)
    if not family_key:
        print("ERROR: family_key required", file=sys.stderr)
        return 2
    entry = mark_promoted(family_key=family_key, threshold=1)
    if entry is None:
        print(
            f"ERROR: could not promote {family_key!r} — family not in registry " f"(or persistence failed)",
            file=sys.stderr,
        )
        return 1
    print(f"OK  promoted family={entry.family_key}  promoted_at={entry.promoted_at:.0f}")
    return 0


def cmd_template_demote(args) -> int:
    """Demote a chained template (mark as regressed). Future bring-ups
    in this family will skip the template and re-synthesize from scratch.

    Idempotent; safe to call repeatedly with updated reasons.
    """
    from ._cli_helpers.family_template_registry import demote_template

    family_key = getattr(args, "family_key", None)
    if not family_key:
        print("ERROR: family_key required", file=sys.stderr)
        return 2
    entry = demote_template(family_key=family_key, reason=getattr(args, "reason", "") or "")
    if entry is None:
        print(f"ERROR: could not demote {family_key!r} — family not in registry", file=sys.stderr)
        return 1
    print(f"OK  demoted family={entry.family_key}  reason={entry.demoted_reason!r}")
    return 0


def _find_demo_dir_safe(model_id: str) -> Optional[Path]:
    """Resolve the demo dir for ``model_id``, returning ``None`` on any
    error.

    Single-source-of-truth wrapper around
    :func:`bringup_loop.find_demo_dir` so the three e2e PCC fail sites
    don't each have to lazy-import + try/except. ``find_demo_dir`` itself
    already returns ``None`` on no-match, but it transitively imports
    ``discovery.BRINGUP_ROOT()`` which can raise in misconfigured envs;
    catching here keeps the diagnostic site one-liners.
    """
    try:
        from .bringup_loop import find_demo_dir

        return find_demo_dir(model_id)
    except Exception:
        return None


def _exit_if_hf_weight_failure(model_id: str, captured_output: str) -> None:
    """Short-circuit the bring-up flow if captured pytest output shows
    an HF weight download/load failure.

    HF failures (gated repo without login, network unavailable,
    corrupted .safetensors, wrong model id, etc.) are ENVIRONMENTAL —
    no amount of LLM iteration on TT code will fix them. Detecting
    them at the capture point and exiting with a clear "please
    download the weights locally and re-run" message saves the user's
    iter budget and gives them actionable remediation up front.

    Wired at every ``_run_prepare_capture`` call site (Path 2 / Path B
    / Path A post-success verification). Returns silently when no
    match — the caller then handles the failure through the normal
    repair / escalation path.

    Pattern matching lives in ``_cli_helpers/error_patterns.py``
    (generic across models, single source of truth alongside the
    other failure-signature regexes).
    """
    from ._cli_helpers.error_patterns import (
        detect_hf_weight_failure,
        format_hf_weight_failure_message,
    )

    failure = detect_hf_weight_failure(captured_output)
    if failure is None:
        return
    sys.stderr.write(format_hf_weight_failure_message(model_id, failure))
    sys.stderr.flush()
    # rc=2 distinguishes "tool bailed for setup reasons" from rc=1
    # (per-component PCC failure) and rc=0 (success). CI/downstream
    # can route on this code.
    sys.exit(2)


from ._cli_helpers.runtime_repair import _runtime_repair_loop, _PCC_FAIL_RC  # noqa: F401


def _run_strict_pcc_gate(
    args: argparse.Namespace,
    model_id: str,
    captured_output: str,
    auto_mode: bool,
) -> Tuple[Optional[Any], Optional[str]]:
    """Run the strict end-to-end PCC gate via the correctness dispatcher.

    Centralizes the gate-running block that Path 2 (ALREADY-SUPPORTED)
    and Path B (cold-start) historically duplicated, and adds the Path A
    (per-component graduate → re-verify end-to-end) wiring point.

    Returns ``(validation_result, repair_prompt)`` from
    :func:`.correctness.run_gate`. Returns ``(None, None)`` when the
    gate should NOT run for this invocation:

      * ``auto_mode`` is False (legacy non-auto invocations don't gate).
      * ``args.strict_pcc`` is False (operator opted out).
      * ``captured_output`` is empty (no demo run to gate on).

    Callers translate ``result.ok == False`` into their own escalation /
    outcome-downgrade logic. The gate intentionally does NOT mutate rc
    or the outcome banner — that belongs at the call site, where the
    routing decision (escalate to Path A, mark UNVERIFIED, etc.) lives.

    Category probe is best-effort: if probe fails, defaults to ``"LLM"``
    (the broad-cover comparator) rather than skipping the gate. Lets the
    text comparator run for LLMs/VLMs even when the probe can't classify.
    """
    if not (auto_mode and getattr(args, "strict_pcc", True)):
        # Loud about which knob disabled the gate — silent skips here
        # used to hide bugs where the SUCCESS banner stamped rc=0 even
        # though strict_pcc was off and no numerical check ever ran.
        if not auto_mode:
            print("  PCC gate: skipped (auto_mode=False — legacy non-auto invocations do not gate).")
        else:
            print("  PCC gate: skipped (--no-strict-pcc / args.strict_pcc=False — operator opted out).")
        return None, None
    if not captured_output:
        # Same — never silent. An empty captured_output here means the
        # pytest-output capture pipeline broke (pump thread, sink path
        # wiring, etc.) so the diagnostic should be visible.
        print(
            "  PCC gate: skipped (captured_output is empty). pytest produced "
            "output to the terminal but the tee'd capture file came back "
            "empty — _pytest_capture_sink wiring is likely broken."
        )
        return None, None
    from .correctness import run_gate as _correctness_run_gate

    _gate_category = "LLM"
    try:
        from .probe import probe_model as _probe_for_gate

        _gp = _probe_for_gate(model_id)
        _gate_category = getattr(_gp, "category", None) or "LLM"
        if _gate_category == "Unknown":
            _gate_category = "LLM"
    except Exception:
        pass
    return _correctness_run_gate(
        category=_gate_category,
        model_id=model_id,
        captured_output=captured_output,
        args=args,
        engine=getattr(args, "pcc_engine", "legacy"),
        compare_tokens=getattr(args, "strict_pcc_tokens", None),
        instruct=not getattr(args, "no_instruct", False),
    )


def _run_pcc_gate(
    *,
    model_id: str,
    captured_output: str,
    args: argparse.Namespace,
    compare_tokens: Optional[int] = None,
    instruct: bool = True,
) -> Tuple[Optional[Any], Optional[str]]:
    """Run the PCC gate against a captured demo run.

    Returns ``(result, prompt_used)`` where ``result`` is an
    :class:`output_validation.ValidationResult` or ``None`` if the
    gate could not run (no captured output, no parseable TT output,
    HF load failed, ...). In the "could not run" cases the caller
    should treat the gate as a soft pass (don't fail the build) but
    print a warning so the user knows the false-green safety net is
    not engaged for this run.

    The gate intentionally does NOT mutate the caller's rc; the
    caller is responsible for translating a failing
    :class:`ValidationResult` into the PCC-fail exit code."""
    from .output_validation import (
        compare_token_sequences,
        DEFAULT_COMPARE_TOKENS,
        extract_demo_user_output,
        generate_hf_reference,
        load_demo_first_prompt,
        tokenize_text_for_compare,
    )

    n = compare_tokens or DEFAULT_COMPARE_TOKENS

    tt_text = extract_demo_user_output(captured_output, user_idx=0)
    if tt_text is None:
        print(
            "  PCC gate: could not find a '==USER 0 - OUTPUT' block in "
            "the pytest output. Skipping the gate (the demo may not be "
            "simple_text_demo, or the test path may not emit the "
            "canonical output marker)."
        )
        return None, None
    if not tt_text.strip():
        print(
            "  PCC gate: TT demo printed an empty output block. "
            "Treating as a FALSE GREEN (the demo passed pytest but "
            "decoded zero tokens)."
        )

        from .output_validation import ValidationResult

        return (
            ValidationResult(
                ok=False,
                reason=(
                    "TT demo produced an empty decoded-text block; "
                    "this is almost certainly a silent generation "
                    "failure that pytest didn't catch."
                ),
                tt_text="",
                hf_text="",
                compared_tokens=0,
            ),
            None,
        )

    prompt = load_demo_first_prompt()
    if not prompt:
        print(
            "  PCC gate: could not load the demo's first prompt from "
            f"{'/'.join(['models', 'tt_transformers', 'demo', 'sample_prompts', 'input_data_questions_prefill_128.json'])}. "
            "Skipping the gate (the prompts file is unexpectedly "
            "missing or malformed)."
        )
        return None, None

    print()
    print(
        "  PCC gate: running HF CPU reference for the same prompt "
        f"(model={model_id}, first {n} tokens, greedy). This adds "
        f"~30s-3min depending on model size."
    )
    try:
        hf = generate_hf_reference(
            model_id,
            prompt,
            max_new_tokens=n,
            instruct=instruct,
        )
    except Exception as exc:
        print(
            f"  PCC gate: HF reference generation FAILED "
            f"({type(exc).__name__}: {exc}). Skipping the gate (the "
            f"build will pass on the basis of pytest alone; this "
            f"could mean a HF cache miss, a VLM that needs special "
            f"loading, or HF_TRUST_REMOTE_CODE not being set)."
        )
        return None, prompt

    if hf.truncated:
        print(
            "  PCC gate: HF reference hit the per-token wall-clock "
            "budget before reaching the full token count. Comparing "
            f"on the {len(hf.token_ids)} tokens we did get."
        )

    try:
        tt_token_ids = tokenize_text_for_compare(model_id, tt_text)
    except Exception as exc:
        print(f"  PCC gate: tokenizer re-load FAILED " f"({type(exc).__name__}: {exc}). Skipping the gate.")
        return None, prompt

    result = compare_token_sequences(
        tt_token_ids,
        hf.token_ids,
        tt_text=tt_text,
        hf_text=hf.text,
        compare_tokens=n,
    )
    print(f"  {result.summary()}")
    if not result.ok:
        print()
        print("  ----- TT-demo output (first 200 chars) -----")
        print(f"  {(result.tt_text or '')[:200]}")
        print("  ----- HF-reference output (first 200 chars) -----")
        print(f"  {(result.hf_text or '')[:200]}")
        print("  --------------------------------------------------")
    return result, prompt


# pcc_repair removed 2026-05-31 — the whole-model retry loop was a
# wrong abstraction. Path 2 (ALREADY-SUPPORTED) now escalates directly
# to Path 1 (scaffold + per-component iterate) via
# _maybe_escalate_pcc_fail. See cli.py:7410 and cli.py:7741 for the
# escalation call sites.


# Outcome labels emitted by ``_final_outcome_banner``. Kept as module
# constants (not an enum) so they grep cleanly out of CI logs and stay
# stable references for downstream tooling that scrapes the banner.
#
# - ``SUCCESS``         — model verified end-to-end; both gates fired and passed
# - ``SUCCESS-PARTIAL`` — components graduated but some on CPU fallback
#                         (KERNEL_MISSING); kernel gap report attached
# - ``UNVERIFIED``      — bring-up completed but strict gate did NOT fire
#                         (logits not captured, comparison incomplete, etc.).
#                         The run is NOT a confirmed SUCCESS — downstream
#                         consumers should treat it as needing re-verification.
# - ``FAIL``            — bring-up did not converge / strict gate failed
OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_SUCCESS_PARTIAL = "SUCCESS-PARTIAL"
OUTCOME_UNVERIFIED = "UNVERIFIED"
OUTCOME_FAIL = "FAIL"


def _final_outcome_banner(
    *,
    rc: int,
    model_id: str,
    path_label: str,
    extra: Optional[List[str]] = None,
    outcome: Optional[str] = None,
    demo_dir: Optional[Path] = None,
) -> None:
    """2026-05-23 (bugfix B3): always emit a single, machine-grep-able
    final-outcome banner before cmd_up returns. The previous
    behaviour was to just return ``rc`` (often ``1`` for pytest
    failures) with NO cmd_up-level summary. The user would see
    pytest's own failure block, then nanobind ref-leak chatter, then
    silence -- with no clear "did tt_hw_planner succeed or fail?"
    signal. This helper prints a fixed-format banner that's also
    easy for ``grep`` and CI logs to spot.

    Lines emitted:
      ===
      TT_HW_PLANNER OUTCOME: SUCCESS|SUCCESS-PARTIAL|UNVERIFIED|FAIL  rc=<int>  model=<id>  path=<label>
      Suggested next steps:
        - <line>
        - <line>
      ===

    Parameters
    ----------
    outcome
        Explicit outcome label. When ``None`` (default), falls back to
        the legacy rc-derived label (``rc==0`` → SUCCESS, else FAIL).
        Pass an explicit value (e.g. ``OUTCOME_UNVERIFIED`` or
        ``OUTCOME_SUCCESS_PARTIAL``) when the rc alone can't express
        the verdict — e.g. per-component PCC passed but the end-to-end
        strict gate could not fire, which is NOT a clean SUCCESS even
        though rc is 0.
    """
    sep = "=" * 72
    print()
    print(sep)
    if outcome is not None:
        label = outcome
    else:
        label = OUTCOME_SUCCESS if rc == 0 else OUTCOME_FAIL
    print(f"  TT_HW_PLANNER OUTCOME: {label}  rc={rc}  " f"model={model_id}  path={path_label}")
    if extra:
        print(f"  Suggested next steps:")
        for line in extra:
            print(f"    - {line}")

    # Surface static-analysis kernel findings (computed once at scaffold
    # time, persisted to <demo_dir>/kernel_findings.json). When a run
    # fails on something the planner had already flagged hours earlier
    # (Phi-3.5 case: head_dim=96 → TT_FATAL at rotary_embedding_hf), the
    # operator needs to see the connection in the final banner instead
    # of scrolling through multi-hour logs to find Step 1 output.
    # If callers don't pass demo_dir explicitly, auto-discover it from
    # the model_id so every existing _final_outcome_banner call benefits
    # without per-site patches.
    _dd = demo_dir
    if _dd is None:
        try:
            from .bringup_loop import find_demo_dir as _find_demo_dir_for_banner

            _dd = _find_demo_dir_for_banner(model_id)
        except Exception:
            _dd = None
    if _dd is not None:
        try:
            from ._cli_helpers.kernel_findings import format_kernel_findings_for_banner, load_kernel_findings

            findings = load_kernel_findings(Path(_dd))
            findings_lines = format_kernel_findings_for_banner(findings)
            if findings_lines:
                print()
                for line in findings_lines:
                    print(f"  {line}")
        except Exception:
            pass

        # 2026-06-03: surface harness-skipped components. These are
        # components whose auto-generated PCC test SKIPPED at the
        # harness layer (couldn't synthesize inputs, path resolved
        # to a container, etc.). They were silently dropped from the
        # iteration queue and would otherwise be counted as graduated.
        # Showing them here makes false-graduation runs honest.
        try:
            import json as _json

            harness_skipped_path = Path(_dd) / "harness_skipped.json"
            if harness_skipped_path.is_file():
                _data = _json.loads(harness_skipped_path.read_text())
                _comps = _data.get("harness_skipped_components", []) if isinstance(_data, dict) else []
                if _comps:
                    print()
                    print(
                        f"  ⚠ HARNESS-SKIPPED ({len(_comps)} component(s)): the auto-PCC test could not "
                        f"build inputs and SKIPPED at the harness layer. These were NOT graduated "
                        f"despite the 'all graduated' banner — they need a test-fixture fix:"
                    )
                    for c in _comps:
                        print(f"    - {c}")
                    print(
                        f"  → Fix candidate submodule paths or _make_arg_for() kwargs in "
                        f"tests/pcc/test_<comp>.py, or invoke the skip-diagnoser."
                    )
        except Exception:
            pass

        # 2026-06-03: surface skip_diagnosis.json verdicts. When the
        # iter loop ran the LLM skip_diagnoser on harness-skipped
        # components at end-of-loop, each component got a verdict:
        # fixed / decompose / manual / unknown. Show verdict counts +
        # any "fixed" actions so the operator knows whether to re-run.
        try:
            import json as _json

            diag_path = Path(_dd) / "skip_diagnosis.json"
            if diag_path.is_file():
                _diag_data = _json.loads(diag_path.read_text())
                _diagnoses = _diag_data.get("diagnoses", []) if isinstance(_diag_data, dict) else []
                if _diagnoses:
                    print()
                    print(f"  SKIP-DIAGNOSER ran on {len(_diagnoses)} harness-skipped component(s):")
                    by_verdict: Dict[str, List[str]] = {}
                    for d in _diagnoses:
                        v = str(d.get("verdict") or "unknown")
                        by_verdict.setdefault(v, []).append(str(d.get("component") or "?"))
                    for v, comps in sorted(by_verdict.items()):
                        glyph = {
                            "fixed": "✓",
                            "decompose": "⤴",
                            "manual": "✋",
                            "unknown": "?",
                        }.get(v, "?")
                        print(f"    {glyph} {v}: {len(comps)} component(s)")
                        for c in comps[:5]:
                            print(f"        - {c}")
                        if len(comps) > 5:
                            print(f"        ... and {len(comps) - 5} more")
                    if by_verdict.get("fixed"):
                        print(f"  → Re-run `up` to pick up the test-fixture fixes " f"and re-test these components.")
        except Exception:
            pass
    print(sep)


def _register_bringup_success(
    model_id: str,
    *,
    path: str,
    sep: str = "=" * 72,
    notes: str = "",
) -> None:
    """Persist a successful bring-up back into the tool's own registry
    so the NEXT similar model gets an `exact` match without going
    through inline auto-onboard.

    Probes the model to derive ``(model_type, category)`` and re-picks
    the backend (which by now is what was actually used to run the
    model). Then delegates to ``learning.register_successful_bringup``
    which writes ``family_backends.py``, ``compatibility.py``, and the
    audit log.

    Best-effort: every step is wrapped so a write failure here can't
    turn a successful bring-up into a failed return code. The user
    just doesn't get the "next time is faster" benefit if this
    function fails."""
    try:
        from .learning import register_successful_bringup
        from .probe import probe_model
        from .family_backends import pick_backend_with_quality
    except Exception as exc:
        print(f"  (learning registration skipped: import failed " f"{type(exc).__name__}: {exc})")
        return
    try:
        probe = probe_model(model_id)
    except Exception as exc:
        print(f"  (learning registration skipped: probe failed " f"{type(exc).__name__}: {exc})")
        return
    model_type = ""
    pipeline_tag = None
    try:
        model_type = str((probe.raw_config or {}).get("model_type") or "")
        pipeline_tag = getattr(probe, "pipeline_tag", None)
    except Exception:
        pass
    backend, quality = pick_backend_with_quality(
        category=probe.category,
        model_type=model_type,
        pipeline_tag=pipeline_tag,
    )
    if backend is None:
        print("  (learning registration skipped: no backend resolved " "post-success, nothing to update)")
        return
    if not model_type:
        print("  (learning registration skipped: model has no " "config.model_type to register)")
        return

    if quality != "exact":
        print(
            f"  (learning registration skipped: backend match quality "
            f"is {quality!r}, not 'exact' -- refusing to canonize a "
            f"closest-fit mapping for {backend.name!r} <- {model_type!r}. "
            f"If this bring-up is genuinely correct, add the model_type "
            f"explicitly to the right backend's model_type_keys.)"
        )
        return
    print()
    print(sep)
    print("  LEARNING  registering this successful bring-up so the next")
    print("  similar model finds it as a sibling and skips inline auto-onboard.")
    print(sep)
    msgs = register_successful_bringup(
        model_id=model_id,
        model_type=model_type,
        category=probe.category,
        backend_name=backend.name,
        sibling_model_id=model_id,
        path=path,
        notes=notes,
    )
    for m in msgs:
        print(f"    {m}")
    print(sep)


def _try_auto_onboard_inline(
    *,
    model_id: str,
    category: str,
    model_type: str,
    pipeline_tag: Optional[str],
    closest_backend: Optional[Any],
) -> Optional[Tuple[Any, str]]:
    """2026-05-23: inline auto-onboard during ``up --auto``. When the
    loud-fallback gate would have aborted (category-default match for a
    template-routing backend) OR there is no backend at all, try to
    LLM-draft a new FamilyBackend entry, splice it into the registry,
    and re-pick.

    Returns ``(backend, quality)`` on success, or ``None`` on any
    failure (no agent in PATH, LLM validation errors, registry write
    fails). The caller is expected to gracefully fall back to a
    closest-template attempt or to cold-start in that case -- NEVER
    abort.

    This is what turns the tool from "if it can't find a sibling, ask
    the user to do something" into "if it can't find a sibling, the
    tool drafts one itself"."""
    try:
        from .auto_onboard import auto_onboard, write_backend_into_registry
    except Exception as exc:
        print(f"  (auto-onboard import failed: {type(exc).__name__}: {exc}; " f"skipping inline draft)")
        return None

    agent_bin = None
    for cand in ("claude", "codex"):
        try:
            from shutil import which

            if which(cand):
                agent_bin = cand
                break
        except Exception:
            pass
    if agent_bin is None:
        print(
            "  (no LLM agent in PATH; skipping inline auto-onboard " "draft and falling back to closest-template path)"
        )
        return None
    sep = "-" * 72
    print()
    print(sep)
    print(f"  INLINE AUTO-ONBOARD  no exact backend match for " f"`{model_id}`; drafting one now.")
    _closest_name = closest_backend.name if closest_backend is not None else "(none)"
    print(
        f"  (model_type={model_type!r}, pipeline_tag={pipeline_tag!r}, "
        f"category={category!r}; closest existing = `{_closest_name}`)"
    )
    print(sep)
    try:
        proposal = auto_onboard(
            model_id,
            agent_bin=agent_bin,
            timeout_s=240,
            skip_llm=False,
        )
    except Exception as exc:
        print(
            f"  inline auto-onboard failed: " f"{type(exc).__name__}: {exc}; falling back to " f"closest-template path."
        )
        return None
    if proposal.validation_errors:
        print(f"  inline auto-onboard produced " f"{len(proposal.validation_errors)} validation error(s):")
        for e in proposal.validation_errors[:5]:
            print(f"    - {e}")
        print("  Falling back to closest-template path.")
        return None
    require_review = os.environ.get("TT_HW_PLANNER_AUTO_ONBOARD_REQUIRE_REVIEW", "1").strip().lower()
    if require_review not in ("0", "false", "no", "off"):
        proposal_dir = REPO_ROOT / "scripts" / "tt_hw_planner" / "_proposals"
        try:
            proposal_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
            proposal_path = proposal_dir / f"{safe_name}.proposal.py"
            header = (
                f"# AUTO-ONBOARD PROPOSAL — NOT YET APPLIED\n"
                f"# model_id   : {model_id}\n"
                f"# category   : {category}\n"
                f"# model_type : {model_type}\n"
                f"# closest    : {_closest_name}\n"
                f"# To accept and write into family_backends.py, run:\n"
                f"#   python -m scripts.tt_hw_planner auto-onboard {model_id} --accept\n"
                f"# To bypass approval inline (NOT RECOMMENDED — see\n"
                f"# family_backends.py:157-166 for the previous empty-backend incident):\n"
                f"#   TT_HW_PLANNER_AUTO_ONBOARD_REQUIRE_REVIEW=0 python -m scripts.tt_hw_planner up ...\n"
            )
            proposal_path.write_text(header + "\n" + (proposal.backend_dataclass_source or ""))
            rel = safe_relative_to_root(proposal_path) if proposal_path.is_absolute() else proposal_path
            print(
                f"  REVIEW REQUIRED: proposal written to `{rel}` but NOT applied. "
                f"Run `python -m scripts.tt_hw_planner auto-onboard {model_id} --accept` "
                f"after reviewing. Falling back to closest-template path for this run."
            )
        except Exception as exc:
            print(
                f"  auto-onboard review-gate save failed: "
                f"{type(exc).__name__}: {exc}; falling back to closest-template path."
            )
        return None
    ok, msg = write_backend_into_registry(proposal)
    print(f"  {msg}")
    if not ok:
        print("  Falling back to closest-template path.")
        return None
    try:
        from .family_backends import pick_backend_with_quality as _repick

        proposal_category = getattr(proposal, "inferred_category", None) or category
        new_backend, new_quality = _repick(
            category=proposal_category,
            model_type=model_type,
            pipeline_tag=pipeline_tag,
        )
    except Exception as exc:
        print(
            f"  re-pick after auto-onboard failed: "
            f"{type(exc).__name__}: {exc}; falling back to "
            f"closest-template path."
        )
        return None
    if new_backend is None or new_quality == "category-default":
        print("  re-pick after auto-onboard still returned " "category-default; falling back to closest-template path.")
        return None
    return new_backend, new_quality


def _emit_and_verify_runnable_demo(
    model_id: str,
    *,
    sep: str = "=" * 72,
    repo_root: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Post-convergence: auto-emit `demo.py` from captured inputs and run
    `pytest demo.py::test_demo` to verify the model is end-to-end runnable
    on device. This closes the long-standing tool gap where convergence
    only meant "PCC tests pass" but the scaffolded `demo.py` (a stale copy
    from the sibling model) didn't match the LLM-written `__call__`
    signatures and crashed on first invocation.

    Returns `(ok, message)` where `ok=True` iff the demo passed pytest.
    Caller is expected to surface `message` in the convergence banner.
    Failures are non-fatal — the model IS still on device, the PCC suite
    proved it; the demo verification is a final usability check on top."""
    rr = repo_root or BRINGUP_ROOT()
    print()
    print(sep)
    print(f"  AUTO-EMIT runnable demo for {model_id}")
    print(sep)
    try:
        demo_path, status = emit_runnable_demo(model_id=model_id, repo_root=rr)
    except Exception as exc:
        msg = (
            f"  demo auto-emit raised {type(exc).__name__}: {exc} — "
            f"the model is on device (PCC suite passed) but the user-"
            f"facing runnable demo could not be regenerated. Edit "
            f"`demo.py` manually or re-run with `--regen-demo-only`."
        )
        print(msg)
        return False, msg
    if status == "no-demo-dir":
        msg = (
            f"  demo auto-emit skipped: model `{model_id}` has not been "
            f"scaffolded yet (no `bringup_status.json` on disk)."
        )
        print(msg)
        return False, msg
    if status == "no-primary":
        msg = (
            f"  demo auto-emit skipped: no graduated NEW component has "
            f"captured inputs on disk (`_captured/<comp>/*.pt`). The "
            f"`up` flow normally captures inputs during scaffolding; "
            f"re-run `python -m scripts.tt_hw_planner capture-inputs "
            f"{model_id}` then `--regen-demo-only`."
        )
        print(msg)
        return False, msg
    if demo_path is None:
        msg = "  demo auto-emit returned no path (internal error)."
        print(msg)
        return False, msg
    rel = demo_path.relative_to(rr) if str(rr) in str(demo_path) else demo_path
    print(f"  emitted: {rel}")

    test_target = f"{rel}::test_demo"
    print(f"  verifying: pytest {test_target} -v")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(rel) + "::test_demo", "-v"],
            cwd=str(rr),
            capture_output=True,
            text=True,
            timeout=300,
        )
        rc = proc.returncode
        combined = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        rc = 124
        combined = f"(demo verification timed out after 300s: {exc})"

    def _extract_pytest_failure_tail(text: str, max_lines: int = 50) -> str:
        """Pull the actually-useful failure context from pytest output,
        skipping post-test nanobind/ttnn diagnostic spam.

        Strategy: prefer lines containing the ACTUAL ERROR signals
        (``RuntimeError:``, ``AssertionError``, ``E   ``, file paths
        like ``.py:NN`` and ``FAILED ``). Always include these even
        if they appear far from the FAILURES header (pytest's
        function-source dump can push the real error 50+ lines down).
        """
        lines = text.splitlines()
        spam_substrings = (
            "leaked function",
            "leaked instance",
            "leaked type",
            "nanobind:",
            "build_cache_telemetry",
            "JitBuildState",
            "BuildKernels",
            "tt_cluster.cpp",
            "Cluster destructor",
            "Closing user mode device",
            "Closing devices in cluster",
            "device_manager.cpp",
            "skipped remainder",
        )
        error_signals = (
            "RuntimeError",
            "AssertionError",
            "TypeError",
            "AttributeError",
            "ValueError",
            "ImportError",
            "ModuleNotFoundError",
            "TT_FATAL",
            "FAILED ",
            "[CPU_FALLBACK]",
            "_CPU_FALLBACK]",
            "E   ",
        )

        def _is_useful(line: str) -> bool:
            s = line.strip()
            if not s:
                return False
            return not any(spam in s for spam in spam_substrings)

        def _is_signal(line: str) -> bool:
            return any(sig in line for sig in error_signals) or bool(re.search(r"\.py:\d+", line))

        useful = [ln for ln in lines if _is_useful(ln)]

        # Find signal lines (the real error info).
        signal_indices = [i for i, ln in enumerate(useful) if _is_signal(ln)]
        if signal_indices:
            # Take a window AROUND every signal line — context above
            # gives traceback frames, context below gives the error
            # message + summary.
            keep = set()
            for idx in signal_indices:
                lo = max(0, idx - 5)
                hi = min(len(useful), idx + 5)
                for j in range(lo, hi):
                    keep.add(j)
            selected = [useful[j] for j in sorted(keep)]
            # Cap to max_lines while ensuring the last signal line
            # is included (the actual error usually appears late).
            if len(selected) > max_lines:
                # Prefer the LAST max_lines of signal+context (tail of
                # the failure block).
                selected = selected[-max_lines:]
            return "\n".join(selected)

        # No specific signals → fall back to last N useful lines.
        return "\n".join(useful[-max_lines:])

    tail = _extract_pytest_failure_tail(combined)
    if rc == 0:
        msg = f"  demo verification PASSED — `pytest {test_target}` is now green."
        print(msg)
        print(sep)
        return True, msg

    # Brain (G8) demo-recovery: parses the pytest tail, identifies the
    # broken wired component (PCC passed but runtime shapes differ), and
    # disables it from the demo's WIRED_COMPONENTS so the rest of the
    # pipeline runs end-to-end in mixed mode. Falls back to retry/archive
    # actions for orphan-sibling and flake cases.
    from .agentic.demo_recovery import (
        archive_demo_files,
        decide_demo_recovery,
        remove_component_from_wiring,
    )

    def _extract_wired_components(demo_p: Path) -> list:
        """Pull display-name strings out of WIRED_COMPONENTS in the
        emitted demo. Best-effort; returns [] on any parse issue."""
        try:
            txt = demo_p.read_text()
        except Exception:
            return []
        # Each tuple: ('submodule_path', '...stubs.<comp>', '<display_name>'),
        names = re.findall(
            r"\(\s*'[^']+'\s*,\s*'[^']+_stubs\.[^']+'\s*,\s*'([^']+)'\s*\)",
            txt,
        )
        return names

    retries_attempted = 0
    # max_retries scales with the number of wired components — the
    # brain's fallback disables one component per retry, so we need
    # enough budget to potentially shrink down to a passing minimal
    # demo. Capped at 6 so a broadly-broken model doesn't loop forever.
    initial_wired_count = len(_extract_wired_components(demo_path))
    max_retries = min(6, max(2, initial_wired_count))
    demo_dir = demo_path.parent
    model_demo_dir = demo_dir.parent if demo_dir.name == "demo" else demo_dir
    # The parser needs the FULL pytest output (not the spam-filtered tail)
    # to find file paths like `_stubs/<comp>.py` that the extractor may
    # have filtered out. Tail stays for user-facing display only.
    full_pytest_output = combined
    while retries_attempted < max_retries:
        wired = _extract_wired_components(demo_path)
        verdict = decide_demo_recovery(
            demo_dir=model_demo_dir,
            canonical_demo=demo_path,
            retries_attempted=retries_attempted,
            max_retries=max_retries,
            pytest_tail=full_pytest_output,
            wired_components=wired,
        )
        print(f"  [brain G8] demo-recovery: {verdict.action} — {verdict.reason}")
        if verdict.action == "give_up":
            break
        if verdict.action == "archive_and_retry" and verdict.archive_paths:
            archived = archive_demo_files(verdict.archive_paths)
            for a in archived:
                print(f"    archived: {a.name}")
        if verdict.action == "disable_component_and_retry":
            broken = getattr(verdict, "broken_component", None)
            if broken:
                if remove_component_from_wiring(demo_path=demo_path, component=broken):
                    print(f"    disabled wiring for `{broken}` in demo")
                else:
                    print(f"    could not disable wiring for `{broken}` — proceeding with retry anyway")
        retries_attempted += 1
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(rel) + "::test_demo", "-v", "--tb=long"],
                cwd=str(rr),
                capture_output=True,
                text=True,
                timeout=300,
            )
            rc = proc.returncode
            combined = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rc = 124
            combined = f"(demo verification retry timed out after 300s: {exc})"
        tail = _extract_pytest_failure_tail(combined)
        full_pytest_output = combined
        if rc == 0:
            msg = f"  demo verification PASSED on retry — `pytest {test_target}` is now green " f"(brain G8 recovered)"
            print(msg)
            # Brain owns end-to-end delivery: if the recovery modified
            # the demo (e.g. removed broken wiring), the modified file
            # must land in main tree so the user can run the demo
            # without manually re-applying the brain's fix.
            try:
                from .agentic.persistence import sync_demo_to_main_tree

                _demo_sync = sync_demo_to_main_tree(worktree_demo_path=demo_path)
                if _demo_sync.status == "synced":
                    print(f"  [brain G8] synced recovered demo to main tree → {_demo_sync.synced_path}")
                elif _demo_sync.status == "sync_failed":
                    # B-FIX #12 (2026-05-31): a sync FAILURE used to look
                    # identical to "not in a worktree". The brain's recovery
                    # work product would silently NOT land in main tree and
                    # the cli would still report PASSED. Now distinguish.
                    print(
                        f"  [brain G8] WARN: recovered demo did NOT sync to main tree — "
                        f"{_demo_sync.reason}. The demo passed in worktree but main "
                        f"tree may not reflect the brain's fix; user must manually "
                        f"copy the demo from {demo_path}."
                    )
                # noop_not_in_worktree / noop_worktree_is_main_tree are
                # legitimate quiet no-ops — no surfacing needed.
            except Exception as _demo_sync_exc:
                print(
                    f"  [brain G8] demo sync non-fatal: " f"{type(_demo_sync_exc).__name__}: {_demo_sync_exc}",
                    file=sys.stderr,
                )
            print(sep)
            return True, msg

    # B1-FIX: call the brain ONE MORE TIME after the retry loop exits
    # so its give_up verdict reason gets surfaced. Without this, the
    # brain's "all N retries exhausted" reasoning was unreachable and
    # the user only saw the generic FAIL message.
    try:
        wired = _extract_wired_components(demo_path)
        give_up_verdict = decide_demo_recovery(
            demo_dir=model_demo_dir,
            canonical_demo=demo_path,
            retries_attempted=max_retries,
            max_retries=max_retries,
            pytest_tail=tail,
            wired_components=wired,
        )
        if give_up_verdict.action == "give_up":
            print(f"  [brain G8] demo-recovery: give_up — {give_up_verdict.reason}")
    except Exception as _give_up_exc:
        print(
            f"  [brain G8] give-up logging non-fatal: " f"{type(_give_up_exc).__name__}: {_give_up_exc}",
            file=sys.stderr,
        )

    msg = (
        f"  demo verification FAILED (pytest rc={rc}). The model is on "
        f"device — the PCC suite already proved it. The demo emitter "
        f"may need a model-specific tweak for the primary entry point. "
        f"Last pytest output:\n{tail}"
    )
    print(msg)
    print(sep)
    return False, msg


def _print_bringup_summary(model_id: str, *, box: str, sep: str = "=" * 72) -> None:
    cats = _classify_components(model_id)
    reuse = cats["reuse"]
    adapt = cats["adapt"]
    native = cats["new_native"]
    fallback = cats["new_fallback"]
    total = len(reuse) + len(adapt) + len(native) + len(fallback)

    from .bringup_loop import find_demo_dir

    _demo_dir = find_demo_dir(model_id)
    _has_status_file = bool(_demo_dir and (_demo_dir / "bringup_status.json").is_file())
    coverage_source = "runtime (bringup_status.json)" if _has_status_file else "static (compat report)"

    print(sep)
    print(f"  Bring-up summary for {model_id}")
    print(f"  Coverage source: {coverage_source}")
    if not _has_status_file and total > 0:
        print(
            "    NOTE: this run took the ALREADY-SUPPORTED path; component "
            "status is derived from the compat report's static analysis, "
            "not from per-op runtime tracking. The pytest run above confirms "
            "the demo executes end-to-end, but doesn't prove every block "
            "fired on TT. For true runtime device-coverage, run with the "
            "agentic probe (TT_PLANNER_PROBE_OUTPUT=<path>) or with "
            "--strict-pcc which captures + compares per-layer activations."
        )
    print(sep)
    print(f"  total tracked components: {total}")
    print(f"    REUSE (existing TTNN op)            : {len(reuse)}")
    print(f"    ADAPT (existing TTNN, light port)   : {len(adapt)}")
    print(f"    NEW   (native TTNN, synthesized)    : {len(native)}")
    print(f"    NEW   (CPU fallback / torch-ref)    : {len(fallback)}")
    print()
    for line in _format_compute_split(model_id, label="components", indent="  "):
        print(line)
    for line in _format_op_split(model_id, label="operations", indent="  ", show_per_component=True):
        print(line)
    if fallback:
        from .overlay_manager import load_locked_modules, load_no_emit_tests

        _locked = set(load_locked_modules(model_id).keys())
        _no_emit = set(load_no_emit_tests(model_id).keys())
        recompose_pending = [n for n in fallback if n in _locked]
        decomposed_pending = [n for n in fallback if n in _no_emit and n not in _locked]
        true_fallback = [n for n in fallback if n not in _locked and n not in _no_emit]

        if recompose_pending:
            print()
            print("  PENDING — recompose in progress (locked; re-iterating the whole module):")
            for n in recompose_pending:
                print(f"    - {n}")
        if decomposed_pending:
            print()
            print("  PENDING — decomposed (recomposes once all its children are on device):")
            for n in decomposed_pending:
                print(f"    - {n}")
        if true_fallback:
            print()
            print("  components still on CPU fallback (torch reference):")
            for n in true_fallback:
                print(f"    - {n}")
        if recompose_pending or true_fallback:
            print()
            print("  to promote these to native TTNN, run:")
            print(f"    python -m scripts.tt_hw_planner promote {model_id} --box {box} --auto")


def _prompt_for_api_key(provider: str) -> Optional[str]:
    env_var = _API_KEY_ENV_VAR.get(provider)
    if not env_var:
        return None
    label = _PROVIDER_LABEL.get(provider, provider)
    if not sys.stdin.isatty():
        print(
            f"\n  --auto requested but {label} credentials are missing and stdin is not a TTY.",
            file=sys.stderr,
        )
        print(
            f"  Set {env_var} in the environment before re-running, or skip --auto.",
            file=sys.stderr,
        )
        return None
    print(f"\n  --auto requested but no {label} credentials were found.")
    print(f"  Paste a {label} API key now to enable the autonomous LLM loop,")
    print(f"  or press Enter to skip (Phase-1 CPU fallback will still be in place).")
    try:
        import getpass

        key = getpass.getpass(f"  {env_var} (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not key:
        return None
    os.environ[env_var] = key
    return key


def _collect_agent_descendant_pids(root_pid: int) -> List[int]:
    """Return the full descendant tree of `root_pid` by walking
    `/proc/<pid>/task/<tid>/children` (the proper Linux kernel API
    for "list this process's direct children" — independent of
    session/process-group membership).

    Why this exists. `_invoke_agent` previously killed the agent via
    `os.killpg(os.getpgid(proc.pid), SIGTERM/SIGKILL)`. That signals
    every process in the SAME process group, which works for normal
    children but FAILS for processes that internally `setsid()` or
    `fork()+exit` to detach from the parent's group — exactly what the
    claude CLI does (the wrapper exits after spawning a long-lived
    worker, leaving the worker reparented to init/PID 1 once the
    wrapper is killed). The orphan we observed on 2026-05-22 ran for
    40+ minutes after the planner declared "killing agent process
    tree" because the worker was outside the killable group.

    /proc-based descendant walking sidesteps this: parent->child links
    persist regardless of session/group changes, AS LONG AS the parent
    is still alive. The caller MUST call this BEFORE killing the
    immediate child — once `proc.pid` exits, its descendants get
    reparented to init and the link is lost forever."""
    out: List[int] = []
    stack: List[int] = [root_pid]
    seen: set = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            task_dir = Path(f"/proc/{pid}/task")
            if not task_dir.is_dir():
                continue
            for task in task_dir.iterdir():
                children_file = task / "children"
                if not children_file.is_file():
                    continue
                try:
                    txt = children_file.read_text().strip()
                except Exception:
                    continue
                for tok in txt.split():
                    try:
                        cpid = int(tok)
                    except ValueError:
                        continue
                    if cpid == pid or cpid in seen:
                        continue
                    out.append(cpid)
                    stack.append(cpid)
        except Exception:
            continue
    return out


def _kill_process_tree(proc, *, label: str) -> None:
    """Robustly terminate `proc` AND its entire descendant tree.

    Strategy, in order:
      1. Collect ALL descendants via /proc walking BEFORE signaling
         anything — once the immediate child dies, its kids get
         reparented to init and we can't trace them anymore.
      2. Send SIGTERM to the process group (catches in-group siblings)
         AND to every descendant PID directly (catches the
         setsid()-detached workers killpg misses). Both signals are
         best-effort — silent failures are fine because the next step
         escalates.
      3. Wait up to 10s for `proc` to exit; many CLIs handle SIGTERM
         cleanly and we want them to print final state if possible.
      4. Escalate to SIGKILL on both the group AND every still-alive
         descendant. SIGKILL cannot be caught, so anything still
         breathing after this is a kernel-level zombie or D-state
         (uninterruptible sleep) and we give up.
      5. Wait 5 more seconds for reaping.

    `label` is used in diagnostic log lines (e.g. `[auto:claude]`,
    `[pytest]`) so a future log audit can attribute kills to their
    source. Returns nothing. All exceptions are swallowed — the
    caller has no recourse if a process refuses to die, but we still
    want the surrounding loop to continue."""
    import os
    import signal
    import subprocess

    descendants = _collect_agent_descendant_pids(proc.pid)

    def _signal_all(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            pass
        for cpid in descendants:
            try:
                os.kill(cpid, sig)
            except Exception:
                pass

    _signal_all(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if descendants:
            print(
                f"  [{label}] SIGTERM didn't reap process + "
                f"{len(descendants)} descendant(s); escalating to "
                f"SIGKILL.",
                file=sys.stderr,
            )
        else:
            print(
                f"  [{label}] SIGTERM didn't reap process; " f"escalating to SIGKILL.",
                file=sys.stderr,
            )
        _signal_all(signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _kill_agent_tree(proc, *, provider: str) -> None:
    """Backward-compatible wrapper around `_kill_process_tree`.

    Kept as a separate entry point so existing call sites (and
    invariant tests) don't have to be touched. New code should
    prefer `_kill_process_tree(proc, label=...)` directly."""
    _kill_process_tree(proc, label=f"auto:{provider}")


def _agent_complexity_timeout(base_timeout_s: int, complexity_bonus: int) -> int:
    """Return the effective agent timeout for a component with the
    given complexity bonus (0..4 from `_component_complexity_bonus`).

    Bumps the base budget by +5 min per complexity unit, capped so a
    base of 900s (15 min) maxes out at 2100s (35 min, +20 min on a
    complexity=4 component). Components with complexity=0 are
    unaffected — the user's `--auto-agent-timeout` value is preserved
    as the floor."""
    if base_timeout_s <= 0 or complexity_bonus <= 0:
        return base_timeout_s
    bumped = base_timeout_s + 300 * int(complexity_bonus)
    return min(bumped, base_timeout_s + 1200)


def _parse_stream_json_event(line: str) -> Optional[Dict[str, object]]:
    """Parse one line of claude `--output-format stream-json` (NDJSON).

    Returns the decoded dict on success, None on any of: empty line,
    non-JSON line, malformed JSON, JSON that isn't an object. We are
    lenient here because the CLI mixes JSON events with the occasional
    plain-text diagnostic line on stderr (which gets redirected to the
    same log via `stderr=STDOUT`). Anything we can't parse is just
    counted as raw log growth elsewhere."""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        evt = json.loads(line)
    except Exception:
        return None
    if not isinstance(evt, dict):
        return None
    return evt


_EDIT_TOOL_NAMES: Tuple[str, ...] = (
    "Edit",
    "Write",
    "MultiEdit",
    "StrReplace",
    "NotebookEdit",
    "edit",
    "write",
    "str_replace",
    "create_file",
)
_READ_TOOL_NAMES: Tuple[str, ...] = (
    "Read",
    "Grep",
    "Glob",
    "LS",
    "Bash",
    "AwaitShell",
    "Shell",
    "WebSearch",
    "WebFetch",
    "read",
    "grep",
    "glob",
    "ls",
    "bash",
    "shell",
)


def _classify_tool_name(name: str) -> str:
    """Return ``"edit"`` for write/edit tools, ``"read"`` for read/
    search/exec tools, ``"other"`` otherwise. Used by the heartbeat
    to track Read-vs-Edit progress separately."""
    if not name:
        return "other"
    n = str(name)
    if n in _EDIT_TOOL_NAMES:
        return "edit"
    if n in _READ_TOOL_NAMES:
        return "read"

    if "edit" in n.lower() or "write" in n.lower() or "create" in n.lower():
        return "edit"
    return "other"


def _summarize_stream_json_event(evt: Dict[str, object], counts: Dict[str, int]) -> None:
    """Mutate `counts` in place to reflect a single stream-json event.

    Counts keys: `tool_use`, `assistant`, `result`, `error`, `other`,
    `edit_count`, `read_count`. The latter two are populated by
    classifying each `tool_use` block's tool name — used by
    `require_edit_progress` to early-kill agents that are stuck in
    a Read-only loop (audit 2026-05-24, P8/L3).

    The `assistant` event from claude stream-json contains a message
    with a `content` list of blocks; each `tool_use` block is counted
    separately so the heartbeat can show "the agent has invoked N
    tools" — that's the real progress signal users care about. The
    bare bytes-on-disk metric is too noisy because tool RESULTS
    (Read/Grep/Bash output that gets fed back to the model) also
    inflate the log size and don't represent forward progress on the
    task."""
    etype = str(evt.get("type") or "other")
    if etype == "assistant":
        counts["assistant"] = counts.get("assistant", 0) + 1
        msg = evt.get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content") or []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        counts["tool_use"] = counts.get("tool_use", 0) + 1
                        bucket = _classify_tool_name(str(block.get("name") or ""))
                        if bucket == "edit":
                            counts["edit_count"] = counts.get("edit_count", 0) + 1
                        elif bucket == "read":
                            counts["read_count"] = counts.get("read_count", 0) + 1
    elif etype == "result":
        counts["result"] = counts.get("result", 0) + 1
    elif etype in ("error", "system_error"):
        counts["error"] = counts.get("error", 0) + 1
    else:
        counts["other"] = counts.get("other", 0) + 1


def _read_proc_rchar(pid: int) -> Optional[int]:
    """Read /proc/<pid>/io's `rchar` (chars read by the process).

    Returns None if the file is unreadable (process already exited
    or kernel was built without CONFIG_TASK_IO_ACCOUNTING). Used as
    a coarse "is the process actually doing I/O" progress signal,
    which catches the case where claude's stream-json output is
    delayed but its `Read`/`Grep` tools are actively reading files."""
    try:
        with open(f"/proc/{pid}/io", "r") as f:
            txt = f.read()
        m = re.search(r"^rchar:\s*(\d+)", txt, flags=re.MULTILINE)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


_HEAVY_FAILURE_CLASSES = frozenset(
    {
        "HANG",
        "L1_OOM",
        "PARTIAL_CPU_FALLBACK",
        "DEVICE_NEEDS_RESET",
        "TT_FATAL_OPAQUE",
    }
)


from ._cli_helpers.agent import _resolve_tiered_model_aliases  # noqa: F401


from ._cli_helpers.agent import _pick_agent_model_for_iter  # noqa: F401


def _snapshot_deliverable_state(
    deliverable_dirs: List[Path],
    expected_files: Optional[List[Path]] = None,
) -> Dict[str, tuple]:
    """Snapshot (mtime, size) for deliverable files.

    Bug D + bug #5 fix (audit 2026-05-22 23:45): when `expected_files`
    is supplied we ONLY track those exact paths (snapshotting missing
    paths as a sentinel so their later creation is detected). This
    prevents the agent from satisfying the deliverable-deadline check
    by writing to a NON-target file (e.g. writing `decoder_head.py`
    while the target was `vision_config` — wrong-file = wasted
    iteration since `apply_all_responses` skips non-matching names).

    Falls back to "all *.py in each deliverable dir" when
    `expected_files` is None, preserving the original 2026-05-22 22:00
    behavior."""
    state: Dict[str, tuple] = {}
    if expected_files:
        for p in expected_files:
            try:
                if p.is_file():
                    st = p.stat()
                    state[str(p)] = (st.st_mtime, st.st_size)
                else:
                    state[str(p)] = (-1.0, -1)
            except Exception:
                state[str(p)] = (-1.0, -1)
        return state
    for d in deliverable_dirs:
        try:
            if not d.is_dir():
                continue
            for p in d.glob("*.py"):
                try:
                    st = p.stat()
                    state[str(p)] = (st.st_mtime, st.st_size)
                except Exception:
                    continue
        except Exception:
            continue
    return state


def _deliverable_changed(baseline: Dict[str, tuple], current: Dict[str, tuple]) -> bool:
    """Return True iff `current` has any path missing from `baseline`,
    OR has a (path, mtime/size) that differs from baseline.

    A path with baseline value `(-1.0, -1)` (sentinel for "didn't
    exist") that now has a real (mtime, size) counts as a change.
    Same path with the same metadata is not a change. New paths not
    in baseline are also a change (only happens in fallback mode
    where `expected_files=None`)."""
    for path, cur_meta in current.items():
        if path not in baseline:
            return True
        if cur_meta != baseline[path]:
            return True
    return False


from ._cli_helpers.agent import _invoke_agent  # noqa: F401


def _test_file_from_classname(classname: str) -> Optional[str]:
    if not classname:
        return None
    path = classname.replace(".", "/") + ".py"
    candidate = BRINGUP_ROOT() / path
    return path if candidate.is_file() else None


def _component_from_test_file(test_file: str) -> Optional[str]:
    stem = Path(test_file).stem
    if not stem.startswith("test_"):
        return None
    comp = stem[len("test_") :].strip()
    return comp or None


def _parse_pytest_report() -> Dict[str, object]:
    xml_path = REPO_ROOT / "generated" / "test_reports" / "most_recent_tests.xml"
    empty_report = {
        "ok": False,
        "summary": "(no test report on disk)",
        "details": "(no test report on disk)",
        "failed_tests": [],
        "failed_components": [],
        "skipped_tests": [],
        "skipped_components": [],
        "passed_tests": [],
        "passed_components": [],
        "saw_any": False,
        "all_passed": False,
    }
    if not xml_path.is_file():
        return empty_report
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(str(xml_path)).getroot()
    except Exception as exc:
        msg = f"(could not parse test report: {exc})"
        empty = dict(empty_report)
        empty["summary"] = msg
        empty["details"] = msg
        return empty

    summary_lines: List[str] = []
    detail_lines: List[str] = []
    failed_tests: List[str] = []
    failed_components: List[str] = []
    skipped_tests: List[str] = []
    skipped_components: List[str] = []
    passed_tests: List[str] = []
    passed_components: List[str] = []
    per_test: Dict[str, Dict[str, object]] = {}
    per_component: Dict[str, List[Dict[str, object]]] = {}
    per_skipped: Dict[str, Dict[str, object]] = {}
    saw_any = False
    has_failures = False

    for case in root.iter("testcase"):
        saw_any = True
        name = case.get("name") or "?"
        classname = case.get("classname") or ""
        test_id = f"{classname}::{name}"
        test_file = _test_file_from_classname(classname)
        component = _component_from_test_file(test_file) if test_file else None
        case_outcome = "passed"
        for child in case:
            tag = child.tag.lower()
            if tag in ("failure", "error"):
                case_outcome = "failed"
                has_failures = True
                full_message = (child.get("message") or "").strip()
                msg_lines = full_message.splitlines()
                head = msg_lines[0] if msg_lines else "(no message)"
                summary_lines.append(f"  - FAILED  {test_id}\n      {head}")
                body_text = (child.text or "").strip()
                body_lines = body_text.splitlines()
                body_head = "\n".join(body_lines[:60]) if body_lines else ""

                msg_body = "\n".join(msg_lines[1:60]).strip() if len(msg_lines) > 1 else ""
                exception_type = (child.get("type") or "").strip()
                pcc_value: Optional[float] = None
                for src in (full_message, body_text):
                    m = re.search(r"\bpcc[^0-9\-]*([-+]?\d*\.\d+|[-+]?\d+)", src, flags=re.IGNORECASE)
                    if m:
                        try:
                            pcc_value = float(m.group(1))
                            break
                        except Exception:
                            pass
                detail_chunks = [f"### {test_id}", head]
                if msg_body:
                    detail_chunks.append(msg_body)
                if body_head:
                    detail_chunks.append(body_head)
                detail_lines.append("\n".join(detail_chunks))
                if test_file and test_file not in failed_tests:
                    failed_tests.append(test_file)
                if component and component not in failed_components:
                    failed_components.append(component)
                entry: Dict[str, object] = {
                    "test_id": test_id,
                    "test_file": test_file or "",
                    "component": component or "",
                    "exception_type": exception_type,
                    "message": full_message,
                    "body": body_text,
                    "pcc_value": pcc_value,
                }
                per_test[test_id] = entry
                if component:
                    per_component.setdefault(component, []).append(entry)
            elif tag == "skipped":
                case_outcome = "skipped"
                full_message = (child.get("message") or "").strip()
                msg = full_message.splitlines()
                head = msg[0] if msg else "(skipped)"
                summary_lines.append(f"  - SKIPPED {test_id}\n      {head}")
                if test_file and test_file not in skipped_tests:
                    skipped_tests.append(test_file)
                if component and component not in skipped_components:
                    skipped_components.append(component)
                per_skipped[test_id] = {
                    "test_id": test_id,
                    "test_file": test_file or "",
                    "component": component or "",
                    "message": full_message,
                    "reason": head,
                }
        if case_outcome == "passed":
            if test_file and test_file not in passed_tests:
                passed_tests.append(test_file)
            if component and component not in passed_components:
                passed_components.append(component)

    return {
        "ok": True,
        "summary": "\n".join(summary_lines) if summary_lines else "(no failures or skips parsed)",
        "details": "\n\n".join(detail_lines) if detail_lines else "(no failure traceback parsed)",
        "failed_tests": failed_tests,
        "failed_components": failed_components,
        "skipped_tests": skipped_tests,
        "skipped_components": skipped_components,
        "per_skipped": per_skipped,
        "passed_tests": passed_tests,
        "passed_components": passed_components,
        "per_test": per_test,
        "per_component": per_component,
        "saw_any": saw_any,
        "all_passed": saw_any and not has_failures,
    }


def _extract_shape_probes(text: str) -> List[Dict[str, str]]:
    """Capability 3: scan a pytest stdout/stderr blob for SHAPE_PROBE
    lines emitted by an LLM-instrumented stub and return them as a
    structured list.

    Each line has the form
        [SHAPE_PROBE <tag>] <name>: shape=... dtype=... layout=... mem=...
    See `_strategy_directive_for_failure(..., 'TT_FATAL_OPAQUE')` for
    the producer template.

    Returns a list of dicts (one per probe line). Returns an empty
    list when no probes are found. The next-iter prompt formatter
    (`_format_shape_probe_block`) folds these back into the LLM input
    so it sees its own instrumentation output WITHOUT needing to scan
    the full pytest log itself.

    NOTE: prefer `_extract_shape_probes_from_report` over passing the
    pre-truncated `details` string to this helper. `details` is
    capped at 60 lines per test in `_parse_pytest_report` (so the
    JUnit body stays bounded), and probe prints can easily live past
    that cap on a stub with many probed call sites. The report-walking
    helper reads the FULL `body` field from `per_test` and feeds it
    here line-by-line — no truncation, no missed probes."""
    out: List[Dict[str, str]] = []
    if not text:
        return out
    for raw_line in text.splitlines():
        m = re.match(
            r"\[SHAPE_PROBE\s+([^\]]+)\]\s+([^:]+):\s+(.*)$",
            raw_line.strip(),
        )
        if not m:
            continue
        tag, name, payload = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        out.append({"tag": tag, "name": name, "payload": payload})
    return out


def _extract_shape_probes_from_report(
    report: Dict[str, object],
) -> List[Dict[str, str]]:
    """Capability 3 (robust harvest): walk a parsed pytest report and
    extract probe lines from the FULL body of every failed test.

    Why this and not just `_extract_shape_probes(details_str)`:
    `_parse_pytest_report` truncates each failure body to 60 lines
    when packing it into `details` (so the prompt stays bounded).
    Probe lines for a heavily-instrumented stub can easily land past
    line 60 — and would be silently dropped. This helper reads the
    pre-truncation `body` field stored in `per_test[*]`, plus the
    pre-truncation `message` field, so no probe is missed regardless
    of where in the failure body it appears."""
    out: List[Dict[str, str]] = []
    if not isinstance(report, dict):
        return out
    per_test = report.get("per_test") or {}
    if not isinstance(per_test, dict):
        return out
    seen_keys: set = set()
    for entry in per_test.values():
        if not isinstance(entry, dict):
            continue
        for field in ("message", "body"):
            payload_text = entry.get(field)
            if not isinstance(payload_text, str) or not payload_text:
                continue
            for probe in _extract_shape_probes(payload_text):
                key = (probe["tag"], probe["name"], probe["payload"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append(probe)
    return out


def _format_shape_probe_block(probes: List[Dict[str, str]]) -> str:
    """Render the SHAPE_PROBE OBSERVATIONS section for the LLM prompt.

    Returns an empty string when `probes` is empty so we don't pollute
    the prompt during normal iters. When non-empty, the block lists
    every probe line grouped by tag, since the LLM may have
    instrumented multiple call sites with distinct tags."""
    if not probes:
        return ""
    by_tag: Dict[str, List[Dict[str, str]]] = {}
    for p in probes:
        by_tag.setdefault(p["tag"], []).append(p)
    lines = [
        "SHAPE_PROBE OBSERVATIONS (from your instrumentation last iter):",
        "  The following SHAPE_PROBE prints were emitted by code you",
        "  added to the stub. Use them to deduce the predicate that",
        "  TT_FATAL was checking, fix the inputs, and REMOVE the",
        "  probe lines from the stub in this iteration.",
        "",
    ]
    for tag in sorted(by_tag):
        lines.append(f"  probe `{tag}`:")
        for p in by_tag[tag]:
            lines.append(f"    {p['name']}: {p['payload']}")
        lines.append("")
    return "\n".join(lines)


def _scope_report_to_demo(report: Dict[str, object], demo_dir: Path) -> Dict[str, object]:
    if not isinstance(report, dict):
        return report
    try:
        demo_rel = str(safe_relative_to_root(demo_dir))
    except Exception:
        demo_rel = str(demo_dir)
    demo_rel = demo_rel.replace("\\", "/").rstrip("/") + "/"

    def under_demo(test_file: object) -> bool:
        if not isinstance(test_file, str) or not test_file:
            return False
        return test_file.replace("\\", "/").startswith(demo_rel)

    scoped: Dict[str, object] = dict(report)
    failed_tests = list(report.get("failed_tests") or [])
    failed_components_in: List[str] = list(report.get("failed_components") or [])
    per_test_in = report.get("per_test") or {}
    per_component_in = report.get("per_component") or {}

    scoped_failed_tests: List[str] = [t for t in failed_tests if under_demo(t)]
    if isinstance(per_test_in, dict):
        scoped_per_test: Dict[str, Dict[str, object]] = {
            tid: entry
            for tid, entry in per_test_in.items()
            if isinstance(entry, dict) and under_demo(entry.get("test_file"))
        }
    else:
        scoped_per_test = {}
    kept_components: set = set()
    for entry in scoped_per_test.values():
        comp = entry.get("component")
        if isinstance(comp, str) and comp:
            kept_components.add(comp)
    if isinstance(per_component_in, dict):
        scoped_per_component: Dict[str, List[Dict[str, object]]] = {
            comp: [e for e in entries if isinstance(e, dict) and under_demo(e.get("test_file"))]
            for comp, entries in per_component_in.items()
            if isinstance(entries, list)
        }
        scoped_per_component = {c: es for c, es in scoped_per_component.items() if es}
    else:
        scoped_per_component = {}
    scoped_failed_components = [c for c in failed_components_in if c in kept_components or c in scoped_per_component]

    detail_chunks: List[str] = []
    summary_lines: List[str] = []
    for entry in scoped_per_test.values():
        test_id = str(entry.get("test_id") or "?")
        msg = str(entry.get("message") or "").strip()
        msg_lines = msg.splitlines() if msg else []
        head = msg_lines[0] if msg_lines else "(no message)"

        msg_body = "\n".join(msg_lines[1:60]).strip() if len(msg_lines) > 1 else ""
        body = str(entry.get("body") or "").strip()
        summary_lines.append(f"  - FAILED  {test_id}\n      {head}")
        chunks = [f"### {test_id}", head]
        if msg_body:
            chunks.append(msg_body)
        if body:
            chunks.append("\n".join(body.splitlines()[:60]))
        detail_chunks.append("\n".join(chunks))

    scoped["failed_tests"] = scoped_failed_tests
    scoped["failed_components"] = scoped_failed_components
    scoped["per_test"] = scoped_per_test
    scoped["per_component"] = scoped_per_component
    scoped["summary"] = "\n".join(summary_lines) if summary_lines else "(no failures or skips parsed for this demo)"
    scoped["details"] = "\n\n".join(detail_chunks) if detail_chunks else "(no failure traceback parsed for this demo)"
    scoped["all_passed"] = bool(report.get("saw_any")) and not scoped_failed_tests

    skipped_tests_in = list(report.get("skipped_tests") or [])
    skipped_components_in = list(report.get("skipped_components") or [])
    per_skipped_in = report.get("per_skipped") or {}
    passed_tests_in = list(report.get("passed_tests") or [])
    passed_components_in = list(report.get("passed_components") or [])

    scoped_skipped_tests = [t for t in skipped_tests_in if under_demo(t)]
    if isinstance(per_skipped_in, dict):
        scoped_per_skipped = {
            tid: entry
            for tid, entry in per_skipped_in.items()
            if isinstance(entry, dict) and under_demo(entry.get("test_file"))
        }
    else:
        scoped_per_skipped = {}
    scoped_skipped_components = [
        c
        for c in skipped_components_in
        if any(isinstance(e, dict) and e.get("component") == c for e in scoped_per_skipped.values())
    ]
    scoped_passed_tests = [t for t in passed_tests_in if under_demo(t)]
    scoped_passed_components: List[str] = []
    if scoped_passed_tests:
        for c in passed_components_in:
            for t in scoped_passed_tests:
                if _component_from_test_file(t) == c and c not in scoped_passed_components:
                    scoped_passed_components.append(c)
                    break

    scoped["skipped_tests"] = scoped_skipped_tests
    scoped["skipped_components"] = scoped_skipped_components
    scoped["per_skipped"] = scoped_per_skipped
    scoped["passed_tests"] = scoped_passed_tests
    scoped["passed_components"] = scoped_passed_components
    return scoped


def _collect_pytest_failure_summary(demo_dir: Path) -> str:
    _ = demo_dir
    report = _parse_pytest_report()
    return str(report.get("summary", "(no test report summary)"))


def _all_tests_passed(demo_dir: Path) -> bool:
    _ = demo_dir
    report = _parse_pytest_report()
    return bool(report.get("all_passed", False))


def _find_handoff_path(demo_dir: Path) -> Optional[Path]:
    handoff_dir = demo_dir / "_handoff"
    if not handoff_dir.is_dir():
        return None
    matches = sorted(handoff_dir.glob("*__handoff.md"))
    return matches[0] if matches else None


def _clear_responses_dir(demo_dir: Path) -> None:
    responses_dir = demo_dir / "_synth_responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    for fp in responses_dir.iterdir():
        if fp.is_file():
            fp.unlink()


_LAST_PYTEST_STAGES: Dict[str, str] = {}
_LAST_PYTEST_PCC: Dict[str, float] = {}


def _run_focused_pytest(
    *,
    model_id: str,
    test_files: List[str],
    timeout_s: Optional[int] = None,
    allow_kill_stale: bool = True,
    allow_device_reset: bool = True,
    _reset_already_attempted: bool = False,
) -> int:
    if not test_files:
        return 0
    import subprocess
    import signal
    import threading
    import time as _time

    _preflight_cleanup_stale_pytest(
        REPO_ROOT, allow_kill=allow_kill_stale, allow_device_reset=allow_device_reset, context="iter:pre-pytest"
    )

    if not _reset_already_attempted:
        _truncate_runtime_fallback_log(model_id)

    if timeout_s is None:
        try:
            timeout_s = int(os.environ.get("TT_PLANNER_PYTEST_TIMEOUT_S", "600"))
        except ValueError:
            timeout_s = 600

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *test_files,
        "-svv",
        "-k",
        "not test_perf_device and not test_e2e_performant and not benchmark and not test_perf and not stress",
    ]
    env = dict(os.environ)
    env["HF_MODEL"] = model_id
    env["PLANNER_TARGET_HF_MODEL"] = model_id
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(BRINGUP_ROOT()),
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )

    _LAST_PYTEST_STAGES.clear()
    _LAST_PYTEST_PCC.clear()
    current_test: Optional[str] = None
    lock_wait_state: Dict[str, object] = {"since": None, "blocker_pid": None, "abort": False}
    captured_tail: collections.deque = collections.deque(maxlen=4000)

    _verbose = os.environ.get("TT_HW_PLANNER_VERBOSE", "") not in ("", "0", "false", "False")

    def _pump():
        nonlocal current_test
        assert proc.stdout is not None
        for raw in proc.stdout:
            captured_tail.append(raw)
            line = raw.rstrip("\n")
            m_test = re.match(r"(models/[^\s:]+::[^\s\[]+)", line)
            if m_test:
                current_test = m_test.group(1)
            if "[bringup] stage=" in line:
                key = current_test or "__global__"
                stage = line.split("[bringup] stage=", 1)[1].strip()
                _LAST_PYTEST_STAGES[key] = stage
                lock_wait_state["since"] = None
                lock_wait_state["blocker_pid"] = None
            if line.lstrip().startswith("[bringup] achieved PCC="):
                m_pcc = re.search(r"achieved\s+PCC=(-?\d+\.\d+).*?component=(\S+)", line)
                if m_pcc:
                    try:
                        _LAST_PYTEST_PCC[m_pcc.group(2)] = float(m_pcc.group(1))
                    except ValueError:
                        pass
            if "Waiting for lock 'CHIP_IN_USE" in line:
                if lock_wait_state["since"] is None:
                    lock_wait_state["since"] = _time.monotonic()
                m_pid = re.search(r"PID:\s*(\d+)", line)
                if m_pid:
                    lock_wait_state["blocker_pid"] = int(m_pid.group(1))
            if _verbose or _pytest_line_interesting(line):
                sys.stdout.write(raw)
                sys.stdout.flush()

    pump_thread = threading.Thread(target=_pump, daemon=True)
    pump_thread.start()

    def _watch_for_lock_stuck():
        threshold_s = 30.0
        while True:
            if proc.poll() is not None:
                return
            since = lock_wait_state.get("since")
            if since is not None and (_time.monotonic() - float(since)) > threshold_s:
                blocker = lock_wait_state.get("blocker_pid")
                print(
                    f"\n  TT device lock 'CHIP_IN_USE_0_PCIe' has been held for "
                    f">{int(threshold_s)}s by PID {blocker}; trying to free it.",
                    file=sys.stderr,
                )
                if allow_kill_stale and isinstance(blocker, int):
                    try:
                        pgid = os.getpgid(blocker)
                        os.killpg(pgid, signal.SIGTERM)
                        print(
                            f"  SIGTERM sent to PGID {pgid} (blocker PID {blocker}); "
                            f"pytest should acquire the device shortly.",
                            file=sys.stderr,
                        )
                        lock_wait_state["since"] = None
                    except (ProcessLookupError, PermissionError) as exc:
                        print(f"  could not signal blocker: {exc}", file=sys.stderr)
                else:
                    print(
                        f"  --no-kill-stale is set OR blocker PID is unknown. "
                        f"Manual recovery:\n"
                        f"    kill -TERM -- -$(ps -o pgid= -p {blocker} | tr -d ' ')",
                        file=sys.stderr,
                    )
            _time.sleep(2.0)

    watcher_thread = threading.Thread(target=_watch_for_lock_stuck, daemon=True)
    watcher_thread.start()

    def _maybe_retry_after_device_reset(rc: int) -> Optional[int]:
        if rc == 0:
            return None
        if _reset_already_attempted or not allow_device_reset:
            return None
        tail_text = "".join(captured_tail)
        if not _output_indicates_device_reset_needed(tail_text):
            return None
        print()
        print("=" * 78)
        print(
            "  Detected stale device state in pytest output "
            "(`Proceeding could lead to undefined behavior` / "
            "`pin_or_map_sysmem_to_device`)."
        )
        print("  This usually means a previously killed orphan left stale IOMMU/sysmem")
        print("  mappings. Running `tt-smi -r` and retrying the same pytest ONCE.")
        print("=" * 78)
        if not _run_tt_smi_reset(context="pytest:post-fail-recovery"):
            print("  device reset did not succeed; leaving rc as is", file=sys.stderr)
            return None
        return _run_focused_pytest(
            model_id=model_id,
            test_files=test_files,
            timeout_s=timeout_s,
            allow_kill_stale=allow_kill_stale,
            allow_device_reset=False,
            _reset_already_attempted=True,
        )

    def _drain_now() -> None:
        try:
            tail_text = "".join(captured_tail)
            drained = _drain_runtime_fallback_log(model_id, stdout_text=tail_text)

            tested: List[str] = []
            for _tf in test_files or []:
                _stem = Path(_tf).stem
                if _stem.startswith("test_"):
                    tested.append(_stem[len("test_") :])

            _persist_runtime_fallbacks(model_id, drained, tested_components=tested or None)
            if drained:
                total_helpers = sum(len(info.get("helpers", [])) for info in drained.values())
                if total_helpers > 0:
                    print(
                        f"  [runtime-fallback] {total_helpers} `_apply_*` helper "
                        f"call(s) ran on CPU at runtime across "
                        f"{len(drained)} component(s); compute split will reflect this."
                    )
        except Exception as _drain_exc:
            print(
                f"  [runtime-fallback] drain failed: {_drain_exc}",
                file=sys.stderr,
            )

    try:
        proc.wait(timeout=timeout_s if timeout_s and timeout_s > 0 else None)
        pump_thread.join(timeout=5)
        rc = proc.returncode
        retried = _maybe_retry_after_device_reset(rc)
        if retried is not None:
            return retried
        _drain_now()
        return rc
    except subprocess.TimeoutExpired:
        print(
            f"  focused pytest WALL-CLOCK BUDGET EXHAUSTED at {timeout_s}s "
            f"— killing process group (likely a hang/deadlock inside the new stub).",
            file=sys.stderr,
        )
        if _LAST_PYTEST_STAGES:
            print("  Last reported stage(s) before hang:", file=sys.stderr)
            for tname, stg in _LAST_PYTEST_STAGES.items():
                print(f"    {tname}: stage={stg}", file=sys.stderr)

        _kill_process_tree(proc, label="pytest")
        pump_thread.join(timeout=5)
        _drain_now()
        return 124


def _safe_component_name(component: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", component).strip("_").lower()
    return safe or "component"


def _component_class_name(component: str) -> str:
    parts = [p for p in _safe_component_name(component).split("_") if p]
    return "".join(p.capitalize() for p in parts) or "Component"


def _component_stub_path(demo_dir: Path, component: str) -> Path:
    return demo_dir / "_stubs" / f"{_safe_component_name(component)}.py"


def _build_stabilized_fallback_stub(component: str) -> str:
    safe = _safe_component_name(component)
    cls = _component_class_name(component)
    from .bringup_loop import _FALLBACK_COERCE_TO_TORCH

    return (
        "import torch\n"
        "import ttnn\n"
        "\n"
        "\n" + _FALLBACK_COERCE_TO_TORCH + "\n\n"
        f"class {cls}:\n"
        "    def __init__(self, device, torch_module):\n"
        "        self.device = device\n"
        "        self.torch_module = torch_module.eval()\n"
        "\n"
        "    def _pick_tensor(self, value):\n"
        "        if torch.is_tensor(value):\n"
        "            return value\n"
        "        if hasattr(value, 'last_hidden_state') and torch.is_tensor(value.last_hidden_state):\n"
        "            return value.last_hidden_state\n"
        "        if isinstance(value, dict):\n"
        "            for v in value.values():\n"
        "                t = self._pick_tensor(v)\n"
        "                if t is not None:\n"
        "                    return t\n"
        "            return None\n"
        "        if isinstance(value, (list, tuple)):\n"
        "            for v in value:\n"
        "                t = self._pick_tensor(v)\n"
        "                if t is not None:\n"
        "                    return t\n"
        "            return None\n"
        "        return None\n"
        "\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        t_args = tuple(_coerce_to_torch(a) for a in args)\n"
        "        t_kwargs = {k: _coerce_to_torch(v) for k, v in kwargs.items()}\n"
        "        with ttnn.manage_config('throw_exception_on_fallback', False):\n"
        "            with ttnn.manage_config('enable_fast_runtime_mode', True):\n"
        "                out = self.torch_module(*t_args, **t_kwargs)\n"
        "        out_t = self._pick_tensor(out)\n"
        "        if out_t is None:\n"
        "            raise RuntimeError('torch fallback produced no tensor output')\n"
        "        return ttnn.from_torch(\n"
        "            out_t.to(torch.bfloat16),\n"
        "            dtype=ttnn.bfloat16,\n"
        "            layout=ttnn.TILE_LAYOUT,\n"
        "            device=self.device,\n"
        "        )\n"
        "\n"
        "\n"
        "def build(device, torch_module):\n"
        f"    return {cls}(device, torch_module)\n"
        "\n"
        "\n"
        "_instance = None\n"
        "\n"
        "\n"
        f"def {safe}(*args, **kwargs):\n"
        "    global _instance\n"
        "    if _instance is None:\n"
        "        raise RuntimeError(\n"
        "            'Synthesized TTNN module requires `build(device, torch_module)`. '\n"
        "            'Call it from the PCC test\\'s `_build_ttnn_port`.'\n"
        "        )\n"
        "    return _instance(*args, **kwargs)\n"
    )


def _rewrite_components_to_stable_fallback(demo_dir: Path, components: List[str]) -> List[str]:
    rewritten: List[str] = []
    for comp in components:
        stub_path = _component_stub_path(demo_dir, comp)
        if not stub_path.is_file():
            continue
        snapshot = Path(str(stub_path) + ".last_good_native")
        if snapshot.is_file():
            stub_path.write_text(snapshot.read_text(encoding="utf-8"), encoding="utf-8")
            continue
        backup = stub_path.with_suffix(stub_path.suffix + ".auto_stabilize.bak")
        if not backup.exists():
            backup.write_text(stub_path.read_text(encoding="utf-8"), encoding="utf-8")
        stub_path.write_text(_build_stabilized_fallback_stub(comp), encoding="utf-8")
        rewritten.append(comp)
    return rewritten


def _only_pcc_threshold_failures(summary: str) -> bool:
    if "FAILED" not in summary:
        return False
    if "AssertionError: PCC" not in summary:
        return False
    non_pcc_markers = (
        "RuntimeError:",
        "TypeError:",
        "ValueError:",
        "AttributeError:",
        "NotImplementedError:",
    )
    return not any(marker in summary for marker in non_pcc_markers)


_DEVICE_RESET_SIGNATURES: Tuple[str, ...] = (
    "Proceeding could lead to undefined behavior",
    "silicon_sysmem_manager.cpp",
    "pin_or_map_sysmem_to_device",
    "Fabric Router Sync: Timeout",
    "fabric_firmware_initializer.cpp",
    "fabric_unavailable",
)


def _output_indicates_device_reset_needed(text: str) -> bool:
    """Return True if `text` (typically pytest stdout/stderr) contains the
    UMD signature for a stale IOMMU/sysmem mapping. This happens when a
    previous TT-device process was killed mid-flight (e.g. an orphan reaped
    by our preflight sweep) and left kernel-side mappings the next process
    sees as inconsistent."""
    if not text:
        return False
    return any(sig in text for sig in _DEVICE_RESET_SIGNATURES)


_CLASS_SEVERITY: Dict[str, int] = {
    "PARTIAL_CPU_FALLBACK": 2,
    "PCC_ONLY": 3,
    "WRAPPER": 5,
    "NO_OP": 5,
    "EMPTY_AGENT": 5,
    "NotImplementedError": 6,
    "NOT_IMPLEMENTED": 6,
    "SHAPE": 7,
    "DTYPE": 7,
    "DTYPE_MISMATCH": 7,
    "RANK_MISMATCH": 7,
    "TT_FATAL_OPAQUE": 9,
    "L1_OOM": 10,
    "L1_SMALL_ZERO": 10,
    "CRASH": 10,
    "HANG": 10,
}


def _failure_class_severity(failure_class: str) -> int:
    """Map a failure-class string to a severity rank. See
    `_CLASS_SEVERITY` for ordering rationale.

    Module-level so the invariant test suite can verify the ordering
    without exercising the full `_run_auto_iterate_loop` closure."""
    if not failure_class:
        return 100
    return _CLASS_SEVERITY.get(failure_class, 100)


def _classify_failure(summary: str, details: str) -> str:
    text = f"{summary}\n{details}"

    if (
        "No chips detected in the cluster" in text
        or "No chips detected" in text
        or re.search(r"num_chips\s*>\s*0", text)
    ):
        return "NO_HARDWARE"
    if _output_indicates_device_reset_needed(text):
        return "DEVICE_NEEDS_RESET"
    if "WALL-CLOCK BUDGET EXHAUSTED" in text or "BUDGET EXHAUSTED" in text or "process group (likely a hang" in text:
        return "HANG"

    if ("L1_SMALL" in text or "L1 small" in text) and "bank size is 0" in text:
        return "L1_SMALL_ZERO"

    if "Input must be UINT32 or BFLOAT16" in text:
        return "EMBEDDING_DTYPE"

    if ("ttnn.concat" in text or "concat(" in text) and "incompatible function arguments" in text:
        return "CONCAT_INCOMPATIBLE"

    # CONFIG_PARAM: NotImplementedError raised from inside canonical
    # tt_transformers code (e.g. rope.py raises NotImplementedError on a
    # `use_qk_fused` config flag that doesn't match the model's head_dim).
    # The fix lives in `models/tt_transformers/tt/model_config.py` or the
    # raising canonical file — NOT in the per-component stub. Detected by
    # the combination of NotImplementedError + tt_transformers in the
    # traceback. The escalation table maps this class to additional
    # editable paths in the canonical tree.
    if "NotImplementedError" in text and ("models/tt_transformers/tt/" in text or "models/common/" in text):
        return "CONFIG_PARAM"

    oom_explicit = ("Out of Memory" in text) or ("Not enough space to allocate" in text)
    oom_via_allocator = ("TT_FATAL" in text) and (
        "bank_manager.cpp" in text
        or "/allocator/" in text
        or "tt_metal/impl/allocator" in text
        or "DRAM buffer" in text
        or "L1 buffer" in text
    )
    if oom_explicit or oom_via_allocator:
        return "L1_OOM"
    if "expected scalar type" in text and (
        "BFloat16" in text or "bfloat16" in text or "Float" in text or "Half" in text
    ):
        return "DTYPE_MISMATCH"
    from ._cli_helpers.error_patterns import (
        matches_state_dict_key_error as _matches_state_dict_key,
        extract_unexpected_kwarg as _extract_unexpected_kwarg,
        extract_missing_args_description as _extract_missing_args,
        matches_tt_fatal_with_predicate as _matches_tt_fatal_pred,
    )

    if _matches_state_dict_key(text):
        return "STATE_DICT_KEY"
    if _extract_unexpected_kwarg(text):
        return "UNEXPECTED_KWARG"
    if _extract_missing_args(text):
        return "MISSING_KWARG"
    if "incompatible function arguments" in text or "TypeError:" in text:
        return "API_SIGNATURE"
    if "AssertionError: PCC" in text:
        return "PCC_ONLY"
    # TT_FATAL with parseable predicate must precede the generic SHAPE
    # bucket so layernorm/matmul/etc. shape-assertion failures get
    # the structured diagnosis (op + predicate) instead of the generic
    # "check reshape/transpose" hint. Caught the 2026-06-01 Qwen2.5-14B
    # decoder_layer rabbit-hole where LLM iter 1→2 saw the same
    # TT_FATAL layernorm shape mismatch and made no targeted fix.
    if _matches_tt_fatal_pred(text):
        return "TT_FATAL_SHAPE"
    if "shape" in text or "dimension" in text:
        return "SHAPE"

    if "TT_FATAL" in text:
        return "TT_FATAL_OPAQUE"
    return "OTHER"


def _detect_no_hardware_failure(report: Dict[str, object]) -> Optional[str]:
    """Scan a parsed pytest report for the 'no Tenstorrent chips visible'
    failure signature and return a short human-readable description if
    found, or None otherwise.

    The pre-flight pytest's per-component error messages all carry the
    same TT_FATAL string when the tt_metal cluster init fails — typically
    because the `tenstorrent` kernel driver isn't loaded or
    `/dev/tenstorrent/*` device nodes don't exist. The LLM cannot fix
    this; it's a host environment issue. The auto-iterate loop must
    bail early instead of spending its full budget invoking Claude on
    what looks like a per-component PCC failure.

    Returns the first matched diagnostic string ("No chips detected in
    the cluster", "num_chips > 0", etc.) so the caller can surface it
    in the bail banner. The full match list is intentionally checked
    in priority order — the most user-readable string wins."""
    if not isinstance(report, dict):
        return None
    text_blobs: List[str] = []
    for key in ("summary", "details"):
        v = report.get(key)
        if isinstance(v, str) and v:
            text_blobs.append(v)
    per_test = report.get("per_test")
    if isinstance(per_test, dict):
        for entry in per_test.values():
            if isinstance(entry, dict):
                for fkey in ("message", "body"):
                    fv = entry.get(fkey)
                    if isinstance(fv, str) and fv:
                        text_blobs.append(fv)
    text = "\n".join(text_blobs)
    if not text:
        return None
    if "No chips detected in the cluster" in text:
        return "No chips detected in the cluster"
    if "No chips detected" in text:
        return "No chips detected"
    if re.search(r"num_chips\s*>\s*0", text):
        return "tt_cluster init failed (num_chips > 0)"
    return None


def _format_no_hardware_diagnostic_banner(error_msg: str) -> List[str]:
    """Produce the multi-line banner printed when the pre-flight detects
    that no Tenstorrent chips are visible. The lines describe what to
    check on the host (kernel driver loaded? device nodes present? user
    in the right group?) so the user can act WITHOUT digging into
    tt_metal internals.

    Returned as a list of lines so callers can print/log it
    consistently (no embedded newlines in any single element)."""
    bar = "=" * 78
    return [
        "",
        bar,
        "  PRE-FLIGHT ABORTED: Tenstorrent hardware not visible to ttnn",
        bar,
        f"  Error: {error_msg}",
        "",
        "  This is a HOST ENVIRONMENT problem, NOT a code problem.",
        "  The LLM cannot fix it; invoking Claude here would burn the entire",
        "  agent budget on every component for no benefit. Skipping the",
        "  auto-iterate loop and exiting with code 2.",
        "",
        "  Diagnose with:",
        "    1. Is the kernel driver loaded?",
        "         lsmod | grep tenstorrent",
        "       (expect: a line starting with `tenstorrent`)",
        "    2. Do the device nodes exist?",
        "         ls /dev/tenstorrent*",
        "       (expect: one node per chip, e.g. /dev/tenstorrent/0)",
        "    3. Are the chips on the PCIe bus?",
        "         lspci | grep -i tenstorrent",
        "       (expect: one line per physical chip)",
        "",
        "  Common fixes:",
        "    - Driver not loaded:   sudo modprobe tenstorrent",
        "    - Driver not installed: re-run tt-installer",
        "    - Permission denied:    add your user to the `tenstorrent` group,",
        "                            then log out and back in",
        "    - Chip not enumerated:  reboot (PCIe re-enumeration)",
        "",
        "  After fixing, re-run the same `tt_hw_planner up ...` command.",
        bar,
        "",
    ]


def _extract_pcc_from_failure(summary: str, details: str) -> Optional[float]:
    """Extract the numeric PCC value from a pytest PCC failure line, if any.

    Used by the auto-iterate loop's progress detector: if the failure class
    is PCC_ONLY and the value strictly improves between attempts, the
    iteration counts as progress and does NOT increment the
    consecutive-same-class counter against the attempt cap. Returns None if
    no PCC value can be parsed.

    Recognised forms:
      "AssertionError: PCC 0.9877806828998408 below target 0.99"
      "PCC -0.0006025997490610422 below target 0.99"
      "pcc achieved      : 0.8845  (target >= 0.99)"
      "[bringup] achieved PCC=0.9991 target=0.99 component=g_l_u"
    """
    text = f"{summary}\n{details}"
    patterns = [
        r"\[bringup\]\s+achieved\s+PCC=(-?\d+\.\d+)",
        r"PCC\s+(-?\d+\.\d+)\s+below\s+target",
        r"pcc\s+achieved\s*:\s*(-?\d+\.\d+)",
        r"comp_pcc.*?\s(-?\d+\.\d+)\s*<\s*0\.99",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _failure_signature(summary: str, details: str) -> str:
    text = f"{summary}\n{details}"

    def _root_cause(s: str) -> str:
        if s.startswith("FAILED ") and " - " in s:
            return s.split(" - ", 1)[1].strip()
        return s

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if (
            "TT_FATAL" in s
            or "Out of Memory" in s
            or "incompatible function arguments" in s
            or "AssertionError: PCC" in s
        ):
            return _root_cause(s)[:220]
    return _root_cause((summary or details or "(unknown)").strip())[:220]


_EDIT_SCOPE_FOR_FAILURE_CLASS: Dict[str, List[str]] = {
    "L1_SMALL_ZERO": ["conftest.py", "tests/pcc/conftest.py"],
    "EMBEDDING_DTYPE": ["tests/pcc/conftest.py", "_captured/**/*.py"],
    "HANG": ["tests/pcc/conftest.py"],
}


# Repo-relative escalation paths (resolved against BRINGUP_ROOT, not demo_dir).
# Used when the failure originates outside the demo dir — e.g. a
# NotImplementedError from `models/tt_transformers/tt/rope.py` whose real
# fix lives in `models/tt_transformers/tt/model_config.py`. Without this,
# the LLM is given no path to fix bugs in canonical code that it didn't
# author. The Phi-3.5 attention failure (use_qk_fused for head_dim=96)
# is the motivating case.
_REPO_RELATIVE_EDIT_SCOPE_FOR_FAILURE_CLASS: Dict[str, List[str]] = {
    "CONFIG_PARAM": [
        "models/tt_transformers/tt/model_config.py",
        "models/tt_transformers/tt/rope.py",
        "models/common/rmsnorm.py",
    ],
}


def _resolve_extra_edit_paths(
    demo_dir: Path,
    failure_class: str,
) -> List[Path]:
    """Expand the failure-class-keyed permission table into concrete
    file paths. Returns an empty list when the class is unknown or has
    no escalation.

    Used by the prompt-assembly path to emit an ESCALATED EDIT SCOPE
    section instructing the LLM that, for this iteration only, it may
    additionally edit the listed files. The default `_stubs/<comp>.py`
    write-scope is always implicitly allowed.

    Two tables are consulted:
      * ``_EDIT_SCOPE_FOR_FAILURE_CLASS`` — paths globbed under demo_dir
        (e.g. conftest.py inside the per-model test tree).
      * ``_REPO_RELATIVE_EDIT_SCOPE_FOR_FAILURE_CLASS`` — paths resolved
        against BRINGUP_ROOT (e.g. canonical tt_transformers source).
        Used when the bug lives in shared code, not the demo-local tree.
    """
    out: List[Path] = []

    # demo_dir-relative patterns (existing behavior).
    patterns = _EDIT_SCOPE_FOR_FAILURE_CLASS.get(failure_class, [])
    for pattern in patterns:
        try:
            for p in demo_dir.glob(pattern):
                if p.is_file():
                    out.append(p)
        except Exception:
            continue

    # repo-relative patterns (canonical source fixes).
    repo_patterns = _REPO_RELATIVE_EDIT_SCOPE_FOR_FAILURE_CLASS.get(failure_class, [])
    if repo_patterns:
        try:
            from .discovery import BRINGUP_ROOT as _BR

            repo_root = _BR()
        except Exception:
            repo_root = None
        if repo_root is not None:
            for rel in repo_patterns:
                p = repo_root / rel
                if p.is_file():
                    out.append(p)

    return sorted(set(out))


def _format_escalated_edit_scope_block(
    demo_dir: Path,
    failure_class: str,
) -> str:
    """Render the ESCALATED EDIT SCOPE section for the LLM prompt.

    Returns an empty string when the failure class does not unlock
    any extra files — keeps the prompt small and avoids tempting the
    LLM to touch files it has no business in."""
    extra = _resolve_extra_edit_paths(demo_dir, failure_class)
    if not extra:
        return ""
    lines = [
        "ESCALATED EDIT SCOPE (failure-class-specific):",
        f"  Failure class `{failure_class}` cannot be fixed inside the per-",
        "  component stub alone. For THIS iteration ONLY, you are also",
        "  permitted to edit the following file(s):",
    ]
    for p in extra:
        try:
            rel = safe_relative_to_root(p)
        except Exception:
            rel = p
        lines.append(f"    - {rel}")
    if failure_class == "L1_SMALL_ZERO":
        lines.extend(
            [
                "",
                "  For L1_SMALL_ZERO specifically: edit the device fixture in",
                "  the listed conftest.py to pass `l1_small_size=16384` (or",
                "  a larger power-of-two if the diagnostic reports a bigger",
                "  buffer requirement) to whatever ttnn device-creation helper",
                "  the fixture uses. Look for `ttnn.open_device(...)`,",
                "  `ttnn.open_mesh_device(...)`, or",
                "  `MeshDevice.create(...)` invocations and add the kwarg.",
                "  This single change unblocks every conv2d/halo/sliding-window",
                "  op the model uses; no stub rewrite is needed once it lands.",
            ]
        )
    elif failure_class == "EMBEDDING_DTYPE":
        lines.extend(
            [
                "",
                "  For EMBEDDING_DTYPE: the test harness may be passing index",
                "  tensors as float32/bfloat16 when ttnn.embedding requires",
                "  uint32. Cast indices to uint32 either inside `__call__`",
                "  (preferred — robust to harness changes) or in the conftest",
                "  fixture's input prep step.",
            ]
        )
    elif failure_class == "HANG":
        lines.extend(
            [
                "",
                "  For HANG: if multiple components hang at the SAME stage",
                "  (e.g. `ttnn_to_torch`), this is likely a harness bug",
                "  rather than a stub bug. Inspect `_ttnn_to_torch_mesh_safe`",
                "  in the test conftest and consider replacing it with a",
                "  per-output type-dispatch (handle Tensor, tuple, list, dict)",
                "  that calls `ttnn.synchronize_device(device)` before drain.",
            ]
        )
    elif failure_class == "CONFIG_PARAM":
        lines.extend(
            [
                "",
                "  For CONFIG_PARAM: a NotImplementedError was raised from",
                "  canonical tt_transformers code because a model_config flag",
                "  (e.g. `use_qk_fused`, `use_hf_rope`) is set inconsistently",
                "  with the model's architecture (e.g. head_dim not divisible",
                "  by 64). The fix is usually a one-line addition to the",
                "  derivation in `model_config.py` to also gate on the",
                "  relevant architecture constraint. Trace the exception back",
                "  from the `raise NotImplementedError` site to wherever the",
                "  offending flag is set; add the missing condition there.",
                "  DO NOT just override the flag in the wrapper — that won't",
                "  stick if downstream code re-reads from configuration.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


_AGENTIC_INVESTIGATION_CLASSES = frozenset(
    {
        "TT_FATAL_OPAQUE",
        "HANG",
        "PCC_ONLY",
        "L1_OOM",
        "DEVICE_NEEDS_RESET",
    }
)


def _format_agentic_affordances_block(
    failure_class: str,
    *,
    consec_count: int = 0,
    has_systemic_pattern: bool = False,
) -> str:
    """Render the AGENTIC AFFORDANCES section for the LLM prompt.

    Returns an empty string when the failure class is in the easy
    bucket AND no systemic pattern has been detected AND the
    consecutive-same-class counter hasn't shown the LLM is stuck.

    Why these triggers:
      - `_AGENTIC_INVESTIGATION_CLASSES`: failure classes where blind
        regen has empirically NOT converged in a single attempt
        without investigation (e.g. PCC_ONLY past iter 1 needs the
        LLM to compare its forward against the HF source).
      - `consec_count >= 2`: the LLM has already failed the same way
        twice; whatever it tried blind didn't work, so it's time to
        switch to read-the-code mode regardless of class.
      - `has_systemic_pattern`: per Capability 4, the bug is in a
        SHARED path the LLM must discover via Grep.
    """
    needs_agentic = failure_class in _AGENTIC_INVESTIGATION_CLASSES or consec_count >= 2 or has_systemic_pattern
    if not needs_agentic:
        return ""
    lines = [
        "AGENTIC AFFORDANCES (use BEFORE writing the response file):",
        "  You are NOT a one-shot code generator. You have full",
        "  read+search+shell access to this repository, and for this",
        "  failure class the OUTER tool EXPECTS you to use it:",
        "",
        "    Read    — open ANY file (the per-component stub, the",
        "              torch reference module, the test harness, any",
        "              exemplar). Use this BEFORE generating code.",
        "    Grep    — search the repo for patterns. Useful for:",
        "                * `class <RefModuleName>` to find the HF",
        "                  reference forward source in a 3rd-party",
        "                  package not inlined in the prompt;",
        "                * `def _apply_<op>` to find sharding /",
        "                  memory_config conventions used elsewhere",
        "                  in this model;",
        "                * `l1_small_size` / `ttnn.open_device` to",
        "                  find the device fixture for L1_SMALL_ZERO;",
        "                * `_ttnn_to_torch_mesh_safe` and similar",
        "                  drain helpers when investigating HANGs.",
        "    Edit    — surgically patch a small region of a file",
        "              (preferred over Write for shared files like",
        "              conftest.py — leaves the rest untouched).",
        "    Write   — for the per-component response file under",
        "              `_synth_responses/<safe>.py`, use Write with",
        "              the COMPLETE file contents (the outer tool",
        "              copies that file over the stub verbatim).",
        "    Bash    — run shell commands. Use SPARINGLY: read-only",
        '              probes (`ls`, `cat`, `find`, `python -c "import',
        '              ttnn; help(ttnn.softmax)"`) are fine; DO NOT',
        "              run pytest — the outer tool runs it after you",
        "              exit and the next iter sees the result.",
        "",
        "  Suggested workflow for a hard failure:",
        "    1. Read the failing component's stub.",
        "    2. Read the FAILURE CONTEXT / SHAPE_PROBE / HF reference",
        "       sections in this prompt.",
        "    3. Grep for any helper / exemplar that solves a similar",
        "       sub-problem. Read those files. They are your model.",
        "    4. Edit/Write the response file once, with confidence.",
        "    5. Exit. Do NOT loop — the outer tool will iterate.",
        "",
        "  The OUTPUT CONTRACT remains: one response file per",
        "  target component at `_synth_responses/<safe>.py`. Edits",
        "  to other files (e.g. `tests/pcc/conftest.py` under an",
        "  ESCALATED EDIT SCOPE) are committed in place.",
        "",
    ]
    return "\n".join(lines)


def _strategy_directive_for_failure(failure_class: str, *, strict_native: bool = False) -> str:
    if failure_class == "PARTIAL_CPU_FALLBACK":
        return (
            "Failure class is PARTIAL_CPU_FALLBACK. This is a "
            "SPECIAL CASE: the PCC test currently PASSES (>= 0.99) — "
            "the stub is numerically CORRECT — but the runtime "
            "instrumentation observed at least one `_apply_*` "
            "helper inside `__call__` falling back to a PyTorch "
            "CPU implementation. The list of offending helpers and "
            "the op kind each one wraps is below in 'PARTIAL-CPU "
            "FALLBACK DETAILS'.\n\n"
            "GROUND RULES — read carefully:\n"
            "  1. **Do NOT** rewrite the entire `__call__`. Only "
            "touch the specific `_apply_*` helper(s) named below. "
            "Everything else in this stub already works.\n"
            "  2. **Do NOT** change the helper's call signature "
            "(its name, arguments, return type/shape/dtype). The "
            "rest of `__call__` calls it; breaking the signature "
            "breaks the working path.\n"
            "  3. The helper currently uses `ttnn.to_torch(...)` + a "
            "torch op + `ttnn.from_torch(...)` to bridge. Replace "
            "that bridge with a pure ttnn op of the same kind "
            "(e.g. `ttnn.conv2d`, `ttnn.matmul`, `ttnn.permute`, "
            "etc.). Match the input/output shapes the existing "
            "torch path produces.\n"
            "  4. If the op is `conv2d`, you almost certainly need "
            "the same `conv_config` / `compute_kernel_config` / "
            "weight + bias pre-bound on `self` that the helper was "
            "given. Use `ttnn.conv2d` with those configs. Set "
            "`return_output_dim=False, return_weights_and_bias="
            "False` so the call returns a single tensor.\n"
            "  5. If a pure ttnn implementation genuinely doesn't "
            "exist for this op kind on this shape (rare — usually "
            "an indexing / advanced gather / dynamic shape thing), "
            "leave the helper alone. The cap-out logic will preserve "
            "the working version. Do NOT replace it with a broken "
            "ttnn call that regresses PCC.\n"
            "  6. After your rewrite, the test must STILL pass PCC "
            "(>= 0.99) AND the helper must no longer route through "
            "`ttnn.to_torch` -> torch -> `ttnn.from_torch`. If you "
            "can't satisfy BOTH, write nothing.\n\n"
            "Inspect the helper body and the pre-bound attributes "
            "on `self.<attribute>` (weight, bias, conv_config, "
            "kernel_size, stride, padding, ...) before deciding "
            "what ttnn call to use."
        )
    if failure_class == "DEVICE_NEEDS_RESET":
        return (
            "Failure class is DEVICE_NEEDS_RESET. This is NOT a code defect — "
            "the previous pytest hit a stale IOMMU / sysmem mapping inherited "
            "from a killed orphan, not anything the new stub did wrong. The "
            "tool has already executed `tt-smi -r` and is feeding you a fresh "
            "traceback. Keep the previously written native ttnn implementation; "
            "do NOT rewrite it to chase this error. Only modify the stub if the "
            "fresh traceback (below) shows a real numerical / shape / API error."
        )
    if failure_class == "HANG":
        return (
            "Failure class is HANG. The previous stub did not complete within "
            "the wall-clock budget. The hang is almost always a DEVICE-SIDE "
            "kernel deadlock (the host enqueued a ttnn op the kernel can't "
            "actually execute, so `ttnn.synchronize_device` and any subsequent "
            "`ttnn.to_torch` block forever). Top causes, in order of frequency:\n"
            "  1. **Tile-misaligned shape** going into a tile op. `ttnn.matmul` / "
            "     `ttnn.linear` / `ttnn.softmax` require the last 2 dims to be "
            "     multiples of 32 in TILE_LAYOUT. Adjust head_dim or pad.\n"
            "  2. **Permute on TILE_LAYOUT** for axes other than the last two. "
            "     Call `ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)` BEFORE permute "
            "     that touches non-tile axes, then `ttnn.to_layout(t, TILE)` "
            "     before the next compute op.\n"
            "  3. **Reshape in TILE_LAYOUT** with non-tile-aligned dims. Switch "
            "     to ROW_MAJOR_LAYOUT for the reshape, then back to TILE.\n"
            "  4. **Mismatched mesh distribution** between operands of one op "
            "     (e.g. input replicated, weight sharded across mesh). Use "
            "     `ttnn.ReplicateTensorToMesh(device)` for both, OR open a "
            "     1x1 sub-mesh for this PCC test.\n"
            "  5. Python-level unbounded loop in `__call__` (rare for native code).\n"
            "Rewrite the forward path with every intermediate produced by a "
            "single tile-aligned ttnn op, and explicit layout transitions around "
            "every permute/reshape."
        )
    if failure_class == "DTYPE_MISMATCH":
        return (
            "Failure class is DTYPE_MISMATCH. PyTorch reported that an op "
            "expected one scalar dtype but received another (typically "
            "`expected scalar type Float but found BFloat16`). This is "
            "usually one of three patterns:\n"
            "  1. Your TTNN forward returns a bfloat16 tensor, but the test "
            "harness compares against a float32 torch reference. Cast your "
            "final output via `ttnn.to_torch(out).to(torch.float32)`, or "
            "`ttnn.typecast(out, ttnn.float32)` before `ttnn.to_torch`. Do "
            "this for EVERY tensor in your return value (dict / tuple / list).\n"
            "  2. Mid-forward, a ttnn op is given a bfloat16 input and a "
            "float32 weight (or vice-versa). Either cast the weight at "
            "`__init__` (`ttnn.from_torch(w, dtype=ttnn.bfloat16, ...)`) or "
            "cast the input right before the op (`ttnn.typecast(x, ttnn.bfloat16)`).\n"
            "  3. A torch operation (e.g. `torch.cat`, `points_t.float()`) is "
            "being called on a ttnn.Tensor object. Use ttnn equivalents, OR "
            "convert to torch first with `ttnn.to_torch(t)`.\n"
            "Read the traceback to find the exact line where dtype assertion "
            "fired, then add the cast immediately above it. Do NOT change the "
            "test scaffold."
        )
    if failure_class == "L1_OOM":
        oom_attention_hint = (
            "  ATTENTION OOM CHECK (the #1 cause for vision transformers):\n"
            "    If your `__call__` builds a Q·K^T attention matrix of shape\n"
            "    (B, H, N, N) where N = H*W is the full spatial token count,\n"
            "    this matrix scales as O(N²). For inputs of 1024×1024 with\n"
            "    stride-4 patches, N = 65536 and Q·K^T is 65536×65536 ≈ 4.3B\n"
            "    elements at bf16 = ~8.6 GB — that is the OOM you are seeing.\n"
            "    Vision-transformer architectures with hierarchical encoders\n"
            "    (Hiera, Swin, ConvNeXt-V2 attention, MViT, etc.) use\n"
            "    WINDOWED attention in their early stages: spatial tokens are\n"
            "    partitioned into windows of size W (e.g. 8×8 or 14×14),\n"
            "    attention is computed within each window, then unpartitioned.\n"
            "    Inspect the HF reference (it likely calls `window_partition`\n"
            "    / `window_unpartition` or has `window_size` in its config /\n"
            "    block). Reproduce the same windowing in your ttnn forward —\n"
            "    do NOT compute a full (N,N) attention. Alternatively, use\n"
            "    `ttnn.transformer.scaled_dot_product_attention` which fuses\n"
            "    softmax and avoids materializing the full attention matrix.\n"
        )
        if strict_native:
            return (
                "Failure class is L1_OOM/DRAM_OOM (allocator could not fit the\n"
                "tensor). The pytest summary may say only `TT_FATAL @\n"
                "bank_manager.cpp:...: false`; the actual root cause is an\n"
                "oversized buffer, almost always either a full-attention\n"
                "(N,N) matrix or an un-sharded conv2d activation.\n" + oom_attention_hint + "  OTHER OOM ROOT CAUSES:\n"
                "    - oversized matmul output (split into smaller matmuls)\n"
                "    - halo-heavy conv2d (use HEIGHT_SHARDED or slice_config)\n"
                "    - replicated weights on every mesh device (shard them)\n"
                "  Avoid memory-aggressive kernels (large max_pool2d, oversized\n"
                "  matmul tiles). Do NOT delegate to a torch wrapper —\n"
                "  strict-native mode forbids it."
            )
        return (
            "Failure class is L1_OOM/DRAM_OOM (allocator could not fit the\n"
            "tensor). The pytest summary may say only `TT_FATAL @\n"
            "bank_manager.cpp:...: false`; the actual root cause is an\n"
            "oversized buffer, almost always either a full-attention (N,N)\n"
            "matrix or an un-sharded conv2d activation.\n" + oom_attention_hint + "  OTHER OOM ROOT CAUSES:\n"
            "    - oversized matmul output (split into smaller matmuls)\n"
            "    - halo-heavy conv2d (use HEIGHT_SHARDED or slice_config)\n"
            "    - replicated weights on every mesh device (shard them)\n"
            "  First try memory-friendly TTNN kernels (smaller tiles, different\n"
            "  sharding, split matmuls, windowed attention). If that still\n"
            "  OOMs after one more attempt, fall back to a torch-reference\n"
            "  wrapper for this component so the model can finish bring-up;\n"
            "  the user can promote it to native later via\n"
            "  `promote ... --strict-native`."
        )
    if failure_class == "L1_SMALL_ZERO":
        return (
            "Failure class is L1_SMALL_ZERO. The TT_FATAL message says\n"
            "`L1_SMALL buffer ... bank size is 0 B`. This is NOT a stub bug —\n"
            "the device was opened with `l1_small_size=0`. Some ttnn ops\n"
            "(notably `ttnn.conv2d` with its halo / sliding-window helper,\n"
            "and `ttnn.max_pool2d`) allocate small scratch buffers from the\n"
            "L1_SMALL bank, which only exists when the device opener was\n"
            "passed a non-zero `l1_small_size` (typical: 16384 or 24576).\n"
            "ACTION: do NOT rewrite the stub to chase this error. Either:\n"
            "  (a) leave the stub's `CONV2D_CPU_FALLBACK` path in place\n"
            "      (already correct — the helper traps the OOM and falls\n"
            "      back to `torch.nn.functional.conv2d` on host); the tool\n"
            "      will report this as a runtime CPU fallback and surface\n"
            "      it to the user, OR\n"
            "  (b) re-architect the op to avoid `ttnn.conv2d` entirely\n"
            "      (rare; only do this if the conv is trivial — e.g. a 1x1\n"
            "      conv expressible as a matmul on a reshaped input).\n"
            "Choice (a) is the right answer for production conv2d ops. Do\n"
            "NOT loop on this; one attempt at (a) and move on."
        )
    if failure_class == "EMBEDDING_DTYPE":
        return (
            "Failure class is EMBEDDING_DTYPE. `ttnn.embedding` requires\n"
            "the *index* tensor (positional arg 1) to be UINT32 or\n"
            "BFLOAT16. Your current code is passing INT32 / INT64 (the\n"
            "default torch dtype for `torch.arange` and most index\n"
            "tensors). SURGICAL FIX (one of the two):\n"
            "  - On host BEFORE upload:\n"
            "       idx_torch = idx_torch.to(torch.int32)  # then\n"
            "       idx = ttnn.from_torch(idx_torch, dtype=ttnn.uint32,\n"
            "                             layout=ttnn.ROW_MAJOR_LAYOUT,\n"
            "                             device=device)\n"
            "  - Or on device:\n"
            "       idx = ttnn.typecast(idx, ttnn.uint32)\n"
            "Note: `ttnn.uint32` indices must be ROW_MAJOR_LAYOUT, not\n"
            "TILE_LAYOUT. The weight (positional arg 2) is BFLOAT16 in\n"
            "TILE_LAYOUT as usual. Output of `ttnn.embedding` is BFLOAT16\n"
            "in ROW_MAJOR — convert back to TILE_LAYOUT before any\n"
            "downstream matmul / linear."
        )
    if failure_class == "CONCAT_INCOMPATIBLE":
        return (
            "Failure class is CONCAT_INCOMPATIBLE. `ttnn.concat(list, dim=...)`\n"
            "raised `TypeError: incompatible function arguments`. This is\n"
            "almost always because at least one element in the list is\n"
            "NOT a `ttnn.Tensor` on the device. SURGICAL FIX:\n"
            "  1. Inspect every branch that produces a tensor that ends up\n"
            "     in the concat list. Any branch that builds a tensor on\n"
            "     host (e.g. `torch.zeros(...)`, `_torch_module(...)` ref\n"
            "     output, or a Python literal) must be wrapped:\n"
            "       t = ttnn.from_torch(t, dtype=ttnn.bfloat16,\n"
            "                           layout=ttnn.TILE_LAYOUT,\n"
            "                           device=device)\n"
            "  2. ALL tensors in the list must share dtype + layout +\n"
            "     mesh distribution. If one is sharded and another is\n"
            "     replicated, replicate the sharded one with\n"
            "     `ttnn.to_memory_config(..., ttnn.DRAM_MEMORY_CONFIG)`\n"
            "     before concat.\n"
            "  3. The `dim` argument MUST be keyword-only (`dim=...`),\n"
            "     not positional. Check that the signature matches\n"
            "     `ttnn.concat(tensors: List[ttnn.Tensor], dim: int)`.\n"
            "Do not rewrite the rest of `__call__` — only the lines that\n"
            "build the concat list and the concat call itself."
        )
    if failure_class == "TT_FATAL_OPAQUE":
        return (
            "Failure class is TT_FATAL_OPAQUE. The traceback shows a bare\n"
            "`TT_FATAL @ <op>.cpp:<line>: false` with no human-readable\n"
            "predicate or shape information. You CANNOT fix this from the\n"
            "traceback alone — you need to know what the inputs to the\n"
            "failing op actually looked like at the call site.\n"
            "\n"
            "AGENTIC ACTION (use your Read/Edit/Bash tools):\n"
            "  1. Open the failing component's stub file (path under\n"
            "     `_stubs/<safe>.py` — listed in the COMPONENTS section).\n"
            "  2. Locate the ttnn op named in the TT_FATAL path (e.g.\n"
            "     `softmax.cpp` -> the `ttnn.softmax(...)` call,\n"
            "     `transpose.cpp` -> the `ttnn.transpose(...)` call,\n"
            "     `concat.cpp` -> the `ttnn.concat(...)` call). If\n"
            "     multiple instances of the op exist, instrument ALL of\n"
            "     them so the first one to hit fires the probe.\n"
            "  3. Inject the shape-probe template (copy verbatim) DIRECTLY\n"
            "     BEFORE the failing ttnn op call:\n"
            "\n"
            "       # SHAPE_PROBE (autoinjected by tt_hw_planner). Tag\n"
            "       # `<probe-tag>` is unique so we can grep the traceback.\n"
            "       def _probe(name, t):\n"
            "           import sys\n"
            "           try:\n"
            "               shape = tuple(t.shape) if hasattr(t, 'shape') else 'no-shape'\n"
            "               dtype = getattr(t, 'dtype', 'no-dtype')\n"
            "               layout = getattr(t, 'layout', 'no-layout')\n"
            "               mem = getattr(t, 'memory_config', lambda: 'no-mem')\n"
            "               try: mem = mem()\n"
            "               except Exception: pass\n"
            "               print(f'[SHAPE_PROBE <probe-tag>] {name}: shape={shape} dtype={dtype} layout={layout} mem={mem}', file=sys.stderr, flush=True)\n"
            "           except Exception as e:\n"
            "               print(f'[SHAPE_PROBE <probe-tag>] {name}: probe-error: {e}', file=sys.stderr, flush=True)\n"
            "       _probe('arg0', <first arg to failing op>)\n"
            "       _probe('arg1', <second arg to failing op>)   # if any\n"
            "       # ... probe every input to the failing op\n"
            "\n"
            "  4. Save the file. The outer tool will re-run pytest; the\n"
            "     SHAPE_PROBE prints will appear in stderr and the next\n"
            "     iter prompt will include them.\n"
            "  5. Use the printed shapes to deduce the predicate that\n"
            "     `TT_FATAL` checked (tile alignment? rank? dtype? memory\n"
            "     config?), then fix the inputs (reshape / typecast /\n"
            "     to_layout) before the call. Remove the probe in that\n"
            "     same iter once you know the answer.\n"
            "\n"
            "If you cannot identify the failing op from the TT_FATAL path\n"
            "(e.g. it points to a generic kernel like `device_operation.cpp`),\n"
            "instrument the LAST 3-5 ttnn ops in `__call__` BEFORE the\n"
            "exception bubbles up — at least one of them is the culprit."
        )
    if failure_class == "API_SIGNATURE":
        return (
            "Failure class is API_SIGNATURE. Use exact TTNN keyword-only signatures from runtime error text "
            "and do not use positional shortcuts."
        )
    if failure_class == "SHAPE":
        return (
            "Failure class is SHAPE. Preserve expected tensor rank/layout at each op boundary and only patch "
            "the minimal shape/layout transforms required for the failing path."
        )
    if failure_class == "PCC_ONLY":
        return (
            "Failure class is PCC_ONLY. The ttnn structure is correct — every op shapes, "
            "every kernel runs, the divergence is purely numerical. DO NOT redesign the "
            "forward; tune numerics/layout/scaling in the SAME structure. If PCC is "
            "improving across attempts (e.g. 0.4 -> 0.88 -> 0.99) you are on the right "
            "track — keep refining the SAME approach.\n"
            "\n"
            "TWO ENRICHED CONTEXT BLOCKS are inlined per-component below (only present for "
            "PCC_ONLY failures):\n"
            "  1. LOCALIZATION HINT — a per-`_apply_*` reference trace from the torch side, "
            "     with expected shape / dtype / mean / std / l2 at every helper boundary. "
            "     Use it as a CHECKLIST: at each `self._apply_X(...)` call site in your "
            "     `__call__`, the output you produce must match the listed (shape, mean, "
            "     std) within bfloat16 noise. The FIRST helper whose stats you cannot "
            "     mentally reproduce is your most likely PCC bug — fix that helper's "
            "     call site (or its body) instead of redesigning the rest. If the "
            "     LOCALIZATION HINT says 'no single helper diverged', the bug is in the "
            "     GLUE code BETWEEN helpers (reshapes, permutes, residual adds, scaling, "
            "     mask handling, activation between helpers).\n"
            "  2. FULL HF REFERENCE SOURCE — the COMPLETE torch class body (constructor, "
            "     helper methods, residual paths), not just `forward()`. Use it to verify "
            "     constants like `sqrt(d_k)`, attention scaling, residual-add order, "
            "     layer-norm eps, and activation variants against your ttnn translation.\n"
            "\n"
            "Common near-miss fixes (in priority order):\n"
            "  (a) missing 1/sqrt(head_dim) scaling on Q·K^T attention\n"
            "  (b) residual-add order mismatch (`x + self.norm(attn(x))` vs `self.norm(x + attn(x))`)\n"
            "  (c) wrong layer_norm/rms_norm eps (HF default 1e-5 vs 1e-6 vs 1e-12 differ at PCC ~0.95)\n"
            "  (d) activation variant (gelu_new vs gelu_pytorch_tanh vs quick_gelu vs silu)\n"
            "  (e) bfloat8_b accumulation where bfloat16 is required (matmul math_fidelity, LayerNorm)\n"
            "  (f) ttnn.softmax called over the wrong axis (must be last axis after permute)\n"
            "  (g) permute axes mismatch — torch (B, H, N, D) often vs ttnn (B, N, H, D)"
        )
    return "Patch only the failing code path and keep all passing modules unchanged."


_TORCH_WRAPPER_PATTERNS = [
    r"self\._torch_module\s*\(",
    r"self\.torch_module\s*\(",
    r"_get_torch_submodule\s*\(",
    r"with\s+torch\.no_grad\(\)\s*:",
]

_FORWARD_METHOD_NAMES = {"__call__", "forward"}


def _stub_uses_torch_wrapper(stub_path: Path) -> bool:
    if not stub_path.is_file():
        return False
    try:
        text = stub_path.read_text(errors="ignore")
    except Exception:
        return False
    try:
        import ast

        tree = ast.parse(text)
    except SyntaxError:
        for pat in _TORCH_WRAPPER_PATTERNS:
            if re.search(pat, text):
                return True
        return False

    def _name_of_call(node: "ast.AST") -> str:
        try:
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    return f.id
                if isinstance(f, ast.Attribute):
                    base = ""
                    if isinstance(f.value, ast.Name):
                        base = f.value.id
                    elif isinstance(f.value, ast.Attribute) and isinstance(f.value.value, ast.Name):
                        base = f"{f.value.value.id}.{f.value.attr}"
                    return f"{base}.{f.attr}" if base else f.attr
        except Exception:
            pass
        return ""

    forbidden_call_names = {
        "_get_torch_submodule",
        "self._torch_module",
        "self.torch_module",
    }
    forbidden_call_basenames = {"_get_torch_submodule"}
    helper_function_names = {"_get_torch_submodule", "_resolve"}

    def _body_calls_torch_fallback(fn) -> bool:
        for node in ast.walk(fn):
            if isinstance(node, ast.With):
                for item in node.items:
                    expr = item.context_expr
                    if isinstance(expr, ast.Call):
                        name = _name_of_call(expr)
                        if name in {"torch.no_grad", "no_grad"}:
                            return True
            if isinstance(node, ast.Call):
                name = _name_of_call(node)
                if name in forbidden_call_names:
                    return True
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_call_basenames:
                    return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in helper_function_names:
                continue
            if _body_calls_torch_fallback(node):
                return True

    try:
        from .commands.emit_e2e import _check_hf_fallback as _g1b_check
    except Exception:  # noqa: BLE001
        _g1b_check = None
    if _g1b_check is not None:
        try:
            if _g1b_check(text):
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _stub_source_excerpt(stub_path: Path, *, max_lines: int = 140) -> str:
    if not stub_path.is_file():
        return "(stub file not found)"
    try:
        text = stub_path.read_text(errors="ignore")
    except Exception:
        return "(stub unreadable)"
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n# ... ({len(lines) - max_lines} more lines truncated)"


def _extract_ops_used(stub_path: Path) -> List[str]:
    if not stub_path.is_file():
        return []
    try:
        text = stub_path.read_text(errors="ignore")
    except Exception:
        return []
    ops = sorted(set(re.findall(r"ttnn\.([a-zA-Z_][a-zA-Z_0-9]*)\s*\(", text)))
    return ops


_SHARDING_KEYWORDS = (
    ("width_sharded", "width-sharded"),
    ("block_sharded", "block-sharded"),
    ("height_sharded", "height-sharded"),
    ("interleaved", "interleaved"),
    ("HEIGHT_SHARDED", "height-sharded"),
    ("WIDTH_SHARDED", "width-sharded"),
    ("BLOCK_SHARDED", "block-sharded"),
)


def _extract_sharding_strategy(stub_path: Path) -> List[str]:
    if not stub_path.is_file():
        return []
    try:
        text = stub_path.read_text(errors="ignore")
    except Exception:
        return []
    found: List[str] = []
    for needle, label in _SHARDING_KEYWORDS:
        if needle in text and label not in found:
            found.append(label)
    return found


def _diagnose_failure(failure_class: str, traceback_excerpt: str, ops_used: List[str]) -> Tuple[str, str]:
    tb_low = (traceback_excerpt or "").lower()
    if failure_class == "DEVICE_NEEDS_RESET":
        return (
            "stale IOMMU/sysmem mapping inherited from a killed orphan pytest "
            "(`pin_or_map_sysmem_to_device` refused to proceed because the "
            "kernel still had the previous process's mapping pinned)",
            "no source change required from the agent — the tool runs `tt-smi -r` "
            "automatically and retries the same pytest once. Leave the stub as is "
            "and wait for the post-reset traceback; if THAT one still fails, the "
            "failure class will be re-classified to whatever actually broke",
        )
    if failure_class == "HANG":
        return (
            "pytest was killed by the wall-clock timeout (the previous stub hung "
            "inside the ttnn forward path — likely an unbounded Python loop, "
            "an infinite recursion, or a device-side deadlock)",
            "remove any `while True:` / unbounded python loops; ensure every "
            "`ttnn.*` op produces a tensor in finite steps; avoid recursive "
            "calls in `__call__`; replace explicit per-element Python iteration "
            "with one batched ttnn op; double-check that `build()` does not "
            "block on a remote download or weight load that never completes",
        )
    if failure_class == "L1_OOM":
        if "matmul" in tb_low or "ttnn.matmul" in ops_used or "ttnn.linear" in ops_used:
            return (
                "matmul/linear ran out of L1 memory on the device",
                "use smaller tiles (16/32 instead of 64), or shard differently "
                "(e.g. block-shard instead of width-shard), or split into two "
                "smaller matmuls",
            )
        if "conv" in tb_low or "ttnn.conv2d" in ops_used:
            return (
                "conv2d ran out of L1 memory",
                "shard input with halo (use ttnn.conv2d's halo-sharded input mode), "
                "or reduce conv tile output size, or use a smaller batch",
            )
        return (
            "device L1 memory exhausted",
            "reduce intermediate buffer sizes via smaller tiles or different sharding",
        )
    if failure_class == "TT_FATAL_SHAPE":
        from ._cli_helpers.error_patterns import extract_tt_fatal_op_and_predicate

        _info = extract_tt_fatal_op_and_predicate(traceback_excerpt or "") or {}
        _op = _info.get("op") or "(unknown op)"
        _predicate = _info.get("predicate") or "(see traceback)"
        _file = _info.get("cpp_file") or ""
        return (
            f"`ttnn.{_op}` failed a device-side shape/dtype assertion. "
            f"The assertion text says EXACTLY what's wrong: `{_predicate}`. "
            f"Translate the predicate to what your stub must produce: a "
            f"comparison like `a.logical_shape()[-1] == gamma.logical_shape()[-1]` "
            f"means the LAST dim of the LHS tensor (the one you pass as `a`/`input`) "
            f"must match the LAST dim of the RHS tensor (the one you pass as "
            f"`gamma`/`weight`/etc.). Cpp source: {_file}",
            f"Step-by-step fix for {_op}-class shape asserts:\n"
            f"  (1) In your stub, print/log the shape of EVERY tensor you pass to "
            f"`ttnn.{_op}(...)` immediately before the call. Identify the two "
            f"named in the predicate `{_predicate}`.\n"
            f"  (2) If the LHS tensor is wrong shape, fix the upstream op: usually a "
            f"missing reshape (e.g. (B,S,H) -> (1,B,S,H) for 4D-required ttnn "
            f"layernorm, or (S,B,H) -> (B,S,H) layout swap).\n"
            f"  (3) If the RHS tensor (gamma/weight/etc.) is wrong shape, fix "
            f"weight loading: load the right slice from state_dict (e.g. "
            f"`state_dict['input_layernorm.weight']` for RMSNorm gamma should "
            f"be `(hidden_size,)` = `(args.dim,)`).\n"
            f"  (4) Do NOT randomly try transposes — the predicate names the "
            f"exact mismatch. Match the named dim, that's it.",
        )
    if failure_class == "SHAPE":
        return (
            "tensor shape mismatch between produced and expected output",
            "check build()'s weight reshape/transpose, check input layout (TILE vs ROW_MAJOR), "
            "verify reshape/permute order matches torch_ref",
        )
    if failure_class == "API_SIGNATURE":
        return (
            "called a ttnn op with wrong arguments",
            "re-check the op's required kwargs (dtype, memory_config, compute_kernel_config); "
            "many ttnn ops require explicit memory_config",
        )
    if failure_class == "PCC_ONLY":
        # Specialize the diagnosis by PCC value when available. The
        # "structurally correct, numerically close" regime
        # (auto_iterate.py:488, agentic/convergence.py:288) already
        # gets CAP RELAXATION via PCC_STUCK_THRESHOLD — but the
        # diagnosis text was generic regardless of PCC. Reuse the
        # existing `_extract_pcc_from_failure` parser (cli.py:4299) to
        # pick the value out of the traceback, then route to the
        # specialized HIGH-PCC hint when PCC >= 0.85 (close enough that
        # the structural code is right; only numerical tuning remains).
        _pcc_val = _extract_pcc_from_failure(traceback_excerpt or "", "")
        if _pcc_val is not None and _pcc_val >= 0.85:
            return (
                f"PCC {_pcc_val:.4f} is in the late-stage numerical "
                f"refinement zone (>= 0.85). The wiring is correct; the "
                f"remaining gap is precision/scaling, NOT structural code. "
                f"At this PCC level, broad rewrites usually REGRESS the "
                f"trajectory — make ONE targeted change per iter and "
                f"observe whether PCC moves up.",
                "Targeted candidates (try ONE per iter, smallest first): "
                "(a) intermediate accumulator dtype: bf16 -> fp32 for any "
                "softmax / sum / mean / scaled-dot-product step "
                "(`compute_kernel_config=ttnn.WormholeComputeKernelConfig"
                "(math_fidelity=ttnn.MathFidelity.HiFi4)` or fp32 dest acc "
                "where supported); (b) scaling factor placement — "
                "Q*K^T scaling must be done BEFORE softmax, not after "
                "(some Llama-family canonicals fold scale into rotary; "
                "verify which); (c) weight transpose order — `(out, in)` "
                "vs `(in, out)` mismatch flips PCC into the 0.7-0.95 "
                "range; (d) RoPE theta — Qwen2 uses `rope_theta=1_000_000` "
                "but standard rotary defaults to 10_000, off by 100x in "
                "the position frequencies; (e) bias/residual omission — "
                "if the model HAS attention.q_proj.bias but the wrapper "
                "ignored it, PCC lands in the 0.85-0.95 region.",
            )
        return (
            "ttnn output is numerically wrong vs torch reference (PCC < 0.99)",
            "check dtype (bf16 vs fp32), weight transpose order, missing scaling factor, "
            "or a missed bias/residual term",
        )
    if failure_class == "STATE_DICT_KEY":
        from ._cli_helpers.error_patterns import extract_missing_state_dict_key

        missing_key = extract_missing_state_dict_key(traceback_excerpt or "") or "<unknown>"
        return (
            f"state_dict does NOT contain the key the canonical class is "
            f"looking up: {missing_key!r}. The canonical class expects its "
            f"sibling-model's naming convention (Llama/Meta-style e.g. "
            f"`layers.N.attention.wq.weight`); the wrapper passed HF naming "
            f"(`q_proj.weight` / `gate_proj.weight` / etc.) without remapping",
            f"in `build()`, BEFORE calling the canonical constructor, REMAP "
            f"the state_dict keys: (a) call the right `convert_hf_to_meta` "
            f"helper from `models.tt_transformers.tt.load_checkpoints` to "
            f"swap HF projection names (q_proj→wq, k_proj→wk, v_proj→wv, "
            f"o_proj→wo, gate_proj→w1, up_proj→w3, down_proj→w2) and (b) "
            f"prefix the remapped keys with `layers.{{layer_num}}.<module>.` "
            f"(e.g. `layers.0.attention.` for Attention, `layers.0.feed_forward.` "
            f"for MLP). Pass the remapped dict as `state_dict=`. The canonical "
            f"class's `__init__` will then find {missing_key!r} (or its remapped "
            f"equivalent) and load weights",
        )
    if failure_class == "UNEXPECTED_KWARG":
        from ._cli_helpers.error_patterns import extract_unexpected_kwarg

        bad_kwarg = extract_unexpected_kwarg(traceback_excerpt or "") or "<unknown>"
        return (
            f"the canonical class's `__init__` does NOT accept the kwarg "
            f"{bad_kwarg!r}. The canonical-wrapper template passes a generic "
            f"set of kwargs (mesh_device, args, state_dict, layer_num, dtype) "
            f"that fits Attention/MLP-style classes — but the wrapped class "
            f"may have a totally different signature (e.g. RMSNorm takes "
            f"`device` not `mesh_device`; RotaryEmbedding takes `dim, "
            f"max_position_embeddings, base, device` and no `args`/`state_dict` "
            f"at all)",
            f"REMOVE {bad_kwarg!r} from the canonical constructor call AND "
            f"replace the rest of the call with the canonical's actual "
            f"`__init__` signature. Read the canonical class's `def __init__` "
            f"DEFINITION line first (grep `^    def __init__` in the "
            f"`tt_reuse_target` file) to see the exact arg list. Some "
            f"components have NO learned weights (e.g. RotaryEmbedding "
            f"computes cos/sin tables from scratch) — for those, drop "
            f"state_dict / torch_module entirely",
        )
    if failure_class == "MISSING_KWARG":
        from ._cli_helpers.error_patterns import extract_missing_args_description

        missing = extract_missing_args_description(traceback_excerpt or "") or "<see traceback>"
        return (
            f"the canonical class's `__init__` needs additional positional "
            f"arguments that the wrapper template did not pass: {missing}",
            f"add the missing args to the canonical constructor call. Common "
            f"ones for tt_transformers classes: `tt_ccl=get_tt_ccl(device)` "
            f"(import from `models.common.modules.tt_ccl`), "
            f"`weight_cache_path=Path(args.weight_cache_path)`, "
            f"`transformation_mats=args.get_rot_mat()` or "
            f"`transformation_mats={{}}` for stubs that don't yet need RoPE, "
            f"`configuration=args` (the ModelArgs instance), "
            f"`model_config=args.model_config`. Read the canonical's full "
            f"`def __init__` signature (grep `^    def __init__` in the "
            f"`tt_reuse_target` file) before guessing",
        )
    return (
        f"failure class={failure_class}; see traceback",
        "re-read the traceback carefully and address the specific error",
    )


def _attempt_log_dir(demo_dir: Path, component_name: str) -> Path:
    return demo_dir / "_attempts" / _safe_id(component_name)


_FAILING_LINE_PATTERN = re.compile(r'File "([^"]+)", line (\d+)')


def _extract_failing_stub_excerpt(traceback_excerpt: str, stub_path: Path) -> Tuple[int, str]:
    """Parse a Python traceback and return (line_number, source_excerpt)
    for the LAST frame that references ``stub_path``.

    The excerpt is centered on the failing line with 6 lines of context
    on each side. Returns (0, "") if no frame in the traceback points to
    the stub or if the stub file isn't readable.

    Generic across all stubs — never references model-specific paths.
    """
    if not traceback_excerpt or not stub_path.is_file():
        return 0, ""
    stub_name = stub_path.name
    failing_line = 0
    for match in _FAILING_LINE_PATTERN.finditer(traceback_excerpt):
        path_str, line_str = match.group(1), match.group(2)
        if stub_name in path_str:
            try:
                failing_line = int(line_str)
            except ValueError:
                continue
    if failing_line <= 0:
        return 0, ""
    try:
        src_lines = stub_path.read_text(errors="ignore").splitlines()
    except Exception:
        return 0, ""
    if failing_line > len(src_lines):
        return 0, ""
    start = max(0, failing_line - 7)
    end = min(len(src_lines), failing_line + 6)
    excerpt_lines: List[str] = []
    for idx in range(start, end):
        marker = ">>> " if (idx + 1) == failing_line else "    "
        excerpt_lines.append(f"{idx + 1:5d} {marker}{src_lines[idx]}")
    return failing_line, "\n".join(excerpt_lines)


def _write_attempt_log(
    *,
    demo_dir: Path,
    component_name: str,
    iter_n: int,
    stub_path: Path,
    exemplar_used: Optional[str],
    model_used: str,
    failure_class: str,
    failure_signature: str,
    traceback_excerpt: str,
    diagnosis_override: Optional[str] = None,
    next_step_override: Optional[str] = None,
    agent_result_text: Optional[str] = None,
) -> None:
    try:
        ops_used = _extract_ops_used(stub_path)
        sharding = _extract_sharding_strategy(stub_path)
        diagnosis, next_step = _diagnose_failure(failure_class, traceback_excerpt, ops_used)
        if diagnosis_override:
            diagnosis = diagnosis_override
        if next_step_override:
            next_step = next_step_override
        try:
            stub_hash = hashlib.sha1(stub_path.read_bytes()).hexdigest()[:12] if stub_path.is_file() else ""
        except Exception:
            stub_hash = ""
        failing_line, failing_line_excerpt = _extract_failing_stub_excerpt(traceback_excerpt or "", stub_path)
        entry = {
            "iter": iter_n,
            "stub_path": str(safe_relative_to_root(stub_path)) if stub_path.is_absolute() else str(stub_path),
            "stub_hash": stub_hash,
            "exemplar_used": exemplar_used or "(none)",
            "model_used": model_used,
            "ops_used": ops_used,
            "sharding_strategy": sharding,
            "failure_class": failure_class,
            "failure_signature": failure_signature,
            "traceback_excerpt": (traceback_excerpt or "")[:2000],
            "diagnosis": diagnosis,
            "next_step": next_step,
            "failing_line": failing_line,
            "failing_line_excerpt": failing_line_excerpt,
            "agent_result_text": (agent_result_text or "").strip()[:1500],
        }
        log_dir = _attempt_log_dir(demo_dir, component_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"iter_{iter_n:03d}.json").write_text(json.dumps(entry, indent=2))
    except Exception as exc:
        print(f"  (could not write attempt log for {component_name}: {exc})", file=sys.stderr)


def _load_attempt_history(demo_dir: Path, component_name: str, max_entries: int = 3) -> List[Dict[str, object]]:
    log_dir = _attempt_log_dir(demo_dir, component_name)
    if not log_dir.is_dir():
        return []
    files = sorted(log_dir.glob("iter_*.json"))
    if not files:
        return []
    entries: List[Dict[str, object]] = []
    for f in files[-max_entries:]:
        try:
            entries.append(json.loads(f.read_text()))
        except Exception:
            continue
    return entries


def _format_attempt_history_block(history: List[Dict[str, object]]) -> str:
    if not history:
        return "  (no prior attempts on file — this is the first try for this component)"
    lines: List[str] = []
    for entry in history:
        it = entry.get("iter", "?")
        ops = ", ".join(entry.get("ops_used", []) or []) or "(none)"
        shard = ", ".join(entry.get("sharding_strategy", []) or []) or "(unspecified)"
        ex = entry.get("exemplar_used", "(none)")
        mdl = entry.get("model_used", "(unknown)")
        fc = entry.get("failure_class", "?")
        diag = entry.get("diagnosis", "(no diagnosis)")
        nxt = entry.get("next_step", "")
        agent_said = str(entry.get("agent_result_text") or "").strip()
        block = (
            f"  Iter {it} (model={mdl}, exemplar={ex}):\n"
            f"    ttnn ops used   : {ops}\n"
            f"    sharding tried  : {shard}\n"
            f"    failure class   : {fc}\n"
            f"    diagnosis       : {diag}\n"
            f"    next step hint  : {nxt}"
        )
        if agent_said:
            preview = agent_said[:400].replace("\n", " ")
            block += f"\n    prior agent said: {preview}{'…' if len(agent_said) > 400 else ''}"
        lines.append(block)
    if history:
        last = history[-1]
        prior_ops = sorted({op for h in history for op in (h.get("ops_used") or [])})
        prior_shard = sorted({s for h in history for s in (h.get("sharding_strategy") or [])})
        prior_ex = sorted(
            {h.get("exemplar_used") for h in history if h.get("exemplar_used") and h.get("exemplar_used") != "(none)"}
        )
        last_excerpt = str(last.get("failing_line_excerpt") or "").strip()
        last_line_no = int(last.get("failing_line") or 0)
        last_tb = str(last.get("traceback_excerpt") or "").strip()
        if last_excerpt and last_tb:
            lines.append("")
            lines.append("  YOUR PRIOR ATTEMPT FAILED AT THIS EXACT LINE (do NOT write this again):")
            lines.append(
                f"    The CURRENT STUB below IS your iter-{last.get('iter', '?')} attempt "
                f"that crashed at line {last_line_no}. Identify the bug and write SOMETHING "
                f"DIFFERENT — re-writing the same operation will reproduce the same failure."
            )
            lines.append("    Failing-line excerpt (>>> marks the crashing line):")
            for src_line in last_excerpt.splitlines():
                lines.append(f"      {src_line}")
            lines.append("    Literal error from that line:")
            tb_tail = last_tb.splitlines()[-6:]
            for tb_line in tb_tail:
                lines.append(f"      {tb_line}")
        lines.append("")
        lines.append("  CONSTRAINTS FOR THIS ITERATION (do NOT repeat what already failed):")
        if prior_shard:
            lines.append(f"    - sharding strategies already tried and failed: {', '.join(prior_shard)}")
        if prior_ex:
            lines.append(f"    - exemplars already used: {', '.join(prior_ex)}")
        if prior_ops:
            lines.append(f"    - ttnn ops you have already used in failing attempts: {', '.join(prior_ops)}")
        lines.append(f"    - latest diagnosis: {last.get('diagnosis', '(none)')}")
        lines.append(f"    - latest next-step hint: {last.get('next_step', '(none)')}")
    return "\n".join(lines)


def _build_cross_component_context_block(
    demo_dir: Path,
    *,
    current_target: Optional[str],
    attempts_per_component: Optional[Dict[str, int]] = None,
    last_failure_class_per_component: Optional[Dict[str, str]] = None,
    max_signatures: int = 6,
    max_failure_rows: int = 6,
) -> str:
    """2026-05-23 (Improvement 1): build a compact "what's happening in
    the REST of the model" block to inject into the per-component LLM
    prompt. Closes the "per-component blinders" gap noted by the
    user (the agentic loop sees one component at a time and so misses
    cross-component context that an interactive Claude session would
    naturally have).

    The block contains, in order:
      1. Bring-up position summary  (X graduated / Y in-progress / Z untouched)
      2. Other-components signatures (their __call__ signatures so the
         agent knows the upstream/downstream interface contract)
      3. Cross-component failure patterns (are OTHER components ALSO
         hitting WRAPPER / PCC / SHAPE? -- signals a shared upstream
         blocker that needs systemic action, not per-component
         iteration)

    Strictly additive: returns "" on any read error so the prompt
    still composes correctly. Bounded output size so it doesn't blow
    out the prompt token budget for trivial-component iters."""
    try:
        status_path = demo_dir / "bringup_status.json"
        if not status_path.is_file():
            return ""
        status = json.loads(status_path.read_text())
    except Exception:
        return ""

    components = status.get("components") or []
    if not isinstance(components, list) or not components:
        return ""

    position_lines: List[str] = []
    try:
        graduated_count = 0
        in_progress_count = 0
        untouched_count = 0
        for c in components:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or ""

            safe = _safe_id(str(name))
            stub_path = demo_dir / "_stubs" / f"{safe}.py"
            if name == current_target:
                in_progress_count += 1
                continue
            if stub_path.is_file() and not _stub_uses_torch_wrapper(stub_path):
                graduated_count += 1
            else:
                untouched_count += 1
        position_lines.append(
            f"  position: {graduated_count} graduated / "
            f"{in_progress_count} in-progress (THIS one) / "
            f"{untouched_count} untouched"
        )
    except Exception:
        pass

    sig_lines: List[str] = []
    try:
        sig_count = 0
        for c in components:
            if sig_count >= max_signatures:
                break
            if not isinstance(c, dict):
                continue
            name = c.get("name") or ""
            if not name or name == current_target:
                continue
            safe = _safe_id(str(name))
            stub_path = demo_dir / "_stubs" / f"{safe}.py"
            if not stub_path.is_file():
                continue
            if _stub_uses_torch_wrapper(stub_path):
                continue

            try:
                src = stub_path.read_text()
            except Exception:
                continue

            m = re.search(
                r"^(\s*def __call__\([^)]*\)\s*(?:->\s*[^:]+)?:)",
                src,
                flags=re.MULTILINE,
            )
            if m:
                sig = m.group(1).strip()

                if len(sig) > 160:
                    sig = sig[:160] + "..."
                sig_lines.append(f"  {name}: {sig}")
                sig_count += 1
        if sig_count == 0:
            sig_lines.append("  (no other components graduated yet)")
    except Exception:
        pass

    failure_lines: List[str] = []
    try:
        if last_failure_class_per_component:
            by_class: Dict[str, List[str]] = {}
            for comp, fc in last_failure_class_per_component.items():
                if not fc:
                    continue
                if comp == current_target:
                    continue
                by_class.setdefault(str(fc), []).append(comp)
            shown = 0
            for fc, comps in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
                if shown >= max_failure_rows:
                    break
                failure_lines.append(
                    f"  {fc}: {len(comps)} other component(s) -- "
                    f"{', '.join(comps[:4])}"
                    f"{'...' if len(comps) > 4 else ''}"
                )
                shown += 1
        if not failure_lines:
            failure_lines.append("  (no recurring failure pattern across components)")
    except Exception:
        pass

    return (
        "\n--- CROSS-COMPONENT CONTEXT (what's happening in the REST of the bring-up) ---\n"
        + "\n".join(position_lines)
        + "\n\n  other components' __call__ signatures (the model's interface contract):\n"
        + "\n".join(sig_lines)
        + "\n\n  cross-component failure pattern (shared blockers, if any):\n"
        + "\n".join(failure_lines)
        + "\n"
    )


def _format_failure_block_for_component(
    per_component_failures: Dict[str, List[Dict[str, object]]],
    component_name: str,
    *,
    body_lines_per_test: int = 80,
) -> str:
    entries = per_component_failures.get(component_name, [])
    if not entries:
        return "(no failure recorded for this component in the latest pytest report)"
    chunks: List[str] = []
    for e in entries:
        test_id = str(e.get("test_id") or "?")
        exc = str(e.get("exception_type") or "").strip()
        msg = str(e.get("message") or "").strip()
        body = str(e.get("body") or "").strip()
        pcc = e.get("pcc_value")
        head = [f"FAILED TEST: {test_id}"]
        if exc:
            head.append(f"  exception type    : {exc}")
        if msg:
            head.append(f"  assertion message : {msg.splitlines()[0]}")
            extra_msg = msg.splitlines()[1:]
            if extra_msg:
                head.append("  full message      :")
                for line in extra_msg[:8]:
                    head.append(f"    {line}")
        if isinstance(pcc, (int, float)):
            head.append(f"  pcc achieved      : {pcc:.4f}  (target >= 0.99)")
        if body:
            body_lines = body.splitlines()
            head.append("  traceback         :")
            for line in body_lines[:body_lines_per_test]:
                head.append(f"    {line}")
            if len(body_lines) > body_lines_per_test:
                head.append(f"    ... ({len(body_lines) - body_lines_per_test} more lines elided)")
        chunks.append("\n".join(head))
    return "\n\n".join(chunks)


def _read_test_source(demo_dir: Path, component_name: str, max_lines: int = 60) -> str:
    safe = _safe_id(component_name)
    test_path = demo_dir / "tests" / "pcc" / f"test_{safe}.py"
    if not test_path.is_file():
        return "(no test file found for this component at "
        f"tests/pcc/test_{safe}.py)"
    try:
        text = test_path.read_text(errors="ignore")
    except Exception as exc:
        return f"(could not read test source: {exc})"
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + f"\n# ... ({len(lines) - max_lines} more lines elided)"
    try:
        rel = safe_relative_to_root(test_path)
    except Exception:
        rel = test_path
    return f"  source: {rel}\n```python\n{text}\n```"


def _component_metadata(demo_dir: Path, component_name: str) -> Optional[Dict[str, object]]:
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        return None
    try:
        data = json.loads(status_path.read_text())
    except Exception:
        return None
    for comp in data.get("components", []):
        if str(comp.get("name", "")).strip() == component_name:
            return comp
    return None


def _format_captured_shape_contract_block(demo_dir: Path, component_name: str) -> str:
    """Surface the captured forward IO shapes from `_captured/<safe>/manifest.json`
    so the agent sees ground-truth tensor shapes BEFORE writing code.

    Without this block the agent has to infer shapes from torch reference
    source (where shapes are runtime values like `x.size(-1)`) and only sees
    real shape data via the post-hoc SHAPE_PROBE block on iter 2+. v12 audit
    showed SHAPE-class failures (encoder_stack, hiera_det_model) persist
    across iters partly because the agent guesses wrong reshape/permute
    orientations that compile but scramble dimensions.

    Returns "" when no captured manifest exists -- harmless for components
    whose capture failed (the test scaffold's _make_arg_for path still
    provides validation, just less precisely)."""
    safe = _safe_id(component_name)
    manifest_path = demo_dir / "_captured" / safe / "manifest.json"
    if not manifest_path.is_file():
        return ""
    try:
        m = json.loads(manifest_path.read_text())
    except Exception:
        return ""

    def _render(v: object, indent: int = 0) -> str:
        if not isinstance(v, dict):
            return repr(v)
        pad = "  " * indent
        kind = v.get("kind", "?")
        if kind == "tensor":
            shape = tuple(v.get("shape", []) or [])
            return f"Tensor shape={shape} dtype={v.get('dtype', '?')}"
        if kind in ("list", "tuple"):
            items = v.get("items", []) or []
            if not items:
                return f"empty {kind}"
            inner = "\n".join(f"{pad}  [{i}]: {_render(it, indent + 1)}" for i, it in enumerate(items))
            return f"{kind}({len(items)}):\n{inner}"
        if kind == "dict":
            items = v.get("items", {}) or {}
            if not items:
                return "empty dict"
            inner = "\n".join(f"{pad}  {k}: {_render(it, indent + 1)}" for k, it in items.items())
            return f"dict({len(items)}):\n{inner}"
        if kind == "none":
            return "None"
        if kind == "scalar":
            return f"{v.get('type', '?')}({v.get('repr', '')})"
        return str(v)

    args = m.get("args") or {"kind": "tuple", "items": []}
    kwargs = m.get("kwargs") or {"kind": "dict", "items": {}}
    output = m.get("output") or {"kind": "none"}
    submodule_path = str(m.get("submodule_path", "?"))

    return (
        f"CAPTURED I/O CONTRACT for `{component_name}` "
        f"(from a real HF forward pass — your forward path will receive these exact shapes):\n"
        f"  submodule resolved at : {submodule_path}\n"
        f"  positional args       : {_render(args, indent=1)}\n"
        f"  keyword args          : {_render(kwargs, indent=1)}\n"
        f"  expected output       : {_render(output, indent=1)}\n"
        f"  Match these shapes/dtypes exactly. Reshape / permute / view operations "
        f"must preserve total element count AND semantic axis ordering. If torch "
        f"uses NCHW and ttnn op needs NHWC, permute explicitly.\n"
    )


def _op_synth_manifest(demo_dir: Path, component_name: str) -> Optional[Dict[str, object]]:
    """Load the op-synth manifest for a component, if one was written by
    `autofill_stubs(..., op_synth=True)`.

    Presence of `<safe>.opplan.json` next to `<safe>.py` is the signal
    that this stub is an op-synth partial port (weights pre-loaded, `_apply_*`
    helpers pre-bound), and that the LLM should only rewrite `__call__`
    instead of the entire file."""
    safe = _safe_id(component_name)
    manifest_path = demo_dir / "_stubs" / f"{safe}.opplan.json"
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return None


def _op_synth_prompt_blocks(demo_dir: Path, component_name: str) -> Tuple[str, str]:
    """Return (palette_block, contract_override) for the per-component
    prompt section. Both strings are empty when no op-synth manifest is
    present for this component, i.e. when the stub is a plain
    torch-fallback wrapper and the original full-file contract applies."""
    manifest = _op_synth_manifest(demo_dir, component_name)
    if not manifest:
        return "", ""

    palette = manifest.get("palette") or []
    llm_gaps = manifest.get("llm_gaps") or []
    cls_name = str(manifest.get("class_name", ""))
    safe = _safe_id(component_name)

    stub_path = demo_dir / "_stubs" / f"{safe}.py"
    fallback_helpers: List[str] = []
    notimpl_helpers: List[str] = []
    try:
        stub_text = stub_path.read_text(errors="ignore") if stub_path.is_file() else ""
    except Exception:
        stub_text = ""
    fallback_kind: Dict[str, str] = {}
    if stub_text:
        helper_re = re.compile(
            r"^(    def _apply_([A-Za-z0-9_]+)\(.*?\):)(.*?)(?=^    def |\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for match in helper_re.finditer(stub_text):
            name = match.group(2)
            body = match.group(3)
            if "CONV2D_CPU_FALLBACK" in body:
                fallback_helpers.append(name)
                fallback_kind[name] = "conv2d"
            elif "ACTIVATION_CPU_FALLBACK" in body:
                fallback_helpers.append(name)
                fallback_kind[name] = "activation"
            if "raise NotImplementedError" in body:
                notimpl_helpers.append(name)

    palette_lines: List[str] = []
    for entry in palette:
        m = re.search(r"self\._apply_([A-Za-z0-9_]+)\(", entry)
        suffix = ""
        if m:
            helper = m.group(1)
            if helper in notimpl_helpers:
                suffix = "    [GAP — body raises NotImplementedError, you MUST implement]"
            elif helper in fallback_helpers:
                kind = fallback_kind.get(helper, "ttnn")
                if kind == "conv2d":
                    suffix = "    [native ttnn.conv2d + CPU fallback; optional to specialize]"
                elif kind == "activation":
                    suffix = "    [CPU fallback via torch ref; optional to replace with raw ttnn ops]"
                else:
                    suffix = "    [native ttnn path + CPU fallback; optional to specialize]"
        palette_lines.append(f"  {entry}{suffix}")
    palette_text = "\n".join(palette_lines) or "  (no pre-bound helpers)"

    if llm_gaps:
        gap_lines: List[str] = []
        for g in llm_gaps:
            name = g.get("name", "?")
            klass = g.get("class", "?")
            note = g.get("notes") or ""
            gap_lines.append(f"  - {name}  ({klass})" + (f"  [{note}]" if note else ""))
        gaps_text = "\n".join(gap_lines)
    else:
        gaps_text = "  (none — fully deterministic; only `__call__` left)"

    counts = manifest.get("counts", {}) or {}
    summary = (
        f"  op-REUSE = {counts.get('op-REUSE', 0)}, "
        f"op-ADAPT = {counts.get('op-ADAPT', 0)}, "
        f"op-NEW   = {counts.get('op-NEW', 0)}"
    )

    palette_block = (
        f"\n--- OP-SYNTH PALETTE (pre-bound helpers; the heavy lifting is done) ---\n"
        f"This stub was emitted by the op-level classifier. Weights are\n"
        f"already loaded as `self.w_*` ttnn tensors inside `__init__`,\n"
        f"and the following helpers are already correct and ready to call:\n"
        f"{palette_text}\n"
        f"\n"
        f"OP-NEW gaps still requiring synthesis (implement these inside\n"
        f"`__call__` directly, or add small private helpers):\n"
        f"{gaps_text}\n"
        f"\n"
        f"Op breakdown: {summary}\n"
    )

    gap_helpers_msg = ""
    if notimpl_helpers:
        gap_helpers_msg = (
            f"  EXTRA: helper(s) {sorted(notimpl_helpers)!r} currently raise\n"
            f"     NotImplementedError — you MUST rewrite their bodies with\n"
            f"     raw ttnn ops (those are the unimplemented op-NEW gaps).\n"
        )
    fallback_helpers_msg = ""
    if fallback_helpers:
        fallback_helpers_msg = (
            f"  OPTIONAL: helper(s) {sorted(fallback_helpers)!r} carry a\n"
            f"     working native-ttnn path AND a CPU-fallback safety net\n"
            f"     (markers `CONV2D_CPU_FALLBACK` / `ACTIVATION_CPU_FALLBACK`).\n"
            f"     They are FUNCTIONAL as-is — PCC will pass even if the\n"
            f"     native path throws. If you want full on-device execution\n"
            f"     for these ops, you MAY specialize them (e.g. tune\n"
            f"     shard_layout / slice_config for conv2d; replace the\n"
            f"     torch round-trip with raw ttnn ops for activations).\n"
            f"     Leaving them unchanged is acceptable.\n"
        )

    contract_override = (
        f"\n*** OP-SYNTH CONTRACT OVERRIDE FOR THIS COMPONENT ***\n"
        f"This stub was machine-pre-synthesized at the op level. You MUST:\n"
        f"  1. PRESERVE everything except `__call__` and (where noted below)\n"
        f"     specific `_apply_*` helpers. Specifically: keep class\n"
        f"     `{cls_name}` (do not rename), keep `__init__` exactly as\n"
        f"     written (all the `self.w_*` weight loads, all `self._torch_w_*`\n"
        f"     fallback tensors, all `self._conv2d_*_params`), keep\n"
        f"     `build()`, keep `_instance`, keep the module-level\n"
        f"     `{safe}(*args, **kwargs)` shim, keep imports.\n"
        f"  2. REWRITE the body of `__call__` to compute the forward pass\n"
        f"     using `self._apply_*` helpers (listed above) and any\n"
        f"     additional ttnn ops you need (transpose, matmul, softmax,\n"
        f"     scaled_dot_product_attention, reshape, etc.). Do NOT call\n"
        f"     `self._torch_module(...)` or `_coerce_to_torch` in the new\n"
        f"     `__call__`; the whole point is that the forward path is\n"
        f"     now pure ttnn.\n"
        f"  3. `_apply_*` helper-rewrite policy:\n"
        f"{gap_helpers_msg}{fallback_helpers_msg}"
        f"     All OTHER helpers (linear, layer_norm, rms_norm, embedding,\n"
        f"     activations) are deterministic and correct — keep them.\n"
        f"  4. For each op-NEW gap listed in the palette, either implement\n"
        f"     it inline in `__call__` with raw ttnn ops, or add a private\n"
        f"     helper `_apply_<name>` that mirrors the pre-bound ones.\n"
        f"  5. Output is STILL the full module file at `_synth_responses/"
        f"{safe}.py` — copy the existing stub verbatim and replace ONLY\n"
        f"     the `__call__` body, any required `_apply_*` gap helpers,\n"
        f"     and (optionally) the fallback helpers from #3.\n"
    )
    return palette_block, contract_override


def _read_file_excerpt(p: Path, *, max_lines: int = 140) -> str:
    if not p.is_file():
        return ""
    try:
        text = p.read_text(errors="ignore")
    except Exception:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n# ... ({len(lines) - max_lines} more lines truncated)"


_EXEMPLAR_ROLE_HINTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("rotary", ("rotary", "rope")),
    ("upsample", ("convtranspose", "deconv", "upsample")),
    ("downsample", ("downsample", "pooling", "pool")),
    ("backbone", ("backbone", "trunk", "stem")),
    ("neck", ("neck", "fpn", "feature_pyramid", "lateral", "projection")),
    ("head", ("lm_head", "predictor", "classifier", "head")),
    ("fuser", ("fuser", "fusion", "merger")),
    ("attention", ("self_attention", "cross_attention", "multihead", "attention", "attn")),
    ("mlp", ("feed_forward", "feedforward", "mlp", "ffn")),
    ("norm", ("layernorm", "rmsnorm", "groupnorm", "norm")),
    ("embed", ("patch_embed", "token_embed", "embedding", "embed")),
    ("decoder", ("mask_decoder", "decoder_head", "decoder")),
    ("encoder", ("vision_encoder", "image_encoder", "encoder")),
    ("conv", ("convnext", "conv_block", "conv2d", "conv")),
    ("transformer_block", ("transformer_block", "layer", "block")),
]


def _exemplar_role_for(name: str, kind: str) -> Optional[str]:
    needle = f"{name} {kind}".lower()
    for role, _hints in _EXEMPLAR_ROLE_HINTS:
        if role in needle:
            return role
        for hint in _hints:
            if hint in needle:
                return role
    if "attn" in needle or "attention" in needle:
        return "attention"
    if "mlp" in needle or "feed" in needle:
        return "mlp"
    if "norm" in needle:
        return "norm"
    return None


def _find_exemplar(component_name: str, kind: str = "", *, demo_dir: Optional[Path] = None) -> Optional[Path]:
    role = _exemplar_role_for(component_name, kind)
    if role is None:
        return None

    if demo_dir is not None:
        stubs_dir = demo_dir / "_stubs"
        if stubs_dir.is_dir():
            for snap in sorted(stubs_dir.glob("*.py.last_good_native")):
                sibling_name = snap.name[: -len(".py.last_good_native")]
                if sibling_name == _safe_id(component_name):
                    continue
                if _exemplar_role_for(sibling_name, "") == role:
                    return snap

    base = BRINGUP_ROOT() / "models" / "demos"
    if not base.is_dir():
        return None
    hint_words = next((hints for r, hints in _EXEMPLAR_ROLE_HINTS if r == role), ())
    tt_dirs = {"tt", "ttnn"}
    scored: List[Tuple[int, int, Path]] = []
    for p in base.rglob("*.py"):
        try:
            parts_low = {q.lower() for q in p.parts}
        except Exception:
            continue
        if not (parts_low & tt_dirs):
            continue
        name_low = p.name.lower()
        score = 0
        for i, w in enumerate(hint_words):
            if w in name_low:
                score += (len(hint_words) - i) * 10
        if score == 0:
            continue
        try:
            head = p.read_text(errors="ignore")[:8000]
        except Exception:
            continue
        if "import ttnn" not in head and "from ttnn" not in head:
            continue
        if "ttnn.linear" in head or "ttnn.matmul" in head or "ttnn.conv2d" in head or "ttnn.layer_norm" in head:
            score += 5
        scored.append((-score, len(p.parts), p))
    if not scored:
        return None
    scored.sort()
    return scored[0][2]


def _exemplar_block(component_name: str, kind: str = "", *, demo_dir: Optional[Path] = None) -> str:
    p = _find_exemplar(component_name, kind, demo_dir=demo_dir)
    if p is None:
        return "(no exemplar found — write the ttnn port from the torch reference below)"
    rel = safe_relative_to_root(p) if p.is_absolute() else p
    src = _read_file_excerpt(p, max_lines=120)
    if not src:
        return "(no exemplar found — write the ttnn port from the torch reference below)"
    return f"  source: {rel}\n" f"```python\n{src}\n```"


def _torch_ref_summary(stub_path: Path, *, max_forward_lines: int = 80, max_state_dict_entries: int = 40) -> str:
    if not stub_path.is_file():
        return "(stub file not found; cannot resolve torch reference)"
    try:
        import importlib.util
        import inspect as _inspect

        module_name = f"_tt_planner_stubprobe_{stub_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(stub_path))
        if spec is None or spec.loader is None:
            return "(could not load stub module spec)"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "_get_torch_submodule"):
            return "(stub has no `_get_torch_submodule`; not an autofilled or scaffolded stub)"
        torch_module = mod._get_torch_submodule()
        if torch_module is None:
            return "(stub `_get_torch_submodule()` returned None)"
        cls = type(torch_module)
        cls_name = f"{cls.__module__}.{cls.__qualname__}"
        try:
            forward_src = _inspect.getsource(cls.forward)
            fwd_lines = forward_src.splitlines()
            if len(fwd_lines) > max_forward_lines:
                forward_src = (
                    "\n".join(fwd_lines[:max_forward_lines])
                    + f"\n# ... ({len(fwd_lines) - max_forward_lines} more lines truncated)"
                )
        except (TypeError, OSError):
            forward_src = "(forward source unavailable; may be from a C extension)"
        try:
            children: List[str] = []
            for name, child in torch_module.named_children():
                children.append(f"    {name}: {type(child).__name__}")
            children_block = "\n".join(children) if children else "    (no named children)"
        except Exception:
            children_block = "    (could not enumerate children)"
        try:
            sd_items: List[str] = []
            for k, v in torch_module.state_dict().items():
                try:
                    shape = tuple(v.shape)
                except Exception:
                    shape = ("?",)
                sd_items.append(f"    {k}: {shape}")
                if len(sd_items) >= max_state_dict_entries:
                    sd_items.append(f"    ... ({len(torch_module.state_dict()) - max_state_dict_entries} more)")
                    break
            sd_block = "\n".join(sd_items) if sd_items else "    (empty state_dict)"
        except Exception as exc:
            sd_block = f"    (state_dict introspection failed: {exc})"
        return (
            f"  torch reference class: {cls_name}\n"
            f"  named children:\n{children_block}\n"
            f"  state_dict (param name -> shape):\n{sd_block}\n"
            f"  forward() source:\n```python\n{forward_src}\n```"
        )
    except Exception as exc:
        return f"(torch reference introspection failed: {type(exc).__name__}: {exc})"


def _full_hf_reference_source(
    stub_path: Path,
    *,
    max_total_lines: int = 600,
    max_child_lines: int = 80,
    model_id: Optional[str] = None,
    demo_dir: Optional[Path] = None,
    component_name: Optional[str] = None,
) -> str:
    """Emit a richer view of the HF reference for PCC_ONLY failures.

    `_torch_ref_summary` already inlines the top-level `forward()` source
    truncated to 80 lines, which is enough for shape/wiring decisions on
    new ttnn ports. But for PCC near-miss debugging the LLM frequently
    needs the FULL class body (the constructor, member assignments,
    helper methods like `_predict_masks` / `_iou_token_logits`, residual
    paths, attention scaling factors, etc.) AND a few of the most
    important named-children classes' `forward()` so it can verify the
    exact computation against the ttnn translation.

    Resolution strategy mirrors `activation_diff._resolve_torch_module`:
    try the stub's `_get_torch_submodule()` first (older / autofilled
    stubs), then fall back to `_CANDIDATE_SUBMODULE_PATHS` + HF AutoModel
    (op-synth partial stubs), then to the per-component cli-level
    resolver if `model_id` + `demo_dir` + `component_name` are provided.

    Truncation budget is shared across the parent class + each child to
    keep the prompt bounded; once `max_total_lines` is hit, remaining
    children are listed by name only.

    Returns "" if introspection fails for any reason (the caller falls
    back to the existing `_torch_ref_summary` block alone).
    """
    if not stub_path.is_file():
        return ""
    try:
        import importlib.util
        import inspect as _inspect

        module_name = f"_tt_planner_hfsrc_{stub_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(stub_path))
        if spec is None or spec.loader is None:
            return ""
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        torch_module = None

        getter = getattr(mod, "_get_torch_submodule", None)
        if getter is not None:
            try:
                torch_module = getter()
            except Exception:
                torch_module = None

        if torch_module is None:
            try:
                from . import activation_diff as _act_diff

                torch_module = _act_diff._resolve_torch_module_from_candidates(
                    mod,
                    demo_dir or stub_path.parent.parent,
                    component_name or stub_path.stem,
                    lambda _msg: None,
                )
            except Exception:
                torch_module = None

        if torch_module is None and model_id and demo_dir and component_name:
            try:
                torch_module = _resolve_torch_submodule_for_component(model_id, demo_dir, component_name)
            except Exception:
                torch_module = None
        if torch_module is None:
            return ""
        cls = type(torch_module)
        cls_name = f"{cls.__module__}.{cls.__qualname__}"

        try:
            cls_src = _inspect.getsource(cls)
        except (TypeError, OSError):
            cls_src = ""
        cls_lines = cls_src.splitlines() if cls_src else []
        if len(cls_lines) > max_total_lines:
            head = cls_lines[:max_total_lines]
            cls_src = "\n".join(head) + (
                f"\n# ... ({len(cls_lines) - max_total_lines} more lines truncated; "
                "see full source via inspect.getsource at runtime)"
            )
        used_lines = min(len(cls_lines), max_total_lines)

        child_blocks: List[str] = []
        children_listed = 0
        deferred_children: List[str] = []
        try:
            for name, child in torch_module.named_children():
                if used_lines >= max_total_lines:
                    deferred_children.append(f"{name}: {type(child).__name__}")
                    continue
                child_cls = type(child)
                child_cls_name = f"{child_cls.__module__}.{child_cls.__qualname__}"
                try:
                    child_src = _inspect.getsource(child_cls.forward)
                except (TypeError, OSError, AttributeError):
                    deferred_children.append(f"{name}: {child_cls_name} (forward source unavailable)")
                    continue
                child_lines = child_src.splitlines()
                budget = max(0, min(max_child_lines, max_total_lines - used_lines))
                if budget <= 0:
                    deferred_children.append(f"{name}: {child_cls_name}")
                    continue
                if len(child_lines) > budget:
                    child_src = (
                        "\n".join(child_lines[:budget]) + f"\n# ... ({len(child_lines) - budget} more lines truncated)"
                    )
                child_blocks.append(
                    f"--- child `{name}` ({child_cls_name}) forward() ---\n" f"```python\n{child_src}\n```"
                )
                used_lines += min(len(child_lines), budget)
                children_listed += 1
        except Exception:
            pass

        parts: List[str] = []
        parts.append(f"FULL HF REFERENCE for PCC debugging (root: {cls_name}):\n" f"```python\n{cls_src}\n```")
        for block in child_blocks:
            parts.append(block)
        if deferred_children:
            parts.append(
                "additional children (source omitted to keep prompt bounded):\n"
                + "\n".join(f"  - {x}" for x in deferred_children)
            )
        return "\n\n".join(parts)
    except Exception:
        return ""


def _resolve_torch_submodule_for_component(model_id: str, demo_dir: Path, component_name: str) -> Optional[object]:
    """Load the HF model and resolve the torch submodule for `component_name`
    using the candidate-paths list embedded in the corresponding pytest test.

    Falls back to introspecting the autofilled stub's `_get_torch_submodule()`
    if the test file doesn't expose `_CANDIDATE_SUBMODULE_PATHS`. Returns None
    if resolution fails."""
    safe = _safe_id(component_name)
    test_path = demo_dir / "tests" / "pcc" / f"test_{safe}.py"
    candidate_paths: List[str] = []
    if test_path.is_file():
        try:
            import ast as _ast

            tree = _ast.parse(test_path.read_text(errors="ignore"))
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, _ast.Name) and tgt.id == "_CANDIDATE_SUBMODULE_PATHS":
                            try:
                                val = _ast.literal_eval(node.value)
                                if isinstance(val, (list, tuple)):
                                    candidate_paths = [str(x) for x in val]
                            except Exception:
                                pass
        except Exception:
            candidate_paths = []

    try:
        import transformers

        model = transformers.AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="bfloat16", low_cpu_mem_usage=True
        )
        model.eval()
    except Exception:
        model = None

    from .module_tree import resolve_dotted as _resolve

    if model is not None:
        for path in candidate_paths:
            try:
                sub = _resolve(model, path)
                if sub is not None:
                    return sub
            except (AttributeError, IndexError, KeyError, TypeError):
                continue

    stub_path = demo_dir / "_stubs" / f"{safe}.py"
    if stub_path.is_file():
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(f"_tt_planner_resolveprobe_{stub_path.stem}", str(stub_path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                getter = getattr(mod, "_get_torch_submodule", None)
                if getter is not None:
                    return getter()
        except Exception:
            return None
    return None


def _numerical_constraints_block(
    stub_path: Path,
    *,
    model_id: Optional[str] = None,
    demo_dir: Optional[Path] = None,
    component_name: Optional[str] = None,
) -> str:
    """Introspect the resolved torch submodule and emit a CHECKLIST of concrete
    numerical hardware constraints (tile alignment, divisibility, etc.) the
    LLM must satisfy.

    Tries `_get_torch_submodule()` from the (autofilled) stub first; if that's
    been replaced by native TTNN code, falls back to resolving via the HF
    model + test's `_CANDIDATE_SUBMODULE_PATHS`. Constraints come from actual
    weight shapes / module attributes, NOT from generic templates. Each line
    ends with PASS / FAIL and (on FAIL) a concrete suggested fix. This is the
    most important block in the prompt for HANG avoidance — device-side
    deadlocks are almost always tile-misaligned matmul / linear / softmax."""

    TILE = 32
    torch_module = None
    if stub_path.is_file():
        try:
            import importlib.util

            module_name = f"_tt_planner_constraintprobe_{stub_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, str(stub_path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                getter = getattr(mod, "_get_torch_submodule", None)
                if getter is not None:
                    torch_module = getter()
        except Exception:
            torch_module = None

    if torch_module is None and model_id and demo_dir and component_name:
        torch_module = _resolve_torch_submodule_for_component(model_id, demo_dir, component_name)

    if torch_module is None:
        return "(could not resolve torch submodule for this component; " "no numerical constraints derived)"

    sd: Dict[str, object] = {}
    try:
        sd = dict(torch_module.state_dict())
    except Exception:
        sd = {}

    lines: List[str] = []
    fixes: List[str] = []

    def shape(t) -> Tuple[int, ...]:
        try:
            return tuple(int(x) for x in t.shape)
        except Exception:
            return ()

    def tile_check(label: str, value: int) -> str:
        ok = value > 0 and value % TILE == 0
        mark = "[ ok ]" if ok else "[FAIL]"
        if not ok:
            fixes.append(
                f"    {label}={value} is NOT a multiple of {TILE}. ttnn.matmul / "
                f"ttnn.linear / ttnn.softmax in TILE_LAYOUT will likely deadlock "
                f"the device on this dim. Pad to {((value // TILE) + 1) * TILE} "
                f"OR pick a different factorisation."
            )
        return f"  {mark} {label} = {value}    (tile alignment: {value} % {TILE} == {value % TILE if value else '?'})"

    attention_keys = [
        ("qkv.weight", "qkv.bias", "fused QKV"),
        ("q_proj.weight", "q_proj.bias", "separate q_proj"),
        ("query.weight", "query.bias", "BERT-style query"),
        ("query_key_value.weight", "query_key_value.bias", "fused QKV (HF)"),
    ]
    embed_dim: Optional[int] = None
    head_dim: Optional[int] = None
    num_heads: Optional[int] = None
    attention_kind: Optional[str] = None
    for w_key, b_key, label in attention_keys:
        if w_key in sd:
            w_shape = shape(sd[w_key])
            if w_shape and len(w_shape) >= 2:
                if "qkv" in w_key or "query_key_value" in w_key:
                    out_dim = w_shape[0]
                    embed_dim_local = out_dim // 3 if out_dim % 3 == 0 else None
                else:
                    embed_dim_local = w_shape[0]
                if embed_dim_local:
                    embed_dim = embed_dim_local
                    attention_kind = label
                    break

    if embed_dim is not None:
        for attr in ("num_heads", "num_attention_heads", "n_head", "n_heads", "heads"):
            v = getattr(torch_module, attr, None)
            if isinstance(v, int) and v > 0 and embed_dim % v == 0:
                num_heads = v
                break
        if num_heads is None:
            for candidate in (12, 8, 16, 4, 6, 3, 2, 1):
                if embed_dim % candidate == 0 and (embed_dim // candidate) >= TILE:
                    num_heads = candidate
                    break
            if num_heads is None and embed_dim % TILE == 0:
                num_heads = max(1, embed_dim // TILE)
            elif num_heads is None:
                num_heads = 1

        head_dim = embed_dim // num_heads if num_heads else embed_dim
        lines.append(f"  ATTENTION detected ({attention_kind}):")
        lines.append(f"    embed_dim    = {embed_dim}")
        lines.append(f"    num_heads    = {num_heads}  (from torch_module attribute or factorisation)")
        lines.append(f"    head_dim     = {head_dim}  (= embed_dim / num_heads)")
        lines.append("")
        lines.append(tile_check("embed_dim", embed_dim))
        lines.append(tile_check("head_dim", head_dim))

    mlp_pairs = [
        ("fc1.weight", "fc2.weight", "fc1/fc2"),
        ("up_proj.weight", "down_proj.weight", "up/down_proj"),
        ("intermediate.dense.weight", "output.dense.weight", "BERT-style intermediate/output"),
        ("dense_h_to_4h.weight", "dense_4h_to_h.weight", "GPT-style h<->4h"),
        ("c_fc.weight", "c_proj.weight", "GPT-2 style c_fc/c_proj"),
    ]
    for w_in_key, w_out_key, label in mlp_pairs:
        if w_in_key in sd:
            w_in_shape = shape(sd[w_in_key])
            if len(w_in_shape) >= 2:
                hidden_in = w_in_shape[1]
                intermediate = w_in_shape[0]
                lines.append("")
                lines.append(f"  MLP detected ({label}):")
                lines.append(f"    hidden_in        = {hidden_in}")
                lines.append(f"    intermediate_dim = {intermediate}")
                lines.append("")
                lines.append(tile_check("hidden_in", hidden_in))
                lines.append(tile_check("intermediate_dim", intermediate))
                break

    for k, v in sd.items():
        w_shape = shape(v)
        if len(w_shape) == 4 and k.endswith(".weight"):
            out_ch, in_ch, kh, kw = w_shape
            lines.append("")
            lines.append(f"  CONV2D detected (`{k}`):")
            lines.append(f"    out_channels = {out_ch}")
            lines.append(f"    in_channels  = {in_ch}")
            lines.append(f"    kernel       = ({kh}, {kw})")
            lines.append("")
            in_ok = in_ch > 0 and in_ch % TILE == 0
            out_ok = out_ch > 0 and out_ch % TILE == 0
            lines.append(f"  {'[ ok ]' if in_ok else '[note]'} in_channels  = {in_ch}    (tile-aligned: {in_ok})")
            lines.append(f"  {'[ ok ]' if out_ok else '[note]'} out_channels = {out_ch}    (tile-aligned: {out_ok})")
            if not in_ok:
                fixes.append(
                    f"    in_channels={in_ch} is NOT a multiple of {TILE}. For ttnn.conv2d this "
                    f"is OK on the FIRST conv ONLY (input projection from images); use "
                    f"`input_channels_alignment` in conv2d kwargs and rely on the auto-shard "
                    f"path. Do NOT call ttnn.matmul on this dim — use ttnn.conv2d."
                )
            if not out_ok:
                fixes.append(
                    f"    out_channels={out_ch} is NOT a multiple of {TILE}. Output projection "
                    f"will need padding to {((out_ch // TILE) + 1) * TILE} OR a non-tile output "
                    f"layout. Verify on conv2d directly; never feed this into ttnn.matmul."
                )
            if kh * kw > 49:
                fixes.append(
                    f"    Large kernel ({kh}x{kw}) on conv2d frequently causes L1 OOM on the "
                    f"first conv. Plan on halo-sharded conv with explicit `slice_config`."
                )
            break

    if not lines:
        lines.append(
            "  (no attention/MLP/conv2d weight pattern recognised; no tile-alignment "
            "checks emitted — fall back to generic ttnn op constraints)"
        )

    out = ["NUMERICAL HARDWARE CONSTRAINTS (computed from this component's actual torch state_dict):"]
    out.extend(lines)
    if fixes:
        out.append("")
        out.append("  SUGGESTED FIXES (act on these BEFORE writing the stub):")
        out.extend(fixes)
    else:
        out.append("")
        out.append(
            "  All recognised dims are tile-aligned. You should NOT need to pad. "
            "If you still see a HANG, the deadlock is in a permute/reshape layout "
            "transition, not in dim alignment."
        )
    return "\n".join(out)


def _stub_forward_body_excerpt(stub_path: Path, *, max_lines: int = 30) -> str:
    if not stub_path.is_file():
        return "(stub file not found)"
    try:
        text = stub_path.read_text(errors="ignore")
    except Exception:
        return "(stub unreadable)"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"\s*def\s+(__call__|forward|__forward__)\s*\(", line):
            end = min(len(lines), idx + max_lines)
            return "\n".join(lines[idx:end])
    return "(no __call__/forward found)"


def _ungraduated_breakdown(demo_dir: Path, components: List[str]) -> str:
    if not components:
        return ""
    verbose = os.environ.get("TT_HW_PLANNER_VERBOSE", "") not in ("", "0", "false", "False")
    lines: List[str] = []
    for comp in components:
        safe = _safe_id(comp)
        stub_path = demo_dir / "_stubs" / f"{safe}.py"
        wrapper = "TORCH WRAPPER" if _stub_uses_torch_wrapper(stub_path) else "AUTOFILL STUB"
        lines.append(f"  - {comp}  [{wrapper}]  ({stub_path})")
        if verbose:
            body = _stub_forward_body_excerpt(stub_path, max_lines=8)
            for bl in body.splitlines():
                lines.append(f"      | {bl}")
    return "\n".join(lines)


def _refinement_directive(tt_reuse_target: str, pcc_value: Optional[float] = None) -> str:
    """Directive for ADAPT components — the canonical TT impl already exists
    and the stub WRAPS it. The LLM's job is to REFINE config/args, NOT to
    rewrite the class or implement ttnn ops from scratch.

    Used when ``component.status == "ADAPT"`` (force_adapt_all-demoted or
    registry-mapped with SUPPORTED+!DROP_IN). Distinct from
    :func:`_native_directive` which targets NEW components (write from
    scratch).
    """
    pcc_str = f" (current per-component PCC = {pcc_value:.4f})" if pcc_value is not None else ""
    return (
        "\nREFINEMENT TARGET (ADAPT — read carefully):\n"
        f"  The canonical TT implementation at `{tt_reuse_target}` ALREADY "
        f"EXISTS and works for sibling models (Llama, Qwen3, etc.). The "
        f"current stub WRAPS it.\n"
        f"\n"
        f"  The per-component PCC test was run{pcc_str}.\n"
        "  The bug is NOT in the canonical class — the bug is in HOW THIS\n"
        "  stub WIRES it for this model's specifics (wrong ModelArgs config,\n"
        "  wrong constructor args, missing tt_ccl / transformation_mats /\n"
        "  configuration, wrong dtype, etc.).\n"
        "\n"
        "INVESTIGATION ORDER — DO NOT read the canonical file end-to-end:\n"
        f"  1. FIRST, grep the canonical file for its `__init__` signature\n"
        f"     so you know what kwargs it accepts:\n"
        f'        Bash: `grep -nE "^    def __init__" {tt_reuse_target}`\n'
        f"     Read the matched lines + the next ~20 lines (the args + their\n"
        f"     immediate usage). DO NOT read the whole file.\n"
        f"  2. If the failure is a `KeyError` on a state_dict key, grep for\n"
        f"     where the canonical reads state_dict:\n"
        f'        Bash: `grep -nE "state_dict\\\\[" {tt_reuse_target}`\n'
        f"     This reveals the EXACT key format the canonical expects\n"
        f"     (typically Llama/Meta-style: `layers.N.<module>.wq.weight`,\n"
        f"     `feed_forward.w1.weight`, etc.). HF naming (`q_proj`,\n"
        f"     `gate_proj`, etc.) must be REMAPPED before passing as\n"
        f"     `state_dict=` — see `models/tt_transformers/tt/load_checkpoints.py`\n"
        f"     for `convert_hf_to_meta(...)`.\n"
        f"  3. If the failure is `unexpected keyword argument 'X'`, the\n"
        f"     canonical does NOT accept `X` — remove it from the call.\n"
        f"     Some components (e.g. `RotaryEmbedding`) take a totally\n"
        f"     different signature `(dim, max_position_embeddings, base,\n"
        f"     device)` with no `mesh_device`/`args`/`state_dict` at all.\n"
        f"  4. ONLY THEN, write the fix to `_synth_responses/<target>.py`.\n"
        f"     Edits should be 1-15 lines.\n"
        "\n"
        "WRITE-FIRST DISCIPLINE: spending more than ~2 minutes reading\n"
        "before writing means you are exploring instead of refining. If\n"
        "the prior-attempts log includes a `prior agent said:` field, that\n"
        "agent's diff already partially fixed the wiring — DO NOT redo it,\n"
        "build on it.\n"
        "\n"
        "ABSOLUTELY FORBIDDEN:\n"
        f"  - Do NOT write a new class to replace `{tt_reuse_target}`'s impl.\n"
        f"  - Do NOT replace the `from {tt_reuse_target}` import with your own ttnn ops.\n"
        "  - Do NOT bypass the canonical impl by re-implementing forward.\n"
        "  - Do NOT delegate to torch (no `_torch_module`, no `_get_torch_submodule`,\n"
        "    no `transformers.AutoModel.from_pretrained`, no submodule walking).\n"
        "  - Do NOT rewrite `__init__` / `build` / `__call__` to NOT delegate to `self._impl`.\n"
        "\n"
        "  The loop WILL reject your stub if it detects the class being\n"
        "  rewritten instead of refined. Convergence comes from small\n"
        "  config-level edits, not from new code.\n"
    )


def _native_directive(forbidden_excerpt: str = "", *, strict_native: bool = False) -> str:
    if strict_native:
        base = (
            "\nBRING-UP TARGET (READ CAREFULLY):\n"
            "  The PCC test contract is `pcc(ttnn_native(x), torch_ref(x)) >= 0.99`. "
            "It is meaningful ONLY when the stub computes with ttnn ops on the "
            "device. If the stub delegates to `self._torch_module(x)` on CPU and "
            "roundtrips through ttnn, the PCC test becomes `pcc(torch_ref(x), "
            "torch_ref(x)_after_bf16_roundtrip)`, which trivially passes and "
            "validates nothing about TTNN. A PASSED test under that pattern is "
            "NOT bring-up — it is a measurement of bfloat16 quantization error.\n"
            "  The loop WILL reject your stub and re-prompt you if it finds the "
            "patterns below.\n"
            "\nMANDATORY CONSTRAINTS for each `_stubs/<comp>.py` you write:\n"
            "  1. Inputs are ttnn tensors on device, outputs must be ttnn tensors on device.\n"
            "  2. Implement the math with `ttnn.*` ops directly (ttnn.linear, ttnn.matmul, "
            "ttnn.layer_norm, ttnn.conv2d, ttnn.silu, ttnn.softmax, ttnn.add, ttnn.mul, etc.).\n"
            "  3. `build(device, torch_module)` extracts weights via `torch_module.state_dict()` "
            "and materializes them onto the device via `ttnn.from_torch(..., layout=ttnn.TILE_LAYOUT)`.\n"
            "  4. Preserve the class signature + `build()` + module-level shim that the scaffold provides.\n"
            "  5. PCC must be >= 0.99 against the torch reference.\n"
            "\nFORBIDDEN PATTERNS (the loop will reject your stub if it finds any of these):\n"
            "  - `self._torch_module(...)` or `self.torch_module(...)` inside `__call__` / `forward` / `__forward__`\n"
            "  - `_get_torch_submodule(...)` inside `__call__` / `forward` / `__forward__`\n"
            "  - `with torch.no_grad():` around any call to a torch.nn module in the forward path\n"
            "  - converting ttnn -> torch -> compute on CPU -> ttnn\n"
        )
        if forbidden_excerpt:
            base += (
                "\nEXAMPLES of the WRONG pattern present in the current stub(s); rewrite to remove these:\n"
                f"{forbidden_excerpt}\n"
            )
        return base
    return (
        "\nBRING-UP TARGET (default mode):\n"
        "  Goal: every PCC test passes on TT hardware. NATIVE ttnn ops are "
        "preferred. A CPU-fallback wrapper (delegating to `self._torch_module` "
        "and roundtripping through ttnn) is acceptable as a last resort if "
        "the native path fails — the user can promote it to native later via "
        "`tt_hw_planner promote ... --strict-native`.\n"
        "\nGuidance for each `_stubs/<comp>.py` you write:\n"
        "  1. Inputs are ttnn tensors on device, outputs must be ttnn tensors on device.\n"
        "  2. Prefer `ttnn.*` ops directly (ttnn.linear, ttnn.matmul, ttnn.layer_norm, "
        "ttnn.conv2d, ttnn.silu, ttnn.softmax, ttnn.add, ttnn.mul). Only use a torch-ref "
        "wrapper if you cannot make the native path pass PCC in this iteration.\n"
        "  3. `build(device, torch_module)` extracts weights via `torch_module.state_dict()`.\n"
        "  4. Preserve the class signature + `build()` + module-level shim that the scaffold provides.\n"
        "  5. PCC must be >= 0.99 against the torch reference.\n"
    )


from ._cli_helpers.auto_iterate import (  # noqa: F401
    add_iter_loop_cli_args,
)

_DEVICE_RESET_COUNT: int = 0
_DEVICE_RESET_MAX_PER_PROCESS: int = 3


def _run_tt_smi_reset(
    *,
    devices: str = "0,1,2,3",
    timeout_s: int = 120,
    force: bool = False,
    context: str = "",
) -> bool:
    """Execute `tt-smi -r <devices>` to clear stale IOMMU / sysmem / NOC
    mappings on the TT cards.

    Safety:
      * Hard-capped at `_DEVICE_RESET_MAX_PER_PROCESS` resets per planner
        process so a recurring failure cannot drive an infinite reset loop.
      * Skipped entirely if env `TT_PLANNER_NO_DEVICE_RESET` is truthy.
      * `force=True` overrides the per-process cap (still subject to the
        env-var override).

    Returns True iff `tt-smi -r` exited cleanly.
    """
    global _DEVICE_RESET_COUNT
    if os.environ.get("TT_PLANNER_NO_DEVICE_RESET", "").lower() in {"1", "true", "yes", "on"}:
        print("  TT_PLANNER_NO_DEVICE_RESET set in env; skipping `tt-smi -r`", file=sys.stderr)
        return False
    if not force and _DEVICE_RESET_COUNT >= _DEVICE_RESET_MAX_PER_PROCESS:
        print(
            f"  device-reset budget exhausted "
            f"({_DEVICE_RESET_COUNT}/{_DEVICE_RESET_MAX_PER_PROCESS}); "
            f"skipping further `tt-smi -r` calls in this process",
            file=sys.stderr,
        )
        return False
    import shutil

    if shutil.which("tt-smi") is None:
        print("  `tt-smi` not on PATH; cannot auto-reset device", file=sys.stderr)
        return False
    import subprocess as _sp

    label = f" [{context}]" if context else ""
    print()
    print("=" * 78)
    print(f"  Auto device-reset{label}: tt-smi -r {devices} (timeout={timeout_s}s)")
    print("=" * 78)
    try:
        proc = _sp.run(
            ["tt-smi", "-r", devices],
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except _sp.TimeoutExpired:
        print(f"  tt-smi -r timed out after {timeout_s}s", file=sys.stderr)
        return False
    except OSError as exc:
        print(f"  failed to launch tt-smi: {exc}", file=sys.stderr)
        return False
    _DEVICE_RESET_COUNT += 1
    tail = (proc.stdout or "") + (proc.stderr or "")
    tail = tail.strip()
    if tail:
        print(tail[-1200:])
    if proc.returncode != 0:
        print(f"  tt-smi -r exited rc={proc.returncode}", file=sys.stderr)
        return False
    print(f"  tt-smi -r completed cleanly (reset #{_DEVICE_RESET_COUNT}/{_DEVICE_RESET_MAX_PER_PROCESS})")
    return True


def _find_stale_pytest_processes(repo_root: Path) -> List[Tuple[int, str, int]]:
    """Return (pid, argv, elapsed_seconds) tuples for pytest processes that
    look like they belong to tt_hw_planner (run a model PCC test under our
    repo) but are NOT descendants of the current process.

    Used to detect orphaned pytest from a previously-aborted shell session
    that's still holding the TT device lock (`CHIP_IN_USE_0_PCIe`)."""
    our_pid = os.getpid()
    try:
        our_uid = os.getuid()
    except AttributeError:
        return []

    def _is_descendant(pid: int, root_pid: int) -> bool:
        cur = pid
        for _ in range(64):
            if cur == root_pid:
                return True
            if cur in (0, 1):
                return False
            try:
                with open(f"/proc/{cur}/status") as f:
                    ppid = None
                    for line in f:
                        if line.startswith("PPid:"):
                            ppid = int(line.split()[1])
                            break
                if ppid is None or ppid == cur:
                    return False
                cur = ppid
            except (FileNotFoundError, PermissionError, ValueError):
                return False
        return False

    try:
        proc_pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return []

    repo_root_s = str(repo_root)
    candidates: List[Tuple[int, str, int]] = []
    try:
        with open("/proc/uptime") as f:
            sys_uptime = float(f.read().split()[0])
    except (OSError, ValueError):
        sys_uptime = 0.0
    try:
        clock_hz = os.sysconf("SC_CLK_TCK")
    except (AttributeError, OSError):
        clock_hz = 100

    for pid in proc_pids:
        if pid == our_pid:
            continue
        try:
            with open(f"/proc/{pid}/status") as f:
                uid_line = next((l for l in f if l.startswith("Uid:")), None)
            if uid_line is None or int(uid_line.split()[1]) != our_uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="ignore").replace("\x00", " ").strip()
            if not cmdline or "pytest" not in cmdline:
                continue
            if (repo_root_s not in cmdline) and ("models/demos/" not in cmdline):
                continue
            if "tests/pcc/" not in cmdline:
                continue
            if _is_descendant(pid, our_pid):
                continue
            elapsed_s = 0
            try:
                with open(f"/proc/{pid}/stat") as f:
                    fields = f.read().split()
                start_ticks = int(fields[21])
                start_s = start_ticks / clock_hz
                elapsed_s = max(0, int(sys_uptime - start_s))
            except (FileNotFoundError, PermissionError, ValueError, IndexError):
                elapsed_s = 0
            candidates.append((pid, cmdline, elapsed_s))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return candidates


def _kill_stale_pytest_processes(stale: List[Tuple[int, str, int]]) -> int:
    """SIGTERM the process group of each stale pid; escalate to SIGKILL if
    still alive after 5s. Returns the number actually reaped."""
    import signal as _signal
    import time as _time

    killed = 0
    for pid, _cmdline, _elapsed in stale:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            killed += 1
            continue
        try:
            os.killpg(pgid, _signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as exc:
            print(f"  could not SIGTERM PGID {pgid}: {exc}", file=sys.stderr)
            continue
        alive = True
        for _ in range(50):
            try:
                os.kill(pid, 0)
                _time.sleep(0.1)
            except ProcessLookupError:
                alive = False
                break
        if not alive:
            killed += 1
            continue
        try:
            os.killpg(pgid, _signal.SIGKILL)
            _time.sleep(0.5)
        except (ProcessLookupError, PermissionError) as exc:
            print(f"  could not SIGKILL PGID {pgid}: {exc}", file=sys.stderr)
            continue
        try:
            os.kill(pid, 0)
            print(f"  PID {pid} still alive after SIGKILL — TT device may still be locked", file=sys.stderr)
        except ProcessLookupError:
            killed += 1
    return killed


def _preflight_cleanup_stale_pytest(
    repo_root: Path, *, allow_kill: bool = True, allow_device_reset: bool = True, context: str = "pre-flight"
) -> int:
    """Print and (optionally) kill stale TT-device-holding pytest processes.
    When orphans are actually reaped AND `allow_device_reset` is true, also
    execute `tt-smi -r` since the killed process's kernel-side IOMMU / sysmem
    mappings can remain pinned and cause the next pytest to crash with
    `Proceeding could lead to undefined behavior`.

    Returns the number of stale processes that were detected. Call this:
      - once at the very start of `up` (catches orphans from prior aborted runs),
      - before each focused pytest invocation (catches orphans across iterations).
    """
    stale = _find_stale_pytest_processes(repo_root)
    if not stale:
        return 0
    print()
    print("=" * 78)
    print(f"  [{context}] {len(stale)} stale pytest process(es) detected that may hold the TT device:")
    print("=" * 78)
    for pid, cmdline, elapsed in stale:
        excerpt = cmdline if len(cmdline) < 160 else cmdline[:157] + "..."
        m, s = divmod(elapsed, 60)
        print(f"    PID {pid}  elapsed={m:02d}:{s:02d}")
        print(f"      argv: {excerpt}")
    if not allow_kill:
        print()
        print("  --no-kill-stale set: leaving them alone. New runs will likely block on")
        print("  'CHIP_IN_USE_0_PCIe'. Kill manually with:")
        for pid, _, _ in stale:
            print(f"    kill -TERM -- -$(ps -o pgid= -p {pid} | tr -d ' ')")
        print()
        return len(stale)
    print()
    print("  Killing process group(s) to release the device lock ...")
    killed = _kill_stale_pytest_processes(stale)
    print(f"  Reaped {killed}/{len(stale)} stale process(es).")
    print()
    if killed > 0 and allow_device_reset:
        print(
            "  Orphans were holding the TT device — their kernel-side IOMMU/sysmem "
            "mappings can persist after kill. Running `tt-smi -r` proactively so the "
            "next pytest doesn't hit `Proceeding could lead to undefined behavior`."
        )
        _run_tt_smi_reset(context=f"{context}:post-orphan-kill")
    elif killed > 0:
        print(
            "  --no-device-reset (or TT_PLANNER_NO_DEVICE_RESET): skipping `tt-smi -r`. "
            "The next pytest may crash inside `pin_or_map_sysmem_to_device`; if it does, "
            "run `tt-smi -r 0,1,2,3` manually.",
            file=sys.stderr,
        )
    return len(stale)


def _post_escalation_bypass(args: argparse.Namespace) -> bool:
    """Return True when :func:`_maybe_escalate_pcc_fail` has re-entered
    ``cmd_up`` and the ALREADY-SUPPORTED classification should be
    ignored so the newly-drafted backend's scaffold step actually
    runs.

    Without this guard, ``closest_supported_model()`` can still match
    the old LLM lineage on the recursive call and route back to the
    same broken fast-path demo that triggered the escalation.

    Lives at module scope so the inline expression in :func:`cmd_up`
    stays short (the structural invariant
    ``test_cmd_up_auto_routes_already_supported_to_prepare_execute``
    requires ``cmd_scaffold(`` to appear within the first 40000 chars
    of ``cmd_up``'s source).
    """
    if not getattr(args, "_escalated_already", False):
        return False
    print(
        "  [post-escalation] ignoring ALREADY-SUPPORTED match so the "
        "newly-onboarded backend can scaffold its own demo."
    )
    return True


def _maybe_escalate_pcc_fail(
    args: argparse.Namespace,
    model_id: str,
    original_rc: int,
    auto_mode: bool,
) -> Optional[int]:
    """Handle the ALREADY-SUPPORTED -> PCC-fail escalation hook.

    When ``--auto`` is set and ``--escalate-on-pcc-fail`` is enabled
    (the default), invoke ``cmd_auto_onboard --accept`` to draft a
    new ``FamilyBackend`` for ``model_id`` and then re-enter
    :func:`cmd_up` so the scaffold + per-component PCC>=0.99 iterate
    loop runs. The sentinel attribute ``_escalated_already`` on
    ``args`` blocks infinite recursion if the freshly-drafted
    backend also routes ALREADY-SUPPORTED and fails the gate.

    Returns:
        * ``None`` if no escalation happened (caller should fall
          through with ``original_rc``).
        * The rc of the re-invoked ``cmd_up`` if the escalation ran.
        * ``original_rc`` if escalation was attempted but the
          auto-onboard step or the recursive cmd_up call failed.

    Lives at module scope (rather than inline in :func:`cmd_up`) so
    the ``test_cmd_up_auto_routes_already_supported_to_prepare_execute``
    invariant's 40000-char source-window check still finds
    ``cmd_scaffold(`` within range.
    """
    can_escalate = (
        auto_mode and getattr(args, "escalate_on_pcc_fail", True) and not getattr(args, "_escalated_already", False)
    )
    if not can_escalate:
        return None
    sep_esc = "=" * 78
    print()
    print(sep_esc)
    print(f"  ESCALATING on PCC fail  model={model_id}")
    print(sep_esc)

    # 2026-05-31 SHORT-CIRCUIT (audit recommendation): if a FamilyBackend
    # already exists for this model with "exact" match quality
    # (ALREADY-SUPPORTED case), skip cmd_auto_onboard entirely. The
    # backend is already mapped; we just need to bypass the
    # ALREADY-SUPPORTED detection on re-entry so cmd_up takes the
    # scaffold + per-component iterate path (SAM2's pattern). Saves
    # the cost of re-drafting a backend that already exists, and avoids
    # the risk of overwriting it.
    try:
        from .family_backends import pick_backend_with_quality
        from .probe import probe_model

        _p_es = probe_model(model_id)
        _be_es, _q_es = pick_backend_with_quality(
            category=getattr(_p_es, "category", None),
            model_type=(_p_es.raw_config or {}).get("model_type") if _p_es.raw_config else None,
            pipeline_tag=getattr(_p_es, "pipeline_tag", None),
        )
        if _be_es is not None and _q_es == "exact":
            print(
                "  Backend ALREADY EXISTS with exact match for this "
                "model — skipping cmd_auto_onboard (no need to draft a "
                "new backend). Re-entering cmd_up with the "
                "ALREADY-SUPPORTED bypass so scaffold + per-component "
                "iterate (Path 1) takes over."
            )
            print(sep_esc)
            setattr(args, "_escalated_already", True)
            try:
                return cmd_up(args)
            except SystemExit as exc:
                return int(exc.code) if exc.code is not None else original_rc
            except Exception as exc:
                print(
                    f"  short-circuit re-entry failed: "
                    f"{type(exc).__name__}: {exc}. Falling back to "
                    f"original rc={original_rc}."
                )
                return original_rc
    except Exception as _sc_exc:
        print(
            f"  short-circuit check non-fatal: "
            f"{type(_sc_exc).__name__}: {_sc_exc}. Falling back to "
            f"original auto-onboard path."
        )

    print(
        "  The ALREADY-SUPPORTED routing produced output the PCC "
        "gate rejected. Drafting a NEW backend via auto-onboard "
        "and re-invoking `up` so the scaffold + per-component "
        "iterate loop runs."
    )
    print(sep_esc)
    ao_args = argparse.Namespace(
        model_id=model_id,
        agent_bin=getattr(args, "auto_agent_bin", None) or "claude",
        auto_model=getattr(args, "auto_model", None) or "sonnet",
        timeout_s=getattr(args, "auto_agent_timeout", 1500) or 1500,
        skip_llm=False,
        accept=True,
    )
    try:
        ao_rc = cmd_auto_onboard(ao_args)
    except SystemExit as exc:
        ao_rc = int(exc.code) if exc.code is not None else 2
    except Exception as exc:
        print(f"  auto-onboard raised: {type(exc).__name__}: {exc}. " f"Falling back to rc={original_rc}.")
        ao_rc = 2
    if ao_rc != 0:
        print(f"  auto-onboard exit={ao_rc}; cannot escalate further. " f"Returning original rc={original_rc}.")
        return original_rc

    try:
        import importlib
        from scripts.tt_hw_planner import family_backends as _fb_mod
        from scripts.tt_hw_planner import compatibility as _compat_mod

        importlib.reload(_fb_mod)
        importlib.reload(_compat_mod)
    except Exception as exc:
        print(
            f"  WARNING: failed to reload registries after auto-onboard "
            f"({type(exc).__name__}: {exc}); recursive cmd_up may "
            f"still see the stale backend list."
        )
    setattr(args, "_escalated_already", True)
    try:
        return cmd_up(args)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else original_rc
    except Exception as exc:
        print(
            f"  re-invocation of cmd_up after auto-onboard raised: "
            f"{type(exc).__name__}: {exc}. Returning original "
            f"rc={original_rc}."
        )
        return original_rc


# Suffixes for session-scoped working-state files that auto_iterate.py
# uses INSIDE a single `up` run (rollback snapshots, pre-iter floors,
# .bak backups). Capturing these into a per-model overlay leaks
# session-local state across runs — e.g. a `.best_native` captured
# from a FAILED iter becomes the next run's starting snapshot and the
# regression-detection logic keeps restoring it (Phi-3.5 attention
# case, 2026-06-03). Per the docstring of ``_snapshot_best_native_stub``
# in ``auto_iterate.py`` these are explicitly described as
# "session-scoped" / "in-session" — they should not survive across
# sessions.
_SESSION_LOCAL_OVERLAY_SUFFIXES = (
    ".py.best_native",
    ".py.preiter_native",
    ".py.last_good_native",
    ".py.auto_stabilize.bak",
    ".py.bak",
)


def _is_session_local_artifact(rel_path: str) -> bool:
    """Whether ``rel_path`` is a per-session working-state artifact
    that should be excluded from overlay capture. See
    :data:`_SESSION_LOCAL_OVERLAY_SUFFIXES` for the suffix list."""
    return any(rel_path.endswith(suf) for suf in _SESSION_LOCAL_OVERLAY_SUFFIXES)


def _capture_worktree_deltas_as_overlay(worktree_path, model_id):
    """Capture every modified file in `worktree_path` as a per-model overlay.

    Fires on BOTH full success (rc=0) and partial success (rc != 0 with at
    least some agent-modified files). v12 demonstrated the production
    pain: a 3h41m run graduated 5 components but ended rc=1, so the
    previous success-only capture path discarded all 5. Capturing on
    partial success too means the next `up` run starts from those
    graduated stubs via the existing pre-flight already-native detection,
    skipping the components that already work.

    Stubs persisted from a failed run may contain broken native code -
    that's OK: pre-flight pytest will detect failure and the convergence
    loop will retry them. The worst case is "same as starting fresh";
    the best case is "5 fewer components to redo."

    Returns (captured_count, capture_ok). On any exception the function
    returns (0, False) and the caller preserves the worktree so deltas
    aren't lost.

    Generic across models -- it just diffs the worktree against base and
    stores patches, no model-specific filtering.

    BUG-1 FIX: The original implementation relied on ``git status --porcelain``
    to find changed files. This silently dropped files in some scenarios
    (untracked files newly written by autofill that weren't in the index,
    files whose changes were already staged by a prior overlay-apply, etc.).
    The fix adds an explicit demo-dir scan as a SECONDARY capture pass:
    every ``_stubs/*.py`` and ``tests/pcc/*.py`` under the model's demo dir
    that differs from HEAD also gets captured. Belt-and-suspenders.

    BUG-2 FIX (2026-06-03): ``.py.best_native`` and the other
    session-scoped working-state snapshots (``_SESSION_LOCAL_OVERLAY_SUFFIXES``)
    are skipped. Capturing them poisoned the next run's
    ``.best_native``-rollback floor with a FAILED iter's stub, making
    the regression-detection logic keep restoring a broken wrapper
    (Phi-3.5 attention TT_FATAL'd on use_hf_rope=True from iter_005,
    overlay captured the bad stub, next run's iter 1 wrote a correct
    wrapper, regression detector rolled back to the bad overlay).
    """
    import subprocess as _subprocess

    from .overlay_manager import store_patch as _store_patch

    captured = 0
    seen_paths: set = set()

    def _capture_one(rel_path: str) -> bool:
        """Stage + diff + store one path. Returns True if captured."""
        if rel_path in seen_paths:
            return False
        seen_paths.add(rel_path)
        if _is_session_local_artifact(rel_path):
            return False
        _subprocess.run(
            ["git", "add", "--", rel_path],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        diff_proc = _subprocess.run(
            ["git", "diff", "HEAD", "--", rel_path],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.returncode != 0 or not diff_proc.stdout.strip():
            return False
        rec = _store_patch(model_id, rel_path, diff_proc.stdout, source="captured_from_bringup")
        return bool(rec)

    try:
        status_proc = _subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(worktree_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if status_proc.returncode != 0:
            raise RuntimeError(f"git status --porcelain failed: {status_proc.stderr.strip()}")
        changed: List[str] = []
        for line in status_proc.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1].strip()
            if path:
                changed.append(path)
        for f in changed:
            if _capture_one(f):
                captured += 1

        # BUG-1 FIX: secondary capture pass that explicitly scans the demo
        # dir for stub + test files. Catches any path git status missed
        # (untracked-but-staged-via-overlay-apply, etc.). Generic — uses
        # the standard tt_hw_planner demo-dir layout.
        try:
            from .bringup_loop import find_demo_dir as _find_demo_dir

            demo_dir = _find_demo_dir(model_id, repo_root=Path(worktree_path))
        except Exception:
            demo_dir = None
        if demo_dir is not None and demo_dir.is_dir():
            scan_paths: List[Path] = []
            for sub in ("_stubs", "tests/pcc", "demo"):
                sub_dir = demo_dir / sub
                if sub_dir.is_dir():
                    scan_paths.extend(sub_dir.glob("*.py"))
            for full_path in scan_paths:
                try:
                    rel_path = str(full_path.relative_to(Path(worktree_path)))
                except ValueError:
                    continue
                if _capture_one(rel_path):
                    captured += 1
        return captured, True
    except Exception as exc:
        print(f"  [isolation] capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return captured, False


def _pytest_line_interesting(s: str) -> bool:
    """True if a pytest stream line is worth showing on a non-verbose screen
    (stage markers, PCC, node ids, pass/fail summaries). Everything else is
    framework noise that still goes to the capture/stream log. Shared by the
    cli.py focused-pytest pump and the prepare.py capture pump."""
    s = s.strip()
    if not s:
        return False
    # Match the real progress marker (a line that STARTS with the marker), not
    # pytest's failure-traceback echo of the test source, where lines like
    # `print("[bringup] stage=...", flush=True)` merely CONTAIN the literal.
    if (
        s.startswith("[bringup] stage=")
        or s.startswith("[bringup] achieved PCC=")
        or "FINAL_PCC" in s
        or "Waiting for lock" in s
    ):
        return True
    if s.startswith("models/") and "::" in s:
        return True
    # A pytest source-echo line for one of our markers is just the print()
    # statement; never show those (they pass the PASSED/FAILED check below via
    # neither — but guard explicitly in case a marker name contains a keyword).
    if s.startswith("print(") and "[bringup] stage=" in s:
        return False
    if re.search(r"\b(PASSED|FAILED|ERROR)\b", s):
        return True
    if re.match(r"=+.*(passed|failed|error|skipped).*=+", s):
        return True
    return False


def _quiet_framework_logging() -> None:
    """Quiet noisy framework logging on the bring-up terminal. No-op when
    TT_HW_PLANNER_VERBOSE=1; full detail always lands in the per-agent and
    pytest stream logs regardless. Suppresses, on screen only:
      - HF "Loading checkpoint shards" progress bars + advisory warnings
        (reprinted by capture + every per-agent prompt-block HF load);
      - the loguru DEBUG dumps emitted at `import ttnn` (ttnn.CONFIG /
        dispatch-core) and the tool's own bringup_plan INFO lines — in BOTH
        this process and the pytest subprocesses (LOGURU_LEVEL is honored by
        a child's fresh loguru import).
    The C++/UMD logger (TT_METAL_LOGGER_LEVEL) is left untouched on purpose:
    the stale-device recovery greps subprocess output for device-log markers,
    so silencing it could mask the very lines that trigger a tt-smi reset.

    DEFAULT IS QUIET: unless TT_HW_PLANNER_VERBOSE is explicitly set to 1/true,
    the screen stays clean (agent/framework chatter suppressed); full detail
    always lands in the per-agent and pytest stream logs regardless.
    Opt into the full/verbose terminal with TT_HW_PLANNER_VERBOSE=1.
    """
    os.environ.setdefault("TT_HW_PLANNER_VERBOSE", "0")
    if os.environ.get("TT_HW_PLANNER_VERBOSE", "") not in ("", "0", "false", "False"):
        return
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    try:
        from transformers.utils import logging as _hf_logging

        _hf_logging.disable_progress_bar()
        _hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from loguru import logger as _loguru_logger

        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass


def _warn_on_registry_drift(args=None) -> None:
    """Remote-first registry sync + non-fatal drift check (fixes-plan Point 2a).

    Resolves ``tenstorrent/tt-metal`` ``main`` to a pinned sha, sha-cached
    shallow/sparse-fetches the reusable-module subtrees, regenerates the
    overlay registry from that snapshot, then drift-checks (missing paths vs
    the local checkout; unmapped modules vs the fetched tree). ``--offline`` or
    ``TT_HW_PLANNER_OFFLINE`` (or no network) falls back to the local checkout
    with a loud stale warning. Records the resolved sha for the run report.
    Never raises: neither a fetch nor a drift check may block bring-up.
    """
    try:
        from .discovery import REPO_ROOT
        from .registry_sync import (
            check_registry_drift,
            fetch_upstream_models,
            format_drift,
            has_hard_drift,
            refresh_registry,
        )

        offline = bool(getattr(args, "offline", False) or getattr(args, "no_registry_sync", False))
        tree = fetch_upstream_models(REPO_ROOT, offline=offline)
        refresh_registry(tree.root, sha=tree.sha)
        os.environ["TT_HW_PLANNER_REGISTRY_SHA"] = f"{tree.source}:{tree.sha}"
        if tree.stale:
            print(
                f"[registry] using LOCAL tree (may be stale) — upstream not fetched (offline/no-network); sha={tree.sha[:12]}"
            )
        elif os.environ.get("TT_HW_PLANNER_VERBOSE"):
            print(f"[registry] synced to tenstorrent/tt-metal@{tree.sha[:12]}")

        issues = check_registry_drift(REPO_ROOT, include_unmapped=False, unmapped_root=tree.root)
        if has_hard_drift(issues):
            n = sum(1 for i in issues if i.kind == "missing_path")
            print(
                f"[registry] {n} mapped registry path(s) are stale on this checkout — run `tt_hw_planner sync-registry` for detail."
            )
            if os.environ.get("TT_HW_PLANNER_VERBOSE"):
                print(format_drift(issues))
    except Exception:
        pass


def cmd_up(args) -> int:
    _quiet_framework_logging()
    _warn_on_registry_drift(args)
    # Resolve local-weights handling BEFORE any subprocess is spawned.
    # Sets HF_HOME / HF_HUB_OFFLINE in os.environ when the user passed
    # --local-dir or --offline-hf; prints an info line when cached
    # weights are auto-detected. Inheriting via os.environ means every
    # pytest subprocess (_run_focused_pytest, _run_prepare_capture,
    # _emit_and_verify_runnable_demo) sees the resolved settings
    # without each call site needing to know.
    _model_id = getattr(args, "model_id", "") or ""
    if _model_id:
        _apply_local_weights_env(args, _model_id)
        # Auto-enable the TT-side activation probe so chain-divergence
        # diagnostic has records to compare against if e2e PCC fails.
        # Idempotent; sets a deterministic per-model path.
        _auto_enable_tt_probe(_model_id)
    if not getattr(args, "isolation", "none") == "worktree":
        return _cmd_up_core(args)
    return _cmd_up_isolated(args)


def cmd_bringup(args) -> int:
    """Zero-flag bring-up: ``tt-hw-planner bringup <model_id>``.

    Wraps :func:`cmd_up` with brain-orchestrated defaults so the user
    doesn't have to remember every ``--auto-*`` knob. The orchestrator
    (brain G8) decides everything from here.

    Tunable knobs are locked in at sensible values:
      - ``--auto`` ON (otherwise the brain doesn't run)
      - ``--auto-agent claude`` (the supported LLM agent)
      - ``--auto-model-tiered`` ON (lighter model for easy iters,
        heavier for stuck ones — saves cost without hurting quality)
      - ``--auto-max-iters 24`` (brain G8 can extend via
        ``should_extend_budget`` if close to converging)
      - ``--auto-max-attempts-per-component 5`` (brain G8 can extend
        per component via ``should_extend_component_cap``)
      - ``--isolation worktree`` (default — clean state, atomic capture)

    The user can override any of these by using the full ``up`` command
    with explicit flags; ``bringup`` exists purely so the common case
    is a one-liner.
    """
    # Build a fully-populated args namespace that cmd_up expects. We
    # take the existing args (which has model_id + box from the
    # bringup subparser) and fill in every other knob with the brain-
    # orchestrated default.
    import argparse as _argparse

    full = _argparse.Namespace(**vars(args))

    # Auto-mesh: if the user didn't pass --mesh, pick the box's
    # LARGEST canonical mesh (most chips, tiebreak toward max-TP
    # shape — favors 1xN over balanced for typical workloads). The
    # default planner picks the SMALLEST viable mesh, which the user
    # explicitly asked us to override.
    #
    # Note: cmd_bringup is bound to BOTH the `auto-up` and `bringup`
    # subcommands. `bringup` (per-component stub bring-up) doesn't
    # carry --box, and cmd_up's Step 4/6 builds an autofill Namespace
    # for cmd_bringup that omits box too. In those cases auto-mesh
    # selection has nothing to do — skip it instead of crashing.
    if not getattr(full, "box", None):
        pass  # no box -> bringup-subcommand path; skip auto-mesh
    elif not getattr(full, "mesh", None):
        try:
            from .hardware import HARDWARE as _HW

            _box = next(b for b in _HW if b.name == full.box)
            if getattr(_box, "default_mesh", None):
                best = _box.default_mesh
            else:
                shapes = list(_box.mesh_shapes)
                # Sort by: (chip_count desc, max_dim desc) so largest mesh
                # with max-TP shape wins.
                shapes.sort(key=lambda s: (s[0] * s[1], max(s)), reverse=True)
                best = shapes[0]
            full.mesh = f"{best[0]},{best[1]}"
            print(
                f"  [auto-up] no --mesh passed → using box's LARGEST canonical mesh: "
                f"{full.mesh} ({best[0]*best[1]} chips on {full.box})"
            )
        except Exception as _mesh_exc:
            print(
                f"  [auto-up] mesh auto-selection failed "
                f"({type(_mesh_exc).__name__}: {_mesh_exc}); planner will default",
                file=sys.stderr,
            )

    # Brain-orchestrated defaults (locked in).
    defaults = {
        "auto": True,
        "auto_agent": "claude",
        "auto_model_tiered": True,
        "auto_max_iters": 24,
        "auto_max_attempts_per_component": 5,
        "isolation": "worktree",
        # Other knobs that `up` reads — set to neutral defaults so
        # argparse-injected attributes don't surprise cmd_up.
        # NOTE: mesh handled above (auto-mesh logic).
        "dtype": None,
        "batch": 1,
        "max_seq_len": 8192,
        "max_generated_tokens": None,
        "accuracy": False,
        "no_trace": False,
        "no_paged_attention": False,
        "no_instruct": False,
        "execute": False,
        "download_first": False,
        "strict": False,
        "no_env_fix": False,
        "auto_model": None,
        "auto_model_light": None,
        "auto_model_heavy": None,
        "auto_model_super_heavy": None,
        "auto_agent_timeout": 600,
        "accept_fallback": False,
        "strict_pcc": True,
        "no_strict_pcc": False,
        "escalate_on_pcc_fail": True,
        "no_escalate_on_pcc_fail": False,
        "strict_pcc_tokens": None,
        "pcc_engine": "agentic",
        "allow_partial_cpu": False,
        "regen_demo_only": False,
        "accept_closest_backend": False,
        "no_meta_plan": False,
        "force_fallback": False,
        "op_synth": True,
        "no_op_synth": False,
        "no_kill_stale": False,
        "no_device_reset": False,
    }
    for k, v in defaults.items():
        if not hasattr(full, k):
            setattr(full, k, v)

    print()
    print("=" * 78)
    print(f"  BRINGUP (brain-orchestrated): {full.model_id}")
    print("=" * 78)
    print(
        f"  Locked defaults: --auto --auto-agent=claude --auto-model-tiered "
        f"--auto-max-iters=24 --auto-max-attempts-per-component=5 "
        f"--isolation=worktree"
    )
    print(
        f"  Brain G8 will orchestrate: budget-extend, cap-extend, " f"phantom-cleanup, sync, demo-emit, demo-recovery"
    )
    print()

    return cmd_up(full)


def _cmd_up_isolated(args) -> int:
    from .overlay_manager import apply_for, store_patch, using_repo
    from .worktree import create as _wt_create, destroy as _wt_destroy

    session = _wt_create(args.model_id)
    print(f"  [isolation] worktree: {session.path}")

    with using_repo(session.path):
        n_shared, shared_files = apply_for("_shared")
        n_model, model_files = apply_for(args.model_id)
    if n_shared or n_model:
        print(f"  [isolation] applied {n_shared} _shared + {n_model} model overlay(s)")
        _stage_proc = subprocess.run(
            ["git", "add", "-u"],
            cwd=session.path,
            capture_output=True,
            text=True,
            check=False,
        )
        if _stage_proc.returncode != 0:
            print(
                f"  [isolation] WARN git add -u failed: {_stage_proc.stderr.strip()}; "
                f"capture may double-count overlay state",
                file=sys.stderr,
            )

    prev_env = {
        "TT_HW_PLANNER_BRINGUP_CWD": os.environ.get("TT_HW_PLANNER_BRINGUP_CWD"),
        "TT_HW_PLANNER_OVERLAY_MODEL": os.environ.get("TT_HW_PLANNER_OVERLAY_MODEL"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
    }
    os.environ["TT_HW_PLANNER_BRINGUP_CWD"] = str(session.path)
    os.environ["TT_HW_PLANNER_OVERLAY_MODEL"] = args.model_id
    _existing_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{session.path}:{_existing_pp}" if _existing_pp else str(session.path)
    prev_cwd = os.getcwd()
    os.chdir(session.path)

    rc = 1
    try:
        rc = _cmd_up_core(args)
        return rc
    finally:
        os.chdir(prev_cwd)
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        capture_label = "full success" if rc == 0 else f"partial success (rc={rc})"
        print(f"  [isolation] {capture_label} — capturing LLM deltas as overlay for {args.model_id}")
        captured, capture_ok = _capture_worktree_deltas_as_overlay(session.path, args.model_id)
        if capture_ok:
            if rc == 0:
                print(f"  [isolation] captured {captured} LLM delta(s); destroying worktree")
                try:
                    _wt_destroy(session)
                except Exception as exc:
                    print(
                        f"  [isolation] WARN worktree destroy failed: {type(exc).__name__}: {exc}. "
                        f"Run `git worktree remove --force {session.path}` to clean up.",
                        file=sys.stderr,
                    )
            else:
                print(f"  [isolation] captured {captured} LLM delta(s); worktree preserved at:")
                print(f"     {session.path}")
                print(
                    f"  [isolation] next `up` run will start from these overlays "
                    f"(skipping already-graduated components). To start clean, run:"
                )
                print(f"     python -m scripts.tt_hw_planner overlay-drop {args.model_id}")
                print(f"  [isolation] cd to worktree to debug. To delete it later:")
                print(f"     git worktree remove --force {session.path}")
        else:
            print(
                f"  [isolation] capture failed; preserving worktree at {session.path} so deltas aren't lost. "
                f"Inspect overlays/{args.model_id.replace('/', '_')}/ for what was captured before the failure.",
                file=sys.stderr,
            )


def _cmd_up_core(args) -> int:
    from ._cli_helpers.bringup_cc import _emit_stop_summary, _reset_summary

    _reset_summary()
    model_id = getattr(args, "model_id", "") or ""
    stop_reason = "bring-up ended"
    try:
        rc = _cmd_up_core_impl(args)
        stop_reason = {
            0: "bring-up completed (gate can_stop)",
            1: "bring-up incomplete — components still not graduated (budget/attempts capped)",
            2: "pre-flight/setup failed — model could not be loaded, scaffolded, or prepared",
        }.get(rc, f"ended with rc={rc}")
        return rc
    except Exception as exc:
        stop_reason = f"aborted by exception: {type(exc).__name__}: {exc}"
        raise
    finally:
        if model_id:
            try:
                _emit_stop_summary(model_id, stop_reason)
            except Exception:
                pass


def _cmd_up_core_impl(args) -> int:
    MODEL = args.model_id
    BOX = args.box
    sep = "=" * 78

    if getattr(args, "auto", False) and "TT_HW_PLANNER_AUTO_ONBOARD_REQUIRE_REVIEW" not in os.environ:
        os.environ["TT_HW_PLANNER_AUTO_ONBOARD_REQUIRE_REVIEW"] = "0"

    def banner(title: str) -> None:
        print()
        print(sep)
        print(f"  {title}")
        print(sep)

    _repair_loop_provider = (
        "claude" if (getattr(args, "auto_agent_bin", None) or "claude").endswith("claude") else "cursor"
    )
    _repair_loop_default_model = getattr(args, "auto_model", None) or (
        "opus" if _repair_loop_provider == "claude" else "sonnet-4"
    )
    (
        _repair_model_light,
        _repair_model_heavy,
        _repair_model_super_heavy,
    ) = _resolve_tiered_model_aliases(
        provider=_repair_loop_provider,
        auto_model=_repair_loop_default_model,
        auto_model_light=getattr(args, "auto_model_light", None),
        auto_model_heavy=getattr(args, "auto_model_heavy", None),
        auto_model_super_heavy=getattr(args, "auto_model_super_heavy", None),
        auto_model_tiered=bool(getattr(args, "auto_model_tiered", False)),
    )
    if _repair_model_light or _repair_model_heavy or _repair_model_super_heavy:
        _super_label = f" → super_heavy={_repair_model_super_heavy}" if _repair_model_super_heavy else ""
        print(
            f"  [auto:{_repair_loop_provider}] tiered model switching "
            f"enabled for repair loops: "
            f"light={_repair_model_light or _repair_loop_default_model}, "
            f"heavy={_repair_model_heavy or _repair_loop_default_model}"
            f"{_super_label}"
        )

    if getattr(args, "regen_demo_only", False):
        banner(f"REGEN-DEMO-ONLY for {MODEL}")
        ok, _ = _emit_and_verify_runnable_demo(MODEL, sep=sep)
        return 0 if ok else 1

    allow_kill_stale = not getattr(args, "no_kill_stale", False)
    allow_device_reset = not getattr(args, "no_device_reset", False)
    _preflight_cleanup_stale_pytest(
        REPO_ROOT, allow_kill=allow_kill_stale, allow_device_reset=allow_device_reset, context="up:preflight"
    )

    banner(f"Step 0/6  Pre-flight checks for {MODEL}")
    ok = _preflight_load_with_autofix(MODEL, allow_fix=not args.no_env_fix)
    if not ok:
        print(
            f"  Pre-flight failed: the installed `transformers` cannot load {MODEL} "
            f"and auto-upgrade was {'disabled (--no-env-fix)' if args.no_env_fix else 'unable to repair the env'}. "
            f"Aborting before scaffold / autofill / prepare.",
            file=sys.stderr,
        )
        return 2

    banner(f"Step 1/6  Static analysis (plan + compat)")
    plan_argv = argparse.Namespace(
        model_id=MODEL,
        box=[BOX],
        dtype=[],
        batch=1,
        seq=8192,
        kv_dtype="bf16",
        all_meshes=False,
        explore_pp=False,
        format="table",
        no_overhead_detail=True,
    )
    try:
        cmd_plan(plan_argv)
    except SystemExit:
        pass
    except Exception as exc:
        print(f"  plan failed: {exc}", file=sys.stderr)
        if args.strict:
            return 2
    compat_argv = argparse.Namespace(
        model_id=MODEL,
        skip_kernel_check=False,
        tp_grid=None,
        format="table",
        verbose=False,
    )
    try:
        compat_rc = cmd_compat(compat_argv)
    except Exception as exc:
        print(f"  compat failed: {exc}", file=sys.stderr)
        if args.strict:
            return 2
        compat_rc = 0
    _strict = getattr(args, "strict", False)
    if compat_rc == 1:
        print(
            f"  Compatibility gate FAILED: could not analyze {MODEL} on {BOX} "
            f"(cmd_compat exit=1) -- config.json unavailable, nothing to build "
            f"against. Aborting before scaffold / autofill / prepare.",
            file=sys.stderr,
        )
        return compat_rc
    if compat_rc in (2, 3):
        _kind = "missing building block(s)" if compat_rc == 2 else "kernel-level constraint(s)"
        if _strict:
            print(
                f"  Compatibility gate FAILED (--strict): {MODEL} on {BOX} has "
                f"{_kind} (cmd_compat exit={compat_rc}). Aborting before scaffold.",
                file=sys.stderr,
            )
            return compat_rc
        print(
            f"  Compatibility gate: {MODEL} on {BOX} has {_kind} (cmd_compat "
            f"exit={compat_rc}). NOT aborting -- recording as work-item(s) and "
            f"continuing into the bring-up loop (ADAPT / REUSE / NEW). Blockers "
            f"listed above are graduated where possible and isolated "
            f"(stub / CPU-fallback) otherwise. Re-run with --strict to hard-fail "
            f"instead.",
            file=sys.stderr,
        )

    mem_fit_rc = _enforce_memory_fit_or_abort(
        MODEL,
        box_name=BOX,
        mesh_str=getattr(args, "mesh", None),
        dtype_override=getattr(args, "dtype", None),
    )
    if mem_fit_rc is not None:
        return mem_fit_rc

    from .compatibility import check_compatibility

    try:
        _early_compat = check_compatibility(MODEL, probe_model(MODEL).raw_config)
    except Exception:
        _early_compat = None

    _compat_already_supported_early = _early_compat is not None and (
        _early_compat.overall.startswith("ALREADY SUPPORTED")
        or (_early_compat.in_external_demo and _early_compat.primary_demo is not None)
    )

    if not _compat_already_supported_early:
        backend_quality_rc = _enforce_backend_match_quality_or_abort(
            MODEL,
            accept_closest=getattr(args, "accept_closest_backend", False),
        )
        if backend_quality_rc is not None:
            return backend_quality_rc

    env_ok_early, env_problems_early = _check_demo_environment_compat()
    _is_external_demo_early = _early_compat is not None and getattr(_early_compat, "in_external_demo", False)
    if not env_ok_early and _is_external_demo_early:
        print()
        print(
            f"  [env] pre-flight advisories target the tt_transformers simple_text_demo "
            f"codepath, which {MODEL} does NOT use (it has its own external demo). "
            f"Non-fatal -- continuing bring-up."
        )
        for line in env_problems_early:
            print(f"  - {line}")
        env_ok_early, env_problems_early = True, []
    if not env_ok_early:
        sep_e = "=" * 72
        print()
        print(sep_e)
        print("  ENVIRONMENT INCOMPATIBLE -- pre-flight found problems in the tt_transformers demo codepath:")
        print(sep_e)
        for line in env_problems_early:
            print(f"  - {line}" if not line.startswith("transformers==") else f"  {line}")
        print()
        print(
            "  This model uses the tt_transformers simple_text_demo path, which needs a "
            "compatible environment. The tool will NOT change your environment automatically "
            "(a prior auto-downgrade removed torch as collateral). Install the versions pinned "
            "in tt_metal/python_env/requirements-dev.txt by hand, e.g.:"
        )
        print("      pip install --extra-index-url https://download.pytorch.org/whl/cpu \\")
        print("          'torch==2.11.0' 'transformers==5.10.2'")
        print("  then re-run. To proceed anyway and accept the risk, pass --no-env-fix.")
        print(sep_e)
        if not getattr(args, "no_env_fix", False):
            return 2
        print("  --no-env-fix set: proceeding despite advisories.")
    _already_supported = (
        _early_compat is not None
        and (
            _early_compat.overall.startswith("ALREADY SUPPORTED")
            or (_early_compat.in_external_demo and _early_compat.primary_demo is not None)
        )
        and not _post_escalation_bypass(args)
    )

    _route_via_generic_llm = False
    if _early_compat is not None and not _already_supported:
        try:
            from .compatibility import Status as _CompatStatus

            _missing = [
                r
                for r in (_early_compat.results or [])
                if getattr(r, "needed", False) and getattr(r, "status", None) == _CompatStatus.MISSING
            ]
        except Exception:
            _missing = []

        _generic_backend_picked = False
        _generic_pick_error: Optional[str] = None
        try:
            from .family_backends import pick_backend
            from .probe import probe_model as _probe_model

            _probe_obj = _probe_model(MODEL)
            if _probe_obj is not None:
                _model_type = getattr(_probe_obj, "model_type", None) or (
                    (_probe_obj.raw_config or {}).get("model_type")
                )
                _pipeline_tag = getattr(_probe_obj, "pipeline_tag", None)
                _category = getattr(_probe_obj, "category", None)
                _backend = pick_backend(
                    category=_category,
                    model_type=_model_type,
                    pipeline_tag=_pipeline_tag,
                )
                _generic_backend_picked = _backend is not None and getattr(_backend, "routing_mode", "") == "generic"
        except Exception as exc:
            _generic_pick_error = f"{type(exc).__name__}: {exc}"
            _generic_backend_picked = False

        try:
            _partial = [
                r
                for r in (_early_compat.results or [])
                if getattr(r, "needed", False) and getattr(r, "status", None) == _CompatStatus.PARTIAL
            ]
        except Exception:
            _partial = []
        if _early_compat.overall == "READY" and not _missing and not _partial and _generic_backend_picked:
            _route_via_generic_llm = True
        elif _generic_backend_picked and _partial:
            print(
                f"  (note: generic LLM backend matched, but "
                f"{len(_partial)} needed building block(s) are "
                f"PARTIAL. The portable demo will attempt the run; "
                f"if it hits `NotImplementedError` on a partial "
                f"path, you'll need to add coverage for that "
                f"block. Listing partials:)"
            )
            for r in _partial[:8]:
                _bn = getattr(getattr(r, "block", None), "name", "?")
                _eff = getattr(getattr(r, "effort", None), "value", "?")
                print(f"    - {_bn} [{_eff}]")
            if len(_partial) > 8:
                print(f"    - ... and {len(_partial) - 8} more")

        _partial_block_names_for_summary = [getattr(getattr(r, "block", None), "name", "?") for r in (_partial or [])]

    if not getattr(args, "no_meta_plan", False) and not _already_supported and not _route_via_generic_llm:
        _run_advisory_meta_plan(
            MODEL,
            box=BOX,
            mesh=getattr(args, "mesh", None),
            agent_bin=getattr(args, "auto_agent_bin", None) or "claude",
            agent_model=getattr(args, "auto_model", None) or "sonnet",
        )

    banner(f"Step 2/6  Scaffold the demo folder for {MODEL}")
    if _route_via_generic_llm and not _already_supported:
        print(
            f"  GENERIC LLM BACKEND. No per-model tt/ folder needed -- "
            f"`tt_transformers/simple_text_demo.py` is architecture-"
            f"portable and reads HF_MODEL from the env. Skipping "
            f"scaffold and routing directly to `prepare --execute`."
        )
        print(
            f"  NOTE: no tuning row exists yet for {MODEL}. The demo "
            f"will use the parametrize defaults (e.g. "
            f"max_seq_len=1024). For tuned performance later, add an "
            f"entry to models/tt_transformers/demo/MAX_PREFILL_CHUNK_"
            f"SIZES_DIV1024 and trace_region_config.py."
        )

        _already_supported = True
        if vars(args).get("_escalated_already"):
            _already_supported = False
    if _already_supported:
        env_ok, env_problems = _check_demo_environment_compat()
        if not env_ok and _early_compat is not None and getattr(_early_compat, "in_external_demo", False):
            print(
                f"  [env] pre-flight advisories target simple_text_demo; {MODEL} uses its own "
                f"external demo -- non-fatal, continuing."
            )
            for line in env_problems:
                print(f"  - {line}")
            env_ok, env_problems = True, []
        if not env_ok:
            sep = "=" * 72
            print()
            print(sep)
            print("  ENVIRONMENT INCOMPATIBLE -- pre-flight found problems in the tt_transformers demo codepath:")
            print(sep)
            for line in env_problems:
                print(f"  - {line}" if not line.startswith("transformers==") else f"  {line}")
            print()
            print(
                "  The tool will NOT change your environment automatically. Install the versions "
                "pinned in tt_metal/python_env/requirements-dev.txt by hand, e.g.:"
            )
            print("      pip install --extra-index-url https://download.pytorch.org/whl/cpu \\")
            print("          'torch==2.11.0' 'transformers==5.10.2'")
            print("  then re-run. To proceed anyway and accept the risk, pass --no-env-fix.")
            print(sep)
            if not getattr(args, "no_env_fix", False):
                return 2
            print("  --no-env-fix set: proceeding despite advisories.")

        print(
            f"  ALREADY SUPPORTED via tt_transformers/simple_text_demo. "
            f"Skipping scaffold (it only adds tuning rows for NEW "
            f"models; this one is already in the table). Auto-routing "
            f"to `prepare --execute`."
        )
        banner(f"Step 2b/6  Prepare + execute {MODEL} via tt_transformers")
        try:
            from .probe import probe_model as _probe_for_embed

            _embed_probe = _probe_for_embed(MODEL)
            _embed_category = getattr(_embed_probe, "category", None) or ""
        except Exception:
            _embed_category = ""
        _is_embed_run = _embed_category == "Embed"
        if _is_embed_run:
            os.environ["TT_HW_PLANNER_EMIT_EMBEDDINGS"] = "1"
            print(
                "  [embed-mode] category=Embed; setting "
                "TT_HW_PLANNER_EMIT_EMBEDDINGS=1 so simple_text_demo "
                "skips decoding and emits ==EMBED markers for the gate, "
                "and forcing no_instruct=True so the chat template is "
                "not applied (HF reference tokenizes raw probes; the TT "
                "side must match)."
            )

        prepare_argv = argparse.Namespace(
            model_id=MODEL,
            box=BOX,
            mesh=getattr(args, "mesh", None),
            dtype=getattr(args, "dtype", None),
            batch=1,
            max_seq_len=1024,
            max_generated_tokens=200,
            accuracy=False,
            no_trace=False,
            no_paged_attention=False,
            no_instruct=_is_embed_run,
            format="text",
            write_script=None,
            execute=True,
            strict=False,
            download_first=False,
        )

        _auto_mode = getattr(args, "auto", False)
        _captured_output = ""
        try:
            if _auto_mode:
                _rc_supp, _captured_output = _run_prepare_capture(prepare_argv)
            else:
                _rc_supp = cmd_prepare(prepare_argv)
        except SystemExit as exc:
            _rc_supp = int(exc.code) if exc.code is not None else 2
        except Exception as exc:
            print(
                f"  prepare --execute failed: {exc}. You can retry "
                f"manually with: python -m scripts.tt_hw_planner "
                f"prepare {MODEL} --box {BOX}"
                f"{' --mesh ' + getattr(args, 'mesh', None) if getattr(args, 'mesh', None) else ''}"
                f" --execute",
                file=sys.stderr,
            )
            _rc_supp = 2

        # HF weight failures are environmental — bail with a clear
        # remediation message BEFORE the runtime-repair / escalation
        # logic kicks in. Saves the user's iter budget and gives
        # actionable next steps. Returns silently when not matched.
        if _auto_mode and _captured_output:
            _exit_if_hf_weight_failure(MODEL, _captured_output)

        if _rc_supp != 0 and _auto_mode and _captured_output:
            _rc_supp = _runtime_repair_loop(
                model_id=MODEL,
                prepare_argv=prepare_argv,
                initial_rc=_rc_supp,
                initial_output=_captured_output,
                agent_bin=getattr(args, "auto_agent_bin", None) or "claude",
                agent_model=getattr(args, "auto_model", None) or "sonnet",
                max_iters=getattr(args, "auto_max_iters", 5),
                agent_timeout_s=getattr(args, "auto_agent_timeout", 1500),
                sep=sep,
                model_light=_repair_model_light,
                model_heavy=_repair_model_heavy,
                model_super_heavy=_repair_model_super_heavy,
            )

        if _rc_supp == 0:
            # Strict end-to-end PCC gate. _run_strict_pcc_gate returns
            # (None, None) when not in auto-mode or strict_pcc is off,
            # in which case we preserve legacy behavior (skip gate).
            # When the gate runs and fails, escalate to Path A via
            # _maybe_escalate_pcc_fail — the per-component iterate
            # flow that brought up Qwen.
            _pcc_result, _pcc_prompt = _run_strict_pcc_gate(args, MODEL, _captured_output, _auto_mode)
            if _pcc_result is not None and not _pcc_result.ok:
                # Chain-divergence diagnostic BEFORE escalation: pure
                # diagnostic, doesn't change routing. Best-effort; any
                # failure inside degrades silently.
                _run_and_log_chain_divergence(MODEL, demo_dir=_find_demo_dir_safe(MODEL))
                _esc_rc = _maybe_escalate_pcc_fail(args, MODEL, _PCC_FAIL_RC, _auto_mode)
                _rc_supp = _esc_rc if _esc_rc is not None else _PCC_FAIL_RC
        if _rc_supp == 0:
            _register_bringup_success(
                MODEL,
                path="ALREADY SUPPORTED (auto-routed to prepare --execute)",
                sep=sep,
                notes="Model was detected as already-supported; cmd_up auto-routed.",
            )
            # WIRING #7 (Path 2 parity): brain G8 sync graduated stubs
            # to main tree. Path 1's auto_iterate calls this on success;
            # Path 2 needs the same wire so worktree-only edits during
            # PCC-repair land in main tree. Non-fatal.
            try:
                from .agentic.persistence import sync_graduated_to_main_tree

                _demo_dir_path2 = find_demo_dir(MODEL)
                if _demo_dir_path2 is not None:
                    _subpath = (
                        _demo_dir_path2.resolve().relative_to(Path.cwd().resolve())
                        if str(Path.cwd().resolve()) in str(_demo_dir_path2.resolve())
                        else None
                    )
                    if _subpath is not None:
                        _sync_p2 = sync_graduated_to_main_tree(
                            worktree_root=Path.cwd(),
                            demo_subpath=str(_subpath),
                            graduated_components=[],  # whole-model path; no per-comp list
                            safe_id_fn=_safe_id,
                        )
                        if _sync_p2.notes:
                            for note in _sync_p2.notes:
                                print(f"  [brain G8 sync] {note}")
            except Exception as _sync_p2_exc:
                print(
                    f"  [brain G8] Path 2 sync non-fatal: " f"{type(_sync_p2_exc).__name__}: {_sync_p2_exc}",
                    file=sys.stderr,
                )
            # WIRING #8 (Path 2 parity): brain G8 e2e demo-emit decision.
            # Path 1 consults should_emit_e2e_demo at success; Path 2
            # needs the same wire so an ALREADY-SUPPORTED model with
            # partial coverage gets the same emit decision quality.
            try:
                from .agentic.e2e import should_emit_e2e_demo as _brain_p2_emit
                from .final_categorization import build_final_categorization

                _cat = build_final_categorization(
                    model_id=MODEL,
                    demo_dir=find_demo_dir(MODEL) or Path("."),
                )
                _emit_p2 = _brain_p2_emit(
                    on_device=_cat.on_device,
                    kernel_missing=_cat.kernel_missing,
                    pending=_cat.pending,
                    final_all_passed=(_pcc_result is not None and _pcc_result.ok),
                )
                print(
                    f"  [brain G8] demo-emit decision (Path 2): "
                    f"{_emit_p2.confidence} confidence — {_emit_p2.reason}"
                )
            except Exception as _emit_p2_exc:
                print(
                    f"  [brain G8] Path 2 emit-decision non-fatal: " f"{type(_emit_p2_exc).__name__}: {_emit_p2_exc}",
                    file=sys.stderr,
                )
            # B-FIX #3 (2026-05-31): surface gate-engagement state in
            # OUTCOME. The PCC gate's soft-skip (None result) is by-design
            # to avoid false-fails on non-text demos — but the user must
            # know the gate didn't actually verify. Without this, SUCCESS
            # rc=0 hides "we passed pytest but never compared decoded
            # output to HF reference."
            _gate_extra = []
            if _pcc_result is None:
                _gate_extra.append(
                    "PCC correctness gate DID NOT engage (soft-skipped — "
                    "see gate logs above for reason). pytest passed but "
                    "decoded output was NOT compared to HF reference; "
                    "the success is on basis of pytest alone."
                )
            elif not _pcc_result.ok:
                # rc=0 with ok=False shouldn't normally happen (the loop
                # would have set rc != 0), but guard for completeness.
                _gate_extra.append(
                    f"PCC gate verdict: ok=False but rc=0 — " f"reason: {getattr(_pcc_result, 'reason', '(none)')}"
                )
            _final_outcome_banner(
                rc=0,
                model_id=MODEL,
                path_label="ALREADY-SUPPORTED -> prepare --execute",
                extra=_gate_extra or None,
            )
        elif _rc_supp == _PCC_FAIL_RC:
            # WIRING #6 (Path 2 parity): consult brain G8's
            # decide_demo_recovery before banner. The recovery primitive
            # may recommend disabling specific wired components and
            # retrying, or simply surface why a retry is futile. Without
            # this, Path 2 has only the escalation hook (auto-onboard a
            # new backend), which doesn't help when the backend exists
            # but a kernel-level op produces wrong tokens.
            _recovery_extra: List[str] = []
            try:
                from .agentic.demo_recovery import decide_demo_recovery as _brain_p2_recover

                _recover_verdict = _brain_p2_recover(
                    demo_dir=(find_demo_dir(MODEL) or Path(".")),
                    canonical_demo=(find_demo_dir(MODEL) or Path(".")) / "demo" / "demo.py",
                    retries_attempted=0,  # we haven't retried in Path 2 yet
                    max_retries=1,
                    pytest_tail=getattr(_pcc_result, "reason", "") if _pcc_result else "",
                    wired_components=[],
                )
                _recovery_extra.append(
                    f"brain G8 demo-recovery verdict: {_recover_verdict.action} — " f"{_recover_verdict.reason}"
                )
            except Exception as _recover_exc:
                print(
                    f"  [brain G8] Path 2 recovery non-fatal: " f"{type(_recover_exc).__name__}: {_recover_exc}",
                    file=sys.stderr,
                )
            _final_outcome_banner(
                rc=_rc_supp,
                model_id=MODEL,
                path_label="ALREADY-SUPPORTED -> PCC GATE REJECTED OUTPUT",
                extra=_recovery_extra
                + [
                    "pytest passed, but the model's actual decoded "
                    "output diverged from the HF CPU reference "
                    "beyond the configured tolerance.",
                    "This is the 'false green' guard: rc=17 means " "the demo ran but produced wrong tokens.",
                    f"To inspect the divergence, re-run with: "
                    f"python -m scripts.tt_hw_planner up {MODEL} "
                    f"--auto --strict-pcc --strict-pcc-tokens 64 "
                    f"--strict-pcc-max-iters 8",
                    "To bypass the gate (NOT recommended; this is "
                    "what hid the bug originally): add --no-strict-pcc.",
                ],
            )
            esc = _maybe_escalate_pcc_fail(args, MODEL, _rc_supp, _auto_mode)
            if esc is not None:
                return esc
        else:
            # WIRING #12 (else-branch escalation, 2026-05-31): also try
            # _maybe_escalate_pcc_fail on non-PCC failure rcs from
            # the ALREADY-SUPPORTED path (e.g. rc=5 from "build broke
            # during PCC repair"). The escalation function already
            # exists; it drafts a new FamilyBackend via cmd_auto_onboard
            # and re-enters cmd_up with _escalated_already=True so the
            # scaffold-and-iterate Path 1 takes over. This is the
            # existing path that SAM2 used — Path 2 needs to delegate
            # to it when the fast-path repair can't converge.
            esc_else = _maybe_escalate_pcc_fail(args, MODEL, _rc_supp, _auto_mode)
            if esc_else is not None:
                return esc_else
            # WIRING #6 (else-branch, 2026-05-31): generic FAILED
            # branch (rc != 0, rc != _PCC_FAIL_RC). Surfaces brain G8
            # demo-recovery verdict in OUTCOME extras.
            # Note: classify+persist is now handled centrally by
            # failure_classifier.classify_and_persist_skip when the
            # repair paths exit; not duplicated here.
            _generic_extra: List[str] = []
            try:
                from .agentic.demo_recovery import decide_demo_recovery as _brain_p2_recover_else
                from .bringup_loop import find_demo_dir as _find_demo

                _demo_dir_else = _find_demo(MODEL) or Path(".")
                _verdict_else = _brain_p2_recover_else(
                    demo_dir=_demo_dir_else,
                    canonical_demo=_demo_dir_else / "demo" / "demo.py",
                    retries_attempted=0,
                    max_retries=1,
                    pytest_tail="",
                    wired_components=[],
                )
                _generic_extra.append(
                    f"brain G8 demo-recovery verdict: {_verdict_else.action} — " f"{_verdict_else.reason}"
                )
            except Exception as _r_exc:
                print(
                    f"  [brain G8] Path 2 else-branch recovery non-fatal: " f"{type(_r_exc).__name__}: {_r_exc}",
                    file=sys.stderr,
                )
            _final_outcome_banner(
                rc=_rc_supp,
                model_id=MODEL,
                path_label="ALREADY-SUPPORTED -> prepare --execute (FAILED)",
                extra=_generic_extra
                + [
                    f"Re-run with: python -m scripts.tt_hw_planner prepare {MODEL}"
                    f" --execute (with TT_PLANNER_PER_TEST_TIMEOUT_S=3600 for"
                    f" larger models)",
                    "Check the pytest stderr above for the actual error.",
                    "If `Failed: Timeout` is in the output, raise the per-test"
                    " timeout: export TT_PLANNER_PER_TEST_TIMEOUT_S=3600",
                ],
            )
        return _rc_supp

    from .scaffold import (
        ColdStartScaffoldError,
        ScaffoldError as _ScaffoldError,
        plan_scaffold as _plan_scaffold_probe,
    )

    _cold_start_signal: Optional[ColdStartScaffoldError] = None
    try:
        _plan_scaffold_probe(MODEL)
    except ColdStartScaffoldError as cs:
        _cold_start_signal = cs
    except _ScaffoldError:
        pass
    except Exception:
        pass

    if _cold_start_signal is not None:
        _cs_missing, _cs_partial = [], []
        try:
            from .compatibility import Status as _CS

            for r in getattr(_early_compat, "results", None) or []:
                if not getattr(r, "needed", False):
                    continue
                _bn = getattr(getattr(r, "block", None), "name", "?")
                _st = getattr(r, "status", None)
                if _st == _CS.MISSING:
                    _cs_missing.append(_bn)
                elif _st == _CS.PARTIAL:
                    _cs_partial.append(_bn)
        except Exception:
            _cs_missing, _cs_partial = [], []
        if _cs_missing or _cs_partial:
            sep = "=" * 72
            print(sep)
            print("  COLD-START BLOCKED — model is not cleanly supported")
            print(sep)
            if _cs_missing:
                print(f"  MISSING (no native ttnn op): {', '.join(_cs_missing)}")
            if _cs_partial:
                print(f"  PARTIAL (needs porting):     {', '.join(_cs_partial)}")
            print()
            print("  The generic cold-start demo runs the whole model and has no")
            print("  per-op CPU fallback, so it would fail on the block(s) above.")
            print("  Routing to per-component bring-up — unsupported ops run on")
            print("  CPU while the rest graduates on device.")
            print(sep)
            if not getattr(args, "_escalated_already", False):
                print()
                print(f"  AUTO-ROUTE: auto-onboard {MODEL} -> re-enter per-component loop")
                ao_args = argparse.Namespace(
                    model_id=MODEL,
                    agent_bin=getattr(args, "auto_agent_bin", None) or "claude",
                    auto_model=getattr(args, "auto_model", None) or "sonnet",
                    timeout_s=getattr(args, "auto_agent_timeout", 1500) or 1500,
                    skip_llm=False,
                    accept=True,
                )
                try:
                    _ao_rc = cmd_auto_onboard(ao_args)
                except SystemExit as _ao_exc:
                    _ao_rc = int(_ao_exc.code) if _ao_exc.code is not None else 2
                except Exception as _ao_exc:
                    print(f"  auto-onboard raised: {type(_ao_exc).__name__}: {_ao_exc}")
                    _ao_rc = 2
                if _ao_rc == 0:
                    try:
                        import importlib
                        from scripts.tt_hw_planner import compatibility as _compat_mod
                        from scripts.tt_hw_planner import family_backends as _fb_mod

                        importlib.reload(_fb_mod)
                        importlib.reload(_compat_mod)
                    except Exception as _rl_exc:
                        print(
                            f"  WARNING: registry reload after auto-onboard failed: "
                            f"{type(_rl_exc).__name__}: {_rl_exc}"
                        )
                    setattr(args, "_escalated_already", True)
                    return _cmd_up_core(args)
                print(f"  auto-onboard exit={_ao_rc}; cannot auto-route. Run manually:")
            else:
                print()
                print("  Already escalated once; not re-routing. Run manually:")
            print(f"      python -m scripts.tt_hw_planner auto-onboard {MODEL}")
            print(f"      python -m scripts.tt_hw_planner auto-up {MODEL}")
            print(sep)
            return 2
        sep = "=" * 72
        print(sep)
        print("  COLD-START PATH (no per-model `tt/` folder needed)")
        print(sep)
        print(f"  Reason: {_cold_start_signal.reason}")
        print()
        print(
            "  Attempting to run this model 'from scratch' via the generic,\n"
            "  architecture-portable demo. The `tt_transformers/simple_text_demo`\n"
            "  reads HF_MODEL from the env and works for any model whose\n"
            "  building blocks (attention, MLP, RoPE, ...) are already supported\n"
            "  by `tt_transformers/tt/`."
        )
        print()
        print(
            "  If this attempt fails, the fix is one of:\n"
            "    * Add the model_type to `compatibility.closest_supported_model()`\n"
            "    * Add a `FamilyBackend` entry in `family_backends.py`\n"
            "    * Use `auto-onboard` to LLM-draft a backend entry"
        )
        print(sep)
        print()
        prepare_argv = argparse.Namespace(
            model_id=MODEL,
            box=BOX,
            mesh=getattr(args, "mesh", None),
            dtype=getattr(args, "dtype", None),
            batch=1,
            max_seq_len=1024,
            max_generated_tokens=200,
            accuracy=False,
            no_trace=False,
            no_paged_attention=False,
            no_instruct=False,
            format="text",
            write_script=None,
            execute=True,
            strict=False,
            download_first=False,
        )

        _auto_mode = getattr(args, "auto", False)
        _captured_output = ""
        try:
            if _auto_mode:
                _rc_cold, _captured_output = _run_prepare_capture(prepare_argv)
            else:
                _rc_cold = cmd_prepare(prepare_argv)
        except SystemExit as exc:
            _rc_cold = int(exc.code) if exc.code is not None else 2
        except Exception as exc:
            print(
                f"  cold-start `prepare --execute` failed: {exc}.\n"
                f"  This means the architecture isn't directly portable to\n"
                f"  `tt_transformers/simple_text_demo`. Next steps:\n"
                f"    1. Run `python -m scripts.tt_hw_planner auto-onboard {MODEL}`\n"
                f"       to LLM-draft a `FamilyBackend` entry.\n"
                f"    2. Or open `compatibility.py` and add a sibling mapping\n"
                f"       for this model_type.",
                file=sys.stderr,
            )
            _rc_cold = 2

        # HF weight failure short-circuit (cold-start variant). See
        # the equivalent guard at the Path 2 call site above.
        if _auto_mode and _captured_output:
            _exit_if_hf_weight_failure(MODEL, _captured_output)

        if _rc_cold != 0 and _auto_mode and _captured_output:
            _rc_cold = _runtime_repair_loop(
                model_id=MODEL,
                prepare_argv=prepare_argv,
                initial_rc=_rc_cold,
                initial_output=_captured_output,
                agent_bin=getattr(args, "auto_agent_bin", None) or "claude",
                agent_model=getattr(args, "auto_model", None) or "sonnet",
                max_iters=getattr(args, "auto_max_iters", 5),
                agent_timeout_s=getattr(args, "auto_agent_timeout", 1500),
                sep=sep,
                model_light=_repair_model_light,
                model_heavy=_repair_model_heavy,
                model_super_heavy=_repair_model_super_heavy,
            )

        if _rc_cold == 0:
            # Strict end-to-end PCC gate (cold-start variant). Same
            # contract as the ALREADY-SUPPORTED site above — escalate
            # to Path A on fail, preserve legacy non-auto bypass.
            _pcc_result, _pcc_prompt = _run_strict_pcc_gate(args, MODEL, _captured_output, _auto_mode)
            if _pcc_result is not None and not _pcc_result.ok:
                _run_and_log_chain_divergence(MODEL, demo_dir=_find_demo_dir_safe(MODEL))
                _esc_rc_cold = _maybe_escalate_pcc_fail(args, MODEL, _PCC_FAIL_RC, _auto_mode)
                _rc_cold = _esc_rc_cold if _esc_rc_cold is not None else _PCC_FAIL_RC
        if _rc_cold == 0:
            _register_bringup_success(
                MODEL,
                path="B. Generic cold-start (prepare --execute via simple_text_demo / hf_eager)",
                sep=sep,
                notes="Auto-routed via ColdStartScaffoldError handler in cmd_up.",
            )
            _final_outcome_banner(
                rc=0,
                model_id=MODEL,
                path_label="COLD-START -> prepare --execute (generic backend)",
            )
        elif _rc_cold == _PCC_FAIL_RC:
            _final_outcome_banner(
                rc=_rc_cold,
                model_id=MODEL,
                path_label="COLD-START -> PCC GATE REJECTED OUTPUT",
                extra=[
                    "Cold-start pytest passed, but the model's "
                    "decoded output diverged from the HF CPU "
                    "reference beyond the configured tolerance "
                    "(false-green guard, rc=17).",
                    "This typically means the generic backend "
                    "produced gibberish because the architecture "
                    "needs a model-specific RoPE / attention / norm "
                    "wiring that the generic path doesn't have.",
                    f"Next step: python -m scripts.tt_hw_planner "
                    f"auto-onboard {MODEL}  (LLM-drafts a "
                    f"model-specific FamilyBackend entry).",
                    "Or to bypass the gate: --no-strict-pcc",
                ],
            )
        else:
            _next_steps = [
                f"Re-run cold-start with a larger per-test timeout:",
                f"    export TT_PLANNER_PER_TEST_TIMEOUT_S=3600  # 1h",
                f"    python -m scripts.tt_hw_planner up {MODEL} --auto"
                f" --auto-agent claude --auto-model-tiered"
                f"{' --mesh ' + getattr(args, 'mesh', None) if getattr(args, 'mesh', None) else ''}",
                f"If the pytest line above says `Failed: Timeout (>NNNN.0s)`,"
                f" the per-test budget needs to be larger than NNNN.",
                f"If it says `NotImplementedError` for one of the PARTIAL blocks"
                f" listed earlier, the cold-start can't proceed without adding"
                f" that block; run:",
                f"    python -m scripts.tt_hw_planner auto-onboard {MODEL}",
            ]

            try:
                _partials = _partial_block_names_for_summary
            except NameError:
                _partials = []
            if _partials:
                _next_steps.insert(
                    0,
                    f"PARTIAL blocks identified at compat-check time: "
                    + ", ".join(_partials[:6])
                    + ("..." if len(_partials) > 6 else ""),
                )
            _final_outcome_banner(
                rc=_rc_cold,
                model_id=MODEL,
                path_label="COLD-START -> prepare --execute (FAILED)",
                extra=_next_steps,
            )
        return _rc_cold

    scaffold_argv = argparse.Namespace(
        model_id=MODEL,
        apply=True,
        format="text",
        no_diff=True,
        # When the escalation hook (_maybe_escalate_pcc_fail) re-enters
        # cmd_up after Path 2 PCC-gate failure, the compat verdict
        # still reads "ALREADY SUPPORTED" — but the whole point of
        # the escalation is to force Path 1, so the scaffold gate
        # must skip the "already supported" early-exit.
        force_already_supported=bool(vars(args).get("_escalated_already")),
    )
    try:
        rc = cmd_scaffold(scaffold_argv)
        if rc not in (0, None):
            print(f"  scaffold returned non-zero ({rc}); aborting.", file=sys.stderr)
            return 2
    except SystemExit as exc:
        if exc.code not in (0, None):
            return int(exc.code) if exc.code is not None else 2
    except Exception as exc:
        # Try the layered setup-step recovery before bailing. Rules
        # catch known shapes (mkdir-missing-parent, etc.); LLM
        # fallback handles novel ones via an allowlist of actions.
        # If a recovery applies cleanly, retry scaffold once. Same
        # re-entry guard as env-fix prevents loops.
        from ._cli_helpers.setup_step_recovery import (
            RecoveryAction as _RA,
            run_setup_step_recovery as _run_setup_recovery,
        )

        _proposal = _run_setup_recovery(
            exc=exc,
            step_name="step2_scaffold",
            work_dir=Path.cwd(),
            repo_root=BRINGUP_ROOT(),
            workspace_summary=f"model_id={MODEL!r} new_demo_dir candidate path",
        )
        if _proposal is not None and _proposal.action != _RA.CANNOT_RECOVER:
            print(f"  scaffold recovery: {_proposal.label()} → retrying scaffold once")
            try:
                rc = cmd_scaffold(scaffold_argv)
                if rc in (0, None):
                    pass  # recovered; fall through to next step
                else:
                    print(f"  scaffold STILL returned non-zero after recovery ({rc}); aborting.", file=sys.stderr)
                    return 2
            except Exception as retry_exc:
                print(f"  scaffold retry failed: {retry_exc}", file=sys.stderr)
                return 2
        else:
            print(f"  scaffold failed: {exc}", file=sys.stderr)
            return 2

    banner(f"Step 3/6  LLM gate — does this model need an LLM to finish bring-up?")
    counts, _rows = _summarize_bringup_status(MODEL)
    # LLM is needed iff there are NEW or ADAPT components.
    # NEW = write from scratch; ADAPT = refine via iterate loop on PCC<0.99.
    # REUSE alone needs no LLM (presumed working as-is).
    llm_components = counts.get("NEW", 0) + counts.get("ADAPT", 0)
    total_components = sum(counts.values())

    if total_components == 0:
        print("  VERDICT: UNKNOWN — no bringup_status.json (scaffold did not produce a plan).")
        print()
        print(
            f"  Component classification: {counts.get('REUSE',0)} REUSE, "
            f"{counts.get('ADAPT',0)} ADAPT, {counts.get('NEW',0)} NEW "
            f"(total {total_components})"
        )
        print("  Skipping the LLM gate; the loop later will treat this as REUSE-only.")
    elif llm_components == 0:
        print("  ============================================================")
        print(f"  >>>  VERDICT: LLM NOT REQUIRED                            <<<")
        print(f"  >>>  ({counts.get('REUSE',0)}/{total_components} components are REUSE — fully supported)  <<<")
        print("  ============================================================")
        print()
        print(
            f"  Component classification: {counts.get('REUSE',0)} REUSE, "
            f"{counts.get('ADAPT',0)} ADAPT, {counts.get('NEW',0)} NEW "
            f"(total {total_components})"
        )
        print()
        print("  This model is fully supported by existing tt-metal modules.")
        if getattr(args, "auto", False):
            print("  --auto requested but no work remains for the LLM; loop will be skipped.")
            args.auto = False
    else:
        print("  ============================================================")
        print(f"  >>>  VERDICT: LLM REQUIRED                                <<<")
        print(
            f"  >>>  ({llm_components}/{total_components} components need LLM: "
            f"{counts.get('NEW',0)} NEW + {counts.get('ADAPT',0)} ADAPT)  <<<"
        )
        print("  ============================================================")
        print()
        print(
            f"  Component classification: {counts.get('REUSE',0)} REUSE, "
            f"{counts.get('ADAPT',0)} ADAPT, {counts.get('NEW',0)} NEW "
            f"(total {total_components})"
        )
        print()
        if getattr(args, "auto", False):
            provider = (getattr(args, "auto_agent", None) or "cursor").lower()
            if provider not in ("cursor", "claude"):
                print(f"  unknown --auto-agent {provider!r}", file=sys.stderr)
                return 2
            ready, why = _check_agent_ready(provider)
            if not ready:
                label = _PROVIDER_LABEL.get(provider, provider)
                env_var = _API_KEY_ENV_VAR.get(provider, "API_KEY")
                print(f"  --auto-agent={provider} but {label} credentials not detected.")
                print(why)
                key = _prompt_for_api_key(provider)
                if key:
                    ready, why = _check_agent_ready(provider)
            if ready:
                print(
                    f"  {_PROVIDER_LABEL.get(provider, provider)} credentials confirmed — "
                    f"LLM loop will run after Phase-1."
                )
            else:
                print(
                    "  No credentials supplied — LLM loop will be SKIPPED.\n"
                    "  Phase-1 CPU fallback will still be installed below."
                )
                args.auto = False
        else:
            print(
                "  Tip: add --auto --auto-agent claude (or cursor) to drive these\n"
                "  components automatically with an LLM. Without --auto, Phase-1 CPU\n"
                "  fallback stubs will be installed; the TTNN ports must be written\n"
                "  by hand afterwards."
            )

    banner(f"Step 4/6  Phase-1 unblock: autofill NEW stubs with torch fallback")

    llm_unavailable = not getattr(args, "auto", False)
    force_fallback = bool(getattr(args, "force_fallback", False))
    risky_native_stubs: List[Tuple[str, Path]] = []
    if llm_unavailable:
        from .bringup_loop import find_demo_dir as _find_demo_dir

        demo_dir_probe = _find_demo_dir(MODEL)
        if demo_dir_probe is not None:
            status_probe = demo_dir_probe / "bringup_status.json"
            if status_probe.is_file():
                try:
                    plan_data = json.loads(status_probe.read_text())
                except Exception:
                    plan_data = {}
                for comp in plan_data.get("components", []) or []:
                    if comp.get("status") != "NEW":
                        continue
                    safe = _safe_id(comp.get("name", ""))
                    stub_path = demo_dir_probe / "_stubs" / f"{safe}.py"
                    if not stub_path.is_file():
                        continue
                    try:
                        text = stub_path.read_text(errors="ignore")
                    except Exception:
                        continue
                    if "_get_torch_submodule" not in text:
                        risky_native_stubs.append((comp.get("name", safe), stub_path))

    if risky_native_stubs:
        if force_fallback:
            print(
                f"  --force-fallback set: {len(risky_native_stubs)} native stub(s) "
                f"will be overwritten with the CPU torch fallback so the demo can "
                f"reach a known-good baseline without an LLM."
            )
            for name, path in risky_native_stubs:
                try:
                    rel = safe_relative_to_root(path)
                except Exception:
                    rel = path
                print(f"    - {name} ({rel})")
        else:
            print(
                "  WARNING: previously LLM- or hand-edited native ttnn stub(s) are "
                "on disk and NO LLM is available to repair them if they hang:"
            )
            for name, path in risky_native_stubs:
                try:
                    rel = safe_relative_to_root(path)
                except Exception:
                    rel = path
                mtime = ""
                try:
                    import datetime as _dt

                    mtime = (
                        " (last modified "
                        + _dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                        + ")"
                    )
                except Exception:
                    pass
                print(f"    - {name}  ->  {rel}{mtime}")
            print(
                "\n  If pytest hangs on one of these, the baseline run will block "
                f"for TT_PLANNER_PYTEST_TIMEOUT_S (default 900s) before being killed.\n"
                "  Recovery options:\n"
                "    (a) Restore the CPU torch fallback for these stubs (safe baseline):\n"
                f"          python -m scripts.tt_hw_planner up {MODEL} --force-fallback ...\n"
                "    (b) Provide LLM credentials and re-run with --auto so the loop\n"
                "        can iterate on the broken stubs and fix them:\n"
                f"          export ANTHROPIC_API_KEY=...\n"
                f"          python -m scripts.tt_hw_planner up {MODEL} --auto --auto-agent claude ...\n"
            )

    _op_synth_effective = (
        False
        if bool(getattr(args, "no_op_synth", False))
        else (True if (bool(getattr(args, "auto", False)) or bool(getattr(args, "op_synth", False))) else False)
    )
    if _op_synth_effective and not bool(getattr(args, "op_synth", False)):
        print(
            "  [auto] op-synth enabled by default for --auto (surgical "
            "__call__-only LLM rewrites). Pass --no-op-synth to disable."
        )

    args.op_synth = _op_synth_effective
    autofill_argv = argparse.Namespace(
        model_id=MODEL,
        next=False,
        component=None,
        autofill=True,
        overwrite_autofill=bool(force_fallback),
        op_synth=_op_synth_effective,
        run_tests=False,
        no_emit_tests=True,
        overwrite_tests=False,
        keep_passing_stubs=True,
        format="text",
        synthesize=False,
        synthesize_component=None,
        llm_provider=None,
        llm_model=None,
        llm_endpoint=None,
        llm_max_retries=2,
        llm_dry_run=False,
        no_fetch_upstream=False,
        emit_prompts=False,
        apply_response=None,
        handoff_to_chat=False,
        apply_all_responses=False,
        list_synth_targets=False,
        # Thread box/mesh through so cmd_bringup's child cmd_up call
        # has them — without this, the wrapped cmd_up crashes with
        # "'Namespace' object has no attribute 'box'" mid-cascade.
        box=BOX,
        mesh=getattr(args, "mesh", None),
    )
    # IMPORTANT: cli.py defines its OWN `cmd_bringup` (the brain-
    # orchestrated auto-up wrapper at line ~6827) which shadows the
    # import from .commands.bringup at the top of this file. Step 4/6
    # autofill needs the PER-COMPONENT dispatcher in commands/bringup.py
    # (which dispatches on args.autofill / args.next / etc.), NOT the
    # wrapper — calling the wrapper here re-enters cmd_up recursively
    # and loops back through Step 0-6 / PCC gate / escalation, wasting
    # time and potentially blowing the call stack. Import the
    # per-component dispatcher directly to bypass the shadowing.
    from .commands.bringup import cmd_bringup as _cmd_bringup_per_component

    try:
        rc = _cmd_bringup_per_component(autofill_argv)
        if rc not in (0, None):
            print(
                f"  autofill returned non-zero ({rc}) — aborting before "
                f"prepare and LLM auto-iterate loop. Inspect the stubs under "
                f"models/demos/.../_stubs/ and re-run after fixing.",
                file=sys.stderr,
            )
            return rc
    except Exception as exc:
        # Layered setup-step recovery (rule registry → LLM allowlist
        # fallback). Same shape as the scaffold-step wiring above.
        # Rules catch the recurring shapes we've already seen
        # (AutoModel cascade for trust_remote_code Phi3 configs,
        # mkdir-missing-parent on fresh demo dirs). LLM handles
        # novel ones via the action allowlist.
        from ._cli_helpers.setup_step_recovery import (
            RecoveryAction as _RA,
            run_setup_step_recovery as _run_setup_recovery,
        )

        _proposal = _run_setup_recovery(
            exc=exc,
            step_name="step4_autofill",
            work_dir=Path.cwd(),
            repo_root=BRINGUP_ROOT(),
            workspace_summary=(
                f"model_id={MODEL!r} op_synth={_op_synth_effective} " f"force_fallback={force_fallback}"
            ),
        )
        if _proposal is not None and _proposal.action != _RA.CANNOT_RECOVER:
            print(f"  autofill recovery: {_proposal.label()} → retrying autofill once")
            try:
                rc = _cmd_bringup_per_component(autofill_argv)
                if rc not in (0, None):
                    print(
                        f"  autofill STILL returned non-zero after recovery ({rc}); " f"aborting.",
                        file=sys.stderr,
                    )
                    return rc
            except Exception as retry_exc:
                print(f"  autofill retry failed: {retry_exc} — aborting.", file=sys.stderr)
                return 2
        else:
            print(f"  autofill failed: {exc} — aborting.", file=sys.stderr)
            return 2

    banner(f"Step 5/6  Build the runnable pytest invocation (prepare)")
    auto_on = getattr(args, "auto", False)
    skip_baseline = auto_on and args.execute
    if skip_baseline:
        print(
            "  --auto is set: skipping the baseline pytest run. The LLM "
            "iteration loop runs pytest itself per-iteration with a hard "
            "wall-clock timeout, so running it again here is wasted time "
            "(and a poisoned stub from a prior session would hang the "
            "whole run before the LLM got a chance to fix it).",
            file=sys.stderr,
        )
    prepare_argv = argparse.Namespace(
        model_id=MODEL,
        box=BOX,
        mesh=getattr(args, "mesh", None),
        dtype=getattr(args, "dtype", None),
        batch=getattr(args, "batch", 1),
        max_seq_len=getattr(args, "max_seq_len", 1024),
        max_generated_tokens=getattr(args, "max_generated_tokens", 200),
        accuracy=getattr(args, "accuracy", False),
        no_trace=getattr(args, "no_trace", False),
        no_paged_attention=getattr(args, "no_paged_attention", False),
        no_instruct=getattr(args, "no_instruct", False),
        format="text",
        write_script=None,
        execute=(args.execute and not skip_baseline),
        download_first=args.download_first,
        strict=args.strict,
        allow_port=False,
    )
    try:
        rc = cmd_prepare(prepare_argv)
    except Exception as exc:
        print(f"  prepare failed: {exc}", file=sys.stderr)
        return 2
    _PLANNING_FAILURE_RC = 2
    if rc not in (0, None):
        recoverable_by_llm = getattr(args, "auto", False) and rc != _PLANNING_FAILURE_RC
        if recoverable_by_llm:
            print(
                f"  prepare exited with code {rc} (PCC tests failed). "
                f"Handing off to the LLM auto-iterate loop below to fix the "
                f"failures — this is the loop's whole purpose.",
                file=sys.stderr,
            )
        else:
            reason = (
                "planning/configuration failure (mesh/arch/box mismatch, " "non-executable plan, or strict refusal)"
                if rc == _PLANNING_FAILURE_RC
                else "PCC tests failed and --auto is not set, so no LLM is " "available to fix them"
            )
            print(
                f"  prepare returned non-zero exit code {rc} — aborting before " f"LLM auto-iterate loop: {reason}.",
                file=sys.stderr,
            )
            return rc
    elif getattr(args, "auto", False):
        seed_report = _parse_pytest_report()
        _, seed_smoke = _auto_iteration_blockers(MODEL)
        if bool(seed_report.get("all_passed", False)) and seed_smoke:
            smoke_txt = ", ".join(seed_smoke)
            print(
                "  prepare/execute passed, but Phase-1 SMOKE tests are still present "
                "(no real PCC validation yet).\n"
                f"  Phase-1 SMOKE tests: {smoke_txt}\n"
                "  continuing --auto to synthesize NEW components and emit real PCC tests."
            )

    if getattr(args, "auto", False):
        provider = (getattr(args, "auto_agent", None) or "cursor").lower()
        if provider not in ("cursor", "claude"):
            banner(f"AUTO-ITERATE: unknown --auto-agent {provider!r}")
            print("  Supported: cursor, claude", file=sys.stderr)
            return 2

        ready, msg = _check_agent_ready(provider)
        if not ready:
            label = _PROVIDER_LABEL.get(provider, provider)
            env_var = _API_KEY_ENV_VAR.get(provider, "API_KEY")
            banner(f"--auto: {label} credentials not detected")
            print(msg)
            key = _prompt_for_api_key(provider)
            if key:
                ready, msg = _check_agent_ready(provider)

        if not ready:
            env_var = _API_KEY_ENV_VAR.get(provider, "API_KEY")
            label = _PROVIDER_LABEL.get(provider, provider)
            banner("LLM iteration not possible — no API keys provided")
            print(
                f"  CPU-fallback scaffold is already installed for {MODEL}, but "
                f"the LLM auto-iterate loop (which converts fallback to native "
                f"ttnn) was SKIPPED because no {label} key was supplied. "
                f"Bring-up is NOT yet complete on the device.\n\n"
                f"  To finish bring-up, set the API key and re-run with --auto:\n"
                f"    export {env_var}=<your-key>\n"
                f"    python -m scripts.tt_hw_planner up {MODEL} \\\n"
                f"        --box {BOX} --execute --auto --auto-agent {provider}\n\n"
                f"  Or switch providers:\n"
                f"    --auto-agent {'claude' if provider == 'cursor' else 'cursor'}\n\n"
                f"  Diagnostic from CLI readiness check:\n"
            )
            print(msg, file=sys.stderr)
            return 0

        agent_bin = msg
        from .bringup_loop import find_demo_dir

        demo_dir = find_demo_dir(MODEL)
        if demo_dir is None:
            print(f"  no scaffolded demo dir found for {MODEL}; cannot auto-iterate.", file=sys.stderr)
            return 0
        model_alias = getattr(args, "auto_model", None)
        if not model_alias:
            model_alias = "opus" if provider == "claude" else "sonnet-4"
        model_light, model_heavy, model_super_heavy = _resolve_tiered_model_aliases(
            provider=provider,
            auto_model=model_alias,
            auto_model_light=getattr(args, "auto_model_light", None),
            auto_model_heavy=getattr(args, "auto_model_heavy", None),
            auto_model_super_heavy=getattr(args, "auto_model_super_heavy", None),
            auto_model_tiered=bool(getattr(args, "auto_model_tiered", False)),
        )
        if model_light or model_heavy or model_super_heavy:
            _super_label = f" → super_heavy={model_super_heavy}" if model_super_heavy else ""
            print(
                f"  [auto:{provider}] tiered model switching enabled: "
                f"light={model_light or model_alias}, heavy={model_heavy or model_alias}{_super_label}"
            )

        seed_report_raw = _parse_pytest_report()
        seed_report = _scope_report_to_demo(seed_report_raw, demo_dir)
        seed_ungrad, seed_smoke = _auto_iteration_blockers(MODEL)
        strict_native = not bool(getattr(args, "accept_fallback", False))
        seed_all_passed = bool(seed_report.get("all_passed", False))

        if seed_all_passed:
            saw_any_demo_test = bool(seed_report.get("per_test"))
            if not saw_any_demo_test:
                seed_all_passed = False
        if strict_native:
            already_done = seed_all_passed and not seed_ungrad and not seed_smoke
            done_msg = "model already runs natively on TT hardware"
            loop_goal = "Auto-iterate until model runs natively on TT hardware"
        else:
            already_done = seed_all_passed and not seed_smoke
            done_msg = (
                "model already runs end-to-end on TT hardware "
                "(--accept-fallback: some components are CPU fallback — see summary)"
                if seed_ungrad
                else "model already runs natively on TT hardware"
            )
            loop_goal = (
                "Auto-iterate until all PCC tests pass on TT hardware " "(--accept-fallback: CPU wrappers permitted)"
            )
        if already_done:
            banner(f"Step 6/6  Bring-up done — {done_msg}")
            _print_bringup_summary(MODEL, box=BOX, sep=sep)
            return 0

        if (getattr(args, "engine", "cc") or "cc") == "cc":
            from ._cli_helpers.bringup_cc import run_bringup_cc
            from .bringup_loop import find_demo_dir

            _dd = find_demo_dir(MODEL)
            if _dd is None:
                print("ERROR: --engine cc requires a scaffolded demo (run bring-up first).", file=sys.stderr)
                return 2
            banner("Step 6/6  Bring-up (cc engine) — harness loop on the per-component gate")
            _cc_rc = run_bringup_cc(
                model_id=MODEL,
                demo_dir=_dd,
                agent_bin=(getattr(args, "auto_agent_bin", None) or "claude"),
                mesh=getattr(args, "mesh", None),
                max_attempts=getattr(args, "auto_max_attempts_per_component", 2),
            )
            try:
                from .run_report import emit_run_report

                _rp = emit_run_report(MODEL, _dd, converged=(_cc_rc == 0))
                if _rp:
                    print(f"  [run-report] wrote {_rp}")
            except Exception:
                pass
            return _cc_rc


from .commands.promote import cmd_promote  # noqa: F401


from .commands.prepare import cmd_prepare  # noqa: F401


from .commands.list_meshes import cmd_list_meshes  # noqa: F401  (re-export)
from .commands.sync_registry import cmd_sync_registry  # noqa: F401


def _load_bringup_status(model_id: str) -> Tuple[Path, Dict]:
    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        raise FileNotFoundError(f"no demo dir for {model_id!r} — run `scaffold --apply` first")
    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"no bringup_status.json at {status_path} — run `scaffold --apply` first")
    return demo_dir, json.loads(status_path.read_text())


def _select_op_synth_targets(status: Dict, *, component: Optional[str], include_adapt: bool) -> List[Dict]:
    # ADAPT removed 2026-05-31. include_adapt kept as a kwarg for
    # callsite compatibility; with ADAPT gone it's always a no-op
    # (the only target status is NEW). Legacy bringup_status.json
    # entries with status="ADAPT" are still accepted when include_adapt
    # is set, to keep stale-overlay loads working.
    components = status.get("components", []) or []
    allowed_status = {"NEW"}
    if include_adapt:
        allowed_status.add("ADAPT")
    if component:
        return [c for c in components if c.get("name") == component]
    return [c for c in components if c.get("status") in allowed_status]


from .commands.capture_inputs import cmd_capture_inputs  # noqa: F401


from .commands.op_synth import cmd_op_synth  # noqa: F401
from .commands.emit_e2e import cmd_emit_e2e  # noqa: F401
from .commands.optimize import cmd_optimize  # noqa: F401
from .commands.auto_onboard import cmd_auto_onboard  # noqa: F401


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    argv = list(sys.argv[1:] if argv is None else argv)

    SUBCOMMANDS = {
        "plan",
        "compat",
        "scaffold",
        "prepare",
        "up",
        "promote",
        "bringup",
        "op-synth",
        "capture-inputs",
        "list-meshes",
        "auto-onboard",
        "-h",
        "--help",
    }
    if argv and argv[0] not in SUBCOMMANDS and ("/" in argv[0] or argv[0].startswith("-")):
        argv = ["plan"] + argv

    parser = argparse.ArgumentParser(
        prog="tt_hw_planner",
        description="Pre-flight memory planner for Tenstorrent hardware.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pup = sub.add_parser(
        "up",
        help=(
            "ONE-SHOT bring-up to native TTNN: plan + compat + scaffold + "
            "autofill + prepare, then (with --auto) iterate with the coding "
            "agent until every NEW component runs natively on TT hardware. "
            "Bring-up means running on device, not via CPU fallback. Use "
            "--execute to actually run on hw."
        ),
    )
    pup.add_argument("model_id", help="HuggingFace model id, e.g. facebook/sam2-hiera-tiny")
    pup.add_argument(
        "--box",
        default="QB2",
        choices=[b.name for b in HARDWARE],
        help="target hardware (default: QB2)",
    )
    pup.add_argument(
        "--mesh",
        default=None,
        help=(
            "Override the mesh shape (e.g. '1,4' or '2x2'). Must be canonical "
            "for the chosen --box; otherwise `up` aborts with the list of "
            "valid shapes before scaffold/autofill/LLM run. Default: planner picks."
        ),
    )
    pup.add_argument(
        "--dtype",
        default=None,
        choices=list(DTYPE_BYTES.keys()),
        help="override dtype for prepare/execute and auto-iterate reruns",
    )
    pup.add_argument("--batch", type=int, default=1, help="batch size for prepare/execute (default: 1)")
    pup.add_argument("--max-seq-len", type=int, default=1024, help="max sequence length for prepare/execute")
    pup.add_argument(
        "--max-generated-tokens",
        type=int,
        default=200,
        help="max generated tokens for prepare/execute",
    )
    pup.add_argument(
        "--accuracy",
        action="store_true",
        help="use accuracy parametrization for prepare/execute and auto reruns",
    )
    pup.add_argument("--no-trace", action="store_true", help="disable trace for prepare/execute and auto reruns")
    pup.add_argument(
        "--no-paged-attention",
        action="store_true",
        help="disable paged attention for prepare/execute and auto reruns",
    )
    pup.add_argument(
        "--no-instruct",
        action="store_true",
        help="disable instruct/chat template for prepare/execute and auto reruns",
    )
    pup.add_argument(
        "--execute",
        action="store_true",
        help="actually run the prepared pytest on TT hardware (default: just print it)",
    )
    pup.add_argument(
        "--download-first",
        action="store_true",
        help="pre-download HuggingFace weights before executing (pairs with --execute)",
    )
    pup.add_argument(
        "--local-dir",
        default=None,
        metavar="PATH",
        help=(
            "Directory containing locally-downloaded HuggingFace weights. "
            "When set, the tool exports HF_HOME=PATH and HF_HUB_OFFLINE=1 for "
            "every subprocess pytest run, so HF loads from the local directory "
            "and never attempts a network call. Use after `huggingface-cli "
            "download <model_id> --local-dir <path>` for weights in a "
            "non-standard location. For the default cache (~/.cache/huggingface/hub/), "
            "no flag is needed — HF will pick it up automatically."
        ),
    )
    pup.add_argument(
        "--offline-hf",
        action="store_true",
        help=(
            "Force HuggingFace offline mode (HF_HUB_OFFLINE=1) without changing "
            "HF_HOME. Useful for reruns where the weights are already in the "
            "default cache and you want to guarantee no network attempt is made."
        ),
    )
    pup.add_argument(
        "--strict",
        action="store_true",
        help="abort on the first non-zero exit from a sub-step (default: continue on plan/compat warnings)",
    )
    pup.add_argument(
        "--no-env-fix",
        action="store_true",
        help=(
            "Do NOT auto-upgrade `transformers` when pre-flight detects the "
            "env can't load the model. Default: auto-upgrades from upstream "
            "main so the single command works for any HF model, including "
            "very new ones (e.g. sam2_video) the shipped transformers "
            "doesn't yet recognize. Use this flag in CI or sealed envs."
        ),
    )
    pup.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        help=(
            "Do NOT fetch the tenstorrent/tt-metal registry from the network; "
            "use the local checkout tree (may be stale). Default: remote-first "
            "sha-pinned sync so the mapping registry tracks upstream without a "
            "manual git pull (fixes-plan Point 2a). Also honored via "
            "TT_HW_PLANNER_OFFLINE=1."
        ),
    )
    pup.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Fully autonomous loop: after the first prepare/execute, invoke "
            "the configured coding-agent CLI (see --auto-agent) to write "
            "every NEW/ADAPT/DEMO file from the handoff, install them, "
            "re-run PCC tests on hardware, and feed failures back to the "
            "agent. Caps at --auto-max-iters. If no credentials are found, "
            "you will be prompted to paste an API key; pressing Enter "
            "skips the LLM loop and keeps the Phase-1 CPU fallback in place."
        ),
    )
    pup.add_argument(
        "--auto-agent",
        choices=("cursor", "claude"),
        default="cursor",
        help=(
            "Which coding-agent CLI drives the --auto loop. "
            "'cursor' uses Cursor's `agent` CLI (requires `agent login`). "
            "'claude' uses Anthropic's Claude Code CLI "
            "(requires ANTHROPIC_API_KEY or one-time `claude` OAuth login). "
            "Default: cursor."
        ),
    )
    pup.add_argument(
        "--auto-max-iters",
        type=int,
        default=5,
        help=(
            "Outer-loop retry budget. Each iteration: agent writes one file "
            "per failing component to `_synth_responses/`, tool applies it, "
            "tool runs pytest, traceback is fed back into the next iteration "
            "(default: 5)."
        ),
    )
    pup.add_argument(
        "--auto-model",
        default=None,
        help=(
            "Model alias the auto-agent uses. "
            "Defaults: 'sonnet-4' for cursor, 'sonnet' for claude. "
            "(See `agent --list-models` or `claude --help` for choices.) "
            "Ignored when --auto-model-tiered or --auto-model-light/--auto-model-heavy "
            "is set."
        ),
    )
    pup.add_argument(
        "--auto-model-light",
        default=None,
        help=(
            "Tiered mode: model alias used for simple / first-attempt iterations. "
            "When set (alone or with --auto-model-heavy), enables automatic "
            "Sonnet/Opus-style switching per iteration based on component "
            "complexity, failure class, and prior attempts. Falls back to "
            "--auto-model when --auto-model-heavy is unset."
        ),
    )
    pup.add_argument(
        "--auto-model-heavy",
        default=None,
        help=(
            "Tiered mode: model alias used for hard iterations (complex "
            "components, repeated failures, HANG / L1_OOM / "
            "PARTIAL_CPU_FALLBACK / DEVICE_NEEDS_RESET / TT_FATAL_OPAQUE). "
            "Falls back to --auto-model when --auto-model-light is unset."
        ),
    )
    pup.add_argument(
        "--auto-model-tiered",
        action="store_true",
        help=(
            "Shortcut: enable tiered model switching with provider defaults "
            "(claude: light=sonnet, heavy=opus; cursor: light=sonnet-4, "
            "heavy=opus). Equivalent to passing --auto-model-light + "
            "--auto-model-heavy with those defaults. Explicit --auto-model-light "
            "/ --auto-model-heavy override these defaults."
        ),
    )
    pup.add_argument(
        "--auto-agent-timeout",
        type=int,
        default=600,
        help=(
            "Wall-clock timeout per agent invocation in seconds (default: "
            "600 = 10 min). The agent should write one file per failing "
            "component and exit; the tool runs pytest. On timeout the agent "
            "process tree is killed and the outer loop continues."
        ),
    )
    pup.add_argument(
        "--auto-max-attempts-per-component",
        type=int,
        default=2,
        help=(
            "Per-component attempt cap (default: 2). The loop targets ONE "
            "component per iter and re-attempts it up to N times in a row. "
            "Once a component fails N times, the loop restores it to a "
            "stable CPU-fallback stub and moves on to the next ungraduated "
            "component. This guarantees forward progress: the loop never "
            "spends all its iterations on a single hopeless component, and "
            "the demo always converges to an end-to-end-running state "
            "(mix of native TTNN + CPU fallback) within --auto-max-iters."
        ),
    )
    pup.add_argument(
        "--accept-fallback",
        action="store_true",
        help=(
            "Allow the loop to converge with CPU-fallback wrappers in place. "
            "Default is strict: a PCC test that passes because the component "
            "delegates to torch on CPU validates nothing (it compares torch to "
            "itself after a bfloat16 roundtrip), so the loop will keep "
            "iterating until every NEW component runs natively in ttnn. Use "
            "this flag only when you explicitly want an end-to-end run that "
            "is partially on CPU."
        ),
    )

    # Iter-loop CLI flags (--parallel-agents, --auto-only-component,
    # --strict-pcc{,-max-iters,-tokens}, --escalate-on-pcc-fail,
    # --no-escalate-on-pcc-fail, --auto-model-super-heavy, --pcc-engine)
    # are defined by the shared helper so `pup` and `pprom` cannot drift.
    # See _cli_helpers/auto_iterate.py:add_iter_loop_cli_args for the
    # canonical list. DO NOT inline these here.
    from ._cli_helpers.auto_iterate import add_iter_loop_cli_args as _add_iter_loop_cli_args

    _add_iter_loop_cli_args(pup)

    pup.add_argument(
        "--allow-partial-cpu",
        action="store_true",
        help=(
            "Stop iterating when every component's PCC test passes, even if "
            "some `_apply_*` helpers still fall back to PyTorch on CPU at "
            "runtime. Default is STRICT: the loop continues until the "
            "compute split reaches 100%% on device (or partial-CPU "
            "components exhaust their attempt cap). Use this flag if you "
            "want to halt as soon as numerical correctness is established, "
            "even at e.g. 99%% on-device — useful for time-bounded bring-"
            "ups where the last few CPU ops are acceptable and the LLM "
            "budget is precious."
        ),
    )
    pup.add_argument(
        "--regen-demo-only",
        action="store_true",
        help=(
            "Skip the entire bring-up loop. Regenerate `demo.py` from the "
            "captured submodule inputs of the largest graduated component "
            "and run `pytest demo.py::test_demo` to verify it. Use this "
            "when an earlier `up --auto` already converged but you (or a "
            "git pull) lost / edited the demo file and want the auto-emit "
            "fresh."
        ),
    )
    pup.add_argument(
        "--accept-closest-backend",
        action="store_true",
        help=(
            "Bypass the backend-match gate. By default, `up --auto` "
            "REFUSES to scaffold a model whose model_type and "
            "pipeline_tag both miss every registered FamilyBackend, "
            "because the silent category-default fallback historically "
            "wasted LLM tokens iterating against the wrong template "
            "(e.g. SAM2 onto SegFormer's component decomposition). "
            "Pass this flag if you've manually verified that the "
            "closest-by-category backend is structurally similar "
            "enough. Recommended alternative: `auto-onboard <model_id>` "
            "to LLM-draft a real backend entry."
        ),
    )
    pup.add_argument(
        "--no-meta-plan",
        action="store_true",
        help=(
            "Disable the pre-loop meta-reasoning planner. By default, "
            "`up --auto` runs ONE LLM call before iterating to "
            "evaluate the bring-up plan as a whole (feasibility "
            "verdict, risks, cheaper alternatives). The verdict is "
            "ADVISORY only -- it never gates the bring-up -- so this "
            "flag is for users who want to skip the ~30s of LLM time "
            "or are running offline."
        ),
    )
    pup.add_argument(
        "--force-fallback",
        action="store_true",
        help=(
            "Force Phase-1 autofill to OVERWRITE any existing NEW stub with the "
            "torch CPU fallback, even if a previous run left a hand- or "
            "LLM-edited native ttnn stub on disk. Use this to recover from a "
            "poisoned native stub (e.g. one that hangs the device) when no LLM "
            "credentials are available to repair it. After --force-fallback "
            "completes, the demo runs end-to-end with the missing components "
            "computed on host CPU, and you can iterate further with --auto."
        ),
    )
    pup.add_argument(
        "--op-synth",
        action="store_true",
        dest="op_synth",
        help=(
            "Force-enable op-level autofill (this is the DEFAULT behavior "
            "when running with --auto; the flag is kept for explicit/"
            "non-auto invocations). Use the op-level classifier during "
            "Phase-1 autofill instead of the plain torch-fallback wrapper. "
            "For each NEW component, walk the live HF submodule, classify "
            "every leaf op against the ttnn primitive set (Linear/"
            "LayerNorm/Conv2d/Embedding/activation/...), and write a "
            "partial native TTNN stub where weights and deterministic op "
            "helpers (`_apply_*`) are pre-bound. `__call__` still falls "
            "back to torch so the smoke test passes; with --auto, the LLM "
            "is then asked to rewrite ONLY `__call__` using the pre-bound "
            "helpers — dramatically smaller unit of work per iteration. "
            "Components whose HF submodule cannot be resolved fall back "
            "to the torch wrapper transparently."
        ),
    )
    pup.add_argument(
        "--no-op-synth",
        action="store_true",
        dest="no_op_synth",
        help=(
            "Opt out of op-level autofill. Op-synth is enabled by default "
            "in --auto mode because the surgical __call__-only LLM rewrite "
            "it unlocks closes the long tail of components (per Tier 1 #1 "
            "of the auto-loop design notes). Pass --no-op-synth to fall "
            "back to plain torch-wrapper autofill + full-file rewrites."
        ),
    )
    pup.add_argument(
        "--no-kill-stale",
        action="store_true",
        help=(
            "Disable the pre-flight and per-iteration zombie-pytest sweep. "
            "By default, `up` scans for orphaned pytest processes that look "
            "like tt_hw_planner runs (matching `models/.../tests/pcc/test_*.py` "
            "in argv, owned by you, not a descendant of the current process), "
            "and reaps them via SIGTERM/SIGKILL on their PGID so they release "
            "the TT device lock (`CHIP_IN_USE_0_PCIe`). Pass this flag if you "
            "want to inspect such processes manually before they're killed."
        ),
    )
    pup.add_argument(
        "--no-device-reset",
        action="store_true",
        help=(
            "Disable automatic `tt-smi -r` recovery. By default, `up` runs "
            "`tt-smi -r 0,1,2,3` (a) after reaping an orphaned pytest process "
            "(its kernel-side IOMMU/sysmem mappings can persist and crash the "
            "next launch with `Proceeding could lead to undefined behavior`), "
            "and (b) if a focused pytest exits and its output contains the UMD "
            "sysmem-mismatch signature (`pin_or_map_sysmem_to_device`), in "
            "which case the same pytest is retried ONCE post-reset. Hard-cap: "
            "3 resets per planner process. Disable via this flag or the "
            "`TT_PLANNER_NO_DEVICE_RESET=1` env var."
        ),
    )
    pup.add_argument(
        "--isolation",
        choices=["worktree", "none"],
        default="worktree",
        help=(
            "Run the bring-up in an isolated git worktree (default: 'worktree'). "
            "Pytest + LLM agents operate inside /tmp/tt_hw_planner_<slug>/, "
            "original repo working tree stays untouched. On success, LLM edits get captured "
            "into scripts/tt_hw_planner/overlays/<model>/ and the worktree is destroyed. "
            "On failure, the worktree is preserved for debug. Pass --isolation none to opt out."
        ),
    )
    pup.set_defaults(func=cmd_up)

    # `auto-up` is a zero-flag entry-point: hands the model_id to `up`
    # with all brain-orchestrated defaults locked in. Power users keep
    # using `up` with explicit flags; everyone else just types
    #     python -m scripts.tt_hw_planner auto-up <model_id>
    # and the orchestrator (brain G8) handles every decision.
    paut = sub.add_parser(
        "auto-up",
        help=(
            "Low-flag entry point: hand a HuggingFace model id plus target "
            "hardware (--box and --mesh, both required) and the brain does "
            "the rest. Sets sane defaults for --auto, --auto-agent, tiered "
            "model selection, iter budget, and per-component cap so the "
            "orchestrator drives every other decision. Power users can fall "
            "back to `up` with explicit flags."
        ),
    )
    paut.add_argument("model_id", help="HuggingFace model id, e.g. facebook/sam2-hiera-tiny")
    paut.add_argument(
        "--box",
        required=True,
        choices=[b.name for b in HARDWARE],
        help="target hardware (required)",
    )
    paut.add_argument(
        "--mesh",
        required=True,
        help="Mesh shape, e.g. '1,4' or '2x2' (required); must be canonical for --box.",
    )
    paut.set_defaults(func=cmd_bringup)

    pprom = sub.add_parser(
        "promote",
        help=(
            "Resume bring-up scoped to remaining CPU-fallback components. "
            "Use when `up --auto` ran out of iters with some components still "
            "on torch-reference fallback — `promote` skips re-scaffolding and "
            "just runs the auto-iterate loop targeting those components. "
            "Same convergence contract as `up`: native ttnn, PCC >= 0.99."
        ),
    )
    pprom.add_argument("model_id", help="HuggingFace model id (must already be bring-up'd via `up`)")

    # Iter-loop CLI flags shared with `pup` — single source of truth.
    # See _cli_helpers/auto_iterate.py:add_iter_loop_cli_args. Without
    # this call, promote silently drops --parallel-agents,
    # --auto-only-component, --auto-model-super-heavy, --strict-pcc,
    # --escalate-on-pcc-fail, --pcc-engine, etc.
    _add_iter_loop_cli_args(pprom)

    pprom.add_argument(
        "--box",
        required=True,
        choices=[b.name for b in HARDWARE],
        help="target hardware (required)",
    )
    pprom.add_argument(
        "--mesh",
        required=True,
        help="mesh shape (e.g. '1,4') (required); must be canonical for --box.",
    )
    pprom.add_argument(
        "--dtype",
        default=None,
        choices=list(DTYPE_BYTES.keys()),
        help="override dtype for reruns",
    )
    pprom.add_argument("--batch", type=int, default=1)
    pprom.add_argument("--max-seq-len", type=int, default=1024)
    pprom.add_argument("--max-generated-tokens", type=int, default=200)
    pprom.add_argument("--accuracy", action="store_true")
    pprom.add_argument("--no-trace", action="store_true")
    pprom.add_argument("--no-paged-attention", action="store_true")
    pprom.add_argument("--no-instruct", action="store_true")
    pprom.add_argument("--download-first", action="store_true")
    pprom.add_argument("--strict", action="store_true")
    pprom.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Drive native-TTNN synthesis via the coding-agent CLI. ON BY "
            "DEFAULT for promote (like auto-up); this flag is kept for "
            "explicitness/back-compat. Pass --no-auto to only list the "
            "fallback components and next steps."
        ),
    )
    pprom.add_argument(
        "--no-auto",
        dest="auto",
        action="store_false",
        help="Opt out of auto-drive: just list fallback components + next steps (status only).",
    )
    pprom.add_argument(
        "--auto-agent",
        choices=("cursor", "claude"),
        default="claude",
        help="Which coding-agent CLI drives the auto loop (default: claude).",
    )
    pprom.add_argument(
        "--auto-max-iters",
        type=int,
        default=24,
        help="Cap for the promote self-iteration loop (default: 24).",
    )
    pprom.add_argument("--auto-model", default=None)
    pprom.add_argument(
        "--auto-model-light",
        default=None,
        help=("Tiered mode: light model alias. See `up --help` for full semantics."),
    )
    pprom.add_argument(
        "--auto-model-heavy",
        default=None,
        help=("Tiered mode: heavy model alias. See `up --help` for full semantics."),
    )
    pprom.add_argument(
        "--auto-model-tiered",
        action="store_true",
        help=(
            "Shortcut: enable tiered model switching with provider defaults "
            "(claude: sonnet -> opus). See `up --help` for full semantics."
        ),
    )
    pprom.add_argument(
        "--auto-agent-timeout",
        type=int,
        default=600,
        help="Wall-clock timeout per agent invocation in seconds (default: 600).",
    )
    pprom.add_argument(
        "--auto-max-attempts-per-component",
        type=int,
        default=5,
        help=(
            "Per-component attempt cap (default: 5 — matches auto-up's "
            "brain-G8 locked default). After N failures, the component is "
            "restored to CPU fallback and the loop moves on. "
            "Same semantics as `up --auto-max-attempts-per-component`."
        ),
    )
    pprom.add_argument(
        "--allow-partial-cpu",
        action="store_true",
        help=(
            "Stop iterating when every component's PCC test passes, even if "
            "some `_apply_*` helpers still fall back to PyTorch on CPU at "
            "runtime. Same semantics as `up --allow-partial-cpu`."
        ),
    )
    pprom.add_argument(
        "--regen-demo-only",
        action="store_true",
        help=(
            "Skip the auto-iterate loop. Regenerate `demo.py` from captured "
            "submodule inputs and verify via `pytest demo.py::test_demo`. "
            "Same semantics as `up --regen-demo-only`."
        ),
    )
    pprom.add_argument(
        "--no-kill-stale",
        action="store_true",
        help=(
            "Disable the per-iteration zombie-pytest sweep that reaps orphan " "TT-device holders (default: enabled)."
        ),
    )
    pprom.add_argument(
        "--no-device-reset",
        action="store_true",
        help=(
            "Disable automatic `tt-smi -r` recovery (after orphan-kill or on "
            "UMD `pin_or_map_sysmem_to_device` errors). See `up --help` for "
            "full details. Hard-cap 3 resets per process; also honors the "
            "`TT_PLANNER_NO_DEVICE_RESET=1` env var."
        ),
    )
    pprom.add_argument(
        "--op-synth",
        action="store_true",
        dest="op_synth",
        help=(
            "Re-autofill CPU-fallback components with op-level partial "
            "stubs BEFORE entering the auto-iterate loop. Same mechanism "
            "as `up --op-synth`, but opt-in here because `promote` "
            "operates on stubs that may already have user edits — the "
            "default is to leave them untouched and let the LLM rewrite "
            "the full file. When set, components whose stub is still a "
            "torch wrapper get a fresh op-synth partial port so the "
            "subsequent LLM rounds get surgical __call__-only rewrites."
        ),
    )
    pprom.add_argument(
        "--no-op-synth",
        action="store_true",
        dest="no_op_synth",
        help=(
            "Symmetric with --no-op-synth on `up`. Kept for scripted "
            "pipelines that pass the flag through unconditionally; in "
            "`promote` it is a no-op because op-synth defaults to off here."
        ),
    )
    pprom.set_defaults(func=cmd_promote, auto=True, auto_model_tiered=True)

    pp = sub.add_parser("plan", help="memory-budget recommendation (default)")
    pp.add_argument("model_id", help="HuggingFace model id, e.g. Qwen/Qwen3-32B")
    pp.add_argument("--batch", type=int, default=1)
    pp.add_argument("--seq", type=int, default=8192)
    pp.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    pp.add_argument("--dtype", action="append", default=[], choices=list(DTYPE_BYTES.keys()))
    pp.add_argument("--box", action="append", default=[], choices=[b.name for b in HARDWARE])
    pp.add_argument(
        "--all-meshes", action="store_true", help="Show every canonical mesh per box, not just the largest TP."
    )
    pp.add_argument("--explore-pp", action="store_true", help="Also enumerate TP×PP combinations (e.g. T3K TP=4,PP=2).")
    pp.add_argument("--format", choices=["table", "json", "markdown"], default="table")
    pp.add_argument("--no-overhead-detail", action="store_true")
    pp.set_defaults(func=cmd_plan)

    pcompat = sub.add_parser(
        "compat",
        help="list which TT building blocks + kernel constraints the model needs",
    )
    pcompat.add_argument("model_id", help="HuggingFace model id, e.g. Qwen/Qwen3-32B")
    pcompat.add_argument("--format", choices=["table", "json"], default="table")
    pcompat.add_argument("--verbose", action="store_true", help="show notes for every block + every kernel finding")
    pcompat.add_argument(
        "--skip-kernel-check",
        action="store_true",
        help="only check building-block availability, not kernel constraints",
    )
    pcompat.add_argument(
        "--tp-grid", type=int, nargs="+", default=None, help="TP values to check for divisibility (default: 1 2 4 8 32)"
    )
    pcompat.add_argument(
        "--mesh",
        default=None,
        help="mesh shape e.g. 2x2; when given, prints the selected TP x DP split for that chip count",
    )
    pcompat.set_defaults(func=cmd_compat)

    pscaf = sub.add_parser(
        "scaffold",
        help="generate first-draft port (table entries + per-model JSONs) for a READY model",
    )
    pscaf.add_argument("model_id", help="HuggingFace model id of the NEW model to port")
    pscaf.add_argument(
        "--apply",
        action="store_true",
        help="actually write the changes to the working tree (default: dry-run)",
    )
    pscaf.add_argument(
        "--format",
        choices=["text", "patch", "json"],
        default="text",
        help="text: human-readable plan + diff; patch: emit `git apply`-compatible diff; json: structured",
    )
    pscaf.add_argument("--no-diff", action="store_true", help="omit the inline diff in text format")
    pscaf.set_defaults(func=cmd_scaffold)

    pbup = sub.add_parser(
        "bringup",
        help=(
            "automated NEW-stub bring-up loop: emit per-component PCC test templates, "
            "optionally run them, and auto-remove stubs whose tests pass. Generalised "
            "for any model already scaffolded via `scaffold --apply`."
        ),
    )
    pbup.add_argument("model_id", help="HuggingFace model id of a previously-scaffolded model")
    pbup.add_argument(
        "--next",
        action="store_true",
        help="print concrete instructions for the next NEW component to implement and exit",
    )
    pbup.add_argument(
        "--component",
        default=None,
        help="when used with --next, target a specific NEW component instead of the first one",
    )
    pbup.add_argument(
        "--autofill",
        action="store_true",
        help=(
            "Phase-1 unblock: replace each NEW-stub body with a torch-fallback "
            "that invokes the corresponding HF PyTorch submodule on host CPU. "
            "Lets the model run end-to-end immediately while you incrementally "
            "port each component to TTNN in Phase 2."
        ),
    )
    pbup.add_argument(
        "--overwrite-autofill",
        action="store_true",
        help="re-emit autofill bodies even for stubs that already have user edits",
    )
    pbup.add_argument(
        "--op-synth",
        action="store_true",
        dest="op_synth",
        help=(
            "When emitting autofill bodies (--autofill), walk the live HF "
            "submodule for each NEW component, classify every leaf op "
            "(Linear/LayerNorm/Conv2d/Embedding/activation/...), and write "
            "a partial native TTNN stub where weights and op-REUSE/op-ADAPT "
            "helpers are pre-bound. `__call__` still torch-falls-back so "
            "the smoke test passes immediately; the LLM only needs to "
            "rewrite `__call__` and fill any op-NEW gaps. Components whose "
            "HF submodule cannot be resolved fall back to the plain torch "
            "wrapper, so every NEW component still gets a runnable stub."
        ),
    )
    pbup.add_argument(
        "--no-op-synth",
        action="store_true",
        dest="no_op_synth",
        help=(
            "Opt out of op-level autofill (default-off here in `bringup`, "
            "kept symmetric with the same flag on `up`). When invoked from "
            "`up --auto` op-synth is on by default — pass --no-op-synth "
            "there to disable; this flag exists on `bringup` so that "
            "scripted pipelines can pass it through without conditionally "
            "stripping it from the argv."
        ),
    )
    pbup.add_argument(
        "--run-tests",
        action="store_true",
        help="actually run pytest on the per-component PCC tests (default: emit-only)",
    )
    pbup.add_argument(
        "--no-emit-tests",
        action="store_true",
        help="skip emitting PCC test templates (only refresh/inspect)",
    )
    pbup.add_argument(
        "--overwrite-tests",
        action="store_true",
        help="overwrite existing per-component PCC test files (default: preserve)",
    )
    pbup.add_argument(
        "--keep-passing-stubs",
        action="store_true",
        help="do not auto-remove stubs whose PCC test passes (default: remove)",
    )
    pbup.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="text: human-readable per-stub summary; json: structured for tooling",
    )
    pbup.add_argument(
        "--list-synth-targets",
        action="store_true",
        dest="list_synth_targets",
        help=(
            "Read scaffold's bring-up plan and print the REUSE / ADAPT / NEW "
            "split, then exit. The LLM is NOT contacted. Use this to see "
            "exactly which components synthesis would touch before opting in."
        ),
    )
    pbup.add_argument(
        "--emit-prompts",
        action="store_true",
        dest="emit_prompts",
        help=(
            "BYO-LLM: write one self-contained prompt .md per NEW component "
            "to `<demo>/_synth_prompts/`. No API key required. Paste each "
            "into the chat assistant of your choice (Cursor's built-in chat, "
            "ChatGPT, Claude, etc.), then feed the response back via "
            "`--apply-response`."
        ),
    )
    pbup.add_argument(
        "--apply-response",
        nargs=2,
        metavar=("COMPONENT", "RESPONSE_FILE"),
        default=None,
        dest="apply_response",
        help=(
            "BYO-LLM: ingest a chat assistant's response from RESPONSE_FILE "
            "and install it as the stub for COMPONENT. Strips markdown "
            "fences, syntax-checks, backs up the previous stub, and adds "
            "the machine-generated banner. Gated to NEW components only."
        ),
    )
    pbup.add_argument(
        "--handoff-to-chat",
        action="store_true",
        dest="handoff_to_chat",
        help=(
            "Generate ONE master prompt file covering every NEW component, "
            "with explicit file-write instructions for the chat agent. The "
            "user @-mentions this file in their Cursor chat once; the agent "
            "writes all response files in one go. Pair with "
            "`--apply-all-responses` to bulk-install everything afterwards."
        ),
    )
    pbup.add_argument(
        "--apply-all-responses",
        action="store_true",
        dest="apply_all_responses",
        help=(
            "Scan `<demo>/_synth_responses/` and apply every `<safe>.py` that "
            "matches a NEW component. Same gating and safety as "
            "`--apply-response`, just batched."
        ),
    )
    pbup.add_argument(
        "--synthesize",
        action="store_true",
        help=(
            "Phase-2: use an LLM to write TTNN module bodies for NEW stubs "
            "ONLY. Gated by scaffold's bring-up plan: REUSE and ADAPT "
            "components are never sent to the LLM. Requires OPENAI_API_KEY "
            "or ANTHROPIC_API_KEY in the env. Combine with --run-tests for "
            "an automated PCC-driven retry loop."
        ),
    )
    pbup.add_argument(
        "--synthesize-component",
        default=None,
        help=(
            "with --synthesize, target a single NEW component by name instead " "of synthesizing every NEW stub at once"
        ),
    )
    pbup.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic"],
        default=None,
        help="override provider auto-detection (default: pick whichever env API key is set)",
    )
    pbup.add_argument(
        "--llm-model",
        default=None,
        help="override the LLM model id (e.g. gpt-4o, claude-3-7-sonnet-latest, qwen2.5-coder)",
    )
    pbup.add_argument(
        "--llm-endpoint",
        default=None,
        help=(
            "override the API base URL (use this to point at a self-hosted "
            "OpenAI-compatible server like Ollama: http://localhost:11434/v1)"
        ),
    )
    pbup.add_argument(
        "--llm-max-retries",
        type=int,
        default=2,
        help="number of retry attempts after the initial synthesis call (default 2; only meaningful with --run-tests)",
    )
    pbup.add_argument(
        "--llm-dry-run",
        action="store_true",
        help="assemble prompts and write the audit log but do not call the LLM (cost-free preview)",
    )
    pbup.add_argument(
        "--no-fetch-upstream",
        action="store_true",
        dest="no_fetch_upstream",
        help=(
            "disable the upstream-source fallback (no network fetch). By default, "
            "if this env's transformers can't load the model, the tool fetches the "
            "matching `modeling_<arch>.py` from huggingface/transformers GitHub so "
            "the prompt is still complete. Use this for offline runs."
        ),
    )
    pbup.set_defaults(func=cmd_bringup)

    pprep = sub.add_parser(
        "prepare",
        help="emit ready-to-run env + pytest invocation for the recommended box",
    )
    pprep.add_argument("model_id", help="HuggingFace model id, e.g. Qwen/Qwen3-32B")
    pprep.add_argument(
        "--box",
        default=None,
        choices=[b.name for b in HARDWARE],
        help="override the planner's box pick",
    )
    pprep.add_argument("--mesh", default=None, help="override the planner's mesh (requires --box, e.g. 1,4)")
    pprep.add_argument(
        "--dtype",
        default=None,
        choices=list(DTYPE_BYTES.keys()),
        help="override the planner's dtype pick",
    )
    pprep.add_argument("--batch", type=int, default=1, help="pytest --batch_size (default 1)")
    pprep.add_argument("--max-seq-len", type=int, default=1024, help="pytest --max_seq_len (default 1024)")
    pprep.add_argument(
        "--max-generated-tokens", type=int, default=200, help="pytest --max_generated_tokens (default 200)"
    )
    pprep.add_argument(
        "--accuracy",
        action="store_true",
        help="use the accuracy parametrization instead of performance",
    )
    pprep.add_argument("--no-trace", action="store_true", help="disable --enable_trace (slower; needed for accuracy)")
    pprep.add_argument("--no-paged-attention", action="store_true", help="disable paged attention")
    pprep.add_argument("--no-instruct", action="store_true", help="use raw completion path instead of chat template")
    pprep.add_argument("--format", choices=["text", "script", "json"], default="text")
    pprep.add_argument(
        "--write-script",
        default=None,
        metavar="PATH",
        help="also write a self-contained bash script to PATH",
    )
    pprep.add_argument(
        "--execute",
        action="store_true",
        help="run the emitted pytest command in-process (requires a runnable plan)",
    )
    pprep.add_argument(
        "--download-first",
        action="store_true",
        help="pre-download Hugging Face model weights before --execute " "(routed canonical id for template backends)",
    )
    pprep.add_argument(
        "--strict",
        action="store_true",
        help="refuse --execute unless compat is ALREADY SUPPORTED or READY (CI-friendly)",
    )
    pprep.add_argument(
        "--allow-port",
        action="store_true",
        help="DEPRECATED: kept for backward compatibility; permissive execution is now the default",
    )
    pprep.set_defaults(func=cmd_prepare)

    pops = sub.add_parser(
        "op-synth",
        help=(
            "Op-level bring-up granularity: walk the HF reference of each "
            "NEW component, classify every leaf op as op-REUSE / op-ADAPT / "
            "op-NEW against the ttnn primitive set, and (with --emit-stub) "
            "write a partial native TTNN stub where weights and "
            "deterministic op helpers are pre-bound. Reuses existing "
            "tt-metal building blocks (ttnn.linear, ttnn.layer_norm, "
            "ttnn.embedding, ttnn.gelu/silu/relu, ...) instead of asking "
            "the LLM to re-derive them. The LLM only needs to rewrite "
            "`__call__` to wire the pre-bound helpers together."
        ),
    )
    pops.add_argument(
        "model_id",
        help="HuggingFace model id of a previously-scaffolded model",
    )
    pops.add_argument(
        "--component",
        default=None,
        help=(
            "target a single component by name (e.g. `prompt_encoder_config`). "
            "Default: every NEW component in bringup_status.json."
        ),
    )
    pops.add_argument(
        "--include-adapt",
        action="store_true",
        help=(
            "Also op-synth ADAPT components (default: NEW only). Useful "
            "when an ADAPT component's sibling tt-port no longer fits "
            "and you want to start from a deterministic op-level skeleton."
        ),
    )
    pops.add_argument(
        "--emit-stub",
        action="store_true",
        help=(
            "Write the partial TTNN stubs to `<demo>/_synth_responses/`. "
            "Default: print the op-plan only, so you can preview the "
            "REUSE / ADAPT / NEW split before generating code."
        ),
    )
    pops.set_defaults(func=cmd_op_synth)

    pci = sub.add_parser(
        "capture-inputs",
        help=(
            "Capture real intermediate tensors per NEW component by running the "
            "HF model once with forward hooks; saves to `<demo>/_captured/<safe>/` "
            "and patches the generated PCC tests to load them instead of "
            "synthetic randoms. This is what makes PCC tests for prompt- "
            "conditioned heads (SAM2 mask_decoder etc.) actually validate "
            "instead of skipping with `synthetic inputs incompatible`."
        ),
    )
    pci.add_argument("model_id", help="HuggingFace model id of a previously-scaffolded model")
    pci.add_argument(
        "--component",
        default=None,
        help="Capture only this component (default: every NEW component).",
    )
    pci.add_argument(
        "--no-upgrade-tests",
        action="store_true",
        help="Skip patching the test files; only write the captured tensors.",
    )
    pci.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the input image_size (default: read from model.config).",
    )
    pci.set_defaults(func=cmd_capture_inputs)

    pl = sub.add_parser("list-meshes", help="print canonical mesh topology")
    pl.set_defaults(func=cmd_list_meshes)

    psr = sub.add_parser(
        "sync-registry",
        help="check the deterministic registry (backends / building blocks) against the checkout for path drift",
    )
    psr.add_argument(
        "--check", action="store_true", help="exit non-zero if any mapped registry path is missing from the tree"
    )
    psr.add_argument("--no-unmapped", action="store_true", help="skip the reverse 'unmapped reusable module' hints")
    psr.add_argument(
        "--add-source",
        action="append",
        metavar="PATH",
        help="register an extra reusable-source root (e.g. models/tt_v3) — persisted and fetched+scanned on every "
        "up/auto-up so a new library extends the reuse map + sibling families without a tool edit (Point 10b). Repeatable.",
    )
    psr.add_argument(
        "--kind",
        choices=("component", "family"),
        default="component",
        help="for --add-source: 'component' (modules become REUSE/ADAPT targets) or 'family' (subdirs become sibling families)",
    )
    psr.add_argument(
        "--default",
        dest="default",
        choices=("reuse", "adapt"),
        default="adapt",
        help="for --add-source: default classification for manifest-declared entries in the root (heuristic stays ADAPT)",
    )
    psr.set_defaults(func=cmd_sync_registry)

    pe2e = sub.add_parser(
        "emit-e2e",
        help=(
            "Emit a chained-pipeline e2e test skeleton (test_e2e.py) for "
            "a model whose components have been brought up. Generates "
            "one TODO[e2e] marker per TT-ported component (with the HF "
            "submodule_path) so the wiring work is mechanical, and ends "
            "with a final PCC check vs the HF reference."
        ),
    )
    pe2e.add_argument("model_id", help="HuggingFace model id (e.g. facebook/sam2-hiera-tiny)")
    pe2e.add_argument(
        "--output",
        default=None,
        help=(
            "Explicit output path. Default: derived from the model's " "canonical demo directory under models/demos/."
        ),
    )
    pe2e.add_argument(
        "--pcc-target",
        type=float,
        default=0.95,
        help="PCC threshold for the final HF-vs-TT comparison (default: 0.95)",
    )
    pe2e.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing tool-owned files (demo/, tt/, tests/test_demo.py, "
            "tests/test_hf_parity.py, evaluation/, reference/, README.md, "
            "requirements.txt, .gitignore, conftest.py). Bring-up artifacts "
            "(_stubs/, bringup_status.json, tests/pcc/, demo.py) are preserved "
            "unless --force is also passed."
        ),
    )
    pe2e.add_argument(
        "--task",
        default=None,
        help=(
            "For multi-task models, emit only this task (e.g. s2tt|t2t|t2s|asr|llm|...). "
            "Default for multi-task models: emit the first registered task. "
            "Use --all-tasks to emit every supported task."
        ),
    )
    pe2e.add_argument(
        "--all-tasks",
        action="store_true",
        dest="all_tasks",
        help=(
            "For multi-task models (e.g. SeamlessM4T), emit one demo per "
            "task head (s2tt + t2t + t2s). Single-task models ignore this flag."
        ),
    )
    pe2e.add_argument(
        "--force",
        action="store_true",
        help=(
            "Additionally allow overwriting preserved files (tests/pcc/ + legacy demo.py). "
            "Required only when re-running emit-e2e wants to clobber the bring-up "
            "tool's outputs or a hand-written demo.py."
        ),
    )
    pe2e.add_argument(
        "--max-iter",
        type=int,
        default=5,
        dest="max_iter",
        help=(
            "LLM diagnose-fix iteration budget when emitted demo output diverges "
            "from HF golden. Default 5 (matches the per-component iter convention)."
        ),
    )
    pe2e.add_argument(
        "--readme-only",
        action="store_true",
        dest="readme_only",
        help=(
            "Re-run only the README synthesis step (e.g. after re-graduation "
            "updated PCC/perf numbers). Does not touch demo/tt/tests/eval/ref."
        ),
    )
    pe2e.add_argument(
        "--mesh",
        default=None,
        help=(
            "Chip mesh to place the pipeline on, e.g. 2x2 (=4 chips). When >1 chip, the tool runs "
            "select_parallelism (per-TP kernel viability) and instructs the builder to open the mesh "
            "at the chosen TP x DP split and map tensors accordingly. Omitted / 1 chip = single-device "
            "(current behavior, no parallelism guidance)."
        ),
    )
    pe2e.add_argument(
        "--max-grade-rounds",
        type=int,
        default=0,
        dest="max_grade_rounds",
        help=(
            "Max agent fix-loop rounds against the combined gate (default 10). Raise it for the hard "
            "host-free work (residency / KV / on-device token feed / decode step) so the agent gets "
            "enough edit-and-recheck attempts."
        ),
    )
    pe2e.set_defaults(func=cmd_emit_e2e)

    popt = sub.add_parser(
        "optimize",
        help=(
            "Run the perf_automation optimization loop on a demo pipeline. "
            "Target is a planner-emitted model_id (resolved via bringup_status.json) "
            "OR a demo directory path (any existing tt-metal demo)."
        ),
    )
    popt.add_argument(
        "target",
        nargs="?",
        help="HF model_id of a planner-emitted demo, OR a demo directory path (omit when using --model-dir)",
    )
    popt.add_argument(
        "--model-dir",
        dest="model_dir",
        help="directory of the model CODE to optimize (existing tt-metal model); isolated in a worktree",
    )
    popt.add_argument(
        "--pcc-test",
        dest="pcc_test",
        help="e2e PCC test node id 'path::test_fn' (tt-metal-root-relative or absolute) used as the "
        "correctness gate; the perf workload is auto-generated from it unless --perf-test is given",
    )
    popt.add_argument(
        "--perf-test",
        dest="perf_test",
        help="explicit perf test node id 'path::test_fn' to profile (overrides auto-generation); use for "
        "models whose e2e test overflows the profiler and that ship a bounded/layer-capped perf test",
    )
    popt.add_argument("--devices", default="0,1", help="single | all | explicit ids like '0,1'")
    popt.add_argument("--mesh", help="mesh shape like '2x2' for roofline calibration (needs --box)")
    popt.add_argument("--box", help="declared TT box for roofline calibration (e.g. p300c, T3K, Galaxy)")
    popt.add_argument("--metric", default="device_ms", help="device_ms | wall_ms | auto")
    popt.add_argument("--max-iter", type=int, default=1000, dest="max_iter")
    popt.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        dest="max_rounds",
        help="cc engine: max claude -p rounds per pipeline (default 3; one round is a full continuous "
        "agent session that climbs the whole ladder). Use 1 for a single pass, raise for models with "
        "lots of headroom. The deterministic gate can still stop earlier via can_stop.",
    )
    popt.add_argument("-k", "--case", dest="case", help="pytest -k case id override (e.g. device_params0)")
    popt.add_argument(
        "--hitl",
        action="store_true",
        dest="hitl",
        help="human-in-the-loop: the agent applies ONE lever at a time, then PAUSES at a block-level "
        "timing + rationale screen for your commit/revert/try decision before continuing (cc engine only). "
        "Interactive — needs a live terminal; slower per lever but you steer every step.",
    )
    popt.add_argument(
        "--in-place",
        action="store_true",
        dest="in_place",
        help="for an EXISTING (non-planner) demo, mutate its source on the current branch instead of "
        "isolating in a worktree. Off by default: existing demos are optimized in a throwaway worktree "
        "on a fresh branch so your working tree stays untouched. Planner-emitted demos are always in-place.",
    )
    popt.add_argument(
        "--e2e-only",
        action="store_true",
        dest="e2e_only",
        help="cc engine: skip ALL optimization — just measure + print the FULL-model end-to-end time (all "
        "layers, trace-replay when a cached decode exists, else eager full-depth). Use to recover the "
        "before/after full-model number if a prior run stopped/was killed before its AFTER bookend fired.",
    )
    popt.add_argument(
        "--sync-catalog",
        action="store_true",
        dest="sync_catalog",
        help="opt-in: pull the shared learned-knob catalog at start and push GRADUATED_* knobs at the end "
        "(cc engine). Off by default — learning stays local unless this is set.",
    )
    popt.add_argument(
        "--catalog-remote",
        default="origin",
        dest="catalog_remote",
        help="git remote (name or URL) to sync the GRADUATED knob catalog to (default: origin). "
        "Use any fork/remote — nothing is hard-coded.",
    )
    popt.add_argument(
        "--catalog-branch",
        default="perf-catalog",
        dest="catalog_branch",
        help="branch on --catalog-remote that holds the shared catalog (default: perf-catalog). "
        "Kept separate from model-optimization commits.",
    )
    popt.add_argument(
        "--module-level",
        action="store_true",
        dest="module_level",
        help="optimize graduated native modules ONE AT A TIME (against each module's per-component PCC "
        "test) instead of the full pipeline. Sidesteps the heavy e2e baseline; a coarse per-module "
        "pre-pass. Composes with --modules, --hitl, --then-e2e.",
    )
    popt.add_argument(
        "--modules",
        default=None,
        help="comma-separated subset of module names for --module-level (default: all graduated modules).",
    )
    popt.add_argument(
        "--then-e2e",
        action="store_true",
        dest="then_e2e",
        help="after --module-level, run one full-pipeline pass to confirm the per-module wins survive " "composition.",
    )
    popt.set_defaults(func=cmd_optimize)

    pao = sub.add_parser(
        "auto-onboard",
        help=(
            "LLM-draft a FamilyBackend entry for a brand-new architecture. "
            "Probes the HF model, walks its nn.Module tree, scores the "
            "structurally-closest existing template, asks the LLM for a "
            "JSON proposal, validates it, and (with --accept) writes it "
            "into family_backends.py so `up --auto` can use it."
        ),
    )
    pao.add_argument(
        "model_id",
        help="HuggingFace model id whose model_type is unknown to the planner",
    )
    pao.add_argument(
        "--accept",
        action="store_true",
        help=(
            "Write the validated proposal directly into "
            "family_backends.py. Default: print the proposal only; the "
            "user can copy-paste or re-run with --accept."
        ),
    )
    pao.add_argument(
        "--skip-llm",
        action="store_true",
        help=(
            "Don't call the LLM; produce a deterministic stub proposal "
            "from probe + module-tree data alone. Used by tests and for "
            "dry-runs in offline environments. The resulting backend "
            "will not have human-friendly naming / pipeline_tags."
        ),
    )
    pao.add_argument(
        "--agent-bin",
        default="claude",
        help="Path to the LLM CLI binary (default: claude on $PATH).",
    )
    pao.add_argument(
        "--auto-model",
        default="sonnet",
        help=(
            "LLM model alias for the one-shot draft call. Defaults to "
            "Claude Sonnet -- the auto-onboard prompt is straightforward "
            "enough that Sonnet handles it reliably and cheaply."
        ),
    )
    pao.add_argument(
        "--timeout-s",
        type=int,
        default=180,
        help="Timeout for the LLM call (default: 180s).",
    )
    pao.set_defaults(func=cmd_auto_onboard)

    from .commands.overlay_apply import cmd_overlay_apply
    from .commands.overlay_drop import cmd_overlay_drop
    from .commands.overlay_extract import cmd_overlay_extract
    from .commands.overlay_list import cmd_overlay_list
    from .commands.overlay_promote import cmd_overlay_promote
    from .commands.overlay_revert import cmd_overlay_revert

    pol = sub.add_parser("overlay-list", help="List captured overlays (per model or all).")
    pol.add_argument("model_id", nargs="?", default=None, help="Optional: filter by model_id")
    pol.set_defaults(func=cmd_overlay_list)

    poa = sub.add_parser("overlay-apply", help="Apply a model's overlays to the working tree.")
    poa.add_argument("model_id")
    poa.set_defaults(func=cmd_overlay_apply)

    por = sub.add_parser("overlay-revert", help="Revert applied overlays (counter to overlay-apply).")
    por.add_argument("model_id")
    por.set_defaults(func=cmd_overlay_revert)

    pod = sub.add_parser(
        "overlay-drop",
        help="Permanently delete stored overlay(s). Omit rel_path to wipe ALL overlays for the scope.",
    )
    pod.add_argument("model_id")
    pod.add_argument(
        "rel_path",
        nargs="?",
        default=None,
        help="Repo-relative path (e.g. models/tt_transformers/tt/rope.py). Omit to drop the entire scope.",
    )
    pod.set_defaults(func=cmd_overlay_drop)

    def _cmd_overlay_clear_skips(args) -> int:
        from .overlay_manager import clear_persistent_skips

        category = getattr(args, "category", None)
        n = clear_persistent_skips(args.model_id, category=category)
        if n:
            scope = f"category={category}" if category else "ALL"
            print(
                f"cleared {n} persistent skip entrie(s) ({scope}) for `{args.model_id}`. "
                f"Next run will re-attempt these components."
            )
        else:
            if category:
                print(f"no persistent skip entries matching category={category!r} " f"found for `{args.model_id}`.")
            else:
                print(f"no persistent skip entries found for `{args.model_id}`.")
        return 0

    pocs = sub.add_parser(
        "overlay-clear-skips",
        help=(
            "Clear the persistent skip-list for a model. Only KERNEL_MISSING "
            "entries are persisted now — use this after TTNN ships the "
            "missing op(s) so the next run re-attempts the affected "
            "components on device."
        ),
    )
    pocs.add_argument("model_id")
    pocs.add_argument(
        "--category",
        default=None,
        help=(
            "Clear only entries matching this category (case-insensitive). "
            "Currently only KERNEL_MISSING is persisted; the flag is kept "
            "for legacy script compatibility."
        ),
    )
    pocs.set_defaults(func=_cmd_overlay_clear_skips)

    from .commands.tackle_skipped import cmd_tackle_skipped

    pts = sub.add_parser(
        "tackle-skipped",
        help=(
            "Phase 2: walk the persistent skip-list for a model and route "
            "each entry to the appropriate unblock strategy. ModuleList "
            "components get dropped (their files moved to _phase2_dropped/), "
            "missing-arg + shape-mismatch components get retried via "
            "capture-inputs with the auto-onboard driver path enabled. "
            "Run AFTER the standard `up --auto` has graduated all "
            "currently-tested components."
        ),
    )
    pts.add_argument("model_id", help="HuggingFace model id")
    pts.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the routing decisions and exit without modifying files or invoking LLM.",
    )
    pts.add_argument(
        "--only-modulelist",
        action="store_true",
        help="Only process ModuleList entries (free, no LLM cost). Skip capture retries.",
    )
    pts.add_argument(
        "--only-capture",
        action="store_true",
        help="Only process capture-retry entries (incurs LLM cost). Skip ModuleList drops.",
    )
    pts.set_defaults(func=cmd_tackle_skipped)

    pop = sub.add_parser(
        "overlay-promote",
        help="Apply an overlay to the shared file and remove the overlay; you then PR the resulting diff normally.",
    )
    pop.add_argument("model_id")
    pop.add_argument("rel_path")
    pop.set_defaults(func=cmd_overlay_promote)

    poe = sub.add_parser(
        "overlay-extract",
        help="Migration: extract uncommitted shared-file changes from working tree into an overlay, then revert the file.",
    )
    poe.add_argument("model_id")
    poe.add_argument("rel_paths", nargs="+", help="One or more repo-relative paths")
    poe.add_argument(
        "--hunks-matching", default=None, help="Only extract hunks whose body matches this regex (e.g. 'gemma3')."
    )
    poe.add_argument(
        "--intended-for-production",
        action="store_true",
        help="Mark the overlay as a production-PR candidate (metadata only; does not change apply behavior). Use for generalized fixes you plan to upstream later.",
    )
    poe.set_defaults(func=cmd_overlay_extract)

    from .commands.worktree_cleanup import cmd_worktree_cleanup
    from .commands.worktree_list import cmd_worktree_list

    pwl = sub.add_parser("worktree-list", help="List active tt_hw_planner bring-up worktrees (active + orphaned).")
    pwl.set_defaults(func=cmd_worktree_list)

    # ─── Chained-template registry management commands ──────────────
    ptl = sub.add_parser(
        "template-list",
        help="List chained-template registry entries (family templates produced by e2e synthesis).",
    )
    ptl.add_argument("--all", action="store_true", help="Include demoted entries.")
    ptl.set_defaults(func=cmd_template_list)

    ptp = sub.add_parser(
        "template-promote",
        help="Force-promote a chained template (skip the multi-model gate threshold).",
    )
    ptp.add_argument("family_key", help="HF model_type the template was registered under.")
    ptp.set_defaults(func=cmd_template_promote)

    ptd = sub.add_parser(
        "template-demote",
        help="Demote a chained template (regressed, force re-synthesis on next bring-up).",
    )
    ptd.add_argument("family_key", help="HF model_type to demote.")
    ptd.add_argument("--reason", default="", help="Operator-supplied reason (stored in entry).")
    ptd.set_defaults(func=cmd_template_demote)

    pwc = sub.add_parser("worktree-cleanup", help="Remove orphan worktrees (creators no longer alive).")
    pwc.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt and remove all orphans.")
    pwc.set_defaults(func=cmd_worktree_cleanup)

    from .commands.commit_tool import cmd_commit_tool

    pct = sub.add_parser(
        "commit-tool",
        help="Stage and commit ONLY scripts/tt_hw_planner/** changes; non-tool files (treated as learnings) are excluded.",
    )
    pct.add_argument("-m", "--message", required=True, help="Commit message")
    pct.add_argument("--dry-run", action="store_true", help="Show what would be staged without committing")
    pct.set_defaults(func=cmd_commit_tool)

    from .commands.decompose import cmd_decompose

    pdec = sub.add_parser(
        "decompose",
        help=(
            "Decompose a stuck large component into its non-trivial children. "
            "Use this when the auto-loop pushed a HOT component to CPU "
            "fallback (verdict AGENT_STUCK / KERNEL_VERIFIED_MISSING / "
            "ITERATION_BUDGET) — the children may graduate independently."
        ),
    )
    pdec.add_argument("model_id", help="HuggingFace model id")
    pdec.add_argument("component", help="Component name (must exist in bringup_status.json)")
    pdec.add_argument(
        "--min-leaf-count",
        type=int,
        default=2,
        help="Filter out children with fewer than N transitive leaf modules (default: 2)",
    )
    pdec.add_argument(
        "--write-plan",
        action="store_true",
        help=(
            "Persist the proposed children to <demo_dir>/decomposition_plan.json "
            "as a PLANNING ARTIFACT for humans. Note: no auto-consumer yet; "
            "the next `up` does NOT automatically re-onboard these children. "
            "Use the JSON to manually update bringup_status.json or as input "
            "to a follow-up scaffold."
        ),
    )
    pdec.set_defaults(func=cmd_decompose)

    from .commands.view_state import cmd_view_skips

    pvs = sub.add_parser(
        "view-skips",
        help=(
            "Pretty-print the persistent skip-list (verified KERNEL_MISSING "
            "components blocked by TTNN op gaps). Read from "
            "`overlays/<model>/skipped_components.json`. No mutations."
        ),
    )
    pvs.add_argument("model_id", help="HuggingFace model id")
    pvs.set_defaults(func=cmd_view_skips)

    args = parser.parse_args(argv)

    _cmd = next((a for a in argv if not a.startswith("-")), "")
    _device_cmds = {
        "auto-up",
        "up",
        "bringup",
        "promote",
        "prepare",
        "emit-e2e",
        "tackle-skipped",
        "op-synth",
        "capture-inputs",
        "decompose",
    }
    if _cmd in _device_cmds and not os.environ.get("_TT_TTNN_PREFLIGHT_DONE"):
        os.environ["_TT_TTNN_PREFLIGHT_DONE"] = "1"
        try:
            from ._cli_helpers.ttnn_preflight import ensure_ttnn_ready

            if not ensure_ttnn_ready():
                return 1
        except Exception:
            pass

    return args.func(args)
