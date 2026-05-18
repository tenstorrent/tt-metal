# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Bring-up bridge: turn a planner verdict into a runnable demo invocation.

`plan` answers "which box fits", `compat` answers "do the building blocks
exist". This module closes the loop: given a HuggingFace model id, it picks
the recommended (box, mesh, dtype) row, maps that to the `MESH_DEVICE`
parametrization used by `models/tt_transformers/demo/simple_text_demo.py`,
checks the two per-model tuning tables that affect first-run behaviour, and
emits a copy-pasteable pytest invocation (or runs it directly).

Static knowledge of the demo wiring lives here so the rest of the planner
remains free of `models/` imports.  The two tuning tables we inspect
(`MAX_PREFILL_CHUNK_SIZES_DIV1024` in `tt/model_config.py` and
`trace_region_size_dict` in `demo/trace_region_config.py`) are loaded by AST
parse, so we stay the single source of truth without importing ttnn.
"""

from __future__ import annotations

import ast
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .compatibility import CompatReport, Status, check_compatibility
from .discovery import ModelDiscovery, discover_model
from .hardware import HARDWARE, find_box
from .kernel_constraints import KernelReport, Severity, evaluate_kernels
from .probe import probe_model
from .verdict import FitRow, evaluate_all


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_CONFIG_PATH = REPO_ROOT / "models" / "tt_transformers" / "tt" / "model_config.py"
TRACE_REGION_PATH = REPO_ROOT / "models" / "tt_transformers" / "demo" / "trace_region_config.py"
DEMO_TEST_PATH = "models/tt_transformers/demo/simple_text_demo.py"


def _quote(s: str) -> str:
    if " " in s or any(c in s for c in "$\"'()"):
        return shlex.quote(s)
    return s


def _format_argv(argv: List[str], *, indent: int) -> str:
    """Group pytest-style flag/value pairs onto the same line and emit a
    backslash-continued multi-line command.  Standalone flags (those that
    start with `-` and aren't followed by a value, or whose next token is
    itself a flag) stay on their own line."""
    pad = " " * indent
    chunks: List[str] = []
    i = 0
    while i < len(argv):
        cur = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if cur.startswith("-") and nxt is not None and not nxt.startswith("-"):
            chunks.append(f"{_quote(cur)} {_quote(nxt)}")
            i += 2
        else:
            chunks.append(_quote(cur))
            i += 1
    return (" \\\n" + pad).join(chunks)


# Mirrors the literal dict in `simple_text_demo.py` that maps the
# MESH_DEVICE env var to the demo's parametrized mesh shape.  Keyed by
# (arch, mesh_shape) so we can resolve a planner row directly.
MESH_DEVICE_MAP: Dict[Tuple[str, Tuple[int, int]], str] = {
    ("Wormhole", (1, 1)): "N150",
    ("Wormhole", (1, 2)): "N300",
    ("Wormhole", (1, 4)): "N150x4",
    ("Wormhole", (1, 8)): "T3K",
    ("Wormhole", (8, 4)): "TG",
    ("Blackhole", (1, 1)): "P150",
    ("Blackhole", (1, 2)): "P300",
    ("Blackhole", (1, 4)): "P150x4",
    ("Blackhole", (1, 8)): "P150x8",
    ("Blackhole", (8, 4)): "BHGLX",
}


def derive_base_model_name(hf_id: str) -> str:
    """Mirror `trace_region_config.get_base_model_name` so we can index
    the same per-model tuning tables the demo does."""
    model_name = hf_id.strip("/").split("/")[-1]
    m = re.search(r"(.*?\d+[bB])-", model_name)
    return m.group(1) if m else model_name


def mesh_device_for(arch: str, mesh_shape: Tuple[int, int]) -> Tuple[Optional[str], str]:
    """Resolve (arch, mesh_shape) to a MESH_DEVICE label.

    Returns (label, note). When `label is None`, the demo has no env-var
    mapping for this physical shape and bring-up must refuse — silently
    substituting a different shape (e.g. [1,4] for [2,2]) used to happen
    here but is now disallowed because the substitute shape changes the
    runtime divisibility constraints (n_kv_heads % cluster_shape[1]).
    """
    direct = MESH_DEVICE_MAP.get((arch, mesh_shape))
    if direct is not None:
        return direct, ""
    chips = mesh_shape[0] * mesh_shape[1]
    same_chip_alternatives = sorted(
        [(shape, label) for (a, shape), label in MESH_DEVICE_MAP.items() if a == arch and shape[0] * shape[1] == chips]
    )
    if same_chip_alternatives:
        alts = ", ".join(f"{label} → {shape}" for shape, label in same_chip_alternatives)
        return None, (
            f"mesh {mesh_shape} has no MESH_DEVICE label on {arch}. "
            f"The demo only exposes these {chips}-chip shapes: {alts}. "
            f"Pick one that satisfies the model's divisibility constraints "
            f"(see `compat`'s Per-TP table), or add a {mesh_shape} entry to "
            f"the demo's mesh_device parametrize before re-running."
        )
    return None, f"no MESH_DEVICE label for {arch} mesh {mesh_shape}; demo cannot be parametrized."


# ---------------------------------------------------------------------------
# Tuning-table inspection
# ---------------------------------------------------------------------------


def _extract_literal_dict(file_path: Path, var_name: str, *, in_func: Optional[str] = None) -> Optional[dict]:
    """Parse a Python source file and return the literal value of an
    assignment to `var_name`.  Restrict to a function body if `in_func` is
    given.  Returns None if the file can't be read, parsed, or the value
    isn't a literal."""
    try:
        src = file_path.read_text()
    except OSError:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    scopes: List[ast.AST] = []
    if in_func is None:
        scopes = [tree]
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == in_func:
                scopes.append(node)

    for scope in scopes:
        for stmt in ast.walk(scope):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        try:
                            return ast.literal_eval(stmt.value)
                        except (ValueError, SyntaxError):
                            return None
    return None


_CHUNK_TABLE: Optional[dict] = None
_TRACE_TABLE: Optional[dict] = None


def _load_tuning_tables() -> Tuple[dict, dict]:
    global _CHUNK_TABLE, _TRACE_TABLE
    if _CHUNK_TABLE is None:
        _CHUNK_TABLE = (
            _extract_literal_dict(
                MODEL_CONFIG_PATH,
                "MAX_PREFILL_CHUNK_SIZES_DIV1024",
                in_func="get_max_prefill_chunk_size",
            )
            or {}
        )
    if _TRACE_TABLE is None:
        _TRACE_TABLE = (
            _extract_literal_dict(
                TRACE_REGION_PATH,
                "trace_region_size_dict",
                in_func="get_supported_trace_region_size",
            )
            or {}
        )
    return _CHUNK_TABLE, _TRACE_TABLE


@dataclass
class TuningCheck:
    table: str
    file_path: str
    base_model_name: str
    mesh_device: str
    found: bool
    value: Optional[Any]
    fallback: str
    auto_resolved_env: Optional[Dict[str, str]] = None


def _lookup_tuning(
    base_model_name: str,
    mesh_device: str,
    table: dict,
    *,
    table_name: str,
    file_path: Path,
    fallback_when_missing: str,
) -> TuningCheck:
    entry = table.get(base_model_name, {})
    val = entry.get(mesh_device) if isinstance(entry, dict) else None
    rel = str(file_path.relative_to(REPO_ROOT)) if file_path.exists() else str(file_path)
    if val is not None:
        return TuningCheck(table_name, rel, base_model_name, mesh_device, True, val, "")
    return TuningCheck(table_name, rel, base_model_name, mesh_device, False, None, fallback_when_missing)


def check_chunk_size(base_model_name: str, mesh_device: str) -> TuningCheck:
    """Look up the prefill chunk size.  If the (model, mesh_device) cell is
    missing but the model has other (non-None) entries in the same row, pick
    the largest of those as a `MAX_PREFILL_CHUNK_SIZE` env-var override —
    that's a more sensible default than the demo's `4×1024` fallback."""
    chunk, _ = _load_tuning_tables()
    check = _lookup_tuning(
        base_model_name,
        mesh_device,
        chunk,
        table_name="MAX_PREFILL_CHUNK_SIZES_DIV1024",
        file_path=MODEL_CONFIG_PATH,
        fallback_when_missing=("would fall back to MAX_PREFILL_CHUNK_SIZE=4 (×1024) — slow."),
    )
    if not check.found:
        row = chunk.get(base_model_name)
        if isinstance(row, dict):
            siblings = [v for v in row.values() if isinstance(v, int) and v > 0]
            if siblings:
                fallback_value = max(siblings)
                check.auto_resolved_env = {"MAX_PREFILL_CHUNK_SIZE": str(fallback_value)}
                check.fallback = (
                    f"auto-set MAX_PREFILL_CHUNK_SIZE={fallback_value} "
                    f"(largest value in the {base_model_name} row of the same table)"
                )
    return check


def check_trace_region(base_model_name: str, mesh_device: str) -> TuningCheck:
    _, trace = _load_tuning_tables()
    return _lookup_tuning(
        base_model_name,
        mesh_device,
        trace,
        table_name="trace_region_size_dict",
        file_path=TRACE_REGION_PATH,
        fallback_when_missing=(
            "will use the demo's parametrize default (50M–100M). "
            "Add an entry if you need to override for a specific model/SKU."
        ),
    )


# ---------------------------------------------------------------------------
# Pytest invocation
# ---------------------------------------------------------------------------


@dataclass
class PytestInvocation:
    test_path: str
    args: List[str]
    env: Dict[str, str]

    def argv(self) -> List[str]:
        return ["pytest", self.test_path] + self.args

    def shell_form(self, *, cwd: Path) -> str:
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in self.env.items())
        return f"{env_lines}\n\ncd {cwd}\n" + _format_argv(self.argv(), indent=4)


def _build_tt_transformers_invocation(
    *,
    hf_model: str,
    mesh_device: str,
    accuracy: bool,
    batch: int,
    max_seq_len: int,
    max_generated_tokens: int,
    trace: bool,
    paged_attention: bool,
    instruct: bool,
) -> PytestInvocation:
    selector = ("accuracy" if accuracy else "performance") + f" and batch-{batch}"
    args: List[str] = [
        "-k",
        selector,
        "--instruct",
        "1" if instruct else "0",
        "--batch_size",
        str(batch),
        "--max_seq_len",
        str(max_seq_len),
        "--max_generated_tokens",
        str(max_generated_tokens),
        "--paged_attention",
        "True" if paged_attention else "False",
        "--stop_at_eos",
        "1",
        "--mode",
        "full",
        "--enable_trace" if trace else "--disable_trace",
    ]
    env = {"HF_MODEL": hf_model, "MESH_DEVICE": mesh_device}
    return PytestInvocation(test_path=DEMO_TEST_PATH, args=args, env=env)


def _build_external_invocation(
    *,
    hf_model: str,
    test_path: str,
) -> PytestInvocation:
    """Build a pytest invocation for a model whose demo lives outside
    `tt_transformers/`. We emit `pytest <test_path> -svv` and pass HF_MODEL
    so any code that reads the env still gets it. MESH_DEVICE is not set
    because external demos typically use a singular `device` fixture; if a
    particular demo needs it, the user can override on the command line."""
    return PytestInvocation(
        test_path=test_path,
        args=["-svv"],
        env={"HF_MODEL": hf_model},
    )


# ---------------------------------------------------------------------------
# Top-level plan
# ---------------------------------------------------------------------------


@dataclass
class BringupPlan:
    model_id: str
    base_model_name: str
    box_name: str
    arch: str
    mesh_shape: Tuple[int, int]
    mesh_device: str
    mesh_device_note: str
    dtype: str
    fit_verdict: str
    fits: bool
    headroom_gb: float
    compat_overall: str
    compat_summary: str
    compat_blocking: List[str]
    compat_porting: List[str]
    kernel_blockers: List[str]
    tuning_checks: List[TuningCheck]
    invocation: Optional[PytestInvocation]
    notes: List[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        """Permissive: True whenever an invocation was built — which happens
        for ALREADY SUPPORTED / READY / FEASIBLE WITH WORK (no MISSING blocks)
        with no TP=1 kernel blockers.  Strict callers should additionally
        require `compat_overall in {"ALREADY SUPPORTED", "READY"}`."""
        return self.invocation is not None and self.fits


class BringupError(RuntimeError):
    """Raised when a runnable command cannot be produced (and the caller
    should surface a clean error message to the user)."""


def prepare_bringup(
    *,
    model_id: str,
    box_override: Optional[str] = None,
    mesh_override: Optional[Tuple[int, int]] = None,
    dtype_override: Optional[str] = None,
    batch: int = 1,
    max_seq_len: int = 1024,
    max_generated_tokens: int = 200,
    accuracy: bool = False,
    trace: bool = True,
    paged_attention: bool = True,
    instruct: bool = True,
    seq_for_planning: int = 8192,
) -> BringupPlan:
    if mesh_override is not None and box_override is None:
        raise BringupError("--mesh requires --box (mesh shapes are box-specific)")

    probe = probe_model(model_id)

    if not probe.raw_config:
        raise BringupError(
            f"could not load config.json for {model_id}. "
            "If it's a gated repo, set HF_TOKEN or `huggingface-cli login`."
        )
    if probe.memory_model is None:
        raise BringupError(
            f"{model_id} is not a transformer-family model "
            f"(category={probe.category}); simple_text_demo doesn't apply."
        )

    compat = check_compatibility(model_id, probe.raw_config)
    kernels: Optional[KernelReport] = evaluate_kernels(probe.raw_config)

    boxes = [find_box(box_override)] if box_override else HARDWARE
    if mesh_override is not None and mesh_override not in boxes[0].mesh_shapes:
        raise BringupError(
            f"mesh {mesh_override} is not canonical for {boxes[0].name}; " f"valid: {boxes[0].mesh_shapes}"
        )

    if dtype_override:
        dtypes = [dtype_override]
    else:
        dtypes = ["bf16", "bfp8_b"] if probe.category in ("LLM", "VLM") else ["bf16"]

    verdict = evaluate_all(
        model=probe.memory_model,
        boxes=boxes,
        dtypes=dtypes,
        batch=batch,
        seq=seq_for_planning,
        kv_dtype_bytes=2.0,
        all_meshes=mesh_override is not None,
        explore_pp=False,
    )

    best: Optional[FitRow]
    if mesh_override is not None:
        rows = [r for r in verdict.rows if r.mesh_shape == mesh_override]
        fitting = [r for r in rows if r.fits]
        best = min(fitting, key=lambda r: r.per_chip_gb) if fitting else (rows[0] if rows else None)
    else:
        best = verdict.best

    if best is None:
        hint = "try --explore-pp, --dtype bfp4_b, or a larger --box"
        raise BringupError(
            f"planner found no fitting configuration for {model_id} on "
            f"{[b.name for b in boxes]} at dtypes {dtypes}. {hint}."
        )

    arch = best.box.arch
    mesh_device, mesh_note = mesh_device_for(arch, best.mesh_shape)
    if mesh_device is None:
        raise BringupError(mesh_note)

    base_name = derive_base_model_name(model_id)
    discovery: ModelDiscovery = compat.discovery if compat.discovery is not None else discover_model(model_id)
    # tt_transformers tuning tables only apply when the demo is the standard
    # simple_text_demo. External demos manage their own tuning.
    if discovery.in_external_demo:
        tuning: List[TuningCheck] = []
    else:
        tuning = [
            check_chunk_size(base_name, mesh_device),
            check_trace_region(base_name, mesh_device),
        ]

    # When `ALREADY SUPPORTED` overrides the per-block checker (the model is
    # already in `tt_transformers`'s verified list, or in the external-demo
    # registry), suppress the per-block warnings — they describe blocks the
    # checker thinks are missing but reality says exist.
    suppress = compat.overall.startswith("ALREADY SUPPORTED")
    compat_blocking = (
        [] if suppress else [r.block.name for r in compat.results if r.needed and r.status == Status.MISSING]
    )
    compat_porting = (
        []
        if suppress
        else [
            f"{r.block.name} [{r.effort.value}] — {r.notes or r.block.notes or 'see compatibility.py'}"
            for r in compat.results
            if r.needed and r.status == Status.PARTIAL
        ]
    )

    kernel_blockers: List[str] = []
    if kernels is not None:
        chosen_tp = max(1, int(best.mesh_shape[1]))
        tps_to_check = {1, chosen_tp}
        for tp in sorted(tps_to_check):
            for f in kernels.findings_by_tp.get(tp, []):
                if not f.passes and f.severity == Severity.BLOCKER:
                    kernel_blockers.append(f"TP={tp}: {f.op}.{f.field}={f.value}: {f.constraint}")

    arch_blocker: Optional[str] = None
    if not discovery.runnable_on_arch(arch):
        supported = ", ".join(sorted(discovery.arch_compatibility)) or "(unknown)"
        arch_blocker = (
            f"demo path is restricted to {supported}; target {best.box.name} is {arch}. "
            f"The Wormhole→Blackhole (or vice versa) port for this demo has not been "
            f"written; kernels will fail to compile (typical symptom: NOC_MAX_BURST_SIZE "
            f"or harvested-core static_asserts)."
        )

    mesh_label_blocker: Optional[str] = None
    if mesh_device is None and mesh_note:
        mesh_label_blocker = mesh_note

    invocation: Optional[PytestInvocation] = None
    has_missing_blocks = any(r.needed and r.status == Status.MISSING for r in compat.results)
    can_run = (
        best.fits
        and compat.overall != "BLOCKED"
        and not compat.overall.startswith("TARGETED")
        and not has_missing_blocks
        and mesh_label_blocker is None
        and not kernel_blockers
        and arch_blocker is None
    )
    if can_run:
        if discovery.in_external_demo and discovery.primary_demo is not None:
            invocation = _build_external_invocation(
                hf_model=model_id,
                test_path=discovery.primary_demo.as_posix(),
            )
        else:
            # tt_transformers path (or fallback when discovery is empty but
            # the model is in SUPPORTED_HF_MODELS — still try the standard
            # demo and let pytest fail loudly if it doesn't apply).
            invocation = _build_tt_transformers_invocation(
                hf_model=model_id,
                mesh_device=mesh_device,
                accuracy=accuracy,
                batch=batch,
                max_seq_len=max_seq_len,
                max_generated_tokens=max_generated_tokens,
                trace=trace,
                paged_attention=paged_attention,
                instruct=instruct,
            )
            for tc in tuning:
                if tc.auto_resolved_env:
                    invocation.env.update(tc.auto_resolved_env)

    notes: List[str] = list(discovery.notes)
    if arch_blocker is not None:
        notes.insert(0, f"ARCH-INCOMPATIBLE: {arch_blocker}")
    if mesh_label_blocker is not None:
        notes.insert(0, f"MESH-LABEL-MISSING: {mesh_label_blocker}")

    return BringupPlan(
        model_id=model_id,
        base_model_name=base_name,
        box_name=best.box.name,
        arch=arch,
        mesh_shape=best.mesh_shape,
        mesh_device=mesh_device,
        mesh_device_note=mesh_note,
        dtype=best.dtype,
        fit_verdict=best.tightness.value,
        fits=best.fits,
        headroom_gb=best.headroom_gb,
        compat_overall=compat.overall,
        compat_summary=compat.effort_summary,
        compat_blocking=compat_blocking,
        compat_porting=compat_porting,
        kernel_blockers=kernel_blockers,
        tuning_checks=tuning,
        invocation=invocation,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_text(plan: BringupPlan) -> str:
    sep = "=" * 78
    out: List[str] = [sep]
    if plan.runnable:
        head = "BRING-UP READY"
    elif plan.invocation is not None:
        head = "BRING-UP READY (will execute; compat is informational)"
    else:
        head = "BRING-UP BLOCKED"
    out.append(f"  {head} — {plan.model_id} on {plan.box_name} mesh [{plan.mesh_shape[0]},{plan.mesh_shape[1]}]")
    out.append(sep)
    out.append("")

    out.append(
        f"  Planner verdict:  {plan.fit_verdict} on {plan.box_name}, "
        f"{plan.dtype}, headroom {plan.headroom_gb:.1f} GB"
    )
    out.append(f"  Compat verdict:   {plan.compat_overall}")
    if plan.compat_summary:
        out.append(f"                    {plan.compat_summary}")
    out.append("")

    if plan.mesh_device_note:
        out.append(f"  Note: {plan.mesh_device_note}")
        out.append("")

    if plan.notes:
        out.append("  Notes:")
        for n in plan.notes:
            out.append(f"    - {n}")
        out.append("")

    if plan.kernel_blockers:
        out.append("  Kernel BLOCKERS (TP=1):")
        for k in plan.kernel_blockers:
            out.append(f"    - {k}")
        out.append("")
    if plan.compat_blocking:
        out.append("  Missing TT building blocks:")
        for b in plan.compat_blocking:
            out.append(f"    - {b}")
        out.append("")
    if plan.compat_porting:
        out.append("  Partial / needs porting:")
        for p in plan.compat_porting:
            out.append(f"    - {p}")
        out.append("")

    if plan.tuning_checks:
        out.append("  Tuning tables checked:")
        for tc in plan.tuning_checks:
            if tc.found:
                glyph = "[ ok ]"
            elif tc.auto_resolved_env:
                glyph = "[auto]"
            else:
                glyph = "[warn]"
            body = (
                f"{tc.table}[{tc.base_model_name}][{tc.mesh_device}] = {tc.value}"
                if tc.found
                else f"{tc.table}[{tc.base_model_name}][{tc.mesh_device}] missing → {tc.fallback}"
            )
            out.append(f"    {glyph}  {body}")
            if not tc.found:
                out.append(f"            ({tc.file_path})")
        out.append("")
    else:
        out.append("  Tuning tables: not applicable (external demo).")
        out.append("")

    if plan.invocation is None:
        out.append("  No bring-up command emitted — fix the blockers above and re-run `prepare`.")
        out.append("")
        out.append(sep)
        return "\n".join(out)

    out.append("  Copy-paste to run from repo root:")
    out.append("")
    for k, v in plan.invocation.env.items():
        out.append(f'    export {k}="{v}"')
    out.append("")
    out.append(f"    cd {REPO_ROOT}")
    out.append("    " + _format_argv(plan.invocation.argv(), indent=8))
    out.append("")
    out.append(sep)
    return "\n".join(out)


def render_script(plan: BringupPlan) -> str:
    if plan.invocation is None:
        return (
            "#!/usr/bin/env bash\n"
            f"# Bring-up blocked for {plan.model_id} (compat={plan.compat_overall}).\n"
            "echo 'tt_hw_planner prepare: no runnable command — see blockers' >&2\n"
            "exit 2\n"
        )
    inv = plan.invocation
    mesh_tag = inv.env.get("MESH_DEVICE") or plan.mesh_device
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by `tt_hw_planner prepare`.",
        f"# Model:   {plan.model_id}",
        f"# Target:  {plan.box_name} mesh [{plan.mesh_shape[0]},{plan.mesh_shape[1]}] " f"({mesh_tag}) @ {plan.dtype}",
        f"# Verdict: {plan.fit_verdict}; compat={plan.compat_overall}",
        "set -euo pipefail",
        "",
    ]
    for k, v in inv.env.items():
        lines.append(f'export {k}="{v}"')
    lines.append("")
    lines.append(f"cd {REPO_ROOT}")
    lines.append(_format_argv(inv.argv(), indent=4))
    lines.append("")
    return "\n".join(lines)


def render_json(plan: BringupPlan) -> str:
    import json

    payload = {
        "model_id": plan.model_id,
        "base_model_name": plan.base_model_name,
        "runnable": plan.runnable,
        "target": {
            "box": plan.box_name,
            "arch": plan.arch,
            "mesh_shape": list(plan.mesh_shape),
            "mesh_device": plan.mesh_device,
            "mesh_device_note": plan.mesh_device_note,
            "dtype": plan.dtype,
        },
        "fit": {
            "verdict": plan.fit_verdict,
            "fits": plan.fits,
            "headroom_gb": plan.headroom_gb,
        },
        "compat": {
            "overall": plan.compat_overall,
            "summary": plan.compat_summary,
            "blocking_blocks": plan.compat_blocking,
            "partial_blocks": plan.compat_porting,
            "kernel_blockers": plan.kernel_blockers,
        },
        "tuning": [
            {
                "table": tc.table,
                "file": tc.file_path,
                "key": [tc.base_model_name, tc.mesh_device],
                "found": tc.found,
                "value": tc.value,
                "fallback": tc.fallback,
            }
            for tc in plan.tuning_checks
        ],
        "notes": plan.notes,
        "invocation": (
            {
                "env": plan.invocation.env,
                "command": plan.invocation.argv(),
                "cwd": str(REPO_ROOT),
            }
            if plan.invocation is not None
            else None
        ),
    }
    return json.dumps(payload, indent=2)
