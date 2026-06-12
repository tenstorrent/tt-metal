from __future__ import annotations

import ast
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .compatibility import CompatReport, Status, check_compatibility
from .discovery import ModelDiscovery, discover_model, safe_relative_to_root
from .family_backends import FamilyBackend, pick_backend
from .hardware import HARDWARE, find_box
from .kernel_constraints import KernelReport, Severity, evaluate_kernels
from .probe import probe_model
from .verdict import FitRow, evaluate_all


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_TEST_PATH = "models/tt_transformers/demo/simple_text_demo.py"


def _model_config_path() -> Path:
    from .discovery import BRINGUP_ROOT

    return BRINGUP_ROOT() / "models" / "tt_transformers" / "tt" / "model_config.py"


def _trace_region_path() -> Path:
    from .discovery import BRINGUP_ROOT

    return BRINGUP_ROOT() / "models" / "tt_transformers" / "demo" / "trace_region_config.py"


def __getattr__(name):
    if name == "MODEL_CONFIG_PATH":
        return _model_config_path()
    if name == "TRACE_REGION_PATH":
        return _trace_region_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _quote(s: str) -> str:
    if " " in s or any(c in s for c in "$\"'()"):
        return shlex.quote(s)
    return s


def _format_argv(argv: List[str], *, indent: int) -> str:
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


MESH_DEVICE_MAP: Dict[Tuple[str, Tuple[int, int]], str] = {
    ("Wormhole", (1, 1)): "N150",
    ("Wormhole", (1, 2)): "N300",
    ("Wormhole", (1, 4)): "N150x4",
    ("Wormhole", (1, 8)): "T3K",
    # Galaxy is a 32-chip box; its canonical large-scale shape is [4,8]
    # (see hardware.py). Both orientations map to the same device label so a
    # request for the canonical (4,8) resolves directly instead of failing.
    ("Wormhole", (4, 8)): "TG",
    ("Wormhole", (8, 4)): "TG",
    ("Blackhole", (1, 1)): "P150",
    ("Blackhole", (1, 2)): "P300",
    ("Blackhole", (1, 4)): "P150x4",
    ("Blackhole", (2, 2)): "P150x4_2x2",
    ("Blackhole", (1, 8)): "P150x8",
    ("Blackhole", (4, 8)): "BHGLX",
    ("Blackhole", (8, 4)): "BHGLX",
}


def derive_base_model_name(hf_id: str) -> str:
    model_name = hf_id.strip("/").split("/")[-1]
    m = re.search(r"(.*?\d+[bB])-", model_name)
    return m.group(1) if m else model_name


def mesh_device_for(arch: str, mesh_shape: Tuple[int, int]) -> Tuple[Optional[str], str]:
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


def _extract_literal_dict(file_path: Path, var_name: str, *, in_func: Optional[str] = None) -> Optional[dict]:
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
                _model_config_path(),
                "MAX_PREFILL_CHUNK_SIZES_DIV1024",
                in_func="get_max_prefill_chunk_size",
            )
            or {}
        )
    if _TRACE_TABLE is None:
        _TRACE_TABLE = (
            _extract_literal_dict(
                _trace_region_path(),
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
    rel = str(safe_relative_to_root(file_path)) if file_path.exists() else str(file_path)
    if val is not None:
        return TuningCheck(table_name, rel, base_model_name, mesh_device, True, val, "")
    return TuningCheck(table_name, rel, base_model_name, mesh_device, False, None, fallback_when_missing)


def check_chunk_size(base_model_name: str, mesh_device: str) -> TuningCheck:
    chunk, _ = _load_tuning_tables()
    check = _lookup_tuning(
        base_model_name,
        mesh_device,
        chunk,
        table_name="MAX_PREFILL_CHUNK_SIZES_DIV1024",
        file_path=_model_config_path(),
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
        file_path=_trace_region_path(),
        fallback_when_missing=(
            "will use the demo's parametrize default (50M–100M). "
            "Add an entry if you need to override for a specific model/SKU."
        ),
    )


@dataclass
class PytestInvocation:
    test_path: str
    args: List[str]
    env: Dict[str, str]

    @staticmethod
    def per_test_timeout_s() -> int:
        import os as _os

        try:
            v = int(_os.environ.get("TT_PLANNER_PER_TEST_TIMEOUT_S", "1800"))
            return v if v > 0 else 1800
        except (TypeError, ValueError):
            return 1800

    def argv(self) -> List[str]:
        return (
            [
                "pytest",
                "-p",
                "scripts.tt_hw_planner.instrumentation",
                self.test_path,
            ]
            + self.args
            + [
                f"--timeout={self.per_test_timeout_s()}",
            ]
        )

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
    env = {"HF_MODEL": hf_model, "MESH_DEVICE": mesh_device, "TT_HW_PLANNER_OVERLAY_MODEL": hf_model}
    return PytestInvocation(test_path=DEMO_TEST_PATH, args=args, env=env)


def _build_external_invocation(
    *,
    hf_model: str,
    test_path: str,
) -> PytestInvocation:
    return PytestInvocation(
        test_path=test_path,
        args=["-svv"],
        env={"HF_MODEL": hf_model},
    )


def _build_scaffolded_stubs_invocation(
    *,
    hf_model: str,
) -> Optional[Tuple[PytestInvocation, str, int]]:
    from .bringup_loop import find_demo_dir
    from .family_backends import DEFAULT_TEMPLATE_PYTEST_EXCLUDE_K

    demo_dir = find_demo_dir(hf_model)
    if demo_dir is None:
        return None

    stubs_dir = demo_dir / "_stubs"
    pcc_dir = demo_dir / "tests" / "pcc"
    stub_files = list(stubs_dir.glob("*.py")) if stubs_dir.is_dir() else []
    if not stub_files:
        return None
    if not pcc_dir.is_dir():
        return None

    matched_tests: List[Path] = []
    for stub in stub_files:
        if stub.name == "__init__.py":
            continue
        candidate = pcc_dir / f"test_{stub.stem}.py"
        if candidate.is_file():
            matched_tests.append(candidate)
    if not matched_tests:
        return None

    from .discovery import BRINGUP_ROOT as _BRINGUP_ROOT

    _root = _BRINGUP_ROOT()

    def _safe_rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(_root.resolve()))
        except ValueError:
            return str(safe_relative_to_root(p))

    rel_tests = [_safe_rel(p) for p in matched_tests]
    invocation = PytestInvocation(
        test_path=rel_tests[0],
        args=[*rel_tests[1:], "-svv", "-k", DEFAULT_TEMPLATE_PYTEST_EXCLUDE_K],
        env={"HF_MODEL": hf_model, "PLANNER_TARGET_HF_MODEL": hf_model},
    )
    return invocation, _safe_rel(demo_dir), len(stub_files)


def _build_family_template_invocation(
    *,
    hf_model: str,
    backend: FamilyBackend,
) -> PytestInvocation:
    test_path = backend.smoke_test_entry or backend.demo_path
    args: List[str] = ["-svv"]
    exclude_k = backend.effective_pytest_exclude_k()
    if exclude_k:
        args.extend(["-k", exclude_k])
    return PytestInvocation(
        test_path=test_path,
        args=args,
        env={
            "HF_MODEL": backend.canonical_hf_id or hf_model,
            "PLANNER_TARGET_HF_MODEL": hf_model,
        },
    )


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
    backend_name: Optional[str] = None
    is_template: bool = False
    template_canonical_hf_id: Optional[str] = None

    @property
    def runnable(self) -> bool:
        return self.invocation is not None and self.fits


class BringupError(RuntimeError):
    pass


def _prepare_non_text_family(
    *,
    probe,
    model_id: str,
    box_override: Optional[str],
    mesh_override: Optional[Tuple[int, int]],
) -> BringupPlan:
    pipeline_tag = getattr(probe, "pipeline_tag", None)
    model_type = (probe.raw_config or {}).get("model_type") if probe.raw_config else None
    backend = pick_backend(category=probe.category, model_type=model_type, pipeline_tag=pipeline_tag)
    if backend is None:
        raise BringupError(
            f"{model_id} has category={probe.category} but no tt-metal demo backend "
            "is registered for that family. Add an entry in "
            "`scripts/tt_hw_planner/family_backends.py` or port a sibling demo first."
        )

    boxes = [find_box(box_override)] if box_override else HARDWARE
    box = boxes[0]
    if mesh_override is not None and mesh_override not in box.mesh_shapes:
        raise BringupError(f"mesh {mesh_override} is not canonical for {box.name}; valid: {box.mesh_shapes}")
    mesh_shape: Tuple[int, int] = mesh_override or (1, 1)
    arch = box.arch
    mesh_device, mesh_note = mesh_device_for(arch, mesh_shape)
    if mesh_device is None:
        mesh_device = "(unset; external demo manages mesh)"

    notes: List[str] = []

    scaffolded = _build_scaffolded_stubs_invocation(hf_model=model_id)
    is_template_run = False
    backend_label = backend.name
    if scaffolded is not None:
        invocation, demo_rel, stub_count = scaffolded
        backend_label = f"scaffolded stubs ({stub_count} component(s)) for {model_id}"
        notes.append(
            f"SCAFFOLDED RUN: targeting {demo_rel}/tests/pcc/ — exercises "
            f"the {stub_count} TTNN stub(s) under {demo_rel}/_stubs/ on TT "
            f"hardware (one PCC test per NEW component). This is the "
            f"model's OWN code, not a sibling template."
        )
    else:
        invocation = _build_family_template_invocation(hf_model=model_id, backend=backend)
        is_template_run = backend.routing_mode == "template"
        if is_template_run:
            notes.append(
                f"TEMPLATE BRING-UP: dispatched to {backend.name} "
                f"({backend.demo_path}). The demo runs canonical_hf_id="
                f"{backend.canonical_hf_id!r}; adapt encoder/decoder/IO for "
                f"{model_id} before expecting correct outputs. (No "
                f"scaffolded stubs found for {model_id}; run "
                f"`scaffold {model_id} --apply` to switch this command to "
                f"running your own TTNN ports instead of a sibling template.)"
            )
        else:
            notes.append(f"Generic backend: {backend.name} ({backend.demo_path}).")
    if backend.notes:
        notes.append(backend.notes)

    compat = check_compatibility(model_id, probe.raw_config or {})

    return BringupPlan(
        model_id=model_id,
        base_model_name=derive_base_model_name(model_id),
        box_name=box.name,
        arch=arch,
        mesh_shape=mesh_shape,
        mesh_device=mesh_device,
        mesh_device_note=mesh_note,
        dtype="bf16",
        fit_verdict="n/a (non-LLM family backend)",
        fits=True,
        headroom_gb=0.0,
        compat_overall=compat.overall,
        compat_summary=compat.effort_summary,
        compat_blocking=[],
        compat_porting=[],
        kernel_blockers=[],
        tuning_checks=[],
        invocation=invocation,
        notes=notes,
        backend_name=backend_label,
        is_template=is_template_run,
        template_canonical_hf_id=backend.canonical_hf_id if is_template_run else None,
    )


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
        return _prepare_non_text_family(
            probe=probe,
            model_id=model_id,
            box_override=box_override,
            mesh_override=mesh_override,
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
    if discovery.in_external_demo:
        tuning: List[TuningCheck] = []
    else:
        tuning = [
            check_chunk_size(base_name, mesh_device),
            check_trace_region(base_name, mesh_device),
        ]

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


def render_text(plan: BringupPlan) -> str:
    sep = "=" * 78
    out: List[str] = [sep]
    if plan.is_template:
        head = "BRING-UP TEMPLATE (closest tt-metal demo for this family)"
    elif plan.runnable:
        head = "BRING-UP READY"
    elif plan.invocation is not None:
        head = "BRING-UP READY (will execute; compat is informational)"
    else:
        head = "BRING-UP BLOCKED"
    out.append(f"  {head} — {plan.model_id} on {plan.box_name} mesh [{plan.mesh_shape[0]},{plan.mesh_shape[1]}]")
    if plan.backend_name:
        out.append(f"  Backend: {plan.backend_name}")
        if plan.template_canonical_hf_id and plan.is_template:
            out.append(f"  Runs canonical HF id out-of-the-box: {plan.template_canonical_hf_id}")
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
    from .discovery import BRINGUP_ROOT

    out.append(f"    cd {BRINGUP_ROOT()}")
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
    from .discovery import BRINGUP_ROOT

    lines.append(f"cd {BRINGUP_ROOT()}")
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
