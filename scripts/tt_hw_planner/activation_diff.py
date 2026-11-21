"""Layer-wise activation diff helper (PCC bisection).

Why this exists
---------------
When a ttnn stub fails its end-of-forward PCC check (e.g. final-output PCC
~ 0.67) but every intermediate op is structurally valid, the LLM has no
signal about WHICH `_apply_*` helper diverges from the torch reference.
A blind re-write tends to oscillate between near-misses ("try bfloat16
accumulation", "try different norm eps") without converging.

This module instruments both the torch reference and the ttnn stub at
every `_apply_*` boundary, runs them on the same captured inputs, and
reports the FIRST helper whose output PCC against the torch reference
submodule drops below a threshold (default 0.95). That single piece of
localization information is then injected into the LLM prompt as a
LOCALIZATION HINT, transforming the LLM's task from
"figure out why the final output is wrong" to "fix `_apply_X`".

The mapping from `_apply_<helper>` to `torch_module.<dotted_name>` is
taken directly from the op-synth manifest sidecar
(`<demo_dir>/_stubs/<safe>.opplan.json`) — `pre_bound[].name` /
`pre_bound[].helper`. So this only works for op-synth partial stubs;
for non-op-synth stubs the function degrades gracefully and returns
None (the loop falls back to the existing strategy directive).

Failure modes are non-fatal: any exception during instrumentation /
forward / PCC compute returns None, so the auto-iter loop never crashes
on a flaky diff and the existing prompt path is preserved.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class HelperDivergence:
    """One row of the layer-wise PCC table.

    `helper` is the `_apply_*` method on the ttnn stub. `submodule_path`
    is the dotted attribute path inside the torch reference module
    (e.g. `transformer.layers.0.self_attn.q_proj`). `pcc` is the cosine
    similarity between the two captured outputs (None if either side
    failed to produce a comparable tensor, or if the ttnn side was
    not available in this run). `note` is a short human-readable
    status (e.g. "ok", "captured-none", "shape-mismatch", "torch-only").

    When the activation diff is run in torch-only mode (device=None),
    `stats` captures the torch reference's per-helper statistics
    (shape, dtype, mean, std, l2 norm). The LLM can use these as
    ground truth to compare against the ttnn output stats it
    mentally derives from its own code.
    """

    helper: str
    submodule_path: str
    ttnn_target: str
    pcc: Optional[float]
    note: str
    stats: Optional[Dict[str, Any]] = None


@dataclass
class LocalizationResult:
    """Output of `localize_pcc_divergence`.

    `first_divergence` is the first helper (in the order ttnn's
    `__call__` executed them) whose PCC dropped below the threshold.
    `table` is the full diff in execution order, useful for the LLM
    prompt. `forward_succeeded` indicates whether both the torch
    reference and the ttnn stub completed without raising.
    """

    first_divergence: Optional[HelperDivergence]
    table: List[HelperDivergence]
    forward_succeeded: bool
    threshold: float
    note: str


def _safe_id(name: str) -> str:
    """Backwards-compat shim. See :func:`module_tree.safe_identifier`."""
    from .module_tree import safe_identifier

    return safe_identifier(name)


def _resolve_dotted(obj: Any, dotted: str) -> Any:
    """Backwards-compat shim. See :func:`module_tree.resolve_dotted`."""
    from .module_tree import resolve_dotted

    return resolve_dotted(obj, dotted)


def _load_opplan_pre_bound(opplan_path: Path) -> List[Dict[str, str]]:
    """Read the op-synth manifest and return the ordered `pre_bound`
    list (each entry has `name`, `helper`, `ttnn_target`, ...).

    Returns [] if the manifest is missing or malformed."""
    if not opplan_path.is_file():
        return []
    try:
        data = json.loads(opplan_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    pb = data.get("pre_bound") or []
    if not isinstance(pb, list):
        return []
    out: List[Dict[str, str]] = []
    for entry in pb:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        helper = entry.get("helper")
        if not name or not helper:
            continue
        out.append(
            {
                "name": str(name),
                "helper": str(helper),
                "ttnn_target": str(entry.get("ttnn_target", "")),
            }
        )
    return out


def _load_captured_io(comp_dir: Path):
    """Load `(args, kwargs, output)` from a capture directory.

    Returns `(None, None, None)` if any file is missing or unreadable.
    Errors here are non-fatal — the caller falls back to the existing
    prompt path."""
    try:
        import torch
    except Exception:
        return None, None, None
    if not comp_dir.is_dir():
        return None, None, None
    args_p = comp_dir / "args.pt"
    kwargs_p = comp_dir / "kwargs.pt"
    output_p = comp_dir / "output.pt"
    if not output_p.is_file():
        return None, None, None
    try:
        args = torch.load(args_p, weights_only=False) if args_p.is_file() else ()
        kwargs = torch.load(kwargs_p, weights_only=False) if kwargs_p.is_file() else {}
        output = torch.load(output_p, weights_only=False)
    except Exception:
        return None, None, None
    return args, kwargs, output


def _pcc(a, b) -> Optional[float]:
    """Cosine-similarity PCC between two torch tensors.

    Best-effort: shape-broadcasts via flatten, casts to float32, and
    returns None on any failure (None tensor, shape mismatch, NaN/Inf
    blow-up). Threshold-based "diverged" decisions are made by the
    caller; this function only reports the raw similarity number."""
    try:
        import torch
    except Exception:
        return None
    if a is None or b is None:
        return None
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    try:
        af = a.detach().to(dtype=torch.float32).flatten()
        bf = b.detach().to(dtype=torch.float32).flatten()
        if af.numel() == 0 or bf.numel() == 0:
            return None
        if af.numel() != bf.numel():
            n = min(af.numel(), bf.numel())
            af = af[:n]
            bf = bf[:n]
        af = af - af.mean()
        bf = bf - bf.mean()
        na = af.norm()
        nb = bf.norm()
        if na.item() == 0.0 or nb.item() == 0.0:
            return None
        return float((af * bf).sum() / (na * nb))
    except Exception:
        return None


def _tensor_stats(t: Any) -> Optional[Dict[str, Any]]:
    """Compact statistics for one tensor — used in torch-only mode to
    inject ground-truth shape/scale into the LLM prompt without needing
    a ttnn-side run for PCC. None inputs / non-tensors return None.

    Canonical shared implementation: also used by the agentic dual-probe
    (HF + TT sides import this to keep the per-module stats schema
    identical across the legacy decode-layer probe and the new
    layer-tree probe)."""
    try:
        import torch
    except Exception:
        return None
    if not isinstance(t, torch.Tensor):
        return None
    try:
        ft = t.detach().to(dtype=torch.float32).flatten()
        if ft.numel() == 0:
            return {
                "shape": tuple(t.shape),
                "dtype": str(t.dtype),
                "numel": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "l2": 0.0,
                "abs_max": 0.0,
            }
        return {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "numel": int(ft.numel()),
            "mean": float(ft.mean()),
            "std": float(ft.std()) if ft.numel() > 1 else 0.0,
            "min": float(ft.min()),
            "max": float(ft.max()),
            "l2": float(ft.norm()),
            "abs_max": float(ft.abs().max()),
        }
    except Exception:
        return None


def _to_torch_best_effort(out: Any, device: Any = None):
    """Convert a ttnn / torch / heterogeneous helper output to a single
    torch tensor for comparison against the torch reference. Returns
    None if the output can't be flattened to a single tensor (e.g.
    a tuple of tensors of incompatible shapes, or an opaque object).

    Heuristics:
    - bare torch.Tensor: return as-is
    - bare ttnn.Tensor: `ttnn.to_torch(...)` (best-effort), with
      mesh-composition handled by stripping the leading mesh dim if
      present.
    - tuple/list of tensors: return the FIRST tensor element (matches
      how most HF submodules return `(hidden, ...)` and ttnn ops return
      a single tensor anyway).
    - dict: try common keys `output`, `last_hidden_state`,
      `hidden_states`, then the first tensor value.
    """
    try:
        import torch
    except Exception:
        return None
    if out is None:
        return None
    if isinstance(out, torch.Tensor):
        return out

    if hasattr(out, "to_torch"):
        try:
            t = out.to_torch()
            if isinstance(t, torch.Tensor):
                return t
        except Exception:
            pass
    try:
        import ttnn

        if isinstance(out, ttnn.Tensor):
            try:
                return ttnn.to_torch(out)
            except Exception:
                try:
                    from models.common.auto_compose import to_torch_auto_compose

                    return to_torch_auto_compose(out, device=device)
                except Exception:
                    return None
    except Exception:
        pass
    if isinstance(out, (tuple, list)):
        for item in out:
            t = _to_torch_best_effort(item)
            if t is not None:
                return t
        return None
    if isinstance(out, dict):
        for key in ("output", "last_hidden_state", "hidden_states"):
            if key in out:
                t = _to_torch_best_effort(out[key])
                if t is not None:
                    return t
        for v in out.values():
            t = _to_torch_best_effort(v)
            if t is not None:
                return t
        return None
    return None


def _install_torch_hooks(
    torch_module,
    pre_bound: List[Dict[str, str]],
) -> Tuple[Dict[str, Any], List[Any]]:
    """Register forward-hooks on every torch submodule named in
    `pre_bound`. Returns a `(captured_outputs, handles)` pair — the
    caller is responsible for removing the handles in a `finally` block.

    `captured_outputs` maps `helper_name` -> raw torch submodule output
    (whatever the submodule returned, no normalization)."""
    captured: Dict[str, Any] = {}
    handles: List[Any] = []
    for entry in pre_bound:
        helper = entry["helper"]
        sub_path = entry["name"]
        try:
            sub = _resolve_dotted(torch_module, sub_path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        if sub is None or not hasattr(sub, "register_forward_hook"):
            continue

        def _mk_hook(helper_name: str) -> Callable[[Any, Any, Any], None]:
            def _hook(_module, _inp, out):
                captured[helper_name] = out

            return _hook

        try:
            h = sub.register_forward_hook(_mk_hook(helper))
            handles.append(h)
        except Exception:
            continue
    return captured, handles


def _install_ttnn_hooks(
    stub_instance,
    pre_bound: List[Dict[str, str]],
) -> Tuple[Dict[str, Any], List[Tuple[str, Any]]]:
    """Monkey-patch each `_apply_<helper>` method on the stub instance
    to capture its return value. Returns `(captured_outputs, originals)`
    so the caller can restore the originals in a `finally` block.

    We only patch helpers that are actually present on the instance —
    op-synth stubs may not bind every entry from `pre_bound` if the
    helper was inlined into `__call__`."""
    captured: Dict[str, Any] = {}
    originals: List[Tuple[str, Any]] = []
    for entry in pre_bound:
        helper = entry["helper"]
        if not hasattr(stub_instance, helper):
            continue
        orig = getattr(stub_instance, helper)
        if not callable(orig):
            continue
        originals.append((helper, orig))

        def _mk_wrapped(helper_name: str, orig_fn: Callable[..., Any]) -> Callable[..., Any]:
            def _wrapped(*args, **kwargs):
                out = orig_fn(*args, **kwargs)

                captured[helper_name] = out
                return out

            return _wrapped

        try:
            setattr(stub_instance, helper, _mk_wrapped(helper, orig))
        except Exception:
            continue
    return captured, originals


def _resolve_torch_module_from_candidates(
    mod,
    demo_dir: Path,
    component_name: str,
    warn: Callable[[str], None],
):
    """Use `_CANDIDATE_SUBMODULE_PATHS` from the stub module + an HF
    AutoModel load to resolve the torch submodule. This is the path
    op-synth partial stubs (which inline resolution into a
    module-level wrapper) need.

    Returns None on any failure — caller should try the next strategy.
    """
    candidates = getattr(mod, "_CANDIDATE_SUBMODULE_PATHS", None)
    if not candidates:
        return None
    hf_model_id = getattr(mod, "HF_MODEL_ID", None)
    if not hf_model_id:
        try:
            state_path = demo_dir / "_state.json"
            if state_path.is_file():
                data = json.loads(state_path.read_text(encoding="utf-8"))
                hf_model_id = data.get("model_id") or data.get("hf_model_id")
        except Exception:
            hf_model_id = None
    if not hf_model_id:
        warn("no HF_MODEL_ID constant or _state.json to load HF model from")
        return None
    try:
        import transformers

        model = transformers.AutoModel.from_pretrained(
            hf_model_id, trust_remote_code=True, torch_dtype="bfloat16", low_cpu_mem_usage=True
        )
        model.eval()
    except Exception as exc:
        warn(f"AutoModel.from_pretrained({hf_model_id!r}) raised: {exc}")
        return None
    for path in candidates:
        try:
            sub = _resolve_dotted(model, path)
            if sub is not None:
                return sub
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
    warn(f"none of _CANDIDATE_SUBMODULE_PATHS resolved on {hf_model_id}")
    return None


def _resolve_torch_module_from_manifest(
    demo_dir: Path,
    component_name: str,
    warn: Callable[[str], None],
):
    """Resolve via the capture manifest's `submodule_path` field. This is
    the most-portable fallback — every component that has captured
    inputs has a manifest pointing at the correct submodule.
    """
    safe = _safe_id(component_name)
    manifest_path = demo_dir / "_captured" / safe / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    sub_path = manifest.get("submodule_path")
    if not isinstance(sub_path, str) or not sub_path:
        return None

    hf_model_id: Optional[str] = None
    stubs_dir = demo_dir / "_stubs"
    for sibling in stubs_dir.glob("*.py"):
        try:
            txt = sibling.read_text(errors="ignore")
            import re as _re

            m = _re.search(r"HF_MODEL_ID\s*=\s*['\"]([^'\"]+)['\"]", txt)
            if m:
                hf_model_id = m.group(1)
                break
        except Exception:
            continue
    if not hf_model_id:
        warn(f"manifest has submodule_path={sub_path!r} but no HF_MODEL_ID resolvable")
        return None
    try:
        import transformers

        model = transformers.AutoModel.from_pretrained(
            hf_model_id, trust_remote_code=True, torch_dtype="bfloat16", low_cpu_mem_usage=True
        )
        model.eval()
        return _resolve_dotted(model, sub_path)
    except Exception as exc:
        warn(f"manifest-based resolution failed: {exc}")
        return None


def _resolve_torch_module(
    mod,
    demo_dir: Path,
    component_name: str,
    warn: Callable[[str], None],
):
    """Three-strategy torch submodule resolution. See callsite docstring."""
    getter = getattr(mod, "_get_torch_submodule", None)
    if getter is not None:
        try:
            t = getter()
            if t is not None:
                return t
        except Exception as exc:
            warn(f"_get_torch_submodule raised: {exc}")
    t = _resolve_torch_module_from_candidates(mod, demo_dir, component_name, warn)
    if t is not None:
        return t
    t = _resolve_torch_module_from_manifest(demo_dir, component_name, warn)
    if t is None:
        warn("all torch-module resolution strategies failed")
    return t


def _load_stub_module(stub_path: Path):
    """Import the stub file as a fresh module so monkey-patching its
    instance methods doesn't bleed into other stubs / probes."""
    if not stub_path.is_file():
        return None
    try:
        module_name = f"_tt_planner_actdiff_{stub_path.stem}_{id(stub_path)}"
        spec = importlib.util.spec_from_file_location(module_name, str(stub_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def localize_pcc_divergence(
    demo_dir: Path,
    component_name: str,
    *,
    threshold: float = 0.95,
    device: Any = None,
) -> Optional[LocalizationResult]:
    """Run layer-wise activation diff for one op-synth component and
    return the first `_apply_*` helper whose output diverges from the
    torch reference.

    Returns None (caller falls back to existing prompt path) if ANY of:
      - the stub is not op-synth (no `.opplan.json` sidecar),
      - the captured inputs directory is missing or empty,
      - the stub cannot be built (e.g. the device arg is None and
        `build()` requires one),
      - the torch reference's forward raises,
      - the ttnn stub's `__call__` raises.

    The function NEVER raises; any internal exception degrades to a
    `None` return with the failure surfaced via stderr only when
    `TT_PLANNER_ACTDIFF_VERBOSE=1` is set.
    """
    safe = _safe_id(component_name)
    opplan = demo_dir / "_stubs" / f"{safe}.opplan.json"
    stub_path = demo_dir / "_stubs" / f"{safe}.py"
    captured_dir = demo_dir / "_captured" / safe

    verbose = bool(os.environ.get("TT_PLANNER_ACTDIFF_VERBOSE"))

    def _warn(msg: str) -> None:
        if verbose:
            print(f"[activation_diff:{component_name}] {msg}", file=sys.stderr)

    pre_bound = _load_opplan_pre_bound(opplan)
    if not pre_bound:
        _warn(f"no op-synth manifest at {opplan}; skipping localization")
        return None

    if not stub_path.is_file():
        _warn(f"stub file missing at {stub_path}")
        return None

    args, kwargs, ref_output = _load_captured_io(captured_dir)
    if args is None and kwargs is None:
        _warn(f"no captured inputs under {captured_dir}")
        return None
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    mod = _load_stub_module(stub_path)
    if mod is None:
        _warn("could not import stub module")
        return None
    torch_module = _resolve_torch_module(mod, demo_dir, component_name, _warn)
    if torch_module is None:
        return None

    torch_captured, torch_handles = _install_torch_hooks(torch_module, pre_bound)
    torch_forward_ok = False
    try:
        try:
            import torch

            with contextlib_no_grad():
                _ = torch_module(*args, **kwargs)
            torch_forward_ok = True
        except Exception as exc:
            _warn(f"torch reference forward raised: {type(exc).__name__}: {exc}")
    finally:
        for h in torch_handles:
            try:
                h.remove()
            except Exception:
                pass

    ttnn_captured: Dict[str, Any] = {}
    ttnn_originals: List[Tuple[str, Any]] = []
    ttnn_forward_ok = False
    stub_instance = None
    if device is not None and hasattr(mod, "build"):
        try:
            stub_instance = mod.build(device, torch_module)
        except Exception as exc:
            _warn(f"stub build() raised: {type(exc).__name__}: {exc}")
            stub_instance = None
    if stub_instance is not None:
        ttnn_captured, ttnn_originals = _install_ttnn_hooks(stub_instance, pre_bound)
        try:
            try:
                _ = stub_instance(*args, **kwargs)
                ttnn_forward_ok = True
            except Exception as exc:
                _warn(f"ttnn stub __call__ raised: {type(exc).__name__}: {exc}")
        finally:
            for name, orig in ttnn_originals:
                try:
                    setattr(stub_instance, name, orig)
                except Exception:
                    pass

    torch_only_mode = stub_instance is None
    table: List[HelperDivergence] = []
    first_divergence: Optional[HelperDivergence] = None
    for entry in pre_bound:
        helper = entry["helper"]
        sub_path = entry["name"]
        ttnn_target = entry["ttnn_target"]
        torch_out_raw = torch_captured.get(helper)
        ttnn_out_raw = ttnn_captured.get(helper)
        torch_t = _to_torch_best_effort(torch_out_raw)
        ttnn_t = _to_torch_best_effort(ttnn_out_raw)
        stats = _tensor_stats(torch_t)
        if torch_only_mode:
            pcc_val: Optional[float] = None
            if torch_t is None:
                note = "torch-none"
            else:
                note = "torch-only"
        else:
            if torch_t is None and ttnn_t is None:
                note = "neither-captured"
                pcc_val = None
            elif torch_t is None:
                note = "torch-none"
                pcc_val = None
            elif ttnn_t is None:
                note = "ttnn-none"
                pcc_val = None
            else:
                pcc_val = _pcc(torch_t, ttnn_t)
                if pcc_val is None:
                    note = "pcc-failed"
                elif pcc_val < threshold:
                    note = "diverged"
                else:
                    note = "ok"
        row = HelperDivergence(
            helper=helper,
            submodule_path=sub_path,
            ttnn_target=ttnn_target,
            pcc=pcc_val,
            note=note,
            stats=stats,
        )
        table.append(row)
        if first_divergence is None and pcc_val is not None and pcc_val < threshold:
            first_divergence = row

    overall_note = "ok"
    if not torch_forward_ok and not ttnn_forward_ok:
        overall_note = "both-forwards-failed"
    elif not torch_forward_ok:
        overall_note = "torch-forward-failed"
    elif not ttnn_forward_ok:
        overall_note = "ttnn-forward-failed"

    return LocalizationResult(
        first_divergence=first_divergence,
        table=table,
        forward_succeeded=torch_forward_ok and ttnn_forward_ok,
        threshold=threshold,
        note=overall_note,
    )


def contextlib_no_grad():
    """Wrapper that returns a `torch.no_grad()` context if torch is
    importable, else a trivial null context. Centralised so a failed
    torch import doesn't crash the activation diff."""
    try:
        import torch

        return torch.no_grad()
    except Exception:
        return contextlib.nullcontext()


def format_localization_hint_block(
    component: str,
    result: Optional[LocalizationResult],
    *,
    max_table_rows: int = 12,
) -> str:
    """Render a compact `LOCALIZATION HINT` block for the LLM prompt.

    Returns "" if `result` is None (no signal — leave the prompt
    unchanged). When `result` is present, the block names the first
    diverging helper and lists the top few rows of the PCC table in
    execution order so the LLM can see whether the upstream helpers
    are clean and the divergence starts where reported."""
    if result is None:
        return ""

    torch_only = all(r.pcc is None for r in result.table) and any(r.stats is not None for r in result.table)

    lines: List[str] = []
    if torch_only:
        lines.append(
            f"REFERENCE ACTIVATION TRACE for `{component}` (torch-side only — ttnn "
            f"diff disabled, this is ground truth at every op boundary):"
        )
        lines.append(
            "  Use these expected (shape, dtype, mean, std, l2) values as a "
            "checklist: every `self._apply_<helper>(...)` in your ttnn "
            "`__call__` MUST produce a tensor of the listed shape (after any "
            "tile-padding is unpadded). If you cannot mentally match the "
            "expected magnitude (mean / std / l2) at a helper, that helper's "
            "call site is your most likely PCC bug. Common root causes: "
            "missing 1/sqrt(d_k) scaling on attention; wrong residual-add "
            "order; eps mismatch in layer_norm/rms_norm; activation variant "
            "(gelu_new vs gelu_pytorch_tanh vs silu); accumulation in "
            "bfloat8_b where bfloat16 is required."
        )
    else:
        lines.append(f"LOCALIZATION HINT for `{component}` (layer-wise PCC bisection):")
        lines.append(f"  diff threshold     : PCC >= {result.threshold:.3f} considered OK")
        lines.append(f"  forwards completed : {'yes' if result.forward_succeeded else 'no — see note below'}")
        if result.note != "ok":
            lines.append(f"  bisection note     : {result.note}")
        if result.first_divergence is not None:
            d = result.first_divergence
            pcc_s = f"{d.pcc:.4f}" if d.pcc is not None else "n/a"
            lines.append(
                f"  FIRST DIVERGENCE  : `{d.helper}` (op {d.ttnn_target}) " f"PCC={pcc_s} vs torch.{d.submodule_path}"
            )
            lines.append(
                "  ACTION: focus your edits on this single helper's call site "
                "in `__call__` and the helper body itself. Upstream helpers "
                "(listed above this row in the table) match the torch "
                "reference within tolerance and should NOT be modified. "
                "Common root causes for a single-helper PCC drop: dtype "
                "(bfloat8_b vs bfloat16), layout transition (TILE vs "
                "ROW_MAJOR), missing scaling (sqrt(head_dim) on attention), "
                "permute/reshape order, residual-add order, eps in "
                "layer_norm/rms_norm."
            )
        else:
            diverged_rows = [r for r in result.table if r.note == "diverged"]
            if not diverged_rows:
                lines.append(
                    "  no single helper diverged below threshold — the final-"
                    "output PCC drop comes from the GLUE code in `__call__` "
                    "(reshapes / permutes / residual adds / mask handling / "
                    "scaling factor / activation choice between helpers). "
                    "Recheck the inter-helper plumbing, NOT the helpers "
                    "themselves."
                )
            else:
                lines.append(
                    f"  {len(diverged_rows)} helpers diverged but execution order "
                    "could not be established (likely the torch hooks fired in "
                    "a different sequence than the ttnn helpers). Inspect the "
                    "table below."
                )
    lines.append("")
    header = (
        "  helper -> torch path -> expected stats" if torch_only else "  helper -> torch path -> PCC (execution order):"
    )
    lines.append(header)
    rendered = 0
    for row in result.table:
        if rendered >= max_table_rows:
            remaining = len(result.table) - rendered
            lines.append(f"    ... ({remaining} more helpers truncated)")
            break
        marker = "  "
        if result.first_divergence is not None and row.helper == result.first_divergence.helper:
            marker = "->"
        elif row.note == "diverged":
            marker = ".."
        if torch_only:
            stats = row.stats
            if stats is None:
                stat_s = f"NOT-CAPTURED  ({row.note})"
            elif "mean" not in stats:
                stat_s = f"shape={stats.get('shape')} dtype={stats.get('dtype')} (empty)"
            else:
                stat_s = (
                    f"shape={stats['shape']} dtype={stats['dtype']} "
                    f"mean={stats['mean']:+.3e} std={stats['std']:.3e} "
                    f"l2={stats['l2']:.3e}"
                )
            lines.append(f"    {marker} {row.helper:55s} -> {row.submodule_path}\n" f"        {stat_s}")
        else:
            pcc_s = "n/a" if row.pcc is None else f"{row.pcc:.4f}"
            lines.append(f"    {marker} {row.helper:55s} {pcc_s:>8s}  {row.note}")
        rendered += 1
    return "\n".join(lines) + "\n"


@dataclass
class DecodeLayerStats:
    """One row of the decode-mode per-layer activation trace.

    `layer_idx` is the 0-based index into the decoder layer list.
    `decode_step` is the 0-based decode position (0 = the FIRST
    generated token after prefill).  `shape` / `dtype` describe the
    layer's hidden-state output tensor at that decode step.  The
    remaining fields are scalar reductions over the flattened tensor
    in float32 -- the LLM uses them as a checklist to verify that
    the TT implementation produces tensors of comparable magnitude
    at every layer boundary.
    """

    layer_idx: int
    decode_step: int
    shape: Tuple[int, ...]
    dtype: str
    mean: float
    std: float
    l2: float
    abs_max: float
    note: str = "ok"


@dataclass
class DecodeLocalizationResult:
    """Output of :func:`localize_decode_divergence`.

    `table` is a list of :class:`DecodeLayerStats` in (decode_step,
    layer_idx) execution order -- iterating in order shows how the
    hidden state evolves both within a single step (top to bottom of
    the network) and across decode steps (the same layer at two
    different positions). `probed_decode_steps` is the set of
    positions actually captured (may differ from the requested set
    if greedy decode ended early or threw). `num_layers` is the
    number of decoder layers in the HF model. `note` is a short
    status string identifying the failure mode if any
    (`"ok"`, `"forward-raised"`, `"layers-not-resolved"`,
    `"hf-model-load-failed"`, `"timeout"`, `"transformers-missing"`).
    """

    table: List[DecodeLayerStats] = field(default_factory=list)
    probed_decode_steps: List[int] = field(default_factory=list)
    hf_model_id: str = ""
    num_layers: int = 0
    decoder_path: str = ""
    note: str = "ok"
    prompt_text: str = ""
    forward_succeeded: bool = False
    collapse_position: Optional[int] = None
    prefix_match_count: Optional[int] = None


_DECODER_LAYERS_CANDIDATES: Tuple[str, ...] = (
    "model.language_model.layers",
    "language_model.layers",
    "model.language_model.model.layers",
    "model.layers",
    "model.model.layers",
    "transformer.h",
    "gpt_neox.layers",
)


def _resolve_decoder_layers(hf_model) -> Tuple[Optional[str], Optional[Any]]:
    """Walk the candidate paths and return ``(dotted_path, layers)``
    for the first one that resolves to a non-empty list-like.

    Returns ``(None, None)`` if no candidate matched -- callers
    should degrade to a soft skip and surface ``note="layers-not-
    resolved"`` so the audit log records the architecture gap.
    """
    try:
        import torch
    except Exception:
        return None, None
    for path in _DECODER_LAYERS_CANDIDATES:
        try:
            layers = _resolve_dotted(hf_model, path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        if isinstance(layers, (list, tuple)):
            if len(layers) > 0:
                return path, layers
        try:
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return path, layers
        except Exception:
            continue
    return None, None


def _torch_dtype_from_string(name: str):
    """Map a short string like ``"bfloat16"`` to ``torch.bfloat16``.

    Returns ``None`` if torch isn't importable or the name is
    unrecognized -- callers degrade gracefully (the HF auto-dtype
    path will still pick something reasonable)."""
    try:
        import torch
    except Exception:
        return None
    canonical = (name or "").lower().strip()
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    return table.get(canonical)


def _hidden_state_from_layer_output(out: Any):
    """Decoder layers return either a Tensor or a tuple whose first
    element is the hidden state. Normalise to a single tensor for
    statistics computation. Returns ``None`` if the layer output is
    neither shape (only seen on broken / experimental architectures)."""
    try:
        import torch
    except Exception:
        return None
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out:
        first = out[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


def _tensor_to_layer_stats(
    t: Any,
    *,
    layer_idx: int,
    decode_step: int,
) -> Optional[DecodeLayerStats]:
    """Reduce a captured hidden-state tensor to the compact stats
    row carried into the repair prompt. Returns ``None`` if the
    tensor is empty / not introspectable.
    """
    try:
        import torch
    except Exception:
        return None
    if not isinstance(t, torch.Tensor):
        return None
    try:
        if t.dim() >= 2:
            slot = t[..., -1, :]
        else:
            slot = t
        ft = slot.detach().to(dtype=torch.float32).flatten()
        if ft.numel() == 0:
            return DecodeLayerStats(
                layer_idx=layer_idx,
                decode_step=decode_step,
                shape=tuple(t.shape),
                dtype=str(t.dtype),
                mean=0.0,
                std=0.0,
                l2=0.0,
                abs_max=0.0,
                note="empty",
            )
        return DecodeLayerStats(
            layer_idx=layer_idx,
            decode_step=decode_step,
            shape=tuple(t.shape),
            dtype=str(t.dtype),
            mean=float(ft.mean()),
            std=float(ft.std()) if ft.numel() > 1 else 0.0,
            l2=float(ft.norm()),
            abs_max=float(ft.abs().max()),
        )
    except Exception:
        return None


def _greedy_step_inputs(
    *,
    prev_input_ids,
    prev_past,
    next_token_id: Optional[int],
):
    """Build the (input_ids, past_key_values) pair for one greedy
    decode step. On the first call (prev_past is None) we pass the
    full prompt; on subsequent calls we pass only the new token and
    rely on the cache."""
    try:
        import torch
    except Exception:
        return None, None
    if prev_past is None:
        return prev_input_ids, None
    return (
        torch.tensor([[next_token_id]], dtype=torch.long),
        prev_past,
    )


def _safe_prompt_from_evidence(prompt_text: str, fallback: str) -> str:
    """Sanitize the evidence-derived prompt for HF tokenization.

    The evidence engine sometimes carries chat-template artifacts in
    its ``input_hint``; we strip the most common leading sentinels
    but keep the body intact. Empty prompts fall back to a stable
    default so the probe still runs (an empty prompt produces zero
    decode steps, which would defeat the purpose).
    """
    p = (prompt_text or "").strip()
    if not p:
        return fallback

    for prefix in ("<|user|>", "<start_of_turn>user", "<|im_start|>user"):
        if p.startswith(prefix):
            p = p[len(prefix) :].lstrip()
    return p or fallback


def localize_decode_divergence(
    *,
    model_id: str,
    prompt_text: str,
    probe_decode_steps: Optional[Sequence[int]] = None,
    max_total_steps: int = 16,
    torch_dtype: str = "bfloat16",
    timeout_s: float = 300.0,
    verbose: bool = False,
    collapse_position: Optional[int] = None,
    prefix_match_count: Optional[int] = None,
) -> Optional[DecodeLocalizationResult]:
    """Drive the HF reference end-to-end (prefill + greedy decode)
    with forward hooks on every decoder layer, and return per-layer
    activation statistics at the requested decode positions.

    Parameters
    ----------
    model_id
        HF repo id (e.g. ``"google/medgemma-4b-it"``). Must be
        loadable via :func:`transformers.AutoModelForCausalLM.from_pretrained`
        or :func:`transformers.AutoModel.from_pretrained`. Gated
        repos require a prior ``huggingface-cli login``.
    prompt_text
        The exact prompt the demo ran. The probe re-tokenizes this
        with the model's own tokenizer and seeds prefill with it.
    probe_decode_steps
        Sequence of 0-based decode positions to snapshot. When
        ``None`` we default to a sensible spread around
        ``collapse_position`` (last good step, first divergent step,
        a step into the collapsed regime).  Steps beyond
        ``max_total_steps`` are silently dropped.
    max_total_steps
        Hard cap on greedy decode iterations. 16 is enough to cover
        the medgemma collapse at step 11 with a few post-collapse
        positions; raise it for collapses that happen later in
        decoding.
    torch_dtype
        ``"bfloat16"`` (default) / ``"float16"`` / ``"float32"``.
        bfloat16 keeps a 4B-parameter model under ~10 GB of RAM
        when loaded on CPU; float32 will OOM on most dev hosts.
    timeout_s
        Wall-clock budget for the entire probe (model load +
        prefill + decode). Exceeded -> ``note="timeout"``, partial
        results are still returned for any positions already
        captured. Default 5 minutes.
    verbose
        Emit short progress lines to stderr while running.
    collapse_position
        Token index where the demo's output collapsed, propagated
        from :class:`TextEvidence`. Used (a) to derive default
        ``probe_decode_steps`` and (b) stashed in the result for
        prompt-block rendering.
    prefix_match_count
        Number of leading tokens that matched HF, propagated from
        :class:`TextEvidence`. Stashed in the result for context.

    Returns
    -------
    A :class:`DecodeLocalizationResult` on success (including
    partial success -- the ``note`` field identifies the failure
    mode and ``table`` carries whatever was captured before the
    failure). Returns ``None`` on configuration errors that prevent
    any meaningful run (transformers package missing, model id
    blank). The caller should treat ``None`` and a result with
    empty ``table`` identically: fall back to the existing prompt
    path.
    """
    import time

    if not model_id:
        return None

    def _log(msg: str) -> None:
        if verbose:
            print(f"[decode_divergence:{model_id}] {msg}", file=sys.stderr)

    try:
        import torch
    except Exception:
        return DecodeLocalizationResult(
            hf_model_id=model_id,
            note="torch-missing",
            prompt_text=prompt_text,
        )

    try:
        import transformers
    except Exception:
        return DecodeLocalizationResult(
            hf_model_id=model_id,
            note="transformers-missing",
            prompt_text=prompt_text,
        )

    start = time.monotonic()

    def _elapsed() -> float:
        return time.monotonic() - start

    def _time_left() -> float:
        return timeout_s - _elapsed()

    if _time_left() <= 0:
        return DecodeLocalizationResult(
            hf_model_id=model_id,
            note="timeout",
            prompt_text=prompt_text,
        )

    dtype_obj = _torch_dtype_from_string(torch_dtype)
    dtype_kwarg: Dict[str, Any] = {"low_cpu_mem_usage": True}
    if dtype_obj is not None:
        dtype_kwarg["torch_dtype"] = dtype_obj

    hf_model = None
    last_load_exc: Optional[BaseException] = None

    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        model_type = (cfg.model_type or "").lower()
    except Exception as exc:
        _log(f"AutoConfig failed: {type(exc).__name__}: {exc}")
        model_type = ""
    candidate_classes: List[str] = []
    if "gemma3" in model_type:
        candidate_classes.append("Gemma3ForConditionalGeneration")
    candidate_classes.extend(["AutoModelForCausalLM", "AutoModel"])
    for cls_name in candidate_classes:
        if _time_left() <= 5.0:
            break
        try:
            if cls_name == "AutoModelForCausalLM":
                from transformers import AutoModelForCausalLM

                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=False,
                    **dtype_kwarg,
                )
            elif cls_name == "AutoModel":
                from transformers import AutoModel

                hf_model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=False,
                    **dtype_kwarg,
                )
            elif cls_name == "Gemma3ForConditionalGeneration":
                from transformers import Gemma3ForConditionalGeneration

                hf_model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id,
                    **dtype_kwarg,
                )
            if hf_model is not None:
                _log(f"loaded via {cls_name}")
                break
        except Exception as exc:
            last_load_exc = exc
            _log(f"{cls_name} failed: {type(exc).__name__}: {exc}")
            continue
    if hf_model is None:
        return DecodeLocalizationResult(
            hf_model_id=model_id,
            note=(f"hf-model-load-failed: " f"{type(last_load_exc).__name__ if last_load_exc else 'unknown'}"),
            prompt_text=prompt_text,
        )

    try:
        hf_model.eval()
    except Exception:
        pass

    decoder_path, layers = _resolve_decoder_layers(hf_model)
    if layers is None or decoder_path is None:
        return DecodeLocalizationResult(
            hf_model_id=model_id,
            note="layers-not-resolved",
            prompt_text=prompt_text,
        )
    num_layers = len(layers)
    _log(f"resolved {num_layers} decoder layers at `{decoder_path}`")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        _log(f"tokenizer load failed: {type(exc).__name__}: {exc}")
        return DecodeLocalizationResult(
            hf_model_id=model_id,
            num_layers=num_layers,
            decoder_path=decoder_path,
            note="tokenizer-load-failed",
            prompt_text=prompt_text,
        )
    fallback_prompt = "How will I get to the moon"
    safe_prompt = _safe_prompt_from_evidence(prompt_text, fallback_prompt)
    try:
        try:
            chat = [{"role": "user", "content": safe_prompt}]
            input_ids = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        except Exception:
            input_ids = tokenizer(safe_prompt, return_tensors="pt").input_ids
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([list(input_ids)], dtype=torch.long)
    except Exception as exc:
        _log(f"tokenization failed: {type(exc).__name__}: {exc}")
        return DecodeLocalizationResult(
            hf_model_id=model_id,
            num_layers=num_layers,
            decoder_path=decoder_path,
            note="tokenization-failed",
            prompt_text=safe_prompt,
        )

    _MAX_DECODE_STEPS_CEILING = 64
    if probe_decode_steps is None:
        if collapse_position is not None and collapse_position > 0:
            input_len = int(input_ids.shape[-1])
            anchor = max(0, int(collapse_position) - input_len)
            need_steps = anchor + 5
            if need_steps > max_total_steps:
                max_total_steps = min(_MAX_DECODE_STEPS_CEILING, need_steps)
                _log(
                    f"auto-extended max_total_steps to {max_total_steps} "
                    f"so collapse anchor decode_step={anchor} (=token "
                    f"{collapse_position} - prompt_len {input_len}) fits"
                )
            steps_set = {
                0,
                max(0, anchor - 1),
                anchor,
                anchor + 1,
                min(max_total_steps - 1, anchor + 4),
            }
        else:
            steps_set = {0, 4, 8, 12}
        probe_decode_steps_list = sorted(s for s in steps_set if 0 <= s < max_total_steps)
    else:
        probe_decode_steps_list = sorted({int(s) for s in probe_decode_steps if 0 <= int(s) < max_total_steps})
    if not probe_decode_steps_list:
        probe_decode_steps_list = [0]

    captured: Dict[int, Any] = {}
    handles: List[Any] = []
    for i, layer in enumerate(layers):
        if not hasattr(layer, "register_forward_hook"):
            continue

        def _mk(i: int) -> Callable[[Any, Any, Any], None]:
            def _hook(_m, _inp, out):
                captured[i] = out

            return _hook

        try:
            handles.append(layer.register_forward_hook(_mk(i)))
        except Exception:
            continue

    table: List[DecodeLayerStats] = []
    captured_steps: List[int] = []
    forward_ok = True
    note = "ok"

    try:
        with contextlib_no_grad():
            past = None

            try:
                out = hf_model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
                past = getattr(out, "past_key_values", None)

                last_logits = getattr(out, "logits", None)
                if last_logits is None:
                    raise RuntimeError("HF model did not return logits")
                next_token = int(last_logits[..., -1, :].argmax(dim=-1).item())
            except Exception as exc:
                _log(f"prefill failed: {type(exc).__name__}: {exc}")
                forward_ok = False
                note = "prefill-failed"
                next_token = -1

            if forward_ok and 0 in probe_decode_steps_list:
                for li in range(num_layers):
                    h = _hidden_state_from_layer_output(captured.get(li))
                    row = _tensor_to_layer_stats(h, layer_idx=li, decode_step=0)
                    if row is not None:
                        table.append(row)
                captured_steps.append(0)

            for k in range(1, max_total_steps):
                if not forward_ok:
                    break
                if _time_left() < 5.0:
                    note = "timeout"
                    forward_ok = False
                    break
                captured.clear()
                try:
                    out = hf_model(
                        input_ids=torch.tensor([[next_token]], dtype=torch.long),
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = getattr(out, "past_key_values", past)
                    last_logits = getattr(out, "logits", None)
                    if last_logits is None:
                        forward_ok = False
                        note = "decode-step-no-logits"
                        break
                    next_token = int(last_logits[..., -1, :].argmax(dim=-1).item())
                except Exception as exc:
                    _log(f"decode step {k} failed: {type(exc).__name__}: {exc}")
                    forward_ok = False
                    note = f"decode-step-failed-at-{k}"
                    break
                if k in probe_decode_steps_list:
                    for li in range(num_layers):
                        h = _hidden_state_from_layer_output(captured.get(li))
                        row = _tensor_to_layer_stats(h, layer_idx=li, decode_step=k)
                        if row is not None:
                            table.append(row)
                    captured_steps.append(k)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

        try:
            del hf_model
        except Exception:
            pass
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    return DecodeLocalizationResult(
        table=table,
        probed_decode_steps=captured_steps,
        hf_model_id=model_id,
        num_layers=num_layers,
        decoder_path=decoder_path,
        note=note,
        prompt_text=safe_prompt,
        forward_succeeded=forward_ok,
        collapse_position=collapse_position,
        prefix_match_count=prefix_match_count,
    )


def format_decode_localization_hint_block(
    model_id: str,
    result: Optional[DecodeLocalizationResult],
    *,
    max_layers_shown: int = 6,
    show_all_layers_at_collapse_step: bool = True,
) -> str:
    """Render the per-layer decode trace as a compact prompt block.

    Returns ``""`` when ``result`` is ``None``, when no rows were
    captured (e.g. timeout before any decode step completed), or
    when the result lacks decoder-layer info -- the caller leaves
    the prompt unchanged so a missing probe is equivalent to the
    pre-probe behavior.

    The block is intentionally compact: by default we show the
    FIRST, MIDDLE, and LAST few decoder layers per decode step plus
    ALL layers at the decode step closest to the collapse position.
    A full N=34 layer x M=5 step dump would dominate the prompt and
    is rarely needed -- the LLM only has to find the FIRST layer
    whose TT-side stats deviate from this reference.
    """
    if result is None:
        return ""
    if not result.table or result.num_layers == 0:
        if result.note and result.note != "ok":
            return f"DECODE-LAYER PROBE for `{model_id}` skipped " f"(note={result.note}).\n"
        return ""

    lines: List[str] = []
    lines.append(
        f"DECODE-LAYER ACTIVATION REFERENCE for `{model_id}` " f"(HF side, last-token slice per decoder layer):"
    )
    lines.append(f"  decoder path        : {result.decoder_path}  " f"({result.num_layers} layers)")
    if result.collapse_position is not None:
        lines.append(
            f"  collapse position   : token {result.collapse_position} " f"(prefix_match={result.prefix_match_count})"
        )
    lines.append(f"  probed decode steps : " f"{', '.join(str(s) for s in result.probed_decode_steps)}")
    if result.note != "ok":
        lines.append(f"  status              : {result.note}")
    lines.append(
        "  USE THIS AS GROUND TRUTH: for each (layer, step) row below, "
        "your ttnn implementation MUST produce a tensor of the listed "
        "shape with (mean, std, l2, abs_max) within ~5%% (bf16) / ~15%% "
        "(bfp8). The FIRST layer at a decode step that exceeds this "
        "tolerance is your bug location. Common root causes for a "
        "single (layer, step) divergence: wrong RoPE freq dispatch on "
        "sliding vs full layers, KV-cache slot off-by-one at a page "
        "boundary, per-head Q/K-norm applied along the wrong dim, "
        "sliding-window mask collapsing to identity, weight-conversion "
        "permutation applied to a 1-D norm vector. Common root causes "
        "for ALL layers diverging starting at a given step: the demo "
        "is reading the wrong token id back from the device (sampler "
        "issue), the page-table is corrupt (paged-attention writes "
        "past the allocated block), or a logit softcap was added "
        "where the HF config has it disabled."
    )
    lines.append("")

    L = result.num_layers
    top_set = list(range(min(max_layers_shown // 3, L)))
    mid_start = max(0, L // 2 - 1)
    mid_set = list(range(mid_start, min(L, mid_start + max_layers_shown // 3)))
    bot_set = list(range(max(0, L - max_layers_shown // 3), L))
    layers_per_step_default = sorted(set(top_set + mid_set + bot_set))
    if not layers_per_step_default:
        layers_per_step_default = list(range(min(L, max_layers_shown)))

    full_step: Optional[int] = None
    if show_all_layers_at_collapse_step and result.collapse_position is not None and result.probed_decode_steps:
        candidates = sorted(
            result.probed_decode_steps,
            key=lambda s: abs(s - 0),
            reverse=True,
        )

        full_step = candidates[0] if candidates else None

    rows_by_step: Dict[int, List[DecodeLayerStats]] = {}
    for row in result.table:
        rows_by_step.setdefault(row.decode_step, []).append(row)

    for step in result.probed_decode_steps:
        rows = rows_by_step.get(step, [])
        if not rows:
            continue
        layers_to_show = list(range(result.num_layers)) if step == full_step else layers_per_step_default
        rows_by_layer = {r.layer_idx: r for r in rows}
        full_or_not = "(all layers)" if step == full_step else "(sampled)"
        lines.append(f"  decode_step={step}  {full_or_not}")
        last_emitted: Optional[int] = None
        for li in layers_to_show:
            r = rows_by_layer.get(li)
            if r is None:
                continue
            if last_emitted is not None and li - last_emitted > 1:
                lines.append(f"      ...  (layers {last_emitted+1}..{li-1} omitted)")
            lines.append(
                f"      layer {r.layer_idx:>2d}  "
                f"shape={r.shape}  dtype={r.dtype}  "
                f"mean={r.mean:+.3e}  std={r.std:.3e}  "
                f"l2={r.l2:.3e}  abs_max={r.abs_max:.3e}"
            )
            last_emitted = r.layer_idx
        lines.append("")

    return "\n".join(lines) + "\n"
