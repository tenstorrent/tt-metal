"""Chained-pipeline end-to-end test emitter.

After a model is brought up component-by-component (per-component PCC
tests under ``tests/pcc/``), the per-component validation doesn't tell
you whether the components compose correctly into the model's actual
inference flow. Hand-written ``test_e2e.py`` files like
``models/demos/vision/segmentation/sam2_hiera_tiny/tests/test_e2e.py``
fill that gap, but the planner has never auto-generated one — so for
every newly-ported model the user has had to write it themselves.

This module emits a skeleton ``test_e2e.py`` for any successfully-
brought-up model. The skeleton:

  * Loads the HF model for reference.
  * Imports every TT component the plan classified as on-device.
  * Provides ``# TODO[e2e]`` markers at each integration point, showing
    the dotted submodule path from the bring-up plan so the wiring
    work is mechanical, not investigative.
  * Falls back to HF for any component the plan classified as CPU.
  * Compares the final output to HF reference at PCC ≥ 0.95 (configurable).
  * Reuses the same probe-input helper functions as the hand-written
    SAM2 e2e (``_to_tt`` / ``_to_torch`` / ``_resolve``) so the emitted
    test is consistent in style with existing in-tree tests.

This is a SKELETON: the dataflow between TT modules can't be inferred
from component metadata alone (it requires knowledge of the model's
forward graph). The TODO markers point at exactly which lines need
human or LLM completion. Future work: extend with a torch.fx-based
dataflow inference pass that fills in the wiring automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_STATUS_REUSE = "REUSE"
_STATUS_ADAPT = "ADAPT"
_STATUS_NEW = "NEW"
_STATUS_CPU_FALLBACK = "CPU-fallback"


@dataclass
class E2EComponentInfo:
    """One row of the emitter's per-component plan-summary table."""

    name: str
    status: str
    submodule_path: Optional[str]
    tt_module_import: Optional[str]
    tt_class_name: Optional[str]
    on_device: bool


_E2E_TEMPLATE = '''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline test for {model_id} on TT hardware.

AUTO-GENERATED SKELETON by ``scripts.tt_hw_planner.e2e_emitter``.

Component plan (from ``BRING_UP_PLAN.md``):
{component_summary}

Each TT-ported component appears below as a ``# TODO[e2e]`` marker
with its dotted submodule_path. Replace each marker with the actual
TT call:

    1. Convert the HF input tensor at that point to a ttnn.Tensor
       via ``_to_tt(...)``.
    2. Call ``<TtModuleClass>(...).forward(tt_input, ...)``.
    3. Convert the output back to torch via ``_to_torch(...)``.
    4. Replace the corresponding ``hf_module(...)`` call with the
       TT output so the rest of the HF forward sees the TT-produced
       activation.

Components classified as CPU-fallback continue to run on CPU via
the HF reference — no wiring required for those.

Usage:
    pytest {test_path} -svv --timeout=600
"""
from __future__ import annotations

import pytest
import torch
import transformers

import ttnn
from models.common.utility_functions import comp_pcc

# Component-specific TT module imports — the planner derived these from
# the per-component bring-up status. Components marked CPU-fallback are
# intentionally absent (they run on CPU via the HF reference forward).
{tt_imports}

HF_MODEL_ID = "{model_id}"
PCC_TARGET = {pcc_target}


# ---------------------------------------------------------------------------
# Boilerplate helpers (same as the SAM2 hand-written e2e — kept here so the
# emitted test is fully self-contained and doesn't add a new dependency).
# ---------------------------------------------------------------------------
def _resolve(obj, dotted):
    """Resolve a dotted attribute path on an object, e.g. ``model.layers.0.attention``."""
    cur = obj
    for tok in dotted.replace("[", ".").replace("]", "").split("."):
        if tok == "":
            continue
        cur = cur[int(tok)] if tok.isdigit() else getattr(cur, tok)
    return cur


def _is_mesh_device(device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


def _to_tt(x, device):
    """Convert a torch.Tensor / scalar / ttnn.Tensor to a ttnn.Tensor on ``device``."""
    if isinstance(x, ttnn.Tensor):
        return x
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        t = x.to(torch.bfloat16)
        if _is_mesh_device(device):
            try:
                return ttnn.from_torch(
                    t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
            except (AttributeError, TypeError):
                pass
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return x


def _to_torch(t, device):
    """Convert a ttnn.Tensor back to a torch.Tensor (mesh-aware)."""
    if not isinstance(t, ttnn.Tensor):
        return t
    try:
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(device)
    except Exception:
        pass
    try:
        from models.common.auto_compose import to_torch_auto_compose
        return to_torch_auto_compose(t)
    except Exception:
        return ttnn.to_torch(t)


# ---------------------------------------------------------------------------
# The chained pipeline test itself.
# ---------------------------------------------------------------------------
def test_e2e_pipeline(device_params, device):
    """Run the full HF forward AND the TT-ported forward, then PCC-compare.

    Implementation plan (mechanically fillable from the TODO[e2e] markers
    below; one marker per TT-ported component in the bring-up plan):

    1. Run HF reference forward and capture the final output.
    2. For each TT-ported component, replace the HF call with a TT call.
       The marker shows the dotted submodule_path so you know WHERE the
       call lives inside the HF model.
    3. Final ``comp_pcc(hf_output, tt_output, pcc=PCC_TARGET)`` decides
       the test verdict.
    """
    # ------------------------------------------------------------------
    # HF reference forward
    # ------------------------------------------------------------------
    hf_model = transformers.AutoModel.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    hf_model.eval()

    # TODO[e2e]: replace this dummy input with the model-specific probe
    # used by the per-component PCC tests (see tests/pcc/test_*.py for
    # the canonical synthetic input shapes for this architecture).
    sample_input = None  # e.g. torch.randn(1, 3, 224, 224) for vision

    with torch.no_grad():
        hf_output_ref = hf_model(sample_input) if sample_input is not None else None

    # ------------------------------------------------------------------
    # TT-ported pipeline — fill in each TODO[e2e] marker below
    # ------------------------------------------------------------------
{tt_pipeline_body}

    # ------------------------------------------------------------------
    # Final PCC check
    # ------------------------------------------------------------------
    if hf_output_ref is None:
        pytest.skip(
            "sample_input is None — fill in the TODO[e2e] above with a "
            "real probe before running this test."
        )

    passed, pcc = comp_pcc(hf_output_ref, tt_output, pcc=PCC_TARGET)
    assert passed, f"end-to-end PCC {{pcc}} < target {{PCC_TARGET}}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__ + "::test_e2e_pipeline", "-svv"]))
'''


_TT_PIPELINE_LINE_PORTED = (
    "    # TODO[e2e]: replace `_resolve(hf_model, {sub_path!r})(...)` with the TT path.\n"
    "    # TT module: {tt_module_import}.{tt_class_name}\n"
    "    # Status: {status} — port present, requires explicit wiring.\n"
)

_TT_PIPELINE_LINE_CPU = (
    "    # CPU-fallback: {name} (submodule path: {sub_path!r})\n"
    "    # No wiring needed; the HF forward computes this on CPU automatically.\n"
)


def _format_component_summary(components: List[E2EComponentInfo]) -> str:
    """Human-readable summary for the docstring header."""
    if not components:
        return "    (no components — empty plan)"
    lines = []
    for c in components:
        marker = "TT  " if c.on_device else "CPU "
        sub = c.submodule_path or "<unknown submodule>"
        lines.append(f"    {marker} [{c.status:<13s}] {c.name:<25s}  →  {sub}")
    return "\n".join(lines)


def _format_tt_pipeline_body(components: List[E2EComponentInfo]) -> str:
    """Per-component TODO markers, ordered by component name for
    determinism. The emitted body is a sequence of comment blocks; the
    actual TT calls are filled in by hand or by a follow-up LLM pass.

    A final dummy assignment is appended so the test compiles even when
    the user hasn't filled in any markers yet (``tt_output = None`` then
    the test pytest-skips at the final PCC check)."""
    if not components:
        return "    # (no components emitted)\n    tt_output = None"

    parts = []
    for c in components:
        sub_path = c.submodule_path or "<unknown.submodule.path>"
        if c.on_device:
            parts.append(
                _TT_PIPELINE_LINE_PORTED.format(
                    sub_path=sub_path,
                    tt_module_import=c.tt_module_import or "(unknown — see plan)",
                    tt_class_name=c.tt_class_name or "(unknown class)",
                    status=c.status,
                )
            )
        else:
            parts.append(_TT_PIPELINE_LINE_CPU.format(name=c.name, sub_path=sub_path))
    parts.append("    tt_output = None  # TODO[e2e]: set this to the final TT-side activation\n")
    return "".join(parts)


def _format_tt_imports(components: List[E2EComponentInfo]) -> str:
    """Emit one COMMENTED import-hint per on-device component.

    Why comments and not real imports: the bring-up plan reliably
    records the TT module *file* (``tt_reuse_target``) but doesn't
    always know the exact *class* exported from that file — for
    registry-resolved components the registry knows the TT class but
    doesn't propagate it through the Component dataclass, and for
    primary-extractor components the class identifier is just the
    lowercase component ``kind`` (``patch_embed``, ``self_attention``),
    not a real Python class.

    Emitting a broken hard import would make pytest collection fail at
    discovery time — the test wouldn't even be visible to the user.
    Emitting commented hints keeps the file collectable AND points the
    user/LLM at the exact module to consult when wiring the body. The
    LLM follow-up pass that fills the TODO[e2e] markers will turn the
    relevant hints into real imports."""
    on_device = [c for c in components if c.on_device]
    if not on_device:
        return "# (no on-device TT modules; nothing to import)"
    lines: List[str] = [
        "# TT module hints (commented so pytest collection always works).",
        "# When you wire a TODO[e2e] marker below, uncomment + correct the",
        "# corresponding import here.",
    ]
    seen = set()
    for c in on_device:
        if c.tt_module_import:
            key = (c.tt_module_import, c.tt_class_name or "?")
            if key in seen:
                continue
            seen.add(key)
            cls_hint = c.tt_class_name or "<ClassName>"
            lines.append(f"# from {c.tt_module_import} import {cls_hint}  " f"# for {c.name} ({c.status})")
        else:
            lines.append(
                f"# TODO[e2e]: locate the TT module for component {c.name!r} "
                f"({c.status}; plan recorded no tt_reuse_target)"
            )
    return "\n".join(lines)


def _looks_like_python_class_name(name: Optional[str]) -> bool:
    """Heuristic: does ``name`` look like a real Python class identifier?

    The bring-up plan's Component.class_name is sometimes the HF class
    (e.g. ``Sam2VisionNeck`` — CamelCase) and sometimes None for
    primary-extractor components, where only the lowercase ``kind``
    (e.g. ``patch_embed``) is available. The emitter can only safely
    emit ``from X import Y`` when ``Y`` is the real class identifier;
    using the lowercase concept name produces an ImportError at pytest
    collection. So distinguish here."""
    if not name:
        return False

    if not name[0].isupper():
        return False
    return all(ch.isalnum() or ch == "_" for ch in name)


def _component_to_e2e_info(c) -> E2EComponentInfo:
    """Adapt a ``bringup_plan.Component`` to our local dataclass. We avoid
    a hard import of bringup_plan to keep this module decoupled (e2e_emitter
    is imported only as a leaf consumer; cycles would be bad)."""
    status = getattr(c, "status", _STATUS_NEW)

    on_device = status in (_STATUS_REUSE, _STATUS_ADAPT, _STATUS_NEW)

    tt_path = getattr(c, "tt_reuse_target", None)
    tt_module_import: Optional[str] = None
    if tt_path:
        p = str(tt_path)
        if p.endswith(".py"):
            p = p[:-3]

        tt_module_import = p.replace("/", ".").lstrip(".")
    raw_class_name = getattr(c, "class_name", None)
    tt_class_name: Optional[str] = raw_class_name if _looks_like_python_class_name(raw_class_name) else None

    if tt_class_name is None:
        tt_module_import = None
    return E2EComponentInfo(
        name=getattr(c, "name", "<unnamed>"),
        status=status,
        submodule_path=getattr(c, "submodule_path", None),
        tt_module_import=tt_module_import,
        tt_class_name=tt_class_name,
        on_device=on_device,
    )


def render_e2e_test(
    *,
    model_id: str,
    components: List,
    test_path: str = "models/demos/<model>/tests/test_e2e.py",
    pcc_target: float = 0.95,
) -> str:
    """Render the e2e test source as a string. Caller writes to disk.

    Args:
        model_id: HF model id (e.g. ``facebook/sam2-hiera-tiny``).
        components: list of ``bringup_plan.Component`` objects.
        test_path: where the file will live, for the docstring's
            usage example. Doesn't affect content correctness.
        pcc_target: PCC threshold for the final comparison.

    Returns:
        Full file content as a string."""
    infos = [_component_to_e2e_info(c) for c in components]
    return _E2E_TEMPLATE.format(
        model_id=model_id,
        test_path=test_path,
        pcc_target=pcc_target,
        component_summary=_format_component_summary(infos),
        tt_imports=_format_tt_imports(infos),
        tt_pipeline_body=_format_tt_pipeline_body(infos),
    )


def emit_e2e_pipeline_test(
    *,
    model_id: str,
    components: List,
    output_path: Path,
    pcc_target: float = 0.95,
    overwrite: bool = False,
) -> Optional[Path]:
    """Write an e2e test skeleton to disk.

    Returns the written path on success, or ``None`` if the file
    already existed and ``overwrite=False`` (the common no-op case
    after the first bring-up run)."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rel_test_path = str(output_path).replace(str(Path.cwd()) + "/", "")
    content = render_e2e_test(
        model_id=model_id,
        components=components,
        test_path=rel_test_path,
        pcc_target=pcc_target,
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _resolve_tt_class_from_file(tt_module_path: Path) -> Optional[str]:
    """Return the first `class X(...):` defined in a TT module file.

    Used to discover the TT class name to import when the bring-up plan
    doesn't propagate it (the registry has ``tt_class`` but Component
    doesn't carry it). Cheap AST scan — no execution of the module.
    Returns None if the file can't be parsed or has no class def.

    Note: many tt_transformers modules expose one primary class per
    file (e.g. ``Attention``, ``MLP``, ``Embedding``, ``RMSNorm``).
    Picking the FIRST class is correct for 95% of cases; for the
    handful of multi-class files (e.g. ``embedding.py`` exposing both
    ``Embedding`` and ``ScaledEmbedding``), the user can edit the
    emitted import. Worth the simplicity over a full registry-lookup
    integration."""
    import ast

    try:
        src = tt_module_path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node.name
    return None


def _components_with_captured_io(
    captured_dir: Path,
    components: List[E2EComponentInfo],
) -> List[E2EComponentInfo]:
    """Return the subset of components for which capture_inputs.py
    successfully recorded ``_captured/<comp>/output.pt``. Order is
    preserved from the component list (which is the order the bring-up
    plan produced — already roughly forward-pass order)."""
    out: List[E2EComponentInfo] = []
    for c in components:
        if not c.on_device:
            continue

        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in c.name)
        comp_dir = captured_dir / safe
        if (comp_dir / "output.pt").exists():
            out.append(c)
    return out


_WIRED_COMPONENT_BLOCK = """    # -- {name} ({status}) --------------------------------------------------
    # captured during HF forward at submodule {sub_path!r}
    _comp_dir = _CAPTURED_ROOT / {safe_name!r}
    _args, _kwargs, _ref_output = _load_captured_io(_comp_dir)
    # TODO[e2e-wired]: convert _args[0] (and any tensor kwargs) to ttnn
    # and pass them through the TT module's forward(). The reference
    # input/output shapes are recorded in _comp_dir/manifest.json.
    _tt_inputs = tuple(_to_tt(a, device) for a in _args)
    _tt_kwargs = {{k: (_to_tt(v, device) if hasattr(v, "shape") else v) for k, v in _kwargs.items()}}
    {tt_call_expr}
    _tt_output_torch = _to_torch(_tt_output, device)
    _passed_{safe_idx}, _pcc_{safe_idx} = comp_pcc(_ref_output, _tt_output_torch, pcc=PCC_TARGET)
    _component_results.append(({name!r}, _passed_{safe_idx}, _pcc_{safe_idx}))
"""


def _format_wired_pipeline_body(
    components_with_io: List[E2EComponentInfo],
    repo_root: Path,
) -> str:
    """Generate one wired block per component that has captured I/O."""
    if not components_with_io:
        return (
            "    pytest.skip(\n"
            '        "No captured I/O found. Run "\n'
            '        "`python -m scripts.tt_hw_planner capture-inputs <model-id>` first."\n'
            "    )\n"
            "    _component_results = []  # unreachable\n"
        )
    parts: List[str] = ["    _component_results = []\n"]
    for idx, c in enumerate(components_with_io):
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in c.name)

        tt_class = c.tt_class_name
        if not tt_class and c.tt_module_import:
            rel = Path(*c.tt_module_import.split(".")).with_suffix(".py")
            tt_class = _resolve_tt_class_from_file(repo_root / rel)
        if tt_class and c.tt_module_import:
            tt_call_expr = (
                f"_tt_output = {tt_class}(device=device).forward(*_tt_inputs, **_tt_kwargs) "
                f" # FROM {c.tt_module_import}"
            )
        else:
            tt_call_expr = (
                "_tt_output = _args[0]  # TODO[e2e-wired]: replace with the real TT class call "
                f"(no TT class resolved for {c.name!r})"
            )
        parts.append(
            _WIRED_COMPONENT_BLOCK.format(
                name=c.name,
                status=c.status,
                sub_path=c.submodule_path or "<unknown>",
                safe_name=safe_name,
                safe_idx=idx,
                tt_call_expr=tt_call_expr,
            )
        )
    parts.append(
        "    _failures = [n for n, p, _ in _component_results if not p]\n"
        "    if _failures:\n"
        "        pytest.fail(\n"
        '            f"per-component PCC failed for: {_failures}. "\n'
        '            f"Full results: {_component_results}"\n'
        "        )\n"
        "    tt_output = _component_results[-1][2]  # last component's PCC value (placeholder)\n"
    )
    return "".join(parts)


def _format_wired_tt_imports(components_with_io: List[E2EComponentInfo], repo_root: Path) -> str:
    """Emit real import lines for components whose TT class can be
    resolved. Components whose class can't be resolved get a TODO
    comment (same defensive policy as the skeleton emitter)."""
    seen: set = set()
    lines: List[str] = []
    for c in components_with_io:
        tt_class = c.tt_class_name
        if not tt_class and c.tt_module_import:
            rel = Path(*c.tt_module_import.split(".")).with_suffix(".py")
            tt_class = _resolve_tt_class_from_file(repo_root / rel)
        if c.tt_module_import and tt_class:
            key = (c.tt_module_import, tt_class)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"from {c.tt_module_import} import {tt_class}  # for {c.name}")
        else:
            lines.append(
                f"# TODO[e2e-wired]: import TT class for {c.name!r} "
                f"(tt_module={c.tt_module_import!r}, class unresolved)"
            )
    return "\n".join(lines) if lines else "# (no resolvable TT imports)"


_WIRED_E2E_TEMPLATE = '''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wired end-to-end pipeline test for {model_id} on TT hardware.

AUTO-GENERATED by ``scripts.tt_hw_planner.e2e_emitter.render_e2e_wired``.

Unlike the skeleton emitter, this version uses CAPTURED per-component
I/O produced by ``capture-inputs``. Each block below:
  1. Loads ``args.pt`` / ``kwargs.pt`` / ``output.pt`` from
     ``_CAPTURED_ROOT/<component>/``.
  2. Calls the corresponding TT module with the captured input.
  3. Compares the TT output to the captured reference via ``comp_pcc``.

A failing block fails the whole test; the failure message names which
components are broken.

Component coverage:
{component_summary}

Usage:
    pytest {test_path} -svv --timeout=600
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import transformers

import ttnn
from models.common.utility_functions import comp_pcc
from scripts.tt_hw_planner.activation_diff import _load_captured_io

# TT module imports — resolved via AST scan of each component's
# tt_reuse_target. Unresolved imports surface as TODO comments below.
{tt_imports}

HF_MODEL_ID = "{model_id}"
PCC_TARGET = {pcc_target}
_CAPTURED_ROOT = Path(__file__).resolve().parent.parent / "_captured"


# ---------------------------------------------------------------------------
# Boilerplate helpers (same convention as the skeleton emitter and the
# SAM2 hand-written e2e).
# ---------------------------------------------------------------------------
def _is_mesh_device(device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


def _to_tt(x, device):
    if isinstance(x, ttnn.Tensor):
        return x
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        t = x.to(torch.bfloat16)
        if _is_mesh_device(device):
            try:
                return ttnn.from_torch(
                    t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
            except (AttributeError, TypeError):
                pass
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return x


def _to_torch(t, device):
    if not isinstance(t, ttnn.Tensor):
        return t
    try:
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(device)
    except Exception:
        pass
    try:
        from models.common.auto_compose import to_torch_auto_compose
        return to_torch_auto_compose(t)
    except Exception:
        return ttnn.to_torch(t)


# ---------------------------------------------------------------------------
# Wired test — one block per captured component.
# ---------------------------------------------------------------------------
def test_e2e_pipeline_wired(device_params, device):
    """Validate every TT-ported component against its captured HF I/O.

    This is the auto-generated 'wired' variant — each component is
    exercised with the EXACT input the HF forward produced at that point
    in the real graph, and the TT output is compared to the captured
    reference output. A true serial chain (feeding TT_A's output into
    TT_B's input) is a TODO — for now the captured input is used at
    every step, which already validates that every TT impl produces
    the right output when given the right input."""
{pipeline_body}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__ + "::test_e2e_pipeline_wired", "-svv"]))
'''


def render_e2e_wired(
    *,
    model_id: str,
    components: List,
    captured_dir: Path,
    repo_root: Path,
    test_path: str = "models/demos/<model>/tests/test_e2e_wired.py",
    pcc_target: float = 0.95,
) -> Optional[str]:
    """Render a *wired* e2e test as a string.

    Returns None when no components have captured I/O (caller should
    fall back to the skeleton emitter and warn the user to run
    capture-inputs first)."""
    infos = [_component_to_e2e_info(c) for c in components]
    components_with_io = _components_with_captured_io(captured_dir, infos)
    if not components_with_io:
        return None
    return _WIRED_E2E_TEMPLATE.format(
        model_id=model_id,
        test_path=test_path,
        pcc_target=pcc_target,
        component_summary=_format_component_summary(components_with_io),
        tt_imports=_format_wired_tt_imports(components_with_io, repo_root),
        pipeline_body=_format_wired_pipeline_body(components_with_io, repo_root),
    )


def emit_e2e_pipeline_test_wired(
    *,
    model_id: str,
    components: List,
    captured_dir: Path,
    repo_root: Path,
    output_path: Path,
    pcc_target: float = 0.95,
    overwrite: bool = False,
) -> Optional[Path]:
    """Write a *wired* e2e test to disk. Returns None if no captured
    I/O is present, OR if the file already exists and overwrite=False."""
    rendered = render_e2e_wired(
        model_id=model_id,
        components=components,
        captured_dir=captured_dir,
        repo_root=repo_root,
        test_path=str(output_path),
        pcc_target=pcc_target,
    )
    if rendered is None:
        return None
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


_HARNESS_E2E_TEMPLATE = '''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline test for {model_id} on TT hardware.

AUTO-GENERATED by ``scripts.tt_hw_planner.e2e_emitter.render_e2e_harness``.

This is the HARNESS variant: instead of trying to wire each component
call by hand, this test invokes ``e2e_harness.run_e2e_pipeline``, which
   * loads the HF model,
   * for each on-device component: parses its per-component PCC test to
     extract the existing TT-construction recipe, evaluates it in a
     sandbox, and registers a forward hook on the matching HF submodule
     that swaps the HF output with the TT-computed output,
   * runs HF.forward() once without hooks (reference) and once with
     hooks installed (TT-substituted), and
   * comp_pcc-compares the final outputs.

The advantage over manually-wired e2e tests: HF's own forward graph
provides the dataflow + any non-TT glue between components. We don't
have to invent it.

Component coverage:
{component_summary}

Usage:
    pytest {test_path} -svv --timeout=600
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.tt_hw_planner.bringup_plan import build_bringup_plan
from scripts.tt_hw_planner.e2e_harness import run_e2e_pipeline
from scripts.tt_hw_planner.family_backends import pick_backend
from scripts.tt_hw_planner.probe import probe_model

HF_MODEL_ID = "{model_id}"
PCC_TARGET = {pcc_target}
DEMO_DIR = Path(__file__).resolve().parent.parent


def _build_sample_input():
    """Synthesize a plausible top-level input for {model_id}.

    Vision models: pixel_values of (1, 3, H, W) inferred from config.
    Text models:   token ids from a tokenizer probe.
    Override TT_PLANNER_E2E_PROBE_TEXT for a custom text probe.
    """
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    vc = getattr(cfg, "vision_config", None) or cfg
    image_size = getattr(vc, "image_size", None) or getattr(cfg, "image_size", None)
    if image_size:
        return torch.randn(1, 3, int(image_size), int(image_size)), {{}}
    try:
        tok = transformers.AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        import os
        probe = os.environ.get("TT_PLANNER_E2E_PROBE_TEXT",
                               "The quick brown fox jumps over the lazy dog.")
        enc = tok(probe, return_tensors="pt")
        return enc["input_ids"], {{"attention_mask": enc.get("attention_mask")}}
    except Exception:
        return None, {{}}


def test_e2e_pipeline_harness(device_params, device):
    """Full TT-substitution e2e test via the harness."""
    import transformers

    sample, sample_kwargs = _build_sample_input()
    if sample is None:
        pytest.skip("Could not synthesize a sample input — model is neither vision nor text.")

    # Rebuild the bring-up plan so the harness knows which components to substitute.
    probe = probe_model(HF_MODEL_ID)
    cfg = probe.raw_config if probe is not None else {{}}
    backend = pick_backend(
        category=getattr(probe, "category", "Other"),
        model_type=(cfg or {{}}).get("model_type"),
        pipeline_tag=getattr(probe, "pipeline_tag", None),
    )
    if backend is None:
        pytest.skip(f"No backend resolved for {{HF_MODEL_ID}} — run auto-onboard first.")
    plan = build_bringup_plan(
        new_model_id=HF_MODEL_ID,
        new_cfg=cfg,
        backend=backend,
        repo_root=Path.cwd(),
    )

    hf_model = transformers.AutoModel.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    hf_model.eval()

    result = run_e2e_pipeline(
        hf_model=hf_model,
        plan=plan,
        demo_dir=DEMO_DIR,
        device=device,
        sample_input=sample,
        pcc_target=PCC_TARGET,
        sample_kwargs={{k: v for k, v in sample_kwargs.items() if v is not None}},
    )

    print(f"  per-component: {{result.per_component_status}}")
    for note in result.notes:
        print(f"  {{note}}")
    assert result.passed, (
        f"end-to-end PCC={{result.pcc:.4f}} < target {{PCC_TARGET}}. "
        f"per-component status: {{result.per_component_status}}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__ + "::test_e2e_pipeline_harness", "-svv"]))
'''


def render_e2e_harness(
    *,
    model_id: str,
    components: List,
    test_path: str = "models/demos/<model>/tests/test_e2e_harness.py",
    pcc_target: float = 0.95,
) -> str:
    """Render the HARNESS-variant e2e test source.

    Unlike the skeleton and wired variants, the harness emits a thin
    pytest file (~80 lines) that delegates the actual TT-substitution
    work to ``e2e_harness.run_e2e_pipeline``. The harness reads each
    per-component PCC test's TT-construction recipe at runtime and
    builds the substitution hooks dynamically — no per-model
    construction synthesis needed in the generated file."""
    infos = [_component_to_e2e_info(c) for c in components]
    on_device = [c for c in infos if c.on_device]
    return _HARNESS_E2E_TEMPLATE.format(
        model_id=model_id,
        test_path=test_path,
        pcc_target=pcc_target,
        component_summary=_format_component_summary(on_device or infos),
    )


def emit_e2e_pipeline_test_harness(
    *,
    model_id: str,
    components: List,
    output_path: Path,
    pcc_target: float = 0.95,
    overwrite: bool = False,
) -> Optional[Path]:
    """Write a HARNESS-variant e2e test to disk.

    Returns None if the file exists and overwrite=False."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_e2e_harness(
        model_id=model_id,
        components=components,
        test_path=str(output_path),
        pcc_target=pcc_target,
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


__all__ = [
    "E2EComponentInfo",
    "emit_e2e_pipeline_test",
    "emit_e2e_pipeline_test_harness",
    "emit_e2e_pipeline_test_wired",
    "render_e2e_harness",
    "render_e2e_test",
    "render_e2e_wired",
]
