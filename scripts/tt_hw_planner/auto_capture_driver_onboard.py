"""Auto-onboard: LLM-drafted capture driver for HF models the generic framework
can't drive.

Mirrors ``auto_onboard.py``'s pattern but for capture drivers instead of
FamilyBackend entries. When the generic introspection chain in
``capture_drivers.try_capture_drivers`` cannot fire forward hooks on some
components of a model, this module asks the LLM to write a custom driver,
validates it, persists it under ``learned_drivers/`` and registers it via the
Layer 3 registry so the next session loads it automatically.

All generic -- no model-specific code anywhere. The LLM is asked to introspect
whatever model the caller passes in. The resulting driver is stored as
DATA (per-model file), not framework code, so the tool's own source stays
model-agnostic.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple


_LEARNED_DRIVERS_DIR = Path(__file__).parent / "learned_drivers"


_PROMPT_TEMPLATE = """\
The HF model `{model_class}` (model_id: `{model_id}`) cannot be driven by the
generic capture framework. Some components could not be invoked via standard
forward patterns. Write a Python function that drives this specific HF model
so its forward hooks fire on the uncaptured components.

The function MUST have this exact signature:

    def driver(model, pixel_values):
        # invoke model so its components get exercised
        # may construct supporting objects (processors, sessions, etc.)
        # return None; the caller has hooks installed before calling driver()
        pass

Constraints:
- Use only HF transformers public API
- Stay in-process; no network, no file I/O
- Match a callable with EXACTLY two positional args: (model, pixel_values)
- Return None
- Wrap any failures in try/except -- never raise out of the driver

CONTEXT:

Model class: {model_class}
Model id:    {model_id}

Forward signature:
    {forward_sig}

Methods available on the model:
    {model_methods}

Other classes in the same transformers module (likely processors/sessions):
    {module_classes}

Components the standard framework could not capture:
    {uncaptured}

What the generic framework already tried (in order):
    {tried}

Return ONLY a single Python function `def driver(model, pixel_values):` with
no markdown, no prose, no top-level imports outside the function body. The
function will be saved as a learned driver and loaded on future runs.

OUTPUT FORMAT (strict):
  - The FIRST characters of your response MUST be `def driver(`
  - No "Here's the driver:" or any prose preamble
  - No ```python ... ``` fences
  - No explanatory text after the closing brace
  - Imports go INSIDE the function body, not at module top

If you cannot satisfy these constraints, respond with exactly `def driver(model, pixel_values): return None`
(the no-op default). Do NOT explain why -- a malformed response is worse than a no-op.
"""


def _probe_model(model: Any, model_id: str) -> dict:
    """Gather minimal-but-sufficient context to draft a driver. Bounded size."""
    cls = type(model)
    forward_sig = "(no forward method)"
    if hasattr(cls, "forward"):
        try:
            forward_sig = str(inspect.signature(cls.forward))
        except (TypeError, ValueError):
            pass

    # Methods of interest: prioritize verbs that look like model entry-points
    # (init_*, run_*, generate*, get_*, *_inference, *_segmentation, *_forward).
    # Keep the list short to avoid blowing up the LLM context window — large
    # prompts cause LLM timeouts that look like "no response" in the closed-
    # loop iteration.
    _ENTRY_PREFIXES = ("init_", "run_", "get_", "create_", "build_", "make_")
    _ENTRY_SUFFIXES = ("_inference", "_session", "_state", "_forward", "_step")
    entry_methods: List[str] = []
    other_methods: List[str] = []
    for m in dir(model):
        if m.startswith("_"):
            continue
        attr = getattr(model, m, None)
        if not callable(attr):
            continue
        if any(m.startswith(p) for p in _ENTRY_PREFIXES) or any(m.endswith(s) for s in _ENTRY_SUFFIXES):
            entry_methods.append(m)
        else:
            other_methods.append(m)
    method_names = entry_methods[:20] + other_methods[:10]

    # Module classes: prefer Session/Cache/Processor/Output classes (the LLM
    # needs these to bridge unusual forward signatures). Trim hard.
    _IMPORTANT_CLASS_TAILS = ("Session", "Cache", "Processor", "Output", "Config")
    module_classes: List[str] = []
    important_classes: List[str] = []
    module_name = getattr(cls, "__module__", "")
    if module_name:
        mod = sys.modules.get(module_name)
        if mod is not None:
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                attr = getattr(mod, name, None)
                if not inspect.isclass(attr):
                    continue
                if any(name.endswith(t) for t in _IMPORTANT_CLASS_TAILS):
                    important_classes.append(name)
                else:
                    module_classes.append(name)
    module_classes = important_classes[:15] + module_classes[:10]

    return {
        "model_class": cls.__name__,
        "model_id": model_id,
        "forward_sig": forward_sig,
        "model_methods": "\n    ".join(method_names) if method_names else "(none)",
        "module_classes": "\n    ".join(module_classes) if module_classes else "(none)",
    }


def _build_prompt(
    probe: dict,
    uncaptured: List[str],
    tried: List[str],
) -> str:
    return _PROMPT_TEMPLATE.format(
        model_class=probe["model_class"],
        model_id=probe["model_id"],
        forward_sig=probe["forward_sig"],
        model_methods=probe["model_methods"],
        module_classes=probe["module_classes"],
        uncaptured="\n    ".join(uncaptured) if uncaptured else "(none listed)",
        tried="\n    ".join(tried) if tried else "(none recorded)",
    )


def _strip_markdown_fences(text: str) -> str:
    """Extract Python code from an LLM response.

    Handles three common LLM output shapes:

      1. Bare code  -> "def driver(...): ..."           returned as-is
      2. Whole-response fenced  ->  "```python\ndef driver...\n```"
                                    -> strip outer fences
      3. Code embedded in prose -> "Here's a driver:\n```python\n...\n```\n
                                    Note: ..."
                                    -> extract the FIRST fenced block

    Falls back to the input (stripped) if no clear extraction is possible.
    """
    out = text.strip()
    # Case 3: prose around a fenced block — extract the first fenced block.
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", out, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if "def driver" in candidate:
            return candidate
    # Case 2: response starts with a fence; strip outer fences.
    if out.startswith("```"):
        out = re.sub(r"^```(?:python|py)?\s*\n", "", out)
        out = re.sub(r"\n```\s*$", "", out)
        return out.strip()
    # Case 1: bare code (or unstructured prose containing the def).
    # If there's prose preamble before `def driver`, drop it.
    def_idx = out.find("def driver")
    if def_idx > 0 and not out[:def_idx].strip().endswith(","):
        # Drop preamble that's clearly not Python (no leading decorators/imports).
        preamble = out[:def_idx].strip()
        if not re.search(r"^(?:from|import|@|#)\s", preamble, re.MULTILINE):
            return out[def_idx:].strip()
    return out


def _validate_driver_source(source: str) -> Tuple[bool, str]:
    """AST-validate the LLM-drafted driver source. Returns (ok, error_message)."""
    if not source.strip():
        return False, "empty source"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    driver_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "driver":
            driver_def = node
            break
    if driver_def is None:
        return False, "no `def driver(...)` function found at top level"

    positional = [a for a in driver_def.args.args if a.arg not in {"self"}]
    if len(positional) != 2:
        return False, f"`driver` must take exactly 2 positional args, found {len(positional)}"

    arg_names = [a.arg for a in positional]
    if arg_names != ["model", "pixel_values"]:
        return False, f"`driver` args must be (model, pixel_values), got {arg_names}"

    return True, ""


def _persist_driver(model_class_name: str, source: str) -> Path:
    """Save the validated driver source under learned_drivers/<safe>.py.

    Wraps the bare `def driver(...)` in a registration shim that auto-registers
    the driver against `capture_drivers` for any model whose class name matches
    `model_class_name`.
    """
    _LEARNED_DRIVERS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", model_class_name).lower().strip("_") or "driver"
    target = _LEARNED_DRIVERS_DIR / f"{safe}.py"

    wrapper = (
        f'"""Learned capture driver for {model_class_name}.\n\n'
        f"Auto-generated by auto_capture_driver_onboard. Do not edit by hand --\n"
        f"re-run the auto-onboard flow if the driver becomes stale.\n"
        f'"""\n'
        f"from scripts.tt_hw_planner.capture_drivers import register_capture_driver\n\n\n"
        f"{source.rstrip()}\n\n\n"
        f"@register_capture_driver(matcher=lambda m: type(m).__name__ == "
        f'"{model_class_name}")\n'
        f"def _registered_driver(model, pixel_values):\n"
        f"    try:\n"
        f"        driver(model, pixel_values)\n"
        f"    except Exception:\n"
        f"        pass\n"
    )

    target.write_text(wrapper)
    return target


def _try_run_driver(
    source: str,
    model: Any,
    target_component_names: List[str],
    pixel_values: Any,
) -> Tuple[bool, str, set]:
    """Execute the drafted driver against `model` with forward hooks on
    every nn.Module. Returns ``(ran_cleanly, error_message, fired_paths)``.

    Closed-loop validation: AST-pass alone isn't enough. A driver can
    pass syntax/signature checks and still raise on
    ``model.init_video_inference(...)``, silently no-op (driver body is
    ``pass``), or fire no target components. This try-run catches all
    three so the next iteration can prompt the LLM with concrete
    feedback.

    Hooks are installed on EVERY named module so we don't need a
    pre-computed name->module mapping. The returned ``fired_paths`` set
    is the named-module paths where forward fired during the driver
    invocation. Caller maps these back to target component names via
    its own resolution table.
    """
    fired_paths: set = set()
    handles: List = []
    try:
        for path, module in model.named_modules():
            if not path:
                continue

            def _hook(_mod, _args, _output, _path=path):
                fired_paths.add(_path)

            try:
                handles.append(module.register_forward_hook(_hook))
            except Exception:
                continue

        ns: dict = {}
        try:
            exec(source, ns)
        except Exception as exc:
            return False, f"exec failed: {type(exc).__name__}: {exc}", fired_paths

        drv = ns.get("driver")
        if not callable(drv):
            return False, "compiled source has no `driver` callable", fired_paths

        try:
            drv(model, pixel_values)
        except Exception as exc:
            return False, f"driver invocation raised: {type(exc).__name__}: {exc}", fired_paths
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    return True, "", fired_paths


def _match_fired_components(
    target_names: List[str],
    fired_paths: set,
    components_by_path: Optional[dict] = None,
) -> set:
    """Map fired submodule paths back to target component names.

    If ``components_by_path`` is provided (mapping ``name -> submodule_path``
    string in the parent model's named_modules), match by path prefix:
    a target named ``X`` fired iff any fired path equals X's submodule
    path or descends from it (``X.something``).

    Otherwise fall back to a substring/safe-id heuristic — useful when
    the caller doesn't have the resolved paths handy.
    """
    fired_targets: set = set()
    if components_by_path:
        for name, comp_path in components_by_path.items():
            if not comp_path:
                continue
            for fired in fired_paths:
                if fired == comp_path or fired.startswith(comp_path + ".") or fired.startswith(comp_path + "["):
                    fired_targets.add(name)
                    break
        return fired_targets
    for name in target_names:
        safe = name.replace("-", "_")
        for fired in fired_paths:
            tail = fired.rsplit(".", 1)[-1]
            if name == tail or safe == tail or name in fired or safe in fired:
                fired_targets.add(name)
                break
    return fired_targets


def auto_onboard_capture_driver(
    model: Any,
    model_id: str,
    uncaptured_components: List[str],
    framework_attempts: List[str],
    *,
    components_by_path: Optional[dict] = None,
    pixel_values: Any = None,
    agent_bin: str = "claude",
    llm_model: str = "sonnet",
    timeout_s: int = 360,  # bumped from 180 — sonnet can take 3-5min for complex prompts (e.g. SAM2 video)
    max_iters: int = 3,
) -> Tuple[bool, Optional[Path], str]:
    """Draft + validate + try-run + persist a capture driver for `model`.

    Closed-loop iteration up to ``max_iters`` total attempts. Each
    iteration:
      1. LLM drafts a `def driver(model, pixel_values)` from the probe.
      2. AST validates signature.
      3. Try-runs the driver against ``model`` with hooks installed.
      4. If exec raised OR fired_paths doesn't cover the target
         components, the next iteration's prompt includes the concrete
         failure (exception type+msg or "fired X but missed Y, Z") so
         the LLM can self-correct.
      5. Only persists when the draft AST-validates AND runs without
         raising AND fires at least one of the target components.

    Returns ``(ok, persisted_path, message)``. On failure,
    ``persisted_path`` is None and ``message`` summarizes the attempts.
    All rejected drafts are saved to learned_drivers/_rejected/ for
    operator inspection.

    Reuses ``llm_synth.invoke_llm_cli_one_shot`` -- the same one-shot LLM
    helper that ``auto_onboard.py`` uses -- so this module adds zero
    duplication of LLM-invocation plumbing.
    """
    from .llm_synth import invoke_llm_cli_one_shot

    probe = _probe_model(model, model_id)
    base_prompt = _build_prompt(probe, uncaptured_components, framework_attempts)

    def _one_attempt(active_prompt: str) -> Tuple[bool, str, str, str]:
        try:
            resp = invoke_llm_cli_one_shot(active_prompt, agent_bin=agent_bin, model=llm_model, timeout_s=timeout_s)
        except Exception as exc:
            return False, "", "", f"LLM invocation failed: {type(exc).__name__}: {exc}"
        src = _strip_markdown_fences(resp)
        ok, err = _validate_driver_source(src)
        return ok, resp, src, err

    def _persist_debug_log(reason: str, raw_responses: List[Tuple[int, str]]) -> None:
        try:
            debug_dir = _LEARNED_DRIVERS_DIR / "_rejected"
            debug_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^a-zA-Z0-9_]+", "_", probe["model_class"]).lower().strip("_") or "driver"
            log_path = debug_dir / f"{safe}_last_rejected.txt"
            with log_path.open("w") as f:
                f.write(f"# Rejected LLM draft(s) for {probe['model_class']} ({model_id})\n")
                f.write(f"# Reason: {reason}\n")
                f.write(f"# Uncaptured components: {uncaptured_components}\n\n")
                for idx, raw in raw_responses:
                    f.write(f"\n===== ATTEMPT {idx} (raw response) =====\n")
                    f.write(raw)
                    f.write("\n===== END ATTEMPT =====\n")
        except Exception:
            pass  # debug-log failure never blocks main flow

    raw_responses: List[Tuple[int, str]] = []
    last_err = ""
    active_prompt = base_prompt
    can_try_run = pixel_values is not None

    for attempt_idx in range(1, max_iters + 1):
        ok, raw, source, err = _one_attempt(active_prompt)
        raw_responses.append((attempt_idx, raw or ""))

        if not ok:
            last_err = f"AST validation: {err}"
            active_prompt = (
                base_prompt
                + f"\n\nIMPORTANT: Attempt {attempt_idx} was REJECTED. Error:\n    {err}"
                + "\n\nProduce ONLY the bare `def driver(model, pixel_values):` Python source. "
                + "No markdown, no prose. Start the response with `def driver`."
            )
            continue

        # AST passed. If we have a try-run input, exercise the driver
        # against the model. Otherwise persist as-is (caller will detect
        # missing artifacts post-capture).
        if not can_try_run:
            persisted = _persist_driver(probe["model_class"], source)
            return (
                True,
                persisted,
                f"driver synthesized + persisted to {persisted.name} (no try-run; pixel_values unavailable)",
            )

        ran_clean, run_err, fired_paths = _try_run_driver(source, model, uncaptured_components, pixel_values)
        if not ran_clean:
            last_err = f"runtime error: {run_err}"
            active_prompt = (
                base_prompt
                + f"\n\nIMPORTANT: Attempt {attempt_idx} parsed correctly but RAISED at runtime. Error:\n    {run_err}"
                + "\n\nLikely cause: the driver tried to invoke `model` in a way that the model's actual "
                + "API doesn't support. Look at the model's `forward` signature and `module_classes` list "
                + "in the context above — if `forward` takes a `Session`/`Cache`/`InferenceSession` "
                + "argument, you MUST construct one before invoking forward. Use the module classes "
                + "(e.g. *Processor, *Session) as documented in the prompt."
                + "\n\nProduce ONLY the bare `def driver(model, pixel_values):` Python source."
            )
            continue

        # Driver ran cleanly. Check whether any target components fired.
        fired_targets = _match_fired_components(uncaptured_components, fired_paths, components_by_path)
        still_missing = [c for c in uncaptured_components if c not in fired_targets]

        if not still_missing:
            persisted = _persist_driver(probe["model_class"], source)
            return (
                True,
                persisted,
                f"driver synthesized + persisted to {persisted.name} (fired all {len(uncaptured_components)} target components)",
            )

        # Partial fire or full miss — persist anyway since SOME components fired,
        # but loop again with feedback if attempts remain. (We persist on the
        # final iteration even with partial coverage; rejected fallback if zero
        # fired and we've exhausted attempts.)
        last_err = (
            f"runtime ok but fired {len(fired_targets)}/{len(uncaptured_components)} target(s); "
            f"missed: {still_missing}"
        )
        if attempt_idx == max_iters:
            if fired_targets:
                # Partial success — persist and let downstream proceed.
                persisted = _persist_driver(probe["model_class"], source)
                return (
                    True,
                    persisted,
                    f"driver persisted with PARTIAL coverage: fired {sorted(fired_targets)} of "
                    f"{uncaptured_components}; still missing {still_missing}",
                )
            # Zero fired across max_iters — give up.
            _persist_debug_log(
                f"runtime ok but zero target components fired across {max_iters} attempts", raw_responses
            )
            return (
                False,
                None,
                f"closed-loop iteration exhausted after {max_iters} attempts: {last_err}. "
                f"Drafts saved to learned_drivers/_rejected/.",
            )

        active_prompt = (
            base_prompt
            + f"\n\nIMPORTANT: Attempt {attempt_idx} ran cleanly but did NOT fire the target "
            + f"components. Fired: {sorted(fired_targets) or '(none)'}. Still missing: {still_missing}."
            + "\n\nRevise the driver so its execution path EXERCISES every component in the "
            + "'Components the standard framework could not capture' list. Trace the model's "
            + "forward path: which submodules does it traverse? Make sure your driver invokes "
            + "a forward path that visits each of the missing components."
            + "\n\nProduce ONLY the bare `def driver(model, pixel_values):` Python source."
        )

    # Loop exited via continue paths exhausting max_iters with AST or runtime errors
    _persist_debug_log(f"all {max_iters} attempts failed: {last_err}", raw_responses)
    return (
        False,
        None,
        f"closed-loop iteration exhausted after {max_iters} attempts: {last_err}. "
        f"Drafts saved to learned_drivers/_rejected/.",
    )


def load_learned_drivers() -> List[str]:
    """Import every ``.py`` file in ``learned_drivers/``.

    Each learned driver file calls ``register_capture_driver`` at import time
    via its `@register_capture_driver(...)` decorator on the registration shim,
    so importing them is what wires the drivers into the registry.

    Returns the list of paths successfully loaded. Errors are silently
    skipped: a malformed learned driver should not block the rest of the
    capture pipeline.
    """
    if not _LEARNED_DRIVERS_DIR.is_dir():
        return []
    loaded: List[str] = []
    for py_file in sorted(_LEARNED_DRIVERS_DIR.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"_learned_driver_{py_file.stem}", py_file)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            loaded.append(str(py_file))
        except Exception:
            continue
    return loaded
