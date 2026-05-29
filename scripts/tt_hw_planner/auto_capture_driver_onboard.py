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

    method_names: List[str] = []
    for m in dir(model):
        if m.startswith("_"):
            continue
        attr = getattr(model, m, None)
        if callable(attr):
            method_names.append(m)
    method_names = method_names[:40]

    module_classes: List[str] = []
    module_name = getattr(cls, "__module__", "")
    if module_name:
        mod = sys.modules.get(module_name)
        if mod is not None:
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                attr = getattr(mod, name, None)
                if inspect.isclass(attr):
                    module_classes.append(name)
    module_classes = module_classes[:30]

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


def auto_onboard_capture_driver(
    model: Any,
    model_id: str,
    uncaptured_components: List[str],
    framework_attempts: List[str],
    *,
    agent_bin: str = "claude",
    llm_model: str = "sonnet",
    timeout_s: int = 180,
) -> Tuple[bool, Optional[Path], str]:
    """Draft + validate + persist a capture driver for `model`.

    Returns ``(ok, persisted_path, message)``. On failure, ``persisted_path`` is
    None and ``message`` describes which validation step rejected the LLM's
    output. Idempotent: re-running with the same model overwrites the persisted
    driver with the new draft.

    Reuses ``llm_synth.invoke_llm_cli_one_shot`` -- the same one-shot LLM helper
    that ``auto_onboard.py`` uses to draft FamilyBackend entries -- so this
    module adds zero duplication of the LLM-invocation plumbing.
    """
    from .llm_synth import invoke_llm_cli_one_shot

    probe = _probe_model(model, model_id)
    prompt = _build_prompt(probe, uncaptured_components, framework_attempts)

    attempts: List[Tuple[str, str]] = []  # list of (raw_response_first_200_chars, error_message)

    def _one_attempt(active_prompt: str) -> Tuple[bool, str, str, str]:
        try:
            resp = invoke_llm_cli_one_shot(active_prompt, agent_bin=agent_bin, model=llm_model, timeout_s=timeout_s)
        except Exception as exc:
            return False, "", "", f"LLM invocation failed: {type(exc).__name__}: {exc}"
        src = _strip_markdown_fences(resp)
        ok, err = _validate_driver_source(src)
        return ok, resp, src, err

    def _persist_debug_log(reason: str, raw_responses: List[Tuple[int, str]]) -> None:
        """Save the rejected LLM responses to a per-model debug log so
        the operator can SEE what the LLM produced. Without this, the
        only artifact of a failed onboarding is a one-line error msg
        and the raw text is lost."""
        try:
            debug_dir = _LEARNED_DRIVERS_DIR / "_rejected"
            debug_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^a-zA-Z0-9_]+", "_", probe["model_class"]).lower().strip("_") or "driver"
            log_path = debug_dir / f"{safe}_last_rejected.txt"
            with log_path.open("w") as f:
                f.write(f"# Rejected LLM draft(s) for {probe['model_class']} ({model_id})\n")
                f.write(f"# Reason: {reason}\n")
                f.write(f"# Uncaptured components: {uncaptured_components}\n")
                f.write("\n")
                for attempt_idx, raw in raw_responses:
                    f.write(f"\n===== ATTEMPT {attempt_idx} (raw response) =====\n")
                    f.write(raw)
                    f.write("\n===== END ATTEMPT =====\n")
        except Exception:
            pass  # debug log failure must never block the main flow

    ok, raw, source, err = _one_attempt(prompt)
    raw_responses: List[Tuple[int, str]] = [(1, raw)]
    if not ok:
        attempts.append((raw[:200].replace("\n", " ") if raw else "(empty)", err))
        # One retry with explicit error feedback so the LLM can self-correct.
        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: A previous draft was REJECTED. Error was:\n    "
            + err
            + "\n\nProduce ONLY the bare `def driver(model, pixel_values):` Python source. "
            + "No markdown fences, no prose, no preamble. Start the response with `def driver`."
        )
        ok, raw, source, err = _one_attempt(retry_prompt)
        raw_responses.append((2, raw))
        if not ok:
            attempts.append((raw[:200].replace("\n", " ") if raw else "(empty)", err))
            _persist_debug_log(f"validation failed after retry: {err}", raw_responses)
            return (
                False,
                None,
                f"validation failed after retry. attempt 1: {attempts[0][1]} (resp={attempts[0][0]!r}); "
                f"attempt 2: {attempts[1][1]} (resp={attempts[1][0]!r}). "
                f"Raw drafts saved to learned_drivers/_rejected/.",
            )

    persisted = _persist_driver(probe["model_class"], source)
    return True, persisted, f"driver synthesized + persisted to {persisted.name}"


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
