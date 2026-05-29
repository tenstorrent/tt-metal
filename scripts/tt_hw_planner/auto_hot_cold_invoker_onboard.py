"""Auto-onboard: LLM-drafted invoker for HOT/COLD profiling of models
whose forward signature isn't pixel_values-compatible.

Mirrors ``auto_capture_driver_onboard.py``'s pattern but for HOT/COLD
profilers instead of capture drivers. When ``hot_cold_profiler._invoke_model_for_profile``
exhausts its standard chain (pixel_values → try_capture_drivers → bare
sample_input) without firing any hooks, this module asks the LLM to
draft a model-specific invoker that DOES drive the workload forward
path so hooks fire.

The drafted invoker is persisted under ``learned_invokers/<safe>.py``
and auto-loaded on future runs via the registry pattern. Result: the
HOT/COLD profiler eventually handles any model — image, video session,
text, audio — as long as the LLM can write a one-shot invocation for it.

Generic and model-agnostic. The framework itself never references SAM2
or any specific model. Per-model invokers live as DATA (learned files),
not framework code.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple


_LEARNED_INVOKERS_DIR = Path(__file__).parent / "learned_invokers"


_INVOKER_REGISTRY: List[Tuple[Callable[[Any], bool], Callable]] = []


def register_hot_cold_invoker(matcher: Callable[[Any], bool]):
    """Register a custom HOT/COLD invoker for models matching ``matcher(model)``.

    The decorated callable receives ``(model, sample_input)`` and returns
    nothing (raises on failure). Forward hooks are already installed by
    the profiler before the invoker runs, so any forward path the
    invoker exercises will fire them.
    """

    def deco(fn: Callable) -> Callable:
        _INVOKER_REGISTRY.append((matcher, fn))
        return fn

    return deco


def resolve_custom_invoker(model: Any) -> Optional[Callable]:
    for matcher, fn in _INVOKER_REGISTRY:
        try:
            if matcher(model):
                return fn
        except Exception:
            continue
    return None


_PROMPT_TEMPLATE = """\
The HF model `{model_class}` (model_id: `{model_id}`) cannot be invoked
via the standard ``model(pixel_values=...)`` or generic session-pattern
chain. Write a Python function that drives this model's forward path
so its components' forward hooks fire.

The function MUST have this exact signature:

    def invoke_for_profile(model, sample_input):
        # invoke model so that its components' forward methods get called
        # may construct supporting objects (sessions, configs, processors)
        # return None; the caller has hooks installed before calling this
        pass

Constraints:
- Use only HF transformers public API
- Stay in-process; no network, no file I/O
- Match the signature EXACTLY: (model, sample_input)
- Return None
- Wrap any failures in try/except — never raise out of the invoker
- Goal: drive the IMAGE-PATH forward (the typical demo workload)
  for vision models. For text/audio models, drive the primary
  inference path that the demo would use.

CONTEXT:

Model class: {model_class}
Model id:    {model_id}

Forward signature:
    {forward_sig}

Methods available on the model:
    {model_methods}

Other classes in the same transformers module (likely sessions/processors):
    {module_classes}

Return ONLY a single Python function `def invoke_for_profile(model, sample_input):`
with no markdown, no prose, no top-level imports outside the function body.
The function will be saved as a learned invoker and loaded on future runs.
"""


def _probe_model(model: Any, model_id: str) -> dict:
    """Same shape as ``auto_capture_driver_onboard._probe_model``. Kept
    independent so the two onboarders can evolve their context
    requirements separately."""
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


def _build_prompt(probe: dict) -> str:
    return _PROMPT_TEMPLATE.format(**probe)


def _strip_markdown_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:python)?\s*\n", "", out)
        out = re.sub(r"\n```\s*$", "", out)
    return out.strip()


def _validate_invoker_source(source: str) -> Tuple[bool, str]:
    """AST-validate the LLM-drafted source. Returns (ok, error_message)."""
    if not source.strip():
        return False, "empty source"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    invoker_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "invoke_for_profile":
            invoker_def = node
            break
    if invoker_def is None:
        return False, "no `def invoke_for_profile(...)` function found at top level"

    positional = [a for a in invoker_def.args.args if a.arg not in {"self"}]
    if len(positional) != 2:
        return False, f"`invoke_for_profile` must take exactly 2 positional args, found {len(positional)}"

    arg_names = [a.arg for a in positional]
    if arg_names != ["model", "sample_input"]:
        return False, f"`invoke_for_profile` args must be (model, sample_input), got {arg_names}"

    return True, ""


def _persist_invoker(model_class_name: str, source: str) -> Path:
    """Save the validated invoker under learned_invokers/<safe>.py with
    a registration shim that auto-registers it for any model whose class
    name matches ``model_class_name``."""
    _LEARNED_INVOKERS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", model_class_name).lower().strip("_") or "invoker"
    target = _LEARNED_INVOKERS_DIR / f"{safe}.py"

    wrapper = (
        f'"""Learned HOT/COLD invoker for {model_class_name}.\n\n'
        f"Auto-generated by auto_hot_cold_invoker_onboard. Do not edit by\n"
        f"hand -- re-run the auto-onboard flow if the invoker becomes stale.\n"
        f'"""\n'
        f"from scripts.tt_hw_planner.auto_hot_cold_invoker_onboard import register_hot_cold_invoker\n\n\n"
        f"{source.rstrip()}\n\n\n"
        f"@register_hot_cold_invoker(matcher=lambda m: type(m).__name__ == "
        f'"{model_class_name}")\n'
        f"def _registered_invoker(model, sample_input):\n"
        f"    try:\n"
        f"        invoke_for_profile(model, sample_input)\n"
        f"    except Exception:\n"
        f"        pass\n"
    )

    target.write_text(wrapper)
    return target


def auto_onboard_hot_cold_invoker(
    model: Any,
    model_id: str,
    *,
    agent_bin: str = "claude",
    llm_model: str = "sonnet",
    timeout_s: int = 180,
) -> Tuple[bool, Optional[Path], str]:
    """Draft + validate + persist a HOT/COLD invoker for ``model``.

    Returns ``(ok, persisted_path, message)``. On failure, ``persisted_path``
    is None and ``message`` describes which validation step rejected the
    LLM's output. Idempotent: re-running with the same model overwrites
    the persisted invoker with the new draft.

    Reuses ``llm_synth.invoke_llm_cli_one_shot`` -- the same one-shot LLM
    helper that ``auto_onboard.py`` and ``auto_capture_driver_onboard.py``
    use -- so this module adds zero duplication of the LLM-invocation
    plumbing.
    """
    from .llm_synth import invoke_llm_cli_one_shot

    probe = _probe_model(model, model_id)
    prompt = _build_prompt(probe)

    try:
        response = invoke_llm_cli_one_shot(prompt, agent_bin=agent_bin, model=llm_model, timeout_s=timeout_s)
    except Exception as exc:
        return False, None, f"LLM invocation failed: {type(exc).__name__}: {exc}"

    source = _strip_markdown_fences(response)
    ok, err = _validate_invoker_source(source)
    if not ok:
        return False, None, f"validation failed: {err}"

    persisted = _persist_invoker(probe["model_class"], source)
    return True, persisted, f"invoker synthesized + persisted to {persisted.name}"


def load_learned_invokers() -> List[str]:
    """Import every ``.py`` file in ``learned_invokers/``. Each file's
    ``@register_hot_cold_invoker`` decorator wires the invoker into the
    registry at import time. Errors are silently skipped: a malformed
    learned invoker should not block the rest of the HOT/COLD pipeline."""
    if not _LEARNED_INVOKERS_DIR.is_dir():
        return []
    loaded: List[str] = []
    for py_file in sorted(_LEARNED_INVOKERS_DIR.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"_learned_invoker_{py_file.stem}", py_file)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            loaded.append(str(py_file))
        except Exception:
            continue
    return loaded
