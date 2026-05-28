"""Generic capture-time drivers for HF models requiring non-standard invocation.

Some HF models can't be invoked by the standard ``model(pixel_values=...)`` path
because their forward requires additional arguments -- typically a session
or state object that has to be constructed via an init_*_session method.

This module provides three layers of generic, model-agnostic drivers that
``capture_inputs.py`` can try AFTER its built-in driver chain has failed:

  Layer 1 - try_introspected_forward
            Inspects ``model.forward()`` signature, synthesizes required
            arguments via type annotations + a small recursive synthesizer,
            and attempts the call.

  Layer 2 - SessionDriverPattern
            Discovers methods on ``model`` matching the name patterns
            ``(init|create|new)_.*(session|state)`` and ``(propagate|process
            |step|run)_.*``. Invokes them in order to fire forward hooks on
            session-style models.

  Layer 3 - register_capture_driver / resolve_custom_driver
            Plugin registry for caller-supplied drivers when introspection
            isn't enough. The registry is empty by default -- the framework
            ships zero per-model drivers. Add via the ``@register_capture_driver``
            decorator only when a specific model can't be driven generically.

All three layers are model-agnostic. No HF model class names are referenced
anywhere. The patterns work for any model that follows the usual HF naming
conventions for session-style invocation.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, List, Optional, Tuple


class _Omit:
    pass


_OMIT = _Omit()


_INIT_SESSION_PATTERN = re.compile(r"(?:init|create|new)_.*(?:session|state)", re.IGNORECASE)
_DRIVE_METHOD_PATTERN = re.compile(r"^(?:propagate|process|step|run|track)_", re.IGNORECASE)


_CUSTOM_DRIVER_REGISTRY: List[Tuple[Callable[[Any], bool], Callable]] = []


def register_capture_driver(matcher: Callable[[Any], bool]):
    """Register a custom capture driver for models matching `matcher(model)`.

    The decorated callable receives `(model, pixel_values)` and returns nothing
    (raises on failure). Hooks are already installed by the caller before the
    driver runs, so any forward path the driver exercises will fire them.
    """

    def deco(driver_fn: Callable) -> Callable:
        _CUSTOM_DRIVER_REGISTRY.append((matcher, driver_fn))
        return driver_fn

    return deco


def resolve_custom_driver(model: Any) -> Optional[Callable]:
    for matcher, driver in _CUSTOM_DRIVER_REGISTRY:
        try:
            if matcher(model):
                return driver
        except Exception:
            continue
    return None


def _is_optional_or_none(ann: Any) -> bool:
    if ann is type(None):
        return True
    args = getattr(ann, "__args__", ())
    return type(None) in args


def _is_tensor_annotation(ann: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if ann is torch.Tensor:
        return True
    args = getattr(ann, "__args__", ())
    return any(a is torch.Tensor for a in args)


def _synthesize_for_param(
    name: str,
    param: inspect.Parameter,
    model: Any,
    pixel_values: Any,
) -> Any:
    """Best-effort synthesis of a single required parameter.

    Strategy: name-based dispatch first (mirrors the existing _arg_for in
    capture_inputs.py for the common cases), then type-annotation fallback,
    then recursive session-object synthesis for session/state-named params.
    """
    name_lower = name.lower()

    if name == "pixel_values":
        return pixel_values

    if "session" in name_lower or "state" in name_lower:
        built = _try_build_session_object(model, pixel_values)
        if built is not _OMIT:
            return built

    ann = param.annotation
    if ann is not inspect.Parameter.empty:
        if _is_optional_or_none(ann):
            return None
        if _is_tensor_annotation(ann):
            try:
                import torch

                return torch.randn(1, 64, 768)
            except ImportError:
                return _OMIT
        if ann is bool:
            return False
        if ann is int:
            return 0
        if ann is str:
            return ""

    return _OMIT


def _try_build_session_object(model: Any, pixel_values: Any) -> Any:
    """Find an init/create/new_*(session|state) method and invoke it generically.

    Recursive arg synthesis: if the init method itself needs args, introspect
    its signature and synthesize them too.
    """
    for method_name in dir(model):
        if not _INIT_SESSION_PATTERN.search(method_name):
            continue
        method = getattr(model, method_name, None)
        if not callable(method):
            continue
        result = _invoke_with_synthesized_args(method, model, pixel_values)
        if result is not None:
            return result
    return _OMIT


def _invoke_with_synthesized_args(
    method: Callable,
    model: Any,
    pixel_values: Any,
) -> Any:
    """Invoke `method` by introspecting its signature and synthesizing required args.

    Returns None on any failure (caller decides what that means). Never raises.
    """
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return None
    kwargs: dict = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        synth = _synthesize_for_param(pname, param, model, pixel_values)
        if synth is _OMIT:
            return None
        kwargs[pname] = synth
    try:
        return method(**kwargs)
    except Exception:
        return None


class SessionDriverPattern:
    """Drive any model that follows 'init_*_session → propagate_*' pattern.

    Discovers methods by name pattern, invokes them via recursive arg synthesis.
    No model-specific code -- works for SAM2 video, future streaming HF models,
    and anything that follows the same naming convention.
    """

    def _find_init_method(self, model: Any) -> Optional[str]:
        for m in dir(model):
            if _INIT_SESSION_PATTERN.search(m) and callable(getattr(model, m, None)):
                return m
        return None

    def _find_drive_method(self, model: Any) -> Optional[str]:
        for m in dir(model):
            if _DRIVE_METHOD_PATTERN.match(m) and callable(getattr(model, m, None)):
                return m
        return None

    def can_drive(self, model: Any) -> bool:
        return self._find_init_method(model) is not None

    def drive(self, model: Any, pixel_values: Any) -> Tuple[bool, Optional[str]]:
        init_name = self._find_init_method(model)
        if not init_name:
            return False, "no init_*_session method found"
        init_method = getattr(model, init_name)
        session = _invoke_with_synthesized_args(init_method, model, pixel_values)
        if session is None:
            return False, f"init {init_name} returned None or raised"

        drive_name = self._find_drive_method(model)
        if not drive_name:
            return False, "no propagate/process/step method found"
        drive_method = getattr(model, drive_name)

        try:
            result = drive_method(session)
        except TypeError:
            try:
                result = _invoke_with_synthesized_args(drive_method, model, pixel_values)
                if result is None:
                    return False, f"drive {drive_name} returned None"
            except Exception as e:
                return False, f"drive {drive_name}: {type(e).__name__}: {e}"
        except Exception as e:
            return False, f"drive {drive_name}: {type(e).__name__}: {e}"

        try:
            if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
                for _ in result:
                    pass
        except Exception:
            pass

        return True, None


def try_introspected_forward(model: Any, pixel_values: Any) -> Tuple[bool, Optional[str]]:
    """Layer-1 driver. Pure introspection of model.forward()."""
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError) as exc:
        return False, f"signature inspection failed: {exc}"

    kwargs: dict = {}
    missing: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        synth = _synthesize_for_param(name, param, model, pixel_values)
        if synth is _OMIT:
            missing.append(name)
        else:
            kwargs[name] = synth

    if missing:
        return False, f"could not synthesize required args: {missing}"

    try:
        model(**kwargs)
        return True, None
    except Exception as exc:
        return False, f"forward raised: {type(exc).__name__}: {exc}"


def try_capture_drivers(model: Any, pixel_values: Any) -> Tuple[bool, List[str]]:
    """Try Layer 3 (custom) -> Layer 2 (pattern) -> Layer 1 (introspection).

    Returns (any_succeeded, attempts_with_results). The caller has already
    installed forward hooks; any successful invocation here fires them and
    populates the capture state.
    """
    attempts: List[str] = []

    custom = resolve_custom_driver(model)
    if custom is not None:
        try:
            custom(model, pixel_values)
            attempts.append(f"custom_driver[{getattr(custom, '__name__', 'unknown')}]: ok")
            return True, attempts
        except Exception as exc:
            attempts.append(f"custom_driver[{getattr(custom, '__name__', 'unknown')}]: " f"{type(exc).__name__}: {exc}")

    pattern = SessionDriverPattern()
    if pattern.can_drive(model):
        ok, err = pattern.drive(model, pixel_values)
        if ok:
            attempts.append("SessionDriverPattern: ok")
            return True, attempts
        attempts.append(f"SessionDriverPattern: {err}")

    ok, err = try_introspected_forward(model, pixel_values)
    if ok:
        attempts.append("introspected_forward: ok")
        return True, attempts
    attempts.append(f"introspected_forward: {err}")

    return False, attempts
