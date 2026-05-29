"""Capture-and-replay helper for the per-component PCC test scaffold.

Why this exists
---------------
The auto-generated `tests/pcc/test_<component>.py` files synthesize a
plausible-looking input tensor for every required forward arg via
`_make_arg_for(...)`. The heuristic works for image classifiers and
plain encoders, but it falls over for prompt-conditioned models like
SAM2 where the test scaffolder cannot reasonably fabricate a 6-tensor
multi-modal kwargs set that matches what the real model passes between
its submodules. Result: those tests SKIP with "synthetic inputs are
incompatible", and the bring-up loop has no way to PCC-validate the
generated stub.

This module fixes the problem at the source. We load the HF model ONCE,
register forward hooks on every submodule that has a NEW stub on disk,
run a single forward pass with a realistic top-level input (typically a
pixel_values tensor of the right size), and dump each captured submodule
IO triple `(args, kwargs, output)` to `<demo_dir>/_captured/<safe>/...`
as torch tensor files.

The PCC test files are then patched (idempotently) to prefer these
captured tensors over the synthetic ones. When a component's captured
inputs are available, the test runs with the EXACT tensors the model
itself produces, and the reference output is the captured `output.pt`
- so we get a true PCC value, not a synthetic stand-in.

Design constraints
------------------
* No model-specific code. We resolve submodules via the same
  `_CANDIDATE_SUBMODULE_PATHS` list the generated tests already use, and
  we drive the model via a single `pixel_values` input that we size off
  `model.config.image_size`. This works for every vision HF model in
  the codebase today; for non-vision models we degrade gracefully and
  the test scaffolder's synthetic-input path stays in effect.
* Best-effort. If the model raises during forward, we still save
  whatever submodule IO we captured up to that point. If a particular
  submodule never fires (e.g. lazy branches), its directory simply
  doesn't appear and the PCC test stays on the synthetic path.
* Lightweight on disk. We save tensors as `torch.save` blobs (compact,
  no double-encoding) under `<demo_dir>/_captured/<safe>/`. Each
  directory has `args.pt`, `kwargs.pt`, `output.pt` and a `manifest.json`
  with shape/dtype metadata for human inspection.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _safe_id(name: str) -> str:
    """Backwards-compat shim. See :func:`module_tree.safe_identifier`."""
    from .module_tree import safe_identifier

    return safe_identifier(name)


def _resolve_attr(obj: Any, dotted: str) -> Any:
    """Backwards-compat shim. See :func:`module_tree.resolve_dotted`."""
    from .module_tree import resolve_dotted

    return resolve_dotted(obj, dotted)


def _read_candidates_from_test(test_path: Path) -> List[str]:
    """Pull `_CANDIDATE_SUBMODULE_PATHS = [...]` out of a generated test file.

    Uses bracket-balanced extraction so paths containing `[0]`, `[1]` etc.
    inside the list strings don't trip up the regex."""
    if not test_path.is_file():
        return []
    txt = test_path.read_text(errors="ignore")
    m = re.search(r"_CANDIDATE_SUBMODULE_PATHS\s*=\s*\[", txt)
    if not m:
        return []
    i = m.end() - 1
    depth = 0
    j = i
    in_str: Optional[str] = None
    while j < len(txt):
        c = txt[j]
        if in_str is not None:
            if c == "\\" and j + 1 < len(txt):
                j += 2
                continue
            if c == in_str:
                in_str = None
        else:
            if c in ("'", '"'):
                in_str = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    if depth != 0:
        return []
    literal = txt[i : j + 1]
    try:
        import ast

        return [str(x) for x in ast.literal_eval(literal)]
    except Exception:
        return []


class _Omit:
    pass


_OMIT = _Omit()


def _arg_for(
    name: str,
    *,
    model: Any,
    pixel_values: Any,
    image_size: int,
    captured: Dict[str, Dict[str, Any]],
    components_by_path: Dict[str, str],
) -> Any:
    """Fabricate a plausible value for one forward arg of a submodule we are
    driving directly. Prefers tensors we've already captured from upstream
    components over fresh random tensors, so downstream submodules see
    realistic image_embeddings / prompt_embeddings instead of pure noise.

    Mirrors the synthetic-input policy in the auto-generated PCC tests,
    but is reused here so the capture loop produces the same shapes the
    tests would have produced — and the captured tensors are then a
    drop-in replacement for the synthetic inputs."""
    import torch

    def _captured_output_of(target_path: str) -> Optional[Any]:
        comp_name = components_by_path.get(target_path)
        if comp_name and comp_name in captured and "output" in captured[comp_name]:
            return captured[comp_name]["output"]
        return None

    cfg = getattr(model, "config", None)
    hidden_size = getattr(cfg, "hidden_size", None) or 768
    if name == "pixel_values":
        return pixel_values
    if name in ("hidden_states", "inputs_embeds", "embeddings"):
        return torch.randn(1, 64, hidden_size)
    if name in ("x", "features", "input", "inputs", "hidden_states_in"):
        return torch.randn(1, 64, hidden_size)
    if name == "input_ids":
        vocab = getattr(cfg, "vocab_size", None) or 32000
        return torch.randint(low=1, high=min(vocab, 1000), size=(1, 64), dtype=torch.long)
    if name == "attention_mask":
        return torch.ones(1, 64, dtype=torch.long)
    if name in ("position_ids", "token_type_ids"):
        return torch.zeros(1, 64, dtype=torch.long)
    if name == "image_embeddings":
        cached = captured.get("__image_embeddings_list__")
        if isinstance(cached, list) and cached:
            return cached[-1]
        v = _captured_output_of("vision_encoder")
        if v is not None:
            for attr in ("last_hidden_state", "vision_features", "image_embeddings", "image_embedding"):
                t = getattr(v, attr, None)
                if isinstance(t, torch.Tensor):
                    return t
            if isinstance(v, (tuple, list)) and v and isinstance(v[-1], torch.Tensor):
                return v[-1]
        return torch.randn(1, 256, 64, 64)
    if name == "image_positional_embeddings":
        if hasattr(model, "shared_image_embedding"):
            sie = model.shared_image_embedding
            for fn_name in ("get_dense_positional_embedding", "get_dense_pe"):
                fn = getattr(sie, fn_name, None)
                if callable(fn):
                    try:
                        return fn((64, 64))
                    except Exception:
                        pass
        return torch.randn(1, 256, 64, 64)
    if name == "sparse_prompt_embeddings":
        v = _captured_output_of("prompt_encoder")
        if isinstance(v, (tuple, list)) and len(v) >= 1 and isinstance(v[0], torch.Tensor):
            return v[0]
        return torch.randn(1, 1, 256)
    if name == "dense_prompt_embeddings":
        v = _captured_output_of("prompt_encoder")
        if isinstance(v, (tuple, list)) and len(v) >= 2 and isinstance(v[1], torch.Tensor):
            return v[1]
        return torch.randn(1, 256, 64, 64)
    if name == "high_resolution_features":
        cached = captured.get("__image_embeddings_list__")
        if isinstance(cached, list) and len(cached) >= 2:
            return list(cached[:-1])
        v = _captured_output_of("vision_encoder")
        if v is not None:
            hrf = getattr(v, "fpn_hidden_states", None)
            if isinstance(hrf, (list, tuple)) and len(hrf) >= 2:
                return list(hrf[:2])
        return [torch.randn(1, 32, 256, 256), torch.randn(1, 64, 128, 128)]
    if name == "input_points":
        return torch.tensor([[[[512.0, 512.0]]]])
    if name == "input_labels":
        return torch.tensor([[[1]]])
    if name in ("input_boxes", "input_masks"):
        return None
    if name == "multimask_output":
        return True
    if name.startswith("output_"):
        return False
    if name in {
        "past_key_values",
        "cache_position",
        "use_cache",
        "return_dict",
        "head_mask",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "labels",
    }:
        return None
    return _OMIT


def _detect_image_size(model) -> int:
    cfg = getattr(model, "config", None)
    for path in ("vision_config.image_size", "image_size", "vision_config.input_size"):
        cur = cfg
        ok = True
        for tok in path.split("."):
            cur = getattr(cur, tok, None)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, int):
            return int(cur)
    return 224


def _summarize_value(v: Any) -> Dict[str, Any]:
    """Produce a JSON-able shape/dtype summary of one tensor / structure."""
    import torch

    if isinstance(v, torch.Tensor):
        return {"kind": "tensor", "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return {"kind": type(v).__name__, "items": [_summarize_value(e) for e in v]}
    if isinstance(v, dict):
        return {"kind": "dict", "items": {k: _summarize_value(val) for k, val in v.items()}}
    if v is None:
        return {"kind": "none"}
    return {"kind": "scalar", "type": type(v).__name__, "repr": repr(v)[:120]}


def _resolve_submodule(model: Any, component_name: str, *, demo_dir: Path) -> Optional[Tuple[Any, str]]:
    """Try the same candidate paths the test scaffold uses, then a few
    auto-derived ones, to find the torch submodule for `component_name`.

    We pull `_CANDIDATE_SUBMODULE_PATHS` from both the PCC test file AND
    the stub file — whichever one has it. (Phase-1 SMOKE test files do
    NOT carry the candidate list, but the stub always does.)"""
    safe = _safe_id(component_name)
    test_path = demo_dir / "tests" / "pcc" / f"test_{safe}.py"
    stub_path = demo_dir / "_stubs" / f"{safe}.py"
    candidates: List[str] = []
    for source in (test_path, stub_path):
        for c in _read_candidates_from_test(source):
            if c and c not in candidates:
                candidates.append(c)

    for variant in (component_name, safe, safe.replace("_", ".")):
        if variant and variant not in candidates:
            candidates.append(variant)
    seen: set = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            sub = _resolve_attr(model, path)
            return sub, path
        except (AttributeError, IndexError, KeyError, TypeError):
            continue

    try:
        comp_norm = re.sub(r"[^a-z0-9]+", "", safe.lower())
        for path, mod in model.named_modules():
            if not path:
                continue
            cls_norm = re.sub(r"[^a-z0-9]+", "", type(mod).__name__.lower())
            if comp_norm and comp_norm in cls_norm:
                return mod, path
    except Exception:
        pass
    return None


def capture_real_inputs(
    *,
    model_id: str,
    demo_dir: Path,
    components: List[str],
    image_size_override: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run the HF model ONCE and capture per-component forward IO.

    Returns a `{component: {status, path, manifest}}` dict. `status` is
    one of `"captured"`, `"submodule_not_resolved"`, `"never_fired"`, or
    `"capture_failed"`.
    """
    import os

    import torch
    import transformers

    try:
        _capture_seed = int(os.environ.get("TT_PLANNER_CAPTURE_SEED", "0"))
    except ValueError:
        _capture_seed = 0
    torch.manual_seed(_capture_seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(_capture_seed)

    captured_root = demo_dir / "_captured"
    captured_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"  [capture] loading {model_id} (seed={_capture_seed}) ...",
            file=sys.stderr,
        )

    from scripts.tt_hw_planner.agentic.probe import load_hf_model_cascade

    model, loader_or_err = load_hf_model_cascade(model_id, torch_dtype="float32", verbose=verbose)
    if model is None:
        if verbose:
            print(f"  [capture] {loader_or_err}", file=sys.stderr)
        return {c: {"status": "model_load_failed", "error": loader_or_err or "load failed"} for c in components}

    resolved: List[Tuple[str, Any, str]] = []
    out: Dict[str, Dict[str, Any]] = {}
    for comp in components:
        res = _resolve_submodule(model, comp, demo_dir=demo_dir)
        if res is None:
            if verbose:
                print(f"  [capture] {comp}: submodule not resolved; skipping.", file=sys.stderr)
            out[comp] = {"status": "submodule_not_resolved"}
            continue
        sub, path = res
        resolved.append((comp, sub, path))
        out[comp] = {"status": "pending", "submodule_path": path}

    if not resolved:
        return out

    state: Dict[str, Dict[str, Any]] = {}

    def _make_hook(comp_name: str) -> Tuple[Callable, Callable]:
        def pre_hook(_module, args, kwargs):
            state.setdefault(comp_name, {})
            state[comp_name]["args"] = args
            state[comp_name]["kwargs"] = dict(kwargs) if isinstance(kwargs, dict) else {}
            return None

        def post_hook(_module, _input, output):
            state.setdefault(comp_name, {})
            state[comp_name]["output"] = output
            return None

        return pre_hook, post_hook

    handles: List[Any] = []
    for comp_name, sub, _path in resolved:
        pre, post = _make_hook(comp_name)

        try:
            handles.append(sub.register_forward_pre_hook(pre, with_kwargs=True))
        except TypeError:
            handles.append(
                sub.register_forward_pre_hook(
                    lambda m, a, _c=comp_name: state.setdefault(_c, {}).update({"args": a, "kwargs": {}})
                )
            )
        handles.append(sub.register_forward_hook(post))

    image_size = image_size_override or _detect_image_size(model)
    pixel_values = torch.randn(1, 3, image_size, image_size)
    if verbose:
        print(
            f"  [capture] running drivers with pixel_values shape "
            f"{tuple(pixel_values.shape)} on {len(resolved)} hook(s) ...",
            file=sys.stderr,
        )

    forward_errors: List[str] = []

    def _try(label: str, fn: Callable[[], Any]) -> None:
        try:
            with torch.no_grad():
                fn()
            if verbose:
                print(f"  [capture] driver `{label}`: ok", file=sys.stderr)
        except Exception as exc:
            forward_errors.append(f"{label}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  [capture] driver `{label}`: {type(exc).__name__}: {exc}", file=sys.stderr)

    try:
        try:
            from transformers import AutoTokenizer

            _tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            _probe_text = os.environ.get(
                "TT_PLANNER_CAPTURE_PROBE_TEXT",
                "The quick brown fox jumps over the lazy dog.",
            )
            _enc = _tok(_probe_text, return_tensors="pt")
            _input_ids = _enc.get("input_ids")
            _attn_mask = _enc.get("attention_mask")
            if _input_ids is not None:

                def _run_text():
                    _kw = {"input_ids": _input_ids}
                    if _attn_mask is not None:
                        _kw["attention_mask"] = _attn_mask
                    model(**_kw)

                _try(
                    f"model(input_ids=..., attention_mask=...) " f"[{_input_ids.shape[-1]} tokens]",
                    _run_text,
                )
        except Exception as _exc:
            if verbose:
                print(
                    f"  [capture] text-driver setup skipped " f"({type(_exc).__name__}: {_exc})",
                    file=sys.stderr,
                )

        if hasattr(model, "get_image_embeddings"):

            def _run_gie():
                out = model.get_image_embeddings(pixel_values=pixel_values)
                if isinstance(out, list):
                    state["__image_embeddings_list__"] = out

            _try("model.get_image_embeddings(pixel_values)", _run_gie)

        _try("model(pixel_values=...)", lambda: model(pixel_values=pixel_values))

        try:
            from .capture_drivers import try_capture_drivers as _try_capture_drivers
            from .auto_capture_driver_onboard import (
                load_learned_drivers as _load_learned_drivers,
            )
        except Exception:
            _try_capture_drivers = None
            _load_learned_drivers = None

        if _load_learned_drivers is not None:
            _loaded = _load_learned_drivers()
            if _loaded and verbose:
                print(f"  [capture] loaded {len(_loaded)} learned driver(s)", file=sys.stderr)

        _generic_attempts: List[str] = []
        if _try_capture_drivers is not None:
            _ok_generic, _generic_attempts = _try_capture_drivers(model, pixel_values)
            for _line in _generic_attempts:
                if verbose:
                    print(f"  [capture] generic-driver: {_line}", file=sys.stderr)
            if _ok_generic:
                forward_errors.append("generic-driver: ok")

        _generic_attempts_for_onboard = list(_generic_attempts)

        import inspect as _inspect

        def _build_kwargs_for(submodule: Any) -> Dict[str, Any]:
            sig = _inspect.signature(submodule.forward)
            kw: Dict[str, Any] = {}
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                if param.default is not _inspect.Parameter.empty:
                    is_required = False
                else:
                    is_required = True
                if not is_required and pname not in {
                    "pixel_values",
                    "hidden_states",
                    "input_ids",
                    "image_embeddings",
                    "image_positional_embeddings",
                    "sparse_prompt_embeddings",
                    "dense_prompt_embeddings",
                    "high_resolution_features",
                    "input_points",
                    "input_labels",
                    "input_boxes",
                    "input_masks",
                    "multimask_output",
                    "inputs_embeds",
                    "embeddings",
                }:
                    continue
                val = _arg_for(
                    pname,
                    model=model,
                    pixel_values=pixel_values,
                    image_size=image_size,
                    captured=state,
                    components_by_path={p: c for (c, _s, p) in resolved},
                )
                if val is _OMIT:
                    if is_required:
                        return {}
                    continue
                kw[pname] = val
            return kw

        for comp_name, sub, path in resolved:
            if comp_name in state and "output" in state[comp_name]:
                continue
            try:
                kw = _build_kwargs_for(sub)
            except Exception as exc:
                forward_errors.append(f"build_kwargs[{comp_name}]: {exc}")
                continue
            if not kw:
                continue
            _try(f"submodule[{path}](**{list(kw.keys())})", lambda _sub=sub, _kw=kw: _sub(**_kw))

        try:
            from .auto_capture_driver_onboard import auto_onboard_capture_driver as _auto_onboard_drv
        except Exception:
            _auto_onboard_drv = None

        _still_uncaptured = [
            comp_name for comp_name, _s, _p in resolved if comp_name not in state or "output" not in state[comp_name]
        ]
        if (
            _auto_onboard_drv is not None
            and _still_uncaptured
            and bool(os.environ.get("TT_PLANNER_AUTO_ONBOARD_DRIVER"))
        ):
            if verbose:
                print(
                    f"  [capture] generic framework left {len(_still_uncaptured)} "
                    f"component(s) un-captured; invoking auto-onboard to draft a "
                    f"custom driver via LLM (TT_PLANNER_AUTO_ONBOARD_DRIVER=1)",
                    file=sys.stderr,
                )
            try:
                ok, path, msg = _auto_onboard_drv(
                    model=model,
                    model_id=model_id,
                    uncaptured_components=_still_uncaptured,
                    framework_attempts=_generic_attempts_for_onboard,
                )
                if verbose:
                    print(f"  [capture] auto-onboard: {msg}", file=sys.stderr)
                if ok:
                    from .capture_drivers import try_capture_drivers as _retry_drivers

                    _ok_retry, _retry_attempts = _retry_drivers(model, pixel_values)
                    for _line in _retry_attempts:
                        if verbose:
                            print(f"  [capture] post-onboard: {_line}", file=sys.stderr)
            except Exception as exc:
                if verbose:
                    print(
                        f"  [capture] auto-onboard raised: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    forward_error: Optional[str] = "; ".join(forward_errors) if forward_errors else None

    for comp_name, _sub, path in resolved:
        if comp_name not in state or "output" not in state[comp_name]:
            out[comp_name] = {
                "status": "never_fired",
                "submodule_path": path,
                "forward_error": forward_error,
            }
            continue
        capture = state[comp_name]
        safe = _safe_id(comp_name)
        comp_dir = captured_root / safe
        comp_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(capture.get("args", ()), comp_dir / "args.pt")
            torch.save(capture.get("kwargs", {}), comp_dir / "kwargs.pt")
            torch.save(capture["output"], comp_dir / "output.pt")
            manifest = {
                "component": comp_name,
                "submodule_path": path,
                "args": _summarize_value(capture.get("args", ())),
                "kwargs": _summarize_value(capture.get("kwargs", {})),
                "output": _summarize_value(capture["output"]),
            }
            (comp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            out[comp_name] = {
                "status": "captured",
                "submodule_path": path,
                "dir": str(comp_dir),
                "manifest": manifest,
            }
            if verbose:
                print(
                    f"  [capture] {comp_name}: captured "
                    f"args={len(capture.get('args', ()))} kwargs="
                    f"{len(capture.get('kwargs', {}))} output={_summarize_value(capture['output']).get('kind')}",
                    file=sys.stderr,
                )
        except Exception as exc:
            out[comp_name] = {
                "status": "capture_failed",
                "submodule_path": path,
                "error": str(exc),
            }
            if verbose:
                print(f"  [capture] {comp_name}: save failed: {exc}", file=sys.stderr)

    return out


CAPTURE_LOADER_SOURCE = '''
def _captured_submodule_path(component_name):
    """Read the submodule_path the capture step hooked when it saved
    inputs for this component. Returns the path string or ``None``.

    BUG-2 FIX: when capture records args for submodule path A but the
    test resolves a different path B (different entry in
    ``_CANDIDATE_SUBMODULE_PATHS``), the captured args/kwargs don't
    fit B's signature and the test fails silently with a misleading
    ``_make_arg_for() inputs are shape-incompatible`` error. Reading
    the manifest's recorded path and using it as the FIRST candidate
    keeps capture's resolution and test's resolution aligned."""
    import json as _json
    import re as _re
    from pathlib import Path as _Path
    safe = _re.sub(r"[^A-Za-z0-9_]+", "_", component_name).strip("_").lower() or "component"
    here = _Path(__file__).resolve()
    demo_dir = here.parents[2]
    manifest_p = demo_dir / "_captured" / safe / "manifest.json"
    if not manifest_p.is_file():
        return None
    try:
        data = _json.loads(manifest_p.read_text())
        path = data.get("submodule_path")
        if isinstance(path, str) and path:
            return path
    except Exception:
        pass
    return None


def _maybe_load_captured(component_name):
    """Load `(args, kwargs, output)` from `<demo_dir>/_captured/<safe>/...`
    if the planner's capture-inputs step produced them; return `None`
    otherwise. Lets the test bypass the synthetic-input path when we have
    REAL intermediate tensors from a live HF forward pass."""
    import os as _os
    import re as _re
    from pathlib import Path as _Path
    safe = _re.sub(r"[^A-Za-z0-9_]+", "_", component_name).strip("_").lower() or "component"
    here = _Path(__file__).resolve()
    # tests/pcc/test_X.py -> demo_dir = parents[2]
    demo_dir = here.parents[2]
    comp_dir = demo_dir / "_captured" / safe
    if not comp_dir.is_dir():
        return None
    args_p = comp_dir / "args.pt"
    kwargs_p = comp_dir / "kwargs.pt"
    output_p = comp_dir / "output.pt"
    if not (args_p.is_file() and kwargs_p.is_file() and output_p.is_file()):
        return None
    try:
        import torch as _torch
        args = _torch.load(args_p, map_location="cpu", weights_only=False)
        kwargs = _torch.load(kwargs_p, map_location="cpu", weights_only=False)
        output = _torch.load(output_p, map_location="cpu", weights_only=False)
        print(f"[bringup] using captured inputs from {comp_dir}", flush=True)
        return args, kwargs, output
    except Exception as _e:
        print(f"[bringup] captured-inputs load failed for {component_name}: {_e}", flush=True)
        return None
'''


_INJECTION_MARKER_V1 = "# CAPTURE_LOADER_INJECTED_V1"


_CAPTURED_SHORT_CIRCUIT_BLOCK = """
    # CAPTURE_LOADER_INJECTED_V1
    _captured = _maybe_load_captured(COMPONENT_NAME)
    if _captured is not None:
        _cap_args, _cap_kwargs, _cap_output = _captured
        # Drop stateful / cache kwargs that capture saved as live
        # objects (DynamicCache, etc.) -- they carry dtypes that don't
        # round-trip through torch.save/load reliably and cause
        # query/key/value dtype mismatch inside the attention forward.
        # The model will rebuild them internally from `hidden_states`.
        _cap_kwargs = {
            _k: _v for _k, _v in (_cap_kwargs or {}).items()
            if _k not in ("past_key_values", "past_key_value", "use_cache",
                          "cache_position", "output_attentions",
                          "output_hidden_states", "return_dict")
        }
        # Cast captured float tensors to match the live torch_module's
        # parameter dtype. Capture was usually run in float32; the
        # test's HF model often loads in bfloat16 (transformers default
        # for Qwen3 / Llama / etc.). Without this cast we hit
        # ``expected m1 and m2 to have the same dtype, but got: float
        # != c10::BFloat16`` and SKIP the test even though the inputs
        # are otherwise valid.
        _target_dtype = None
        try:
            for _p in torch_module.parameters():
                if _p.is_floating_point():
                    _target_dtype = _p.dtype
                    break
        except Exception:
            _target_dtype = None
        def _cast_to_target(_x):
            if _target_dtype is None or _x is None:
                return _x
            if isinstance(_x, torch.Tensor) and _x.is_floating_point() and _x.dtype != _target_dtype:
                return _x.to(_target_dtype)
            if isinstance(_x, (list, tuple)):
                _seq = [_cast_to_target(_v) for _v in _x]
                return type(_x)(_seq) if not isinstance(_x, tuple) else tuple(_seq)
            return _x
        _cap_args = tuple(_cast_to_target(_v) for _v in _cap_args) if _cap_args else _cap_args
        _cap_kwargs = {_k: _cast_to_target(_v) for _k, _v in (_cap_kwargs or {}).items()}
        kwargs = dict(_cap_kwargs or {})
        if _cap_args:
            _sig = inspect.signature(torch_module.forward)
            _names = [
                _p.name for _p in _sig.parameters.values()
                if _p.name != "self"
                and _p.kind not in (_p.VAR_POSITIONAL, _p.VAR_KEYWORD)
            ]
            for _i, _v in enumerate(_cap_args):
                if _i >= len(_names):
                    break
                if _names[_i] in kwargs:
                    continue
                kwargs[_names[_i]] = _v
        primary = None
        for _name, _val in kwargs.items():
            if isinstance(_val, torch.Tensor):
                primary = (_name, _val)
                break
        if primary is None:
            primary = ("(captured)", torch.zeros(1))
        return torch_module, kwargs, primary
"""


def upgrade_test_to_use_captured_inputs(test_path: Path) -> bool:
    """Idempotently inject the captured-inputs short-circuit into a
    generated PCC test file. Returns True if the file was modified."""
    if not test_path.is_file():
        return False
    src = test_path.read_text(errors="ignore")
    if _INJECTION_MARKER_V1 in src:
        return False

    anchor = "def _build_torch_reference():"
    if anchor not in src:
        return False
    src = src.replace(anchor, CAPTURE_LOADER_SOURCE.lstrip() + "\n\n" + anchor, 1)

    resolved_marker = 'print(f"[bringup] resolved torch submodule via `{resolved_path}`")'
    if resolved_marker not in src:
        guard_marker = '"`_CANDIDATE_SUBMODULE_PATHS`."\n        )'
        if guard_marker in src:
            src = src.replace(
                guard_marker,
                guard_marker + "\n" + _CAPTURED_SHORT_CIRCUIT_BLOCK.rstrip("\n"),
                1,
            )
        else:
            return False
    else:
        src = src.replace(
            resolved_marker,
            resolved_marker + "\n" + _CAPTURED_SHORT_CIRCUIT_BLOCK.rstrip("\n"),
            1,
        )

    test_path.write_text(src)
    return True


def upgrade_test_to_set_l1_small_size(
    test_path: Path,
    *,
    l1_small_size: int = 24576,
) -> bool:
    """Idempotently rewrite a stale `device_params=[{}]` parametrize on a
    generated PCC test to inject `l1_small_size=<N>`. Returns True if the
    file was modified.

    Background. The scaffolder used to emit
        @pytest.mark.parametrize("device_params", [{}], indirect=True)
    on every generated PCC test. The empty dict opens the device with
    `l1_small_size=0`, which makes the FIRST `ttnn.conv2d` /
    `ttnn.max_pool2d` raise `TT_FATAL: bank size is 0 B`. The autofilled
    stub then falls back to torch-on-host for that op — correct, but
    turns a 30-second PCC test into a 5-15 minute one. On SAM2-hiera-
    small (observed 2026-05-22) this blew past the 10-minute pre-flight
    pytest budget without ever completing a single test, leaving the
    auto-iterate loop with no signal to act on.

    The scaffolder templates in `bringup_loop.py` now default to
    `l1_small_size=24576`, but ALL existing scaffolded test files (e.g.
    from prior `up` runs or from someone resuming a demo created with
    the old template) still carry the broken `[{}]` form. Without an
    in-place fixer those stale files would silently keep blowing the
    budget forever — the user's only recovery is to `rm -rf` the demo
    dir and re-scaffold from scratch.

    This helper backfills the kwarg so a single re-run of `up` or
    `promote` repairs every stale demo. It is invoked from
    `upgrade_all_tests_in_demo`, which runs once per auto-iterate
    pre-flight."""
    if not test_path.is_file():
        return False
    src = test_path.read_text(errors="ignore")
    old = '@pytest.mark.parametrize("device_params", [{}], indirect=True)'
    if old not in src:
        return False
    new = f'@pytest.mark.parametrize("device_params", ' f'[{{"l1_small_size": {int(l1_small_size)}}}], indirect=True)'
    src = src.replace(old, new)
    test_path.write_text(src)
    return True


def upgrade_all_tests_in_demo(demo_dir: Path) -> List[Tuple[str, bool]]:
    """Apply `upgrade_test_to_use_captured_inputs` AND
    `upgrade_test_to_set_l1_small_size` to every PCC test in the demo
    and return per-test outcomes.

    A test counts as 'modified' if EITHER upgrader changed it — the
    auto-iterate banner prints whichever number is non-zero. We chain
    both upgraders because they edit DIFFERENT parts of the file
    (captured-inputs short-circuit vs the parametrize header) and are
    both idempotent on their own."""
    out: List[Tuple[str, bool]] = []
    pcc_dir = demo_dir / "tests" / "pcc"
    if not pcc_dir.is_dir():
        return out
    for tp in sorted(pcc_dir.glob("test_*.py")):
        modified_any = False
        try:
            if upgrade_test_to_use_captured_inputs(tp):
                modified_any = True
        except Exception as exc:
            print(f"  [capture] upgrade failed for {tp.name}: {exc}", file=sys.stderr)
        try:
            if upgrade_test_to_set_l1_small_size(tp):
                modified_any = True
                print(
                    f"  [capture] {tp.name}: backfilled "
                    f"l1_small_size=24576 on stale [{{}}] device_params "
                    f"(prevents conv2d CPU-fallback storm)."
                )
        except Exception as exc:
            print(
                f"  [capture] l1_small_size upgrade failed for " f"{tp.name}: {exc}",
                file=sys.stderr,
            )
        out.append((tp.name, modified_any))
    return out
