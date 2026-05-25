"""HF-side universal module-tree probe.

This is the HF-reference half of the G1 dual-probe. It loads the HF
model on CPU, walks ``named_modules()``, installs a forward hook on
every non-container module, runs the same prompt the demo saw, and
returns a list of per-module activation stats.

The TT-side counterpart is :mod:`.tt_probe`, which runs inside the
demo subprocess.

Why not just reuse ``activation_diff.localize_decode_divergence``?
------------------------------------------------------------------
That function:

* Hardcodes a list of "decoder layer" paths (``model.layers``,
  ``model.language_model.layers``, ...). To extend to a new
  architecture you append to the list. That's the category-specific
  hardcoding the user objected to.
* Only hooks decoder layers. Doesn't capture embedding output, lm_head,
  per-attention-head outputs, MLP outputs, etc.
* Returns one row per (decoder_layer, decode_step) -- fine for LLMs,
  but misses VLM vision-tower modules, Whisper encoder layers, SAM2
  hiera blocks, etc.

This module is HF-class-name-agnostic: it walks ``named_modules()`` and
hooks every leaf and intermediate module that produces a tensor. The
selection heuristic (same ``_LAYERISH_SUFFIXES`` as :mod:`.tt_probe`)
narrows to the architecturally-meaningful boundaries so the output
table is short enough to feed into an LLM prompt.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


from scripts.tt_hw_planner.module_tree import (
    _CONTAINER_CLASS_NAMES,
    _HIGH_LEVEL_SUFFIXES as _LAYERISH_SUFFIXES,
)


@dataclass
class HFModuleStats:
    """One row of the HF-side activation trace."""

    qualified_name: str
    class_name: str
    step: int
    shape: Tuple[int, ...]
    dtype: str
    mean: float
    std: float
    l2: float
    abs_max: float


@dataclass
class HFProbeResult:
    """Output of :func:`probe_hf_modules`."""

    model_id: str
    records: List[HFModuleStats] = field(default_factory=list)
    num_modules_hooked: int = 0
    decode_steps: List[int] = field(default_factory=list)
    note: str = "ok"
    prompt_text: str = ""
    elapsed_s: float = 0.0


def _looks_layerish(class_name: str) -> bool:
    """True iff ``class_name`` looks like a compute layer (not a
    container, not a utility). Reuses the same suffix list as
    :func:`module_tree._looks_high_level`; container-class names are
    excluded explicitly so ``ModuleList`` etc. never get probed."""
    if class_name in _CONTAINER_CLASS_NAMES:
        return False
    from scripts.tt_hw_planner.module_tree import _looks_high_level

    return _looks_high_level(class_name)


def _hidden_from_layer_output(out: Any) -> Any:
    """HF decoder layers return Tensor or (Tensor, ...). Normalise via
    the shared helper in :mod:`activation_diff`."""
    from scripts.tt_hw_planner.activation_diff import _hidden_state_from_layer_output

    return _hidden_state_from_layer_output(out)


def _tensor_stats(t: Any, *, name: str, cls: str, step: int) -> Optional[HFModuleStats]:
    """Reduce a tensor to per-module stats. Delegates the actual
    scalar reduction to :func:`activation_diff._tensor_stats` (the
    canonical shared helper) and wraps the dict into an
    :class:`HFModuleStats` dataclass."""
    try:
        import torch
    except Exception:
        return None
    if not isinstance(t, torch.Tensor):
        return None

    if t.dim() == 3:
        slot = t[..., -1, :]
    else:
        slot = t
    from scripts.tt_hw_planner.activation_diff import _tensor_stats as _base_stats

    d = _base_stats(slot)
    if d is None:
        return None
    return HFModuleStats(
        qualified_name=name,
        class_name=cls,
        step=step,
        shape=tuple(t.shape),
        dtype=d["dtype"],
        mean=d.get("mean", 0.0),
        std=d.get("std", 0.0),
        l2=d.get("l2", 0.0),
        abs_max=d.get("abs_max", 0.0),
    )


def load_hf_model_cascade(
    model_id: str,
    *,
    torch_dtype: str = "bfloat16",
    timeout_s: float = 300.0,
    verbose: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    """Universal HF model loader: cascades through every reasonable
    ``transformers.AutoModelFor*`` class until one succeeds.

    Returns ``(model, loader_name)`` on success, ``(None, error_msg)``
    on failure. Extracted from :func:`probe_hf_modules` so
    :mod:`capture_inputs` can reuse the cascade without duplicating
    the loader-class list. Adding a new modality (e.g. TTS) means
    appending one string here -- every caller benefits.
    """
    import time as _t

    start = _t.monotonic()
    try:
        import torch
        import transformers
    except Exception as exc:
        return (None, f"import-failed: {type(exc).__name__}: {exc}")
    from scripts.tt_hw_planner.activation_diff import _torch_dtype_from_string

    dtype = _torch_dtype_from_string(torch_dtype) or torch.bfloat16
    hf_model = None
    last_exc: Optional[BaseException] = None
    used_loader: Optional[str] = None
    for loader in (
        "AutoModelForCausalLM",
        "AutoModelForSpeechSeq2Seq",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForImageClassification",
        "AutoModel",
    ):
        if _t.monotonic() - start > timeout_s - 5:
            break
        try:
            cls = getattr(transformers, loader, None)
            if cls is None:
                continue
            hf_model = cls.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            used_loader = loader
            if verbose:
                print(f"  [hf-loader] {model_id} via {loader}", file=sys.stderr, flush=True)
            break
        except Exception as exc:
            last_exc = exc
            if verbose:
                print(
                    f"  [hf-loader] {loader} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            continue
    if hf_model is None:
        return (
            None,
            f"hf-load-failed: {type(last_exc).__name__ if last_exc else 'unknown'}: "
            f"{last_exc if last_exc else 'all loaders refused this model'}",
        )
    try:
        hf_model.eval()
    except Exception:
        pass
    return (hf_model, used_loader)


def probe_hf_modules(
    *,
    model_id: str,
    prompt_text: str,
    max_total_steps: int = 4,
    torch_dtype: str = "bfloat16",
    timeout_s: float = 300.0,
    verbose: bool = False,
) -> Optional[HFProbeResult]:
    """Load the HF model, install hooks on every layer-like submodule,
    run prefill + ``max_total_steps`` greedy decode iterations, return
    a per-(module, step) stats table.

    Categories supported: any model loadable via
    ``transformers.AutoModelForCausalLM`` / ``AutoModel`` /
    ``AutoModelForSpeechSeq2Seq`` / etc. The function tries
    ``AutoModelForCausalLM`` first (highest LLM/VLM coverage); falls
    back to ``AutoModel``. For non-text categories the caller would
    pass a single-step prompt and the probe still emits one record per
    layer.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"[hf_probe:{model_id}] {msg}", file=sys.stderr, flush=True)

    start = time.monotonic()

    try:
        import torch
    except Exception:
        return HFProbeResult(model_id=model_id, note="torch-missing", prompt_text=prompt_text)

    try:
        import transformers
    except Exception:
        return HFProbeResult(model_id=model_id, note="transformers-missing", prompt_text=prompt_text)

    hf_model, loader_or_err = load_hf_model_cascade(
        model_id, torch_dtype=torch_dtype, timeout_s=timeout_s, verbose=verbose
    )
    if hf_model is None:
        return HFProbeResult(
            model_id=model_id,
            note=loader_or_err or "hf-load-failed",
            prompt_text=prompt_text,
        )
    _log(f"loaded via {loader_or_err}")

    captured: Dict[Tuple[str, str], Any] = {}
    handles: List[Any] = []
    seen_names: List[Tuple[str, str]] = []
    try:
        for path, mod in hf_model.named_modules():
            if not path:
                continue
            cls_name = type(mod).__name__
            if not _looks_layerish(cls_name):
                continue
            if not hasattr(mod, "register_forward_hook"):
                continue

            def _mk(p: str, c: str) -> Callable[[Any, Any, Any], None]:
                def _hook(_m, _inp, out):
                    captured[(p, c)] = out

                return _hook

            try:
                handles.append(mod.register_forward_hook(_mk(path, cls_name)))
                seen_names.append((path, cls_name))
            except Exception:
                continue
    except Exception as exc:
        _log(f"hook install failed: {type(exc).__name__}: {exc}")

    _log(f"hooked {len(seen_names)} modules")
    if not seen_names:
        return HFProbeResult(
            model_id=model_id,
            note="no-layerish-modules-found",
            prompt_text=prompt_text,
            num_modules_hooked=0,
        )

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        except Exception:
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
    except Exception as exc:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        return HFProbeResult(
            model_id=model_id,
            note=f"tokenize-failed: {type(exc).__name__}",
            prompt_text=prompt_text,
            num_modules_hooked=len(seen_names),
        )

    records: List[HFModuleStats] = []
    captured_steps: List[int] = []
    note = "ok"
    try:
        with torch.no_grad():
            captured.clear()
            try:
                out = hf_model(input_ids=input_ids, use_cache=True, return_dict=True)
                past = getattr(out, "past_key_values", None)
                logits = getattr(out, "logits", None)
                if logits is None:
                    raise RuntimeError("HF model produced no logits")
                next_id = int(logits[..., -1, :].argmax(dim=-1).item())
            except Exception as exc:
                _log(f"prefill failed: {type(exc).__name__}: {exc}")
                note = "prefill-failed"
                next_id = -1
                past = None

            def _emit(step: int) -> None:
                for path, cls_name in seen_names:
                    raw = captured.get((path, cls_name))
                    if raw is None:
                        continue
                    h = _hidden_from_layer_output(raw)
                    if h is None:
                        h = raw
                    row = _tensor_stats(h, name=path, cls=cls_name, step=step)
                    if row is not None:
                        records.append(row)

            if note == "ok":
                _emit(0)
                captured_steps.append(0)

                for k in range(1, max_total_steps):
                    if time.monotonic() - start > timeout_s - 5:
                        note = "timeout"
                        break
                    captured.clear()
                    try:
                        out = hf_model(
                            input_ids=torch.tensor([[next_id]], dtype=torch.long),
                            past_key_values=past,
                            use_cache=True,
                            return_dict=True,
                        )
                        past = getattr(out, "past_key_values", past)
                        logits = getattr(out, "logits", None)
                        if logits is None:
                            note = "decode-no-logits"
                            break
                        next_id = int(logits[..., -1, :].argmax(dim=-1).item())
                    except Exception as exc:
                        _log(f"decode step {k} failed: {type(exc).__name__}: {exc}")
                        note = f"decode-step-failed-at-{k}"
                        break
                    _emit(k)
                    captured_steps.append(k)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        try:
            del hf_model
            import gc

            gc.collect()
        except Exception:
            pass

    return HFProbeResult(
        model_id=model_id,
        records=records,
        num_modules_hooked=len(seen_names),
        decode_steps=captured_steps,
        note=note,
        prompt_text=prompt_text,
        elapsed_s=time.monotonic() - start,
    )


__all__ = ["HFModuleStats", "HFProbeResult", "probe_hf_modules"]
