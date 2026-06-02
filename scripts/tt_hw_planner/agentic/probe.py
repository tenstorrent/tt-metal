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


# ─── HF ↔ TT chain divergence comparison ─────────────────────────────
#
# When the end-to-end PCC gate fails, both probes have already run
# (HF live via ``probe_hf_modules``, TT-side via the
# :mod:`.tt_probe` instrumentation that persists records to JSON).
# What's missing is the comparator that pairs records by
# ``(qualified_name, step)`` and surfaces the first module where the
# TT-side activation drifts beyond a relative-tolerance threshold.
#
# Why summary-stat drift, not Pearson PCC?
# ----------------------------------------
# Both probes record SUMMARY STATISTICS per layer output, not raw
# tensors (the raw tensors would be too large to ship across the
# pytest subprocess boundary). So we compute relative drift in
# (mean, std, l2, abs_max) instead of PCC. The metric is coarser than
# per-tensor Pearson, but the goal here is LOCALIZATION ("which
# module diverged"), not VERIFICATION ("is the output numerically
# faithful"). The strict-PCC gate already did the verification step
# and said "no"; this comparator says WHERE.


@dataclass(frozen=True)
class ModuleDivergence:
    """One row of the HF vs TT comparison at a specific
    ``(qualified_name, step)``.

    ``relative_drift`` carries the per-statistic ratio
    ``|hf - tt| / max(|hf|, |tt|, eps)`` for each summary stat.
    ``max_drift`` is the dominant signal, used to compare against the
    threshold.
    """

    qualified_name: str
    class_name: str
    step: int
    hf_stats: Dict[str, Any]
    tt_stats: Dict[str, Any]
    relative_drift: Dict[str, float]
    max_drift: float


@dataclass
class ChainDivergenceResult:
    """Output of :func:`compare_hf_tt_probes`.

    ``first_divergence`` is the first ``(qualified_name, step)`` (in
    HF-trace order) where any summary statistic exceeded
    ``threshold``. ``None`` means every paired module was within
    tolerance — useful for the caller to know the divergence is in an
    UNPAIRED module (i.e. TT side never produced records for it).

    ``unpaired_hf_modules`` / ``unpaired_tt_modules`` flag the
    asymmetry cases: an HF module with no TT counterpart usually
    means TT-side probe didn't hook that module (potential missing
    component); a TT module with no HF counterpart is rare and
    usually means the TT side ran a path HF doesn't.
    """

    first_divergence: Optional[ModuleDivergence]
    table: List[ModuleDivergence]
    paired_modules: int
    unpaired_hf_modules: List[str]
    unpaired_tt_modules: List[str]
    threshold: float
    note: str


_DRIFT_STATS = ("mean", "std", "l2", "abs_max")
_DRIFT_EPS = 1e-8


def _relative_drift(hf_val: float, tt_val: float) -> float:
    """Relative deviation between two scalar stats.

    ``|hf - tt| / max(|hf|, |tt|, eps)`` — symmetric and bounded.
    Eps prevents divide-by-zero when both sides are zero (in which
    case the drift is 0 — they agree). Returns ``inf`` if either
    side is NaN (treat as divergent).
    """
    if hf_val != hf_val or tt_val != tt_val:  # NaN check
        return float("inf")
    denom = max(abs(hf_val), abs(tt_val), _DRIFT_EPS)
    return abs(hf_val - tt_val) / denom


def compare_hf_tt_probes(
    hf_result: HFProbeResult,
    tt_records: List[Dict[str, Any]],
    *,
    threshold: float = 0.05,
) -> ChainDivergenceResult:
    """Pair HF probe records with TT probe records by
    ``(qualified_name, step)`` and return the first module where any
    summary statistic exceeded ``threshold`` relative drift.

    Parameters
    ----------
    hf_result
        Live HF probe output (run on CPU with hooks across every
        non-container module).
    tt_records
        TT-side records as a list of dicts, typically loaded from
        the JSON file ``tt_probe`` writes. Each dict must carry at
        minimum ``qualified_name``, ``step``, and the four summary
        stats in ``_DRIFT_STATS``.
    threshold
        Relative-drift cutoff. ``0.05`` (5%) is a reasonable default
        for "the layer's output is meaningfully different." Tightening
        catches subtler drift; loosening avoids false positives from
        sampling noise.

    Returns
    -------
    A :class:`ChainDivergenceResult`. Never raises; malformed inputs
    produce an empty table with a descriptive ``note``.

    Pure (no I/O, no side effects). Unit-testable with synthetic
    record lists.
    """
    if not isinstance(hf_result, HFProbeResult):
        return ChainDivergenceResult(
            first_divergence=None,
            table=[],
            paired_modules=0,
            unpaired_hf_modules=[],
            unpaired_tt_modules=[],
            threshold=threshold,
            note="invalid hf_result (not an HFProbeResult)",
        )
    if not isinstance(tt_records, list):
        return ChainDivergenceResult(
            first_divergence=None,
            table=[],
            paired_modules=0,
            unpaired_hf_modules=[],
            unpaired_tt_modules=[],
            threshold=threshold,
            note="invalid tt_records (not a list)",
        )

    # Index TT records by (qualified_name, step) for O(1) lookup.
    tt_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in tt_records:
        if not isinstance(rec, dict):
            continue
        qn = rec.get("qualified_name")
        st = rec.get("step")
        if isinstance(qn, str) and isinstance(st, int):
            tt_index[(qn, st)] = rec

    table: List[ModuleDivergence] = []
    first: Optional[ModuleDivergence] = None
    unpaired_hf: List[str] = []
    seen_tt_keys = set()

    for hf_rec in hf_result.records:
        key = (hf_rec.qualified_name, hf_rec.step)
        tt_rec = tt_index.get(key)
        if tt_rec is None:
            unpaired_hf.append(f"{hf_rec.qualified_name}@{hf_rec.step}")
            continue
        seen_tt_keys.add(key)
        drift_map: Dict[str, float] = {}
        for stat_key in _DRIFT_STATS:
            hf_val = getattr(hf_rec, stat_key, None)
            tt_val = tt_rec.get(stat_key)
            if hf_val is None or tt_val is None:
                continue
            try:
                drift_map[stat_key] = _relative_drift(float(hf_val), float(tt_val))
            except (TypeError, ValueError):
                continue
        if not drift_map:
            # No comparable stats; record but don't gate on it.
            continue
        max_drift = max(drift_map.values())
        div = ModuleDivergence(
            qualified_name=hf_rec.qualified_name,
            class_name=hf_rec.class_name,
            step=hf_rec.step,
            hf_stats={k: getattr(hf_rec, k, None) for k in _DRIFT_STATS},
            tt_stats={k: tt_rec.get(k) for k in _DRIFT_STATS},
            relative_drift=drift_map,
            max_drift=max_drift,
        )
        table.append(div)
        if first is None and max_drift > threshold:
            first = div

    unpaired_tt = [f"{qn}@{st}" for (qn, st) in tt_index.keys() if (qn, st) not in seen_tt_keys]

    # Note-resolution table:
    #   • no paired modules        → "no paired modules ..."
    #   • paired, divergence found → "ok" (caller acts on first_divergence)
    #   • paired, no divergence, no orphans → "ok" (genuine clean comparison)
    #   • paired, no divergence, orphans present → "check unpaired modules"
    note = "ok"
    if not table:
        note = "no paired modules — TT and HF probes shared no (qualified_name, step) keys"
    elif first is None and (unpaired_hf or unpaired_tt):
        note = "all paired modules within threshold — divergence may be in unpaired modules"

    return ChainDivergenceResult(
        first_divergence=first,
        table=table,
        paired_modules=len(table),
        unpaired_hf_modules=unpaired_hf,
        unpaired_tt_modules=unpaired_tt,
        threshold=threshold,
        note=note,
    )


__all__ = [
    "HFModuleStats",
    "HFProbeResult",
    "probe_hf_modules",
    "ModuleDivergence",
    "ChainDivergenceResult",
    "compare_hf_tt_probes",
]
