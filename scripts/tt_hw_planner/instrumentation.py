from __future__ import annotations

import os
import sys


def _log(msg: str) -> None:
    print(f"[tt_hw_planner.instrumentation] {msg}", file=sys.stderr, flush=True)


_INSTALLED = False


SHARED_SCOPE: str = "_shared"
_APPLIED_OVERLAYS_SCOPES: list = []


def install_all() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _apply_overlays_for_active_model()
    _install_lightweight_module_probe()
    _install_trace_disable()
    _install_logit_dump()


def _apply_overlays_for_active_model() -> None:
    global _APPLIED_OVERLAYS_SCOPES
    try:
        from .overlay_manager import apply_for
    except Exception as exc:
        _log(f"overlay_manager import failed; overlays skipped: {type(exc).__name__}: {exc}")
        return

    scopes = [SHARED_SCOPE]
    model_id = os.environ.get("TT_HW_PLANNER_OVERLAY_MODEL", "")
    if model_id:
        scopes.append(model_id)

    for scope in scopes:
        n, files = apply_for(scope)
        if n:
            _APPLIED_OVERLAYS_SCOPES.append(scope)
            _log(f"applied {n} overlay(s) for scope={scope}:")
            for f in files:
                _log(f"  + {f}")


def pytest_unconfigure(config) -> None:
    global _APPLIED_OVERLAYS_SCOPES
    if not _APPLIED_OVERLAYS_SCOPES:
        return
    try:
        from .overlay_manager import revert_for
    except Exception:
        return
    for scope in reversed(_APPLIED_OVERLAYS_SCOPES):
        n, _ = revert_for(scope)
        if n:
            _log(f"reverted {n} overlay(s) for scope={scope}")
    _APPLIED_OVERLAYS_SCOPES = []


def _install_lightweight_module_probe() -> None:
    if not os.environ.get("TT_PLANNER_PROBE_OUTPUT"):
        return
    try:
        from models.common.lightweightmodule import LightweightModule
    except Exception as exc:
        _log(f"LightweightModule import failed; probe disabled: {type(exc).__name__}: {exc}")
        return

    if getattr(LightweightModule, "_tt_hw_planner_probe_patched", False):
        return

    orig_call = LightweightModule.__call__

    def patched_call(self, *args, **kwargs):
        if not getattr(LightweightModule, "_tt_hw_planner_probe_done", False):
            LightweightModule._tt_hw_planner_probe_done = True
            try:
                _auto_attach_probe(self)
            except Exception as exc:
                _log(f"probe auto-attach failed: {type(exc).__name__}: {exc}")
        return orig_call(self, *args, **kwargs)

    LightweightModule.__call__ = patched_call
    LightweightModule._tt_hw_planner_probe_patched = True
    _log("LightweightModule.__call__ patched for probe auto-attach")


def _auto_attach_probe(first_call_self) -> None:
    import gc

    from models.common.lightweightmodule import LightweightModule
    from scripts.tt_hw_planner.agentic.tt_probe import maybe_install_global

    all_lwm = [o for o in gc.get_objects() if isinstance(o, LightweightModule)]
    child_ids = set()
    for o in all_lwm:
        try:
            refs = gc.get_referents(o)
        except Exception:
            continue
        for r in refs:
            if isinstance(r, LightweightModule) and r is not o:
                child_ids.add(id(r))
            elif isinstance(r, (list, tuple, dict)):
                try:
                    members = r.values() if isinstance(r, dict) else r
                    for m in members:
                        if isinstance(m, LightweightModule) and m is not o:
                            child_ids.add(id(m))
                except Exception:
                    pass
    roots = [o for o in all_lwm if id(o) not in child_ids] or [first_call_self]
    for root in roots:
        try:
            n = maybe_install_global(root)
            _log(f"probe attached: root={type(root).__name__} wrapped={n}")
        except Exception as exc:
            _log(f"probe attach failed for {type(root).__name__}: {type(exc).__name__}: {exc}")


def _install_trace_disable() -> None:
    if not (os.environ.get("TT_PLANNER_NO_TRACE") == "1" or os.environ.get("TT_PLANNER_PROBE_OUTPUT")):
        return
    try:
        from models.tt_transformers.demo import simple_text_demo as _demo
    except Exception as exc:
        _log(f"simple_text_demo import failed; trace-disable skipped: {type(exc).__name__}: {exc}")
        return

    orig = getattr(_demo, "test_demo_text", None)
    if orig is None or getattr(orig, "_tt_hw_planner_trace_patched", False):
        return

    import functools

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        if "enable_trace" in kwargs and kwargs["enable_trace"]:
            _log("disabling trace mode for this run (NO_TRACE or PROBE_OUTPUT set)")
            kwargs["enable_trace"] = False
        return orig(*args, **kwargs)

    # functools.wraps copies __dict__ which carries `pytestmark` (the
    # parametrize decorators). Without this, replacing _demo.test_demo_text
    # below collapses the 44-variant parametrize expansion to a single
    # unparametrized test, breaking any `-k '<id>'` selector.
    if hasattr(orig, "pytestmark"):
        wrapped.pytestmark = list(orig.pytestmark)
    wrapped._tt_hw_planner_trace_patched = True
    _demo.test_demo_text = wrapped
    _log("test_demo_text wrapped to force enable_trace=False")


# The step-0 logit dump is REQUIRED for the strict end-to-end PCC gate
# in correctness/text.py to fire. Default-on: capture always installs.
# Disabling is opt-OUT only — set the env var to one of the falsy values
# below (case-insensitive) to skip the patch. Disabling means the strict
# gate will fail-closed (UNVERIFIED) because it has no logits to compare.
LOGITS_DUMP_ENV_VAR = "TT_HW_PLANNER_DUMP_LOGITS"
_LOGITS_DUMP_DISABLE_VALUES = frozenset({"0", "false", "no", "off"})


def _should_install_logit_dump() -> bool:
    """Pure check: should the step-0 logit dump patch fire?

    Default-on. Returns False iff the env var ``TT_HW_PLANNER_DUMP_LOGITS``
    is explicitly set to a falsy value (0/false/no/off, case-insensitive).

    Pure (no side effects) so the unit tests can pin the decision logic
    without needing the ``ttnn``/``Generator`` import chain.
    """
    val = os.environ.get(LOGITS_DUMP_ENV_VAR, "").strip().lower()
    return val not in _LOGITS_DUMP_DISABLE_VALUES


def _install_logit_dump() -> None:
    if not _should_install_logit_dump():
        return
    try:
        from models.tt_transformers.tt.generator import Generator
    except Exception as exc:
        _log(f"Generator import failed; logit-dump skipped: {type(exc).__name__}: {exc}")
        return

    if getattr(Generator, "_tt_hw_planner_logit_dump_patched", False):
        return

    orig = Generator.prefill_forward_text

    def wrapped(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        try:
            _dump_first_step_logits(result)
        except Exception as exc:
            _log(f"logit-dump emit failed: {type(exc).__name__}: {exc}")
        return result

    Generator.prefill_forward_text = wrapped
    Generator._tt_hw_planner_logit_dump_patched = True
    _log("Generator.prefill_forward_text wrapped for step-0 logit dump")


def _dump_first_step_logits(result) -> None:
    """Best-effort step-0 logit dump for the strict PCC gate.

    When ``prefill_forward_text`` is called with ``sampling_params=None``,
    it returns the full ``[batch, 1, vocab_size]`` logits tensor — which
    is exactly what the gate needs. But when on-device sampling is
    enabled (typical for the simple_text_demo which sets
    ``temperature/top_p/top_k``), the function returns the SAMPLED TOKEN
    ID instead — a scalar of shape ``(1,)``. Dumping that as "logits"
    feeds the gate garbage and produces "shape mismatch" errors.

    Decision logic:
      * If the captured tensor's flat size is NOT comparable to a
        vocab dimension (>= 1000 elements), we are almost certainly
        looking at sampled tokens rather than logits. Skip the dump
        and log loudly so the gate's UNVERIFIED message carries an
        actionable next step.
      * Otherwise dump the tensor and emit ``==LOGITS PATH:`` so
        the strict gate can compute PCC.
    """
    from pathlib import Path

    import numpy as _np
    import torch

    if getattr(_dump_first_step_logits, "_done", False):
        return

    if isinstance(result, tuple) and len(result) >= 1:
        logits = result[0]
    else:
        logits = result
    if not isinstance(logits, torch.Tensor):
        _log(
            f"logit-dump skipped: prefill_forward_text returned a "
            f"{type(logits).__name__}, not a Tensor — typically means "
            f"on-device sampling is active and the function emitted "
            f"sampled-token-ids instead of logits."
        )
        return

    flat = logits.reshape(-1)
    if int(flat.numel()) < 1000:
        _log(
            f"logit-dump skipped: captured tensor has only {int(flat.numel())} "
            f"elements (expected ~vocab_size ≥ 1000). On-device sampling is "
            f"likely active — prefill_forward_text returned sampled token IDs, "
            f"not full vocab logits. To get strict logit-PCC, either disable "
            f"on-device sampling (sampling_params=None) OR hook a deeper "
            f"forward call that emits the full output_tensor."
        )
        return

    dump_dir = Path(os.environ.get("TT_HW_PLANNER_DUMP_DIR", "/tmp/tt_hw_planner_runs"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"tt_logits_step0_user0_{os.getpid()}.npy"
    arr = logits[0].detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
    _np.save(str(dump_path), arr)
    print(f"==LOGITS PATH: {dump_path}", flush=True)
    _log(f"logit-dump emitted: shape={tuple(logits.shape)} flat={arr.shape[0]} path={dump_path}")
    _dump_first_step_logits._done = True


def pytest_configure(config) -> None:
    install_all()
