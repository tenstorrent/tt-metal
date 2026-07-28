# SPDX-License-Identifier: Apache-2.0
"""One GENERIC PerfAdapter for trace-replay per-token measurement — valid for ANY model.

`measure_adapter` (agent/trace_replay.py) needs a PerfAdapter (setup/step/refresh_inputs). Rather than
a hand-written adapter per model, this module ships a SINGLE adapter that wraps any pipeline conforming
to a tiny standard DECODE CONTRACT — so the only per-model artifact is the pipeline's own `decode_step`,
which the structural decode lever (or emit-e2e) produces; the adapter itself is model-agnostic and
lives here once.

DECODE CONTRACT (duck-typed on the built pipeline object):
    decode_step(state) -> state          REQUIRED. Exactly one steady-state decode token: reads
                                          persistent on-device buffers, samples on-device, writes the
                                          next id back into `state`. NO host reads (to_torch/.item()).
    decode_prefill(input_ids) -> state    OPTIONAL. Process the prompt once, return the initial cache/
                                          state. If absent, `state` starts as None (fixed-input loop).
    self_traced = True                    OPTIONAL (class/instance attr). Declares that the pipeline OWNS
                                          its trace capture internally (persistent-buffer / vLLM-style
                                          decode -- e.g. GLM's decode(enable_trace=True)). Such a
                                          decode_step does host<->device I/O + execute_trace inside
                                          itself, so measure_adapter must NOT begin_trace_capture around it;
                                          it TIMES the native step instead (see trace_replay._measure_native).
                                          The purity rule above is waived.
    trace_path() -> str                   OPTIONAL, only meaningful with self_traced. Returns the real replay
                                          path ("trace+1cq") the pipeline actually took.

A pipeline WITHOUT decode_step (repeat-prefill / host-argmax decode) raises AttributeError in setup;
the perf test's guard then falls back to FORWARD_WALL_MS and the detector reports 'repeat_prefill'.
"""

from __future__ import annotations

import os
import sys
from typing import Callable


class NotTraceCapable(AttributeError):
    """A pipeline that GENUINELY cannot be trace-replayed (repeat-prefill / host-argmax decode, no
    decode_step and no PIPELINE_STAGES). Subclasses AttributeError so existing `except AttributeError`
    fallbacks still catch it, but the distinct type lets measure_adapter emit a STABLE, authoritative
    TRACE_NOT_TRACE_CAPABLE=1 marker — the ONE legitimate eager terminal — instead of being confused
    with an incidental setup/attribute failure (a real bug) that the generation loop must keep
    correcting. Model- and hardware-agnostic: it is about the pipeline's decode contract, not any
    specific model or board."""


def resolve_mesh_shape(default_rows: int = 1, default_cols: int = 1) -> tuple[int, int]:
    """The topology the run should open, as (rows, cols). optimize/emit-e2e export the planned split
    (plan_parallelism -> TP x DP) into TT_PERF_MESH_ROWS/COLS; a model's device-open (or the generated
    perf test's self-open) calls this to honor it, falling back to its own default when unset. This is
    how --devices/--mesh actually reshapes topology: the tool plans, the open reads it here."""
    _r = (os.environ.get("TT_PERF_MESH_ROWS") or "").strip()
    _c = (os.environ.get("TT_PERF_MESH_COLS") or "").strip()
    if _r or _c:
        # A HALF-set pair used to raise on int("") and fall through to the default, so a planned
        # TP=4 mesh opened as 1x1 and the single-chip measurement was reported as the planned
        # topology. Honour whichever side was given rather than discarding both.
        try:
            r = int(_r) if _r else int(default_rows)
            c = int(_c) if _c else int(default_cols)
            if r >= 1 and c >= 1:
                return r, c
        except (TypeError, ValueError):
            print(
                "  [perf_adapter] WARNING: TT_PERF_MESH_ROWS/COLS set but unparseable (rows=%r cols=%r); "
                "falling back to the source default %dx%d -- the measured topology may NOT be the "
                "planned one" % (_r, _c, default_rows, default_cols),
                file=sys.stderr,
                flush=True,
            )
    return default_rows, default_cols


class PipelineDecodeAdapter:
    """Generic PerfAdapter over any pipeline exposing the decode contract above.

    build_fn    device -> pipeline. Builds the pipeline EXACTLY as the demo/perf test does, on the
                measurement device (so the trace captures the real program).
    prompt_ids  small prompt fed to decode_prefill to build the initial state (ignored if the
                pipeline has no decode_prefill).
    batch       users in the batch — forwarded so trace_replay derives tokens_per_sec.
    """

    def __init__(self, build_fn: Callable[[object], object], prompt_ids=None, *, batch: int = 1) -> None:
        self._build = build_fn
        self._prompt = prompt_ids
        self.batch = int(batch or 1)
        self._pipe = None
        self._state = None

    def setup(self, device) -> None:
        self._pipe = self._build(device)
        step = getattr(self._pipe, "decode_step", None)
        if not callable(step):
            raise NotTraceCapable(
                "pipeline exposes no decode_step(state); its decode is repeat-prefill — "
                "run the structural decode lever to add a cached single-token step"
            )
        prefill = getattr(self._pipe, "decode_prefill", None)
        self._state = prefill(self._prompt) if callable(prefill) else None
        if bool(getattr(self._pipe, "self_traced", False)):
            self.self_traced = True
            self.trace_path = getattr(self._pipe, "trace_path", None)

    def step(self):
        self._state = self._pipe.decode_step(self._state)
        return self._state

    def refresh_inputs(self) -> None:
        pass


class _Stage:
    """One profilable unit emit-e2e emitted: a name and a host-op-free traceable step."""

    __slots__ = ("name", "step", "self_traced", "trace_path")

    def __init__(self, name, step, self_traced=False, trace_path=None):
        self.name = name
        self.step = step
        self.self_traced = bool(self_traced)
        self.trace_path = trace_path


class PipelineStageAdapter:
    """Generic PER-STAGE perf adapter — profiles WHATEVER emit-e2e emits, not just decode.

    emit-e2e records `PIPELINE_STAGES = [...]` and, for each stage, exposes the identical
    model-agnostic contract on the pipeline object:
        <stage>_trace_setup(inputs)   host prep + pre-upload of shape-dependent constants OUTSIDE
                                      the trace (pin the variable axis, upload mask/RoPE/KV).
        <stage>_trace_step()          ONE fixed-shape, host-op-free step reading only resident
                                      buffers — this is what gets captured as a trace.

    This adapter binds every such stage so `measure_adapter` traces each one. For a pipeline
    that exposes ONLY the older single-stage decode contract (decode_step / decode_prefill)
    it synthesizes a single "decode" stage, so the legacy path is unchanged.
    A repeat-prefill pipeline (no stages, no decode_step) raises AttributeError in setup, exactly as
    before — the perf test's guard then falls back to FORWARD_WALL_MS.
    """

    def __init__(self, build_fn: Callable[[object], object], prompt_ids=None, *, batch: int = 1) -> None:
        self._build = build_fn
        self._prompt = prompt_ids
        self.batch = int(batch or 1)
        self._pipe = None
        self.stages = []

    def _inputs_dict(self):
        if self._prompt is None:
            return None
        import torch

        ids = [int(x) for x in self._prompt]
        return {"input_ids": torch.tensor(ids, dtype=torch.long).reshape(self.batch, -1)}

    def _self_prime_inputs(self, stage):
        """Model-agnostic fallback inputs for a stage whose <stage>_trace_setup needs model-specific
        tensors that generic prompt_ids can't supply (e.g. XTTS's (cond_mel, ref_wav, text, codes)).
        Ask the PIPELINE (or its module) for its OWN reference-grounded inputs via a conventional
        provider -- per-stage first, then pipeline-wide -- so the adapter never has to know the shape.
        These are the same providers the model's own self-test uses (e.g. _default_selftest_inputs,
        which replays the captured reference inputs), so they work whether the reference is HF or a
        model-local loader. Returns None when the pipeline exposes no such provider."""
        import sys as _sys

        p = self._pipe
        mod = _sys.modules.get(type(p).__module__)
        names = [
            n
            for n in (
                "%s_default_inputs" % stage if stage else "",
                "default_selftest_inputs",
                "_default_selftest_inputs",
                "default_inputs",
            )
            if n
        ]
        for owner in (p, mod):
            if owner is None:
                continue
            for nm in names:
                fn = getattr(owner, nm, None)
                if callable(fn):
                    try:
                        return fn()
                    except Exception:  # noqa: BLE001
                        continue
        return None

    def _call_with_inputs(self, fn, primary, stage=None):
        try:
            return fn(primary)
        except (AttributeError, TypeError, KeyError, IndexError, ValueError):
            pass
        _last = None
        inp = self._inputs_dict()
        if inp is not None:
            try:
                return fn(inp)
            except (AttributeError, TypeError, KeyError, IndexError, ValueError) as _e:
                _last = _e
        primed = self._self_prime_inputs(stage)
        if primed is not None:
            try:
                return fn(primed)
            except (AttributeError, TypeError, KeyError, IndexError, ValueError) as _e:
                _last = _e
        if _last is not None:
            raise _last
        return fn(primary)

    def setup(self, device) -> None:
        p = self._pipe = self._build(device)
        stages = []
        _stage_names = getattr(p, "PIPELINE_STAGES", None)
        if not _stage_names:
            import sys as _sys

            _pmod = _sys.modules.get(type(p).__module__)
            _stage_names = getattr(_pmod, "PIPELINE_STAGES", []) if _pmod else []
        for name in list(_stage_names or []):
            step = getattr(p, "%s_trace_step" % name, None)
            if not callable(step):
                continue
            setup = getattr(p, "%s_trace_setup" % name, None)
            if callable(setup):
                self._call_with_inputs(setup, None, stage=name)
            # Propagate self_traced: a pipeline that OWNS its capture must be timed natively, never
            # wrapped in a second begin_trace_capture. The decode fallback below already does this;
            # omitting it here meant declaring PIPELINE_STAGES turned a working self-traced pipeline
            # into a nested-capture TT_FATAL.
            _selft = bool(getattr(p, "%s_self_traced" % name, None) or getattr(p, "self_traced", False))
            stages.append(
                _Stage(
                    name,
                    step,
                    _selft,
                    getattr(p, "trace_path", None) if _selft else None,
                )
            )
        if stages:
            self.stages = stages
            return
        # Fallback: older single-stage decode contract wrapped as one "decode" stage.
        step = getattr(p, "decode_step", None)
        if not callable(step):
            raise NotTraceCapable(
                "pipeline exposes neither PIPELINE_STAGES trace hooks nor decode_step(state); "
                "its decode is repeat-prefill — run the structural decode lever to add a cached step"
            )
        prefill = getattr(p, "decode_prefill", None)
        box = {"state": self._call_with_inputs(prefill, self._prompt, stage="decode") if callable(prefill) else None}

        def _dstep():
            box["state"] = step(box["state"])

        if bool(getattr(p, "self_traced", False)):
            self.stages = [_Stage("decode", _dstep, True, getattr(p, "trace_path", None))]
            return
        self.stages = [_Stage("decode", _dstep)]
