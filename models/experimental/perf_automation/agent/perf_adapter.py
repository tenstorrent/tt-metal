# SPDX-License-Identifier: Apache-2.0
"""One GENERIC PerfAdapter for trace-replay per-token measurement — valid for ANY model.

`measure_adapter` (agent/trace_replay.py) needs a PerfAdapter (setup/step/refresh_inputs, + optional
write_inputs). Rather than a hand-written adapter per model, this module ships a SINGLE adapter that
wraps any pipeline conforming to a tiny standard DECODE CONTRACT — so the only per-model artifact is
the pipeline's own `decode_step`, which the structural decode lever (or emit-e2e) produces; the
adapter itself is model-agnostic and lives here once.

DECODE CONTRACT (duck-typed on the built pipeline object):
    decode_step(state) -> state          REQUIRED. Exactly one steady-state decode token: reads
                                          persistent on-device buffers, samples on-device, writes the
                                          next id back into `state`. NO host reads (to_torch/.item()).
    decode_prefill(input_ids) -> state    OPTIONAL. Process the prompt once, return the initial cache/
                                          state. If absent, `state` starts as None (fixed-input loop).
    decode_write_inputs(state) -> None    OPTIONAL. Stage the next step's inputs (issued on CQ1). Its
                                          presence flips measure_adapter's auto mode into the 2CQ path.

A pipeline WITHOUT decode_step (repeat-prefill / host-argmax decode) raises AttributeError in setup;
the perf test's guard then falls back to FORWARD_WALL_MS and the detector reports 'repeat_prefill'.
"""

from __future__ import annotations

from typing import Callable


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
            raise AttributeError(
                "pipeline exposes no decode_step(state); its decode is repeat-prefill — "
                "run the structural decode lever to add a cached single-token step"
            )
        prefill = getattr(self._pipe, "decode_prefill", None)
        self._state = prefill(self._prompt) if callable(prefill) else None
        wi = getattr(self._pipe, "decode_write_inputs", None)
        if callable(wi):
            self.write_inputs = lambda: wi(self._state)

    def step(self):
        self._state = self._pipe.decode_step(self._state)
        return self._state

    def refresh_inputs(self) -> None:
        pass


class _Stage:
    """One profilable unit emit-e2e emitted: a name, a host-op-free traceable step, and an
    optional CQ1 input-staging hook (its presence flips that stage into the 2CQ path)."""

    __slots__ = ("name", "step", "write")

    def __init__(self, name, step, write=None):
        self.name = name
        self.step = step
        self.write = write


class PipelineStageAdapter:
    """Generic PER-STAGE perf adapter — profiles WHATEVER emit-e2e emits, not just decode.

    emit-e2e records `PIPELINE_STAGES = [...]` and, for each stage, exposes the identical
    model-agnostic contract on the pipeline object:
        <stage>_trace_setup(inputs)   host prep + pre-upload of shape-dependent constants OUTSIDE
                                      the trace (pin the variable axis, upload mask/RoPE/KV).
        <stage>_trace_step()          ONE fixed-shape, host-op-free step reading only resident
                                      buffers — this is what gets captured as a trace.
        <stage>_write_inputs()        stage the next input on CQ1 — presence flips 2CQ for the stage.

    This adapter binds every such stage so `measure_adapter` traces (+2CQ) each one. For a pipeline
    that exposes ONLY the older single-stage decode contract (decode_step / decode_prefill /
    decode_write_inputs) it synthesizes a single "decode" stage, so the legacy path is unchanged.
    A repeat-prefill pipeline (no stages, no decode_step) raises AttributeError in setup, exactly as
    before — the perf test's guard then falls back to FORWARD_WALL_MS.
    """

    def __init__(self, build_fn: Callable[[object], object], prompt_ids=None, *, batch: int = 1) -> None:
        self._build = build_fn
        self._prompt = prompt_ids
        self.batch = int(batch or 1)
        self._pipe = None
        self.stages = []

    def setup(self, device) -> None:
        p = self._pipe = self._build(device)
        stages = []
        for name in list(getattr(p, "PIPELINE_STAGES", []) or []):
            step = getattr(p, "%s_trace_step" % name, None)
            if not callable(step):
                continue
            setup = getattr(p, "%s_trace_setup" % name, None)
            if callable(setup):
                setup(None)  # host prep + pre-upload, OUTSIDE the trace
            write = getattr(p, "%s_write_inputs" % name, None)
            stages.append(_Stage(name, step, write if callable(write) else None))
        if stages:
            self.stages = stages
            return
        # Fallback: older single-stage decode contract wrapped as one "decode" stage.
        step = getattr(p, "decode_step", None)
        if not callable(step):
            raise AttributeError(
                "pipeline exposes neither PIPELINE_STAGES trace hooks nor decode_step(state); "
                "its decode is repeat-prefill — run the structural decode lever to add a cached step"
            )
        prefill = getattr(p, "decode_prefill", None)
        box = {"state": prefill(self._prompt) if callable(prefill) else None}

        def _dstep():
            box["state"] = step(box["state"])

        wi = getattr(p, "decode_write_inputs", None)
        _dwrite = (lambda: wi(box["state"])) if callable(wi) else None
        self.stages = [_Stage("decode", _dstep, _dwrite)]
