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
    decode_input_buffer(state) -> Tensor  OPTIONAL (preferred, model+hardware agnostic). Return the
                                          PERSISTENT on-device input tensor that decode_step reads. The
                                          adapter pins it and stages every step's write in-place into
                                          THAT tensor on CQ1 (copy_host_to_device_tensor), so the address
                                          the captured trace baked in never goes stale across candidates.
                                          The pipeline owns the buffer (it alone knows the correct
                                          shape/dtype/layout/sharding for this model on this device); the
                                          adapter only pins + writes it, so one code path covers every
                                          model and board. Its presence flips auto mode into 2CQ.
    decode_write_inputs(state) -> None    OPTIONAL (fallback). Model-authored staging of the next step's
                                          inputs on CQ1. Used only when decode_input_buffer is absent.
                                          Also flips auto mode into 2CQ.
    self_traced = True                    OPTIONAL (class/instance attr). Declares that the pipeline OWNS
                                          its trace capture + CQ1 input-staging internally (persistent-
                                          buffer / vLLM-style decode -- e.g. GLM's decode(enable_trace=True)).
                                          Such a decode_step does host<->device I/O + execute_trace inside
                                          itself, so measure_adapter must NOT begin_trace_capture around it;
                                          it TIMES the native step instead (see trace_replay._measure_native).
                                          The purity rule above is waived and the tool does no CQ1 staging.
    trace_path() -> str                   OPTIONAL, only meaningful with self_traced. Returns the real replay
                                          path ("trace+2cq"/"trace+1cq") the pipeline actually took, so the
                                          headline honestly reflects whether CQ1 overlap engaged.

A pipeline WITHOUT decode_step (repeat-prefill / host-argmax decode) raises AttributeError in setup;
the perf test's guard then falls back to FORWARD_WALL_MS and the detector reports 'repeat_prefill'.
"""

from __future__ import annotations

import os
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
    try:
        r = int(os.environ.get("TT_PERF_MESH_ROWS", ""))
        c = int(os.environ.get("TT_PERF_MESH_COLS", ""))
        if r >= 1 and c >= 1:
            return r, c
    except (TypeError, ValueError):
        pass
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
            return
        if self._bind_persistent_write(self._state):
            return
        wi = getattr(self._pipe, "decode_write_inputs", None)
        if callable(wi):
            self.write_inputs = lambda: wi(self._state)

    def _bind_persistent_write(self, state) -> bool:
        buf_fn = getattr(self._pipe, "decode_input_buffer", None)
        if not callable(buf_fn):
            return False
        try:
            import ttnn

            buf = buf_fn(state)
            if buf is None:
                return False
            host_seed = ttnn.from_device(buf)

            def _write():
                ttnn.copy_host_to_device_tensor(host_seed, buf, cq_id=1)

            self.write_inputs = _write
            return True
        except Exception:  # noqa: BLE001
            return False

    def step(self):
        self._state = self._pipe.decode_step(self._state)
        return self._state

    def refresh_inputs(self) -> None:
        pass


class _Stage:
    """One profilable unit emit-e2e emitted: a name, a host-op-free traceable step, and an
    optional CQ1 input-staging hook (its presence flips that stage into the 2CQ path)."""

    __slots__ = ("name", "step", "write", "self_traced", "trace_path")

    def __init__(self, name, step, write=None, self_traced=False, trace_path=None):
        self.name = name
        self.step = step
        self.write = write
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
            raise NotTraceCapable(
                "pipeline exposes neither PIPELINE_STAGES trace hooks nor decode_step(state); "
                "its decode is repeat-prefill — run the structural decode lever to add a cached step"
            )
        prefill = getattr(p, "decode_prefill", None)
        box = {"state": prefill(self._prompt) if callable(prefill) else None}

        def _dstep():
            box["state"] = step(box["state"])

        if bool(getattr(p, "self_traced", False)):
            self.stages = [_Stage("decode", _dstep, None, True, getattr(p, "trace_path", None))]
            return
        wi = getattr(p, "decode_write_inputs", None)
        _dwrite = (lambda: wi(box["state"])) if callable(wi) else None
        self.stages = [_Stage("decode", _dstep, _dwrite)]
