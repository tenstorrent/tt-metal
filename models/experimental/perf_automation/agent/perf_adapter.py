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

from . import stage_seams as _seams


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


def headline_unit(stage_names, pipeline=None) -> str:
    """Which unit of work the headline measures: "token", "step", or "inference".

    STRUCTURE FIRST. This was decided by `"decode" in stage_name.lower()` -- a substring test on free
    text emit-e2e wrote, which is a guess wearing an observation's clothes: a pipeline whose recurring
    stage is called `generate` reads as one-pass, and one that names any stage `decode` reads as
    autoregressive whether it loops or not.

    A pipeline exposing decode_step(state) retires ONE TOKEN PER CALL by definition -- that is the
    decode contract PipelineDecodeAdapter requires and raises NotTraceCapable without. It is a fact
    about the built object, so it is checked first. Underneath it a stage-adapter pipeline, which
    exposes per-stage hooks rather than the single decode contract, is asked what one call retires.

    NOTHING HERE READS A NAME. Every question is put to the object; a stage that answers none of them
    is counted as retiring a whole inference, which is what an unstated stage is.

    Lives here rather than in trace_replay because that module imports ttnn at module scope and so
    cannot be imported, let alone tested, without a device.
    """
    # 1. WHAT THE PIPELINE SAYS IT RETIRES. PIPELINE_UNIT is the model stating its own unit, the way
    # PIPELINE_STAGES states its own stages -- the only source that cannot be wrong about a model
    # nobody anticipated. Optional: absent, the contract below answers for every pipeline in tree.
    _declared = str(getattr(pipeline, "PIPELINE_UNIT", "") or "").strip().lower()
    if not _declared and pipeline is not None:
        import sys as _sys

        _mod = _sys.modules.get(type(pipeline).__module__)
        _declared = str(getattr(_mod, "PIPELINE_UNIT", "") or "").strip().lower() if _mod else ""
    if _declared in ("token", "step", "inference"):
        return _declared
    # 2. THE DECODE CONTRACT, a fact about the built object: decode_step(state) retires one token per
    # call by definition. Every stage-adapter pipeline in tree also keeps it, so the name matches
    # below are already unreachable for them -- they remain for a pipeline that declares stages and
    # keeps no contract, which is the only case left where nothing structural has spoken.
    if pipeline is not None and callable(getattr(pipeline, "decode_step", None)):
        return "token"
    # 3. WHAT ONE CALL RETIRES, asked of the stage instead of read off its name. <stage>_trace_items()
    # is the same optional seam _Stage derives `recurring` from: a stage that states it retires exactly
    # one item per call IS the recurring one -- that is what recurring means. Absent, it states nothing
    # and is skipped, so an unstated stage can never be mistaken for a recurring one. What that item is
    # CALLED -- token, denoising step -- is not a structural fact and is not guessed; a model whose item
    # is not a token declares PIPELINE_UNIT, which is answered first.
    for _n in stage_names or []:
        _items = getattr(pipeline, _seams.hook(_n, _seams.ITEMS), None) if pipeline is not None else None
        if not callable(_items):
            continue
        try:
            if int(_items() or 0) == 1:
                return "token"
        except Exception:  # noqa: BLE001 -- a stage that cannot count says nothing; it never breaks a run
            continue
    return "inference"


def resolve_batch(pipeline, requested: int = 0) -> int:
    """How many users this pipeline serves: the request, else what the pipeline declares, else 1.

    The generated perf test used to hardcode `batch=1`, so a pipeline emit-e2e built for 8 users was
    measured serving one and its aggregate throughput under-reported eightfold. Batch is a property
    of the ARTIFACT, not of the harness, so the artifact is asked.

    `requested > 0` always wins, which is what makes a batch sweep possible without rebuilding the
    demo. Names are tried in decreasing specificity and every model that declares any of them is
    covered; a pipeline that declares none is batch 1, exactly as before.
    """
    if int(requested or 0) > 0:
        return int(requested)
    # `B` IS IN THIS LIST BECAUSE A PIPELINE USED IT AND WAS READ AS BATCH 1. Voxtral-Mini-3B declares
    # DECODE_BATCH = 8 and stores it as `self.B`, nothing else. The generated perf test resolved that
    # correctly -- it checks B -- and printed PERF_BATCH_STREAMS=8, while this function, which is what
    # the ADAPTER asks and therefore what the scorecard is built from, stopped at "max_batch". So a run
    # serving 8 users reported batch=1 and its aggregate throughput eightfold low (11.42 tok/s against
    # ~91.4), on 2026-08-15. The device work was correct throughout; only the number published was not.
    #
    # Two lists naming the same property is how they drifted. This one is the authority -- the test's
    # copy is written per-model by an agent and cannot be relied on to agree -- so it carries every
    # name any pipeline here has used.
    for attr in ("max_batch_size", "batch_size", "batch", "max_batch", "B"):
        try:
            v = int(getattr(pipeline, attr, 0) or 0)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return 1


class PipelineDecodeAdapter:
    """Generic PerfAdapter over any pipeline exposing the decode contract above.

    build_fn    device -> pipeline. Builds the pipeline EXACTLY as the demo/perf test does, on the
                measurement device (so the trace captures the real program).
    prompt_ids  small prompt fed to decode_prefill to build the initial state (ignored if the
                pipeline has no decode_prefill).
    batch       users in the batch — forwarded so trace_replay derives tokens_per_sec.
    """

    def __init__(self, build_fn: Callable[[object], object], prompt_ids=None, *, batch: int = 0) -> None:
        self._build = build_fn
        self._prompt = prompt_ids
        # 0 = ask the pipeline in setup(), once it exists. Resolving here is impossible: the pipeline
        # is not built until setup(device).
        self._requested_batch = int(batch or 0)
        self.batch = max(1, self._requested_batch)
        self._pipe = None
        self._state = None

    def setup(self, device) -> None:
        self._pipe = self._build(device)
        self.batch = resolve_batch(self._pipe, self._requested_batch)
        step = getattr(self._pipe, "decode_step", None)
        if not callable(step):
            raise NotTraceCapable(
                "pipeline exposes no decode_step(state); its decode is repeat-prefill — "
                "run the structural decode lever to add a cached single-token step"
            )
        prefill = getattr(self._pipe, "decode_prefill", None)
        # PREFILL THE WHOLE BATCH, not one sequence. This passed the bare prompt whatever the batch,
        # so a batch-8 run built single-user state and then had its throughput multiplied by 8 --
        # the same manufactured speedup PipelineStageAdapter._inputs_dict had, by a different route.
        # Replication is attempted only for batch > 1, and falls back to the raw prompt: a pipeline
        # whose decode_prefill wants a 1-D sequence must keep working exactly as before.
        self._state = self._request_batch(prefill) if callable(prefill) else None
        if bool(getattr(self._pipe, "self_traced", False)):
            self.self_traced = True
            self.trace_path = getattr(self._pipe, "trace_path", None)

    def _request_batch(self, prefill):
        """Prefill state for ALL `batch` users, falling back to the single-sequence call.

        Whether a pipeline accepts a batched prompt is settled by CALLING it, not by inspecting a
        signature: the decode contract is duck-typed across every model emit-e2e produces, so the
        only reliable test is whether the call goes through.

        When it does not, `self.batch` is CORRECTED TO 1. trace_replay reads that attribute to derive
        tokens_per_sec, so leaving it at 8 after serving one user would report an 8x aggregate that
        never happened -- the same manufactured speedup this whole change exists to remove.
        """
        if self.batch <= 1 or self._prompt is None:
            return prefill(self._prompt)
        try:
            import torch

            ids = torch.as_tensor(self._prompt).reshape(-1)
            batched = ids.unsqueeze(0).expand(self.batch, ids.numel()).contiguous()
        except Exception:  # noqa: BLE001
            self.batch = 1
            return prefill(self._prompt)
        try:
            return prefill(batched)
        except Exception:  # noqa: BLE001
            self.batch = 1
            return prefill(self._prompt)

    def step(self):
        self._state = self._pipe.decode_step(self._state)
        return self._state

    def refresh_inputs(self) -> None:
        pass


class _Stage:
    """One profilable unit emit-e2e emitted: a name and a host-op-free traceable step."""

    __slots__ = ("name", "step", "self_traced", "trace_path", "items", "recurring")

    def __init__(self, name, step, self_traced=False, trace_path=None, items=0, recurring=None):
        self.name = name
        self.step = step
        self.self_traced = bool(self_traced)
        self.trace_path = trace_path
        # HOW MANY ITEMS ONE CALL RETIRES: tokens for a prompt-consuming stage, frames for an
        # encoder, 1 for a recurring step. The compute ceiling is 2 x params x THIS, so a stage that
        # cannot state it is priced at one item -- which is right for a recurring stage and is why
        # the audio encoder's compute roof read 0.041 ms against a measurement in the tens of ms.
        # 0 means "not stated", which the reader turns into 1; it is not a claim of one.
        self.items = max(0, int(items or 0))
        # THE STAGE THE HEADLINE IS PER, derived rather than matched. A stage that retires exactly
        # one item per call IS the recurring one -- that is what recurring means -- and the reader
        # picked it by `"decode" in name.lower()` instead, so a pipeline whose loop is called
        # `generate` read as one-pass and one that names any stage `decode` read as autoregressive
        # whether it looped or not. Explicit for the legacy contract, which retires one token per
        # call by definition; None when the stage stated no count, and the caller falls back.
        self.recurring = (self.items == 1) if recurring is None else bool(recurring)


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

    def __init__(self, build_fn: Callable[[object], object], prompt_ids=None, *, batch: int = 0) -> None:
        self._build = build_fn
        self._prompt = prompt_ids
        # 0 = ask the pipeline in setup(); see resolve_batch.
        self._requested_batch = int(batch or 0)
        self.batch = max(1, self._requested_batch)
        self._pipe = None
        self.stages = []

    def _inputs_dict(self):
        """The prompt, REPLICATED to `batch` rows -- every user gets the FULL sequence.

        This reshaped instead: `torch.tensor(ids).reshape(self.batch, -1)`, which SPLITS one prompt
        across the rows. At batch 8 a 128-token prompt became eight sequences of 16, so ISL silently
        fell to a eighth of what the test declared while the scorecard still multiplied throughput by
        8 -- a batch speedup manufactured out of a shorter sequence. It also raised outright whenever
        ISL was not divisible by batch.

        Batch B means B users each doing the DECLARED work, so the row count is the only thing that
        changes with B; the sequence length does not.
        """
        if self._prompt is None:
            return None
        import torch

        ids = torch.tensor([int(x) for x in self._prompt], dtype=torch.long).reshape(-1)
        return {"input_ids": ids.unsqueeze(0).expand(self.batch, ids.numel()).contiguous()}

    def _call_with_inputs(self, fn, primary):
        try:
            return fn(primary)
        except (AttributeError, TypeError, KeyError, IndexError):
            inp = self._inputs_dict()
            if inp is None:
                raise
            return fn(inp)

    def setup(self, device) -> None:
        p = self._pipe = self._build(device)
        self.batch = resolve_batch(p, self._requested_batch)
        # SAY WHEN THE ANSWER IS A GUESS. `batch` scales the aggregate throughput the scorecard
        # publishes, so an unresolved one does not look like a missing value -- it looks like a
        # single-user run. On 2026-08-15 that reported 11.42 tok/s for a pipeline serving 8, and
        # nothing in the log hinted the number had been defaulted rather than measured. If the
        # pipeline declares none of the names resolve_batch knows, the operator should hear it.
        if self._requested_batch <= 0 and self.batch == 1:
            print(
                "PERF_BATCH_UNRESOLVED=1 pipeline=%s declares none of "
                "(max_batch_size, batch_size, batch, max_batch, B); assuming 1 user -- aggregate "
                "throughput is reported per-user until it does" % type(p).__name__,
                flush=True,
            )
        stages = []
        _stage_names = getattr(p, "PIPELINE_STAGES", None)
        if not _stage_names:
            import sys as _sys

            _pmod = _sys.modules.get(type(p).__module__)
            _stage_names = getattr(_pmod, "PIPELINE_STAGES", []) if _pmod else []
        for name in list(_stage_names or []):
            step = getattr(p, _seams.hook(name, _seams.STEP), None)
            if not callable(step):
                # A DECLARED STAGE THAT CANNOT BE MEASURED SAYS SO. This was a bare `continue`, so a
                # stage the model listed in PIPELINE_STAGES but never gave a step hook disappeared
                # here -- out of adapter.stages, out of stage_ms, and out of the roofline, which
                # renders exactly the stages that measured. The report then showed two towers for a
                # three-tower model and nothing anywhere said one was missing. The contract warns at
                # preflight; this is the same fact at the moment it costs a row.
                print(
                    "  [perf-adapter] stage %r declared but %s is missing; it cannot be measured and "
                    "gets no row. The other stages are unaffected." % (name, _seams.hook(name, _seams.STEP)),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            setup = getattr(p, _seams.hook(name, _seams.SETUP), None)
            if callable(setup):
                # Standard emit-e2e hook: <stage>_trace_inputs() returns exactly the args this stage's
                # trace_setup takes (assembled by the pipeline from its captured reference tensors). This
                # is the model-agnostic seam -- the adapter never has to know the shape or the model. Fall
                # back to the generic None/prompt_ids path for pipelines that don't expose it (a stage
                # whose trace_setup self-derives, or a text model driven by prompt_ids).
                _tin = getattr(p, _seams.hook(name, _seams.INPUTS), None)
                # ONE STAGE THAT CANNOT PRODUCE ITS INPUTS MUST NOT COST THE OTHERS THEIR STAGE.
                # This was unguarded, so a single hook that raised took the whole adapter down and
                # every stage lost its boundary. Measured: voxtral's encode_trace_inputs torch.loads a
                # captured golden tensor, and on a tree without one it raised FileNotFoundError --
                # taking prefill and decode with it, though both derive their own inputs and needed
                # nothing from disk. The run then had no per-stage split at all rather than two of
                # three, and the roofline shared one math-fidelity peak across every stack.
                try:
                    if callable(_tin):
                        setup(_tin())
                    else:
                        self._call_with_inputs(setup, None)
                except Exception as exc:  # noqa: BLE001
                    print(
                        "  [perf-adapter] stage %r cannot prepare its own inputs (%s: %s); it gets no "
                        "stage, the others are unaffected" % (name, type(exc).__name__, str(exc)[:120]),
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
            # Propagate self_traced: a pipeline that OWNS its capture must be timed natively, never
            # wrapped in a second begin_trace_capture. The decode fallback below already does this;
            # omitting it here meant declaring PIPELINE_STAGES turned a working self-traced pipeline
            # into a nested-capture TT_FATAL.
            _selft = bool(getattr(p, "%s_self_traced" % name, None) or getattr(p, "self_traced", False))
            # <stage>_trace_items() -- the same optional, model-agnostic seam as _trace_inputs, for
            # the one number the compute ceiling needs and nothing else could supply. Only the
            # pipeline knows what one call of this stage retires: an encoder's frame count is not the
            # prompt length and is not derivable from the byte model, which is why every stage but
            # the prompt-consuming one was priced at a single item. Absent, nothing changes.
            _n = 0
            _items_fn = getattr(p, _seams.hook(name, _seams.ITEMS), None)
            if callable(_items_fn):
                try:
                    _n = max(0, int(_items_fn() or 0))
                except Exception:  # noqa: BLE001 -- an unstated count is 0, never a broken run
                    _n = 0
            stages.append(
                _Stage(
                    name,
                    step,
                    _selft,
                    getattr(p, "trace_path", None) if _selft else None,
                    _n,
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
        box = {"state": self._call_with_inputs(prefill, self._prompt) if callable(prefill) else None}

        def _dstep():
            box["state"] = step(box["state"])

        # ONE TOKEN PER CALL, BY DEFINITION -- that is what decode_step(state) is, and reaching here
        # means the pipeline keeps that contract. Stated rather than left to a fallback, so the
        # legacy path feeds the same derived machinery the declared path does: `recurring` selects
        # the headline stage, `items` prices the compute ceiling. This is the ONLY place on the
        # measurement path that still names a stage, and it names it because the contract it just
        # verified is spelled that way.
        if bool(getattr(p, "self_traced", False)):
            self.stages = [_Stage("decode", _dstep, True, getattr(p, "trace_path", None), 1, True)]
            return
        self.stages = [_Stage("decode", _dstep, items=1, recurring=True)]
