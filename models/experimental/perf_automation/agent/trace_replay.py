# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic trace-replay latency measurement for optimize/perf.

`measure_adapter(adapter, device)` drives a PipelineStageAdapter (see perf_adapter.py): for EACH stage
emit-e2e emitted (adapter.stages, from the pipeline's PIPELINE_STAGES) it captures one steady-state,
host-op-free step as a device trace and replays it under trace on a single command queue. It prints,
per stage:

    TRACE_STAGE_MS[<stage>]=<float> path=trace+1cq

plus the headline clean, GPU-comparable wall the harness parses:

    TRACE_PER_TOKEN_MS=<float>     (the AR/decode stage if present, else the whole-pipeline sum)

which the harness (agent/tracy_tool.py + cc_optimize/perf_mcp.py) reads as the `trace` metric source
(vs the `eager` fallback FORWARD_WALL_MS). This is the companion of perf_adapter.py: the adapter is
the shell (setup + per-stage step), this module is the engine (warmup -> capture -> timed replay ->
emit the numbers). A legacy single-step adapter (PipelineDecodeAdapter, no .stages) is wrapped as one
"decode" stage, so the old path is unchanged.

Measurement is trace+1cq end to end. Caller contract (the generated perf test): call inside the
`if _PERF_TRACE:` block, on the SAME device the test opened WITH `trace_region_size`. Any failure here
(notably a repeat-prefill pipeline with no `decode_step`, which raises in `adapter.setup`) propagates
out; the perf test's guard catches it, prints `TRACE_REPLAY_SKIPPED=...`, and falls back to
FORWARD_WALL_MS.
"""

from __future__ import annotations

import os
import time

import ttnn

# Warmup compiles kernels + populates RoPE/mask/KV caches (trace capture can neither compile nor
# upload from host); replay iters are averaged for a stable per-token number. Both env-tunable.
_WARMUP_ITERS = max(1, int(os.environ.get("TT_TRACE_WARMUP_ITERS", "3")))
_REPLAY_ITERS = max(1, int(os.environ.get("TT_TRACE_REPLAY_ITERS", "16")))


def _capture_step_trace(device, step):
    """Warm up, then capture exactly one host-op-free, fixed-shape step as a trace on cq0."""
    for _ in range(_WARMUP_ITERS):
        step()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    step()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    return tid


def _replay_1cq(device, tid, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


# A REPLAYED TRACE DISPATCHES ONE OP. An eager pass dispatches one per ttnn call in the model --
# 3,564 of them for a 48-layer gemma-3 prefill. Nothing else about the two paths differs by three
# orders of magnitude, so the dispatch count is the signal, and anything above this is eager.
_TRACED_OP_DISPATCH_MAX = int(os.environ.get("TT_TRACE_DISPATCH_MAX", "32"))


def _count_op_dispatches(fn):
    """Run `fn` once; return (dispatches, working_set_bytes) or (None, 0) if the hook did not install.

    WHAT A STAGE ACTUALLY READS, MEASURED. The roofline needs the bytes one unit of work streams, and
    until now nothing measured it. Every route was a guess about the checkpoint: the tower name list
    (`audio_tower|vision_tower|...`), or the stage_roots section map -- which was tried and dropped
    lm_head on every untied model. _stage_roofs already said why the profile could not settle it:
    "_top_ops keys on (op_code, shape, memory) and records nothing about which phase an op ran in",
    and summary._stage_measured_bytes consequently returns 0 on every real profile, so the
    "observed-first" line in _bytes_for has always taken its fallback branch.

    But the hook that counts dispatches sees EVERY op the stage runs, with its arguments -- and
    trace_replay already runs each stage in isolation. So the read set is right here: the distinct
    DEVICE tensors this stage's ops touched. That is an observation of this stage on this build, not
    an inference from names, and it needs no new pass -- it rides the warmup call that was happening
    anyway.

    DISTINCT, BY IDENTITY. A weight used by two ops is one resident tensor, and the quantity this
    feeds -- params x width -- counts each weight once, so the working set is the comparable figure.
    Counting every use instead would be a different (larger) number and would not be comparable to
    the ceiling it is meant to replace.

    Host tensors are excluded by asking the tensor, via the census's own _on_device: voxtral keeps an
    fp32 host copy of its weights alive while the device holds bf16, and counting those reported
    29.96 GB for an 11.3 GB device footprint.

    Best-effort throughout. A tensor that raises when inspected is skipped rather than ending the
    step, because this is instrumentation riding a measurement: it may return less than the truth,
    never an exception.
    """
    try:
        from ttnn import decorators as _dec

        target = _dec.Operation
        orig = target.__call__
    except Exception:  # noqa: BLE001
        fn()
        return None, 0
    try:
        from .weight_census import _on_device as _dev, bytes_per_elem as _bpe, dtype_name as _dtn
    except Exception:  # noqa: BLE001
        _dev = None
    n = [0]
    seen: dict = {}

    def _note(x):
        if _dev is None:
            return
        try:
            if not _dev(x):
                return
            key = id(x)
            if key in seen:
                return
            shape = getattr(x, "shape", None) or getattr(x, "padded_shape", None)
            dt = getattr(x, "dtype", None)
            if shape is None or dt is None:
                return
            ne = 1
            for d in tuple(shape):
                ne *= int(d)
            if ne > 0:
                seen[key] = ne * _bpe(_dtn(dt))
        except Exception:  # noqa: BLE001
            return

    def counting(self, *a, **kw):
        n[0] += 1
        for _x in a:
            _note(_x)
        for _x in kw.values():
            _note(_x)
        return orig(self, *a, **kw)

    target.__call__ = counting
    try:
        fn()
    finally:
        target.__call__ = orig
    return n[0], int(sum(seen.values()))


def _measure_native(device, stage):
    """Time a SELF-TRACED stage: the pipeline owns its trace capture (persistent-buffer / vLLM-style
    decode, e.g. GLM's decode(enable_trace=True)), so we must NOT begin_trace_capture around it --
    doing so raises "Writes/Reads are not supported during trace capture" because the step does
    host<->device I/O + execute_trace internally. Instead warm it (the pipeline lazily captures its own
    trace on the first call) and time steady-state replays.

    THE PATH IS OBSERVED, NOT ASSUMED. This returned "trace+1cq" for every self-traced stage, on the
    strength of the stage having DECLARED itself self-traced -- but declaring the capability is not
    the same as exercising it. gemma-3's prefill declares it and then falls back to eager, because
    `can_enable_trace` consults an allow-list that the model config sets EMPTY for every device; the
    harness recorded 91.33 ms as `path=trace+1cq` while the model logged `trace: False` on every call.
    Half that number is Python dispatch, and the roofline then graded it against a band that assumes
    a traced measurement.

    So the last warmup is counted rather than merely run. `stage.trace_path()` still wins when a
    pipeline reports its own path; the count is what decides when nothing does."""
    for _ in range(max(0, _WARMUP_ITERS - 1)):
        stage.step()
    dispatched, _ws_bytes = _count_op_dispatches(stage.step)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(_REPLAY_ITERS):
        stage.step()
    ttnn.synchronize_device(device)
    per_s = (time.perf_counter() - t0) / _REPLAY_ITERS
    tp = getattr(stage, "trace_path", None)
    if callable(tp):
        try:
            path = str(tp())
        except Exception:  # noqa: BLE001
            path = "unknown"
    elif dispatched is None:
        # The counter could not be installed, so nothing was observed. "unknown" is the honest label:
        # it withholds the verdict downstream instead of asserting a path on no evidence.
        path = "unknown"
    else:
        path = "trace+1cq" if dispatched <= _TRACED_OP_DISPATCH_MAX else "eager"
    if dispatched is not None:
        print("TRACE_STAGE_OPS[%s]=%d path=%s" % (stage.name, dispatched, path), flush=True)
    # THE BYTES THIS STAGE READ, beside the time that read them. The one quantity the roofline could
    # never measure -- every other route infers it from the checkpoint and a naming convention.
    # Silent when zero: nothing observed is not "this stage reads nothing", and a stage with no
    # reported bytes keeps the existing estimate rather than being given a ceiling of infinity.
    if _ws_bytes > 0:
        print("TRACE_STAGE_BYTES[%s]=%d" % (stage.name, _ws_bytes), flush=True)
    return per_s * 1000.0, path


def _measure_stage(device, stage):
    """Capture stage.step as a trace, replay it on a single command queue, return (ms, path)."""
    if getattr(stage, "self_traced", False):
        return _measure_native(device, stage)
    tid = _capture_step_trace(device, stage.step)
    try:
        per_s = _replay_1cq(device, tid, _REPLAY_ITERS)
        path = "trace+1cq"
    finally:
        try:
            ttnn.release_trace(device, tid)
        except Exception:
            pass
    return per_s * 1000.0, path


class _LegacyStage:
    """Wrap a legacy single-step adapter (PipelineDecodeAdapter: .step()) as a stage."""

    def __init__(self, adapter):
        # The legacy adapter IS the decode contract -- PipelineDecodeAdapter raises NotTraceCapable
        # without decode_step(state) -- so one token per call is a fact here, not a guess.
        self.name = "decode"
        self.items = 1
        self.recurring = True
        self.step = adapter.step
        self.self_traced = bool(getattr(adapter, "self_traced", False))
        self.trace_path = getattr(adapter, "trace_path", None)


def _checkpoint_for_census(pipeline=None):
    """Where this model's weights are, for the census's checkpoint-name vocabulary. None if unknown.

    FOUND FROM THE PIPELINE ITSELF. The obvious source is an env var naming the model root, and the
    harness does not export one -- checked against a live run's whole process tree: no
    PERF_MCP_MODEL_ROOT anywhere in it. A fix resting on that would have been inert and looked
    installed.

    The object being measured knows where it lives: its class's module file sits inside the model's
    own directory, and walking up from there reaches the root whose source names the hub repo. The
    walk stops at a repository boundary so this can never wander up and scan a whole monorepo.
    """
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    cands = []
    root = (_os.environ.get("PERF_MCP_MODEL_ROOT") or _os.environ.get("TT_PERF_MODEL_ROOT") or "").strip()
    if root:
        cands.append(_Path(root))
    _mod = _sys.modules.get(type(pipeline).__module__) if pipeline is not None else None
    _f = getattr(_mod, "__file__", None) if _mod else None
    if _f:
        for _par in list(_Path(_f).resolve().parents)[:4]:
            if (_par / ".git").exists():
                break  # the repository root: everything above the model, and far too big to scan
            cands.append(_par)
    for _c in cands:
        try:
            from .stack_survey import model_id_from_source

            _mid = str(model_id_from_source(_c) or "").strip()
            if _mid:
                return _mid
        except Exception:  # noqa: BLE001
            continue
    return None


def measure_adapter(adapter, device) -> float:
    """Trace-replay per-stage latency for WHATEVER the pipeline emitted. Traces every stage in
    adapter.stages; prints TRACE_STAGE_MS[<stage>] per stage and TRACE_PER_TOKEN_MS (the AR/decode
    stage if present, else the whole-pipeline sum). Returns that headline ms.

    Raises (propagating to the perf test's guard) if the pipeline has no traceable step at all."""
    # setup() builds the pipeline + binds stages. A pipeline that GENUINELY cannot trace (repeat-prefill /
    # no decode_step) raises NotTraceCapable — the ONE legitimate eager terminal. Emit a STABLE marker so
    # the generation-time validator can tell this apart from an incidental setup bug (which it must keep
    # correcting) and accept the eager path, then re-raise so the perf test guard falls to FORWARD_WALL_MS.
    try:
        from .perf_adapter import NotTraceCapable
    except Exception:  # pragma: no cover - perf_adapter always importable alongside this module
        NotTraceCapable = ()
    try:
        adapter.setup(device)
        # THE ONLY MOMENT THE SERVED WIDTH EXISTS. setup() has built the model and the loader has
        # decided each tensor's dtype; the checkpoint records what was on disk, and every byte rule
        # downstream is a PREDICTION of what happened here. Walk it once and state it. Best-effort:
        # a census that cannot run leaves the ceiling on its existing fallbacks rather than failing
        # a measurement run.
        try:
            from .weight_census import census as _census, marker as _cmarker, sections_marker as _smarker

            # WITH THE CHECKPOINT, so the split is recorded in the vocabulary stage_roots speaks.
            #
            # census() records a subtree's bytes under TWO names: the attribute it was reached
            # through, and the checkpoint section it came from. Only the second can be looked up by
            # a stage_roots entry -- and the checkpoint argument was never passed, so only the first
            # existed. Run 10, 2026-08-19, published a complete device_section_bytes keyed
            # enc_a / enc_b / lm_layers / lm_head / embed / kv / mlp / attn, with no `audio_tower`
            # and no `language_model` in it. The measured split was present, correct, and unusable:
            # every stage fell back to apportioning by the checkpoint's disk ratio, which is the
            # single input the census exists to replace.
            _c = _census(
                getattr(adapter, "_pipe", None) or adapter,
                scope="pipeline",
                checkpoint=_checkpoint_for_census(getattr(adapter, "_pipe", None)),
            )
            if _c.get("weight_bytes"):
                print(_cmarker(_c), flush=True)
                # The split, measured in the same walk. Printed beside the total rather than derived
                # later from the checkpoint, which is the only other place a split was available and
                # states disk precision, not served precision.
                # WHICH RESIDENT WEIGHTS ARE GATHERED, from the only thing that knows. The census
                # sums every weight and the ceiling assumes each is streamed once per unit -- true
                # of a matmul weight, false of a lookup table, and nothing in the checkpoint tells
                # them apart (voxtral's embed_tokens and lm_head are both [131072, 3072]). Optional:
                # a pipeline that returns nothing changes no number.
                try:
                    _gw = getattr(getattr(adapter, "_pipe", None) or adapter, "gathered_weight_bytes", None)
                    _g = _gw() if callable(_gw) else None
                    if isinstance(_g, dict) and _g:
                        _pairs = ",".join(
                            "%s:%d" % (k, int(v))
                            for k, v in _g.items()
                            if str(k).strip() and "," not in str(k) and ":" not in str(k) and int(v) > 0
                        )
                        if _pairs:
                            print("TRACE_GATHERED_WEIGHTS=%s" % _pairs, flush=True)
                except Exception:  # noqa: BLE001 -- an unstated split is the old behaviour
                    pass
                _sm = _smarker(_c)
                if _sm:
                    print(_sm, flush=True)
        except Exception:  # noqa: BLE001
            pass
    except NotTraceCapable as exc:
        print("TRACE_NOT_TRACE_CAPABLE=1", flush=True)
        print("TRACE_REPLAY_SKIPPED=%r" % (exc,), flush=True)
        raise

    # Report the topology the trace ACTUALLY ran on (from the mesh device the perf test opened), so the
    # full-pipeline scorecard reflects the REAL mesh (DP x TP) instead of falling back to its default.
    # A >1-chip mesh means the pipeline ran tensor-parallel (sharded). Emitted here, in the tool, so it
    # is correct for EVERY generated perf test without relying on the test to print it.
    def _mesh_dims(dev):
        s = getattr(dev, "shape", None)
        if s is not None:
            try:
                d = [int(x) for x in tuple(s)]
                if len(d) >= 2:
                    return d[0], d[1]
                if len(d) == 1:
                    return 1, d[0]
            except Exception:  # noqa: BLE001
                pass
            if hasattr(s, "num_rows") and hasattr(s, "num_cols"):
                return int(s.num_rows), int(s.num_cols)
        return None

    _dims = _mesh_dims(device)
    if _dims is None:
        try:
            from .perf_adapter import resolve_mesh_shape

            _dims = tuple(int(x) for x in resolve_mesh_shape(1, 1))
        except Exception:  # noqa: BLE001
            _dims = (1, 1)
    _dp, _tp = int(_dims[0]), int(_dims[1])
    print("DP=%d TP=%d shard_active=%s" % (_dp, _tp, bool(_dp * _tp > 1)), flush=True)

    stages = list(getattr(adapter, "stages", None) or [])
    if not stages:
        # Legacy PipelineDecodeAdapter (exposes .step(), no .stages): one decode stage.
        if not callable(getattr(adapter, "step", None)):
            raise AttributeError("adapter exposes neither .stages nor a callable .step()")
        stages = [_LegacyStage(adapter)]

    results = []
    for st in stages:
        ms, path = _measure_stage(device, st)
        results.append((st.name, ms, path))
        print("TRACE_STAGE_MS[%s]=%.4f path=%s" % (st.name, ms, path), flush=True)
        # BESIDE THE TIME THAT MEASURED IT. The compute ceiling is 2 x params x items-in-the-unit,
        # and the run recorded that number for exactly one stage -- the parser wrote it to the key
        # "prefill", by hand -- so every other stage was priced at one item however much work it did.
        # Printed per stage, from what the pipeline declares, so a third tower can have a real
        # arithmetic ceiling instead of a placeholder. Silent when unstated: 1 stays the fallback.
        _n = int(getattr(st, "items", 0) or 0)
        if _n > 0:
            print("TRACE_STAGE_ITEMS[%s]=%d" % (st.name, _n), flush=True)

    pipeline_ms = sum(ms for _, ms, _ in results)
    # THE UNIT IS A STRUCTURAL FACT, so read the STRUCTURE, not a name. This matched
    # `"decode" in stage_name` -- a substring test on free text emit-e2e wrote -- which is a guess
    # wearing an observation's clothes: a pipeline whose recurring stage is called `generate` reads
    # as one-pass, and one that names any stage `decode` reads as autoregressive whether it loops or
    # not. The decode CONTRACT is the real signal: a pipeline exposing decode_step(state) retires one
    # token per call by definition, which is what PipelineDecodeAdapter already requires and raises
    # NotTraceCapable without. The name match stays BELOW it, for stage-adapter pipelines that expose
    # per-stage hooks rather than the single decode contract.
    from models.experimental.perf_automation.agent.perf_adapter import headline_unit as _hu

    _unit = _hu([r[0] for r in results], getattr(adapter, "_pipe", None))
    # THE RECURRING STAGE, AS THE STAGE ITSELF REPORTED IT. `recurring` is set from the item count
    # the pipeline declares (one item per call is what recurring means) and unconditionally for the
    # legacy decode contract. The name match below is what this used to do first, and it is a guess
    # in both directions: a loop called `generate` read as one-pass, and any stage called `decode`
    # read as the recurring one whether it looped or not.
    _rec = {st.name for st in stages if getattr(st, "recurring", False)}
    decode = next((r for r in results if r[0] in _rec), None)
    if decode is None:
        decode = next((r for r in results if "decode" in r[0].lower()), None)
    if _unit == "token" and decode is None and results:
        decode = results[-1]
    # WHICH UNIT OF WORK THE HEADLINE MEASURES. This selection already knew -- a decode stage means
    # the number is per TOKEN, no decode stage means it is one whole pipeline pass -- but it never
    # said so, and the roofline band on the other end assumed "token" unconditionally. A step-unit
    # (diffusion) or inference-unit (classifier) model therefore had its per-unit ceiling scored
    # against whatever this printed, silently: the same per-token-vs-per-profile unit mix that once
    # made every module read ABOVE_BAND, on a different axis. Name it here, at the only place that
    # can know, so the band can refuse a mismatch instead of guessing.
    step = next((r for r in results if any(k in r[0].lower() for k in ("step", "denoise", "diffus"))), None)
    if decode:
        headline_ms, headline_path, headline_unit = decode[1], decode[2], "token"
    elif step:
        headline_ms, headline_path, headline_unit = step[1], step[2], "step"
    else:
        headline_ms, headline_path, headline_unit = pipeline_ms, "trace+pipeline", "inference"
    batch = int(getattr(adapter, "batch", 1) or 1)
    per_s = headline_ms / 1000.0
    tokens_per_sec = (batch / per_s) if per_s > 0 else 0.0
    # THE line the harness parses (tracy_tool.py:_PER_TOKEN_RE / perf_mcp.py). Keep the name verbatim.
    print("TRACE_PER_TOKEN_MS=%.4f" % headline_ms, flush=True)
    print("TRACE_HEADLINE_UNIT=%s" % headline_unit, flush=True)
    print(
        "TRACE_PIPELINE_MS=%.4f TRACE_STAGES=%d%s"
        % (pipeline_ms, len(results), "" if decode else " (no decode stage: per-token=pipeline sum)"),
        flush=True,
    )
    print(
        "TRACE_REPLAY_PATH=%s TRACE_TOKENS_PER_SEC=%.2f batch=%d warmup=%d iters=%d"
        % (headline_path, tokens_per_sec, batch, _WARMUP_ITERS, _REPLAY_ITERS),
        flush=True,
    )
    return headline_ms
