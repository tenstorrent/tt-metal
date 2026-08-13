# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERF gate for Call 1 `text_generation` — prompt -> continuation.

Built and run EXACTLY as `demo/demo_text_generation.py` does: the demo's own
device open (`selftest_device.open_selftest_device`, which is what this model
SELF-OPENS with — one chip, one command queue, a trace region), the module-level
`build_pipeline(device, layers=...)` factory from `tt/pipeline.py`, and
`run_generate(stage_inputs(prompt))` as the forward. The demo's HF golden /
comp_pcc tail is deliberately absent: this file measures, it does not gate
correctness.

Everything runs IN-PROCESS so tracy sees every device op.
"""
from __future__ import annotations

import os
import time

import torch

# tt/pipeline.py repairs sys.path so `import ttnn` resolves to the BUILT package
# rather than the same-named source directory, so it is imported first.
from models.demos.voxtral_tts_backbone.tt.pipeline import _pad_to_tile, build_pipeline  # noqa: E402
from models.demos.voxtral_tts_backbone.selftest_device import (  # noqa: E402
    L1_SMALL_SIZE,
    TRACE_REGION_SIZE,
    close_selftest_device,
    open_selftest_device,
)

import ttnn  # noqa: E402

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS. 128 in / 128 out is the industry-standard
# short-context benchmark point; both are env-overridable and both are echoed below so a
# reader never has to guess what was measured. PERF_OSL_TOKENS is the ONE decode-horizon
# unit: it is what the loop runs AND what this file prints.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# BATCH BELONGS TO THE MODEL. 0 = ask the pipeline (resolve_batch), any positive value overrides.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream does
# not overflow the profiler; the tool sends that number for tracy runs. The variable being ABSENT
# means ALL LAYERS -- so PERF_LAYERS is None then, which is `build_pipeline`'s own all-layers value.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# THE HEAVY AXIS FOR THIS MODEL IS TOKENS. `capacity` (KV / trace capacity C) pins the sequence
# axis of every profiled forward: `prefill_trace_setup` stages at exactly seq_len=C, the resident
# KV is C deep, and `decode_horizon` bounds the loop by C - prompt_len. A generation of ISL in /
# OSL out needs room for BOTH, so C is derived from those DECLARED numbers -- a small,
# representative 128 + 128 -- and NEVER from the checkpoint's production shape
# (max_position_embeddings=128000, which under tracy would never finish). Env-overridable.
_cap = (os.environ.get("TT_PERF_CAPACITY") or "").strip()
PERF_CAPACITY = int(_cap) if (_cap.isdigit() and int(_cap) > 0) else _pad_to_tile(PERF_ISL_TOKENS + PERF_OSL_TOKENS)

# TOPOLOGY. --devices/--mesh are planned by the tool and exported as TT_PERF_MESH_ROWS/COLS;
# resolve_mesh_shape honours them, defaulting to THIS source's own shape (open_selftest_device
# opens a single chip), so an unset env behaves exactly as the demo does.
from models.experimental.perf_automation.agent.perf_adapter import (  # noqa: E402
    resolve_batch,
    resolve_mesh_shape,
)

_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# The source ALWAYS opens with a trace region (26 traced decoder layers at [1, C, 3072] have to fit),
# so the region is never smaller than the source's own sizing; a larger request still wins.
_TRACE_REGION = max(int(os.environ.get("TT_PERF_TRACE_REGION") or 0), TRACE_REGION_SIZE)

#: Filler used only to hit an EXACT token count; ISL is the tool's condition, not a sentence choice.
_ISL_FILLER = (
    "The quick brown fox jumps over the lazy dog. Tenstorrent builds AI accelerators and the "
    "compiler stack that runs models on them, from kernels up to whole transformer pipelines. "
)


def prompt_ids_for_isl(tokenizer, n_tokens: int):
    """A prompt of EXACTLY `n_tokens` ids, [1, n_tokens]."""
    try:  # prefer the tool's own helper when this checkout ships one
        from models.experimental.perf_automation.agent.perf_test_gen import (
            prompt_ids_for_isl as _tool_impl,
        )

        ids = _tool_impl(tokenizer, n_tokens)
        return torch.as_tensor(ids).reshape(1, -1).to(torch.int64)
    except (ImportError, AttributeError):
        pass
    n = max(1, int(n_tokens))
    base = torch.as_tensor(tokenizer(_ISL_FILLER, return_tensors="pt")["input_ids"]).reshape(-1)
    if base.numel() == 0:
        raise RuntimeError("tokenizer produced no ids for the ISL filler text")
    reps = (n + base.numel() - 1) // base.numel()
    return base.repeat(reps)[:n].reshape(1, n).to(torch.int64)


def _open_perf_device():
    """Open the device the SOURCE opens. `open_selftest_device` is this model's own opener
    (single chip, one command queue, trace region); a tool-planned multi-chip mesh opens the same
    way through open_mesh_device so the trace and the eager forward share one topology."""
    rows, cols = _MESH_SHAPE
    if rows * cols > 1:
        return (
            ttnn.open_mesh_device(
                ttnn.MeshShape(rows, cols),
                l1_small_size=L1_SMALL_SIZE,
                trace_region_size=_TRACE_REGION,
                num_command_queues=1,
            ),
            True,
        )
    return open_selftest_device(trace_region_size=_TRACE_REGION), False


def _close_perf_device(device, is_mesh: bool) -> None:
    if is_mesh:
        try:
            ttnn.close_mesh_device(device)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the measurement
            print("[perf] close_mesh_device failed: %r" % (exc,), flush=True)
    else:
        close_selftest_device(device)


def test_text_generation_perf():
    device, _is_mesh = _open_perf_device()
    # ONE resident build per device: the eager forward and the trace pass measure the SAME
    # pipeline object, so a full-depth run never builds the model twice on one chip.
    _built = {}

    def _build_for_perf(dev):
        pipe = _built.get(id(dev))
        if pipe is None:
            # EXACTLY the demo's factory + build args (demo: build_pipeline(device, layers=...)).
            pipe = build_pipeline(dev, layers=PERF_LAYERS, capacity=PERF_CAPACITY)
            _built[id(dev)] = pipe
        return pipe
    try:

        def _eager_forward():
            pipeline = _build_for_perf(device)
            # SETUP (host->device staging), outside the timed/wrapped region, exactly as the demo
            # does before its forward.
            prompt_ids = prompt_ids_for_isl(pipeline.tokenizer, PERF_ISL_TOKENS)
            staged = pipeline.stage_inputs(input_ids=prompt_ids)
            print("PERF_ISL_TOKENS=%d" % staged["prompt_len"], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            print(
                "PERF_DEPTH=%d PERF_SEQ_LEN=%d PERF_CAPACITY=%d PERF_BATCH_RESOLVED=%d"
                % (pipeline.depth, staged["seq_len"], pipeline.capacity, resolve_batch(pipeline, PERF_BATCH)),
                flush=True,
            )
            counter = [0]
            _orig = []

            def _draining(fn):
                def inner(*a, **k):
                    r = fn(*a, **k)
                    counter[0] += 1
                    if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                        try:
                            ttnn.ReadDeviceProfiler(device)
                        except Exception:
                            pass
                    return r

                return inner

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                # Call 1 `text_generation`: prefill + free-running greedy decode, BOUNDED by the
                # declared PERF_OSL_TOKENS -- the same number printed above, never a smaller one.
                out = pipeline.run_generate(staged, max_new_tokens=PERF_OSL_TOKENS)
                ttnn.synchronize_device(device)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            print("PERF_DECODED_TOKENS=%d" % len(out["tokens"]), flush=True)
            assert out is not None and out["tokens"]  # perf only — NO PCC

        def _traced_forward():
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            pipeline = _build_for_perf(device)
            # ISL: EXACTLY PERF_ISL_TOKENS tokens, and the traced prefill runs at seq_len=capacity,
            # which is derived from that same number (plus the decode horizon the KV must hold).
            _prompt_ids = prompt_ids_for_isl(pipeline.tokenizer, PERF_ISL_TOKENS)
            print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            print(
                "PERF_DEPTH=%d PERF_CAPACITY=%d" % (pipeline.depth, pipeline.capacity),
                flush=True,
            )
            # Stage adapter profiles WHATEVER emit-e2e emitted: PIPELINE_STAGES = [prefill, decode],
            # each with its own host-op-free <stage>_trace_setup / <stage>_trace_step.
            measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), device)

        def _try_traced():
            try:
                _traced_forward()
                return True
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
                return False

        _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
        if _PERF_TRACE and not _PROFILING:
            if not _try_traced():
                print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly", flush=True)
                _eager_forward()
        else:
            _eager_forward()
            if _PERF_TRACE:
                _try_traced()
    finally:
        _close_perf_device(device, _is_mesh)
