# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the gemma-3 TEXT ('main') pipeline.

Derived from tests/e2e/test_pcc_hf.py, but with the reference/HF model construction and EVERY PCC
comparison removed: this file runs ONLY the on-device TTNN forward and times it.
"""
from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn

# Same model identity the e2e gate pins (tests/e2e/conftest.py does this too, unconditionally).
os.environ.setdefault("HF_MODEL", "google/gemma-3-12b-it")
HF_MODEL_ID = os.environ.get("HF_MODEL", "google/gemma-3-12b-it")

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS. 128 in / 128 out is the industry-standard short-context
# benchmark point; both are env-overridable and both are echoed below so a reader of the log never
# has to guess what context length the throughput number describes.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# The KV/model window. NOT the model's 128k maximum -- a bounded, representative operating point
# (the demo's own value), env-overridable.
PERF_MAX_SEQ_LEN = int(os.environ.get("TT_PERF_MAX_SEQ_LEN", "1024"))
BATCH_SIZE = 1

# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window so a deep model's marker stream (x mesh
# chips) does not overflow the profiler; the tool sends that number for tracy runs. The variable being
# ABSENT means ALL LAYERS -- the tool expresses "whole model" by REMOVING the cap, never by sending a
# sentinel, because "0" arrives as a truthy string and gets read as "build zero layers".
# Pass PERF_LAYERS straight to the builder: None is every builder's own all-layers value. Do NOT
# default it to a number here -- that would silently cap the full-depth gate.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# TOPOLOGY. The source uses the `mesh_device` FIXTURE with an explicit (1, 1) parametrize, so the
# fixture + decorator are kept and resolve_mesh_shape feeds the tuple -- that is what lets
# --devices/--mesh reshape the run while an unset env behaves exactly as the source does.
from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape  # noqa: E402

_DEMO_MESH = {"P150": (1, 1), "N150": (1, 1), "N300": (1, 1)}.get(os.environ.get("MESH_DEVICE"), (1, 1))
_MESH_SHAPE = resolve_mesh_shape(default_rows=_DEMO_MESH[0], default_cols=_DEMO_MESH[1])

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# The source opens through the plain mesh_device fixture (no extra device_params), so only the
# trace-region reservation is added here. No fabric: the source never sets it and this mesh is 1x1.
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE, for baseline and every candidate. The tool
    # measures trace+1cq end to end, so the device opens with a single command queue.
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = 1

PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}


def _prompt_ids_for_isl(tokenizer, n_tokens: int) -> torch.Tensor:
    """EXACTLY n_tokens ids, as a 1-D LongTensor.

    The tool owns ISL, not this file: no example sentence is written and no length is picked here.
    Prefer the tool's own helper when this tree ships it; otherwise build the same thing locally by
    tokenizing filler and truncating to the requested length.
    """
    try:
        from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

        ids = prompt_ids_for_isl(tokenizer, n_tokens)
        ids = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
        return ids.reshape(-1).long()
    except (ImportError, AttributeError):
        pass

    filler = "The capital of France is Paris, and the history of the city stretches back many years. "
    text = filler
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    while len(ids) < n_tokens:
        text += filler
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_main_perf(device_params, mesh_device):
    from transformers import AutoTokenizer

    device = mesh_device  # every ttnn call below dispatches on the mesh the fixture opened

    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    # ISL: build the prompt to EXACTLY PERF_ISL_TOKENS tokens rather than writing an example
    # sentence, so the measurement condition is the tool's choice and not the generator's.
    _prompt_ids = _prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
    print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
    print("PERF_LAYERS=%s" % (PERF_LAYERS,), flush=True)

    def _build_for_perf(dev):
        """Return the RESIDENT, STAGE-EXPOSING pipeline object (PIPELINE_STAGES + trace hooks)."""
        from models.demos.multimodal.gemma3.tt.pipeline import build_pipeline

        # build_pipeline takes no depth argument, so the depth cap is left to the builder/env --
        # never a literal 0, which would be read as a zero-layer model.
        pipe = build_pipeline(
            dev,
            max_seq_len=PERF_MAX_SEQ_LEN,
            batch_size=BATCH_SIZE,
            instruct=True,
        )
        # emit-e2e's per-stage input hook: without it PipelineStageAdapter calls
        # <stage>_trace_setup(None) and the pipeline falls back to its own 6-token default prompt,
        # which would silently discard PERF_ISL_TOKENS.
        _ids = _prompt_ids.reshape(1, -1)
        pipe.prefill_trace_inputs = lambda: {"input_ids": _ids}
        pipe.decode_trace_inputs = lambda: {"input_ids": _ids}
        return pipe

    def _eager_forward():
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
            pipe = _build_for_perf(device)
            # BOUNDED: one prefill over the ISL prompt + PERF_MAX_NEW_TOKENS decode steps.
            state = pipe.decode_prefill(_prompt_ids.reshape(1, -1))
            for _ in range(max(1, PERF_MAX_NEW_TOKENS)):
                state = pipe.decode_step(state)
            out = state
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        # Stage adapter profiles WHATEVER emit-e2e emitted: every PIPELINE_STAGES entry gets
        # traced. Falls back to the single decode contract for pipelines that expose only decode_step.
        measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=BATCH_SIZE), device)

    def _try_traced():
        try:
            _traced_forward()
            return True
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
            return False

    # MEASUREMENT ORDER — trace by default, eager only as its fallback. Never both at full depth on
    # one device: the second build has no memory left for its KV cache.
    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not _PROFILING:
        if not _try_traced():
            print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly", flush=True)
            _eager_forward()
    else:
        _eager_forward()
        if _PERF_TRACE:
            _try_traced()
