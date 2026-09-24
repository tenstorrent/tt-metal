# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the Qwen-Image-Edit 'main' pipeline (Call 1: image_edit) on TT.

Built and run exactly as tests/e2e/test_e2e_image_edit.py does (same mesh_device fixture, same
device_params, same build_pipeline / encode / prepare / host_op_selftest forward), but ONLY the
on-device TTNN forward is kept: no HF golden, no PCC, no correctness gates.

The heavy axis for this diffusion model is the scheduler TIMESTEP count, so the profiled forward runs
a SHORT schedule (TT_PERF_STEPS, default 2) instead of the source's 50.
"""
from __future__ import annotations

import importlib
import os
import sys
import time

import pytest
import torch

import ttnn

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# diffusion's heavy axis: scheduler timesteps. SMALL default; the source's full schedule is 50.
PERF_STEPS = int(os.environ.get("TT_PERF_STEPS", "2"))
# the batch the source test drives (E2E_BATCH); TT_PERF_BATCH>0 overrides
_SOURCE_BATCH = 4

_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
_pl_vision_encode = (os.environ.get("TT_PERF_VISION_ENCODE_LAYERS") or "").strip()
PERF_VISION_ENCODE_LAYERS = (
    int(_pl_vision_encode) if (_pl_vision_encode.isdigit() and int(_pl_vision_encode) > 0) else None
)
_pl_text_encode = (os.environ.get("TT_PERF_TEXT_ENCODE_LAYERS") or "").strip()
PERF_TEXT_ENCODE_LAYERS = int(_pl_text_encode) if (_pl_text_encode.isdigit() and int(_pl_text_encode) > 0) else None
_pl_vae_encode = (os.environ.get("TT_PERF_VAE_ENCODE_LAYERS") or "").strip()
PERF_VAE_ENCODE_LAYERS = int(_pl_vae_encode) if (_pl_vae_encode.isdigit() and int(_pl_vae_encode) > 0) else None
_pl_denoise = (os.environ.get("TT_PERF_DENOISE_LAYERS") or "").strip()
PERF_DENOISE_LAYERS = int(_pl_denoise) if (_pl_denoise.isdigit() and int(_pl_denoise) > 0) else None
_pl_vae_decode = (os.environ.get("TT_PERF_VAE_DECODE_LAYERS") or "").strip()
PERF_VAE_DECODE_LAYERS = int(_pl_vae_decode) if (_pl_vae_decode.isdigit() and int(_pl_vae_decode) > 0) else None

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402,F401


def prompt_ids_for_isl(tokenizer, n_tokens):
    """Prompt ids of EXACTLY n_tokens. Uses the tool's helper when this checkout ships it."""
    try:
        from models.experimental.perf_automation.agent import perf_test_gen as _ptg

        _fn = getattr(_ptg, "prompt_ids_for_isl", None)
        if _fn is not None:
            return _fn(tokenizer, n_tokens)
    except Exception:  # noqa: BLE001
        pass
    n_tokens = max(1, int(n_tokens))
    filler = (
        "Replace the background with a quiet mountain lake at sunrise, keep the subject sharp, "
        "preserve the original lighting on the face and add soft reflections on the water. "
    )
    ids = []
    if tokenizer is not None:
        chunk = tokenizer(filler, add_special_tokens=False)["input_ids"]
        if chunk and isinstance(chunk[0], (list, tuple)):
            chunk = chunk[0]
        chunk = list(chunk)
        while chunk and len(ids) < n_tokens:
            ids.extend(chunk)
    if not ids:
        ids = list(range(1, n_tokens + 1))
    return torch.tensor(ids[:n_tokens], dtype=torch.long).unsqueeze(0)


# The source tree the demo lives in; used to graft sub-packages missing from this checkout.
_SOURCE_ROOT = os.environ.get("TT_PERF_SOURCE_ROOT", "/home/ttuser/apande/tt-metal")


def _graft_missing(missing):
    """Extend the deepest already-importable parent package's __path__ with the source tree's copy."""
    parts = missing.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:i])
        try:
            pkg = importlib.import_module(parent)
        except Exception:  # noqa: BLE001
            continue
        path = getattr(pkg, "__path__", None)
        if path is None:
            return False
        cand = os.path.join(_SOURCE_ROOT, *parts[:i])
        if os.path.isdir(cand) and cand not in list(path):
            path.append(cand)
            importlib.invalidate_caches()
            return True
        return False
    return False


def _import_pipeline():
    for _ in range(8):
        try:
            return importlib.import_module("models.demos.qwen_image_edit.tt.pipeline")
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or ""
            if not missing or not _graft_missing(missing):
                raise
            for k in [k for k in sys.modules if k.startswith("models.demos.qwen_image_edit.tt")]:
                sys.modules.pop(k, None)
    return importlib.import_module("models.demos.qwen_image_edit.tt.pipeline")


try:
    _P0 = _import_pipeline()
    _SRC_MESH = tuple(_P0.MESH_SHAPE)
    _SRC_TRACE_REGION = int(_P0.DEVICE_PARAMS["trace_region_size"])
except Exception as _ie:  # noqa: BLE001 -- the real import error surfaces inside the test
    print("PIPELINE_IMPORT_DEFERRED=%r" % (_ie,), flush=True)
    _SRC_MESH = (2, 4)
    _SRC_TRACE_REGION = 41943040

# TOPOLOGY: the source's mesh_device fixture shape is P.MESH_SHAPE; --devices/--mesh can reshape it.
_DEMO_MESH = _SRC_MESH
_MESH_SHAPE = tuple(resolve_mesh_shape(default_rows=_DEMO_MESH[0], default_cols=_DEMO_MESH[1]))

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576, "trace_region_size": _SRC_TRACE_REGION}
if _MESH_SHAPE[0] * _MESH_SHAPE[1] > 1:
    _DEV_PARAMS["fabric_config"] = ttnn.FabricConfig.FABRIC_1D  # the source sets it
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(
        os.environ.get("TT_PERF_TRACE_REGION", str(max(41943040, _DEV_PARAMS["trace_region_size"])))
    )
    _DEV_PARAMS["num_command_queues"] = 1


def _batch():
    return PERF_BATCH if PERF_BATCH > 0 else _SOURCE_BATCH


def _config():
    from models.demos.qwen_image_edit.tt.inputs import EditConfig

    return EditConfig(batch=_batch(), num_inference_steps=PERF_STEPS)


def _build_kwargs():
    return dict(
        layers=PERF_LAYERS,
        vision_encode_layers=PERF_VISION_ENCODE_LAYERS,
        text_encode_layers=PERF_TEXT_ENCODE_LAYERS,
        vae_encode_layers=PERF_VAE_ENCODE_LAYERS,
        denoise_layers=PERF_DENOISE_LAYERS,
        vae_decode_layers=PERF_VAE_DECODE_LAYERS,
    )


@pytest.fixture(scope="module")
def hf_pipe():
    # weights source for build_pipeline (the source passes it as model=); no golden is built
    P = _import_pipeline()
    return P.load_hf_reference(torch.float32)


@pytest.mark.timeout(2 * 3600)
@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_main_perf(mesh_device, hf_pipe):
    P = _import_pipeline()
    print("PERF_MESH_SHAPE=%dx%d" % _MESH_SHAPE, flush=True)
    print("PERF_STEPS=%d PERF_BATCH=%d" % (PERF_STEPS, _batch()), flush=True)

    def _eager_forward():
        cfg = _config()
        pipe = P.build_pipeline(mesh_device, model=hf_pipe, cfg=cfg, **_build_kwargs())
        # --- per-stage marks (injected) ---------------------------------------------------
        # Runs HERE, at the end of the function that built the pipeline, because that object is a LOCAL of
        # this scope: an earlier version copied the test's own PipelineStageAdapter(...) arguments into the
        # profiling branch and raised NameError, since the generator had defined them inside another
        # function. Handed locals() rather than a name, so nothing depends on how the test spells things.
        print("STAGE_MARKS_ENTER", flush=True)
        try:
            from models.experimental.perf_automation.agent import stage_marks as _tt_sm2

            print("STAGE_MARKS_RESULT=%d" % _tt_sm2.mark_stages_in_scope(locals(), device), flush=True)
        except Exception as _tt_e2:  # noqa: BLE001
            print("STAGE_MARKS_SKIPPED=%r" % (_tt_e2,), flush=True)
        enc = pipe.encode(cfg)
        p = pipe.prepare(enc)
        print(
            f"[perf] batch={p.B} steps={p.num_steps} size={enc.width}x{enc.height} cfg_scale={p.cfg_scale}", flush=True
        )

        counter = [0]
        _orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(mesh_device)
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
            _verdict, out = pipe.host_op_selftest(p)  # the same on-device forward the source runs
            ttnn.synchronize_device(mesh_device)
            try:
                ttnn.ReadDeviceProfiler(mesh_device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC
        print(f"[perf] steps_run={pipe.steps_run}", flush=True)

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        def _build_for_perf(dev):
            build_pipeline = _import_pipeline().build_pipeline

            return build_pipeline(dev, model=hf_pipe, cfg=_config(), **_build_kwargs())

        _prompt_ids = prompt_ids_for_isl(getattr(hf_pipe, "tokenizer", None), PERF_ISL_TOKENS)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), mesh_device)

    def _try_traced():
        try:
            _traced_forward()
            return True
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
            return False

    print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)

    _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    if _PERF_TRACE and not _PROFILING:
        if not _try_traced():
            print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly", flush=True)
            _eager_forward()
    else:
        # --- stage marks (injected by perf_test_gen) -------------------------------------
        # The measured region is bracketed by the conventional start/stop pair so the main report
        # slices exactly the ops run_head emitted; the pass below is additive and feeds per-stage
        # fidelity only. Injected rather than written by the generator: the skeleton is advisory and
        # a generated test simply omitted this, which is why five earlier attempts measured nothing.
        try:
            from models.experimental.perf_automation.agent import stage_marks as _tt_sm
        except Exception:  # noqa: BLE001
            _tt_sm = None
        if _tt_sm is not None:
            _tt_sm.signpost("start")
        _eager_forward()
        if _tt_sm is not None:
            _tt_sm.signpost("stop")
        if _PERF_TRACE:
            _try_traced()
