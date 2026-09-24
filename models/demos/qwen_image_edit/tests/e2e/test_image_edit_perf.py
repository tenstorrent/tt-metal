import inspect
import os
import time

import torch

import ttnn
from models.demos.qwen_image_edit.mesh import close_mesh, open_mesh
from models.demos.qwen_image_edit.tt import pipeline as P
from models.demos.qwen_image_edit.tt.inputs import EditConfig, sample_images, sample_prompts, sample_seeds
from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
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

# Diffusion: TIMESTEPS drive the op count. Small, env-overridable (demo default is 50).
PERF_STEPS = int(os.environ.get("TT_PERF_STEPS", "2"))
PERF_AREA = int(os.environ.get("TT_PERF_AREA", "256"))  # demo --area default
PERF_CFG_SCALE = float(os.environ.get("TT_PERF_CFG_SCALE", "4.0"))
PERF_NEGATIVE_PROMPT = " "

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402

# the demo's open_mesh() opens the T3K as a 2x4 mesh
_MESH_SHAPE = resolve_mesh_shape(default_rows=2, default_cols=4)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))


def _open_perf_mesh():
    rows, cols = _MESH_SHAPE
    try:
        params = inspect.signature(open_mesh).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kw = {}
    if _PERF_TRACE:
        if "trace_region_size" in params or accepts_var_kw:
            kw["trace_region_size"] = _TRACE_REGION
        if "num_command_queues" in params or accepts_var_kw:
            kw["num_command_queues"] = 1
    shape_ok = (rows, cols) == (2, 4)
    if not shape_ok:
        for name in ("mesh_shape", "shape"):
            if name in params:
                kw[name] = (rows, cols)
                shape_ok = True
                break
        if not shape_ok and "rows" in params and "cols" in params:
            kw["rows"], kw["cols"] = rows, cols
            shape_ok = True
    if shape_ok:
        # open exactly as the demo does
        return open_mesh(**kw)
    # planned topology differs from the source's (--devices/--mesh): open it directly.
    if rows * cols > 1:
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        except Exception:  # noqa: BLE001
            pass
    dkw = {"l1_small_size": 24576}
    if _PERF_TRACE:
        dkw["trace_region_size"] = _TRACE_REGION
        dkw["num_command_queues"] = 1
    return ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **dkw)


def _close_perf_mesh(device):
    try:
        close_mesh(device)
    except Exception:  # noqa: BLE001
        ttnn.close_mesh_device(device)


def _even(n):
    return n + (n % 2)


def _default_batch():
    if PERF_BATCH > 0:
        return PERF_BATCH
    try:
        b = int(getattr(EditConfig(), "batch", 0) or 0)
    except Exception:  # noqa: BLE001
        b = 0
    return b if b > 0 else 32  # demo's --batch default


def _tokenizer(hf):
    for path in (("tokenizer",), ("processor", "tokenizer"), ("text_tokenizer",)):
        obj = hf
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _make_cfg(batch, steps):
    return EditConfig(
        batch=batch,
        area=PERF_AREA * PERF_AREA,
        num_inference_steps=steps,
        true_cfg_scale=PERF_CFG_SCALE,
        negative_prompt=PERF_NEGATIVE_PROMPT,
    )


def _build_kwargs():
    return dict(
        layers=PERF_LAYERS,
        vision_encode_layers=PERF_VISION_ENCODE_LAYERS,
        text_encode_layers=PERF_TEXT_ENCODE_LAYERS,
        vae_encode_layers=PERF_VAE_ENCODE_LAYERS,
        denoise_layers=PERF_DENOISE_LAYERS,
        vae_decode_layers=PERF_VAE_DECODE_LAYERS,
    )


def test_image_edit_perf():
    hf = P.load_hf_reference(torch.float32)
    tok = _tokenizer(hf)
    # the VAE runs batch-parallel over the 2 mesh rows: keep the batch even (as the demo pads)
    batch = _even(_default_batch())

    _prompt_ids = None
    if tok is not None:
        _prompt_ids = prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
    print("PERF_ISL_TOKENS=%d" % (_prompt_ids.shape[-1] if _prompt_ids is not None else PERF_ISL_TOKENS), flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
    print("PERF_BATCH=%d PERF_STEPS=%d PERF_AREA=%d" % (batch, PERF_STEPS, PERF_AREA), flush=True)

    def _inputs(n):
        images = sample_images(n)
        if _prompt_ids is not None and tok is not None:
            ids = _prompt_ids.reshape(-1).tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            prompts = [text] * n
        else:
            prompts = sample_prompts(n)
        seeds = sample_seeds(n)
        return images, prompts, seeds

    device = _open_perf_mesh()
    try:

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
                    if type(_op).__name__ == "FastOperation":
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            out = None
            image = None
            _fw0 = time.monotonic()
            try:
                steps = max(1, min(PERF_STEPS, _EAGER_OSL_TOKENS))
                cfg = _make_cfg(batch, steps)
                pipe = P.build_pipeline(device, model=hf, cfg=cfg, **_build_kwargs())
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
                images, prompts, seeds = _inputs(cfg.batch)
                enc = pipe.encode(cfg, images=images, prompts=prompts, seeds=seeds)
                p = pipe.prepare(enc)
                _fw0 = time.monotonic()
                out = pipe.run_image_edit(p)
                image = P.to_host(out).to(torch.float32)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None
            assert image is not None and image.numel() > 0  # perf only — NO PCC

        def _traced_forward():
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                from models.demos.qwen_image_edit.tt.pipeline import build_pipeline

                cfg = _make_cfg(batch, max(1, PERF_STEPS))
                return build_pipeline(dev, model=hf, cfg=cfg, **_build_kwargs())

            ids = _prompt_ids
            if ids is None:
                ids = torch.zeros((1, PERF_ISL_TOKENS), dtype=torch.long)
            measure_adapter(PipelineStageAdapter(_build_for_perf, ids, batch=PERF_BATCH), device)

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
    finally:
        _close_perf_mesh(device)
