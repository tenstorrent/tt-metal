# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2b: generate a video where the DiT denoise loop runs on the Blackhole
(ttnn); Qwen text-encode, VAE decode and the scheduler run on CPU.

    HY_H=64 HY_W=64 HY_FRAMES=1 HY_STEPS=4 HY_TRUNC=16 HY_OUT=/path \
        pytest tests/e2e/test_stage2b_gen.py::test_stage2b_gen -s

One-chip DRAM note: the 16.6GB bf16 weights + the real prompt's long text-stream
activations exceed one Blackhole's ~34GB DRAM at real resolution. It completes on
ONE chip only with a truncated text seq at tiny res (HY_TRUNC).

Full-res, untruncated-text generation needs QB2 multi-chip sharding -- see
`test_stage2b_gen_qb2` below and `real_weights/README.md` "RESUME ON QB2":

    HY_H=480 HY_W=848 HY_FRAMES=13 HY_STEPS=50 HY_OUT=/path \
        pytest tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2 -s

Width 848 matches this checkpoint's own computed default (HunyuanVideo15Pipeline's
default_aspect_ratio + target_size); 832 (the earlier bring-up value) was off by
16px. num_frames stays at 13, NOT this checkpoint's real default of 121: 121
frames needs ~40GB per chip just for one layer's joint-attention score matrix
(measured on live QB2 hardware) -- over a single chip's entire ~34GB DRAM,
because attention here is unsharded along the sequence dimension (TP=4 shards
only the head dimension, 16->4 heads/chip). 61 and 33 frames were also tried on
live hardware and both OOM'd too, with a failure pattern that didn't scale as
cleanly as buffer-size math predicts (deeper into the 54-layer stack before
failing as the per-layer buffer shrinks, suggesting per-layer memory isn't
fully freed/reused across layers) -- so this isn't just "pick a smaller number,"
it's a real gap that needs either a flash-attention-style kernel (never
materialize the full seq x seq score matrix) or sequence/context parallelism
(shard attention along tokens too, not just heads) to close. Out of scope here.
"""
import contextlib
import json
import os
import shutil
import subprocess
import time

import pytest

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.tests.e2e.test_real_weight_pcc import coerce_bf16
from models.demos.hf_eager.hunyuanvideo_1_5.tt import pipeline as P
from models.demos.hf_eager.hunyuanvideo_1_5.tt.media_writeout import save_generated_frames

_COMMUNITY = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
_I2V_COMMUNITY = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"

# Phase attribution. Import time is the earliest point this process can observe,
# so `t_start_of_run - _T_IMPORT` bounds collection + mesh open + fixture setup,
# which is otherwise invisible from inside the test body.
_T_IMPORT = time.perf_counter()
_PHASES = {}  # sequential top-level phases; these sum to the function total
_NESTED = {}  # accumulators for work nested INSIDE a phase (subsets, not addends)
_CALL_LOG = {}  # per-call durations, so first-call JIT can be split from steady state


@contextlib.contextmanager
def _phase(name):
    start = time.perf_counter()
    try:
        yield
    finally:
        _PHASES[name] = _PHASES.get(name, 0.0) + (time.perf_counter() - start)


def _record_nested(name, seconds):
    _NESTED[name] = _NESTED.get(name, 0.0) + seconds
    _CALL_LOG.setdefault(name, []).append(round(seconds, 4))


def _time_module_forward(module, name):
    """Time an ``nn.Module`` via hooks, without touching its class."""
    pending = []

    def pre_hook(*_):
        pending.append(time.perf_counter())

    def post_hook(*_):
        if pending:
            _record_nested(name, time.perf_counter() - pending.pop())

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)


def _time_call(obj, name):
    """Accumulate wall seconds for ``obj(...)``.

    ``__call__`` is looked up on the type, so the bespoke-adapter branch patches
    the class -- safe only because each TT adapter class backs exactly one live
    instance. It is never valid for an ``nn.Module`` (that shares one ``__call__``
    with every module in the process), so those go through forward hooks instead.
    """
    if obj is None:
        return
    import torch

    if isinstance(obj, torch.nn.Module):
        _time_module_forward(obj, name)
        return
    cls = type(obj)
    original = cls.__call__
    if getattr(original, "_hy_timed", False):
        return

    def timed(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            _record_nested(name, time.perf_counter() - start)

    timed._hy_timed = True
    cls.__call__ = timed


def _time_method(obj, attr, name):
    """Shadow one bound method with a timing wrapper via an instance attribute."""
    if obj is None or not hasattr(obj, attr):
        return
    original = getattr(obj, attr)

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _record_nested(name, time.perf_counter() - start)

    with contextlib.suppress(AttributeError):
        setattr(obj, attr, timed)


def _pipeline_path(repo=_COMMUNITY):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    return snapshot_download(repo)  # cached if present, else downloads (~50GB)


def _load_i2v_image(height, width, label=""):
    """i2v needs a conditioning first-frame image. HY_IMAGE=<path> overrides; otherwise
    fall back to a saved real frame, else a synthetic gradient, so the test is self-
    contained. The pipeline's image_processor resizes to the target bucket internally."""
    import numpy as np
    from PIL import Image

    candidates = [os.environ.get("HY_IMAGE"), "/home/tt-admin/sdawle/hunyuanvideo1.5/hy720p_FIXED_frame060.png"]
    for c in candidates:
        if c and os.path.exists(c):
            img = Image.open(c).convert("RGB")
            print(f"[{label}] i2v conditioning image: {c} {img.size}", flush=True)
            return img
    arr = np.zeros((height, width, 3), dtype="uint8")
    arr[..., 0] = 120
    arr[..., 1] = np.linspace(40, 200, height)[:, None]
    img = Image.fromarray(arr)
    print(f"[{label}] i2v conditioning image: synthetic gradient {img.size}", flush=True)
    return img


def _run_stage2b_gen(device, *, height, width, frames, steps, trunc, outdir, label, use_trace):
    import numpy as np
    import torch
    from diffusers import HunyuanVideo15Pipeline
    from PIL import Image

    coerce_bf16()
    # HY_I2V=1: image->video. Uses a DIFFERENT diffusers pipeline class
    # (HunyuanVideo15ImageToVideoPipeline): it adds a SigLIP image_encoder -> image_embeds
    # AND VAE-encodes the input image into the DiT's conditioning channels (in_channels=65 =
    # 32 noise + 32 image-cond + 1 mask). Loading the i2v checkpoint directly means its
    # scheduler already carries the correct shift (480p_i2v=5.0) -- no override needed (unlike
    # the t2v HY_720P transformer-swap below, which must fix the reused 480p scheduler).
    _t_fn_start = time.perf_counter()
    _i2v = os.environ.get("HY_I2V") == "1"
    with _phase("host_checkpoint_load_s"):
        if _i2v:
            from diffusers import HunyuanVideo15ImageToVideoPipeline

            pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
                _pipeline_path(_I2V_COMMUNITY), torch_dtype=torch.bfloat16
            )
        else:
            pipe = HunyuanVideo15Pipeline.from_pretrained(_pipeline_path(), torch_dtype=torch.bfloat16)
    # HY_720P=1: swap the 480p DiT for the 720p_t2v transformer (same arch, target_size=960).
    # VAE + text encoders are identical across tiers, so we reuse the cached 480p ones and
    # only load the 720p transformer (its shards must already be in the HF cache). Pass
    # HY_H=720 HY_W=1280 for the 720p 16:9 bucket.
    if os.environ.get("HY_720P") == "1":
        import gc

        from huggingface_hub import snapshot_download

        _repo720 = (
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v"
            if _i2v
            else "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
        )
        _720 = snapshot_download(
            _repo720,
            allow_patterns=["transformer/*", "model_index.json"],
        )
        _cls = type(pipe.transformer)
        pipe.transformer = None  # free the 480p DiT before loading the 720p one
        gc.collect()
        # low_cpu_mem_usage=False fully materializes the weights into RAM (no lazy
        # mmap) -- the deferred mmap page-fault access was SIGBUS-ing during the
        # ttnn.from_torch device upload in build_pipeline.
        pipe.transformer = _cls.from_pretrained(
            _720, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        )
        # The i2v pipeline caches target_size at __init__ from the (480p) transformer and uses
        # it to pick the output resolution bucket -- refresh it from the swapped-in 720p
        # transformer (640 -> 960) or i2v keeps generating at 480p. (t2v takes explicit H/W so
        # it has no target_size attr; the guard skips it.)
        if hasattr(pipe, "target_size"):
            pipe.target_size = pipe.transformer.config.target_size
        # The 720p checkpoint uses flow-match scheduler shift=9.0 (480p uses 5.0); keeping
        # the 480p shift under-shifts the 720p denoise trajectory -> washed-out/oversaturated
        # output. VAE scaling_factor + guidance are identical, so only the shift needs fixing.
        # NOTE: FlowMatchEulerDiscreteScheduler.set_timesteps reads `self.shift` (the property
        # backed by `self._shift`), NOT `self.config.shift`. register_to_config only updates the
        # config dict, so it's a silent no-op -- must go through set_shift()/`_shift`.
        _shift = float(os.environ.get("HY_SCHED_SHIFT", "7.0" if _i2v else "9.0"))
        if hasattr(pipe.scheduler, "set_shift"):
            pipe.scheduler.set_shift(_shift)
        else:
            pipe.scheduler._shift = _shift
        try:
            pipe.scheduler.register_to_config(shift=_shift)  # keep config in sync (cosmetic)
        except Exception:
            pass
        print(
            f"[{label}] HY_720P: swapped in 720p transformer "
            f"(target_size={pipe.transformer.config.target_size}, "
            f"sched shift set to {pipe.scheduler.shift})",
            flush=True,
        )

    # Encode text before constructing the full-mesh TT DiT. This permits a safe
    # sequential encoder load -> encode -> unload -> DiT load lifecycle on the
    # same mesh, avoiding overlapping submesh contexts and reducing peak DRAM.
    qwen_adapter = None
    if os.environ.get("HY_TT_QWEN", "0") == "1":
        from models.demos.hf_eager.hunyuanvideo_1_5.tt import qwen_encoder as _qe

        qwen_dev = _qe.HY_QWEN_SUBMESH
        if qwen_dev is None and os.environ.get("HY_TT_QWEN_SHARED", "0") == "1":
            qwen_dev = device
        if qwen_dev is not None:
            placement = "separate submesh" if _qe.HY_QWEN_SUBMESH is not None else "shared DiT mesh (TP+FSDP)"
            with _phase("tt_qwen_weight_upload_s"):
                qwen_adapter = _qe.TTQwenTextEncoderAdapter(pipe.text_encoder, qwen_dev)
            pipe.text_encoder = qwen_adapter
            print(f"[{label}] Qwen text-encode: ON DEVICE ({placement})", flush=True)
        else:
            print(
                f"[{label}] no Qwen submesh; using host Qwen "
                "(HY_TT_QWEN_SHARED=1 enables sequential full-mesh encoding)",
                flush=True,
            )

    byt5_adapter = None
    if os.environ.get("HY_TT_BYT5", "0") == "1":
        from models.demos.hf_eager.hunyuanvideo_1_5.tt import byt5_encoder as _be

        # `select_byt5_device` prefers the reserved disjoint submesh and falls back
        # to the current mesh only when that mesh is itself a legal 1/2-device byT5
        # placement. It never carves a view out of the live DiT mesh.
        byt5_dev, byt5_support = _be.select_byt5_device(pipe.text_encoder_2.config, device)
        if byt5_dev is None:
            print(f"[{label}] byT5 text-encode: HOST ({byt5_support.reason})", flush=True)
        else:
            with _phase("tt_byt5_weight_upload_s"):
                byt5_adapter = _be.TTByT5EncoderAdapter(
                    pipe.text_encoder_2, byt5_dev, max_prompt_length=int(pipe.tokenizer_2_max_length)
                )
            pipe.text_encoder_2 = byt5_adapter
            print(
                f"[{label}] byT5 text-encode: ON DEVICE "
                f"({byt5_support.strategy} on a dedicated {tuple(byt5_dev.shape)} mesh)",
                flush=True,
            )

    from models.demos.hf_eager.hunyuanvideo_1_5.tt.text_conditioning import encode_prompt_pair

    _prompt = os.environ.get("HY_PROMPT", "A cat walks on the grass, realistic")
    _neg = os.environ.get("HY_NEG_PROMPT") or None
    _cache_prompts = os.environ.get("HY_PROMPT_CACHE", "0") == "1"
    _time_call(pipe.text_encoder, "qwen_encode_s")
    _time_call(pipe.text_encoder_2, "byt5_encode_s")
    with _phase("text_encode_s"):
        text_kwargs, cache_hit = encode_prompt_pair(pipe, _prompt, _neg, use_cache=_cache_prompts)
    print(
        f"[{label}] text conditioning: {'cache hit' if cache_hit else 'encoded'} "
        f"({'persistent cache enabled' if _cache_prompts else 'cache disabled'})",
        flush=True,
    )
    if qwen_adapter is not None and os.environ.get("HY_TT_QWEN_KEEP_RESIDENT", "0") != "1":
        qwen_adapter.deallocate_weights()
        print(f"[{label}] Qwen weights unloaded before DiT construction", flush=True)
    if byt5_adapter is not None:
        byt5_adapter.deallocate_weights()
        print(f"[{label}] byT5 weights unloaded before DiT construction", flush=True)

    real_tf = pipe.transformer
    with _phase("tt_dit_weight_upload_s"):
        tt = P.build_pipeline(device, real_tf)
    calls = {"n": 0, "device_runs": 0}

    # The reusable adapter owns diffusers' per-condition CFG grouping. The
    # opt-in resident mode additionally coordinates with a scheduler wrapper:
    # model outputs and the evolving SP latent stay on device; only the final
    # latent is gathered for VAE handoff.
    _resident = os.environ.get("HY_DEVICE_RESIDENT_DENOISE", "0") == "1"
    tt_adapter = P.TTTransformerAdapter(
        real_tf,
        tt,
        getattr(pipe, "guider", None),
        use_trace=use_trace,
        device_resident=_resident,
        task=("i2v" if _i2v else "t2v"),
        trunc=trunc,
        counters=calls,
    )
    pipe.transformer = tt_adapter
    if _resident:
        pipe.scheduler = P.DeviceResidentFlowMatchScheduler(pipe.scheduler, tt_adapter, pipe.guider)
        print(
            f"[{label}] FlowMatch Euler + latent: DEVICE RESIDENT "
            f"({'trace' if use_trace else 'eager'}, final gather only)",
            flush=True,
        )

    # Optional: run VAE decode on device too (HY_TT_VAE=1). Replaces the CPU
    # AutoencoderKLHunyuanVideo15.decode with the ttnn port, replicated across the
    # mesh. See tt/vae_decoder.py.
    if os.environ.get("HY_TT_VAE", "0") == "1":
        from models.demos.hf_eager.hunyuanvideo_1_5.tt import vae_decoder as _vd

        # Prefer a SEPARATE submesh (carved by the fixture on different chips) so the
        # VAE decode has its own DRAM and doesn't co-reside with the resident DiT.
        vae_dev = _vd.HY_VAE_SUBMESH or device
        with _phase("tt_vae_weight_upload_s"):
            pipe.vae = _vd.TTVAEDecodeAdapter(pipe.vae, vae_dev)
        placement = "separate chips" if vae_dev is not device else "shared with DiT"
        print(f"[{label}] VAE decode: ON DEVICE (ttnn) on {list(vae_dev.get_device_ids())} ({placement})", flush=True)

    # Optional: run the i2v SigLIP image-encoder on device (HY_TT_SIGLIP=1). The 27-layer
    # vision transformer runs REPLICATED on the full DiT mesh -- shared with the resident DiT,
    # the same way the VAE-decode and text-encode adapters run -- so it needs no spare chip and
    # works at sp=8xtp=4. (patch-embed + post-LN stay on host; the adapter PCC-self-checks vs
    # the host encoder on first call. See tt/siglip_encoder.py, validated by
    # tests/pcc/test_siglip.py at PCC ~0.995.)
    if _i2v and os.environ.get("HY_TT_SIGLIP", "0") == "1":
        from models.demos.hf_eager.hunyuanvideo_1_5.tt import siglip_encoder as _se

        with _phase("tt_siglip_weight_upload_s"):
            pipe.image_encoder = _se.TTSiglipImageEncoderAdapter(pipe.image_encoder, device)
        print(f"[{label}] SigLIP image-encode: ON DEVICE (ttnn, replicated on full mesh)", flush=True)

    _pkw = {}
    if _i2v:
        # i2v derives the output resolution from the input image + the checkpoint's
        # target_size (640->480p, 960->720p), so it takes NO height/width -- it computes
        # them via video_processor.calculate_default_height_width and crop-resizes the image.
        _pkw["image"] = _load_i2v_image(height, width, label)
    _dims = {} if _i2v else {"height": height, "width": width}
    print(f"[{label}] prompt: {_prompt}\n[{label}] negative: {_neg}", flush=True)
    # Attribute the three device-capable stages inside the single pipe() call.
    _time_call(pipe.transformer, "dit_denoise_s")
    _time_method(pipe.vae, "decode", "vae_decode_s")
    _time_method(pipe.vae, "encode", "vae_encode_s")
    if getattr(pipe, "image_encoder", None) is not None:
        _time_call(pipe.image_encoder, "siglip_encode_s")
    try:
        with _phase("generate_total_s"):
            generated = pipe(
                prompt=None,
                num_frames=frames,
                num_inference_steps=steps,
                generator=torch.Generator().manual_seed(0),
                **_dims,
                **_pkw,
                **text_kwargs,
            )
    finally:
        # Multi-shape CFG owns one trace region per exact text shape. Release
        # every region on both success and generation failure.
        if use_trace:
            tt_adapter.release_trace()
    out = generated.frames[0]
    print(
        f"\n[{label}] generated {len(out)} frames; transformer __call__s={calls['n']}, "
        f"real on-device runs={calls['device_runs']}",
        flush=True,
    )

    # Media writeout via tt/media_writeout.py. At 121 frames this stage is ~50s,
    # comparable to VAE decode, and most of it is the animated GIF: a host profile
    # measured Pillow GIF at 2.43s per 13 frames (~22.6s at 121) against 0.65s for
    # serial PNG and 0.079s for 16-thread PNG. The gates below are therefore worth
    # real time on long generations:
    #   HY_FAST_WRITEOUT=1  thread the PNG writes   (~6.1s -> ~0.7s at 121f)
    #   HY_SAVE_GIF=0       skip the GIF entirely   (~22.6s at 121f; mp4 still written)
    # Defaults reproduce the previous inline behaviour byte for byte.
    fps = int(os.environ.get("HY_FPS", "24"))
    with _phase("frame_writeout_s"):
        timings = save_generated_frames(out, outdir, fps=fps)

    n = timings["frames"]
    parts = [f"{n} frames"]
    if timings.get("gif"):
        parts.append("tt_blackhole.gif")
    if timings.get("mp4"):
        parts.append(f"tt_blackhole.mp4 ({fps}fps)")
    detail = " + ".join(parts)
    suffix = "" if timings.get("mp4") else " (ffmpeg not found, no mp4)"
    print(f"[{label}] SAVED {detail} -> {outdir}{suffix}", flush=True)
    print(f"[{label}] writeout timings: {timings}", flush=True)

    in_function = time.perf_counter() - _t_fn_start
    phase_sum = sum(_PHASES.values())
    dit_calls = _CALL_LOG.get("dit_denoise_s", [])
    steady = dit_calls[1:]
    nested_in_generate = sum(
        _NESTED.get(k, 0.0) for k in ("dit_denoise_s", "vae_decode_s", "vae_encode_s", "siglip_encode_s")
    )
    summary = {
        "config": {
            "frames": frames,
            "steps": steps,
            "height": height,
            "width": width,
            "i2v": _i2v,
            "trace": bool(use_trace),
            "tt_qwen": os.environ.get("HY_TT_QWEN", "0") == "1",
            "tt_siglip": os.environ.get("HY_TT_SIGLIP", "0") == "1",
            "tt_vae": os.environ.get("HY_TT_VAE", "0") == "1",
            "tt_byt5": os.environ.get("HY_TT_BYT5", "0") == "1",
            "resident": _resident,
            "cfg_policy": os.environ.get("HY_CFG_PADDING_POLICY", "separate"),
        },
        # Bounds collection + mesh open + fixture setup, which the test body cannot see.
        "pretest_setup_and_mesh_open_s": round(_t_fn_start - _T_IMPORT, 3),
        "phases_s": {k: round(v, 3) for k, v in _PHASES.items()},
        "phase_sum_s": round(phase_sum, 3),
        "in_function_total_s": round(in_function, 3),
        "unattributed_in_function_s": round(in_function - phase_sum, 3),
        # Subsets of generate_total_s, NOT additional time.
        "nested_s": {k: round(v, 3) for k, v in _NESTED.items()},
        "generate_residual_s": round(_PHASES.get("generate_total_s", 0.0) - nested_in_generate, 3),
        "dit_calls": len(dit_calls),
        "dit_first_call_s": round(dit_calls[0], 3) if dit_calls else None,
        "dit_steady_mean_s": round(sum(steady) / len(steady), 4) if steady else None,
    }
    print(f"[{label} phase breakdown] " + json.dumps(summary, sort_keys=True), flush=True)


# Single-device trace+2CQ defaults OFF: unlike the mesh case (weights sharded
# 4-way, ~4GB/chip, verified working with a 256MB trace region), a single chip
# holds the FULL ~16.6GB bf16 model with much less headroom, and a
# trace_region_size of 256MB OOMs during plain weight loading there (measured;
# only 1MB was confirmed to get past weight loading, and even that wasn't
# confirmed sufficient for an actual trace capture). Opt in at your own risk
# via HY_TRACE=1 -- don't flip this default without re-verifying on hardware.
_use_trace_single = os.environ.get("HY_TRACE", "0") == "1"
_SINGLE_DEVICE_PARAMS = {}
if _use_trace_single:
    _SINGLE_DEVICE_PARAMS = {
        "num_command_queues": 2,
        "trace_region_size": int(os.environ.get("HY_TRACE_REGION_SIZE", str(1 * 1024 * 1024))),
    }


@pytest.mark.parametrize("device_params", [_SINGLE_DEVICE_PARAMS], indirect=True)
def test_stage2b_gen(device):
    """One-chip smoke test: tiny resolution + truncated text (documented DRAM
    limitation -- see module docstring)."""
    _run_stage2b_gen(
        device,
        height=int(os.environ.get("HY_H", "64")),
        width=int(os.environ.get("HY_W", "64")),
        frames=int(os.environ.get("HY_FRAMES", "1")),
        steps=int(os.environ.get("HY_STEPS", "4")),
        trunc=int(os.environ.get("HY_TRUNC", "0")),
        outdir=os.environ.get("HY_OUT", "/tmp/hy15_stage2b"),
        label="stage2b",
        use_trace=_use_trace_single,
    )


# Mesh trace+2CQ is validated but defaults OFF pending a post-fix one-shot A/B.
# The matched 480p I2V/121f run showed a real ~5% steady-state gain (2.33 ->
# 2.21 s/step), but the old discarded warmup made crossover ~71 steps. The
# adapter now captures without that warmup and explicitly executes the first
# trace; keep this opt-in until 13f then 121f measurements confirm the new
# 50-step crossover on an idle Galaxy.
_use_trace_qb2 = os.environ.get("HY_TRACE", "0") == "1"
_QB2_DEVICE_PARAMS = {"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}
if _use_trace_qb2:
    # Only reserve a 2nd command queue + trace region when trace+2CQ is actually
    # requested -- both eat into the same tight per-chip DRAM budget the eager
    # path doesn't need (see the OOM notes in the module docstring), so don't
    # pay for them in a HY_TRACE=0 run.
    _QB2_DEVICE_PARAMS = {
        **_QB2_DEVICE_PARAMS,
        "num_command_queues": 2,
        "trace_region_size": int(os.environ.get("HY_TRACE_REGION_SIZE", str(256 * 1024 * 1024))),
    }


@pytest.mark.timeout(3600)  # full-res, 50-step, 54-layer x4-chip denoise loop is far slower than pytest.ini's 300s
@pytest.mark.parametrize("device_params", [_QB2_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_stage2b_gen_qb2(mesh_device):
    """QB2 flat-TP=4 variant: full resolution, NO text truncation -- the DiT is
    sharded (Megatron-style) across all 4 mesh devices, so the full-length text
    stream + real-resolution latent activations fit (see
    `real_weights/README.md` "RESUME ON QB2"). height/width default to this
    checkpoint's own computed default (480x848); num_frames defaults to 13, NOT
    this checkpoint's real default of 121 -- 121 (and 61, and 33) all OOM on
    live QB2 hardware, see module docstring. Override via the same HY_* env
    vars as `test_stage2b_gen` (HY_TRUNC is intentionally not read here)."""
    _run_stage2b_gen(
        mesh_device,
        height=int(os.environ.get("HY_H", "480")),
        width=int(os.environ.get("HY_W", "848")),
        frames=int(os.environ.get("HY_FRAMES", "13")),
        steps=int(os.environ.get("HY_STEPS", "50")),
        trunc=0,
        outdir=os.environ.get("HY_OUT", "/tmp/hy15_stage2b_qb2"),
        label="stage2b-qb2",
        use_trace=_use_trace_qb2,
    )
