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
import os
import shutil
import subprocess

import pytest

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.tests.e2e.test_real_weight_pcc import coerce_bf16
from models.demos.hf_eager.hunyuanvideo_1_5.tt import pipeline as P

_COMMUNITY = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
_I2V_COMMUNITY = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"


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
    _i2v = os.environ.get("HY_I2V") == "1"
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
    real_tf = pipe.transformer
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
        pipe.vae = _vd.TTVAEDecodeAdapter(pipe.vae, vae_dev)
        placement = "separate chips" if vae_dev is not device else "shared with DiT"
        print(f"[{label}] VAE decode: ON DEVICE (ttnn) on {list(vae_dev.get_device_ids())} ({placement})", flush=True)

    if os.environ.get("HY_TT_QWEN", "0") == "1":
        from models.demos.hf_eager.hunyuanvideo_1_5.tt import qwen_encoder as _qe

        # Prefer a dedicated Qwen submesh (carved on spare chips at sp<=3). When none
        # exists (sp=4 fills every mesh row), text-encode stays on CPU by default --
        # for a ONE-SHOT video that's actually faster (A/B: host 5:59 vs shared-mesh
        # device 6:58, the 7B weight-load + first-compile + FSDP-gather outweigh the
        # one-time CPU encode). Opt in with HY_TT_QWEN_SHARED=1 to run Qwen on the DiT's
        # OWN full mesh (TP=4 + FSDP across the other axis, weights co-resident, no
        # overlapping context) -- worthwhile only for a SERVED / multi-prompt setup
        # where the load+compile amortize.
        qwen_dev = _qe.HY_QWEN_SUBMESH
        if qwen_dev is None and os.environ.get("HY_TT_QWEN_SHARED", "0") == "1":
            qwen_dev = device
        if qwen_dev is not None:
            _pl = "separate submesh" if _qe.HY_QWEN_SUBMESH is not None else "shared DiT mesh (TP=4+FSDP)"
            pipe.text_encoder = _qe.TTQwenTextEncoderAdapter(pipe.text_encoder, qwen_dev)
            print(f"[{label}] Qwen text-encode: ON DEVICE ({_pl}) on {list(qwen_dev.get_device_ids())}", flush=True)
        else:
            print(
                f"[{label}] no Qwen submesh at this mesh; text-encode on CPU (HY_TT_QWEN_SHARED=1 to reuse the DiT mesh)",
                flush=True,
            )

    # Optional: run the i2v SigLIP image-encoder on device (HY_TT_SIGLIP=1). The 27-layer
    # vision transformer runs REPLICATED on the full DiT mesh -- shared with the resident DiT,
    # the same way the VAE-decode and text-encode adapters run -- so it needs no spare chip and
    # works at sp=8xtp=4. (patch-embed + post-LN stay on host; the adapter PCC-self-checks vs
    # the host encoder on first call. See tt/siglip_encoder.py, validated by
    # tests/pcc/test_siglip.py at PCC ~0.995.)
    if _i2v and os.environ.get("HY_TT_SIGLIP", "0") == "1":
        from models.demos.hf_eager.hunyuanvideo_1_5.tt import siglip_encoder as _se

        pipe.image_encoder = _se.TTSiglipImageEncoderAdapter(pipe.image_encoder, device)
        print(f"[{label}] SigLIP image-encode: ON DEVICE (ttnn, replicated on full mesh)", flush=True)

    _prompt = os.environ.get("HY_PROMPT", "A cat walks on the grass, realistic")
    _neg = os.environ.get("HY_NEG_PROMPT") or None
    _pkw = {"negative_prompt": _neg} if _neg is not None else {}
    if _i2v:
        # i2v derives the output resolution from the input image + the checkpoint's
        # target_size (640->480p, 960->720p), so it takes NO height/width -- it computes
        # them via video_processor.calculate_default_height_width and crop-resizes the image.
        _pkw["image"] = _load_i2v_image(height, width, label)
    _dims = {} if _i2v else {"height": height, "width": width}
    print(f"[{label}] prompt: {_prompt}\n[{label}] negative: {_neg}", flush=True)
    out = pipe(
        prompt=_prompt,
        num_frames=frames,
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(0),
        **_dims,
        **_pkw,
    ).frames[0]
    print(
        f"\n[{label}] generated {len(out)} frames; transformer __call__s={calls['n']}, "
        f"real on-device runs={calls['device_runs']}",
        flush=True,
    )

    os.makedirs(outdir, exist_ok=True)
    pil = [
        f if isinstance(f, Image.Image) else Image.fromarray((np.asarray(f).clip(0, 1) * 255).astype("uint8"))
        for f in out
    ]
    for i, im in enumerate(pil):
        im.save(f"{outdir}/frame_{i:03d}.png")
    pil[0].save(f"{outdir}/tt_blackhole.gif", save_all=True, append_images=pil[1:], duration=125, loop=0)

    fps = int(os.environ.get("HY_FPS", "24"))
    mp4_path = f"{outdir}/tt_blackhole.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                f"{outdir}/frame_%03d.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                mp4_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(
            f"[{label}] SAVED {len(pil)} frames + tt_blackhole.gif + tt_blackhole.mp4 ({fps}fps) -> {outdir}",
            flush=True,
        )
    else:
        print(
            f"[{label}] SAVED {len(pil)} frames + tt_blackhole.gif -> {outdir} (ffmpeg not found, no mp4)", flush=True
        )


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


# Mesh trace+2CQ is validated but defaults OFF for best one-shot latency. On the
# compute/CCL-bound real workload, steady-state was unchanged and capture/warmup
# made a measured 13-frame/50-step run slower overall (166s eager vs 187s traced).
# Set HY_TRACE=1 to exercise the resident-buffer trace infrastructure.
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
