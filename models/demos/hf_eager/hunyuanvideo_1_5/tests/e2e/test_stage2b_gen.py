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


def _pipeline_path():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    return snapshot_download(_COMMUNITY)  # cached if present, else downloads (~50GB)


def _run_stage2b_gen(device, *, height, width, frames, steps, trunc, outdir, label, use_trace):
    import numpy as np
    import torch
    from diffusers import HunyuanVideo15Pipeline
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from PIL import Image

    coerce_bf16()
    pipe = HunyuanVideo15Pipeline.from_pretrained(_pipeline_path(), torch_dtype=torch.bfloat16)
    real_tf = pipe.transformer
    tt = P.build_pipeline(device, real_tf)
    calls = {"n": 0, "device_runs": 0}

    class TTTransformer:
        """diffusers' Guider (`pipe.guider`) calls the transformer once PER
        CONDITION (conditional, unconditional, ...) as separate batch=1 Python
        calls within the same denoise step -- never as one pre-batched call.
        `guider.num_conditions` (read live, since CFG can be step-range-gated
        and so can vary across steps) says how many calls belong to the
        current step's group. We defer every call but the last one in a group,
        then run the WHOLE group as one real on-device batch and backfill the
        earlier calls' already-returned tensors in place via `Tensor.set_`
        (diffusers just stashed the reference; nothing reads it until after
        the group's last call returns). `_build_attn_bias`/`_trim_to_valid` in
        tt/pipeline.py were built for exactly this per-batch-item case.

        The captured-trace + 2CQ write/replay path (tt/pipeline.py's
        denoise_trace_setup/step/write_inputs/trace_execute), controlled by
        the use_trace param below, can replace the eager per-step tt.run()
        path. Defaults differ by caller (see _use_trace_single/_use_trace_qb2
        near the bottom of this file): ON for test_stage2b_gen_qb2 (mesh,
        verified correct on live QB2 hardware -- coherent video, same real-
        on-device-run count), OFF for test_stage2b_gen (single device --
        a 256MB trace region OOMs during plain weight loading there since the
        full unsharded model has far less headroom than the mesh's per-chip
        sharded weights; unverified even at the smaller 1MB size that does
        get past weight loading). Where verified (mesh, 13 frames/50 steps),
        traced measured NO speedup over eager: steady-state ~3.20s/it eager
        vs ~3.25s/it traced, and the one-time setup+warmup+capture overhead
        makes the total denoise loop slightly SLOWER (166s vs 187s) -- this
        workload is compute/CCL-bound, not host-dispatch-bound, so skipping
        repeated op dispatch has nothing to save here. Kept on for the mesh
        case anyway as validated infrastructure in case a future compute-side
        optimization shifts the bottleneck enough for this to start paying
        off."""

        def __init__(self, real, ttpipe, guider, use_trace=True):
            self.__dict__["_real"], self.__dict__["_tt"], self.__dict__["_guider"] = real, ttpipe, guider
            self.__dict__["_pending"] = []  # [(inp_dict, dtype, placeholder)] for the in-flight CFG group
            # Command-3 trace+2CQ path (default varies by caller, see below): captured
            # once on the first denoise step's batched group, then write+replay for
            # every step after. None means "not captured yet".
            self.__dict__["_use_trace"] = use_trace
            self.__dict__["_trace_id"] = None
            self.__dict__["_trace_out"] = None
            self.config, self.dtype = real.config, real.dtype

        def __getattr__(self, k):
            return getattr(self.__dict__["_real"], k)

        def __call__(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep_r=None,
            encoder_hidden_states_2=None,
            encoder_attention_mask_2=None,
            image_embeds=None,
            attention_kwargs=None,
            return_dict=True,
            **kw,
        ):
            calls["n"] += 1
            if trunc:  # smoke test: shrink text-stream activations to fit one chip
                encoder_hidden_states = encoder_hidden_states[:, :trunc]
                encoder_attention_mask = encoder_attention_mask[:, :trunc]
                encoder_hidden_states_2 = encoder_hidden_states_2[:, : max(1, trunc // 4)]
                encoder_attention_mask_2 = encoder_attention_mask_2[:, : max(1, trunc // 4)]
            inp = dict(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states_2=encoder_hidden_states_2,
                encoder_attention_mask_2=encoder_attention_mask_2,
                image_embeds=image_embeds,
                task="t2v",
            )
            dtype = hidden_states.dtype
            guider = self.__dict__["_guider"]
            n = int(guider.num_conditions) if guider is not None else 1

            if n <= 1:  # CFG disabled this step (or no guider) -- nothing to batch
                calls["device_runs"] += 1
                out = self.__dict__["_tt"].run(inp, granularity="composite").to(dtype)
                return Transformer2DModelOutput(sample=out) if return_dict else (out,)

            pending = self.__dict__["_pending"]
            placeholder = torch.empty(0, dtype=dtype)
            pending.append((inp, dtype, placeholder))
            if len(pending) < n:  # more conditions still to arrive this step
                return Transformer2DModelOutput(sample=placeholder) if return_dict else (placeholder,)

            group, self.__dict__["_pending"] = pending, []
            keys = group[0][0]
            batched_inp = {
                k: (torch.cat([g[0][k] for g in group], dim=0) if torch.is_tensor(keys[k]) else keys[k]) for k in keys
            }
            calls["device_runs"] += 1
            tt = self.__dict__["_tt"]
            if not self.__dict__["_use_trace"]:
                batched_out = tt.run(batched_inp, granularity="composite")
            elif self.__dict__["_trace_id"] is None:
                # First step's group: encode once (text/image conditioning, RoPE,
                # attn_bias -- all step-invariant, see tt/pipeline.py), warm up
                # (compiles/caches kernels, untraced), then capture ONE step.
                # This capture run's own execution IS step 0's real result.
                tt.denoise_trace_setup(batched_inp)
                tt.denoise_trace_step()
                tid = ttnn.begin_trace_capture(tt.device, cq_id=0)
                trace_out = tt.denoise_trace_step()
                ttnn.end_trace_capture(tt.device, tid, cq_id=0)
                self.__dict__["_trace_id"], self.__dict__["_trace_out"] = tid, trace_out
                batched_out = tt._unpatchify(trace_out, tt._resident["out_shape"])
            else:
                # Every later step: write the new latent/timestep on CQ1, replay
                # the captured trace on CQ0, read the (in-place refreshed) output.
                tt.denoise_write_inputs(batched_inp["hidden_states"], batched_inp["timestep"])
                tt.denoise_trace_execute(self.__dict__["_trace_id"], blocking=True)
                batched_out = tt._unpatchify(self.__dict__["_trace_out"], tt._resident["out_shape"])
            for (_, g_dtype, g_placeholder), row in zip(group[:-1], batched_out[:-1]):
                g_placeholder.set_(row.unsqueeze(0).to(g_dtype).clone())
            out = batched_out[-1].unsqueeze(0).to(dtype)
            return Transformer2DModelOutput(sample=out) if return_dict else (out,)

    pipe.transformer = TTTransformer(real_tf, tt, getattr(pipe, "guider", None), use_trace=use_trace)

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

    out = pipe(
        prompt="A cat walks on the grass, realistic",
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(0),
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


# Mesh trace+2CQ defaults ON: verified end-to-end on live QB2 hardware (480x848,
# 13 frames, 50 steps) -- coherent output, correct trace capture/replay across
# the 4-device mesh with CCL all-reduce embedded in the trace. Set HY_TRACE=0
# to fall back to the eager per-step path.
_use_trace_qb2 = os.environ.get("HY_TRACE", "1") == "1"
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
