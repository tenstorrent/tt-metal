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

    HY_H=480 HY_W=832 HY_FRAMES=13 HY_STEPS=50 HY_OUT=/path \
        pytest tests/e2e/test_stage2b_gen.py::test_stage2b_gen_qb2 -s
"""
import os

import pytest

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.tests.e2e.test_real_weight_pcc import coerce_bf16
from models.demos.hf_eager.hunyuanvideo_1_5.tt import pipeline as P

_COMMUNITY = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"


def _pipeline_path():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    return snapshot_download(_COMMUNITY)  # cached if present, else downloads (~50GB)


def _run_stage2b_gen(device, *, height, width, frames, steps, trunc, outdir, label):
    import numpy as np
    import torch
    from diffusers import HunyuanVideo15Pipeline
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from PIL import Image

    coerce_bf16()
    pipe = HunyuanVideo15Pipeline.from_pretrained(_pipeline_path(), torch_dtype=torch.bfloat16)
    real_tf = pipe.transformer
    tt = P.build_pipeline(device, real_tf)
    calls = {"n": 0}

    class TTTransformer:
        def __init__(self, real, ttpipe):
            self.__dict__["_real"], self.__dict__["_tt"] = real, ttpipe
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
            outs = []
            for b in range(hidden_states.shape[0]):  # per-sample (handles CFG batching)
                inp = dict(
                    hidden_states=hidden_states[b : b + 1],
                    timestep=timestep[b : b + 1],
                    encoder_hidden_states=encoder_hidden_states[b : b + 1],
                    encoder_attention_mask=encoder_attention_mask[b : b + 1],
                    encoder_hidden_states_2=encoder_hidden_states_2[b : b + 1],
                    encoder_attention_mask_2=encoder_attention_mask_2[b : b + 1],
                    image_embeds=image_embeds[b : b + 1],
                    task="t2v",
                )
                outs.append(self.__dict__["_tt"].run(inp, granularity="composite").to(hidden_states.dtype))
            out = torch.cat(outs, dim=0)
            return Transformer2DModelOutput(sample=out) if return_dict else (out,)

    pipe.transformer = TTTransformer(real_tf, tt)

    out = pipe(
        prompt="A cat walks on the grass, realistic",
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(0),
    ).frames[0]
    print(f"\n[{label}] generated {len(out)} frames; on-device transformer calls={calls['n']}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    pil = [
        f if isinstance(f, Image.Image) else Image.fromarray((np.asarray(f).clip(0, 1) * 255).astype("uint8"))
        for f in out
    ]
    for i, im in enumerate(pil):
        im.save(f"{outdir}/frame_{i:03d}.png")
    pil[0].save(f"{outdir}/tt_blackhole.gif", save_all=True, append_images=pil[1:], duration=125, loop=0)
    print(f"[{label}] SAVED {len(pil)} frames + tt_blackhole.gif -> {outdir}", flush=True)


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
    )


@pytest.mark.timeout(3600)  # full-res, 50-step, 54-layer x4-chip denoise loop is far slower than pytest.ini's 300s
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [4], indirect=True)
def test_stage2b_gen_qb2(mesh_device):
    """QB2 flat-TP=4 variant: full resolution, NO text truncation -- the DiT is
    sharded (Megatron-style) across all 4 mesh devices, so the full-length text
    stream + real-resolution latent activations fit (see
    `real_weights/README.md` "RESUME ON QB2"). Defaults match the README's
    real-resolution target; override via the same HY_* env vars as
    `test_stage2b_gen` (HY_TRUNC is intentionally not read here)."""
    _run_stage2b_gen(
        mesh_device,
        height=int(os.environ.get("HY_H", "480")),
        width=int(os.environ.get("HY_W", "832")),
        frames=int(os.environ.get("HY_FRAMES", "13")),
        steps=int(os.environ.get("HY_STEPS", "50")),
        trunc=0,
        outdir=os.environ.get("HY_OUT", "/tmp/hy15_stage2b_qb2"),
        label="stage2b-qb2",
    )
