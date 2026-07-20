# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hybrid text->image generation for `tencent/HunyuanImage-3.0` on Tenstorrent.

HunyuanImage-3.0 generates images by DIFFUSION-IN-TRANSFORMER (`gen_image` mode):
a FlowMatch (Euler) denoising loop over VAE latent tokens, classifier-free
guidance (cfg_factor=2), a timestep-conditioned velocity head (`ragged_final_layer`),
then `AutoencoderKLConv3D.decode` -> pixels. This is a DIFFERENT forward from the
`gen_text` greedy/prefill path the rest of this port targets.

HYBRID split (this module):
  * TT Galaxy runs the 32 MoE decoder layers  -> HunyuanImage3Pipeline.forward_image
    (via a monkeypatch of HunyuanImage3Model.forward; see install_tt_layer_stack).
  * Host (torch) runs EVERYTHING else, reusing the upstream code verbatim:
    tokenizer + chat template (prepare_model_inputs), wte, image-token
    instantiation (patch_embed) + timestep tokens, the 2D image RoPE, the block
    attention mask, the velocity head (time_embed_2 + final_layer), the
    FlowMatchDiscreteScheduler + CFG, and the VAE decode + PNG postprocess.

The diffusion loop here is a TRIMMED copy of HunyuanImage3Text2ImagePipeline.__call__
that (1) forces `first_step=True` and re-feeds the FULL sequence EVERY step so no
cross-step KV cache is needed across the host/TT boundary (mathematically identical
to upstream -- the fixed text prefix is cheaply recomputed each step vs the ~4096
image tokens), and (2) calls model.forward directly to avoid the upstream
`torch.autocast(device_type="cuda")` (this box has no CUDA).

STATUS: first-draft, pending on-device validation (the device was occupied during
authoring). Assumptions flagged with `# VERIFY:` are the ones most likely to need a
tweak on first run.

Correctness gate: `step_pcc` compares the TT per-step velocity against the pure-host
(first-N HF layers) velocity at reduced depth -- the Stage-2 milestone.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from models.common.utility_functions import comp_pcc
from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as ttpipe

HF_MODEL_ID = ttpipe.HF_MODEL_ID
DEFAULT_PROMPT = "A serene mountain lake at sunrise, photorealistic, ultra detailed."

# Submodules that stay on host for the image path (everything except the 32
# decoder layers). Floated to fp32 so the AdaGN convs / embeds run on CPU.
_HOST_HEAD_ATTRS = (
    "model.wte",
    "time_embed",
    "time_embed_2",
    "timestep_emb",
    "patch_embed",
    "final_layer",
    "vae",
)


# --------------------------------------------------------------------------
# model loading + host/device wiring
# --------------------------------------------------------------------------
def _get_submodule(model, dotted):
    obj = model
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _float_host_heads(model):
    """Cast the host-side (non-layer) submodules to fp32 for CPU execution.
    The 32 decoder layers are left as-is (they run on TT, not host)."""
    for attr in _HOST_HEAD_ATTRS:
        try:
            _get_submodule(model, attr).float()
        except AttributeError:
            pass  # optional submodule (e.g. no separate vae attr) -> skip


def load_model(num_layers=32, model=None):
    """Load the HF model (host) + tokenizer; float the host heads; truncate the
    decoder stack to `num_layers` (full=32). Returns the model."""
    if model is None:
        model = ttpipe.load_reference_model()
    # tokenizer wrapper is required by prepare_model_inputs (_tkwrapper).
    if getattr(model, "_tkwrapper", None) is None:
        for meth in ("load_tokenizer", "set_tokenizer"):
            fn = getattr(model, meth, None)
            if callable(fn):
                try:
                    fn(HF_MODEL_ID)  # VERIFY: exact tokenizer-load entrypoint/arg
                    break
                except Exception:
                    pass
    if num_layers is not None and num_layers < len(model.model.layers):
        model.model.layers = nn.ModuleList(list(model.model.layers[:num_layers]))
    _float_host_heads(model)
    model.eval()
    return model


def install_tt_layer_stack(model, tt_pipe):
    """Monkeypatch HunyuanImage3Model.forward so the decoder-layer loop runs on
    TT via tt_pipe.forward_image. Handles CFG (bsz>1) by running one TT forward
    per sample (each has its own rope row + mask row). Returns an uninstall fn."""
    from transformers.modeling_outputs import BaseModelOutputWithPast

    inner = model.model  # HunyuanImage3Model
    had_own = "forward" in inner.__dict__  # was there an instance-level override?
    prev = inner.__dict__.get("forward")

    def _tt_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        custom_pos_emb=None,
        mode="gen_text",
        first_step=None,
        gen_timestep_scatter_index=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        cos, sin = custom_pos_emb  # each [bsz, S, head_dim] (already gathered)
        bsz = inputs_embeds.shape[0]
        outs = []
        for i in range(bsz):
            mask_i = attention_mask[i] if attention_mask is not None else None  # [1,S,S]
            h_i = tt_pipe.forward_image(
                inputs_embeds[i : i + 1].to(torch.float32),
                cos[i].to(torch.float32),
                sin[i].to(torch.float32),
                mask_i,
            )
            outs.append(h_i)
        hidden = torch.cat(outs, dim=0).to(inputs_embeds.dtype).to(inputs_embeds.device)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=None, hidden_states=None, attentions=None
        )

    import types

    inner.forward = types.MethodType(_tt_forward, inner)

    def _uninstall():
        if had_own:
            inner.forward = prev
        else:
            del inner.__dict__["forward"]

    return _uninstall


def build_tt_backed_model(mesh_device, num_layers=32, model=None):
    """Load the model + build the TT decoder pipeline + install the TT layer
    stack. Returns (model, tt_pipe, uninstall_fn)."""
    model = load_model(num_layers=num_layers, model=model)
    tt_pipe = ttpipe.HunyuanImage3Pipeline(mesh_device, model, num_layers=num_layers)
    uninstall = install_tt_layer_stack(model, tt_pipe)
    return model, tt_pipe, uninstall


# --------------------------------------------------------------------------
# gen_image inputs + a single velocity step (shared by the gate and the loop)
# --------------------------------------------------------------------------
def prepare_gen_image_inputs(model, prompt, image_size, seed=0):
    """Build the static (across diffusion steps) gen_image model inputs + the
    block attention mask. Reuses the upstream chat template / rope / mask code."""
    kwargs = model.prepare_model_inputs(
        prompt=prompt,
        mode="gen_image",
        image_size=image_size,  # [H, W]; snapped to the nearest supported reso
        bot_task="image",
        seed=seed,
    )
    input_ids = kwargs["input_ids"]
    attn_mask = model._prepare_attention_mask_for_generation(
        input_ids, model.generation_config, model_kwargs=kwargs
    )  # [bsz, 1, S, S] bool: text causal + image-block bidirectional
    return kwargs, attn_mask


def run_velocity_once(model, kwargs, attn_mask, latents, t, cfg_factor):
    """One denoising-step forward -> velocity `diffusion_prediction` [cfg,32,h,w].
    Forces first_step=True + full sequence (works with or without the TT stack)."""
    latent_in = torch.cat([latents] * cfg_factor).to(torch.float32)
    t_expand = t.repeat(latent_in.shape[0])
    out = model(
        input_ids=kwargs["input_ids"],
        attention_mask=attn_mask,
        position_ids=kwargs["position_ids"],
        past_key_values=None,
        use_cache=False,
        custom_pos_emb=kwargs["custom_pos_emb"],
        mode="gen_image",
        first_step=True,
        images=latent_in,
        image_mask=kwargs["image_mask"],
        timestep=t_expand,
        gen_timestep_scatter_index=kwargs["gen_timestep_scatter_index"],
        return_dict=True,
    )
    return out.diffusion_prediction.to(torch.float32)  # [cfg, 32, h, w]


# --------------------------------------------------------------------------
# Stage 2 gate: per-step velocity PCC (TT vs pure-host) at reduced depth
# --------------------------------------------------------------------------
def step_pcc(mesh_device, prompt=DEFAULT_PROMPT, image_size=(1024, 1024), num_layers=2, seed=0, pcc_target=0.95):
    """Compare the TT per-diffusion-step velocity against the pure-host (first-N
    HF layers) velocity, at reduced depth. Returns a dict with pcc + tensors.

    Reduced depth keeps it cheap (N layers) while exercising the exact gen_image
    forward (2D image RoPE + block mask + velocity head)."""
    model = load_model(num_layers=num_layers)
    kwargs, attn_mask = prepare_gen_image_inputs(model, prompt, list(image_size), seed=seed)
    cfg = 2  # gen_image always CFG=2 here

    # deterministic latents at a mid timestep
    sched = model.pipeline.scheduler
    sched.set_timesteps(50, device="cpu")
    t = sched.timesteps[len(sched.timesteps) // 2]
    h, w = kwargs["batch_gen_image_info"][0].token_height, kwargs["batch_gen_image_info"][0].token_width
    g = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(1, int(model.config.vae["latent_channels"]), h, w, generator=g, dtype=torch.float32)

    # host golden: original forward over the (truncated-to-N) HF layers
    vel_ref = run_velocity_once(model, kwargs, attn_mask, latents, t, cfg)

    # TT: same inputs through the TT decoder stack
    tt_pipe = ttpipe.HunyuanImage3Pipeline(mesh_device, model, num_layers=num_layers)
    uninstall = install_tt_layer_stack(model, tt_pipe)
    try:
        vel_tt = run_velocity_once(model, kwargs, attn_mask, latents, t, cfg)
    finally:
        uninstall()

    ok, pcc = comp_pcc(vel_ref, vel_tt, pcc_target)
    return {
        "pcc": pcc,
        "pcc_ok": ok,
        "num_layers": num_layers,
        "token_hw": (h, w),
        "seq_len": int(kwargs["input_ids"].shape[1]),
    }


# --------------------------------------------------------------------------
# Stage 3: full hybrid diffusion loop -> PNG
# --------------------------------------------------------------------------
def generate_image(
    model,
    tt_pipe,
    prompt=DEFAULT_PROMPT,
    image_size=(1024, 1024),
    num_inference_steps=None,
    guidance_scale=None,
    seed=0,
    out_path="hunyuan_t2i.png",
):
    """Full hybrid text->image render -> saves a PNG. Assumes the TT layer stack
    is already installed on `model` (via build_tt_backed_model). Returns the PIL
    image + timing dict."""
    import time

    gc = model.generation_config
    steps = int(num_inference_steps or gc.diff_infer_steps)
    guidance = float(guidance_scale if guidance_scale is not None else gc.diff_guidance_scale)
    cfg = 2 if guidance > 1.0 else 1

    kwargs, attn_mask = prepare_gen_image_inputs(model, prompt, list(image_size), seed=seed)

    pipe = model.pipeline
    sched = pipe.scheduler
    sched.set_timesteps(steps, device="cpu")
    timesteps = sched.timesteps

    h = kwargs["batch_gen_image_info"][0].token_height
    w = kwargs["batch_gen_image_info"][0].token_width
    g = torch.Generator("cpu").manual_seed(seed)
    latents = torch.randn(1, int(model.config.vae["latent_channels"]), h, w, generator=g, dtype=torch.float32)

    per_step_ms = []
    for i, t in enumerate(timesteps):
        t0 = time.time()
        pred = run_velocity_once(model, kwargs, attn_mask, latents, t, cfg)  # [cfg,32,h,w]
        if cfg == 2:
            pred_cond, pred_uncond = pred.chunk(2)
            pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        latents = sched.step(pred, t, latents, return_dict=False)[0]
        per_step_ms.append(1000.0 * (time.time() - t0))
        print(f"  step {i + 1}/{steps}  t={float(t):.2f}  {per_step_ms[-1]:.0f} ms")

    # VAE decode (host, fp32) -> pixels
    sf = model.vae.config.scaling_factor
    if sf:
        latents = latents / sf
    if hasattr(model.vae, "ffactor_temporal"):
        latents = latents.unsqueeze(2)  # [1,32,1,h,w]
    with torch.no_grad():
        image = model.vae.decode(latents.float(), return_dict=False)[0]  # [1,3,1,H,W]
    if hasattr(model.vae, "ffactor_temporal"):
        image = image.squeeze(2)
    image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])[0]

    out_path = os.path.abspath(out_path)
    image.save(out_path)
    timing = {
        "steps": steps,
        "s_per_image": sum(per_step_ms) / 1000.0,
        "ms_per_step_mean": sum(per_step_ms) / len(per_step_ms),
        "token_hw": (h, w),
        "seq_len": int(kwargs["input_ids"].shape[1]),
        "out_path": out_path,
    }
    print(f"saved {out_path}  ({timing['s_per_image']:.1f} s/image, {timing['ms_per_step_mean']:.0f} ms/step)")
    return image, timing
