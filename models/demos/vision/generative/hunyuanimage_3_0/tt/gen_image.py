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


def _force_trust_remote_code():
    """Make HF custom-code loads non-interactive (we've already accepted this
    model's code by loading it). Prevents the `Do you wish to run the custom
    code? [y/N]` prompt from hanging/aborting a non-TTY run."""
    try:
        import transformers.dynamic_module_utils as _dmu

        _dmu.resolve_trust_remote_code = lambda *a, **k: True
    except Exception:
        pass


def load_model(num_layers=32, model=None, float_layers=None):
    """Load the HF model (host) + tokenizer; float the host heads; truncate the
    decoder stack to `num_layers` (full=32). Returns the model.

    float_layers: cast the (truncated) decoder layers to fp32 so a HOST-golden
    forward has a consistent dtype with the fp32 heads (the fp32 inputs_embeds
    would otherwise hit `F.linear` fp32-input vs bf16-weight). Default = auto:
    True only when small (<=8 layers, i.e. the host-golden PCC regime); at full
    depth the layers run on TT (never on host), so leave them bf16 (fp32 x32 =
    ~320GB is infeasible). Matches hf_reference_prefill, which floats the golden."""
    torch.set_grad_enabled(False)
    _force_trust_remote_code()
    if model is None:
        model = ttpipe.load_reference_model()
    # tokenizer wrapper is required by prepare_model_inputs (_tkwrapper).
    # HF's HunyuanImage3ForCausalMM.load_tokenizer(tokenizer) does
    # `self._tkwrapper = TokenizerWrapper(tokenizer)` -- it wants a tokenizer
    # OBJECT, not a model-id string. Passing HF_MODEL_ID made TokenizerWrapper
    # raise, the bare `except Exception: pass` swallowed it, `_tkwrapper` stayed
    # None, and every render then died in prepare_model_inputs with
    # "'NoneType' object has no attribute 'apply_chat_template'" -- AFTER paying
    # a ~29-min weight-tilize build. Pass the real tokenizer, and fail LOUDLY.
    if getattr(model, "_tkwrapper", None) is None:
        tok = ttpipe.load_tokenizer()
        for meth in ("load_tokenizer", "set_tokenizer"):
            fn = getattr(model, meth, None)
            if callable(fn):
                fn(tok)
                break
        else:
            raise RuntimeError(
                "HF model exposes neither load_tokenizer nor set_tokenizer; "
                "cannot initialise _tkwrapper for prepare_model_inputs."
            )
        if getattr(model, "_tkwrapper", None) is None:
            raise RuntimeError(
                "tokenizer wrapper still unset after load_tokenizer(); "
                "prepare_model_inputs would fail at apply_chat_template."
            )
    if num_layers is not None and num_layers < len(model.model.layers):
        model.model.layers = nn.ModuleList(list(model.model.layers[:num_layers]))
    _float_host_heads(model)
    if float_layers is None:
        float_layers = len(model.model.layers) <= 8
    if float_layers:
        model.model.layers.float()
    model.eval()
    return model


def install_tt_layer_stack(model, tt_pipe, use_trace=False):
    """Monkeypatch HunyuanImage3Model.forward so the decoder-layer loop runs on
    TT. Handles CFG (bsz>1) by running one TT forward per sample. use_trace=True
    uses the host-free traced replay (image_trace_setup on first call, then
    image_trace_step); the trace is captured once and reused for every sample/step
    (cos/sin/mask are constant). Returns an uninstall fn."""
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
        # CFG-parallel: run cond+uncond as ONE bsz=2 forward. cos/sin/mask are identical
        # across CFG samples (position-based, not token-content), so share sample 0's.
        # Halves the per-layer TP collectives. HUNYUAN_CFG_PARALLEL=1 to enable.
        if os.environ.get("HUNYUAN_CFG_PARALLEL") == "1" and bsz > 1:
            emb = inputs_embeds.to(torch.float32)  # [bsz, S, H]
            cos0, sin0 = cos[0].to(torch.float32), sin[0].to(torch.float32)
            mask0 = attention_mask[0] if attention_mask is not None else None  # [1,S,S], shared
            if use_trace:
                if getattr(tt_pipe, "_img_trace", None) is None:
                    tt_pipe.image_trace_setup(emb, cos0, sin0, mask0)
                hidden = tt_pipe.image_trace_step(emb)
            else:
                hidden = tt_pipe.forward_image(emb, cos0, sin0, mask0)
            hidden = hidden.to(inputs_embeds.dtype).to(inputs_embeds.device)
        else:
            outs = []
            for i in range(bsz):
                emb_i = inputs_embeds[i : i + 1].to(torch.float32)
                cos_i, sin_i = cos[i].to(torch.float32), sin[i].to(torch.float32)
                mask_i = attention_mask[i] if attention_mask is not None else None  # [1,S,S]
                if use_trace:
                    if getattr(tt_pipe, "_img_trace", None) is None:
                        tt_pipe.image_trace_setup(emb_i, cos_i, sin_i, mask_i)
                    h_i = tt_pipe.image_trace_step(emb_i)
                else:
                    h_i = tt_pipe.forward_image(emb_i, cos_i, sin_i, mask_i)
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


def build_tt_backed_model(mesh_device, num_layers=32, model=None, use_trace=False):
    """Load the model + build the TT decoder pipeline + install the TT layer
    stack. use_trace=True enables the host-free traced replay. Returns
    (model, tt_pipe, uninstall_fn)."""
    model = load_model(num_layers=num_layers, model=model)
    tt_pipe = ttpipe.HunyuanImage3Pipeline(mesh_device, model, num_layers=num_layers)
    uninstall = install_tt_layer_stack(model, tt_pipe, use_trace=use_trace)
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
# e2e prompt->image PCC: run the WHOLE diffusion loop (->VAE->pixels) twice --
# pure-host (HF layers) vs pure-TT -- and compare the final latent + decoded
# image. This exercises the REAL image path (loop + CFG + scheduler + VAE), not
# just one step.
#
# Correctness strategy for the FULL 32-layer model (why the default is reduced
# depth): a full-depth HOST golden image is infeasible (running the 80B MoE on
# CPU for ~4096 tokens x N steps x 2 CFG is many hours), so this gate runs at
# REDUCED depth (num_layers, default 2) where the host trajectory is tractable,
# and the full-image path itself is fully exercised. Full-DEPTH correctness rests
# on: (a) per-block PCC >= 0.997 (the model's established methodology -- 32
# identical blocks each validated => the stack is correct), (b) this reduced-depth
# full-image-path gate, and (c) a full-depth render sanity/CLIP check
# (generate_image + finite/non-degenerate output, optional CLIP-vs-prompt). Set
# num_layers=32 to run full depth (slow: host trajectory dominates).
# --------------------------------------------------------------------------
def _reset_scheduler(sched, num_steps):
    sched.set_timesteps(num_steps, device="cpu")
    # defensive: ensure a clean step counter for a fresh trajectory
    for attr in ("_step_index", "_begin_index"):
        if hasattr(sched, attr):
            setattr(sched, attr, None)
    return sched.timesteps


def _run_trajectory(model, kwargs, attn_mask, sched, timesteps, latents, guidance, cfg):
    """Advance one full diffusion trajectory with WHATEVER layer stack is
    currently active (host HF layers, or TT if the monkeypatch is installed).
    Returns (final_latents, step0_velocity)."""
    vel0 = None
    for i, t in enumerate(timesteps):
        pred = run_velocity_once(model, kwargs, attn_mask, latents, t, cfg)  # [cfg,32,h,w]
        if i == 0:
            vel0 = pred
        if cfg == 2:
            pc, pu = pred.chunk(2)
            pred = pu + guidance * (pc - pu)
        latents = sched.step(pred, t, latents, return_dict=False)[0]
    return latents, vel0


def e2e_image_pcc(
    mesh_device,
    prompt=DEFAULT_PROMPT,
    image_size=(1024, 1024),
    num_layers=2,
    num_steps=8,
    seed=0,
    pcc_target=0.95,
    decode=True,
    out_prefix=None,
    use_trace_tt=False,
):
    """Full prompt->image e2e PCC (TT vs pure-host) over the real diffusion loop.
    Returns pcc_final_latent, pcc_image, pcc_velocity_step0, image_finite, etc."""
    model = load_model(num_layers=num_layers)
    kwargs, attn_mask = prepare_gen_image_inputs(model, prompt, list(image_size), seed=seed)
    cfg = 2
    guidance = float(model.generation_config.diff_guidance_scale)
    sched = model.pipeline.scheduler
    h = kwargs["batch_gen_image_info"][0].token_height
    w = kwargs["batch_gen_image_info"][0].token_width
    lc = int(model.config.vae["latent_channels"])

    def latents0():
        g = torch.Generator("cpu").manual_seed(seed)
        return torch.randn(1, lc, h, w, generator=g, dtype=torch.float32)

    # pure-host trajectory (HF layers; monkeypatch NOT installed)
    ts = _reset_scheduler(sched, num_steps)
    host_lat, host_vel0 = _run_trajectory(model, kwargs, attn_mask, sched, ts, latents0(), guidance, cfg)

    # pure-TT trajectory (TT stubs), same seed/start (optionally traced)
    tt_pipe = ttpipe.HunyuanImage3Pipeline(mesh_device, model, num_layers=num_layers)
    uninstall = install_tt_layer_stack(model, tt_pipe, use_trace=use_trace_tt)
    try:
        ts = _reset_scheduler(sched, num_steps)
        tt_lat, tt_vel0 = _run_trajectory(model, kwargs, attn_mask, sched, ts, latents0(), guidance, cfg)
    finally:
        if use_trace_tt and hasattr(tt_pipe, "image_trace_release"):
            tt_pipe.image_trace_release()
        uninstall()

    _, pcc_vel0 = comp_pcc(host_vel0, tt_vel0, pcc_target)  # step-0 velocity (identical start)
    _, pcc_lat = comp_pcc(host_lat, tt_lat, pcc_target)  # final latent (whole trajectory)
    result = {
        "pcc_velocity_step0": pcc_vel0,
        "pcc_final_latent": pcc_lat,
        "num_layers": num_layers,
        "num_steps": num_steps,
        "token_hw": (h, w),
    }

    if decode:

        def _dec(lat):
            lat = lat / model.vae.config.scaling_factor
            if hasattr(model.vae, "ffactor_temporal"):
                lat = lat.unsqueeze(2)
            with torch.no_grad():
                img = model.vae.decode(lat.float(), return_dict=False)[0]
            if hasattr(model.vae, "ffactor_temporal"):
                img = img.squeeze(2)
            return img

        host_img, tt_img = _dec(host_lat), _dec(tt_lat)
        _, pcc_img = comp_pcc(host_img, tt_img, pcc_target)
        result["pcc_image"] = pcc_img
        result["image_finite"] = bool(torch.isfinite(tt_img).all())
        result["image_std"] = float(tt_img.float().std())  # non-degenerate sanity
        if out_prefix:
            pp = model.pipeline.image_processor.postprocess
            pp(host_img, output_type="pil", do_denormalize=[True] * host_img.shape[0])[0].save(f"{out_prefix}_host.png")
            pp(tt_img, output_type="pil", do_denormalize=[True] * tt_img.shape[0])[0].save(f"{out_prefix}_tt.png")

    result["pcc_ok"] = bool(result["pcc_final_latent"] >= pcc_target and (not decode or result["image_finite"]))
    return result


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

    # TOTAL end-to-end latency = the diffusion loop + VAE decode + postprocess
    # (excludes the one-time cold model load). This is the real "s/image"
    # wall-clock that replaces the earlier projection.
    t_gen_start = time.time()
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
    loop_s = time.time() - t_gen_start

    # VAE decode (host, fp32) -> pixels
    t_vae = time.time()
    sf = model.vae.config.scaling_factor
    if sf:
        latents = latents / sf
    if hasattr(model.vae, "ffactor_temporal"):
        latents = latents.unsqueeze(2)  # [1,32,1,h,w]
    # EXPERIMENT (VAE precision): HUNYUAN_VAE_AUTOCAST = "bf16" | "fp16" wraps the
    # host VAE decode in CPU autocast (fp32 baseline otherwise). bf16 conv3d can hit
    # oneDNN AVX512-BF16 on host -> may cut the ~57s decode. Correctness = image
    # sanity (lossy; fine for a demo). Isolated lever, run separately.
    _vae_ac = os.environ.get("HUNYUAN_VAE_AUTOCAST", "").lower()
    with torch.no_grad():
        if _vae_ac in ("bf16", "fp16"):
            _dt = torch.bfloat16 if _vae_ac == "bf16" else torch.float16
            with torch.autocast(device_type="cpu", dtype=_dt):
                image = model.vae.decode(latents.float(), return_dict=False)[0]  # [1,3,1,H,W]
        else:
            image = model.vae.decode(latents.float(), return_dict=False)[0]  # [1,3,1,H,W]
    if hasattr(model.vae, "ffactor_temporal"):
        image = image.squeeze(2)
    image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])[0]
    vae_decode_s = time.time() - t_vae

    out_path = os.path.abspath(out_path)
    image.save(out_path)
    total_latency_s = time.time() - t_gen_start
    timing = {
        "steps": steps,
        "total_latency_s": total_latency_s,  # THE headline: full e2e s/image
        "loop_s": loop_s,
        "vae_decode_s": vae_decode_s,
        "ms_per_step_mean": sum(per_step_ms) / len(per_step_ms),
        "per_step_ms": per_step_ms,
        "token_hw": (h, w),
        "seq_len": int(kwargs["input_ids"].shape[1]),
        "out_path": out_path,
    }
    if hasattr(tt_pipe, "image_trace_release"):
        tt_pipe.image_trace_release()

    # Parseable sentinel (mirrors the manual track's TRACE_PER_TOKEN_MS convention).
    print(f"E2E_T2I_TOTAL_LATENCY_S={total_latency_s:.4f}")
    print(
        f"saved {out_path}  (total {total_latency_s:.1f} s/image | loop {loop_s:.1f} s "
        f"@ {timing['ms_per_step_mean']:.0f} ms/step x{steps} | vae {vae_decode_s:.1f} s)"
    )
    return image, timing
