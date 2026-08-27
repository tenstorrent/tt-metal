# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stage 3 of the host-glue port: keep the transformer hidden ON-DEVICE across the
whole diffusion loop (the #1 T2I perf lever, ~57% host-glue).

Today per step the HF path runs patch_embed + final_layer on host CPU and round-trips
the [cfg,~4116,4096] inputs_embeds (up) + hidden (down) every step. This module does it
all on-device:

  setup (once):   base = wte(input_ids) [static]; find the contiguous image block
                  [start:end]; pre-upload RoPE cos/sin + attn mask + the static suffix;
                  build PatchEmbedTT + FinalLayerTT.
  per step:       img = PatchEmbedTT(latent, time_embed(t)) on-device; assemble
                  inputs_embeds on-device via ROW_MAJOR concat([prefix, img, suffix])
                  (image block is contiguous) with the per-step timestep_emb(t) token
                  patched into the tiny prefix; -> tilize -> run decoder layers (hidden
                  stays on device) -> slice the image-position hidden -> FinalLayerTT
                  (+time_embed_2(t)) -> download ONLY the velocity [cfg,32,64,64].

Removes host patch_embed + final_layer + the two ~68 MB/step transfers. Correctness:
test_host_glue_stage3.py compares this velocity vs the existing host-reference path
(PatchEmbedTT/FinalLayerTT each already PCC 0.9997 standalone).

NOTE embedders: patch_embed uses `time_embed(t)`, the <timestep> meta-token uses a
SEPARATE `timestep_emb(t)`, and final_layer uses `time_embed_2(t)` — three distinct heads.
"""
from __future__ import annotations

import torch

import ttnn

from .host_glue_tt import _repl, build_final_layer, build_patch_embed
from .pipeline import _mesh_to_torch


def _forward_image_device(tt_pipe, inputs_embeds_tt, cos_tt, sin_tt, mask_tt):
    """Run the N graduated decoder layers on an ALREADY-on-device inputs_embeds and
    return the DEVICE hidden [1,S,hidden] (no embed upload, no download)."""
    hidden = inputs_embeds_tt
    for layer in tt_pipe.layers:
        hidden = layer(hidden, custom_pos_emb=(cos_tt, sin_tt), return_l_aux=False, attn_mask=mask_tt)
    return hidden


def _rm_upload(tt_pipe, t):
    """Upload a host [1,n,hidden] slice as ROW_MAJOR bf16 (replicated) for on-device concat."""
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=tt_pipe.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **tt_pipe._mesh_kw(),
    )


def setup_ondevice_headglue_static(model, tt_pipe, token_h=64, token_w=64):
    """PROMPT-INDEPENDENT on-device conv heads (patch_embed + final_layer) -- the ~114s
    conv-kernel compile. Build ONCE at server startup and pass as `heads=` to
    setup_ondevice_headglue / generate_image_ondevice to reuse across renders (the
    weights are the fixed VAE conv weights, independent of prompt/text)."""
    return {
        "token_h": token_h,
        "token_w": token_w,
        "patch_embed_tt": build_patch_embed(tt_pipe.device, model, token_h, token_w),
        "final_layer_tt": build_final_layer(tt_pipe.device, model, token_h, token_w),
    }


def setup_ondevice_headglue(model, tt_pipe, kwargs, attn_mask, token_h=64, token_w=64, heads=None):
    """Precompute the static gen_image sequence pieces + build the on-device conv heads.
    Returns a ctx dict consumed by run_velocity_once_ondevice."""
    input_ids = kwargs["input_ids"]  # [cfg, S]
    image_mask = kwargs["image_mask"]  # [cfg, S] bool (contiguous <img> run)
    cos, sin = kwargs["custom_pos_emb"]  # [cfg, S, head_dim]
    ts_idx = kwargs["gen_timestep_scatter_index"]  # [cfg, n_ts]
    cfg, S = int(input_ids.shape[0]), int(input_ids.shape[1])
    with torch.no_grad():
        base = model.model.wte(input_ids).to(torch.float32)  # [cfg, S, hidden] static base embeds
    if heads is None:
        heads = setup_ondevice_headglue_static(model, tt_pipe, token_h, token_w)
    ctx = {
        "cfg": cfg,
        "S": S,
        "token_h": token_h,
        "token_w": token_w,
        "patch_embed_tt": heads["patch_embed_tt"],
        "final_layer_tt": heads["final_layer_tt"],
        "rows": [],
    }
    for i in range(cfg):
        m = image_mask[i].reshape(-1).bool()
        idx = torch.nonzero(m).flatten()
        start, end = int(idx[0]), int(idx[-1]) + 1  # contiguous image block [start:end], end-start=token_h*token_w
        cos_i, sin_i = cos[i].to(torch.float32), sin[i].to(torch.float32)
        cos_tt, sin_tt = tt_pipe._upload_pos_img(cos_i, sin_i, S)
        mask_tt = tt_pipe._upload_mask(attn_mask[i] if attn_mask is not None else None, S)
        suffix_tt = _rm_upload(tt_pipe, base[i : i + 1, end:])  # static, resident
        ctx["rows"].append(
            {
                "start": start,
                "end": end,
                "prefix_base": base[i : i + 1, :start].clone(),  # host; <timestep> slot patched per step
                "ts_pos": int(ts_idx[i].reshape(-1)[0]),
                "cos": cos_tt,
                "sin": sin_tt,
                "mask": mask_tt,
                "suffix": suffix_tt,
            }
        )
    ctx["cfg_parallel_ok"] = False
    if cfg == 2:
        r0, r1 = ctx["rows"][0], ctx["rows"][1]
        ctx["cfg_parallel_ok"] = bool(
            r0["start"] == r1["start"]
            and r0["end"] == r1["end"]
            and torch.equal(cos[0], cos[1])
            and torch.equal(sin[0], sin[1])
            and (attn_mask is None or torch.equal(attn_mask[0], attn_mask[1]))
        )
    return ctx


def run_velocity_once_ondevice(model, ctx, tt_pipe, latents, t, cfg_factor):
    """One denoising step, fully on-device head-glue. latents: host [1,32,H,W] (single;
    same for all CFG). Returns velocity [cfg,32,H,W] (only tensor downloaded)."""
    dev = tt_pipe.device
    pe_tt, fl_tt = ctx["patch_embed_tt"], ctx["final_layer_tt"]
    th, tw = ctx["token_h"], ctx["token_w"]
    with torch.no_grad():
        te = model.time_embed(t.reshape(-1)[:1]).to(torch.float32)  # patch_embed emb
        te2 = model.time_embed_2(t.reshape(-1)[:1]).to(torch.float32)  # final_layer emb
        ts_tok = model.timestep_emb(t.reshape(-1)[:1]).to(torch.float32)[0]  # <timestep> meta-token embed
    emb_pe, emb_fl = _repl(dev, te), _repl(dev, te2)
    img_embeds = pe_tt(latents.to(torch.float32), emb_pe)  # [1,H*W,hidden] TILE (same for all CFG)
    img_rm = ttnn.to_layout(img_embeds, ttnn.ROW_MAJOR_LAYOUT)
    import os as _os

    if _os.environ.get("HUNYUAN_CFG_PARALLEL", "0") == "1" and cfg_factor == 2 and ctx.get("cfg_parallel_ok", False):
        # CFG-parallel: cond+uncond as ONE bsz=2 forward -> ~half the per-layer TP
        # collectives across the decoder stack. Eligible only when the 2 rows share
        # cos/sin/mask + image-block position (asserted in setup_ondevice_headglue).
        seqs = []
        for i in range(cfg_factor):
            r = ctx["rows"][i]
            pref = r["prefix_base"].clone()
            pref[0, r["ts_pos"]] = ts_tok.to(pref.dtype)
            pref_rm = _rm_upload(tt_pipe, pref)
            seq_rm = ttnn.concat([pref_rm, img_rm, r["suffix"]], dim=1)
            ttnn.deallocate(pref_rm)
            if tt_pipe._sp:
                S = int(seq_rm.shape[1])
                _, npad = tt_pipe._sp_pad(S)
                if npad:
                    zpad = _rm_upload(tt_pipe, torch.zeros(int(seq_rm.shape[0]), npad, int(seq_rm.shape[2])))
                    seq_rm_p = ttnn.concat([seq_rm, zpad], dim=1)
                    ttnn.deallocate(zpad)
                    ttnn.deallocate(seq_rm)
                    seq_rm = seq_rm_p
                seq_repl = ttnn.to_layout(seq_rm, ttnn.TILE_LAYOUT)
                ttnn.deallocate(seq_rm)
                seq_tt = tt_pipe._sp_scatter_seq(seq_repl)
                ttnn.deallocate(seq_repl)
            else:
                seq_tt = ttnn.to_layout(seq_rm, ttnn.TILE_LAYOUT)
                ttnn.deallocate(seq_rm)
            seqs.append(seq_tt)
        seq_b = ttnn.concat(seqs, dim=0)  # [cfg, S_pad/sp, hidden]
        for s in seqs:
            ttnn.deallocate(s)
        r0 = ctx["rows"][0]
        hidden_b = _forward_image_device(tt_pipe, seq_b, r0["cos"], r0["sin"], r0["mask"])
        ttnn.deallocate(seq_b)
        vels = []
        for i in range(cfg_factor):
            hi = ttnn.slice(hidden_b, [i, 0, 0], [i + 1, int(hidden_b.shape[1]), int(hidden_b.shape[2])])
            if tt_pipe._sp:
                hg = tt_pipe._seq_gather(hi)
                ttnn.deallocate(hi)
                hi = hg
            hidden_rm = ttnn.to_layout(hi, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(hi)
            r = ctx["rows"][i]
            hid_img_rm = ttnn.slice(hidden_rm, [0, r["start"], 0], [1, r["end"], int(hidden_rm.shape[-1])])
            ttnn.deallocate(hidden_rm)
            hid_img = ttnn.to_layout(hid_img_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(hid_img_rm)
            vel = fl_tt(hid_img, emb_fl)
            ttnn.deallocate(hid_img)
            vt = _mesh_to_torch(vel, dev).to(torch.float32).reshape(1, th, tw, 32).permute(0, 3, 1, 2).contiguous()
            ttnn.deallocate(vel)
            vels.append(vt)
        ttnn.deallocate(hidden_b)
        for x in (emb_pe, emb_fl, img_embeds, img_rm):
            ttnn.deallocate(x)
        return torch.cat(vels, dim=0)
    vels = []
    for i in range(cfg_factor):
        r = ctx["rows"][i]
        pref = r["prefix_base"].clone()
        pref[0, r["ts_pos"]] = ts_tok.to(pref.dtype)  # patch the per-step <timestep> token
        pref_rm = _rm_upload(tt_pipe, pref)
        seq_rm = ttnn.concat([pref_rm, img_rm, r["suffix"]], dim=1)  # [1,S,hidden] ROW_MAJOR (replicated)
        if tt_pipe._sp:
            # SP: these embeds are assembled REPLICATED on device, but the decoder runs
            # sequence-parallel (rope/mask/K,V-gather all assume S_pad/sp per device).
            # Zero-pad the sequence to S_pad (mult of sp*32) in ROW_MAJOR (tile-safe),
            # tilize, then reshard replicated -> per-device seq shard. Pad tokens are
            # masked out (see _upload_mask) and their positions are never read back
            # (only the image block [start:end] is sliced out below).
            S = int(seq_rm.shape[1])
            _, npad = tt_pipe._sp_pad(S)
            if npad:
                # replicated zero pad (same upload path as _rm_upload -> matches seq_rm)
                zpad = _rm_upload(tt_pipe, torch.zeros(int(seq_rm.shape[0]), npad, int(seq_rm.shape[2])))
                seq_rm_p = ttnn.concat([seq_rm, zpad], dim=1)
                ttnn.deallocate(zpad)
                ttnn.deallocate(seq_rm)
                seq_rm = seq_rm_p
            seq_repl = ttnn.to_layout(seq_rm, ttnn.TILE_LAYOUT)  # [1,S_pad,hidden] replicated
            seq_tt = tt_pipe._sp_scatter_seq(seq_repl)  # [1,S_pad/sp,hidden] seq-sharded
            ttnn.deallocate(seq_repl)
        else:
            seq_tt = ttnn.to_layout(seq_rm, ttnn.TILE_LAYOUT)
        hidden = _forward_image_device(tt_pipe, seq_tt, r["cos"], r["sin"], r["mask"])  # [1,S_pad/sp,hidden] under SP
        if tt_pipe._sp:
            hidden_full = tt_pipe._seq_gather(hidden)  # gather seq shard -> [1,S_pad,hidden] replicated
            ttnn.deallocate(hidden)
            hidden = hidden_full
        hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        hid_img_rm = ttnn.slice(hidden_rm, [0, r["start"], 0], [1, r["end"], int(hidden.shape[-1])])
        hid_img = ttnn.to_layout(hid_img_rm, ttnn.TILE_LAYOUT)  # [1,H*W,hidden]
        vel = fl_tt(hid_img, emb_fl)  # [1,1,H*W,32] device
        vt = _mesh_to_torch(vel, dev).to(torch.float32).reshape(1, th, tw, 32).permute(0, 3, 1, 2).contiguous()
        vels.append(vt)  # [1,32,H,W]
        for x in (pref_rm, seq_rm, seq_tt, hidden, hidden_rm, hid_img_rm, hid_img, vel):
            ttnn.deallocate(x)
    for x in (emb_pe, emb_fl, img_embeds, img_rm):
        ttnn.deallocate(x)
    return torch.cat(vels, dim=0)  # [cfg,32,H,W]


def generate_image_ondevice(
    model,
    tt_pipe,
    prompt,
    image_size=(1024, 1024),
    num_inference_steps=None,
    guidance_scale=None,
    seed=0,
    out_path="hunyuan_t2i_ondevice.png",
    heads=None,
):
    """Full hybrid text->image render using the STAGE-3 fully-on-device head-glue path
    (hidden never leaves the device across the loop). Mirrors gen_image.generate_image
    but swaps run_velocity_once -> run_velocity_once_ondevice. Returns (PIL image, timing).
    Honors HUNYUAN_VAE_AUTOCAST (bf16/fp16) + HUNYUAN_CCL_LINKS like the baseline."""
    import os
    import time

    from .gen_image import prepare_gen_image_inputs

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

    ctx = setup_ondevice_headglue(model, tt_pipe, kwargs, attn_mask, token_h=h, token_w=w, heads=heads)

    # Prepare the on-device VAE decoder HERE (model setup), before the timed render, so the
    # ~19.4s one-time prepare-mesh-weights build is not charged to the per-image path — the
    # loop and vae timers below then measure decode-only. Cached on `model`; a no-op on later
    # images (and when host VAE is used).
    if os.environ.get("HUNYUAN_ONDEVICE_VAE", "").lower() in ("1", "true", "yes"):
        from .vae_decode import prebuild_ondevice_vae

        prebuild_ondevice_vae(model, tt_pipe.device, latent_h=h, latent_w=w, latent_t=1)

    t_gen = time.time()
    per_step_ms = []
    for i, t in enumerate(timesteps):
        t0 = time.time()
        pred = run_velocity_once_ondevice(model, ctx, tt_pipe, latents, t, cfg)  # [cfg,32,h,w]
        if cfg == 2:
            pc, pu = pred.chunk(2)
            pred = pu + guidance * (pc - pu)
        latents = sched.step(pred, t, latents, return_dict=False)[0]
        per_step_ms.append(1000.0 * (time.time() - t0))
        print(f"  step {i + 1}/{steps} {per_step_ms[-1]:.0f} ms", flush=True)
    loop_s = time.time() - t_gen

    t_vae = time.time()
    sf = model.vae.config.scaling_factor
    if sf:
        latents = latents / sf
    if hasattr(model.vae, "ffactor_temporal"):
        latents = latents.unsqueeze(2)
    _vae_ac = os.environ.get("HUNYUAN_VAE_AUTOCAST", "").lower()
    _ondevice_vae = os.environ.get("HUNYUAN_ONDEVICE_VAE", "").lower() in ("1", "true", "yes")
    with torch.no_grad():
        if _ondevice_vae:
            # On-device mesh VAE decode (perf lever #1). Host model.vae.decode stays the oracle.
            from .vae_decode import ondevice_vae_decode

            image = ondevice_vae_decode(model, tt_pipe.device, latents.float())  # [1,3,1,H,W]
        elif _vae_ac in ("bf16", "fp16"):
            _dt = torch.bfloat16 if _vae_ac == "bf16" else torch.float16
            with torch.autocast(device_type="cpu", dtype=_dt):
                image = model.vae.decode(latents.float(), return_dict=False)[0]
        else:
            image = model.vae.decode(latents.float(), return_dict=False)[0]
    if hasattr(model.vae, "ffactor_temporal"):
        image = image.squeeze(2)
    image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])[0]
    vae_s = time.time() - t_vae

    total = loop_s + vae_s
    ms = sum(per_step_ms) / len(per_step_ms)
    image.save(out_path)
    print(
        f"ONDEVICE_E2E_TOTAL_LATENCY_S={total:.3f} loop={loop_s:.1f}s @ {ms:.0f} ms/step x{steps} "
        f"vae={vae_s:.1f}s token_hw=({h},{w}) out={out_path}",
        flush=True,
    )
    return image, {"total_s": total, "loop_s": loop_s, "vae_s": vae_s, "ms_per_step": ms, "steps": steps}
