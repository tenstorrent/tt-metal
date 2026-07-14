# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN diffusion pipeline for HunyuanImage-3.0 — composes the already-ported,
# individually-PCC-gated blocks into a denoise step and a multi-step loop.
#
# Host-routed composition is in tests/pcc/test_denoise.py; on-device step in test_pipeline.py:
# that test proves a single step end to end but round-trips every module
# hand-off through host. Here the hand-offs stay ON DEVICE — the image-token
# scatter into the [text | image | text] sequence is a device-side concat, and
# the backbone -> final_layer hand-off is a device-side slice.
#
# Mirrors the gen_image path of HunyuanImage3ForCausalMM.forward
# (modeling_hunyuan_image_3.py): instantiate_vae_image_tokens -> self.model
# (ln_f SKIPPED for gen_image) -> ragged_final_layer.
#
#     noised latent --patch_embed(UNetDown)+time_embed--> image tokens
#     image tokens scattered into [text | image | text] sequence
#     sequence --backbone (N MoE layers, NO ln_f)--> hidden
#     hidden[image span] --final_layer(UNetUp)+time_embed_2--> velocity pred
#
# Scope / status:
#   * Single contiguous gen-image span for on-device scatter (T2I and I2I).
#     T2I uses ``text_pre | img | text_post``; I2I uses ``base_embeds`` with
#     cond tokens pre-scattered on host, overwriting only the gen span each step.
#   * Multi-span attention masks and multi-span 2D RoPE are supported via
#     ``bundle_to_denoise_cond`` (cond joint + gen bidirectional blocks).
#   * On-device scatter uses ttnn.concat, so span boundaries should be
#     TILE-aligned (multiples of 32) in TILE layout when possible.
#   * VAE decode (latent -> pixels) and tokenizer/sequence construction are
#     separate milestones — see README. This module stops at the latent.

import os
import time

import torch
import ttnn

from .denoise_dual_cq import DenoiseDualCQCoordinator, denoise_2cq_enabled, latent_tt_to_torch
from .scheduler import classifier_free_guidance_tt
from .vae_dual_cq import VaeDualCQCoordinator, vae_2cq_enabled


def _ttnn_embed_dtype(ref: ttnn.Tensor) -> ttnn.DataType:
    """Return the ttnn dtype of a device embedding tensor (not torch.dtype)."""
    if not isinstance(ref, ttnn.Tensor):
        raise TypeError(
            "expected a device ttnn.Tensor for embedding dtype; "
            "upload host tensors via upload_denoise_cond_mesh / replicate_fn first"
        )
    return ref.dtype

try:
    from tracy import signpost
except ImportError:

    def signpost(header="", message=None):
        pass


# Tracy region markers for HunyuanTtDenoiseStep.__call__. Filter CSV with:
#   python models/tt_transformers/scripts/op_perf_results.py ops_perf_results_*.csv \
#     --signpost start_patch_embed
# Regions: start_patch_embed … stop_patch_embed, start_scatter … stop_scatter,
# start_backbone … stop_backbone, start_final_layer … stop_final_layer,
# start_denoise_step … stop_denoise_step (whole step).


class HunyuanTtDenoiseStep:
    """One on-device diffusion step: latent + conditioning -> velocity prediction.

    Holds references to the three ported sub-modules (built by the caller) and
    the static sequence layout. Stateless across steps except for those modules;
    the latent, timestep embeddings and text embeddings are passed in each call.
    """

    def __init__(
        self,
        device,
        *,
        patch_embed,  # HunyuanTtUNetDown
        backbone,  # HunyuanTtModel (apply_final_norm=False)
        final_layer,  # HunyuanTtUNetUp
        img_slice: slice,  # contiguous image-token span in the sequence
        grid_hw: tuple,  # (token_h, token_w) of the latent grid
        seq_len: int,
    ):
        self.device = device
        self.patch_embed = patch_embed
        self.backbone = backbone
        self.final_layer = final_layer
        self.img_slice = img_slice
        self.token_h, self.token_w = grid_hw
        self.seq_len = seq_len
        self.n_img = self.token_h * self.token_w
        assert img_slice.stop - img_slice.start == self.n_img, "img_slice must cover grid_h*grid_w tokens"

    # -- sequence scatter ---------------------------------------------------
    def _scatter(self, img_tokens, text_pre, text_post):
        """[text_pre | img_tokens | text_post] -> [B, S, H], on device.

        img_tokens: ttnn [1,1,n_img,H] (NHWC flat from UNetDown).
        text_pre/text_post: ttnn [B, *, H] TILE (precomputed prompt embeddings).
        """
        B = self.backbone_batch
        H = self.backbone.hidden_size
        # UNetDown emits NHWC-flat tokens whose dtype/layout can differ from the
        # TILE/bf16 text embeddings; concat requires all pieces to match. Align
        # to the text reference (and the backbone expects TILE inputs_embeds).
        ref = text_pre if text_pre is not None else text_post
        toks = ttnn.reshape(img_tokens, [B, self.n_img, H])
        toks = ttnn.to_layout(toks, ttnn.TILE_LAYOUT)
        ref_dtype = _ttnn_embed_dtype(ref)
        if toks.dtype != ref_dtype:
            toks = ttnn.typecast(toks, ref_dtype)
        pieces = [p for p in (text_pre, toks, text_post) if p is not None]
        seq = ttnn.concat(pieces, dim=1)
        ttnn.deallocate(toks)
        return seq

    def _scatter_from_base(self, img_tokens, base_embeds):
        """Overwrite gen span inside prebuilt ``base_embeds`` (I2I path)."""
        B = self.backbone_batch
        H = self.backbone.hidden_size
        toks = ttnn.reshape(img_tokens, [B, self.n_img, H])
        toks = ttnn.to_layout(toks, ttnn.TILE_LAYOUT)
        base_dtype = _ttnn_embed_dtype(base_embeds)
        if toks.dtype != base_dtype:
            toks = ttnn.typecast(toks, base_dtype)

        pre_len = self.img_slice.start
        post_len = self.seq_len - self.img_slice.stop
        pieces = []
        if pre_len > 0:
            pieces.append(ttnn.slice(base_embeds, [0, 0, 0], [B, pre_len, H]))
        pieces.append(toks)
        if post_len > 0:
            pieces.append(ttnn.slice(base_embeds, [0, self.img_slice.stop, 0], [B, self.seq_len, H]))
        seq = ttnn.concat(pieces, dim=1)
        ttnn.deallocate(toks)
        return seq

    # -- one step -----------------------------------------------------------
    def __call__(
        self,
        latent_bchw,  # torch [B,C,h,w] or ttnn NHWC flat — UNetDown entry
        *,
        text_pre=None,  # ttnn [B, text_pre_len, H] TILE (T2I path)
        text_post=None,  # ttnn [B, text_post_len, H] TILE; may be None
        base_embeds=None,  # ttnn [B, S, H] TILE — I2I path (cond pre-scattered)
        t_emb1,  # ttnn [1,1,B,H] timestep embedding for patch_embed
        t_emb2,  # ttnn [1,1,B,H] timestep embedding for final_layer
        image_infos,  # 2D-RoPE spans, e.g. [[(slice,(h,w)), ...]] per batch row
        attention_mask,  # ttnn [B,1,S,S] additive mask
        batch: int = 1,
    ):
        self.backbone_batch = batch
        if base_embeds is not None and (text_pre is not None or text_post is not None):
            raise ValueError("pass base_embeds OR text_pre/text_post, not both")
        if base_embeds is None and text_pre is None:
            raise ValueError("text_pre or base_embeds is required")

        verbose = os.environ.get("HY_VERBOSE", "1") != "0"

        signpost("start_denoise_step")

        # 1) patch_embed: noised latent -> image tokens [1,1,n_img,H]
        if verbose:
            print("[denoise_step] patch_embed ...", flush=True)
        signpost("start_patch_embed")
        img_tok, th, tw = self.patch_embed(latent_bchw, t_emb1)
        signpost("stop_patch_embed")
        assert (th, tw) == (self.token_h, self.token_w), f"grid mismatch: got {th}x{tw}"

        # 2) scatter into the sequence, run the backbone (NO ln_f)
        signpost("start_scatter")
        if base_embeds is not None:
            seq = self._scatter_from_base(img_tok, base_embeds)
        else:
            seq = self._scatter(img_tok, text_pre, text_post)
        signpost("stop_scatter")
        ttnn.deallocate(img_tok)
        if verbose:
            print(f"[denoise_step] backbone forward seq_len={self.seq_len} ...", flush=True)
        signpost("start_backbone")
        hidden = self.backbone.forward(
            inputs_embeds=seq,
            seq_len=self.seq_len,
            image_infos=image_infos,
            attention_mask=attention_mask,
        )
        signpost("stop_backbone")
        ttnn.deallocate(seq)

        # 3) final_layer: image-span hidden -> velocity prediction (NHWC flat)
        if verbose:
            print("[denoise_step] final_layer ...", flush=True)
        H = self.backbone.hidden_size
        signpost("start_final_layer")
        img_out = ttnn.slice(
            hidden,
            [0, self.img_slice.start, 0],
            [batch, self.img_slice.stop, H],
        )
        ttnn.deallocate(hidden)
        img_out = ttnn.reshape(img_out, [1, 1, self.n_img, H])
        pred, _, _ = self.final_layer(img_out, t_emb2, th, tw, B=batch)
        ttnn.deallocate(img_out)
        signpost("stop_final_layer")
        signpost("stop_denoise_step")
        return pred  # ttnn NHWC flat [1,1,B*h*w,latent_ch]

    def forward_device(
        self,
        latent_flat,
        *,
        text_pre=None,
        text_post=None,
        base_embeds=None,
        t_emb1,
        t_emb2,
        image_infos,
        attention_mask,
        batch: int = 1,
        cos_sin=None,
    ):
        """On-device denoise forward from a pre-uploaded NHWC-flat latent (trace path)."""
        self.backbone_batch = batch
        if base_embeds is not None and (text_pre is not None or text_post is not None):
            raise ValueError("pass base_embeds OR text_pre/text_post, not both")
        if base_embeds is None and text_pre is None:
            raise ValueError("text_pre or base_embeds is required")

        B, H, W = batch, self.token_h, self.token_w
        img_tok, th, tw = self.patch_embed.forward_latent(latent_flat, t_emb1, B, H, W)
        assert (th, tw) == (self.token_h, self.token_w), f"grid mismatch: got {th}x{tw}"

        if base_embeds is not None:
            seq = self._scatter_from_base(img_tok, base_embeds)
        else:
            seq = self._scatter(img_tok, text_pre, text_post)
        ttnn.deallocate(img_tok)

        hidden = self.backbone.forward(
            inputs_embeds=seq,
            seq_len=self.seq_len,
            image_infos=image_infos,
            attention_mask=attention_mask,
            cos_sin=cos_sin,
        )
        ttnn.deallocate(seq)

        Hdim = self.backbone.hidden_size
        img_out = ttnn.slice(
            hidden,
            [0, self.img_slice.start, 0],
            [batch, self.img_slice.stop, Hdim],
        )
        ttnn.deallocate(hidden)
        img_out = ttnn.reshape(img_out, [1, 1, self.n_img, Hdim])
        pred, _, _ = self.final_layer(img_out, t_emb2, th, tw, B=batch)
        ttnn.deallocate(img_out)
        return pred


def denoise_loop(
    step: HunyuanTtDenoiseStep,
    scheduler,  # HunyuanTtScheduler, set_timesteps() already called
    init_latent,  # torch [B, C, h, w] — initial noise
    *,
    time_embed,  # HunyuanTtTimestepEmbedder for patch_embed (UNetDown)
    time_embed_2,  # HunyuanTtTimestepEmbedder for final_layer (UNetUp)
    cond,  # conditioning dict for the conditional pass (see below)
    uncond=None,  # same dict for the unconditional pass; None => no CFG
    guidance_scale: float = 1.0,
    timestep_emb=None,  # host ref TimestepEmbedder for gen_timestep_scatter_index
    guidance_emb=None,  # host ref TimestepEmbedder for guidance_scatter_index (distil)
    timestep_r_emb=None,  # host ref TimestepEmbedder for gen_timestep_r_scatter_index (meanflow)
    cfg_distilled: bool = False,
    use_meanflow: bool = False,
    mesh_device=None,  # pass the MeshDevice when the backbone is mesh-resident
    dual_cq: DenoiseDualCQCoordinator | None = None,
):
    """Run the diffusion denoise loop, returning the final latent (torch NCHW).

    `cond`/`uncond` carry the static, timestep-independent conditioning forwarded
    to `step.__call__`:
        text_pre, text_post, base_embeds, image_infos, attention_mask, batch.
    Use ``text_pre``/``text_post`` for T2I or ``base_embeds`` for I2I (cond
    tokens pre-scattered on host). ``image_infos`` should list all 2D-RoPE spans
    (cond + gen) for I2I.
    The timestep embeddings ARE recomputed each step from `time_embed` /
    `time_embed_2` (both passes share them — the timestep is identical), and the
    per-step velocity prediction drives the on-device Euler update via the
    scheduler. With CFG, the conditional and unconditional predictions are
    combined on device before the update.

    When ``timestep_emb`` is set and ``cond`` carries ``base_embeds_host`` plus
    ``gen_timestep_scatter_index``, the gen-image timestep token is re-scattered
    on host each step before upload (I2I/T2I gen path). For Instruct-Distil
    (``cfg_distilled``), also scatters ``guidance`` at ``1000 * guidance_scale``.
    For meanflow (``use_meanflow``), scatters ``timestep_r`` from
    ``scheduler.get_timestep_r(t)`` each step.

    Latent representation: the scheduler operates on device NHWC-flat tensors,
    but UNetDown's entry only accepts torch NCHW, so the (small) latent makes one
    host hop per step. The heavy compute (backbone, MoE) stays on device. A
    future patch_embed change to accept a device NHWC-flat latent would remove
    this hop; the latent is tiny relative to the backbone, so it is not on the
    critical path.
    """
    import torch

    from models.experimental.hunyuan_image_3_0.tt.stage_trace import DenoiseStepTracer
    from models.experimental.hunyuan_image_3_0.tt.trace_config import denoise_execute_trace_enabled

    B, C, h, w = init_latent.shape
    scheduler.set_begin_index(0)
    num_steps = len(scheduler.timesteps) if scheduler.timesteps is not None else 0

    if denoise_execute_trace_enabled(steps=num_steps) and mesh_device is not None:
        print("[denoise] execute_trace loop on CQ0 (HY_TRACE=1)", flush=True)
        tracer = DenoiseStepTracer(
            step.device,
            step,
            time_embed=time_embed,
            time_embed_2=time_embed_2,
            scheduler=scheduler,
            cond=cond,
            uncond=uncond,
            guidance_scale=guidance_scale,
            mesh_device=mesh_device,
            batch=B,
            channels=C,
            h=h,
            w=w,
        )
        try:
            latent = tracer.run(init_latent, scheduler.timesteps)
        finally:
            tracer.release()
        return latent

    if dual_cq is None and mesh_device is not None and denoise_2cq_enabled(mesh_device):
        dual_cq = DenoiseDualCQCoordinator(mesh_device)
        print("[denoise] 2CQ latent D2H on CQ1 (HY_TRACE=1)", flush=True)

    do_cfg = uncond is not None and guidance_scale != 1.0 and not cfg_distilled
    distill_guidance = 1000.0 * guidance_scale if cfg_distilled else None
    latent = init_latent  # torch NCHW (canonical host form; small tensor)
    timesteps = list(scheduler.timesteps)
    total_steps = len(timesteps)
    verbose = os.environ.get("HY_VERBOSE", "1") != "0"
    loop_t0 = time.time()

    # Host<->device helpers — replicate to / gather from the mesh when resident.
    def _up(t_host, dtype):
        if mesh_device is not None:
            return ttnn.from_torch(
                t_host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        return ttnn.from_torch(t_host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=step.device)

    def _down(t_dev):
        if mesh_device is not None:
            out = ttnn.to_torch(t_dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            return out[:1]  # one replica (flat leading dim is 1)
        return ttnn.to_torch(t_dev)

    def _down_latent_bchw(t_dev):
        return latent_tt_to_torch(t_dev, mesh_device, batch=B, channels=C, h=h, w=w)

    def _prepare_base_embeds(c, t_scalar):
        host = c.get("base_embeds_host")
        if host is not None:
            emb = host
            idx = c.get("gen_timestep_scatter_index")
            needs_scatter = (
                (idx is not None and timestep_emb is not None)
                or (cfg_distilled and c.get("guidance_scatter_index") is not None and guidance_emb is not None)
                or (use_meanflow and c.get("gen_timestep_r_scatter_index") is not None and timestep_r_emb is not None)
            )
            if needs_scatter:
                from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import (
                    scatter_distill_step_embeds,
                )

                t_r = float(scheduler.get_timestep_r(t_scalar)) if use_meanflow else None
                emb = scatter_distill_step_embeds(
                    emb,
                    t_scalar=float(t_scalar),
                    gen_timestep_scatter_index=idx,
                    timestep_emb=timestep_emb,
                    guidance_scalar=distill_guidance,
                    guidance_scatter_index=c.get("guidance_scatter_index"),
                    guidance_emb=guidance_emb,
                    t_r_scalar=t_r,
                    gen_timestep_r_scatter_index=c.get("gen_timestep_r_scatter_index"),
                    timestep_r_emb=timestep_r_emb,
                )
            return _up(emb, ttnn.bfloat16)
        return c.get("base_embeds")

    for step_i, t in enumerate(timesteps):
        step_t0 = time.time()
        if dual_cq is not None and step_i > 0:
            latent = dual_cq.consume_latent_torch(mesh_device, batch=B, channels=C, h=h, w=w)
        if verbose:
            cfg_note = " +CFG" if do_cfg else ""
            print(
                f"[denoise] step {step_i + 1}/{total_steps} t={float(t):.0f}{cfg_note} ...",
                flush=True,
            )
        # Timestep embedding: pass a (replicated) device tensor [1,1,B,1] so the
        # embedder doesn't host-upload internally (which ignores the mesh).
        tvec = _up(torch.tensor([float(t)] * B, dtype=torch.float32).reshape(1, 1, B, 1), ttnn.float32)
        te1 = time_embed.forward(tvec)  # [1,1,B,H]
        te2 = time_embed_2.forward(tvec)
        ttnn.deallocate(tvec)

        def _one(c):
            kwargs = dict(
                t_emb1=te1,
                t_emb2=te2,
                image_infos=c["image_infos"],
                attention_mask=c["attention_mask"],
                batch=c.get("batch", B),
            )
            base = _prepare_base_embeds(c, t)
            if base is not None:
                kwargs["base_embeds"] = base
            else:
                kwargs["text_pre"] = c["text_pre"]
                kwargs["text_post"] = c.get("text_post")
            pred = step(latent, **kwargs)
            if base is not None and c.get("base_embeds_host") is not None:
                ttnn.deallocate(base)
            return pred

        pred = _one(cond)  # device NHWC flat [1,1,B*h*w,C]
        if do_cfg:
            if verbose:
                print("[denoise] CFG uncond pass ...", flush=True)
            pred_uncond = _one(uncond)
            combined = classifier_free_guidance_tt(pred, pred_uncond, guidance_scale)
            ttnn.deallocate(pred)
            ttnn.deallocate(pred_uncond)
            pred = combined

        # On-device Euler update: prev = sample + (sigma_next - sigma) * pred.
        sample = _up(latent.permute(0, 2, 3, 1).reshape(1, 1, B * h * w, C).contiguous(), pred.dtype)
        nxt = scheduler.step(pred, t, sample)
        if dual_cq is not None:
            dual_cq.launch_latent_d2h(nxt)
        else:
            latent = _down_latent_bchw(nxt)
            ttnn.deallocate(nxt)

        ttnn.deallocate(pred)
        ttnn.deallocate(sample)
        ttnn.deallocate(te1)
        ttnn.deallocate(te2)
        if verbose:
            cq_note = " [2CQ D2H queued]" if dual_cq is not None else ""
            print(
                f"[denoise] step {step_i + 1}/{total_steps} done "
                f"({time.time() - step_t0:.1f}s, total {time.time() - loop_t0:.0f}s){cq_note}",
                flush=True,
            )

    if dual_cq is not None:
        latent = dual_cq.consume_latent_torch(mesh_device, batch=B, channels=C, h=h, w=w)
        print(f"[denoise] 2CQ completed {dual_cq.steps} latent D2H transfers on CQ1", flush=True)

    return latent  # torch [B, C, h, w] — feed to VAE decode


def upload_denoise_cond_mesh(
    cond_row: dict,
    *,
    seq_len: int,
    mesh_device,
    replicate_fn,
    full_attn_slices=None,
):
    """Upload host cond dict entries needed on device for ``denoise_loop``."""
    del seq_len, full_attn_slices  # reserved for TT attention-mask build
    out = dict(cond_row)
    mask = out.get("attention_mask")
    if mask is not None and not isinstance(mask, ttnn.Tensor):
        out["attention_mask"] = replicate_fn(mask)
    base = out.get("base_embeds")
    if base is None:
        base = out.get("base_embeds_host")
    if base is not None and not isinstance(base, ttnn.Tensor):
        host = base.to(torch.bfloat16) if isinstance(base, torch.Tensor) else base
        out["base_embeds"] = replicate_fn(host)
    return out


def decode_latent(
    mesh_device,  # ttnn.MeshDevice (replicated) — the VAE decoder's device context
    latent,  # torch [B, C, h, w] diffusion latent (single frame)
    *,
    scaling_factor: float = 0.562679178327931,  # config.json vae.scaling_factor
    decoder=None,  # optional prebuilt VAEDecoderTTNN (reuse across calls)
    grid_hw=None,  # (h, w) latent grid; when set, build the decoder for this size
    dtype=ttnn.bfloat16,
    ccl_manager=None,  # set (with h/w_mesh_axis) to run the decoder H/W-spatial-parallel
    h_mesh_axis=None,
    w_mesh_axis=None,
    dual_cq: VaeDualCQCoordinator | None = None,
):
    """Decode a diffusion latent to an RGB image in [0, 1] via the TTNN VAE.

    The VAE decoder runs on a (replicated) MeshDevice — a SEPARATE device context
    from the single-device backbone — so this is a distinct stage and the latent
    crosses host between the denoise loop and here. Mirrors the reference image
    decode (hunyuan_image_3_pipeline.py):
        latent = latent / scaling_factor
        image  = vae.decode(latent)          # T_in=1 -> T_out=4
        image  = image[:, :, -1]             # keep last temporal frame (HF behavior)
        image  = (image / 2 + 0.5).clamp(0, 1)

    Returns torch [B, 3, H, W] in [0, 1].
    """
    from models.experimental.hunyuan_image_3_0.ref.vae.decoder import vae_decode_output_to_rgb
    from .vae.decoder import VAEDecoderTTNN, bcthw_to_bthwc, bthwc_to_bcthw

    if dual_cq is None and vae_2cq_enabled(mesh_device):
        dual_cq = VaeDualCQCoordinator(mesh_device)
        print("[vae] 2CQ decode active (CQ0 forward, CQ1 async RGB D2H)", flush=True)

    if decoder is None:
        vae_kwargs = {}
        if grid_hw is not None:
            vae_kwargs = dict(latent_h=int(grid_hw[0]), latent_w=int(grid_hw[1]))
        decoder = VAEDecoderTTNN(mesh_device, dtype=dtype, **vae_kwargs)

    spatial = ccl_manager is not None and (h_mesh_axis is not None or w_mesh_axis is not None)
    if spatial:
        from models.experimental.hunyuan_image_3_0.tt.trace_config import vae_execute_trace_enabled

        if vae_execute_trace_enabled():
            return _decode_latent_spatial_traced(
                mesh_device,
                latent,
                decoder,
                ccl_manager,
                h_mesh_axis,
                w_mesh_axis,
                scaling_factor=scaling_factor,
                dtype=dtype,
            )

        img = _decode_latent_spatial(
            mesh_device,
            latent,
            decoder,
            ccl_manager,
            h_mesh_axis,
            w_mesh_axis,
            scaling_factor=scaling_factor,
            dtype=dtype,
            dual_cq=dual_cq,
        )
        if dual_cq is not None:
            print(
                f"[vae] 2CQ completed d2h={dual_cq.d2h_transfers}",
                flush=True,
            )
        return img

    from models.experimental.hunyuan_image_3_0.tt.trace_config import vae_execute_trace_enabled

    if vae_execute_trace_enabled():
        return _decode_latent_replicated_traced(
            mesh_device,
            latent,
            decoder,
            scaling_factor=scaling_factor,
            dtype=dtype,
        )

    # [B, C, h, w] -> scaled BCTHW [B, C, 1, h, w]
    z = (latent / scaling_factor).unsqueeze(2)
    host = z.bfloat16() if dtype == ttnn.bfloat16 else z.float()
    x_bcthw = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x_bthwc = bcthw_to_bthwc(x_bcthw)
    ttnn.deallocate(x_bcthw, force=False)

    if dual_cq is not None:
        dual_cq.fence_compute_before_forward()
    out_bthwc = decoder(x_bthwc)
    ttnn.deallocate(x_bthwc, force=False)

    out_bcthw = bthwc_to_bcthw(out_bthwc)
    if dual_cq is not None:
        dual_cq.launch_output_d2h(out_bcthw)
        out_host = dual_cq.consume_output_host()
        img = ttnn.to_torch(
            out_host,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        print(
            f"[vae] 2CQ completed d2h={dual_cq.d2h_transfers}",
            flush=True,
        )
    else:
        img = ttnn.to_torch(
            ttnn.from_device(out_bcthw),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
    img = img[: img.shape[0] // mesh_device.get_num_devices()].float()  # [B, 3, T, H, W]

    return vae_decode_output_to_rgb(img)  # last temporal frame -> [B, 3, H, W] in [0, 1]


def _decode_latent_spatial_traced(
    mesh_device,
    latent,
    decoder,
    ccl,
    h_mesh_axis,
    w_mesh_axis,
    *,
    scaling_factor,
    dtype,
):
    from models.experimental.hunyuan_image_3_0.ref.vae.decoder import vae_decode_output_to_rgb
    from models.experimental.hunyuan_image_3_0.tt.stage_trace import VaeDecodeTracer
    from .vae.spatial import enable_vae_spatial

    print("[vae] execute_trace spatial decode on CQ0 (HY_VAE_DECODE_TRACE=1)", flush=True)
    enable_vae_spatial(decoder, ccl, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
    mesh_shape = tuple(mesh_device.shape)
    dims = [None, None]
    if h_mesh_axis is not None:
        dims[h_mesh_axis] = 2
    if w_mesh_axis is not None:
        dims[w_mesh_axis] = 3
    dims = [d if d is not None else (3 if 2 in dims else 2) for d in dims]

    tracer = VaeDecodeTracer(
        mesh_device,
        decoder,
        ccl=ccl,
        h_mesh_axis=h_mesh_axis,
        w_mesh_axis=w_mesh_axis,
        mesh_shape=mesh_shape,
        dims=dims,
        dtype=dtype,
        spatial=True,
    )
    try:
        out_bthwc = tracer.capture_and_run(latent, scaling_factor=scaling_factor)
        img_bthwc = ttnn.to_torch(
            ttnn.from_device(out_bthwc),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=dims),
        ).float()
        ttnn.deallocate(out_bthwc, force=False)
    finally:
        tracer.release()

    img = img_bthwc.permute(0, 4, 1, 2, 3)
    return vae_decode_output_to_rgb(img)


def _decode_latent_replicated_traced(
    mesh_device,
    latent,
    decoder,
    *,
    scaling_factor,
    dtype,
):
    from models.experimental.hunyuan_image_3_0.ref.vae.decoder import vae_decode_output_to_rgb
    from models.experimental.hunyuan_image_3_0.tt.stage_trace import VaeDecodeTracer
    from .vae.decoder import bthwc_to_bcthw

    print("[vae] execute_trace decode on CQ0 (HY_VAE_DECODE_TRACE=1)", flush=True)
    tracer = VaeDecodeTracer(mesh_device, decoder, dtype=dtype, spatial=False)
    try:
        out_bthwc = tracer.capture_and_run(latent, scaling_factor=scaling_factor)
        out_bcthw = bthwc_to_bcthw(out_bthwc)
        img = ttnn.to_torch(
            ttnn.from_device(out_bcthw),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        ttnn.deallocate(out_bthwc, force=False)
        ttnn.deallocate(out_bcthw, force=False)
    finally:
        tracer.release()

    img = img[: img.shape[0] // mesh_device.get_num_devices()].float()
    return vae_decode_output_to_rgb(img)


def _decode_latent_spatial(
    mesh_device,
    latent,
    decoder,
    ccl,
    h_mesh_axis,
    w_mesh_axis,
    *,
    scaling_factor,
    dtype,
    dual_cq: VaeDualCQCoordinator | None = None,
):
    """H/W-spatial-parallel decode: shard the latent across the mesh (H->axis0,
    W->axis1), run the spatially-sharded decoder (convs keep a halo, norms/attn
    gather), then ConcatMesh2dToTensor the output back to full resolution."""
    from models.experimental.hunyuan_image_3_0.ref.vae.decoder import vae_decode_output_to_rgb
    from .vae.spatial import enable_vae_spatial

    enable_vae_spatial(decoder, ccl, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
    mesh_shape = tuple(mesh_device.shape)

    # [B,C,h,w] -> BTHWC [B,1,h,w,C] on host; H,W must divide the axis sizes (64 / 2 ok).
    z = (latent / scaling_factor).unsqueeze(2)  # [B,C,1,h,w]
    host = (z.bfloat16() if dtype == ttnn.bfloat16 else z.float()).permute(0, 2, 3, 4, 1).contiguous()  # BTHWC

    # ShardTensor2dMesh dims: index = mesh_axis, value = tensor dim. H=dim2, W=dim3.
    dims = [None, None]
    if h_mesh_axis is not None:
        dims[h_mesh_axis] = 2
    if w_mesh_axis is not None:
        dims[w_mesh_axis] = 3
    dims = [d if d is not None else (3 if 2 in dims else 2) for d in dims]  # fill unused axis w/ a unique dummy dim
    x_bthwc = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    if dual_cq is not None:
        dual_cq.fence_compute_before_forward()
    out_bthwc = decoder(x_bthwc)  # sharded [B,T,H_out/h,W_out/w,3]
    ttnn.deallocate(x_bthwc, force=False)

    if dual_cq is not None:
        dual_cq.launch_output_d2h(out_bthwc)
        img_host = dual_cq.consume_output_host()
        img_bthwc = ttnn.to_torch(
            img_host,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=dims),
        ).float()
    else:
        img_bthwc = ttnn.to_torch(
            ttnn.from_device(out_bthwc),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=dims),
        ).float()  # [B,T,H_out,W_out,3]
        ttnn.deallocate(out_bthwc, force=False)

    img = img_bthwc.permute(0, 4, 1, 2, 3)  # [B,3,T,H_out,W_out]
    return vae_decode_output_to_rgb(img)  # [B,3,H_out,W_out] in [0,1]
