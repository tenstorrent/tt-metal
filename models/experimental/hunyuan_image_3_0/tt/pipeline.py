# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN diffusion pipeline for HunyuanImage-3.0 — composes the already-ported,
# individually-PCC-gated blocks into a denoise step and a multi-step loop.
#
# This is the "next milestone" referenced in tests/pcc/test_denoise_step.py:
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
#   * Single contiguous image span (the T2I case). The on-device scatter uses
#     ttnn.concat, so the span boundaries must be TILE-aligned (multiples of 32)
#     in TILE layout. T2I layouts satisfy this; ragged/multi-image (Instruct)
#     layouts will need the host-scatter fallback (`scatter_on_host=True`).
#   * VAE decode (latent -> pixels) and tokenizer/sequence construction are
#     separate milestones — see README. This module stops at the latent.

import ttnn

from .scheduler import classifier_free_guidance_tt


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
        if toks.dtype != ref.dtype:
            toks = ttnn.typecast(toks, ref.dtype)
        pieces = [p for p in (text_pre, toks, text_post) if p is not None]
        seq = ttnn.concat(pieces, dim=1)
        ttnn.deallocate(toks)
        return seq

    # -- one step -----------------------------------------------------------
    def __call__(
        self,
        latent_bchw,  # torch [B,C,h,w] or ttnn NHWC flat — UNetDown entry
        *,
        text_pre,  # ttnn [B, text_pre_len, H] TILE  (sequence before image span)
        text_post,  # ttnn [B, text_post_len, H] TILE (after image span); may be None
        t_emb1,  # ttnn [1,1,B,H] timestep embedding for patch_embed
        t_emb2,  # ttnn [1,1,B,H] timestep embedding for final_layer
        image_infos,  # 2D-RoPE image span info, e.g. [[(img_slice,(h,w))]]
        attention_mask,  # ttnn [B,1,S,S] additive mask
        batch: int = 1,
    ):
        self.backbone_batch = batch

        # 1) patch_embed: noised latent -> image tokens [1,1,n_img,H]
        img_tok, th, tw = self.patch_embed(latent_bchw, t_emb1)
        assert (th, tw) == (self.token_h, self.token_w), f"grid mismatch: got {th}x{tw}"

        # 2) scatter into the sequence, run the backbone (NO ln_f)
        seq = self._scatter(img_tok, text_pre, text_post)
        ttnn.deallocate(img_tok)
        hidden = self.backbone.forward(
            inputs_embeds=seq,
            seq_len=self.seq_len,
            image_infos=image_infos,
            attention_mask=attention_mask,
        )
        ttnn.deallocate(seq)

        # 3) final_layer: image-span hidden -> velocity prediction (NHWC flat)
        H = self.backbone.hidden_size
        img_out = ttnn.slice(
            hidden,
            [0, self.img_slice.start, 0],
            [batch, self.img_slice.stop, H],
        )
        ttnn.deallocate(hidden)
        img_out = ttnn.reshape(img_out, [1, 1, self.n_img, H])
        pred, _, _ = self.final_layer(img_out, t_emb2, th, tw, B=batch)
        ttnn.deallocate(img_out)
        return pred  # ttnn NHWC flat [1,1,B*h*w,latent_ch]


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
):
    """Run the diffusion denoise loop, returning the final latent (torch NCHW).

    `cond`/`uncond` carry the static, timestep-independent conditioning forwarded
    to `step.__call__`:
        text_pre, text_post, image_infos, attention_mask, batch.
    The timestep embeddings ARE recomputed each step from `time_embed` /
    `time_embed_2` (both passes share them — the timestep is identical), and the
    per-step velocity prediction drives the on-device Euler update via the
    scheduler. With CFG, the conditional and unconditional predictions are
    combined on device before the update.

    Latent representation: the scheduler operates on device NHWC-flat tensors,
    but UNetDown's entry only accepts torch NCHW, so the (small) latent makes one
    host hop per step. The heavy compute (backbone, MoE) stays on device. A
    future patch_embed change to accept a device NHWC-flat latent would remove
    this hop; the latent is tiny relative to the backbone, so it is not on the
    critical path.
    """
    import torch

    B, C, h, w = init_latent.shape
    scheduler.set_begin_index(0)
    do_cfg = uncond is not None and guidance_scale != 1.0
    latent = init_latent  # torch NCHW (canonical host form; small tensor)

    for t in scheduler.timesteps:
        tvec = torch.tensor([float(t)] * B)
        te1 = time_embed.forward(tvec)  # [1,1,B,H]
        te2 = time_embed_2.forward(tvec)

        def _one(c):
            return step(
                latent,
                text_pre=c["text_pre"],
                text_post=c.get("text_post"),
                t_emb1=te1,
                t_emb2=te2,
                image_infos=c["image_infos"],
                attention_mask=c["attention_mask"],
                batch=c.get("batch", B),
            )

        pred = _one(cond)  # device NHWC flat [1,1,B*h*w,C]
        if do_cfg:
            pred_uncond = _one(uncond)
            combined = classifier_free_guidance_tt(pred, pred_uncond, guidance_scale)
            ttnn.deallocate(pred)
            ttnn.deallocate(pred_uncond)
            pred = combined

        # On-device Euler update: prev = sample + (sigma_next - sigma) * pred.
        sample = ttnn.from_torch(
            latent.permute(0, 2, 3, 1).reshape(1, 1, B * h * w, C).contiguous(),
            dtype=pred.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=step.device,
        )
        nxt = scheduler.step(pred, t, sample)
        latent = ttnn.to_torch(nxt).reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

        ttnn.deallocate(pred)
        ttnn.deallocate(sample)
        ttnn.deallocate(nxt)
        ttnn.deallocate(te1)
        ttnn.deallocate(te2)

    return latent  # torch [B, C, h, w] — feed to VAE decode
