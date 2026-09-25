# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QwenImageTransformer2DModel forward, chained from the graduated transformer ports.

    img  = img_in(cat[latents, image_latents])                       patch_embed port (Linear 64 -> 3072)
    txt  = txt_in(txt_norm(prompt_embeds))                           RMSNorm(3584) + the same Linear port
    temb = qwen_timestep_proj_embeddings(t / 1000)                   timesteps -> timestep_embedding
    rope = qwen_embed_rope([(1,h,w), (1,h,w)], txt_len)              computed once per run
    txt, img = qwen_image_transformer_block_i(img, txt, temb, rope, mask)   x 60 (attention + feed_forward)
    out  = proj_out(ada_layer_norm_continuous(img, temb))            decoder_head port
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import (
    ada_layer_norm_continuous,
    decoder_head,
    patch_embed,
    qwen_embed_rope,
    qwen_image_transformer_block,
    qwen_timestep_proj_embeddings,
)


def _hifi(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class TtRMSNorm32:
    """diffusers RMSNorm(dim, eps, elementwise_affine=True) in float32 (txt_norm)."""

    def __init__(self, device, torch_module):
        self.eps = float(torch_module.eps)
        w = torch_module.weight.detach().to(torch.float32).reshape(1, -1)
        self.w = ttnn.from_torch(
            w,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self.cfg = _hifi(device)

    def __call__(self, x):
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True, compute_kernel_config=self.cfg)
        return ttnn.multiply(ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, self.eps))), self.w)


class TtQwenImageTransformer:
    def __init__(self, device, torch_module, layers=None, tracker=None):
        self.device = device
        self.hf_config = torch_module.config
        n = (
            len(torch_module.transformer_blocks)
            if layers is None
            else max(1, min(int(layers), len(torch_module.transformer_blocks)))
        )
        self.num_layers = n
        tr = tracker

        self.img_in = patch_embed.build(device, torch_module.img_in)
        self.txt_norm = TtRMSNorm32(device, torch_module.txt_norm)
        self.txt_in = patch_embed.build(device, torch_module.txt_in)

        self.time_text_embed = qwen_timestep_proj_embeddings.build(device, torch_module.time_text_embed)
        self.pos_embed = qwen_embed_rope.build(device, torch_module.pos_embed)
        # The repeated stack: a plain list of same-typed graduated block ports.
        self.transformer_blocks = [
            qwen_image_transformer_block.build(device, blk) for blk in list(torch_module.transformer_blocks)[:n]
        ]
        self.norm_out = ada_layer_norm_continuous.build(device, torch_module.norm_out)
        self.proj_out = decoder_head.build(device, torch_module.proj_out)

        if tr is not None:
            from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import (
                feed_forward,
                timestep_embedding,
                timesteps,
            )

            tr.track("qwen_timestep_proj_embeddings", self.time_text_embed, stub_module=qwen_timestep_proj_embeddings)
            tr.track("timesteps", self.time_text_embed.time_proj, stub_module=timesteps)
            tr.track("timestep_embedding", self.time_text_embed.timestep_embedder, stub_module=timestep_embedding)
            tr.track("qwen_embed_rope", self.pos_embed, stub_module=qwen_embed_rope)
            for blk in self.transformer_blocks:
                tr.track("qwen_image_transformer_block", blk, stub_module=qwen_image_transformer_block)
                inner = blk.stack.blocks[0]
                tr.track("feed_forward", inner.img_ff, stub_module=feed_forward)
                tr.track("feed_forward", inner.txt_ff, stub_module=feed_forward)
            tr.track("ada_layer_norm_continuous", self.norm_out, stub_module=ada_layer_norm_continuous)

    def rope(self, img_shapes, txt_len):
        """(img_freqs [S_img, 128], txt_freqs [txt_len, 128]) float32 [cos | sin] device tables."""
        return self.pos_embed([list(img_shapes)], max_txt_seq_len=int(txt_len))

    def __call__(self, hidden_states, encoder_hidden_states, timestep, rotary, attention_mask=None):
        """hidden_states [B, S_img, 64] fp32, encoder_hidden_states [B, L, 3584] fp32, timestep [B] (= t/1000),
        rotary from rope(), attention_mask [B, 1, 1, L + S_img] (1 = attend) or None. -> [B, S_img, 64] fp32."""
        img = self.img_in(hidden_states)
        txt = self.txt_in(self.txt_norm(encoder_hidden_states))
        temb = self.time_text_embed(timestep, img)
        jak = {"attention_mask": attention_mask} if attention_mask is not None else None
        for blk in self.transformer_blocks:
            txt, img = blk(
                img, encoder_hidden_states=txt, temb=temb, image_rotary_emb=rotary, joint_attention_kwargs=jak
            )
        img = self.norm_out(img, temb)
        return self.proj_out(img)
