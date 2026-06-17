# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of the HunyuanImage-3.0 transformer backbone.
# Mirrors HunyuanImage3Model in
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#
#     hidden = wte(input_ids)                # or caller-supplied inputs_embeds
#     for layer in layers:                   # 32 identical MoE decoder layers
#         hidden = layer(hidden, rope, mask)
#     hidden = ln_f(hidden)                  # OPTIONAL — see note below
#
# Note on ln_f: upstream applies the final RMSNorm *outside* the model in the
# image-generation path (see HunyuanImage3Model.forward — the ln_f call is
# commented out). We mirror that by making the final norm opt-in via
# `apply_final_norm` (default True for a standalone LM backbone; pass False to
# match the image-gen call site).
#
# Memory: with stream_experts=True each HunyuanTtDecoderLayer keeps a host-RAM
# reference to its layer's torch expert weights and rebuilds experts from them
# every forward. Holding all 32 layers resident is therefore ~150GB of host
# RAM. For the full 32-layer model this loader must be backed by on-demand disk
# streaming (a future change); the small-stack PCC test holds only a few layers
# and fits comfortably. `layer_loader(i)` returns the state_dict for layer i,
# keyed `model.layers.{i}.*`.

import ttnn
from models.common.lightweightmodule import LightweightModule

from .transformer_layer import HunyuanTtDecoderLayer
from .attention.rms_norm import HunyuanTtRMSNorm


class HunyuanTtModel(LightweightModule):
    def __init__(
        self,
        device,
        *,
        num_layers: int = 32,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        num_experts: int = 64,
        moe_topk: int = 8,
        use_qk_norm: bool = True,
        use_mixed_mlp_moe: bool = True,
        norm_topk_prob: bool = True,
        rms_norm_eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        stream_experts: bool = True,
        layer_loader=None,
        embed_state_dict: dict = None,
        norm_state_dict: dict = None,
        apply_final_norm: bool = True,
        ccl_manager=None,
        expert_mesh_axis: int = 1,
        bf16_layers=None,
    ):
        """
        Args:
            device:           TTNN device.
            num_layers:       number of decoder layers to stack (1..32).
            layer_loader:     callable(layer_idx) -> state_dict for that layer,
                              keyed `model.layers.{layer_idx}.*`. Called once per
                              layer at construction.
            embed_state_dict: dict containing `model.wte.weight` ([V, H]). Required
                              only if forward() is called with input_ids.
            norm_state_dict:  dict containing `model.ln_f.weight`. Required only if
                              apply_final_norm is True.
            apply_final_norm: apply ln_f at the end (LM backbone). Pass False to
                              match the image-generation call site.
        """
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.apply_final_norm = apply_final_norm

        # Token embedding table (ROW_MAJOR weight; ttnn.embedding emits TILE).
        self.embed_weight = None
        if embed_state_dict is not None:
            w = embed_state_dict["model.wte.weight"]  # [V, H]
            self.embed_weight = ttnn.from_torch(
                w,
                dtype=weight_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if layer_loader is None:
            raise ValueError("layer_loader is required")

        # Per-layer mixed precision: layers in `bf16_layers` keep bf16 expert
        # weights (more accurate, 2x memory); the rest use `weight_dtype` (bf8).
        # Lets us trade DRAM headroom for accuracy on the most sensitive layers.
        bf16_layers = set(bf16_layers or [])

        self.layers = []
        for i in range(num_layers):
            sd = layer_loader(i)
            layer_dtype = ttnn.bfloat16 if i in bf16_layers else weight_dtype
            self.layers.append(
                HunyuanTtDecoderLayer(
                    device,
                    sd,
                    layer_num=i,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    num_experts=num_experts,
                    moe_topk=moe_topk,
                    use_qk_norm=use_qk_norm,
                    use_mixed_mlp_moe=use_mixed_mlp_moe,
                    norm_topk_prob=norm_topk_prob,
                    rms_norm_eps=rms_norm_eps,
                    weight_dtype=layer_dtype,
                    stream_experts=stream_experts,
                    ccl_manager=ccl_manager,
                    expert_mesh_axis=expert_mesh_axis,
                )
            )

        self.ln_f = None
        if apply_final_norm:
            if norm_state_dict is None:
                raise ValueError("norm_state_dict with 'model.ln_f.weight' is required when apply_final_norm=True")
            self.ln_f = HunyuanTtRMSNorm(device, hidden_size, norm_state_dict, "model.ln_f", eps=rms_norm_eps)

    def embed(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Embed input_ids ([B, S] uint32, ROW_MAJOR) -> [B, S, H] TILE."""
        if self.embed_weight is None:
            raise ValueError("model was built without embed_state_dict; pass inputs_embeds to forward() instead")
        emb = ttnn.embedding(input_ids, self.embed_weight, layout=ttnn.TILE_LAYOUT)
        # Normalise to rank-3 [B, S, H] (ttnn.embedding may emit a leading 1-dim);
        # downstream attention assumes a 3-D hidden tensor.
        bsz = input_ids.shape[0]
        seq = input_ids.shape[-1]
        return ttnn.reshape(emb, [bsz, seq, self.hidden_size])

    def forward(
        self,
        input_ids: ttnn.Tensor = None,
        *,
        inputs_embeds: ttnn.Tensor = None,
        seq_len: int,
        image_infos=None,
        attention_mask=None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids:      [B, S] uint32 ROW_MAJOR token ids (embedded on device).
            inputs_embeds:  [B, S, H] TILE — pre-embedded hidden states; takes
                            precedence over input_ids (matches the image-gen path).
            seq_len:        S — used to build the shared 2D RoPE tables.
            image_infos:    per-batch image span info for 2D RoPE (None => text-only).
            attention_mask: optional additive mask [B,1,S,S]; None => causal SDPA.
        Returns:
            [B, S, H] TTNN tensor (final hidden states, ln_f applied iff
            apply_final_norm).
        """
        # Track whether `hidden` is caller-owned (inputs_embeds) — if so, don't
        # free it after the first layer.
        if inputs_embeds is not None:
            hidden = inputs_embeds
            caller_owns_hidden = True
        elif input_ids is not None:
            hidden = self.embed(input_ids)
            caller_owns_hidden = False
        else:
            raise ValueError("provide either input_ids or inputs_embeds")

        # Build the 2D RoPE tables once and share them across all layers.
        cos_tt, sin_tt = self.layers[0].self_attn.rope.prepare_cos_sin(seq_len, image_infos=image_infos)

        for layer in self.layers:
            nxt = layer(
                hidden,
                seq_len=seq_len,
                image_infos=image_infos,
                attention_mask=attention_mask,
                cos_sin=(cos_tt, sin_tt),
            )
            if not caller_owns_hidden:
                ttnn.deallocate(hidden)
            caller_owns_hidden = False
            hidden = nxt

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        if self.ln_f is not None:
            normed = self.ln_f(hidden)
            ttnn.deallocate(hidden)
            hidden = normed

        return hidden
