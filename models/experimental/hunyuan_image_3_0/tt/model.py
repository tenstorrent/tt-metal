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
from .parallel_utils import sp_gather, sp_shard


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
        tp_axis: int = 1,
        tp_factor: int = 1,
        sp_axis: int = 0,
        sp_factor: int = 1,
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
        # Sequence parallel: split the token sequence across `sp_axis`. All SP
        # plumbing is contained in forward() — inputs arrive replicated (full S) and
        # outputs are returned replicated, so the pipeline/demo are unaffected.
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.sp_factor = sp_factor

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
                    tp_axis=tp_axis,
                    tp_factor=tp_factor,
                    sp_axis=sp_axis,
                    sp_factor=sp_factor,
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

        # --- Sequence-parallel entry reshard --------------------------------
        # Split the replicated inputs across sp_axis: hidden + cos/sin on the seq
        # dim, the mask on its QUERY dim (keys stay full so each device attends to
        # the whole sequence). Outputs are gathered back to full S before return.
        sp = self.sp_factor > 1
        sp_owned = []
        sp_pad = 0  # tokens padded so each shard is tile-aligned (sliced off at exit)
        if sp:
            n, ax, ccl = self.sp_factor, self.sp_axis, self.ccl_manager
            # Each shard must be tile-aligned, so pad S up to a multiple of n*TILE.
            # The real gen sequence (e.g. 4107) is neither even nor tile-aligned.
            TILE = 32
            mult = n * TILE
            S_pad = ((seq_len + mult - 1) // mult) * mult
            sp_pad = S_pad - seq_len
            if sp_pad:
                # hidden/cos/sin: zero-pad the seq dim (padded query outputs are
                # discarded at exit; padded keys are masked out below).
                hidden = ttnn.pad(hidden, [(0, 0), (0, sp_pad), (0, 0)], value=0.0)
                cos_tt = ttnn.pad(cos_tt, [(0, 0), (0, 0), (0, sp_pad), (0, 0)], value=0.0)
                sin_tt = ttnn.pad(sin_tt, [(0, 0), (0, 0), (0, sp_pad), (0, 0)], value=0.0)
                if attention_mask is not None:
                    # Mask the padded KEY columns (-1e30) so real queries ignore the
                    # padding; padded query ROWS can be anything (sliced off later).
                    attention_mask = ttnn.pad(attention_mask, [(0, 0), (0, 0), (0, 0), (0, sp_pad)], value=-1.0e30)
                    attention_mask = ttnn.pad(attention_mask, [(0, 0), (0, 0), (0, sp_pad), (0, 0)], value=0.0)
            hidden = sp_shard(ccl, hidden, dim=1, mesh_axis=ax, n=n)  # [B, S_pad/n, H]
            caller_owns_hidden = False  # we created a fresh sharded tensor
            cos_tt = sp_shard(ccl, cos_tt, dim=2, mesh_axis=ax, n=n)  # [1,1,S_pad/n,hd]
            sin_tt = sp_shard(ccl, sin_tt, dim=2, mesh_axis=ax, n=n)
            if attention_mask is not None:
                attention_mask = sp_shard(ccl, attention_mask, dim=2, mesh_axis=ax, n=n)  # [B,1,S_pad/n,S_pad]
                sp_owned.append(attention_mask)

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

        # --- Sequence-parallel exit gather ----------------------------------
        # Re-assemble the full (replicated) sequence so the caller/pipeline sees the
        # same [B, S, H] contract as the non-SP path.
        if sp:
            full = sp_gather(self.ccl_manager, hidden, dim=1, mesh_axis=self.sp_axis, n=self.sp_factor)
            ttnn.deallocate(hidden)
            hidden = full  # [B, S_pad, H]
            if sp_pad:
                # Drop the padding rows -> back to the real [B, S, H] contract.
                shp = list(hidden.shape)
                unpadded = ttnn.slice(hidden, [0, 0, 0], [shp[0], shp[1] - sp_pad, shp[2]])
                ttnn.deallocate(hidden)
                hidden = unpadded
            for t in sp_owned:
                ttnn.deallocate(t)

        if self.ln_f is not None:
            normed = self.ln_f(hidden)
            ttnn.deallocate(hidden)
            hidden = normed

        return hidden
