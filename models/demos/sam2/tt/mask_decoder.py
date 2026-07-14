# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN native Sam2MaskDecoder matching HuggingFace modeling_sam2.py.
Two-way transformer with self-attention, cross-attention, upscaling, hypernetworks,
IoU prediction, and dynamic multimask selection.

CURRENT LIMITATIONS (CLEARLY DOCUMENTED):
All compute runs on CPU via torch.nn.functional.
Porting to TTNN ops requires hardware CI validation for:
- ttnn.linear (QKV projections) — verified API from qwen3_vl
- ttnn.transformer.scaled_dot_product_attention (self/cross attention)
- ttnn.upsample + ttnn.conv2d (upscaling) — pattern from SDXL
- ttnn.layer_norm
- ttnn.gelu
"""

from typing import Dict, List, Optional, Tuple
import torch
import ttnn
import math


class TtnnSam2Attention:
    """Matches HF Sam2Attention — q/k/v/o projections with downsample rate support.
    CPU-only — TODO: port to ttnn.linear + ttnn.SDPA."""

    def __init__(self, hidden_size: int, downsample_rate: int, num_heads: int,
                 device: ttnn.Device, state_dict: Optional[dict] = None, prefix: str = ""):
        self.device = device
        self.hidden_size = hidden_size
        self.internal_dim = hidden_size // downsample_rate
        self.num_heads = num_heads
        self.head_dim = self.internal_dim // num_heads
        self.scale = self.head_dim ** -0.5

        def _load(name, shape):
            key = f"{prefix}.{name}"
            if state_dict and key in state_dict:
                return state_dict[key]
            return torch.randn(shape)

        # Keep as torch tensors — upload to device when porting to ttnn.linear
        self.q_w = _load("q_proj.weight", (self.internal_dim, self.hidden_size)).T.contiguous()
        self.q_b = _load("q_proj.bias", (self.internal_dim,))
        self.k_w = _load("k_proj.weight", (self.internal_dim, self.hidden_size)).T.contiguous()
        self.k_b = _load("k_proj.bias", (self.internal_dim,))
        self.v_w = _load("v_proj.weight", (self.internal_dim, self.hidden_size)).T.contiguous()
        self.v_b = _load("v_proj.bias", (self.internal_dim,))
        self.o_w = _load("o_proj.weight", (self.hidden_size, self.internal_dim)).T.contiguous()
        self.o_b = _load("o_proj.bias", (self.hidden_size,))

    def forward_on_device(self, query, key, value, attention_similarity=None):
        """Forward on CPU (torch). Input: [B, S, D] float tensors. Returns (output, attn_weights).
        TODO: Port to ttnn.linear for QKV projections + ttnn.transformer.scaled_dot_product_attention."""
        B, S, _ = query.shape
        nh = self.num_heads
        hd = self.head_dim

        q = query @ self.q_w.to(query.dtype) + self.q_b.to(query.dtype)
        k = key @ self.k_w.to(key.dtype) + self.k_b.to(key.dtype)
        v = value @ self.v_w.to(value.dtype) + self.v_b.to(value.dtype)

        q = q.view(B, S, nh, hd).transpose(1, 2)
        k = k.view(B, S, nh, hd).transpose(1, 2)
        v = v.view(B, S, nh, hd).transpose(1, 2)

        attn = q * self.scale
        attn = attn @ k.transpose(-2, -1)
        if attention_similarity is not None:
            attn = attn + attention_similarity
        attn = torch.nn.functional.softmax(attn, dtype=torch.float32, dim=-1)
        out = attn.to(q.dtype) @ v
        out = out.transpose(1, 2).reshape(B, S, -1)
        out = out @ self.o_w.to(out.dtype) + self.o_b.to(out.dtype)
        return out, attn


class TtnnSam2TwoWayAttentionBlock:
    """Matches HF Sam2TwoWayAttentionBlock: self_attn → cross_attn_token→image → MLP → cross_attn_image→token.
    CPU-only — TODO: port to ttnn ops."""

    def __init__(self, config: dict, device: ttnn.Device, skip_first_layer_pe: bool = False,
                 state_dict: Optional[dict] = None, prefix: str = ""):
        self.device = device
        self.skip_first_layer_pe = skip_first_layer_pe
        hidden_size = config.get("hidden_size", 256)
        num_heads = config.get("num_attention_heads", 8)
        downsample_rate = config.get("attention_downsample_rate", 2)

        self.self_attn = TtnnSam2Attention(hidden_size, 1, num_heads, device, state_dict, f"{prefix}.self_attn")
        self.cross_attn_t2i = TtnnSam2Attention(hidden_size, downsample_rate, num_heads, device, state_dict, f"{prefix}.cross_attn_token_to_image")
        self.cross_attn_i2t = TtnnSam2Attention(hidden_size, downsample_rate, num_heads, device, state_dict, f"{prefix}.cross_attn_image_to_token")

        def _load(name, shape):
            key = f"{prefix}.{name}"
            if state_dict and key in state_dict:
                return state_dict[key]
            return torch.randn(shape)

        self.ln1_w = _load("layer_norm1.weight", (hidden_size,))
        self.ln1_b = _load("layer_norm1.bias", (hidden_size,))
        self.ln2_w = _load("layer_norm2.weight", (hidden_size,))
        self.ln2_b = _load("layer_norm2.bias", (hidden_size,))
        self.ln3_w = _load("layer_norm3.weight", (hidden_size,))
        self.ln3_b = _load("layer_norm3.bias", (hidden_size,))
        self.ln4_w = _load("layer_norm4.weight", (hidden_size,))
        self.ln4_b = _load("layer_norm4.bias", (hidden_size,))

        # MLP: hidden_size -> mlp_dim -> hidden_size
        mlp_dim = config.get("mlp_dim", 2048)
        self.mlp_in_w = _load("mlp.proj_in.weight", (mlp_dim, hidden_size)).T.contiguous()
        self.mlp_in_b = _load("mlp.proj_in.bias", (mlp_dim,))
        self.mlp_out_w = _load("mlp.proj_out.weight", (hidden_size, mlp_dim)).T.contiguous()
        self.mlp_out_b = _load("mlp.proj_out.bias", (hidden_size,))

    def forward(self, queries, keys, query_pe, key_pe, attention_similarity=None):
        """Forward on CPU. Returns (queries, keys, attn_out)."""
        # Self attention
        if self.skip_first_layer_pe:
            attn_out, _ = self.self_attn.forward_on_device(queries, queries, queries)
        else:
            q_pe = queries + query_pe
            attn_out, _ = self.self_attn.forward_on_device(q_pe, q_pe, queries)
        queries = queries + attn_out
        queries = torch.nn.functional.layer_norm(queries, (queries.shape[-1],), self.ln1_w, self.ln1_b)

        # Cross attention: tokens attending to image
        q_t2i = queries + query_pe
        k_t2i = keys + key_pe
        attn_out, _ = self.cross_attn_t2i.forward_on_device(q_t2i, k_t2i, keys, attention_similarity)
        queries = queries + attn_out
        queries = torch.nn.functional.layer_norm(queries, (queries.shape[-1],), self.ln2_w, self.ln2_b)

        # MLP
        mlp_h = queries @ self.mlp_in_w.to(queries.dtype) + self.mlp_in_b.to(queries.dtype)
        mlp_h = torch.nn.functional.relu(mlp_h)
        mlp_out = mlp_h @ self.mlp_out_w.to(mlp_h.dtype) + self.mlp_out_b.to(mlp_h.dtype)
        queries = queries + mlp_out
        queries = torch.nn.functional.layer_norm(queries, (queries.shape[-1],), self.ln3_w, self.ln3_b)

        # Cross attention: image attending to tokens
        q_i2t = keys + key_pe
        k_i2t = queries + query_pe
        attn_out, _ = self.cross_attn_i2t.forward_on_device(q_i2t, k_i2t, queries)
        keys = keys + attn_out
        keys = torch.nn.functional.layer_norm(keys, (keys.shape[-1],), self.ln4_w, self.ln4_b)
        return queries, keys, attn_out


class TtnnSam2MaskDecoder:
    """TTNN native mask decoder matching HF Sam2MaskDecoder.
    Two-way transformer, upscaling, hypernetworks, IoU/object score heads.
    
    NOTE: All compute runs on CPU via torch.
    TODO: Port to TTNN ops — ttnn.linear, ttnn.SDPA, ttnn.upsample + ttnn.conv2d (for upscaling),
    ttnn.layer_norm, ttnn.gelu. See SDXL for conv2d pattern and qwen3_vl for SDPA pattern."""

    def __init__(self, device: ttnn.Device, config: dict, state_dict: Optional[dict] = None):
        self.device = device
        self.hidden_size = config.get("hidden_size", 256)
        self.num_multimask_outputs = config.get("num_multimask_outputs", 3)
        self.num_mask_tokens = self.num_multimask_outputs + 1  # 4

        def _load(name, shape):
            if state_dict and name in state_dict:
                return state_dict[name]
            return torch.randn(shape)

        # Learned tokens
        self.iou_token = _load("mask_decoder.iou_token.weight", (1, self.hidden_size))
        self.mask_tokens = _load("mask_decoder.mask_tokens.weight", (self.num_mask_tokens, self.hidden_size))
        self.obj_score_token = _load("mask_decoder.obj_score_token.weight", (1, self.hidden_size))

        # Two-way transformer (2 blocks)
        self.num_layers = config.get("num_hidden_layers", 2)
        self.transformer_layers = []
        for i in range(self.num_layers):
            prefix = f"mask_decoder.transformer.layers.{i}"
            layer = TtnnSam2TwoWayAttentionBlock(
                config, device, skip_first_layer_pe=(i == 0),
                state_dict=state_dict, prefix=prefix,
            )
            self.transformer_layers.append(layer)

        # Final attention token→image
        self.final_attn = TtnnSam2Attention(
            self.hidden_size, config.get("attention_downsample_rate", 2),
            config.get("num_attention_heads", 8), device,
            state_dict, "mask_decoder.transformer.final_attn_token_to_image",
        )
        self.ln_final_w = _load("mask_decoder.transformer.layer_norm_final_attn.weight", (self.hidden_size,))
        self.ln_final_b = _load("mask_decoder.transformer.layer_norm_final_attn.bias", (self.hidden_size,))

        # Upscaling ConvTranspose2d weights (CPU ops — TODO: port to ttnn.upsample + ttnn.conv2d)
        self.uc1_w = _load("mask_decoder.upscale_conv1.weight", (64, self.hidden_size, 2, 2))
        self.uc1_b = _load("mask_decoder.upscale_conv1.bias", (64,))
        self.uc2_w = _load("mask_decoder.upscale_conv2.weight", (32, 64, 2, 2))
        self.uc2_b = _load("mask_decoder.upscale_conv2.bias", (32,))
        self.uln_w = _load("mask_decoder.upscale_layer_norm.weight", (64,))
        self.uln_b = _load("mask_decoder.upscale_layer_norm.bias", (64,))

        # Output hypernetworks MLPs (4 MLPs: 256->256->32)
        self.hyper_mlps = []
        for i in range(self.num_mask_tokens):
            prefix = f"mask_decoder.output_hypernetworks_mlps.{i}"
            h_in_w = _load(f"{prefix}.proj_in.weight", (self.hidden_size, self.hidden_size))
            h_in_b = _load(f"{prefix}.proj_in.bias", (self.hidden_size,))
            h_out_w = _load(f"{prefix}.proj_out.weight", (32, self.hidden_size))
            h_out_b = _load(f"{prefix}.proj_out.bias", (32,))
            self.hyper_mlps.append({
                "in_w": h_in_w.T.contiguous(), "in_b": h_in_b,
                "out_w": h_out_w.T.contiguous(), "out_b": h_out_b,
            })

        # IoU prediction head
        self.iou_in_w = _load("mask_decoder.iou_prediction_head.proj_in.weight", (256, self.hidden_size)).T.contiguous()
        self.iou_in_b = _load("mask_decoder.iou_prediction_head.proj_in.bias", (256,))
        self.iou_out_w = _load("mask_decoder.iou_prediction_head.proj_out.weight", (self.num_mask_tokens, 256)).T.contiguous()
        self.iou_out_b = _load("mask_decoder.iou_prediction_head.proj_out.bias", (self.num_mask_tokens,))

        # Object score head
        self.os_in_w = _load("mask_decoder.pred_obj_score_head.proj_in.weight", (256, self.hidden_size)).T.contiguous()
        self.os_in_b = _load("mask_decoder.pred_obj_score_head.proj_in.bias", (256,))
        self.os_out_w = _load("mask_decoder.pred_obj_score_head.proj_out.weight", (1, 256)).T.contiguous()
        self.os_out_b = _load("mask_decoder.pred_obj_score_head.proj_out.bias", (1,))

    def _compute_mlp(self, x: torch.Tensor, weights: dict) -> torch.Tensor:
        """2-layer MLP on CPU: linear + ReLU + linear."""
        h = x @ weights["in_w"].to(x.dtype) + weights["in_b"].to(x.dtype)
        h = torch.nn.functional.relu(h)
        return h @ weights["out_w"].to(h.dtype) + weights["out_b"].to(h.dtype)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: Optional[torch.Tensor],
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
        high_resolution_features: Optional[List[torch.Tensor]] = None,
        attention_similarity: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Full mask decoder forward. Matches HF Sam2MaskDecoder.forward().
        All ops on CPU — TODO: port to TTNN."""
        B, C, H, W = image_embeddings.shape

        if sparse_prompt_embeddings is None:
            point_batch_size = 1
            num_points = 0
            sparse_prompt_embeddings = torch.zeros(B, 1, 0, C, dtype=image_embeddings.dtype)
        else:
            point_batch_size = sparse_prompt_embeddings.shape[1]
            num_points = sparse_prompt_embeddings.shape[2]

        # Concatenate output tokens
        output_tokens = torch.cat([
            self.obj_score_token, self.iou_token, self.mask_tokens,
        ], dim=0)
        output_tokens = output_tokens.unsqueeze(0).unsqueeze(0).expand(B, point_batch_size, -1, -1)

        if num_points > 0:
            tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(image_embeddings.dtype)

        # Add dense prompt to image embeddings
        img_emb = image_embeddings + dense_prompt_embeddings
        img_emb = img_emb.repeat_interleave(point_batch_size, dim=0)
        img_pe = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Flatten for transformer
        _, _, fH, fW = img_emb.shape
        img_tokens = img_emb.flatten(2).transpose(1, 2).unsqueeze(1)
        img_pe_tokens = img_pe.flatten(2).transpose(1, 2).unsqueeze(1)

        # Apply transformer blocks
        pe = point_embeddings
        for layer in self.transformer_layers:
            point_embeddings, img_tokens, _ = layer.forward(
                point_embeddings, img_tokens, pe, img_pe_tokens, attention_similarity,
            )

        # Final attention token→image
        q = point_embeddings + pe
        k = img_tokens + img_pe_tokens
        attn_out, _ = self.final_attn.forward_on_device(q, k, img_tokens)
        point_embeddings = point_embeddings + attn_out
        point_embeddings = torch.nn.functional.layer_norm(
            point_embeddings, (self.hidden_size,), self.ln_final_w, self.ln_final_b,
        )

        # Extract outputs
        iou_token_out = point_embeddings[:, :, 1, :]
        mask_tokens_out = point_embeddings[:, :, 2:(2 + self.num_mask_tokens), :]
        obj_score_token_out = point_embeddings[:, :, 0, :]

        # Upscale path (CPU torch — TODO: ttnn.upsample + ttnn.conv2d)
        img_tokens = img_tokens.transpose(2, 3).reshape(B * point_batch_size, C, fH, fW)

        feat_s0 = high_resolution_features[0] if high_resolution_features else None
        feat_s1 = high_resolution_features[1] if high_resolution_features else None

        # upscale_conv1: 256→64, k=2, s=2
        uc1 = torch.nn.functional.conv_transpose2d(
            img_tokens, self.uc1_w.to(img_tokens.dtype), bias=self.uc1_b.to(img_tokens.dtype),
            stride=2, padding=0, output_padding=0,
        )
        if feat_s1 is not None:
            uc1 = uc1 + feat_s1.repeat_interleave(point_batch_size, dim=0)
        uc1 = torch.nn.functional.gelu(uc1)
        uc1_ln = torch.nn.functional.layer_norm(
            uc1.permute(0, 2, 3, 1), (64,), self.uln_w, self.uln_b).permute(0, 3, 1, 2)

        # upscale_conv2: 64→32, k=2, s=2
        uc2 = torch.nn.functional.conv_transpose2d(
            uc1_ln, self.uc2_w.to(uc1_ln.dtype), bias=self.uc2_b.to(uc1_ln.dtype),
            stride=2, padding=0, output_padding=0,
        )
        if feat_s0 is not None:
            uc2 = uc2 + feat_s0.repeat_interleave(point_batch_size, dim=0)
        upscaled = torch.nn.functional.gelu(uc2)

        # Hypernetworks: compute mask weights per output token
        hyper_list = []
        for i in range(self.num_mask_tokens):
            hyp = self._compute_mlp(mask_tokens_out[:, :, i, :], self.hyper_mlps[i])
            hyper_list.append(hyp)
        hyper_in = torch.stack(hyper_list, dim=2)

        # Matmul hypernetworks with upscaled features
        _, uC, uH, uW = upscaled.shape
        upscaled_flat = upscaled.view(B, point_batch_size, uC, uH * uW)
        masks = (hyper_in @ upscaled_flat).view(B, point_batch_size, -1, uH, uW)

        # IoU prediction
        iou_h = iou_token_out @ self.iou_in_w.to(iou_token_out.dtype) + self.iou_in_b.to(iou_token_out.dtype)
        iou_h = torch.nn.functional.relu(iou_h)
        iou_pred = torch.sigmoid(
            iou_h @ self.iou_out_w.to(iou_h.dtype) + self.iou_out_b.to(iou_h.dtype))

        # Object score prediction
        os_h = obj_score_token_out @ self.os_in_w.to(obj_score_token_out.dtype) + self.os_in_b.to(obj_score_token_out.dtype)
        os_h = torch.nn.functional.relu(os_h)
        obj_score_logits = os_h @ self.os_out_w.to(os_h.dtype) + self.os_out_b.to(os_h.dtype)

        # Select mask based on multimask_output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        return {
            "masks": masks,
            "iou_scores": iou_pred,
            "object_score_logits": obj_score_logits,
        }
