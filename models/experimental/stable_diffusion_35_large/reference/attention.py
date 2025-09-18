# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from .normalization import RmsNorm


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class Attention(torch.nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        dim_head: int,
        heads: int,
        out_dim: int,
        qk_norm: str,
        added_kv_proj_dim: int = 0,
        context_pre_only: bool = True,
    ) -> None:
        super().__init__()

        if qk_norm != "rms_norm":
            msg = "invalid qk_norm"
            raise ValueError(msg)

        eps = 1e-6

        self.context_pre_only = context_pre_only
        self.head_dim = dim_head
        self.num_heads = heads

        self.norm_q = RmsNorm(dim=dim_head, eps=eps)
        self.norm_k = RmsNorm(dim=dim_head, eps=eps)

        self.to_q = torch.nn.Linear(query_dim, out_dim)
        self.to_k = torch.nn.Linear(query_dim, out_dim)
        self.to_v = torch.nn.Linear(query_dim, out_dim)

        if added_kv_proj_dim > 0:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, out_dim)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, out_dim)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, out_dim)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(out_dim, out_dim))

        if not self.context_pre_only:
            self.to_add_out = torch.nn.Linear(out_dim, out_dim)

        if added_kv_proj_dim > 0:
            self.norm_added_q = RmsNorm(dim=dim_head, eps=eps)
            self.norm_added_k = RmsNorm(dim=dim_head, eps=eps)
        else:
            self.norm_added_q = None
            self.norm_added_k = None

    def forward(
        self,
        *,
        spatial: torch.Tensor,
        prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = spatial.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim

        residual = spatial

        query = self.to_q(spatial)
        key = self.to_k(spatial)
        value = self.to_v(spatial)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if prompt is not None:
            prompt_query_proj = self.add_q_proj(prompt)
            prompt_key_proj = self.add_k_proj(prompt)
            prompt_value_proj = self.add_v_proj(prompt)

            prompt_query_proj = prompt_query_proj.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            prompt_key_proj = prompt_key_proj.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            prompt_value_proj = prompt_value_proj.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            if self.norm_added_q is not None:
                prompt_query_proj = self.norm_added_q(prompt_query_proj)
            if self.norm_added_k is not None:
                prompt_key_proj = self.norm_added_k(prompt_key_proj)

            query = torch.cat([query, prompt_query_proj], dim=2)
            key = torch.cat([key, prompt_key_proj], dim=2)
            value = torch.cat([value, prompt_value_proj], dim=2)

        spatial = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        spatial = spatial.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)
        spatial = spatial.to(query.dtype)

        if prompt is not None:
            spatial, prompt = (
                spatial[:, : residual.shape[1]],
                spatial[:, residual.shape[1] :],
            )
            if not self.context_pre_only:
                prompt = self.to_add_out(prompt)

        spatial = self.to_out[0](spatial)

        return spatial, prompt
