# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import ttnn

from . import utils
from .linear import Linear, LinearParameters
from .normalization import RmsNorm, RmsNormParameters
from .substate import has_substate, substate


@dataclass
class AttentionPartParameters:
    mesh_width: int
    qkv_proj: LinearParameters
    norm_q: RmsNormParameters
    norm_k: RmsNormParameters
    out_proj: LinearParameters | None


@dataclass
class AttentionParameters:
    mesh_width: int
    spatial: AttentionPartParameters
    prompt: AttentionPartParameters | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> AttentionParameters:
        _, mesh_width = device.shape

        return cls(
            spatial=AttentionPartParameters(
                qkv_proj=LinearParameters.from_torch(
                    _merge_qkv_proj(substate(state, "to_q"), substate(state, "to_k"), substate(state, "to_v")),
                    dtype=dtype,
                    device=device,
                    mesh_sharding_dim=0,
                    chunks=3,
                ),
                norm_q=RmsNormParameters.from_torch(substate(state, "norm_q"), dtype=dtype, device=device),
                norm_k=RmsNormParameters.from_torch(substate(state, "norm_k"), dtype=dtype, device=device),
                out_proj=(
                    LinearParameters.from_torch(
                        substate(state, "to_out.0"),
                        dtype=dtype,
                        device=device,
                        mesh_sharding_dim=0,
                    )
                    if has_substate(state, "to_out.0")
                    else None
                ),
                mesh_width=mesh_width,
            ),
            prompt=AttentionPartParameters(
                qkv_proj=(
                    LinearParameters.from_torch(
                        _merge_qkv_proj(
                            substate(state, "add_q_proj"), substate(state, "add_k_proj"), substate(state, "add_v_proj")
                        ),
                        dtype=dtype,
                        device=device,
                        mesh_sharding_dim=0,
                        chunks=3,
                    )
                    if has_substate(state, "add_q_proj")
                    else None
                ),
                norm_q=RmsNormParameters.from_torch(substate(state, "norm_added_q"), dtype=dtype, device=device),
                norm_k=RmsNormParameters.from_torch(substate(state, "norm_added_k"), dtype=dtype, device=device),
                out_proj=(
                    LinearParameters.from_torch(
                        substate(state, "to_add_out"),
                        dtype=dtype,
                        device=device,
                        mesh_sharding_dim=0,
                    )
                    if has_substate(state, "add_q_proj")
                    else None
                ),
                mesh_width=mesh_width,
            )
            if has_substate(state, "add_q_proj")
            else None,
            mesh_width=mesh_width,
        )


class AttentionPart:
    def __init__(self, parameters: AttentionPartParameters) -> None:
        super().__init__()

        eps = 1e-6

        self._qkv_proj = Linear(parameters.qkv_proj)
        self._out_proj = Linear(parameters.out_proj) if parameters.out_proj is not None else None
        self._norm_q = RmsNorm(parameters.norm_q, eps=eps)
        self._norm_k = RmsNorm(parameters.norm_k, eps=eps)

        self._mesh_width = parameters.mesh_width

    def qkv(self, x: ttnn.Tensor, *, num_heads: int) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        utils.signpost("qkv preparation")

        # moving the tensor to L1 is slower
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        x = self._qkv_proj.forward(x, skip_reduce_scatter=True)
        x = self._qkv_proj.reduce_scatter(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            x,
            num_heads=num_heads // self._mesh_width,
            transpose_key=False,
        )
        del x

        q = self._norm_q.forward(q)
        k = self._norm_k.forward(k)

        return q, k, v

    def out_proj(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._out_proj is None:
            return x

        # moving the tensor to L1 is slower
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        return self._out_proj.forward(x)


class Attention:
    def __init__(self, parameters: AttentionParameters, *, num_heads: int) -> None:
        super().__init__()

        self._num_heads = num_heads

        self._spatial_attn = AttentionPart(parameters.spatial)
        self._prompt_attn = AttentionPart(parameters.prompt) if parameters.prompt is not None else None
        self._mesh_width = parameters.mesh_width

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor | None = None,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        spatial: N ⊗ S1 ⊗ (H * E1)
        prompt: N ⊗ S2 ⊗ (H * E2)
        """
        device = spatial.device()

        opt = self._mesh_width > 1
        _, spatial_seq_len, _ = spatial.shape

        if opt:
            spatial = ttnn.to_memory_config(spatial, ttnn.L1_MEMORY_CONFIG)

        q, k, v = self._spatial_attn.qkv(spatial, num_heads=self._num_heads)

        if image_rotary_emb is not None:
            emb = (
                (
                    image_rotary_emb[0][-spatial_seq_len:],
                    image_rotary_emb[1][-spatial_seq_len:],
                )
                if prompt is not None
                else image_rotary_emb
            )
            q = _apply_rotary_emb(q, emb)
            k = _apply_rotary_emb(k, emb)

        # operands to SDPA are required to be in DRAM
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)

        if prompt is None:
            utils.signpost("dot product attention path I")
            attn = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, **self._sdpa_settings(device=device)
            )
            del q, k, v
            if opt:
                attn = ttnn.to_memory_config(attn, ttnn.L1_MEMORY_CONFIG)

            attn = ttnn.transformer.concatenate_heads(attn)

            spatial = self._spatial_attn.out_proj(attn)
            return spatial, None

        assert self._prompt_attn is not None

        q2, k2, v2 = self._prompt_attn.qkv(prompt, num_heads=self._num_heads)

        if image_rotary_emb is not None:
            emb = (
                image_rotary_emb[0][:-spatial_seq_len],
                image_rotary_emb[1][:-spatial_seq_len],
            )
            q2 = _apply_rotary_emb(q2, emb)
            k2 = _apply_rotary_emb(k2, emb)

        # operands to SDPA are required to be in DRAM
        q2 = ttnn.to_memory_config(q2, ttnn.DRAM_MEMORY_CONFIG)
        k2 = ttnn.to_memory_config(k2, ttnn.DRAM_MEMORY_CONFIG)
        v2 = ttnn.to_memory_config(v2, ttnn.DRAM_MEMORY_CONFIG)

        utils.signpost("dot product attention path II")
        prompt, spatial = ttnn.transformer.joint_scaled_dot_product_attention(
            q2, k2, v2, q, k, v, joint_strategy="rear", **self._sdpa_settings(device=device)
        )
        del q, k, v, q2, k2, v2
        if opt:
            prompt = ttnn.to_memory_config(prompt, ttnn.L1_MEMORY_CONFIG)
            spatial = ttnn.to_memory_config(spatial, ttnn.L1_MEMORY_CONFIG)

        spatial = ttnn.transformer.concatenate_heads(spatial)
        prompt = ttnn.transformer.concatenate_heads(prompt)

        spatial = self._spatial_attn.out_proj(spatial)
        prompt = self._prompt_attn.out_proj(prompt)

        return spatial, prompt

    def _sdpa_settings(self, *, device: ttnn.Device) -> dict[str, Any]:
        return dict(
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=256,
                k_chunk_size=512,
                exp_approx_mode=True,
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            ),
            # setting this gives wrong results:
            # memory_config=ttnn.L1_MEMORY_CONFIG,
        )


def _merge_qkv_proj(
    q_state: dict[str, torch.Tensor | None],
    k_state: dict[str, torch.Tensor | None],
    v_state: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.cat([q_state["weight"], k_state["weight"], v_state["weight"]]),
        "bias": torch.cat([q_state["bias"], k_state["bias"], v_state["bias"]]),
    }


def _apply_rotary_emb(x: ttnn.Tensor, freqs_cis: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
    cos, sin = freqs_cis
    cos = cos.reshape([1, 1, *cos.shape])
    sin = sin.reshape([1, 1, *sin.shape])

    return x * cos + ttnn.alt_complex_rotate90(x) * sin
