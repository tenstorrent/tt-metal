# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from .linear import TtLinear, TtLinearParameters
from .normalization import TtRmsNorm, TtRmsNormParameters
from .substate import has_substate, substate
from .utils import all_gather


@dataclass
class TtAttentionPartParameters:
    qkv_proj: TtLinearParameters
    norm_q: TtRmsNormParameters
    norm_k: TtRmsNormParameters
    out_proj: TtLinearParameters | None


@dataclass
class TtAttentionParameters:
    spatial: TtAttentionPartParameters
    prompt: TtAttentionPartParameters | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        num_heads: int,
        unpadded_num_heads: int,
        hidden_dim_padding: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtAttentionParameters:
        spatial_qkv_proj = _merge_qkv_proj(substate(state, "to_q"), substate(state, "to_k"), substate(state, "to_v"))
        prompt_qkv_proj = _merge_qkv_proj(
            substate(state, "add_q_proj"), substate(state, "add_k_proj"), substate(state, "add_v_proj")
        )
        n_local_heads = num_heads // device.get_num_devices()
        return cls(
            spatial=TtAttentionPartParameters(
                qkv_proj=TtLinearParameters.from_torch_col_parallel(
                    state=spatial_qkv_proj,
                    n_local_heads=n_local_heads,
                    unpadded_num_heads=unpadded_num_heads,
                    hidden_dim_padding=hidden_dim_padding,
                    dtype=dtype,
                    device=device,
                ),
                norm_q=TtRmsNormParameters.from_torch(substate(state, "norm_q"), dtype=dtype, device=device),
                norm_k=TtRmsNormParameters.from_torch(substate(state, "norm_k"), dtype=dtype, device=device),
                out_proj=TtLinearParameters.from_torch(
                    substate(state, "to_out.0"), dtype=dtype, device=device, shard_dim=-1
                ),
            ),
            prompt=TtAttentionPartParameters(
                qkv_proj=TtLinearParameters.from_torch_col_parallel(
                    state=prompt_qkv_proj,
                    n_local_heads=n_local_heads,
                    unpadded_num_heads=unpadded_num_heads,
                    hidden_dim_padding=hidden_dim_padding,
                    dtype=dtype,
                    device=device,
                ),
                norm_q=TtRmsNormParameters.from_torch(substate(state, "norm_added_q"), dtype=dtype, device=device),
                norm_k=TtRmsNormParameters.from_torch(substate(state, "norm_added_k"), dtype=dtype, device=device),
                out_proj=TtLinearParameters.from_torch(
                    substate(state, "to_add_out"), dtype=dtype, device=device, shard_dim=-1
                )
                if has_substate(state, "to_add_out")
                else None,
            )
            if has_substate(state, "add_q_proj")
            else None,
        )


class TtAttentionPart:
    def __init__(self, parameters: TtAttentionPartParameters, device: ttnn.MeshDevice) -> None:
        super().__init__()

        eps = 1e-6

        self._device = device
        self._qkv_proj = TtLinear(parameters.qkv_proj)
        self._out_proj = TtLinear(parameters.out_proj) if parameters.out_proj is not None else None
        self._norm_q = TtRmsNorm(parameters.norm_q, eps=eps)
        self._norm_k = TtRmsNorm(parameters.norm_k, eps=eps)

    def qkv(self, x: ttnn.Tensor, *, num_heads: int, deallocate: bool) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        # tracy.signpost("enter TtAttentionPart")

        _batch_size, sequence_length, _embedding_dim = x.shape

        # Input sharding
        if sequence_length > 1024:
            # sharding leads to worse PCC, so disable it until further investigation
            mm_a_x = 8
            mm_a_y = 8
            mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG
        elif sequence_length >= 512:
            mm_a_y = 8
            mm_a_x = 8
            # mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG
            # x = to_memory_config(
            #     x,
            #     memory_config=ttnn.create_sharded_memory_config(
            #         x.shape,
            #         core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            #         strategy=mm_a_x_strategy,
            #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
            #     ),
            #     deallocate=deallocate,
            # )
            deallocate = True
        else:
            mm_a_x = 8
            mm_a_y = 6
            mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG

        qkv = self._qkv_proj(
            x,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            dtype=ttnn.bfloat16,
            deallocate=deallocate,
        )

        #        qkv = ttnn.reallocate(qkv)
        #        qkv = to_memory_config(qkv, ttnn.DRAM_MEMORY_CONFIG, deallocate=True)

        num_local_heads = num_heads // self._device.get_num_devices()
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=num_local_heads, transpose_key=False
        )

        ttnn.deallocate(qkv)

        q = self._norm_q(q, deallocate=True)
        k = self._norm_k(k, deallocate=True)

        # q = to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG, deallocate=True)
        # k = to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG, deallocate=True)
        # v = to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG, deallocate=True)

        # tracy.signpost("exit TtAttentionPart")

        return q, k, v

    def out_proj(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._out_proj is None:
            return x

        # Input sharding
        """
        if x.shape[-2] >= 512:
            mm_a_y = 8
            mm_a_x = 8
            mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
            mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            x = ttnn.to_memory_config(
                x,
                memory_config=ttnn.create_sharded_memory_config(
                    x.shape,
                    core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                    strategy=mm_a_x_strategy,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )
        else:
            mm_a_x = 8
            mm_a_y = 6
            mm_a_x_memory_config = ttnn.L1_MEMORY_CONFIG

        return self._out_proj(
                             x,
                             memory_config=mm_a_x_memory_config,
                             core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
                             dtype=ttnn.bfloat8_b,
                             )
        """

        grid_size = x.device().compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)
        result = self._out_proj(x, core_grid=core_grid)

        # return to_memory_config(result, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, deallocate=True)
        return result


class TtAttention:
    def __init__(self, parameters: TtAttentionParameters, *, num_heads: int, device) -> None:
        super().__init__()

        self._device = device
        self._num_heads = num_heads

        self._spatial_attn = TtAttentionPart(parameters.spatial, device=self._device)
        self._prompt_attn = (
            TtAttentionPart(parameters.prompt, device=self._device) if parameters.prompt is not None else None
        )

    def __call__(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor | None = None,
        deallocate: bool = False,
        N: int,
        L: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        spatial: N ⊗ S1 ⊗ (H * E1)
        prompt: N ⊗ S2 ⊗ (H * E2)
        """
        device = spatial.device()

        spatial = ttnn.squeeze(spatial, 1)
        prompt = ttnn.squeeze(prompt, 1)

        # tracy.signpost("enter TtAttention")

        q, k, v = self._spatial_attn.qkv(spatial, num_heads=self._num_heads, deallocate=deallocate)

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=768,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        if prompt is None:
            # operands must be in DRAM
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            # attn = ttnn.to_memory_config(attn, memory_config=ttnn.L1_MEMORY_CONFIG)

            concatenated_attn = ttnn.transformer.concatenate_heads(
                attn,
                # memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
            )
            ttnn.deallocate(attn)

            spatial = self._spatial_attn.out_proj(concatenated_attn)

            spatial = ttnn.unsqueeze(spatial, 1)
            return spatial, None

        assert self._prompt_attn is not None

        q2, k2, v2 = self._prompt_attn.qkv(prompt, num_heads=self._num_heads, deallocate=deallocate)

        spatial, prompt = ttnn.transformer.joint_scaled_dot_product_attention(
            q,
            k,
            v,
            q2,
            k2,
            v2,
            joint_strategy="rear",
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

        # spatial = ttnn.to_memory_config(spatial, memory_config=ttnn.L1_MEMORY_CONFIG)
        # prompt = ttnn.to_memory_config(prompt, memory_config=ttnn.L1_MEMORY_CONFIG)

        spatial = ttnn.transformer.concatenate_heads(
            spatial,
            # memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        )
        prompt = ttnn.transformer.concatenate_heads(
            prompt,
            # memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        )

        spatial = ttnn.unsqueeze(spatial, 1)
        prompt = ttnn.unsqueeze(prompt, 1)

        if self._device.get_num_devices() > 1:
            spatial = all_gather(spatial, dim=-1)
            prompt = all_gather(prompt, dim=-1)

        spatial = self._spatial_attn.out_proj(spatial)
        prompt = self._prompt_attn.out_proj(prompt)

        # tracy.signpost("exit TtAttention")
        return spatial, prompt


def _merge_qkv_proj(
    q_state: dict[str, torch.Tensor],
    k_state: dict[str, torch.Tensor],
    v_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.cat([q_state["weight"], k_state["weight"], v_state["weight"]]) if "weight" in q_state else None,
        "bias": torch.cat([q_state["bias"], k_state["bias"], v_state["bias"]]) if "bias" in q_state else None,
    }
