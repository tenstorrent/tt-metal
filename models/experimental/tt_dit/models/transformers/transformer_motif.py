# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import ttnn

from ...layers.attention import all_gather
from ...layers.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, PatchEmbed
from ...layers.linear import ColParallelLinear, Linear
from ...layers.module import Module, Parameter
from ...layers.normalization import DistributedLayerNorm
from ...layers.transformer_block import TransformerBlock

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import Any

    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager
    from ...utils.padding import PaddingConfig


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class MotifTransformer(Module):
    ENCODED_TEXT_DIM = 4096
    SD3_LATENT_CHANNEL = 16

    def __init__(
        self,
        *,
        patch_size: int,
        num_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        pooled_projection_dim: int,
        pos_embed_max_size: int,
        modulation_dim: int,
        time_embed_dim: int,
        register_token_num: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        init: bool = False,
    ) -> None:
        super().__init__()

        in_channels = self.SD3_LATENT_CHANNEL
        out_channels = self.SD3_LATENT_CHANNEL
        inner_dim = num_attention_heads * attention_head_dim
        sample_size = 64

        self.patch_size = patch_size
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            sp_mesh_axis=parallel_config.sequence_parallel.mesh_axis,
            init=init,
        )

        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            mesh_device=mesh_device,
            init=init,
        )
        # self.time_text_embed.timestep_embedder.linear_1 = torch.nn.Linear(time_embed_dim, 4 * time_embed_dim)
        # self.time_text_embed.timestep_embedder.linear_2 = torch.nn.Linear(4 * time_embed_dim, modulation_dim)
        # self.time_text_embed.text_embedder.linear_1 = torch.nn.Linear(pooled_projection_dim, 4 * pooled_projection_dim)
        # self.time_text_embed.text_embedder.linear_2 = torch.nn.Linear(4 * pooled_projection_dim, modulation_dim)
        # self.time_text_embed.time_proj = Timesteps(time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)

        self.context_embedder = ColParallelLinear(
            self.ENCODED_TEXT_DIM,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            init=init,
        )

        self.transformer_blocks = [
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=i == num_layers - 1,
                ff_activation_fn="linear-silu",
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=mesh_device,
                init=init,
            )
            for i in range(num_layers)
        ]

        self.time_embed_out = Linear(
            inner_dim,
            2 * inner_dim,
            mesh_device=mesh_device,
            init=init,
        )

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            init=init,
        )
        # self.norm_out = AdaLayerNormContinuous(self.inner_dim, modulation_dim, elementwise_affine=False, eps=1e-6)

        self.proj_out = Linear(
            inner_dim,
            patch_size * patch_size * out_channels,
            mesh_device=mesh_device,
            init=init,
        )

        self.register_tokens = Parameter(
            shape=[1, register_token_num, self.inner_dim],
            device=mesh_device,
            init=init,
        )

    # We do not shard the last dimension of spatial, because its dimension is less than the tile
    # size for a device count of four and more. This requires padding, which is not currently
    # supported by `reduce_scatter_minimal_async`.
    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        # combined_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        """Run the model forward.

        Args:
            spatial: Tensor with shape [batch_size, spatial_sequence_length / sp_factor, in_channels].
            prompt: Tensor with shape [batch_size, prompt_sequence_length, joint_attention_dim].
            pooled: Tensor with shape [batch_size, pooled_projection_dim].
            timestep: Tensor with shape [batch_size, 1].
            guidance: Optional tensor with shape [batch_size, 1].
            spatial_rope: Tuple of two tensors with shape [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: Tuple of two tensors with shape [prompt_sequence_length, head_dim] (sequence is not sharded!).
        """
        time_embed = self.time_text_embed(timestep=timestep, guidance=guidance, pooled_projection=pooled)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        for i, block in enumerate(self.transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation=True,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(spatial.device())

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        # combined = ttnn.concat([prompt, spatial], dim=1)
        # del prompt, spatial

        for i, block in enumerate(self.single_transformer_blocks, start=1):
            spatial, prompt = block.forward(
                # combined=combined,
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                # rope=combined_rope,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                # sequence_length=spatial_sequence_length + prompt_sequence_length,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation=True,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(self.mesh_device)

        # spatial = combined[:, prompt_sequence_length:]
        # del combined

        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = _chunk_time3d(spatial_time, 2)

        spatial = all_gather(
            spatial, dim=2, parallel_factor=self.parallel_config.tensor_parallel, ccl_manager=self.ccl_manager
        )

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)

    def _prepare_torch_state(self, state: MutableMapping[str, Any]) -> None:
        def convert_ada_norm(prefix: str, *, pre_only: bool) -> None:
            ws = state[f"{prefix}.weight"].chunk(6)
            bs = state[f"{prefix}.bias"].chunk(6)

            if pre_only:
                state[f"{prefix}.weight"] = torch.concat(ws[:2])
                state[f"{prefix}.bias"] = torch.concat(bs[:2])
            else:
                state[f"{prefix}.weight"] = torch.concat([ws[1], ws[0], ws[2], ws[4], ws[3], ws[5]])
                state[f"{prefix}.bias"] = torch.concat([bs[1], bs[0], bs[2] + 1, bs[4], bs[3] - 1, bs[5]])

        pos_embed = state.pop("pos_embed", {})
        pos_embed = pos_embed.unsqueeze(0) if pos_embed is not None else None

        state["pos_embed"] = {
            "pos_embed": pos_embed,
            "proj": state.get("patching", {}).pop("projection_SD3", {}),
        }
        state["time_text_embed"] = {
            "timestep_embedder": state.get("time_emb", {}).pop("time_emb", {}),
            "text_embedder": state.get("time_emb", {}).pop("pooled_text_emb", {}),
        }
        state["context_embedder"] = state.get("text_cond", {}).pop("projection", {})
        state["proj_out"] = state.pop("final_linear_SD3", {})
        state["norm_out"] = {"linear": state.pop("final_modulation", {})}

        state["transformer_blocks"] = blocks = state.pop("mmdit_blocks", {})

        block_count = len(blocks)

        for i_str, block in blocks:
            i = int(i_str)

            attn = block.get("attn")
            if attn is not None:
                to_out_weight = attn.get("o_proj", {}).pop("weight")
                to_add_out_weight = attn.get("add_o_proj", {}).pop("weight")

                if to_out_weight is not None:
                    attn.setdefault("to_out", {})["0"] = {
                        "weight": to_out_weight,
                        "bias": torch.zeros_like(to_out_weight[0]),
                    }
                if to_add_out_weight is not None and i != block_count - 1:  # unused in the last block
                    attn["to_add_out"] = {
                        "weight": to_add_out_weight,
                        "bias": torch.zeros_like(to_add_out_weight[0]),
                    }

                attn["norm_q"] = attn.pop("q_norm_x", {})
                attn["norm_k"] = attn.pop("k_norm_x", {})
                attn["norm_added_q"] = attn.pop("q_norm_c", {})
                attn["norm_added_k"] = attn.pop("k_norm_c", {})

            block["norm1_context"] = {"linear": state.get("affine_params_c", {}).pop("projection", {})}
            block["norm1"] = {"linear": state.get("affine_params_x", {}).pop("projection", {})}

            convert_ada_norm(block["norm1.linear"], pre_only=False)
            convert_ada_norm(block["norm1_context"]["linear"], pre_only=i == block_count - 1)

            ff_gate = state.get("mlp_3_x", {}).pop("gate_proj", {})
            ff_down = state.get("mlp_3_x", {}).pop("down_proj", {})
            ff_context_gate = state.get("mlp_3_c", {}).pop("gate_proj", {})
            ff_context_down = state.get("mlp_3_c", {}).pop("down_proj", {})

            block["ff"] = {"net": {"0": {"proj": ff_gate}, "2": ff_down}}
            if i != block_count - 1:  # unused in the last block
                block["ff_context"] = {"net": {"0": {"proj": ff_context_gate}, "2": ff_context_down}}

            x_weight = block.get("linear_1_x", {}).pop("weight")
            x_bias = block.get("linear_1_x", {}).pop("bias")

            if x_weight is not None and x_bias is not None:
                q_weight = attn.get("q_proj", {}).pop("weight")
                k_weight = attn.get("k_proj", {}).pop("weight")
                v_weight = attn.get("v_proj", {}).pop("weight")

                if q_weight is not None:
                    attn["to_q"] = {"weight": q_weight @ x_weight, "bias": q_weight @ x_bias}
                if k_weight is not None:
                    attn["to_k"] = {"weight": k_weight @ x_weight, "bias": k_weight @ x_bias}
                if v_weight is not None:
                    attn["to_v"] = {"weight": v_weight @ x_weight, "bias": v_weight @ x_bias}

            c_weight = block.get("linear_1_c", {}).pop("weight")
            c_bias = block.get("linear_1_c", {}).pop("bias")

            if c_weight is not None and c_bias is not None:
                add_q_weight = attn.get("add_q_proj", {}).pop("weight")
                add_k_weight = attn.get("add_k_proj", {}).pop("weight")
                add_v_weight = attn.get("add_v_proj", {}).pop("weight")

                if add_q_weight is not None:
                    attn["add_q_proj"] = {"weight": add_q_weight @ c_weight, "bias": add_q_weight @ c_bias}
                if add_k_weight is not None:
                    attn["add_k_proj"] = {"weight": add_k_weight @ c_weight, "bias": add_k_weight @ c_bias}
                if add_v_weight is not None:
                    attn["add_v_proj"] = {"weight": add_v_weight @ c_weight, "bias": add_v_weight @ c_bias}

        # diffusers to TT-NN
        state["time_embed_out"] = state.get("norm_out", {}).pop("linear", {})  # chunks=2 if sharded
        state["norm_out"] = state.get("norm_out", {}).pop("norm", {})


def _chunk_time3d(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
