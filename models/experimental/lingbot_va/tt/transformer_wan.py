# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN WanTransformer3DModel for Lingbot-VA.

Reuses WanPatchEmbed, WanTimeTextImageEmbedding, and utilities from models.tt_dit.
Defines WanTransformerBlock and WanAttention locally (no wan2_2 dependency).
Adds Lingbot-VA-specific: in_channels=48, action_embedder, condition_embedder_action,
action_proj_out, and dual forward paths (video + action).
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

import ttnn

from models.tt_dit.layers.embeddings import WanPatchEmbed, WanTimeTextImageEmbedding
from models.tt_dit.layers.feedforward import ParallelFeedForward
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import DistributedLayerNorm
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.padding import pad_vision_seq_parallel
from models.tt_dit.utils.substate import pop_substate, rename_substate
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor, from_torch, unflatten

from .attention_wan import WanAttention
from .wan_RoPE import WanRotaryPosEmbed


# Lingbot-VA config (from reference model.py)
DIM = 3072  # num_attention_heads * attention_head_dim
FFN_DIM = 14336
NUM_HEADS = 24
HEAD_DIM = 128
IN_CHANNELS = 48
OUT_CHANNELS = 48
ACTION_DIM = 30
TEXT_DIM = 4096
FREQ_DIM = 256
NUM_LAYERS = 30
PATCH_SIZE = (1, 2, 2)
CROSS_ATTN_NORM = True
EPS = 1e-6
ROPE_MAX_SEQ_LEN = 1024


class WanTransformerBlock(Module):
    """Transformer block for Lingbot-VA (same structure as tt_dit wan2_2, no wan2_2 dependency)."""

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attention_norm: bool = True,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.cross_attention_norm = cross_attention_norm
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn1 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=True,
        )

        self.attn2 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=False,
        )

        self.norm2 = (
            DistributedLayerNorm(
                dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
            if cross_attention_norm
            else None
        )

        self.ffn = ParallelFeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu",
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self.norm3 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 6, dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "ffn.net.0.proj", "ffn.ff1")
        rename_substate(state, "ffn.net.2", "ffn.ff2")

        if "scale_shift_table" in state:
            state["scale_shift_table"] = state["scale_shift_table"].unsqueeze(0)

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        temb_1BTD: ttnn.Tensor,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        temb_1BTD: replicated on SP, fractured D on TP
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        assert temb_1BTD.shape[2] == 6, "wan expects 6 chunks in timestep embedding"

        shifted_temb_1BTD = self.scale_shift_table.data + temb_1BTD
        shift_msa_1B1D, scale_msa_1B1D, gate_msa_1B1D, c_shift_msa_1B1D, c_scale_msa_1B1D, c_gate_msa_1B1D = ttnn.chunk(
            shifted_temb_1BTD, 6, dim=2
        )

        # NOTE: workaround - addcmul (fused and unfused) is less accurate with fp32 gate input
        gate_msa_1B1D = ttnn.typecast(gate_msa_1B1D, dtype=ttnn.bfloat16)
        c_gate_msa_1B1D = ttnn.typecast(c_gate_msa_1B1D, dtype=ttnn.bfloat16)

        spatial_normed_1BND = self.norm1(
            spatial_1BND, dynamic_weight=(1.0 + scale_msa_1B1D), dynamic_bias=shift_msa_1B1D
        )

        # Self attention on spatial with fused residual addcmul
        spatial_1BND = self.attn1(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=spatial_1BND,
            addcmul_gate=gate_msa_1B1D,
        )

        # Cross attention on prompt
        spatial_normed_1BND = self.norm2(spatial_1BND)

        attn_output_1BND = self.attn2(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            prompt_1BLP=prompt_1BLP,
        )
        spatial_1BND = spatial_1BND + attn_output_1BND

        # Feed Forward
        spatial_normed_1BND = self.norm3(
            spatial_1BND, dynamic_weight=(1.0 + c_scale_msa_1B1D), dynamic_bias=c_shift_msa_1B1D
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_ff_1BND = self.ffn(spatial_normed_1BND, compute_kernel_config=self.ff_compute_kernel_config)

        spatial_1BND = ttnn.addcmul(spatial_1BND, spatial_ff_1BND, c_gate_msa_1B1D)

        return spatial_1BND


class WanTransformer3DModel(Module):
    """
    TTNN WanTransformer3DModel for Lingbot-VA.

    Supports:
    - Video path: spatial (B, in_channels, F, H, W) -> patch_embed -> blocks -> proj_out -> (B, out_channels, F, H, W)
    - Action path: spatial (B, action_dim, F, H, W) -> action_embedder -> blocks -> action_proj_out -> (B, N, action_dim)
    """

    def __init__(
        self,
        *,
        patch_size: tuple[int, ...] = PATCH_SIZE,
        num_heads: int = NUM_HEADS,
        dim: int = DIM,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        action_dim: int = ACTION_DIM,
        text_dim: int = TEXT_DIM,
        freq_dim: int = FREQ_DIM,
        ffn_dim: int = FFN_DIM,
        num_layers: int = NUM_LAYERS,
        cross_attn_norm: bool = CROSS_ATTN_NORM,
        eps: float = EPS,
        rope_max_seq_len: int = ROPE_MAX_SEQ_LEN,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = True,
    ) -> None:
        super().__init__()

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_fsdp = is_fsdp
        self.fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None
        self.cached_rope_features = {}

        self.patch_size = patch_size
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.action_dim = action_dim

        head_dim = dim // num_heads
        self.rope = WanRotaryPosEmbed(
            mesh_device=self.mesh_device,
            attention_head_dim=head_dim,
            patch_size=patch_size,
            max_seq_len=rope_max_seq_len,
        )

        self.patch_embedding = WanPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.action_embedder = Linear(
            action_dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
        )

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            time_freq_dim=freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=text_dim,
            mesh_device=self.mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.condition_embedder_action = WanTimeTextImageEmbedding(
            dim=dim,
            time_freq_dim=freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=text_dim,
            mesh_device=self.mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.blocks = ModuleList(
            WanTransformerBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                cross_attention_norm=cross_attn_norm,
                eps=eps,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
            )
            for _ in range(num_layers)
        )

        self.norm_out = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(
            dim,
            patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
            bias=True,
            mesh_device=mesh_device,
        )

        self.action_proj_out = Linear(
            dim,
            action_dim,
            bias=True,
            mesh_device=mesh_device,
        )

        self.scale_shift_table = Parameter(
            total_shape=[1, 2, dim],
            device=mesh_device,
            mesh_axes=[None, None, parallel_config.tensor_parallel.mesh_axis],
            dtype=ttnn.float32,
        )

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().save(directory, prefix=prefix)
        directory = Path(directory)
        torch.save(self.rope.state_dict(), directory / f"{prefix}rope.pt")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().load(directory, prefix=prefix)
        directory = Path(directory)
        self.rope.load_state_dict(torch.load(directory / f"{prefix}rope.pt"))

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        self.rope.load_state_dict(pop_substate(state, "rope"))

        # Map reference patch_embedding_mlp (Linear) -> patch_embedding (WanPatchEmbed expects conv-form weight)
        rename_substate(state, "patch_embedding_mlp", "patch_embedding")
        patch_weight = state.get("patch_embedding.weight")
        if patch_weight is not None:
            # Reference: (inner_dim, in_channels * p0*p1*p2); WanPatchEmbed expects (embed_dim, in_c, kt, kh, kw)
            embed_dim, flat = patch_weight.shape
            p0, p1, p2 = self.patch_size
            assert flat == self.in_channels * p0 * p1 * p2
            state["patch_embedding.weight"] = patch_weight.reshape(embed_dim, self.in_channels, p0, p1, p2)
        patch_bias = state.get("patch_embedding.bias")
        if patch_bias is not None:
            state["patch_embedding.bias"] = patch_bias.reshape(1, -1)

    def get_rope_features(self, hidden_states: torch.Tensor):
        """Build RoPE cos/sin and transformation matrix from spatial tensor shape."""
        if tuple(hidden_states.shape) not in self.cached_rope_features:
            rope_features = self.prepare_rope_features(hidden_states)
            self.cached_rope_features[tuple(hidden_states.shape)] = rope_features
        return self.cached_rope_features[tuple(hidden_states.shape)]

    def prepare_rope_features(self, hidden_states: torch.Tensor):
        """Given video/action input (B, C, F, H, W), compute RoPE features. Returns tensors on device."""
        logger.info("Preparing rope features for shape %s", hidden_states.shape)
        rope_cos, rope_sin = self.rope(hidden_states)
        rope_cos_1HND = rope_cos.permute(0, 2, 1, 3)
        rope_sin_1HND = rope_sin.permute(0, 2, 1, 3)
        rope_cos_1HND = pad_vision_seq_parallel(
            rope_cos_1HND, num_devices=self.parallel_config.sequence_parallel.factor
        )
        rope_sin_1HND = pad_vision_seq_parallel(
            rope_sin_1HND, num_devices=self.parallel_config.sequence_parallel.factor
        )
        trans_mat = get_rot_transformation_mat()
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tt_rope_cos_1HND = from_torch(
            rope_cos_1HND,
            device=self.mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., sp_axis, None],
        )
        tt_rope_sin_1HND = from_torch(
            rope_sin_1HND,
            device=self.mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., sp_axis, None],
        )
        tt_trans_mat = bf16_tensor(trans_mat, device=self.mesh_device)
        return tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

    def prepare_text_conditioning(
        self,
        encoder_hidden_states: torch.Tensor | ttnn.Tensor,
        action_mode: bool = False,
    ):
        embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        if isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = from_torch(
                encoder_hidden_states,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
            )
        return embedder.forward_text(encoder_hidden_states)

    def prepare_timestep_conditioning(self, timestep: torch.Tensor, action_mode: bool = False):
        assert timestep.ndim == 1
        timestep = float32_tensor(timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=self.mesh_device)
        embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        tt_temb_11BD, tt_timestep_proj_1BTD = embedder.forward_timestep(timestep, timestep_seq_len=None)
        tt_timestep_proj_1BTD = unflatten(ttnn.squeeze(tt_timestep_proj_1BTD, -2), -1, (6, -1))
        return tt_temb_11BD, tt_timestep_proj_1BTD

    def prepare_conditioning(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        action_mode: bool = False,
    ):
        assert timestep.ndim == 1
        tt_temb_11BD, tt_timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep, action_mode=action_mode)
        tt_prompt_1BLP = self.prepare_text_conditioning(encoder_hidden_states, action_mode=action_mode)
        return tt_temb_11BD, tt_timestep_proj_1BTD, tt_prompt_1BLP

    def preprocess_spatial_input_host(self, spatial: torch.Tensor):
        """Patchify video (B, C, F, H, W) -> (1, B, N, C*p0*p1*p2) and pad for SP."""
        B, C, F, H, W = spatial.shape
        logger.info("Preprocessing spatial input with shape %s", spatial.shape)
        assert B == 1, "Batch size must be 1"
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W
        spatial = spatial.reshape(B, C, patch_F, pF, patch_H, pH, patch_W, pW)
        spatial = spatial.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, B, N, pF * pH * pW * C)
        spatial = pad_vision_seq_parallel(spatial, num_devices=self.parallel_config.sequence_parallel.factor)
        return spatial, N

    def preprocess_spatial_input(self, spatial: torch.Tensor):
        spatial, N = self.preprocess_spatial_input_host(spatial)
        spatial = bf16_tensor(
            spatial,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )
        return spatial, N

    def preprocess_action_input(self, spatial: torch.Tensor):
        """Rearrange (B, action_dim, F, H, W) -> (1, B, N, action_dim) on device."""
        B, C, F, H, W = spatial.shape
        assert C == self.action_dim
        N = F * H * W
        # (B, C, F, H, W) -> (B, F*H*W, C)
        spatial = spatial.permute(0, 2, 3, 4, 1).reshape(1, B, N, self.action_dim)
        spatial = pad_vision_seq_parallel(spatial, num_devices=self.parallel_config.sequence_parallel.factor)
        spatial = bf16_tensor(
            spatial,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )
        return spatial, N

    def postprocess_spatial_output_host(self, spatial_1BND: torch.Tensor, F: int, H: int, W: int, N: int):
        """Reverse of preprocess_spatial_input_host. (1, B, N, out) -> (B, out_channels, F, H, W)."""
        assert len(spatial_1BND.shape) == 4 and spatial_1BND.shape[0] == 1
        B = spatial_1BND.shape[1]
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        spatial_BND = spatial_1BND.squeeze(0)
        spatial_BND = spatial_BND[:, :N]
        spatial_patches = spatial_BND.reshape(B, patch_F, patch_H, patch_W, pF, pH, pW, self.out_channels)
        spatial_BCFHW = spatial_patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_channels, F, H, W)
        return spatial_BCFHW

    def postprocess_spatial_output(self, spatial_1BND: ttnn.Tensor, F: int, H: int, W: int, N: int) -> torch.Tensor:
        spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )
        spatial_1BND = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0])
        return self.postprocess_spatial_output_host(spatial_1BND, F, H, W, N)

    def postprocess_action_output(self, spatial_1BND: ttnn.Tensor, N: int) -> torch.Tensor:
        """Gather SP and return (B, N, action_dim)."""
        spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )
        out = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0])
        out = out[0, :, :N, :]  # (1, B, N, action_dim) -> (B, N, action_dim)
        return out

    def forward(
        self,
        spatial: torch.Tensor,
        prompt: torch.Tensor,
        timestep: torch.Tensor,
        grid_id: torch.Tensor,
        action_mode: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            spatial: Video path (B, in_channels, F, H, W) or action path (B, action_dim, F, H, W).
            prompt: Encoder hidden states (text).
            timestep: 1D timestep tensor.
            action_mode: If True, use action embedder and action_proj_out.

        Returns:
            Video path: (B, out_channels, F, H, W). Action path: (B, N, action_dim).
        """
        B, C, F, H, W = spatial.shape
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        rope_cos_1HND, rope_sin_1HND, trans_mat = self.get_rope_features(spatial)
        temb_11BD, timestep_proj_1BTD, prompt_1BLP = self.prepare_conditioning(
            timestep, prompt, action_mode=action_mode
        )

        if action_mode:
            spatial_1BNI, N = self.preprocess_action_input(spatial)
            spatial_1BND = self.action_embedder(
                spatial_1BNI,
                compute_kernel_config=self.hifi4_compute_kernel_config,
                dtype=ttnn.bfloat16,
            )
        else:
            spatial_1BNI, N = self.preprocess_spatial_input(spatial)
            spatial_1BND = self.patch_embedding(spatial_1BNI)

        for block in self.blocks:
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=timestep_proj_1BTD,
                N=N,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
            )

        scale_shift_1BSD = self.scale_shift_table.data + temb_11BD
        shift_11BD, scale_11BD = ttnn.chunk(scale_shift_1BSD, 2, -2)
        spatial_norm_1BND = self.norm_out(
            spatial_1BND,
            dynamic_weight=(1 + scale_11BD),
            dynamic_bias=shift_11BD,
            dtype=ttnn.float32,
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm_1BND,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        if action_mode:
            proj_out_1BNA = self.action_proj_out(
                spatial_norm_1BND,
                compute_kernel_config=self.hifi4_compute_kernel_config,
                dtype=ttnn.float32,
            )
            return self.postprocess_action_output(proj_out_1BNA, N)
        else:
            proj_out_1BNI = self.proj_out(
                spatial_norm_1BND,
                compute_kernel_config=self.hifi4_compute_kernel_config,
                dtype=ttnn.float32,
            )
            return self.postprocess_spatial_output(proj_out_1BNI, F, H, W, N)
