import torch
import torch.nn as nn
from einops import rearrange
import ttnn
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from models.utility_functions import nearest_32
from models.common.lightweightmodule import LightweightModule
from models.experimental.mochi.block import TtAsymmetricJointBlock
from models.experimental.mochi.final_layer import TtFinalLayer
from models.experimental.mochi.common import to_tt_tensor, unsqueeze_to_4d, replicate_attn_mask, stack_cos_sin
from models.tt_transformers.tt.common import get_rot_transformation_mat
from genmo.mochi_preview.dit.joint_model.layers import (
    PatchEmbed,
    TimestepEmbedder,
)
from genmo.mochi_preview.dit.joint_model.utils import AttentionPool
from genmo.mochi_preview.dit.joint_model.rope_mixed import (
    compute_mixed_rotation,
    create_position_matrix,
)
from models.experimental.mochi.embed import TtPatchEmbed


class TtAsymmDiTJoint(LightweightModule):
    """DiT model implemented for TensorTorch with asymmetric attention."""

    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path: Path,
        *,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size_x: int = 1152,
        hidden_size_y: int = 1152,
        depth: int = 48,
        num_heads: int = 16,
        mlp_ratio_x: float = 8.0,
        mlp_ratio_y: float = 4.0,
        t5_feat_dim: int = 4096,
        t5_token_length: int = 256,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        timestep_scale: Optional[float] = None,
        use_extended_posenc: bool = False,
        rope_theta: float = 10000.0,
        **block_kwargs,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.state_dict = state_dict
        self.weight_cache_path = weight_cache_path

        # Save configuration
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.head_dim = hidden_size_x // num_heads
        self.use_extended_posenc = use_extended_posenc
        self.t5_token_length = t5_token_length
        self.t5_feat_dim = t5_feat_dim
        self.rope_theta = rope_theta

        # Create PyTorch embedders (these run on CPU) and load their weights
        # self.x_embedder = TtPatchEmbed(
        #     mesh_device=mesh_device,
        #     state_dict=state_dict,
        #     weight_cache_path=weight_cache_path,
        #     dtype=ttnn.bfloat16,
        #     patch_size=patch_size,
        #     in_chans=in_channels,
        #     embed_dim=hidden_size_x,
        #     bias=patch_embed_bias,
        #     state_dict_prefix="x_embedder",
        # )
        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size_x,
            bias=patch_embed_bias,
        )
        x_embedder_state_dict = {
            k[len("x_embedder.") :]: v for k, v in state_dict.items() if k.startswith("x_embedder.")
        }
        self.x_embedder.load_state_dict(x_embedder_state_dict)

        self.t_embedder = TimestepEmbedder(hidden_size_x, bias=timestep_mlp_bias, timestep_scale=timestep_scale)
        t_embedder_state_dict = {
            k[len("t_embedder.") :]: v for k, v in state_dict.items() if k.startswith("t_embedder.")
        }
        self.t_embedder.load_state_dict(t_embedder_state_dict)

        self.t5_y_embedder = AttentionPool(t5_feat_dim, num_heads=8, output_dim=hidden_size_x)
        t5_y_embedder_state_dict = {
            k[len("t5_y_embedder.") :]: v for k, v in state_dict.items() if k.startswith("t5_y_embedder.")
        }
        self.t5_y_embedder.load_state_dict(t5_y_embedder_state_dict)

        self.t5_yproj = nn.Linear(t5_feat_dim, hidden_size_y, bias=True)
        t5_yproj_state_dict = {k[len("t5_yproj.") :]: v for k, v in state_dict.items() if k.startswith("t5_yproj.")}
        self.t5_yproj.load_state_dict(t5_yproj_state_dict)

        # Initialize position frequencies parameter
        self.pos_frequencies = state_dict["pos_frequencies"]

        # Create DiT blocks
        MAX_DEPTH = 48
        self.blocks = []
        for b in range(depth):
            update_y = b < MAX_DEPTH - 1
            block = TtAsymmetricJointBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                state_dict_prefix=f"blocks.{b}",
                weight_cache_path=weight_cache_path,
                layer_num=b,
                dtype=ttnn.bfloat16,
                hidden_size_x=hidden_size_x,
                hidden_size_y=hidden_size_y,
                num_heads=num_heads,
                mlp_ratio_x=mlp_ratio_x,
                mlp_ratio_y=mlp_ratio_y,
                update_y=update_y,
                **block_kwargs,
            )
            self.blocks.append(block)

        # Create final layer
        self.final_layer = TtFinalLayer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="final_layer",
            weight_cache_path=weight_cache_path,
            dtype=ttnn.bfloat16,
            hidden_size=hidden_size_x,
            patch_size=patch_size,
            out_channels=in_channels,
        )

    def prepare_text_features(self, t5_feat, t5_mask):
        """
        Prepare text features for the model.
        """
        # assert t5_feat.size(1) == self.t5_token_length
        t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)
        y_feat_BLY = self.t5_yproj(t5_feat)

        tt_y_feat_1BLY = to_tt_tensor(unsqueeze_to_4d(y_feat_BLY), self.mesh_device)  # Add dim for ttnn format
        tt_y_pool_11BX = to_tt_tensor(unsqueeze_to_4d(t5_y_pool), self.mesh_device)

        return tt_y_feat_1BLY, tt_y_pool_11BX

    def prepare_rope_features(self, T, H, W):
        pH, pW = H // self.patch_size, W // self.patch_size
        N = T * pH * pW
        pos = create_position_matrix(T, pH=pH, pW=pW, device="cpu", dtype=torch.float32)
        rope_cos_NHD, rope_sin_NHD = compute_mixed_rotation(freqs=self.pos_frequencies, pos=pos)
        rope_cos_1HND, rope_sin_1HND = stack_cos_sin(
            rope_cos_NHD.unsqueeze(0).permute(0, 2, 1, 3), rope_sin_NHD.unsqueeze(0).permute(0, 2, 1, 3)
        )

        trans_mat = get_rot_transformation_mat(None)

        tt_rope_cos_1HND = to_tt_tensor(unsqueeze_to_4d(rope_cos_1HND), self.mesh_device, shard_dim=-3)
        tt_rope_sin_1HND = to_tt_tensor(unsqueeze_to_4d(rope_sin_1HND), self.mesh_device, shard_dim=-3)
        tt_trans_mat = to_tt_tensor(trans_mat, self.mesh_device)

        return tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

    def prepare_step_independent(
        self,
        T,
        H,
        W,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ):
        """
        Prepare inputs for the model which don't change across steps.
        - rope cos/sin
        - transformation mat
        - y_feat
        - y_pool

        Returns ttnn tensors
        """
        # Construct position array
        pH, pW = H // self.patch_size, W // self.patch_size
        N = T * pH * pW
        pos = create_position_matrix(T, pH=pH, pW=pW, device="cpu", dtype=torch.float32)
        rope_cos_NHD, rope_sin_NHD = compute_mixed_rotation(freqs=self.pos_frequencies, pos=pos)
        rope_cos_1HND, rope_sin_1HND = stack_cos_sin(
            rope_cos_NHD.unsqueeze(0).permute(0, 2, 1, 3), rope_sin_NHD.unsqueeze(0).permute(0, 2, 1, 3)
        )

        trans_mat = get_rot_transformation_mat(None)

        assert t5_feat.size(1) == self.t5_token_length
        t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)
        y_feat_BLY = self.t5_yproj(t5_feat)

        tt_y_feat_1BLY = to_tt_tensor(unsqueeze_to_4d(y_feat_BLY), self.mesh_device)  # Add dim for ttnn format
        tt_y_pool_11BX = to_tt_tensor(unsqueeze_to_4d(t5_y_pool), self.mesh_device)
        tt_rope_cos_1HND = to_tt_tensor(unsqueeze_to_4d(rope_cos_1HND), self.mesh_device, shard_dim=-3)
        tt_rope_sin_1HND = to_tt_tensor(unsqueeze_to_4d(rope_sin_1HND), self.mesh_device, shard_dim=-3)
        tt_trans_mat = to_tt_tensor(trans_mat, self.mesh_device)

        return tt_y_feat_1BLY, tt_y_pool_11BX, tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

    def prepare_step_dependent(
        self,
        x: torch.Tensor,
        sigma: float,
        tt_y_pool: ttnn.Tensor,
    ):
        """
        Prepare inputs for the model which change across steps.
        Accepts x as torch Tensor. TODO: Process x_embedder on device.
        - x
        - c
        """
        # Visual patch embeddings with positional encoding
        B, C, T, H, W = x.shape

        pH, pW = H // self.patch_size, W // self.patch_size
        # Flatten x for embedding
        x = x.reshape(C, T, pH, self.patch_size, pW, self.patch_size)
        x_1BNI = x.permute(1, 2, 4, 0, 3, 5).reshape(1, B, T * pH * pW, C * self.patch_size * self.patch_size)
        x_1BNI = to_tt_tensor(x_1BNI, self.mesh_device)
        x_1BNX = self.x_embedder(x_1BNI)

        # Global vector embedding for conditionings
        c_t_BX = self.t_embedder(1 - sigma)
        c_t_11BX = to_tt_tensor(unsqueeze_to_4d(c_t_BX), self.mesh_device)  # Add dims for ttnn format
        c_11BX = c_t_11BX + tt_y_pool

        return x_1BNX, c_11BX

    def _prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deprecated. Prepare input tensors (runs in PyTorch on CPU).
        """

        # Visual patch embeddings with positional encoding
        T, H, W = x.shape[-3:]
        pH, pW = H // self.patch_size, W // self.patch_size
        x = self.x_embedder(x)  # (B, N, D)
        B = x.size(0)

        # Construct position array
        N = T * pH * pW
        pos = create_position_matrix(T, pH=pH, pW=pW, device=x.device, dtype=torch.float32)
        rope_cos, rope_sin = compute_mixed_rotation(freqs=self.pos_frequencies, pos=pos)

        # Global vector embedding for conditionings
        c_t = self.t_embedder(1 - sigma)

        # Pool T5 tokens using attention pooler
        # assert t5_feat.size(1) == self.t5_token_length
        t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)
        assert t5_y_pool.size(0) == B

        c = c_t + t5_y_pool
        y_feat = self.t5_yproj(t5_feat)

        return x, c, y_feat, rope_cos, rope_sin

    def preprocess_input(self, x):
        B, C, T, H, W = x.shape
        assert B == 1, "Batch size must be 1"
        pH, pW = H // self.patch_size, W // self.patch_size
        x = x.reshape(B, C, T, pH, self.patch_size, pW, self.patch_size)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).reshape(1, B, T * pH * pW, C * self.patch_size * self.patch_size)
        # x = to_tt_tensor(x, self.mesh_device)
        return x

    def reverse_preprocess(self, x, T, H, W):
        """
        This is the reverse of preprocess_input. It differs from postprocess_output
        in that the input x is of shape: B (T pH pW) (C p p)
        returns shape: B C T (pH p) (pW p)
        """
        assert len(x.shape) == 4
        assert x.shape[0] == 1
        B = x.shape[1]
        pH, pW = H // self.patch_size, W // self.patch_size
        x = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).squeeze(0)
        x = x.reshape(B, T, pH, pW, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).reshape(B, self.out_channels, T, H, W)
        return x

    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given torch inputs, compose step-independent and step-dependent input preparation.
        Returns ttnn tensors on device.
        """
        T, H, W = x.shape[-3:]
        tt_y_feat, tt_y_pool, tt_rope_cos, tt_rope_sin, tt_trans_mat = self.prepare_step_independent(
            T, H, W, t5_feat, t5_mask
        )
        x_1BNX, c_11BX = self.prepare_step_dependent(x, sigma, tt_y_pool)
        return x_1BNX, c_11BX, tt_y_feat, tt_rope_cos, tt_rope_sin, tt_trans_mat

    def forward_inner(
        self,
        x_1BNI: ttnn.Tensor,
        sigma: float,
        y_feat_1BLY: ttnn.Tensor,
        y_pool_11BX: ttnn.Tensor,
        rope_cos_1HND: ttnn.Tensor,
        rope_sin_1HND: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        uncond: bool = False,
    ):
        """
        Optimized forward pass which depends on outer loop to prepare inputs.
        Returns
        """
        x_1BNX = self.x_embedder(x_1BNI)
        x_1BNX = to_tt_tensor(unsqueeze_to_4d(x_1BNX), self.mesh_device)

        # Global vector embedding for conditionings
        c_t_BX = self.t_embedder(1 - sigma)
        c_t_11BX = to_tt_tensor(unsqueeze_to_4d(c_t_BX), self.mesh_device)  # Add dims for ttnn format
        c_11BX = c_t_11BX + y_pool_11BX

        # Run blocks
        for block in self.blocks:
            x_1BNX, y_feat_1BLY = block(
                x_1BNX,
                c_11BX,
                y_feat_1BLY,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
                uncond=uncond,
            )

        # Run final layer
        x_1BNI = self.final_layer(x_1BNX, c_11BX)

        return x_1BNI

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y_feat: List[torch.Tensor],
        y_mask: List[torch.Tensor],
        uncond: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        User-friendly, unoptimized forward.
        """
        _, _, T, H, W = x.shape

        x_1BNX, c_11BX, y_feat_1BLY, rope_cos_1HND, rope_sin_1HND, trans_mat = self.prepare(
            x, sigma, y_feat[0], y_mask[0]
        )
        # TODO: c shape should be broadcastable to x shape, with batch dim matching. It currently doesn't.

        # Run blocks
        for block in self.blocks:
            x_1BNX, y_feat_1BLY = block(
                x_1BNX,
                c_11BX,
                y_feat_1BLY,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
                uncond=uncond,
            )

        # Run final layer
        x_1BND = self.final_layer(x_1BNX, c_11BX)

        # Converts output back to torch and expected shape
        x_BCTHW = self.reverse_preprocess(x_1BND, T, H, W)
        return x_BCTHW
