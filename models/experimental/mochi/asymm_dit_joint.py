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
        )

    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare input tensors (runs in PyTorch on CPU)."""

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

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y_feat: List[torch.Tensor],
        y_mask: List[torch.Tensor],
        uncond: bool = False,
    ) -> torch.Tensor:
        """Forward pass of DiT."""
        B, _, T, H, W = x.shape

        # Run prepare() in PyTorch on CPU
        x_BNX, c_BX, y_feat_BLY, rope_cos_NHD, rope_sin_NHD = self.prepare(x, sigma, y_feat[0], y_mask[0])
        x_N = x_BNX.shape[1]

        rope_cos_1HND, rope_sin_1HND = stack_cos_sin(
            rope_cos_NHD.unsqueeze(0).permute(0, 2, 1, 3), rope_sin_NHD.unsqueeze(0).permute(0, 2, 1, 3)
        )

        trans_mat = get_rot_transformation_mat(None)

        # TODO: c shape should be broadcastable to x shape, with batch dim matching. It currently doesn't.
        # Convert tensors to ttnn format using common function
        tt_x_1BNX = to_tt_tensor(unsqueeze_to_4d(x_BNX), self.mesh_device)
        tt_c_11BX = to_tt_tensor(unsqueeze_to_4d(c_BX), self.mesh_device)  # Add dims for ttnn format
        tt_y_feat_1BLY = to_tt_tensor(unsqueeze_to_4d(y_feat_BLY), self.mesh_device)  # Add dim for ttnn format
        tt_rope_cos_1HND = to_tt_tensor(unsqueeze_to_4d(rope_cos_1HND), self.mesh_device, shard_dim=-3)
        tt_rope_sin_1HND = to_tt_tensor(unsqueeze_to_4d(rope_sin_1HND), self.mesh_device, shard_dim=-3)
        tt_trans_mat = to_tt_tensor(trans_mat, self.mesh_device)
        # Run blocks
        for block in self.blocks:
            tt_x_1BNX, tt_y_feat_1BLY = block(
                tt_x_1BNX,
                tt_c_11BX,
                tt_y_feat_1BLY,
                rope_cos=tt_rope_cos_1HND,
                rope_sin=tt_rope_sin_1HND,
                trans_mat=tt_trans_mat,
                uncond=uncond,
            )

        # Run final layer
        tt_x_1BND = self.final_layer(tt_x_1BNX, tt_c_11BX)

        # Convert output back to PyTorch
        # output is already replicated.
        # TODO: Don't do this concat here, just get a ref to single device tensor
        x_1BND = ttnn.to_torch(tt_x_1BND, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0:1]
        x_D = self.patch_size**2 * self.out_channels
        x_BND = x_1BND[..., :x_N, :].reshape(B, x_N, x_D)

        # Rearrange output in PyTorch on CPU
        x_BCTHW = rearrange(
            x_BND,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=T,
            hp=H // self.patch_size,
            wp=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )

        return x_BCTHW
