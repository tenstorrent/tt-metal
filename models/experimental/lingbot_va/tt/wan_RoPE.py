import ttnn
import torch
from models.tt_dit.layers.module import Module
from models.tt_transformers.tt.common import get_rot_transformation_mat


class WanRotaryPosEmbed(Module):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Calculate dimensions for f, h, w
        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute frequency bases and store as device tensors
        self.f_freqs_base, self.h_freqs_base, self.w_freqs_base = self._precompute_freqs_base()

        # Precompute transformation matrix
        self.transformation_mat = ttnn.from_torch(
            get_rot_transformation_mat(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    def _precompute_freqs_base(self):
        # Compute frequency bases using torch, then convert to TTNN tensors
        f_freqs_base = 1.0 / (self.theta ** (torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta ** (torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta ** (torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double() / self.w_dim))

        # Convert to TTNN tensors and store on device
        f_freqs_base_tt = ttnn.from_torch(
            f_freqs_base.float(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        h_freqs_base_tt = ttnn.from_torch(
            h_freqs_base.float(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        w_freqs_base_tt = ttnn.from_torch(
            w_freqs_base.float(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

        return f_freqs_base_tt, h_freqs_base_tt, w_freqs_base_tt

    def forward(self, grid_ids: ttnn.Tensor) -> ttnn.Tensor:
        # grid_ids should be a TTNN tensor of shape [batch, 3, seq_len]

        # Extract f, h, w grid IDs and reshape for broadcasting
        f_grid_ids = grid_ids[:, 0, :]  # [batch, seq_len]
        h_grid_ids = grid_ids[:, 1, :]  # [batch, seq_len]
        w_grid_ids = grid_ids[:, 2, :]  # [batch, seq_len]

        # Get shapes for reshaping
        batch_size = grid_ids.shape[0]
        seq_len = grid_ids.shape[2]

        # Reshape grid_ids to [batch, seq_len, 1] for broadcasting
        f_grid_ids = ttnn.reshape(f_grid_ids, (batch_size, seq_len, 1))
        h_grid_ids = ttnn.reshape(h_grid_ids, (batch_size, seq_len, 1))
        w_grid_ids = ttnn.reshape(w_grid_ids, (batch_size, seq_len, 1))

        # Reshape freqs_base to [1, 1, dim//2] for broadcasting
        f_freqs_base_shape = list(self.f_freqs_base.shape)
        f_freqs_base_reshaped = ttnn.reshape(self.f_freqs_base, (1, 1, f_freqs_base_shape[-1]))

        h_freqs_base_shape = list(self.h_freqs_base.shape)
        h_freqs_base_reshaped = ttnn.reshape(self.h_freqs_base, (1, 1, h_freqs_base_shape[-1]))

        w_freqs_base_shape = list(self.w_freqs_base.shape)
        w_freqs_base_reshaped = ttnn.reshape(self.w_freqs_base, (1, 1, w_freqs_base_shape[-1]))

        # Multiply: [batch, seq_len, 1] * [1, 1, dim//2] -> [batch, seq_len, dim//2]
        f_freqs = ttnn.multiply(f_grid_ids, f_freqs_base_reshaped)
        h_freqs = ttnn.multiply(h_grid_ids, h_freqs_base_reshaped)
        w_freqs = ttnn.multiply(w_grid_ids, w_freqs_base_reshaped)

        # Concatenate frequencies along the last dimension
        freqs = ttnn.concat([f_freqs, h_freqs, w_freqs], dim=-1)  # [batch, seq_len, attention_head_dim//2]

        # Create complex frequencies (cos + i*sin)
        cos_freqs = ttnn.cos(freqs)
        sin_freqs = ttnn.sin(freqs)

        # Stack cos and sin to match the expected format for rotary embedding
        freqs_cis = ttnn.concat([cos_freqs, sin_freqs], dim=-1)  # [batch, seq_len, attention_head_dim]

        return freqs_cis
