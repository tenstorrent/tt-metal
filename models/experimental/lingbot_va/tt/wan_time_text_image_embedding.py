from models.tt_dit.layers.module import Module
from models.tt_dit.layers.embeddings import Timesteps, TimestepEmbedding
from models.tt_dit.layers.linear import RowParallelLinear
import ttnn


class TtWanTimeTextImageEmbedding(Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        mesh_device: ttnn.MeshDevice,
    ):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, dtype=ttnn.float32, mesh_device=mesh_device)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim, dtype=ttnn.bfloat16, mesh_device=mesh_device
        )
        self.act_fn = ttnn.silu
        self.time_proj = RowParallelLinear(
            dim,
            time_proj_dim,
            bias=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
        )
        # TODO: Implement ttnn version of PixArtAlphaTextProjection
        # self.text_embedder = TTTextProjection(text_embed_dim, dim, mesh_device)

    def forward(
        self,
        timestep: ttnn.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep_flat = ttnn.reshape(timestep, (B * L,))
        # Add 3 leading singleton dimensions
        timestep_4d = ttnn.reshape(timestep_flat, (1, 1, 1, B * L))
        # Get batch and sequence length from input shape
        input_shape = list(timestep.shape)

        # Extract dimensions and determine total_size
        if len(input_shape) == 4:
            # For 4D input, find the dimension that contains the sequence length
            # It could be (1, 1, 1, total_size) or (1, 1, total_size, 1) or (B, 1, 1, L) etc.
            # Find the largest non-1 dimension (excluding leading 1s)
            non_one_dims = [dim for dim in input_shape if dim != 1]
            if non_one_dims:
                total_size = max(non_one_dims)
            else:
                total_size = input_shape[-1]  # Fallback to last dimension
            B = 1
            L = total_size
        else:
            # Fallback: assume 2D input (B, L)
            B = input_shape[0] if len(input_shape) > 0 else 1
            L = input_shape[1] if len(input_shape) > 1 else 1
            total_size = B * L

        # ALWAYS reshape to (1, 1, total_size, 1) for proper broadcasting
        # This ensures: (1, 1, 257, 1) * (1, 1, 1, 128) -> (1, 1, 257, 128)
        timestep = ttnn.reshape(timestep, (1, 1, total_size, 1))

        # Ensure dtype is float32 to match Timesteps expectation
        if timestep.dtype != ttnn.float32:
            timestep = ttnn.typecast(timestep, ttnn.float32)

        # Now timestep is (1, 1, total_size, 1) and can broadcast with time_proj_factor (1, 1, 1, 128)
        timestep = self.timesteps_proj(timestep)  # Output: (1, 1, total_size, 256)

        # CRITICAL FIX: Reshape to 2D (total_size, 256) to reduce memory usage
        # This matches the reference implementation which processes flattened timesteps
        timestep_shape = list(timestep.shape)
        timestep = ttnn.reshape(timestep, (total_size, timestep_shape[-1]))  # (B*L, 256)

        # Now process through time_embedder with 2D tensor (much more memory efficient)
        temb = self.time_embedder(timestep)  # Input: (B*L, 256), Output: (B*L, dim)

        if dtype:
            temb = temb.to(dtype)

        timestep_proj = self.time_proj(self.act_fn(temb))  # Input: (B*L, dim), Output: (B*L, time_proj_dim)

        # Get feature dimensions explicitly
        temb_shape = list(temb.shape)
        temb_feat_dim = temb_shape[-1]

        timestep_proj_shape = list(timestep_proj.shape)
        timestep_proj_feat_dim = timestep_proj_shape[-1]

        # Reshape outputs back to (B, L, feat_dim)
        temb = ttnn.reshape(temb, (B, L, temb_feat_dim))
        timestep_proj = ttnn.reshape(timestep_proj, (B, L, timestep_proj_feat_dim))

        return temb, timestep_proj
