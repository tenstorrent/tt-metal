import pytest
import torch
from loguru import logger
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
import ttnn
from ....utils.tensor import bf16_tensor_2dshard, bf16_tensor
from ....utils.check import assert_quality
from ....utils.mochi import get_rot_transformation_mat


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links",
    [
        [(1, 1), 0, 1, 1],
        [(1, 2), 0, 1, 1],
        [(1, 2), 1, 0, 1],
        [(2, 1), 0, 1, 1],
        [(2, 1), 1, 0, 1],
        [(2, 2), 0, 1, 1],
        [(2, 2), 1, 0, 1],
        [(2, 4), 0, 1, 1],
        [(2, 4), 1, 0, 1],
        [(4, 8), 0, 1, 4],
        [(4, 8), 1, 0, 4],
    ],
    ids=[
        "1x1sp0tp1",
        "1x2sp0tp1",
        "1x2sp1tp0",
        "2x1sp0tp1",
        "2x1sp1tp0",
        "2x2sp0tp1",
        "2x2sp1tp0",
        "2x4sp0tp1",
        "2x4sp1tp0",
        "4x8sp0tp1",
        "4x8sp1tp0",
    ],
    indirect=["mesh_device"],
)
def test_wan_rotary_pos_embed(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int, num_links: int):
    """
    Test WanRotaryPosEmbed from diffusers with default arguments and reasonable input shape.
    """

    # Instantiate WanRotaryPosEmbed with default args
    # Default arguments based on diffusers implementation

    patch_size = (1, 2, 2)
    attention_head_dim = 128
    rope_max_seq_len = 1024
    dim = 5120
    num_heads = dim // attention_head_dim

    rope = WanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=rope_max_seq_len,
    )

    logger.info(f"Created WanRotaryPosEmbed: {rope}")

    # Create a reasonable input shape

    B, C, F, H, W = 1, 16, 21, 90, 160
    p_t, p_h, p_w = patch_size
    patch_F, patch_H, patch_W = F // p_t, H // p_h, W // p_w

    hidden_states = torch.randn(B, C, F, H, W, dtype=torch.float32)

    rotary_emb = rope(hidden_states)
    logger.info(f"cos_shape: {rotary_emb[0].shape}")
    logger.info(f"sin_shape: {rotary_emb[1].shape}")

    # Create input tensor
    patchify_seq_len = patch_F * patch_H * patch_W
    input_tensor = torch.randn(B, patchify_seq_len, num_heads, attention_head_dim, dtype=torch.float32)
    logger.info(f"patchified input shape: {input_tensor.shape}")

    def apply_rotary_emb(
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(hidden_states)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(hidden_states)

    # Call RoPE on the input
    # WanRotaryPosEmbed typically expects (batch, seq_len, hidden_dim) and returns cos/sin embeddings
    output_tensor = apply_rotary_emb(input_tensor, rotary_emb[0], rotary_emb[1])

    # Create TT inputs
    cos = rotary_emb[0].permute(0, 2, 1, 3)
    sin = rotary_emb[1].permute(0, 2, 1, 3)
    input_tensor = input_tensor.permute(0, 2, 1, 3)
    tt_cos = bf16_tensor(cos, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_sin = bf16_tensor(sin, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    tt_input = bf16_tensor_2dshard(input_tensor, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_transformation_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # Create TT outputs
    tt_output = ttnn.experimental.rotary_embedding_llama(
        tt_input, tt_cos, tt_sin, tt_transformation_mat  # , compute_kernel_config=self.rope_compute_kernel_config
    )

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    tt_output = tt_output.permute(0, 2, 1, 3)

    # Check that the TT and torch outputs are close
    assert_quality(output_tensor, tt_output, pcc=0.99)
