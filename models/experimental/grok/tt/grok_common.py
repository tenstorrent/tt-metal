# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh
from models.utility_functions import nearest_32


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions, grok-style.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Grok-1 uses 10000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = torch.cos(emb), torch.sin(emb)
    return cos, sin


def freq_row_to_rotation_matrix(cos_row, sin_row):
    """
    Transform cos/sin frequency rows to a dim x dim rotation matrix
    that implements cos + rotate_half * sin
    """

    d = len(sin_row)
    m_cos = torch.diag(cos_row)
    m_sin = torch.diag(sin_row)
    d = len(sin_row)
    m_rot_sin = torch.cat([m_sin[d // 2 :], -m_sin[: d // 2]])
    return m_cos + m_rot_sin


def get_rotation_mat(dhead, end):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = [freq_row_to_rotation_matrix(c, s) for c, s in zip(cos, sin)]
    return rot_mat


def prepare_inputs_ttnn(x_bsh, hidden_size, current_pos, mesh_device):
    """
    Prepare inputs for decode mode.
    x: (batch, seq, hidden_dim)
    B: batch (32)
    S: sequence len (1)
    H: dim (4096)
    """
    assert x_bsh.size(2) == hidden_size
    assert len(x_bsh.size()) == 3

    batch = x_bsh.size(0)
    seq_len = x_bsh.size(1)
    assert seq_len == 1, "Only supporting decode mode"

    x_1SBH = x_bsh.view(1, seq_len, batch, hidden_size)

    # input goes to L1
    xs_1SBH = ttnn.from_torch(
        x_1SBH,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    # Attention mask
    padded_layer_past_len = nearest_32(current_pos + 1)
    attn_mask = torch.zeros(seq_len, 32, 32, padded_layer_past_len)  # [SB4P]

    # Fill mask with -inf outside the processed tokens
    attn_mask[:, :, :, current_pos + 1 :] = torch.finfo(attn_mask.dtype).min

    attn_mask = ttnn.from_torch(
        attn_mask,
        device=mesh_device,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    ATTN_MASK_MEMCFG = ttnn.create_sharded_memory_config(
        shape=(32, padded_layer_past_len),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    attn_mask = ttnn.interleaved_to_sharded(attn_mask, ATTN_MASK_MEMCFG)

    return xs_1SBH, attn_mask


def prepare_rotation_mat_ttnn(head_dim, max_seq_len, mesh_device):
    """
    Prepare rotation matricies for decode mode.
    """
    rot_mat = get_rotation_mat(dhead=head_dim, end=max_seq_len * 2)
    rot_mats = [
        ttnn.from_torch(
            rot_mat_i.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        for rot_mat_i in rot_mat
    ]

    return rot_mats


# Sample logits from a distribution
def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs.squeeze(), top_p)
    else:
        next_token = torch.argmax(logits, dim=-1)

    return next_token


def cache_attention(mesh_device, state_dict, model_args, rot_emb_matrix_list, seq_start, seq_len, dtype):
    logger.info(f"Caching attention ops for iterations {seq_start} to {seq_start + seq_len}...")
    from models.experimental.grok.tt.grok_attention import TtGrokAttention

    attention_inputs = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    tt_attn = TtGrokAttention(
        mesh_device,
        state_dict,
        model_args,
        layer_num=0,
        dtype=dtype,
    )
    for iter in range(seq_start, seq_start + seq_len):
        logger.info(f"Caching iteration {iter}...")
        pos = iter

        padded_layer_past_len = min(nearest_32(pos + 1), model_args.sliding_window)
        attn_mask = ttnn.from_torch(
            # torch.zeros(1, 1, 32, padded_layer_past_len),
            torch.zeros(1, 32, 32, padded_layer_past_len),
            device=mesh_device,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        ATTN_MASK_MEMCFG = ttnn.create_sharded_memory_config(
            shape=(32, padded_layer_past_len),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        attn_mask = ttnn.interleaved_to_sharded(attn_mask, ATTN_MASK_MEMCFG)

        _ = tt_attn(
            attention_inputs,
            pos,
            pos + 1,
            attn_mask,
            rot_emb_matrix_list,
        )
        # ttnn.deallocate(tt_out[0])

    logger.info("Attention ops cached")
