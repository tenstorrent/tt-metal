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


def precompute_freqs(dim: int, end: int, theta: float = 1000000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def freqs_to_rotation_matrix(cos_freqs, sin_freqs):
    """
    Transform cos/sin frequencies to a rotation matrix.
    """
    emb_size, emb_dim = cos_freqs.shape
    dhead = emb_dim * 2
    rot_emb_matrix = torch.zeros(emb_size, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    rot_emb_matrix = rot_emb_matrix.transpose(-1, -2)  # Necessary for correct rotation when applied as (x @ R)
    return rot_emb_matrix, emb_size


def get_rotation_mat(dhead, end):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat, emb_size = freqs_to_rotation_matrix(cos, sin)
    rot_mat = [rot_mat[i, :, :] for i in range(emb_size)]
    return rot_mat


def prepare_inputs_ttnn(x_bsh, hidden_size, current_pos, sliding_window, device_mesh):
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
        device=device_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    # Attention mask
    padded_layer_past_len = min(nearest_32(current_pos + 1), sliding_window)
    current = current_pos % sliding_window
    n_local_heads = 4  # model_args.n_heads // 8 # num_devices TODO pass the arguments instead of hard coding

    attn_mask = torch.zeros(seq_len, 32, 32, padded_layer_past_len)  # [SB4P]

    # Fill mask with -inf outside the processed tokens
    if current_pos < sliding_window:
        attn_mask[:, :, :, current + 1 :] = torch.finfo(attn_mask.dtype).min
    else:
        # TODO Double check this logic.
        # As current position increases closer to sliding window we're masking [: curr] and [window - curr : ]
        #  so for a window = 32, and we're on position 55, curr: 55%32=23. we're then masking [:23] and [9:] which is masking everything.
        attn_mask[:, :, :, :current] = torch.finfo(attn_mask.dtype).min
        attn_mask[:, :, :, sliding_window - current :] = torch.finfo(attn_mask.dtype).min

    attn_mask = ttnn.from_torch(
        attn_mask,
        device=device_mesh,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    ATTN_MASK_MEMCFG = ttnn.create_sharded_memory_config(
        shape=(32, padded_layer_past_len),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    attn_mask = ttnn.experimental.tensor.interleaved_to_sharded(attn_mask, sharded_mem_config=ATTN_MASK_MEMCFG)

    return xs_1SBH, attn_mask


def prepare_rotation_mat_ttnn(head_dim, max_seq_len, device_mesh):
    """
    Prepare rotation matricies for decode mode.
    """
    rot_mat = get_rotation_mat(dhead=head_dim, end=max_seq_len * 2)
    rot_mats = [
        ttnn.from_torch(
            rot_mat_i.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
            device=device_mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
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


def cache_attention(device_mesh, state_dict, model_args, rot_emb_matrix_list, seq_start, seq_len, dtype):
    logger.info(f"Caching attention ops for iterations {seq_start} to {seq_start + seq_len}...")
    from models.experimental.grok.tt.grok_attention import TtGrokAttention

    attention_inputs = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    tt_attn = TtGrokAttention(
        device_mesh,
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
            device=device_mesh,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        ATTN_MASK_MEMCFG = ttnn.create_sharded_memory_config(
            shape=(32, padded_layer_past_len),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        attn_mask = ttnn.experimental.tensor.interleaved_to_sharded(attn_mask, sharded_mem_config=ATTN_MASK_MEMCFG)

        _ = tt_attn(
            attention_inputs,
            pos,
            pos + 1,
            attn_mask,
            rot_emb_matrix_list,
        )
        # ttnn.deallocate(tt_out[0])

    logger.info("Attention ops cached")
