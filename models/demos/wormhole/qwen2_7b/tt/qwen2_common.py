# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import nearest_32
from safetensors.torch import load_file as safetensors_load_file
import json


def generate_cos_sin_cache_ttnn(
    tt_devices,
    head_dim,
    max_position_embeddings=2048,
    base=10000,
    dtype=None,
):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(
        max_position_embeddings,
        device=inv_freq.device,
        dtype=inv_freq.dtype,
    )
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    emb_cos = emb.cos()[None, None, :, :]
    emb_sin = emb.sin()[None, None, :, :]

    tt_cos_cached = [
        ttnn.from_torch(
            emb_cos,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        for tt_device in tt_devices
    ]

    tt_sin_cached = [
        ttnn.from_torch(
            emb_sin,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        for tt_device in tt_devices
    ]

    return tt_cos_cached, tt_sin_cached


def precompute_freqs(dim: int, end: int, theta: float = 1000000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 1000000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    # The default theta is sourced from the official config file: https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json, where `rope_theta` is referred.

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

    """
    The cosine and sine frequency values are arranged into the rotating matrix `rot_emb_matrix` as follows.
    The arrangment sources from Qwen2 codes in the Transformer library: https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/qwen2/modeling_qwen2.py#L182.
    For Qwen2-7B, the head dimension `dhead` is 128, whereas 64 cosine and 64 sine frequency values are needed. Let's shorten `cos_freqs` as `cos` and `sin_freqs` as `sin`. The rotating matrix looks as

    cos[0]     0      0       0  sin[0]     0      0       0
        0  cos[1]                       sin[1]
        0         ......
        0                cos[63]                      sin[63]
    -sin[0]                      cos[0]
        0  -sin[1]                      cos[1]
        0         ......                       ......
        0                -sin[63]                     cos[63]

    , which is a 128x128 matrix.

    For a head vector of length 128, it should be viewed as 64 distinct 2D vectors, with each 2D vector to be rotated by a unique designated frequency. Assume the head vector is named `head`. Then, (head[0], head[64]) forms the first 2D vector, (head[1], head[65]) forms the second, and so on. All 2D vectors are rotated simultaneously using the above rotation matrix. As a result, the output vector `output` will have its first 2D vector rotated to (output[0], output[64]) = (head[0] * cos[0] - head[64] * sin[0], head[0] * sin[0] + head[64] * cos[0]).
    """
    rot_emb_matrix = torch.zeros(emb_size, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, emb_dim), torch.arange(0, emb_dim)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(emb_dim, dhead), torch.arange(emb_dim, dhead)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(emb_dim, dhead), torch.arange(0, emb_dim)] = -sin_freqs.clone()
    rot_emb_matrix[..., torch.arange(0, emb_dim), torch.arange(emb_dim, dhead)] = sin_freqs.clone()

    return rot_emb_matrix


def gather_rotary_emb(rot_emb_matrix, position_ids):
    """
    Gather the rotary embeddings for a given position_ids
    """
    batch_size, seqlen = position_ids.shape
    emb_size, _, dhead = rot_emb_matrix.shape
    position_ids = position_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, dhead, dhead)
    rot_emb = rot_emb_matrix.gather(0, position_ids).view(batch_size, seqlen, dhead, dhead)
    return rot_emb


def get_rotation_mat(dhead, end, start_pos, seqlen, batch):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


def prepare_inputs_ttnn(x, current_pos, hidden_size, sliding_window, device):
    """
    Prepare inputs for decode mode. Assume that current token is at
    start_pos, and KV cache has valid data up to start_pos.
    x: (batch, seq, hidden_dim)
    start_pos: int
    """

    assert len(x.shape) == 3
    assert x.shape[2] == hidden_size

    batch = x.shape[0]
    seq_len = x.shape[1]
    hidden_size = x.shape[2]
    assert seq_len == 1, "Only supporting decode mode"

    # Support input on device
    if torch.is_tensor(x):  # Input on host -> Use torch
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
    else:  # Input on device -> Use ttnn
        x = ttnn.reshape(
            x, (batch, seq_len, 1, hidden_size)
        )  # [batch, seqlen, hidden_dim] -> [batch, seqlen, 1, hidden_dim]
        x = ttnn.permute(x, (1, 2, 0, 3))  # [seq_len, 1, batch, hidden_dim]
    # Pad small batches to 32
    if batch < 32:
        zeros = torch.zeros(1, seq_len, 32, hidden_size)
        zeros[:, :, :batch, :] = x
        x = zeros

    current = current_pos % sliding_window

    # expected shapes:
    # x: (seq_len, 1, batch, hidden_dim)
    # start_pos: int
    # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
    # rot_mat: [1, 1, head_dim, head_dim]
    # assert x.size() == (seq_len, 1, batch, hidden_size)

    if torch.is_tensor(x):
        x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    else:  # Convert the row major layout from embedding back to tile layout
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    return (
        x,
        current,
    )


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


from models.demos.wormhole.qwen2_7b.tt.qwen2_attention import TtQwen2Attention


def cache_attention(device, state_dict, model_args, rot_emb_matrix_list, dtype, iterations):
    attention_input = ttnn.from_torch(
        torch.randn(1, 1, 32, model_args.dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_model = TtQwen2Attention(
        [device],
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        configuration=model_args,
        rot_mat=rot_emb_matrix_list,
        start_pos=0,
    )
    for iter in range(iterations):
        pos = iter
        tt_out = tt_model(
            [attention_input],
            pos,
        )

    ttnn.deallocate(tt_model.wqkv_list[0])
    ttnn.deallocate(tt_model.wo_list[0])
    ttnn.deallocate(tt_model.layer_past_list[0][0])
    ttnn.deallocate(tt_model.layer_past_list[0][1])
    ttnn.deallocate(tt_model.head_dims[0])
    ttnn.deallocate(tt_model.expand_D_8D[0])
    ttnn.deallocate(tt_model.reduce_8D_D[0])
    ttnn.deallocate(tt_model.mask_Q_8D[0])
    ttnn.deallocate(attention_input)


def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def get_prefill_rot_mat(head_dim, max_seq_len, device, seq_len):
    cos, sin = precompute_freqs(head_dim, max_seq_len * 2)
    cos_gathered, sin_gathered = gather_cos_sin(torch.arange(0, seq_len), cos, sin)
    assert cos_gathered.size() == (1, 1, seq_len, head_dim)
    assert sin_gathered.size() == (1, 1, seq_len, head_dim)

    cos_gathereds = ttnn.from_torch(
        cos_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    sin_gathereds = ttnn.from_torch(
        sin_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    rot_mats = [cos_gathereds, sin_gathereds]
    return rot_mats


#  Add-Multiply method of rotary embeddings for prefill
def get_rot_transformation_mat(dhead):
    # ROPE op uses a single tile
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def prepare_inputs_ttnn_prefill(x_bsh, device):
    """
    Prepare inputs for prefill mode.
    x: (batch, seq, hidden_dim)
    B: batch (32)
    S: sequence len (1)
    H: dim (4096)
    """
    batch = x_bsh.size(0)
    seq_len = x_bsh.size(1)

    x_1BSH = x_bsh.unsqueeze(0)

    # Attention mask
    attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
    attn_mask_torch = torch.triu(attn_mask, diagonal=1)
    attn_mask = attn_mask_torch.view(1, 1, seq_len, seq_len).expand(8, 1, seq_len, seq_len)

    attn_mask = ttnn.from_torch(
        attn_mask,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # input goes to L1
    xs_1BSH = ttnn.from_torch(
        x_1BSH,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return xs_1BSH, attn_mask, attn_mask_torch


def load_safetensor_weights(weights_path):
    # Read the index file which contains the file names of the weight files.
    index_path = weights_path + "/model.safetensors.index.json"
    with open(index_path, "r") as f:
        index_data = json.load(f)

    # Retrieve the weight file names from the index JSON
    weight_map = index_data["weight_map"]
    safetensor_files = set(weight_map.values())

    # Read each safetensors file mentioned in the index
    loaded_weights = {}
    for file in safetensor_files:
        safetensor_path = weights_path + "/" + file
        weights = safetensors_load_file(safetensor_path)
        loaded_weights.update(weights)  # Merge weights into a single dictionary

    return loaded_weights
