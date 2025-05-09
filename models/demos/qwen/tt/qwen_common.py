# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json

import torch
from safetensors.torch import load_file as safetensors_load_file

import ttnn


class HostEmbedding(torch.nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.emb = torch.nn.Embedding(model_args.vocab_size, model_args.dim)

    def forward(self, x):
        return self.emb(x)


def precompute_freqs(dim: int, end: int, theta: float = 1000000.0, use_scaled: bool = False):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 1000000.0.

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
    rot_emb_matrix[..., torch.arange(0, emb_dim), torch.arange(0, emb_dim)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(emb_dim, dhead), torch.arange(emb_dim, dhead)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(emb_dim, dhead), torch.arange(0, emb_dim)] = -sin_freqs.clone()
    rot_emb_matrix[..., torch.arange(0, emb_dim), torch.arange(emb_dim, dhead)] = sin_freqs.clone()

    # rot_emb_matrix = rot_emb_matrix.transpose(-1, -2)  # Necessary for correct rotation when applied as (x @ R)
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


def get_rotation_mat_batched(rot_mat, start_pos, seqlen, batch):
    if isinstance(start_pos, int):
        start_pos = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
    position_ids = start_pos.view(seqlen, batch)
    rot_emb = gather_rotary_emb(rot_mat, position_ids)
    return rot_emb


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


def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def get_prefill_rot_mat(head_dim, max_seq_len, mesh_device, seq_len):
    cos, sin = precompute_freqs(head_dim, max_seq_len * 2)
    cos_gathered, sin_gathered = gather_cos_sin(torch.arange(0, seq_len), cos, sin)
    assert cos_gathered.size() == (1, 1, seq_len, head_dim)
    assert sin_gathered.size() == (1, 1, seq_len, head_dim)

    cos_gathereds = ttnn.from_torch(
        cos_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_gathereds = ttnn.from_torch(
        sin_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    rot_mats = [cos_gathereds, sin_gathereds]
    return rot_mats


#  Add-Multiply method of rotary embeddings for prefill
def get_rot_transformation_mat(dhead):
    # ROPE op uses a single tile
    dhead = 32
    emb_dim = dhead // 2
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    # rot_emb_matrix[..., torch.arange(emb_dim, dhead), torch.arange(0, emb_dim)] = 1
    # rot_emb_matrix[..., torch.arange(0, emb_dim), torch.arange(emb_dim, dhead)] = -1
    return rot_emb_matrix


def get_single_rot_mat(
    dhead, mesh_device, num_devices, start_pos=0, theta: float = 1000000.0, use_scaled=False, on_host=False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dhead, 2)[: (dhead // 2)].float() / dhead))
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    emb_dim = cos_freqs.shape[0]
    dhead = emb_dim * 2
    rot_matrix = torch.zeros(dhead, dhead)
    rot_matrix[..., torch.arange(0, emb_dim), torch.arange(0, emb_dim)] = cos_freqs.clone()
    rot_matrix[..., torch.arange(emb_dim, dhead), torch.arange(emb_dim, dhead)] = cos_freqs.clone()
    rot_matrix[..., torch.arange(emb_dim, dhead), torch.arange(0, emb_dim)] = -sin_freqs.clone()
    rot_matrix[..., torch.arange(0, emb_dim), torch.arange(emb_dim, dhead)] = sin_freqs.clone()

    # Support for start_pos different than 0
    freqs = start_pos * freqs
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    current_rot_mat = torch.zeros(dhead, dhead)
    current_rot_mat[..., torch.arange(0, emb_dim), torch.arange(0, emb_dim)] = cos_freqs.clone()
    current_rot_mat[..., torch.arange(emb_dim, dhead), torch.arange(emb_dim, dhead)] = cos_freqs.clone()
    current_rot_mat[..., torch.arange(emb_dim, dhead), torch.arange(0, emb_dim)] = -sin_freqs.clone()
    current_rot_mat[..., torch.arange(0, emb_dim), torch.arange(emb_dim, dhead)] = sin_freqs.clone()

    return ttnn.from_torch(
        current_rot_mat.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
        device=mesh_device if not on_host else None,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if num_devices > 1 or not on_host else None,
    ), ttnn.from_torch(
        rot_matrix.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
        device=mesh_device if not on_host else None,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if num_devices > 1 or not on_host else None,
    )


def num_to_core_range_set(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_x - 1, num_y - 1),
            ),
        }
    )


def calculate_hidden_dim(dim, ffn_dim_multiplier, multiple_of):
    """Helper function based on logic used in reference model:
    https://github.com/meta-llama/llama-models/blob/e4a6ed52a142bb9b5106dcbf48e41f97f8e7378e/models/llama3/reference_impl/model.py#L227C7-L231C83
    """
    hidden_dim = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def get_out_subblock_w(per_core_N, out_subblock_h):
    """
    Helper function to calculate the out_subblock_w based on the per_core_N and out_subblock_h
    """
    out_subblock_w = 4  # TODO: Check with LLK team if this is the true bound, might be 8 now
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_N % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def first_five(tensor, mesh_device):
    """
    Helper function to return the first 5 elements of a tensor via torch
    """
    return torch.Tensor(ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)))[0, 0, 0, :5]


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
