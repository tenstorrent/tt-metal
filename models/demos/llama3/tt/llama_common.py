# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn


class HostEmbedding(torch.nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.emb = torch.nn.Embedding(model_args.vocab_size, model_args.dim)

    def forward(self, x):
        return self.emb(x)


# Default configuration for Paged Attention
class PagedAttentionConfig:
    def __init__(self, block_size=32, max_num_blocks=1024):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks


def encode_prompt_llama_instruct(tokenizer, prompt_text, system_prompt_text=None):
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {{ model_answer_1 }}<|eot_id|>
    """
    begin_of_text = [tokenizer.special_tokens["<|begin_of_text|>"]]
    start_header = [tokenizer.special_tokens["<|start_header_id|>"]]
    end_header = [tokenizer.special_tokens["<|end_header_id|>"]]
    end_turn = [tokenizer.special_tokens["<|eot_id|>"]]
    system = tokenizer.encode("system", bos=False, eos=False)
    user = tokenizer.encode("user", bos=False, eos=False)
    assistant = tokenizer.encode("assistant", bos=False, eos=False)
    prompt = tokenizer.encode(prompt_text, bos=False, eos=False)

    system_prompt = start_header + system + end_header + system_prompt_text + end_turn if system_prompt_text else []
    user_prompt = start_header + user + end_header + prompt + end_turn
    assistant_reply = start_header + assistant + end_header
    return begin_of_text + system_prompt + user_prompt + assistant_reply


def apply_scaling(freqs: torch.Tensor, scale_factor: float = 8):
    # Llama-3.x specific scaling
    # Values obtained from grid search
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = True, scale_factor: float = 8):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 500000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    if use_scaled:
        freqs = apply_scaling(freqs, scale_factor)
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
    return rot_emb_matrix


def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


def get_prefill_rot_mat(head_dim, max_seq_len, mesh_device, seq_len, scale_factor, start_pos=0):
    cos, sin = precompute_freqs(head_dim, max_seq_len * 2, scale_factor=scale_factor)
    cos_gathered, sin_gathered = gather_cos_sin(torch.arange(start_pos, start_pos + seq_len), cos, sin)
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
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def get_single_rot_mat(
    dhead,
    mesh_device,
    num_devices,
    start_pos=0,
    theta: float = 500000.0,
    use_scaled=True,
    on_host=False,
):
    freqs_unscaled = 1.0 / (theta ** (torch.arange(0, dhead, 2)[: (dhead // 2)].float() / dhead))
    if use_scaled:
        freqs = apply_scaling(freqs_unscaled)
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    rot_matrix = torch.zeros(dhead, dhead)
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()
    rot_matrix = rot_matrix.transpose(-1, -2)

    # Support for start_pos different than 0
    freqs = start_pos * freqs_unscaled
    if use_scaled:
        freqs = apply_scaling(freqs)
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    current_rot_mat = torch.zeros(dhead, dhead)
    current_rot_mat[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    current_rot_mat[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    current_rot_mat[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    current_rot_mat[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    return ttnn.from_torch(
        current_rot_mat.T.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
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


def copy_host_to_device(host_tensors, device_tensors=None, mesh_device=None):
    """
    Helper function which copies host tensors to device tensors.
    If no device_tensors are provided, it creates new device tensors and returns them.
    """
    if device_tensors is None:
        assert mesh_device is not None, "mesh_device is required when device_tensors is None"
        ret = []
        for i in range(len(host_tensors)):
            on_device = ttnn.to_device(host_tensors[i], device=mesh_device) if host_tensors[i] else None
            ret.append(on_device)
        return ret
    else:
        for i in range(len(host_tensors)):
            if host_tensors[i] is None:
                assert device_tensors[i] is None
                continue
            ttnn.copy_host_to_device_tensor(host_tensors[i], device_tensors[i])
        return device_tensors


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


def sample_host(tt_input, mesh_device, temperature=0.6, top_p=0.08, on_host=True):
    vocab_size = tt_input.shape[-1]
    if mesh_device:
        pt_input = ttnn.to_torch(
            tt_input,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=list(mesh_device.shape)),
        )[:, :1, :, :vocab_size]
    else:  # input already on host
        pt_input = tt_input[..., :vocab_size]

    if temperature > 0:
        probs = torch.softmax(pt_input / temperature, dim=-1)
        pt_out = sample_top_p(probs.squeeze(), top_p)
        if mesh_device:
            pt_out = pt_out.view(1, 1, 1, -1)
    else:
        if mesh_device:
            pt_out = torch.argmax(pt_input, dim=-1, keepdim=True).transpose(-1, -2)
        else:
            pt_out = torch.argmax(pt_input, dim=-1)

    if mesh_device is None:
        return pt_out
    if on_host:
        return (
            ttnn.as_tensor(
                pt_out,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
            ),
            pt_out,
        )
    else:
        return (
            ttnn.from_torch(
                pt_out,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            pt_out,
        )


def get_padded_prefill_len(seq_len):
    """
    If seq_len is less than 128, pad to 128
    If seq_len is more than 128, pad to whichever is smaller: a power of 2 or a multiple of 2048
    TODO: Generalize for max_mm_seq_len different from 2048
    """
    if seq_len <= 128:
        return 128
    pow_2_pad = nearest_pow_2(seq_len)
    mult_2048_pad = 2048 * math.ceil(seq_len / 2048)
    min_extended_pad = min(pow_2_pad, mult_2048_pad)
    return min_extended_pad


def get_block_size(kv_cache):
    return kv_cache[0][0].shape[2]


def num_blocks_in_seq(seq_len, block_size):
    return math.ceil(seq_len / block_size)


def nearest_pow_2(x):
    return 2 ** math.ceil(math.log2(x))


def get_max_prefill_chunk_size(seq_len, max_prefill_seq_len):
    """
    Determine the largest multiple of 2048 that divides `seq_len` and is less than or equal to `max_prefill_seq_len`.

    **Assumptions**:
    - `seq_len` is a multiple of 2048.
    - `max_prefill_seq_len` is a multiple of 2048.
    """
    MIN_CHUNK_SIZE = 2048

    if not isinstance(seq_len, int) or not isinstance(max_prefill_seq_len, int):
        raise TypeError("Both seq_len and max_prefill_seq_len must be integers.")
    if seq_len <= 0 or max_prefill_seq_len <= 0:
        raise ValueError("Both seq_len and max_prefill_seq_len must be positive integers.")

    if seq_len % MIN_CHUNK_SIZE != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of {MIN_CHUNK_SIZE}.")
    if max_prefill_seq_len % MIN_CHUNK_SIZE != 0:
        raise ValueError(f"max_prefill_seq_len ({max_prefill_seq_len}) must be a multiple of {MIN_CHUNK_SIZE}.")

    # Calculate the maximum possible chunk size
    # It cannot exceed either max_prefill_seq_len or seq_len
    max_possible_chunk = min(max_prefill_seq_len, seq_len)

    # Iterate from the largest possible multiple of MIN_CHUNK_SIZE down to MIN_CHUNK_SIZE
    for chunk_size in range(max_possible_chunk, 0, -MIN_CHUNK_SIZE):
        if seq_len % chunk_size == 0:
            return chunk_size

    raise ValueError("No valid chunk size found")
