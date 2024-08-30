# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import ttnn
from ttnn import ReplicateTensorToMesh
from models.utility_functions import nearest_32
import json
import math


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["prompt"])
    return in_prompt


def preprocess_inputs(input_prompts, tokenizer, model_args, dtype, instruct, device_mesh):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    # Pad the inputs to the max length prompt
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(input_prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.int32)

    logger.info(f"# of users: {len(encoded_prompts)}")
    for i, encoded in enumerate(encoded_prompts):
        # Right padding
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)

    input_mask_bool = input_tokens != tokenizer.pad_id
    input_mask = input_mask_bool.int()  # from_torch doesn't support bool type

    # convert to ttnn tensor
    # Encoded input tokens need to be uint32 for embedding. Otherwise the dtype conversion to bfloat16 will change the tokenizer ID
    input_tokens_tt = [
        ttnn.from_torch(
            input_tokens[:, i].unsqueeze(0),
            device=device_mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        for i in range(max_prompt_len)
    ]
    input_mask_tt = [
        ttnn.from_torch(
            input_mask[:, i].unsqueeze(0),
            device=device_mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        for i in range(max_prompt_len)
    ]
    return input_tokens_tt, max_prompt_len, input_mask_tt, input_tokens, input_mask_bool


def preprocess_inputs_prefill(
    input_prompts,
    tokenizer,
    model_args,
    dtype,
    instruct,
    device_mesh,
    max_generated_tokens,
    is_ci_env=False,
    max_prefill_len=32768,
):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    # The maximum KV-cache len supported is 32k. To avoid going out of memory, clip the max prefill length by the maximum number of tokens that will be generated
    if max_prefill_len == 32768:
        max_prefill_len = 32768 - max_generated_tokens

    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # The large input demo we provide contains more tokens than the maximum (32k tokens)
    # To avoid running out of memory, clip to max_prefill_len
    if min_prompt_len > max_prefill_len:
        logger.info(
            f"Clipping prompts to {max_prefill_len} tokens to avoid running out of memory. Also avoids `prefill-as-decode` mode for the entire demo (since it only computes 120 iterations)"
        )
        if instruct:  # When clipping, make sure to add the ` [/INST]` token at the end (4 tokens)
            encoded_prompts = [encod[: max_prefill_len - 4] for encod in encoded_prompts]
            dec_prompts = [tokenizer.decode(encod) + " [/INST]" for encod in encoded_prompts]
            encoded_prompts = [tokenizer.encode(prompt) for prompt in dec_prompts]
        else:
            encoded_prompts = [encod[:max_prefill_len] for encod in encoded_prompts]

        # Update prompt lengths
        prompt_lens = [len(x) for x in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)

    assert (
        max_prompt_len <= model_args.max_seq_len
    ), f"Max prompt length {max_prompt_len} exceeds model max seq len {model_args.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    logger.info(f"# of users: {len(encoded_prompts)}")
    input_tokens_prefill = []
    decoding_pos = []
    prefill_lens = []

    # Always prefill the nearest power of 2 for each user. This means that the majority of cases we will prefill more tokens than needed.
    # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
    for i, encoded in enumerate(encoded_prompts):
        # Prefill size is nearest power of 2
        prefill_seq_len = max(2 ** math.ceil(math.log(len(encoded), 2)), 128)

        # Initial prefill tensors full of pad tokens
        input_tokens_prefill_i = torch.full((1, prefill_seq_len), 0, dtype=torch.int32)
        input_tokens_prefill_i[0, : len(encoded[:])] = torch.tensor(encoded[:]).to(input_tokens_prefill_i)
        input_tokens_prefill.append(input_tokens_prefill_i)

        # Keep the correct decoding position of each user
        decoding_pos.append(len(encoded))
        prefill_lens.append(prefill_seq_len)

    return (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    )


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def prepare_inputs_ttnn(x_bsh, hidden_size, device_mesh):
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
    # Pad small batches to 32
    if batch < 32:
        zeros = torch.zeros(1, seq_len, 32, hidden_size)
        zeros[:, :, :batch, :] = x_1SBH
        x_1SBH = zeros

    # input goes to L1
    xs_1SBH = ttnn.from_torch(
        x_1SBH,
        device=device_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    return xs_1SBH


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


def cache_attention(device_mesh, state_dict, model_args, current_rot_mat, rot_matrix, dtype):
    from models.demos.t3000.mixtral8x7b.tt.mixtral_attention import TtMixtralAttention

    logger.info(f"Caching attention...")

    attention_inputs = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    tt_attn = TtMixtralAttention(
        device_mesh,
        state_dict,
        model_args,
        layer_num=0,
        dtype=dtype,
    )

    # SDPA in attention only supports chunk sizes of 32, 256, 512. This loop caches all 3 variants of SDPA
    for iter in [32, 200, 1024]:
        if iter > 0:
            current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)
        pos = iter

        _ = tt_attn(
            attention_inputs,
            [pos] * model_args.max_batch_size,
            None,
            current_rot_mat,
        )
    logger.info("Attention ops cached")


def get_single_rot_mat_torch(dhead, start_pos=0, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dhead, 2)[: (dhead // 2)].float() / dhead))
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    rot_matrix = torch.zeros(dhead, dhead)
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()
    rot_matrix = rot_matrix.transpose(-1, -2)

    freqs = start_pos * freqs
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    current_rot_mat = torch.zeros(dhead, dhead)
    current_rot_mat[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    current_rot_mat[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    current_rot_mat[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    current_rot_mat[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    return current_rot_mat.unsqueeze(0).unsqueeze(0).transpose(-1, -2), rot_matrix.unsqueeze(0).unsqueeze(0)


def get_single_rot_mat(dhead, device_mesh, start_pos=0, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dhead, 2)[: (dhead // 2)].float() / dhead))
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    rot_matrix = torch.zeros(dhead, dhead)
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()
    rot_matrix = rot_matrix.transpose(-1, -2)

    # Support for start_pos different than 0
    freqs = start_pos * freqs
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    current_rot_mat = torch.zeros(dhead, dhead)
    current_rot_mat[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    current_rot_mat[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    current_rot_mat[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    current_rot_mat[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    return ttnn.from_torch(
        current_rot_mat.unsqueeze(0).unsqueeze(0).transpose(-1, -2),  # 1,1,head_dim,head_dim
        device=device_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    ), ttnn.from_torch(
        rot_matrix.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
        device=device_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )


def get_single_rot_mat_multi_pos(dhead, device_mesh, start_pos_ids, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dhead, 2)[: (dhead // 2)].float() / dhead))
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    rot_matrix = torch.zeros(dhead, dhead)
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()
    rot_matrix = rot_matrix.transpose(-1, -2)

    current_rot_mat = torch.zeros(len(start_pos_ids), dhead, dhead)
    # Support for start_pos different than 0
    for i, start_pos in enumerate(start_pos_ids):
        freqs_i = start_pos * freqs
        sin_freqs, cos_freqs = torch.sin(freqs_i), torch.cos(freqs_i)
        current_rot_mat[i, torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
        current_rot_mat[i, torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
        current_rot_mat[i, torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
        current_rot_mat[i, torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    return ttnn.from_torch(
        current_rot_mat.unsqueeze(0).transpose(-1, -2),  # 1,B,head_dim,head_dim
        device=device_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    ), ttnn.from_torch(
        rot_matrix.unsqueeze(0).unsqueeze(0).repeat(1, len(start_pos_ids), 1, 1),  # 1,1,head_dim,head_dim
        device=device_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )


def precompute_freqs(dim: int, end: int, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


#  Add-Multiply method of rotary embeddings for prefill
def get_rot_transformation_mat(dhead):
    # ROPE op uses a single tile
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def get_prefill_rot_mat(head_dim, max_seq_len, device_mesh, seq_len):
    cos, sin = precompute_freqs(head_dim, max_seq_len * 2)
    cos_gathered, sin_gathered = gather_cos_sin(torch.arange(0, seq_len), cos, sin)
    assert cos_gathered.size() == (1, 1, seq_len, head_dim)
    assert sin_gathered.size() == (1, 1, seq_len, head_dim)

    cos_gathereds = ttnn.from_torch(
        cos_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
        device=device_mesh,
    )
    sin_gathereds = ttnn.from_torch(
        sin_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
        device=device_mesh,
    )

    rot_mats = [cos_gathereds, sin_gathereds]
    return rot_mats


def prepare_inputs_ttnn_prefill(x_bsh, device_mesh, num_tokens=None):
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
    if num_tokens is not None:  # Mask any additional tokens that were prefilled
        attn_mask_torch[num_tokens:, num_tokens:] = torch.finfo(torch.float32).min
    attn_mask = attn_mask_torch.view(1, 1, seq_len, seq_len)

    if seq_len == 128:
        attn_mask_dtype = ttnn.bfloat16
    else:
        attn_mask_dtype = ttnn.bfloat8_b
    attn_mask = ttnn.from_torch(
        attn_mask,
        device=device_mesh,
        dtype=attn_mask_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    # input goes to L1
    xs_1BSH = ttnn.from_torch(
        x_1BSH,
        device=device_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    return xs_1BSH, attn_mask, attn_mask_torch


# Updates some model arg parameters above their default values
def set_model_args(model_args, seq_len):
    if seq_len >= 8192:  # for seqlen larger than 8k we can't fit 32 users in a batch
        model_args.max_seq_len = seq_len
        model_args.max_batch_size = 32 // (seq_len // 8192)
        if seq_len > 8192 * 2:  # For seqlen higher than 16k, we can only fit 1 user in a batch
            model_args.max_batch_size = 1
    return model_args
