import torch
import ttnn
from models.tt_transformers.tt.common import gather_cos_sin, PagedAttentionConfig
from models.experimental.phi3_mini.tt.model_config import Phi3MiniModelArgs
from loguru import logger
import math


def precompute_freqs(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scale_factor: int = 1.0,
    ext_scale_tensor: torch.tensor = torch.tensor([1.0]),
):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation.
        scale_factor (float, optional): Factor based on Ratio of target to original context length.
        ext_scale_tensor (torch.tesnor, optional): Scaling tensor applied to RoPE frequencies to modulate positional encoding.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (ext_scale_tensor * theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs) * scale_factor, torch.sin(freqs) * scale_factor


def get_prefill_rot_mat(
    head_dim, mesh_device, seq_len, theta, scale_factor, ext_scale_tensors, orig_context_len, start_pos=0
):
    if seq_len > orig_context_len:
        ext_scale_tensor = torch.tensor(ext_scale_tensors["long_factor"], dtype=torch.float32)
    else:
        ext_scale_tensor = torch.tensor(ext_scale_tensors["short_factor"], dtype=torch.float32)
    cos, sin = precompute_freqs(head_dim, seq_len, theta, scale_factor, ext_scale_tensor)
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


def preprocess_inputs_prefill(
    input_prompts,
    tokenizer,
    model_args,
    instruct,
    max_generated_tokens,
    max_prefill_len=128 * 1024,
):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    # To avoid going out of memory, clip the max prefill length by the maximum number of tokens that will be generated
    if max_prefill_len == 128 * 1024:
        max_prefill_len = 128 * 1024 - max_generated_tokens

    encoded_prompts = [
        model_args[idx % len(model_args)].encode_prompt(prompt, instruct=instruct)
        for idx, prompt in enumerate(input_prompts)
    ]

    # Print the length of encoded prompts
    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in encoded_prompts))

    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # To avoid running out of memory when giving prompts larger than the maximum, clip to max_prefill_len
    if min_prompt_len > max_prefill_len:
        logger.info(f"Left-clipping prompts to {max_prefill_len}")
        if instruct:
            # We need to allow a few tokens for the system prompt and the special turn tokens for assistant and user;
            # to find out how big those will be, we will:
            # 1. Tokenize the entire prompt with non-instruct tokenization
            # 2. Calculate overhead = length of instruct tokenization - length of non-instruct tokenization
            # 3. Shorten the tokenized clipped prompt by the overhead and convert back to text
            # 4. Tokenize the result with instruct tokenization
            # 5. Assert that the length of this is equal to the max_prefill_len
            raw_prompts = [
                model_args[idx % len(model_args)].encode_prompt(prompt, instruct=False)
                for idx, prompt in enumerate(input_prompts)
            ]
            overhead = [len(e) - len(r) for e, r in zip(encoded_prompts, raw_prompts)]
            shortened = [
                tokenizer[idx % len(model_args)].decode(e[-(max_prefill_len - o) :])
                for idx, e, o in enumerate(zip(raw_prompts, overhead))
            ]
            encoded_prompts = [
                model_args[idx % len(model_args)].encode_prompt(prompt, instruct=instruct)
                for idx, prompt in enumerate(shortened)
            ]
            assert all(
                len(e) == max_prefill_len for e in encoded_prompts
            ), f"Clipped prompts are not of the correct length, expected {max_prefill_len} but got {[len(e) for e in encoded_prompts]}"
        else:
            encoded_prompts = [encod[-max_prefill_len:] for encod in encoded_prompts]

        # Update prompt lengths
        prompt_lens = [len(x) for x in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)
    for m in model_args:
        assert (
            max_prompt_len <= m.max_seq_len
        ), f"Max prompt length {max_prompt_len} exceeds model max seq len {m.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    logger.info(f"# of users: {len(encoded_prompts)}")
    input_tokens_prefill = []
    decoding_pos = []
    prefill_lens = []

    # Always prefill the nearest power of 2 for each user. This means that the majority of cases we will prefill more tokens than needed.
    # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
    # Prefill size is padded to nearest power of 2 of max prompt lenght
    prefill_seq_len = max(2 ** math.ceil(math.log(max_prompt_len, 2)), 128)
    for i, encoded in enumerate(encoded_prompts):
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


def get_max_prefill_chunk_size(seq_len, max_prefill_seq_len, min_chunk_size=None):
    """
    Determine the largest multiple of min_chunk_size or default : 2048 that divides `seq_len` and is less than or equal to `max_prefill_seq_len`.

    **Assumptions**:
    - `seq_len` is a multiple of min_chunk_size or default:2048.
    - `max_prefill_seq_len` is a multiple of min_chunk_size or default:2048.
    """
    MIN_CHUNK_SIZE = 2048 if min_chunk_size is None else min_chunk_size

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


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
):
    from models.experimental.phi3_mini.tt.phi3_mini_model import Phi3Transformer

    tt_model_args = Phi3MiniModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    if num_layers is not None:
        tt_model_args.n_layers = num_layers

    # Avoid loading state_dict for every DP model
    if not state_dict:
        state_dict = tt_model_args.load_state_dict()

    model = Phi3Transformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None

    return tt_model_args, model, tt_kv_cache, state_dict
