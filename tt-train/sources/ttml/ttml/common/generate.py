import ttnn
import ttml
from ttml.common.utils import round_up_to_tile
import numpy as np


def create_causal_mask_kv_cache(
    query_seq_len: int, prompt_len: int = 0
) -> ttml.autograd.Tensor:
    """Create a causal attention mask for autoregressive generation with KV cache.

    This matches the C++ implementation exactly.

    Args:
        query_seq_len: Length of query sequence
        prompt_len: Length of prompt (for decode mode, this is the cache position)

    Returns:
        Causal mask tensor
    """
    whole_seq_len = prompt_len + query_seq_len
    padded_query_len = round_up_to_tile(query_seq_len)
    padded_whole_len = round_up_to_tile(whole_seq_len)

    # Mask shape: [padded_query_len, padded_whole_len] - query_len x key_len
    mask_data = np.zeros((padded_query_len, padded_whole_len), dtype=np.float32)

    # Fill mask: token i can attend to positions 0 through i + prompt_len (inclusive)
    # This matches C++: for (uint32_t j = 0; j <= i + prompt_len; ++j)
    # range(n) gives [0, 1, 2, ..., n-1], so range(prompt_len + i + 1) gives [0, 1, 2, ..., prompt_len + i]
    for i in range(query_seq_len):
        for j in range(prompt_len + i + 1):
            mask_data[i, j] = 1.0

    # Reshape to [1, 1, padded_query_len, padded_whole_len]
    mask_data = mask_data.reshape(1, 1, padded_query_len, padded_whole_len)
    mask_tensor = ttml.autograd.Tensor.from_numpy(
        mask_data, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )

    return mask_tensor


def tokens_to_tensor_kv_cache(tokens: list) -> ttml.autograd.Tensor:
    """Create tensor from token IDs with padding to nearest multiple of 32.

    This matches the C++ implementation exactly.

    Args:
        tokens: List of token IDs

    Returns:
        Token tensor with padding
    """
    actual_len = len(tokens)
    padded_len = round_up_to_tile(actual_len)

    # Pad tokens with zeros to reach padded length
    padded_tokens = np.zeros(padded_len, dtype=np.uint32)
    for i in range(actual_len):
        padded_tokens[i] = tokens[i]

    # Reshape to [1, 1, 1, padded_len]
    padded_tokens = padded_tokens.reshape(1, 1, 1, padded_len)
    tokens_tensor = ttml.autograd.Tensor.from_numpy(
        padded_tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )

    return tokens_tensor


def generate(
    model,
    prompt_tokens,
    transformer_config,
    temperature=0.0,
    vocab_size=None,
    composer=None,
):
    """Generate text with KV cache for efficient inference.

    Args:
        model: LLaMA model instance
        prompt_tokens: Initial prompt token IDs
        transformer_config: Model config with num_blocks, num_groups, embedding_dim, max_sequence_length
        use_sampling: Whether to use temperature sampling (if False, uses greedy decoding)
    """

    ttml.autograd.AutoContext.get_instance().set_gradient_mode(
        ttml.autograd.GradMode.DISABLED
    )
    model.eval()

    # Create KV cache
    batch_size = 1
    num_layers = transformer_config.num_blocks
    num_groups = transformer_config.num_groups
    max_seq_len = transformer_config.max_sequence_length
    head_dim = transformer_config.embedding_dim // transformer_config.num_heads

    kv_cache_config = ttml.models.KvCacheConfig(
        num_layers, batch_size, num_groups, max_seq_len, head_dim
    )
    kv_cache = ttml.models.KvCache(kv_cache_config)

    # Reset KV cache for new sequence
    kv_cache.reset()

    generated_tokens = prompt_tokens.copy()
    prompt_len = len(prompt_tokens)

    logits_mask_tensor = None
    padded_vocab_size = round_up_to_tile(vocab_size)
    if padded_vocab_size != vocab_size:
        logits_mask_tensor = build_logits_mask(vocab_size, padded_vocab_size)

    # Generate tokens one by one
    for step in range(max_seq_len - prompt_len):
        # For first step (prefill): use all prompt tokens
        # For subsequent steps (decode): use only the last generated token
        processed_tokens = 0
        if kv_cache.get_cache_position() == 0:
            # Prefill: process entire prompt
            input_tokens = generated_tokens
        else:
            # Decode: process only last token
            input_tokens = [generated_tokens[-1]]
            processed_tokens = len(generated_tokens) - 1

        token_tensor = tokens_to_tensor_kv_cache(input_tokens)

        # Create causal mask
        # For prefill: query_len = prompt_len, prompt_len = 0 (all tokens can attend to previous)
        # For decode: query_len = 1, prompt_len = cache_position (new token can attend to all cached tokens)
        # This matches C++: create_causal_mask(device, input_tokens.size(), processed_tokens)
        mask = create_causal_mask_kv_cache(len(input_tokens), processed_tokens)
        new_tokens = len(input_tokens)
        logits = model(token_tensor, mask, kv_cache=kv_cache, new_tokens=new_tokens)

        # Sample next token
        # The logits tensor has shape [1, 1, seq_len, vocab_size] where seq_len may be padded
        # We need to extract the token at the last actual position (len(input_tokens) - 1)
        next_token_tensor = ttml.ops.sample.sample_op(
            logits, temperature, np.random.randint(low=1e7), logits_mask_tensor
        )
        next_token_idx = len(input_tokens) - 1
        next_token = int(
            next_token_tensor.to_numpy(composer=composer).flatten()[next_token_idx]
        )

        generated_tokens = np.append(generated_tokens, next_token)

    kv_cache.reset()

    return generated_tokens
