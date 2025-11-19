# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch
from loguru import logger

try:
    from transformers import AutoTokenizer

    transformers_available = True
except ImportError:
    transformers_available = False
    logger.warning("transformers library not available. Tokenizer utilities will not function.")


def load_minicpm_tokenizer(model_name: str = "openbmb/MiniCPM-o-2_6", trust_remote_code: bool = True):
    """
    Load the MiniCPM-o-2_6 tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        trust_remote_code: Whether to trust remote code

    Returns:
        AutoTokenizer instance
    """
    if not transformers_available:
        raise ImportError("transformers library is required for tokenizer functionality")

    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def encode_prompt(tokenizer, prompt_text: str, system_prompt_text: Optional[str] = None):
    """
    Encode text prompt to token IDs.

    Adapted from encode_prompt_hf in models/tt_transformers/tt/common.py:242

    Args:
        tokenizer: HuggingFace tokenizer
        prompt_text: User prompt text
        system_prompt_text: Optional system prompt

    Returns:
        List of token IDs
    """
    chat = []
    if isinstance(prompt_text, str):
        if system_prompt_text:
            chat.append({"role": "system", "content": system_prompt_text})
        if prompt_text:
            chat.append({"role": "user", "content": prompt_text})
        return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
    else:
        return tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)


def preprocess_inputs_prefill(
    input_prompts: List[str],
    tokenizer,
    model_args,
    instruct: bool = True,
    max_generated_tokens: int = 200,
    max_prefill_len: Optional[int] = None,
):
    """
    Preprocess input prompts for prefill phase.

    Adapted from preprocess_inputs_prefill in models/tt_transformers/tt/common.py:166-239

    Args:
        input_prompts: List of input prompt strings
        tokenizer: HuggingFace tokenizer
        model_args: Model arguments containing configuration
        instruct: Whether to use instruct mode
        max_generated_tokens: Maximum tokens to generate
        max_prefill_len: Maximum prefill length

    Returns:
        Tuple of (input_tokens_prefill, encoded_prompts, decoding_pos, prefill_lens)
    """
    # Encode all prompts
    encoded_prompts = []
    for prompt in input_prompts:
        if isinstance(prompt, str):
            # Use our encode_prompt function for consistent formatting
            tokens = encode_prompt(tokenizer, prompt)
        else:
            # Assume it's already tokenized
            tokens = prompt
        encoded_prompts.append(tokens)

    # Determine maximum sequence length
    max_seq_len = model_args.max_seq_len
    if max_prefill_len is not None:
        max_seq_len = min(max_seq_len, max_prefill_len)

    # Check sequence length constraints
    for m_args in model_args if isinstance(model_args, list) else [model_args]:
        if m_args.max_context_len < max_seq_len:
            logger.warning(
                f"Requested max_seq_len {max_seq_len} exceeds model max context len {m_args.max_context_len}. "
                f"Clipping to {m_args.max_context_len}"
            )
            max_seq_len = m_args.max_context_len

    # Handle variable length prompts
    if len(set(len(p) for p in encoded_prompts)) > 1:
        # Pad shorter prompts to the maximum length among all prompts
        max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
        assert (
            max_generated_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

        # Update prompt lengths
        prompt_lens = [len(x) for x in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)

        for m in model_args if isinstance(model_args, list) else [model_args]:
            assert (
                max_prompt_len <= m.max_seq_len
            ), f"Max prompt length {max_prompt_len} exceeds model max seq len {m.max_seq_len}"
        assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
        assert (
            min_prompt_len <= max_prompt_len
        ), f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

        logger.info(f"# of users: {len(encoded_prompts)}")
        input_tokens_prefill = []
        decoding_pos = []
        prefill_lens = []

        # Pad each prompt to the maximum length among all prompts.
        # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
        for i, encoded in enumerate(encoded_prompts):
            # Initial prefill tensors full of pad tokens
            input_tokens_prefill_i = torch.full((1, max_prompt_len), tokenizer.pad_token_id, dtype=torch.int32)
            input_tokens_prefill_i[0, : len(encoded[:])] = torch.tensor(encoded[:]).to(input_tokens_prefill_i)
            input_tokens_prefill.append(input_tokens_prefill_i)

            # Keep the correct decoding position of each user
            decoding_pos.append(len(encoded))
            prefill_lens.append(max_prompt_len)

    else:
        # All prompts have the same length
        max_encoded_prompt_len = len(encoded_prompts[0])
        assert (
            max_generated_tokens + max_encoded_prompt_len <= max_seq_len
        ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

        input_tokens_prefill = []
        decoding_pos = []
        prefill_lens = []

        for encoded in encoded_prompts:
            input_tokens_prefill.append(torch.tensor(encoded, dtype=torch.int32).unsqueeze(0))
            decoding_pos.append(len(encoded))
            prefill_lens.append(len(encoded))

    return (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    )


def decode_tokens(tokenizer, token_ids: List[int], skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs to text.

    Args:
        tokenizer: HuggingFace tokenizer
        token_ids: List of token IDs
        skip_special_tokens: Whether to skip special tokens

    Returns:
        Decoded text string
    """
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def format_multimodal_prompt(
    text: str, has_audio: bool = False, has_image: bool = False, system_prompt: str = None
) -> str:
    """
    Format text prompt for multimodal MiniCPM-o architecture.

    Args:
        text: Main user prompt text
        has_audio: Whether audio input is present
        has_image: Whether image input is present
        system_prompt: Optional system prompt

    Returns:
        Formatted prompt string with special tokens
    """
    formatted_prompt = ""

    # Add system prompt if provided
    if system_prompt:
        formatted_prompt += f"System: {system_prompt}\n\n"

    # Add modality indicators using MiniCPM special tokens
    modalities = []
    if has_audio:
        modalities.append("<|audio|>")
    if has_image:
        modalities.append("<|image|>")

    if modalities:
        formatted_prompt += f"Input modalities: {' '.join(modalities)}\n\n"

    # Add main text prompt
    formatted_prompt += f"User: {text}"

    return formatted_prompt


def create_attention_mask_for_multimodal(
    text_tokens: torch.Tensor, audio_feature_len: int = 0, image_feature_len: int = 0
) -> torch.Tensor:
    """
    Create attention mask for multimodal sequences.

    MiniCPM-o architecture processes multimodal inputs by concatenating
    features from different modalities. This function creates appropriate
    attention masks to handle cross-modal interactions.

    Args:
        text_tokens: Text token IDs tensor [batch, seq_len]
        audio_feature_len: Length of audio features (0 if no audio)
        image_feature_len: Length of image features (0 if no image)

    Returns:
        Attention mask tensor [batch, total_seq_len]
    """
    batch_size, text_seq_len = text_tokens.shape

    # Calculate total sequence length
    total_seq_len = text_seq_len + audio_feature_len + image_feature_len

    # Create full attention mask (all positions can attend to each other)
    # In multimodal models, we typically allow full attention across modalities
    attention_mask = torch.ones(batch_size, total_seq_len, dtype=torch.int64)

    return attention_mask


def get_special_tokens(tokenizer):
    """
    Get special token IDs from tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Dict with special token IDs
    """
    return {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
