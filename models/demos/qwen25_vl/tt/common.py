# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import os
from loguru import logger
import ttnn
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta


def merge_vision_tokens(
    input_ids,
    input_embeds,
    image_embeds,
    hf_config,
):
    """
    input_ids are the input ids of the text tokens
    input_embeds are torch embedded text tokens
    image_embeds are torch embedded vision tokens
    """
    n_image_tokens = (input_ids == hf_config.image_token_id).sum().item()
    n_image_features = image_embeds.shape[0]
    if n_image_tokens != n_image_features:
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        )

    mask = input_ids == hf_config.image_token_id
    mask_unsqueezed = mask.unsqueeze(-1)
    mask_expanded = mask_unsqueezed.expand_as(input_embeds)
    image_mask = mask_expanded.to(input_embeds.device)

    input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)
    return input_embeds


def preprocess_inputs_prefill(
    input_embeds,
    model_args,
    max_prefill_len=None,
):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    # To avoid going out of memory, clip the max prefill length by the maximum number of tokens that will be generated
    if max_prefill_len is None:
        max_prefill_len = model_args.max_seq_len

    # Print the length of encoded prompts
    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in input_embeds))

    max_prompt_len = max(len(x) for x in input_embeds)
    assert (
        max_prompt_len <= max_prefill_len
    ), f"Max prompt length {max_prompt_len} exceeds max prefill len {max_prefill_len} and clipping and retokenizing is not supported for Qwen2.5 VL"

    logger.info(f"# of users: {len(input_embeds)}")
    input_prefill = []
    decoding_pos = []
    prefill_lens = []

    # Always prefill the nearest power of 2 for each user. This means that the majority of cases we will prefill more tokens than needed.
    # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
    pad_embedding = model_args.tokenizer.encode(model_args.tokenizer.pad_token, add_special_tokens=False)[0]
    for input_embed in input_embeds:
        # Prefill size is nearest power of 2 - FIXME: *really*? power of 2? surely we only need it to be a multiple of 1024 or whatever?
        prefill_seq_len = max(2 ** math.ceil(math.log(len(input_embed), 2)), 128)

        # Initial prefill tensors full of pad tokens
        input_prefill_i = torch.full((prefill_seq_len, input_embeds.shape[-1]), pad_embedding, dtype=input_embeds.dtype)
        input_prefill_i[: len(input_embed), :] = torch.tensor(input_embed).to(input_prefill_i)
        input_prefill.append(input_prefill_i)

        # Keep the correct decoding position of each user
        decoding_pos.append(len(input_embed))
        prefill_lens.append(prefill_seq_len)

    input_prefill = torch.stack(input_prefill)  # [batch_size, prefill_seq_len, embed_dim]

    return (
        input_prefill,
        decoding_pos,
        prefill_lens,
    )


def multimodal_rope_from_hf(
    inputs,
    input_embeds,
    reference_model,
    max_seq_len,
):
    # Unlike the reference model, we will precompute cos and sin for the entire sequence length including the generated tokens
    padded_inputs = torch.nn.functional.pad(inputs.input_ids, (0, max_seq_len - inputs.input_ids.shape[-1]))
    padded_attention_mask = torch.nn.functional.pad(
        inputs.attention_mask, (0, max_seq_len - inputs.attention_mask.shape[-1]), value=1.0
    )

    # Qwen2_5_VLForConditionalGeneration.forward:
    position_ids, rope_deltas = reference_model.get_rope_index(
        padded_inputs,
        inputs.image_grid_thw if "image_grid_thw" in inputs else None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=padded_attention_mask,
    )
    # Qwen2_5_VLModel.forward:
    cos, sin = reference_model.model.rotary_emb(input_embeds, position_ids)
    # apply_multimodal_rotary_pos_emb:
    mrope_section = reference_model.config.rope_scaling["mrope_section"] * 2
    unsqueeze_dim = 1
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    # convert to meta-style interleaved format:
    cos, sin = convert_rope_style_hf_to_meta(cos, sin)
    # we have precomputed embeddings for the entire sequence length and converted to 1D so we no longer need to track rope_deltas
    return cos, sin


def check_tensor(ttnn_tensor, name, mesh_device):
    our = torch.Tensor(ttnn.to_torch(ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)))
    if not os.path.exists(f"{name}.pt"):
        torch.save(our, f"{name}.pt")
        return

    ref = torch.load(f"{name}.pt")
    if not torch.allclose(our, ref):
        logger.error(f"Tensor {name} mismatch")
        breakpoint()
    else:
        logger.info(f"Tensor match: {name}")


# [INFO]: copied from tt_transformers/tt/common.py only the functions that Qwen-VL uses
class PagedAttentionConfig:
    def __init__(self, block_size=32, max_num_blocks=1024):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks


def nearest_pow_2(x):
    return 2 ** math.ceil(math.log2(x))


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


def num_blocks_in_seq(seq_len, block_size):
    return math.ceil(seq_len / block_size)


def get_block_size(kv_cache):
    return kv_cache[0][0].shape[2]


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
        if pt_out.dim() == 1:  # if sampling a single token re-add the batch dim to the tensor
            pt_out = pt_out.unsqueeze(0)
        return None, pt_out
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


def nearest_multiple(x, multiple_of):
    return math.ceil(x / multiple_of) * multiple_of
