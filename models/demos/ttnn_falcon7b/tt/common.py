# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
import ttnn
import torch
from pathlib import Path
from loguru import logger
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, AttentionMaskConverter


def strip_state_dict_prefix(state_dict, prefix):
    return {k[len(prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def from_torch_cached(
    filename,
    torch_tensor,
    device=None,
    dtype=None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    layout=ttnn.TILE_LAYOUT,
    unsqueeze_to_4d=False,
):
    filename = f"{filename}_{dtype.name}.bin"
    try:
        tensor = ttnn.load_tensor(filename)
        if tuple(tensor.shape) != tuple(torch_tensor.shape):
            logger.warning(
                f"Cached file {filename} has shape {tensor.shape}, expected {torch_tensor.shape}, regenerating cache"
            )
            raise RuntimeError
        logger.info(f"Loaded cache for {filename} of shape {tensor.shape}")
    except (FileNotFoundError, RuntimeError):
        tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)
        logger.info(f"Generating cache for {filename} of shape {tensor.shape}")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        ttnn.dump_tensor(filename, tensor)
    if unsqueeze_to_4d:
        tensor = ttnn.unsqueeze_to_4D(tensor)
    tensor = ttnn.to_device(tensor, device, memory_config=memory_config)
    return tensor


def create_custom_preprocessor(model_config, tt_cache_path, device, base_file_name=None):
    def rotary_embedding_custom_processor(torch_model, name):
        parameters = {}
        if base_file_name:
            base_file_path = f"{tt_cache_path}/{base_file_name}.{name}"
        else:
            base_file_path = f"{tt_cache_path}/{name}"

        if isinstance(torch_model, transformers.models.falcon.modeling_falcon.FalconRotaryEmbedding):
            parameters["cos_cached"] = from_torch_cached(
                f"{base_file_path}.cos_cached",
                torch_model.cos_cached,
                device=device,
                dtype=model_config["COS_CACHED_WEIGHTS_DTYPE"],
                unsqueeze_to_4d=True,
            )
            parameters["sin_cached"] = from_torch_cached(
                f"{base_file_path}.sin_cached",
                torch_model.sin_cached,
                device=device,
                dtype=model_config["SIN_CACHED_WEIGHTS_DTYPE"],
                unsqueeze_to_4d=True,
            )
        elif isinstance(torch_model, torch.nn.Linear):
            linear_weight_file_name = f"{base_file_path}.weight"
            parameters["weight"] = from_torch_cached(
                linear_weight_file_name, torch_model.weight.T.contiguous(), device=device, dtype=ttnn.bfloat8_b
            )
            if torch_model.bias is not None:
                linear_bias_file_name = f"{base_file_path}.bias"
                parameters["bias"] = from_torch_cached(
                    linear_bias_file_name, torch_model.bias.reshape((1, -1)).contiguous(), device=device
                )
        elif isinstance(torch_model, torch.nn.LayerNorm):
            parameters["weight"] = from_torch_cached(
                f"{base_file_path}.weight", torch_model.weight.reshape((1, -1)), device=device, dtype=ttnn.bfloat16
            )
            parameters["bias"] = from_torch_cached(
                f"{base_file_path}.bias", torch_model.bias.reshape((1, -1)), device=device, dtype=ttnn.bfloat16
            )

        return parameters

    return rotary_embedding_custom_processor


def create_attention_input(llm_mode, dtype, batch, sequence_length, hidden_size, device):
    torch_attention_input = (torch.rand(batch, sequence_length, hidden_size) * 2) - 1

    q_len = sequence_length
    if llm_mode == "prefill":
        tt_attention_input = ttnn.from_torch(
            torch_attention_input.unsqueeze(1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
    elif llm_mode == "decode":
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        tt_attention_input = ttnn.from_torch(
            torch_attention_input.unsqueeze(1).transpose(0, 2),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")
    return torch_attention_input, tt_attention_input


def create_attention_mask(
    llm_mode, dtype, attention_input, batch, sequence_length, num_attention_heads, kv_cache_length, device
):
    if llm_mode == "prefill":
        q_len, kv_len = sequence_length, sequence_length
        assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_length == 0, "For prefill, no kv_cache is passed in!"

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        attention_mask = attn_mask_converter.to_causal_4d(
            batch,
            sequence_length,
            sequence_length,
            dtype=torch.bfloat16,
        )

        tt_attention_mask = ttnn.from_torch(
            attention_mask.expand(-1, num_attention_heads, -1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    elif llm_mode == "decode":
        q_len, kv_len = sequence_length, kv_cache_length + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        kv_len_padded = (kv_len + 31) // 32 * 32

        attention_mask = torch.zeros(batch, 1, sequence_length, kv_len, dtype=bool)
        attention_mask_padded = (
            torch.cat(
                (
                    attention_mask,
                    torch.ones(batch, 1, sequence_length, kv_len_padded - kv_len, dtype=bool),
                ),
                dim=-1,
            ).to(torch.float32)
            * -1e3
        )

        tt_attention_mask = ttnn.from_torch(
            (attention_mask_padded.transpose(0, 2)).expand(-1, num_attention_heads, -1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    return attention_mask, tt_attention_mask


def create_kv_cache(llm_mode, dtype, batch, kv_cache_length, config, device):
    head_dim = config.hidden_size // config.num_attention_heads

    # Pre-initialize KV-cache for prefill mode
    torch_k_cache = torch.zeros(batch, 1, config.max_position_embeddings, head_dim)
    torch_v_cache = torch.zeros(batch, 1, config.max_position_embeddings, head_dim)

    if llm_mode == "prefill":
        layer_past = None
        ttnn_k_cache = ttnn.from_torch(torch_k_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        ttnn_v_cache = ttnn.from_torch(torch_v_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    elif llm_mode == "decode":
        k_cache_data = torch.rand(batch, 1, kv_cache_length, head_dim)
        v_cache_data = torch.rand(batch, 1, kv_cache_length, head_dim)
        layer_past = (k_cache_data, v_cache_data)

        torch_k_cache[:, :, :kv_cache_length, :] = k_cache_data
        torch_v_cache[:, :, :kv_cache_length, :] = v_cache_data
        ttnn_k_cache = ttnn.from_torch(torch_k_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        ttnn_v_cache = ttnn.from_torch(torch_v_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    tt_layer_past = (ttnn_k_cache, ttnn_v_cache)
    return layer_past, tt_layer_past


def create_position_ids(llm_mode, kv_cache_length):
    if llm_mode == "prefill":
        position_ids = None
    elif llm_mode == "decode":
        position_ids = torch.LongTensor([kv_cache_length])
    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")
    return position_ids
