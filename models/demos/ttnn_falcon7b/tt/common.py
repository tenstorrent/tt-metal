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


def create_custom_preprocessor(model_config, tt_cache_path, device, base_file_name=None, weights_mesh_mapper=None):
    def rotary_embedding_custom_processor(torch_model, name):
        parameters = {}
        if base_file_name:
            base_file_path = f"{tt_cache_path}/{base_file_name}.{name}"
        else:
            base_file_path = f"{tt_cache_path}/{name}"

        if isinstance(torch_model, transformers.models.falcon.modeling_falcon.FalconRotaryEmbedding):
            parameters["cos_cached"] = ttnn.as_tensor(
                torch_model.cos_cached,
                dtype=model_config["COS_CACHED_WEIGHTS_DTYPE"],
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=model_config["DEFAULT_MEMCFG"],
                cache_file_name=f"{base_file_path}.cos_cached",
                preprocess=lambda tensor: tensor.unsqueeze(0).unsqueeze(0),
                mesh_mapper=weights_mesh_mapper,
            )
            parameters["sin_cached"] = ttnn.as_tensor(
                torch_model.sin_cached,
                dtype=model_config["SIN_CACHED_WEIGHTS_DTYPE"],
                device=device,
                memory_config=model_config["DEFAULT_MEMCFG"],
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=f"{base_file_path}.sin_cached",
                preprocess=lambda tensor: tensor.unsqueeze(0).unsqueeze(0),
                mesh_mapper=weights_mesh_mapper,
            )
        elif isinstance(torch_model, torch.nn.Linear):
            parameters["weight"] = ttnn.as_tensor(
                torch_model.weight,
                dtype=ttnn.bfloat8_b,
                device=device,
                memory_config=model_config["DEFAULT_MEMCFG"],
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=f"{base_file_path}.weight",
                preprocess=lambda tensor: tensor.T.contiguous(),
                mesh_mapper=weights_mesh_mapper,
            )
            if torch_model.bias is not None:
                parameters["bias"] = ttnn.as_tensor(
                    torch_model.bias,
                    dtype=ttnn.bfloat8_b,
                    device=device,
                    memory_config=model_config["DEFAULT_MEMCFG"],
                    layout=ttnn.TILE_LAYOUT,
                    cache_file_name=f"{base_file_path}.bias",
                    preprocess=lambda tensor: tensor.reshape((1, -1)),
                    mesh_mapper=weights_mesh_mapper,
                )
        elif isinstance(torch_model, torch.nn.LayerNorm):
            parameters["weight"] = ttnn.as_tensor(
                torch_model.weight,
                dtype=ttnn.bfloat16,
                device=device,
                memory_config=model_config["DEFAULT_MEMCFG"],
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=f"{base_file_path}.weight",
                preprocess=lambda tensor: tensor.reshape((1, -1)),
                mesh_mapper=weights_mesh_mapper,
            )
            parameters["bias"] = ttnn.as_tensor(
                torch_model.bias,
                dtype=ttnn.bfloat16,
                device=device,
                memory_config=model_config["DEFAULT_MEMCFG"],
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=f"{base_file_path}.bias",
                preprocess=lambda tensor: tensor.reshape((1, -1)),
                mesh_mapper=weights_mesh_mapper,
            )

        return parameters

    return rotary_embedding_custom_processor


def create_attention_input(llm_mode, dtype, batch, sequence_length, hidden_size, device, mesh_mapper=None):
    torch_attention_input = (torch.rand(batch, sequence_length, hidden_size) * 2) - 1

    q_len = sequence_length
    if llm_mode == "prefill":
        tt_attention_input = ttnn.from_torch(
            torch_attention_input.unsqueeze(1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        )
    elif llm_mode == "decode":
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        tt_attention_input = ttnn.from_torch(
            torch_attention_input.unsqueeze(1).transpose(0, 2),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        )

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")
    return torch_attention_input, tt_attention_input


def create_attention_mask(
    llm_mode,
    dtype,
    attention_input,
    batch,
    sequence_length,
    num_attention_heads,
    kv_cache_length,
    device,
    mesh_mapper=None,
):
    if llm_mode == "prefill":
        q_len, kv_len = sequence_length, sequence_length
        assert batch == 1 or mesh_mapper, "For prefill, batch must be 1!"
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
            mesh_mapper=mesh_mapper,
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
            mesh_mapper=mesh_mapper,
        )
    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    return attention_mask, tt_attention_mask


def create_kv_cache(llm_mode, dtype, batch, kv_cache_length, config, device, mesh_mapper=None):
    head_dim = config.hidden_size // config.num_attention_heads

    # Pre-initialize KV-cache for prefill mode
    torch_k_cache = torch.zeros(batch, 1, config.max_position_embeddings, head_dim)
    torch_v_cache = torch.zeros(batch, 1, config.max_position_embeddings, head_dim)

    if llm_mode == "prefill":
        layer_past = None
        ttnn_k_cache = ttnn.from_torch(
            torch_k_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=mesh_mapper
        )
        ttnn_v_cache = ttnn.from_torch(
            torch_v_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=mesh_mapper
        )
    elif llm_mode == "decode":
        k_cache_data = torch.rand(batch, 1, kv_cache_length, head_dim)
        v_cache_data = torch.rand(batch, 1, kv_cache_length, head_dim)
        layer_past = (k_cache_data, v_cache_data)

        torch_k_cache[:, :, :kv_cache_length, :] = k_cache_data
        torch_v_cache[:, :, :kv_cache_length, :] = v_cache_data
        ttnn_k_cache = ttnn.from_torch(
            torch_k_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=mesh_mapper
        )
        ttnn_v_cache = ttnn.from_torch(
            torch_v_cache, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=mesh_mapper
        )
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
