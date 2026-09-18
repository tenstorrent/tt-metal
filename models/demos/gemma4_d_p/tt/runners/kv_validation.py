# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-less golden checks for Gemma4's exported migration table."""

from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.gemma4_d_p.tt.attention.global_kv_cache import pack_global_kv_reference, sliding_kv_indices
from models.demos.gemma4_d_p.tt.runners.adapter import Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import CONFIG_NAMES


def golden_cache_heads(key, value, layer, real_len):
    global_layer = layer % 6 == 5
    heads, width = (4, 512) if global_layer else (16, 256)
    for tensor in (key, value):
        if tensor.ndim != 4 or tensor.shape[:2] != (1, heads) or tensor.shape[2] < real_len or tensor.shape[3] != width:
            raise ValueError(f"Layer {layer}: expected golden [1, {heads}, >= {real_len}, {width}], got {tensor.shape}")
    key, value = key[0, :, :real_len].float(), value[0, :, :real_len].float()
    if global_layer:
        packed = pack_global_kv_reference(key, value)
        return {head: packed[head] for head in range(4)}
    key = key.index_select(-1, sliding_kv_indices())
    return {**{4 + head: key[head] for head in range(16)}, **{20 + head: value[head] for head in range(16)}}


def cache_pcc(expected, actual):
    if expected.shape != actual.shape or not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError("KV comparison requires matching shapes and finite values")
    expected, actual = expected.float().flatten(), actual.float().flatten()
    if torch.equal(expected, actual):
        return 1.0
    if expected.std() == 0 or actual.std() == 0:
        return 0.0
    pcc = float(torch.corrcoef(torch.stack((expected, actual)))[0, 1])
    if not torch.isfinite(torch.tensor(pcc)):
        raise ValueError("KV correlation is not finite")
    return pcc


def read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir):
    from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk, _resolve_unique_id

    if tuple(table.config_name(i) for i in range(table.num_configs())) != CONFIG_NAMES:
        raise ValueError("Gemma4 requires its 36 named cache configurations")
    if not 0 < real_len <= table.config(0).max_sequence_length:
        raise ValueError("Golden verification length must fit the cache")
    minima = {"global_k_rotary": 1.0, "global_v": 1.0, "sliding_k": 1.0, "sliding_v": 1.0}
    for layer in range(Gemma4ServiceConfig.NUM_LAYERS):
        with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as golden:
            expected_heads = golden_cache_heads(
                golden.get_tensor(f"key_cache_layer_{layer}"),
                golden.get_tensor(f"value_cache_layer_{layer}"),
                layer,
                real_len,
            )
        for config_id, expected in expected_heads.items():
            config = table.config(config_id)
            if config.chunk_n_tokens != 32 or config.num_layers != Gemma4ServiceConfig.NUM_LAYERS:
                raise ValueError("Gemma4 migration requires 32-token chunks and all model layers")
            rows = []
            for position in range(0, real_len, config.chunk_n_tokens):
                location = table.lookup(layer, position, slot_id, config_id)
                nodes = table.get_device_group(location.device_group_index).fabric_node_ids
                unique_id = _resolve_unique_id(nodes, device_map)
                if location.size_bytes != config.chunk_size_bytes:
                    raise ValueError(f"Missing or invalid KV chunk: layer={layer} config={config_id} pos={position}")
                raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, location.noc_addr, location.size_bytes)
                rows.append(_decode_bfp8_chunk(bytes(raw), expected.shape[-1]))
            actual = torch.cat(rows)[:real_len]
            if config_id < 4:
                scores = {
                    "global_k_rotary": cache_pcc(expected[:, :128], actual[:, :128]),
                    "global_v": cache_pcc(expected[:, 128:], actual[:, 128:]),
                }
            else:
                scores = {"sliding_k" if config_id < 20 else "sliding_v": cache_pcc(expected, actual)}
            for name, score in scores.items():
                minima[name] = min(minima[name], score)
        logger.info(f"[Gemma4 KV PCC] slot={slot_id} layer={layer} minima={minima}")
    return minima
