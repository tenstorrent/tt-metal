# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare exported Gemma4 KV against the row-sharded GPU trace."""

import json
import os
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.gemma4_d_p.tt.attention.global_kv_cache import pack_global_kv_reference, sliding_kv_indices
from models.demos.gemma4_d_p.tt.runners.adapter import Gemma4ServiceConfig
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import CONFIG_NAMES


def load_gpu_cache_heads(trace_dir, layer, real_len):
    heads, width = (4, 512) if layer % 6 == 5 else (16, 256)
    directory = Path(trace_dir) / "kv_cache" / f"layer_{layer}"
    shards = sorted(directory.glob("rows_*.safetensors"), key=lambda path: int(path.stem.split("_")[1]))
    parts, position = [], 0
    for shard in shards:
        start, end = map(int, shard.stem.split("_")[1:])
        if position >= real_len:
            break
        if start != position or end <= start:
            raise ValueError(f"Layer {layer}: noncontiguous GPU trace at {shard}")
        with safe_open(str(shard), framework="pt") as tensors:
            rows = tensors.get_slice(f"kv_post_transform_layer_{layer}")
            if rows.get_shape() != [end - start, 2 * heads * width]:
                raise ValueError(f"Layer {layer}: invalid GPU KV shape in {shard}: {rows.get_shape()}")
            parts.append(rows[: min(end, real_len) - start])
        position = min(end, real_len)
    if position != real_len:
        raise ValueError(f"Layer {layer}: GPU trace covers {position}/{real_len} tokens")
    key, value = torch.cat(parts).split(heads * width, dim=-1)
    key = key.reshape(real_len, heads, width).permute(1, 0, 2).unsqueeze(0)
    value = value.reshape(real_len, heads, width).permute(1, 0, 2).unsqueeze(0)
    return golden_cache_heads(key, value, layer, real_len)


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


def read_cache_head(table, device_map, layer, slot_id, config_id, real_len, width):
    from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk, _resolve_unique_id

    config = table.config(config_id)
    if config.chunk_n_tokens != 32 or config.num_layers != Gemma4ServiceConfig.NUM_LAYERS:
        raise ValueError("Gemma4 migration requires 32-token chunks and all model layers")
    if config.chunk_size_bytes != width // 32 * 1088:
        raise ValueError(f"Invalid BFP8 chunk size for config {config_id}")
    banks = {}
    for position in range(0, real_len, 32):
        location = table.lookup(layer, position, slot_id, config_id)
        if location.size_bytes != config.chunk_size_bytes:
            raise ValueError(f"Missing or invalid KV chunk: layer={layer} config={config_id} pos={position}")
        bank = (int(location.device_group_index), location.noc_addr >> 32)
        banks.setdefault(bank, []).append((location.noc_addr, position // 32))
    actual = torch.empty(((real_len + 31) // 32, 32, width))
    for (group, _), locations in banks.items():
        locations.sort()
        address = locations[0][0]
        if any(current != address + index * config.chunk_size_bytes for index, (current, _) in enumerate(locations)):
            raise ValueError(f"Noncontiguous Gemma4 KV bank: layer={layer} config={config_id}")
        nodes = table.get_device_group(ttnn.experimental.disaggregation.DeviceGroupIndex(group)).fabric_node_ids
        unique_id = _resolve_unique_id(nodes, device_map)
        count = len(locations)
        raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, address, count * config.chunk_size_bytes)
        # Each bank stores complete 32-token blocks consecutively.
        decoded = _decode_bfp8_chunk(bytes(raw), width * count).reshape(32, count, width).permute(1, 0, 2)
        actual[[index for _, index in locations]] = decoded
    return actual.reshape(-1, width)[:real_len]


def read_slot_kv_and_check_pcc(table, device_map, slot_id, real_len, trace_dir):
    if tuple(table.config_name(i) for i in range(table.num_configs())) != CONFIG_NAMES:
        raise ValueError("Gemma4 requires its 36 named cache configurations")
    if not 0 < real_len <= table.config(0).max_sequence_length:
        raise ValueError("Golden verification length must fit the cache")

    def read_heads(layer):
        configs = range(4) if layer % 6 == 5 else range(4, 36)
        for config_id in configs:
            width = 640 if config_id < 4 else 256
            yield config_id, read_cache_head(table, device_map, layer, slot_id, config_id, real_len, width)

    return compare_slot_cache(read_heads, slot_id, real_len, trace_dir)


def read_cache_tensor(tensor, slot_id, real_len):
    """Gather one slot's populated prefix and restore chunk-major CP order."""
    cp, tp = Gemma4ServiceConfig.MESH_SHAPE
    local_chunk = Gemma4ServiceConfig.CHUNK_SIZE // cp
    if real_len % Gemma4ServiceConfig.CHUNK_SIZE:
        raise ValueError("Command-queue validation requires complete prefill chunks")
    selected = ttnn.slice(
        tensor,
        (slot_id, 0, 0, 0),
        (slot_id + 1, tensor.shape[1], real_len // cp, tensor.shape[3]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        row_major = ttnn.untilize(selected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    finally:
        ttnn.deallocate(selected)
    try:
        host = ttnn.from_device(row_major, blocking=True)
    finally:
        ttnn.deallocate(row_major)
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(host)]
    gathered = torch.cat([torch.cat(shards[row * tp : (row + 1) * tp], dim=1) for row in range(cp)], dim=2)[0]
    heads, _, width = gathered.shape
    return (
        gathered.reshape(heads, cp, real_len // Gemma4ServiceConfig.CHUNK_SIZE, local_chunk, width)
        .permute(0, 2, 1, 3, 4)
        .reshape(heads, real_len, width)
        .float()
    )


def check_table_samples(table, device_map, layer, slot_id, config_id, actual):
    """Check each CP rank's first and last populated block against the table."""
    from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk, _resolve_unique_id

    chunk_size = Gemma4ServiceConfig.CHUNK_SIZE
    local_chunk = chunk_size // Gemma4ServiceConfig.MESH_SHAPE[0]
    last_chunk = actual.shape[0] - chunk_size
    for row in range(Gemma4ServiceConfig.MESH_SHAPE[0]):
        for position in (row * local_chunk, last_chunk + (row + 1) * local_chunk - 32):
            location = table.lookup(layer, position, slot_id, config_id)
            if location.size_bytes != actual.shape[1] // 32 * 1088:
                raise ValueError(f"Invalid table sample: layer={layer} config={config_id} pos={position}")
            nodes = table.get_device_group(location.device_group_index).fabric_node_ids
            unique_id = _resolve_unique_id(nodes, device_map)
            raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, location.noc_addr, location.size_bytes)
            decoded = _decode_bfp8_chunk(bytes(raw), actual.shape[1])
            torch.testing.assert_close(decoded, actual[position : position + 32], rtol=0, atol=0)


def compare_slot_cache(read_heads, slot_id, real_len, trace_dir):
    started = time.perf_counter()
    timings = dict(reference_seconds=0.0, readback_seconds=0.0, comparison_seconds=0.0)
    minima = {"global_k_rotary": 1.0, "global_v": 1.0, "sliding_k": 1.0, "sliding_v": 1.0}
    measurements = []
    for layer in range(Gemma4ServiceConfig.NUM_LAYERS):
        start = time.perf_counter()
        expected_heads = load_gpu_cache_heads(trace_dir, layer, real_len)
        timings["reference_seconds"] += time.perf_counter() - start
        layer_minima = {}
        start = time.perf_counter()
        for config_id, actual in read_heads(layer):
            timings["readback_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
            expected = expected_heads[config_id]
            if config_id < 4:
                scores = {
                    "global_k_rotary": cache_pcc(expected[:, :128], actual[:, :128]),
                    "global_v": cache_pcc(expected[:, 128:], actual[:, 128:]),
                }
            else:
                scores = {"sliding_k" if config_id < 20 else "sliding_v": cache_pcc(expected, actual)}
            for name, score in scores.items():
                minima[name] = min(minima[name], score)
                layer_minima[name] = min(layer_minima.get(name, 1.0), score)
            measurements.append(dict(layer=layer, config=CONFIG_NAMES[config_id], pcc=scores))
            timings["comparison_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
        logger.info(
            f"[Gemma4 KV PCC] slot={slot_id} layer={layer} layer_minima={layer_minima} "
            f"running_min_pcc={min(minima.values()):.8f}"
        )
    timings["total_seconds"] = time.perf_counter() - started
    if summary_dir := os.getenv("PREFILL_PCC_SUMMARY_DIR"):
        directory = Path(summary_dir)
        directory.mkdir(parents=True, exist_ok=True)
        result = dict(
            trace_dir=str(trace_dir),
            slot=slot_id,
            tokens=real_len,
            minima=minima,
            timings=timings,
            measurements=measurements,
        )
        (directory / f"gemma4_slot{slot_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info(f"[Gemma4 KV PCC] slot={slot_id} final_min_pcc={min(minima.values()):.8f} timings={timings}")
    return minima
