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

PREPARED_GPU_TRACE_LAYOUT = "gemma4_kv_heads_v1"


def load_prepared_gpu_cache_heads(path, layer, real_len):
    configs = range(4) if layer % 6 == 5 else range(4, 36)
    width = 640 if layer % 6 == 5 else 256
    with safe_open(str(path), framework="pt") as tensors:
        if (tensors.metadata() or {}).get("layout") != PREPARED_GPU_TRACE_LAYOUT:
            raise ValueError(f"Layer {layer}: invalid prepared GPU layout in {path}")
        if set(tensors.keys()) != {CONFIG_NAMES[config] for config in configs}:
            raise ValueError(f"Layer {layer}: invalid prepared GPU head names in {path}")
        result = {}
        for config in configs:
            rows = tensors.get_slice(CONFIG_NAMES[config])
            shape = rows.get_shape()
            if len(shape) != 2 or shape[1] != width or not 0 < real_len <= shape[0]:
                raise ValueError(f"Layer {layer}: invalid prepared GPU KV shape {shape} for {real_len} tokens")
            if rows.get_dtype() != "BF16":
                raise ValueError(f"Layer {layer}: prepared GPU KV must be BF16")
            result[config] = rows[:real_len].float()
    return result


def load_gpu_cache_heads(trace_dir, layer, real_len):
    prepared = Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"
    if prepared.is_file():
        return load_prepared_gpu_cache_heads(prepared, layer, real_len)
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
    """Compute whole-head PCC with a bounded FP32 workspace."""
    if expected.shape != actual.shape or expected.numel() == 0:
        raise ValueError("KV comparison requires matching nonempty shapes")
    expected = expected.reshape(expected.shape[0], -1)
    actual = actual.reshape(actual.shape[0], -1)
    rows_per_block = max(1, 1024 * 1024 // expected.shape[1])
    buffer = torch.empty((2, min(rows_per_block, expected.shape[0]), expected.shape[1]), dtype=torch.float32)
    count = 0
    mean = torch.zeros(2, dtype=torch.float32)
    covariance = torch.zeros(2, 2, dtype=torch.float32)
    identical = True
    for start in range(0, expected.shape[0], rows_per_block):
        end = min(start + rows_per_block, expected.shape[0])
        values = buffer[:, : end - start]
        values[0].copy_(expected[start:end])
        values[1].copy_(actual[start:end])
        values = values.reshape(2, -1)
        if not torch.isfinite(values).all():
            raise ValueError("KV comparison requires finite values")
        identical = identical and torch.equal(values[0], values[1])
        block_count = values.shape[1]
        block_mean = values.mean(dim=1, keepdim=True)
        values.sub_(block_mean)
        delta = block_mean.flatten() - mean
        total_count = count + block_count
        covariance += values @ values.T + torch.outer(delta, delta) * (count * block_count / total_count)
        mean += delta * (block_count / total_count)
        count = total_count
    if identical:
        return 1.0
    scale = covariance[0, 0].sqrt() * covariance[1, 1].sqrt()
    if scale == 0:
        return 0.0
    pcc = covariance[0, 1] / scale
    if not torch.isfinite(pcc):
        raise ValueError("KV correlation is not finite")
    return float(pcc.clamp(-1, 1))


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
    local_heads, width = shards[0].shape[1], shards[0].shape[3]
    chunks = real_len // Gemma4ServiceConfig.CHUNK_SIZE
    gathered = torch.empty((local_heads * tp, chunks, cp, local_chunk, width), dtype=shards[0].dtype)
    for row in range(cp):
        for column in range(tp):
            shard = shards[row * tp + column].reshape(local_heads, chunks, local_chunk, width)
            gathered[column * local_heads : (column + 1) * local_heads, :, row].copy_(shard)
    return gathered.reshape(local_heads * tp, real_len, width)


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
            torch.testing.assert_close(decoded, actual[position : position + 32].float(), rtol=0, atol=0)


def compare_slot_cache(read_heads, slot_id, real_len, trace_dir):
    started = time.perf_counter()
    timings = dict(reference_seconds=0.0, readback_seconds=0.0, comparison_seconds=0.0)
    minima = {"global_k_rotary": 1.0, "global_v": 1.0, "sliding_k": 1.0, "sliding_v": 1.0}
    measurements = []
    layer_timings = []
    for layer in range(Gemma4ServiceConfig.NUM_LAYERS):
        previous_timings = timings.copy()
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
        layer_seconds = {name: timings[name] - previous_timings[name] for name in previous_timings}
        layer_timings.append(dict(layer=layer, **layer_seconds))
        logger.info(
            f"[Gemma4 KV PCC] slot={slot_id} layer={layer} layer_minima={layer_minima} "
            f"running_min_pcc={min(minima.values()):.8f} "
            f"reference={layer_seconds['reference_seconds']:.2f}s "
            f"readback={layer_seconds['readback_seconds']:.2f}s "
            f"pcc={layer_seconds['comparison_seconds']:.2f}s"
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
            layer_timings=layer_timings,
            measurements=measurements,
        )
        (directory / f"gemma4_slot{slot_id}.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info(f"[Gemma4 KV PCC] slot={slot_id} final_min_pcc={min(minima.values()):.8f} timings={timings}")
    return minima
