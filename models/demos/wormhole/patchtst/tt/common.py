# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn

MEMORY_TIER = Literal["dram", "l1"]

TT_DTYPE = ttnn.bfloat16
DEFAULT_TRACE_REGION_SIZE = 5_000_000
DEFAULT_L1_SMALL_SIZE = 24_576
DEFAULT_NUM_COMMAND_QUEUES = 2
HIGH_CHANNEL_DRAM_THRESHOLD = 100
_L1_DEBUG_DEVICE_INFO = ttnn._ttnn.reports.get_device_info

MEMORY_CONFIG_BY_TIER: dict[MEMORY_TIER, ttnn.MemoryConfig] = {
    "l1": ttnn.L1_MEMORY_CONFIG,
    "dram": ttnn.DRAM_MEMORY_CONFIG,
}


@dataclass
class TTLinear:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None

    def release(self) -> None:
        ttnn.deallocate(self.weight)
        if self.bias is not None:
            ttnn.deallocate(self.bias)


@dataclass(frozen=True)
class PatchTSTRuntimePolicy:
    activation_memory_tier: MEMORY_TIER = "dram"
    weight_memory_tier: MEMORY_TIER = "dram"
    use_sdpa: bool = True
    use_sharded_attention_inputs: bool = False
    allow_shard_fallback: bool = False
    sharding_max_rows_per_chunk: int = 4096
    use_fused_ffn: bool = True
    use_device_patching: bool = False
    sdpa_q_chunk_size: int = 128
    sdpa_k_chunk_size: int = 128


def resolve_runtime_policy_for_workload(
    policy: PatchTSTRuntimePolicy | None,
    dataset_num_channels: int,
) -> PatchTSTRuntimePolicy:
    effective_policy = policy or PatchTSTRuntimePolicy()
    if policy is not None:
        return policy

    if dataset_num_channels >= HIGH_CHANNEL_DRAM_THRESHOLD and effective_policy.activation_memory_tier == "l1":
        # Traffic-scale channel counts overflow the default L1 QKV projection path, so the implicit default
        # policy must move activations to DRAM to keep the high-channel workload runnable without weakening math.
        return replace(effective_policy, activation_memory_tier="dram")
    return effective_policy


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def _bf16_tile_padded_bytes(rows: int, width: int) -> int:
    # PatchTST activation tensors in this demo are bfloat16 tile-layout tensors.
    # TT tile storage rounds each logical shard to whole 32x32 tiles, so the packed shard bytes are:
    #   roundup(rows, 32) * roundup(width, 32) * sizeof(bfloat16)
    # This matches the physical volume the allocator must back for the shard buffer.
    padded_rows = _round_up(int(rows), ttnn.TILE_SIZE)
    padded_width = _round_up(int(width), ttnn.TILE_SIZE)
    return padded_rows * padded_width * 2


def ensure_interleaved(tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    if not tensor.is_sharded():
        return tensor
    interleaved = ttnn.sharded_to_interleaved(tensor, memory_config=memory_config)
    ttnn.deallocate(tensor)
    return interleaved


def to_tt_param(
    tensor: torch.Tensor,
    *,
    device,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    param = tensor.detach().to(torch.float32)
    if param.ndim == 1:
        param = param.reshape(1, 1, -1)
    elif param.ndim == 2:
        param = param.reshape(1, *param.shape)
    return ttnn.from_torch(
        param,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def build_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    device,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> TTLinear:
    weight_tt = ttnn.to_device(preprocess_linear_weight(weight, dtype=dtype), device, memory_config=memory_config)
    bias_tt = None
    if bias is not None:
        bias_tt = ttnn.to_device(preprocess_linear_bias(bias, dtype=dtype), device, memory_config=memory_config)
    return TTLinear(weight_tt, bias_tt)


def build_linear_from_state(
    state: dict[str, torch.Tensor],
    prefix: str,
    *,
    device,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> TTLinear:
    bias_key = f"{prefix}.bias"
    return build_linear(
        state[f"{prefix}.weight"],
        state[bias_key] if bias_key in state else None,
        device=device,
        dtype=dtype,
        memory_config=memory_config,
    )


def maybe_height_shard_3d_tensor(
    tensor: ttnn.Tensor,
    device,
    enable: bool,
    allow_fallback: bool = True,
) -> tuple[ttnn.Tensor, bool]:
    if not enable or len(tensor.shape) != 3:
        return tensor, False

    batch_size = int(tensor.shape[0])
    sequence_length = int(tensor.shape[1])
    hidden_size = int(tensor.shape[2])
    padded_batch_size = int(tensor.padded_shape[0])
    padded_sequence_length = int(tensor.padded_shape[1])

    if hidden_size % ttnn.TILE_SIZE != 0:
        return tensor, False
    if batch_size * sequence_length < ttnn.TILE_SIZE:
        return tensor, False
    if (batch_size * sequence_length) % ttnn.TILE_SIZE != 0:
        return tensor, False

    rows = padded_batch_size * padded_sequence_length
    compute_grid = device.compute_with_storage_grid_size()
    total_compute_cores = int(compute_grid.x) * int(compute_grid.y)
    rows_per_core = math.ceil(rows / total_compute_cores)
    padded_rows_per_core = _round_up(rows_per_core, ttnn.TILE_SIZE)
    input_shard_bytes = _bf16_tile_padded_bytes(padded_rows_per_core, hidden_size)
    qkv_shard_bytes = _bf16_tile_padded_bytes(padded_rows_per_core, hidden_size * 3)

    # Height-sharded attention-input compute is only safe while the per-core working set stays below one L1 bank.
    # PatchTST does not use TTNN's explicit DRAM prefetcher today, and TTNN sharded MemoryConfig objects are L1-only.
    # That means there is a hard binary choice:
    #   - small enough: shard into L1 across compute cores
    #   - too large: keep the tensor interleaved so kernels read from DRAM/L1 through the normal path
    #
    # We use a strict no-exception boundary instead of probing until OOM:
    #   estimated_working_set = 2 * input_shard + 2 * qkv_shard
    #
    # Reasoning:
    #   - the QKV linear needs an input shard and an output shard resident,
    #   - TT kernels commonly double-buffer their circular buffers,
    #   - qkv_shard is the largest logical activation in this path because width expands from D to 3D.
    #
    # On this Wormhole device the allocator reports:
    #   worker_l1_size  = 1,499,136 B
    #   cb_limit        = 1,395,424 B
    #   l1_bank_size    = 1,370,848 B
    #
    # The practical hard line for a single sharded buffer is the smaller of cb_limit and bank_size.
    # We saw this explicitly in a failing experiment:
    #   requested_per_bank = 1,409,024 B > bank_size = 1,370,848 B
    # so the allocator failed before execution could recover.
    info = _L1_DEBUG_DEVICE_INFO(device)
    l1_effective_limit_bytes = min(int(info.l1_bank_size), int(info.cb_limit))
    estimated_working_set_bytes = 2 * input_shard_bytes + 2 * qkv_shard_bytes
    if estimated_working_set_bytes > l1_effective_limit_bytes:
        if allow_fallback:
            return tensor, False
        raise RuntimeError(
            "Height-sharded attention input exceeds the strict per-core L1 budget and shard fallback is disabled. "
            f"rows={rows}, rows_per_core={rows_per_core}, padded_rows_per_core={padded_rows_per_core}, "
            f"width={hidden_size}, input_shard_bytes={input_shard_bytes}, qkv_shard_bytes={qkv_shard_bytes}, "
            f"estimated_working_set_bytes={estimated_working_set_bytes}, "
            f"l1_effective_limit_bytes={l1_effective_limit_bytes}"
        )

    core_coord = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(
        total_compute_cores,
        core_coord,
        row_wise=True,
    )
    sharded_mem_cfg = ttnn.create_sharded_memory_config_(
        shape=(padded_rows_per_core, hidden_size),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
        tile_layout=True,
    )
    # Interleaved [B, S, D] tile tensors carry independent padding on the first two logical axes.
    # If we shard that rank-3 view directly, TTNN can count shards off the padded sequence axis alone and
    # reject otherwise valid shapes. Flattening to [1, B*S, D] first makes the sharding math operate on the
    # true padded token rows, then we reshape the sharded tensor back to [B, S, D] on the same full core grid.
    flattened = ttnn.reshape(tensor, (1, batch_size * sequence_length, hidden_size))
    sharded_flattened = ttnn.interleaved_to_sharded(flattened, sharded_mem_cfg)
    reshaped = ttnn.reshape(sharded_flattened, (batch_size, sequence_length, hidden_size))
    if flattened is not tensor:
        ttnn.deallocate(flattened)
    if reshaped is not sharded_flattened:
        ttnn.deallocate(sharded_flattened)
    return reshaped, True


def _aligned_batch_for_height_shard(batch_size: int, sequence_length: int) -> int:
    if sequence_length <= 0:
        return batch_size
    if (batch_size * sequence_length) % ttnn.TILE_SIZE == 0:
        return batch_size
    step = ttnn.TILE_SIZE // math.gcd(sequence_length, ttnn.TILE_SIZE)
    return ((batch_size + step - 1) // step) * step


def sharded_layout_roundtrip_3d_tensor(
    tensor: ttnn.Tensor,
    device,
    enable: bool,
    max_rows_per_chunk: int = 4096,
    interleaved_memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
    allow_fallback: bool = True,
) -> tuple[ttnn.Tensor, int]:
    if not enable or len(tensor.shape) != 3:
        return tensor, 0

    batch_size = int(tensor.shape[0])
    sequence_length = int(tensor.shape[1])
    hidden_size = int(tensor.shape[2])
    if batch_size <= 0 or sequence_length <= 0:
        return tensor, 0

    rows = batch_size * sequence_length
    if rows <= max_rows_per_chunk:
        aligned_batch = _aligned_batch_for_height_shard(batch_size=batch_size, sequence_length=sequence_length)
        padded = tensor
        did_pad = False
        if aligned_batch > batch_size:
            padded = ttnn.pad(
                tensor,
                padding=((0, aligned_batch - batch_size), (0, 0), (0, 0)),
                value=0.0,
            )
            did_pad = True

        sharded, used_shard = maybe_height_shard_3d_tensor(
            padded,
            device=device,
            enable=True,
            allow_fallback=allow_fallback,
        )
        if not used_shard:
            if did_pad:
                ttnn.deallocate(padded)
            return tensor, 0

        interleaved = ttnn.sharded_to_interleaved(sharded, memory_config=interleaved_memory_config)
        ttnn.deallocate(sharded)
        if did_pad:
            ttnn.deallocate(padded)
            trimmed = ttnn.slice(interleaved, (0, 0, 0), (batch_size, sequence_length, hidden_size))
            ttnn.deallocate(interleaved)
            interleaved = trimmed
        return interleaved, 1

    chunk_batch = max(1, max_rows_per_chunk // sequence_length)
    pieces: list[ttnn.Tensor] = []
    shard_ops = 0
    for start in range(0, batch_size, chunk_batch):
        end = min(batch_size, start + chunk_batch)
        chunk = ttnn.slice(tensor, (start, 0, 0), (end, sequence_length, hidden_size))
        roundtripped, ops = sharded_layout_roundtrip_3d_tensor(
            chunk,
            device=device,
            enable=True,
            max_rows_per_chunk=max_rows_per_chunk,
            interleaved_memory_config=interleaved_memory_config,
            allow_fallback=allow_fallback,
        )
        shard_ops += ops
        pieces.append(roundtripped)
        if roundtripped is not chunk:
            ttnn.deallocate(chunk)

    if len(pieces) == 1:
        return pieces[0], shard_ops

    merged = pieces[0]
    for part in pieces[1:]:
        combined = ttnn.concat([merged, part], dim=0)
        ttnn.deallocate(merged)
        ttnn.deallocate(part)
        merged = combined
    return merged, shard_ops
