# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

import ttnn
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import _accuracy_summary, _tensor_summary
from models.demos.deepseek_v4_flash.real_traceable_decode_smoke import TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE

PAGED_SDPA_DECODE_TRACE_SMOKE_SCHEMA_VERSION = 1
DEFAULT_PAGED_SDPA_DECODE_TRACE_REGION_SIZE = 64 * 1024 * 1024
DEFAULT_PAGED_SDPA_DECODE_BATCH = 8
DEFAULT_PAGED_SDPA_DECODE_NUM_HEADS = 16
DEFAULT_PAGED_SDPA_DECODE_NUM_KV_HEADS = 4
DEFAULT_PAGED_SDPA_DECODE_SEQ_LEN = 128
DEFAULT_PAGED_SDPA_DECODE_HEAD_DIM = 128
DEFAULT_PAGED_SDPA_DECODE_BLOCK_SIZE = 64
DEFAULT_PAGED_SDPA_DECODE_GRID_SIZE = (8, 2)
DEFAULT_PAGED_SDPA_DECODE_POSITIONS = (31, 95)
DEFAULT_PAGED_SDPA_DECODE_Q_VALUES = (0.25, -0.5)
DEFAULT_PAGED_SDPA_DECODE_PCC = 0.99
DEFAULT_PAGED_SDPA_DECODE_RTOL = 8e-2
DEFAULT_PAGED_SDPA_DECODE_ATOL = 8e-1


@dataclass(frozen=True)
class PagedSdpaDecodeTraceInputs:
    q_by_step: tuple[torch.Tensor, ...]
    k: torch.Tensor
    v: torch.Tensor
    paged_k: torch.Tensor
    paged_v: torch.Tensor
    page_table: torch.Tensor
    permutation: torch.Tensor
    positions: tuple[int, ...]
    q_values: tuple[float, ...]


def run_paged_sdpa_decode_trace_smoke(
    *,
    device_id: int = 0,
    trace_region_size: int = DEFAULT_PAGED_SDPA_DECODE_TRACE_REGION_SIZE,
    batch: int = DEFAULT_PAGED_SDPA_DECODE_BATCH,
    num_heads: int = DEFAULT_PAGED_SDPA_DECODE_NUM_HEADS,
    num_kv_heads: int = DEFAULT_PAGED_SDPA_DECODE_NUM_KV_HEADS,
    seq_len: int = DEFAULT_PAGED_SDPA_DECODE_SEQ_LEN,
    head_dim: int = DEFAULT_PAGED_SDPA_DECODE_HEAD_DIM,
    block_size: int = DEFAULT_PAGED_SDPA_DECODE_BLOCK_SIZE,
    positions: tuple[int, ...] = DEFAULT_PAGED_SDPA_DECODE_POSITIONS,
    q_values: tuple[float, ...] = DEFAULT_PAGED_SDPA_DECODE_Q_VALUES,
    pcc: float = DEFAULT_PAGED_SDPA_DECODE_PCC,
    rtol: float = DEFAULT_PAGED_SDPA_DECODE_RTOL,
    atol: float = DEFAULT_PAGED_SDPA_DECODE_ATOL,
) -> dict[str, Any]:
    """Trace paged SDPA decode once and replay with a mutable device cur_pos_tensor.

    This is a focused attention-read proof for DeepSeek V4 Flash bring-up. It is
    intentionally not the full DeepSeek sparse/indexer path and does not claim
    production autoregressive decode.
    """

    visible_devices = _default_single_visible_device_on_galaxy()
    _validate_paged_sdpa_decode_args(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        block_size=block_size,
        positions=positions,
        q_values=q_values,
        pcc=pcc,
    )
    inputs = _build_paged_sdpa_decode_trace_inputs(
        batch=batch,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        block_size=block_size,
        positions=positions,
        q_values=q_values,
    )
    references = tuple(
        _paged_sdpa_decode_reference(
            q=inputs.q_by_step[step],
            k=inputs.k,
            v=inputs.v,
            cur_pos=int(position),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            k_chunk_size=block_size,
            scale=head_dim**-0.5,
        )
        for step, position in enumerate(inputs.positions)
    )

    device = None
    trace_id = None
    try:
        device = ttnn.open_device(
            device_id=int(device_id),
            num_command_queues=1,
            trace_region_size=int(trace_region_size),
        )
        tt_q = ttnn.as_tensor(
            inputs.q_by_step[0],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_paged_sdpa_q_memory_config(
                device=device,
                batch=batch,
                num_heads=num_heads,
                head_dim=head_dim,
            ),
        )
        tt_k = ttnn.as_tensor(
            inputs.paged_k,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_v = ttnn.as_tensor(
            inputs.paged_v,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_page_table = ttnn.Tensor(inputs.page_table, ttnn.int32).to(device)
        tt_cur_pos = ttnn.Tensor(
            torch.full((int(batch),), int(inputs.positions[0]), dtype=torch.int32),
            ttnn.int32,
        ).to(device)

        op_kwargs = {
            "page_table_tensor": tt_page_table,
            "cur_pos_tensor": tt_cur_pos,
            "scale": head_dim**-0.5,
            "program_config": ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=DEFAULT_PAGED_SDPA_DECODE_GRID_SIZE,
                q_chunk_size=_padded_num_heads(num_heads),
                k_chunk_size=int(block_size),
                exp_approx_mode=False,
            ),
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

        ttnn.transformer.paged_scaled_dot_product_attention_decode(tt_q, tt_k, tt_v, **op_kwargs)
        ttnn.synchronize_device(device)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        output = ttnn.transformer.paged_scaled_dot_product_attention_decode(tt_q, tt_k, tt_v, **op_kwargs)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)

        actuals: list[torch.Tensor] = []
        for step, position in enumerate(inputs.positions):
            _copy_q_to_device(inputs.q_by_step[step], tt_q)
            _copy_cur_pos_to_device(int(position), batch=batch, tt_cur_pos=tt_cur_pos)
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            actuals.append(ttnn.to_torch(output).float()[:, :, : int(num_heads), :].contiguous())

        ttnn.release_trace(device, trace_id)
        trace_id = None
    finally:
        if device is not None and trace_id is not None:
            ttnn.release_trace(device, trace_id)
        if device is not None:
            ttnn.close_device(device)

    accuracy_by_step = [
        {
            "step": step,
            "position": int(position),
            "accuracy": {
                "attention_output": _accuracy_summary(
                    references[step],
                    actuals[step],
                    pcc_threshold=pcc,
                    rtol=rtol,
                    atol=atol,
                )
            },
            "expected_mean": float(references[step].float().mean().item()),
            "actual_mean": float(actuals[step].float().mean().item()),
            "q_value_loaded_before_replay": float(inputs.q_values[step]),
            "cache_rows_read": _cache_rows_read(int(position)),
            "cache_pages_read": _cache_pages_read(
                int(position),
                block_size=block_size,
                page_table=inputs.page_table,
            ),
        }
        for step, position in enumerate(inputs.positions)
    ]
    output_delta = (actuals[-1].float() - actuals[0].float()).abs()
    output_difference = {
        "first_position": int(inputs.positions[0]),
        "last_position": int(inputs.positions[-1]),
        "max_abs": float(output_delta.max().item()),
        "mean_abs": float(output_delta.mean().item()),
        "passed": bool(float(output_delta.max().item()) > 1.0),
        "reason": "attention output changes when cur_pos_tensor expands the paged cache rows visible to SDPA",
    }
    passed = bool(
        output_difference["passed"] and all(step["accuracy"]["attention_output"]["passed"] for step in accuracy_by_step)
    )

    return {
        "schema_version": PAGED_SDPA_DECODE_TRACE_SMOKE_SCHEMA_VERSION,
        "mode": "ttnn-trace",
        "passed": passed,
        "proof": "paged_sdpa_decode_dynamic_cur_pos_trace_replay",
        "attention_read_api": TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE,
        "attention_read_api_kind": "paged_sdpa_decode",
        "one_trace_capture_replayed_across_positions": len(inputs.positions) > 1,
        "positions": [int(value) for value in inputs.positions],
        "positions_used": [int(value) for value in inputs.positions],
        "device_scope": {
            "device_id": int(device_id),
            "tt_visible_devices": visible_devices,
            "uses_single_device_from_galaxy_reservation": os.environ.get("IRD_NUM_PCIE_CHIPS") == "32",
        },
        "cache_rows_pages_read_per_step": [
            {
                "step": step,
                "position": int(position),
                "rows": _cache_rows_read(int(position)),
                "pages": _cache_pages_read(int(position), block_size=block_size, page_table=inputs.page_table),
            }
            for step, position in enumerate(inputs.positions)
        ],
        "dynamic_cache_write_position": {
            "status": "not_in_scope",
            "reason": "focused attention-read proof uses preseeded paged K/V caches",
        },
        "dynamic_rope_position": {
            "status": "not_in_scope",
            "reason": "focused attention-read proof does not apply RoPE; dynamic RoPE is covered by the traceable decode smoke",
        },
        "dynamic_cache_read_current_position": {
            "status": "proved",
            "dynamic": True,
            "api": TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE,
            "position_input": "mutable device cur_pos_tensor",
            "page_table_input": "device page_table_tensor",
            "page_table_dynamic": False,
            "inside_trace": True,
        },
        "cache_read": {
            "cache_layout": "[num_blocks, num_kv_heads, block_size, head_dim]",
            "q_layout": "[1, batch, num_heads, head_dim]",
            "cur_pos_tensor_shape": [int(batch)],
            "page_table_shape": [int(batch), int(seq_len) // int(block_size)],
            "page_table": inputs.page_table.tolist(),
            "permutation": [int(value) for value in inputs.permutation.tolist()],
            "sliding_window_size": None,
            "causal": True,
            "scale": head_dim**-0.5,
        },
        "trace_capture": {
            "attempted": True,
            "capture_passed": True,
            "execute_replay_attempted": True,
            "execute_replay_passed": True,
            "trace_region_size": int(trace_region_size),
            "capture_count": 1,
            "single_capture_replayed_across_positions": len(inputs.positions) > 1,
            "positions_used": [int(value) for value in inputs.positions],
            "q_input_dynamic": True,
            "cur_pos_tensor_dynamic": True,
            "page_table_dynamic": False,
            "cache_read_current_position_dynamic": True,
            "attention_read_api": TRACEABLE_DECODE_ATTENTION_READ_PAGED_SDPA_DECODE,
            "host_boundaries_inside_trace": [],
            "traced_operations": [
                "ttnn.transformer.paged_scaled_dot_product_attention_decode(q,k_cache,v_cache,page_table,cur_pos_tensor)"
            ],
        },
        "attention_path": {
            "simplified_dense_paged_attention_stepping_stone": True,
            "host_supplied_attention_output": False,
            "device_cache_read_contributes": True,
            "reads_different_cache_rows_across_replay": output_difference["passed"],
            "true_deepseek_sparse_indexer_semantics": False,
            "true_deepseek_attention_sink_semantics": False,
            "true_deepseek_kv_split": False,
            "production_autoregressive_decode": False,
            "remaining_integration_step": (
                "reshape the DeepSeek decode Q/K/V path to SDPA decode tensors and replace the fixed Python "
                "ttnn.slice cache-window read with this paged attention read API"
            ),
        },
        "host_boundaries_inside_trace": [],
        "host_boundaries_outside_trace": [
            "synthetic_paged_kv_cache_host_seed_to_device",
            "static_page_table_host_to_device",
            "q_input_host_to_device_before_capture_and_replay",
            "cur_pos_tensor_host_to_device_before_capture_and_replay",
            "trace_output_readback_after_replay",
        ],
        "inputs": {
            "q_by_step": [_tensor_summary(value) for value in inputs.q_by_step],
            "k_contiguous": _tensor_summary(inputs.k),
            "v_contiguous": _tensor_summary(inputs.v),
            "paged_k": _tensor_summary(inputs.paged_k),
            "paged_v": _tensor_summary(inputs.paged_v),
            "page_table": _tensor_summary(inputs.page_table),
        },
        "ttnn_by_step": [
            {
                "step": step,
                "position": int(position),
                "attention_output": _tensor_summary(actuals[step]),
            }
            for step, position in enumerate(inputs.positions)
        ],
        "reference_by_step": [
            {
                "step": step,
                "position": int(position),
                "attention_output": _tensor_summary(references[step]),
            }
            for step, position in enumerate(inputs.positions)
        ],
        "accuracy_by_step": accuracy_by_step,
        "accuracy": accuracy_by_step[0]["accuracy"],
        "output_difference": output_difference,
        "limitations": [
            "focused proof uses synthetic dense paged K/V tensors, not DeepSeek sparse/indexer cache selection",
            "focused proof uses one visible device from the Galaxy reservation",
            "dynamic cache write and dynamic RoPE are not part of this proof",
            "page table is device resident but static across replay; cur_pos_tensor is the dynamic read position",
            "the full DeepSeek traceable decode subpath still uses ttnn.slice Python bounds for cache-window reads",
        ],
    }


def _build_paged_sdpa_decode_trace_inputs(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    block_size: int,
    positions: tuple[int, ...],
    q_values: tuple[float, ...],
) -> PagedSdpaDecodeTraceInputs:
    q_by_step = tuple(
        torch.full((1, int(batch), int(num_heads), int(head_dim)), float(value), dtype=torch.bfloat16)
        for value in q_values
    )
    k = torch.zeros((int(batch), int(num_kv_heads), int(seq_len), int(head_dim)), dtype=torch.bfloat16)
    row_values = torch.arange(int(seq_len), dtype=torch.float32).reshape(1, 1, int(seq_len), 1)
    dim_offsets = torch.linspace(-8.0, 8.0, steps=int(head_dim), dtype=torch.float32).reshape(1, 1, 1, int(head_dim))
    v = (row_values + dim_offsets).expand(int(batch), int(num_kv_heads), int(seq_len), int(head_dim))
    v = v.contiguous().to(torch.bfloat16)
    blocks_per_seq = int(seq_len) // int(block_size)
    paged_k = _to_paged_cache(k, batch=batch, blocks_per_seq=blocks_per_seq, block_size=block_size)
    paged_v = _to_paged_cache(v, batch=batch, blocks_per_seq=blocks_per_seq, block_size=block_size)
    max_num_blocks = int(batch) * blocks_per_seq
    permutation = torch.tensor(list(reversed(range(max_num_blocks))), dtype=torch.long)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(int(batch), blocks_per_seq).to(torch.int32)
    return PagedSdpaDecodeTraceInputs(
        q_by_step=q_by_step,
        k=k,
        v=v,
        paged_k=paged_k[permutation].contiguous(),
        paged_v=paged_v[permutation].contiguous(),
        page_table=page_table,
        permutation=permutation,
        positions=tuple(int(value) for value in positions),
        q_values=tuple(float(value) for value in q_values),
    )


def _paged_sdpa_decode_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cur_pos: int,
    num_heads: int,
    num_kv_heads: int,
    k_chunk_size: int,
    scale: float,
) -> torch.Tensor:
    batch = int(q.shape[1])
    head_dim = int(q.shape[-1])
    padded_num_heads = _padded_num_heads(num_heads)
    padded_layer_len = _nearest_n(int(cur_pos) + 1, int(k_chunk_size))
    attn_mask = torch.zeros((batch, padded_num_heads, 1, padded_layer_len), dtype=torch.float32)
    attn_mask[:, :, :, int(cur_pos) + 1 :] = torch.finfo(torch.float32).min
    q_slice = q[:, :, : int(num_heads), :].float().permute(1, 2, 0, 3)
    k_slice = k[:, :, :padded_layer_len, :].float()
    v_slice = v[:, :, :padded_layer_len, :].float()
    if int(num_kv_heads) < int(num_heads):
        if int(num_heads) % int(num_kv_heads) != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        repeat = int(num_heads) // int(num_kv_heads)
        k_slice = torch.cat(
            [k_slice[:, head : head + 1, :, :].repeat(1, repeat, 1, 1) for head in range(num_kv_heads)],
            dim=1,
        )
        v_slice = torch.cat(
            [v_slice[:, head : head + 1, :, :].repeat(1, repeat, 1, 1) for head in range(num_kv_heads)],
            dim=1,
        )
    expected = F.scaled_dot_product_attention(
        q_slice,
        k_slice,
        v_slice,
        attn_mask=attn_mask[:, : int(num_heads), :, :],
        scale=float(scale),
        is_causal=False,
    )
    return expected.squeeze(2).unsqueeze(0).reshape(1, batch, int(num_heads), head_dim).contiguous()


def _to_paged_cache(cache: torch.Tensor, *, batch: int, blocks_per_seq: int, block_size: int) -> torch.Tensor:
    return (
        cache.reshape(int(batch), int(cache.shape[1]), int(blocks_per_seq), int(block_size), int(cache.shape[-1]))
        .transpose(1, 2)
        .reshape(int(batch) * int(blocks_per_seq), int(cache.shape[1]), int(block_size), int(cache.shape[-1]))
    )


def _paged_sdpa_q_memory_config(*, device, batch: int, num_heads: int, head_dim: int):
    shard_grid = ttnn.num_cores_to_corerangeset(
        int(batch),
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        (_padded_num_heads(num_heads), int(head_dim)),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _copy_q_to_device(q: torch.Tensor, tt_q) -> None:
    host_q = ttnn.from_torch(q.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(host_q, tt_q)


def _copy_cur_pos_to_device(cur_pos: int, *, batch: int, tt_cur_pos) -> None:
    host_cur_pos = ttnn.from_torch(
        torch.full((int(batch),), int(cur_pos), dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.copy_host_to_device_tensor(host_cur_pos, tt_cur_pos)


def _default_single_visible_device_on_galaxy() -> str | None:
    if os.environ.get("IRD_NUM_PCIE_CHIPS") == "32":
        os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
    return os.environ.get("TT_VISIBLE_DEVICES")


def _cache_rows_read(cur_pos: int) -> dict[str, Any]:
    return {
        "start": 0,
        "end_exclusive": int(cur_pos) + 1,
        "count": int(cur_pos) + 1,
    }


def _cache_pages_read(cur_pos: int, *, block_size: int, page_table: torch.Tensor) -> dict[str, Any]:
    logical_pages = list(range((int(cur_pos) // int(block_size)) + 1))
    return {
        "block_size": int(block_size),
        "logical_pages": logical_pages,
        "physical_pages_by_batch": [
            [int(page_table[batch, page].item()) for page in logical_pages] for batch in range(int(page_table.shape[0]))
        ],
    }


def _padded_num_heads(num_heads: int) -> int:
    return _nearest_pow_2(_nearest_n(int(num_heads), ttnn.TILE_SIZE))


def _nearest_n(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _nearest_pow_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _validate_paged_sdpa_decode_args(
    *,
    batch: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    block_size: int,
    positions: tuple[int, ...],
    q_values: tuple[float, ...],
    pcc: float,
) -> None:
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if num_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(f"head counts must be positive, got num_heads={num_heads}, num_kv_heads={num_kv_heads}")
    if int(num_heads) % int(num_kv_heads) != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if seq_len <= 0 or head_dim <= 0:
        raise ValueError(f"seq_len and head_dim must be positive, got seq_len={seq_len}, head_dim={head_dim}")
    if int(seq_len) % int(block_size) != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by block_size {block_size}")
    if len(positions) < 2:
        raise ValueError("positions must contain at least two replay positions")
    if len(positions) != len(q_values):
        raise ValueError("positions and q_values must have the same length")
    if any(int(position) < 0 or int(position) >= int(seq_len) for position in positions):
        raise ValueError(f"positions must be in [0, {seq_len}), got {positions}")
    if any(int(positions[index]) <= int(positions[index - 1]) for index in range(1, len(positions))):
        raise ValueError(f"positions must be strictly increasing, got {positions}")
    if not 0.0 <= float(pcc) <= 1.0:
        raise ValueError(f"pcc must be in [0, 1], got {pcc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace/replay a paged SDPA decode attention-read proof.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--trace-region-size", type=int, default=DEFAULT_PAGED_SDPA_DECODE_TRACE_REGION_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_PAGED_SDPA_DECODE_BATCH)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_PAGED_SDPA_DECODE_NUM_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_PAGED_SDPA_DECODE_NUM_KV_HEADS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_PAGED_SDPA_DECODE_SEQ_LEN)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_PAGED_SDPA_DECODE_HEAD_DIM)
    parser.add_argument("--block-size", type=int, default=DEFAULT_PAGED_SDPA_DECODE_BLOCK_SIZE)
    parser.add_argument("--positions", type=int, nargs="+", default=list(DEFAULT_PAGED_SDPA_DECODE_POSITIONS))
    parser.add_argument("--q-values", type=float, nargs="+", default=list(DEFAULT_PAGED_SDPA_DECODE_Q_VALUES))
    parser.add_argument("--pcc", type=float, default=DEFAULT_PAGED_SDPA_DECODE_PCC)
    parser.add_argument("--rtol", type=float, default=DEFAULT_PAGED_SDPA_DECODE_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_PAGED_SDPA_DECODE_ATOL)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_paged_sdpa_decode_trace_smoke(
        device_id=args.device_id,
        trace_region_size=args.trace_region_size,
        batch=args.batch,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        block_size=args.block_size,
        positions=tuple(args.positions),
        q_values=tuple(args.q_values),
        pcc=args.pcc,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
