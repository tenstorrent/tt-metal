# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generic expert weight loading and management."""

import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig, Mode
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from ..weight_source import env_path
from .config import ExpertConfig


@dataclass(frozen=True)  # ✅ Make immutable to prevent accidental modification
class ExpertWeights:
    """Container for expert weight tensors - immutable after creation"""

    gate_proj: ttnn.Tensor
    up_proj: ttnn.Tensor
    down_proj: ttnn.Tensor
    gate_proj_bias: ttnn.Tensor
    up_proj_bias: ttnn.Tensor
    down_proj_bias: ttnn.Tensor
    intermediate_size_per_device: int


def load_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat4_b,
    tensor_cache_path=None,
) -> ExpertWeights:
    """
    Load and shard expert weights.

    Args:
        mesh_device: TTNN mesh device
        config: Expert configuration
        state_dict: Dictionary with expert weights
        mesh_config: Mesh parallelization configuration
        weight_dtype: Data type for weights
        tensor_cache_path: Optional path for weight caching

    Returns:
        ExpertWeights with loaded and sharded tensors
    """
    # Calculate sharded dimensions
    intermediate_size_per_device = mesh_config.shard_size(config.intermediate_size, mode=Mode.DECODE)

    if state_dict:
        # Extract gate and up projections from fused weight
        gate_proj = state_dict["gate_up_proj"][..., ::2].reshape(
            1, config.num_experts, config.hidden_size, config.intermediate_size
        )
        up_proj = state_dict["gate_up_proj"][..., 1::2].reshape(
            1, config.num_experts, config.hidden_size, config.intermediate_size
        )
        gate_proj_bias = state_dict["gate_up_proj_bias"][..., ::2].reshape(
            1, config.num_experts, config.intermediate_size
        )
        up_proj_bias = state_dict["gate_up_proj_bias"][..., 1::2].reshape(
            1, config.num_experts, config.intermediate_size
        )
    else:
        gate_proj = None
        up_proj = None
        gate_proj_bias = None
        up_proj_bias = None
    # Get mesh mappers
    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # Load gate projection
    gate_proj_tt = ttnn.as_tensor(
        gate_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load up projection
    up_proj_tt = ttnn.as_tensor(
        up_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias_dtype = ttnn.bfloat16
    # Load gate bias
    gate_proj_bias_tt = ttnn.as_tensor(
        gate_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load up bias
    up_proj_bias_tt = ttnn.as_tensor(
        up_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load down projection
    if state_dict:
        down_proj = state_dict["down_proj"].reshape(1, config.num_experts, config.intermediate_size, config.hidden_size)
        down_proj_bias = state_dict["down_proj_bias"].reshape(1, config.num_experts, config.hidden_size)
        # Handle row-parallel bias (must not be replicated across TP devices)
        if mesh_config.decode.tp > 1:
            down_proj_bias = torch.cat(
                [down_proj_bias] + [torch.zeros_like(down_proj_bias)] * (mesh_config.decode.tp - 1), dim=-1
            )
    else:
        down_proj = None
        down_proj_bias = None

    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    down_proj_bias_tt = ttnn.as_tensor(
        down_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj_bias"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return ExpertWeights(
        gate_proj=gate_proj_tt,
        up_proj=up_proj_tt,
        down_proj=down_proj_tt,
        gate_proj_bias=gate_proj_bias_tt,
        up_proj_bias=up_proj_bias_tt,
        down_proj_bias=down_proj_bias_tt,
        intermediate_size_per_device=intermediate_size_per_device,
    )


@dataclass(frozen=True)
class DenseDecodeExpertWeights:
    """Per device, every expert's TP shard side by side (experts/decode.py decode_forward_dense):
    gate / up: [H, E * I_local] with expert e in columns [e * I_local, (e + 1) * I_local), DRAM width-sharded over the
    banks; their biases expanded to [32, E * I_local] (one row per token, DRAM interleaved);
    down: [E * I_local, H] with expert e in rows [e * I_local, (e + 1) * I_local), DRAM width-sharded;
    down_bias: [E, H] (real on the first TP device only: row-parallel), applied as routing_weights @ down_bias;
    expand: [E, E * I_local] with expand[e, e * I_local + i] = 1 (routing weight of expert e on its columns)."""

    gate: ttnn.Tensor
    up: ttnn.Tensor
    down: ttnn.Tensor
    gate_bias: ttnn.Tensor
    up_bias: ttnn.Tensor
    down_bias: ttnn.Tensor
    expand: ttnn.Tensor
    intermediate_local: int


def _dram_width_sharded(mesh_device, k, n):
    banks = mesh_device.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    shard_n = -(-n // (ttnn.TILE_SIZE * banks)) * ttnn.TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, shard_n), ttnn.ShardOrientation.ROW_MAJOR),
    )


def load_dense_decode_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    tokens,
    weight_dtype=ttnn.bfloat4_b,
    tensor_cache_path=None,
) -> DenseDecodeExpertWeights:
    tp = mesh_config.decode.tp
    E, H, I = config.num_experts, config.hidden_size, config.intermediate_size
    il = I // tp
    columns = mesh_config.column_parallel(mesh_device)
    rows = mesh_config.row_parallel(mesh_device)
    gate = up = down = gate_bias = up_bias = down_bias = None
    if state_dict:
        gate_up = state_dict["gate_up_proj"]  # [E, H, 2 * I], gate / up interleaved
        gate_up_bias = state_dict["gate_up_proj_bias"]  # [E, 2 * I]

        def side_by_side(w):
            """[E, H, I] -> [1, 1, H, tp * E * il]: TP shard d of every expert, experts side by side, shards in order."""
            return torch.cat(
                [w[:, :, d * il : (d + 1) * il].permute(1, 0, 2).reshape(H, E * il) for d in range(tp)], dim=-1
            ).reshape(1, 1, H, tp * E * il)

        def bias_rows(b):
            """[E, I] -> [1, 1, tokens, tp * E * il], one identical row per token."""
            flat = torch.cat([b[:, d * il : (d + 1) * il].reshape(E * il) for d in range(tp)])
            return flat.reshape(1, 1, 1, -1).expand(1, 1, tokens, tp * E * il).contiguous()

        gate = side_by_side(gate_up[..., ::2])
        up = side_by_side(gate_up[..., 1::2])
        gate_bias = bias_rows(gate_up_bias[..., ::2])
        up_bias = bias_rows(gate_up_bias[..., 1::2])
        down_w = state_dict["down_proj"]  # [E, I, H]
        down = torch.cat([down_w[:, d * il : (d + 1) * il, :].reshape(E * il, H) for d in range(tp)], dim=0)
        down = down.reshape(1, 1, tp * E * il, H)
        db = state_dict["down_proj_bias"].reshape(1, 1, E, H)
        down_bias = torch.cat([db] + [torch.zeros_like(db)] * (tp - 1), dim=-1)
    expand = torch.zeros(1, 1, E, E * il)
    for e in range(E):
        expand[0, 0, e, e * il : (e + 1) * il] = 1

    def weight(t, name, mapper, k, n):
        return ttnn.as_tensor(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, name),
            memory_config=_dram_width_sharded(mesh_device, k, n),
        )

    def bias(t, name, mapper):
        return ttnn.as_tensor(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return DenseDecodeExpertWeights(
        gate=weight(gate, "dense_decode_gate", columns, H, E * il),
        up=weight(up, "dense_decode_up", columns, H, E * il),
        down=weight(down, "dense_decode_down", rows, E * il, H),
        gate_bias=bias(gate_bias, f"dense_decode_gate_bias_t{tokens}", columns),
        up_bias=bias(up_bias, f"dense_decode_up_bias_t{tokens}", columns),
        down_bias=bias(down_bias, "dense_decode_down_bias", columns),
        expand=ttnn.from_torch(
            expand,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        intermediate_local=il,
    )


# Decode-only expert weights for the streamed MoE (fused_decode/stream.py). Each TP shard of the intermediate dimension
# is zero-padded up to a multiple of DECODE_INTERMEDIATE_ALIGN so the gate|up column pairs split evenly over the DRAM
# banks (720 -> 768 = 24 tiles = 3 pairs per bank for gpt-oss-20b at TP=4). Padded gate/up columns are zero (weight
# and bias), so their SwiGLU output is exactly zero and the padded down rows contribute nothing.
DECODE_INTERMEDIATE_ALIGN = 256


@dataclass(frozen=True)
class DecodeExpertWeights:
    # Per device: packed [gate | up] (+ bias) and down (+ bias, real on the first TP device only) of every expert, in
    # the per-DRAM-bank streamed layouts of fused_decode/stream.py.
    gate_up_stream: ttnn.Tensor
    down_stream: ttnn.Tensor
    intermediate_padded: int
    down_dtype: object = (
        None  # the down stream's weight dtype when it differs from gate|up (GPT_OSS_DECODE_DOWN_BF8_DIR)
    )
    gate_up_dtype: object = None  # the gate|up stream's weight dtype (GPT_OSS_DECODE_GATEUP_BF8_DIR)


def load_decode_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat4_b,
    tensor_cache_path=None,
) -> DecodeExpertWeights:
    from ..fused_decode.stream import (
        EXPERT_BANK_ROT,
        NBIAS,
        as_stream_tensor,
        columns_per_bank,
        stream_cache_exists,
        stream_down_layout,
        stream_gate_up_layout,
    )

    tp = mesh_config.decode.tp
    inter_local = config.intermediate_size // tp
    inter_pad = -(-inter_local // DECODE_INTERMEDIATE_ALIGN) * DECODE_INTERMEDIATE_ALIGN
    E, H = config.num_experts, config.hidden_size
    banks = mesh_device.dram_grid_size().x
    import re

    from ..fused_decode.config import decode_experts_split, decode_split_dtype

    layer_match = re.search(r"layers\.(\d+)", str(tensor_cache_path))
    split = decode_experts_split(layer_match.group(1) if layer_match else None)
    if split:
        split_streams = _load_split_decode_expert_weights(
            mesh_device,
            config,
            state_dict,
            tp,
            inter_local,
            inter_pad,
            banks,
            tensor_cache_path,
            split,
            layer_match.group(1) if layer_match else None,
        )
        if len(split) == 2:
            return DecodeExpertWeights(
                gate_up_stream=split_streams["gate_up"],
                down_stream=split_streams["down"],
                intermediate_padded=inter_pad,
                down_dtype=decode_split_dtype(layer_match.group(1) if layer_match else None, "down"),
                gate_up_dtype=decode_split_dtype(layer_match.group(1) if layer_match else None, "gate_up"),
            )
    # GPT_OSS_DECODE_BF4_ROUNDED: a directory of layer_<L>.pt holding the experts already rounded onto the bfloat4_b
    # grid by GPTQ (tools/gptq_experts.py: the HF-layout gate_up_proj / down_proj and their biases). The device
    # conversion keeps such values exactly, so decode runs the same bytes at the same speed with ~half the rounding
    # error (top-1 vs the FP32 reference 90.0% -> 94.1%, effective temperature 1.10 -> 1.01). Own cache names.
    rounded_tag = ""
    rounded = env_path("GPT_OSS_DECODE_BF4_ROUNDED")  # a local directory or hf://... (weight_source.py)
    if rounded and weight_dtype == ttnn.bfloat4_b:
        import re

        match = re.search(r"layers\.(\d+)", str(tensor_cache_path))
        if match is None:
            raise ValueError(
                f"GPT_OSS_DECODE_BF4_ROUNDED needs the layer index in tensor_cache_path ({tensor_cache_path})"
            )
        state_dict = {k: v.float() for k, v in torch.load(f"{rounded}/layer_{match.group(1)}.pt").items()}
        rounded_tag = "_r" + os.path.basename(rounded.rstrip("/"))
    from ..fused_decode.config import decode_down_bf8_dir

    down_dtype, down_tag = weight_dtype, ""
    down_dir = decode_down_bf8_dir()
    if down_dir and weight_dtype == ttnn.bfloat4_b:
        import re

        match = re.search(r"layers\.(\d+)", str(tensor_cache_path))
        if match is None:
            raise ValueError(
                f"GPT_OSS_DECODE_DOWN_BF8_DIR needs the layer index in tensor_cache_path ({tensor_cache_path})"
            )
        down_dtype, down_tag = ttnn.bfloat8_b, "_d8" + os.path.basename(down_dir.rstrip("/"))
        if state_dict:
            exact = torch.load(f"{down_dir}/layer_{match.group(1)}.pt")
            state_dict = {
                **state_dict,
                "down_proj": exact["down_proj"].float(),
                "down_proj_bias": exact["down_proj_bias"].float(),
            }
    from ..fused_decode.config import decode_gateup_bf8_dir

    gu_dtype, gu_tag = weight_dtype, ""
    gu_dir = decode_gateup_bf8_dir()
    if gu_dir and weight_dtype == ttnn.bfloat4_b:
        import re

        match = re.search(r"layers\.(\d+)", str(tensor_cache_path))
        if match is None:
            raise ValueError(
                f"GPT_OSS_DECODE_GATEUP_BF8_DIR needs the layer index in tensor_cache_path ({tensor_cache_path})"
            )
        gu_dtype, gu_tag = ttnn.bfloat8_b, "_g8" + os.path.basename(gu_dir.rstrip("/"))
        if state_dict:
            exact = torch.load(f"{gu_dir}/layer_{match.group(1)}.pt")
            state_dict = {
                **state_dict,
                "gate_up_proj": exact["gate_up_proj"].float(),
                "gate_up_proj_bias": exact["gate_up_proj_bias"].float(),
            }
    tile = ttnn.TILE_SIZE

    def pad_shards(t, dim):
        """Split t along dim into tp shards, zero-pad each shard to inter_pad, concatenate."""
        chunks = torch.chunk(t, tp, dim=dim)
        pad = [0, 0] * (t.dim() - 1 - (dim % t.dim())) + [0, inter_pad - inter_local]
        return torch.cat([torch.nn.functional.pad(c, pad) for c in chunks], dim=dim)

    gu_layouts = dn_layouts = None
    if state_dict:
        gate_up = state_dict["gate_up_proj"]
        gate_up_bias = state_dict["gate_up_proj_bias"]
        gate = pad_shards(gate_up[..., ::2].reshape(1, E, H, config.intermediate_size), -1)
        up = pad_shards(gate_up[..., 1::2].reshape(1, E, H, config.intermediate_size), -1)
        gate_bias = pad_shards(gate_up_bias[..., ::2].reshape(E, 1, config.intermediate_size), -1)
        up_bias = pad_shards(gate_up_bias[..., 1::2].reshape(E, 1, config.intermediate_size), -1)
        down = pad_shards(state_dict["down_proj"].reshape(1, E, config.intermediate_size, H), -2)
        down_bias = state_dict["down_proj_bias"].reshape(E, 1, H)
        # Row-parallel down: the bias is added once, on the first TP device.
        down_bias = torch.cat([down_bias] + [torch.zeros_like(down_bias)] * (tp - 1), dim=-1)
        # One streamed layout per TP shard: [E, H, 2 * I_pad] = [gate shard | up shard] and its [E, 2 * I_pad] bias;
        # [E, I_pad, H] down rows and the [E, H] bias.
        gu_layouts = [
            stream_gate_up_layout(
                torch.cat([g[0], u[0]], -1), torch.cat([gb[:, 0], ub[:, 0]], -1), banks, EXPERT_BANK_ROT
            )
            for g, u, gb, ub in zip(
                torch.chunk(gate, tp, -1),
                torch.chunk(up, tp, -1),
                torch.chunk(gate_bias, tp, -1),
                torch.chunk(up_bias, tp, -1),
            )
        ]
        dn_layouts = [
            stream_down_layout(w[0], b[:, 0], banks, EXPERT_BANK_ROT)
            for w, b in zip(torch.chunk(down, tp, -2), torch.chunk(down_bias, tp, -1))
        ]

    suffix0 = f"decode_i{inter_pad}_b{banks}_nb{NBIAS}{rounded_tag}"  # the unrotated layout's cache name
    suffix = f"{suffix0}_rot{EXPERT_BANK_ROT}"
    rows = {
        "gate_up_stream": E * 2 * inter_pad // tile // banks * (H // tile + 1) * tile,
        "down_stream": E * columns_per_bank(H, banks) * (inter_pad // tile + 1) * tile,
    }

    def stream(name, layouts, weight_dtype=weight_dtype, tag=""):
        cache = get_cache_file_name(tensor_cache_path, f"{name}_{suffix}{tag}")
        if layouts is None and cache is not None and not stream_cache_exists(cache, weight_dtype):
            # no rotated cache and no state dict (a cache built before the bank rotation): rotate the unrotated
            # cache's bank shards on the host (exact: the BFP4 blocks stay within a bank's tiles)
            base = as_stream_tensor(
                mesh_device, None, rows[name], weight_dtype, get_cache_file_name(tensor_cache_path, f"{name}_{suffix0}")
            )
            host = ttnn.to_torch(base, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[0]
            ttnn.deallocate(base)
            r = EXPERT_BANK_ROT * tile
            layouts = [torch.cat([h[:, r:], h[:, :r]], dim=-1) for h in host]
        return as_stream_tensor(mesh_device, layouts, rows[name], weight_dtype, cache)

    if "gate_up" in split:
        gate_up_stream, gu_dtype = split_streams["gate_up"], decode_split_dtype(
            layer_match.group(1) if layer_match else None, "gate_up"
        )
    else:
        gate_up_stream = stream("gate_up_stream", gu_layouts, gu_dtype, gu_tag)
    if "down" in split:
        down_stream, down_dtype = split_streams["down"], decode_split_dtype(
            layer_match.group(1) if layer_match else None, "down"
        )
    else:
        down_stream = stream("down_stream", dn_layouts, down_dtype, down_tag)
    return DecodeExpertWeights(
        gate_up_stream=gate_up_stream,
        down_stream=down_stream,
        intermediate_padded=inter_pad,
        down_dtype=down_dtype,
        gate_up_dtype=gu_dtype,
    )


def _split_stream_tensor(mesh_device, per_device, name, tensor_cache_path, bank_bytes):
    """per_device: one list per TP device of `banks` equal-length uint8 byte streams (or None: load the cache) -> one
    DRAM width-sharded uint8 tensor per device whose bank b shard is that bank's bytes, contiguous (the readers address
    them by byte offset). ROW_MAJOR pages of 64 bytes."""
    W = 64
    rows = bank_bytes // W
    banks = mesh_device.dram_grid_size().x
    t = None
    if per_device is not None:
        t = torch.stack([torch.cat([b.reshape(rows, W) for b in dev], dim=-1) for dev in per_device]).unsqueeze(0)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (rows, W),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return ttnn.as_tensor(
        t,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint8,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, 1)),
        cache_file_name=get_cache_file_name(tensor_cache_path, name),
        memory_config=mem,
    )


def _load_split_decode_expert_weights(
    mesh_device,
    config,
    state_dict,
    tp,
    inter_local,
    inter_pad,
    banks,
    cache_path,
    streams=("gate_up", "down"),
    layer_idx=None,
):
    """TEMP experiment (GPT_OSS_DECODE_EXPERTS_SPLIT): {stream name: tensor} for the decode expert streams in `streams`,
    holding the checkpoint's MXFP4 weights exactly (fused_decode/mxfp4_split.py), in the bank / expert / column order of
    stream_gate_up_layout and stream_down_layout, each column (down: each expert column segment) = [BFP4 part][BFP2
    part]. state_dict must hold the checkpoint values (read before any GPTQ substitution).

    GPT_OSS_DECODE_SPLIT_ORDERS=<dir of order_<L>.pt [E, I]>: reorder each expert's intermediate neurons as the GPTQ
    searched-order build did (neuron r = checkpoint neuron order[e][r]: gate / up columns 2 r, 2 r + 1, their bias, down
    row r), so a split stream pairs with a GPTQ stream of that build (GPT_OSS_DECODE_BF4_ROUNDED); own cache names."""
    import os

    from ..fused_decode import mxfp4_split as ms
    from ..fused_decode.config import DECODE_SPLIT_RANKS
    from ..fused_decode.stream import EXPERT_BANK_ROT, columns_per_bank

    E, H, I = config.num_experts, config.hidden_size, config.intermediate_size
    tile = ttnn.TILE_SIZE
    kx, kd = H // tile, inter_pad // tile
    with_round = DECODE_SPLIT_RANKS is not None  # + the BFP4 rounding read by the inexact routed ranks
    gu_col = (kx + 1) * ms.BFP4_BYTES * (2 if with_round else 1) + kx * ms.BFP2_BYTES
    dn_seg = (kd + 1) * ms.BFP4_BYTES * (2 if with_round else 1) + kd * ms.BFP2_BYTES
    P = inter_pad // tile // banks  # gate|up column pairs per bank
    Cd = columns_per_bank(H, banks)
    gu_bank = E * 2 * P * gu_col
    dn_bank = E * Cd * dn_seg

    def shard_pad(t, dim, d):
        c = torch.chunk(t, tp, dim=dim)[d]
        pad = [0, 0] * (t.dim() - 1 - (dim % t.dim())) + [0, inter_pad - inter_local]
        return torch.nn.functional.pad(c, pad)

    def pack(w, b, wr=None, br=None, wp=None, bp=None, chunk=4):
        sl = lambda t, i: None if t is None else t[i : i + chunk]
        if wp is not None:  # rest-pkg: the third part = the package's bfloat4_b tiles (normal orientation)
            parts = [
                (
                    *ms.pack_columns(w[i : i + chunk], b[i : i + chunk]),
                    ms.pack_columns_pkg(wp[i : i + chunk], bp[i : i + chunk]),
                )
                for i in range(0, w.shape[0], chunk)
            ]
        else:
            parts = [
                ms.pack_columns(w[i : i + chunk], b[i : i + chunk], with_round, sl(wr, i), sl(br, i))
                for i in range(0, w.shape[0], chunk)
            ]
        return torch.cat([torch.cat(list(p), -1) for p in parts], 0)  # [E, n_tiles, column bytes]

    def bank_streams(cols):  # [E, banks * C, bytes] -> bank b's bytes = logical bank (b + rot) % banks, expert-major
        C = cols.shape[1] // banks
        return [cols[:, L * C : (L + 1) * C].reshape(-1) for L in ((b + EXPERT_BANK_ROT) % banks for b in range(banks))]

    orders = os.environ.get("GPT_OSS_DECODE_SPLIT_ORDERS", "").strip()
    tag = f"decode_split_i{inter_pad}_b{banks}_rot{EXPERT_BANK_ROT}"
    if with_round:
        tag += "_rnd"
    if orders:
        tag += "_o" + os.path.basename(orders.rstrip("/"))
    # GPT_OSS_DECODE_SPLIT_ROUND_DIR: the rounded part from a dir of layer_<L>.pt on the K-grouped grid (gptq_k.py,
    # checkpoint channel order) instead of round_k of the checkpoint
    round_dir = env_path("GPT_OSS_DECODE_SPLIT_ROUND_DIR") if with_round else ""
    if round_dir:
        assert layer_idx is not None, "GPT_OSS_DECODE_SPLIT_ROUND_DIR needs the layer index"
        tag += "_g" + os.path.basename(round_dir.rstrip("/"))
    # GPT_OSS_DECODE_SPLIT_ROUND_ORDERS: the rounded part (ROUND_DIR, already in this order) uses another neuron order than
    # the exact parts: an expert's gate|up and down read the same part at one rank, so each part needs only its own order
    round_orders = os.environ.get("GPT_OSS_DECODE_SPLIT_ROUND_ORDERS", "").strip() if round_dir else ""
    from ..fused_decode.config import DECODE_SPLIT_REST_PKG

    rest_pkg = DECODE_SPLIT_REST_PKG and with_round
    pkg_dir = env_path("GPT_OSS_DECODE_BF4_ROUNDED") if rest_pkg else ""
    if rest_pkg:
        assert pkg_dir and layer_idx is not None, "GPT_OSS_DECODE_SPLIT_REST_PKG needs GPT_OSS_DECODE_BF4_ROUNDED"
        tag = tag.replace("_rnd", "_rnd_pk" + os.path.basename(pkg_dir.rstrip("/")))
    names = {s: f"{s}_stream_{tag}" for s in streams}
    # as_tensor's cache file names (ttnn/operations/core.py): a cached stream is loaded, so skip the CPU packing
    cached = cache_path is not None and all(
        os.path.isfile(f"{get_cache_file_name(cache_path, n)}_dtype_UINT8_layout_ROW_MAJOR.tensorbin")
        for n in names.values()
    )
    gu_dev = dn_dev = None
    if state_dict and not cached:
        gu, gub = state_dict["gate_up_proj"].float(), state_dict["gate_up_proj_bias"].float()  # [E, H, 2I], [E, 2I]
        dn, dnb = state_dict["down_proj"].float(), state_dict["down_proj_bias"].float()  # [E, I, H], [E, H]
        gu_p = gub_p = dn_p = dnb_p = None
        if rest_pkg:  # the package's GPTQ experts (searched neuron order, biases in that order)
            pw = torch.load(f"{pkg_dir}/layer_{int(layer_idx)}.pt")
            gu_p, gub_p = pw["gate_up_proj"].float(), pw["gate_up_proj_bias"].float()
            dn_p, dnb_p = pw["down_proj"].float(), pw["down_proj_bias"].float()
        gu_r = dn_r = gub_r = None
        if round_dir and not rest_pkg:
            rw = torch.load(f"{round_dir}/layer_{int(layer_idx)}.pt")
            gu_r, dn_r = rw["gate_up_proj"].float(), rw["down_proj"].float()  # [E, H, 2I], [E, I, H]
            if round_orders:
                pr = torch.load(f"{round_orders}/order_{int(layer_idx)}.pt").long()
                gub_r = torch.gather(gub, 1, torch.stack([2 * pr, 2 * pr + 1], -1).reshape(E, 2 * I))
        if orders:
            assert layer_idx is not None, "GPT_OSS_DECODE_SPLIT_ORDERS needs the layer index in the cache path"
            pi = torch.load(f"{orders}/order_{int(layer_idx)}.pt").long()  # [E, I]
            pi2 = torch.stack([2 * pi, 2 * pi + 1], -1).reshape(E, 2 * I)  # interleaved gate / up columns
            gu = torch.gather(gu, 2, pi2.unsqueeze(1).expand(-1, H, -1))
            gub = torch.gather(gub, 1, pi2)
            dn = torch.gather(dn, 1, pi.unsqueeze(-1).expand(-1, -1, H))
            if gu_r is not None:  # K-grouped rounding is per output column: reordering columns keeps it on the grid
                gu_r = torch.gather(gu_r, 2, pi2.unsqueeze(1).expand(-1, H, -1))
                dn_r = torch.gather(dn_r, 1, pi.unsqueeze(-1).expand(-1, -1, H))
        gu_dev, dn_dev = [], []
        half = inter_pad // tile
        order = [c for bank in range(banks) for j in range(bank * P, (bank + 1) * P) for c in (j, half + j)]
        npad = Cd * banks * tile - H
        for d in range(tp):
            if "gate_up" in streams:
                w = torch.cat([shard_pad(gu[..., ::2], -1, d), shard_pad(gu[..., 1::2], -1, d)], -1)
                b = torch.cat([shard_pad(gub[..., ::2], -1, d), shard_pad(gub[..., 1::2], -1, d)], -1)
                wr = br = wp = bp = None
                if gu_r is not None:
                    wr = torch.cat([shard_pad(gu_r[..., ::2], -1, d), shard_pad(gu_r[..., 1::2], -1, d)], -1)
                if gub_r is not None:
                    br = torch.cat([shard_pad(gub_r[..., ::2], -1, d), shard_pad(gub_r[..., 1::2], -1, d)], -1)
                if gu_p is not None:
                    wp = torch.cat([shard_pad(gu_p[..., ::2], -1, d), shard_pad(gu_p[..., 1::2], -1, d)], -1)
                    bp = torch.cat([shard_pad(gub_p[..., ::2], -1, d), shard_pad(gub_p[..., 1::2], -1, d)], -1)
                gu_dev.append(bank_streams(pack(w, b, wr, br, wp, bp)[:, order]))
            if "down" in streams:
                wd = torch.nn.functional.pad(shard_pad(dn, -2, d), (0, npad))
                bd = torch.nn.functional.pad(dnb if d == 0 else torch.zeros_like(dnb), (0, npad))  # row-parallel bias
                wr = None if dn_r is None else torch.nn.functional.pad(shard_pad(dn_r, -2, d), (0, npad))
                wp = bp = None
                if dn_p is not None:
                    wp = torch.nn.functional.pad(shard_pad(dn_p, -2, d), (0, npad))
                    bp = torch.nn.functional.pad(dnb_p if d == 0 else torch.zeros_like(dnb_p), (0, npad))
                dn_dev.append(bank_streams(pack(wd, bd, wr, None, wp, bp)))
    # GPT_OSS_DECODE_SPLIT_ROUND_ONLY: a stream with no exact rank keeps only its rounded parts (the last part4 bytes of
    # every segment), cut from the full stream; own cache names "_ro"
    from ..fused_decode.config import DECODE_SPLIT_ROUND_ONLY, DECODE_SPLIT_SEPARATE, decode_split_dtype

    part4 = {"gate_up": (kx + 1) * ms.BFP4_BYTES, "down": (kd + 1) * ms.BFP4_BYTES}
    seg = {"gate_up": gu_col, "down": dn_seg}
    full_bank = {"gate_up": gu_bank, "down": dn_bank}
    per_device = {"gate_up": gu_dev, "down": dn_dev}
    out = {}
    for s in ("gate_up", "down"):
        if s not in streams:
            continue
        dtype = decode_split_dtype(layer_idx, s)
        round_only = DECODE_SPLIT_ROUND_ONLY and with_round and dtype == "split_r0"
        # GPT_OSS_DECODE_SPLIT_SEPARATE: 0 < k exact ranks: per bank every segment's exact parts, then every rounded part
        separate = DECODE_SPLIT_SEPARATE and with_round and not rest_pkg and dtype not in ("split", "split_r0")
        if not (round_only or separate):
            out[s] = _split_stream_tensor(mesh_device, per_device[s], names[s], cache_path, full_bank[s])
            continue
        ro_name = names[s] + ("_ro" if round_only else "_sep")
        ro_bank = full_bank[s] // seg[s] * part4[s] if round_only else full_bank[s]
        ro_dev = None
        if not os.path.isfile(f"{get_cache_file_name(cache_path, ro_name)}_dtype_UINT8_layout_ROW_MAJOR.tensorbin"):
            full = per_device[s]
            if full is None:  # the full stream from its cache, back to host bytes per device and bank
                t = _split_stream_tensor(mesh_device, None, names[s], cache_path, full_bank[s])
                host = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[0]
                ttnn.deallocate(t)
                full = [[h[:, b * 64 : (b + 1) * 64].reshape(-1) for b in range(banks)] for h in host]
            if round_only:
                ro_dev = [[b.reshape(-1, seg[s])[:, -part4[s] :].reshape(-1) for b in dev] for dev in full]
            else:
                cut = lambda v: torch.cat([v[:, : -part4[s]].reshape(-1), v[:, -part4[s] :].reshape(-1)])
                ro_dev = [[cut(b.reshape(-1, seg[s])) for b in dev] for dev in full]
        out[s] = _split_stream_tensor(mesh_device, ro_dev, ro_name, cache_path, ro_bank)
    return out


def load_decode_down_bal(
    mesh_device, config: ExpertConfig, mesh_config: MeshConfig, dw, weight_dtype, tensor_cache_path
):
    """The down weights of DecodeExpertWeights dw again, in the balanced layout of the fused verify MoE
    (fused_decode/stream.py down_bal_plan / stream_down_bal_layout: more down columns on the DRAM bank whose gate|up
    pairs are padding, no all-padding K tiles), from its cache or rebuilt from dw.down_stream (exact: whole BFP4
    tiles move). Returns (tensor, (bank_cols, seg_inter))."""
    from ..fused_decode.stream import (
        EXPERT_BANK_ROT,
        NBIAS,
        as_stream_tensor,
        bank_workers,
        columns_per_bank,
        down_bal_plan,
        stream_cache_exists,
        stream_down_bal_layout,
    )

    tp = mesh_config.decode.tp
    inter_local = config.intermediate_size // tp
    inter_pad = dw.intermediate_padded
    E, H = config.num_experts, config.hidden_size
    banks = mesh_device.dram_grid_size().x
    tile = ttnn.TILE_SIZE
    # banks whose optimal reader cores share a row (the MoE's row placement puts their readers on it)
    workers = bank_workers(mesh_device)
    rows = {}
    for b, w in enumerate(workers):
        rows.setdefault(w.y, []).append(b)
    shared = [g for g in rows.values() if len(g) > 1]
    plan = down_bal_plan(H, inter_pad, inter_local, banks, EXPERT_BANK_ROT, shared_rows=shared)
    bank_cols, seg_inter = plan
    C = max(n for _, n in bank_cols)
    n_tiles = -(-H // tile)
    suffix = f"decode_i{inter_pad}_b{banks}_nb{NBIAS}_rot{EXPERT_BANK_ROT}"
    cache = get_cache_file_name(
        tensor_cache_path, f"down_bal_{suffix}_s{seg_inter}_" + "_".join(str(n) for _, n in bank_cols)
    )
    layouts = None
    if not stream_cache_exists(cache, weight_dtype):
        Cs = columns_per_bank(H, banks)
        seg = inter_pad // tile + 1
        hosts = ttnn.to_torch(dw.down_stream, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[0]
        layouts = []
        for h in hosts:  # [E, columns, K tiles (I_pad / 32 + bias), 32, 32] of this device, logical column order
            cols = torch.zeros(E, Cs * banks, seg, tile, tile, dtype=h.dtype)
            for b in range(banks):
                L = (b + EXPERT_BANK_ROT) % banks
                cols[:, L * Cs : (L + 1) * Cs] = h[:, b * tile : (b + 1) * tile].reshape(E, Cs, seg, tile, tile)
            cols = torch.cat([cols[:, :n_tiles, :seg_inter], cols[:, :n_tiles, -1:]], dim=2)
            layouts.append(stream_down_bal_layout(cols, bank_cols))
    t = as_stream_tensor(mesh_device, layouts, E * C * (seg_inter + 1) * tile, weight_dtype, cache)
    return t, plan
