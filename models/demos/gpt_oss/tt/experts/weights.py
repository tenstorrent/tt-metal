# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generic expert weight loading and management."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig, Mode
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from ..bfp4_quant import bfp4_cache_tag, mxfp4_to_bfp4_k, n_grouped, prepare_bfp4_expert_weight
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
    nd=False,
    down_weight_dtype=None,
    rounding="max",
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
        nd: Store gate / up / down ND-sharded in one [1, 1, K, 32] DRAM shard per (expert, output tile column)
            (fused_decode/nd.py: the TP=1 fused decode streams these same tensors; prefill reads them through the
            sparse matmuls' tensor accessors)
        down_weight_dtype: dtype of down_proj (default weight_dtype)
        rounding: BFP4 rounding of the fused decode path (../bfp4_quant.py; these copies are N-grouped), "max" (ttnn's
            packer) for the original path

    Returns:
        ExpertWeights with loaded and sharded tensors
    """
    # Calculate sharded dimensions
    intermediate_size_per_device = mesh_config.shard_size(config.intermediate_size, mode=Mode.DECODE)
    down_weight_dtype = down_weight_dtype or weight_dtype
    gate_up_memory_config = down_memory_config = ttnn.DRAM_MEMORY_CONFIG
    nd_suffix = ""
    tp = mesh_config.decode.tp
    # ND experts at TP > 1 (gpt-oss-120b on 1x4): each TP shard of the intermediate dimension (720) zero-padded to whole
    # tiles (736), so the [1, 1, K, 32] ND shards hold whole tile columns; the padded gate / up columns (weight and
    # bias 0) give SwiGLU outputs of exactly 0 and the padded down rows are 0, in prefill and decode alike.
    inter_pad = intermediate_size_per_device
    if nd:
        from ..fused_decode.nd import nd_expert_memory_config

        inter_pad = -(-intermediate_size_per_device // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        gate_up_memory_config = nd_expert_memory_config(mesh_device, config.hidden_size)
        down_memory_config = nd_expert_memory_config(mesh_device, inter_pad)
        nd_suffix = "_nd1x32" + (f"_i{inter_pad}" if inter_pad != intermediate_size_per_device else "")

    def pad_shards(t, dim):
        """Each of the tp shards of t along dim zero-padded to inter_pad (identity when no padding)."""
        if t is None or inter_pad == intermediate_size_per_device:
            return t
        pad = [0, 0] * (t.dim() - 1 - (dim % t.dim())) + [0, inter_pad - intermediate_size_per_device]
        return torch.cat([torch.nn.functional.pad(c, pad) for c in torch.chunk(t, tp, dim=dim)], dim=dim)

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
        gate_proj = pad_shards(prepare_bfp4_expert_weight(gate_proj, weight_dtype, rounding), -1)
        up_proj = pad_shards(prepare_bfp4_expert_weight(up_proj, weight_dtype, rounding), -1)
        gate_proj_bias = pad_shards(gate_proj_bias, -1)
        up_proj_bias = pad_shards(up_proj_bias, -1)
    else:
        gate_proj = None
        up_proj = None
        gate_proj_bias = None
        up_proj_bias = None
    qtag = bfp4_cache_tag(weight_dtype, n_grouped(rounding))
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
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj{nd_suffix}{qtag}"),
        memory_config=gate_up_memory_config,
    )

    # Load up projection
    up_proj_tt = ttnn.as_tensor(
        up_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj{nd_suffix}{qtag}"),
        memory_config=gate_up_memory_config,
    )
    bias_dtype = ttnn.bfloat16
    # padded biases (ND experts at TP > 1) under their own cache names
    bias_suffix = f"_i{inter_pad}" if inter_pad != intermediate_size_per_device else ""
    # Load gate bias
    gate_proj_bias_tt = ttnn.as_tensor(
        gate_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj_bias{bias_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load up bias
    up_proj_bias_tt = ttnn.as_tensor(
        up_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=bias_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj_bias{bias_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load down projection
    if state_dict:
        down_proj = state_dict["down_proj"].reshape(1, config.num_experts, config.intermediate_size, config.hidden_size)
        down_proj = pad_shards(prepare_bfp4_expert_weight(down_proj, down_weight_dtype, rounding), -2)
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
        dtype=down_weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(
            tensor_cache_path, f"down_proj{nd_suffix}{bfp4_cache_tag(down_weight_dtype, n_grouped(rounding))}"
        ),
        memory_config=down_memory_config,
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
        intermediate_size_per_device=inter_pad,
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
    # the copies' dtype and BFP4 rounding (../bfp4_quant.py)
    dtype: object = ttnn.bfloat4_b
    rounding: str = "max"

    @property
    def k_grouped(self):
        """K-grouped BFP4 copies: every tile stored transposed (fused_decode/stream.py transposed=True)."""
        return self.rounding == "kmse" and self.dtype == ttnn.bfloat4_b


def load_decode_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat4_b,
    tensor_cache_path=None,
    rounding="max",
) -> DecodeExpertWeights:
    """rounding: of BFP4 copies (../bfp4_quant.py): "kmse" = K-grouped (tiles stored transposed, BFP4 groups of 16
    along K), "mse" / "max" N-grouped."""
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
    tile = ttnn.TILE_SIZE
    kgroup = rounding == "kmse" and weight_dtype == ttnn.bfloat4_b

    def pad_shards(t, dim):
        """Split t along dim into tp shards, zero-pad each shard to inter_pad, concatenate."""
        chunks = torch.chunk(t, tp, dim=dim)
        pad = [0, 0] * (t.dim() - 1 - (dim % t.dim())) + [0, inter_pad - inter_local]
        return torch.cat([torch.nn.functional.pad(c, pad) for c in chunks], dim=dim)

    def rnd(t):
        # BFP4 rounding on the logical [.., K, N] weights: the streamed layouts move whole tiles, so the groups stay
        return mxfp4_to_bfp4_k(t) if kgroup else prepare_bfp4_expert_weight(t, weight_dtype, rounding)

    gu_layouts = dn_layouts = None
    if state_dict:
        gate_up = state_dict["gate_up_proj"]
        gate_up_bias = state_dict["gate_up_proj_bias"]
        gate = pad_shards(rnd(gate_up[..., ::2].reshape(1, E, H, config.intermediate_size)), -1)
        up = pad_shards(rnd(gate_up[..., 1::2].reshape(1, E, H, config.intermediate_size)), -1)
        gate_bias = pad_shards(gate_up_bias[..., ::2].reshape(E, 1, config.intermediate_size), -1)
        up_bias = pad_shards(gate_up_bias[..., 1::2].reshape(E, 1, config.intermediate_size), -1)
        down = pad_shards(rnd(state_dict["down_proj"].reshape(1, E, config.intermediate_size, H)), -2)
        down_bias = state_dict["down_proj_bias"].reshape(E, 1, H)
        # Row-parallel down: the bias is added once, on the first TP device.
        down_bias = torch.cat([down_bias] + [torch.zeros_like(down_bias)] * (tp - 1), dim=-1)
        # One streamed layout per TP shard: [E, H, 2 * I_pad] = [gate shard | up shard] and its [E, 2 * I_pad] bias;
        # [E, I_pad, H] down rows and the [E, H] bias.
        gu_layouts = [
            stream_gate_up_layout(
                torch.cat([g[0], u[0]], -1),
                torch.cat([gb[:, 0], ub[:, 0]], -1),
                banks,
                EXPERT_BANK_ROT,
                transposed=kgroup,
            )
            for g, u, gb, ub in zip(
                torch.chunk(gate, tp, -1),
                torch.chunk(up, tp, -1),
                torch.chunk(gate_bias, tp, -1),
                torch.chunk(up_bias, tp, -1),
            )
        ]
        dn_layouts = [
            stream_down_layout(w[0], b[:, 0], banks, EXPERT_BANK_ROT, transposed=kgroup)
            for w, b in zip(torch.chunk(down, tp, -2), torch.chunk(down_bias, tp, -1))
        ]

    # Cache names: ttnn-rounded BFP4 copies keep the original names, other roundings / dtypes get their own.
    suffix0 = f"decode_i{inter_pad}_b{banks}_nb{NBIAS}{bfp4_cache_tag(weight_dtype, rounding)}"  # unrotated layout
    suffix = f"{suffix0}_rot{EXPERT_BANK_ROT}" + ("_bfp8" if weight_dtype == ttnn.bfloat8_b else "")
    legacy = suffix0 == f"decode_i{inter_pad}_b{banks}_nb{NBIAS}" and weight_dtype == ttnn.bfloat4_b
    rows = {
        "gate_up_stream": E * 2 * inter_pad // tile // banks * (H // tile + 1) * tile,
        "down_stream": E * columns_per_bank(H, banks) * (inter_pad // tile + 1) * tile,
    }

    def stream(name, layouts):
        cache = get_cache_file_name(tensor_cache_path, f"{name}_{suffix}")
        if layouts is None and cache is not None and legacy and not stream_cache_exists(cache, weight_dtype):
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

    return DecodeExpertWeights(
        gate_up_stream=stream("gate_up_stream", gu_layouts),
        down_stream=stream("down_stream", dn_layouts),
        intermediate_padded=inter_pad,
        dtype=weight_dtype,
        rounding=rounding,
    )


def load_decode_down_bal(mesh_device, config: ExpertConfig, mesh_config: MeshConfig, dw, tensor_cache_path):
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
    weight_dtype = dw.dtype
    suffix = f"decode_i{inter_pad}_b{banks}_nb{NBIAS}{bfp4_cache_tag(weight_dtype, dw.rounding)}_rot{EXPERT_BANK_ROT}"
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
