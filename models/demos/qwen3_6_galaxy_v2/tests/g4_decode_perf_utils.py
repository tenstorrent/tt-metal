# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for G=4 decode perf assessment microbenchmarks."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn

# Qwen3.6-27B dense LM geometry
_DIM = 5120
_INTER = 17408
_N_LAYERS = 64
_N_MLP_PARAMS_PER_LAYER = 3 * _DIM * _INTER  # gate + up + down per layer
_TOTAL_MODEL_PARAMS = 27e9  # Qwen3.6-27B dense
_MLP_PARAM_FRACTION = (_N_MLP_PARAMS_PER_LAYER * _N_LAYERS) / _TOTAL_MODEL_PARAMS

# G=4 stage: 4-chip line mesh
G4_MESH_SHAPE = (1, 4)
G4_CLUSTER_AXIS = 1
G4_BW_BYTES_PER_S = 4 * 512 * (1024**3)  # 2.048 TB/s aggregate

TARGET_TOK_S = 70.0
TARGET_MS_PER_TOKEN = 1000.0 / TARGET_TOK_S  # ~14.29 ms

BYTES_PER_WEIGHT_DTYPE = {
    ttnn.bfloat16: 2,
    ttnn.bfloat8_b: 1,
    ttnn.bfloat4_b: 0.5,
}

WDTYPE_NAMES = {
    ttnn.bfloat16: "bf16",
    ttnn.bfloat8_b: "bf8",
    ttnn.bfloat4_b: "bf4",
}

DEFAULT_SNAPSHOT = Path(
    os.environ.get(
        "QWEN36_HF_SNAPSHOT",
        "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
        "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    )
)


@dataclass(frozen=True)
class AnalyticFloor:
    mlp_ms: float
    full_model_ms: float
    mlp_tok_s: float
    full_model_tok_s: float


def tile_pad_m(batch: int, tile: int = 32) -> int:
    """Tile-padded M dim for decode (batch lives in dim-2 for qwen3.6 decode)."""
    return max(tile, int(math.ceil(batch / tile) * tile))


def analytic_mlp_floor_ms(wdtype: ttnn.DataType) -> AnalyticFloor:
    bpe = BYTES_PER_WEIGHT_DTYPE[wdtype]
    mlp_bytes_per_layer = _N_MLP_PARAMS_PER_LAYER * bpe
    mlp_ms = (mlp_bytes_per_layer / G4_BW_BYTES_PER_S) * 1000.0
    mlp_all_layers_ms = mlp_ms * _N_LAYERS
    full_model_bytes = _TOTAL_MODEL_PARAMS * bpe
    full_ms = (full_model_bytes / G4_BW_BYTES_PER_S) * 1000.0
    return AnalyticFloor(
        mlp_ms=mlp_ms,
        full_model_ms=full_ms,
        mlp_tok_s=1000.0 / mlp_all_layers_ms,
        full_model_tok_s=1000.0 / full_ms,
    )


def torch_swiglu_mlp(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """x [B,1,M,H]; weights transposed matmul layout [H, inter] / [inter, H]."""
    gate = torch.matmul(x, w1)
    up = torch.matmul(x, w3)
    hidden = torch.nn.functional.silu(gate) * up
    return torch.matmul(hidden, w2)


def load_mlp_weights_torch(snapshot_dir: Path, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load gate/up/down for one layer; return transposed [dim,inter] / [inter,dim]."""
    import json

    from safetensors.torch import load_file as load_st

    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
        prefixes = [
            f"model.language_model.layers.{layer}.mlp.gate_proj.weight",
            f"model.language_model.layers.{layer}.mlp.up_proj.weight",
            f"model.language_model.layers.{layer}.mlp.down_proj.weight",
        ]
        files = sorted({weight_map[k] for k in prefixes})
        sd: dict[str, torch.Tensor] = {}
        for fn in files:
            shard = load_st(str(snapshot_dir / fn))
            for k in prefixes:
                if k in shard:
                    sd[k] = shard[k]
        gate = sd[prefixes[0]]
        up = sd[prefixes[1]]
        down = sd[prefixes[2]]
    else:
        from models.demos.qwen3_6_galaxy_v2.tt.load_checkpoints import load_hf_state_dict

        ckpt = os.environ.get("HF_MODEL", str(snapshot_dir))
        sd = load_hf_state_dict(ckpt)
        gate = sd[f"model.layers.{layer}.mlp.gate_proj.weight"]
        up = sd[f"model.layers.{layer}.mlp.up_proj.weight"]
        down = sd[f"model.layers.{layer}.mlp.down_proj.weight"]

    w1 = gate.T.contiguous()
    w3 = up.T.contiguous()
    w2 = down.T.contiguous()
    return w1, w3, w2


def _shard_cols_2d(full: torch.Tensor, cols: int) -> torch.Tensor:
    """Pack last-dim col-shard for a (1, cols) mesh as 4D [1, cols, M, H/cols].

    `full` is [1, 1, M, H]; per-chip tensors are [1, 1, M, H/cols] after
    ShardTensor2dMesh(dims=(0, 1)).
    """
    assert full.shape[0] == 1 and full.shape[1] == 1
    _, _, m, h = full.shape
    assert h % cols == 0
    hpc = h // cols
    out = torch.zeros(1, cols, m, hpc, dtype=full.dtype)
    for c in range(cols):
        out[0, c] = full[0, 0, :, c * hpc : (c + 1) * hpc]
    return out


def make_col_sharded_activation(
    mesh: ttnn.MeshDevice,
    batch_rows: int,
    *,
    seed: int = 42,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> tuple[torch.Tensor, ttnn.Tensor]:
    """Col-sharded [1,1,M,H/4] activation matching qwen3.6 decode contract."""
    torch.manual_seed(seed)
    m = tile_pad_m(batch_rows)
    full = torch.randn(1, 1, m, _DIM, dtype=torch.bfloat16) * 0.02
    packed = _shard_cols_2d(full, G4_MESH_SHAPE[1])
    x = ttnn.from_torch(
        packed,
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=G4_MESH_SHAPE),
    )
    return full, x


def _shard_w_col_parallel(w: torch.Tensor) -> torch.Tensor:
    """Column-parallel pack of a [K, N] weight: chip c gets [K, N/cols].

    For gate/up (w1/w3): full K=H contraction on each chip, output cols split.
    Packed 4D [1, cols, K, N/cols] -> per-chip [1, 1, K, N/cols].
    """
    cols = G4_MESH_SHAPE[1]
    k, n = w.shape
    assert n % cols == 0
    npc = n // cols
    packed = torch.zeros(1, cols, k, npc, dtype=w.dtype)
    for c in range(cols):
        packed[0, c] = w[:, c * npc : (c + 1) * npc]
    return packed


def _shard_w_row_parallel(w: torch.Tensor) -> torch.Tensor:
    """Row-parallel pack of a [K, N] weight: chip c gets [K/cols, N].

    For down (w2): K=I contraction split across chips (partial sums), output
    full N=H per chip (reduced by the following reduce_scatter/all_reduce).
    Packed 4D [1, cols, K/cols, N] -> per-chip [1, 1, K/cols, N].
    """
    cols = G4_MESH_SHAPE[1]
    k, n = w.shape
    assert k % cols == 0
    kpc = k // cols
    packed = torch.zeros(1, cols, kpc, n, dtype=w.dtype)
    for c in range(cols):
        packed[0, c] = w[c * kpc : (c + 1) * kpc, :]
    return packed


def upload_mlp_weights(
    mesh: ttnn.MeshDevice,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    wdtype: ttnn.DataType,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    def _up(packed: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            packed,
            device=mesh,
            dtype=wdtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=G4_MESH_SHAPE),
        )

    # w1/w3 column-parallel (output split), w2 row-parallel (contraction split).
    return _up(_shard_w_col_parallel(w1)), _up(_shard_w_col_parallel(w3)), _up(_shard_w_row_parallel(w2))


def mlp_g4_forward(
    x: ttnn.Tensor,
    w1: ttnn.Tensor,
    w3: ttnn.Tensor,
    w2: ttnn.Tensor,
    *,
    act_dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """Correct single-axis 4-chip TP SwiGLU MLP (Megatron column/row parallel).

    Input  x:  col-sharded [1,1,M,H/4] (axis=1, 4 chips).
    Output:    col-sharded [1,1,M,H/4].

    1. all_gather(axis=1) -> full [1,1,M,H] on each chip
    2. w1/w3 column-parallel  -> [1,1,M,I/4]  (no comm)
    3. SwiGLU silu(w1)*w3     -> [1,1,M,I/4]
    4. w2 row-parallel        -> partial [1,1,M,H]
    5. reduce_scatter(axis=1) -> [1,1,M,H/4]  (sum partials + scatter H)
    """
    mem = ttnn.DRAM_MEMORY_CONFIG
    x_full = ttnn.all_gather(
        x,
        dim=3,
        cluster_axis=G4_CLUSTER_AXIS,
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=mem,
    )
    w1_out = ttnn.linear(x_full, w1, dtype=act_dtype, memory_config=mem)
    w3_out = ttnn.linear(x_full, w3, dtype=act_dtype, memory_config=mem)
    ttnn.deallocate(x_full)
    ff = ttnn.mul(
        w1_out,
        w3_out,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        dtype=act_dtype,
        memory_config=mem,
    )
    ttnn.deallocate(w1_out)
    ttnn.deallocate(w3_out)
    w2_out = ttnn.linear(ff, w2, dtype=act_dtype, memory_config=mem)
    ttnn.deallocate(ff)
    out = ttnn.reduce_scatter(
        w2_out,
        dim=3,
        cluster_axis=G4_CLUSTER_AXIS,
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=mem,
    )
    ttnn.deallocate(w2_out)
    return out


def gather_g4_output_to_torch(mesh: ttnn.MeshDevice, out: ttnn.Tensor) -> torch.Tensor:
    """Reconstruct full [1,1,M,H] from col-sharded per-chip [1,1,M,H/4]."""
    return ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=G4_MESH_SHAPE),
    )


def print_decision_table(results: dict) -> None:
    print("\n" + "=" * 72)
    print("G=4 DECODE PERF ASSESSMENT — DECISION TABLE")
    print("=" * 72)
    print(f"Target: {TARGET_TOK_S:.0f} tok/s ({TARGET_MS_PER_TOKEN:.2f} ms/token)")
    print(f"{'wdtype':<8} {'batch':<6} {'mlp_ms':<10} {'x64_mlp_ms':<12} {'pcc':<8} {'floor_ms':<10} {'GO?':<6}")
    print("-" * 72)
    go_precisions = set()
    for (wdtype, batch), row in sorted(results.items(), key=lambda kv: (WDTYPE_NAMES[kv[0][0]], kv[0][1])):
        floor = analytic_mlp_floor_ms(wdtype)
        x64 = row["mlp_ms"] * _N_LAYERS
        pcc = row.get("pcc", float("nan"))
        full_proj_ms = x64 / _MLP_PARAM_FRACTION if _MLP_PARAM_FRACTION > 0 else float("inf")
        tok_s = 1000.0 / full_proj_ms if full_proj_ms > 0 else 0.0
        go = pcc >= 0.99 and full_proj_ms <= TARGET_MS_PER_TOKEN and batch == 1
        if go:
            go_precisions.add(WDTYPE_NAMES[wdtype])
        print(
            f"{WDTYPE_NAMES[wdtype]:<8} {batch:<6} {row['mlp_ms']:<10.3f} {x64:<12.1f} "
            f"{pcc:<8.4f} {floor.full_model_ms:<10.2f} {'YES' if go else 'no':<6}"
        )
        if batch == 1:
            print(f"         -> projected full-model: {full_proj_ms:.1f} ms ({tok_s:.0f} tok/s)")
    print("-" * 72)
    if go_precisions:
        print(f"GO: G=4 pipeline viable at precision(s): {', '.join(sorted(go_precisions))}")
    else:
        print("NO-GO at G=4 for 70 tok/s with tested precisions (bf16 expected to fail floor).")
    print("=" * 72 + "\n")


def unshard_cols_from_packed(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """Inverse of _shard_cols_2d: [1, cols, M, H/cols] -> [1, 1, M, H]."""
    _, _, m, hpc = packed.shape
    full = torch.zeros(1, 1, m, hpc * cols, dtype=packed.dtype)
    for c in range(cols):
        full[0, 0, :, c * hpc : (c + 1) * hpc] = packed[0, c]
    return full


def evaluate_go_nogo(results: dict) -> tuple[bool, set[str]]:
    """Return (any_go, go_precisions) from measured {(wdtype,batch): {mlp_ms,pcc}}."""
    go_precisions: set[str] = set()
    for (wdtype, batch), row in results.items():
        if batch != 1:
            continue
        pcc = row.get("pcc", 0.0)
        x64 = row["mlp_ms"] * _N_LAYERS
        full_proj_ms = x64 / _MLP_PARAM_FRACTION
        if pcc >= 0.99 and full_proj_ms <= TARGET_MS_PER_TOKEN:
            go_precisions.add(WDTYPE_NAMES[wdtype])
    return bool(go_precisions), go_precisions
