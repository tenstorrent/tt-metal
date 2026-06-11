# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-DP prefill KV-population unit test (1-layer full_attention, paged).

Isolates the bug behind the batch-32 decode layer-3 explosion: the full-model
decode SDPA reads GARBAGE KV for users 1..31 (only user 0 is valid), so the
hidden explodes at the first full-attention layer. The decode COMPUTE path runs
(fused rs_create_heads etc.), so the suspect is the PREFILL KV POPULATION:
``_forward_prefill_qwen36`` writes user u's KV via ``paged_fill_cache(batch_idx=u)``
with a REPLICATED page_table, and the decode reads it back via a COLUMN-sharded
page_table.

This test prefills N IDENTICAL users (loop user_id=0..N-1, same x) into a 1-layer
full_attention model, then reads back the on-device KV cache and asserts that the
blocks belonging to EACH user (page_table[u]) hold the SAME K/V as user 0's
blocks. If users 1..N-1's blocks are zero/garbage, the population is the bug.

Run (fast — 1 layer):
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate \\
      && QWEN36_KVPOP_USERS=32 python -m pytest --noconftest -v -s \\
         models/demos/qwen3_6_galaxy_v2/tests/test_batchdp_prefill_kv_population.py
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_T_PREFILL = int(os.environ.get("QWEN36_KVPOP_T", "128"))
_N_USERS = int(os.environ.get("QWEN36_KVPOP_USERS", "32"))
_LAYER_IDX = 3
_H = 5120
_PAGED_BLOCK_SIZE = 32
# Enough blocks for 32 users × ceil(T/block) + slack, reshaped [max_batch, blocks/user].
_BLOCKS_PER_USER = max(4, (_T_PREFILL + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE + 1)
_PAGED_MAX_NUM_BLOCKS = 32 * _BLOCKS_PER_USER


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _load_state_dict_for_layer(snapshot_dir: pathlib.Path, layer_idx: int) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        f"model.language_model.layers.{layer_idx}.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _relabel_layer_idx(sd: dict, src_idx: int, dst_idx: int) -> dict:
    src_prefix = f"model.language_model.layers.{src_idx}."
    dst_prefix = f"model.language_model.layers.{dst_idx}."
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        out[(dst_prefix + k[len(src_prefix) :]) if k.startswith(src_prefix) else k] = v
    return out


def _build_tt_model_paged(mesh, state_dict, paged_attention_config, max_batch):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = ["full_attention"]
    if max_batch != 1:
        args.max_batch_size = max_batch
        args.tile_padded_batch_rows = args.tile_size * int(math.ceil(max_batch / args.tile_size))
        if getattr(args, "num_device_groups", 0):
            args.batch_size_per_device_group = max(max_batch // args.num_device_groups, 1)
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,
    )
    return model, args


def _build_page_table(args, paged_attention_config):
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    return reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )


def _rope_tt(mesh, T: int):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    mk = lambda t: ttnn.from_torch(
        t.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return mk(cos_ref), mk(sin_ref)


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
    B, T, H = t.shape
    return ttnn.from_torch(
        t.reshape(1, 1, T, H),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


@pytest.mark.hardware
def test_batchdp_prefill_kv_population(bh_glx_mesh):
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    sd = _relabel_layer_idx(_load_state_dict_for_layer(_SNAPSHOT, _LAYER_IDX), _LAYER_IDX, 0)
    paged = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged(bh_glx_mesh, sd, paged, max_batch=_N_USERS)
    attn = model.layers[0].attention
    print(
        f"[kvpop] N_USERS={_N_USERS} T={_T_PREFILL} max_num_blocks={_PAGED_MAX_NUM_BLOCKS} "
        f"blocks/user={_PAGED_MAX_NUM_BLOCKS // args.max_batch_size}"
    )

    page_table_torch = _build_page_table(args, paged)
    page_table_tt = ttnn.from_torch(
        page_table_torch,
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(bh_glx_mesh, dims=(None, None), mesh_shape=args.cluster_shape),
    )
    cos_tt, sin_tt = _rope_tt(bh_glx_mesh, _T_PREFILL)

    # Identical prompt for every user.
    torch.manual_seed(43)
    x_cpu = torch.randn(1, _T_PREFILL, _H, dtype=torch.bfloat16)

    # --- PREFILL all N users (same x), routing each to user_id=u ---
    for u in range(_N_USERS):
        x_tt = _send_col_sharded_hidden(x_cpu, bh_glx_mesh, args)
        out = model.forward(
            x_tt,
            current_pos=None,
            rot_mats=(cos_tt, sin_tt),
            user_id=u,
            mode="prefill",
            page_table=page_table_tt,
            chunk_page_table=None,
            chunk_start_idx=ttnn.from_torch(
                torch.tensor([0], dtype=torch.int32),
                device=bh_glx_mesh,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
            ),
            start_pos=0,
            get_last_token=-1,
            kv_cache=None,
            batch_size=1,
        )
        if isinstance(out, (list, tuple)):
            out = out[0]
        ttnn.deallocate(out)
    print(f"[kvpop] prefilled {_N_USERS} identical users")

    # --- Read back KV cache from device shards. The cache is row-sharded on n_kv
    # (dims=(1,None)) and COLUMN-REPLICATED: devices 0,1,2,3 = (row0, col0..3) all
    # hold n_kv head 0 and MUST be identical if prefill replicated across columns.
    # The col-sharded decode reads users [8c:8c+8] from COLUMN c, so cols 1-3 must
    # also be populated — checking only dev0 (col0) MISSED this. n_real_blocks is
    # the per-user block count holding real tokens.
    keys_cache = attn.layer_past[0]
    n_real_blocks = (_T_PREFILL + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE  # blocks holding real tokens
    dev_shards = ttnn.get_device_tensors(keys_cache)

    def _read_col(col):  # device id of (row0, col) = col (row-major 8x4)
        kc = ttnn.to_torch(dev_shards[col]).float().reshape(_PAGED_MAX_NUM_BLOCKS, -1, _PAGED_BLOCK_SIZE, attn.head_dim)
        return kc

    k_cols = [_read_col(c) for c in range(4)]

    # CROSS-COLUMN check: user u's blocks must be IDENTICAL on every column (col-replicated).
    print(f"[kvpop] cross-column replication check (dev0..3 = col0..3, all n_kv head 0):")
    col_bad = []
    for c in range(1, 4):
        d = (k_cols[c] - k_cols[0]).abs().max().item()
        am = k_cols[c].abs().mean().item()
        print(f"[kvpop]   col{c}: absmean={am:.4f}  maxdiff_vs_col0={d:.4f}")
        if d > 1e-2:
            col_bad.append((c, d, am))
    assert not col_bad, (
        f"KV cache NOT column-replicated by prefill — cols {[c for c,_,_ in col_bad]} differ from col0. "
        f"Decode col c reads users [8c:8c+8] from col c, so unreplicated cols ⇒ garbage KV ⇒ layer-3 explosion."
    )

    k_dev0 = k_cols[0]

    def gather_user_k(u):
        blks = page_table_torch[u, :n_real_blocks].tolist()  # physical block ids for user u
        return torch.stack([k_dev0[b] for b in blks], dim=0)  # [n_real_blocks, 1, block, hd]

    k0 = gather_user_k(0)
    print(f"[kvpop] user0 cached-K absmean={k0.abs().mean():.4f} absmax={k0.abs().max():.4f}")
    bad = []
    for u in range(1, _N_USERS):
        ku = gather_user_k(u)
        d = (ku - k0).abs().max().item()
        am = ku.abs().mean().item()
        if u in (1, 7, 8, 15, 16, 31) or d > 1e-2:
            print(f"[kvpop] user{u:2d}: cached-K absmean={am:.4f} maxdiff_vs_u0={d:.4f}")
        if d > 1e-2:
            bad.append((u, d, am))
    assert not bad, f"{len(bad)} users have KV != user0 (population bug). first few: {bad[:6]}"
    print("[kvpop] PASS — all users' KV blocks populated identically")
