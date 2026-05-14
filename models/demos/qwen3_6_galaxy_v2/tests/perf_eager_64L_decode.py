# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-perf-eager — full 64-layer decode eager wall-clock measurement.

Adapts ``test_decode_eager_64L_pcc.py`` to skip PCC and time only:
  * one prefill T=128 (seeds KV/DeltaNet state)
  * multiple decode T=1 steps (mean/std)

PCC may fail (known 64L decode numerical bug being chased elsewhere), but
wall-clock latency is meaningful since the same ops execute.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/perf_eager_64L_decode.py \\
            -v -s
"""
from __future__ import annotations

import json
import pathlib
import statistics
import time

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

_T_PREFILL = 128
_N_LAYERS = 64
_H = 5120
_N_WARMUP_DECODE = 1
_N_RUNS_DECODE = 5


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


def _load_state_dict_all_layers(snapshot_dir):
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        "model.language_model.layers.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _build_tt_model(mesh, state_dict):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = _N_LAYERS
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    return model, args


def _build_partial_rope_cos_sin_tt(mesh, positions):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    cos_tt = ttnn.from_torch(
        cos_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


def _send_col_sharded_hidden(t, mesh, args):
    B, T, H = t.shape
    t_4d = t.reshape(1, 1, T, H)
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


@pytest.mark.hardware
def test_perf_64L_decode_eager(bh_glx_mesh):
    """Measure prefill + decode wall-clock for full 64L model (eager mode)."""
    state_dict = _load_state_dict_all_layers(_SNAPSHOT)
    print(f"[64L-decode-perf] loaded {len(state_dict)} weights")

    model, args = _build_tt_model(bh_glx_mesh, state_dict)
    print(f"[64L-decode-perf] TT 64L model built")

    torch.manual_seed(44)
    x_prefill_cpu = torch.randn(1, _T_PREFILL, args.dim, dtype=torch.bfloat16)
    x_decode_cpu = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)

    # --- Prefill (seeds KV/DeltaNet state) ---
    x_prefill_tt = _send_col_sharded_hidden(x_prefill_cpu, bh_glx_mesh, args)
    cos_pref, sin_pref = _build_partial_rope_cos_sin_tt(bh_glx_mesh, torch.arange(_T_PREFILL, dtype=torch.long))
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    t0 = time.perf_counter()
    _ = model.forward(
        x_prefill_tt,
        current_pos=None,
        rot_mats=(cos_pref, sin_pref),
        user_id=0,
        mode="prefill",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"[64L-decode-perf] PREFILL (T={_T_PREFILL}): {prefill_ms:.2f} ms")

    # --- Decode (timed) ---
    cos_dec, sin_dec = _build_partial_rope_cos_sin_tt(bh_glx_mesh, torch.tensor([_T_PREFILL], dtype=torch.long))

    def _decode_once():
        x_dec_tt = _send_col_sharded_hidden(x_decode_cpu, bh_glx_mesh, args)
        out = model.forward(
            x_dec_tt,
            current_pos=_T_PREFILL,
            rot_mats=(cos_dec, sin_dec),
            user_id=0,
            mode="decode",
            page_table=None,
            chunk_page_table=None,
            chunk_start_idx=None,
            start_pos=0,
            get_last_token=-1,
            kv_cache=None,
            batch_size=1,
        )
        # Force sync via host pull.
        if isinstance(out, list):
            _ = ttnn.to_torch(
                out[0],
                mesh_composer=ttnn.ConcatMesh2dToTensor(bh_glx_mesh, dims=(3, 0), mesh_shape=args.cluster_shape),
            )
        else:
            _ = ttnn.to_torch(
                out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(bh_glx_mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
            )

    # Warmup
    for i in range(_N_WARMUP_DECODE):
        t0 = time.perf_counter()
        _decode_once()
        dt = (time.perf_counter() - t0) * 1000
        print(f"[64L-decode-perf] DECODE WARMUP {i}: {dt:.2f} ms")

    # Timed
    latencies_ms = []
    for i in range(_N_RUNS_DECODE):
        t0 = time.perf_counter()
        _decode_once()
        dt = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt)
        print(f"[64L-decode-perf] DECODE RUN {i}: {dt:.2f} ms")

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    tok_s = 1000.0 / mean_ms
    print(
        f"\n[64L-decode-perf] SUMMARY DECODE T=1: "
        f"mean = {mean_ms:.2f} ms  std = {std_ms:.2f} ms  "
        f"tok/s/user = {tok_s:.2f}"
    )
    print(f"[64L-decode-perf] PREFILL T={_T_PREFILL}: {prefill_ms:.2f} ms")
    print(f"[64L-decode-perf] RAW DECODE: {latencies_ms}")
