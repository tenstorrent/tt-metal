# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-perf-eager — 1L DeltaNet prefill eager wall-clock measurement.

Builds the same 1-layer TtTransformer used by
``test_layer0_deltanet_forward_pcc.py`` but skips PCC verification and times
the forward call only.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/perf_eager_1L_prefill.py \\
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

_B = 1
_T_PREFILL = 128
_H = 5120
_LAYER_IDX = 0
_LAYER_TYPE = "linear_attention"
_N_WARMUP = 1
_N_RUNS = 5


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


def _load_state_dict_for_layer(snapshot_dir, layer_idx):
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
    sd = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _build_tt_model(mesh, state_dict, layer_idx, layer_type):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = [layer_type]
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


def _build_partial_rope_cos_sin_tt(mesh, T):
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
def test_perf_1L_prefill_eager(bh_glx_mesh):
    """Measure eager wall-clock of 1-layer DeltaNet prefill at T=128."""
    state_dict = _load_state_dict_for_layer(_SNAPSHOT, _LAYER_IDX)
    print(f"[1L-prefill-perf] loaded {len(state_dict)} weights")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, _LAYER_IDX, _LAYER_TYPE)
    print(f"[1L-prefill-perf] TT 1L model built")

    torch.manual_seed(42)
    x_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)

    # Build inputs once; we re-use them every run (eager, no trace).
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(bh_glx_mesh, _T_PREFILL)
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )

    def _run_once():
        x_tt = _send_col_sharded_hidden(x_cpu, bh_glx_mesh, args)
        out = model.forward(
            x_tt,
            current_pos=None,
            rot_mats=(cos_tt, sin_tt),
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
        # Force-finish the device work: pull a tile to host.
        _ = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(bh_glx_mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
        )

    # --- Warmup ---
    for i in range(_N_WARMUP):
        t0 = time.perf_counter()
        _run_once()
        dt = (time.perf_counter() - t0) * 1000
        print(f"[1L-prefill-perf] WARMUP {i}: {dt:.2f} ms")

    # --- Timed runs ---
    latencies_ms = []
    for i in range(_N_RUNS):
        t0 = time.perf_counter()
        _run_once()
        dt = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt)
        print(f"[1L-prefill-perf] RUN {i}: {dt:.2f} ms")

    mean_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    ms_per_token = mean_ms / _T_PREFILL
    print(
        f"\n[1L-prefill-perf] SUMMARY T={_T_PREFILL}: "
        f"mean = {mean_ms:.2f} ms  std = {std_ms:.2f} ms  "
        f"ms/token = {ms_per_token:.3f}"
    )
    print(f"[1L-prefill-perf] RAW: {latencies_ms}")
