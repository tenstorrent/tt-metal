# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-tracy-4 driver — 1L DeltaNet-only, prefill T=128 + 2 decode steps.

Fork of ``tracy_perf_4L_4T.py`` (V2-tracy-3 baseline) with two changes:

1. ``n_layers = 1`` (was 4): isolate a single decoder block so the
   per-op profile is attributable to ONE block type only.
2. ``linear_attention_pattern = ["linear_attention"]``: this 1L slot
   becomes a DeltaNet (a.k.a. *linear-attention*) block.  The companion
   driver ``tracy_perf_1L_fullattn.py`` uses
   ``["full_attention"]`` so the two captures can be compared.

V2-tracy-3 used the mixed 4L pattern ``[lin, lin, lin, full]`` which
averages both block types together — users cannot tell which costs
come from DeltaNet vs full-attention.  V2-tracy-4 separates them.

Default-off baseline: NO ``QWEN36_*`` env vars set.  Same env contract
as V2-tracy-3.

Invocation:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m tracy -p -v -r -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/demo/tracy_perf_1L_delta.py \\
            -v -s
"""
from __future__ import annotations

import json
import pathlib
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
_N_LAYERS = 1
_PATTERN = ["linear_attention"]
_N_DECODE_STEPS = 2  # V2-tracy-4: 2 warm decode steps (matches task spec)
# Layer 0 is DeltaNet in the hybrid pattern, so it has the matching weights
# already cached under qwen3_6_galaxy_v2.
_LAYER_IDX = 0


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


def _load_state_dict_for_layers(snapshot_dir: pathlib.Path, layer_indices: list[int]) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
    ] + [f"model.language_model.layers.{i}." for i in layer_indices]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _build_tt_model(mesh, state_dict, pattern, n_layers):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = n_layers
    args.linear_attention_pattern = pattern
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


def _build_partial_rope_cos_sin_tt(mesh, positions: torch.Tensor):
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


def _do_prefill(model, mesh, args, x_prefill_cpu):
    x_tt = _send_col_sharded_hidden(x_prefill_cpu, mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(mesh, torch.arange(x_prefill_cpu.shape[1], dtype=torch.long))
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return model.forward(
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


def _do_decode(model, mesh, args, x_decode_cpu, cur_pos):
    x_dec_tt = _send_col_sharded_hidden(x_decode_cpu, mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(mesh, torch.tensor([cur_pos], dtype=torch.long))
    out = model.forward(
        x_dec_tt,
        current_pos=cur_pos,
        rot_mats=(cos_tt, sin_tt),
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
    # Force device sync via host pull (lm_head returns list[Tensor]).
    if isinstance(out, list):
        _ = ttnn.to_torch(
            out[0],
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(3, 0), mesh_shape=args.cluster_shape),
        )
    else:
        _ = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
        )


@pytest.mark.hardware
def test_demo_perf_1L_delta(bh_glx_mesh):
    """V2-tracy-4 per-op profile: 1L DeltaNet prefill (T=128) + 2 decode steps.

    Isolates DeltaNet device-time attribution by running a single
    linear-attention layer between the embedding and lm_head.

    Profiled region bracketing:
        signpost("start") --> prefill --> signpost("prefill_done")
            --> decodes (compile + warm) --> signpost("stop")
    """
    try:
        from tracy import signpost
    except ImportError:  # Allow non-tracy runs (signposts become no-ops).
        signpost = lambda *_args, **_kwargs: None  # noqa: E731

    import os as _os

    flags = {
        "QWEN36_RESIDUAL_BUF_ON": _os.environ.get("QWEN36_RESIDUAL_BUF_ON", "0"),
        "QWEN36_DELTA_LAR": _os.environ.get("QWEN36_DELTA_LAR", "0"),
        "QWEN36_FULLATTN_LAR": _os.environ.get("QWEN36_FULLATTN_LAR", "0"),
        "QWEN36_TT_LANG_BETA_G": _os.environ.get("QWEN36_TT_LANG_BETA_G", "0"),
    }
    print(f"[v2-tracy-1L-delta] env flag snapshot: {flags}")

    layer_indices = [_LAYER_IDX]
    state_dict = _load_state_dict_for_layers(_SNAPSHOT, layer_indices)
    print(f"[v2-tracy-1L-delta] loaded {len(state_dict)} weights for layers {layer_indices}")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS)
    print(f"[v2-tracy-1L-delta] TT 1L model built (pattern={_PATTERN})")

    # Deterministic dummy inputs.
    torch.manual_seed(44)
    x_prefill_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)
    x_decode_cpu = torch.randn(_B, 1, _H, dtype=torch.bfloat16)

    # --- Warmup: 1 prefill ONLY BEFORE signposts (compile pass) ---
    print("[v2-tracy-1L-delta] WARMUP prefill (compile pass, NOT signposted) ...")
    t0 = time.perf_counter()
    pf_out_w = _do_prefill(model, bh_glx_mesh, args, x_prefill_cpu)
    if pf_out_w is not None and not isinstance(pf_out_w, list):
        try:
            pf_out_w.deallocate(True)
        except Exception:
            pass
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[v2-tracy-1L-delta] warmup prefill done in {(time.perf_counter() - t0) * 1000.0:.2f} ms")

    # Flush the device profiler DRAM buffer after the long warmup prefill.
    print("[v2-tracy-1L-delta] flushing device profiler buffers after warmup ...")
    ttnn.ReadDeviceProfiler(bh_glx_mesh)

    # --- Profiled region ---
    signpost("start")

    print("[v2-tracy-1L-delta] PROFILED prefill (T=128, warm) ...")
    t0 = time.perf_counter()
    pf_out = _do_prefill(model, bh_glx_mesh, args, x_prefill_cpu)
    if pf_out is not None and not isinstance(pf_out, list):
        try:
            pf_out.deallocate(True)
        except Exception:
            pass
    ttnn.synchronize_device(bh_glx_mesh)
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    signpost("prefill_done")
    print(f"[v2-tracy-1L-delta] PROFILED prefill (warm): {prefill_ms:.2f} ms")

    n_total_decodes = _N_DECODE_STEPS + 1  # +1 compile-pass slot
    print(f"[v2-tracy-1L-delta] PROFILED decode x {n_total_decodes} (#0 = compile, #1..{_N_DECODE_STEPS} = warm) ...")
    decode_times_ms = []
    for step in range(n_total_decodes):
        cur_pos = _T_PREFILL + step
        # Bracket only the WARM steps with decode_warm_start / stop so the
        # per-op tracy analysis can isolate steady-state device kernel time.
        if step == 1:
            signpost("decode_warm_start")
        t0 = time.perf_counter()
        _do_decode(model, bh_glx_mesh, args, x_decode_cpu, cur_pos)
        ttnn.synchronize_device(bh_glx_mesh)
        dt = (time.perf_counter() - t0) * 1000.0
        decode_times_ms.append(dt)
        label = "COMPILE" if step == 0 else "TIMED"
        print(f"[v2-tracy-1L-delta]   decode step {step} ({label}, cur_pos={cur_pos}): {dt:.2f} ms")

    signpost("stop")

    print("\n[v2-tracy-1L-delta] === summary ===")
    print(f"[v2-tracy-1L-delta]   prefill (T={_T_PREFILL}, warm) : {prefill_ms:.2f} ms")
    print(f"[v2-tracy-1L-delta]   decode steps (incl compile)  : {n_total_decodes}")
    print(f"[v2-tracy-1L-delta]   decode #0 (compile)          : {decode_times_ms[0]:.2f} ms")
    warm_ms = decode_times_ms[1:]
    if warm_ms:
        print(f"[v2-tracy-1L-delta]   decode #1..N (warm) mean     : {sum(warm_ms)/len(warm_ms):.2f} ms")
    print(f"[v2-tracy-1L-delta]   decode raw                    : {decode_times_ms}")
