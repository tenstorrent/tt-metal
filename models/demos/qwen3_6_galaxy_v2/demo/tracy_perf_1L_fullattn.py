# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-tracy-4 driver — 1L full-attention-only, prefill T=128 + 2 decode steps.

Fork of ``tracy_perf_4L_4T.py`` (V2-tracy-3 baseline) with two changes:

1. ``n_layers = 1`` (was 4): isolate a single decoder block.
2. ``linear_attention_pattern = ["full_attention"]``: this slot becomes
   a full-attention (softmax-attn + KV cache) block.  Companion driver
   ``tracy_perf_1L_delta.py`` runs the linear-attention variant.

V2-tracy-3 used the mixed 4L pattern ``[lin, lin, lin, full]`` which
averages both block types together.  V2-tracy-4 separates them so the
per-op tables are attributable to ONE block type only.

We load layer 3 weights (full_attention slot in the canonical
``[lin, lin, lin, full]`` x16 pattern) so the HF state-dict keys for
self_attn (q_proj / k_proj / v_proj / o_proj / q_norm / k_norm) line
up with what TtAttention expects.

Default-off baseline: NO ``QWEN36_*`` env vars set.

Invocation:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m tracy -p -v -r -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/demo/tracy_perf_1L_fullattn.py \\
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
_PATTERN = ["full_attention"]
_N_DECODE_STEPS = 2  # V2-tracy-4: 2 warm decode steps (matches task spec)
# Layer 3 is the full-attention slot in the canonical pattern.
_LAYER_IDX = 3


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
    """Load HF state_dict but RE-INDEX the chosen layer to layer 0 so
    TtTransformer (which only builds 1 decoder block) can find the keys.
    """
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
    # Re-index layer keys so the chosen layer index → 0.
    if layer_indices != [0]:
        remapped: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            new_k = k
            for src_idx, dst_idx in zip(layer_indices, range(len(layer_indices))):
                if src_idx == dst_idx:
                    continue
                src_prefix = f"model.language_model.layers.{src_idx}."
                dst_prefix = f"model.language_model.layers.{dst_idx}."
                if k.startswith(src_prefix):
                    new_k = dst_prefix + k[len(src_prefix) :]
                    break
            remapped[new_k] = v
        sd = remapped
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
def test_demo_perf_1L_fullattn(bh_glx_mesh):
    """V2-tracy-4 per-op profile: 1L full-attention prefill (T=128) + 2 decode steps."""
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *_args, **_kwargs: None  # noqa: E731

    import os as _os

    flags = {
        "QWEN36_RESIDUAL_BUF_ON": _os.environ.get("QWEN36_RESIDUAL_BUF_ON", "0"),
        "QWEN36_DELTA_LAR": _os.environ.get("QWEN36_DELTA_LAR", "0"),
        "QWEN36_FULLATTN_LAR": _os.environ.get("QWEN36_FULLATTN_LAR", "0"),
        "QWEN36_TT_LANG_BETA_G": _os.environ.get("QWEN36_TT_LANG_BETA_G", "0"),
    }
    print(f"[v2-tracy-1L-fullattn] env flag snapshot: {flags}")

    layer_indices = [_LAYER_IDX]
    state_dict = _load_state_dict_for_layers(_SNAPSHOT, layer_indices)
    print(f"[v2-tracy-1L-fullattn] loaded {len(state_dict)} weights for layers {layer_indices} (remapped to 0)")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS)
    print(f"[v2-tracy-1L-fullattn] TT 1L model built (pattern={_PATTERN})")

    torch.manual_seed(44)
    x_prefill_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)
    x_decode_cpu = torch.randn(_B, 1, _H, dtype=torch.bfloat16)

    # --- Warmup: 1 prefill BEFORE signposts (compile pass) ---
    print("[v2-tracy-1L-fullattn] WARMUP prefill (compile pass, NOT signposted) ...")
    t0 = time.perf_counter()
    pf_out_w = _do_prefill(model, bh_glx_mesh, args, x_prefill_cpu)
    if pf_out_w is not None and not isinstance(pf_out_w, list):
        try:
            pf_out_w.deallocate(True)
        except Exception:
            pass
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[v2-tracy-1L-fullattn] warmup prefill done in {(time.perf_counter() - t0) * 1000.0:.2f} ms")

    print("[v2-tracy-1L-fullattn] flushing device profiler buffers after warmup ...")
    ttnn.ReadDeviceProfiler(bh_glx_mesh)

    # --- Profiled region ---
    signpost("start")

    print("[v2-tracy-1L-fullattn] PROFILED prefill (T=128, warm) ...")
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
    print(f"[v2-tracy-1L-fullattn] PROFILED prefill (warm): {prefill_ms:.2f} ms")

    n_total_decodes = _N_DECODE_STEPS + 1
    print(
        f"[v2-tracy-1L-fullattn] PROFILED decode x {n_total_decodes} (#0 = compile, #1..{_N_DECODE_STEPS} = warm) ..."
    )
    decode_times_ms = []
    for step in range(n_total_decodes):
        cur_pos = _T_PREFILL + step
        t0 = time.perf_counter()
        _do_decode(model, bh_glx_mesh, args, x_decode_cpu, cur_pos)
        ttnn.synchronize_device(bh_glx_mesh)
        dt = (time.perf_counter() - t0) * 1000.0
        decode_times_ms.append(dt)
        label = "COMPILE" if step == 0 else "TIMED"
        print(f"[v2-tracy-1L-fullattn]   decode step {step} ({label}, cur_pos={cur_pos}): {dt:.2f} ms")

    signpost("stop")

    print("\n[v2-tracy-1L-fullattn] === summary ===")
    print(f"[v2-tracy-1L-fullattn]   prefill (T={_T_PREFILL}, warm) : {prefill_ms:.2f} ms")
    print(f"[v2-tracy-1L-fullattn]   decode steps (incl compile)  : {n_total_decodes}")
    print(f"[v2-tracy-1L-fullattn]   decode #0 (compile)          : {decode_times_ms[0]:.2f} ms")
    warm_ms = decode_times_ms[1:]
    if warm_ms:
        print(f"[v2-tracy-1L-fullattn]   decode #1..N (warm) mean     : {sum(warm_ms)/len(warm_ms):.2f} ms")
    print(f"[v2-tracy-1L-fullattn]   decode raw                    : {decode_times_ms}")
