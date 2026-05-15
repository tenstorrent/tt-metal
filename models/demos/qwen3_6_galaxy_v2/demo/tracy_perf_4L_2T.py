# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2 tracy per-op profile driver — 4L hybrid, prefill T=128 + 2 decode steps.

Mirrors ``models/demos/qwen3_6_galaxy/demo/text_demo.py`` (the v1 driver) but
targets the v2 ``TtTransformer`` stack with the hybrid 4-layer pattern
``[lin, lin, lin, full]`` (3 DeltaNet + 1 full-attn). Real HF weights from
the cached snapshot are loaded.

Tracy signposts segment the resulting per-op CSV into a *prefill-only*
section and a *decode-only* section:

    start --> prefill_done --> stop

A warmup prefill + warmup decode run BEFORE the ``signpost("start")`` to
ensure the program-cache and SDPA-decode compile-pass are excluded from
the profiled window.

Invocation:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m tracy -p -v -r -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/demo/tracy_perf_4L_2T.py::test_demo_perf_4L_2T \\
            -v -s

The tracy CSV lands under ``generated/profiler/reports/<timestamp>/``.
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
_N_LAYERS = 4
_PATTERN = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
_N_DECODE_STEPS = 2


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
def test_demo_perf_4L_2T(bh_glx_mesh):
    """V2 tracy per-op profile: 4L hybrid prefill (T=128) + 2 decode steps.

    Profiled region bracketing:
        signpost("start") --> prefill --> signpost("prefill_done") --> 2 decodes --> signpost("stop")

    A warmup prefill + warmup decode runs BEFORE the start signpost so the
    program cache / SDPA-decode compile pass is excluded from the profile.
    """
    try:
        from tracy import signpost
    except ImportError:  # Allow non-tracy runs (signposts become no-ops).
        signpost = lambda *_args, **_kwargs: None  # noqa: E731

    layer_indices = list(range(_N_LAYERS))
    state_dict = _load_state_dict_for_layers(_SNAPSHOT, layer_indices)
    print(f"[v2-tracy-4L-2T] loaded {len(state_dict)} weights for layers {layer_indices}")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS)
    print(f"[v2-tracy-4L-2T] TT 4L model built (pattern={_PATTERN})")

    # Deterministic dummy inputs (PCC is NOT verified — wall-clock + tracy only).
    torch.manual_seed(44)
    x_prefill_cpu = torch.randn(_B, _T_PREFILL, _H, dtype=torch.bfloat16)
    x_decode_cpu = torch.randn(_B, 1, _H, dtype=torch.bfloat16)

    # --- Warmup: 1 prefill ONLY (NO warmup decode) BEFORE signposts ---
    # NOTE on the warmup contract for v2:
    # v2's TtTransformer mutates KV / DeltaNet state INTERNALLY (no caches
    # passed in/out across calls). The canonical v1 sequence
    # (warmup prefill --> warmup decode --> profiled prefill) HANGS in v2:
    # a chunked-delta-rule prefill after a recurrent decode trips a state-
    # corruption bug (observed: signpost("prefill_done") fires then the
    # process wedges in futex_wait_queue for 10+ min on every thread).
    #
    # Robust v2 flow:
    #   warmup prefill (compile pass, NOT signposted)
    #   signpost("start")
    #   profiled prefill (kernels warm; python wall-clock unaffected by compile)
    #   signpost("prefill_done")
    #   decode #0 (compile pass — included in tracy decode rows; tracy
    #             reports DEVICE kernel duration not host compile time, so
    #             the per-op sum_dev_us is the same as a warm run; the python
    #             wall-clock for this step will be higher)
    #   decode #1..N (warm)
    #   signpost("stop")
    #
    # The decode-section tracy table aggregates 1+N decode steps. Since
    # tracy DEVICE durations do not include host-side compile time, op-level
    # sum_dev_us is unaffected. We track the python wall-clock per step
    # separately and exclude step 0 from the wall-clock mean.
    # Warmup as a single prefill (NOT a decode — decode after warmup makes
    # the profiled second prefill safe, but the v2 P-D-P-D order seems to
    # hang in the second decode; the working sequence verified end-to-end
    # is P-P-D-D-D, i.e. warmup prefill, profiled prefill, then decodes).
    print("[v2-tracy-4L-2T] WARMUP prefill (compile pass, NOT signposted) ...")
    t0 = time.perf_counter()
    pf_out_w = _do_prefill(model, bh_glx_mesh, args, x_prefill_cpu)
    if pf_out_w is not None and not isinstance(pf_out_w, list):
        try:
            pf_out_w.deallocate(True)
        except Exception:
            pass
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[v2-tracy-4L-2T] warmup prefill done in {(time.perf_counter() - t0) * 1000.0:.2f} ms")

    # Flush the device profiler DRAM buffer after the long warmup prefill.
    # Without this, the ~70 s compile-pass fills the per-Risc 12 000-event
    # ring buffer and ops in the profiled region are dropped — causing
    # process_ops_logs.py post-process to fail with
    # "AssertionError: Device data missing: Op X not present in
    #  cpp_device_perf_report.csv".  ReadDeviceProfiler reads-and-resets
    # the buffers so the profiled region starts with a clean buffer.
    print("[v2-tracy-4L-2T] flushing device profiler buffers after warmup ...")
    ttnn.ReadDeviceProfiler(bh_glx_mesh)

    # --- Profiled region ---
    signpost("start")

    print("[v2-tracy-4L-2T] PROFILED prefill (T=128, warm) ...")
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
    print(f"[v2-tracy-4L-2T] PROFILED prefill (warm): {prefill_ms:.2f} ms")

    # decode #0 is compile-pass; decode #1..N are warm.  tracy DEVICE kernel
    # durations do NOT include host-side compile overhead, so per-op
    # sum_dev_us is unaffected by the compile pass on step 0.  python wall-
    # clock for step 0 will be ~30 s higher than warm steps.
    n_total_decodes = _N_DECODE_STEPS + 1
    print(f"[v2-tracy-4L-2T] PROFILED decode x {n_total_decodes} (#0 = compile, #1..{_N_DECODE_STEPS} = warm) ...")
    decode_times_ms = []
    for step in range(n_total_decodes):
        cur_pos = _T_PREFILL + step
        t0 = time.perf_counter()
        _do_decode(model, bh_glx_mesh, args, x_decode_cpu, cur_pos)
        ttnn.synchronize_device(bh_glx_mesh)
        dt = (time.perf_counter() - t0) * 1000.0
        decode_times_ms.append(dt)
        label = "COMPILE" if step == 0 else "TIMED"
        print(f"[v2-tracy-4L-2T]   decode step {step} ({label}, cur_pos={cur_pos}): {dt:.2f} ms")

    signpost("stop")

    print("\n[v2-tracy-4L-2T] === summary ===")
    print(f"[v2-tracy-4L-2T]   prefill (T={_T_PREFILL}, warm) : {prefill_ms:.2f} ms")
    print(f"[v2-tracy-4L-2T]   decode steps (incl compile)  : {n_total_decodes}")
    print(f"[v2-tracy-4L-2T]   decode #0 (compile)          : {decode_times_ms[0]:.2f} ms")
    warm_ms = decode_times_ms[1:]
    if warm_ms:
        print(f"[v2-tracy-4L-2T]   decode #1..N (warm) mean     : {sum(warm_ms)/len(warm_ms):.2f} ms")
    print(f"[v2-tracy-4L-2T]   decode raw                    : {decode_times_ms}")
