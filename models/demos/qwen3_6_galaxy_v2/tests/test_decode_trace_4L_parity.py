# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-9 — 4L decode trace-capture + speedup on BH GLX 8x4.

Builds the 4L hybrid (``[lin, lin, lin, full]``) TtTransformer that
``test_decode_eager_pcc.py`` verifies at PCC > 0.99, runs prefill to seed
the KV cache + DeltaNet state, then:

  1. Runs a compile-pass eager decode at cur_pos=T_PREFILL → eager_logits.
     Wall-clock recorded as ``eager_ms`` (includes the SDPA-decode compile
     hit on the FIRST call — second decode would be ~25% faster but the
     v1/v2 PERF.md eager baseline is the cold first-call latency).
  2. Refreshes the SDPA mask buffer for cur_pos=T_PREFILL OUTSIDE the trace
     boundary, enters trace-decode mode (skips inline mask refresh), and
     captures the trace at the SAME cur_pos.
  3. Replays the trace once → traced_logits.  Compares argmax token and
     reports PCC vs eager.
  4. Times subsequent replays → traced_ms.  Reports speedup.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_trace_4L_parity.py \\
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
_N_LAYERS = 4
_PATTERN = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
_PCC_PARITY = 0.99  # relaxed; bit-identical replay would be 0.9999 but eager+traced may differ slightly
_TIMING_REPS = 5


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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_tt_model(mesh, state_dict, pattern: list[str], n_layers: int):
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


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
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


def _do_prefill(model, mesh, args, x_prefill: torch.Tensor):
    x_tt = _send_col_sharded_hidden(x_prefill, mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(mesh, torch.arange(x_prefill.shape[1], dtype=torch.long))
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


def _do_decode(model, x_decode_tt, cos_tt, sin_tt, cur_pos: int):
    return model.forward(
        x_decode_tt,
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


def _gather_logits_to_cpu(tt_logits_list, mesh, args):
    out0 = tt_logits_list[0] if isinstance(tt_logits_list, list) else tt_logits_list
    logits_torch = ttnn.to_torch(
        out0,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(3, 0), mesh_shape=args.cluster_shape),
    )
    n_cols = args.cluster_shape[1]
    logits_torch = logits_torch[: logits_torch.shape[0] // n_cols]
    while logits_torch.dim() > 3 and logits_torch.shape[0] == 1:
        logits_torch = logits_torch.squeeze(0)
    if logits_torch.dim() == 3:
        logits_decode = logits_torch[:, 0:1, : args.vocab_size]
    else:
        logits_decode = logits_torch[..., : args.vocab_size]
    return logits_decode


def _run_trace_parity(bh_glx_mesh, n_layers: int, pattern: list[str], pcc_threshold: float, label: str):
    """Build N-layer model, prefill once, eager decode @ T_PREFILL, then capture
    and replay the trace at the SAME cur_pos.  KV/DN state is mutated by the
    eager step; the trace decode runs at the SAME cur_pos which writes into
    the SAME cache slot (overwriting the eager step's writes — that's
    deterministically equivalent because the inputs match).

    Returns ``(eager_ms, traced_ms, parity_pcc)``.
    """
    layer_indices = list(range(n_layers))
    state_dict = _load_state_dict_for_layers(_SNAPSHOT, layer_indices)
    print(f"[V2-9 trace {label}] loaded {len(state_dict)} weights for layers {layer_indices}")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, pattern, n_layers)
    print(f"[V2-9 trace {label}] TT {n_layers}-layer model built (pattern len={len(pattern)})")

    torch.manual_seed(44)
    T_full = _T_PREFILL + 1
    x_full = torch.randn(_B, T_full, _H, dtype=torch.bfloat16)
    x_prefill = x_full[:, :_T_PREFILL, :]
    x_decode_torch = x_full[:, _T_PREFILL : _T_PREFILL + 1, :]

    # ---- (A) Prefill (seeds KV+DN state) ----
    pf_out = _do_prefill(model, bh_glx_mesh, args, x_prefill)
    if pf_out is not None and not isinstance(pf_out, list):
        # prefill output (col-sharded hidden) — not needed.
        try:
            pf_out.deallocate(True)
        except Exception:
            pass
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[V2-9 trace {label}] prefill done (T={_T_PREFILL}), KV/DN seeded")

    # ---- (B) Eager decode @ _T_PREFILL ----
    # NB: this is the COMPILE pass for the decode SDPA + DeltaNet recurrent
    # kernels; the first call is slow.  Subsequent decode calls (and trace
    # replays) reuse the cached programs.
    x_decode_tt_eager = _send_col_sharded_hidden(x_decode_torch, bh_glx_mesh, args)
    cos_tt_de, sin_tt_de = _build_partial_rope_cos_sin_tt(bh_glx_mesh, torch.tensor([_T_PREFILL], dtype=torch.long))
    model.set_trace_decode_mode(False)
    ttnn.synchronize_device(bh_glx_mesh)
    t0 = time.perf_counter()
    eager_out = _do_decode(model, x_decode_tt_eager, cos_tt_de, sin_tt_de, _T_PREFILL)
    ttnn.synchronize_device(bh_glx_mesh)
    eager_ms = (time.perf_counter() - t0) * 1000.0
    eager_logits_cpu = _gather_logits_to_cpu(eager_out, bh_glx_mesh, args)
    eager_logits_flat = eager_logits_cpu.reshape(-1)[: args.vocab_size].float().clone()
    pred_eager = int(eager_logits_flat.argmax().item())
    print(f"[V2-9 trace {label}] eager decode ms = {eager_ms:.2f}, pred = {pred_eager}")
    # Dealloc the eager output to free L1/DRAM slots before trace capture.
    if isinstance(eager_out, list):
        for t in eager_out:
            try:
                t.deallocate(True)
            except Exception:
                pass
    try:
        x_decode_tt_eager.deallocate(True)
    except Exception:
        pass

    # ---- (C) Refresh mask buffer + enter trace mode BEFORE begin_trace_capture ----
    # (The eager call at the same cur_pos already wrote into the SAME mask buf
    # via the in-forward refresh, so this is a no-op for trace correctness,
    # but it's the explicit contract.)
    model.refresh_decode_per_step_buffers(_T_PREFILL)
    model.set_trace_decode_mode(True)
    if hasattr(model, "tt_ccl") and hasattr(model.tt_ccl, "reset_gather_and_buffer_idx"):
        model.tt_ccl.reset_gather_and_buffer_idx()
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[V2-9 trace {label}] mask buf primed; entering trace mode")

    # Fresh input buffer for the trace.  Its address is bound into the trace
    # so subsequent execute_trace calls re-read this same address.
    x_decode_tt_capture = _send_col_sharded_hidden(x_decode_torch, bh_glx_mesh, args)

    trace_id = None
    parity_pcc = None
    traced_ms = float("nan")
    try:
        trace_id = ttnn.begin_trace_capture(bh_glx_mesh, cq_id=0)
        traced_out = _do_decode(model, x_decode_tt_capture, cos_tt_de, sin_tt_de, _T_PREFILL)
        ttnn.end_trace_capture(bh_glx_mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(bh_glx_mesh)
        print(f"[V2-9 trace {label}] capture SUCCESS — trace_id = {trace_id}")

        # First replay → PCC parity check.
        ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(bh_glx_mesh)
        traced_logits_cpu = _gather_logits_to_cpu(traced_out, bh_glx_mesh, args)
        traced_logits_flat = traced_logits_cpu.reshape(-1)[: args.vocab_size].float()
        pred_traced = int(traced_logits_flat.argmax().item())
        parity_pcc = _pcc(eager_logits_flat, traced_logits_flat)
        print(
            f"[V2-9 trace {label}] traced pred = {pred_traced} (eager pred = {pred_eager}, "
            f"argmax match = {pred_traced == pred_eager})"
        )
        print(f"[V2-9 trace {label}] eager-vs-traced PCC = {parity_pcc:.6f}")

        # Time replays.
        ttnn.synchronize_device(bh_glx_mesh)
        t0 = time.perf_counter()
        for _ in range(_TIMING_REPS):
            ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(bh_glx_mesh)
        traced_ms = ((time.perf_counter() - t0) * 1000.0) / _TIMING_REPS
        print(f"[V2-9 trace {label}] mean traced decode ms (over {_TIMING_REPS} reps) = {traced_ms:.2f}")

        return eager_ms, traced_ms, parity_pcc
    finally:
        model.set_trace_decode_mode(False)
        if trace_id is not None:
            try:
                ttnn.release_trace(bh_glx_mesh, trace_id)
            except Exception as e:
                print(f"[V2-9 trace {label}] release_trace failed (ignored): {e}")


@pytest.mark.hardware
def test_qwen36_4_layer_decode_trace_parity(bh_glx_mesh):
    """4L trace capture + parity + speedup vs eager."""
    eager_ms, traced_ms, parity_pcc = _run_trace_parity(bh_glx_mesh, _N_LAYERS, _PATTERN, _PCC_PARITY, "4L")
    speedup = eager_ms / max(traced_ms, 1e-9)
    print(f"[V2-9 trace 4L] eager={eager_ms:.2f} ms, traced={traced_ms:.2f} ms, speedup={speedup:.2f}x")
    assert parity_pcc is not None and parity_pcc >= _PCC_PARITY, (
        f"4L trace parity drift: PCC {parity_pcc} < {_PCC_PARITY}; "
        f"eager_ms={eager_ms:.2f}, traced_ms={traced_ms:.2f}"
    )
    print(f"[V2-9 trace 4L] PASSED")
