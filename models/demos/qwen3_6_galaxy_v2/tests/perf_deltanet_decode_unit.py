# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DeltaNet single-layer decode latency micro-benchmark (BH GLX 8×4).

Measures the end-to-end decode latency of one ``TtQwen36DeltaAttention``
block running on the production 32-device Galaxy mesh.  Acts as a
regression gate for DeltaNet decode optimizations:

  - DRAM-sharded w_qkvz + w_out (V2-DRAM-P2)
  - Fully-fused decode kernel (atupe `gdn_full_fused_inplace` pattern)
  - CCL fusion / reduction (out_proj all_reduce, entry-gather elimination)

The block accounts for **48 of 64** decoder layers and ~**71%** of decode
step time (per V2-tracy-4), so a per-layer improvement here scales to the
full model.

Approach
--------
Builds a full ``TtTransformer`` with ``n_layers=1`` and
``linear_attention_pattern=["linear_attention"]`` — the proven scaffold —
then reaches into ``model.layers[0].attention`` and times its
``forward(mode="decode")`` in isolation (no norm, no MLP, no residual,
no LM head).  The persistent ``dn_state_buffer`` / ``conv_state_buffer``
are exercised exactly as they would be during a real decode loop.

Modes
-----
- **Eager** (default): ``forward_decode`` N times with ``synchronize_device``
  between calls.  Use with Tracy for per-op breakdown.
- **Traced** (``QWEN36_PERF_TRACE=1``): capture trace once, replay N times.
  Closer to production traced perf.
- **With CCL** (``QWEN36_PERF_INCLUDE_CCL=1``): wrap the block in the
  col-sharded → full-H gather + full-H → col-sharded scatter pair that
  ``llama_decoder.py`` runs around DeltaNet in the V2-TP path.  Off by
  default so timing isolates the block.

Mesh
----
Always opens the production 8×4 BH Galaxy mesh.  This is the 32-device
setup; do not parameterize away from it (perf characteristics differ on
smaller meshes).

Usage
-----
    # default: 200 iters, eager, layer 0, random weights
    pytest models/demos/qwen3_6_galaxy_v2/tests/perf_deltanet_decode_unit.py -s

    # traced (closer to production)
    QWEN36_PERF_TRACE=1 pytest ... -s

    # include CCL gather/scatter around the block
    QWEN36_PERF_INCLUDE_CCL=1 pytest ... -s

    # 500 iters on layer 4 with real HF weights
    QWEN36_PERF_ITERS=500 QWEN36_PERF_LAYER=4 QWEN36_PERF_REAL_WEIGHTS=1 pytest ... -s

The test prints mean / stdev / min / max / p99 per-iteration latency
plus a projection to the full 48-DeltaNet-layer contribution to a decode
step.  Use it as a before/after gate for every DeltaNet decode change.
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import time

import pytest
import torch

import ttnn

# --- Tunables --------------------------------------------------------------
MESH_SHAPE = (8, 4)  # production BH GLX
_LAYER = int(os.environ.get("QWEN36_PERF_LAYER", "0"))
_ITERS = int(os.environ.get("QWEN36_PERF_ITERS", "200"))
_WARMUP = int(os.environ.get("QWEN36_PERF_WARMUP", "10"))
_USE_TRACE = os.environ.get("QWEN36_PERF_TRACE", "0") == "1"
_INCLUDE_CCL = os.environ.get("QWEN36_PERF_INCLUDE_CCL", "0") == "1"
_USE_REAL_WEIGHTS = os.environ.get("QWEN36_PERF_REAL_WEIGHTS", "0") == "1"
_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

# DeltaNet hyperparams (fixed per Qwen3.6-27B HF config).
_H = 5120  # hidden dim
_HD = 128  # DeltaNet head_dim
_NK = 16  # K heads
_NV = 48  # V heads
_CONV_K = 4
_QKV_DIM = _NK * _HD + _NK * _HD + _NV * _HD  # 10240
_Z_DIM = _NV * _HD  # 6144


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    # trace_region_size enables QWEN36_PERF_TRACE=1 (begin_trace_capture) +
    # clean per-op device profiling of the traced replay.
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE), trace_region_size=50_000_000)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _make_random_state_dict(layer_num: int, seed: int = 17) -> dict[str, torch.Tensor]:
    """Build a synthetic state_dict with the exact DeltaNet weight keys
    ``_extract_layer_dn_weights`` reads + the few non-DeltaNet entries
    ``TtTransformer.__init__`` accesses at layer-construction time.

    Small-scale random weights (std=0.02) keep the recurrence numerically
    stable; perf doesn't depend on weight values.
    """
    g = torch.Generator().manual_seed(seed + layer_num)

    def _r(shape, dtype=torch.bfloat16):
        return torch.empty(*shape, dtype=torch.float32).normal_(0.0, 0.02, generator=g).to(dtype)

    pfx = f"layers.{layer_num}.linear_attn."
    sd = {
        # DeltaNet block weights
        pfx + "in_proj_qkvz.weight": _r((_QKV_DIM + _Z_DIM, _H)),
        pfx + "in_proj_ba.weight": _r((2 * _NV, _H)),
        pfx + "conv1d.weight": _r((_QKV_DIM, 1, _CONV_K)),
        pfx + "A_log": torch.empty(_NV, dtype=torch.float32).normal_(0.0, 0.02, generator=g),
        pfx + "dt_bias": torch.empty(_NV, dtype=torch.float32).normal_(0.0, 0.02, generator=g),
        pfx + "norm.weight": torch.empty(_HD, dtype=torch.float32).normal_(1.0, 0.01, generator=g),
        pfx + "out_proj.weight": _r((_H, _NV * _HD)),
        # Required by the surrounding TtTransformer / decoder scaffolding
        # (norm + MLP + lm_head are built but never invoked in the timed path).
        f"layers.{layer_num}.attention_norm.weight": torch.ones(_H, dtype=torch.bfloat16) * 0.01,
        f"layers.{layer_num}.ffn_norm.weight": torch.ones(_H, dtype=torch.bfloat16) * 0.01,
        f"layers.{layer_num}.feed_forward.w1.weight": _r((17_408, _H)),
        f"layers.{layer_num}.feed_forward.w2.weight": _r((_H, 17_408)),
        f"layers.{layer_num}.feed_forward.w3.weight": _r((17_408, _H)),
        "tok_embeddings.weight": _r((248_320, _H)),
        "norm.weight": torch.ones(_H, dtype=torch.bfloat16) * 0.01,
        "output.weight": _r((248_320, _H)),
    }
    return sd


def _load_real_state_dict(layer_num: int) -> dict[str, torch.Tensor]:
    """Load just the layers needed (DeltaNet + norms + MLP for one layer +
    top-level embed/norm/output) from the HF snapshot."""
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        f"model.language_model.layers.{layer_num}.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    raw: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(_SNAPSHOT / fn))
        for k in needed_keys:
            if k in shard:
                raw[k] = shard[k]
    # Re-key to the layout the V2 loader expects (the model's
    # ``convert_hf_to_meta`` does this for the real boot path; we relabel
    # the small slice we need to layer 0).
    out: dict[str, torch.Tensor] = {}
    src = f"model.language_model.layers.{layer_num}."
    dst = f"layers.{layer_num}."
    for k, v in raw.items():
        if k.startswith(src):
            out[dst + k[len(src) :]] = v
        elif k.startswith("model.language_model.embed_tokens."):
            out["tok_embeddings." + k[len("model.language_model.embed_tokens.") :]] = v
        elif k.startswith("model.language_model.norm."):
            out["norm." + k[len("model.language_model.norm.") :]] = v
        elif k.startswith("lm_head."):
            out["output." + k[len("lm_head.") :]] = v
    return out


def _build_one_layer_model(mesh, layer_num: int):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = 1
    args.linear_attention_pattern = ["linear_attention"]

    state_dict = _load_real_state_dict(layer_num) if _USE_REAL_WEIGHTS else _make_random_state_dict(layer_num)

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


def _send_full_h(t_4d: torch.Tensor, mesh, args) -> ttnn.Tensor:
    """Full-H replicated layout (DeltaNet's forward_decode input contract).

    Expects ``t_4d`` already shaped [B, 1, T, H]; we forward as-is.
    """
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _send_col_sharded(t_4d: torch.Tensor, mesh, args) -> ttnn.Tensor:
    """Col-sharded H/4 layout (V2-TP residual-stream contract)."""
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _do_one_step(attention, x_input_tt, *, include_ccl: bool):
    """One DeltaNet decode step.

    When ``include_ccl`` is True, ``x_input_tt`` must be COL-SHARDED H/4 —
    we gather it to full-H, run the block, and scatter back to col-sharded
    (matching the production V2-TP decoder path).  Otherwise ``x_input_tt``
    must be FULL-H replicated.
    """
    if include_ccl:
        x_full = ttnn.all_gather(
            x_input_tt,
            dim=3,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_full = attention.forward(
            x_full,
            current_pos=0,
            rot_mats=None,
            user_id=0,
            mode="decode",
            page_table=None,
            kv_cache=None,
            batch_size=1,
        )
        ttnn.deallocate(x_full)
        out = ttnn.mesh_partition(out_full, dim=-1, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_full)
        return out
    return attention.forward(
        x_input_tt,
        current_pos=0,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        kv_cache=None,
        batch_size=1,
    )


def _summary(latencies_ms: list[float], label: str) -> None:
    mean = statistics.mean(latencies_ms)
    stdev = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    sorted_l = sorted(latencies_ms)
    p99 = sorted_l[int(0.99 * (len(latencies_ms) - 1))]
    p50 = sorted_l[len(latencies_ms) // 2]
    projected_48l_ms = mean * 48
    print(f"\n[perf-dn-unit][{label}] DeltaNet 1L decode latency  (n={len(latencies_ms)}):")
    print(f"  mean   = {mean:7.3f} ms")
    print(f"  median = {p50:7.3f} ms")
    print(f"  stdev  = {stdev:7.3f} ms")
    print(f"  min    = {min(latencies_ms):7.3f} ms")
    print(f"  max    = {max(latencies_ms):7.3f} ms")
    print(f"  p99    = {p99:7.3f} ms")
    print(f"  ---")
    print(f"  projected 48-layer DeltaNet contribution / step = {projected_48l_ms:7.2f} ms")
    print(f"  ---")


@pytest.mark.hardware
def test_perf_deltanet_decode_unit(bh_glx_mesh):
    """Measure DeltaNet decode step latency on the 8x4 BH GLX mesh.

    Reports mean/std/p99 wall-clock over ``_ITERS`` iterations after
    ``_WARMUP`` warm-up calls.  ``include_ccl`` controls whether the
    timing wraps the col-sharded↔full-H gather/scatter pair that surrounds
    DeltaNet in the V2-TP decoder.
    """
    torch.manual_seed(11)

    print(f"\n[perf-dn-unit] mesh={MESH_SHAPE}  layer={_LAYER}  iters={_ITERS}  warmup={_WARMUP}")
    print(f"[perf-dn-unit] trace={_USE_TRACE}  include_ccl={_INCLUDE_CCL}  " f"real_weights={_USE_REAL_WEIGHTS}")

    model, args = _build_one_layer_model(bh_glx_mesh, _LAYER)
    attention = model.layers[0].attention
    print(f"[perf-dn-unit] 1L TT model built (attention type = {type(attention).__name__})")
    assert "DeltaAttention" in type(attention).__name__, (
        f"Expected a DeltaNet attention block on layer 0; got {type(attention).__name__}.  "
        "Did the linear_attention_pattern fail to apply?"
    )

    # Decode input: [B=1, 1, T=1, H], COL-SHARDED H/4 — production residual
    # stream layout AND the new V2-DN-TP DeltaNet input contract.
    # ``include_ccl`` adds an extra gather + scatter around the block;
    # post-V2-DN-TP the gather+scatter shouldn't be needed in production,
    # so this flag mostly stays off.
    x_cpu = torch.randn(1, 1, 1, _H, dtype=torch.bfloat16) * 0.5
    x_input_tt = _send_col_sharded(x_cpu, bh_glx_mesh, args)

    # ---- Warmup ----
    for _ in range(_WARMUP):
        out = _do_one_step(attention, x_input_tt, include_ccl=_INCLUDE_CCL)
        ttnn.deallocate(out)
    ttnn.synchronize_device(bh_glx_mesh)

    # ---- Timed runs ----
    if _USE_TRACE:
        trace_id = ttnn.begin_trace_capture(bh_glx_mesh, cq_id=0)
        out_trace = _do_one_step(attention, x_input_tt, include_ccl=_INCLUDE_CCL)
        ttnn.end_trace_capture(bh_glx_mesh, trace_id, cq_id=0)

        latencies_ms: list[float] = []
        for _ in range(_ITERS):
            t0 = time.perf_counter()
            ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(bh_glx_mesh)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        ttnn.release_trace(bh_glx_mesh, trace_id)
        _summary(latencies_ms, "TRACED")
    else:
        latencies_ms = []
        for _ in range(_ITERS):
            t0 = time.perf_counter()
            out = _do_one_step(attention, x_input_tt, include_ccl=_INCLUDE_CCL)
            ttnn.synchronize_device(bh_glx_mesh)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            ttnn.deallocate(out)
        _summary(latencies_ms, "EAGER")

    # Flush device-side profiler data to host so a `python -m tracy -p -r`
    # capture has device kernel durations to merge (otherwise the report
    # post-process reports "No device logs found").
    ttnn.synchronize_device(bh_glx_mesh)
    ttnn.ReadDeviceProfiler(bh_glx_mesh)

    ttnn.deallocate(x_input_tt)


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-x"])
