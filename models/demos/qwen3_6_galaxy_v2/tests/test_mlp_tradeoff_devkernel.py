# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DEVICE-KERNEL tradeoff: col-REPLICATED decode MLP vs the REAL ring-config SHARDED MLP.

Run under Tracy, one variant per invocation (QWEN36_TRADEOFF_VARIANT=rep|sharded),
looping the MLP inside a signpost("start")/("stop") window. Aggregate the
ops_perf CSV (per-op max-over-32-devices critical path, /iters) for the DEVICE-KERNEL
us/iter — NOT wall-clock.

  SHARDED uses the REAL ring config: w1/w3 ring matmul (FF1_3_RING40_PROGCFG, K=1280
    on the 40-core ring) -> line_reduce_scatter(cols, 640) x2 -> SwiGLU -> line_all_gather
    (cols, 2560) -> w2 ring matmul (L1 w2) -> line_all_reduce(rows).   [col RS+gather+row AR]
  REPLICATED: w1/w3 LOCAL (K=5120 full, row-shard inter) -> SwiGLU -> w2 LOCAL ->
    line_all_reduce(rows).                                              [row AR only]

Run (both):
  for V in sharded rep; do
    MESH_DEVICE=BH_GLX QWEN36_TRADEOFF_VARIANT=$V python -m tracy -p -v -r \
      --op-support-count 20000 -m pytest --noconftest \
      models/demos/qwen3_6_galaxy_v2/tests/test_mlp_tradeoff_devkernel.py -s
  done
then aggregate each run's ops_perf csv over the signpost window.
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401
    _PAGED_BLOCK_SIZE,
    _PAGED_MAX_NUM_BLOCKS,
    _SNAPSHOT,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    bh_glx_mesh,
)

try:
    from tracy import signpost
except ImportError:
    signpost = lambda *_a, **_k: None  # noqa: E731

_CLUSTER = (8, 4)
_ITERS = int(os.environ.get("QWEN36_TRADEOFF_ITERS", "10"))
_VARIANT = os.environ.get("QWEN36_TRADEOFF_VARIANT", "rep").lower()


@pytest.mark.hardware
def test_mlp_tradeoff_devkernel(bh_glx_mesh):
    mesh = bh_glx_mesh
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FUSE_RS_MATMUL": "1",  # build the ring-40 weights
    }.items():
        os.environ.setdefault(_k, _v)
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    sd = _load_full_state_dict(_SNAPSHOT)
    pfx0 = "model.language_model.layers.0."
    allp = "model.language_model.layers."
    sd = {k: v for k, v in sd.items() if (not k.startswith(allp)) or k.startswith(pfx0)}
    paged = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(mesh, sd, ["linear_attention"], 1, paged)
    model.switch_mode("decode")
    mlp = model.layers[0].feed_forward
    mc = model.model_config
    tt_ccl = mlp.tt_ccl
    ck = args.compute_kernel_config_hifi2
    M, dim, inter = 32, args.dim, args.intermediate_dim
    rows, cols = _CLUSTER
    print(f"[devk] VARIANT={_VARIANT} iters={_ITERS} M={M} dim={dim} inter={inter}", flush=True)

    if _VARIANT == "sharded":
        # Call the REAL fused decode MLP (ring matmul + all_gather_matmul + CCL) — the
        # actual current implementation, not a hand-rebuild. FUSE_RS_MATMUL=1 is set above.
        os.environ["QWEN36_FUSE_RS_MATMUL"] = "1"
        block = model.layers[0]
        # ff_in: col-sharded so each chip holds dim_per_tp=1280 (= dim 5120 / 4 cols). Feed the
        # full dim=5120 and shard the last dim across the 4 cols -> 1280/chip (what the decoder
        # feeds _mlp_decode_qwen36; the branch reshards it to the ring layout internally).
        x = torch.randn(1, 1, M, args.dim) * 0.5
        x_t = ttnn.from_torch(
            x,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, -1), mesh_shape=_CLUSTER),
        )

        def run():
            return block._mlp_decode_qwen36(ttnn.clone(x_t), batch_size=1)

    else:
        # REPLICATED: x replicated; w1/w3 row-shard inter + replicate dim over cols; w2 row-shard inter-K.
        torch.manual_seed(0)
        w1 = torch.randn(dim, inter) * 0.02
        w3 = torch.randn(dim, inter) * 0.02
        w2 = torch.randn(inter, dim) * 0.02
        x = torch.randn(1, 1, M, dim) * 0.5
        x_t = ttnn.from_torch(
            x,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def _sr(t, d):
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(0),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(d, None), mesh_shape=_CLUSTER),
            )

        w1t, w3t, w2t = _sr(w1, -1), _sr(w3, -1), _sr(w2, -2)

        def run():
            h1 = ttnn.linear(
                x_t, w1t, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            h3 = ttnn.linear(
                x_t, w3t, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            sg = ttnn.mul(
                h1,
                h3,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            h1.deallocate(True)
            h3.deallocate(True)
            op = ttnn.linear(
                sg, w2t, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            sg.deallocate(True)
            out = tt_ccl.line_all_reduce(op, cluster_axis=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            op.deallocate(True)
            return out

    o = run()  # warmup/compile
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)  # flush warmup
    signpost("start")
    for _ in range(_ITERS):
        o = run()
    ttnn.synchronize_device(mesh)
    signpost("stop")
    print(
        f"[devk] {_VARIANT}: {_ITERS} iters done — aggregate ops_perf csv over the signpost window (/{_ITERS}).",
        flush=True,
    )
    assert o is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
