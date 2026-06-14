# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Check the RMSAllGather hypothesis (device-kernel) on 32-chip BH GLX 8x4.

In-flow Tracy attribution found RMSAllGatherDeviceOperation (= ttnn.fused_rms_minimal,
the decode distributed RMSNorm over cluster_axis=1 cols ring-4) is the #1 decode collective:
~722us/FA-layer, ~334us/GDN-layer (device-kernel).

HYPOTHESIS to check: is it expensive because it ALL-GATHERS THE FULL VECTOR (1280/5120 wide,
wasteful), or just the small RMS stats (sum-of-squares)? The decode wrapper builds the
"LAYERNORM" stats buffer as (1,1,32,128) -> looks MINIMAL (gathers 128-wide stats, not the
full vector). This test confirms that on device + measures the per-call device-kernel, and
tests the HARDCODED num_links=1 (tt_sharded_distributed_rmsnorm:1915) vs num_links=2 (everything
else uses GALAXY_NUM_LINKS=2) — a candidate lever if the gather is bandwidth-bound.

Run under Tracy (device-kernel), one num_links per invocation:
  for NL in 1 2; do
    MESH_DEVICE=BH_GLX QWEN36_RMS_NUM_LINKS=$NL python -m tracy -p -v -r --op-support-count 20000 \
      -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_rmsnorm_allgather_devk.py -s
  done
then aggregate RMSAllGatherDeviceOperation device-kernel over the signpost window for each.
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

_ITERS = int(os.environ.get("QWEN36_RMS_ITERS", "20"))
_NUM_LINKS = int(os.environ.get("QWEN36_RMS_NUM_LINKS", "1"))


@pytest.mark.hardware
def test_rmsnorm_allgather_devk(bh_glx_mesh):
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
    }.items():
        os.environ.setdefault(_k, _v)
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    sd = _load_full_state_dict(_SNAPSHOT)
    pfx0, allp = "model.language_model.layers.0.", "model.language_model.layers."
    sd = {k: v for k, v in sd.items() if (not k.startswith(allp)) or k.startswith(pfx0)}
    paged = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(bh_glx_mesh, sd, ["linear_attention"], 1, paged)
    model.switch_mode("decode")
    mesh = bh_glx_mesh
    norm = model.layers[0].attention_norm  # DistributedNorm (decode rms_allgather)
    tt_ccl = norm.tt_ccl
    wsd = tt_ccl.worker_sub_device_id

    # Decode norm input: full residual dim=5120 col-sharded across 4 cols -> dim_per_tp=1280/chip
    # (matches gamma's per-chip width; feeding 1280 then re-sharding would give 320/chip).
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, args.dim) * 0.1
    x_t = ttnn.from_torch(
        x,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=norm.gather_in_mem_cfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=(8, 4)),
    )

    # Warmup forward -> lazily builds the "LAYERNORM" stats buffer + confirms the path.
    _ = norm.forward(ttnn.clone(x_t), None, "decode")
    ttnn.synchronize_device(mesh, sub_device_ids=[wsd])

    stats = tt_ccl.all_gather_buffers["LAYERNORM"]
    print(
        f"\n[rms] HYPOTHESIS CHECK: stats buffer shape = {list(stats.shape)} "
        f"(128-wide => MINIMAL stats gather, NOT the full {args.dim}-wide vector)",
        flush=True,
    )
    print(
        f"[rms] num_links={_NUM_LINKS} iters={_ITERS} input={list(x_t.shape)} dim_per_tp={args.dim_per_tp}", flush=True
    )

    res0 = ttnn.zeros_like(x_t)
    gamma = norm.norm.weight_distributed
    eps = norm.norm.eps
    omc = norm.norm.output_mem_config
    topo = norm.ccl_topology

    def one():
        cax = 1
        sem = tt_ccl.gather_semaphore_handles[cax][tt_ccl.gather_idx[cax]]
        out = ttnn.fused_rms_minimal(
            ttnn.clone(x_t),
            norm.ln_prg_cfg,
            cax,
            mesh,
            sem,
            topology=topo,
            residual_input_tensor=res0,
            num_links=_NUM_LINKS,
            epsilon=eps,
            weight=gamma,
            stats=stats,
            memory_config=omc,
        )
        tt_ccl.gather_idx[cax] = (tt_ccl.gather_idx[cax] + 1) % tt_ccl.num_cbs
        return out

    o = one()
    ttnn.synchronize_device(mesh, sub_device_ids=[wsd])
    ttnn.ReadDeviceProfiler(mesh)
    signpost("start")
    for _ in range(_ITERS):
        o = one()
    ttnn.synchronize_device(mesh, sub_device_ids=[wsd])
    signpost("stop")
    print(
        f"[rms] done num_links={_NUM_LINKS}: aggregate RMSAllGatherDeviceOperation device-kernel /{_ITERS}", flush=True
    )
    assert o is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
