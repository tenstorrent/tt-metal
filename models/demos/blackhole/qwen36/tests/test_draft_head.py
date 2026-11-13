# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the drafter's bfloat4_b DRAM-sharded LM head (tt/draft_head.py) at the Qwen3.8 TP=8 shape.

On a random head [248320, 5120] and the drafter's row counts (1 and 8 draft rows, 32 the most it takes):
* build_draft_head takes the shape;
* DraftHead.linear returns the logits contract the drafter's fused argmax takes ([1,1,R,vocab/tp] float32, TILE,
  DRAM interleaved), close to the plain matmul on an interleaved bfloat4_b copy of the same weight (same quantized
  values) and, with bfloat4_b error, to the shared bfloat8_b head the drafter read before;
* the trace replay time of the copy against the shared head.

    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_draft_head.py
"""

import time

import pytest
import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape

MESH = _resolve_mesh_shape(8)
K, VOCAB = 5120, 248320


def _parametrize(fn):
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _replay_us(mesh, tid, n=50):
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / n * 1e6


def _logits(t, rows):
    """[rows, vocab] float32: the devices' vocab shards side by side."""
    return torch.cat([ttnn.to_torch(d).reshape(-1, d.shape[-1])[:rows] for d in ttnn.get_device_tensors(t)], -1)


@torch.no_grad()
@_parametrize
def test_the_bfp4_head_matches_the_plain_heads(mesh_device):
    from models.demos.blackhole.qwen36.tt import tp_common as tpc
    from models.demos.blackhole.qwen36.tt.draft_head import build_draft_head
    from models.demos.blackhole.qwen36.tt.spec_argmax_fused import ROWS, FusedShardArgmax

    nd = mesh_device.get_num_devices()
    g = torch.Generator().manual_seed(0)
    head_w = (torch.randn(VOCAB, K, generator=g) * 0.02).to(torch.bfloat16)
    head = build_draft_head(head_w, mesh_device, None)
    assert head is not None, "the TP=8 shape must take the bfloat4_b head"

    def plain(dtype):
        return ttnn.as_tensor(
            head_w,
            preprocess=lambda t: t.T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=tpc.tp_shard_mapper(mesh_device, -1),
        )

    bfp4_plain, bf8_shared = plain(ttnn.bfloat4_b), plain(ttnn.bfloat8_b)
    cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    heads = {
        "bfp4-dram-sharded": lambda x: head.linear(x, cfg),
        "bfp4-plain": lambda x: ttnn.linear(x, bfp4_plain, dtype=ttnn.float32, compute_kernel_config=cfg),
        "bf8-shared": lambda x: ttnn.linear(x, bf8_shared, dtype=ttnn.float32, compute_kernel_config=cfg),
    }
    fused = FusedShardArgmax(mesh_device, VOCAB, nd)
    for rows in (1, 8, 32):
        x_host = torch.randn(1, 1, rows, K, generator=g).to(torch.bfloat16)
        x = ttnn.from_torch(
            x_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        assert head.applies(x)
        outs = {tag: fn(x) for tag, fn in heads.items()}
        new = outs["bfp4-dram-sharded"]
        assert tuple(new.shape) == (1, 1, rows, VOCAB // nd) and new.dtype == ttnn.float32
        assert fused.supports(new) is None, fused.supports(new)
        got = {tag: _logits(t, rows) for tag, t in outs.items()}
        ref = x_host.reshape(rows, K).float() @ head_w.float().T
        ids_tt = fused(new, ROWS)
        ids = ttnn.to_torch(ttnn.get_device_tensors(ids_tt)[0]).reshape(-1)[:rows].to(torch.int64)
        ttnn.deallocate(ids_tt)
        assert torch.equal(ids, got["bfp4-dram-sharded"].argmax(-1)), "fused argmax != argmax of the logits"

        ok, pcc = comp_pcc(got["bfp4-plain"], got["bfp4-dram-sharded"], 0.9999)
        assert ok, f"rows={rows}: DRAM-sharded vs plain bfloat4_b PCC {pcc}"
        same = int((got["bfp4-plain"] == got["bfp4-dram-sharded"]).sum())
        _, pcc_shared = comp_pcc(got["bf8-shared"], got["bfp4-dram-sharded"], 0.99)
        _, pcc_ref4 = comp_pcc(ref, got["bfp4-dram-sharded"], 0.99)
        _, pcc_ref8 = comp_pcc(ref, got["bf8-shared"], 0.99)
        top_agree = int((got["bf8-shared"].argmax(-1) == ids).sum())
        logger.info(
            f"[DRAFT_HEAD] rows={rows}: vs plain bfloat4_b PCC {pcc} ({same} of {got['bfp4-plain'].numel()} equal); "
            f"vs shared bfloat8_b PCC {pcc_shared}, top-1 {top_agree}/{rows}; vs fp32 torch PCC bfloat4_b {pcc_ref4} "
            f"bfloat8_b {pcc_ref8}"
        )
        assert pcc_shared > 0.99, f"rows={rows}: bfloat4_b vs bfloat8_b head PCC {pcc_shared}"
        for t in outs.values():
            ttnn.deallocate(t)

        ttnn.synchronize_device(mesh_device)
        timed = ("bfp4-dram-sharded", "bf8-shared")
        tids, held = {}, []
        with trace_allocation_tracker.corruptible_allocation_scope(mesh_device):
            for tag in timed:
                tids[tag] = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                held.append(heads[tag](x))
                ttnn.end_trace_capture(mesh_device, tids[tag], cq_id=0)
        us = {tag: _replay_us(mesh_device, tids[tag]) for tag in tids}
        logger.info(f"[DRAFT_HEAD] rows={rows}: replay " + ", ".join(f"{t} {u:.1f} us" for t, u in us.items()))
        for tag in tids:
            ttnn.release_trace(mesh_device, tids[tag])
        for t in held + [x]:
            ttnn.deallocate(t)
    for t in (head.weight, bfp4_plain, bf8_shared):
        ttnn.deallocate(t)
