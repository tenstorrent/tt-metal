# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test for QWEN36_DECODE_AR_FUSE_RESIDUAL: all_reduce_async(residual_tensor=) on the decode RES40 layout.

For each batch B, the same per-device bf16 partials and the same RES40 residual go through
  unfused: dar.all_reduce(partial) then dar.add(residual, ar)          (today's decode path)
  fused:   dar.arm_residual(residual); dar.all_reduce(partial)        (residual added in the reduction kernel)
and the test asserts
  * fused == unfused byte for byte on every device (bf16 DST: the same FPU add, one rounding);
  * every device holds the same bytes (replicated result);
  * fused is deterministic over repeated calls and under trace replay;
  * PCC vs a torch fp32 reference (sanity; the byte check is the real gate).

Run (Blackhole Galaxy, inside the container, one 1x8 TP submesh):
    MESH_DEVICE=P150x8 QWEN36_FABRIC_RING=1 HF_MODEL=<Qwen3.8-27B snapshot> \\
        pytest models/demos/blackhole/qwen36/tests/test_decode_ar_fuse.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def _per_device(mesh_device, t):
    """[ndev, ...] torch tensor, one slice per device (no reduction)."""
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))


def _partial(mesh_device, parts):
    """parts [tp, B, dim] -> device i holds parts[i] as [1,1,B,dim] (DRAM, bf16), like a row-parallel output."""
    tp, B, dim = parts.shape
    return ttnn.from_torch(
        parts.reshape(1, 1, B, tp * dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _residual(mesh_device, res, dar):
    t = ttnn.from_torch(
        res.reshape(1, 1, *res.shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_memory_config(t, dar.res_mem)


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [1, 8, 32], ids=["B1", "B8", "B32"])
def test_decode_ar_fuse_residual(mesh_device, B, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", model_path())
    tp = mesh_device.get_num_devices()
    if tp < 2:
        pytest.skip("needs a TP mesh")
    from models.tt_transformers.tt.ccl import TT_CCL

    args = Qwen36ModelArgs(mesh_device, max_batch_size=32, max_seq_len=256)
    tt_ccl = TT_CCL(mesh_device)
    dar = tpc.DecodeResidualAllReduce(mesh_device, args, tt_ccl, gdn_fp32=False, fuse_residual=True)
    dim = args.dim

    parts = (torch.randn(tp, B, dim) * 0.5).to(torch.bfloat16)
    res = (torch.randn(B, dim) * 4.0).to(torch.bfloat16)
    residual = _residual(mesh_device, res, dar)

    # Unfused reference: today's decode path (all_reduce, then the RES40 add).
    ar = dar.all_reduce(_partial(mesh_device, parts))
    unfused = dar.add(residual, ar)
    ttnn.deallocate(ar)
    ref_dev = _per_device(mesh_device, unfused)
    ttnn.deallocate(unfused)

    # Fused, eager, three times (program cache hit on the 2nd/3rd).
    for it in range(3):
        dar.arm_residual(residual)
        out = dar.all_reduce(_partial(mesh_device, parts))
        h = dar.add_residual(residual, out)
        assert h is out, "flag on + bf16 RES40 residual must take the fused path"
        got = _per_device(mesh_device, h)
        ttnn.deallocate(h)
        assert got.shape == ref_dev.shape
        for d in range(tp):
            assert torch.equal(got[d], got[0]), f"device {d} differs from device 0 (iter {it})"
        assert torch.equal(got, ref_dev), (
            f"fused != unfused (iter {it}): max abs diff {(got.float() - ref_dev.float()).abs().max().item()}, "
            f"mismatching elems {(got != ref_dev).sum().item()}"
        )

    ref = res.float() + parts.float().sum(0)
    passing, pcc = comp_pcc(ref, got[0].reshape(B, dim).float(), 0.999)
    logger.info(f"AR-fuse B={B}: byte-identical to unfused on {tp} devices; PCC vs fp32 ref {pcc}")
    assert passing, pcc

    # Traced: capture one fused call, replay twice, same bytes.
    partial_in = _partial(mesh_device, parts)
    packed = ttnn.to_memory_config(partial_in, dar.res_mem)  # pre-pack outside the trace (no alloc in trace)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = ttnn.experimental.all_reduce_async(
        packed,
        dar.buffer,
        cluster_axis=1,
        mesh_device=mesh_device,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
        topology=dar.topology,
        num_links=dar.num_links,
        memory_config=dar.res_mem,
        residual_tensor=residual,
    )
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for rep in range(2):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got_t = _per_device(mesh_device, traced)
        assert torch.equal(got_t, ref_dev), f"traced replay {rep} != unfused"
    ttnn.release_trace(mesh_device, tid)
