# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the GR write folded into the row_reduce v2 program (QWEN38_RR_WRITE=1): the new residual and the block
output of ``row_reduce`` + ``gr_write`` vs the folded program, bit for bit on rows < rows, bf16 (rows 1, 2; DRAM and L1
sharded partial) and fp32 (row 1: the block is the bf16 narrowing of the fp32 sum, as QSA's typecast), repeated calls, a
trace replayed with parity carried, and the traced time of [reduce, gr_write] vs the folded program."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import test_fused_row_reduce_device as base
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_row_reduce_device import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_write as gw
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_reduce as rr

TP = MESH_SHAPE[1]
CASES = [("bf16", 1, False), ("bf16", 2, False), ("bf16", 1, True), ("bf16", 2, True), ("fp32", 1, False)]
RESULTS = os.environ.get("ROW_REDUCE_RESULTS")


def _mesh_tensor(mesh, x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        x,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _host(mesh, t, rows):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:, :, :rows, :]


def _bits(x):
    return x.to(torch.bfloat16).view(torch.int16) if x.dtype != torch.float32 else x.view(torch.int32)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [base.DEVICE_PARAMS], indirect=True)
def test_rr_write_is_bitwise_reduce_plus_gr_write(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    os.environ[rr.ENV] = "2"
    os.environ[rr.RR_WRITE_ENV] = "1"
    rr.prepare(mesh)
    results = []
    for dtype, rows, sharded in CASES:
        g = torch.Generator().manual_seed(40 + rows)
        x = base._partials(dtype, rows, seed=500 + rows)
        partial = base._to_mesh(mesh, x, dtype, sharded)
        residual_h = (
            torch.randn(TP, 4, rows, 320, generator=g)
            * torch.exp2(torch.randint(-4, 5, (TP, 4, rows, 320), generator=g).float())
        ).to(torch.bfloat16)
        injection_h = torch.randn(TP, 1, rows, 4, generator=g).to(torch.bfloat16)
        residual, injection = _mesh_tensor(mesh, residual_h), _mesh_tensor(mesh, injection_h)
        state = SimpleNamespace(residual=residual, injection=injection)

        def chain():
            os.environ[rr.RR_WRITE_ENV] = "0"
            block = rr.row_reduce(partial, cluster_axis=1)
            block_bf = block if dtype == "bf16" else ttnn.typecast(block, ttnn.bfloat16)
            new = gw.gr_write(block_bf, residual, injection)
            os.environ[rr.RR_WRITE_ENV] = "1"
            return block, new

        def folded():
            rr.set_pending(state)
            block = rr.row_reduce(partial, cluster_axis=1)
            new = rr.take_written(block, state)
            assert new is not None, "the folded program was not taken"
            return block, new

        ref_block, ref_new = chain()
        ref_block_h, ref_new_h = _host(mesh, ref_block, rows), _host(mesh, ref_new, rows)
        report = {"dtype": dtype, "rows": rows, "l1_sharded": sharded}
        block, new = folded()
        report["new_residual_mismatch"] = base._mismatches(_host(mesh, new, rows), ref_new_h, "bf16")
        report["block_mismatch"] = base._mismatches(_host(mesh, block, rows), ref_block_h, dtype)
        again = 0
        for _ in range(6):
            b2, n2 = folded()
            again += base._mismatches(_host(mesh, n2, rows), ref_new_h, "bf16") + base._mismatches(
                _host(mesh, b2, rows), ref_block_h, dtype
            )
        report["repeat_mismatch"] = again
        os.environ[rr.RR_WRITE_ENV] = "1"
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = [folded() for _ in range(5)]
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        bad = 0
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            bad += sum(base._mismatches(_host(mesh, n, rows), ref_new_h, "bf16") for _, n in outs)
        ttnn.release_trace(mesh, tid)
        report["trace_replay_mismatch"] = bad

        def free(pair):
            for t in pair:
                ttnn.deallocate(t)

        def timed(fn):
            def call():
                block, new = fn()
                ttnn.deallocate(block)
                return new

            return base._traced_us(mesh, call)

        report["us_slope_and_mean12"] = {"reduce_plus_gr_write": timed(chain), "folded": timed(folded)}
        print("RR_WRITE_RESULT " + json.dumps(report))
        results.append(report)
        assert (
            report["new_residual_mismatch"] == 0 and report["block_mismatch"] == 0 and again == 0 and bad == 0
        ), report
    if RESULTS:
        with open(RESULTS, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
