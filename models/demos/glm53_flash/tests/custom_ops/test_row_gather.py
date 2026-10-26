# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/row_gather`` (tt-blaze G53F-0186): bitwise vs torch and vs ``ttnn.embedding(..., ROW_MAJOR)`` on every
chip, at the prefill MoE regions path's shapes (H 4096; a [T, H] token table gathered into the region buffer, and a
[max_buf, H] expert-output table gathered back for every pick), plus the timing of both ops.

    GLM53_MESH_SHAPE=8,4 pytest models/demos/glm53_flash/tests/custom_ops/test_row_gather.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")
from models.demos.glm53_flash.custom_ops.row_gather.op import RowGather
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
H = 4096
# (table rows, gathered rows): the region gather at REGIONS_T 8192 full / fit-like, and the pick gather back
CASES = [(8192, 67840), (8192, 20480), (67840, 65536), (20480, 65536), (777, 1000)]
REPS = int(os.environ.get("ROW_GATHER_REPS", "5"))


def _time(fn, mesh_device, reps):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / reps


@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_row_gather(mesh_device):
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    op = RowGather(mesh_device)
    torch.manual_seed(0)
    for R, N in CASES:
        table = torch.randn(R, H).to(torch.bfloat16)
        idx = torch.randint(0, R, (1, N), dtype=torch.int32)
        t_tt = ttnn.from_torch(
            table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=rep
        )
        i_tt = ttnn.from_torch(
            idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=rep
        )
        ref = RowGather.golden(table, idx)
        got = op(i_tt, t_tt)
        emb = ttnn.embedding(i_tt, t_tt, layout=ttnn.ROW_MAJOR_LAYOUT)
        devs_g, devs_e = ttnn.get_device_tensors(got), ttnn.get_device_tensors(emb)
        ok_ref = all(torch.equal(ttnn.to_torch(d).reshape(ref.shape), ref) for d in devs_g)
        ok_emb = all(
            torch.equal(ttnn.to_torch(a).reshape(ref.shape), ttnn.to_torch(b).reshape(ref.shape))
            for a, b in zip(devs_g, devs_e)
        )
        ttnn.deallocate(got)
        ttnn.deallocate(emb)
        t_op = _time(lambda: op(i_tt, t_tt), mesh_device, REPS)
        t_emb = _time(lambda: ttnn.embedding(i_tt, t_tt, layout=ttnn.ROW_MAJOR_LAYOUT), mesh_device, REPS)
        gb = N * H * 2 / 1e9
        logger.info(
            f"[row_gather] R={R} N={N}: ==torch {ok_ref} ==embedding {ok_emb} on {len(devs_g)} chips; "
            f"row_gather {t_op * 1e3:.3f} ms ({gb / t_op:.0f} GB/s written)  embedding {t_emb * 1e3:.3f} ms "
            f"({gb / t_emb:.0f} GB/s)  speedup {t_emb / t_op:.2f}x"
        )
        assert ok_ref and ok_emb, (R, N, ok_ref, ok_emb)
        ttnn.deallocate(t_tt)
        ttnn.deallocate(i_tt)
