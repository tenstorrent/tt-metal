# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where a model-local ``ttnn.generic_op`` call's fixed ~1 ms goes (tt-blaze G53F-0206).

``custom_ops/untilize_used`` with ``run = 0`` does no device work (every core gets zero tile rows, core 0 writes one
8 KB zero row), yet costs ~1.05 ms per call on the 8x4 mesh (G53F-0203); the mHC mix kernels cost ~1.25 ms per call
at 64 rows and at 2048 rows alike (G53F-0202). This splits one call (``UU_N`` back-to-back calls, one synchronize):

* ``full``: build the ProgramDescriptor in Python + ``ttnn.generic_op`` (what the model does);
* ``build``: the Python descriptor build alone (no device call);
* ``reuse``: ``ttnn.generic_op`` with ONE prebuilt descriptor (same tensors) -- the C++ path: descriptor hash,
  program-cache hit, runtime-args override on every core of 32 devices, enqueue;
* ``reuse_hash``: the same with ``custom_program_hash`` set (no descriptor hash);
* ``enqueue`` columns: the host time until the last call returned (before the synchronize).

    pytest models/demos/glm53_flash/tests/test_generic_op_overhead.py -s
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.untilize_used.op import UntilizeUsed
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
N = int(os.environ.get("UU_N", "50"))
R, H, E = 67840, 4096, 288


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_generic_op_overhead(mesh_device):
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, H]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    zeros = torch.zeros(1, E, dtype=torch.int32)
    counts, offsets = [
        ttnn.from_torch(zeros, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=rep)
        for _ in range(2)
    ]
    op = UntilizeUsed(mesh_device)
    y = op(out, counts, offsets)  # compile + first program
    ttnn.synchronize_device(mesh_device)

    def timed(fn):
        fn()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(N):
            fn()
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        return round((t2 - t0) / N * 1e3, 3), round((t1 - t0) / N * 1e3, 3)

    row = {}
    row["full"], row["full_enqueue"] = timed(lambda: ttnn.deallocate(op(out, counts, offsets)))
    row["build"], _ = timed(lambda: op.descriptor(out, counts, offsets, y))
    _, desc = op.descriptor(out, counts, offsets, y)
    row["reuse"], row["reuse_enqueue"] = timed(lambda: ttnn.generic_op([out, counts, offsets, y], desc))
    desc.custom_program_hash = 0x6E5E_D0_0206
    row["reuse_hash"], row["reuse_hash_enqueue"] = timed(lambda: ttnn.generic_op([out, counts, offsets, y], desc))
    row["calls"] = N
    logger.info(f"[generic-op] {json.dumps(row)}")
