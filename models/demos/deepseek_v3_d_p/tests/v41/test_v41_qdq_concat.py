# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The output projection's head concatenation folded into the FP8 QDQ's output shape (``tt/v41/qdq.py``
``fp8_qdq(x, out_shape=...)``, ``tt/v41/attention.py::_o_proj`` with ``V41_DECODE_CONCAT=qdq``)
against ``nlp_concat_heads`` followed by the QDQ.

No weights: ``V41_QC_TRIALS`` random bf16 ``[1, g, 32, rank]`` tiles a chip (the decode tile's ``wo_a``
output: one
tile row a head group), the two paths compared bit for bit on every chip, then each path under trace
(``V41_QC_REPLAYS`` replays of {calls} calls) for the per-layer device time. Results: ``V41_QC_RESULT``,
``V41_QC_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp8_qdq


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_qdq_concat(mesh_device, device_params):
    trials = int(os.environ.get("V41_QC_TRIALS", 4))
    replays = int(os.environ.get("V41_QC_REPLAYS", 20))
    calls = int(os.environ.get("V41_QC_CALLS", 32))  # calls a trace: the replay's fixed cost divided by this
    sp, tp = tuple(mesh_device.shape)
    groups, rank = C.O_GROUPS // tp, C.O_LORA_RANK // C.O_GROUPS  # wo_a: [1, G/tp, 32, rank a group] a chip
    torch.manual_seed(13)

    def tile(seed: int):
        gen = torch.Generator().manual_seed(seed)
        t = torch.randn(sp * groups, 32, tp * rank, generator=gen) * 3.0
        return ttnn.from_torch(
            t.view(1, sp * groups, 32, tp * rank),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(1, 3)),
        )

    op = lambda x: fp8_qdq(ttnn.experimental.nlp_concat_heads(x))
    qdq = lambda x: fp8_qdq(x, out_shape=(1, 1, 32, groups * rank))
    result = {"mesh": [sp, tp], "shape": [1, groups, 32, rank], "trials": []}
    for trial in range(trials):
        x = tile(trial)
        ref, out = op(x), qdq(x)
        assert tuple(out.shape) == tuple(ref.shape) == (1, 1, 32, groups * rank), (tuple(out.shape), tuple(ref.shape))
        diffs = [
            int((ttnn.to_torch(a) != ttnn.to_torch(b)).sum())
            for a, b in zip(ttnn.get_device_tensors(ref), ttnn.get_device_tensors(out))
        ]
        entry = {"trial": trial, "words_differ": sum(diffs), "chips_differ": sum(1 for d in diffs if d)}
        result["trials"].append(entry)
        logger.info(f"qdq concat trial {json.dumps(entry)}")
        assert sum(diffs) == 0, entry
        for t in (x, ref, out):
            ttnn.deallocate(t)
    x = tile(99)
    timing = {}
    for name, fn in (("op", lambda: op(x)), ("qdq", lambda: qdq(x))):
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [fn() for _ in range(calls)]
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        best = None
        for _ in range(replays):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) / calls
            best = dt if best is None else min(best, dt)
        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            ttnn.deallocate(o)
        timing[name] = round(best * 1e6, 1)
        logger.info(f"qdq concat traced {name}: {timing[name]} us a layer (min of {replays} replays of {calls} calls)")
    result["layer_us"] = timing
    logger.info(f"V41_QC_RESULT {json.dumps(result)}")
    out_path = os.environ.get("V41_QC_OUT")
    if out_path:
        with open(os.path.expanduser(out_path), "w") as f:
            json.dump(result, f, indent=1)
