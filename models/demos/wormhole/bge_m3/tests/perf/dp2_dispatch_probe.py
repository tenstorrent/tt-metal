# SPDX-License-Identifier: Apache-2.0
"""DP2 dispatch attribution: does the ~146ms wall gap belong to 2x1 mesh trace
dispatch (FDMeshCommandQueue walking both devices sequentially)?

V0: current 2x1 mesh, one trace, B6/chip (the shipping DP=2 path).
V1: two 1x1 submeshes, two traces (DP-replica mode, B6 each), sequential
    blocking=False launch + concurrent sync.
V2: same two submeshes, launched concurrently from 2 host threads.

All measure Forward-only trace-replay wall (no H2D in the timed region).
If V1/V2 wall << V0 wall, the gap is 2x1 mesh dispatch serialization.
Requires BGE_M3_DATA_PARALLEL=1.
"""
import concurrent.futures as cf
import os
import time

import pytest
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tests.perf.dp2_perf import _to_batchsharded_tensors
from models.demos.wormhole.bge_m3.tests.perf.perf import NUM_ITERATIONS, prepare_inputs
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

SEQ_LEN = 8192
PER_CHIP_BATCH = 6


def _one_chip_tensors(tokenizer, sub, pad_token_id):
    # Build B6/S8192 inputs on a single 1x1 submesh (replicate mapper, 1 device).
    import torch

    ids = torch.randint(1, 1000, (PER_CHIP_BATCH, SEQ_LEN), dtype=torch.long)
    ttids = torch.zeros(PER_CHIP_BATCH, SEQ_LEN, dtype=torch.long)
    mask = (ids != pad_token_id).to(torch.int64)
    pos = (torch.cumsum(mask, dim=1) * mask + pad_token_id).to(torch.long)

    def to_dev(t):
        return ttnn.from_torch(
            t.int(), device=sub, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    return {"input_ids": to_dev(ids), "token_type_ids": to_dev(ttids), "position_ids": to_dev(pos)}


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 1}],
    indirect=True,
)
def test_dp2_dispatch_probe(mesh_device):
    assert os.environ.get("BGE_M3_DATA_PARALLEL", "0") == "1", "set BGE_M3_DATA_PARALLEL=1"

    # ── V0: current 2x1 mesh, one trace ──────────────────────────────────
    args, model_v0, shared_sd = create_tt_model(
        mesh_device=mesh_device, max_batch_size=12, max_seq_len=SEQ_LEN, dtype=ttnn.bfloat8_b
    )
    assert model_v0._data_parallel
    inp = prepare_inputs(args.tokenizer, 12, SEQ_LEN, args.pad_token_id)
    dt0 = _to_batchsharded_tensors(inp, mesh_device, device=True)
    model_v0.forward(**dt0)
    ttnn.synchronize_device(mesh_device)
    model_v0.capture_trace(**dt0, mesh_device=mesh_device, cq_id=0)
    for _ in range(3):
        model_v0.execute_trace(blocking=True)
    v0 = []
    for _ in range(NUM_ITERATIONS):
        t = time.perf_counter()
        model_v0.execute_trace(blocking=True)
        v0.append((time.perf_counter() - t) * 1e3)
    model_v0.release_trace()
    logger.info(f"V0 2x1-mesh-trace: avg={sum(v0)/len(v0):.2f}ms best={min(v0):.2f}ms")

    # ── V1/V2: two 1x1 submeshes, DP-replica mode ────────────────────────
    os.environ["BGE_M3_DP_REPLICA"] = "1"
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(1, 1))
    assert len(submeshes) == 2
    submodels, subtensors = [], []
    for i, sub in enumerate(submeshes):
        _, m, _ = create_tt_model(
            mesh_device=sub,
            max_batch_size=PER_CHIP_BATCH,
            max_seq_len=SEQ_LEN,
            dtype=ttnn.bfloat8_b,
            state_dict=shared_sd,
        )
        assert m._data_parallel, f"submesh {i} did not activate DP-replica head-fold"
        st = _one_chip_tensors(args.tokenizer, sub, args.pad_token_id)
        m.forward(**st)
        ttnn.synchronize_device(sub)
        m.capture_trace(**st, mesh_device=sub, cq_id=0)
        submodels.append(m)
        subtensors.append(st)
    for _ in range(3):
        for m in submodels:
            m.execute_trace(blocking=True)

    # V1: sequential nonblocking launch, sync both at end
    v1 = []
    for _ in range(NUM_ITERATIONS):
        t = time.perf_counter()
        for m in submodels:
            m.execute_trace(blocking=False, synchronize=False)
        for sub in submeshes:
            ttnn.synchronize_device(sub)
        v1.append((time.perf_counter() - t) * 1e3)
    logger.info(f"V1 two-1x1-seq-launch: avg={sum(v1)/len(v1):.2f}ms best={min(v1):.2f}ms")

    # V2: threaded concurrent launch
    pool = cf.ThreadPoolExecutor(max_workers=2)
    v2 = []
    for _ in range(NUM_ITERATIONS):
        t = time.perf_counter()
        futs = [pool.submit(m.execute_trace, blocking=True, synchronize=True) for m in submodels]
        for f in futs:
            f.result()
        v2.append((time.perf_counter() - t) * 1e3)
    pool.shutdown()
    logger.info(f"V2 two-1x1-threaded: avg={sum(v2)/len(v2):.2f}ms best={min(v2):.2f}ms")

    for m in submodels:
        m.release_trace()

    logger.info(f"SUMMARY  V0={min(v0):.1f}ms  V1={min(v1):.1f}ms  V2={min(v2):.1f}ms  (best-of)")
