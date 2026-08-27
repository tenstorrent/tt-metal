"""Measure the traced-replay wall breakdown: enqueue-return vs full-completion.
Separates host-launch latency, device execution, and completion-wait."""
import time

import pytest
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tests.perf.dp2_perf import _to_batchsharded_tensors, prepare_inputs
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

SEQ = 8192
BATCH = 12


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 50_000_000, "num_command_queues": 1}], indirect=True)
def test_trace_timing(mesh_device):
    args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=BATCH,
        max_seq_len=SEQ,
        dtype=ttnn.bfloat8_b,
        data_parallel=True,
        use_experimental_encoder_sdpa=True,
        encoder_sdpa_q256_vbf4=True,
        use_qkv_scatter_matmul=True,
    )
    inp = prepare_inputs(args.tokenizer, BATCH, SEQ, args.pad_token_id)
    dev = _to_batchsharded_tensors(inp, mesh_device, device=True)
    out = model.forward(**dev)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    model.capture_trace(**dev, mesh_device=mesh_device, cq_id=0)
    for _ in range(3):
        model.execute_trace(blocking=True)

    N = 15
    # A: enqueue-only (non-blocking) — host time to ISSUE the trace command
    enq = []
    full = []
    for _ in range(N):
        t0 = time.perf_counter()
        ttnn.execute_trace(model._trace_device, model._trace_id, cq_id=0, blocking=False)
        t1 = time.perf_counter()  # returned after enqueue, before device done
        ttnn.synchronize_device(model._trace_device)
        t2 = time.perf_counter()  # device fully done
        enq.append((t1 - t0) * 1e3)
        full.append((t2 - t0) * 1e3)
    burst = 4
    ttnn.synchronize_device(model._trace_device)
    t0 = time.perf_counter()
    for _ in range(burst):
        ttnn.execute_trace(model._trace_device, model._trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(model._trace_device)
    burst_ms = (time.perf_counter() - t0) * 1e3

    model.release_trace()
    enq.sort()
    full.sort()
    logger.info(f"TRACE enqueue-return (host issue): min={min(enq):.2f} med={enq[len(enq)//2]:.2f} ms")
    logger.info(f"TRACE full (issue+device+sync):   min={min(full):.2f} med={full[len(full)//2]:.2f} ms")
    logger.info(f"=> device-execution+completion (full - enqueue): {min(full)-min(enq):.2f} ms")
    logger.info(
        f"TRACE burst throughput: {burst} replays in {burst_ms:.2f} ms = "
        f"{burst_ms / burst:.2f} ms/replay, {1000.0 * burst / burst_ms:.3f} req/s"
    )
