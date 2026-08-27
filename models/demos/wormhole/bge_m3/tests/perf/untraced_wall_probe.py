"""Measure UNTRACED full-forward wall (host-streamed) vs the traced wall.
If untraced wall ~= device span (~1008ms) and traced ~= 1164ms, the ~156ms is
trace-replay dispatch overhead."""
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
def test_untraced_wall(mesh_device):
    args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=BATCH,
        max_seq_len=SEQ,
        dtype=ttnn.bfloat8_b,
        data_parallel=True,
        use_experimental_encoder_sdpa=True,
        mlp_wi_output_dtype=ttnn.bfloat4_b,
    )
    inp = prepare_inputs(args.tokenizer, BATCH, SEQ, args.pad_token_id)
    dev = _to_batchsharded_tensors(inp, mesh_device, device=True)
    # warmup
    for _ in range(3):
        out = model.forward(**dev)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)
    N = 10
    t = []
    enqueue = []
    sync = []
    for _ in range(N):
        t0 = time.perf_counter()
        out = model.forward(**dev)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        enqueue.append((t1 - t0) * 1e3)
        sync.append((t2 - t1) * 1e3)
        t.append((t2 - t0) * 1e3)
        ttnn.deallocate(out)
    t.sort()
    enqueue.sort()
    sync.sort()
    logger.info(f"UNTRACED full-forward wall: min={min(t):.1f} med={t[len(t)//2]:.1f} ms")
    logger.info(f"UNTRACED enqueue: med={enqueue[len(enqueue)//2]:.1f} ms; sync: med={sync[len(sync)//2]:.1f} ms")
