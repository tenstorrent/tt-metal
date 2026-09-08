# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH decode per-op profiling harness (TP, real checkpoint layers). A layer subset keeps the device-profiler buffers
from overflowing (the 64-layer eager prefill alone exhausts them before decode starts): 128-token traced prefill, then N eager
decode steps (each op gets host+device records) and finally a traced decode timed over replays.

  MESH_DEVICE=P150x4 python -m tracy -r -v --op-support-count 5000 -m pytest \
      models/demos/blackhole/qwen36/tests/test_decode_profile_scratch.py -k decode_L8 -s
"""
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import copy_host_to_device

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 1 << 30,
    }
]
_MESH_SHAPE = (1, 4)
BLOCK_SIZE = 64  # same paging as test_prefill_profile_scratch.py
CHUNK = 2048


def _blocks_for(isl):
    bucket = ((isl + CHUNK - 1) // CHUNK) * CHUNK
    blocks = max(64, (bucket + 256 + BLOCK_SIZE - 1) // BLOCK_SIZE)
    return ((blocks + 31) // 32) * 32


@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "layers, steps",
    [
        pytest.param("0,1,2,3,4,5,6,7", 6, id="decode_L8"),
        pytest.param("0,1,2,3", 6, id="decode_L4"),
        pytest.param("all", 4, id="decode_all"),
        pytest.param("0", 2, id="decode_L1"),
    ],
)
def test_decode_profile(mesh_device, layers, steps):
    device = mesh_device
    device.enable_program_cache()
    layer_indices = None if layers == "all" else [int(x) for x in layers.split(",")]
    isl = 128
    num_blocks = _blocks_for(isl)
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len, layer_indices=layer_indices)
    logger.info(f"[DECODE_PROF] model load {time.time() - t0:.1f}s layers={model.layer_indices}")
    logger.info(f"[DECODE_PROF] ccl links axis0={model.tt_ccl.get_num_links(0)} axis1={model.tt_ccl.get_num_links(1)}")
    kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    # no masked-bucket warm-ups: the tracy ops report stops after ~4600 device ops, so keep the pre-decode op count small
    model.capture_prefill_trace_chunked(model.mesh_device, page_table, chunk_size=CHUNK, warmup_masked_buckets=False)

    torch.manual_seed(0)
    token_ids = torch.randint(1000, 20000, (1, isl), dtype=torch.long)
    _ = model.prefill_traced_chunked(token_ids, page_table, actual_len=isl)
    nxt = 1000  # the token value is irrelevant for profiling
    T = isl

    dev = model.prepare_inputs_decode(
        torch.tensor([[nxt]], dtype=torch.int32), torch.tensor([T], dtype=torch.int32), page_table=page_table
    )

    logger.info(
        f"[DECODE_PROF] dev shapes: tokens {dev[0].shape} padded {dev[0].padded_shape} pos {dev[1].shape} rope {dev[2].shape}"
    )
    _x = model.embd(dev[0])
    logger.info(f"[DECODE_PROF] embd out shape {_x.shape} padded {_x.padded_shape} dtype {_x.dtype} layout {_x.layout}")
    ttnn.deallocate(_x)

    def _decode_fwd():
        return model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])[0]

    def _update(token, position):
        host = model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int32),
            page_table=page_table,
        )
        copy_host_to_device(host[:3], device_tensors=dev[:3])

    # eager decode steps (compile on the first)
    pos = T
    eager_times = []
    dumps, toks = [], []
    for i in range(steps + 1):
        signpost(f"decode_eager_{i}")
        ttnn.synchronize_device(device)
        t1 = time.time()
        out = _decode_fwd()
        ttnn.synchronize_device(device)
        eager_times.append(time.time() - t1)
        if i == 0:
            logger.info(f"[DECODE_PROF] logits out shape {out.shape} padded {out.padded_shape}")
        lg = model.process_output_decode(out, B=1, S=1).reshape(-1)
        nxt = int(torch.argmax(lg).item())
        dumps.append(lg.float().clone())
        toks.append(nxt)
        pos += 1
        _update(nxt, pos)
    signpost("decode_eager_end")
    logger.info(f"[DECODE_PROF] eager step wall ms: {['%.1f' % (t * 1e3) for t in eager_times]}")
    logger.info(f"[DECODE_PROF] tokens {toks}")
    if os.environ.get("DECODE_PROF_DUMP"):
        torch.save({"logits": torch.stack(dumps), "tokens": toks}, os.environ["DECODE_PROF_DUMP"])

    # traced decode: capture once, time replays (skip under the device profiler: tracy's report asserts on trace ops)
    if os.environ.get("DECODE_PROF_NO_TRACE") == "1":
        print(
            f"DECODE_PROF_RESULT layers={len(model.layers)} traced_step_ms=nan eager_step_ms={min(eager_times[1:]) * 1e3:.2f}"
        )
        return
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    tt_logits = _decode_fwd()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    signpost("decode_trace")
    reps = 10
    t1 = time.time()
    for _ in range(reps):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    per = (time.time() - t1) / reps
    signpost("decode_trace_end")
    n_layers = len(model.layers)
    logger.info(f"[DECODE_PROF] traced step {per * 1e3:.2f} ms for {n_layers} layers")
    print(
        f"DECODE_PROF_RESULT layers={n_layers} traced_step_ms={per * 1e3:.3f} eager_step_ms={min(eager_times[1:]) * 1e3:.2f}"
    )
    ttnn.release_trace(device, trace_id)
