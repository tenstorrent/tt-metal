# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 B12/S8192 DATA-parallel (DP=2) benchmark on one N300.

Each chip is an independent replica running the full single-chip forward on its
batch shard (B/2 = 6), full sequence 8192. NO inter-chip collectives (no K/V
all-gather) - attention is standard single-chip SDPA over the full local
sequence, i.e. exact full attention. Inputs are sharded on the BATCH dim (0).

Activated via create_tt_model(..., data_parallel=True).
"""

import time

import pytest
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tests.perf.perf import NUM_ITERATIONS, prepare_inputs
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

BATCH = 12
SEQ_LEN = 8192


def _to_batchsharded_tensors(inputs, mesh_device, *, device):
    # Shard input_ids / token_type / position on the BATCH dim (tensor dim 0)
    # across the 2-chip mesh. Each chip gets B/2 sequences, full length.
    mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    kwargs = {"mesh_mapper": mapper}
    if device:
        kwargs.update(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def convert(tensor, dtype, layout):
        return ttnn.from_torch(tensor, dtype=dtype, layout=layout, **kwargs)

    return {
        "input_ids": convert(inputs["input_ids"].int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "token_type_ids": convert(inputs["token_type_ids"].int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "position_ids": convert(inputs["position_ids"].int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    }


def _copy_inputs(host_tensors, device_tensors):
    for key in host_tensors:
        ttnn.copy_host_to_device_tensor(host_tensors[key], device_tensors[key])


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_embedding_perf_b12_s8192_dp2(mesh_device):
    assert tuple(mesh_device.shape) == (2, 1)
    assert mesh_device.get_num_devices() == 2

    t0 = time.perf_counter()
    args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=BATCH,
        max_seq_len=SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        data_parallel=True,
        use_experimental_encoder_sdpa=True,
    )
    assert model._data_parallel, "DP mode not active"
    logger.info(f"DP2 model built in {time.perf_counter() - t0:.1f}s")

    inputs = prepare_inputs(args.tokenizer, BATCH, SEQ_LEN, args.pad_token_id)
    host_tensors = _to_batchsharded_tensors(inputs, mesh_device, device=False)
    device_tensors = _to_batchsharded_tensors(inputs, mesh_device, device=True)

    logger.info("Compiling DP2 forward")
    out = model.forward(**device_tensors)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)

    logger.info("Capturing DP2 trace")
    model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)
    for _ in range(3):
        model.execute_trace(blocking=True)

    times = []
    for _ in range(NUM_ITERATIONS):
        fresh = prepare_inputs(args.tokenizer, BATCH, SEQ_LEN, args.pad_token_id)
        fresh_host = _to_batchsharded_tensors(fresh, mesh_device, device=False)
        _copy_inputs(fresh_host, device_tensors)
        t0 = time.perf_counter()
        model.execute_trace(blocking=True)
        times.append((time.perf_counter() - t0) * 1000.0)

    model.release_trace()
    avg_ms = sum(times) / len(times)
    best_ms = min(times)
    logger.info(f"Avg DP2 prefill time: {avg_ms:.3f}ms")
    logger.info(f"Best DP2 prefill time: {best_ms:.3f}ms")
    logger.info(f"Avg DP2 tokens/s: {BATCH * SEQ_LEN / (avg_ms / 1000.0):.0f}")
