# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op device profile of K3.1's first layers as the runner runs them, on an 8x4 stage and a 4x4 stage.

Built through the runner's own runtime (`k31_stage.window_params` -> `ADAPTER.build_runtime`): real weights from
the layout's TTNN cache, the runner's mesh and fabric (`k31_stage.PLACEMENT`), its context (1,064,960), chunk,
scaled-FP8 KV, TP-sharded KV and shared-expert overlap, traced. Layers [0, 4): KDA layers 0-2 and DSA layer 3,
the first rank of either layout, so every layer sits at the same AttnRes position on both.

Run it under the device profiler (`scripts/run_safe_pytest.sh --profile`). Two marked regions:

  K31_PROF_EAGER   one eager forward (the traced path's own warm-up forward), with a signpost before each layer
                   (K31_LAYER_<i>), which gives the op sequence of each layer;
  K31_PROF_TRACED  the runner's 10K request (the K3.1 golden prompt, 2 x 5120 or 4 x 2560 chunks) replayed
                   K31_PROF_REPEATS times from position 0, one trace replay per chunk.

The traced replays run the eager forward's op sequence, so the eager signposts split them into layers.
Without the profiler the test only checks that the replays complete.
"""

import dataclasses
import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_K31, resolve_trace

FIRST_LAYER, NUM_LAYERS = 0, 4
# The runner's single 10K request (sc1): 10240 tokens.
REQUEST_LEN = 10240
REPEATS = int(os.environ.get("K31_PROF_REPEATS", "3"))


def _metadata_msg(mesh_device, slot_id: int, actual_start: int, actual_end: int) -> ttnn.Tensor:
    """The packed (slot, start, end) record the traced path reads on device; the socket's in serving."""
    return ttnn.from_torch(
        torch.tensor([slot_id, actual_start, actual_end], dtype=torch.int64).reshape(1, 1, 1, 3),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", k31_stage.PLACEMENT, indirect=["mesh_device", "device_params"])
def test_k31_layer_profile(mesh_device, device_params):
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    trace = resolve_trace(TRACE_K31)
    assert trace is not None, f"no K3.1 golden at {TRACE_K31}"
    ids = trace.metadata["token_ids"] + trace.metadata["completion_token_ids"]
    ids = (ids + [MTP_PAD_TOKEN_ID] * REQUEST_LEN)[:REQUEST_LEN]

    mesh_shape = tuple(mesh_device.shape)
    hf_config, params = k31_stage.window_params(
        mesh_shape, FIRST_LAYER, NUM_LAYERS, k31_stage.MAX_SEQ_LEN, is_first_rank=True, is_last_rank=False
    )
    params = dataclasses.replace(params, use_trace=True)
    chunk = params.chunk_size
    num_chunks = REQUEST_LEN // chunk
    runtime = k31_stage.ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    try:
        kv = k31_stage.ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
        if profiling:
            ttnn.ReadDeviceProfiler(mesh_device)  # the weight upload alone can fill the buffer
        runtime.compile(kv)  # traced: persistent buffers + one warm forward
        ttnn.synchronize_device(mesh_device)
        if profiling:
            ttnn.ReadDeviceProfiler(mesh_device)

        # One more eager forward of the traced path, each layer behind a signpost.
        layers = runtime.model.layers
        originals = [layer.forward for layer in layers]

        def marked(i, forward):
            def forward_after_signpost(*args, **kwargs):
                signpost(header=f"K31_LAYER_{FIRST_LAYER + i}")
                return forward(*args, **kwargs)

            return forward_after_signpost

        for i, layer in enumerate(layers):
            layer.forward = marked(i, originals[i])
        signpost(header="K31_PROF_EAGER")
        try:
            runtime._forward_traced(kv)
            ttnn.synchronize_device(mesh_device)
        finally:
            for layer, forward in zip(layers, originals):
                layer.forward = forward
        signpost(header="K31_PROF_EAGER_END")
        if profiling:
            ttnn.ReadDeviceProfiler(mesh_device)

        runtime.capture_trace(kv)
        ttnn.synchronize_device(mesh_device)
        if profiling:
            ttnn.ReadDeviceProfiler(mesh_device)

        signpost(header="K31_PROF_TRACED")
        for r in range(REPEATS):
            for c in range(num_chunks):
                msg = _metadata_msg(mesh_device, 0, c * chunk, (c + 1) * chunk)
                runtime.prefill_chunk(
                    runtime.make_chunk_input(ids[c * chunk : (c + 1) * chunk]),
                    kv,
                    slot_id=0,
                    actual_start=c * chunk,
                    actual_end=(c + 1) * chunk,
                    request_id=c,
                    metadata_msg=msg,
                )
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(msg)
                # One chunk's replay per read keeps the device profiler buffer from overflowing.
                if profiling:
                    ttnn.ReadDeviceProfiler(mesh_device)
            logger.info(f"K31_PROF {mesh_shape} repeat {r}: {num_chunks} x {chunk}-token chunks replayed")
        signpost(header="K31_PROF_TRACED_END")
    finally:
        runtime.release_trace()
