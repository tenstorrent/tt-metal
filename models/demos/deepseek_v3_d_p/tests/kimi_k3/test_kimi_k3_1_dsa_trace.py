# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One captured K3.1 forward serves every user slot, chunk position and valid end of its DSA layers.

PREFILL_USE_TRACE=1 records the chunk forward once and replays it with the per-chunk slot, start and end
read on device from the persistent `metadata` the runtime rewrites before every replay
(`TtPrefillRuntime._forward_traced`). A Gated DSA layer makes five per-request decisions from them: the
KVPE and index-key cache writes (slot, start, real-token clamp), the indexer's causal score extent and
cache slot, the top-k valid count (the scored prefix, ceil32(actual_end) capped at the chunk end), and the
KVPE prefix gather. Any of them frozen at capture leaves the capture-time chunk exact and every other chunk
wrong, so the capture is recorded on slot 0 at position 0 (valid count 5120) and replayed on:

  user A (slot 0) at 0, user B (slot 1) at 0, B at 5120, A at 5120, A at 10240 with a partial chunk, then
  user C starting a new 1000-token request on B's slot 1.

Within each chunk at 0 the causal mask leaves rows below 2048 fewer than top-k keys (the rest sentinel
indices); the later chunks select from a 10240- and 11264-token prefix of their own slot. User C's whole
context is below 2048: its valid count, 1024, takes top-k's short search (at most 2048 keys) under a
capture whose count took the long one. It also reuses a slot without clearing it, and in the KDA window
its head must load a zero carry over B's.

Reference: the same model run eagerly from host ints over the same schedule, from the same pristine
caches. Every replay's output and both caches (every slot, every row) must match eager bit for bit after
that chunk: a replay that wrote or read the wrong slot or rows leaves a different cache or output.
Sensitivity, measured with the capture reading a stand-in word fixed at 0 (eager untouched): a pinned
slot fails chunk 1 (slot 1) and a pinned start fails chunk 2 (5120), each at the KVPE cache check.

Windows: the first DSA layer alone, two DSA layers (rank-local KV slots 0 and 1 of both caches), and
KDA + DSA (layers 0-3), where the KDA carries take their slot from the same metadata. Synthetic weights
and dense-FFN stand-in for the MoE from `synthetic.py`, as `test_kimi_k3_1_dsa_block.py`.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import (
    CHUNK,
    build_stack,
    compose,
    placements,
    release,
    synthetic_state_dict,
    token_ids,
    upload_tokens,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_kimi_k3_1_dsa_block import _shards
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

NUM_USERS = 2
# (request, slot, actual_start, actual_end): requests A (0) and B (1) interleave their chunks, A ends on a
# partial chunk, then request C (2) starts a short request on B's slot. Each request has its own tokens.
CHUNKS = [
    (0, 0, 0, CHUNK),
    (1, 1, 0, CHUNK),
    (1, 1, CHUNK, 2 * CHUNK),
    (0, 0, CHUNK, 2 * CHUNK),
    (0, 0, 2 * CHUNK, 2 * CHUNK + 1000),
    (2, 1, 0, 1000),
]
NUM_REQUESTS = 3
# A few layers' capture fits well inside the runner's 256 MB trace region.
TRACE_REGION_SIZE = 64 * 1024 * 1024
# (first_layer_idx, num_layers): one DSA layer, two DSA layers (KV slots 0 and 1), and KDA + DSA layers 0-3.
TRACE_WINDOWS = {"dsa_L3": (3, 1), "dsa_L3_L7": (3, 5), "hybrid_L0_3": (0, 4)}


def _scalar(mesh_device, value: int) -> ttnn.Tensor:
    """A metadata word as the runtime allocates it (`TtPrefillRuntime._meta1_dev`)."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    placements(trace_region_size=TRACE_REGION_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("window", list(TRACE_WINDOWS), ids=list(TRACE_WINDOWS))
def test_k31_dsa_traced_replay_matches_eager(mesh_device, device_params, window):
    first_layer_idx, num_layers = TRACE_WINDOWS[window]
    state_dict = synthetic_state_dict(first_layer_idx, num_layers)
    stack, caches = build_stack(mesh_device, state_dict, first_layer_idx, num_layers, num_users=NUM_USERS)
    kvpe, index = caches.kvpe, caches.index
    # Fresh caches, restored in place before the traced pass so both passes start from the same state.
    pristine = (kvpe.storage.cpu(), index.cpu())

    streams = [token_ids(3 * CHUNK, seed=7 + request) for request in range(NUM_REQUESTS)]

    def tokens(request, start):
        return upload_tokens(mesh_device, streams[request][start : start + CHUNK])

    def caches_now():
        return _shards(mesh_device, kvpe.storage).view(torch.uint8), _shards(mesh_device, index)

    input_tt = tokens(0, 0)
    metadata = ChunkMetadata(_scalar(mesh_device, 0), _scalar(mesh_device, 0), _scalar(mesh_device, CHUNK), None)

    def stage(request, slot, start, end):
        """Rewrite the persistent input and metadata in place, as the runtime does before every replay."""
        for source, destination in (
            (tokens(request, start), input_tt),
            (_scalar(mesh_device, slot), metadata.slot_id),
            (_scalar(mesh_device, start), metadata.actual_start),
            (_scalar(mesh_device, end), metadata.actual_end),
        ):
            ttnn.copy(source, destination)
            ttnn.deallocate(source)

    def forward_traced():
        # As `TtPrefillRuntime._forward_traced`: host bounds absent, the slot word on device.
        return stack.forward(
            input_tt,
            kvpe_cache=kvpe,
            index_kv_cache=index,
            actual_isl=CHUNK,
            actual_start=None,
            actual_end=None,
            cache_user_id=0,
            metadata=metadata,
        )

    controller = SubDeviceTraceController(mesh_device)
    captured = False
    traced_out = None
    try:
        eager = []
        for request, slot, start, end in CHUNKS:
            logger.info(f"{window}: eager slot {slot} [{start}, {end})")
            chunk_tokens = tokens(request, start)
            out = stack.forward(
                chunk_tokens,
                kvpe_cache=kvpe,
                index_kv_cache=index,
                actual_isl=end - start,
                actual_start=start,
                actual_end=end,
                cache_user_id=slot,
            )
            ttnn.deallocate(chunk_tokens)
            eager.append((compose(mesh_device, out, CHUNK), *caches_now()))
            ttnn.deallocate(out)
        for (_, kvpe_a, index_a), (_, kvpe_b, index_b) in zip(eager, eager[1:]):
            assert not torch.equal(kvpe_a, kvpe_b) and not torch.equal(index_a, index_b), "a chunk wrote no cache"

        # Capture on slot 0 at position 0, after two warm forwards settle programs and persistent resources.
        stack.set_trace_controller(controller)
        stage(0, 0, 0, CHUNK)
        for _ in range(2):
            ttnn.deallocate(forward_traced())
        ttnn.synchronize_device(mesh_device)
        controller.begin_capture()
        traced_out = forward_traced()
        controller.end_capture()
        captured = True
        ttnn.synchronize_device(mesh_device)
        logger.info(f"{window}: captured {controller.num_segments} segments")

        ttnn.copy_host_to_device_tensor(pristine[0], kvpe.storage)
        ttnn.copy_host_to_device_tensor(pristine[1], index)
        for i, (request, slot, start, end) in enumerate(CHUNKS):
            logger.info(f"{window}: replay slot {slot} [{start}, {end})")
            stage(request, slot, start, end)
            controller.replay()
            out_want, kvpe_want, index_want = eager[i]
            kvpe_got, index_got = caches_now()
            label = f"{window} chunk {i} (slot {slot}, [{start}, {end}))"
            assert torch.equal(kvpe_got, kvpe_want), f"{label}: traced KVPE cache differs from eager"
            assert torch.equal(index_got, index_want), f"{label}: traced index cache differs from eager"
            assert torch.equal(compose(mesh_device, traced_out, CHUNK), out_want), f"{label}: traced output differs"
    finally:
        if captured:
            controller.release()
        stack.set_trace_controller(None)
        if traced_out is not None:
            ttnn.deallocate(traced_out)
        for tensor in (input_tt, *metadata.scalars):
            ttnn.deallocate(tensor)
        release(stack, caches)
