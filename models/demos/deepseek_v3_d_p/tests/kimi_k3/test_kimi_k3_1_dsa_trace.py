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
KDA + DSA (layers 0-3), where the KDA carries take their slot from the same metadata. Dense-FFN stand-in
for the MoE and synthetic weights, as `test_kimi_k3_1_dsa_block.py`.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config, kimi_k3_1_hf_config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_kimi_k3_1_dsa_block import (
    CHUNK,
    MAX_SEQ_LEN,
    WINDOWS,
    _DenseFfnK31,
    _shards,
    _synthetic_state_dict,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import SP_AXIS, TP_AXIS, _compose
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
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
# The block test's placement plus a trace region; a few layers' capture fits well inside the runner's 256 MB.
PLACEMENTS = [
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "l1_small_size": KimiK31Config.L1_SMALL_SIZE,
            "trace_region_size": 64 * 1024 * 1024,
        },
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]
TRACE_WINDOWS = ["dsa_L3", "dsa_L3_L7", "hybrid_L0_3"]


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
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("window", TRACE_WINDOWS, ids=TRACE_WINDOWS)
def test_k31_dsa_traced_replay_matches_eager(mesh_device, device_params, window):
    first_layer_idx, num_layers, kv_only = WINDOWS[window]
    assert not kv_only
    adapter = get_adapter("kimi_k3_1")
    config = kimi_k3_1_hf_config(max_seq=MAX_SEQ_LEN)
    params = PrefillRunParams(
        mesh_shape=tuple(mesh_device.shape),
        num_layers=num_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=CHUNK,
        num_users=NUM_USERS,
        capacity_factor=2,
        num_links=1,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        tp_shard_kv=True,
        sparse_kv_cache_format=None,
    )
    model = TtKimiK3Transformer(
        mesh_device,
        config,
        _DenseFfnK31,
        _synthetic_state_dict(config, first_layer_idx, num_layers),
        num_layers=num_layers,
        seq_len=CHUNK,
        first_layer_idx=first_layer_idx,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=MAX_SEQ_LEN,
        is_chunked=True,
        slot_num=NUM_USERS,
        sparse_kv_cache_format=adapter.resolve_sparse_kv_cache_format(params.sparse_kv_cache_format),
        tp_shard_kv=True,
    )
    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=config, params=params)
    kvpe, index = caches.kvpe, caches.index
    model.kda_states = caches.kda_states
    # Fresh caches, restored in place before the traced pass so both passes start from the same state.
    pristine = (kvpe.storage.cpu(), index.cpu())

    mesh_shape = tuple(mesh_device.shape)
    streams = [
        torch.randint(0, config.vocab_size, (3 * CHUNK,), generator=torch.Generator().manual_seed(7 + request)).tolist()
        for request in range(NUM_REQUESTS)
    ]

    def tokens(request, start):
        return prepare_prefill_input_tensor(
            streams[request][start : start + CHUNK], mesh_device, mesh_shape[SP_AXIS], False, mesh_shape, SP_AXIS
        )

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
        return model.forward(
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
            out = model.forward(
                chunk_tokens,
                kvpe_cache=kvpe,
                index_kv_cache=index,
                actual_isl=end - start,
                actual_start=start,
                actual_end=end,
                cache_user_id=slot,
            )
            ttnn.deallocate(chunk_tokens)
            eager.append((_compose(mesh_device, out), *caches_now()))
            ttnn.deallocate(out)
        for (_, kvpe_a, index_a), (_, kvpe_b, index_b) in zip(eager, eager[1:]):
            assert not torch.equal(kvpe_a, kvpe_b) and not torch.equal(index_a, index_b), "a chunk wrote no cache"

        # Capture on slot 0 at position 0, after two warm forwards settle programs and persistent resources.
        model.set_trace_controller(controller)
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
            assert torch.equal(_compose(mesh_device, traced_out), out_want), f"{label}: traced output differs"
    finally:
        if captured:
            controller.release()
        model.set_trace_controller(None)
        if traced_out is not None:
            ttnn.deallocate(traced_out)
        if caches.kda_states is not None:
            caches.kda_states.deallocate()
        for tensor in (input_tt, *metadata.scalars):
            ttnn.deallocate(tensor)
        model.release_sub_device_managers()
