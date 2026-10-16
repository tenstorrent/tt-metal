# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded sliding attention (GEMMA4_D_P_BOUNDED_SWA) against the unbounded sliding cache.

One trace, captured at chunk 0, replays the chunks of interleaved requests with only the device metadata refreshed
between replays. Each replay writes the chunk to a bounded (two-chunk, circular) and an unbounded sliding cache and
attends both through the CP ring. The first request wraps the bounded cache twice. The requests share each cache, so a
write that missed the wrap would land in another request's slot.

The first two requests end in a partial chunk, padded to the full chunk as the runtime pads it. The padding is large,
so a real token that read a pad row would fail the reference check. One request ends inside a middle CP rank's slab,
the other inside rank 0's slab. A third request then reuses the first one's slot, whose bounded slabs still hold that
request's last two chunks; its second chunk would read them if its first chunk had been written to the wrong slab.
A fourth request, shorter than one chunk, reuses the second one's slot: its only chunk must ignore the stale previous
slab, which still holds the second request's chunk.

The chunk size sets how many CP predecessors the 1024-token window reaches (slab = chunk / CP). 2048 reaches several.
1024 reaches all CP ranks, so the window wraps onto the rank's own slab of the previous chunk, which the op reads from
the local circular cache. The production chunk, 8192, reaches one neighbour; test_bounded_swa_model.py covers it.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_d_p.config import GALAXY_MESH_SHAPES, MeshConfig
from models.demos.gemma4_d_p.tt.attention.ring_prefill import (
    bounded_sliding_cache_capacity,
    init_sliding_ring_kv_cache,
    sliding_ring_prefill_attention,
    write_chunk_to_sliding_ring_cache,
)
from models.demos.gemma4_d_p.tt.ccl import CCLManager
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata

# Gemma4-31B sliding attention.
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, SLIDING_WINDOW = 32, 16, 256, 1024
# Request -> (slot, chunks, CP rank its final chunk ends in as a fraction of CP, or None for a full final chunk).
# Request 0 ends inside a middle rank's slab and request 1 inside rank 0's. Request 2 reuses slot 0 after request 0.
# Request 3, shorter than one chunk, reuses slot 1 after request 1 and ends inside a late rank's slab.
REQUESTS = ((0, 5, 0.5), (1, 3, 0.0), (0, 2, None), (1, 1, 0.75))
# (request, chunk) in replay order: request 0 wraps the two-slab cache twice while request 1 starts between its chunks.
SCHEDULE = ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (0, 3), (1, 2), (0, 4), (2, 0), (2, 1), (3, 0))
# Scale of the rows that pad a partial chunk.
PAD_SCALE = 100.0
MIN_PCC = 0.99
# Query rows per reference block, which keeps the score matrix small at the 8192 chunk.
REFERENCE_BLOCK = 1024


def _real_tokens(request, chunk, chunk_size, cp):
    """Real tokens in a chunk. A partial final chunk ends partway into its rank's slab, off a tile boundary."""
    _, chunks, end_rank = REQUESTS[request]
    if chunk != chunks - 1 or end_rank is None:
        return chunk_size
    slab = chunk_size // cp
    return int(end_rank * cp) * slab + slab // 2 + 7


def _reference(q, k, v, start, scale):
    """Causal sliding-window GQA for the chunk at ``start``; k and v hold every token of its request up to the
    chunk end."""
    group = q.shape[0] // k.shape[0]
    out = torch.empty(q.shape, dtype=torch.float32)
    for block in range(0, q.shape[1], REFERENCE_BLOCK):
        q_block = q[:, block : block + REFERENCE_BLOCK].float()
        q_start = start + block
        q_end = q_start + q_block.shape[1]
        first = max(0, q_start - SLIDING_WINDOW + 1)
        k_block = k[:, first:q_end].float().repeat_interleave(group, dim=0)
        v_block = v[:, first:q_end].float().repeat_interleave(group, dim=0)
        query_pos = torch.arange(q_start, q_end)[:, None]
        key_pos = torch.arange(first, q_end)[None, :]
        visible = (key_pos <= query_pos) & (query_pos - key_pos < SLIDING_WINDOW)
        scores = (q_block @ k_block.transpose(-1, -2) * scale).masked_fill(~visible, float("-inf"))
        out[:, block : block + REFERENCE_BLOCK] = torch.softmax(scores, dim=-1) @ v_block
    return out


@torch.no_grad()
@pytest.mark.timeout(540)
@pytest.mark.parametrize("chunk_size", [1024, 2048], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            shape,
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 50_000_000,
                "require_exact_physical_num_devices": True,
            },
            id=f"{shape[0]}x{shape[1]}",
        )
        for shape in GALAXY_MESH_SHAPES
    ],
    indirect=True,
)
def test_bounded_swa_matches_unbounded_cache(mesh_device, chunk_size):
    mesh_config = MeshConfig(mesh_device)
    ccl_manager = CCLManager(mesh_config)
    tp = mesh_config.tp_degree
    cp = mesh_config.cp_degree
    num_users = 1 + max(slot for slot, _, _ in REQUESTS)
    max_chunks = max(chunks for _, chunks, _ in REQUESTS)
    bounded_capacity = bounded_sliding_cache_capacity(chunk_size)
    unbounded_capacity = max_chunks * chunk_size
    scale = HEAD_DIM**-0.5

    torch.manual_seed(0)
    q_all, k_all, v_all = [], [], []
    for request, (_, chunks, _) in enumerate(REQUESTS):
        tokens = chunks * chunk_size
        real_end = tokens - chunk_size + _real_tokens(request, chunks - 1, chunk_size, cp)
        for tensors, heads in ((q_all, NUM_Q_HEADS), (k_all, NUM_KV_HEADS), (v_all, NUM_KV_HEADS)):
            tensor = torch.randn(heads, tokens, HEAD_DIM, dtype=torch.bfloat16)
            tensor[:, real_end:] *= PAD_SCALE
            tensors.append(tensor)

    # Sequence on the CP rows, heads on the TP columns: each rank holds one contiguous slab of the chunk.
    mesh_dims = [None, None]
    mesh_dims[mesh_config.cp_axis] = 2
    mesh_dims[mesh_config.tp_axis] = 1
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=mesh_dims)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), dims=mesh_dims)

    def host_chunk(tensors, request, chunk):
        rows = tensors[request][:, chunk * chunk_size : (chunk + 1) * chunk_size].unsqueeze(0)
        return ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper)

    tt_q, tt_k, tt_v = (ttnn.to_device(host_chunk(t, 0, 0), mesh_device) for t in (q_all, k_all, v_all))
    metadata = PrefillMetadata(mesh_config, circular_capacity=bounded_capacity)
    bounded = init_sliding_ring_kv_cache(
        mesh_config, NUM_KV_HEADS // tp, HEAD_DIM, bounded_capacity, num_users=num_users
    )
    unbounded = init_sliding_ring_kv_cache(
        mesh_config, NUM_KV_HEADS // tp, HEAD_DIM, unbounded_capacity, num_users=num_users
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    def stage(request, chunk):
        for tensors, device_tensor in ((q_all, tt_q), (k_all, tt_k), (v_all, tt_v)):
            ttnn.copy_host_to_device_tensor(host_chunk(tensors, request, chunk), device_tensor)
        metadata.update(slot_idx=REQUESTS[request][0], kv_actual_global=chunk * chunk_size)

    def attend(cache, capacity, circular, buffer_key):
        write_chunk_to_sliding_ring_cache(
            cache.k,
            cache.v,
            tt_k,
            tt_v,
            mesh_config,
            kv_actual_global=0,
            prefill_metadata=metadata,
            circular_kv_cache=circular,
        )
        return sliding_ring_prefill_attention(
            tt_q,
            cache.k,
            cache.v,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            prefill_metadata=metadata,
            num_local_kv_heads=NUM_KV_HEADS // tp,
            head_dim=HEAD_DIM,
            max_seq_len=capacity,
            logical_n=capacity,
            kv_actual_global=0,
            gather_buffer_key=buffer_key,
            sliding_window_size=SLIDING_WINDOW,
            scale=scale,
            compute_kernel_config=compute_kernel_config,
            circular_kv_cache=circular,
        )

    def forward():
        return attend(bounded, bounded_capacity, True, 0), attend(unbounded, unbounded_capacity, False, 1)

    # Compile, then capture with the first chunk staged. The capture writes chunk 0 of request 0 again, harmlessly.
    stage(*SCHEDULE[0])
    for output in forward():
        output.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = forward()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for request, chunk in SCHEDULE:
            stage(request, chunk)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            out_bounded, out_unbounded = (ttnn.to_torch(output, mesh_composer=mesh_composer)[0] for output in outputs)
            real = _real_tokens(request, chunk, chunk_size, cp)
            start = chunk * chunk_size
            end = start + real
            q, k, v = (tensors[request] for tensors in (q_all, k_all, v_all))
            expected = _reference(q[:, start:end], k[:, :end], v[:, :end], start, scale)
            label = f"request {request} chunk {chunk} ({real} real tokens, slab {chunk % 2})"
            assert torch.equal(out_bounded, out_unbounded), f"{label}: bounded output differs from unbounded"
            passed, pcc = comp_pcc(expected, out_bounded[:, :real].float(), MIN_PCC)
            assert passed, f"{label}: PCC {pcc} against the torch reference"
    finally:
        ttnn.release_trace(mesh_device, trace_id)
