# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-model bounded sliding attention (GEMMA4_D_P_BOUNDED_SWA) against the unbounded model.

Two models share one set of device weights: one with the unbounded sliding caches, one with the bounded (two-chunk,
circular) ones. Each is traced once and replays the same real-text requests, with only the tokens, RoPE positions and
ring metadata refreshed between replays. The bounded model must match the unbounded one bit for bit:

- the final hidden states of every chunk, over its real rows;
- after the last replay, the global caches, and each sliding cache slab against the unbounded rows of the chunk it
  holds.

Request 0 fills the context and wraps the bounded caches repeatedly. Request 1 interleaves with it on the other slot.
Both end in a partial chunk padded with token 0. Request 2 then reuses request 0's slot, and request 3, shorter than
one chunk, reuses request 1's.
"""

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import (
    MODEL_DTYPE,
    _cp_gather_torch,
    _cp_or_replicate_mapper,
    _get_prefill_tokens,
    _hf_model_id,
    _shared_device_weights,
)
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model

NUM_SLOTS = 2
# Two full-model traces, one per model.
TRACE_REGION_SIZE = 512_000_000
# Layers whose caches are compared after the last replay: the first and last sliding and global layers.
CHECKED_LAYERS = (0, 5, 58, 59)
# (context, chunk) per mesh: 8x4 at the production chunk, where the window reaches one CP neighbour and sliding
# SDPA does not split K, and 4x8 at the short slab, where it reaches two and K is split.
CASES = {(8, 4): (65536, 8192), (4, 8): (32768, 2048)}


def _requests(context_len, chunk_size, cp):
    """(slot, real tokens) per request. Request 0 fills the context and ends inside a middle CP rank's slab;
    request 1 runs three chunks and ends inside rank 0's; request 2 reuses slot 0 with two full chunks; request 3
    reuses slot 1 with a single partial chunk that ends inside a late rank's slab."""
    slab = chunk_size // cp
    return (
        (0, context_len - chunk_size + (cp // 2) * slab + slab // 2 + 7),
        (1, 2 * chunk_size + slab // 2 + 7),
        (0, 2 * chunk_size),
        (1, (3 * cp // 4) * slab + slab // 2 + 7),
    )


def _schedule(requests, chunk_size):
    """(request, chunk) in replay order: request 1's chunks interleave with request 0's from its third chunk on,
    then requests 2 and 3 run."""
    chunks = [-(-real // chunk_size) for _, real in requests]
    assert chunks[1] <= chunks[0] - 2, "request 1 must finish while request 0 runs"
    order = [(0, 0), (0, 1)]
    for chunk in range(2, chunks[0]):
        if chunk - 2 < chunks[1]:
            order.append((1, chunk - 2))
        order.append((0, chunk))
    order += [(2, chunk) for chunk in range(chunks[2])]
    order += [(3, chunk) for chunk in range(chunks[3])]
    return order


def _device_rows(tensor, slot, row_start, row_end):
    """One slot's local cache rows on every device, in mesh order."""
    selected = ttnn.slice(
        tensor,
        (slot, 0, row_start, 0),
        (slot + 1, tensor.shape[1], row_end, tensor.shape[3]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(selected)]
    finally:
        ttnn.deallocate(selected)


@torch.no_grad()
@pytest.mark.timeout(540)
@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_bounded_swa_model_matches_unbounded(mesh_device):
    context_len, chunk_size = CASES[tuple(mesh_device.shape)]
    mesh_config = MeshConfig(mesh_device)
    hf_model_id = _hf_model_id()
    with _shared_device_weights():
        models = [
            create_tt_model(
                mesh_config=mesh_config,
                prefill_chunk_size=chunk_size,
                max_batch_size=NUM_SLOTS,
                max_seq_len=context_len,
                dtype=MODEL_DTYPE,
                hf_model_id=hf_model_id,
                bounded_sliding_window_attention=bounded,
            )[1]
            for bounded in (False, True)
        ]
    unbounded, bounded = models
    vocab_size = unbounded.vocab_size
    # Distinct requests: the text stream at different offsets.
    stream = _get_prefill_tokens(hf_model_id, 2 * context_len, vocab_size, "text")
    requests = _requests(context_len, chunk_size, mesh_config.cp_degree)
    request_tokens = []
    for request, (_, real) in enumerate(requests):
        padded = -(-real // chunk_size) * chunk_size
        tokens = stream[:, request * chunk_size : request * chunk_size + padded].clone()
        tokens[:, real:] = 0
        request_tokens.append(tokens)

    def host_ids(values):
        return ttnn.from_torch(
            values.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_cp_or_replicate_mapper(mesh_config, seq_dim=-1),
        )

    device_tokens = ttnn.to_device(host_ids(request_tokens[0][:, :chunk_size]), mesh_device)
    device_positions = ttnn.to_device(host_ids(torch.arange(chunk_size).unsqueeze(0)), mesh_device)
    for model in models:
        model.set_prefill_rope_positions(device_positions)
        model._prefill_metadata_external = True

    def stage(request, chunk):
        start = chunk * chunk_size
        ttnn.copy_host_to_device_tensor(host_ids(request_tokens[request][:, start : start + chunk_size]), device_tokens)
        ttnn.copy_host_to_device_tensor(
            host_ids(torch.arange(start, start + chunk_size).unsqueeze(0)), device_positions
        )
        for model in models:
            model.prefill_metadata.update(slot_idx=requests[request][0], kv_actual_global=start)

    def forward(model):
        return model(hidden_states=model.transform_and_embed_prefill_inputs_device(device_tokens))

    trace_ids, outputs = [], []
    try:
        for model in models:
            stage(0, 0)
            forward(model).deallocate(True)
            stage(0, 0)
            trace_ids.append(ttnn.begin_trace_capture(mesh_device, cq_id=0))
            outputs.append(forward(model))
            ttnn.end_trace_capture(mesh_device, trace_ids[-1], cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for request, chunk in _schedule(requests, chunk_size):
            stage(request, chunk)
            for trace_id in trace_ids:
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            real = min(chunk_size, requests[request][1] - chunk * chunk_size)
            hidden_unbounded, hidden_bounded = (
                _cp_gather_torch(output, mesh_config)[..., :real, :] for output in outputs
            )
            label = f"request {request} chunk {chunk} ({real} real tokens)"
            assert torch.isfinite(hidden_unbounded).all(), f"{label}: unbounded hidden states are not finite"
            assert torch.equal(hidden_bounded, hidden_unbounded), (
                f"{label}: bounded hidden states differ, max abs diff "
                f"{(hidden_bounded - hidden_unbounded).abs().max().item()}"
            )
    finally:
        for trace_id in trace_ids:
            ttnn.release_trace(mesh_device, trace_id)

    # Every slot's last request has written its last chunks. Each bounded slab holds chunk g at local rows
    # (g % 2) * L; the unbounded cache holds it at g * L.
    cp = mesh_config.cp_degree
    local = chunk_size // cp
    last_request = {slot: request for request, (slot, _) in enumerate(requests)}
    for layer_idx in CHECKED_LAYERS:
        unbounded_cache, bounded_cache = (model.tt_kv_cache[layer_idx] for model in models)
        sliding = hasattr(bounded_cache, "k")
        pairs = (
            ((unbounded_cache.k, bounded_cache.k), (unbounded_cache.v, bounded_cache.v))
            if sliding
            else ((unbounded_cache.kv, bounded_cache.kv),)
        )
        for slot, request in last_request.items():
            num_chunks = -(-requests[request][1] // chunk_size)
            for chunk in range(max(0, num_chunks - 2), num_chunks):
                bounded_start = (chunk % 2) * local if sliding else chunk * local
                for unbounded_tensor, bounded_tensor in pairs:
                    expected = _device_rows(unbounded_tensor, slot, chunk * local, (chunk + 1) * local)
                    actual = _device_rows(bounded_tensor, slot, bounded_start, bounded_start + local)
                    for device, (want, got) in enumerate(zip(expected, actual)):
                        assert torch.equal(
                            got, want
                        ), f"layer {layer_idx} slot {slot} chunk {chunk} device {device}: bounded cache rows differ"
