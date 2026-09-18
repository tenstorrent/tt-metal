# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What does one attention call cost on the ring, and does the prefix still show up in the bill?

The gather route had to move the whole active prefix onto one chip per layer, so its per-call cost
grew with the prefix even after the transfer was scoped to the populated chunks. The ring route keeps
the prefix sequence-parallel and never assembles it, so the only prefix-dependent work left is the
SDPA arithmetic itself. This measures both routes through one attention instance, at prefixes from
1024 to the cache capacity, and reports them side by side.

Q/K chunk sizes are swept as well: they set how the op tiles the SDPA loop, and the right pair is not
obvious for a 256-row local Q against a prefix that grows by three orders of magnitude.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.attention import FullCausalAttention
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache, write_kv_chunk

MESH_SHAPE = (4, 8)
SP, TP = MESH_SHAPE
GLOBAL_CHUNK = 1024
LOCAL_SEQUENCE = GLOBAL_CHUNK // SP
HEAD_DIM = Llama31_8BConfig.HEAD_DIM
NUM_KV_HEADS = Llama31_8BConfig.NUM_KEY_VALUE_HEADS
NUM_Q_HEADS = Llama31_8BConfig.NUM_ATTENTION_HEADS
LOCAL_Q_HEADS = NUM_Q_HEADS // TP
CAPACITY = 8192
LAYER = 0
SLOT = 0
REPS = 10


def _kv_chunk(mesh_device):
    # The writer stages in bfloat16 and converts into the cache dtype itself.
    payload = torch.randn(NUM_KV_HEADS, GLOBAL_CHUNK, HEAD_DIM) * 0.1
    return ttnn.from_torch(
        payload.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _local_q(mesh_device):
    values = torch.randn(NUM_Q_HEADS, GLOBAL_CHUNK, HEAD_DIM) * 0.1
    return ttnn.from_torch(
        values.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _time(call, mesh_device):
    for _ in range(3):  # warm the program cache for this shape
        call().deallocate(True)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(REPS):
        call().deallocate(True)
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - start) / REPS * 1e3


def _gather_route(attention, cache, q, *, end):
    """The retained reference route, timed the way production used to run it."""
    logical_n = end
    batch_index = SLOT * Llama31_8BConfig.NUM_LAYERS + LAYER
    natural_k = attention._gather_and_reorder(
        cache.k, attention.gathered_k, batch_index=batch_index, logical_n=logical_n
    )
    natural_v = attention._gather_and_reorder(
        cache.v, attention.gathered_v, batch_index=batch_index, logical_n=logical_n
    )
    mask, query_valid = attention._build_mask(actual_start=end - GLOBAL_CHUNK, actual_end=end, logical_n=logical_n)
    output = ttnn.transformer.scaled_dot_product_attention(
        q,
        natural_k,
        natural_v,
        attn_mask=mask,
        is_causal=False,
        scale=HEAD_DIM**-0.5,
        program_config=attention.program_config,
        compute_kernel_config=attention.compute_kernel_config,
    )
    masked = ttnn.multiply(output, query_valid)
    for tensor in (natural_k, natural_v, mask, query_valid, output):
        tensor.deallocate(True)
    return masked


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
def test_ring_vs_gather_attention_cost(mesh_device):
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    cache_dtype = ttnn.bfloat8_b
    attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype, max_seq_len=CAPACITY)
    cache = allocate_kv_cache(mesh_device, mesh_config, max_seq_len=CAPACITY, cache_dtype=cache_dtype)

    for index in range(CAPACITY // GLOBAL_CHUNK):
        tt_k, tt_v = _kv_chunk(mesh_device), _kv_chunk(mesh_device)
        write_kv_chunk(
            cache,
            tt_k,
            tt_v,
            slot_idx=SLOT,
            layer_idx=LAYER,
            actual_start=index * GLOBAL_CHUNK,
            actual_end=(index + 1) * GLOBAL_CHUNK,
        )
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    q = _local_q(mesh_device)
    ttnn.synchronize_device(mesh_device)

    logger.info(f"{'prefix':>8} {'ring ms':>9} {'gather ms':>10} {'speedup':>8}")
    for end in (1024, 2048, 4096, 8192):
        start = end - GLOBAL_CHUNK
        ring_ms = _time(
            lambda: attention(q, cache, slot_idx=SLOT, layer_idx=LAYER, actual_start=start, actual_end=end),
            mesh_device,
        )
        gather_ms = _time(lambda: _gather_route(attention, cache, q, end=end), mesh_device)
        logger.info(f"{end:>8} {ring_ms:>9.3f} {gather_ms:>10.3f} {gather_ms / ring_ms:>7.2f}x")

    # A 32-layer chunk pays this 32 times, so the shape of the curve matters more than any single
    # point: report what each route charges for the last 7168 tokens of prefix.
    logger.info("chunk sizes at the full prefix (q_chunk, k_chunk -> ms):")
    base_program_config = attention.ring_program_config
    grid = mesh_device.compute_with_storage_grid_size()
    for q_chunk in (32, 128, 256):
        for k_chunk in (128, 256, 512):
            attention.ring_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
            try:
                ms = _time(
                    lambda: attention(
                        q,
                        cache,
                        slot_idx=SLOT,
                        layer_idx=LAYER,
                        actual_start=CAPACITY - GLOBAL_CHUNK,
                        actual_end=CAPACITY,
                    ),
                    mesh_device,
                )
            except Exception as error:  # an unsupported pair should not end the sweep
                logger.info(f"  q{q_chunk:<4} k{k_chunk:<4} unsupported: {type(error).__name__}")
                continue
            logger.info(f"  q{q_chunk:<4} k{k_chunk:<4} {ms:>8.3f} ms")
    attention.ring_program_config = base_program_config

    q.deallocate(True)
    cache.k.deallocate(True)
    cache.v.deallocate(True)
