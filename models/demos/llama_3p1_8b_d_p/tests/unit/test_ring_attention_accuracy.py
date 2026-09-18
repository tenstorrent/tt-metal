# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""How close can the ring path get to the FP32 gather route, and does any knob close the gap?

The cache-read ring requires the streaming compute path (``kv_actual_isl`` asserts
``fp32_dest_acc_en=false`` in ring_joint_sdpa_program_factory.cpp:1341), so its running max and sum
live in BF16 destination registers while the gather route accumulates in FP32. That costs fidelity,
and the question is how much and whether the remaining knobs -- packer L1 accumulation, the SDPA
chunk shape, math fidelity -- buy any of it back.

Both routes run on the same cache through the same attention instance, and the FP32 gather route is
the reference: it is the numerics production shipped before the ring, so a delta here is exactly what
the ring costs. Metrics match the suite's gates (PCC and normalized L2 over valid rows only).
"""

import itertools

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
NUM_LAYERS = Llama31_8BConfig.NUM_LAYERS
CAPACITY = 2048
SLOT = 0
LAYER = 0


def _owned_positions(start):
    """Absolute query positions each SP rank owns for the chunk beginning at ``start``."""
    owned = [[] for _ in range(SP)]
    for position in range(start, start + GLOBAL_CHUNK):
        owned[(position % GLOBAL_CHUNK) // LOCAL_SEQUENCE].append(position)
    return owned


def _shard(mesh_device, values, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        values.unsqueeze(0).to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(2, 1)),
    )


def _metrics(expected, actual):
    expected, actual = expected.flatten().double(), actual.flatten().double()
    centered_e, centered_a = expected - expected.mean(), actual - actual.mean()
    pcc = float(
        torch.dot(centered_e, centered_a)
        / (torch.linalg.vector_norm(centered_e) * torch.linalg.vector_norm(centered_a))
    )
    nl2 = float(torch.linalg.vector_norm(expected - actual) / torch.linalg.vector_norm(expected))
    return pcc, nl2


def _gather_reference(attention, cache, q, *, actual_start, actual_end):
    """The branch's FP32 production numerics, used here as the reference."""
    logical_n = ((actual_end + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    batch_index = SLOT * NUM_LAYERS + LAYER
    natural_k = attention._gather_and_reorder(
        cache.k, attention.gathered_k, batch_index=batch_index, logical_n=logical_n
    )
    natural_v = attention._gather_and_reorder(
        cache.v, attention.gathered_v, batch_index=batch_index, logical_n=logical_n
    )
    mask, query_valid = attention._build_mask(actual_start=actual_start, actual_end=actual_end, logical_n=logical_n)
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


def _compare(mesh_device, reference, candidate, *, actual_start, actual_end):
    """Worst PCC / NL2 across chips, over valid query rows only."""
    owned = _owned_positions(actual_start)
    reference_shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(reference)]
    candidate_shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(candidate)]
    worst_pcc, worst_nl2 = 1.0, 0.0
    for sp_coord in range(SP):
        rows = [row for row, position in enumerate(owned[sp_coord]) if position < actual_end]
        if not rows:
            continue
        for tp_coord in range(TP):
            index = sp_coord * TP + tp_coord
            expected = reference_shards[index][:, :LOCAL_Q_HEADS, rows, :HEAD_DIM]
            actual = candidate_shards[index][:, :LOCAL_Q_HEADS, rows, :HEAD_DIM]
            pcc, nl2 = _metrics(expected, actual)
            worst_pcc, worst_nl2 = min(worst_pcc, pcc), max(worst_nl2, nl2)
    return worst_pcc, worst_nl2


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH_SHAPE, id="galaxy-4x8")], indirect=True)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
def test_ring_accuracy_against_fp32_gather(mesh_device, cache_dtype):
    torch.manual_seed(0)
    mesh_config = MeshConfig(MESH_SHAPE, TP)
    attention = FullCausalAttention(mesh_device, mesh_config, cache_dtype=cache_dtype, max_seq_len=CAPACITY)
    cache = allocate_kv_cache(mesh_device, mesh_config, max_seq_len=CAPACITY, cache_dtype=cache_dtype)

    # Keep the payload zero-mean and wide: a near-constant prefix makes the output near-constant too,
    # and PCC centers its inputs, so it reports noise rather than agreement on such a fixture.
    ranges = ((0, 1024), (1024, 1537))
    for actual_start, actual_end in ranges:
        payload = torch.randn(NUM_KV_HEADS, GLOBAL_CHUNK, HEAD_DIM) * 0.5
        tt_k, tt_v = _shard(mesh_device, payload), _shard(mesh_device, payload)
        write_kv_chunk(
            cache, tt_k, tt_v, slot_idx=SLOT, layer_idx=LAYER, actual_start=actual_start, actual_end=actual_end
        )
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    grid = mesh_device.compute_with_storage_grid_size()
    base_program, base_compute = attention.ring_program_config, attention.ring_compute_kernel_config
    logger.info(f"[{cache_dtype}] knob sweep against the FP32 gather route (worst chip, valid rows only)")
    logger.info(f"{'range':>12} {'q':>4} {'k':>4} {'packer':>7} {'fidelity':>9} {'PCC':>10} {'NL2':>9}")

    for actual_start, actual_end in ranges:
        q = _shard(mesh_device, torch.randn(NUM_Q_HEADS, GLOBAL_CHUNK, HEAD_DIM) * 0.1)
        reference = _gather_reference(attention, cache, q, actual_start=actual_start, actual_end=actual_end)
        ttnn.synchronize_device(mesh_device)
        for q_chunk, k_chunk, packer, fidelity in itertools.product(
            (128, 256), (128, 512), (False, True), (ttnn.MathFidelity.HiFi4,)
        ):
            attention.ring_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
            attention.ring_compute_kernel_config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=False,  # the cache-read ring rejects FP32 accumulation
                packer_l1_acc=packer,
            )
            try:
                candidate = attention(
                    q, cache, slot_idx=SLOT, layer_idx=LAYER, actual_start=actual_start, actual_end=actual_end
                )
                ttnn.synchronize_device(mesh_device)
            except Exception as error:
                logger.info(f"  q{q_chunk} k{k_chunk} packer={packer}: unsupported {type(error).__name__}: {error}")
                continue
            pcc, nl2 = _compare(mesh_device, reference, candidate, actual_start=actual_start, actual_end=actual_end)
            candidate.deallocate(True)
            logger.info(
                f"{f'[{actual_start},{actual_end})':>12} {q_chunk:>4} {k_chunk:>4} {str(packer):>7} "
                f"{str(fidelity).split('.')[-1]:>9} {pcc:>10.7f} {nl2:>9.5f}"
            )
        reference.deallocate(True)
        q.deallocate(True)

    attention.ring_program_config, attention.ring_compute_kernel_config = base_program, base_compute
    cache.k.deallocate(True)
    cache.v.deallocate(True)
