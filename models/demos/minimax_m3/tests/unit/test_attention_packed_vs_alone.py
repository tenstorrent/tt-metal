# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed forward through the ATTENTION LAYER, SP=8 x TP=4, dense + sparse: one forward of 4 x 1024-token chunk
chunks must match running each chunk alone, in order, against an identical cache.

Each composition first fills the slots' prefixes (single-chunk calls, identically in both caches), then runs the
packed forward in one cache and the chunks one by one in the other. The chunks' output rows and the K / V /
index_k caches must agree. Compositions: mixed depths, consecutive chunks of one slot, a partial last chunk,
interleaved partial chunks, an empty tail (its slot's bytes stay unchanged) and two consecutive mid-slab chunks
of one slot. Random weights, the runtime's whole-cache indexed RoPE.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.attention import allocate_kv_caches
from models.demos.minimax_m3.tt.prefill_chunk import PrefillChunk, validate_chunks

from ..test_factory import M3_HIDDEN, packed_attention_forward, parametrize_mesh_with_fabric, random_sp_attention

C, F, USERS = 1024, 4, 4
CAP = 12 * C
PCC = 0.998


def prefix_fills(start, end):
    """Single-chunk prefix fills covering [start, end) in chunk-sized steps."""
    return [(a, min(a + C, end)) for a in range(start, end, C)]


# name -> (prefix fills per slot, the forward's chunks)
CASES = {
    "mixed_depths": (
        {1: prefix_fills(0, 2 * C), 2: prefix_fills(0, 8 * C), 3: prefix_fills(0, 4 * C)},
        [
            PrefillChunk(0, 0, C),
            PrefillChunk(1, 2 * C, 3 * C),
            PrefillChunk(2, 8 * C, 9 * C),
            PrefillChunk(3, 4 * C, 5 * C),
        ],
    ),
    "same_slot_run": (
        {1: prefix_fills(0, C)},
        [PrefillChunk(0, 0, C), PrefillChunk(0, C, 2 * C), PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, C, 2 * C)],
    ),
    "partial_last": (
        {0: prefix_fills(0, 3 * C), 2: prefix_fills(0, 2 * C)},
        [
            PrefillChunk(1, 0, C),
            PrefillChunk(2, 2 * C, 3 * C),
            PrefillChunk(0, 3 * C, 4 * C),
            PrefillChunk(0, 4 * C, 4 * C + 300),
        ],
    ),
    "interleaved_partials": (
        {0: prefix_fills(0, C)},
        [PrefillChunk(0, C, C + 700), PrefillChunk(1, 0, C), PrefillChunk(1, C, C + 333), PrefillChunk(2, 0, 300)],
    ),
    "empty_tail": (
        {0: prefix_fills(0, 2 * C), 3: prefix_fills(0, 2 * C)},
        [PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, 0, 600), PrefillChunk(3, 0, 0), PrefillChunk(3, 0, 0)],
    ),
    "mid_slab": (
        {0: prefix_fills(0, 2 * C + 96), 1: prefix_fills(0, C + 512)},
        [
            PrefillChunk(0, 2 * C + 96, 3 * C + 96),
            PrefillChunk(0, 3 * C + 96, 4 * C + 96),
            PrefillChunk(1, C + 512, 2 * C + 512),
            PrefillChunk(2, 0, C),
        ],
    ),
}


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("layer_kind", ["dense", "sparse"])
def test_attention_packed_vs_alone(mesh_device, device_params, layer_kind, reset_seeds):
    assert tuple(mesh_device.shape) == (8, 4)
    is_sparse = layer_kind == "sparse"

    layer = random_sp_attention(mesh_device, chunk_size=C, max_seq_len=CAP, is_sparse=is_sparse, global_layer_idx=0)

    def read_cache(kvc):
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
        out = {}
        for name in ("k", "v", "index_k"):
            t = getattr(kvc, name)
            sl = ttnn.slice(t, [0, 0, 0, 0], list(t.shape), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out[name] = ttnn.to_torch(sl, mesh_composer=composer).float()
            ttnn.deallocate(sl)
        return out

    gen = torch.Generator().manual_seed(1)
    rand_rows = lambda: torch.randn(C, M3_HIDDEN, generator=gen) * 0.1
    failures, worst = [], 1.0
    for name, (prefixes, chunks) in CASES.items():
        validate_chunks(chunks, chunk_size=C, forward_chunks=F, num_users=USERS, max_seq_len=CAP)
        packed_kv = allocate_kv_caches(mesh_device, num_layers=1, max_seq_len=CAP, num_users=USERS)
        alone_kv = allocate_kv_caches(mesh_device, num_layers=1, max_seq_len=CAP, num_users=USERS)
        for slot, fills in prefixes.items():
            for a, b in fills:
                x = rand_rows()
                for kvc in (packed_kv, alone_kv):
                    packed_attention_forward(layer, [x], [PrefillChunk(slot, a, b)], kvc)
        ttnn.synchronize_device(mesh_device)
        before = read_cache(packed_kv)

        xs = [rand_rows() for _ in chunks]
        packed = packed_attention_forward(layer, xs, chunks, packed_kv)
        alone = [
            None if e.is_empty else packed_attention_forward(layer, [x], [e], alone_kv)[0] for x, e in zip(xs, chunks)
        ]
        ttnn.synchronize_device(mesh_device)
        after, ref = read_cache(packed_kv), read_cache(alone_kv)

        for k, e in enumerate(chunks):
            if e.is_empty:
                ok = bool(torch.all(packed[k] == 0))
                logger.info(f"[{layer_kind}/{name}] chunk {k} empty: zero rows {ok}")
                if not ok:
                    failures.append(f"{name} chunk {k}: empty chunk rows are not zero")
                continue
            _, pcc = comp_pcc(alone[k], packed[k], PCC)
            worst = min(worst, pcc)
            logger.info(f"[{layer_kind}/{name}] chunk {k} {e}: out pcc={pcc:.6f}")
            if pcc < PCC:
                failures.append(f"{name} chunk {k}: out pcc {pcc:.6f}")
        names = ("k", "v", "index_k") if is_sparse else ("k", "v")
        for cache in names:
            _, pcc = comp_pcc(ref[cache], after[cache], PCC)
            diff = (ref[cache] - after[cache]).abs().max().item()
            worst = min(worst, pcc)
            mean = ref[cache].abs().mean().item()
            logger.info(f"[{layer_kind}/{name}] cache {cache}: pcc={pcc:.6f} max|diff|={diff:.4g} mean|ref|={mean:.4g}")
            if pcc < PCC:
                failures.append(f"{name} cache {cache}: pcc {pcc:.6f}")
        for slot in {e.slot_id for e in chunks if e.is_empty} - {e.slot_id for e in chunks if not e.is_empty}:
            for cache in names:
                if not torch.equal(before[cache][slot], after[cache][slot]):
                    failures.append(f"{name}: empty chunk wrote slot {slot} ({cache})")
        packed_kv.deallocate()
        alone_kv.deallocate()

    logger.info(f"[{layer_kind}] worst pcc={worst:.6f}")
    assert not failures, f"{layer_kind} packed vs alone: {failures}"
