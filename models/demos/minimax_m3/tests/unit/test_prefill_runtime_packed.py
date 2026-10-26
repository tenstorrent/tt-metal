# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed forwards through TtPrefillRuntime.prefill_forward: 4 x 1024-token chunks per forward must fill the
KV cache like the same chunks prefilled one per forward (forward_tokens_budget == chunk_size, today's path).

One schedule of five forwards over four slots covers consecutive chunks of one slot, mixed depths, interleaved
partial chunks, an empty tail and mid-slab resumes (two consecutive mid-slab chunks of one slot). The same
chunks then run one per forward through the same runtime (empty tail) and through a runtime built with
forward_tokens_budget == chunk_size, each into its own cache. Every slot is compared per layer (K, V and, on the MSA
layers, index_k) in natural order. After compile() the packed schedule must add no program-cache entries.

Real weights, first 5 layers (3 dense + 2 MSA/MoE), from the warm tilized weight cache (HF_MODEL / TT_CACHE_PATH);
skipped when that cache is not available.
"""

import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.prefill_chunk import PrefillChunk

from ..test_factory import parametrize_mesh_with_fabric

NUM_LAYERS = 5
C, W, USERS = 1024, 4096, 4
CAPACITY = 8 * C
PCC = 0.998  # packing only: same forward width, one chunk per forward
PCC_CHUNK_WIDTH = 0.997  # vs forward_tokens_budget == chunk_size: the auto matmul configs follow the forward width (M)
EMPTY = PrefillChunk(0, 0, 0)
SCHEDULE = [
    [PrefillChunk(0, 0, C), PrefillChunk(0, C, 2 * C), PrefillChunk(1, 0, C), PrefillChunk(2, 0, C)],
    [PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, C, C + 700), PrefillChunk(2, C, 2 * C), PrefillChunk(3, 0, 300)],
    [PrefillChunk(2, 2 * C, 3 * C), PrefillChunk(2, 3 * C, 4 * C), PrefillChunk(0, 3 * C, 3 * C + 500), EMPTY],
    [
        PrefillChunk(1, C + 672, 2 * C + 672),
        PrefillChunk(1, 2 * C + 672, 3 * C + 300),
        PrefillChunk(3, 288, C + 288),
        PrefillChunk(2, 4 * C, 5 * C),
    ],
    [PrefillChunk(0, 3 * C + 480, 4 * C + 480), EMPTY, EMPTY, EMPTY],
]


@pytest.mark.timeout(2400)  # two 5-layer real-weight builds + compiles + per-slot read-backs
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
def test_prefill_runtime_packed(mesh_device, device_params, reset_seeds):
    if not os.getenv("HF_MODEL"):
        pytest.skip("needs real MiniMax-M3 weights (HF_MODEL + a warm TT_CACHE_PATH tilized cache)")
    from models.demos.minimax_m3.tt.attention.kv_cache import allocate_kv_caches
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    model_args = ModelArgs(mesh_device=mesh_device)
    hf_config = model_args.hf_config
    hf_config.num_hidden_layers = NUM_LAYERS
    os.environ.setdefault("M3_LOAD_NLAYERS", str(NUM_LAYERS))
    cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
    if not weight_cache_is_complete(cache_path, hf_config, NUM_LAYERS, ttnn.bfloat4_b):
        pytest.skip(f"tilized weight cache for the first {NUM_LAYERS} layers is not complete at {cache_path}")

    gen = torch.Generator().manual_seed(0)
    tokens = [torch.randint(0, hf_config.vocab_size, (CAPACITY,), generator=gen).tolist() for _ in range(USERS)]
    ends = {}
    for forward in SCHEDULE:
        for e in forward:
            if not e.is_empty:
                ends[e.slot_id] = e.actual_end

    def chunk_tokens(e):
        if e.is_empty:
            return [0] * C
        return tokens[e.slot_id][e.actual_start : e.actual_end] + [0] * (C - (e.actual_end - e.actual_start))

    def build(forward_tokens_budget):
        cfg = TtPrefillRuntimeConfig(
            num_layers=NUM_LAYERS,
            max_seq_len=CAPACITY,
            mesh_shape=tuple(mesh_device.shape),
            chunk_size=C,
            forward_tokens_budget=forward_tokens_budget,
            num_users=USERS,
            expert_weight_dtype=ttnn.bfloat4_b,
            weight_cache_path=cache_path,
            topology=ttnn.Topology.Linear,
        )
        runtime = TtPrefillRuntime(mesh_device, hf_config, {}, cfg)
        kv_cache = allocate_kv_caches(
            mesh_device, num_layers=NUM_LAYERS, max_seq_len=CAPACITY, num_users=USERS, head_dim=hf_config.head_dim
        )
        runtime.compile(kv_cache)
        return runtime, kv_cache

    def natural(runtime, kv_cache):
        out = {}
        for slot, end in ends.items():
            blocks = runtime.read_slot_kv(kv_cache, slot)
            out[slot] = [
                [naturalize_kv_block(b[layer], end, runtime.config.sp_factor, C, CAPACITY) for b in blocks]
                for layer in range(NUM_LAYERS)
            ]
        return out

    runtime, kv_cache = build(W)
    ttnn.synchronize_device(mesh_device)
    programs = mesh_device.num_program_cache_entries()
    for forward in SCHEDULE:
        inp = runtime.make_forward_input([chunk_tokens(e) for e in forward], [e.actual_start for e in forward])
        runtime.prefill_forward(inp, kv_cache, forward)
    ttnn.synchronize_device(mesh_device)
    new_programs = mesh_device.num_program_cache_entries() - programs
    packed = natural(runtime, kv_cache)
    kv_cache.deallocate()
    # Same forward width, one chunk per forward (empty tail): isolates packing from the width's matmul numerics.
    kv_cache = allocate_kv_caches(
        mesh_device, num_layers=NUM_LAYERS, max_seq_len=CAPACITY, num_users=USERS, head_dim=hf_config.head_dim
    )
    for forward in SCHEDULE:
        for e in forward:
            if not e.is_empty:
                solo = [e] + [EMPTY] * (W // C - 1)
                inp = runtime.make_forward_input([chunk_tokens(x) for x in solo], [x.actual_start for x in solo])
                runtime.prefill_forward(inp, kv_cache, solo)
    ttnn.synchronize_device(mesh_device)
    solo_same_width = natural(runtime, kv_cache)
    kv_cache.deallocate()
    runtime.release_sub_device_managers()
    del runtime, kv_cache
    gc.collect()

    runtime, kv_cache = build(C)
    for forward in SCHEDULE:
        for e in forward:
            if not e.is_empty:
                inp = runtime.make_chunk_input(chunk_tokens(e), e.actual_start)
                runtime.prefill_chunk(
                    inp, kv_cache, slot_id=e.slot_id, actual_start=e.actual_start, actual_end=e.actual_end
                )
    ttnn.synchronize_device(mesh_device)
    alone = natural(runtime, kv_cache)
    runtime.release_sub_device_managers()

    def worst_pcc(ref_all, tag):
        worst = 1.0
        for slot in sorted(ends):
            for layer in range(NUM_LAYERS):
                names = ("k", "v", "index_k") if layer >= 3 else ("k", "v")
                for name, ref, got in zip(names, ref_all[slot][layer], packed[slot][layer]):
                    _, p = comp_pcc(ref.float(), got.float(), PCC)
                    logger.info(f"[packed vs {tag}] slot {slot} (end {ends[slot]}) layer {layer} {name}: pcc={p:.5f}")
                    worst = min(worst, p)
        return worst

    worst_same_width = worst_pcc(solo_same_width, "one chunk per forward")
    worst_chunk_width = worst_pcc(alone, "forward_tokens_budget == chunk_size")
    logger.info(
        f"[packed] worst pcc vs one chunk per forward {worst_same_width:.5f}, vs forward_tokens_budget == chunk_size "
        f"{worst_chunk_width:.5f}; programs added after compile(): {new_programs}"
    )
    assert worst_same_width >= PCC, f"packing changes a slot's KV (worst pcc={worst_same_width:.5f})"
    assert (
        worst_chunk_width >= PCC_CHUNK_WIDTH
    ), f"packed forwards diverge from one chunk per forward (worst pcc={worst_chunk_width:.5f})"
    assert new_programs == 0, f"the packed schedule compiled {new_programs} programs after compile()"
