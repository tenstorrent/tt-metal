# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed forwards through TtPrefillRuntime.prefill_forward: 4 x 1024-token chunks per forward must fill the
KV cache like the same chunks prefilled one per forward (forward_tokens_budget == chunk_size, today's path).

One schedule of seven forwards over four slots covers consecutive chunks of one slot, mixed depths, interleaved
partial chunks, an empty tail and mid-slab resumes (two consecutive mid-slab chunks of one slot), with 4, 3, 2 and 1
real chunks per forward: a 2-chunk forward whose ragged mid-slab chunk precedes a full one, and a 3-chunk forward
of mid-slab chunks. The same chunks then run one per forward through the same runtime (empty tail) and through a
runtime built with forward_tokens_budget == chunk_size, each into its own cache. Every slot is compared per layer
(K, V and, on the MSA layers, index_k) in natural order. After compile(), neither pass of the packed runtime may add
a program-cache entry.

A forward runs its layers on its non-empty chunks only (widths 4, 3, 2 and 1 in this schedule), so the
one-chunk-per-forward pass (one real chunk + three empties, ragged and mid-slab chunks included) runs at one
chunk's width: it must reproduce the forward_tokens_budget == chunk_size runtime (the check also logs whether it is
bit-identical).

test_prefill_runtime_packed_two_ranks runs the same schedule through two pipeline ranks on one mesh (layers 0-2 on a
first, non-last rank, layers 3-4 on a non-first, last rank): the first rank's output keeps the full forward width
with zero rows for the empty chunks, no program compiles after compile(), and the KV matches one chunk per forward.

Real weights, first 5 layers (3 dense + 2 MSA/MoE), from the warm tilized weight cache (HF_MODEL / TT_CACHE_PATH);
skipped when that cache is not available.
"""

import copy
import gc
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.prefill_chunk import PrefillChunk

from ..test_factory import parametrize_mesh_with_fabric

NUM_LAYERS = 5
C, W, USERS = 1024, 4096, 4
CAPACITY = 8 * C
# Packed vs one chunk per forward: the forward width differs (a forward runs at its real chunks' rows), and the auto
# matmul configs follow the width (M), so the sums run in a different order.
PCC_ACROSS_WIDTHS = 0.997
# A one-chunk forward runs at one chunk's width: the same programs as a runtime built for one chunk per forward.
PCC_SAME_WORK = 0.99999
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
    # 2 real chunks: a ragged mid-slab chunk, then a full one.
    [PrefillChunk(3, C + 288, C + 688), PrefillChunk(2, 5 * C, 6 * C), EMPTY, EMPTY],
    # 3 real chunks, all mid-slab: two consecutive chunks of one slot (the second partial), then another slot.
    [
        PrefillChunk(1, 3 * C + 288, 4 * C + 288),
        PrefillChunk(1, 4 * C + 288, 4 * C + 888),
        PrefillChunk(0, 4 * C + 480, 5 * C + 480),
        EMPTY,
    ],
]


def _setup(mesh_device):
    """Model config, weight cache path and per-slot tokens; skips without weights or a complete cache."""
    if not os.getenv("HF_MODEL"):
        pytest.skip("needs real MiniMax-M3 weights (HF_MODEL + a warm TT_CACHE_PATH tilized cache)")
    from models.demos.minimax_m3.tt.model_config import ModelArgs
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
    return hf_config, cache_path, tokens


def _schedule_ends():
    ends = {}
    for forward in SCHEDULE:
        for e in forward:
            if not e.is_empty:
                ends[e.slot_id] = e.actual_end
    return ends


def _one_per_forward():
    """The schedule's chunks one per forward, the rest of each forward empty."""
    return [[e] + [EMPTY] * (W // C - 1) for forward in SCHEDULE for e in forward if not e.is_empty]


def _chunk_tokens(tokens, e):
    if e.is_empty:
        return [0] * C
    return tokens[e.slot_id][e.actual_start : e.actual_end] + [0] * (C - (e.actual_end - e.actual_start))


def _worst_pcc(ref_all, got_all, layers, tag):
    worst = 1.0
    for slot in sorted(ref_all):
        for layer in layers:
            names = ("k", "v", "index_k") if layer >= 3 else ("k", "v")
            for name, ref, got in zip(names, ref_all[slot][layer], got_all[slot][layer]):
                _, p = comp_pcc(ref.float(), got.float(), PCC_ACROSS_WIDTHS)
                logger.info(f"[packed vs {tag}] slot {slot} layer {layer} {name}: pcc={p:.5f}")
                worst = min(worst, p)
    return worst


@pytest.mark.timeout(2400)  # two 5-layer real-weight builds + compiles + per-slot read-backs
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
def test_prefill_runtime_packed(mesh_device, device_params, reset_seeds):
    hf_config, cache_path, tokens = _setup(mesh_device)
    from models.demos.minimax_m3.tt.attention.kv_cache import allocate_kv_caches
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    ends = _schedule_ends()

    def chunk_tokens(e):
        return _chunk_tokens(tokens, e)

    def new_kv_cache():
        return allocate_kv_caches(
            mesh_device, num_layers=NUM_LAYERS, max_seq_len=CAPACITY, num_users=USERS, head_dim=hf_config.head_dim
        )

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
        kv_cache = new_kv_cache()
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

    new_programs = {}

    def counted(tag, forwards, runtime, kv_cache):
        """Run forwards (no cache read-back) and record the program-cache entries they add."""
        ttnn.synchronize_device(mesh_device)
        before = mesh_device.num_program_cache_entries()
        for forward in forwards:
            inp = runtime.make_forward_input([chunk_tokens(e) for e in forward], [e.actual_start for e in forward])
            runtime.prefill_forward(inp, kv_cache, forward)
        ttnn.synchronize_device(mesh_device)
        new_programs[tag] = mesh_device.num_program_cache_entries() - before

    runtime, kv_cache = build(W)
    counted("packed", SCHEDULE, runtime, kv_cache)
    packed = natural(runtime, kv_cache)
    kv_cache.deallocate()
    # One chunk per forward (empty tail): each forward runs at one chunk's width.
    kv_cache = new_kv_cache()
    counted("one chunk per forward", _one_per_forward(), runtime, kv_cache)
    one_per_forward = natural(runtime, kv_cache)
    kv_cache.deallocate()
    runtime.release_sub_device_managers()
    del runtime, kv_cache
    gc.collect()

    # A runtime built for one chunk per forward. It shares the mesh program cache with the runtime above, so its
    # program count would prove nothing and is not checked.
    runtime, kv_cache = build(C)
    for forward in SCHEDULE:
        for e in forward:
            if not e.is_empty:
                inp = runtime.make_chunk_input(chunk_tokens(e), e.actual_start)
                runtime.prefill_chunk(
                    inp, kv_cache, slot_id=e.slot_id, actual_start=e.actual_start, actual_end=e.actual_end
                )
    one_chunk_build = natural(runtime, kv_cache)
    runtime.release_sub_device_managers()

    layers = range(NUM_LAYERS)
    worst_vs_one_per_forward = _worst_pcc(one_per_forward, packed, layers, "one chunk per forward")
    worst_vs_one_chunk_build = _worst_pcc(one_chunk_build, packed, layers, "forward_tokens_budget == chunk_size")
    logger.info(
        f"[packed] mesh={tuple(mesh_device.shape)}: worst pcc vs one chunk per forward "
        f"{worst_vs_one_per_forward:.5f}, vs forward_tokens_budget == chunk_size {worst_vs_one_chunk_build:.5f}; "
        f"programs added after compile(), per pass: {new_programs}"
    )
    # A forward of one real chunk runs at one chunk's width: the same work as a runtime built for one chunk.
    worst_one_chunk, identical = 1.0, True
    for slot in sorted(ends):
        for layer in layers:
            for ref, got in zip(one_chunk_build[slot][layer], one_per_forward[slot][layer]):
                identical = identical and torch.equal(ref, got)
                worst_one_chunk = min(worst_one_chunk, comp_pcc(ref.float(), got.float(), PCC_SAME_WORK)[1])
    logger.info(
        f"[packed] one-chunk forwards vs forward_tokens_budget == chunk_size: worst pcc {worst_one_chunk:.6f}, "
        f"bit-identical {identical}"
    )
    assert (
        worst_one_chunk >= PCC_SAME_WORK
    ), f"a one-chunk forward diverges from a runtime built for one chunk ({worst_one_chunk:.6f})"
    assert (
        worst_vs_one_per_forward >= PCC_ACROSS_WIDTHS
    ), f"packing changes a slot's KV (worst pcc={worst_vs_one_per_forward:.5f})"
    assert (
        worst_vs_one_chunk_build >= PCC_ACROSS_WIDTHS
    ), f"packed forwards diverge from a runtime built for one chunk (worst pcc={worst_vs_one_chunk_build:.5f})"
    assert not any(new_programs.values()), f"passes compiled programs after compile(): {new_programs}"


@pytest.mark.timeout(2400)  # two 2-rank real-weight builds + compiles + per-slot read-backs
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
def test_prefill_runtime_packed_two_ranks(mesh_device, device_params, reset_seeds):
    """Two pipeline ranks on one mesh: layers 0-2 on a first, non-last rank (embeds the non-empty chunks' tokens,
    pads its output back to the full width), layers 3-4 on a non-first, last rank (slices the received activation).
    The activation goes from one runtime to the other directly, without the D2D socket."""
    hf_config, cache_path, tokens = _setup(mesh_device)
    from models.demos.minimax_m3.tt.attention.kv_cache import allocate_kv_caches
    from models.demos.minimax_m3.tt.residual import use_sharded_residual
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    ranks = [(0, 3, True, False), (3, 2, False, True)]  # (first_layer_idx, num_layers, is_first_rank, is_last_rank)
    ends = _schedule_ends()
    sp = mesh_device.shape[0]
    rows_per_chip, chunk_rows = W // sp, C // sp
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))

    def new_kv_caches():
        return [
            allocate_kv_caches(
                mesh_device, num_layers=n, max_seq_len=CAPACITY, num_users=USERS, head_dim=hf_config.head_dim
            )
            for _, n, _, _ in ranks
        ]

    runtimes = []
    for first, n, is_first, is_last in ranks:
        cfg = TtPrefillRuntimeConfig(
            num_layers=n,
            max_seq_len=CAPACITY,
            mesh_shape=tuple(mesh_device.shape),
            chunk_size=C,
            forward_tokens_budget=W,
            num_users=USERS,
            expert_weight_dtype=ttnn.bfloat4_b,
            weight_cache_path=cache_path,
            topology=ttnn.Topology.Linear,
            first_layer_idx=first,
            is_first_rank=is_first,
            is_last_rank=is_last,
            # The activation layout between ranks, as the adapter sets it (see the runner's D2D mapper).
            pipeline_activation_emb_tp_sharded=use_sharded_residual(),
        )
        runtimes.append(TtPrefillRuntime(mesh_device, copy.deepcopy(hf_config), {}, cfg))
    for runtime, kv_cache in zip(runtimes, new_kv_caches()):
        runtime.compile(kv_cache)
        kv_cache.deallocate()

    def run(forwards):
        """Forwards through both ranks; returns per-slot natural KV by global layer, the programs added, and per
        forward whether the first rank's output rows of the empty chunks are all zero."""
        kv_caches = new_kv_caches()
        pad_rows_zero = []
        ttnn.synchronize_device(mesh_device)
        before = mesh_device.num_program_cache_entries()
        for forward in forwards:
            inp = runtimes[0].make_forward_input(
                [_chunk_tokens(tokens, e) for e in forward], [e.actual_start for e in forward]
            )
            out = runtimes[0].prefill_forward(inp, kv_caches[0], forward)
            assert out.shape[2] == rows_per_chip, f"first rank output has {out.shape[2]} rows, not {rows_per_chip}"
            n_real = sum(not e.is_empty for e in forward)
            host = ttnn.to_torch(out, mesh_composer=composer)
            pads = [host[:, :, r * rows_per_chip + n_real * chunk_rows : (r + 1) * rows_per_chip] for r in range(sp)]
            pad_rows_zero.append(all(bool((p == 0).all()) for p in pads))
            assert runtimes[1].prefill_forward(out, kv_caches[1], forward) is None
        ttnn.synchronize_device(mesh_device)
        programs = mesh_device.num_program_cache_entries() - before
        kv = {slot: {} for slot in ends}
        for (first, n, _, _), runtime, kv_cache in zip(ranks, runtimes, kv_caches):
            for slot, end in ends.items():
                blocks = runtime.read_slot_kv(kv_cache, slot)
                for layer in range(n):
                    kv[slot][first + layer] = [
                        naturalize_kv_block(b[layer], end, runtime.config.sp_factor, C, CAPACITY) for b in blocks
                    ]
            kv_cache.deallocate()
        return kv, programs, pad_rows_zero

    packed, packed_programs, packed_pad_zero = run(SCHEDULE)
    one_per_forward, one_programs, one_pad_zero = run(_one_per_forward())
    for runtime in runtimes:
        runtime.release_sub_device_managers()

    worst = _worst_pcc(one_per_forward, packed, range(NUM_LAYERS), "one chunk per forward, two ranks")
    logger.info(
        f"[packed two ranks] mesh={tuple(mesh_device.shape)}: worst pcc vs one chunk per forward {worst:.5f}; "
        f"programs added after compile(): packed {packed_programs}, one chunk per forward {one_programs}; "
        f"empty-chunk rows zero: packed {packed_pad_zero}, one chunk per forward {one_pad_zero}"
    )
    assert all(packed_pad_zero) and all(one_pad_zero), "the first rank's rows of the empty chunks are not zero"
    assert packed_programs == 0 and one_programs == 0, "programs compiled after compile()"
    assert worst >= PCC_ACROSS_WIDTHS, f"two-rank packed forwards change a slot's KV (worst pcc={worst:.5f})"
