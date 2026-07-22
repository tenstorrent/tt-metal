# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers, resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    blockcyclic_cache_host,
    blockcyclic_positions,
    rotated_chip_positions,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.chunked_prefill_utils import (
    cpu_mla_reference,
    discover_traces,
    load_trace,
    partition_iters,
    single_trace,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_mla_inference(
    config,
    weights,
    mesh_device,
    seq_len,
    mesh_shape,
    sp_axis,
    tp_axis,
    topology,
    tt_kvpe_cache,
    return_indices=False,
    inject_indices=None,
):
    """
    Utility function to run MLA inference without host comparison.

    Args:
        config: Model configuration
        weights: Model weights dictionary
        mesh_device: Mesh device for TT
        seq_len: Sequence length
        mesh_shape: Shape of mesh device
        sp_axis: Sequence parallel axis
        tp_axis: Tensor parallel axis
        topology: Topology (Linear or Ring)
        tt_kvpe_cache: Initialized KVPE cache on device

    Returns:
        Tuple of (tt_output, hidden_states, shard_dims)
    """
    # Create TT MLA
    logger.info("Creating TT MLA...")

    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        topology=topology,
        # Match the single-layer test cache (num_kvpe_cache_layers=1): the single-shot write goes through
        # update_padded_kv_cache, which asserts cache_batch % layer_num == 0.
        layer_num=1,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis)
    # Sparse (DSA) single-shot is folded onto the block-cyclic chunked path (one full-seq chunk at offset
    # 0): it uses the indexed rope tables and a caller-owned indexer key cache, exactly like the chunked
    # path. This helper drives only sparse variants (deepseek_v32 / glm_5_1 / glm_5_2), so the indexer is
    # always present.
    has_indexer = resolve_has_indexer(config)
    assert has_indexer, "run_mla_inference is the sparse single-shot driver; dense prefill is chunked-only"
    rope_tensors = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len, chunk_size_global=seq_len)
    # Layer-slot count mirrors the serving adapter: the indexer strides the folded user-major cache by
    # num_full_indexer_layers (GLM-5.2 cross-layer reuse), so the cache must carry that many slots for
    # update_padded_kv_cache's cache_batch % num_layers check. Falls back to 1 (no indexer_types).
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    # Verify TT MLA exists
    assert mla_tt is not None, "TT MLA should exist"

    # Create test inputs
    batch_size = 1
    hidden_size = config.hidden_size

    logger.info(f"Creating test inputs: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

    # Create random input tensor (generate in float32, then convert to bfloat16)
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)

    tt_input = hidden_states.unsqueeze(0)  # [1, batch, seq, hidden]

    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[sp_axis] = -2
    tt_hidden_states = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    # GLM-5.2 indexer reuse (return_indices / inject_indices): capture this layer's top-k selection, or
    # feed a prior layer's to skip the indexer. Defaults leave the single-shot forward unchanged.
    mla_out = mla_tt.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=rope_tensors,
        kvpe_cache=tt_kvpe_cache,
        indexer_indices=inject_indices,
        return_indexer_indices=return_indices,
        index_kv_cache=index_kv_cache,
    )
    indices = None
    if return_indices:
        tt_output, indices = mla_out
    else:
        tt_output = mla_out

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    if return_indices:
        return tt_output, hidden_states, shard_dims, indices
    return tt_output, hidden_states, shard_dims


# ---------------------------------------------------------------------------------------------------
# Unified chunked-prefill driver. One loop (preload -> N iters of write+rope+ring_mla -> compare)
# parametrized by where the prefix/reference come from. See test_mla_chunked_prefill below.
# ---------------------------------------------------------------------------------------------------
# Set MLA_CHUNKED_TRACE_DIR to the ROOT dir holding one subdir per layer-0 GPU trace (each with
# mla_io/ + kv_cache/). It enables the prefill>0 scenarios (load the prior KV + reference from the
# real GPU run); multi-user pulls one trace per user, cycling if there are fewer traces than users.
# The root may mix kimi and deepseek traces as siblings; discover_traces filters by variant name.
MLA_CHUNKED_TRACE_DIR = os.environ.get("MLA_CHUNKED_TRACE_DIR")
# MLA_CHUNKED_TRACE_PATH points straight at ONE specific trace dir (the leaf holding mla_io/ +
# kv_cache/, not the root). It takes precedence over MLA_CHUNKED_TRACE_DIR and skips the root
# scan/variant-filter entirely; the single trace is shared (cycled) across all users.
MLA_CHUNKED_TRACE_PATH = os.environ.get("MLA_CHUNKED_TRACE_PATH")

# Per-iteration VALID token counts for the rotation/padding edge cases, tuned for the TARGET 8x4 mesh
# (sp=8, chunk_local=640, chunk=5120). Each cumulative kv_actual lands on a distinct rotation edge:
# which chip the boundary falls on (0..7), chip-aligned vs mid-chip straddle (offset != 0), single vs
# multi-slab, and how much of the chunk is pad. All values are tile-aligned (multiple of 32).
ROTATED_VALID_LISTS = [
    [640, 5120],  # aligned_min: iter0 = 1 chip valid (7 chips pad), then chip-1 rotated full
    [672, 5120],  # midchip_straddle: frontier 1 tile into chip 1, then rotated with offset=32 straddle
    [4480, 5120],  # lastchip: iter0 = 7 chips, rotation at the LAST chip (chip 7)
    [1280, 1920, 5120],  # rot_partial: iter1 is rotated AND partial (3-chip valid, 5-chip pad)
    [5120, 1280, 5120],  # multislab: rotation in slab 1 (multi-slab), partial then full
    [5120, 5120],  # allfull: sanity, two full chunks at slab boundaries (aligned, no rotation)
]
ROTATED_VALID_IDS = ["aligned_min", "midchip_straddle", "lastchip", "rot_partial", "multislab", "allfull"]


def _run_chunked_prefill(
    request,
    mesh_device,
    *,
    iters_isl,
    reference="cpu",
    chunk_size_global=5120,
    prefill_len=0,
    num_users=1,
    use_pretrained=False,
    topology=ttnn.Topology.Linear,
):
    """Unified chunked-prefill scenario, decoupled from the reference.

    `reference` selects how inputs + ground truth are produced -- independent of prefill_len / env:
      * "cpu"   -> synthetic inputs + torch MLA reference (k_pe in Meta basis). Partial-chunk iters
                   (rotation) allowed; any prefix is preloaded from the CPU reference KV.
      * "trace" -> GPU-trace inputs + reference (k_pe in HF basis, re-interleaved to compare). TRACE
                    ONLY: requires MLA_CHUNKED_TRACE_DIR or MLA_CHUNKED_TRACE_PATH (skips if both unset); supports partial iters.
      * None    -> no reference (functional/perf): random inputs + random prefix, finite-output check.
    Multi-user partitions iters_isl across users (last gets the remainder); each user is independent in
    its own cache slot, so cross-user contamination surfaces as a per-user output PCC drop.
    """
    assert reference in ("cpu", "trace", None), f"reference must be 'cpu'|'trace'|None, got {reference!r}"
    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tile = ttnn.TILE_SIZE
    chunk_local = chunk_size_global // sp

    assert chunk_size_global % (tile * sp) == 0, f"chunk_size_global {chunk_size_global} % (TILE*sp={tile * sp}) != 0"
    for v in iters_isl:
        assert 0 < v <= chunk_size_global and v % tile == 0, f"iter isl {v}: tile-aligned and <= {chunk_size_global}"
    assert prefill_len % tile == 0, f"prefill_len {prefill_len} must be tile-aligned"

    use_trace = reference == "trace"
    if use_trace:
        if MLA_CHUNKED_TRACE_DIR is None and MLA_CHUNKED_TRACE_PATH is None:
            pytest.skip(
                "reference='trace' requires MLA_CHUNKED_TRACE_DIR (root) or "
                "MLA_CHUNKED_TRACE_PATH (single trace) -- trace-only scenario"
            )
        # The trace is a DENSE token sequence; iters_isl just chunks it variably. Partial iters pad
        # the device's fixed-width chunk (masked by causality) -- they are not pad in the sequence --
        # so any iters_isl / prefill works exactly like the CPU ref. The only trace constraint is
        # total_len <= trace length, asserted per-user below.
        use_pretrained = True  # the GPU trace was generated with the real checkpoint

    groups = partition_iters(iters_isl, num_users)
    # Resolve trace dirs: a single explicit trace (MLA_CHUNKED_TRACE_PATH) wins; otherwise scan the
    # root (MLA_CHUNKED_TRACE_DIR) and filter the kimi/deepseek siblings by variant name.
    if not use_trace:
        traces = None
    elif MLA_CHUNKED_TRACE_PATH is not None:
        traces = single_trace(MLA_CHUNKED_TRACE_PATH, num_users)
    else:
        variant_name = request.getfixturevalue("variant").name
        traces = discover_traces(MLA_CHUNKED_TRACE_DIR, num_users, variant_name)

    # Cache holds the max (kv_actual + chunk) window across all users/iters, slab-aligned, >= 2 slabs.
    max_window = chunk_size_global * 2
    for g in groups:
        ka = prefill_len
        for v in g:
            max_window = max(max_window, ka + chunk_size_global)
            ka += v
    seq_len_cache = ((max_window + chunk_size_global - 1) // chunk_size_global) * chunk_size_global

    if use_pretrained:
        config, sd = request.getfixturevalue("pretrained_transformer_weights")
        weights = sd["layers"][0]["mla_weights"]
    else:
        config, weights = request.getfixturevalue("random_weights")
    config.max_seq_len = seq_len_cache
    kvpe_dim = config.kv_lora_rank + config.qk_rope_head_dim
    hidden_size = config.hidden_size

    logger.info(
        f"chunked prefill: mesh={tuple(mesh_device.shape)} chunk={chunk_size_global} prefill={prefill_len} "
        f"iters={iters_isl} users={num_users} reference={reference} "
        f"weights={'pretrained' if use_pretrained else 'random'} seq_len_cache={seq_len_cache}"
    )

    # ---- per-user inputs + references. Each source provides hidden + (ref_out, ref_kvpe); the prior
    #      prefix KV is carved from that same reference (random for the functional, ref-less mode). ----
    users = []  # each: {group, total_len, hidden, ref_out|None, kv_prior|None, kv_post|None}
    for u in range(num_users):
        g = groups[u]
        total_len = prefill_len + sum(g)
        if reference == "trace":
            mi, mo, kv = load_trace(traces[u])
            assert total_len <= mi.shape[0], f"user {u}: prefill+iters {total_len} > trace len {mi.shape[0]}"
            hidden, ref_out, ref_kvpe = mi[:total_len], mo[:total_len], kv[:total_len]
        elif reference == "cpu":
            torch.manual_seed(42 + u)
            hidden = torch.randn(total_len, hidden_size, dtype=torch.bfloat16)
            ref_out, ref_kvpe = cpu_mla_reference(config, weights, hidden)
        else:  # None -> functional / perf, no reference
            torch.manual_seed(100 + u)
            hidden = torch.randn(total_len, hidden_size, dtype=torch.bfloat16)
            ref_out, ref_kvpe = None, None

        if prefill_len == 0:
            kv_prior = None
        elif ref_kvpe is not None:
            kv_prior = ref_kvpe[:prefill_len]  # preload the reference's prior KV (cpu or trace)
        else:
            kv_prior = torch.randn(prefill_len, kvpe_dim, dtype=torch.bfloat16)  # functional: random prefix
        users.append(
            dict(group=g, total_len=total_len, hidden=hidden, ref_out=ref_out, kv_prior=kv_prior, kv_post=ref_kvpe)
        )

    # ---- device setup ----
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        topology=topology,
        slot_num=num_users,
        layer_num=1,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis)
    indexed_rope = rope_setup.get_rope_tensors_indexed(
        cache_seq_len_global=seq_len_cache, chunk_size_global=chunk_size_global
    )
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=num_users,
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[tp_axis] = -1
    hidden_shard_dims[sp_axis] = -2
    out_concat_dims = [None, None]
    out_concat_dims[tp_axis] = -1
    out_concat_dims[sp_axis] = -2
    cache_shard_dims = [None, None]
    cache_shard_dims[sp_axis] = 2

    # ---- preload the prior prefix (trace or random) into each slot, block-cyclic ----
    if prefill_len > 0:
        logger.info(f"Preloading {prefill_len}-token prefix into {num_users} slot(s) (block-cyclic host->device)...")
        cache_host = torch.zeros(num_users, 1, seq_len_cache, kvpe_dim, dtype=torch.bfloat16)
        for u in range(num_users):
            kv_prior = users[u]["kv_prior"]
            if use_trace:
                # The GPU trace stores k_pe in the HF half-split basis; the device cache is the Meta
                # interleaved basis. Re-interleave the k_pe block before preload (k_nope is basis-
                # agnostic) -- same transform the post-run cache comparison applies. Without this the
                # 50k preloaded prefix attends in the wrong basis and only the output PCC (not the
                # cache PCC, which checks just the new region) shows the ~0.92 drop.
                kv_prior = kv_prior.clone()
                d = kvpe_dim - config.kv_lora_rank
                pe = kv_prior[:, config.kv_lora_rank :]
                kv_prior[:, config.kv_lora_rank :] = torch.stack([pe[:, : d // 2], pe[:, d // 2 :]], dim=-1).reshape(
                    pe.shape[0], d
                )
            cache_host[u, 0] = blockcyclic_cache_host(kv_prior, sp, chunk_size_global, seq_len_cache, kvpe_dim)[0, 0]
        cache_host_tt = ttnn.from_torch(
            cache_host,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=cache_shard_dims),
        )
        ttnn.copy_host_to_device_tensor(cache_host_tt, tt_kvpe_cache)
        ttnn.synchronize_device(mesh_device)

    mesh_device.enable_program_cache()
    # Accumulated natural-order output per user (only the measured region is filled).
    out_accum = [torch.zeros(1, 1, users[u]["total_len"], hidden_size, dtype=torch.bfloat16) for u in range(num_users)]

    # ---- iterate: interleave users by local iter index (exercises cross-user isolation) ----
    n_iters = max(len(u["group"]) for u in users)
    logger.info(f"Starting DEVICE chunked prefill: up to {n_iters} iters x {num_users} user(s)")
    for i in range(n_iters):
        for u in range(num_users):
            g = users[u]["group"]
            if i >= len(g):
                continue
            isl = g[i]
            kv_actual = prefill_len + sum(g[:i])
            valid_end = kv_actual + isl
            total_len = users[u]["total_len"]

            positions = rotated_chip_positions(kv_actual, sp, chunk_local)
            flat = [positions[c][r] for c in range(sp) for r in range(chunk_local)]
            gather_idx = torch.tensor([min(gp, total_len - 1) for gp in flat], dtype=torch.long)
            chunk_in = users[u]["hidden"][gather_idx].clone()
            chunk_in[torch.tensor([gp >= valid_end for gp in flat])] = 0.0

            tt_h = ttnn.from_torch(
                chunk_in.reshape(1, 1, chunk_size_global, hidden_size),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=hidden_shard_dims
                ),
            )
            tt_out = mla_tt.forward(
                hidden_states=tt_h,
                rope_tensors=indexed_rope,
                kvpe_cache=tt_kvpe_cache,
                actual_start=kv_actual,
                cache_user_id=u,
            )
            out_flat = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape
                ),
            ).to(torch.bfloat16)[0, 0]

            assert torch.isfinite(out_flat).all(), f"user {u} iter {i}: non-finite output"
            valid_pairs = [(row, gp) for row, gp in enumerate(flat) if gp < valid_end]
            src = torch.tensor([row for row, _ in valid_pairs], dtype=torch.long)
            dst = torch.tensor([gp for _, gp in valid_pairs], dtype=torch.long)
            out_accum[u][0, 0, dst, :] = out_flat[src, :]

            if users[u]["ref_out"] is not None:
                _, msg = assert_with_pcc(
                    users[u]["ref_out"][kv_actual:valid_end].reshape(1, 1, isl, hidden_size),
                    out_accum[u][:, :, kv_actual:valid_end, :],
                    0.98,
                )
                rot = "rotated" if kv_actual % chunk_size_global != 0 else "aligned"
                logger.info(f"  user {u} iter {i} (kv_actual={kv_actual} isl={isl} {rot}): out PCC {msg}")
        ttnn.synchronize_device(mesh_device)
        ttnn.distributed_context_barrier()

    if reference is None:
        logger.success(f"✓ Functional chunked prefill ran ({num_users} user(s), finite output)")
        return

    # ---- per-user full-measured-region output PCC ----
    for u in range(num_users):
        if users[u]["ref_out"] is None:
            continue
        meas = out_accum[u][:, :, prefill_len:, :]
        ref_meas = users[u]["ref_out"][prefill_len:].reshape(1, 1, -1, hidden_size)
        _, msg = assert_with_pcc(ref_meas, meas, 0.98)
        logger.info(f"  user {u} full measured output PCC: {msg}")

    # ---- check the measured KV cache vs the reference. The rotation accumulates into the canonical
    #      block-cyclic layout, so blockcyclic_positions un-rotates the final cache (incl. partial
    #      chunks). k_nope is compared directly; k_pe is direct for the CPU ref (mla_reference is
    #      Meta-style) but re-interleaved for the GPU trace (HF half-split -> device Meta basis). ----
    if any(users[u]["kv_post"] is not None for u in range(num_users)):
        cache_sr = ttnn.to_torch(
            tt_kvpe_cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
        ).to(torch.float32)[
            :, :1
        ]  # TP replica 0 -> [num_users, 1, seq_cache, kvpe]
        p = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
        kv_lora = config.kv_lora_rank
        d = kvpe_dim - kv_lora
        for u in range(num_users):
            if users[u]["kv_post"] is None:
                continue
            nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
            nat[p] = cache_sr[u, 0]
            dev = nat[prefill_len : users[u]["total_len"]]
            ref = users[u]["kv_post"][prefill_len:].to(torch.float32)
            ref_pe = ref[:, kv_lora:]
            if use_trace:  # GPU trace stores k_pe HF half-split -> re-interleave to the device Meta basis
                ref_pe = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(-1, d)
            _, nope_msg = assert_with_pcc(ref[:, :kv_lora], dev[:, :kv_lora], 0.98)
            _, pe_msg = assert_with_pcc(ref_pe, dev[:, kv_lora:], 0.98)
            basis = "Meta-aligned" if use_trace else "direct"
            logger.info(f"  user {u} KV cache PCC -- k_nope: {nope_msg}  k_pe[{basis}]: {pe_msg}")

    logger.success(f"✓ Chunked prefill passed ({'trace' if use_trace else 'cpu'} ref, {num_users} user(s))")


# Functionality scenarios (id, kwargs) -- PURE FUNCTIONALITY: no mesh, no reference. Mesh and
# reference are SEPARATE pytest axes below (chunk=5120 is valid for sp in {2,4,8}), so the same
# scenario runs on any mesh and is validated against either ground truth (or run functional) without
# duplicating the case.
_CHUNKED_SCENARIOS = (
    [(f"rot-{rid}", dict(iters_isl=lst)) for rid, lst in zip(ROTATED_VALID_IDS, ROTATED_VALID_LISTS)]
    # One representative case packing the most sp=8 edges: iter0 aligned partial, iter1 rotated
    # chip-aligned (offset=0) partial, iter2 rotated mid-chip straddle (offset=32) + multi-slab + full.
    # NOTE: ids must not nest as substrings, else `-k <id>` can't isolate one (pytest -k is substring).
    # Convention: "-Nu" = N users. "maxedge"/"deep" are intentional families (single- + multi-user).
    + [("maxedge-1u", dict(iters_isl=[2560, 2592, 5120]))]
    + [
        ("production-50k+5k", dict(iters_isl=[5120] * 11)),
        ("fullchunk-2u", dict(iters_isl=[5120] * 4, num_users=2)),
        # Multi-user WITH padding/rotation: each user runs the full maxedge pattern in its own slot
        # (partition splits [..]*2 into one maxedge per user), exercising rotation + cross-user isolation.
        ("maxedge-2u", dict(iters_isl=[2560, 2592, 5120] * 2, num_users=2)),
        ("deep-50k+5k", dict(iters_isl=[5120], prefill_len=50 * 1024)),
        ("deep-2u", dict(iters_isl=[5120, 5120], prefill_len=50 * 1024, num_users=2)),
    ]
)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        },
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        },
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
    ],
    ids=["line", "ring", "fabric2d"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4), (8, 4)], ids=["2x2", "2x4", "8x4"], indirect=True)
@pytest.mark.parametrize("reference", ["cpu", "trace", None], ids=["cpu", "trace", "func"])
@pytest.mark.parametrize("kwargs", [kw for _, kw in _CHUNKED_SCENARIOS], ids=[sid for sid, _ in _CHUNKED_SCENARIOS])
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p", "kimi_k2_6"], indirect=True, ids=["dsv3", "kimi"])
@pytest.mark.timeout(0)
def test_mla_chunked_prefill(request, mesh_device, kwargs, reference, device_params, variant):
    """Unified chunked-prefill driver crossed with independent mesh and reference axes. Each
    functionality scenario (rotation edges, production depth, multi-user, deep prefix) runs on any mesh
    and is validated against the CPU torch reference ('cpu'), the GPU trace ('trace', skips without
    MLA_CHUNKED_TRACE_DIR/PATH), or run with no reference ('func'). Select with e.g.
    -k 'maxedge-1u and trace and 8x4'. See _run_chunked_prefill.

    Real weights on the CPU-reference path: point the variant's HF env var (DEEPSEEK_V3_HF_MODEL /
    KIMI_K2_6_HF_MODEL) at a checkpoint to validate the chunked path against the CPU torch reference
    with pretrained weights instead of random. create_mla_reference is config-driven and
    architecture-agnostic (Kimi's YaRN/theta flow through, absorbed-MLA math matches the variant's own
    reference), so this works for both variants. It complements the deepseek GPU-trace path, which only
    replays full-chunk iters and so never exercises real weights across the rotation/partial-chunk edge
    scenarios that the cpu path covers. Without the env var, fall back to random (mirroring
    test_kimi_mla). kimi_k2_6 also runs the trace path (loader + k_pe re-interleave are arch-agnostic; needs kimi
    GPU traces in MLA_CHUNKED_TRACE_DIR). It otherwise runs the same
    config-driven driver on any arch/mesh."""
    # Opt into real weights on the cpu path when the variant's checkpoint env var is set. The "trace"
    # path already forces pretrained; "func" is ref-less so weights don't matter. The pretrained
    # fixture skips the test if the env var is set but the checkpoint is incomplete.
    if reference == "cpu" and os.environ.get(variant.env_var) and not kwargs.get("use_pretrained"):
        kwargs = {**kwargs, "use_pretrained": True}
    topology = (
        ttnn.Topology.Ring
        if device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_1D_RING
        else ttnn.Topology.Linear
    )
    _run_chunked_prefill(request, mesh_device, reference=reference, topology=topology, **kwargs)
