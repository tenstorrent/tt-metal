# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-user chunked prefill for Kimi-K3: the oracle for packed (batched) forwards.

A packed forward holds several chunks, possibly from different users at different depths. Each layer runs its
attention once per chunk against that chunk's own user state -- an MLA layer its KV slab, a KDA layer its slot's
recurrent and convolution carries in the engine's slabs -- and its FFN (ffn_norm, MoE or dense FFN) once over
every chunk's rows. AttnRes needs nothing per chunk: every one of its reductions is over the hidden dimension, so
one walk covers all rows. Whatever the packing, each chunk's result must equal what that chunk gets alone. These
tests pin that down device against device, so they need the TTNN weight cache but no golden trace.

* test_k3_multiuser_interleaved_matches_sequential: one chunk per forward. N users prefilled chunk by chunk in
  round-robin order into one set of slots, and each user start to finish into another, must agree: last-layer
  output, every KDA layer's carries and every MLA layer's KV.
* test_k3_packed_forward_matches_one_chunk_forwards: two chunks per forward (FORWARD_CHUNKS). A schedule with
  partial chunks, a partial chunk followed by a real one, one user's consecutive chunks in the same forward (the
  second continues the first's KDA carry and reads its KV), users at different depths and empty tail chunks must
  match the same chunks each alone in its own forward.
* test_k3_packed_traced_matches_eager: the same schedule through TtKimiK3Runtime.prefill_forward with use_trace
  (one capture, replayed per forward; each chunk position's slot and bounds and the MoE padding config read on
  device) must leave the same KDA carries and KV as the eager packed forward.

Depth: 4 layers hold every Kimi-K3 layer kind: 0 KDA + dense FFN, 1-2 KDA + MoE, 3 MLA + MoE.

What packing may change: only rounding in AttnRes. With the plain residual stream (PlainResidualStream, the
bisection arm) a packed forward is bit-exact against one chunk per forward at every layer, KDA carry and KV slab,
MoE layers included (no #63555 effect here). TtAttnRes rounds a row differently depending on where the row sits in
the call: a chunk at position 1 of a forward differs from the same chunk at position 0 by ~1e-3 in ~1% of rows
after layer 0 (job 169400), which then compounds. So the packed test runs both: plain asserts bit-exact everywhere,
AttnRes asserts PCC bars.
"""

import copy

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotated_chunk_positions
from models.demos.common.prefill.prefill_chunk import PrefillChunk, validate_chunks
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, moe_fabric_payload
from models.demos.deepseek_v3_d_p.tests.kda.utils import read_slab_layer
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_slabs import attach_kda_states
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates, deallocate_state
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import PlainResidualStream
from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, assert_gate_mode_matches_adapter
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import KimiK3KvCaches
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural
from tests.ttnn.utils_for_testing import comp_pcc

NUM_LAYERS = 4
SP_AXIS, TP_AXIS = 0, 1
# 640 rows per SP chip on the 8x4 mesh: the chunk the KDA recurrence and the TTNN cache are built for.
CHUNK = 5120
N_CHUNKS = 3
NUM_USERS = 3
# Two chunks per forward: 1,280 MoE rows per chip. At four, the MoE's shared-expert matmuls (TtSharedExpert's
# get_program_configs, per_core_M = ceil(80 M-tiles / 9 rows) = 9) need 1,714,112 B of static circular buffers on
# Kimi-K3's widths, past the 1,572,864 B of L1; that program-config change belongs to the shared MoE.
FORWARD_CHUNKS = 2
GATE_MODE = GateComputeMode.DEVICE_FP32

# Interleaving changes only the order of whole forward() calls: bit-exact up to a reduction-order tie.
INTERLEAVED_PCC = 0.9999
# The AttnRes packed case (module docstring): bars with margin over the measured output PCC >= 0.99705 and KDA carry /
# KV PCC >= 0.99985 (job 168987), tight enough to catch a packing error. The plain-residual case is asserted bit-exact.
PACKED_STATE_PCC = 0.998
PACKED_OUTPUT_PCC = 0.995
# Two different users must NOT agree; catches a comparison that reads the same slot twice.
DISTINCT_USERS_MAX_PCC = 0.99
# Traced and eager run the same programs (the traced ones read their scalars on device).
TRACED_PCC = 0.9999
K3_TRACE_REGION_SIZE = 256 * 1024 * 1024

MESH_PARAMS = [
    pytest.param(
        (8, 4),
        fabric2d_device_params(
            fabric_payload_size=moe_fabric_payload(KimiK3Config),
            l1_small_size=KimiK3Config.L1_SMALL_SIZE,
            trace_region_size=K3_TRACE_REGION_SIZE,
        ),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    ),
]

_COMMON = [
    pytest.mark.parametrize(
        "mesh_device, device_params, num_links", MESH_PARAMS, indirect=["mesh_device", "device_params"]
    ),
    pytest.mark.parametrize("variant", ["kimi_k3"], indirect=True, ids=["kimi_k3"]),
    pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 requires Blackhole"),
    pytest.mark.timeout(0),
]


def _common(fn):
    for mark in reversed(_COMMON):
        fn = mark(fn)
    return fn


def _cache_path(mesh_device, weight_cache_path, num_layers=NUM_LAYERS):
    sp, tp = tuple(mesh_device.shape)
    path = weight_cache_path / f"{sp}x{tp}"
    assert TtKimiK3Transformer.check_cache_complete(
        path,
        num_layers,
        experts_per_chip=KimiK3Config.NUM_ROUTED_EXPERTS // (sp * tp),
        first_k_dense=KimiK3Config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {path}"
    return path


def _build(
    mesh_device, config, device_params, num_links, weight_cache_path, num_slots, forward_chunks=1, residual="attnres"
):
    """The 4-layer model from the TTNN cache, its KV cache and KDA slabs with ``num_slots`` user slots. ``residual``:
    "attnres" (the model) or "plain" (PlainResidualStream, the bisection arm)."""
    seq_cache = N_CHUNKS * CHUNK
    extra = {}
    if residual == "plain":
        extra["residual_factory"] = lambda hidden, block_residual=None: PlainResidualStream(hidden)
    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        {},
        num_layers=NUM_LAYERS,
        seq_len=CHUNK,
        max_seq_len=seq_cache,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        gate_fallback_mode=GATE_MODE,
        weight_cache_path=_cache_path(mesh_device, weight_cache_path),
        is_chunked=True,
        slot_num=num_slots,
        routing_use_l1_small_for_semaphores=True,
        tp_shard_kv=True,
        # No final norm: the comparisons read the residual stream's model-level AttnRes read.
        build_tail=False,
        forward_chunks=forward_chunks,
        **extra,
    )
    kvpe = allocate_mla_kvpe_cache(
        mesh_device=mesh_device,
        hf_config=config,
        max_seq_len=seq_cache,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_layers=model.schedule.num_mla_layers,
        num_users=num_slots,
        tp_axis=TP_AXIS,
        bank_aligned=False,  # paging alignment wants multiples of 8,192 tokens; this cache holds 15,360
    )
    attach_kda_states(mesh_device, model, sp_axis=SP_AXIS, tp_axis=TP_AXIS, num_slots=num_slots)
    for slot in range(num_slots):
        model.reset_streams(slot)
    return model, kvpe


def _token_ids(config, num_users, length):
    """Distinct deterministic in-vocab ids per user, [length] each."""
    return [
        torch.randint(0, config.vocab_size, (length,), generator=torch.Generator().manual_seed(1000 + u))
        for u in range(num_users)
    ]


def _forward_input(mesh_device, chunks, ids_by_slot):
    """A forward's tokens in the packed per-chip layout (pack_forward_tokens), SP-sharded on device. A partial
    chunk's tail and an empty chunk are padded with token 0."""
    sp = mesh_device.shape[SP_AXIS]
    chunk_tokens = []
    for c in chunks:
        real = [] if c.is_empty else ids_by_slot[c.slot_id][c.actual_start : c.actual_end].tolist()
        chunk_tokens.append(real + [0] * (CHUNK - len(real)))
    packed = pack_forward_tokens(chunk_tokens, [c.actual_start for c in chunks], sp)
    return ttnn.from_torch(
        torch.tensor(packed, dtype=torch.int64).reshape(sp, 1, len(chunks) * CHUNK // sp),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(SP_AXIS, None)),
    )


def _run_forward(model, kvpe, mesh_device, chunks, ids_by_slot):
    """One forward of ``chunks``; the residual stream's final read on host, [n * CHUNK, emb], in the packed
    per-chip row order. A one-chunk forward on a one-chunk model takes the regular (non-packed) path."""
    tokens = _forward_input(mesh_device, chunks, ids_by_slot)
    for c in chunks:
        if not c.is_empty and c.actual_start == 0:
            model.reset_streams(c.slot_id)
    if len(chunks) == 1 and model.forward_chunks == 1:
        c = chunks[0]
        out = model.forward(
            tokens,
            kvpe,
            actual_isl=CHUNK,
            actual_start=c.actual_start,
            actual_end=c.actual_end,
            cache_user_id=c.slot_id,
        )
    else:
        out = model.forward(tokens, kvpe, actual_isl=CHUNK, chunks=chunks)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(tokens)
    host = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    ).to(torch.float32)
    ttnn.deallocate(out)
    return host[0, 0]


def _chunk_real_rows(hidden, k, num_chunks, chunk_obj, sp):
    """Chunk k's real rows out of a forward's host output (chip-major: chip c holds every chunk's rows back to
    back), as [n_real, emb]."""
    chunk_local = CHUNK // sp
    positions = rotated_chunk_positions(chunk_obj.actual_start, sp, chunk_local)
    rows = []
    for c in range(sp):
        base = (c * num_chunks + k) * chunk_local
        for r in range(chunk_local):
            if positions[c][r] < chunk_obj.actual_end:
                rows.append(hidden[base + r])
    return torch.stack(rows)


def _shards(tensor):
    return torch.cat([ttnn.to_torch(t).flatten().to(torch.float32) for t in ttnn.get_device_tensors(tensor)])


def _kda_slot(mesh_device, slabs, slot, layer_idx):
    """Layer ``layer_idx``'s carries at ``slot``, every device's shard flattened: (recurrent, convolution)."""
    state = read_slab_layer(mesh_device, slabs, slot, layer_idx)
    try:
        return _shards(state.recurrent), _shards(state.convolution)
    finally:
        deallocate_state(state)


def _kv_slots(kvpe, mesh_device, num_mla_layers, num_slots=None):
    """Per-slot KV rows. ``num_slots`` defaults to what the cache holds: a traced packed runtime's cache has one more
    slot than the KDA slabs, the scratch slot an empty chunk position writes."""
    rows, _ = gather_cache_natural(kvpe.storage, mesh_device, tp_shard_kv=True)
    if num_slots is None:
        num_slots = rows.shape[0] // num_mla_layers
    assert rows.shape[0] == num_slots * num_mla_layers, (rows.shape, num_slots, num_mla_layers)
    return [rows[s * num_mla_layers : (s + 1) * num_mla_layers] for s in range(num_slots)]


def _pcc(a, b):
    _, pcc = comp_pcc(a, b)
    return float(pcc)


def _compare_state(label, mesh_device, model, kvpe, pairs, *, exact_layers, threshold):
    """Every (reference slot, slot under test) pair: each KDA layer's two carries and each MLA layer's KV. Layers in
    ``exact_layers`` must be bit-exact, the rest at least ``threshold`` PCC. Returns the failures."""
    failures = []
    slabs = model.kda_states
    kv = _kv_slots(kvpe, mesh_device, model.schedule.num_mla_layers)

    def check(name, layer, ref, got, ref_slot, got_slot):
        if layer in exact_layers:
            diff = (ref - got).abs().max().item()
            logger.info(f"[{label}] slot {got_slot} vs {ref_slot} {name}[{layer}] max|diff| {diff:.3e}")
            if diff != 0:
                failures.append(f"slot {got_slot} {name} layer {layer} not bit-exact (max|diff| {diff:.3e})")
        else:
            pcc = _pcc(ref, got)
            logger.info(f"[{label}] slot {got_slot} vs {ref_slot} {name}[{layer}] PCC {pcc:.6f}")
            if pcc < threshold:
                failures.append(f"slot {got_slot} {name} layer {layer} PCC {pcc:.6f} < {threshold}")

    for ref_slot, got_slot in pairs:
        for layer in model.schedule.rank_kda_layer_ids():
            ref = _kda_slot(mesh_device, slabs, ref_slot, layer)
            got = _kda_slot(mesh_device, slabs, got_slot, layer)
            check("kda_recurrent", layer, ref[0], got[0], ref_slot, got_slot)
            check("kda_convolution", layer, ref[1], got[1], ref_slot, got_slot)
        # The schedule lists the whole model's MLA layers; this rank holds the first num_mla_layers of them.
        for rank, layer in enumerate(model.schedule.mla_layer_ids[: model.schedule.num_mla_layers]):
            check("kvpe", layer, kv[ref_slot][rank], kv[got_slot][rank], ref_slot, got_slot)
    return failures


def _packed_schedule():
    """(user lengths, forwards) for four users. Every forward holds FORWARD_CHUNKS (2) chunks; user u lives in slot
    u. Covers an empty tail chunk, one user's consecutive chunks in the same forward (the second continues the first's
    KDA carry and reads its KV), a partial chunk followed by a real one at a different depth, and a partial chunk
    alone. Starts are 32-aligned (KDA); ends are not."""
    C = CHUNK
    lengths = [3 * C, C + 704, 896, 2 * C]  # A: 3 full, B: full + partial, C: one partial, D: 2 full
    E = PrefillChunk(0, 0, 0)
    forwards = [
        [PrefillChunk(0, 0, C), PrefillChunk(1, 0, C)],
        [PrefillChunk(0, C, 2 * C), E],
        [PrefillChunk(3, 0, C), PrefillChunk(3, C, 2 * C)],
        [PrefillChunk(2, 0, 896), PrefillChunk(0, 2 * C, 3 * C)],
        [PrefillChunk(1, C, C + 704), E],
    ]
    return lengths, forwards


def _config(config_only):
    config = copy.deepcopy(config_only)  # config_only is shared process-wide
    config.max_seq_len = N_CHUNKS * CHUNK
    return config


@_common
def test_k3_multiuser_interleaved_matches_sequential(
    variant, config_only, mesh_device, device_params, num_links, weight_cache_path
):
    """Round-robin over users (slots N..2N-1) must reproduce each user run alone (slots 0..N-1)."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    num_slots = 2 * NUM_USERS
    mesh_device.enable_program_cache()
    config = _config(config_only)
    model, kvpe = _build(mesh_device, config, device_params, num_links, weight_cache_path, num_slots)
    try:
        ids = _token_ids(config, NUM_USERS, N_CHUNKS * CHUNK)
        ids_by_slot = {s: ids[s % NUM_USERS] for s in range(num_slots)}

        def run(slot, c):
            return _run_forward(model, kvpe, mesh_device, [PrefillChunk(slot, c * CHUNK, (c + 1) * CHUNK)], ids_by_slot)

        sequential = {(u, c): run(u, c) for u in range(NUM_USERS) for c in range(N_CHUNKS)}
        interleaved = {}
        for c in range(N_CHUNKS):
            for u in range(NUM_USERS):
                interleaved[(u, c)] = run(NUM_USERS + u, c)

        failures = []
        for key, ref in sequential.items():
            pcc = _pcc(ref, interleaved[key])
            logger.info(f"[interleaved] user {key[0]} chunk {key[1]} output PCC {pcc:.6f}")
            if pcc < INTERLEAVED_PCC:
                failures.append(f"user {key[0]} chunk {key[1]} output PCC {pcc:.6f} < {INTERLEAVED_PCC}")
        failures += _compare_state(
            "interleaved",
            mesh_device,
            model,
            kvpe,
            [(u, NUM_USERS + u) for u in range(NUM_USERS)],
            exact_layers=(),
            threshold=INTERLEAVED_PCC,
        )
        cross = _pcc(sequential[(0, N_CHUNKS - 1)], sequential[(1, N_CHUNKS - 1)])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users 0 and 1 produced the same output (PCC {cross:.6f})"
        assert not failures, f"[interleaved] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        model.release_sub_device_managers()


@pytest.mark.parametrize("residual", ["attnres", "plain"])
@_common
def test_k3_packed_forward_matches_one_chunk_forwards(
    variant, config_only, mesh_device, device_params, num_links, weight_cache_path, residual
):
    """Packed forwards (slots 0..3) must reproduce every chunk run alone in its own forward (slots 4..7).

    Both sides use the same packed model; the reference forward is the chunk followed by empty chunks, so the
    comparison isolates what sharing a forward with other chunks does to a chunk. Plain residual: bit-exact
    everywhere. AttnRes: PCC bars, since it rounds a row by its position in the call (module docstring).
    """
    exact = residual == "plain"
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    sp = mesh_device.shape[SP_AXIS]
    lengths, forwards = _packed_schedule()
    num_users = len(lengths)
    num_slots = 2 * num_users
    mesh_device.enable_program_cache()
    config = _config(config_only)
    model, kvpe = _build(
        mesh_device,
        config,
        device_params,
        num_links,
        weight_cache_path,
        num_slots,
        forward_chunks=FORWARD_CHUNKS,
        residual=residual,
    )
    try:
        for chunks in forwards:
            validate_chunks(
                chunks,
                chunk_size=CHUNK,
                forward_chunks=FORWARD_CHUNKS,
                num_users=num_slots,
                max_seq_len=N_CHUNKS * CHUNK,
            )
        ids = _token_ids(config, num_users, max(lengths))
        ids_by_slot = {s: ids[s % num_users] for s in range(num_slots)}
        empty = PrefillChunk(0, 0, 0)

        packed = {}
        for f, chunks in enumerate(forwards):
            logger.info(f"[packed] forward {f}: {chunks}")
            hidden = _run_forward(model, kvpe, mesh_device, chunks, ids_by_slot)
            for k, c in enumerate(chunks):
                if not c.is_empty:
                    packed[(c.slot_id, c.actual_start)] = _chunk_real_rows(hidden, k, FORWARD_CHUNKS, c, sp)

        failures = []
        for chunks in forwards:
            for c in chunks:
                if c.is_empty:
                    continue
                ref_chunk = PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end)
                hidden = _run_forward(
                    model, kvpe, mesh_device, [ref_chunk] + [empty] * (FORWARD_CHUNKS - 1), ids_by_slot
                )
                ref = _chunk_real_rows(hidden, 0, FORWARD_CHUNKS, ref_chunk, sp)
                got = packed[(c.slot_id, c.actual_start)]
                pcc = _pcc(ref, got)
                diff = (ref - got).abs().max().item()
                logger.info(f"[packed] {residual} {c} output PCC {pcc:.6f} max|diff| {diff:.3e}")
                if exact and diff != 0:
                    failures.append(f"{c} output not bit-exact (max|diff| {diff:.3e})")
                elif pcc < PACKED_OUTPUT_PCC:
                    failures.append(f"{c} output PCC {pcc:.6f} < {PACKED_OUTPUT_PCC}")

        failures += _compare_state(
            f"packed-{residual}",
            mesh_device,
            model,
            kvpe,
            [(u + num_users, u) for u in range(num_users)],
            exact_layers=tuple(range(NUM_LAYERS)) if exact else (),
            threshold=PACKED_STATE_PCC,
        )
        cross = _pcc(packed[(0, 0)], packed[(3, 0)])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users A and D produced the same output (PCC {cross:.6f})"
        assert not failures, f"[packed] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        model.release_sub_device_managers()


@_common
def test_k3_packed_traced_matches_eager(variant, config_only, mesh_device, device_params, num_links, weight_cache_path):
    """Traced packed forwards (slots 0..3) must leave the same KDA carries and KV as the eager packed forward of the
    same chunks on the same model (slots 4..7)."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    lengths, forwards = _packed_schedule()
    num_users = len(lengths)
    num_slots = 2 * num_users
    seq_cache = N_CHUNKS * CHUNK
    config = _config(config_only)
    rt_config = TtPrefillRuntimeConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=seq_cache,
        mesh_shape=tuple(mesh_device.shape),
        chunk_size=CHUNK,
        forward_tokens_budget=FORWARD_CHUNKS * CHUNK,
        num_users=num_slots,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=GATE_MODE,
        model_cfg=KimiK3Config,
        routing_use_l1_small_for_semaphores=True,
        weight_cache_path=_cache_path(mesh_device, weight_cache_path),
        # Layer 3 is MLA, so the last layer may be KV-only: the caches are the output.
        kv_only_last_layer=True,
        use_trace=True,
        tp_shard_kv=True,
        # Kimi-K3's MLA cache is dense (the K3 adapter resolves this to None); the config default is GLM's.
        sparse_kv_cache_format=None,
    )
    mesh_device.enable_program_cache()
    runtime = TtKimiK3Runtime(mesh_device, config, {}, rt_config)
    model = runtime.model
    caches = KimiK3KvCaches(
        kvpe=allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=seq_cache,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            # The users plus the traced forward's scratch slot for an empty chunk position.
            num_users=rt_config.cache_num_users,
            tp_axis=TP_AXIS,
            bank_aligned=False,  # paging alignment wants multiples of 8,192 tokens; this cache holds 15,360
        ),
        # num_users exactly: the scratch slot is out of range for the KDA copies, which skip it on device.
        kda_states=KdaStates.allocate(
            mesh_device,
            kimi_k3_kda_geometry(tuple(mesh_device.shape), SP_AXIS, TP_AXIS),
            layer_ids=model.schedule.rank_kda_layer_ids(),
            num_slots=num_slots,
        ),
    )
    try:
        runtime.compile(caches)
        runtime.capture_trace(caches)
        for slot in range(num_slots):
            model.reset_streams(slot)
        ids = _token_ids(config, num_users, max(lengths))

        def forward_input(chunks, user_of):
            tokens = []
            for c in chunks:
                real = [] if c.is_empty else ids[user_of(c.slot_id)][c.actual_start : c.actual_end].tolist()
                tokens.append(real + [0] * (CHUNK - len(real)))
            return runtime.make_forward_input(tokens, [c.actual_start for c in chunks])

        for f, chunks in enumerate(forwards):
            logger.info(f"[traced] forward {f}: {chunks}")
            runtime.prefill_forward(forward_input(chunks, lambda s: s), caches, chunks)
        ttnn.synchronize_device(mesh_device)

        # Eager reference on the same model: forward_packed called directly runs the host-scalar programs.
        for chunks in forwards:
            ref = [
                c if c.is_empty else PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end) for c in chunks
            ]
            for c in ref:
                if not c.is_empty and c.actual_start == 0:
                    model.reset_streams(c.slot_id)
            inp = forward_input(ref, lambda s: s - num_users)
            model.forward(inp, caches.kvpe, actual_isl=CHUNK, chunks=ref)
            ttnn.deallocate(inp)
        ttnn.synchronize_device(mesh_device)

        failures = _compare_state(
            "traced",
            mesh_device,
            model,
            caches.kvpe,
            [(u + num_users, u) for u in range(num_users)],
            exact_layers=(),
            threshold=TRACED_PCC,
        )
        kv = _kv_slots(caches.kvpe, mesh_device, model.schedule.num_mla_layers, rt_config.cache_num_users)
        cross = _pcc(kv[0][-1], kv[3][-1])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users A and D hold the same KV (PCC {cross:.6f})"
        assert not failures, f"[traced] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        runtime.release_trace()


# A pipeline split must land on an AttnRes block edge (a multiple of 12), so the smallest is rank 0 = layers 0..11 and
# rank 1 = layers 12..15: every K3 layer kind on both sides of the boundary, the last layer MLA (KV-only), and a sealed
# AttnRes snapshot crossing it.
PIPELINE_LAYERS = 16
PIPELINE_SPLIT = 12
D2H_READ_TIMEOUT_S = 120
# Why two ranks round differently from one under AttnRes, although the planes cross in the same dtype and layout: the
# call shape at the boundary. In one rank, layer 11's output is still a pending write when layer 12's read runs, and
# attn_res_gather_softmax(pending=...) scores the stream with that addend folded in inside the kernel. At a split,
# rank 0's handoff settles it first with a separate ttnn.add (TtAttnResStream.detach), so rank 1's first read scores an
# already-materialized sum. Everything after it follows. Rank 0 stays bit-exact; rank 1 measured PCC >= 0.999743 (job
# 169741). This is a serving property of any Kimi-K3 split, packed or not -- its own one-chunk split test
# (test_transformer_pipeline_split.py) gates on PCC too -- and handing the pending addend over as its own plane would
# remove it. With the plain residual stream there is no deferred write, and the split is asserted bit-exact.
PIPELINE_PCC = 0.9995


class _PlainK3Transformer(TtKimiK3Transformer):
    """Kimi-K3 on the plain residual stream (the bisection arm), as a runtime builds it: the handoff is the live sum."""

    HANDS_OFF_SEALED = False

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, residual_factory=lambda hidden, block_residual=None: PlainResidualStream(hidden), **kwargs
        )


class _PlainK3Runtime(TtKimiK3Runtime):
    MODEL_CLS = _PlainK3Transformer


def _read_d2h_records(service, count):
    """``count`` 12 B (slot, start, end) records off the D2H FIFO, failing (not hanging) if fewer arrive."""
    import struct
    import threading

    records, errors = [], []

    def read():
        try:
            for _ in range(count):
                records.append(struct.unpack("<3I", service.read_metadata()[:12]))
        except Exception as e:  # surfaced below
            errors.append(e)

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(D2H_READ_TIMEOUT_S)
    assert not errors, errors
    assert not t.is_alive(), f"only {len(records)} of {count} D2H ack records arrived in {D2H_READ_TIMEOUT_S}s"
    return records


def _pipeline_rt_config(mesh_device, device_params, num_links, cache_path, first, n, num_slots, use_trace):
    return TtPrefillRuntimeConfig(
        num_layers=n,
        max_seq_len=N_CHUNKS * CHUNK,
        mesh_shape=tuple(mesh_device.shape),
        chunk_size=CHUNK,
        forward_tokens_budget=FORWARD_CHUNKS * CHUNK,
        num_users=num_slots,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=8,
        gate_fallback_mode=GATE_MODE,
        model_cfg=KimiK3Config,
        routing_use_l1_small_for_semaphores=True,
        weight_cache_path=cache_path,
        kv_only_last_layer=first + n == PIPELINE_LAYERS,
        use_trace=use_trace,
        tp_shard_kv=True,
        sparse_kv_cache_format=None,
        first_layer_idx=first,
        is_first_rank=first == 0,
        is_last_rank=first + n == PIPELINE_LAYERS,
    )


def _pipeline_caches(mesh_device, config, runtime):
    """A rank's caches: its MLA layers' KV at cache_num_users slots (the traced scratch slot included), its KDA
    layers' slabs at num_users."""
    model = runtime.model
    return KimiK3KvCaches(
        kvpe=allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=N_CHUNKS * CHUNK,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=runtime.config.cache_num_users,
            tp_axis=TP_AXIS,
            bank_aligned=False,  # 15,360 tokens is not a multiple of 8,192
        ),
        kda_states=KdaStates.allocate(
            mesh_device,
            kimi_k3_kda_geometry(tuple(mesh_device.shape), SP_AXIS, TP_AXIS),
            layer_ids=model.schedule.rank_kda_layer_ids(),
            num_slots=runtime.config.num_users,
        ),
    )


def _rank_state(mesh_device, runtime, caches, slots):
    """{(global layer, kind): [per-slot host tensor]} for every KDA carry and MLA KV slab this rank holds."""
    model = runtime.model
    state = {}
    for layer in model.schedule.rank_kda_layer_ids():
        per = [_kda_slot(mesh_device, caches.kda_states, s, layer) for s in slots]
        state[(layer, "kda_recurrent")] = [p[0] for p in per]
        state[(layer, "kda_convolution")] = [p[1] for p in per]
    kv = _kv_slots(caches.kvpe, mesh_device, model.schedule.num_mla_layers)
    first = model.first_layer_idx
    mla = [l for l in model.schedule.mla_layer_ids if first <= l < first + model.num_layers]
    for rank, layer in enumerate(mla):
        state[(layer, "kvpe")] = [kv[s][rank] for s in slots]
    return state


def _release(runtimes):
    """Every runtime's traces before any sub-device manager (the models share one mesh here; in the runner each rank is
    its own process), then each model's managers."""
    for r in runtimes:
        if r._controller is not None:
            r._controller.release()
    for r in runtimes:
        r.release_trace()  # also the model's sub-device managers


@pytest.mark.parametrize(
    "traced, residual",
    [(False, "attnres"), (False, "plain"), (True, "attnres")],
    ids=["eager-attnres", "eager-plain", "traced-attnres"],
)
@_common
def test_k3_packed_pipeline_matches_one_rank(
    variant, config_only, mesh_device, device_params, num_links, weight_cache_path, traced, residual
):
    """Packed forwards split over two pipeline ranks (layers 0..11 and 12..15) must leave the same KDA carries and KV as
    one rank holding all 16 layers, with rank 0's packed activation -- the live AttnRes stream and its sealed snapshot,
    over both chunks' rows -- handed to rank 1 the way the runner's D2D socket hands it over.

    Rank 0 acks through a real D2H service: one 12 B (slot, start, end) record per non-empty chunk per layer, each
    chunk's records contiguous and in pack order (#63438's packed contract), after the forward's writes.

    Eager, plain residual stream: every layer on both ranks bit-exact against one rank -- the handoff itself is exact.
    Eager, AttnRes: rank 0 bit-exact (packing puts every chunk at the same position on both sides), rank 1 at
    PIPELINE_PCC (see there: the boundary settles AttnRes's deferred write differently). The one-rank reference runs
    first and is released before the ranks are built, so the mesh never holds both. Traced: the ranks replay their
    captures into slots 0..3 against the same two ranks' models run eagerly into slots 4..7, at TRACED_PCC."""
    assert_gate_mode_matches_adapter(variant, GATE_MODE)
    lengths, forwards = _packed_schedule()
    num_users = len(lengths)
    num_slots = num_users * (2 if traced else 1)
    config = _config(config_only)
    cache_path = _cache_path(mesh_device, weight_cache_path, PIPELINE_LAYERS)
    mesh_device.enable_program_cache()
    ids = _token_ids(config, num_users, max(lengths))

    def tokens(runtime, chunks):
        toks = []
        for c in chunks:
            real = [] if c.is_empty else ids[c.slot_id % num_users][c.actual_start : c.actual_end].tolist()
            toks.append(real + [0] * (CHUNK - len(real)))
        return runtime.make_forward_input(toks, [c.actual_start for c in chunks])

    runtime_cls = _PlainK3Runtime if residual == "plain" else TtKimiK3Runtime

    def rt(first, n, use_trace):
        return runtime_cls(
            mesh_device,
            config,
            {},
            _pipeline_rt_config(mesh_device, device_params, num_links, cache_path, first, n, num_slots, use_trace),
        )

    ref_state = None
    if not traced:
        ref = rt(0, PIPELINE_LAYERS, False)
        try:
            ref_caches = _pipeline_caches(mesh_device, config, ref)
            ref.compile(ref_caches)
            for chunks in forwards:
                ref.prefill_forward(tokens(ref, chunks), ref_caches, chunks)
            ttnn.synchronize_device(mesh_device)
            ref_state = _rank_state(mesh_device, ref, ref_caches, range(num_users))
        finally:
            _release([ref])
        del ref, ref_caches

    split = [(0, PIPELINE_SPLIT), (PIPELINE_SPLIT, PIPELINE_LAYERS - PIPELINE_SPLIT)]
    ranks = [rt(first, n, traced) for first, n in split]
    rank_caches = [_pipeline_caches(mesh_device, config, r) for r in ranks]
    d2h = ttnn.D2HStreamService(
        mesh_device,
        global_spec=None,
        fifo_size_bytes=16 * 1024,
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=12,
        metadata_ring=True,
    )
    try:
        planes = 1 if residual == "plain" else 2
        assert ranks[1].activation_planes == planes, f"rank 1 should receive {planes} plane(s)"
        for r, caches in zip(ranks, rank_caches):
            r.compile(caches)
        acks_per_chunk = ranks[0]._packed_ack_count()
        if traced:
            ranks[0].set_d2h_ack_service(d2h)
            for r, caches in zip(ranks, rank_caches):
                r.capture_trace(caches)
            warm = _read_d2h_records(d2h, ranks[0].warmup_ack_count())
            logger.info(f"[pipeline] drained {len(warm)} warm-up D2H records")

        for f, chunks in enumerate(forwards):
            logger.info(f"[pipeline {'traced' if traced else 'eager'}] forward {f}: {chunks}")
            out = ranks[0].prefill_forward(tokens(ranks[0], chunks), rank_caches[0], chunks, d2h_service=d2h)
            act = ttnn.clone(out)  # what the D2D socket delivers: a copy of rank 0's per-chip activation
            if not traced:
                ttnn.deallocate(out)
            assert ranks[1].prefill_forward(act, rank_caches[1], chunks) is None
            want = [
                (c.slot_id, c.actual_start, c.actual_end)
                for c in chunks
                if not c.is_empty
                for _ in range(acks_per_chunk)
            ]
            got = _read_d2h_records(d2h, len(want))
            assert got == want, f"forward {f}: D2H records {got} != {want}"

        if traced:
            # Eager reference on the same two models (their host-scalar programs), slots 4..7.
            for chunks in forwards:
                shifted = [
                    c if c.is_empty else PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end)
                    for c in chunks
                ]
                for r in ranks:
                    for c in shifted:
                        if not c.is_empty and c.actual_start == 0:
                            r.model.reset_streams(c.slot_id)
                inp = tokens(ranks[0], shifted)
                h = ranks[0].model.forward_packed(inp, rank_caches[0].kvpe, shifted)
                ttnn.deallocate(inp)
                ranks[1].model.forward_packed(h, rank_caches[1].kvpe, shifted)
                ttnn.deallocate(h)
        ttnn.synchronize_device(mesh_device)

        failures = []
        for r, caches in zip(ranks, rank_caches):
            state = _rank_state(mesh_device, r, caches, range(num_slots))
            for (layer, kind), per_slot in sorted(state.items()):
                for u in range(num_users):
                    want = per_slot[u + num_users] if traced else ref_state[(layer, kind)][u]
                    got = per_slot[u]
                    diff = (got - want).abs().max().item()
                    pcc = _pcc(want, got)
                    logger.info(f"[pipeline] slot {u} {kind} layer {layer} max|diff| {diff:.3e} PCC {pcc:.6f}")
                    if traced:
                        bad = pcc < TRACED_PCC
                    elif residual == "plain" or layer < PIPELINE_SPLIT:
                        bad = diff != 0
                    else:
                        bad = pcc < PIPELINE_PCC
                    if bad:
                        failures.append(f"slot {u} {kind} layer {layer}: max|diff| {diff:.3e} PCC {pcc:.6f}")
        assert not failures, f"[pipeline] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        _release(ranks)
        d2h = None
