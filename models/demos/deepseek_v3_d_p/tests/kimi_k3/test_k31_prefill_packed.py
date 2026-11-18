# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed (multi-chunk) prefill forwards for Kimi-K3.1, on synthetic weights.

K3.1 is K3 with Gated DSA on its full-attention layers, so a packed forward runs each DSA layer's attention once per
chunk with that chunk's own indexer: the chunk's index keys go to its slot of the index cache, and its top-k runs
over its own slot's prefix. Nothing indexer-related crosses chunks, layers or pipeline ranks: K3.1 has no
`indexer_types`, so every DSA layer owns its indexer (passes_indexer_indices is False). These tests are the K3.1
counterparts of `test_k3_prefill_multiuser.py`, with the index cache compared beside the KV cache and the KDA carries.

* test_k31_packed_forward_matches_one_chunk_forwards: the K3 packed schedule (partial chunks, one user's consecutive
  chunks in one forward, users at different depths, empty tail chunks) against each chunk alone in its own forward on
  the same model, then against a model built for one chunk per forward. Plain residual stream: bit-exact output, KV,
  index cache and KDA carries. AttnRes: PCC bars (it rounds a row by its position in the call; see the K3 test).
* test_k31_packed_traced_matches_eager: the same schedule through TtKimiK3Runtime.prefill_forward with use_trace
  (one capture, replayed per forward, every chunk position's slot and bounds read on device, an empty position in
  the scratch slot) must leave the caches and carries the eager packed forward leaves, bit for bit.
* test_k31_packed_pipeline_matches_one_rank: the K3.1 runtime split over two pipeline ranks on one mesh (layers 0-11
  and 12-15, two DSA layers each side), packed activations handed rank to rank and D2H acks from rank 0, must match
  one rank (plain residual, eager: bit-exact), and traced must match eager on both ranks (bit-exact). No indexer
  indices cross the boundary.

Synthetic weights and the dense-FFN stand-in for the MoE (synthetic.py): the MoE is K3's, covered by the K3 test.
"""

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.prefill_chunk import (
    PrefillChunk,
    packed_forward_capabilities,
    packed_forward_refusals,
    validate_chunks,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import read_slab_layer
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import (
    CACHE_LEN,
    CHUNK,
    DENSE_FFN_CONFIG,
    SP_AXIS,
    TP_AXIS,
    build_stack,
    hf_config,
    placements,
    release,
    run_params,
    synthetic_state_dict,
    token_ids,
    window,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_k3_prefill_multiuser import (
    _PlainK3Transformer,
    _read_d2h_records,
)
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import deallocate_state
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import PlainResidualStream
from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntimeConfig
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.test_utils import gather_cache_natural
from tt_py_test_utils_common.utils_for_testing import comp_pcc

# Layers 0-7: KDA 0-2 and 4-6, DSA 3 and 7 (two DSA layers, so the index cache has two layer slots per user).
NUM_LAYERS = 8
FORWARD_CHUNKS = 2
# The K3 test's bars, which K3.1 shares: AttnRes rounds a row by its position in the call (packed output >= 0.995,
# state >= 0.998); a model built for one chunk per forward runs every matmul on fewer rows (>= 0.99).
PACKED_OUTPUT_PCC = 0.995
PACKED_STATE_PCC = 0.998
SINGLE_MODEL_OUTPUT_PCC = 0.99
DISTINCT_USERS_MAX_PCC = 0.99
TRACE_REGION_SIZE = 256 * 1024 * 1024
# A pipeline split lands on an AttnRes block edge (a multiple of 12); layers 0-11 and 12-15 hold DSA layers 3, 7, 11
# and 15.
PIPELINE_LAYERS = 16
PIPELINE_SPLIT = 12


def _schedule():
    """(user lengths, forwards): the K3 packed schedule at K3.1's synthetic chunk. User u lives in slot u. Starts are
    chunk-aligned (no rotated token layout) and 32-aligned (KDA); ends are not."""
    C = CHUNK
    lengths = [3 * C, C + 704, 896, 2 * C]
    E = PrefillChunk(0, 0, 0)
    forwards = [
        [PrefillChunk(0, 0, C), PrefillChunk(1, 0, C)],
        [PrefillChunk(0, C, 2 * C), E],
        [PrefillChunk(3, 0, C), PrefillChunk(3, C, 2 * C)],
        [PrefillChunk(2, 0, 896), PrefillChunk(0, 2 * C, 3 * C)],
        [PrefillChunk(1, C, C + 704), E],
    ]
    return lengths, forwards


def _streams(num_users, length):
    return [torch.tensor(token_ids(length, seed=100 + u)) for u in range(num_users)]


def _chunk_tokens(chunks, streams, user_of):
    """Each chunk's natural-order tokens, a partial or empty chunk padded with token 0 to CHUNK."""
    out = []
    for c in chunks:
        real = [] if c.is_empty else streams[user_of(c.slot_id)][c.actual_start : c.actual_end].tolist()
        out.append(real + [0] * (CHUNK - len(real)))
    return out


def _upload(mesh_device, chunks, streams, user_of):
    """A forward's tokens in the packed per-chip layout (every start here is chunk-aligned, so no rotation): chip c's
    rows are slice c of every chunk, in chunk order. SP-sharded on mesh axis 0."""
    sp = mesh_device.shape[SP_AXIS]
    per_chip = [[] for _ in range(sp)]
    for tokens in _chunk_tokens(chunks, streams, user_of):
        s = CHUNK // sp
        for c in range(sp):
            per_chip[c] += tokens[c * s : (c + 1) * s]
    return ttnn.from_torch(
        torch.tensor(per_chip, dtype=torch.int64).reshape(sp, 1, -1),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )


def _host(mesh_device, out):
    """A `[1, 1, n * CHUNK / sp, emb / tp]` output as host `[n * CHUNK, emb]`, chip-major rows."""
    host = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
    ).to(torch.float32)
    return host[0, 0]


def _chunk_rows(hidden, k, n, chunk, sp):
    """Chunk k's real rows out of a forward's chip-major output (chunk-aligned start: chip c holds positions
    [c * s, (c + 1) * s) of the chunk)."""
    s = CHUNK // sp
    rows = []
    for c in range(sp):
        for r in range(s):
            if chunk.actual_start + c * s + r < chunk.actual_end:
                rows.append(hidden[(c * n + k) * s + r])
    return torch.stack(rows)


def _forward(model, caches, mesh_device, chunks, streams, user_of=lambda s: s):
    """One forward of ``chunks`` on a stack; host output. One chunk on a one-chunk model takes the regular path."""
    tokens = _upload(mesh_device, chunks, streams, user_of)
    if len(chunks) == 1 and model.forward_chunks == 1:
        c = chunks[0]
        out = model.forward(
            tokens,
            caches.kvpe,
            index_kv_cache=caches.index,
            actual_isl=CHUNK,
            actual_start=c.actual_start,
            actual_end=c.actual_end,
            cache_user_id=c.slot_id,
        )
    else:
        out = model.forward(tokens, caches.kvpe, index_kv_cache=caches.index, actual_isl=CHUNK, chunks=chunks)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(tokens)
    if out is None:
        return None
    host = _host(mesh_device, out)
    ttnn.deallocate(out)
    return host


def _pcc(a, b):
    return float(comp_pcc(a, b)[1])


def _state(mesh_device, schedule, first_layer_idx, num_layers, caches, slots):
    """{(global layer, kind): [per-slot host tensor]}: each KDA layer's two carries, each DSA layer's KV and index
    keys. Cache layer slots are user-major (slot * num_mla_layers + rank-local MLA index)."""
    state = {}
    for layer in schedule.rank_kda_layer_ids():
        per = []
        for s in slots:
            carry = read_slab_layer(mesh_device, caches.kda_states, s, layer)
            try:
                per.append(tuple(_flat(t) for t in (carry.recurrent, carry.convolution)))
            finally:
                deallocate_state(carry)
        state[(layer, "kda_recurrent")] = [p[0] for p in per]
        state[(layer, "kda_convolution")] = [p[1] for p in per]
    n_mla = schedule.num_mla_layers
    mla = [l for l in schedule.mla_layer_ids if first_layer_idx <= l < first_layer_idx + num_layers]
    assert len(mla) == n_mla, (mla, n_mla)
    for kind, cache in (("kvpe", caches.kvpe), ("index", caches.index)):
        rows, _ = gather_cache_natural(cache, mesh_device, tp_shard_kv=True, sp_axis=SP_AXIS)
        for local, layer in enumerate(mla):
            state[(layer, kind)] = [rows[s * n_mla + local] for s in slots]
    return state


def _flat(tensor):
    return torch.cat([ttnn.to_torch(t).flatten().to(torch.float32) for t in ttnn.get_device_tensors(tensor)])


def _compare(label, state, pairs, *, exact, threshold=PACKED_STATE_PCC):
    """Every (reference slot index, slot index under test) pair of every state entry. Returns the failures."""
    failures = []
    for (layer, kind), per_slot in sorted(state.items()):
        for ref, got in pairs:
            want, have = per_slot[ref], per_slot[got]
            diff = (want - have).abs().max().item()
            pcc = _pcc(want, have)
            logger.info(f"[{label}] {kind} layer {layer} slot {got} vs {ref}: max|diff| {diff:.3e} PCC {pcc:.6f}")
            if (exact and diff != 0) or (not exact and pcc < threshold):
                failures.append(f"{kind} layer {layer} slot {got} vs {ref}: max|diff| {diff:.3e} PCC {pcc:.6f}")
    return failures


def _plain(hidden, block_residual=None):
    return PlainResidualStream(hidden)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", placements(), indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("residual", ["plain", "attnres"])
def test_k31_packed_forward_matches_one_chunk_forwards(mesh_device, device_params, residual):
    """Packed forwards (slots 0..3) against each chunk alone in its own packed forward (slots 4..7), then against a
    model built for one chunk per forward. Plain residual: bit-exact. AttnRes: PCC bars."""
    exact = residual == "plain"
    factory = _plain if exact else None
    lengths, forwards = _schedule()
    num_users = len(lengths)
    num_slots = 2 * num_users
    sp = mesh_device.shape[SP_AXIS]
    for chunks in forwards:
        validate_chunks(
            chunks, chunk_size=CHUNK, forward_chunks=FORWARD_CHUNKS, num_users=num_slots, max_seq_len=CACHE_LEN
        )
    mesh_device.enable_program_cache()
    state_dict = synthetic_state_dict(0, NUM_LAYERS)
    streams = _streams(num_users, max(lengths))
    user_of = lambda s: s % num_users
    empty = PrefillChunk(0, 0, 0)

    model, caches = build_stack(
        mesh_device,
        state_dict,
        0,
        NUM_LAYERS,
        num_users=num_slots,
        forward_chunks=FORWARD_CHUNKS,
        residual_factory=factory,
    )
    failures, alone = [], {}
    try:
        assert caches.index is not None, "K3.1 allocates a DSA index cache"
        packed = {}
        for f, chunks in enumerate(forwards):
            logger.info(f"[packed] forward {f}: {chunks}")
            hidden = _forward(model, caches, mesh_device, chunks, streams, user_of)
            for k, c in enumerate(chunks):
                if not c.is_empty:
                    packed[(c.slot_id, c.actual_start)] = _chunk_rows(hidden, k, FORWARD_CHUNKS, c, sp)

        for chunks in forwards:
            for c in chunks:
                if c.is_empty:
                    continue
                ref_chunk = PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end)
                hidden = _forward(
                    model, caches, mesh_device, [ref_chunk] + [empty] * (FORWARD_CHUNKS - 1), streams, user_of
                )
                ref = _chunk_rows(hidden, 0, FORWARD_CHUNKS, ref_chunk, sp)
                alone[(c.slot_id, c.actual_start)] = ref
                got = packed[(c.slot_id, c.actual_start)]
                diff, pcc = (ref - got).abs().max().item(), _pcc(ref, got)
                logger.info(f"[packed] {residual} {c} output PCC {pcc:.6f} max|diff| {diff:.3e}")
                if exact and diff != 0:
                    failures.append(f"{c} output not bit-exact (max|diff| {diff:.3e})")
                elif pcc < PACKED_OUTPUT_PCC:
                    failures.append(f"{c} output PCC {pcc:.6f} < {PACKED_OUTPUT_PCC}")

        state = _state(mesh_device, model.schedule, 0, NUM_LAYERS, caches, range(num_slots))
        failures += _compare(
            f"packed-{residual}", state, [(u + num_users, u) for u in range(num_users)], exact=exact
        )
        # Two users' index keys must differ: catches a comparison that reads one slot twice.
        index_layer = max(l for l, kind in state if kind == "index")
        cross = _pcc(state[(index_layer, "index")][0], state[(index_layer, "index")][3])
        assert cross < DISTINCT_USERS_MAX_PCC, f"users A and D hold the same index keys (PCC {cross:.6f})"
    finally:
        release(model, caches)
    del model, caches
    gc.collect()

    single, single_caches = build_stack(
        mesh_device, state_dict, 0, NUM_LAYERS, num_users=num_users, forward_chunks=1, residual_factory=factory
    )
    try:
        for chunks in forwards:
            for c in chunks:
                if c.is_empty:
                    continue
                hidden = _forward(single, single_caches, mesh_device, [c], streams, user_of)
                ref = _chunk_rows(hidden, 0, 1, c, sp)
                pcc = _pcc(ref, alone[(c.slot_id, c.actual_start)])
                logger.info(f"[single] {residual} {c} output PCC {pcc:.6f}")
                if pcc < SINGLE_MODEL_OUTPUT_PCC:
                    failures.append(f"{c} output PCC {pcc:.6f} vs forward_chunks=1 < {SINGLE_MODEL_OUTPUT_PCC}")
    finally:
        release(single, single_caches)
    assert not failures, f"[packed] {len(failures)} mismatch(es):\n" + "\n".join(failures)


class _PlainK31Runtime(TtKimiK3Runtime):
    """K3.1's runtime on the plain residual stream: the handoff is the live sum alone."""

    MODEL_CLS = _PlainK3Transformer

    @property
    def activation_planes(self) -> int:
        return 1


def _rt_config(mesh_device, device_params, first, n, total, num_users, use_trace):
    return TtPrefillRuntimeConfig(
        num_layers=n,
        max_seq_len=CACHE_LEN,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        chunk_size=CHUNK,
        forward_tokens_budget=FORWARD_CHUNKS * CHUNK,
        num_users=num_users,
        num_links=1,
        topology=per_axis_topology(device_params["fabric_config"]),
        capacity_factor=2,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        model_cfg=DENSE_FFN_CONFIG,
        routing_use_l1_small_for_semaphores=True,
        weight_cache_path=None,
        kv_only_last_layer=False,
        use_trace=use_trace,
        tp_shard_kv=True,
        sparse_kv_cache_format=MlaKvCacheFormat.SCALED_FP8,
        first_layer_idx=first,
        is_first_rank=first == 0,
        is_last_rank=first + n == total,
    )


def _runtime(mesh_device, device_params, state_dict, first, n, total, num_users, use_trace, runtime_cls=TtKimiK3Runtime):
    """A K3.1 runtime over layers [first, first + n) of a ``total``-layer synthetic model, and its caches from the
    serving allocator (the index and KV caches with the traced scratch slot, the KDA slabs at num_users)."""
    rt = runtime_cls(
        mesh_device,
        hf_config(),
        window(state_dict, first, n),
        _rt_config(mesh_device, device_params, first, n, total, num_users, use_trace),
    )
    params = run_params(
        mesh_device,
        first,
        n,
        is_first_rank=first == 0,
        is_last_rank=first + n == total,
        num_users=num_users,
        forward_chunks=FORWARD_CHUNKS,
        use_trace=use_trace,
    )
    caches = KimiK31Adapter().allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config(), params=params)
    assert params.num_users + (1 if use_trace else 0) == rt.config.cache_num_users
    return rt, caches


def _release_runtimes(pairs):
    """Every runtime's trace before any sub-device manager (the models share one mesh here), then the slabs."""
    for rt, _ in pairs:
        if rt._controller is not None:
            rt._controller.release()
    for rt, caches in pairs:
        rt.release_trace()  # also the model's sub-device managers
        if caches.kda_states is not None:
            caches.kda_states.deallocate()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    placements(trace_region_size=TRACE_REGION_SIZE),
    indirect=["mesh_device", "device_params"],
)
def test_k31_packed_traced_matches_eager(mesh_device, device_params):
    """Traced packed forwards (slots 0..3) must leave the KV, index cache and KDA carries the eager packed forward of
    the same chunks on the same model leaves (slots 4..7), bit for bit."""
    lengths, forwards = _schedule()
    num_users = len(lengths)
    num_slots = 2 * num_users
    mesh_device.enable_program_cache()
    state_dict = synthetic_state_dict(0, NUM_LAYERS)
    streams = _streams(num_users, max(lengths))
    rt, caches = _runtime(mesh_device, device_params, state_dict, 0, NUM_LAYERS, NUM_LAYERS, num_slots, True)
    try:
        model = rt.model
        rt.compile(caches)
        rt.capture_trace(caches)

        def forward_input(chunks, user_of):
            return rt.make_forward_input(_chunk_tokens(chunks, streams, user_of), [c.actual_start for c in chunks])

        for f, chunks in enumerate(forwards):
            logger.info(f"[traced] forward {f}: {chunks}")
            rt.prefill_forward(forward_input(chunks, lambda s: s), caches, chunks)
        ttnn.synchronize_device(mesh_device)

        for chunks in forwards:
            ref = [c if c.is_empty else PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end) for c in chunks]
            inp = forward_input(ref, lambda s: s - num_users)
            out = model.forward(inp, caches.kvpe, index_kv_cache=caches.index, actual_isl=CHUNK, chunks=ref)
            ttnn.deallocate(inp)
            if out is not None:
                ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        state = _state(mesh_device, model.schedule, 0, NUM_LAYERS, caches, range(num_slots))
        failures = _compare("traced", state, [(u + num_users, u) for u in range(num_users)], exact=True)
        assert not failures, f"[traced] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        _release_runtimes([(rt, caches)])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    placements(trace_region_size=TRACE_REGION_SIZE),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("traced, residual", [(False, "plain"), (True, "attnres")], ids=["eager-plain", "traced"])
def test_k31_packed_pipeline_matches_one_rank(mesh_device, device_params, traced, residual):
    """Packed forwards over two pipeline ranks (layers 0-11, 12-15) with D2H acks from rank 0.

    Eager on the plain residual stream: every layer's KV, index keys and KDA carries on both ranks bit-exact against
    one rank holding all 16 layers (run first and released before the ranks are built). Traced (AttnRes): the ranks'
    replays into slots 0..3 bit-exact against the same two models run eagerly into slots 4..7. The ranks pass no
    indexer indices: each DSA layer runs its own indexer, so the runner's PP split needs no
    supports_packed_forward_indexer_indices."""
    lengths, forwards = _schedule()
    num_users = len(lengths)
    num_slots = num_users * (2 if traced else 1)
    mesh_device.enable_program_cache()
    state_dict = synthetic_state_dict(0, PIPELINE_LAYERS)
    streams = _streams(num_users, max(lengths))
    runtime_cls = _PlainK31Runtime if residual == "plain" else TtKimiK3Runtime
    user_of = lambda s: s % num_users

    adapter = KimiK31Adapter()
    assert not adapter.passes_indexer_indices
    assert (
        packed_forward_refusals(
            FORWARD_CHUNKS, indexer_indices=adapter.passes_indexer_indices, **packed_forward_capabilities(adapter)
        )
        == []
    )

    def tokens(rt, chunks):
        return rt.make_forward_input(_chunk_tokens(chunks, streams, user_of), [c.actual_start for c in chunks])

    ref_state = None
    if not traced:
        ref = _runtime(
            mesh_device, device_params, state_dict, 0, PIPELINE_LAYERS, PIPELINE_LAYERS, num_slots, False, runtime_cls
        )
        try:
            ref[0].compile(ref[1])
            for chunks in forwards:
                ref[0].prefill_forward(tokens(ref[0], chunks), ref[1], chunks)
            ttnn.synchronize_device(mesh_device)
            ref_state = _state(mesh_device, ref[0].model.schedule, 0, PIPELINE_LAYERS, ref[1], range(num_users))
        finally:
            _release_runtimes([ref])
        del ref
        gc.collect()

    split = [(0, PIPELINE_SPLIT), (PIPELINE_SPLIT, PIPELINE_LAYERS - PIPELINE_SPLIT)]
    ranks = [
        _runtime(mesh_device, device_params, state_dict, first, n, PIPELINE_LAYERS, num_slots, traced, runtime_cls)
        for first, n in split
    ]
    d2h = ttnn.D2HStreamService(
        mesh_device,
        global_spec=None,
        fifo_size_bytes=16 * 1024,
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=12,
        metadata_ring=True,
    )
    try:
        for rt, caches in ranks:
            assert not rt.forwards_indexer_indices, "a K3.1 rank passes no indexer indices"
            rt.compile(caches)
        acks_per_chunk = ranks[0][0]._packed_ack_count()
        if traced:
            ranks[0][0].set_d2h_ack_service(d2h)
            for rt, caches in ranks:
                rt.capture_trace(caches)
            warm = _read_d2h_records(d2h, ranks[0][0].warmup_ack_count())
            logger.info(f"[pipeline] drained {len(warm)} warm-up D2H records")

        (rt0, caches0), (rt1, caches1) = ranks
        for f, chunks in enumerate(forwards):
            logger.info(f"[pipeline {'traced' if traced else 'eager'}] forward {f}: {chunks}")
            out = rt0.prefill_forward(tokens(rt0, chunks), caches0, chunks, d2h_service=d2h)
            act = ttnn.clone(out)  # what the D2D socket delivers
            if not traced:
                ttnn.deallocate(out)
            assert rt1.prefill_forward(act, caches1, chunks) is None
            want = [
                (c.slot_id, c.actual_start, c.actual_end) for c in chunks if not c.is_empty for _ in range(acks_per_chunk)
            ]
            got = _read_d2h_records(d2h, len(want))
            assert got == want, f"forward {f}: D2H records {got} != {want}"

        if traced:
            for chunks in forwards:
                shifted = [
                    c if c.is_empty else PrefillChunk(c.slot_id + num_users, c.actual_start, c.actual_end)
                    for c in chunks
                ]
                inp = tokens(rt0, shifted)
                h = rt0.model.forward_packed(inp, caches0.kvpe, shifted, index_kv_cache=caches0.index)
                ttnn.deallocate(inp)
                out = rt1.model.forward_packed(h, caches1.kvpe, shifted, index_kv_cache=caches1.index)
                ttnn.deallocate(h)
                if out is not None:
                    ttnn.deallocate(out)
        ttnn.synchronize_device(mesh_device)

        failures = []
        for (rt, caches), (first, n) in zip(ranks, split):
            state = _state(mesh_device, rt.model.schedule, first, n, caches, range(num_slots))
            if traced:
                failures += _compare(
                    f"pipeline-traced rank@{first}", state, [(u + num_users, u) for u in range(num_users)], exact=True
                )
                continue
            for key, per_slot in sorted(state.items()):
                for u in range(num_users):
                    want, got = ref_state[key][u], per_slot[u]
                    diff = (want - got).abs().max().item()
                    logger.info(f"[pipeline] {key} slot {u}: max|diff| {diff:.3e}")
                    if diff != 0:
                        failures.append(f"{key} slot {u}: max|diff| {diff:.3e} PCC {_pcc(want, got):.6f}")
        assert not failures, f"[pipeline] {len(failures)} mismatch(es):\n" + "\n".join(failures)
    finally:
        _release_runtimes(ranks)
        d2h = None
