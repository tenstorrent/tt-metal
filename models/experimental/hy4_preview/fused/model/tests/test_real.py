# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A decode step through the REAL path in one process (PLAN.md section 19), first form: the prefill stack writes
the cache rows, its weights are freed, the fused decode weights are loaded, the fused layers read the prefill
stack's rows in place.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>              (the fused side only: the stack is not built)
    HY4_TEST_TIMEOUT=2400 TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x

Env: HY4_CHAIN_LAYERS (6; 78), HY4_CHAIN_PROMPT (1022: the prompt of HF's decode golden; a prompt below the top-k,
so the selection is every cached position and no append is needed for the steps the golden has),
HY4_REAL_USERS (8), HY4_REAL_MAX_SEQ (8192).

Flow and what is measured (wall time and DRAM per chip after each phase; lines HY4_MK11 real):
  1. the fused step's L1 state (AttnLayer, both MlpLayers) FIRST: the prefill stack's ops must find it allocated
  2. PrefillBridge.build(): the prefill stack for the same layers, its caches for USERS users of MAX_SEQ positions
  3. prefill: the golden's prompt and HF's own continuation (N + steps - 1 tokens) into every user's slot, so user
     u can decode at position N - 1 + u over rows the prefill stack wrote
  4. PrefillBridge.release_model(): the prefill weights go, the caches stay
  5. the fused decode weights of the layers are loaded
  6. one decode step for the 8 users at positions N - 1 .. (step j of the golden for user j; users beyond the
     golden's steps repeat the first ones), the selection every position up to the user's own, on the prefill
     stack's storage tensor in its own layout (slot = user x layers + layer; 128-token stripes over the chips):
     launched program by program, every status read; per layer against HF, then logits and greedy tokens.
Not in this form: the append (the rows of the decode positions are the prefill stack's), the full-indexer layers'
scan (the selection is the host's), the paged pool (ordinary slots).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.model.tests import test_chain as TC
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt import prefill_bridge as pb
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

USERS = int(os.environ.get("HY4_REAL_USERS", "8"))
# "truncate": P1 appends the new token's cache row itself (AttnLayer(append=...)): at a position the prefill stack has
# already written, the row is rewritten, and must come out as the prefill stack left it wherever the layer's input
# is the prefill stack's own (layer 0); "" (default): the rows of the decode positions stay the prefill stack's
APPEND = os.environ.get("HY4_REAL_APPEND") or None
MAX_SEQ = int(os.environ.get("HY4_REAL_MAX_SEQ", "8192"))
# "1": sparse_mla's row reader takes its addresses from fused/kv_store's frame tables (AttnLayer(reader_frames=True):
# identity tables on the prefill stack's slot-major storage)
READER_FRAMES = os.environ.get("HY4_REAL_READER_FRAMES") == "1"


def attn_kwargs():
    """The attention half's switches of this run (the latent scale is test_chain's: HY4_LATENT_SCALE)."""
    kw = {}
    if APPEND:
        kw["append"] = APPEND
    if READER_FRAMES:
        kw["reader_frames"] = True
    return kw or None


TRACE_REGION = 256 << 20
galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), TC.MESH)],
    indirect=True,
    ids=["8x4"],
)


def gb(mesh):
    return round(pb.dram(mesh)["allocated_gb"], 3)


def all_rows(mesh, kv, cfg):
    """Every stored row of the prefill stack's storage, [chips, slots, rows per slot, 576], read back through the
    stack's unpack."""
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, reconstruct_scaled_fp8_kv_cache

    ttnn.synchronize_device(mesh)
    physical = ttnn.to_torch(kv, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    geometry = MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope)
    return (
        reconstruct_scaled_fp8_kv_cache(physical, geometry)[:, 0]
        .float()
        .reshape(TC.CHIPS, -1, cfg.rows_per_slot, cfg.width)
    )


def all_fields(mesh, kv, cfg):
    """The storage's three fields as stored: (e4m3 latent values before their scales [.., 512], the 4 scales of a row,
    the RoPE part in BF16 [.., 64]), each [chips, slots, rows per slot, n]."""
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, unpack_scaled_fp8_kv_cache

    ttnn.synchronize_device(mesh)
    physical = ttnn.to_torch(kv, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    geometry = MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope)
    return tuple(
        t[:, 0].float().reshape(TC.CHIPS, -1, cfg.rows_per_slot, t.shape[-1])
        for t in unpack_scaled_fp8_kv_cache(physical, geometry)
    )


def append_parts(
    rows_before, rows_after, fields_before, fields_after, finished, cfg, layout, positions, layers, label, gold=None
):
    """Where the appended rows' gain against the prefill stack's rows comes from. Per layer, over the 8 decode rows:
    the RoPE part (BF16 in both stacks, no cast: a difference there is before the cast), the 4 scales of a row (the
    maxima of its groups before the cast), the latent part as stored, and our BF16 row BEFORE the cast (the append
    core's finished row) against our own stored row (the cast alone) and against the stack's stored row."""
    lat = cfg.latent
    for layer in range(layers):
        idx = [row_index(cfg, layout, layer, u, p) for u, p in enumerate(positions)]
        pick = lambda t: torch.stack([t[i] for i in idx])
        b, a = pick(rows_before), pick(rows_after)
        pg = lambda x, y: f"{TC.pcc(x, y):.6f}/{TC.gain(x, y):.4f}"
        sb, sa = pick(fields_before[1]), pick(fields_after[1])
        text = (
            f"stack's stored row vs ours stored (pcc/gain): RoPE part {pg(b[:, lat:], a[:, lat:])}, latent part {pg(b[:, :lat], a[:, :lat])}; "
            f"scales ours / the stack's: mean {float((sa / sb).mean()):.4f} (min {float((sa / sb).min()):.4f}, max {float((sa / sb).max()):.4f})"
        )
        if finished.get(layer) is not None:
            f = finished[layer]
            text += (
                f"; our row BEFORE the cast vs our stored row: latent {pg(f[:, :lat], a[:, :lat])}, RoPE part equal "
                f"{bool(torch.equal(f[:, lat:], a[:, lat:]))}; the stack's stored row vs our row before the cast: latent "
                f"{pg(b[:, :lat], f[:, :lat])}"
            )
        if gold is not None:  # the golden's own rows of those positions (HF's cache, in the device frame)
            h = gold(layer)
            text += (
                f"; against the golden's rows: RoPE part the stack's {pg(h[:, lat:], b[:, lat:])}, ours {pg(h[:, lat:], a[:, lat:])}; "
                f"latent part the stack's stored {pg(h[:, :lat], b[:, :lat])}, ours stored {pg(h[:, :lat], a[:, :lat])}"
                + (
                    f", ours before the cast {pg(h[:, :lat], finished[layer][:, :lat])}"
                    if finished.get(layer) is not None
                    else ""
                )
            )
        logger.info(f"HY4_MK11 {label} append parts layer {layer}: {text}")


def row_index(cfg, layout, layer, user, position):
    return layout.chip(position), cfg.slot(user, layer), layout.slot(position)


def append_check(rows_before, rows_after, cfg, layout, positions, layers, label):
    """What a step with the append did to the storage: the rows of the decode positions (rewritten by P1: each must
    be near the row the prefill stack wrote there, pcc > 0.999), their neighbours in the same shard on the owner chips
    (layers 0, 2, 5: must not have changed), and every other row of every slot, layer and chip (must not have
    changed). Returns the failures."""
    failed = []
    own = torch.zeros(rows_before.shape[:3], dtype=torch.bool)
    for layer in range(layers):
        idx = [row_index(cfg, layout, layer, u, p) for u, p in enumerate(positions)]
        for i in idx:
            own[i] = True
        b = torch.stack([rows_before[i] for i in idx])
        a = torch.stack([rows_after[i] for i in idx])
        same = sum(torch.equal(x, y) for x, y in zip(b, a))
        logger.info(
            f"HY4_MK11 {label} append layer {layer}: the rows of the 8 decode positions, rewritten by P1, against the prefill "
            f"stack's own: {same} of {len(idx)} equal, pcc {TC.pcc(b, a):.6f}, gain {TC.gain(b, a):.4f}"
        )
        # (not bit-equal, layer 0 included: "truncate" is the prefill stack's PACK bit for bit on the same BF16 row; the
        # row before the pack comes from two stacks' arithmetic. Measured 2026-11-20: pcc 0.99988 to 0.99998.)
        if TC.pcc(b, a) < 0.999:
            failed.append(f"layer {layer}: the appended rows against the prefill stack's: pcc {TC.pcc(b, a):.6f}")
    for layer in sorted({0, 2, layers - 1} & set(range(layers))):
        near = [
            row_index(cfg, layout, layer, u, q)
            for u, p in enumerate(positions)
            for q in (p - 1, p + 1)
            if q not in positions and layout.chip(q) == layout.chip(p)
        ]
        changed = sum(not torch.equal(rows_before[i], rows_after[i]) for i in near)
        logger.info(
            f"HY4_MK11 {label} append layer {layer}: the rows just below and above the decode positions in the same shard on "
            f"the owner chips ({len(near)} rows that are no decode position of this step): {changed} changed"
        )
        if changed:
            failed.append(f"layer {layer}: {changed} neighbour rows changed")
    differ = (rows_before != rows_after).any(dim=-1)  # [chips, slots, rows]
    stray = differ & ~own
    logger.info(
        f"HY4_MK11 {label} append: rows of the whole storage that changed: {int(differ.sum())} of {differ.numel()} "
        f"({int(own.sum())} are decode positions of this step); changed rows that are NOT decode positions: {int(stray.sum())} "
        f"{stray.nonzero()[:6].tolist()}"
    )
    if int(stray.sum()):
        failed.append(f"{int(stray.sum())} rows changed that the append must not touch")
    return failed


def l1_view(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    return v.total_bytes_allocated_per_bank, v.largest_contiguous_bytes_free_per_bank


def run_sanity(mesh, when):
    """A program of ours that cannot stall anything, before the first fused program: one data-movement kernel on one
    core of every chip, which counts."""
    from models.experimental.hy4_preview.fused.layer_common import transport as tp
    from models.experimental.hy4_preview.fused.model import program as MP

    words = tp.core_buffer(mesh, [MP.CANARY_CORE], 256, ttnn.uint32, width=16)
    ttnn.synchronize_device(mesh)
    before = MP.sanity_counts(words)
    Stack.launch([("sanity", *MP.sanity_program(mesh, words))])
    ttnn.synchronize_device(mesh)
    after = MP.sanity_counts(words)
    ok = sum(a[0] == b[0] + 1 and a[1] == b[1] + 1 for a, b in zip(after, before))
    logger.info(f"HY4_MK11 real sanity program {when}: ran and counted on {ok} of {len(after)} chips")
    assert ok == len(after), f"the sanity program {when}: {after}"
    ttnn.deallocate(words)


def prepare_step(chain, mesh, kv, N, steps, layout):
    """Everything a decode step of the 8 rows needs on the device beside the weights: row j decodes step j of the
    golden (position N - 1 + j) in user slot j of ``kv`` (the prefill stack's storage; the mock: random rows), its
    selection every position up to its own. Returns the storage."""
    st, n = chain.st, chain.n
    chain.cfg = cfg = SL.SparseMlaConfig(rows_per_slot=MAX_SEQ // 32, num_layers=n)
    chain.J = J = [j % steps for j in range(USERS)]
    chain.positions = [N - 1 + j for j in J]
    if is_mock():
        latent, rope = hs.random_latents(cfg.users * n, cfg.rows_per_slot, cfg, seed=3)
        packed = hs.pack_rows(mesh, latent[:, None], rope[:, None])
        kv = ttnn.to_memory_config(packed, hs.kv_memory_config(mesh, cfg))
    tables = []
    for chip in range(TC.CHIPS):
        lists = []
        for p_u in chain.positions:
            mine = [p for p in range(p_u + 1) if layout.chip(p) == chip]
            lists.append(torch.tensor([layout.slot(p) for p in mine], dtype=torch.long))
        tables.append(SL.selection_table(cfg, lists))
    shards = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tables]
    chain.sel = ttnn.to_device(ttnn.from_host_shards(shards, mesh.shape), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    st.set_cache(cfg, kv, chain.sel)
    if APPEND:
        st.a.set_positions(chain.positions)
    set_rope(chain)
    chain.reset_streams()
    ttnn.synchronize_device(mesh)
    return kv


def set_rope(chain):
    chain.st.set_rope(*TC.rope_tiles(chain.positions, chain.acfg))


@galaxy
@pytest.mark.timeout(0)
def test_real(mesh_device):
    mesh = mesh_device
    marks = {}

    def mark(name, t0):
        marks[name] = (round(time.perf_counter() - t0, 1), gb(mesh)) + l1_view(mesh)
        logger.info(
            f"HY4_MK11 real {name}: {marks[name][0]} s, DRAM per chip {marks[name][1]} GB, L1 tensors {marks[name][2]} B per core, "
            f"largest free L1 block {marks[name][3]} B"
        )

    g = TC.hf_harness.find_decode_golden(TC.LAYERS, TC.PROMPT)
    N, n, steps = g["meta"]["prompt_len"], TC.LAYERS, g["meta"]["steps"]
    layout = BlockCyclicLayout()
    tokens = [int(v) for v in g["input_ids"][: N + steps - 1]]
    bridge = chain = None
    kv = None
    try:
        mark("0 mesh open", time.perf_counter())
        if not is_mock():
            # 1. the prefill stack, with nothing of ours on the device
            t0 = time.perf_counter()
            bridge = pb.PrefillBridge(mesh, num_layers=n, num_users=USERS, max_seq_len=MAX_SEQ)
            bridge.build()
            mark("1 prefill stack built", t0)
            # 2. prefill
            t0 = time.perf_counter()
            for user in range(USERS):
                bridge.prefill(tokens, user=user)
            mark(f"2 prefill of {len(tokens)} tokens into {USERS} slots", t0)
            # 3. the swap
            t0 = time.perf_counter()
            bridge.release_model()
            mark("3 prefill weights released", t0)
            kv = bridge.kv_caches.kvpe.storage
            run_sanity(mesh, "after the release")
        # 4. the fused step's L1 state and the decode weights
        t0 = time.perf_counter()
        chain = TC.Chain(mesh, load=False, attn_kwargs=attn_kwargs())
        st = chain.st
        mark("4 fused L1 state (arena 307,200 B and the rest)", t0)
        t0 = time.perf_counter()
        chain.load_layers()
        mark("5 fused decode weights loaded", t0)
        # 6. the step on the prefill stack's storage
        kv = prepare_step(chain, mesh, kv, N, steps, layout)
        cfg = chain.cfg
        rows_before = fields_before = None
        finished = {}
        if APPEND and not is_mock():
            rows_before, fields_before = all_rows(mesh, kv, cfg), all_fields(mesh, kv, cfg)

        def keep_finished(kind, layer):
            """After a layer's P1: the BF16 rows its append core finished (before the cast), from the owner chips."""
            if not (APPEND and kind == "p1" and not is_mock()):
                return
            rows = torch.zeros(USERS, cfg.width)
            images = {}
            for u, p in enumerate(chain.positions):
                coord = next(c for c in st.topo.chips if st.topo.chip(c) == layout.chip(p))
                if coord not in images:
                    images[coord] = st.a.read_append(coord)["rows"]
                rows[u] = images[coord][u]
            finished[layer] = rows

        if is_mock():
            Stack.launch(st.step_programs(audit=True))
            return
        label = (
            f"real path, {n} layers, P{TC.PROMPT} [{TC.MOE_FIDELITY}], rows by the prefill stack"
            + (f", append {APPEND}" if APPEND else "")
            + (", reader frames" if READER_FRAMES else "")
            + (f", latent scale {TC.LATENT_SCALE}" if TC.LATENT_SCALE else "")
        )
        head, lm = TC.head_reference()
        records = []
        t0 = time.perf_counter()
        out = TC.run_step_untraced(chain, label, records, after_program=keep_finished)
        heads = [TC.check_logits(chain, head, lm, out[(n - 1, "streams_out")], label)]
        mark("6 the fused step, untraced, with the host's reads", t0)
        append_failed = []
        if APPEND:
            rows_after = all_rows(mesh, kv, cfg)
            append_failed = append_check(rows_before, rows_after, cfg, layout, chain.positions, n, label)
            gold_rows = lambda layer: torch.stack(
                [
                    frames.tail_to_device(g["cache"][layer]["latent"][p].float(), chain.acfg.qk_rope_head_dim)
                    for p in chain.positions
                ]
            )
            append_parts(
                rows_before,
                rows_after,
                fields_before,
                all_fields(mesh, kv, cfg),
                finished,
                cfg,
                layout,
                chain.positions,
                n,
                label,
                gold=gold_rows,
            )
        TC.table(records, heads, label)
        canary = st.canary_summary()
        logger.info(
            f"HY4_MK11 real canary: {canary}; phases (s, DRAM GB per chip, L1 tensors B, largest free L1 block B): {marks}; DRAM "
            f"high-water {max(v[1] for v in marks.values())} GB"
        )
        for kind, v in canary.items():
            assert v["found"] == 0 and v["timeouts"] == 0 and v["checks"] == v["runs"], f"canary {kind}: {v}"
        assert not append_failed, append_failed
    finally:
        if chain is not None and chain.sel is not None:
            ttnn.deallocate(chain.sel)
        if bridge is not None:
            bridge.close()
