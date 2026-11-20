# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conversations of DIFFERENT lengths, one to three frames of 8192 tokens each, in fused/kv_store's paged pool, served
by the traced fused step beside the prefill stack's trace, with requests arriving while the rows decode (PLAN.md
section 41). The form is tests/test_coexist.py::test_coexist_trace with the fused ends (HY4_COEXIST_FORM=fused: ids in
DRAM -> IN -> the layers -> HEAD -> ids; nothing of ours before the prefill capture), the pool, the append "truncate",
reader_frames and the restore program; what is new here is data:

  the mix        row u serves a conversation of its own length (HY4_POOL_MIX: the AgentX contexts x 1 / 22, the longest cut to 16,500 tokens: three frames), each a
                 6-layer golden of its own (prefixes of one text, HF's half-split pairing: the prefill stack's);
                 stripe rows beyond the first frame reach the readers and the append through the frame tables
  the selection  a context above the top-k reads by the golden's picks: one selection table per full-indexer layer
                 (layers 0, 1, 5 of 6), rewritten in place when a row's conversation changes
  a request      engine.begin of a NEW conversation while all 8 rows decode: after every prefill pass of it (a replay
                 into the workbench) one replay of the restore trace and DECODES decode replays, every row against the
                 run before the request, BIT FOR BIT (AttnLayer(fixed_order=True): the same step on the same rows);
                 then the conversation takes a decode row (one call: the tables and the restore image), and the one
                 that left can be evicted by the next request that needs its frames; the rows that kept their
                 conversation are compared bit for bit again, the row that changed against its golden only
  the requests   1: a 2-frame conversation into free frames; 2: a 3-frame one the pool has no room for: the store
                 evicts the least recently used conversation without a row; 3: a 2-frame one that does not fit even
                 so: begin returns None and nothing changes (it waits); 4: a 1-frame one (evicts); 5: the waiting one
                 again, which now fits. No row is ever idle: a row keeps its conversation until the next one is in
                 the pool, and a conversation's frames are reused only after its row was given another table.

    HY4_MOCK=1 HY4_CHAIN_PROMPT=2073 pytest <this file>
    HY4_CHAIN_PROMPT=2073 HY4_TEST_TIMEOUT=2400 TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.fused.kv_store.manager import KvStore
from models.experimental.hy4_preview.fused.model import reserve as RS
from models.experimental.hy4_preview.fused.model.program import is_mock
from models.experimental.hy4_preview.fused.model.tests import test_chain as TC
from models.experimental.hy4_preview.fused.model.tests import test_coexist as CX
from models.experimental.hy4_preview.fused.model.tests import test_real as TR
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MIX = [int(v) for v in os.environ.get("HY4_POOL_MIX", "2073,3240,4213,5258,6546,7976,9778,16500").split(",")]
POOL_FRAMES = int(os.environ.get("HY4_POOL_FRAMES", "13"))
MAX_SEQ = int(os.environ.get("HY4_POOL_MAX_SEQ", "24576"))  # the workbench: three frames
DECODES = int(os.environ.get("HY4_POOL_DECODES", "4"))  # decode replays back to back after a prefill pass
ROUNDS = int(os.environ.get("HY4_POOL_ROUNDS", "5"))  # requests of the script below
# 1: stop after decode replay A and its checks (the stack with a 3-frame workbench beside our L1 state, the conversations
# of 1 to 3 frames, the per-layer selections, rows of the 2nd and 3rd frame read and appended); 2: the requests too
STAGE = int(os.environ.get("HY4_POOL_STAGE", "1"))
REPEATS = int(
    os.environ.get("HY4_POOL_REPEATS", "0")
)  # single decode replays after replay A, each row against replay A's
MIN_PCC = float(os.environ.get("HY4_POOL_MIN_PCC", "0.9"))  # a row's logits against its golden's (a measurement here)
# "0": a row that kept its conversation while ANOTHER row changed its own is only logged against the run before (the
# MoE's union of experts changed); default: asserted bit-equal at the end of the test, every violation listed
STRICT_OTHERS = os.environ.get("HY4_POOL_STRICT_OTHERS", "1") != "0"
USERS = 8
TOPK = 2048


class Conv:
    """A conversation: its golden (prompt length N), the tokens a request prefills (the prompt and the golden's
    continuation) and the golden step its decode row replays."""

    def __init__(self, key, golden, step):
        self.key, self.g, self.step = key, golden, step
        self.N, self.steps = golden["meta"]["prompt_len"], golden["meta"]["steps"]
        self.tokens = [int(v) for v in golden["input_ids"][: self.N + self.steps - 1]]
        self.position = self.N - 1 + step

    def picks(self, layer):
        """The positions this row reads on ``layer``: the reference indexer's picks of the step (every position while
        the context is below the top-k), ascending."""
        chosen = sorted(
            {
                int(v)
                for v in self.g["layers"][layer]["topk_indices"][self.step].tolist()
                if 0 <= int(v) <= self.position
            }
        )
        assert len(chosen) == min(self.position + 1, TOPK), (self.key, layer, len(chosen), self.position)
        return chosen


def full_layers(n):
    """The checkpoint's full-indexer layers below n (0, 1, then every fourth): each has a selection of its own, the
    shared layers behind it read by its picks."""
    return [layer for layer in range(n) if layer < 2 or layer % 4 == 1]


def selection_tables(cfg, convs, layer):
    """Per chip, sparse_mla's selection table [1 + users, top-k] (row 0 the counts, row 1 + u the stripe rows) of the 8
    rows' picks on ``layer``. sparse_mla/layout.py selection_table without its bound on the stripe row: a row of the
    second or third frame of a conversation is addressed through the row's frame table (kv_frames::locate)."""
    picks = [conv.picks(layer) for conv in convs]
    out = []
    for chip in range(TC.CHIPS):
        table = torch.zeros(1 + cfg.users, cfg.topk, dtype=torch.int32)
        for u, chosen in enumerate(picks):
            rows = [fr.stripe_row(p) for p in chosen if fr.chip_of(p) == chip]
            assert len(rows) <= cfg.topk and all(0 <= r < fr.TABLE_FRAMES * fr.FRAME_ROWS for r in rows)
            table[0, u] = len(rows)
            table[1 + u, : len(rows)] = torch.tensor(rows, dtype=torch.int32)
        out.append(table)
    return out, [len(c) for c in picks]


def host_tables(tables, mesh):
    shards = [ttnn.from_torch(t[None, None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT) for t in tables]
    return ttnn.from_host_shards(shards, mesh.shape)


class Hooked:
    """The engine's prefill callable with a call after every pass (what the decode rows do between two passes)."""

    def __init__(self, inner):
        self.inner, self.after = inner, None

    def __call__(self, slot, start, end, ids):
        result = self.inner(slot, start, end, ids)
        if self.after is not None:
            self.after(slot, start, end, result)
        return result


@TR.galaxy
@pytest.mark.timeout(0)
def test_pool_mix(mesh_device, monkeypatch):
    from models.experimental.hy4_preview.fused.common import tracing
    from models.experimental.hy4_preview.fused.model import ends

    mesh, n = mesh_device, TC.LAYERS
    goldens = {N: hf_harness.find_decode_golden(n, N) for N in sorted(set(MIX))}
    missing = [N for N, g in goldens.items() if g is None]
    if missing:
        pytest.skip(
            f"no {n}-layer decode golden of {missing} tokens under {hf_harness.GOLDEN_ROOT} (mk11_scripts/mk11_goldens_pool.sh)"
        )
    assert len(MIX) == USERS
    steps = goldens[MIX[0]]["meta"]["steps"]
    convs = {
        u: Conv(u, goldens[N], u % steps) for u, N in enumerate(MIX)
    }  # key -> Conv; row u starts on conversation u
    rows = [convs[u] for u in range(USERS)]  # the conversation of each decode row
    frames_of = lambda conv: fr.frames_for(len(conv.tokens) + conv.steps)
    assert all(frames_of(c) * fr.FRAME_TOKENS <= MAX_SEQ for c in rows), "a conversation is longer than the workbench"
    from models.experimental.hy4_preview.reference.attention import rope_pairing

    label = f"pool mix, {n} layers, contexts {MIX}, pool of {POOL_FRAMES} frames, workbench {MAX_SEQ} tokens"
    # the prefill stack writes rows with HF's half-split pairing: this run is the OLD reference end to end (goldens,
    # weights' row order, RoPE tiles): a regression and a test of the mechanics, not accuracy
    if rope_pairing() != "half_split":
        pytest.skip("the prefill stack's rows are half-split: this test runs without HY4_ROPE_PAIRING")
    logger.info(
        f"HY4_MK11 {label}: stage {STAGE}; RoPE pairing {rope_pairing()} (the prefill stack's; goldens under {hf_harness.GOLDEN_ROOT}: "
        f"the OLD reference, so 'against its golden' below is a regression figure)"
    )
    marks = {}
    mark = CX.marker(mesh, marks)
    b = CX.Both()
    b.trace_id = b.restore_trace = b.restore_program = None
    reports = {}
    kind = fr.Kind("latent", n, fr.LATENT.width, fr.LATENT.shard_bytes, fr.LATENT.row_stride)
    pool_g = fr.Geometry.pool(kind)
    full = full_layers(n)

    def rep(text):
        prev = list(reports.values())[-1] if reports else None
        reports[text] = RS.report(mesh, f"pool: {text}", prev=prev)

    try:
        rep("mesh open")
        banks = RS.dram_view(mesh)["banks"]
        store = engine = None
        if not is_mock():
            # a. the prefill stack in trace mode with a workbench of MAX_SEQ tokens, the store's engine made between its
            # cache allocation and its compile (as tests/test_coexist.py build_both)
            import models.demos.common.prefill.adapter as AD
            from models.experimental.hy4_preview.fused.kv_store.engine import Engine, StackPrefill

            params = AD.PrefillRunParams
            monkeypatch.setattr(AD, "PrefillRunParams", lambda **kw: params(**dict(kw, use_trace=True)))
            adapter = AD.get_adapter("hy4_preview")
            allocate = adapter.allocate_kv_cache
            made = {}

            def allocate_with_engine(**kw):
                caches = allocate(**kw)
                made["store"] = KvStore(POOL_FRAMES, rows=USERS)
                made["engine"] = Engine(mesh, made["store"], caches.kvpe.storage, caches.index, None, bench_users=USERS)
                return caches

            monkeypatch.setattr(adapter, "allocate_kv_cache", allocate_with_engine)
            t0 = time.perf_counter()
            bridge = b.bridge = pb.PrefillBridge(mesh, num_layers=n, num_users=USERS, max_seq_len=MAX_SEQ)
            bridge.build()
            monkeypatch.undo()
            store, engine = made["store"], made["engine"]
            mark("a prefill stack built in trace mode, warmed", t0)
            t0 = time.perf_counter()
            engine.warm()
            mark(f"a2 the store's engine warmed (pool of {POOL_FRAMES} frames)", t0)
        else:
            store = KvStore(POOL_FRAMES, rows=USERS)
        # b. the reserve
        reserve = b.reserve = RS.DramReserve(mesh, CX.RESERVE // banks)
        hooked = None
        if not is_mock():
            # c. the prefill trace; d. the 8 requests through the engine, each against a slot prefilled alone
            t0 = time.perf_counter()
            runtime, caches = bridge.runtime, bridge.kv_caches
            runtime.capture_trace(caches)
            ttnn.synchronize_device(mesh)
            mark(
                f"c prefill trace captured ({runtime._controller.num_segments} segments, {runtime._controller.trace_bytes()} B)",
                t0,
            )
            hooked = engine.prefill = Hooked(StackPrefill(runtime, caches, mesh, pb.MESH, trace=True))
            t0 = time.perf_counter()
            recs = [engine.begin(u, rows[u].tokens, max_new=steps) for u in range(USERS)]
            assert all(r is not None for r in recs), "the pool has no room for the 8 conversations"
            for u in range(USERS):
                store.bind(u, u)
            engine.sync_tables()
            mark(f"d {USERS} requests of {[len(c.tokens) for c in rows]} tokens through the engine", t0)
            cmp = {}
            for u in range(USERS):
                engine.reference(u, 1)
                cmp[u] = engine.compare(u, 1)
            bad = {u: {k: v for k, v in c.items() if isinstance(v, dict) and v.get("bad")} for u, c in cmp.items()}
            logger.info(
                f"HY4_MK11 {label} d: requests (frames, passes, copy-out ms) {[(r['frames'], len(r['passes']), round(1e3 * r['copy_out_s'], 1)) for r in recs]}; "
                f"pass ms {[round(1e3 * p['chunk_s'], 1) for r in recs for p in r['passes']]}; frames of the conversations "
                f"{[list(store.conversation(u).frames) for u in range(USERS)]}; each conversation's pool frames against slot 1 prefilled "
                f"alone with the same passes (the workbench held a longer or shorter conversation before): "
                f"{ {u: {k: v for k, v in c.items() if k != 'bad'} for u, c in cmp.items()} }; conversations with rows that differ: "
                f"{ {u: v for u, v in bad.items() if v} }"
            )
            assert not any(bad.values()), bad
            kv = engine.pool[engine.latent]
            b.base = {"bench": CX.storage(mesh, bridge)["kvpe"]}
            TR.run_sanity(mesh, "after the capture and the requests, the prefill model built")
        else:
            for u in range(USERS):
                assert store.begin(u, len(rows[u].tokens) + steps) is not None
                store.bind(u, u)
        rep("prefill stack built, captured, pool filled")
        base_report = reports["prefill stack built, captured, pool filled"]

        # e. ours, inside the reserve
        t0 = time.perf_counter()
        before = RS.buffers(mesh, "DRAM")
        reserve.plug()
        reserve.open()
        chain = b.chain = TC.Chain(
            mesh,
            load=False,
            attn_kwargs={"append": "truncate", "reader_frames": True},
            head_site=True,
            wait_limit=CX.FUSED_WAIT,
        )
        st = chain.st
        mark("e1 fused L1 state", t0)
        t0 = time.perf_counter()
        chain.load_layers()
        # (a frame is a slot of 256 stripe rows for the readers: the pool's geometry; the rows of a conversation's later
        # frames are reached through its table)
        chain.cfg = cfg = SL.SparseMlaConfig(rows_per_slot=fr.FRAME_ROWS, num_layers=n)
        if is_mock():
            latent, rope = hs.random_latents(cfg.users * n, cfg.rows_per_slot, cfg, seed=3)
            packed = hs.pack_rows(mesh, latent[:, None], rope[:, None])
            kv = ttnn.to_memory_config(packed, hs.kv_memory_config(mesh, cfg))
        b.kv = kv
        sels, counts = [], []
        written = {"sel": {}, "table": None, "positions": None}  # what the host last wrote, for validate()
        for i, layer in enumerate(full):
            tables, count = selection_tables(cfg, rows, layer)
            sels.append(ttnn.to_device(host_tables(tables, mesh), mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            written["sel"][i] = tables
            counts.append(count)
        chain.sel, b.sels = sels[0], sels
        st.set_cache(cfg, kv, sels[0])
        for layer in range(n):
            st.layers[layer].sel = sels[max(i for i, f in enumerate(full) if f <= layer)]
        logger.info(
            f"HY4_MK11 {label}: {len(sels)} selection tables (full-indexer layers {full}); positions read per row on each: {counts}; "
            f"visible positions per row {[c.position + 1 for c in rows]}; stripe rows up to {max(fr.stripe_row(c.position) for c in rows)}"
        )

        def frame_tables():
            table = list(store.table(pool_g))
            st.set_frame_tables(table)
            written["table"] = table

        def inputs():
            """What the host gives a step: the rows' positions (the append), their RoPE tiles; at rest otherwise."""
            chain.positions = [c.position for c in rows]
            chain.J = [c.step for c in rows]
            st.a.set_positions(chain.positions)
            written["positions"] = list(chain.positions)
            TR.set_rope(chain)
            ttnn.synchronize_device(mesh)

        def validate(what):
            """Before any launch or replay, from what the host LAST WROTE to the device (tables, selections,
            positions): what sparse_mla's selection_table asserts, for the frame form. Per row and chip: at most top-k
            ids; every id a stripe row of a position the row may see (at or below its own); every id's frame mapped in
            the row's table (a reader sent to an unmapped frame would read through a garbage address); and the frame of
            the row's own position mapped (the append)."""
            table, positions = written["table"], written["positions"]
            assert table is not None and positions is not None and len(table) == USERS * fr.TABLE_FRAMES
            ids_seen = frames_seen = 0
            for u, p in enumerate(positions):
                row_table = table[u * fr.TABLE_FRAMES : (u + 1) * fr.TABLE_FRAMES]
                assert (
                    row_table[fr.stripe_row(p) >> 8] != fr.UNMAPPED
                ), f"{what}: row {u}: the frame of its own position {p} is not mapped"
                for i, tables in written["sel"].items():
                    total = 0
                    for chip, t in enumerate(tables):
                        count = int(t[0, u])
                        assert 0 <= count <= cfg.topk, (what, u, chip, count)
                        ids = t[1 + u, :count].long()
                        if count:
                            assert int(ids.min()) >= 0 and len(set(ids.tolist())) == count, (what, u, chip, "ids")
                            seen = torch.tensor([fr.position_of(chip, int(r)) for r in ids.tolist()])
                            assert (
                                int(seen.max()) <= p
                            ), f"{what}: row {u} chip {chip} table {i}: a position {int(seen.max())} beyond its own {p}"
                            frames = set((ids >> 8).tolist())
                            unmapped = [f for f in frames if f >= fr.TABLE_FRAMES or row_table[f] == fr.UNMAPPED]
                            assert (
                                not unmapped
                            ), f"{what}: row {u} chip {chip} table {i}: ids in frames {unmapped} that its table does not map"
                            frames_seen = max(frames_seen, max(frames) + 1)
                        total += count
                    assert total <= cfg.topk, (what, u, i, total)
                    ids_seen += total
            return ids_seen, frames_seen

        def bindings():
            """After a row's conversation changed: the tables (both of the attention's tensors and the restore image, one
            call), the selections and the token ids, each written in place."""
            frame_tables()
            for i, layer in enumerate(full):
                tables, _ = selection_tables(cfg, rows, layer)
                ttnn.copy_host_to_device_tensor(host_tables(tables, mesh), sels[i])
                written["sel"][i] = tables
            fe.write_ids([c.g["input_ids"][c.position] for c in rows])  # (the fed ids: DRAM, in place)
            ttnn.synchronize_device(mesh)

        frame_tables()
        inputs()
        seen = validate("before the eager step")
        logger.info(
            f"HY4_MK11 {label}: the selections, tables and positions as written: {seen[0]} ids over the {len(full)} tables, all at or below their row's position and in mapped frames; frames a row reads from: up to {seen[1]}"
        )
        if not is_mock():
            b.pool_rows = TR.all_rows(mesh, kv, cfg)
            b.base["bench"] = CX.storage(mesh, bridge)["kvpe"]
        rep("fused L1 state, weights, step tensors (reserve open)")
        # the fused ends, inside the reserve: the embedding table, the head's streams, the id and logits tensors (DRAM),
        # EndsHead.sync (L1, registered)
        t0 = time.perf_counter()
        fe = b.fe = chain.fused_ends = CX.FusedEnds(mesh, chain)
        fe.write_ids([c.g["input_ids"][c.position] for c in rows])
        ttnn.synchronize_device(mesh)
        mark("e3 the fused ends: the embedding table, the head's streams, sync words, id and logits tensors", t0)
        rep("the fused ends' tensors (reserve open)")
        exact = bool(st.a.fixed_order)  # (two replays of the same step give the same bits: PLAN.md section 49)
        violations = []  # rows that kept their conversation and came out with other bits than the run before

        def one_step():
            fe.step(audit=True)

        ends_seen = {"steps": 0, "off": []}
        tail_cache = []

        def tail_reference():
            if not tail_cache:
                tail_cache.append(TC.head_reference())
            return tail_cache[0]

        def check(what, reference=None, same=(), changed=(), log_only=False):
            """Every row against its own golden's step. ``reference`` (an earlier check): rows ``same`` decode the same
            thing as then and must come out equal in every bit of streams, logits and id (fixed order); with ``changed``
            (rows whose conversation changed since: another row's MoE union) a difference of the others is collected in
            ``violations`` and asserted at the end of the test, so that one run lists them all. log_only: the comparison
            is a measurement (the untraced eager step against the traced replay)."""
            streams, dev, got = fe.read()
            status = st.status()
            # the detector of PLAN.md section 44: the device's logits against the reference tail on the step's own streams
            tail = TC.host_logits(*tail_reference(), streams)[1]
            off = ends.off_shards(dev, tail)
            ends_seen["steps"] += 1
            if off:
                ends_seen["off"].append((what, off))
                logger.warning(
                    f"HY4_MK11 {label} {what}: ENDS OFF on this step: chips' shards of the vocabulary that leave the reference tail on the step's own streams {off}"
                )
            per_row = []
            for u, c in enumerate(rows):
                gl, want = c.g["logits"][c.step].float(), int(c.g["cached_logits_argmax"][c.step])
                top2 = gl.topk(2).values
                per_row.append(
                    (
                        round(TC.pcc(gl, dev[u]), 5),
                        int(got[u]) == want,
                        round(float(top2[0] - top2[1]), 3),
                        round(TC.pcc(c.g["layers"][n - 1]["streams_out"][c.step].float(), streams[u]), 5),
                    )
                )
            text, differ = "", []
            if reference is not None:
                bits = CX.same_rows(reference, (streams, dev, got))
                differ = [u for u in same if not bits[u]]
                eq = [int(got[u]) == int(reference[2][u]) for u in same]
                pc = [round(TC.pcc(reference[1][u], dev[u]), 6) for u in same]
                text = (
                    f"; rows {list(same)} against the run before: equal in every bit of streams, logits and id on {len(same) - len(differ)} of "
                    f"{len(same)} (rows that differ {differ}), tokens equal {sum(eq)} of {len(eq)}, logits pcc min {min(pc):.6f}"
                    + (f"; rows {list(changed)} changed their conversation" if changed else "")
                )
            hc = fe.head.canary_summary()
            logger.info(
                f"HY4_MK11 {label} {what}: per row (logits pcc vs its golden, token = the golden's, the golden's margin, streams pcc): "
                f"{per_row}; rows {[c.key for c in rows]}; tokens {sum(p[1] for p in per_row)} of {USERS}{text}; device logits vs the "
                f"reference tail on the same streams {TC.pcc(tail, dev):.6f}; ids = the first maximum of the device's logits on "
                f"{int((got == dev.argmax(-1)).sum())} of {USERS}; the HEAD's canary {hc}; status {status}"
            )
            assert not status, f"{what}: a status is left: {status}"
            assert bool(torch.isfinite(streams).all())
            assert not off, f"{what}: the device's logits leave the reference tail on the step's own streams: {off}"
            assert (
                int((got == dev.argmax(-1)).sum()) == USERS and hc["failed"] == 0
            ), f"{what}: ids / the HEAD's canary {hc}"
            assert min(p[0] for p in per_row) > MIN_PCC, f"{what}: a row's logits are far from its golden's: {per_row}"
            if reference is not None:
                if exact and differ and not changed and not log_only:
                    raise AssertionError(
                        f"{what}: nothing changed since the run before and rows {differ} differ from it"
                    )
                if exact and differ and changed:
                    violations.append((what, differ, [pc[list(same).index(u)] for u in differ]))
                firm = [u for u in same if per_row[u][2] > 1.0]
                assert all(
                    int(got[u]) == int(reference[2][u]) for u in firm
                ), f"{what}: a row with a firm token changed it: {eq} {pc}"
            return streams, dev, got

        one_step()
        ttnn.synchronize_device(mesh)
        if not is_mock():
            first = check("the eager step (during the reserve)")
        rep("the eager step")
        st.hw.mark_per_step(st.a.rope)
        b.restore_program = st.make_restore()
        b.restore_program.launch()
        ttnn.synchronize_device(mesh)
        logger.info(f"HY4_MK11 {label}: {b.restore_program.summary()}")
        if not is_mock():
            b.restore_trace, _ = tracing.capture(mesh, b.restore_program.launch)
            ttnn.synchronize_device(mesh)
            rep("the restore program made, launched once and captured")
            t0 = time.perf_counter()
            b.trace_id, _ = tracing.capture(mesh, one_step)
            ttnn.synchronize_device(mesh)
            mark("f our step captured as a trace", t0)
            rep("our trace captured")
        used, outside = reserve.close(before)
        mark(f"e2 fused decode weights and the step's tensors ({used * banks >> 20} MB per chip of the reserve)", t0)
        assert not outside, f"DRAM tensors of ours outside the reserved range: {outside}"
        rep("reserve closed, plugs freed")

        # the start-up rule (PLAN.md section 24) as an assert, before the first replay
        now = RS.report(mesh, "", log=False)
        declared = set(st.hw.live()) | {
            int(ttnn.get_global_semaphore_address(sem)) for sem, _, _, _ in st.hw.semaphores
        }
        late = RS.late_allocations(base_report, now, (reserve.address, reserve.address + reserve.size), declared)
        plan = st.hw.plan()
        logger.info(
            f"HY4_MK11 {label}: the registry: {st.hw.summary()}; a restore zeroes {len(plan['zero'])} ranges ({plan['bytes']['zero']} B per "
            f"chip), stores {len(plan['store'])} tensors ({plan['bytes']['store']} B per chip), resets {plan['semaphores']} semaphores; alive "
            f"now and allocated after the prefill capture outside what was declared for it: {late}"
        )
        assert not any(late.values()), f"allocated after the first capture outside what was declared: {late}"
        text, ok = fe.registry_lines(st.hw, plan)
        logger.info(
            f"HY4_MK11 {label}: the fused ends in the registry: {text}; L1_SMALL buffers now {len(now['L1_SMALL'])}, at the prefill capture {len(base_report['L1_SMALL'])}"
        )
        assert ok, text

        if is_mock():
            # the host side of a round: a conversation leaves its row, a longer one takes it, everything in place
            store.unbind(7)
            convs["x"] = Conv("x", goldens[MIX[6]], 2)
            assert store.begin("x", len(convs["x"].tokens) + steps) is not None
            store.bind(7, "x")
            rows[7] = convs["x"]
            bindings()
            inputs()
            validate("the mock's rebinding")
            assert not any(r.stale() for r in getattr(st, "restores", []))
            logger.info(
                f"HY4_MK11 {label} mock: restore wrote (tensors, bytes per chip) {st.restore()}; tables after a rebinding {[store.frame_ids(u)[:3] for u in range(USERS)]}"
            )
            return

        def decode():
            validate("before a decode replay")
            return (
                tracing.replay_ms(mesh, b.trace_id, DECODES) if DECODES > 1 else tracing.replay_ms(mesh, b.trace_id, 1)
            )

        def restore():
            ttnn.execute_trace(mesh, b.restore_trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            bad = st.hw.verify(mesh, ranges_only=True)
            assert not bad, f"registered tensors that are not what the host last wrote after the restore: {bad[:8]}"

        def pool_check(what, fresh, written=()):
            """The rows of the pool that changed since the last snapshot: only decode positions, each in the frame the
            store maps it to, and whole frames listed in ``written`` (a request's copy-out); the bench not at all by
            decode. ``fresh``: rows whose decode row had not been appended yet (it must have changed)."""
            now_rows = TR.all_rows(mesh, kv, cfg)  # [chips, frames * layers, 256, width]
            differ = (b.pool_rows != now_rows).any(dim=-1)
            found = {tuple(i) for i in differ.nonzero().tolist()}
            want, must = set(), set()
            for u, c in enumerate(rows):
                frame = store.conversation(c.key).frames[c.position // fr.FRAME_TOKENS]
                for layer in range(n):
                    spot = (fr.chip_of(c.position), frame * n + layer, fr.stripe_row(c.position) % fr.FRAME_ROWS)
                    want.add(spot)
                    if u in fresh:
                        must.add(spot)
            outside = sorted(f for f in found - want if f[1] // n not in written)
            logger.info(
                f"HY4_MK11 {label} pool after {what}: rows that changed {len(found)} of {differ.numel()}; the decode positions' rows in "
                f"the frames the store maps them to: {len(want)}; changed rows outside them and outside the frames a request wrote "
                f"{sorted(written)}: {outside[:6]}; decode rows of {sorted(fresh)} that did not change: {sorted(must - found)[:6]}; frames by "
                f"decode row {[list(store.conversation(c.key).frames) for c in rows]}"
            )
            assert not outside and must <= found, (outside[:6], sorted(must - found)[:6])
            b.pool_rows = now_rows

        def bench_check(what):
            bench = CX.storage(mesh, bridge)["kvpe"]
            changed = int((bench != b.base["bench"]).sum())
            logger.info(f"HY4_MK11 {label} bench after {what}: bytes that changed {changed}")
            assert changed == 0, f"{what}: decode wrote the bench"

        bindings()  # (the same tables, selections and ids again, in place: the writes a rebinding makes)
        inputs()
        ms_a = decode()
        ref = check(
            f"decode replay A ({DECODES} back to back)", reference=first, same=range(USERS), log_only=True
        )  # (the eager step: the same step, untraced)
        pool_check("the eager step and decode replay A", fresh=set(range(USERS)))
        bench_check("the eager step and decode replay A")
        rep("decode replay A")
        beyond = [
            (u, c.position, fr.stripe_row(c.position), store.conversation(c.key).frames[c.position // fr.FRAME_TOKENS])
            for u, c in enumerate(rows)
            if c.position >= fr.FRAME_TOKENS
        ]
        logger.info(
            f"HY4_MK11 {label} STAGE 1: decode replay A {ms_a:.3f} ms per replay; rows whose own position lies beyond the first frame "
            f"(row, position, stripe row, pool frame the append wrote): {beyond}; canary {st.canary_summary()}; phases {marks}"
        )
        if REPEATS:
            # Replay against replay on this mix (PLAN.md section 42 found two things with the ttnn ends and the arrival
            # order: rows with several states, and replays whose logits were off on every row). Three ways of getting to
            # a replay, REPEATS of each, EVERY one read: "single": a replay, a wait, the read; "restore": the restore
            # trace's replay and the decode replay enqueued together; "pair": two decode replays enqueued together, the
            # second read. Counted: replays with a row that differs from replay A in any bit of streams, logits or id.
            summary = {}
            for mode in ("single", "restore", "pair"):
                differ = []
                for i in range(REPEATS):
                    validate("before a decode replay")
                    if mode == "restore":
                        ttnn.execute_trace(mesh, b.restore_trace, cq_id=0, blocking=False)
                    if mode == "pair":
                        ttnn.execute_trace(mesh, b.trace_id, cq_id=0, blocking=False)
                    ttnn.execute_trace(mesh, b.trace_id, cq_id=0, blocking=False)
                    bits = CX.same_rows(ref, fe.read())
                    if not all(bits):
                        differ.append((i, [u for u in range(USERS) if not bits[u]]))
                summary[mode] = differ
            status = st.status()
            logger.info(
                f"HY4_MK11 {label} REPLAY AGAINST REPLAY (fixed_order {st.a.fixed_order}): {REPEATS} replays each way, every one read; replays "
                f"with a row that differs from replay A in any bit (replay, rows): { {m: v[:8] for m, v in summary.items()} }; counts "
                f"{ {m: len(v) for m, v in summary.items()} }; the HEAD's canary {fe.head.canary_summary()}; status {status}"
            )
            assert not status
            assert not exact or not any(summary.values()), summary
        if STAGE < 2:
            logger.info(
                f"HY4_MK11 {label} ENDS: steps read through the device's ends {ends_seen['steps']}, off {len(ends_seen['off'])}: {ends_seen['off']}"
            )
            assert fe.head.canary_summary()["failed"] == 0
            for which, v in st.canary_summary().items():
                assert v["found"] == 0 and v["timeouts"] == 0, f"canary {which}: {v}"
            return

        # the requests: (key, the mix entry of its prompt, the row it takes, the golden step it decodes, does it fit)
        script = [
            ("x1", 6, 7, 2, True),
            ("x2", 7, 0, 3, True),
            ("x3", 6, None, 4, False),
            ("x4", 0, 6, 4, True),
            ("x3", 6, 1, 5, True),
        ][:ROUNDS]
        timeline = []
        for r, (key, entry, row, step, fits) in enumerate(script, 1):
            conv = convs[key] = Conv(key, goldens[MIX[entry]], step)
            passes = []

            def after_pass(slot, start, end, result):
                if slot != engine.bench_user:
                    return
                t = time.perf_counter()
                restore()
                ms_restore = (time.perf_counter() - t) * 1e3
                inputs()
                ms = decode()
                check(
                    f"request {r}: {DECODES} decode replays after the pass [{start}, {end}) of conversation {key!r}",
                    reference=ref,
                    same=range(USERS),
                )
                passes.append((start, end, round(result["chunk_s"] * 1e3, 1), round(ms_restore, 1), round(ms, 3)))

            hooked.after = after_pass
            free_before, evictions = store.pool.free, store.stats.evictions
            frames_before = {k: list(c.frames) for k, c in store.conversations.items()}
            rec = engine.begin(key, conv.tokens, max_new=steps)
            hooked.after = None
            if not fits:
                # no room even with every conversation without a row evicted: nothing changed, the request waits
                now_frames = {k: list(c.frames) for k, c in store.conversations.items() if c.frames}
                assert (
                    rec is None
                    and now_frames == {k: v for k, v in frames_before.items() if v}
                    and store.stats.evictions == evictions
                )
                logger.info(
                    f"HY4_MK11 {label} request {r}: {key!r} of {len(conv.tokens)} tokens ({frames_of(conv)} frames) does not fit: free "
                    f"frames {free_before}, frames of conversations without a row "
                    f"{ {k: len(c.frames) for k, c in store.conversations.items() if c.frames and c.row is None} }; begin returned None, no "
                    f"frame list changed, no eviction, no pass; waits {store.stats.waits}"
                )
                timeline.append((key, len(conv.tokens), None, [], [], None, None))
                continue
            assert rec is not None, f"request {r}: {key!r} does not fit"
            evicted = [k for k, v in frames_before.items() if v and not store.conversation(k).frames]
            new_frames = list(store.conversation(key).frames)
            # the request's rows against the same passes into an ordinary slot, then the pool: only its frames and the
            # decode rows changed
            engine.reference(key, 1)
            cmp = engine.compare(key, 1)
            assert not any(isinstance(v, dict) and v.get("bad") for v in cmp.values()), cmp
            restore()
            pool_check(
                f"request {r}: {key!r} with decode replays between its passes", fresh=set(), written=set(new_frames)
            )
            b.base["bench"] = CX.storage(mesh, bridge)["kvpe"]  # (the passes wrote the workbench and slot 1)
            # the conversation takes its row; the one that leaves it has no request in flight any more
            left = rows[row].key
            store.unbind(row)
            engine.end(left)
            store.bind(row, key)
            rows[row] = conv
            t = time.perf_counter()
            bindings()
            ms_bind = (time.perf_counter() - t) * 1e3
            restore()
            inputs()
            ms = decode()
            others = [u for u in range(USERS) if u != row]
            new = check(
                f"request {r}: row {row} now serves {key!r} ({len(conv.tokens)} tokens, frames {new_frames})",
                reference=ref,
                same=others,
                changed=[row],
            )
            pool_check(f"request {r}: the decode replays after the rebinding", fresh={row})
            bench_check(f"request {r}")
            ref = new
            timeline.append((key, len(conv.tokens), new_frames, evicted, passes, round(ms_bind, 1), round(ms, 3)))
            logger.info(
                f"HY4_MK11 {label} request {r}: {key!r} of {len(conv.tokens)} tokens: free frames before {free_before}, conversations "
                f"evicted for it {evicted} (evictions {store.stats.evictions - evictions}), its frames {new_frames}; passes (start, end, "
                f"pass ms, restore replay with its read-back ms, ms per decode replay) {passes}; copy-in {1e3 * rec['copy_in_s']:.1f} ms, "
                f"copy-out {1e3 * rec['copy_out_s']:.1f} ms; against slot 1 prefilled alone { {k: v for k, v in cmp.items() if k != 'bad'} }; "
                f"row {row} from {left!r} to {key!r}: tables, selections and ids in place {ms_bind:.1f} ms from the host; decode replay "
                f"after it {ms:.3f} ms"
            )
        rep("after the rounds")
        logger.info(
            f"HY4_MK11 {label} ENDS: steps read through the device's ends {ends_seen['steps']}, off {len(ends_seen['off'])}: {ends_seen['off']}"
        )
        canary = st.canary_summary()
        logger.info(
            f"HY4_MK11 {label} REQUESTS: decode replay before any request {ms_a:.3f} ms; per request (key, tokens, frames, evicted, "
            f"passes, binding ms, decode ms) {timeline}; store {store.stats}; canary {canary}; phases {marks}"
        )
        for which, v in canary.items():
            assert v["found"] == 0 and v["timeouts"] == 0, f"canary {which}: {v}"
        logger.info(
            f"HY4_MK11 {label} BIT-EQUALITY: rows that kept their conversation while another row changed its own and came out with other "
            f"bits than the run before (check, rows, their logits pcc against it): {violations}"
        )
        assert not (STRICT_OTHERS and violations), violations
    finally:
        if b.trace_id is not None:
            ttnn.release_trace(mesh, b.trace_id)
        if b.restore_trace is not None:
            ttnn.release_trace(mesh, b.restore_trace)
        if b.restore_program is not None:
            b.restore_program.free()
        for t in getattr(b, "sels", [])[1:]:
            ttnn.deallocate(t)
        CX.free_both(b)
