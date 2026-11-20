# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill stack and the fused decode step in one process WITHOUT releasing the prefill model: a prefill pass
(a replay of the stack's trace) between two decode steps (PLAN.md sections 24 and 26).

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>     (the fused side, the reserve, the snapshots)
    HY4_TEST_TIMEOUT=2400 TT_METAL_OPERATION_TIMEOUT_SECONDS=30 pytest <this file> -x

Env: as tests/test_real.py; HY4_COEXIST_RESERVE_GB (3: the DRAM held for the fused step's tensors, per chip).

Sequence (lines HY4_MK11 coexist):
  a. the prefill stack through its adapter in TRACE mode (PrefillRunParams(use_trace=True)), its caches for 8 users,
     runtime.compile (the eager warm-up: every program of the stack is enqueued here, with nothing of ours in L1)
  b. a DRAM ballast holds the range the fused step's tensors will take (model/reserve.py: the Stack makes its L1
     state before its DRAM weights, so the reserve stands in for "our DRAM tensors before the capture")
  c. runtime.capture_trace
  d. prefill by replay: the golden's prompt and HF's continuation into the 8 slots; slot 1 twice (is a pass
     bit-equal to its repeat?); both caches read on all 32 chips
  e. plugs over all free DRAM, the ballast freed, the fused L1 state, the decode weights, the step's tensors (all DRAM
     of ours inside the reserved range, checked by address), the plugs freed
  f. decode step A on the 8 slots' rows, program by program, against HF; then every device tensor of ours is read
     (L1: all chips, kept; DRAM: checksums, the large streams on three chips) with the stack's two caches
  g. ONE prefill pass by replay for slot 1 (the same tokens) with our tensors allocated. A throw is reported with
     its origin (the enqueue-time clash check or elsewhere) and ends the test
  h. the same reads: what the replay changed of ours (per tensor: chips, cores, byte range) and of the caches (the
     slots it did not touch; slot 1 against the same pass of d, all chips)
  i. Stack.reset() (the host's restore of the hand-over words), what still differs from f, then the host writes f's
     contents back into every L1 tensor that still differs; decode step B against HF and against step A
"""

import os
import time
import traceback

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.model import reserve as RS
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.model.tests import test_chain as TC
from models.experimental.hy4_preview.fused.model.tests import test_real as TR
from models.experimental.hy4_preview.tt import prefill_bridge as pb
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

RESERVE = int(float(os.environ.get("HY4_COEXIST_RESERVE_GB", "3")) * (1 << 30))
# "full": after Stack.reset() the host writes step A's contents back into every L1 tensor that still differs (the upper
# bound of a restore); "cut": Stack.reset() and zeros in the status buffers and the canary's counters only, everything
# else stays as the replay left it (the restore PLAN.md section 26 reads from the first run's list); "registry": no
# Stack.reset(): Stack.restore() rewrites every L1 tensor the host ever wrote with what it last wrote (hostwrites.py),
# read back and compared before step B
RESTORE = os.environ.get("HY4_COEXIST_RESTORE", "full")


def storage(mesh, bridge):
    """Both caches of the prefill stack as bytes, [32 chips, ...] each, user slot major (the stack's own reader: a
    single chip's fp8 tensor cannot be read)."""
    ttnn.synchronize_device(mesh)
    out = {}
    for name, t in (("kvpe", bridge.kv_caches.kvpe.storage), ("index", bridge.kv_caches.index)):
        if t is not None:
            slabs = bridge.read_slabs(t).contiguous()
            out[name] = slabs.view(torch.uint8).reshape(slabs.shape[0], -1)
    return out


def slot_differences(a, b, slots):
    """Per cache, the slots (of ``slots`` equal parts of a chip's bytes) and chips where a and b differ."""
    out = {}
    for name in a:
        x, y = a[name].reshape(TC.CHIPS, slots, -1), b[name].reshape(TC.CHIPS, slots, -1)
        diff = (x != y).any(dim=-1)  # [chips, slots]
        out[name] = {
            "slots": sorted(diff.any(dim=0).nonzero().flatten().tolist()),
            "chips": int(diff.any(dim=1).sum()),
            "bytes": int((x != y).sum()),
        }
    return out


def cut_restore(mesh, chain, tensors, ours_l1, snap, rest):
    """The cut restore's host writes beside Stack.reset(): zeros in the three projection groups' status words (arena
    ranges on their hubs), in the MoE status buffers and in the canary's counters; and the dense FFN's slot words (its
    hub's meta buffer: the host writes them once when the layer is set, MlpLayer.set_layer, and no step rewrites
    them: mk11_coexist_cut ran without this and layer 0's MLP output came out at gain 0.88, 5 chips' slots gone).
    ``snap``: the L1 contents now; ``rest``: the contents after a step."""
    from models.experimental.hy4_preview.fused.projections import program as PJ

    a = chain.st.a
    state = chain.st.layers[0].attn
    address = lambda t: int(ttnn.get_device_tensors(t)[0].buffer_address())
    dense_meta = {address(m.op.meta) for m in chain.st.mlps.values() if m.dense}
    cut = {}
    for path in ours_l1:
        if address(ours_l1[path]) in dense_meta:
            cut[path] = rest[path]
            continue
        if path.endswith(".a.arena"):
            raws = [r.clone() for r in snap[path]["raw"]]
            for prog in (state.pre, state.query, state.post):
                start = a._core_index((prog.hub.x, prog.hub.y)) * A.ARENA_BYTES + prog.offsets["status"]
                for r in raws:
                    r[start : start + 4 * PJ.STATUS_WORDS] = 0
            cut[path] = dict(snap[path], raw=raws)
        elif path.endswith(".status") or path.endswith(".canary"):
            cut[path] = dict(snap[path], raw=[torch.zeros_like(r) for r in snap[path]["raw"]])
    RS.write_back(mesh, tensors, cut, list(cut))
    return sorted(cut)


class Both:
    """What build_both leaves: the prefill stack (trace mode, captured, 8 slots filled by replay) and the fused step
    with its weights, both built, the fused step's DRAM inside the reserved range."""

    bridge = chain = reserve = prefill = kv = base = None


def build_both(b, mesh, monkeypatch, mark, during_reserve=None):
    """Phases a to e of this module's sequence into ``b`` (filled as it goes: the caller's finally frees what
    exists). during_reserve(b): called while the reserved range is the only free DRAM, after the weights and the
    step's tensors (whatever else of ours must persist is allocated there)."""
    g = b.g = TC.hf_harness.find_decode_golden(TC.LAYERS, TC.PROMPT)
    N, n, steps = g["meta"]["prompt_len"], TC.LAYERS, g["meta"]["steps"]
    b.N, b.n, b.steps = N, n, steps
    layout = b.layout = BlockCyclicLayout()
    tokens = b.tokens = [int(v) for v in g["input_ids"][: N + steps - 1]]
    kv = None
    mark("0 mesh open", time.perf_counter())
    banks = b.banks = RS.dram_view(mesh)["banks"]
    if not is_mock():
        # a. the prefill stack in trace mode (the bridge builds it; the one parameter it does not pass is use_trace)
        import models.demos.common.prefill.adapter as AD
        from models.experimental.hy4_preview.fused.kv_store.engine import StackPrefill

        params = AD.PrefillRunParams
        monkeypatch.setattr(AD, "PrefillRunParams", lambda **kw: params(**dict(kw, use_trace=True)))
        if POOL:
            # the store's Engine between the stack's cache allocation and its compile: its two L1 table tensors must
            # exist before the stack's programs size their buffers (kv_store's own test builds in this order)
            from models.experimental.hy4_preview.fused.kv_store.engine import Engine
            from models.experimental.hy4_preview.fused.kv_store.manager import KvStore

            adapter = AD.get_adapter("hy4_preview")
            allocate = adapter.allocate_kv_cache

            def allocate_with_engine(**kw):
                caches = allocate(**kw)
                b.store = KvStore(POOL_FRAMES, rows=TR.USERS)
                b.engine = Engine(mesh, b.store, caches.kvpe.storage, caches.index, None, bench_users=TR.USERS)
                return caches

            monkeypatch.setattr(adapter, "allocate_kv_cache", allocate_with_engine)
        t0 = time.perf_counter()
        bridge = b.bridge = pb.PrefillBridge(mesh, num_layers=n, num_users=TR.USERS, max_seq_len=TR.MAX_SEQ)
        bridge.build()
        monkeypatch.undo()
        mark("a prefill stack built in trace mode, warmed", t0)
        if POOL:
            t0 = time.perf_counter()
            b.engine.warm()
            mark(
                f"a2 the store's engine warmed (pool of {POOL_FRAMES} frames, its copy and compare programs launched once)",
                t0,
            )
    # b. the reserve
    t0 = time.perf_counter()
    reserve = b.reserve = RS.DramReserve(mesh, RESERVE // banks)
    mark(f"b ballast of {RESERVE >> 20} MB per chip", t0)
    if not is_mock():
        # c. the capture
        t0 = time.perf_counter()
        runtime, caches = bridge.runtime, bridge.kv_caches
        runtime.capture_trace(caches)
        ttnn.synchronize_device(mesh)
        mark(
            f"c prefill trace captured ({runtime._controller.num_segments} segments, {runtime._controller.trace_bytes()} B)",
            t0,
        )
        # d. prefill by replay
        prefill = b.prefill = StackPrefill(runtime, caches, mesh, pb.MESH, trace=True)
        if POOL:
            # d (pool). 8 requests through the engine: passes by replay into the workbench (user slot 0), kv_copy out to
            # the conversation's pool frames; decode row u serves conversation u
            engine, store = b.engine, b.store
            engine.prefill = prefill
            t0 = time.perf_counter()
            recs = [engine.begin(u, tokens, max_new=steps) for u in range(TR.USERS)]
            assert all(r is not None for r in recs), "the pool has no room for the 8 conversations"
            for u in range(TR.USERS):
                store.bind(u, u)
            engine.sync_tables()
            mark(f"d {TR.USERS} requests of {len(tokens)} tokens through the engine (workbench, then pool frames)", t0)
            # each conversation's pool frames against an ordinary slot prefilled alone (the store's compare program)
            cmp = {}
            for u in range(TR.USERS):
                engine.reference(u, 1)
                cmp[u] = engine.compare(u, 1)
            bad = {u: {k: v for k, v in c.items() if isinstance(v, dict) and v.get("bad")} for u, c in cmp.items()}
            logger.info(
                f"HY4_MK11 coexist d (pool): requests: {[(r['frames'], len(r['passes']), round(1e3 * r['copy_out_s'], 1)) for r in recs]} "
                f"(frames, passes, copy-out ms); frames of the conversations {[list(store.conversation(u).frames) for u in range(TR.USERS)]}; "
                f"each conversation's pool frames against slot 1 prefilled alone: {cmp[0]}; conversations with rows that differ: "
                f"{ {u: v for u, v in bad.items() if v} }"
            )
            assert not any(bad.values()), bad
            kv = engine.pool[engine.latent]
            b.base = {"bench": storage(mesh, bridge)["kvpe"]}
        else:
            t0 = time.perf_counter()
            passes = [prefill(user, 0, len(tokens), tokens) for user in range(TR.USERS)]
            mark(f"d prefill of {len(tokens)} tokens into {TR.USERS} slots by replay", t0)
            first = storage(mesh, bridge)
            t0 = time.perf_counter()
            passes.append(prefill(1, 0, len(tokens), tokens))
            ttnn.synchronize_device(mesh)
            pass_ms = (time.perf_counter() - t0) * 1e3
            logger.info(
                f"HY4_MK11 coexist d: the replayed passes before anything of ours exists, ms each: {[pass_text(r) for r in passes]}"
            )
            b.base = storage(mesh, bridge)
            again = {k: slot_differences({k: first[k]}, {k: b.base[k]}, 1)[k] for k in first}
            logger.info(
                f"HY4_MK11 coexist d: slot 1's pass repeated before anything of ours exists ({pass_ms:.0f} ms with its "
                f"input): the caches against the first pass (bytes that differ, chips): {again}"
            )
            del first
            kv = caches.kvpe.storage
        TR.run_sanity(mesh, "after the capture and the replays, the prefill model built")
    if getattr(b, "after_prefill_capture", None) is not None:
        b.after_prefill_capture()
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.L1_SMALL)
    b.small_prefill = {"allocated": v.total_bytes_allocated_per_bank, "buffers": RS.buffers(mesh, "L1_SMALL")}
    # e. ours: nothing free in DRAM but the reserved range while it is allocated
    t0 = time.perf_counter()
    before = RS.buffers(mesh, "DRAM")
    l1_before = RS.buffers(mesh, "L1")
    t_plug = time.perf_counter()
    reserve.plug()
    reserve.open()
    t_plug = time.perf_counter() - t_plug
    # (b.head_site: the site table gets the model's head as its last entry, for the fused ends' HEAD program;
    # b.wait_limit: the bound of the step's waits, where the caller sets one)
    chain_kw = {"wait_limit": b.wait_limit} if getattr(b, "wait_limit", None) else {}
    chain = b.chain = TC.Chain(
        mesh, load=False, attn_kwargs=TR.attn_kwargs(), head_site=bool(getattr(b, "head_site", False)), **chain_kw
    )
    st = chain.st
    mark("e1 fused L1 state", t0)
    t0 = time.perf_counter()
    chain.load_layers()
    kv = b.kv = TR.prepare_step(chain, mesh, kv, N, steps, layout)
    if POOL and not is_mock():
        # the store's tables in place of the identity tables: the one call through which a binding reaches the step
        st.set_frame_tables(b.store.table(b.engine.pool_g[b.engine.latent]))
        # (the pool and the bench before any decode step of ours has appended)
        b.pool_rows = TR.all_rows(mesh, kv, chain.cfg)
        b.base["bench"] = storage(mesh, b.bridge)["kvpe"]
    if during_reserve is not None:
        during_reserve(b)
    t_close = time.perf_counter()
    used, outside = reserve.close(before)
    t_close = time.perf_counter() - t_close
    logger.info(
        f"HY4_MK11 coexist e: wall time of the plug allocations and the ballast's release {t_plug * 1e3:.1f} ms, of the "
        f"address check and the plugs' release {t_close * 1e3:.1f} ms"
    )
    mark(f"e2 fused decode weights and the step's tensors ({used * banks >> 20} MB per chip of the reserve)", t0)
    assert not outside, f"DRAM tensors of ours outside the reserved range: {outside}"
    tensors = b.tensors = RS.device_tensors(chain, "chain")
    b.ours_l1 = {p: t for p, t in tensors.items() if RS.kind_of(t) == "L1"}
    address = lambda t: int(ttnn.get_device_tensors(t)[0].buffer_address())
    b.ours_dram = {p: t for p, t in tensors.items() if RS.kind_of(t) == "DRAM" and address(t) != address(kv)}
    found = {address(t) for t in b.ours_l1.values()}
    unwalked = {k: v for k, v in RS.buffers(mesh, "L1").items() if k not in l1_before and k[1] not in found}
    logger.info(
        f"HY4_MK11 coexist e: {len(b.ours_l1)} L1 tensors and {len(b.ours_dram)} DRAM tensors of ours; L1 buffers of ours that "
        f"are not tensors of the step's objects (global semaphores, a trace's or an op's own tensors): {unwalked}"
    )
    return b


def pass_text(r):
    return f"{r['chunk_s'] * 1e3:.1f} (+ input {r['prep_s'] * 1e3:.1f})"


def free_both(b):
    if b.reserve is not None:
        b.reserve.release()
    if b.chain is not None and b.chain.sel is not None:
        ttnn.deallocate(b.chain.sel)
    if b.bridge is not None:
        b.bridge.close()


def marker(mesh, marks):
    def mark(name, t0):
        marks[name] = (round(time.perf_counter() - t0, 1), TR.gb(mesh)) + TR.l1_view(mesh)
        logger.info(
            f"HY4_MK11 coexist {name}: {marks[name][0]} s, DRAM per chip {marks[name][1]} GB, L1 tensors {marks[name][2]} B per core, "
            f"largest free L1 block {marks[name][3]} B"
        )

    return mark


@TR.galaxy
@pytest.mark.timeout(0)
def test_coexist(mesh_device, monkeypatch):
    mesh = mesh_device
    marks = {}
    mark = marker(mesh, marks)
    b = Both()
    try:
        build_both(b, mesh, monkeypatch, mark)
        bridge, chain, prefill, kv, base = b.bridge, b.chain, b.prefill, b.kv, b.base
        st, n, tokens, tensors, ours_l1, ours_dram = chain.st, b.n, b.tokens, b.tensors, b.ours_l1, b.ours_dram
        ms = pass_text
        if is_mock():
            Stack.launch(st.step_programs(audit=True))
            logger.info(
                f"HY4_MK11 coexist mock: registry restore wrote (tensors, bytes per chip) {st.restore()}; after the launch: {st.hw.summary()}"
            )
            some = dict(list(ours_l1.items())[:27:3])
            snap = RS.snapshot(mesh, some)
            assert not RS.differences(tensors, snap, snap)
            other = {
                p: {"chips": v["chips"], "sha": list(v["sha"]), "raw": [r.clone() for r in v["raw"]]}
                for p, v in snap.items()
            }
            for v in other.values():  # (the comparison and the write-back, on made-up changes)
                v["raw"][5][v["raw"][5].numel() // 2 + 7] += 1
                v["sha"][5] = "changed"
            diff = RS.differences(tensors, snap, other)
            assert len(diff) == len(some) and all(d["bytes"] == 1 for d in diff.values()), diff
            logger.info(f"HY4_MK11 coexist mock: {RS.describe(diff)}")
            RS.write_back(mesh, tensors, snap, list(diff))
            logger.info(
                f"HY4_MK11 coexist mock: the cut restore writes {cut_restore(mesh, chain, tensors, ours_l1, *[RS.snapshot(mesh, ours_l1)] * 2)}"
            )
            return
        label = f"coexist, {n} layers, P{TC.PROMPT} [{TC.MOE_FIDELITY}], rows by the prefill stack's replay"
        head, lm = TC.head_reference()

        def step(name):
            records = []
            chain.reset_streams()
            TR.set_rope(chain)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            out = TC.run_step_untraced(chain, f"{label}, step {name}", records)
            res = TC.check_logits(chain, head, lm, out[(n - 1, "streams_out")], f"{label}, step {name}")
            TC.table(records, [res], f"{label}, step {name}")
            mark(f"step {name}, untraced, with the host's reads", t0)
            return out, res

        # f. step A and the state at rest
        out_a, res_a = step("A")

        def read_all():
            l1 = RS.snapshot(mesh, ours_l1)
            dram = RS.snapshot(mesh, ours_dram, keep=False, max_elements=64_000_000)
            return l1, dram, storage(mesh, bridge)

        t0 = time.perf_counter()
        l1_f, dram_f, caches_f = read_all()
        mark("f every tensor of ours and both caches read", t0)
        again = slot_differences(base, caches_f, 1)
        logger.info(
            f"HY4_MK11 coexist f: the caches after step A against before it (no append: nothing may differ): {again}"
        )
        # g. the prefill pass with our tensors allocated
        t0 = time.perf_counter()
        try:
            r = prefill(1, 0, len(tokens), tokens)
            ttnn.synchronize_device(mesh)
            logger.info(f"HY4_MK11 coexist g: the replayed pass with our tensors allocated: {ms(r)} ms")
        except Exception as e:
            text = str(e)
            origin = "the enqueue-time clash check" if "clash with L1 buffers" in text else "elsewhere"
            frames_ = traceback.format_exc().strip().splitlines()
            logger.info(
                f"HY4_MK11 coexist g: the replay with our tensors allocated THREW ({type(e).__name__}) from {origin}: "
                f"{text[:1500]!r}; Python frames: {frames_[-12:]}"
            )
            pytest.fail(f"the prefill replay threw from {origin}: {text[:300]}")
        mark("g one prefill pass by replay with our tensors allocated", t0)
        # h. what it changed
        t0 = time.perf_counter()
        l1_h, dram_h, caches_h = read_all()
        mark("h the same reads", t0)
        d_l1 = RS.differences(tensors, l1_f, l1_h)
        d_dram = RS.differences(tensors, dram_f, dram_h)
        slots = TR.USERS  # (a chip's bytes are user-major in both caches)
        d_caches = slot_differences(caches_f, caches_h, slots)
        logger.info(f"HY4_MK11 coexist h: L1 tensors of ours changed by the replay: {len(d_l1)} of {len(ours_l1)}")
        for line in RS.describe(d_l1):
            logger.info(f"HY4_MK11 coexist h L1: {line}")
        logger.info(
            f"HY4_MK11 coexist h: DRAM tensors of ours changed by the replay: {len(d_dram)} of {len(ours_dram)} "
            f"{ {p: d['chips'] for p, d in d_dram.items()} }; frame tables: none allocated in this form (no append, no pool)"
        )
        logger.info(
            f"HY4_MK11 coexist h/j: the prefill stack's caches after the replay against before it, {slots} user slots, all "
            f"32 chips (slot 1 is the replayed one, the same tokens; the other slots were not touched): {d_caches}"
        )
        # i. the restore from the host and step B
        t0 = time.perf_counter()
        if RESTORE != "registry":
            st.reset()
        t_reset = time.perf_counter() - t0
        l1_r = RS.snapshot(mesh, ours_l1)
        d_reset = RS.differences(tensors, l1_f, l1_r)
        logger.info(
            f"HY4_MK11 coexist i: after Stack.reset(), L1 tensors that still differ from their contents after step A: "
            f"{len(d_reset)}; changed by the replay and restored by the reset: {sorted(set(d_l1) - set(d_reset))}; not changed by "
            f"the replay but by the reset itself (not at zero after a step): {sorted(set(d_reset) - set(d_l1))}"
        )
        for line in RS.describe(d_reset):
            logger.info(f"HY4_MK11 coexist i still differs: {line}")
        t_write = time.perf_counter()
        if RESTORE == "cut":
            wrote = cut_restore(mesh, chain, tensors, ours_l1, l1_r, l1_f)
        elif RESTORE == "registry":
            count, nbytes = st.restore()
            t_restore = time.perf_counter() - t_write
            t_verify = time.perf_counter()
            bad = st.hw.verify(mesh)
            logger.info(
                f"HY4_MK11 coexist i [registry]: Stack.restore() wrote {count} tensors, {nbytes} B per chip, in {t_restore * 1e3:.1f} ms; "
                f"read back and compared in {(time.perf_counter() - t_verify) * 1e3:.0f} ms: {len(bad)} differ {bad[:8]}"
            )
            assert not bad, f"registered tensors that are not what the host last wrote after the restore: {bad[:8]}"
            wrote = [e.where for e in st.hw.live().values() if not e.per_step]
            t_write = time.perf_counter() - t_restore  # (the line below then reports the restore's time alone)
        else:
            RS.write_back(mesh, tensors, l1_f, list(d_reset))
            wrote = sorted(d_reset)
        t_write = time.perf_counter() - t_write
        logger.info(
            f"HY4_MK11 coexist i [{RESTORE}]: wall time of Stack.reset() {t_reset * 1e3:.1f} ms, of the host's writes into "
            f"{len(wrote)} tensors {t_write * 1e3:.1f} ms (through the host, all chips; a program of ours would replace both)"
        )
        l1_w = RS.snapshot(mesh, ours_l1)
        left = RS.differences(tensors, l1_f, l1_w)
        mark(f"i Stack.reset() and the host's writes [{RESTORE}]", t0)
        if RESTORE != "full":
            logger.info(
                f"HY4_MK11 coexist i [{RESTORE}]: wrote {wrote}; step B starts with {len(left)} L1 tensors as the replay left them: {sorted(left)}"
            )
        else:
            assert not left, f"the write-back left differences: {RS.describe(left)}"
        out_b, res_b = step("B")
        key = (n - 1, "streams_out")
        logger.info(
            f"HY4_MK11 coexist i: step B against step A: last streams pcc {TC.pcc(out_a[key], out_b[key]):.6f}, logits pcc vs HF "
            f"A {res_a['logits']:.6f} B {res_b['logits']:.6f}, tokens equal to HF's A {res_a['tokens']} B {res_b['tokens']} of 8; "
            f"status after B {st.status()}; canary {st.canary_summary()}"
        )
        logger.info(f"HY4_MK11 coexist phases (s, DRAM GB per chip, L1 tensors B, largest free L1 block B): {marks}")
        assert not st.status()
        for kind, v in st.canary_summary().items():
            assert v["found"] == 0 and v["timeouts"] == 0, f"canary {kind}: {v}"
        assert not d_dram, f"the replay changed DRAM tensors of ours: {list(d_dram)}"
        assert abs(res_a["logits"] - res_b["logits"]) < 2e-3 and res_b["tokens"] >= res_a["tokens"] - 1, (res_a, res_b)
    finally:
        free_both(b)


ROUNDS = int(os.environ.get("HY4_COEXIST_ROUNDS", "3"))
# The traced form's variants:
#   "fused"    (default) the whole step with the ends as programs of this piece (ends_fused.py; PLAN.md sections 46 to
#              49): ids (DRAM) -> IN -> P1(post) .. MLP of every layer -> HEAD -> ids. Nothing of ours exists before the
#              prefill capture: the embedding table, the head's streams, the id tensors and the kept logits are DRAM
#              inside the reserve, EndsHead.sync is one more registered L1 tensor, IN and HEAD are launched once
#              eagerly with the other fused programs, and our trace is captured while the reserved range is still the
#              only free DRAM.
#   "no_ends"  the trace of ours is the 3N + 1 fused programs alone (streams written by the host): its capture
#              allocates nothing and compiles nothing new (PLAN.md section 28, E-A)
# (The forms with the ttnn-module ends, "ends_first" and "ends_late" of PLAN.md section 28, ended with the fused ends:
# their 38 + 10 carried semaphores, their 9 DRAM buffers and the order "the ends before the prefill stack" are gone.)
FORM = os.environ.get("HY4_COEXIST_FORM", "fused")
# "1": after a prefill pass the fused step's L1 is restored by ONE program on the device (kernels/restore.cpp,
# model/restore.py, made and launched once inside the reserve) and no host write; else Stack.restore() from the host
RESTORE_PROGRAM = os.environ.get("HY4_COEXIST_RESTORE_PROGRAM") == "1"
# decode replays back to back between two prefill passes (and before the first: the run without a prefill pass that
# the rounds are compared with). More than one needs the ends: without them a replay starts from the streams the one
# before left (the site updates them in place)
DECODES = int(os.environ.get("HY4_COEXIST_DECODES", "1"))
# single decode replays after each round's back-to-back replays, EACH read and compared with replay A bit for bit
SINGLES = int(os.environ.get("HY4_COEXIST_SINGLES", "0"))
# "1": every conversation lives in fused/kv_store's paged pool: the prefill passes go into the workbench slot (user 0 of
# the stack's caches) and are copied out to pool frames, decode reads and appends in the POOL through the store's frame
# tables (needs HY4_REAL_APPEND and HY4_REAL_READER_FRAMES). HY4_COEXIST_POOL_FRAMES: the pool's size.
POOL = os.environ.get("HY4_COEXIST_POOL") == "1"
POOL_FRAMES = int(os.environ.get("HY4_COEXIST_POOL_FRAMES", "12"))
FUSED_WAIT = 30_000_000  # the bounded waits of the step with the fused ends (tests/test_ends_fused.py test_step's)


class FusedEnds:
    """The fused ends beside a Chain (ends_fused.py), as the coexistence forms use them: one step per trace, the fed
    ids in T[0] (DRAM, written by the host once and when a row's conversation changes), the step's ids in T[1], the
    step's logits kept in DRAM by the units' second write. Everything is made at construction (a trace capture
    refuses device writes): call it while the reserved range is the only free DRAM."""

    def __init__(self, mesh, chain):
        from models.experimental.hy4_preview.fused.model import ends_fused as EF

        self.EF, self.mesh, self.chain = EF, mesh, chain
        st = chain.st
        table, head_w = TC.hy4_tensors(["model.embed_tokens.weight", "lm_head.weight"])
        self.ein = EF.EndsIn(st, table)
        stream = EF.head_stream(mesh, lambda c: head_w[c * EF.VOCAB_PER_CHIP : (c + 1) * EF.VOCAB_PER_CHIP])
        self.head = EF.EndsHead(st, 2 * chain.n, stream)
        self.T = [EF.id_tensor(mesh), EF.id_tensor(mesh)]
        self.kept = EF.logits_tensor(mesh)
        # IN zeroes the streams before layer 0's P1 reads them (PLAN.md section 46): the tensor is scratch for a
        # restore, whatever the host wrote into it when the Stack was made
        st.hw.scratch(st.a.streams, [], "fused/model/PLAN.md section 46")
        self.programs = None
        ttnn.synchronize_device(mesh)

    def write_ids(self, tokens):
        """The 8 rows' fed tokens, in place (a DRAM write: allowed after every capture)."""
        self.ein.write_ids(self.T[0], [int(v) for v in tokens])

    def step(self, audit=False):
        """One whole step: 3 N + 2 programs. The layer programs' descriptors are built at the first call."""
        st = self.chain.st
        if self.programs is None:
            self.programs = st.step_programs(first_post=True, last_post=False, audit=audit)
        self.ein.launch(self.T[0])
        Stack.launch(self.programs)
        self.head.launch(self.T[1], self.kept)

    def read(self):
        """(streams after the last layer [8, 4, hidden], the device's logits [8, vocab], the device's ids [8])."""
        st = self.chain.st
        ttnn.synchronize_device(self.mesh)
        streams = st.read_streams()
        logits = (
            self.EF.read_logits(self.kept).permute(1, 0, 2).reshape(8, -1)
        )  # the chips' shards in the order of their ids
        ids = torch.tensor(self.EF.read_ids(self.T[1], chip=st.topo.chip(st.topo.root)))
        return streams, logits, ids

    def registry_lines(self, hw, plan):
        """What the registry and a restore's plan hold of the ends: (text, the sync tensor's entry is a stored one)."""
        address = lambda t: int(ttnn.get_device_tensors(t)[0].buffer_address())
        sync, streams = hw.live().get(address(self.head.sync)), hw.live().get(address(self.chain.st.a.streams))
        stored = [s for s in plan["store"] if s[0] == address(self.head.sync)]
        text = (
            f"EndsHead.sync: {'registered' if sync is not None else 'NOT REGISTERED'}"
            + (f" ({sync.where}, {sync.bytes()} B per chip, written {sync.writes} times)" if sync is not None else "")
            + f", in the restore's plan as {'a stored tensor' if stored else 'NOT stored'}; the streams tensor: "
            + (
                "scratch (IN writes it), nothing restored"
                if streams is not None and streams.scratch is not None
                else f"{streams and streams.where}"
            )
        )
        return text, sync is not None and bool(stored)


def same_rows(a, b):
    """Per row: (streams, logits, id) of two reads of a step are equal in every bit."""
    return [
        bool(torch.equal(a[0][u], b[0][u]) and torch.equal(a[1][u], b[1][u]) and int(a[2][u]) == int(b[2][u]))
        for u in range(len(a[2]))
    ]


@TR.galaxy
@pytest.mark.timeout(0)
def test_coexist_trace(mesh_device, monkeypatch):
    """The serving form: the fused step as a trace of ours beside the prefill stack's trace, both models built. Decode
    replay A; then ROUNDS times: a prefill pass by replay, the restore (the registry's list, read back and compared),
    the per-step inputs, DECODES decode replays; every decode replay read against replay A (bit for bit with
    AttnLayer(fixed_order=True): the same step on the same rows) and against HF. The buffer report (DRAM, L1, L1_SMALL,
    trace region) is logged after every phase with what was allocated and freed since the one before, and everything
    allocated since the prefill capture is asserted to lie in what was declared for it (PLAN.md section 24)."""
    from models.experimental.hy4_preview.fused.common import tracing
    from models.experimental.hy4_preview.fused.model import ends

    assert FORM in ("no_ends", "fused"), FORM
    fused = FORM == "fused"
    mesh = mesh_device
    marks = {}
    mark = marker(mesh, marks)
    b = Both()
    b.trace_id = None
    b.head_site = fused
    if fused:
        b.wait_limit = FUSED_WAIT
    reports = {}

    def rep(label):
        prev = list(reports.values())[-1] if reports else None
        now = reports[label] = RS.report(mesh, f"{FORM}: {label}", prev=prev)
        if label == "prefill stack built, captured, slots filled":
            b.prefill_l1 = {a for a in now["L1"] if prev is None or a not in prev["L1"]}
            if getattr(b, "engine", None) is not None:
                b.prefill_l1 -= {
                    int(ttnn.get_device_tensors(t.tensor)[0].buffer_address())
                    for t in (b.engine.latent_tables, b.engine.index_tables)
                }
        if getattr(b, "prefill_l1", None) is not None:
            # every L1 buffer that is neither the prefill stack's, nor a registered tensor or semaphore of the fused
            # step: whatever nobody restores (the ttnn ends' carried counters were such; the fused ends have none)
            declared = set()
            if b.chain is not None:
                hw = b.chain.st.hw
                declared = set(hw.live()) | {
                    int(ttnn.get_global_semaphore_address(sem)) for sem, _, _, _ in hw.semaphores
                }
            address = lambda t: int(ttnn.get_device_tensors(t)[0].buffer_address())
            engine = getattr(b, "engine", None)
            own = (
                {}
                if engine is None
                else {
                    address(engine.latent_tables.tensor): "latent tables",
                    address(engine.index_tables.tensor): "index tables",
                }
            )
            other = {
                a: n
                for a, n in sorted(now["L1"].items())
                if a not in b.prefill_l1 and a not in declared and a not in own
            }
            logger.info(
                f"HY4_MK11 buffers [{FORM}: {label}]: L1 buffers that are neither the prefill stack's ({len(b.prefill_l1)}) nor "
                f"registered by the fused step ({len(declared)}): {len(other)} {other}"
                + (
                    f"; the store engine's OWN two table tensors (made before the Stack, written by the engine, read by no "
                    f"kernel of this form, overwritten by every prefill pass and not restored: NOT state): "
                    f"{ {a: (name, now['L1'].get(a)) for a, name in own.items()} }"
                    if own
                    else ""
                )
            )

    try:

        def one_step():
            if fused:
                b.fe.step(audit=True)
            else:
                Stack.launch(b.programs)

        def capture():
            t0 = time.perf_counter()
            b.trace_id, _ = tracing.capture(mesh, one_step)
            ttnn.synchronize_device(mesh)
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
            mark(
                f"f our step captured as a trace [{FORM}] (trace buffers now {view.total_bytes_allocated_per_bank} B per bank)",
                t0,
            )
            rep("our trace captured")

        def in_reserve(b):
            chain, st = b.chain, b.chain.st
            rep("fused L1 state, weights, step tensors (reserve open)")
            if fused:
                t0 = time.perf_counter()
                b.fe = chain.fused_ends = FusedEnds(mesh, chain)  # (on the chain: RS.device_tensors finds its tensors)
                b.fe.write_ids(b.g["input_ids"][chain.positions])
                ttnn.synchronize_device(mesh)
                mark(
                    "e3 the fused ends: the embedding table, the head's streams, sync words, id and logits tensors", t0
                )
                rep("the fused ends' tensors (reserve open)")
            else:
                b.programs = st.step_programs(audit=True)
            one_step()
            ttnn.synchronize_device(mesh)
            if not is_mock():
                b.first_out = b.check("the eager step (during the reserve)")
            rep("the eager step")
            st.hw.mark_per_step(
                st.a.rope, *([] if fused else [st.a.streams])
            )  # (written by the host before every step)
            if RESTORE_PROGRAM:
                b.restore_program = st.make_restore()  # (its DRAM image inside the reserved range)
                b.restore_program.launch()  # (compiled here, before our capture; the step is at rest: it changes nothing)
                ttnn.synchronize_device(mesh)
                logger.info(f"HY4_MK11 coexist trace [{FORM}]: {b.restore_program.summary()}")
                if not is_mock():  # (as a trace of its own: one command between a prefill replay and a decode replay)
                    b.restore_trace, _ = tracing.capture(mesh, b.restore_program.launch)
                    ttnn.synchronize_device(mesh)
                rep("the restore program made, launched once and captured")
            capture()  # (its temporaries inside the reserved range)

        rep("mesh open")
        b.after_prefill_capture = lambda: rep("prefill stack built, captured, slots filled")
        label = (
            f"coexist trace [{FORM}], {TC.LAYERS} layers, P{TC.PROMPT} [{TC.MOE_FIDELITY}]"
            + (f", append {TR.APPEND}" if TR.APPEND else "")
            + (", reader frames" if TR.READER_FRAMES else "")
            + (f", latent scale {TC.LATENT_SCALE}" if TC.LATENT_SCALE else "")
            + (f", pool of {POOL_FRAMES} frames" if POOL else "")
        )
        head, lm = TC.head_reference() if not is_mock() else (None, None)
        ends_seen = [0, []]

        def check(what):
            """One read of the step the device last ran: (streams after the last layer, logits, ids)."""
            chain, st, g, n = b.chain, b.chain.st, b.g, b.n
            J = chain.J
            gl, want = g["logits"][J].float(), g["cached_logits_argmax"][J]
            gold = g["layers"][n - 1]["streams_out"][J].float()
            ttnn.synchronize_device(mesh)
            if fused:
                streams, dev_logits, got = b.fe.read()
                # the detector of PLAN.md section 44: the device's logits against the reference tail on the step's
                # own streams, per chip's shard of the vocabulary
                tail = TC.host_logits(head, lm, streams)[1]
                off = ends.off_shards(dev_logits, tail)
                ends_seen[0] += 1
                if off:
                    ends_seen[1].append((what, off))
                    logger.warning(
                        f"HY4_MK11 {label} {what}: ENDS OFF on this step: chips' shards of the vocabulary that leave the reference tail on the step's own streams {off}"
                    )
                extra = (
                    f", vs the reference tail on the same streams {TC.pcc(tail, dev_logits):.6f} (max abs {float((tail - dev_logits).abs().max()):.3f}); ids = the "
                    f"first maximum of the device's own logits on {int((got == dev_logits.argmax(-1)).sum())} of 8; the HEAD's canary {b.fe.head.canary_summary()}"
                )
            else:
                streams = st.read_streams()
                _, dev_logits = TC.host_logits(head, lm, streams)
                got = dev_logits.argmax(-1)
                extra = ""
            status = st.status()
            logger.info(
                f"HY4_MK11 {label} {what}: streams after the last layer vs HF pcc={TC.pcc(gold, streams):.6f}; logits "
                f"({'device' if fused else 'the reference tail on the device streams'}) vs HF pcc={TC.pcc(gl, dev_logits):.6f}{extra}; "
                f"greedy token = HF's on {int((got == want).sum())} of 8 rows; status {status}"
            )
            assert not status, f"{what}: a status is left: {status}"
            assert bool(torch.isfinite(streams).all())
            if fused:
                assert not off, f"{what}: the device's logits leave the reference tail on the step's own streams: {off}"
                assert (
                    int((got == dev_logits.argmax(-1)).sum()) == 8
                ), f"{what}: the ids are not the first maximum of the device's logits"
                assert b.fe.head.canary_summary()["failed"] == 0, f"{what}: the HEAD's canary"
            return streams, dev_logits, got

        b.check = check
        build_both(b, mesh, monkeypatch, mark, during_reserve=in_reserve)
        rep("reserve closed, plugs freed")
        chain, st, prefill, tokens = b.chain, b.chain.st, b.prefill, b.tokens
        plan = st.hw.plan()
        logger.info(
            f"HY4_MK11 {label}: the registry: {st.hw.summary()}; L1 tensors of the Stack the host never wrote: "
            f"{sorted(st.hw.unregistered(b.ours_l1))}; writes into L1 from outside the fused model, not registered: {st.hw.refused}; "
            f"a restore on the device would zero {len(plan['zero'])} ranges ({plan['bytes']['zero']} B per chip), store "
            f"{len(plan['store'])} tensors ({plan['bytes']['store']} B per chip) and reset {plan['semaphores']} semaphores"
        )
        if fused:
            text, ok = b.fe.registry_lines(st.hw, plan)
            logger.info(f"HY4_MK11 {label}: the fused ends in the registry: {text}")
            assert ok, text

        # everything allocated since the prefill capture, per memory class, against what was declared for it: the rule
        # of PLAN.md section 24 as an assert (DRAM: the reserved range; L1: the registry's tensors and semaphores;
        # L1_SMALL: nothing)
        base = reports.get("prefill stack built, captured, slots filled") or reports["mesh open"]
        now = RS.report(mesh, "", log=False)
        declared = set(st.hw.live()) | {
            int(ttnn.get_global_semaphore_address(sem)) for sem, _, _, _ in st.hw.semaphores
        }
        late = RS.late_allocations(base, now, (b.reserve.address, b.reserve.address + b.reserve.size), declared)
        logger.info(
            f"HY4_MK11 {label}: alive now and allocated after the prefill capture, OUTSIDE what was declared for it (DRAM: the "
            f"reserved range; L1: the registry's tensors and semaphores, {len(declared)} in all; L1_SMALL: nothing): {late}; "
            f"L1_SMALL buffers now {len(now['L1_SMALL'])}, at the prefill capture {len(base['L1_SMALL'])}"
        )
        assert not any(late.values()), f"allocated after the first capture outside what was declared: {late}"
        if is_mock():
            logger.info(f"HY4_MK11 {label} mock: restore wrote (tensors, bytes per chip) {st.restore()}")
            if (
                TR.APPEND and TR.READER_FRAMES
            ):  # (the host side of a rebinding: the tables and the restore image in one call)
                g_ = st.a.append_geometry()
                st.set_frame_tables([v for u in reversed(range(TR.USERS)) for v in g_.identity_table(u)])
                assert not any(r.stale() for r in getattr(st, "restores", []))
            return

        assert DECODES == 1 or fused, "several decode replays back to back: the form with the ends"
        exact = bool(
            st.a.fixed_order
        )  # (the arrival order of X4's partials makes two replays differ: PLAN.md section 15)

        def decode():
            """ms per decode replay: one blocking replay, or DECODES replays enqueued back to back and one wait."""
            if DECODES > 1:
                return tracing.replay_ms(mesh, b.trace_id, DECODES)
            t = time.perf_counter()
            ttnn.execute_trace(mesh, b.trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            return (time.perf_counter() - t) * 1e3

        def pool_rows_check(what, fresh):
            """After decode replays on the pool: the rows that changed in the POOL lie only at the decode positions IN
            THE FRAMES THE STORE'S MANAGER MAPS THEM TO (frame id from the manager against where a changed row was
            found), and the bench (the stack's own slots, the workbench among them) did not change. ``fresh``: the
            decode rows whose rows were prefill-written before (their appended rows must all have changed; a row our
            own step wrote before may come out bit-equal again)."""
            engine, store, cfg, layout = b.engine, b.store, chain.cfg, b.layout
            now_rows = TR.all_rows(mesh, b.kv, cfg)  # [chips, frames * layers, 256, width]
            differ = (b.pool_rows != now_rows).any(dim=-1)
            found = {tuple(i) for i in differ.nonzero().tolist()}
            want, must = set(), set()
            for u, p in enumerate(chain.positions):
                frames = store.conversation(store.binding[u]).frames
                frame = frames[p // engine.store.frame_tokens]
                for layer in range(b.n):
                    want.add((layout.chip(p), frame * b.n + layer, layout.slot(p) % 256))
                    if u in fresh:
                        must.add((layout.chip(p), frame * b.n + layer, layout.slot(p) % 256))
            bench_now = storage(mesh, b.bridge)["kvpe"]
            bench_diff = int((bench_now != b.base["bench"]).sum())
            logger.info(
                f"HY4_MK11 {label} pool after {what}: rows of the pool that changed {len(found)} of {differ.numel()}; the decode "
                f"positions' rows in the frames the manager maps them to: {len(want)}; changed rows outside them: "
                f"{sorted(found - want)[:6]}; rows of decode rows {sorted(fresh)} (prefill-written before) that did not change: "
                f"{sorted(must - found)[:6]}; frames by decode row "
                f"{[list(store.conversation(store.binding[u]).frames) for u in range(TR.USERS)]}; bytes of the bench that changed: "
                f"{bench_diff}"
            )
            assert found <= want and must <= found, (sorted(found - want)[:6], sorted(must - found)[:6])
            assert bench_diff == 0, "decode wrote the bench"
            b.pool_rows, b.base["bench"] = now_rows, bench_now

        def inputs():
            TR.set_rope(chain)
            if TR.APPEND:
                st.a.set_positions(chain.positions)
            if not fused:
                chain.reset_streams()
            ttnn.synchronize_device(mesh)

        if POOL:
            assert TR.APPEND and TR.READER_FRAMES, "the pool: HY4_REAL_APPEND and HY4_REAL_READER_FRAMES"
        inputs()
        ms_a = decode()
        ref = check(f"decode replay A ({DECODES} back to back, no prefill pass before)")
        streams_a, logits_a, tokens_a = ref
        eager = same_rows(b.first_out, ref)
        logger.info(
            f"HY4_MK11 {label}: decode replay A against the eager step (the same step, untraced): rows equal in every bit of streams, logits and id {eager}"
        )
        rep("decode replay A")
        per_round = []
        singles = [0, 0]
        if POOL:
            pool_rows_check("the eager step and decode replay A", fresh=set(range(TR.USERS)))
        for r in range(ROUNDS):
            rebuilt = set()  # decode rows whose cache rows were made again by another prefill since replay A
            if POOL:
                # a request with a REBINDING: conversation 1 leaves its decode row, is evicted, another conversation takes
                # frames, conversation 1 is created again (other frames), prefilled through the engine (copy-in, passes by
                # replay in the workbench, copy-out to the pool) and bound again; its table row changes
                engine, store = b.engine, b.store
                old = list(store.conversation(1).frames)
                store.unbind(1)
                engine.end(1)
                engine.evict(1)
                extra = engine.begin(100 + r, tokens)
                rec = engine.begin(1, tokens, max_new=b.steps)
                assert extra is not None and rec is not None, "no room in the pool for the rebinding"
                store.bind(1, 1)
                # (no engine.sync_tables() here: the engine's own two table tensors are read by no kernel of this form,
                # and a host write into them now would enter the registry as state of the step)
                new = list(store.conversation(1).frames)
                t = time.perf_counter()
                st.set_frame_tables(
                    store.table(engine.pool_g[engine.latent])
                )  # (the tables and the restore image in one call)
                ms_tables = (time.perf_counter() - t) * 1e3
                p = rec["passes"][-1]
                logger.info(
                    f"HY4_MK11 {label} round {r + 1}: conversation 1 rebound: frames {old} -> {new} (another conversation took "
                    f"{list(store.conversation(100 + r).frames)}); the request through the engine: {len(rec['passes'])} pass(es) "
                    f"{pass_text(p)} ms, copy-in {1e3 * rec['copy_in_s']:.1f} ms, copy-out {1e3 * rec['copy_out_s']:.1f} ms; the new "
                    f"tables into both table tensors and the restore image refreshed: {ms_tables:.1f} ms from the host"
                )
                assert new != old, "the rebinding gave conversation 1 its old frames: the tables did not change"
                b.pool_rows = TR.all_rows(mesh, b.kv, chain.cfg)  # (the pool as the request left it)
                b.base["bench"] = storage(mesh, b.bridge)["kvpe"]
                rebuilt = {1}
            else:
                p = prefill(1, 0, len(tokens), tokens)
                ttnn.synchronize_device(mesh)
            logger.info(
                f"HY4_MK11 {label} round {r + 1}: the prefill pass by replay after our replay: {pass_text(p)} ms"
            )
            t = time.perf_counter()
            if RESTORE_PROGRAM:
                ttnn.execute_trace(mesh, b.restore_trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                count, nbytes = (
                    len(b.restore_program.stamp),
                    b.restore_program.counts["zero_bytes"] + b.restore_program.counts["store_bytes"],
                )
            else:
                count, nbytes = st.restore()
            ms_restore = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            bad = st.hw.verify(mesh, ranges_only=RESTORE_PROGRAM)
            ms_verify = (time.perf_counter() - t) * 1e3
            assert (
                not bad
            ), f"round {r}: registered tensors that are not what the host last wrote after the restore: {bad[:8]}"
            t = time.perf_counter()
            inputs()
            ms_inputs = (time.perf_counter() - t) * 1e3
            ms_b = decode()
            now_b = check(f"{DECODES} decode replay(s) after prefill pass {r + 1}")
            streams_b, logits_b, tokens_b = now_b
            same = same_rows(ref, now_b)
            for i in range(SINGLES):  # (every replay that is counted on is read: PLAN.md section 45)
                ttnn.execute_trace(mesh, b.trace_id, cq_id=0, blocking=False)
                got = check(f"single decode replay {i + 1} of round {r + 1}")
                singles[0] += 1
                singles[1] += int(not all(s or u in rebuilt for u, s in enumerate(same_rows(ref, got))))
            per_round.append((p["chunk_s"] * 1e3, ms_restore, ms_inputs, ms_b))
            if POOL:
                pool_rows_check(f"the decode replay(s) of round {r + 1}", fresh={1})
            line = (
                f"prefill pass {pass_text(p)} ms, restore [{'ONE program on the device, launched from the host' if RESTORE_PROGRAM else 'host writes'}] "
                f"{count} tensors / {nbytes} B per chip {ms_restore:.1f} ms (read back "
                f"and compared in {ms_verify:.0f} ms: all equal), per-step inputs from the host {ms_inputs:.1f} ms, {DECODES} decode "
                f"replay(s) {ms_b:.3f} ms each; vs replay A: rows equal in every bit of streams, logits and id {same}"
                + (
                    f" (rows {sorted(rebuilt)}: their cache rows were made again by another prefill, in other frames)"
                    if rebuilt
                    else ""
                )
                + f", streams pcc {TC.pcc(streams_a, streams_b):.6f}, logits pcc {TC.pcc(logits_a, logits_b):.6f}, "
                f"tokens equal {int((tokens_a == tokens_b).sum())} of 8"
            )
            logger.info(f"HY4_MK11 {label} round {r + 1}: {line}")
            assert int((tokens_a == tokens_b).sum()) == 8, line
            if exact:
                # the same step on the same rows: the same bits. (A row whose cache rows another prefill made again
                # is named and not asserted: only its token, above.)
                assert all(s for u, s in enumerate(same) if u not in rebuilt), line
            else:
                assert TC.pcc(logits_a, logits_b) > 0.9995, line
        if SINGLES:
            logger.info(
                f"HY4_MK11 {label} SINGLES: {singles[0]} single decode replays read, {singles[1]} with a row that differs from replay A in any bit"
            )
            assert not exact or singles[1] == 0
        rep("after the rounds")
        if RESTORE_PROGRAM:
            # the restore's cost where it will be used: enqueued between a prefill replay and a decode replay
            n = 20
            t0 = tracing.sync(mesh)
            for _ in range(n):
                b.restore_program.launch()
            eager_us = 1e6 * (tracing.sync(mesh) - t0) / n
            traced_us = 1e3 * tracing.replay_ms(mesh, b.restore_trace, n)
            alone_ms = tracing.replay_ms(mesh, b.trace_id, n) if fused else None
            pair_ms = None
            if fused:
                t0 = tracing.sync(mesh)
                for _ in range(n):
                    ttnn.execute_trace(mesh, b.restore_trace, cq_id=0, blocking=False)
                    ttnn.execute_trace(mesh, b.trace_id, cq_id=0, blocking=False)
                pair_ms = 1e3 * (tracing.sync(mesh) - t0) / n
                last = check("after the restore timings")
                assert (
                    not exact or POOL or all(same_rows(ref, last))
                ), "the step behind a restore replay differs from replay A"
            logger.info(
                f"HY4_MK11 {label} RESTORE as a program: {n} launches back to back from the host {eager_us:.0f} us each; as a trace "
                f"of its own, {n} replays back to back {traced_us:.0f} us each; "
                + (
                    f"decode replay alone {alone_ms:.3f} ms, [restore replay + decode replay] {pair_ms:.3f} ms: the restore in place costs "
                    f"{1e3 * (pair_ms - alone_ms):.0f} us"
                    if fused
                    else ""
                )
            )
        if fused:
            logger.info(
                f"HY4_MK11 {label} ENDS: steps read through the device's ends {ends_seen[0]}, off {len(ends_seen[1])}: {ends_seen[1]}"
            )
        after = storage(mesh, b.bridge)
        if POOL:
            b.base, after = {"kvpe": b.base["bench"]}, {"kvpe": after["kvpe"]}
        canary = st.canary_summary()
        col = lambda i: [round(v[i], 3) for v in per_round]
        logger.info(
            f"HY4_MK11 {label} ROUNDS: {ROUNDS} x [prefill pass, restore, per-step inputs, {DECODES} decode replays back to back]; "
            f"ms per round: prefill pass {col(0)}, restore {col(1)}, inputs {col(2)}, per decode replay {col(3)}; per decode replay "
            f"before any prefill pass {ms_a:.3f} (6 layers on the prefill stack's rows, the logits kept: a regression figure, not the step's time of record)"
        )
        logger.info(
            f"HY4_MK11 {label}: decode replay A {ms_a:.3f} ms ({'back to back' if DECODES > 1 else 'blocking, with the host wait'}); the prefill stack's caches after "
            f"{ROUNDS} passes between our replays against before anything of ours existed: {slot_differences(b.base, after, TR.USERS)}; "
            f"canary since the last restore {canary}; phases {marks}"
        )
        for kind, v in canary.items():
            assert v["found"] == 0 and v["timeouts"] == 0, f"canary {kind}: {v}"
    finally:
        if b.trace_id is not None:
            ttnn.release_trace(mesh, b.trace_id)
        if getattr(b, "restore_trace", None) is not None:
            ttnn.release_trace(mesh, b.restore_trace)
        if getattr(b, "restore_program", None) is not None:
            b.restore_program.free()
        free_both(b)
