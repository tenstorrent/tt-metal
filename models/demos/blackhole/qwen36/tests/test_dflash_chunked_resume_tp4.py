# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""D1 gate of the DFlash2 scheduler-driven chunked prefill (QWEN36_DFLASH_CHUNKED_PREFILL, profiles/tp4_chunked/DESIGN.md
section 9.2): a prompt prefilled in chunks -- with riders, rider-only calls, live speculative sessions stepping in both
verify buckets and multi-chunk resume calls in between -- must leave EXACTLY the state the unchunked prefill leaves:

  * the anchor logits (host row, torch.equal);
  * the durable GDN row of the slot: recurrent state and every conv tap, every GDN layer, every device;
  * the drafter ring of the slot over the readable window [max(0, L-2048), L) (every drafter layer, K and V, every
    device) and
    ``dec.ctx_len``;
  * the paged target K/V over [0, L) (first and last full-attention layer);
  * after ``begin``, the first MAX_NEW committed tokens of the chunked slot equal the unchunked slot's.

RIDER cases (review F2): a short prompt prefilled WHILE a partial is held (a rider-only call with the partial parked,
the plugin's oversized-rider step; or a rider sharing a call with the partial's intermediate chunk, the served chunk
step) must leave exactly the state the same prompt leaves when no partial is held (slots RREF / RCAND, same
snapshot and host-copy comparison). A preempted request's replay reaches the model as exactly such a prompt (prompt +
outputs as prompt ids, admitted whole). A second NEGATIVE CONTROL (the rider's GDN reset skipped, so it starts from the
partial's state) must be DETECTED.

The REFERENCE (unchunked, today's path) and the CANDIDATE (chunked) run in DIFFERENT physical slots (1 and 6) with
disjoint page-table rows, and are compared through host copies, so a candidate that skips its slot write or writes
the wrong slot cannot pass on the reference's leftover bytes (review M1). The live sessions' committed tokens must
equal a replay without the interleaved chunks (interleave invariance), the program cache must not grow after the
spec captures (the resume / park programs compile in the pre-capture warm sequence, as in the server), and the
scratch ownership must be clear after the warm-up and at the end. A NEGATIVE CONTROL (the unpark after a rider
disabled) must be DETECTED as a mismatch (review M2).

Everything runs through the serving class's own orchestration (Qwen36DFlashForCausalLM._prefill_planned and
_spec_prefill, the ChunkedPrefillPlanner, park / unpark), on a Qwen36DFlashForCausalLM shell over a directly built
model + DFlash2DualBucketDecoder (buckets 8x4, 4x8; the batch8-dflash2 geometry).

Run (from the tree, the JIT resolves kernel sources from the CWD):
  MESH_DEVICE=P150x4 QWEN36_GDN_SPEC_FUSED=1 QWEN36_DFLASH_FOLD_SEED=1 QWEN36_DFLASH_CHUNKED_PREFILL=1 \
    pytest models/demos/blackhole/qwen36/tests/test_dflash_chunked_resume_tp4.py -x -q -s
Knobs: D1_LONG (default 32768; the long case's prompt length), D1_SKIP_NEGATIVE=1.
"""

import gc
import glob
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_spec_batched import _batch_prompts
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

S = 8
BUCKETS = ((8, 4), (4, 8))
C = 2048
REF, CAND = 1, 6  # physical slots of the unchunked reference and the chunked candidate
LIVE = (0, 2, 3)  # slots of the live speculative sessions
RIDER = 4  # slot of the short riders
RREF, RCAND = 5, 7  # slots of the rider-state cases: rider with no partial held / rider while a partial is held
MAX_NEW = 48
L_LONG = int(os.environ.get("D1_LONG", "32768"))
BPU_BIG = ((-(-(L_LONG + 2 * MAX_NEW + 64) // BLOCK_SIZE)) + 31) // 32 * 32
BPU_SMALL = 16
BPU_MID = 48  # rider-state slots: prompts up to 3072 tokens


def _tables():
    """Per-slot page-table rows, disjoint, all BPU_BIG wide (the verify trace's row width): slots REF / CAND own
    BPU_BIG blocks each, the others BPU_SMALL blocks (zero-padded, as vLLM pads)."""
    rows, nxt = [], 0
    for u in range(S):
        n = BPU_BIG if u in (REF, CAND) else (BPU_MID if u in (RREF, RCAND) else BPU_SMALL)
        r = torch.zeros(BPU_BIG, dtype=torch.int32)
        r[:n] = torch.arange(nxt, nxt + n, dtype=torch.int32)
        nxt += n
        rows.append(r)
    return torch.stack(rows), nxt


def _long_ids(tokenizer, n, k):
    """n token ids of real text (tech_reports markdown; a different offset per k)."""
    root = os.environ.get("TT_METAL_HOME", ".")
    text = ""
    for f in sorted(glob.glob(os.path.join(root, "tech_reports", "**", "*.md"), recursive=True)):
        text += open(f, errors="ignore").read() + "\n\n"
        if len(text) > 8 * (n + 4096):
            break
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    off = (k * 7919) % max(1, len(ids) - n)
    out = ids[off : off + n]
    assert len(out) == n, f"corpus too short for {n} tokens"
    return out


def _programs(device):
    ttnn.synchronize_device(device)
    return int(device.num_program_cache_entries())


def _all_dev(t):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


def _gdn_row(model, slot):
    """Durable batched GDN row ``slot`` of every GDN layer, every device: (rec [Nv,Dk,Dv], [K taps [D]])."""
    out = []
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        dn = layer.attention
        rec = [r[slot].clone() for r in _all_dev(dn.rec_state)]
        taps = [[c.reshape(-1, c.shape[-1])[slot].clone() for c in _all_dev(t)] for t in dn.conv_states]
        out.append((rec, taps))
    return out


def _ring(dec, phys, L):
    """Drafter ring K/V rows of slot ``phys`` for the positions a later query can read, [max(0, L-window), L) with
    window = ring - 2 blocks (the sliding window; every drafter layer and device). The ring's 2 extra blocks are
    scratch: an ingested-but-not-begun slot seated in the identity bucket drafts a dummy block at its frontier
    (positions L.., dflash2_serving._draft_inputs), which wraps onto ring slots of positions [L-ring, L-window) --
    in unchunked serving too -- and no query reads them again."""
    d = dec.drafter
    bs, ring = int(d.block_size), int(d.ring)
    window = ring - 2 * bs
    pos = torch.arange(max(0, L - window), L)
    slots = pos % ring
    blocks = d.phys_tables[phys][slots // bs].long()
    rows = (slots % bs).long()
    out = []
    for kc, vc in d.kv:
        ks, vs = _all_dev(kc), _all_dev(vc)
        out.append(([k[blocks, :, rows].clone() for k in ks], [v[blocks, :, rows].clone() for v in vs]))
    return out


def _kv(model, pt_row, L):
    """Target paged K and V over [0, L) of the first and last full-attention layer, every device."""
    atts = [layer.attention for layer in model.layers if layer.is_full_attention]
    pos = torch.arange(L)
    blocks = pt_row[pos // BLOCK_SIZE].long()
    rows = (pos % BLOCK_SIZE).long()
    out = []
    for att in (atts[0], atts[-1]):
        for cache in (att.paged_k, att.paged_v):
            out.append([x[blocks, :, rows].clone() for x in _all_dev(cache)])
    return out


def _snap(model, dec, slot, pt_row, L, logits):
    return {
        "logits": logits.clone(),
        "ctx": int(dec.ctx_len[slot]),
        "gdn": _gdn_row(model, slot),
        "ring": _ring(dec, slot, L),
        "kv": _kv(model, pt_row, L),
    }


def _diff(a, b):
    """First difference between two snapshots (None when bit-identical)."""
    if not torch.equal(a["logits"], b["logits"]):
        return f"anchor logits (max |d| {float((a['logits'] - b['logits']).abs().max()):.3e})"
    if a["ctx"] != b["ctx"]:
        return f"ctx_len {a['ctx']} != {b['ctx']}"
    for li, ((ra, ta), (rb, tb)) in enumerate(zip(a["gdn"], b["gdn"])):
        for d, (x, y) in enumerate(zip(ra, rb)):
            if not torch.equal(x, y):
                return f"GDN layer {li} rec (device {d}, max |d| {float((x - y).abs().max()):.3e})"
        for m, (xs, ys) in enumerate(zip(ta, tb)):
            for d, (x, y) in enumerate(zip(xs, ys)):
                if not torch.equal(x, y):
                    return f"GDN layer {li} conv tap {m} (device {d})"
    for li, ((ka, va), (kb, vb)) in enumerate(zip(a["ring"], b["ring"])):
        for d in range(len(ka)):
            if not (torch.equal(ka[d], kb[d]) and torch.equal(va[d], vb[d])):
                return f"drafter ring layer {li} (device {d})"
    for i, (xs, ys) in enumerate(zip(a["kv"], b["kv"])):
        for d, (x, y) in enumerate(zip(xs, ys)):
            if not torch.equal(x, y):
                return f"target KV tensor {i} (device {d})"
    return None


class _Shell:
    """The serving class's orchestration over a directly built model + decoder (no vLLM)."""

    def __init__(self, model, dec):
        from models.demos.blackhole.qwen36.tt import qwen36_vllm_dflash as D

        o = D.Qwen36DFlashForCausalLM.__new__(D.Qwen36DFlashForCausalLM)
        o.model = [model]
        o._spec = dec
        o._in_warmup = False
        o._B = S
        o._phys = list(range(S))
        o._pending = [None] * S
        o._carry = [[] for _ in range(S)]
        o._stopped = [False] * S
        o._prev_tail = [None] * S
        o._cp_on = True
        o._cp_owner_phys = None
        o._forbid_plain = False
        self.o, self.model, self.dec = o, model, dec

    def call(self, rows):
        """One prefill call: rows = [(prompt_ids, pt_row, phys, start, end, resume, final)]. Returns logits by row."""
        prompts = [torch.tensor([p], dtype=torch.int32) for p, *_ in rows]
        out = self.o._prefill_planned(
            self.model,
            self.dec,
            prompts,
            [r[1] for r in rows],
            [r[2] for r in rows],
            [r[3] for r in rows],
            [r[4] for r in rows],
            [r[5] for r in rows],
            [r[6] for r in rows],
            seat=False,
        )
        return [x.reshape(1, -1) for x in out]

    def owner_clear(self):
        return self.model._chunked_prefill_planner().owner is None and self.o._cp_owner_phys is None


def _chunk_calls(ids, pt_row, phys, bounds, riders=()):
    """The candidate's prefill calls for chunk ``bounds`` [(start, end)...]; ``riders`` = {call index: [(ids, pt,
    phys)]} rides in that call (a rider-only call is ``bounds`` entry None)."""
    L = len(ids)
    calls = []
    for k, b in enumerate(bounds):
        rows = []
        if b is not None:
            s, e = b
            rows.append((ids, pt_row, phys, s, e, s > 0, e == L))
        for r_ids, r_pt, r_phys in dict(riders).get(k, []):
            rows.append((r_ids, r_pt, r_phys, 0, len(r_ids), False, True))
        calls.append(rows)
    return calls


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dflash_chunked_prefill_equals_unchunked(mesh_device, monkeypatch):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    if os.environ.get("QWEN36_GDN_SPEC_FUSED", "0") != "1":
        pytest.skip("the dual-bucket decoder runs the fused GDN verify: set QWEN36_GDN_SPEC_FUSED=1")
    monkeypatch.setenv("QWEN36_DFLASH_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("QWEN36_DFLASH_BUCKET_DOWN_STEPS", "1000000")  # buckets change only when the test says so
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.dflash2_serving import DFlash2DualBucketDecoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=S, max_seq_len=BPU_BIG * BLOCK_SIZE)
    assert len(model.layers) >= 62, "the DFlash2 taps live at layers 5..61: run the full model"
    model.set_gdn_fused_decode(True)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    short = _batch_prompts(S, tokenizer)  # 130 .. 255 tokens
    pt, n_blocks = _tables()
    model.free_kv_caches()
    model.allocate_kv_caches(
        [n_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=S
    )
    model.ensure_gdn_park_buffer()
    dec = DFlash2DualBucketDecoder(model, num_blocks=BPU_BIG, buckets=BUCKETS)
    ids = {bt: bid for bid, b in dec.buckets.items() for bt in [(int(b.B), int(b.T))]}
    BIG, SMALL = ids[(8, 4)], ids[(4, 8)]
    sh = _Shell(model, dec)
    S0 = len(short[0])
    try:
        # ---- server-order warm-up: alloc, eager warm, the resume/park warm sequence, capture, drafter traces ----
        dec.alloc()
        dec.warm(warm_position=S0 + 1)
        sh.o._cp_warm_sequence(model, dec, [pt[REF], pt[CAND]], 128, "D1 warm-up")
        assert sh.owner_clear()
        for u in range(1, S):  # every other slot once (per-slot GDN write / drafter fill programs), as _spec_prepare
            sh.call([(short[u], pt[u], u, 0, len(short[u]), False, True)])
        first_w = int(sh.call([(short[0], pt[0], 0, 0, S0, False, True)])[0].argmax())
        dec.capture(warm_position=S0 + 1)
        dec.plan({0})
        dec.begin(0, first_w, S0, pt[0])
        for _ in range(2):
            dec.step()
        dec.set_bucket(SMALL)
        for _ in range(2):
            dec.step()
        dec.set_bucket(BIG)
        dec.step()
        dec.end(0)
        ttnn.synchronize_device(device)
        n0 = device.num_program_cache_entries()
        logger.info(f"[D1] traces captured; program cache {n0} entries")

        results = {}
        programs = {}

        def run_case(name, L, bounds, riders=(), live=None, bucket=None, negative=False, check_decode=False):
            """REF: unchunked into slot REF. CAND: the chunk calls into slot CAND, with live sessions (slots LIVE[:live])
            stepping twice between calls in ``bucket``; their committed tokens must equal a replay without the calls."""
            p = _long_ids(tokenizer, L, len(results))
            n_a = _programs(device)
            lr = sh.call([(p, pt[REF], REF, 0, L, False, True)])[0]
            n_ref = _programs(device)
            ref = _snap(model, dec, REF, pt[REF], L, lr)
            calls = _chunk_calls(p, pt[CAND], CAND, bounds, riders)

            def live_run(with_calls):
                outs = {}
                lc = None
                n_live = live or 0
                slots = LIVE[:n_live]
                if n_live:
                    for u in slots:
                        f = int(sh.call([(short[u], pt[u], u, 0, len(short[u]), False, True)])[0].argmax())
                        if bucket is None:
                            dec.plan({v for v in range(S) if dec.active[v]} | {u})
                        dec.begin(u, f, len(short[u]), pt[u])
                        outs[u] = [f]
                    if bucket is not None and dec.cur_id != bucket:
                        dec.set_bucket(bucket)
                todo = list(calls) if with_calls else []
                n_steps = 2 * len(calls) + 2
                for i in range(n_steps if n_live else 0):
                    if i % 2 == 0 and todo:
                        r = sh.call(todo.pop(0))
                        if r and todo == [] and with_calls:
                            lc = r[0]
                    com = dec.step()
                    for u, t in com.items():
                        outs[u].extend(int(x) for x in t)
                while todo:  # no live sessions: the calls back to back
                    r = sh.call(todo.pop(0))
                    lc = r[0]
                for u in slots:
                    dec.end(u)
                return outs, lc

            if negative:
                monkeypatch.setattr(model, "_unpark_gdn_scratch", lambda: None)
            try:
                outs_c, lc = live_run(True)
                n_cand = _programs(device)
            finally:
                if negative:
                    monkeypatch.undo()
                    monkeypatch.setenv("QWEN36_DFLASH_CHUNKED_PREFILL", "1")
                    monkeypatch.setenv("QWEN36_DFLASH_BUCKET_DOWN_STEPS", "1000000")
            if negative:
                # the stale owner (the planner committed it) is dead now; clear it for the next case
                model._chunked_prefill_planner().owner = None
                sh.o._cp_owner_phys = None
            cand = _snap(model, dec, CAND, pt[CAND], L, lc)
            d = _diff(ref, cand)
            # The chunked candidate must not need a program the unchunked reference (today's path) did not: the
            # program-cache growth over the candidate run (rider / live sessions pre-warmed) must be 0.
            programs[name] = (n_ref - n_a, n_cand - n_ref)
            logger.info(
                f"[D1] {name}: programs compiled by the unchunked ref {n_ref - n_a}, by the chunked run {n_cand - n_ref}"
            )
            if negative:
                assert d is not None, f"[D1] NEGATIVE CONTROL {name}: a resume without its unpark was NOT detected"
                logger.info(f"[D1] negative control {name}: mismatch detected as required ({d})")
                results[name] = f"negative: detected ({d})"
                return
            assert d is None, f"[D1] {name}: chunked prefill differs from unchunked: {d}"
            if live:
                outs_r, _ = live_run(False)
                for u in outs_c:
                    n = min(len(outs_c[u]), len(outs_r[u]))
                    assert n >= 3 and outs_c[u][:n] == outs_r[u][:n], (
                        f"[D1] {name}: live slot {u}'s tokens changed with the chunks interleaved: "
                        f"{outs_c[u][:n]} vs {outs_r[u][:n]}"
                    )
            if check_decode:
                toks = {}
                for slot, lg in ((REF, lr), (CAND, lc)):
                    f = int(lg.argmax())
                    dec.plan({slot})
                    dec.begin(slot, f, L, pt[slot])
                    out = [f]
                    while len(out) < MAX_NEW:
                        out.extend(int(x) for x in dec.step()[slot])
                    dec.end(slot)
                    toks[slot] = out[:MAX_NEW]
                assert toks[REF] == toks[CAND], f"[D1] {name}: decode after begin differs: {toks}"
            assert sh.owner_clear(), f"[D1] {name}: scratch owner left behind"
            results[name] = (
                "bit-identical" + (" + live invariant" if live else "") + (" + decode" if check_decode else "")
            )
            logger.info(f"[D1] {name}: {results[name]}")

        rider = (short[5], pt[RIDER], RIDER)
        # (a) one chunk + a 1-token tail, a rider-only call between (park / unpark)
        run_case("L2049_rider_only_call", 2049, [(0, C), None, (C, 2049)], riders={1: [rider]})
        # (b) exact multiple; a rider shares the final (resume) call
        run_case("L4096_rider_in_resume_call", 4096, [(0, C), (C, 2 * C)], riders={1: [rider]})
        # (c) a 4096-token scheduler chunk (two 2048 sub-chunks in one intermediate call) + a 1-token tail
        run_case("L4097_chunk4096", 4097, [(0, 2 * C), (2 * C, 4097)])
        # (d) one chunk, then the remainder in ONE resume call (the last decoder left), 3 live sessions in 4x8
        run_case(
            "L6200_multichunk_resume_live3_4x8",
            6200,
            [(0, C), (C, 6200)],
            live=3,
            bucket=SMALL,
            check_decode=True,
        )
        # (e) chunk by chunk with a rider, 2 live sessions in the IDENTITY bucket 8x4 (the partial's row holds)
        run_case(
            "L6200_chunks_rider_live2_8x4",
            6200,
            [(0, C), (C, 2 * C), (2 * C, 3 * C), (3 * C, 6200)],
            riders={2: [rider]},
            live=2,
            bucket=BIG,
        )
        # (f) the long prompt: 2048 chunks, riders in two calls, 1 live session in 4x8
        nfull = L_LONG // C
        bounds = [(k * C, (k + 1) * C) for k in range(nfull)]
        if L_LONG % C:
            bounds.append((nfull * C, L_LONG))
        run_case(
            f"L{L_LONG}_chunks_riders_live1",
            L_LONG,
            bounds,
            riders={3: [rider], len(bounds) - 1: [rider]},
            live=1,
            bucket=SMALL,
            check_decode=True,
        )

        def run_rider_case(name, R, L, mode, negative=False):
            """Review F2: rider state while a partial is held == the same prompt with no partial held. REF: the rider
            alone into RREF (no scratch owner). CAND: a long prompt's first chunk into CAND, then the rider into RCAND
            in a rider-only call (``mode`` "only": the partial parked) or next to the partial's intermediate chunk
            (``mode`` "with_chunk": park after that chunk); the partial then finishes. Compared: the rider's anchor
            logits, GDN row, drafter ring window, ctx_len and target KV."""
            assert sh.owner_clear(), f"[D1] {name}: scratch owner before the case"
            p_r = _long_ids(tokenizer, R, 100 + len(results))
            p_l = _long_ids(tokenizer, L, 200 + len(results))
            n_a = _programs(device)
            lr = sh.call([(p_r, pt[RREF], RREF, 0, R, False, True)])[0]
            n_ref = _programs(device)
            ref = _snap(model, dec, RREF, pt[RREF], R, lr)
            sh.call([(p_l, pt[CAND], CAND, 0, C, False, False)])
            rider_row = (p_r, pt[RCAND], RCAND, 0, R, False, True)
            if mode == "only":
                rider_call, rest = [rider_row], [(C, L)]
            else:
                rider_call, rest = [(p_l, pt[CAND], CAND, C, 2 * C, True, False), rider_row], [(2 * C, L)]
            if negative:
                monkeypatch.setattr(model, "_reset_gdn_state_for_new_sequence", lambda: None)
            try:
                lc = sh.call(rider_call)[-1]
            finally:
                if negative:
                    monkeypatch.undo()
                    monkeypatch.setenv("QWEN36_DFLASH_CHUNKED_PREFILL", "1")
                    monkeypatch.setenv("QWEN36_DFLASH_BUCKET_DOWN_STEPS", "1000000")
            n_cand = _programs(device)
            cand = _snap(model, dec, RCAND, pt[RCAND], R, lc)
            for s_, e_ in rest:
                sh.call([(p_l, pt[CAND], CAND, s_, e_, True, e_ == L)])
            assert sh.owner_clear(), f"[D1] {name}: scratch owner left behind"
            d = _diff(ref, cand)
            programs[name] = (n_ref - n_a, n_cand - n_ref)
            logger.info(
                f"[D1] {name}: programs compiled by the rider alone {n_ref - n_a}, by the rider next to a partial "
                f"{n_cand - n_ref}"
            )
            if negative:
                assert d is not None, f"[D1] NEGATIVE CONTROL {name}: a rider without its GDN reset was NOT detected"
                logger.info(f"[D1] negative control {name}: mismatch detected as required ({d})")
                results[name] = f"negative: detected ({d})"
                return
            assert d is None, f"[D1] {name}: the rider's state differs with a partial held: {d}"
            results[name] = "rider bit-identical"
            logger.info(f"[D1] {name}: {results[name]}")

        # (g) rider state while a partial is held: a 600-token rider in a rider-only call (partial parked); a 2048-token
        #     rider (one full chunk, the largest short) next to the partial's intermediate chunk; a 2600-token whole
        #     prompt in a rider-only call (model level: the plugin admits a long replay only with no partial in flight)
        run_rider_case("RIDER_600_only_call", 600, 4200, "only")
        run_rider_case("RIDER_2048_with_chunk", 2048, 6200, "with_chunk")
        run_rider_case("RIDER_2600_only_call", 2600, 4200, "only")
        # negative controls: the rider's reset is not undone (no unpark) -> must be detected; a rider that skips its own
        # GDN reset (starts from the partial's state) -> must be detected
        if os.environ.get("D1_SKIP_NEGATIVE", "0") != "1":
            run_case("NEG_L4096_no_unpark", 4096, [(0, C), None, (C, 2 * C)], riders={1: [rider]}, negative=True)
            run_rider_case("NEG_RIDER_2048_no_reset", 2048, 6200, "with_chunk", negative=True)

        ttnn.synchronize_device(device)
        n1 = device.num_program_cache_entries()
        logger.info(f"[D1] results {results}; programs (ref, chunked) per case {programs}; cache {n0} -> {n1}")
        bad = {k: v for k, v in programs.items() if v[1] and not k.startswith("NEG")}
        assert not bad, f"[D1] the chunked runs compiled programs the unchunked reference did not: {bad}"
        if n1 != n0 and os.environ.get("D1_STRICT_CACHE", "0") == "1":
            raise AssertionError(f"[D1] {n1 - n0} program(s) compiled after the spec captures (cache {n0} -> {n1})")
    finally:
        dec.release()
        model.free_kv_caches()
    dec = model = None  # noqa: F841 (drop the references before gc)
    gc.collect()
