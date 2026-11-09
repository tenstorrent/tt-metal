# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Production-loop harness for QWEN36_SPEC_ADAPTIVE_K (tt/spec_adaptive.py): the deferred chained fused loop vLLM's async
scheduler drives (decode_forward(defer_read=True) of step n+1, chained on step n, BEFORE step n is read; then the reader's
handle.wait() and propose_draft_tokens(hidden=handle)), with REAL prefills of real text so acceptance means something.

The block is [B, 1 + K_api] (K_api = engine.K_api: 3 flag off, the wide family's K flag on) and each row carries at most
SpecPlan.k_by_rows' cap of the step's live rows (the plugin's draft_cap). Run once per arm (QWEN36_SPEC_ADAPTIVE_K=0 / 1 in
the environment; the engine reads it at construction) and compare the two logs / JSON files.

Per run it logs, over the steady-state steps: ms/step (time between consecutive step reads), committed tokens per live row
per step, tok/s/user (= committed / ms), the family of every step (flag on), the engine's switch / compaction counters and
each user's generated tokens (written to AK_OUT/<tag>.json for the md5 / agreement comparison across arms).

Env: AK_CTX (prompt tokens per user, default 32768), AK_RUNS (comma list of <users>[:sampled], default "1,2,4,8"), AK_STEPS
(read steps per run, default 80), AK_WARM (steps excluded from the stats, default 8), AK_CHURN (1 = also run the join / leave
schedule that crosses the 4 <-> 5 user boundary: 6 users, 2 leave from the middle rows (a compaction), 2 join, ...),
AK_OUT (directory for the per-run JSON, default /tmp/ak/out).
"""

import hashlib
import json
import math
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tt import spec_engine as se
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.tp_common import spec_adaptive_k_wide

B, K = 8, 3
CTX = int(os.environ.get("AK_CTX", "32768"))
RUNS = [r.strip() for r in os.environ.get("AK_RUNS", "1,2,4,8").split(",") if r.strip()]
STEPS = int(os.environ.get("AK_STEPS", "80"))
WARM = int(os.environ.get("AK_WARM", "8"))
CHURN = os.environ.get("AK_CHURN", "0") == "1"
OUT = os.environ.get("AK_OUT", "/tmp/ak/out")
MAX_NEW = 10**6


class AKDriver:
    """The plugin side of the deferred depth-1 pipeline at the runner block's width (1 + K_api columns)."""

    def __init__(self, eng, nb, sampled=False, seed=0):
        self.eng, self.nb = eng, nb
        self.B, self.Ka = eng._B_state, eng.K_api
        self.W = self.Ka + 1
        self.ak = eng._ak
        self.reqs = [None] * self.B
        self.pend, self.host_rows, self.next0 = [], set(), {}
        self.sampled = sampled
        self.gen = torch.Generator().manual_seed(4321 + seed)
        self.rec = []  # per read step: dict(wall, wait, rows, counts, fam)
        self.done = {}
        self.steps = 0

    def cap(self, n):
        """SpecPlan.draft_cap: the wide family's K at <= its width live rows, the base K above (flag off: K)."""
        if self.ak is None:
            return self.Ka
        return self.ak.wide.K if n <= self.ak.wide.B else self.ak.base.K

    def table_row(self, row):
        return torch.arange(1 + row * self.nb, 1 + (row + 1) * self.nb, dtype=torch.int32)

    def live(self):
        return [r for r in range(self.B) if self.reqs[r] is not None]

    def admit(self, row, rid, prompt, max_new=MAX_NEW):
        assert self.reqs[row] is None
        logits = self.eng.prefill(row, prompt, self.table_row(row)[None])
        self.reqs[row] = dict(rid=rid, P=len(prompt), out=[int(torch.as_tensor(logits).argmax())], acc=1, proposal=[])
        self.reqs[row]["max_new"] = max_new
        self.host_rows.add(row)

    def finish(self, row):
        q = self.reqs[row]
        self.eng.release(row)
        self.done[q["rid"]] = list(q["out"])
        self.reqs[row] = None
        self.host_rows.discard(row)

    def _spec_sampling(self):
        B, Ka = self.B, self.Ka
        return {
            "temperature": torch.full((B,), 0.6, dtype=torch.float32),
            "top_k": torch.full((B,), 20, dtype=torch.int32),
            "top_p": torch.full((B,), 0.95, dtype=torch.float32),
            "greedy": torch.zeros(B, dtype=torch.bool),
            "uniforms": torch.rand((B, Ka + 2), generator=self.gen, dtype=torch.float32).clamp(max=0.9999999),
        }

    def enq(self):
        B, W = self.B, self.W
        live = self.live()
        cap = self.cap(len(live))
        prev = self.pend[-1][0] if self.pend else None
        tok = torch.zeros((B, W), dtype=torch.int32)
        pos = torch.full((B, W), -1, dtype=torch.int32)
        nvd = torch.zeros(B, dtype=torch.int32)
        acc = torch.ones(B, dtype=torch.int32)
        pt = torch.zeros((B, self.nb), dtype=torch.int32)
        cr = torch.zeros(B, dtype=torch.bool)
        for r in live:
            q = self.reqs[r]
            pt[r] = self.table_row(r)
            if prev is None or r in self.host_rows:
                p0 = q["P"] + len(q["out"]) - 1
                self.next0[r] = p0
                d = list(q["proposal"])[:cap]
                n = len(d)
                tok[r] = -1
                tok[r, : 1 + n] = torch.tensor([q["out"][-1]] + d, dtype=torch.int32)
                pos[r, : 1 + n] = torch.arange(p0, p0 + 1 + n, dtype=torch.int32)
                nvd[r], acc[r] = n, q["acc"]
            else:
                up = self.next0[r] + W  # bound of this step's column-0 position (the pending step commits <= W)
                cr[r] = True
                tok[r] = -1
                pos[r] = up + torch.arange(W, dtype=torch.int32)
                nvd[r], acc[r] = cap, -1
        kw = {}
        if bool(cr.any()):
            kw.update(chain_rows=cr, chain_tip=prev.tip_id)
        if self.sampled:
            kw["spec_sampling"] = self._spec_sampling()
        res = self.eng.decode_forward(
            tok,
            pos,
            num_valid_drafts=nvd,
            accepted_counts=acc,
            page_table=pt,
            spec_mode="fused_sample",
            defer_read=True,
            **kw,
        )
        assert res.handle is not None
        self.pend.append((res.handle, list(live), {r: self.reqs[r]["rid"] for r in live}))
        self.host_rows.clear()
        self.steps += 1

    def resolve(self):
        h, rows, rids = self.pend.pop(0)
        ids, counts, p0 = h.wait()
        assert tuple(ids.shape) == (self.B, self.W), tuple(ids.shape)
        # every row, padding rows included, must satisfy the plugin's AcceptResult rule (vllm_tt_plugin/spec_accept.py):
        # count in [1, W], real ids exactly inside the count, PLACEHOLDER (-1) exactly past it
        cols = torch.arange(self.W)[None, :]
        inside = cols < counts.to(torch.int64)[:, None]
        assert bool(((counts >= 1) & (counts <= self.W)).all()), counts.tolist()
        assert not bool((ids[inside] == -1).any()) and bool((ids[~inside] == -1).all()), (ids.tolist(), counts.tolist())
        cpos = torch.full_like(ids, -1)
        for u in range(self.B):
            if int(p0[u]) >= 0:
                cpos[u, : int(counts[u])] = int(p0[u]) + 1 + torch.arange(int(counts[u]), dtype=torch.int32)
        dr = self.eng.propose_draft_tokens(self.Ka, ids.clone(), cpos, counts.clone(), hidden=h)
        assert tuple(dr.draft_token_ids.shape) == (self.B, self.Ka)
        got = {}
        for r in rows:
            q = self.reqs[r]
            if q is None or q["rid"] != rids[r]:
                continue
            cnt = int(counts[r])
            assert 1 <= cnt <= self.W and int(p0[r]) >= 0, (r, cnt, int(p0[r]))
            new = ids[r, :cnt].tolist()[: q["max_new"] - len(q["out"])]
            assert min(new) >= 0, (r, new)
            q["out"].extend(new)
            q["acc"] = cnt
            self.next0[r] = int(p0[r]) + cnt
            q["proposal"] = dr.draft_token_ids[r, : int(dr.num_valid[r])].tolist()
            got[r] = cnt
        fam = getattr(getattr(h, "ak_fam", None), "name", "base")
        return got, fam

    def pipeline(self, n, on_step=None):
        """n enqueues; the step enqueued first is read once the next one is enqueued (vLLM's async depth 1)."""
        t_prev = time.perf_counter()
        for i in range(n):
            if not self.live():
                break
            t0 = time.perf_counter()
            self.enq()
            if len(self.pend) == 2:
                ev = self.pend[0][0].event
                tw = time.perf_counter()
                if ev is not None:
                    ttnn.event_synchronize(ev)
                tw1 = time.perf_counter()
                got, fam = self.resolve()
                t2 = time.perf_counter()
                self.rec.append(
                    dict(
                        wall=1e3 * (t2 - t_prev),
                        wait=1e3 * (tw1 - tw),
                        enq=1e3 * (tw - t0),
                        rows=len(got),
                        counts=list(got.values()),
                        fam=fam,
                    )
                )
                t_prev = t2
                if on_step is not None:
                    on_step(self, len(self.rec))
        self.drain()

    def drain(self):
        while self.pend:
            self.resolve()


def _stats(xs):
    xs = sorted(xs)
    if not xs:
        return "n/a"
    return f"median {statistics.median(xs):.2f} p10 {xs[len(xs) // 10]:.2f} p90 {xs[(9 * len(xs)) // 10]:.2f}"


def _summ(rec):
    rec = [r for r in rec if r["rows"] > 0]
    if not rec:
        return {}
    walls = [r["wall"] for r in rec]
    per_row = [statistics.mean(r["counts"]) for r in rec]
    tps = [1e3 * c / w for c, w in zip(per_row, walls)]
    fams = {}
    for r in rec:
        fams[r["fam"]] = fams.get(r["fam"], 0) + 1
    hist = {}  # committed tokens per row-step -> count (acceptance at each draft depth: P(count > d) / P(count >= d))
    for r in rec:
        for c in r["counts"]:
            hist[int(c)] = hist.get(int(c), 0) + 1
    return dict(
        steps=len(rec),
        ms_med=statistics.median(walls),
        ms_mean=statistics.mean(walls),
        committed_per_row=statistics.mean(per_row),
        tok_s_user=statistics.mean(per_row) * 1e3 / statistics.mean(walls),
        tok_s_user_med=statistics.median(tps),
        wait_med=statistics.median(r["wait"] for r in rec),
        fams=fams,
        hist=dict(sorted(hist.items())),
    )


def _md5(toks):
    return hashlib.md5(",".join(str(t) for t in toks).encode()).hexdigest()[:12]


def _ak_counters(eng):
    st = eng._ak
    if st is None:
        return {}
    return dict(
        base_steps=st.base.steps,
        wide_steps=st.wide.steps,
        switches=dict(st.switch),
        compactions=getattr(st, "compactions", 0),
        truncated=st.truncated,
    )


def _prompts(tokenizer):
    base = _get_prompt(CTX, tokenizer)[0].tolist()
    # user u: the same document, its first 13 * u tokens dropped (distinct streams, nearly the same length)
    return [base[13 * u :] for u in range(B)]


def _run(eng, tag, prompts, users, sampled, nb):
    drv = AKDriver(eng, nb, sampled=sampled, seed=users)
    t0 = time.perf_counter()
    for u in range(users):
        drv.admit(u, u, prompts[u])
    t_pref = time.perf_counter() - t0
    c0 = _ak_counters(eng)
    drv.pipeline(STEPS + WARM)
    s = _summ(drv.rec[WARM:])
    for r in drv.live():
        drv.finish(r)
    gens = {str(k): v for k, v in sorted(drv.done.items())}
    md5s = {k: _md5(v[:256]) for k, v in gens.items()}
    logger.info(
        f"[ak] {tag} users={users} {'sampled' if sampled else 'greedy'} ctx={CTX}: ms/step {s.get('ms_med', 0):.2f} "
        f"(mean {s.get('ms_mean', 0):.2f}) committed/row/step {s.get('committed_per_row', 0):.3f} tok/s/user "
        f"{s.get('tok_s_user', 0):.1f} (median {s.get('tok_s_user_med', 0):.1f}) wait {s.get('wait_med', 0):.2f} "
        f"families {s.get('fams')} counts {s.get('hist')} prefill {t_pref:.1f}s ak {c0} -> {_ak_counters(eng)} "
        f"md5(256) {md5s}"
    )
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(dict(tag=tag, users=users, sampled=sampled, ctx=CTX, stats=s, gens=gens, ak=_ak_counters(eng)), f)
    return s


def _churn(eng, tag, prompts, nb):
    """Join / leave schedule across the 4 <-> 5 boundary (flag on: wide <-> base switches, a compaction when the users
    left in the middle rows leave holes)."""
    drv = AKDriver(eng, nb, seed=99)
    for u in range(6):
        drv.admit(u, u, prompts[u])
    nxt = [6]
    plan = {40: ("leave", [2, 5]), 80: ("join", 2), 120: ("leave", [0, 3]), 160: ("join", 1), 200: ("leave", [1])}

    def on_step(d, i):
        act = plan.get(i)
        if act is None:
            return
        d.drain()  # the plugin applies finishes / admissions between steps; drained here so prefills are simple
        kind, arg = act
        if kind == "leave":
            for r in arg:
                if d.reqs[r] is not None:
                    d.finish(r)
        else:
            for _ in range(arg):
                free = [r for r in range(d.B) if d.reqs[r] is None]
                d.admit(free[0], nxt[0], prompts[nxt[0] % len(prompts)])
                nxt[0] += 1
        logger.info(f"[ak] {tag} churn step {i}: {kind} {arg} -> live rows {d.live()}")

    c0 = _ak_counters(eng)
    drv.pipeline(240, on_step=on_step)
    s = _summ(drv.rec[WARM:])
    for r in drv.live():
        drv.finish(r)
    by_fam = {}
    for r in drv.rec[WARM:]:
        by_fam.setdefault((r["fam"], r["rows"]), []).append(r)
    detail = {
        f"{f}/{n}": f"{statistics.median(x['wall'] for x in v):.2f} ms {statistics.mean(statistics.mean(x['counts']) for x in v):.2f} tok"
        for (f, n), v in sorted(by_fam.items())
    }
    logger.info(
        f"[ak] {tag} churn ctx={CTX}: ms/step {s.get('ms_med', 0):.2f} committed/row/step {s.get('committed_per_row', 0):.3f} "
        f"tok/s/user {s.get('tok_s_user', 0):.1f} families {s.get('fams')} ak {c0} -> {_ak_counters(eng)} "
        f"per (family/rows): {detail}"
    )
    os.makedirs(OUT, exist_ok=True)
    gens = {str(k): v for k, v in sorted(drv.done.items())}
    with open(f"{OUT}/{tag}.json", "w") as f:
        json.dump(dict(tag=tag, churn=True, ctx=CTX, stats=s, gens=gens, ak=_ak_counters(eng), detail=detail), f)


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_adaptive_k_production_loop(mesh_device, monkeypatch):
    from transformers import AutoTokenizer

    for k, v in (
        ("QWEN36_SPEC_FUSED", "1"),
        ("QWEN36_SPEC_CHAIN", "1"),
        ("QWEN36_SPEC_CHAIN_SAMPLED", "1"),
        ("QWEN36_SPEC_FUSED_AUDIT_EVERY", os.environ.get("QWEN36_SPEC_FUSED_AUDIT_EVERY", "8")),
        ("QWEN36_SPEC_CHAIN_AUDIT_EVERY", os.environ.get("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "8")),
    ):
        monkeypatch.setenv(k, v)
    arm = "on" if os.environ.get("QWEN36_SPEC_ADAPTIVE_K", "0") == "1" else "off"
    Ka = spec_adaptive_k_wide(B)[1] if arm == "on" else K
    new = (STEPS + WARM) * (Ka + 1) * 4 + 256 if CHURN else (STEPS + WARM + 4) * (Ka + 1) + 128
    nb = math.ceil((CTX + new + 2 * (Ka + 1)) / BLOCK_SIZE) + 1
    num_blocks = B * nb + 40
    mesh = mesh_device
    mesh.enable_program_cache()
    t0 = time.time()
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=B, max_seq_len=nb * BLOCK_SIZE, enable_mtp=None)
    assert model.mtp is not None, "no MTP head"
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    model.allocate_kv_caches(
        [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=B
    )
    model.set_gdn_fused_decode(True)
    eng = se.get_spec_engine(model, B, K)
    eng.prepare(torch.stack([torch.arange(1 + u * nb, 1 + (u + 1) * nb, dtype=torch.int32) for u in range(B)]))
    eng.capture()
    logger.info(
        f"[ak] arm={arm} K_api={eng.K_api} model + engine ready in {time.time() - t0:.0f}s: ctx={CTX} runs={RUNS} nb={nb} "
        f"trace/DRAM log {getattr(model, '_vfy_mem_log', None)} wide={None if eng._ak is None else (eng._ak.wide.model or {}).get('_vfy_mem_log')}"
    )
    n_prog = mesh.num_program_cache_entries()
    mesh.set_program_cache_misses_allowed(False)
    try:
        prompts = _prompts(tokenizer)
        for r in RUNS:
            users, _, mode = r.partition(":")
            _run(eng, f"{arm}_ctx{CTX}_u{users}{'_s' if mode == 'sampled' else ''}", prompts, int(users), mode == "sampled", nb)
        if CHURN:
            _churn(eng, f"{arm}_ctx{CTX}_churn", prompts, nb)
        assert mesh.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh.set_program_cache_misses_allowed(True)
    logger.info(f"[ak] fused line tail: {eng._ak_note_text()} chain_breaks={dict(getattr(eng, 'chain_breaks', {}))}")
