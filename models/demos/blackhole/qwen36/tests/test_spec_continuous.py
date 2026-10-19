# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""B > 1 continuous batching on MTPSpecEngine, driven directly (no vLLM).
Rows are independent, so a request's greedy tokens must not depend on which other rows are live, padding
or finishing: each request must equal its output in a full simultaneous batch.
"""

import math
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.spec_helpers import PAD, _prompt_pool
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _N_LAYERS, NEAR_TIE_GAP, _top2_gap
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import get_spec_engine

B = 8
K = 3
NB = 16  # page-table width in blocks (1024 tokens)
NUM_BLOCKS = 160  # KV pool; block 0 is the null block and is never handed to a request
MAX_NEW = 48
PROMPT_LENS = (130, 257, 389, 512, 600, 211, 345, 460)


def _setup(mesh_device):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=NB * BLOCK_SIZE, n_layers=_N_LAYERS)
    assert model.mtp is not None, "MTP head not built"
    model.set_gdn_fused_decode(True)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    pool = _prompt_pool(tokenizer, 4096)
    prompts = []
    for i, n in enumerate(PROMPT_LENS):
        prompts.append(pool[97 * i : 97 * i + n])
        assert len(prompts[i]) == n
    model.allocate_kv_caches(
        [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=B
    )
    eng = get_spec_engine(model, B, K)
    eng.prepare(_warm_table())
    eng.capture()
    return model, eng, tokenizer, prompts


def _warm_table():
    return torch.stack([torch.arange(1 + u * NB, 1 + (u + 1) * NB, dtype=torch.int32) for u in range(B)])


# The join schedule of test_join_running_batch: step -> [(row, request id, prompt index)]; requests join a running
# batch, and request 100 reuses row 0 (with prompt 3) once request 0 has finished.
_JOIN_SCHEDULE = {
    0: [(0, 0, 0), (5, 5, 5)],
    3: [(2, 2, 2)],
    7: [(1, 1, 1), (7, 7, 7)],
    12: [(3, 3, 3), (4, 4, 4), (6, 6, 6)],
    14: [(0, 100, 3)],
}


def _join_schedule(prompts):
    return {step: [(row, rid, prompts[p]) for row, rid, p in adms] for step, adms in _JOIN_SCHEDULE.items()}


def _step_classes(drv):
    """(plain, remap-only, join) step times in ms from a BatchDriver (join = a fresh row seeded in the step)."""
    plain = [ms for ms, rm, fr in drv.step_ms if not rm and not fr]
    remap_only = [ms for ms, rm, fr in drv.step_ms if rm and not fr]
    joins = [ms for ms, rm, fr in drv.step_ms if fr]
    return plain, remap_only, joins


class BatchDriver:
    """Host mirror of the plugin runner at B > 1 (greedy); row == state slot, None = padding row."""

    def __init__(self, eng, seed, lookahead=True, drafts=True, recycle=True):
        self.eng = eng
        self.lookahead = lookahead  # False: a row's blocks cover only its real tokens (no lookahead, as ngram)
        self.drafts = drafts  # False: proposals are withheld, every step verifies n = 0
        self.recycle = recycle  # False: finished requests keep their blocks, so their KV can be read after the run
        self.reqs = [None] * B
        g = torch.Generator().manual_seed(seed)
        self.free = (1 + torch.randperm(NUM_BLOCKS - 1, generator=g)).tolist()
        self.outs = {}
        self.steps = 0
        self.stats = []
        self.blocks_of = {}
        self.remapped = False  # a remap since the last step
        self.step_ms = []  # per step: (decode_forward ms, remap applied, fresh row seeded)

    def _blocks_for(self, req, last_pos):
        """[NB] int32 row covering positions up to last_pos + (1+K) + (K+1) lookahead (up to last_pos without
        lookahead); unused entries 0."""
        extra = (1 + K) + (K + 1) if self.lookahead else 1
        need = min(NB, math.ceil((last_pos + extra) / BLOCK_SIZE))
        while len(req["blocks"]) < need:
            req["blocks"].append(self.free.pop())
        row = req["blocks"] + [0] * (NB - len(req["blocks"]))
        return torch.tensor(row, dtype=torch.int32)

    def live(self):
        return [r for r in range(B) if self.reqs[r] is not None]

    def _finish(self, row):
        req = self.reqs[row]
        self.eng.release(row)
        self.outs[req["rid"]] = req["out"]
        self.blocks_of[req["rid"]] = list(req["blocks"])
        if self.recycle:
            self.free.extend(req["blocks"])
        self.reqs[row] = None

    def admit(self, row, rid, prompt, max_new=MAX_NEW):
        assert self.reqs[row] is None, f"row {row} is live"
        req = {"rid": rid, "prompt": list(prompt), "P": len(prompt), "blocks": [], "proposal": [], "acc": 1}
        req["max_new"] = max_new
        table_row = self._blocks_for(req, len(prompt) - 1)
        logits = self.eng.prefill(row, prompt, table_row[None])
        req["out"] = [int(logits.argmax())]
        self.reqs[row] = req
        if max_new == 1:
            self._finish(row)

    def step(self, skip=frozenset()):
        """One decode step; requests whose id is in skip are left out (padding rows) and keep their state."""
        eng = self.eng
        tokens = torch.zeros((B, 1 + K), dtype=torch.int32)
        start = torch.full((B, 1 + K), PAD, dtype=torch.int32)
        nvd = torch.zeros(B, dtype=torch.int32)
        acc = torch.ones(B, dtype=torch.int32)
        pt = torch.zeros((B, NB), dtype=torch.int32)
        live = [r for r in self.live() if self.reqs[r]["rid"] not in skip]
        pos0, drafts, ns = {}, {}, {}
        for r in live:
            q = self.reqs[r]
            pos0[r] = q["P"] + len(q["out"]) - 1
            drafts[r] = list(q["proposal"]) if self.drafts else []
            n = ns[r] = len(drafts[r])
            tokens[r] = PAD
            tokens[r, : 1 + n] = torch.tensor([q["out"][-1]] + drafts[r], dtype=torch.int32)
            start[r, : 1 + n] = torch.arange(pos0[r], pos0[r] + 1 + n, dtype=torch.int32)
            nvd[r] = n
            acc[r] = q["acc"]
            pt[r] = self._blocks_for(q, pos0[r])
        fresh = bool(eng._fresh)
        t0 = time.perf_counter()
        res = eng.decode_forward(
            tokens, start, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt, spec_mode="argmax_ids"
        )
        self.step_ms.append((1e3 * (time.perf_counter() - t0), self.remapped, fresh))
        self.remapped = False
        ids_all = res.argmax_ids
        assert tuple(ids_all.shape) == (B, 1 + K)
        cross = [r for r in live if int(pt[r, (pos0[r] + K) // BLOCK_SIZE]) == 0]  # padded columns past the blocks

        committed = torch.full((B, 1 + K), PAD, dtype=torch.int32)
        cpos = torch.full((B, 1 + K), PAD, dtype=torch.int32)
        acc_vec = torch.ones(B, dtype=torch.int32)
        pt2 = torch.zeros((B, NB), dtype=torch.int32)
        ms = {}
        for r in live:
            q, n = self.reqs[r], ns[r]
            ids = ids_all[r].tolist()
            assert all(i == PAD for i in ids[n + 1 :]), f"row {r}: ids past n={n} not -1: {ids}"
            assert all(i != PAD for i in ids[: n + 1]), f"row {r}: -1 inside valid ids {ids} (n={n})"
            m = 0
            while m < n and drafts[r][m] == ids[m]:
                m += 1
            ms[r] = m
            new = (drafts[r][:m] + [ids[m]])[: q["max_new"] - len(q["out"])]
            q["out"].extend(new)
            q["acc"] = len(new)
            committed[r, : len(new)] = torch.tensor(new, dtype=torch.int32)
            cpos[r, : len(new)] = torch.arange(pos0[r] + 1, pos0[r] + 1 + len(new), dtype=torch.int32)
            acc_vec[r] = len(new)
            pt2[r] = self._blocks_for(q, pos0[r] + len(new))
        d = eng.propose_draft_tokens(K, committed, cpos, acc_vec, pt2)
        assert tuple(d.draft_token_ids.shape) == (B, K)
        assert tuple(d.num_valid.shape) == (B,)
        for r in range(B):
            nv = int(d.num_valid[r])
            if r in pos0:
                assert 0 <= nv <= K
                self.reqs[r]["proposal"] = d.draft_token_ids[r, :nv].tolist()
            else:
                assert nv == 0, f"padding row {r} proposed {nv} drafts"
        for r in live:
            if len(self.reqs[r]["out"]) >= self.reqs[r]["max_new"]:
                self._finish(r)
        self.stats.append({"live": live, "n": ns, "m": ms, "cross": cross, "no_draft": sorted(eng._no_draft)})
        self.steps += 1

    def remap(self, perm):
        self.eng.remap_slots(perm)
        self.reqs = [self.reqs[j] for j in perm]
        self.remapped = True

    def condense(self):
        """The plugin's input_batch.condense: the last live row moves into the lowest hole, as one swap remap.
        Returns the remap, or None when the live rows are already front-packed."""
        live = self.live()
        holes = [r for r in range(len(live)) if self.reqs[r] is None]
        if not holes:
            return None
        perm = list(range(B))
        for h, t in zip(holes, [r for r in reversed(live) if r >= len(live)]):
            perm[h], perm[t] = t, h
        self.remap(perm)
        return perm

    def run(self, schedule, remaps=None, skips=None):
        """schedule: step index -> [(row, rid, prompt)] admissions, made at that step or the first later one
        where the row is free. remaps: step index -> slot_remap permutation, applied after that step's admissions
        and before its decode. skips: step index -> request ids left out of that step's decode."""
        waiting = []
        while True:
            waiting += schedule.get(self.steps, [])
            for adm in list(waiting):
                if self.reqs[adm[0]] is None:
                    self.admit(*adm)
                    waiting.remove(adm)
            if remaps and self.steps in remaps:
                self.remap(remaps[self.steps])
            if not self.live() and not waiting and not any(s > self.steps for s in schedule):
                break
            self.step((skips or {}).get(self.steps, frozenset()))
        return self.outs


def _accept_rate(driver):
    """Mean committed tokens per live row-step."""
    committed = sum(m + 1 for s in driver.stats for m in s["m"].values())
    rows = sum(len(s["live"]) for s in driver.stats)
    return committed / rows


def _assert_same(name, got, ref):
    if got != ref:
        i = next((j for j in range(min(len(got), len(ref))) if got[j] != ref[j]), min(len(got), len(ref)))
        raise AssertionError(f"{name}: first divergence at {i}\n got {got}\n ref {ref}")


def _mtp_kv_rows(eng, blocks, start, stop):
    """Device-0 MTP K and V rows [stop - start, heads, head_dim] at positions start..stop-1 over ``blocks``."""
    att = eng.mtp.attention
    out = []
    for cache in (att.paged_k, att.paged_v):
        c = ttnn.to_torch(ttnn.get_device_tensors(cache)[0]).float()
        out.append(torch.stack([c[blocks[q // BLOCK_SIZE], :, q % BLOCK_SIZE] for q in range(start, stop)]))
    return out


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_padding_rows_match_full_batch(mesh_device):
    """A partial batch (padding rows, and rows finishing early) gives each request exactly its full-batch tokens."""
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = BatchDriver(eng, seed=1)
        full = drv.run({0: [(u, u, prompts[u]) for u in range(B)]})
        for u in range(B):
            assert len(full[u]) == MAX_NEW, f"request {u}: {len(full[u])} tokens"
            logger.info(f"[full] request {u}: {tokenizer.decode(full[u])!r}")
        rate = _accept_rate(drv)
        logger.info(f"[full] steps={drv.steps} mean committed/row-step={rate:.2f}")
        assert rate > 1.0, f"no drafts accepted (rate {rate:.2f})"

        part = BatchDriver(eng, seed=2).run({0: [(r, r, prompts[r]) for r in (1, 4, 6)]})
        for r in (1, 4, 6):
            _assert_same(f"request {r} in 3-row batch", part[r], full[r])

        part2 = BatchDriver(eng, seed=3).run({0: [(0, 0, prompts[0])]})
        _assert_same("request 0 alone", part2[0], full[0])
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("fast_join", ["0", "1"])
def test_join_running_batch(mesh_device, fast_join, monkeypatch):
    """Requests joining a running batch (including into a just-finished row) get exactly their full-batch tokens.
    QWEN36_SPEC_FAST_JOIN=1 (the fresh rows seeded in place and verified inside the running rows' replay) must also
    make a join step cheap: median join step < 1.8x the median plain step (the eager seed's is ~5x)."""
    monkeypatch.setenv("QWEN36_SPEC_FAST_JOIN", fast_join)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    assert eng.fast_join is (fast_join == "1")
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        full = BatchDriver(eng, seed=1).run({0: [(u, u, prompts[u]) for u in range(B)]})
        drv = BatchDriver(eng, seed=4)
        stag = drv.run(_join_schedule(prompts))
        for u in range(B):
            _assert_same(f"request {u} joined late", stag[u], full[u])
        _assert_same("request 100 (prompt 3 in row 0)", stag[100], full[3])
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    plain, _, joins = _step_classes(drv)
    logger.info(
        f"[join fast_join={fast_join}] plain={len(plain)} (median {statistics.median(plain):.1f} ms) "
        f"join={len(joins)} (median {statistics.median(joins):.1f} ms, max {max(joins):.1f} ms)"
    )
    if fast_join == "1":
        assert statistics.median(joins) < 1.8 * statistics.median(plain), "a fast-join step is not cheap"


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fast_join_vs_eager_seed(mesh_device, monkeypatch):
    """QWEN36_SPEC_FAST_JOIN 0 vs 1 on the SAME model and prompts (note 23, D2): the join schedule under both flags,
    one engine after the other. Every request's tokens identical between the flags (the seeded first token comes
    from the T-row verify kernels instead of the eager T = 1 forward: a bf16 near-tie flip would show here;
    QWEN36_FAST_JOIN_STRICT_TOKENS=0 turns that into a log line), mean committed tokens per row-step within 0.1,
    and the fast path's join / seed steps under half the eager seed's and under 1.8x a plain step."""
    monkeypatch.setenv("QWEN36_SPEC_FAST_JOIN", "0")
    model, eng, tokenizer, prompts = _setup(mesh_device)
    runs = {}
    for flag in ("0", "1"):
        if flag == "1":
            eng.shutdown()
            monkeypatch.setenv("QWEN36_SPEC_FAST_JOIN", "1")
            eng = get_spec_engine(model, B, K)
            eng.prepare(_warm_table())
            eng.capture()
        assert eng.fast_join is (flag == "1")
        n_prog = mesh_device.num_program_cache_entries()
        mesh_device.set_program_cache_misses_allowed(False)
        try:
            full = BatchDriver(eng, seed=1).run({0: [(u, u, prompts[u]) for u in range(B)]})
            drv = BatchDriver(eng, seed=4)
            stag = drv.run(_join_schedule(prompts))
            assert mesh_device.num_program_cache_entries() == n_prog, f"flag {flag}: a program compiled after capture"
        finally:
            mesh_device.set_program_cache_misses_allowed(True)
        for u in range(B):
            _assert_same(f"[fast_join={flag}] request {u} joined late", stag[u], full[u])
        plain, _, joins = _step_classes(drv)
        runs[flag] = dict(
            full=full, stag=stag, rate=_accept_rate(drv), plain=statistics.median(plain), join=statistics.median(joins)
        )
        logger.info(
            f"[fast_join={flag}] committed/row-step={runs[flag]['rate']:.3f} plain median {runs[flag]['plain']:.1f} ms "
            f"join median {runs[flag]['join']:.1f} ms (n={len(joins)}, max {max(joins):.1f} ms)"
        )
    strict = os.environ.get("QWEN36_FAST_JOIN_STRICT_TOKENS", "1") == "1"
    for rid, ref in runs["0"]["stag"].items():
        got = runs["1"]["stag"][rid]
        if got != ref:
            i = next((j for j in range(min(len(got), len(ref))) if got[j] != ref[j]), min(len(got), len(ref)))
            # Near-tie check: token i is predicted from prompt + out[:i]; prefill that sequence eagerly (the engine's
            # eager path, QWEN36_SPEC_FAST_PREFILL numerics aside) and read the plain top-2 gap at token i.
            p_idx = next(p for adms in _JOIN_SCHEDULE.values() for _, r, p in adms if r == rid)
            seq = list(prompts[p_idx]) + [int(t) for t in ref[:i]]
            was_fast = eng.fast_prefill
            eng.fast_prefill = False
            lg = eng.prefill(0, seq, _warm_table()[0:1])
            eng.release(0)
            eng.fast_prefill = was_fast
            gap = _top2_gap(lg)
            msg = (
                f"request {rid}: fast join diverges from the eager seed at token {i}\n fast  {got}\n eager {ref}\n"
                f" fast: {tokenizer.decode(got)!r}\n eager: {tokenizer.decode(ref)!r}\n"
                f" eager top-2 gap at token {i}: {gap:.3f} ({'near-tie' if gap < NEAR_TIE_GAP else 'CONFIDENT'}), "
                f"eager argmax {int(lg.argmax())}"
            )
            if strict:
                raise AssertionError(msg)
            logger.warning(msg)
    assert abs(runs["0"]["rate"] - runs["1"]["rate"]) < 0.1, "acceptance changed with the fast join"
    assert runs["1"]["join"] < 0.5 * runs["0"]["join"], "the fast join is not faster than the eager seed"
    assert runs["1"]["join"] < 1.8 * runs["1"]["plain"], "a fast-join step is not cheap"


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_row_moves_keep_state(mesh_device):
    """Rows moved by slot_remap (including a prefilled request before its first decode) keep their requests' tokens."""
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        full = BatchDriver(eng, seed=1).run({0: [(u, u, prompts[u]) for u in range(B)]})
        moved = BatchDriver(eng, seed=5).run(
            {
                0: [(u, u, prompts[u]) for u in (0, 1, 3, 4, 5, 6)],
                4: [(2, 2, prompts[2])],
                9: [(7, 7, prompts[7])],
            },
            remaps={
                4: [1, 0, 7, 3, 4, 5, 6, 2],  # rid 2 moves to row 7 before its seed; rows 0 and 1 swap
                6: [3, 4, 5, 6, 0, 1, 2, 7],
                11: [7, 6, 5, 4, 3, 2, 1, 0],
            },
        )
        for u in range(B):
            _assert_same(f"request {u} after row moves", moved[u], full[u])
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("fast_join", ["0", "1"])
def test_churn_row_moves_match_full_batch(mesh_device, fast_join, monkeypatch):
    """Production-shaped churn (note 23, D3): short outputs, a finish about every second step, each hole condensed
    by the runner's swap remap and refilled by a new request. Every request's tokens equal its full-batch tokens,
    no program compiles after capture, and a remap costs nothing on the device: a step with a remap and no fresh
    row takes < 1.3x the median plain step (QWEN36_SPEC_SLOTMAP=1; the device remap of =0 adds ~200 ms). With
    QWEN36_SPEC_FAST_JOIN=1 a join step (a fresh row seeded) takes < 1.8x the median plain step as well."""
    monkeypatch.setenv("QWEN36_SPEC_FAST_JOIN", fast_join)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    assert eng.fast_join is (fast_join == "1")
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        full = BatchDriver(eng, seed=1).run({0: [(u, u, prompts[u]) for u in range(B)]})
        drv = BatchDriver(eng, seed=11)
        N_REQ = 24
        lens = {rid: 5 + (rid * 7) % 12 for rid in range(N_REQ)}  # 5..16 tokens, staggered finishes
        home = {}  # rid -> prompt index
        for u in range(B):
            drv.admit(u, u, prompts[u], max_new=lens[u])
            home[u] = u
        rid = B
        while drv.live() or rid < N_REQ:
            drv.condense()
            # one admission into the lowest free row every other step (at once when the batch ran dry)
            if rid < N_REQ and (drv.steps % 2 == 0 or not drv.live()):
                free = [r for r in range(B) if drv.reqs[r] is None]
                if free:
                    home[rid] = rid % B
                    drv.admit(free[0], rid, prompts[rid % B], max_new=lens[rid])
                    rid += 1
            if drv.live():
                drv.step()
        for r, p in home.items():
            assert len(drv.outs[r]) == lens[r], f"request {r}: {len(drv.outs[r])} tokens, wanted {lens[r]}"
            _assert_same(f"request {r} (prompt {p}) under churn", drv.outs[r], full[p][: lens[r]])
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
        plain, remap_only, joins = _step_classes(drv)
        logger.info(
            f"[churn fast_join={fast_join}] steps={drv.steps} plain={len(plain)} (median {statistics.median(plain):.1f} ms) "
            f"remap-only={len(remap_only)} (median {statistics.median(remap_only) if remap_only else 0:.1f} ms) "
            f"join={len(joins)} (median {statistics.median(joins) if joins else 0:.1f} ms, "
            f"max {max(joins) if joins else 0:.1f} ms)"
        )
        assert len(remap_only) >= 3, "the churn schedule produced no remap-only steps"
        assert statistics.median(remap_only) < 1.3 * statistics.median(plain), "a remap step is not free"
        if fast_join == "1":
            assert joins, "the churn schedule produced no join steps"
            assert statistics.median(joins) < 1.8 * statistics.median(plain), "a fast-join step is not cheap"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_left_out_rows_keep_state(mesh_device):
    """Requests left out of decode steps (running, or before their first decode) keep their full-batch tokens."""
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        full = BatchDriver(eng, seed=1).run({0: [(u, u, prompts[u]) for u in range(B)]})
        out = BatchDriver(eng, seed=6).run(
            {
                0: [(u, u, prompts[u]) for u in (0, 1, 2, 3, 4)],
                3: [(5, 5, prompts[5])],
                4: [(6, 6, prompts[6])],
                8: [(7, 7, prompts[7])],
            },
            skips={
                2: {1, 3},  # running requests left out for a step
                3: {1, 5},  # request 1 for a second step; request 5 left out before its first decode
                4: {5},  # request 6 seeds while 5 waits: 5 is re-prefilled when it returns
                6: {0, 1, 2, 3, 4, 6},  # every live request but one left out
                9: {2},
                10: {2},  # request 2 left out across a row move
            },
            remaps={10: [7, 6, 5, 4, 3, 2, 1, 0]},
        )
        for u in range(B):
            _assert_same(f"request {u} left out of some steps", out[u], full[u])
        assert not eng._parked, f"parked slots left behind: {sorted(eng._parked)}"
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_padded_columns_past_reservation(mesh_device):
    """No-draft steps whose padded columns cross into unallocated blocks (max_model_len, ngram) give the same
    tokens as with lookahead blocks, and two such rows in one call do not trip the KV-write collision check."""
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        short = prompts[0][:60]
        # rows 0 and 3 run in lockstep: their padded columns cross 64 in the same call
        sched = {0: [(0, 0, short), (3, 1, short), (5, 2, prompts[5])]}
        ref = BatchDriver(eng, seed=7, drafts=False).run(sched)
        drv = BatchDriver(eng, seed=8, lookahead=False, drafts=False)
        out = drv.run(sched)
        assert max(len(s["cross"]) for s in drv.stats) >= 2, "no step had two rows past their blocks"
        for rid in ref:
            _assert_same(f"request {rid} without lookahead", out[rid], ref[rid])
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_withheld_rows_keep_mtp_kv(mesh_device):
    """Rows marked no_draft (drafts withheld past the streak) still write their MTP KV: the drafter cache over the
    generated positions matches a run where no row is ever marked."""
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    streak = eng.no_draft_streak
    # rid 2 joins unmarked while rids 0 and 1 are marked: the recorded drafter then runs with kept rows
    sched = {0: [(0, 0, prompts[0]), (3, 1, prompts[3])], 20: [(5, 2, prompts[5])]}
    runs = {}
    try:
        for name, limit, seed in (("unmarked", 10**9, 9), ("marked", streak, 10)):
            eng.no_draft_streak = limit
            drv = BatchDriver(eng, seed=seed, drafts=False, recycle=False)
            outs = drv.run(sched)
            marked = (
                max(len(s["no_draft"]) for s in drv.stats),
                any(0 < len(s["no_draft"]) < len(s["live"]) for s in drv.stats),
            )
            kv = {}
            for rid, r in ((0, 0), (1, 3), (2, 5)):
                P = len(prompts[r])
                # anchor slots P .. P + len(out) - 2 (one per step); the last drafts past them are junk
                kv[rid] = _mtp_kv_rows(eng, drv.blocks_of[rid], P, P + len(outs[rid]) - 1)
            runs[name] = (outs, marked, kv)
    finally:
        eng.no_draft_streak = streak
        mesh_device.set_program_cache_misses_allowed(True)
    assert runs["unmarked"][1][0] == 0 and runs["marked"][1] == (3, True), "rows not marked no_draft as expected"
    for rid in (0, 1, 2):
        _assert_same(f"request {rid} marked vs unmarked", runs["marked"][0][rid], runs["unmarked"][0][rid])
        for which, a, b in zip("KV", runs["unmarked"][2][rid], runs["marked"][2][rid]):
            cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
            worst = int(cos.argmin())
            logger.info(f"[mtp_kv] request {rid} {which}: min cos {float(cos.min()):.5f} at offset {worst}")
            assert float(cos.min()) > 0.99, f"request {rid}: MTP {which} differs at generated offset {worst}"
    assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"


def _mtp_kv_slot_all_devices(eng, blocks, pos):
    """MTP paged K and V at position ``pos`` over ``blocks``, one tensor per device (bf16, exact bytes)."""
    att = eng.mtp.attention
    out = []
    for cache in (att.paged_k, att.paged_v):
        for t in ttnn.get_device_tensors(cache):
            c = ttnn.to_torch(t)
            out.append(c[blocks[pos // BLOCK_SIZE], :, pos % BLOCK_SIZE].clone())
    return out


def _join_scenario(eng, prompts, R, join_fast, monkeypatch):
    """R fresh rows (rows 0..R-1) join in ONE step (QWEN36_SPEC_JOIN_FAST=join_fast), then run to completion.
    Returns tokens, the MTP KV of slot Tp-1 per fresh row right after the join step, the join step's verify logits
    (fresh rows) and the join step's host ms."""
    monkeypatch.setenv("QWEN36_SPEC_JOIN_FAST", join_fast)
    logger.info(f"[joinfast] QWEN36_SPEC_JOIN_FAST={os.environ['QWEN36_SPEC_JOIN_FAST']} R={R}")
    drv = BatchDriver(eng, seed=11)
    snap = {}
    orig = eng.decode_forward

    def hooked(*a, **kw):
        first = not snap
        if first:
            kw["spec_mode"] = "logits"
            assert sorted(eng._fresh) == list(range(R)), f"fresh rows {sorted(eng._fresh)} != 0..{R - 1}"
        t0 = time.perf_counter()
        res = orig(*a, **kw)
        if first:
            snap["ms"] = 1e3 * (time.perf_counter() - t0)
            snap["logits"] = res.logits[:R].clone()
            snap["kv"] = [
                _mtp_kv_slot_all_devices(eng, drv.reqs[r]["blocks"], len(drv.reqs[r]["prompt"]) - 1) for r in range(R)
            ]
        return res

    eng.decode_forward = hooked
    try:
        for r in range(R):
            drv.admit(r, r, prompts[r])
        outs = drv.run({})
    finally:
        del eng.decode_forward
    snap["outs"] = outs
    return snap


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_join_fast_multirow_bit_identical(mesh_device, monkeypatch):
    """QWEN36_SPEC_JOIN_FAST 1 vs 0 with R = 2 and R = 4 fresh rows joining in one step (temperature 0): identical
    token streams, bit-identical MTP KV of slot Tp-1 for every joined row (every device), identical verify logits of
    the join step (torch.equal). Logs the join step host ms for both flags (3 repeats each, interleaved)."""
    monkeypatch.setenv("QWEN36_SPEC_FAST_JOIN", "1")
    monkeypatch.setenv("QWEN36_SPEC_JOIN_FAST", "1")  # prepare compiles the batched warm programs
    model, eng, tokenizer, prompts = _setup(mesh_device)
    assert eng.fast_join
    for R in (2, 4):
        runs = {"0": [], "1": []}
        for rep in range(3):
            for flag in ("1", "0"):
                runs[flag].append(_join_scenario(eng, prompts, R, flag, monkeypatch))
        ref = runs["0"][0]
        for flag in ("1", "0"):
            for i, s in enumerate(runs[flag]):
                tag = f"R={R} flag={flag} rep={i}"
                for rid in range(R):
                    _assert_same(f"{tag} request {rid}", s["outs"][rid], ref["outs"][rid])
                for r in range(R):
                    for d, (a, b) in enumerate(zip(s["kv"][r], ref["kv"][r])):
                        assert torch.equal(a, b), f"{tag}: MTP KV row {r} tensor {d} differs"
                    assert any(bool(t.any()) for t in s["kv"][r]), f"{tag}: MTP KV row {r} is all zero"
                assert torch.equal(s["logits"], ref["logits"]), f"{tag}: join-step verify logits differ"
        for flag in ("1", "0"):
            ms = [s["ms"] for s in runs[flag]]
            logger.info(
                f"[joinfast] R={R} QWEN36_SPEC_JOIN_FAST={flag} join step host ms: "
                + ", ".join(f"{m:.1f}" for m in ms)
                + f" (median {statistics.median(ms):.1f})"
            )
