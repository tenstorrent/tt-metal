# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""B > 1 continuous batching on MTPSpecEngine, driven directly (no vLLM).
Rows are independent, so a request's greedy tokens must not depend on which other rows are live, padding
or finishing: each request must equal its output in a full simultaneous batch.
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.spec_helpers import PAD, _prompt_pool
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _N_LAYERS
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
    warm = torch.stack([torch.arange(1 + u * NB, 1 + (u + 1) * NB, dtype=torch.int32) for u in range(B)])
    eng.prepare(warm)
    eng.capture()
    return model, eng, tokenizer, prompts


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
        res = eng.decode_forward(
            tokens, start, num_valid_drafts=nvd, accepted_counts=acc, page_table=pt, spec_mode="argmax_ids"
        )
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
def test_join_running_batch(mesh_device):
    """Requests joining a running batch (including into a just-finished row) get exactly their full-batch tokens."""
    model, eng, tokenizer, prompts = _setup(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        full = BatchDriver(eng, seed=1).run({0: [(u, u, prompts[u]) for u in range(B)]})
        stag = BatchDriver(eng, seed=4).run(
            {
                0: [(0, 0, prompts[0]), (5, 5, prompts[5])],
                3: [(2, 2, prompts[2])],
                7: [(1, 1, prompts[1]), (7, 7, prompts[7])],
                12: [(3, 3, prompts[3]), (4, 4, prompts[4]), (6, 6, prompts[6])],
                14: [(0, 100, prompts[3])],  # reuses row 0 once request 0 has finished
            }
        )
        for u in range(B):
            _assert_same(f"request {u} joined late", stag[u], full[u])
        _assert_same("request 100 (prompt 3 in row 0)", stag[100], full[3])
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
    finally:
        mesh_device.set_program_cache_misses_allowed(True)


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
