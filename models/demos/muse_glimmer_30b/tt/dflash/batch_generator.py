# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 speculative decoding for several users in one 32-row step: user slots own row groups of a RowLayout.

K (drafts per user) follows the number of active users (K_BY_USERS); R = K + 1 rows per user, and each R has its own
traces. The drafter context pass of the rows committed by the last verify runs inside the next draft step, or eagerly
with the old layout when a user's rows move (new R or new group). The vLLM adapter drives start_slot / propose_slots /
verify_slots / commit_slot / release_slot directly."""

import time
from dataclasses import dataclass, field

import torch
from loguru import logger

import ttnn
from models.demos.muse_glimmer_30b.tt.attention import kv_page_table
from models.demos.muse_glimmer_30b.tt.dflash.generator import _WARM_SAMPLING, DFlash2Generator, _mean_ms
from models.demos.muse_glimmer_30b.tt.dflash.layout import RowLayout
from models.demos.muse_glimmer_30b.tt.dflash.sampling import rejection_sample
from models.demos.muse_glimmer_30b.tt.dflash.topk import TOPK_PER_DEVICE, device_topk, read_topk
from models.demos.muse_glimmer_30b.tt.model import DECODE_B_PAD
from models.demos.muse_glimmer_30b.tt.sampling import draw, filtered_probs, normalize_sampling

K_BY_USERS = ((2, 15), (4, 7), (8, 3))  # (max active users, drafts per user K)


def rows_for_users(n, k_by_users=K_BY_USERS):
    """Rows per user R = K + 1 for n active users."""
    for users, k in k_by_users:
        if n <= users:
            return k + 1
    raise ValueError(f"{n} users > {k_by_users[-1][0]}: use plain decode")


def plan_layout(active, R, old=None):
    """RowLayout with 32 // R groups for the active slots. With the same R as `old`, each slot keeps its group and a
    new slot takes the lowest free group. With a new R, the slots keep their old group order (new slots after them,
    by slot id)."""
    G = DECODE_B_PAD // R
    active = list(active)
    assert len(active) <= G, f"{len(active)} users do not fit {G} groups of {R} rows"
    old_group = {u: old.group_of(u) for u in active if old is not None and u in old.slots}
    groups = [None] * G
    if old is not None and old.R == R:
        for u, g in old_group.items():
            groups[g] = u
        rest = sorted(u for u in active if u not in old_group)
    else:
        rest = sorted(active, key=lambda u: (old_group.get(u, G), u))
    free = [g for g in range(G) if groups[g] is None]
    for g, u in zip(free, rest):
        groups[g] = u
    return RowLayout(R, tuple(groups))


@dataclass
class _Slot:
    p: int  # next position (tokens in the target KV cache)
    anchor: (
        int  # last committed token: row 0 of the next block (None until the first verify after a return_logits prefill)
    )
    sp: tuple  # normalized sampling (temperature, top_p, k) or None (greedy)
    gen: torch.Generator  # host RNG (sampling) or None
    budget: int  # max new tokens, the first token included
    ignore_eos: bool = False
    tokens: list = field(default_factory=list)
    accept: list = field(default_factory=list)  # new tokens per round
    pending: tuple = None  # (p_prev, produced) of the last verify; its taps are in the verifier rows
    ctx_pending: bool = False  # the pending rows still need the drafter context pass
    done: bool = False


class DFlash2BatchGenerator(DFlash2Generator):
    """Several users per 32-row step (generate_batch). The inherited single-user API (start / round / generate) runs
    on slot 0 when no batch is running."""

    def __init__(
        self,
        model,
        drafter,
        selector,
        page_rows,
        page_table_tt,
        stop_ids,
        enable_trace=False,
        k_by_users=K_BY_USERS,
    ):
        """page_rows: host int32 [max_users, nb], row u = slot u's target page row; page_table_tt: the device prefill
        page table (row u = slot u). The drafter needs max_users >= page_rows.shape[0]."""
        super().__init__(model, drafter, selector, page_rows, page_table_tt, stop_ids, enable_trace)
        self.max_users = int(page_rows.shape[0])
        assert drafter.max_users >= self.max_users, (drafter.max_users, self.max_users)
        assert model.args.max_batch_size >= self.max_users, (model.args.max_batch_size, self.max_users)
        self.k_by_users = tuple(k_by_users)
        self.slots = {}  # slot -> _Slot
        self.layout = None  # layout of the last verify (the verifier tap rows follow it); None before the first round
        self._ctx_rows = {}  # (R, groups) -> R device tensors [groups] int32: context-pass write rows
        self.batch_rounds = []
        self._placed = None  # the last layout used by a draft or verify step (keeps slots in their groups)
        self._traced_draft = self._traced_verify = False
        # Persistent tensors of every layout shape exist before any trace capture: a buffer allocated after a capture
        # can sit on memory the trace uses for intermediates, and every replay overwrites it.
        for R in sorted({k + 1 for _, k in self.k_by_users}):
            G = DECODE_B_PAD // R
            self.verifier.reserve(R, G)
            self._ctx_rows_for(R, G)
            drafter.set_block_rows(R)  # builds the conv shift matrix of this block size
        drafter.set_block_rows(drafter.layout.R)

    def _ctx_rows_for(self, R, groups):
        key = (R, groups)
        if key not in self._ctx_rows:  # created once, kept (persistent for trace replay)
            assert not self.traces, f"context rows {key} allocated after a trace capture (reserve them in __init__)"
            self._ctx_rows[key] = [
                ttnn.from_torch(
                    torch.full((groups,), -1, dtype=torch.int32),
                    device=self.mesh,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self.mapper,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(R)
            ]
        return self._ctx_rows[key]

    def start_slot(
        self,
        slot,
        prompt_ids,
        max_new_tokens=None,
        sampling=None,
        seed=0,
        ignore_eos=False,
        return_logits=False,
        warm=False,
    ):
        """Prefill `slot` with layer taps, fill its drafter context and draw its first token (returned). With
        return_logits=True, return the last-position logits (torch float32 [vocab]) instead: the slot then has no
        anchor until its first verify, whose row 0 is the caller's token. max_new_tokens None: no token budget.
        warm=True (warm_prefill): run every drafter context chunk of the bucket."""
        assert 0 <= slot < self.max_users and (max_new_tokens is None or max_new_tokens >= 1)
        m, d, cfg = self.model, self.drafter, self.cfg
        sp = normalize_sampling(sampling, m.args.max_top_k)
        gen = torch.Generator().manual_seed(seed) if sp is not None else None
        prompt = torch.tensor(list(prompt_ids), dtype=torch.int64)
        ids = list(cfg.target_layer_ids)
        if self.traces and m.prefill_is_new(prompt.numel()):
            logger.info(f"prefill length {prompt.numel()} is new: releasing traces (recaptured on next use)")
            self.release_traces(keep_seen=True)
        logits_tt, taps, t0, s0, s_real = m.prefill_with_taps(
            prompt, slot, self.page_table_tt, ids, tap_window=cfg.sliding_window
        )
        if return_logits:
            first = m.logits_to_torch(logits_tt, 1)[0]
        elif sp is None:
            tok_tt = m.sample_on_device(logits_tt)
            first = int(m.tokens_to_torch(tok_tt, 1)[0])
            ttnn.deallocate(tok_tt)
        else:
            temperature, top_p, k = sp
            vals_tt, ids_tt = device_topk(logits_tt, self.verifier.vocab_bias)
            values, tids = read_topk(
                vals_tt, ids_tt, [0], vocab_local=m.args.vocab_local, vocab_size=m.args.vocab_size, k=TOPK_PER_DEVICE
            )
            ttnn.deallocate(vals_tt)
            ttnn.deallocate(ids_tt)
            probs, pids = filtered_probs(values, tids, temperature, top_p, k)
            first = draw(probs[0], pids[0], gen)
        ttnn.deallocate(logits_tt)
        self._fill_context(taps, t0, s0, s_real, d.fill_table(slot), 1, all_chunks=warm)
        budget = max_new_tokens if max_new_tokens is not None else 1 << 62
        if return_logits:
            s = _Slot(p=s_real, anchor=None, sp=sp, gen=gen, budget=budget, ignore_eos=ignore_eos)
        else:
            s = _Slot(p=s_real, anchor=first, sp=sp, gen=gen, budget=budget, ignore_eos=ignore_eos, tokens=[first])
            s.done = budget <= 1 or (not ignore_eos and first in self.stop_ids)
        self.slots[slot] = s
        return first

    def _flush_context(self, old):
        """Eager drafter context pass of the pending rows with the layout `old` of the verify that produced them
        (rows p < produced of each slot; the other rows are skipped)."""
        d = self.drafter
        d.set_layout(old)
        pos = [0] * DECODE_B_PAD
        widx = [[-1] * old.groups for _ in range(old.R)]
        for g, u in enumerate(old.slots):
            s = self.slots.get(u)
            if s is None or s.done or not s.ctx_pending:
                continue
            p_prev, produced = s.pending
            for p in range(old.R):
                pos[g * old.R + p] = p_prev + p
                if p < produced:
                    widx[p][g] = p_prev + p
            s.ctx_pending = False
        rope_idx = d.make_rope_idx(pos)
        rows = [
            ttnn.from_torch(
                torch.tensor(w, dtype=torch.int32),
                device=self.mesh,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for w in widx
        ]
        d.context_pass(self.verifier.tap_list(), rope_idx, rows)
        for t in [rope_idx] + rows:
            ttnn.deallocate(t)

    def _set_batch_inputs(self, lay):
        """Host copies for the draft step on layout `lay`: block tokens / rope / SDPA positions, the verifier's write
        rows (= the block rows, reused by the verify pass), and the context rows of the pending slots (all R rows: the
        block pass rewrites the rows past the accepted ones)."""
        R, cfg = lay.R, self.cfg
        toks = torch.zeros(1, DECODE_B_PAD, dtype=torch.int32)
        sdpa = torch.full((DECODE_B_PAD,), -1, dtype=torch.int32)
        rope, block_pos, ctx_rope = [0] * DECODE_B_PAD, [-1] * DECODE_B_PAD, [0] * DECODE_B_PAD
        ctx = [[-1] * lay.groups for _ in range(R)]
        for g, u in enumerate(lay.slots):
            if u is None:
                continue
            s, r0 = self.slots[u], g * R
            toks[0, r0] = s.anchor
            toks[0, r0 + 1 : r0 + R] = cfg.mask_token_id
            sdpa[r0 : r0 + R] = s.p + R - 1
            for p in range(R):
                rope[r0 + p] = block_pos[r0 + p] = s.p + p
            if s.ctx_pending:
                for p in range(R):
                    ctx_rope[r0 + p] = ctx[p][g] = s.pending[0] + p
                s.ctx_pending = False
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(toks, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper),
            self.block_tokens_tt,
        )
        self._copy_rope(rope, self.block_rope_idx)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(sdpa, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper),
            self.sdpa_pos,
        )
        self.verifier.write_rows(block_pos)
        self._copy_rope(ctx_rope, self.ctx_rope_idx)
        for vals, dev in zip(ctx, self._ctx_rows_for(R, lay.groups)):
            self._copy_i32(vals, dev)

    def _batch_draft_device(self):
        """Context pass + block pass + draft head on the persistent inputs of the drafter's layout (traceable)."""
        m, d = self.model, self.drafter
        lay = d.layout
        d.context_pass(self.verifier.tap_list(), self.ctx_rope_idx, self._ctx_rows_for(lay.R, lay.groups))
        x = d.embed_block(self.block_tokens_tt, m.embedding_weight)
        h = d.block_forward(x, self.block_rope_idx, self.verifier.kv_write_rows, self.sdpa_pos)
        out = d.draft_head(h, m.lm_head)
        ttnn.deallocate(x)
        ttnn.deallocate(h)
        return out

    def _run_stage(self, key, fn, read):
        """Device stage `key` (eager the first time, then captured and replayed) -> (read(outputs), seconds, traced)."""
        plan = self._plan(key)
        if plan == "capture":
            self._capture(key, fn)
        t0 = time.perf_counter()
        if plan == "eager":
            out = fn()
            res = read(out)
            for t in out if isinstance(out, (tuple, list)) else [out]:
                ttnn.deallocate(t)
        else:
            ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
            res = read(self.trace_out[key])
        return res, time.perf_counter() - t0, plan != "eager"

    def _apply_layout(self, lay):
        if self.drafter.layout != lay:
            self.drafter.set_layout(lay)
        if self.verifier.layout != lay:
            self.verifier.set_layout(lay)
        self._placed = lay

    def _flush_moved_context(self, lay):
        """Feed the pending context rows eagerly, with the layout of the verify that produced them, when `lay` moves a
        pending slot (new R or new group) or leaves it out; otherwise the next draft step consumes them."""
        old = self.layout
        if old is not None and any(
            s.ctx_pending and (old.R != lay.R or u not in lay.slots or old.group_of(u) != lay.group_of(u))
            for u, s in self.slots.items()
            if not s.done
        ):
            self._flush_context(old)

    def propose_slots(self, slots=None):
        """Draft for `slots` (default: every active slot with an anchor) in one 32-row step; R follows their number.
        Consumes the context rows of the last verify. Returns {slot: (drafts [R-1], q or None, cands)}."""
        if slots is None:
            slots = [u for u, s in self.slots.items() if not s.done and s.anchor is not None]
        slots = sorted(slots)
        if not slots:
            return {}
        R = rows_for_users(len(slots), self.k_by_users)
        lay = plan_layout(slots, R, self._placed)
        self._flush_moved_context(lay)
        self._apply_layout(lay)
        self._set_batch_inputs(lay)
        d = self.drafter
        groups, self._t_draft, self._traced_draft = self._run_stage(
            f"bdraft_R{R}", self._batch_draft_device, lambda o: d.read_draft_groups(*o)
        )
        tb = time.perf_counter()
        sel = {}
        for u, (scores, cands, hp) in groups.items():
            s = self.slots[u]
            if s.sp is None:
                sel[u] = (self.selector.select(scores, cands, hp, s.anchor), None, cands)
            else:
                drafts, q = self.selector.sample(scores, cands, hp, s.anchor, s.sp[0], s.gen)
                sel[u] = (drafts, q, cands)
        self._t_select = time.perf_counter() - tb
        return sel

    def verify_slots(self, rows, mode):
        """Target verify of each slot's [anchor] + drafts (1 <= len <= R tokens, padded with its last token) at
        positions p .. p+R-1, in one 32-row step; R follows the number of slots. Sets each slot's anchor to its row 0
        and does not move p. Returns {slot: argmax -> R ints | topk -> (values [R, 32], ids [R, 32]) | logits ->
        torch float32 [R, vocab]}."""
        slots = sorted(rows)
        R = rows_for_users(len(slots), self.k_by_users)
        if self.layout is not None and any(s.ctx_pending for s in self.slots.values() if not s.done):
            self._flush_context(
                self.layout
            )  # no draft step since the last commit: feed its rows before the taps change
        lay = plan_layout(slots, R, self._placed)
        tokens, positions = [0] * DECODE_B_PAD, [-1] * DECODE_B_PAD
        for g, u in enumerate(lay.slots):
            if u is None:
                continue
            s, toks = self.slots[u], list(rows[u])
            assert 1 <= len(toks) <= R, (u, len(toks), R)
            toks += [toks[-1]] * (R - len(toks))
            s.anchor = toks[0]
            for p in range(R):
                tokens[g * R + p] = toks[p]
                positions[g * R + p] = s.p + p
        self._apply_layout(lay)
        v = self.verifier
        v.prepare(tokens, positions)
        res, self._t_verify, self._traced_verify = self._run_stage(
            f"bverify_{mode}_R{R}", lambda: v.forward(mode), lambda o: v.read_out(o, mode)
        )
        self.layout = lay
        out = {}
        for g, u in enumerate(lay.slots):
            if u is not None:
                r = slice(g * R, (g + 1) * R)
                out[u] = (res[0][r], res[1][r]) if mode == "topk" else res[r]
        return out

    def commit_slot(self, slot, count, last_token):
        """Commit `count` tokens (accepted drafts + the bonus) of the slot's last verify; last_token is its new anchor."""
        assert 1 <= count <= self.rows, count
        s = self.slots[slot]
        s.pending, s.ctx_pending = (s.p, count), True
        s.p += count
        s.anchor = last_token

    def release_slot(self, slot):
        """Forget the slot's request; the next prefill into the slot rewrites its KV rows and drafter context."""
        self.slots.pop(slot, None)

    def _warm_start(self, prompt, sampling, return_logits):
        self.start_slot(0, prompt, max_new_tokens=1, sampling=sampling, return_logits=return_logits, warm=True)
        self.release_slot(0)

    def _warm_decode_runs(self, prompt):
        """Every row layout (one per K) in both verify modes, then the inherited single-user path."""
        for users, _ in self.k_by_users:
            for sampling in (None, _WARM_SAMPLING):
                self.generate_batch([prompt] * users, 4, sampling=sampling, ignore_eos=True)
        super()._warm_decode_runs(prompt)

    def batch_round(self):
        """One draft + verify round for every active slot -> {slot: new tokens}."""
        self._t_capture = 0.0
        t0 = time.perf_counter()
        active = sorted(u for u, s in self.slots.items() if not s.done)
        sel = self.propose_slots(active)
        mode = "topk" if any(self.slots[u].sp is not None for u in active) else "argmax"
        ver = self.verify_slots({u: [self.slots[u].anchor] + list(sel[u][0]) for u in active}, mode)
        te = time.perf_counter()
        out = {}
        for u in active:
            s = self.slots[u]
            drafts, q, cands = sel[u]
            if s.sp is not None:
                tp, tids = filtered_probs(ver[u][0], ver[u][1], *s.sp)
                acc, bonus = rejection_sample(drafts, q, cands, tp, tids, s.gen)
            elif mode == "argmax":
                acc, bonus = self._accept_greedy(drafts, ver[u])
            else:  # greedy slot in a sampled batch: top-1 of the target top-k
                vals, ids = ver[u]
                acc, bonus = self._accept_greedy(drafts, [int(ids[i, int(vals[i].argmax())]) for i in range(len(vals))])
            budget = s.budget - len(s.tokens)
            new = (drafts[:acc] + [bonus])[:budget] if s.ignore_eos else self._emit(drafts, acc, bonus, budget)
            self.commit_slot(u, acc + 1, bonus)
            s.tokens += new
            s.accept.append(len(new))
            s.done = (
                len(s.tokens) >= s.budget
                or (not s.ignore_eos and new[-1] in self.stop_ids)
                or s.p + self.rows > self.model.args.max_seq_len
            )
            out[u] = new
        t_accept = time.perf_counter() - te
        total = time.perf_counter() - t0 - self._t_capture
        t_draft, t_select, t_verify = self._t_draft, self._t_select, self._t_verify
        self.batch_rounds.append(
            dict(
                total=total,
                draft=t_draft,
                select=t_select,
                verify=t_verify,
                accept=t_accept,
                other=total - t_draft - t_select - t_verify - t_accept,
                n=sum(len(x) for x in out.values()),
                users=len(active),
                R=self.layout.R,
                traced=self._traced_draft and self._traced_verify,
            )
        )
        return out

    def generate_batch(self, prompts, max_new_tokens, sampling=None, seeds=None, ignore_eos=False):
        """Run len(prompts) users together (slot u = prompts[u]). sampling: None (greedy), one dict for every user, or
        a list of per-user dicts / None; seeds: per-user host RNG seeds (default: the slot id).
        Returns (tokens per user, stats); full_tok_s is the total rate over the traced rounds with every user active."""
        n = len(prompts)
        assert 1 <= n <= min(self.max_users, self.k_by_users[-1][0]), n
        samp = list(sampling) if isinstance(sampling, (list, tuple)) else [sampling] * n
        seeds = list(range(n)) if seeds is None else list(seeds)
        self.slots, self.layout, self._placed, self.batch_rounds = {}, None, None, []
        t0 = time.perf_counter()
        ttft = []
        for u, prompt in enumerate(prompts):
            self.start_slot(u, prompt, max_new_tokens, samp[u], seeds[u], ignore_eos)
            self._sync()
            ttft.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        while any(not s.done for s in self.slots.values()):
            self.batch_round()
        self._sync()
        decode_s = time.perf_counter() - t1
        full = [r for r in self.batch_rounds if r["traced"] and r["users"] == n]
        full_s = sum(r["total"] for r in full)
        tokens = [self.slots[u].tokens for u in range(n)]
        accept = [self.slots[u].accept for u in range(n)]
        stats = {
            "users": n,
            "R": rows_for_users(n, self.k_by_users),
            "ttft_s": ttft,
            "decode_s": decode_s,
            "rounds": len(self.batch_rounds),
            "full_rounds": len(full),
            "full_tok_s": sum(r["n"] for r in full) / full_s if full_s > 0 else 0.0,
            "decode_tok_s_all": sum(len(t) - 1 for t in tokens) / decode_s if decode_s > 0 else 0.0,
            "mean_accept": [sum(a) / max(1, len(a)) for a in accept],
            "mean_ms": {
                k: _mean_ms([r[k] for r in full]) for k in ("draft", "select", "verify", "accept", "other", "total")
            },
        }
        self._end_batch()
        mm = stats["mean_ms"]
        logger.info(
            f"BATCH SUMMARY: users={n} R={stats['R']} | {stats['full_tok_s']:.1f} tok/s total "
            f"({stats['full_tok_s'] / n:.1f}/user, {len(full)} full traced rounds) | all {stats['decode_tok_s_all']:.1f} "
            f"tok/s | mean_accept={[round(a, 2) for a in stats['mean_accept']]} | round ms: draft {mm['draft']:.1f} "
            f"select {mm['select']:.1f} verify {mm['verify']:.1f} accept {mm['accept']:.1f} other {mm['other']:.1f} "
            f"total {mm['total']:.1f}"
        )
        return tokens, stats

    def _end_batch(self):
        """Back to the single-user layout (slot 0, 16 rows) for the inherited single-user API."""
        default = RowLayout(self.rows, (0,))
        self.drafter.set_layout(default)
        self.verifier.set_layout(default)
        self.layout = self._placed = None

    def close(self):
        for rows in self._ctx_rows.values():
            for t in rows:
                ttnn.deallocate(t)
        self._ctx_rows = {}
        super().close()


def build_batch_generator(model, max_users, enable_trace=True, k_by_users=K_BY_USERS, warm=True):
    """Drafter (max_users rings) + selector + DFlash2BatchGenerator on `model` (args.max_batch_size >= max_users).
    Slot u owns row u of kv_page_table(args, max_users).
    warm (with enable_trace): compile every prefill and decode program first (warm)."""
    from models.demos.muse_glimmer_30b.tt.dflash.drafter import TtDFlash2Drafter
    from models.demos.muse_glimmer_30b.tt.dflash.selector import HostCandidateSelector

    args, mesh = model.args, model.mesh_device
    drafter = TtDFlash2Drafter(mesh, args, model.ccl, model.rope, max_users=max_users)
    page_rows = kv_page_table(args, max_users)
    page_table_tt = ttnn.from_torch(
        page_rows,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gen = DFlash2BatchGenerator(
        model,
        drafter,
        HostCandidateSelector.from_checkpoint(),
        page_rows,
        page_table_tt,
        list(args.eos_token_ids),
        enable_trace=enable_trace,
        k_by_users=k_by_users,
    )
    if enable_trace and warm:
        gen.warm()
    return gen
