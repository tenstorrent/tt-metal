# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 phase 2 (P2-3): one served class for 1-32 greedy users, DFlash2 speculation while few users are live
(qwen36_resident/adaptive.py's two geometries in one process), vllm-tt-plugin's runner-model spec contract with
narrow decode (docs/SPEC_DECODE_CONTRACT.md).

Modes, chosen at propose time (the plugin cannot take a proposal back, so a mode change takes effect at the next
call): SPEC (live users <= 8): every live user is a user of the 8 x 4 verify-rows build; propose drafts K = 3 for each,
verify calls run the vrows build. PLAIN (more than 8 live users, until PLAIN_HOLD propose calls at <= SPEC_BACK live
users): every user is in the plain 32-row build; propose offers no drafts, so the runner sends ordinary [B, 1] calls,
served by the rows adapter (device greedy, the plain build). A switch to PLAIN moves the spec users at the next call
(a resolving verify: their commits from accepted_counts; or a narrow call); a switch back moves them at propose time
(state + option-B drafter refill) and drafts for them at once.

Every live request owns a plain row (the rows adapter's SlotMap; its KV pages are its block table in the shared pool);
a spec user index is extra (spec_of[plain row]). Prefill: in SPEC mode while the spec build has room and the live
count stays <= 8, through SpecRowsSession.start_many; otherwise through the rows adapter (the plain build). A verify
call carrying rows of plain-geometry users (no drafts: column 0 only) runs a plain launch for them as well (mixed
steps, right after an arrival that ends SPEC). Greedy only (phase 2a); a token history per request (prompt + every
committed token) feeds the drafter refill.
"""
import os

import torch
from loguru import logger

from qwen36_resident import greedy_only
from qwen36_resident import weights as Q
from qwen36_resident.model import TOP_K
from qwen36_resident.vllm_rows import CHUNK_UNIT, ResidentRowsAdapter, RowsEngine, expand_blocks

SPEC_USERS = 8
SPEC_K = 3
SPEC_BACK = int(os.environ.get("RESIDENT_ADAPTIVE_SPEC_BACK", "6"))  # back to SPEC at <= this many live users ...
PLAIN_HOLD = int(os.environ.get("RESIDENT_ADAPTIVE_PLAIN_HOLD", "32"))  # ... for this many propose calls in a row
PLAIN_ROWS = 32


class _PlainRowsEngine(RowsEngine):
    """the rows adapter's engine over the plain build (the PP prefill's batched pass runs with the drafter's tap writes
    sent to scratch: plain-geometry prompts need no drafter context)"""


class Qwen36ResidentAdaptiveForCausalLM(ResidentRowsAdapter):
    greedy_only = True
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,  # (the plain build's narrow calls: the hub's greedy)
        "max_device_top_k": TOP_K,
        "supports_device_penalties": False,
        "supports_spec_decode": True,
        "spec_requirements": ["device_propose"],
        "output_tokens_per_step": 1,
    }

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if int(max_num_seqs) > PLAIN_ROWS:
            return SpecReject(reason=f"up to {PLAIN_ROWS} users (max_num_seqs {max_num_seqs})", supported_k=())
        if int(requested_k) < 1:
            return SpecReject(reason="no drafts requested", supported_k=(SPEC_K,))
        return SpecPlan(
            effective_k=min(SPEC_K, int(requested_k)),
            lanes_per_request=SPEC_K + 1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, **kwargs):
        from vllm.transformers_utils.config import uses_mrope

        path = Q.checkpoint_dir(
            os.environ.get("HF_MODEL") or hf_config._name_or_path, getattr(hf_config, "_commit_hash", None)
        )
        assert max_batch_size <= PLAIN_ROWS, f"{cls.__name__}: max_num_seqs <= {PLAIN_ROWS}"
        return cls(mesh_device, Q.Checkpoint(path), max_seq_len, max_batch_size, uses_mrope(hf_config))

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, max_num_seqs=None, **kwargs):
        if os.environ.get("RESIDENT_MAX_TOKENS_ALL_USERS"):
            return int(os.environ["RESIDENT_MAX_TOKENS_ALL_USERS"])
        return int(max_model_len) * int(max_num_seqs or 1)

    def __init__(self, mesh, ck, max_seq_len, max_batch, mrope):
        greedy_only.install()
        self.mesh, self.ck = mesh, ck
        super().__init__(None, ck.config["vocab_size"], max_seq_len, PLAIN_ROWS, CHUNK_UNIT, mrope, rows=PLAIN_ROWS)
        self.ae = self.sess = None
        self.spec_of = {}  # plain row (resident slot) -> spec user index
        self.history = {}  # resident slot -> every token so far (prompt + committed); [-1] is the next to run
        self.last_acc = {}  # resident slot -> tokens its last spec step committed (its next verify applies them)
        self.mode, self.low_calls = "SPEC", 0

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        from qwen36_resident.adaptive import AdaptiveEngine

        nb = int(kv_cache_shape[0])
        logger.info(f"Resident adaptive: {SPEC_USERS} x 4 verify rows + {PLAIN_ROWS} plain rows, KV pool {nb} blocks")
        self.ae = AdaptiveEngine(self.mesh, self.ck, self.max_seq_len + 2 * 4 + 64, nb)
        self.sess = self.ae.sess
        self.ae.plain_tables = self.tables  # (one list: the engine's moves and the rows adapter write the same tables)
        d = self.ae.d
        self.engine = _PlainRowsEngine(
            self.ae.plain, self.ae.pp, self.ae.emb, self.vocab, d.vocab_chip, PLAIN_ROWS, CHUNK_UNIT, self.mesh
        )
        return []

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass

    # ------------------------------------------------------------------ helpers
    def _free_spec(self):
        used = set(self.spec_of.values())
        return [u for u in range(SPEC_USERS) if u not in used]

    def _note(self, r, pos, tok):
        """the history of slot r: token tok at position pos (earlier entries kept, later dropped)"""
        h = self.history.setdefault(r, [])
        if pos < len(h):
            del h[pos:]
        if pos == len(h):
            h.append(int(tok))

    def _to_plain_all(self, acc_of=None):
        for r, u in sorted(self.spec_of.items()):
            acc = (acc_of or {}).get(r, self.last_acc.get(r, 1))
            self.ae.move_to_plain(u, r, self.tables[r], acc)
            self.last_acc[r] = 1
        self.spec_of.clear()

    def _to_spec_all(self):
        free = self._free_spec()
        for r in sorted(self.live):
            if r in self.spec_of:
                continue
            u = free.pop(0)
            self.ae.move_to_spec(r, u, self.tables[r], self.history[r])
            self.spec_of[r], self.last_acc[r] = u, 1

    def _plain_tokens(self, feed):
        """one plain launch of the slots in feed {slot: (token, position)} (the others held): their device greedy"""
        sitting = {s for s in self.live if s not in feed}
        self.engine.decode(feed, [0], preserve_launches={0} if sitting else ())
        self.engine.sync()
        return dict(zip(feed, self.engine.tokens_from(self.engine.read_tokens(list(feed), False))))

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, empty_slots=None, **kwargs):
        n = int(len(prompt_lens))
        if start_pos is not None and any(int(x) > 0 for x in start_pos[:n]):
            raise RuntimeError("the adaptive profile prefills whole prompts (serve without the TT chunk core)")
        if empty_slots is None:
            raise RuntimeError("a stateful multi-user model needs the plugin's empty_slots (destination slots)")
        live_after = len(self.live) + n
        if self.mode == "PLAIN" and not self.live:  # (nobody to move: an idle server starts speculating again)
            self.mode, self.low_calls = "SPEC", 0
        to_spec = self.mode == "SPEC" and live_after <= SPEC_USERS and len(self._free_spec()) >= n
        out = torch.zeros(n, 1, self.vocab)
        if to_spec:
            free, items = self._free_spec(), []
            for j in range(n):
                T = int(prompt_lens[j])
                r = self._start_prompt(int(empty_slots[j]))
                blocks = expand_blocks(page_table[j].tolist(), self.pages_per_block)
                self.tables[r] = list(blocks)
                u = free.pop(0)
                self.spec_of[r], self.last_acc[r] = u, 1
                self.ae._set_spec_table(u, blocks)  # (the engine's table list stays the one source)
                ids = [int(t) for t in tokens[j, :T]]
                self.history[r] = list(ids)
                items.append((u, ids, None))
            for j, lg in enumerate(self.sess.start_many(items)):
                masked = torch.full_like(lg, float("-inf"))
                masked[int(lg.argmax())] = 0.0  # (greedy: the runner samples the argmax)
                out[j, 0] = masked
                self.live.add(self.slots.res_of[int(empty_slots[j])])
            self._write_tables()
        else:
            if self.mode == "SPEC" and self.spec_of and live_after > SPEC_USERS:
                logger.info(f"adaptive: {live_after} live users -> PLAIN")
                self.mode = "PLAIN"
            res = super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, empty_slots=empty_slots, **kwargs)
            out = res[0] if isinstance(res, tuple) else res
            for j in range(n):
                r = self.slots.res_of[int(empty_slots[j])]
                self.history[r] = [int(t) for t in tokens[j, : int(prompt_lens[j])]]
        return (out, torch.zeros(n, dtype=torch.long)) if self.mrope else out

    # ------------------------------------------------------------------ decode
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        *args,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        slot_remap=None,
        **kwargs,
    ):
        res_of = self.slots.remapped(slot_remap) if slot_remap is not None else dict(self.slots.res_of)
        B = int(tokens.shape[0])
        W = int(tokens.shape[1]) if tokens.dim() > 1 else 1
        pos = start_pos.reshape(B, -1)
        rows = {i: res_of[i] for i in range(B) if int(pos[i, 0]) >= 0 and res_of.get(i) in self.live}
        tok0 = tokens.reshape(B, -1)[:, 0].tolist()
        for i, r in rows.items():
            self._note(r, int(pos[i, 0]), tok0[i])
        # (the spec users' tables here; the plain rows' the rows adapter keeps — every row's 4,096-entry table converted
        # and compared each call cost several ms a step at 32 users)
        if page_table is not None and (spec_mode is not None or self.spec_of):
            changed = False
            for i, r in rows.items():
                if r not in self.spec_of and spec_mode is None:
                    continue
                tbl = expand_blocks(page_table[i].tolist(), self.pages_per_block)
                if tbl != self.tables[r]:
                    self.tables[r], changed = tbl, True
                    if r in self.spec_of:
                        self.ae._set_spec_table(self.spec_of[r], tbl)
            if changed:
                self._write_tables()
        if spec_mode is None:
            if self.mode == "PLAIN" and self.spec_of:
                self._to_plain_all()
            if not self.spec_of:  # (every user plain: the rows adapter's own path)
                return super().decode_forward(tokens, start_pos, page_table, *args, slot_remap=slot_remap, **kwargs)
            out = self._spec_row0(rows, tokens.reshape(B, -1)[:, 0], pos[:, 0], B)
            self._commit_remap(slot_remap, res_of)
            return out
        return self._verify(tokens, pos, rows, B, W, num_valid_drafts, accepted_counts, spec_mode, slot_remap, res_of)

    def _commit_remap(self, slot_remap, res_of):
        if slot_remap is not None:
            remap = [int(j) for j in slot_remap]
            self._expect_moves = {j: i for i, j in enumerate(remap) if i != j}
            self.slots.res_of = res_of

    def _spec_row0(self, rows, tok0, pos0, B):
        """a narrow call while users are in the spec build (right after their start, before any drafts): a vrows step
        of row 0 (the drafts' rows run a dummy) for them, a plain launch for the plain ones; host token ids [B, 1]"""
        U, R = SPEC_USERS, 4
        toks, upos, act, of_u = [[0] * R for _ in range(U)], [0] * U, [False] * U, {}
        feed = {}
        for i, r in rows.items():
            if r in self.spec_of:
                u = self.spec_of[r]
                toks[u], upos[u], act[u], of_u[i] = [int(tok0[i])] + [0] * (R - 1), int(pos0[i]), True, u
            else:
                feed[r] = (int(tok0[i]), int(pos0[i]))
        res = torch.zeros(B, 1, dtype=torch.int32)
        if of_u:
            g = self.sess.verify(toks, upos, act)
            for i, u in of_u.items():
                res[i, 0] = g[u][0]
                self.last_acc[rows[i]] = 1
        if feed:
            got = self._plain_tokens(feed)
            for i, r in rows.items():
                if r in got:
                    res[i, 0] = got[r]
        return res

    def _verify(self, tokens, pos, rows, B, W, num_valid_drafts, accepted_counts, spec_mode, slot_remap, res_of):
        from vllm_tt_plugin.spec_decode import VerifyOutput

        U, R = SPEC_USERS, 4
        tk = tokens.reshape(B, W)
        acc_of = {r: int(accepted_counts[i]) for i, r in rows.items()} if accepted_counts is not None else {}
        nvd = {i: int(num_valid_drafts[i]) for i in rows} if num_valid_drafts is not None else {}
        if self.mode == "PLAIN" and self.spec_of and not any(nvd.values()):
            self._to_plain_all(acc_of)  # (a resolving verify: the spec users' commits, then the plain build)
        out = torch.zeros(B, W, dtype=torch.int32)
        toks, upos, act, of_u, feed = [[0] * R for _ in range(U)], [0] * U, [False] * U, {}, {}
        for i, r in rows.items():
            row = [int(t) for t in tk[i].tolist()]
            if r in self.spec_of:
                u = self.spec_of[r]
                row = [t if 0 <= t < self.vocab else 0 for t in row] + [0] * (R - W)
                toks[u], upos[u], act[u], of_u[i] = row[:R], int(pos[i, 0]), True, u
            else:
                assert nvd.get(i, 0) == 0, "a plain-geometry row with drafts"
                feed[r] = (row[0], int(pos[i, 0]))
        if of_u:
            g = self.sess.verify(toks, upos, act)
            for i, u in of_u.items():
                out[i] = torch.tensor(g[u][:W], dtype=torch.int32)
        if feed:
            got = self._plain_tokens(feed)
            for i, r in rows.items():
                if r in got:
                    out[i, 0] = got[r]
        self._commit_remap(slot_remap, res_of)
        return VerifyOutput(spec_mode=spec_mode, argmax_ids=out)

    # ------------------------------------------------------------------ propose
    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        B, K = int(committed_tokens.shape[0]), int(num_drafts)
        res_of = dict(self.slots.res_of)
        rows = {}
        for i in range(B):
            a = int(accepted_counts[i])
            p = int(committed_positions[i, a - 1])
            r = res_of.get(i)
            if p < 0 or r is None or r not in self.live:
                continue
            for c in range(a):
                self._note(r, int(committed_positions[i, c]), int(committed_tokens[i, c]))
            rows[i] = (r, a, int(committed_tokens[i, a - 1]), p)
            self.last_acc[r] = a
        live = len(self.live)
        # ---- the mode for the next step
        if live > SPEC_USERS:
            if self.mode == "SPEC":
                logger.info(f"adaptive: {live} live users -> PLAIN")
            self.mode, self.low_calls = "PLAIN", 0
        elif self.mode == "PLAIN":
            self.low_calls = self.low_calls + 1 if live <= SPEC_BACK else 0
            if self.low_calls >= PLAIN_HOLD and live > 0:
                logger.info(f"adaptive: {live} live users for {self.low_calls} steps -> SPEC")
                self._to_spec_all()  # (state moves + drafter refill; their drafts below)
                self.mode, self.low_calls = "SPEC", 0
        ids = torch.zeros(B, K, dtype=torch.int32)
        valid = torch.zeros(B, dtype=torch.int32)
        if self.mode == "SPEC" and self.spec_of:
            U = SPEC_USERS
            anchors, positions, accepted, active, row_u = [0] * U, [0] * U, [1] * U, [False] * U, {}
            for i, (r, a, tok, p) in rows.items():
                if r in self.spec_of:
                    u = self.spec_of[r]
                    anchors[u], positions[u], accepted[u], active[u], row_u[i] = tok, p, a, True, u
            drafts = self.sess.propose(anchors, positions, accepted, active)
            for i, u in row_u.items():
                ids[i] = torch.tensor(drafts[u][:K], dtype=torch.int32)
                valid[i] = K
        return DraftOutput(draft_token_ids=ids, num_valid=valid)

    # ------------------------------------------------------------------ lifecycle
    def release_request(self, vslot):
        r = self.slots.res_of.get(int(vslot))
        super().release_request(vslot)
        if r is not None:
            if r in self.spec_of:
                u = self.spec_of.pop(r)
                self.ae._set_spec_table(u, [0])
            self.history.pop(r, None)
            self.last_acc.pop(r, None)
