# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 phase 1: greedy DFlash2 for up to 8 users in one launch, served through vllm-tt-plugin's runner-model spec
contract (vllm-tt-plugin docs/SPEC_DECODE_CONTRACT.md).

Qwen36ResidentSpecRowsForCausalLM drives dflash.SpecRowsSession over a ResidentDFlashRows (verify rows: 8 users x 4
rows, the PP prefill with the drafter's taps, the drafter's 8 x 4 layout):
  prefill_forward   a whole prompt per row: SpecRowsSession.start into the row's user (its resident slot); returns its
                    last row's logits masked to the argmax (greedy only, as the single-user DFlash2 profile);
  decode_forward    the verify call ([B, 1 + K] tokens / positions, num_valid_drafts, accepted_counts, spec_mode):
                    one verify step of every live user, VerifyOutput(argmax_ids [B, 1 + K]);
  propose_draft_tokens  after every commit: DraftOutput(drafts [B, K]) from each live user's last committed token.
The rows adapter keeps the slots: a wire row's resident slot is its user. Whole prompts only (the profile runs without
the chunk core); no narrow decode, no async scheduling; requests are decoded greedily.
"""
import os

import torch
from loguru import logger

from qwen36_resident import greedy_only
from qwen36_resident import weights as Q
from qwen36_resident.vllm_rows import ResidentRowsAdapter, expand_blocks

SPEC_R = int(os.environ.get("RESIDENT_SPEC_ROWS", "4"))  # verify rows per user (4: up to 8 users per launch)
SPEC_USERS = 32 // SPEC_R
# offer no drafts (num_valid 0): every verify commits row 0's greedy token, the same engine's plain greedy decode
# (the served reference a speculative run's texts must equal)
NO_DRAFTS = os.environ.get("RESIDENT_SPEC_NO_DRAFTS", "0") == "1"


class Qwen36ResidentSpecRowsForCausalLM(ResidentRowsAdapter):
    greedy_only = True
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
        "supports_device_penalties": False,
        "supports_spec_decode": True,
        "spec_requirements": ["device_propose"],
        "output_tokens_per_step": 1,
    }

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        k = SPEC_R - 1
        if int(max_num_seqs) > SPEC_USERS:
            return SpecReject(
                reason=f"verify rows of {SPEC_R} serve up to {SPEC_USERS} users (max_num_seqs {max_num_seqs})",
                supported_k=(),
            )
        if int(requested_k) < 1:
            return SpecReject(reason="no drafts requested", supported_k=(k,))
        return SpecPlan(
            effective_k=min(k, int(requested_k)),
            lanes_per_request=SPEC_R,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
        )

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, **kwargs):
        from vllm.transformers_utils.config import uses_mrope

        path = Q.checkpoint_dir(
            os.environ.get("HF_MODEL") or hf_config._name_or_path, getattr(hf_config, "_commit_hash", None)
        )
        assert max_batch_size <= SPEC_USERS, f"{cls.__name__}: max_num_seqs <= {SPEC_USERS}"
        return cls(mesh_device, Q.Checkpoint(path), max_seq_len, max_batch_size, uses_mrope(hf_config))

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, max_num_seqs=None, **kwargs):
        users = int(max_num_seqs or 1)
        if os.environ.get("RESIDENT_MAX_TOKENS_ALL_USERS"):
            return int(os.environ["RESIDENT_MAX_TOKENS_ALL_USERS"])
        return int(max_model_len) * users

    def __init__(self, mesh, ck, max_seq_len, max_batch, mrope):
        greedy_only.install()
        self.mesh, self.ck, self.max_seq_len = mesh, ck, max_seq_len
        cfg = ck.config
        super().__init__(None, cfg["vocab_size"], max_seq_len, SPEC_USERS, 1024, mrope, rows=SPEC_USERS)
        self.sess = None

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """the verify-rows engine on a KV pool of the plugin's block count (its block ids index the pool)"""
        from qwen36_resident.dflash import ResidentDFlashRows, SpecRowsSession

        nb = int(kv_cache_shape[0])
        logger.info(f"Resident spec rows: {SPEC_USERS} users x {SPEC_R} verify rows, KV pool {nb} blocks")
        eng = ResidentDFlashRows(
            self.mesh,
            self.ck,
            users=SPEC_USERS,
            max_pos=self.max_seq_len + 2 * SPEC_R + 64,
            R=SPEC_R,
            prefill=True,
            kv_pool_blocks=nb,
        )
        self.sess = SpecRowsSession(eng)
        self.engine = _SpecEngine(eng)
        return []

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass

    # ---- prefill: a whole prompt per row into its user
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, empty_slots=None, **kwargs):
        n = int(len(prompt_lens))
        if start_pos is not None and any(int(x) > 0 for x in start_pos[:n]):
            raise RuntimeError("the spec rows profile prefills whole prompts (serve without the TT chunk core)")
        if empty_slots is None:
            raise RuntimeError("a stateful multi-user model needs the plugin's empty_slots (destination slots)")
        out = torch.zeros(n, 1, self.vocab)
        for j in range(n):
            T = int(prompt_lens[j])
            r = self._start_prompt(int(empty_slots[j]))
            # (the whole row: the first verify's rows write past the prompt; vLLM's padding is its null block 0)
            blocks = expand_blocks(page_table[j].tolist(), self.pages_per_block)
            self.tables[r] = list(blocks)
            lg = self.sess.start(r, [int(t) for t in tokens[j, :T]], table=blocks)
            self.live.add(r)
            masked = torch.full_like(lg, float("-inf"))
            masked[int(lg.argmax())] = 0.0  # (greedy: the runner samples the argmax)
            out[j, 0] = masked
        return (out, torch.zeros(n, dtype=torch.long)) if self.mrope else out

    # ---- verify
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
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if spec_mode is None:
            raise RuntimeError("the spec rows model serves the verify call only (no narrow decode)")
        res_of = self.slots.remapped(slot_remap) if slot_remap is not None else dict(self.slots.res_of)
        B, W = int(tokens.shape[0]), int(tokens.shape[1])
        pos = start_pos.reshape(B, W)
        U, R = self.sess.e.U, self.sess.e.R
        toks = [[0] * R for _ in range(U)]
        upos, active, row_user = [0] * U, [False] * U, {}
        changed = False
        for i in range(B):
            p0 = int(pos[i, 0])
            if p0 < 0:
                continue
            u = res_of.get(i)
            if u is None or u not in self.live:
                raise RuntimeError(f"verify row {i} has no live resident slot (map {res_of})")
            row = [int(t) for t in tokens[i].tolist()]
            # (padded draft columns carry a placeholder id: any valid id stands in, their outputs are not read)
            row = [t if 0 <= t < self.vocab else 0 for t in row] + [0] * (R - W)
            toks[u], upos[u], active[u], row_user[i] = row[:R], p0, True, u
            if page_table is not None:
                tbl = expand_blocks(page_table[i].tolist(), self.pages_per_block)
                if tbl != self.tables[u]:
                    self.tables[u], changed = tbl, True
        if changed:
            e = self.sess.e
            full = [list(t) for t in e.m.block_tables]
            for u in range(U):
                if self.tables[u]:
                    for r in range(u * R, (u + 1) * R):
                        full[r] = list(self.tables[u])
            e.m.set_page_table(full)
        g = self.sess.verify(toks, upos, active)
        out = torch.zeros(B, W, dtype=torch.int32)
        for i, u in row_user.items():
            out[i] = torch.tensor(g[u][:W], dtype=torch.int32)
        if slot_remap is not None:  # accepted: commit the remap (exactly once), expect the plugin's note of its moves
            remap = [int(j) for j in slot_remap]
            self._expect_moves = {j: i for i, j in enumerate(remap) if i != j}
            self.slots.res_of = res_of
        return VerifyOutput(spec_mode=spec_mode, argmax_ids=out)

    # ---- propose
    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        B, K = int(committed_tokens.shape[0]), int(num_drafts)
        U = self.sess.e.U
        anchors, positions, accepted, active, row_user = [0] * U, [0] * U, [1] * U, [False] * U, {}
        res_of = dict(self.slots.res_of)
        for i in range(B):
            a = int(accepted_counts[i])
            p = int(committed_positions[i, a - 1])
            u = res_of.get(i)
            if p < 0 or u is None or u not in self.live:
                continue
            anchors[u], positions[u], accepted[u], active[u], row_user[i] = (
                int(committed_tokens[i, a - 1]),
                p,
                a,
                True,
                u,
            )
        drafts = self.sess.propose(anchors, positions, accepted, active)
        ids = torch.zeros(B, K, dtype=torch.int32)
        valid = torch.zeros(B, dtype=torch.int32)
        for i, u in row_user.items():
            ids[i] = torch.tensor(drafts[u][:K], dtype=torch.int32)
            valid[i] = 0 if NO_DRAFTS else K
        return DraftOutput(draft_token_ids=ids, num_valid=valid)


class _SpecEngine:
    """the bits of the rows engine the adapter's slot bookkeeping asks for"""

    def __init__(self, eng):
        self.e = eng
        self.device_sampling = False

    def row_order(self):
        return list(range(self.e.U))

    def set_page_table(self, tables):
        pass

    def sync(self):
        import ttnn

        ttnn.synchronize_device(self.e.mesh)
