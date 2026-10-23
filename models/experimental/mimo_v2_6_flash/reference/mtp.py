# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fp32 reference of MiMo's MTP layers (model.mtp.layers.i), as vLLM's ``MiMoV2MTP`` runs them, and of the drafter.

HF's modeling code ignores the MTP weights, so vLLM's implementation is the reference: at position t a layer takes
the target's final normed hidden state at t and the embedding of a token, projects cat(enorm(embedding),
hnorm(hidden)) with ``eh_proj``, runs a decoder block (SWA attention with sinks, a dense MLP) and normalizes with its
own ``final_layernorm``; the target's LM head turns that into a token. The embedding and LM head are the target's.

ReferenceMTPDrafter drafts MAX_DRAFTS tokens per step in one of two ways (reference/spec_decode.py has the equations):

* one layer (vLLM's MTP proposer, upstream vLLM for MiMo): layer 0 at t takes the token at t + 1 and predicts the
  token at t + 2; drafts 2 and 3 feed it its own output as the hidden, at the next positions;
* every layer (vLLM 0.31's MultiModuleMTPSpeculator with the num_nextn_predict_layers patch): draft i + 1 from layer i,
  each taking the target's hidden at t and the token at t + i + 1, predicting the token at t + i + 2, every layer's
  query at the position before the last committed token.

and, for the two requests of a lane that drafts for both (reference/spec_decode.py ``plan_mtp_pair``), one token
each: layer 0 over each request's committed rows, the draft from its last one. Every request keeps its own caches
(its MTP ring); the lane's stash serves the request that drafts alone.
"""

import torch

from .model import ReferenceDecoderLayer, load_layer_weights, rms_norm
from .spec_decode import (
    MAX_DRAFTS,
    MTP_RING_TOKENS,
    SPEC_SLOTS,
    MtpLayersInputs,
    MtpPairInputs,
    MtpStash,
    mtp_layer_tokens,
    mtp_stash,
    plan_mtp,
    plan_mtp_layers,
    plan_mtp_pair,
)


def cache_ring(cache_position, ring=MTP_RING_TOKENS):
    """The MTP ring a cache position (spec_decode.mtp_cache_position) addresses."""
    return int(cache_position) // (2 * ring)


def load_mtp_weights(ckpt, cfg, i):
    w = load_layer_weights(ckpt, cfg, cfg.mtp_layer_index(i))
    w["head"] = ckpt.mtp_head(i)
    return w


class ReferenceMTP:
    def __init__(self, cfg, i=0, ckpt=None, weights=None):
        """MTP layer ``i`` from ``ckpt``, or prepared ``weights`` (load_mtp_weights' keys) for synthetic tests."""
        self.cfg = cfg
        if weights is None:
            weights = load_mtp_weights(ckpt, cfg, i)
        self.head = weights["head"]
        self.layer = ReferenceDecoderLayer(cfg, cfg.mtp_layer_index(i), weights=weights)

    def __call__(self, hidden, embeds, positions, history=None):
        """hidden [S, hidden]: the target's (or, one-layer drafting, this layer's) final normed hidden at
        ``positions`` [S]; embeds [S, hidden]: the embeddings of the tokens this layer takes there (layer i: i + 1
        positions later). Returns the layer's final normed output [S, hidden]; its LM head logits predict the tokens
        one position after the embedded ones. ``history`` keeps this layer's K / V."""
        eps = self.cfg.layernorm_epsilon
        e = rms_norm(embeds.float(), self.head["enorm"], eps)
        h = rms_norm(hidden.float(), self.head["hnorm"], eps)
        x = torch.cat((e, h), dim=-1) @ self.head["eh_proj"].T
        y = self.layer(x, positions, history)
        return rms_norm(y, self.head["final_layernorm"], eps)


class PositionalHistory:
    """An MTP layer's KV cache on the host with the device ring's semantics: K / V kept by position, a later write at
    a position replaces the earlier one (reference.model's KVHistory appends)."""

    def __init__(self):
        self.entries = {}

    def append(self, k, v, pos):
        for i, p in enumerate(pos.tolist()):
            self.entries[p] = (k[:, i], v[:, i])
        keys = sorted(self.entries)
        k_all = torch.stack([self.entries[p][0] for p in keys], dim=1)
        v_all = torch.stack([self.entries[p][1] for p in keys], dim=1)
        return k_all, v_all, torch.tensor(keys)


class ReferenceMTPDrafter:
    """The adapter's propose step for one lane, on the host. ``mtps``: the ReferenceMTP layers, one (one-layer
    drafting: layer 0 drafts every token) or MAX_DRAFTS (draft i + 1 from layer i); ``embedding`` [vocab, hidden] and
    ``lm_head`` [vocab, hidden] the target's. Each layer keeps its K / V in a PositionalHistory per MTP ring (one per
    request, ``ring_histories``); with every layer the drafter also keeps the lane's MtpStash rows (target hidden,
    layer-0 token) for the next re-prefill.

    plan() plans a request's draft from where the last step left it (reference.spec_decode.plan_mtp or
    plan_mtp_layers), plan_pair() the pair draft of two (plan_mtp_pair); draft() runs either from the slots' tokens and
    target hidden rows, as the device program would."""

    def __init__(self, mtps, embedding, lm_head):
        assert len(mtps) in (1, MAX_DRAFTS), f"{len(mtps)} MTP layers: one, or one per draft"
        self.mtps, self.embedding, self.lm_head = list(mtps), embedding, lm_head
        self.reset()

    @property
    def multi_layer(self):
        return len(self.mtps) > 1

    def reset(self):
        """A new request's prompt: no K / V, nothing stashed."""
        self.rings = {}
        self.stash = MtpStash.empty() if self.multi_layer else None
        self.stash_rows = [None] * SPEC_SLOTS  # (hidden, layer-0 token) of each stash slot
        self.outputs = None  # each draft's final normed output at its row (the anchor), of the last draft

    def ring_histories(self, ring):
        """The caches (one PositionalHistory per layer) of MTP ring ``ring``."""
        return self.rings.setdefault(int(ring), [PositionalHistory() for _ in self.mtps])

    @property
    def histories(self):
        """Ring 0's caches (a lane serving one request in slot 0)."""
        return self.ring_histories(0)

    def plan(self, source, committed_positions, accepted, ring=0):
        if self.multi_layer:
            return plan_mtp_layers(source, committed_positions, accepted, self.stash, len(self.mtps), ring=ring)
        return plan_mtp(source, committed_positions, accepted, ring=ring)

    def plan_pair(self, sources, committed_positions, accepted, rings):
        return plan_mtp_pair(sources, committed_positions, accepted, rings)

    def _greedy(self, h):
        return int(torch.argmax(h @ self.lm_head.T))

    def draft(self, plan, slot_tokens, slot_hidden):
        """``slot_tokens`` / ``slot_hidden``: per slot of the last step, the token committed there and the target's
        final normed hidden that chose it. Returns the MAX_DRAFTS drafts."""
        if isinstance(plan, MtpLayersInputs):
            return self._draft_layers(plan, slot_tokens, slot_hidden)
        if isinstance(plan, MtpPairInputs):
            return self._draft_pair(plan, slot_tokens, slot_hidden)
        mtp, history = self.mtps[0], self.ring_histories(plan.ring)[0]
        fill = [s for s in range(SPEC_SLOTS) if plan.fill[s] >= 0]
        out = mtp(
            torch.stack([slot_hidden[s] for s in fill]),
            self.embedding[torch.tensor([slot_tokens[s] for s in fill])],
            torch.tensor([plan.fill[s] for s in fill]),
            history,
        )
        h = out[fill.index(plan.anchor)]
        drafts, self.outputs = [self._greedy(h)], [h]
        for positions in (plan.step2, plan.step3):
            position = torch.tensor([positions[plan.anchor]])
            h = mtp(h[None], self.embedding[torch.tensor([drafts[-1]])], position, history)[0]
            drafts.append(self._greedy(h))
            self.outputs.append(h)
        return drafts

    def _draft_layers(self, plan, slot_tokens, slot_hidden):
        """Every layer over the plan's rows (stash rows, then the step's), layer i on the tokens shifted by i; the
        rows become the stash."""
        rows = plan.rows
        assert rows and rows[-1] == plan.anchor
        hidden, tokens0 = [], []
        for s in rows:
            kind, k = plan.sources[s]
            if kind == "stash":
                h, t = self.stash_rows[k]
            else:
                h, t = slot_hidden[k], slot_tokens[k]
            hidden.append(h)
            tokens0.append(int(t))
        hidden = torch.stack(hidden)
        positions = torch.tensor([plan.positions[s] for s in rows])
        drafts, self.outputs = [], []
        for i, (mtp, history) in enumerate(zip(self.mtps, self.ring_histories(plan.ring))):
            tokens = mtp_layer_tokens(tokens0, drafts, i)
            h = mtp(hidden, self.embedding[torch.tensor(tokens)], positions, history)[-1]
            drafts.append(self._greedy(h))
            self.outputs.append(h)
        self.stash = mtp_stash(plan)
        self.stash_rows = [
            (hidden[rows.index(s)], tokens0[rows.index(s)]) if s in rows else None for s in range(SPEC_SLOTS)
        ]
        return drafts

    def _draft_pair(self, plan, slot_tokens, slot_hidden):
        """Layer 0 over each source's rows, in its own ring; returns each source's one draft (None: it drafts nothing).
        The stash is the lone request's: a pair draft leaves it."""
        drafts, self.outputs = [], []
        for anchor in plan.anchors:
            if anchor is None:
                drafts.append(None)
                self.outputs.append(None)
                continue
            ring = plan.rings[anchor]
            rows = [s for s in plan.rows if plan.rings[s] == ring]
            out = self.mtps[0](
                torch.stack([slot_hidden[s] for s in rows]),
                self.embedding[torch.tensor([slot_tokens[s] for s in rows])],
                torch.tensor([plan.fill[s] for s in rows]),
                self.ring_histories(ring)[0],
            )
            h = out[rows.index(anchor)]
            drafts.append(self._greedy(h))
            self.outputs.append(h)
        return drafts


class MtpProgramMirror:
    """The every-layer draft program of one replica (tt/mtp.py ``_body_layers``) on the host, step for step on the
    kernels' integer mirrors (reference/spec_chain.py): the plan step on the packed upload and the stash page, the
    hidden gather from the table [the step's final norm rows, the stash rows] (which becomes the stash), and pass i on
    MTP layer i over the plan's slots with the tokens step's tokens, each writing its own cache in the ring its cache
    positions address (PositionalHistory per ring, the device ring's semantics), its greedy token per used slot into
    head i; then the drafts step. ``run_pair``: the pair draft program (tt/mtp.py ``_body_pair``). ``mtps``:
    MAX_DRAFTS ReferenceMTP layers; ``embedding`` / ``lm_head`` the target's; ``table_rows`` T, the final norm buffer's
    rows."""

    def __init__(self, mtps, embedding, lm_head, table_rows=32):
        from .spec_chain import stash_page

        assert len(mtps) == MAX_DRAFTS
        self.mtps, self.embedding, self.lm_head, self.table_rows = list(mtps), embedding, lm_head, table_rows
        self.rings = {}
        self.stash = stash_page(table_rows)
        self.stash_hidden = torch.zeros(table_rows, embedding.shape[-1])
        self.heads = [[0] * table_rows for _ in self.mtps]
        self.outputs = None

    def reset(self):
        """TtMiMoMTP.reset: zero rings (no K / V), an empty stash page."""
        from .spec_chain import stash_page

        self.rings = {}
        self.stash = stash_page(self.table_rows)

    def ring_histories(self, ring):
        return self.rings.setdefault(int(ring), [PositionalHistory() for _ in self.mtps])

    @property
    def histories(self):
        return self.ring_histories(0)

    def run_pair(self, mtp, step_hidden):
        """One replay of the pair draft program: ``mtp`` the packed upload (pack_mtp_pair's layout), ``step_hidden``
        the final norm buffer [T, hidden]. MTP layer 0 over the fill's slots, each ring's rows on that ring's cache,
        the hidden rows the slots' own; returns the result's drafts section (drafts_step, pair)."""
        from .spec_chain import drafts_step
        from .spec_inputs import MTP_ROWS

        b = len(mtp) // MTP_ROWS
        fill, tokens, rope = mtp[b : 2 * b], mtp[:b], mtp[4 * b : 5 * b]
        head = [0] * self.table_rows
        self.outputs = []
        used = [s for s in range(SPEC_SLOTS) if fill[s] >= 0]
        for ring in sorted({cache_ring(fill[s]) for s in used}):
            rows = [s for s in used if cache_ring(fill[s]) == ring]
            out = self.mtps[0](
                step_hidden.float()[torch.tensor(rows)],
                self.embedding[torch.tensor([tokens[s] for s in rows])],
                torch.tensor([rope[s] for s in rows]),
                self.ring_histories(ring)[0],
            )
            for k, s in enumerate(rows):
                head[s] = int(torch.argmax(out[k] @ self.lm_head.T))
            self.outputs.append(out[-1])
        self.heads[0] = head
        return drafts_step(mtp, [head], pair=True)

    def run(self, mtp, step_hidden):
        """One replay: ``mtp`` the packed upload (int32 values [MTP_ROWS * B], B = table_rows), ``step_hidden`` the
        final norm buffer [T, hidden]. Returns the result's drafts section (drafts_step)."""
        from .spec_chain import drafts_step, plan_step, tokens_step

        inputs, self.stash = plan_step(mtp, self.stash, self.table_rows)
        rows = torch.cat((step_hidden.float(), self.stash_hidden))[torch.tensor(inputs["hidden"])]
        self.stash_hidden = rows.clone()
        used = [j for j in range(SPEC_SLOTS) if inputs["pos"][j] >= 0]
        positions = torch.tensor([inputs["rope"][j] for j in used])
        rings = {cache_ring(inputs["pos"][j]) for j in used}
        assert len(rings) <= 1, f"one request's rows in rings {rings}"
        histories = self.ring_histories(rings.pop() if rings else 0)
        self.outputs = []
        for i, (mtp_layer, history) in enumerate(zip(self.mtps, histories)):
            tokens = inputs["tokens0"] if i == 0 else tokens_step(inputs["tokens0"], self.heads, i)
            head = [0] * self.table_rows
            if used:
                embeds = self.embedding[torch.tensor([tokens[j] for j in used])]
                out = mtp_layer(rows[used], embeds, positions, history)
                for k, j in enumerate(used):
                    head[j] = int(torch.argmax(out[k] @ self.lm_head.T))
                self.outputs.append(out[-1])
            self.heads[i] = head
        return drafts_step(mtp, self.heads, layers=MAX_DRAFTS)


def backfill_rings(mtps, histories, embedding, history, upload, stash=None, stash_hidden=None, verify=None):
    """The backfill trace (tt/mtp_backfill.py) on the host: its plan (reference.mtp_backfill.backfill_plan) on the
    lane's ``history`` (reference.mtp_backfill.History), the upload and (a relayed upload) the verify upload, then
    every MTP layer over the plan's rows, row r writing its K / V at position S0 + r of that layer's cache in the
    request's ring (``histories``, its PositionalHistory per layer; the outputs are not used). With every layer it also
    rewrites the stash page and the stash's hidden rows (``stash``, ``stash_hidden`` [T, hidden], changed in place).
    Returns the new stash page."""
    from .mtp_backfill import BACKFILL_ROWS, backfill_plan, backfill_start

    plan, stash = backfill_plan(upload, history.page, len(mtps), stash, history.rows, verify=verify)
    if plan["a"] is None:
        return stash
    a = plan["a"]
    positions = backfill_start(a, history.rows) + torch.arange(BACKFILL_ROWS)
    rows = history.hidden[torch.tensor(plan["hidden"])]
    for mtp, cache, tokens in zip(mtps, histories, plan["tokens"]):
        mtp(rows, embedding[torch.tensor(tokens)], positions, cache)
    for k, e in plan["stash_rows"]:
        stash_hidden[k] = history.hidden[e]
    return stash
