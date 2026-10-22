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
"""

import torch

from .model import ReferenceDecoderLayer, load_layer_weights, rms_norm
from .spec_decode import (
    MAX_DRAFTS,
    SPEC_SLOTS,
    MtpLayersInputs,
    MtpStash,
    mtp_layer_tokens,
    mtp_stash,
    plan_mtp,
    plan_mtp_layers,
)


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
    """The adapter's propose step for one lane's request, on the host. ``mtps``: the ReferenceMTP layers, one
    (one-layer drafting: layer 0 drafts every token) or MAX_DRAFTS (draft i + 1 from layer i); ``embedding`` [vocab,
    hidden] and ``lm_head`` [vocab, hidden] the target's. Each layer keeps its K / V in its own PositionalHistory; with
    every layer the drafter also keeps the MtpStash rows (target hidden, layer-0 token) for the next re-prefill.

    plan() plans a draft from where the last step left the request (reference.spec_decode.plan_mtp or
    plan_mtp_layers), draft() runs it from the slots' tokens and target hidden rows, as the device program would."""

    def __init__(self, mtps, embedding, lm_head):
        assert len(mtps) in (1, MAX_DRAFTS), f"{len(mtps)} MTP layers: one, or one per draft"
        self.mtps, self.embedding, self.lm_head = list(mtps), embedding, lm_head
        self.reset()

    @property
    def multi_layer(self):
        return len(self.mtps) > 1

    def reset(self):
        """A new request's prompt: no K / V, nothing stashed."""
        self.histories = [PositionalHistory() for _ in self.mtps]
        self.stash = MtpStash.empty(len(self.mtps)) if self.multi_layer else None
        self.stash_rows = [None] * (len(self.mtps) - 1)  # (hidden, layer-0 token) of each stash row
        self.outputs = None  # each draft's final normed output at its row (the anchor), of the last draft

    def plan(self, source, committed_positions, accepted):
        if self.multi_layer:
            return plan_mtp_layers(source, committed_positions, accepted, self.stash, len(self.mtps))
        return plan_mtp(source, committed_positions, accepted)

    def _greedy(self, h):
        return int(torch.argmax(h @ self.lm_head.T))

    def draft(self, plan, slot_tokens, slot_hidden):
        """``slot_tokens`` / ``slot_hidden``: per slot of the last step, the token committed there and the target's
        final normed hidden that chose it. Returns the MAX_DRAFTS drafts."""
        if isinstance(plan, MtpLayersInputs):
            return self._draft_layers(plan, slot_tokens, slot_hidden)
        mtp, history = self.mtps[0], self.histories[0]
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
        last rows become the stash."""
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
        for i, (mtp, history) in enumerate(zip(self.mtps, self.histories)):
            tokens = mtp_layer_tokens(tokens0, drafts, i)
            h = mtp(hidden, self.embedding[torch.tensor(tokens)], positions, history)[-1]
            drafts.append(self._greedy(h))
            self.outputs.append(h)
        self.stash = mtp_stash(plan, len(self.mtps))
        last = len(plan.positions) - len(self.stash_rows)
        self.stash_rows = [
            (hidden[rows.index(s)], tokens0[rows.index(s)]) if s in rows else None
            for s in range(last, len(plan.positions))
        ]
        return drafts
