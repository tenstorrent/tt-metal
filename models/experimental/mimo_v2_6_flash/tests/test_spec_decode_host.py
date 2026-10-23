# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side MTP speculative decoding (reference/spec_decode.py, reference/mtp.py); no device, no ttnn.

The end-to-end tests drive the vLLM adapter's bookkeeping (verify passes, greedy acceptance, MTP draft positions)
against a small random fp32 MiMo: speculative greedy decoding must produce exactly the plain greedy tokens, with
any drafter."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.model import ReferenceDecoderLayer, rms_norm
from models.experimental.mimo_v2_6_flash.reference.mtp import ReferenceMTP
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    MAX_DRAFTS,
    SPEC_SLOTS,
    DraftSource,
    accept_greedy,
    draft_source,
    mtp_cache_position,
    mtp_table_row,
    plan_mtp,
    plan_verify_passes,
    serial_positions,
)

BLOCK = 64


# ---------------------------------------------------------------------- planning
def test_one_row_fills_one_pass():
    (p,) = plan_verify_passes([(5, [10, 11, 12, 13], [40, 41, 42, 43], 3)])
    assert p.tokens == [10, 11, 12, 13] and p.positions == [40, 41, 42, 43] and p.table_rows == [5] * 4
    assert p.answers == [(0, 5, 0), (1, 5, 1), (2, 5, 2), (3, 5, 3)]
    assert draft_source([p]) == DraftSource(row=5, slots=[0, 1, 2, 3])


def test_short_drafts_leave_slots_idle():
    (p,) = plan_verify_passes([(0, [7, 8, -1, -1], [9, 10, -1, -1], 1)])
    assert p.positions == [9, 10, -1, -1] and p.table_rows == [0, 0, None, None]
    assert draft_source([p]).slots == [0, 1]


def test_rows_share_a_pass_while_they_fit():
    rows = [(0, [1, 2, -1, -1], [5, 6, -1, -1], 1), (1, [3, -1, -1, -1], [9, -1, -1, -1], 0)]
    (p,) = plan_verify_passes(rows)
    assert p.positions == [5, 6, 9, -1] and p.table_rows == [0, 0, 1, None]
    assert draft_source([p]) is None  # two requests: no drafting


def test_a_joining_request_gets_its_own_pass():
    rows = [(0, [1, 2, 3, 4], [5, 6, 7, 8], 3), (1, [9, -1, -1, -1], [20, -1, -1, -1], 0)]
    passes = plan_verify_passes(rows)
    assert [p.positions for p in passes] == [[5, 6, 7, 8], [20, -1, -1, -1]]
    assert passes[1].answers == [(0, 1, 0)]
    assert draft_source(passes) is None


def test_plan_rejects_a_block_wider_than_a_pass():
    with pytest.raises(ValueError):
        plan_verify_passes([(0, [1] * 5, [1, 2, 3, 4, 5], 4)])
    with pytest.raises(ValueError):
        plan_verify_passes([(0, [1, 2], [1, -1], 1)])


def test_serial_positions_keep_one_slot_each():
    assert serial_positions([4, 5, -1, 7, -1, -1]) == [
        [4, -1, -1, -1, -1, -1],
        [-1, 5, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 7, -1, -1],
    ]


def test_mtp_after_a_verify():
    # The verify ran columns at positions 40..43 in slots 0..3 and committed two tokens, at 41 and 42.
    plan = plan_mtp(DraftSource(row=3, slots=[0, 1, 2, 3]), [41, 42, -1, -1], 2)
    assert plan.fill == [40, 41, -1, -1] and plan.anchor == 1
    assert plan.step2 == [-1, 42, -1, -1] and plan.step3 == [-1, 43, -1, -1]


def test_mtp_after_a_plain_step():
    plan = plan_mtp(DraftSource(row=6, slots=[2]), [17, -1, -1, -1], 1)
    assert plan.fill == [-1, -1, 16, -1] and plan.anchor == 2
    assert plan.step2 == [-1, -1, 17, -1] and plan.step3 == [-1, -1, 18, -1]
    with pytest.raises(ValueError):
        plan_mtp(DraftSource(row=6, slots=[2]), [17, 18, -1, -1], 2)


def test_accept_greedy():
    assert accept_greedy([5, 6, 7, 8], [5, 6, 9], 3) == [5, 6, 7]
    assert accept_greedy([5, 6, 7, 8], [5, 6, 7], 3) == [5, 6, 7, 8]
    assert accept_greedy([5, 6, 7, 8], [4, 6, 7], 3) == [5]
    assert accept_greedy([5, 0, 0, 0], [0, 0, 0], 0) == [5]


@pytest.mark.parametrize("t", [0, 1, 63, 64, 127, 128, 255, 256, 1000, 1 << 20])
def test_mtp_ring_addresses_every_window_position(t):
    """A write at true position t and every later read of t (from a query up to 127 positions later) land on the
    same physical (block, row), and no other position in such a window shares it."""
    row, blocks = mtp_table_row(BLOCK)
    assert len(row) <= blocks  # paged_update_cache: the table is no wider than the cache

    def physical(cache_pos):
        assert 0 <= cache_pos < len(row) * BLOCK
        return row[cache_pos // BLOCK], cache_pos % BLOCK

    written = physical(mtp_cache_position(t))
    for u in range(t, t + 128):
        seen = {}
        query = mtp_cache_position(u)
        for d in range(128):  # the SWA window: the query and the 127 positions before it
            if u - d < 0:
                break
            seen[u - d] = physical(query - d)
        assert seen[t] == written
        assert len(set(seen.values())) == len(seen)


# ---------------------------------------------------------------------- end to end against a small fp32 MiMo
def tiny_config():
    return MiMoConfig(
        {
            "hidden_size": 64,
            "num_hidden_layers": 3,
            "vocab_size": 101,
            "layernorm_epsilon": 1e-6,
            "hybrid_layer_pattern": [0, 1, 1],
            "moe_layer_freq": [0, 0, 0],
            "intermediate_size": 96,
            "moe_intermediate_size": 32,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "n_group": 1,
            "topk_group": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 24,
            "v_head_dim": 16,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 24,
            "swa_v_head_dim": 24,
            "partial_rotary_factor": 1 / 3,
            "rope_theta": 10000.0,
            "swa_rope_theta": 10000.0,
            "sliding_window": 8,
            "attention_value_scale": 0.7,
            "add_swa_attention_sink_bias": True,
            "add_full_attention_sink_bias": False,
            "max_position_embeddings": 512,
            "num_nextn_predict_layers": 1,
        }
    )


def layer_weights(cfg, idx, g):
    dims, h = cfg.attention_dims(idx), cfg.hidden_size

    def w(*shape):
        return torch.randn(*shape, generator=g) * shape[-1] ** -0.5

    attn = {
        "q_proj": w(dims.num_heads * dims.head_dim, h),
        "k_proj": w(dims.num_kv_heads * dims.head_dim, h),
        "v_proj": w(dims.num_kv_heads * dims.v_head_dim, h),
        "o_proj": w(h, dims.num_heads * dims.v_head_dim),
        "attention_sink_bias": torch.randn(dims.num_heads, generator=g) if dims.has_sink else None,
    }
    mlp = {"gate_proj": w(cfg.intermediate_size, h), "up_proj": w(cfg.intermediate_size, h)}
    mlp["down_proj"] = w(h, cfg.intermediate_size)
    norm = lambda: 1 + 0.1 * torch.randn(h, generator=g)  # noqa: E731
    return {"input_layernorm": norm(), "post_attention_layernorm": norm(), "attn": attn, "mlp": mlp}


class TinyMiMo:
    """Greedy target: full recompute of a prefix (no cache), so a verify column sees exactly its own prefix."""

    def __init__(self, cfg, g):
        h = cfg.hidden_size
        self.cfg = cfg
        self.embedding = torch.randn(cfg.vocab_size, h, generator=g)
        self.layers = [ReferenceDecoderLayer(cfg, i, weights=layer_weights(cfg, i, g)) for i in range(3)]
        self.final_norm = 1 + 0.1 * torch.randn(h, generator=g)
        self.lm_head = torch.randn(cfg.vocab_size, h, generator=g) * h**-0.5

    def hidden(self, tokens):
        """Final normed hidden [S, hidden] of every position of ``tokens``."""
        x = self.embedding[torch.tensor(tokens)]
        positions = torch.arange(len(tokens))
        for layer in self.layers:
            x = layer(x, positions)
        return rms_norm(x, self.final_norm, self.cfg.layernorm_epsilon)

    def greedy(self, tokens):
        h = self.hidden(tokens)[-1]
        return int(torch.argmax(h @ self.lm_head.T)), h


class PositionalHistory:
    """The MTP ring's semantics on the host: K / V kept by position, a later write overwrites (reference.model's
    KVHistory appends)."""

    def __init__(self):
        self.entries = {}

    def append(self, k, v, pos):
        for i, p in enumerate(pos.tolist()):
            self.entries[p] = (k[:, i], v[:, i])
        keys = sorted(self.entries)
        k_all = torch.stack([self.entries[p][0] for p in keys], dim=1)
        v_all = torch.stack([self.entries[p][1] for p in keys], dim=1)
        return k_all, v_all, torch.tensor(keys)


class MtpDrafter:
    """The adapter's propose step on the host: the MTP fill over the slots plan_mtp fills, then two steps on the
    anchor slot, from the slots' committed tokens and the target hidden the last step left in them."""

    def __init__(self, cfg, target, g):
        w = layer_weights(cfg, cfg.mtp_layer_index(0), g)
        h = cfg.hidden_size
        w["head"] = {
            "eh_proj": torch.randn(h, 2 * h, generator=g) * (2 * h) ** -0.5,
            "enorm": 1 + 0.1 * torch.randn(h, generator=g),
            "hnorm": 1 + 0.1 * torch.randn(h, generator=g),
            "final_layernorm": 1 + 0.1 * torch.randn(h, generator=g),
        }
        self.mtp = ReferenceMTP(cfg, weights=w)
        self.target = target
        self.history = PositionalHistory()

    def draft(self, plan, slot_tokens, slot_hidden):
        t = self.target
        fill = [s for s in range(SPEC_SLOTS) if plan.fill[s] >= 0]
        out = self.mtp(
            torch.stack([slot_hidden[s] for s in fill]),
            t.embedding[torch.tensor([slot_tokens[s] for s in fill])],
            torch.tensor([plan.fill[s] for s in fill]),
            self.history,
        )
        h = out[fill.index(plan.anchor)]
        drafts = [int(torch.argmax(h @ t.lm_head.T))]
        for positions in (plan.step2, plan.step3):
            position = torch.tensor([positions[plan.anchor]])
            h = self.mtp(h[None], t.embedding[torch.tensor([drafts[-1]])], position, self.history)[0]
            drafts.append(int(torch.argmax(h @ t.lm_head.T)))
        return drafts


class OracleDrafter:
    """Drafts the target's own greedy continuation: every draft is accepted (the full-width verify path).
    ``corrupt``: spoil draft k of the k-th proposal (mod MAX_DRAFTS + 1), so the accepted counts cycle 1 .. 4."""

    def __init__(self, target, corrupt=False):
        self.target = target
        self.corrupt = corrupt
        self.seq = None
        self.calls = 0

    def draft(self, plan, slot_tokens, slot_hidden):
        seq, out = list(self.seq), []
        for _ in range(MAX_DRAFTS):
            tok, _ = self.target.greedy(seq)
            seq.append(tok)
            out.append(tok)
        bad = self.calls % (MAX_DRAFTS + 1)
        self.calls += 1
        if self.corrupt and bad < MAX_DRAFTS:
            out[bad] = (out[bad] + 1) % self.target.cfg.vocab_size
        return out


def spec_generate(target, drafter, prompt, steps):
    """Greedy generation the way the adapter serves one request: a plain step first, then propose / verify."""
    seq = list(prompt)
    tok, h = target.greedy(seq)
    seq.append(tok)
    # The plain decode step left its committed token and the hidden that chose it in the request's slot.
    source, slot_tokens, slot_hidden = DraftSource(row=0, slots=[2]), {2: tok}, {2: h}
    committed_positions, accepted, counts = [len(seq) - 1], 1, []
    while len(seq) < len(prompt) + steps:
        plan = plan_mtp(source, committed_positions + [-1] * MAX_DRAFTS, accepted)
        if isinstance(drafter, OracleDrafter):
            drafter.seq = seq
        drafts = drafter.draft(plan, slot_tokens, slot_hidden)
        # The verify block: the last committed token and the drafts at the next positions.
        p = len(seq) - 1
        block = [seq[-1]] + drafts
        (vp,) = plan_verify_passes([(0, block, list(range(p, p + 1 + MAX_DRAFTS)), MAX_DRAFTS)])
        argmax, slot_tokens, slot_hidden = [0] * (1 + MAX_DRAFTS), {}, {}
        for s, _, c in vp.answers:
            argmax[c], slot_hidden[s] = target.greedy(seq[:p] + block[: c + 1])
            slot_tokens[s] = argmax[c]
        committed = accept_greedy(argmax, drafts, MAX_DRAFTS)
        committed_positions = list(range(len(seq), len(seq) + len(committed)))
        seq += committed
        source, accepted = draft_source([vp]), len(committed)
        counts.append(accepted)
    return seq[: len(prompt) + steps], source, counts


@pytest.mark.parametrize("drafter_kind", ["mtp", "oracle", "partial"])
def test_speculative_greedy_equals_plain_greedy(drafter_kind):
    g = torch.Generator().manual_seed(0)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    if drafter_kind == "mtp":
        drafter = MtpDrafter(cfg, target, g)
    else:
        drafter = OracleDrafter(target, corrupt=drafter_kind == "partial")
    prompt, steps = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38], 24
    plain = list(prompt)
    for _ in range(steps):
        plain.append(target.greedy(plain)[0])
    spec, source, counts = spec_generate(target, drafter, prompt, steps)
    assert spec == plain
    assert source.slots == list(range(SPEC_SLOTS))
    if drafter_kind == "oracle":
        assert set(counts) == {MAX_DRAFTS + 1}
    if drafter_kind == "partial":
        assert set(counts) == set(range(1, MAX_DRAFTS + 2))
