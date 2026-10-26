# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side MTP speculative decoding (reference/spec_decode.py, reference/mtp.py); no device, no ttnn.

The end-to-end tests drive the vLLM adapter's bookkeeping (verify passes, greedy acceptance, MTP draft positions)
against a small random fp32 MiMo: speculative greedy decoding must produce exactly the plain greedy tokens, with
any drafter. The multi-layer drafter (one MTP layer per draft) must also draft what a recomputation of every layer
over the whole committed sequence drafts, after steps of every accepted count: its incremental caches, re-prefill
and stash are exact."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.model import ReferenceDecoderLayer, rms_norm
from models.experimental.mimo_v2_6_flash.reference.mtp import ReferenceMTP, ReferenceMTPDrafter
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    MAX_DRAFTS,
    SPEC_SLOTS,
    DraftSource,
    MtpStash,
    accept_greedy,
    draft_source,
    mtp_cache_position,
    mtp_layer_tokens,
    mtp_stash,
    mtp_table_row,
    plan_mtp,
    plan_mtp_layers,
    plan_verify_passes,
    serial_positions,
)
from models.experimental.mimo_v2_6_flash.reference import spec_decode as spec_layout

# The MTP drafter's 4-slot verify (the draft groups: test_spec_groups_host.py, test_spec_groups_adapter_host.py).
pytestmark = pytest.mark.skipif(
    (spec_layout.SPEC_SLOTS, spec_layout.DRAFT_GROUPS) != (4, 1),
    reason="MTP's 4-slot layouts: run without MIMO_SPEC_SLOTS / MIMO_SPEC_GROUPS",
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


def test_plan_rejects_a_block_wider_than_a_pass(expect_error):
    with expect_error(ValueError, "4 drafts need 5 slots, a pass has 4"):
        plan_verify_passes([(0, [1] * 5, [1, 2, 3, 4, 5], 4)])
    with expect_error(ValueError, "column 1 of 2 has no position"):
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


def test_mtp_after_a_plain_step(expect_error):
    plan = plan_mtp(DraftSource(row=6, slots=[2]), [17, -1, -1, -1], 1)
    assert plan.fill == [-1, -1, 16, -1] and plan.anchor == 2
    assert plan.step2 == [-1, -1, 17, -1] and plan.step3 == [-1, -1, 18, -1]
    with expect_error(ValueError, "2 committed tokens, the step answered 1 columns"):
        plan_mtp(DraftSource(row=6, slots=[2]), [17, 18, -1, -1], 2)


def test_mtp_layers_after_a_rejection():
    # Last draft: rows 38 .. 40 in slots 1 .. 3. The verify at 41..44 rejected d_1 and committed one token, at 42: two
    # stash rows re-run before the new row, right-aligned.
    plan = plan_mtp_layers(DraftSource(row=3, slots=[0, 1, 2, 3]), [42, -1, -1, -1], 1, MtpStash([-1, 38, 39, 40]))
    assert plan.positions == [-1, 39, 40, 41] and plan.reprefill == 2 and plan.anchor == 3
    assert plan.sources == [None, ("stash", 2), ("stash", 3), ("step", 0)]
    assert plan.rows == [1, 2, 3]
    assert mtp_stash(plan) == MtpStash([-1, 39, 40, 41])


@pytest.mark.parametrize("m, reprefill", [(1, 2), (2, 1), (3, 0), (4, 0)])
def test_mtp_layers_reprefill_the_rows_later_layers_drafted(m, reprefill):
    committed = [42 + c for c in range(m)] + [-1] * (4 - m)
    plan = plan_mtp_layers(DraftSource(row=0, slots=[0, 1, 2, 3]), committed, m, MtpStash([37, 38, 39, 40]))
    assert plan.reprefill == reprefill
    rows = plan.rows
    assert len(rows) == max(3, m) and rows[-1] == plan.anchor == 3
    assert [plan.positions[s] for s in rows] == list(range(41 - reprefill, 41 + m))
    assert [plan.sources[s] for s in rows] == [("stash", 4 - reprefill + j) for j in range(reprefill)] + [
        ("step", c) for c in range(m)
    ]


def test_mtp_layers_reprefill_only_what_the_stash_holds():
    source = DraftSource(row=0, slots=[0, 1, 2, 3])
    # The last draft ran one row (its first draft): one stash row.
    plan = plan_mtp_layers(source, [42, -1, -1, -1], 1, MtpStash([-1, -1, -1, 40]))
    assert plan.positions == [-1, -1, 40, 41] and plan.reprefill == 1
    # The last step did not follow a draft of these positions (a step without drafting in between): nothing is stale.
    plan = plan_mtp_layers(source, [42, -1, -1, -1], 1, MtpStash([-1, 29, 30, 31]))
    assert plan.positions == [-1, -1, -1, 41] and plan.reprefill == 0
    plan = plan_mtp_layers(source, [42, -1, -1, -1], 1, MtpStash.empty())
    assert plan.reprefill == 0
    # Consecutive from A - 1 down only: 39 without 40 re-runs nothing.
    plan = plan_mtp_layers(source, [42, -1, -1, -1], 1, MtpStash([-1, 38, 39, -1]))
    assert plan.reprefill == 0


def test_mtp_layers_replanning_the_same_commit_finds_the_same_rows():
    """A draft planned again from the commit it was planned from (after it ran: the stash then holds its own rows)
    re-runs the same rows from the same sources: the adapter's chain check drafts each chained draft again."""
    source = DraftSource(row=0, slots=[0, 1, 2, 3])
    for stash in (MtpStash([37, 38, 39, 40]), MtpStash([-1, -1, -1, 40]), MtpStash.empty()):
        for m in range(1, SPEC_SLOTS + 1):
            committed = [42 + c for c in range(m)] + [-1] * (4 - m)
            plan = plan_mtp_layers(source, committed, m, stash)
            again = plan_mtp_layers(source, committed, m, mtp_stash(plan))
            assert again.positions == plan.positions and again.reprefill == plan.reprefill
            # The stash rows are found where the first draft put them (its own slots, which held them).
            for s in plan.rows[: plan.reprefill]:
                assert again.sources[s] == ("stash", s)


def test_mtp_layers_after_a_plain_step(expect_error):
    # A step without drafts after a draft: its one slot, and both stash rows (the draft's d_1, d_2 are unverified).
    plan = plan_mtp_layers(DraftSource(row=6, slots=[2]), [17, -1, -1, -1], 1, MtpStash([-1, 13, 14, 15]))
    assert plan.positions == [-1, 14, 15, 16] and plan.sources[1:] == [("stash", 2), ("stash", 3), ("step", 2)]
    with expect_error(ValueError, "2 committed tokens, the step answered 1 columns"):
        plan_mtp_layers(DraftSource(row=6, slots=[2]), [17, 18, -1, -1], 2, MtpStash([-1, 13, 14, 15]))
    with expect_error(ValueError, "not consecutive"):
        plan_mtp_layers(DraftSource(row=0, slots=[0, 1, 2, 3]), [17, 19, -1, -1], 2, MtpStash([-1, 13, 14, 15]))
    with expect_error(ValueError, "a stash of 2 rows for passes of 4 slots"):
        plan_mtp_layers(DraftSource(row=0, slots=[0, 1, 2, 3]), [17, -1, -1, -1], 1, MtpStash([14, 15]))


def test_mtp_layer_tokens_shift_one_per_layer():
    tokens0, drafts = [10, 11, 12], [20, 21, 22]
    assert mtp_layer_tokens(tokens0, drafts, 0) == [10, 11, 12]
    assert mtp_layer_tokens(tokens0, drafts, 1) == [11, 12, 20]
    assert mtp_layer_tokens(tokens0, drafts, 2) == [12, 20, 21]


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
            "num_nextn_predict_layers": MAX_DRAFTS,
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


def mtp_drafter(cfg, target, g, layers):
    """A ReferenceMTPDrafter over ``layers`` random MTP layers (1: one-layer drafting)."""
    h, mtps = cfg.hidden_size, []
    for i in range(layers):
        w = layer_weights(cfg, cfg.mtp_layer_index(i), g)
        w["head"] = {
            "eh_proj": torch.randn(h, 2 * h, generator=g) * (2 * h) ** -0.5,
            "enorm": 1 + 0.1 * torch.randn(h, generator=g),
            "hnorm": 1 + 0.1 * torch.randn(h, generator=g),
            "final_layernorm": 1 + 0.1 * torch.randn(h, generator=g),
        }
        mtps.append(ReferenceMTP(cfg, i, weights=w))
    return ReferenceMTPDrafter(mtps, target.embedding, target.lm_head)


class OracleDrafter:
    """Drafts the target's own greedy continuation: every draft is accepted (the full-width verify path).
    ``corrupt``: spoil draft k of the k-th proposal (mod MAX_DRAFTS + 1), so the accepted counts cycle 1 .. 4."""

    def __init__(self, target, corrupt=False):
        self.target = target
        self.corrupt = corrupt
        self.seq = None
        self.calls = 0

    def plan(self, source, committed_positions, accepted):
        return plan_mtp(source, committed_positions, accepted)

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
        plan = drafter.plan(source, committed_positions + [-1] * MAX_DRAFTS, accepted)
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


@pytest.mark.parametrize("drafter_kind", ["mtp", "mtp_layers", "oracle", "partial"])
def test_speculative_greedy_equals_plain_greedy(drafter_kind):
    g = torch.Generator().manual_seed(0)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    if drafter_kind.startswith("mtp"):
        drafter = mtp_drafter(cfg, target, g, MAX_DRAFTS if drafter_kind == "mtp_layers" else 1)
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


# ---------------------------------------------------------------------- multi-layer drafts against a recomputation
def recompute_drafts(target, mtps, seq, first):
    """The multi-layer equations without caches: layer i over the rows q = first .. B - 1 of the committed ``seq``
    (the last at B), row q taking the target hidden h_q and the token x~[q + i + 1] (``seq`` then the drafts so far),
    at position q; its output at B - 1 gives draft i + 1. Returns the drafts and those outputs."""
    b = len(seq) - 1
    hidden = target.hidden(seq)[first:b]
    positions = torch.arange(first, b)
    drafts, outputs = [], []
    for i, mtp in enumerate(mtps):
        window = list(seq) + drafts
        tokens = [window[q + i + 1] for q in range(first, b)]
        h = mtp(hidden, target.embedding[torch.tensor(tokens)], positions)[-1]
        drafts.append(int(torch.argmax(h @ target.lm_head.T)))
        outputs.append(h)
    return drafts, outputs


# Accepted counts m per step: every ordered pair of 1 .. 4 (a de Bruijn sequence), and steps without drafts ("plain":
# one token, from another slot) after a draft.
SCHEDULE = [1, 1, 2, 1, 3, 1, 4, 2, 2, 3, 2, 4, 3, 3, 4, 4, 1, "plain", 4, "plain", "plain", 3, 1]


def drive(target, drafter, prompt, schedule, keep_stash=True):
    """Serves one request with ``drafter`` after a plain first step; step k commits per ``schedule[k]``: m - 1 of
    the drafts and then a token that is not the next draft. Yields each draft's plan, the drafter's drafts and
    outputs, and the recomputation's."""
    seq = list(prompt)
    tok, h = target.greedy(seq)
    seq.append(tok)
    source, slot_tokens, slot_hidden = DraftSource(row=0, slots=[2]), {2: tok}, {2: h}
    committed_positions, accepted, first = [len(seq) - 1], 1, None
    vocab = target.cfg.vocab_size
    for k, step in enumerate(schedule):
        plan = drafter.plan(source, committed_positions + [-1] * MAX_DRAFTS, accepted)
        if first is None:
            first = min(p for p in plan.positions if p >= 0)
        drafts = drafter.draft(plan, slot_tokens, slot_hidden)
        if not keep_stash:
            drafter.stash = MtpStash.empty()
        yield plan, drafts, drafter.outputs, recompute_drafts(target, drafter.mtps, seq, first)
        # The verify of the last committed token (at p) and the drafts: commit m tokens at p + 1 .. p + m.
        p = len(seq) - 1
        if step == "plain":
            new, slots = [drafts[0] if k % 2 else (drafts[0] + 1) % vocab], [2]
        else:
            new, slots = drafts[: step - 1], list(range(SPEC_SLOTS))
            new.append(target.greedy(seq + new)[0] if step > MAX_DRAFTS else (drafts[step - 1] + 1 + k) % vocab)
        hidden = target.hidden(seq + new)
        source = DraftSource(row=0, slots=slots)
        slot_tokens = {slots[c]: new[c] for c in range(len(new))}
        slot_hidden = {slots[c]: hidden[p + c] for c in range(len(new))}
        committed_positions, accepted = list(range(p + 1, p + 1 + len(new))), len(new)
        seq += new


def test_mtp_layers_draft_what_a_recomputation_drafts():
    g = torch.Generator().manual_seed(1)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    drafter = mtp_drafter(cfg, target, g, MAX_DRAFTS)
    prompt = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38]
    rows = set()
    for plan, drafts, outputs, (want_drafts, want_outputs) in drive(target, drafter, prompt, SCHEDULE):
        rows.add((plan.reprefill, len(plan.rows)))
        assert drafts == want_drafts
        for got, want in zip(outputs, want_outputs):
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    # Re-prefills of 2, 1 and 0 rows, and every row count a pass takes: 1 for the first draft (after the plain step),
    # 2 for the next (its stash holds one row), then 3, or 4 after all 3 drafts were accepted.
    assert rows == {(0, 1), (1, 2), (2, 3), (1, 3), (0, 3), (0, 4)}


def test_without_reprefill_later_layers_draft_from_stale_entries():
    """The recomputation test's power: dropping the stash (no re-prefill) leaves rejected drafts in the later layers'
    caches, and their outputs drift from the recomputation."""
    g = torch.Generator().manual_seed(1)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    drafter = mtp_drafter(cfg, target, g, MAX_DRAFTS)
    prompt = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38]
    worst = [0.0] * MAX_DRAFTS
    for _, _, outputs, (_, want_outputs) in drive(target, drafter, prompt, SCHEDULE, keep_stash=False):
        for i, (got, want) in enumerate(zip(outputs, want_outputs)):
            worst[i] = max(worst[i], float((got - want).abs().max()))
    assert worst[0] < 1e-5  # layer 0 never drafts into its cache
    assert worst[1] > 0.1 and worst[2] > 0.1
