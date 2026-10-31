# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Draft groups on one Galaxy row (TP=8; no weights): the chain kernels (tt/spec_chain.py, kernels/spec_chain.cpp)
with two requests per verify pass against reference/spec_chain.py bit for bit, eager and replayed from one capture.
Skipped unless the process runs with MIMO_SPEC_SLOTS=8 and MIMO_SPEC_GROUPS=2 (3 drafts per request, every MTP
layer) or 4 (1 draft per request: one MTP layer through the planned program).

test_spec_chain_groups_tp8 (the planned MTP program): the chain and drafts steps on passes of a row per group
(accepted counts, each group's limit on its own, one row alone in any group), the plan and tokens steps on a
description per group and random stash pages, and the relay of the pages into a chained verify.

test_spec_chain_groups_dflash_tp8: the chain and drafts steps with DFlash drafting on the same passes (3 drafts)."""

import pytest
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.reference.sample_verify import NO_TOKEN, OUT_WORDS, control_words, pass_inputs
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    ACCEPT_WORDS,
    RESULT_WORDS,
    DFlashLayout,
    chain_row,
    chain_step,
    dflash_chain_step,
    dflash_drafts_step,
    drafts_step,
    plan_step,
    relay_step,
    stash_page,
    tokens_step,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    DRAFT_GROUPS,
    GROUP_SLOTS,
    MAX_DRAFTS,
    SPEC_SLOTS,
    WIDTHS,
    DraftSource,
    draft_sources,
    plan_verify_passes,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import pack_mtp_commits, pack_verify
from models.experimental.mimo_v2_6_flash.tests.common import dp_submeshes
from models.experimental.mimo_v2_6_flash.tests.seeded_rng import SeededRng
from models.experimental.mimo_v2_6_flash.tests.test_spec_inputs_tp8 import (
    DEVICE_PARAMS,
    REP,
    B,
    int_words,
    replayed,
    upload_ints,
    upload_words,
    words_of,
)
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.mtp import make_layer_inputs
from models.experimental.mimo_v2_6_flash.tt.spec_chain import CTRL_WORDS, SpecChain
from models.experimental.mimo_v2_6_flash.tt.spec_inputs import SpecInputs

pytestmark = pytest.mark.skipif(
    DRAFT_GROUPS < 2 or SPEC_SLOTS != 8, reason="draft groups: run with MIMO_SPEC_SLOTS=8 MIMO_SPEC_GROUPS=2 or 4"
)

MAX_SEQ_LEN = 4096
VOCAB = 152576


def u32(plan, n):
    return ttnn.from_torch(
        torch.zeros(1, n, dtype=torch.int32),
        device=plan.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.replicate(),
    )


def accept_words(groups):
    """The accept page with draft group g's request committing groups[g] = (count, tokens) (None: unused)."""
    words = []
    for g in range(ACCEPT_WORDS // OUT_WORDS):
        n, toks = groups[g] if g < len(groups) and groups[g] is not None else (0, [])
        out = [n] + list(toks) + [NO_TOKEN] * (1 + MAX_DRAFTS - len(toks))
        words += out + [0] * (OUT_WORDS - len(out))
    return words


def pass_cases(rng, width=GROUP_SLOTS):
    """(verify pass, chain row sources, accept page) of passes in draft groups of ``width`` slots with a row per group
    (rows 5, 6, .. in a shuffled group each) and of one row in any group."""
    cases = []
    bases = [1, 300, 255, 257, 1023, 2000, MAX_SEQ_LEN - 1 - MAX_DRAFTS, 40, 500, MAX_SEQ_LEN - 2]
    count = SPEC_SLOTS // width
    for k in range(5 * width):
        rows_ids = [5 + g for g in range(count)]
        order = list(range(count))
        for i in range(count - 1, 0, -1):  # Fisher-Yates (SeededRng has no shuffle)
            j = rng.randrange(0, i + 1)
            order[i], order[j] = order[j], order[i]
        groups = dict(zip(rows_ids, order))
        rows, committed = [], {}
        for r in rows_ids:
            m = (k % width) + 1 if r == 5 else rng.randrange(1, width + 1)
            b = max(rng.choice(bases), m)
            tokens = [rng.randrange(VOCAB) for _ in range(width)]
            rows.append((r, tokens, list(range(b - m, b - m + width)), width - 1))
            committed[r] = (m, [rng.randrange(VOCAB) for _ in range(m)])
        (vp,) = plan_verify_passes(rows, groups=groups, group_slots=width)
        sources = draft_sources([vp], group_slots=width)
        cases.append((vp, sources, accept_words([committed[groups_row(groups, g)] for g in range(count)])))
    for g in range(count):
        tokens, positions = list(range(11, 11 + width)), list(range(700, 700 + width))
        (vp,) = plan_verify_passes([(7, tokens, positions, width - 1)], groups={7: g}, group_slots=width)
        acc = [None] * count
        acc[g] = (min(2, width), [21, 22][: min(2, width)])
        cases.append((vp, draft_sources([vp], group_slots=width), accept_words(acc)))
    return cases


def groups_row(groups, g):
    return next(r for r, k in groups.items() if k == g)


def verify_tensors(vp, sources, limit, relay=False):
    tok = torch.zeros(B, dtype=torch.int32)
    pos = torch.full((B,), -1, dtype=torch.int32)
    tok[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
    pos[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
    return tok, pos, torch.tensor(chain_row(B, sources, limit, relay=relay), dtype=torch.int32)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_chain_groups_tp8(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    spec = SpecInputs(plan, B, REP)
    accept, argmax, heads = u32(plan, ACCEPT_WORDS), u32(plan, B), [u32(plan, B) for _ in range(MAX_DRAFTS)]
    ctrl = u32(plan, CTRL_WORDS)
    li = make_layer_inputs(plan, B, 64)
    chain = SpecChain(plan, spec, accept, argmax, heads, li)
    chain.warm(ctrl)
    rng = SeededRng(3)

    def run_chain():
        chain.chain()
        chain.drafts()

    replay = replayed(mesh, run_chain)
    pages = []
    cases = [case for width in WIDTHS for case in pass_cases(rng, width)]
    for k, (vp, sources, acc) in enumerate(cases):
        for limit in (MAX_SEQ_LEN - MAX_DRAFTS, rng.choice([p for p in vp.positions if p >= 0]) + 1):
            tok, pos, chain_t = verify_tensors(vp, sources, limit)
            ids = [rng.randrange(VOCAB) for _ in range(B)]
            head_rows = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
            want_mtp, header = chain_step(pack_verify(tok, pos, chain_t).tolist(), acc, ids, layers=MAX_DRAFTS)
            want_result = header + drafts_step(want_mtp, head_rows, layers=MAX_DRAFTS)
            for runner in (run_chain, replay):
                ttnn.copy_host_to_device_tensor(spec.verify_host(tok, pos, chain_t), spec.verify_packed)
                upload_words(plan, acc, accept)
                upload_words(plan, ids, argmax)
                for h, words in zip(heads, head_rows):
                    upload_words(plan, words, h)
                runner()
                for c, got in enumerate(words_of(spec.mtp_packed)):
                    assert got == [w & 0xFFFFFFFF for w in want_mtp], f"case {k} limit {limit} chip {c}: upload"
                for c, got in enumerate(words_of(chain.result)):
                    assert got == want_result, f"case {k} limit {limit} chip {c}: result page"
            pages.append((vp, sources, want_result))

    # The relay of each page into the next verify of the same rows (provisional positions above the real ones).
    def run_relay():
        chain.relay(ctrl)

    replay = replayed(mesh, run_relay)
    for k, (vp, sources, page) in enumerate(pages[::3]):
        rows = sorted({s.row for s in sources})
        groups = {s.row: s.group for s in sources}
        width = vp.width
        nxt_rows = [
            (
                r,
                [rng.randrange(VOCAB) for _ in range(width)],
                list(range(3000 + 10 * r, 3000 + 10 * r + width)),
                width - 1,
            )
            for r in rows
        ]
        (nxt,) = plan_verify_passes(nxt_rows, groups=groups, group_slots=width)
        nxt_sources = draft_sources([nxt], group_slots=width)
        tok, pos, chain_t = verify_tensors(nxt, nxt_sources, MAX_SEQ_LEN - MAX_DRAFTS, relay=True)
        ctrl_words = control_words(pass_inputs(nxt, None)[1])
        want_verify, want_ctrl = relay_step(pack_verify(tok, pos, chain_t).tolist(), page, ctrl_words)
        for runner in (run_relay, replay):
            ttnn.copy_host_to_device_tensor(spec.verify_host(tok, pos, chain_t), spec.verify_packed)
            upload_words(plan, page, chain.result)
            upload_words(plan, ctrl_words, ctrl)
            runner()
            for c, got in enumerate(int_words(spec.verify_packed)):
                assert got == want_verify, f"relay {k} chip {c}: verify upload"
            for c, got in enumerate(words_of(ctrl)):
                assert got == want_ctrl, f"relay {k} chip {c}: control page"

    # The plan step, then the tokens steps, on a description per group and random stash pages.
    def run_plan():
        chain.plan_layers()
        for layer in range(1, MAX_DRAFTS):
            chain.tokens(layer)

    replay = replayed(mesh, run_plan)
    outs = {"pos": li.pos, "rope": li.rope, "hidden": li.hidden, "tokens0": li.tokens[0]}
    for k in range(40 * len(WIDTHS)):
        width = WIDTHS[k % len(WIDTHS)]
        count = SPEC_SLOTS // width
        commits = []
        for g in range(count):
            if rng.random() < 0.15:
                continue  # an idle group
            m = rng.randrange(1, width + 1)
            a = rng.choice([2, 255, 256, 3000, MAX_SEQ_LEN - 6])
            slots = list(range(g * width, g * width + m)) if rng.random() < 0.7 else [g * width + 1]
            m = min(m, len(slots))
            positions = [a + 1 + c for c in range(m)] + [-1] * (1 + MAX_DRAFTS - m)
            source = DraftSource(row=5 + g, slots=slots, group=g, width=width)
            commits.append((source, [rng.randrange(VOCAB) for _ in range(m)], positions, m))
        mtp = pack_mtp_commits(B, commits).tolist()
        stash_positions = []
        for g in range(count):
            a = next((c[2][0] - 1 for c in commits if c[0].group == g), 100)
            r = rng.randrange(0, width + 1)
            stash_positions += [-1] * (width - r) + [a - r + j for j in range(r)]
        stash_positions = [p if p >= 0 else -1 for p in stash_positions]
        page = stash_page(B, stash_positions, [rng.randrange(VOCAB) if p >= 0 else 0 for p in stash_positions])
        head_rows = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
        want, new = plan_step(mtp, page, B)
        want_tokens = [want["tokens0"]] + [
            tokens_step(want["tokens0"], head_rows, layer, mtp) for layer in range(1, MAX_DRAFTS)
        ]
        for runner in (run_plan, replay):
            upload_ints(plan, mtp, spec.mtp_packed)
            upload_ints(plan, page, li.stash)
            for h, words in zip(heads, head_rows):
                upload_words(plan, words, h)
            runner()
            for name, t in outs.items():
                for c, got in enumerate(int_words(t)):
                    assert got == want[name], f"plan {k}: {name} chip {c}"
            for j, t in enumerate(li.serial):
                for c, got in enumerate(int_words(t)):
                    assert got == want["serial"][j], f"plan {k}: serial {j} chip {c}"
            for layer, t in enumerate(li.tokens):
                for c, got in enumerate(int_words(t)):
                    assert got == want_tokens[layer], f"plan {k}: tokens {layer} chip {c}"
            for c, got in enumerate(int_words(li.stash)):
                assert got == new, f"plan {k}: stash page chip {c}"


@pytest.mark.skipif(MAX_DRAFTS < 3, reason="DFlash drafts a block of 3 or 7")
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_chain_groups_dflash_tp8(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    spec = SpecInputs(plan, B, REP)
    layout = DFlashLayout(ring=1024, block=8, heads=8, last=MAX_SEQ_LEN - 1, groups=DRAFT_GROUPS)
    accept, argmax, head, packed = u32(plan, ACCEPT_WORDS), u32(plan, B), u32(plan, B), u32(plan, layout.words)
    chain = SpecChain(plan, spec, accept, argmax, [head], dflash=(packed, layout))

    def run():
        chain.chain()
        chain.drafts()

    replay = replayed(mesh, run)
    rng = SeededRng(4)
    for k, (vp, sources, acc) in enumerate(pass_cases(rng)):
        for limit in (MAX_SEQ_LEN - MAX_DRAFTS, rng.choice([p for p in vp.positions if p >= 0]) + 1):
            tok, pos, chain_t = verify_tensors(vp, sources, limit)
            ids = [rng.randrange(VOCAB) for _ in range(B)]
            head_row = [rng.randrange(VOCAB) for _ in range(B)]
            want_packed, header = dflash_chain_step(pack_verify(tok, pos, chain_t).tolist(), acc, ids, layout)
            want_result = header + dflash_drafts_step(want_packed, head_row, layout)
            assert len(want_result) == RESULT_WORDS
            for runner in (run, replay):
                ttnn.copy_host_to_device_tensor(spec.verify_host(tok, pos, chain_t), spec.verify_packed)
                upload_words(plan, acc, accept)
                upload_words(plan, ids, argmax)
                upload_words(plan, head_row, head)
                runner()
                for c, got in enumerate(words_of(packed)):
                    assert got == want_packed, f"case {k} limit {limit} chip {c}: DFlash upload"
                for c, got in enumerate(words_of(chain.result)):
                    assert got == want_result, f"case {k} limit {limit} chip {c}: result page"
