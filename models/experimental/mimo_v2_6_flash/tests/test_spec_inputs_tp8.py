# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-only (one Galaxy row, TP=8; no weights loaded): the packed speculative inputs' derive programs
(tt/spec_inputs.py) write a verify's and an MTP draft's trace inputs exactly as the host built them input by input.

For each step the host copies the packed upload; the derive program runs eagerly, then replayed from a trace of it.
Every derived buffer is read back from every chip of the row and compared bit for bit with the host values
(test_spec_inputs_host.py's expectations: TtMiMoModel.prepare_decode_inputs_host, kv.owned_positions / owner_mask,
serial_positions, the MTP's plan_mtp / mtp_cache_position inputs). With the decode RoPE tables (whole 192-wide rows,
or compact: the 64 rotated dims plus the 1 / 0 tail, as past 128K context), the verify's and the draft's table rows
at the derived indices (TtMiMoModel.rope_table_rows) must equal the host cos / sin they used to upload.

test_rope_table_rows_1m_tp8: the same rows of 1M-position tables (the MTP serving entry's context) near 0, 128K and
1M - 1, past it into the table margin, and at random positions with idle slots, eager and traced.

test_spec_chain_tp8: the chain and drafts steps (tt/spec_chain.py, kernels/spec_chain.cpp) against their host mirror
(reference/spec_chain.py, itself checked against the host plan in test_spec_chain_host.py) bit for bit, eager and
replayed from one capture: the draft's packed upload they write and the result page, for passes of one request with
every accepted count, around the ring's wrap and at the context limit, an idle chain row, and a host-written upload
(the result page now carries the pass's positions).

test_spec_chain_dflash_tp8: the same chain and drafts steps with DFlash drafting (tt/dflash.py): the DFlash draft's
packed upload the chain step writes (reference.spec_chain.dflash_chain_step) and the drafts step's section from the
drafter's GreedyHead row, bit for bit, eager and replayed, and a host-written upload (dflash_inputs).

test_greedy_verify_tp8: GreedyVerify (tt/verify_sampler.py, kernels/greedy_verify.cpp), the accept step of a verify the
device sampler does not serve, against reference.sample_verify.verify_pass on all-greedy control pages bit for bit,
eager and replayed: every pass shape plan_verify_passes makes of 1-4 requests, every accepted count.

test_spec_relay_tp8: the relay step (a chained verify's tokens and positions into ``verify_packed`` and its drafts into
the accept step's control page, from the lane's last result page) against its mirror (reference.spec_chain.relay_step)
bit for bit, eager and traced: every count of the step chained onto (its page written by the one-layer or the
every-layer chain and drafts steps) and every number of drafts relayed, around the ring's wrap and at the context
limit, a page that drafted nothing, host content beside the relayed slots, and an upload without the relay word
(untouched).

test_spec_chain_layers_tp8: the same with every MTP layer (MIMO_MTP_LAYERS=3): the chain step's description and the
drafts step, then the plan step (plan_mtp_layers on the device stash: every commit count, a plain step's slot, stash
hits, misses and holes, the ring's wrap, the context end, an idle description) and the tokens steps, every output
buffer and the stash page bit for bit against reference/spec_chain.py, eager and replayed from one capture.

test_mtp_history_tp8: the MTP history's kernel steps (tt/mtp_backfill.py, kernels/mtp_history.cpp) against
reference/mtp_backfill.py bit for bit: a decode step's, a verify pass's and a prefill list's writes (page and hidden
rows), and the backfill's plan (rows, RoPE indices, tokens, the fill table, the stash page and hidden rows), eager and
traced."""

import itertools
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.sample_verify import (
    GROUP_WORDS,
    GROUPS,
    NO_TOKEN,
    OUT_WORDS,
    control_words,
    pass_inputs,
    verify_pass,
)
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    ACCEPT_WORDS,
    DRAFTS_AT,
    RESULT_WORDS,
    STATUS_AT,
    DFlashLayout,
    chain_row,
    chain_step,
    dflash_chain_step,
    dflash_drafts_step,
    dflash_inputs,
    drafts_step,
    relay_step,
    plan_step,
    stash_page,
    tokens_step,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    MAX_DRAFTS,
    SPEC_SLOTS,
    DraftSource,
    draft_source,
    plan_mtp,
    plan_verify_passes,
)
from models.experimental.mimo_v2_6_flash.reference.mtp_backfill import (
    MTP_WINDOW,
    History,
    backfill_plan,
    backfill_row,
    decode_writes,
    entry,
    prefill_writes,
    verify_writes,
)
from models.experimental.mimo_v2_6_flash.reference.spec_chain import CHAIN_HSLOTS
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import pack_mtp_layers, pack_verify
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, dp_submeshes, model_path
from models.experimental.mimo_v2_6_flash.tests.seeded_rng import SeededRng
from models.experimental.mimo_v2_6_flash.tests.test_spec_inputs_host import (
    REP,
    B,
    mtp_host_values,
    random_positions,
    verify_host_values,
)
from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.model import ROPE_TABLE_MARGIN, TtMiMoModel
from models.experimental.mimo_v2_6_flash.tt.mtp import make_layer_inputs
from models.experimental.mimo_v2_6_flash.tt.mtp_backfill import MtpHistory
from models.experimental.mimo_v2_6_flash.tt.rope import MiMoRotarySetup
from models.experimental.mimo_v2_6_flash.tt.spec_chain import SpecChain
from models.experimental.mimo_v2_6_flash.tt.spec_inputs import SpecInputs
from models.experimental.mimo_v2_6_flash.tt.verify_sampler import GreedyVerify

MAX_SEQ_LEN = 4096
DEVICE_PARAMS = {**GALAXY_DEVICE_PARAMS, "trace_region_size": 20_000_000}


def chips(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def same(t, want, what):
    """Every chip's logical values of device tensor ``t`` equal ``want`` (one tensor, or one per chip)."""
    for c, got in enumerate(chips(t)):
        w = want[c] if isinstance(want, list) else want
        assert torch.equal(
            got.reshape(w.shape).to(w.dtype), w
        ), f"{what}: chip {c}\n{got.flatten()[:8]}\n{w.flatten()[:8]}"


def model_stub(plan, cfg, rope_tables, max_seq_len=MAX_SEQ_LEN):
    """The TtMiMoModel attributes prepare_decode_inputs_host, setup_decode_rope_tables and rope_table_rows read;
    ``rope_tables`` None, "whole" or "compact"."""
    rope_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    stub = SimpleNamespace(
        plan=plan,
        cfg=cfg,
        mesh_device=plan.mesh_device,
        max_batch=B,
        max_seq_len=max_seq_len,
        decode_trace=None,
        rope_tables=None,
        rope_table_tails=None,
        rope_setups={
            "swa": MiMoRotarySetup(plan.mesh_device, B, cfg.head_dim, rope_dim, cfg.swa_rope_theta),
            "full": MiMoRotarySetup(plan.mesh_device, B, cfg.head_dim, rope_dim, cfg.rope_theta),
        },
    )
    if rope_tables is not None:
        TtMiMoModel.setup_decode_rope_tables(stub, compact=rope_tables == "compact")
    return stub


def same_rows(stub, idx, tables, host, what):
    """The table rows at ``idx`` (TtMiMoModel.rope_table_rows) equal the host cos / sin ``host`` once uploaded."""
    mesh = stub.mesh_device
    for k, (row, h) in enumerate(zip(TtMiMoModel.rope_table_rows(stub, idx, tables), host)):
        uploaded = ttnn.to_device(h, mesh)
        same(row, chips(uploaded)[0], f"{what} table {k}")
        row.deallocate(True)
        uploaded.deallocate(True)


def replayed(mesh, run):
    """Run ``run`` eagerly (compiling it), capture it, and return a function replaying the capture."""
    run()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    run()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return lambda: ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)


@pytest.mark.parametrize("rope_tables", [None, "whole", "compact"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_inputs_tp8(mesh_device, rope_tables):
    cfg = MiMoConfig.from_model_path(model_path())
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    rep = kv_lib.kv_replication(plan.tp, cfg.num_key_value_heads)
    stub = model_stub(plan, cfg, rope_tables)
    rope = rope_tables is not None
    spec = SpecInputs(plan, B, rep)
    g = torch.Generator().manual_seed(0)

    # Verify: the trace's inputs, allocated as capture_verify_trace does.
    idle = torch.full((B,), -1, dtype=torch.int32)
    host = TtMiMoModel.prepare_decode_inputs_host(stub, torch.zeros(B, dtype=torch.int32), idle, serial=True)
    inputs = [ttnn.to_device(t, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in host]
    replay = replayed(mesh, lambda: spec.derive_verify(inputs, rope))
    cases = [(torch.zeros(B, dtype=torch.int32), idle)]
    cases += [
        (torch.randint(0, 152576, (B,), generator=g, dtype=torch.int32), random_positions(g, a, 0, MAX_SEQ_LEN))
        for a in (1, 4, 17, B)
    ]
    for k, (tokens, positions) in enumerate(cases):
        for run in (lambda: spec.derive_verify(inputs, rope), replay):
            ttnn.copy_host_to_device_tensor(spec.verify_host(tokens, positions), spec.verify_packed)
            run()
            want = verify_host_values(tokens, positions, rope)
            same(inputs[0], want[0]["tokens"], f"case {k} tokens")
            same(inputs[1], want[0]["cur_pos"], f"case {k} cur_pos")
            same(inputs[6], (positions >= 0).to(torch.bfloat16).reshape(1, 1, -1, 1), f"case {k} mask")
            same(inputs[7], [w["pos_owned"] for w in want], f"case {k} pos_owned")
            owner = kv_lib.owner_mask(positions, plan.tp, rep)
            same(inputs[8], [owner[c] for c in range(plan.tp)], f"case {k} owner_mask")
            if rope:
                same(inputs[9], want[0]["rope_pos"], f"case {k} rope_pos")
                # The four rows the verify reads equal the cos / sin the host would upload for these positions.
                host_rope = TtMiMoModel.prepare_decode_inputs_host(stub, tokens, positions)[2:6]
                same_rows(stub, inputs[9], stub.rope_tables, host_rope, f"case {k} verify RoPE")
            for j, s in enumerate(want[0]["serial"]):
                same(inputs[len(inputs) - SPEC_SLOTS + j], s, f"case {k} serial {j}")

    # MTP draft: tokens, fill, SPEC_SLOTS serial, step2, step3 (TtMiMoMTP._idle_inputs), and the RoPE indices.
    def buf(dtype, shape, value):
        return kv_lib.replicated(plan, torch.full(shape, value, dtype=torch.int32), dtype)

    tokens_tt = buf(ttnn.uint32, (1, B), 0)
    ints = [buf(ttnn.int32, (B,), -1) for _ in range(SPEC_SLOTS + 3)]
    fill, serial, step2, step3 = ints[0], ints[1 : 1 + SPEC_SLOTS], ints[-2], ints[-1]
    idx = [buf(ttnn.uint32, (1, B), 0) for _ in range(3)] if rope else None
    run_mtp = lambda: spec.derive_mtp(tokens_tt, fill, serial, step2, step3, idx)  # noqa: E731
    replay = replayed(mesh, run_mtp)
    drafts = [(None, None)]
    for bonus in (1, 255, 256, 257, 511, 512, MAX_SEQ_LEN - 4):
        drafts.append(
            ([11, 12, 13, 14], plan_mtp(DraftSource(row=0, slots=[0, 1, 2, 3]), [bonus - 2, bonus - 1, bonus, -1], 3))
        )
        drafts.append(([0, 0, 15, 0], plan_mtp(DraftSource(row=0, slots=[2]), [bonus, -1, -1, -1], 1)))
    for k, (toks, mplan) in enumerate(drafts):
        for run in (run_mtp, replay):
            ttnn.copy_host_to_device_tensor(spec.mtp_host(toks, mplan), spec.mtp_packed)
            run()
            want = mtp_host_values(toks, mplan)
            same(tokens_tt, want["tokens"], f"draft {k} tokens")
            same(fill, want["fill"], f"draft {k} fill")
            for j in range(SPEC_SLOTS):
                same(serial[j], want["serial"][j], f"draft {k} serial {j}")
            same(step2, want["step2"], f"draft {k} step2")
            same(step3, want["step3"], f"draft {k} step3")
            if rope:
                for p, (i, w) in enumerate(zip(idx, want["rope_idx"])):
                    same(i, w, f"draft {k} rope index {p}")
                    # The table rows the draft reads equal the host cos / sin it uploaded before.
                    host_pos = w.reshape(-1).to(torch.int32)
                    host_rope = stub.rope_setups["swa"].decode_host_inputs(host_pos)
                    same_rows(stub, i, stub.rope_tables[:2], host_rope, f"draft {k} RoPE pass {p}")


def words_of(t):
    """A replicated row-major tensor's values on every chip, as uint32 words (int64)."""
    return [(ttnn.to_torch(d).reshape(-1).to(torch.int64) & 0xFFFFFFFF).tolist() for d in ttnn.get_device_tensors(t)]


def upload_words(plan, words, t):
    """Copy uint32 ``words`` into the replicated row-major uint32 tensor ``t``."""
    host = torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32).reshape(1, -1)
    rep = plan.replicate()
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), t
    )


def accept_page(count, tokens):
    """SampleVerify's output page with group 0 committing ``tokens`` (count words), the other groups unused."""
    words = []
    for g in range(ACCEPT_WORDS // OUT_WORDS):
        n, toks = (count, tokens) if g == 0 else (0, [])
        out = [n] + list(toks) + [NO_TOKEN] * (1 + MAX_DRAFTS - len(toks))
        words += out + [0] * (OUT_WORDS - len(out))
    return words


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_chain_tp8(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    spec = SpecInputs(plan, B, REP)

    def u32(n):
        return ttnn.from_torch(
            torch.zeros(1, n, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=plan.replicate(),
        )

    accept, argmax, heads = u32(ACCEPT_WORDS), u32(B), [u32(B) for _ in range(MAX_DRAFTS)]
    chain = SpecChain(plan, spec, accept, argmax, heads)

    def run():
        chain.chain()
        chain.drafts()

    replay = replayed(mesh, run)
    rng = SeededRng(0)
    cases = []  # (verify pass, chain row or None, accept words)
    for bonus in (1, 255, 256, 257, 511, 512, MAX_SEQ_LEN - 1 - MAX_DRAFTS, MAX_SEQ_LEN - 2):
        for n in range(MAX_DRAFTS + 1):
            for m in range(1, n + 2):
                if bonus - m < 0:
                    continue
                toks = [rng.randrange(152576) for _ in range(n + 1)]
                (vp,) = plan_verify_passes([(3, toks, list(range(bonus - m, bonus - m + n + 1)), n)])
                committed = [rng.randrange(152576) for _ in range(m)]
                row = chain_row(B, draft_source([vp]), MAX_SEQ_LEN - MAX_DRAFTS)
                cases.append((vp, row, accept_page(m, committed)))
    (two,) = plan_verify_passes([(0, [1, 2], [40, 41], 1), (1, [3], [90], 0)])
    cases.append((two, None, accept_page(2, [5, 6])))
    for k, (vp, row, acc) in enumerate(cases):
        tok = torch.zeros(B, dtype=torch.int32)
        pos = torch.full((B,), -1, dtype=torch.int32)
        tok[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
        pos[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
        chain_t = None if row is None else torch.tensor(row, dtype=torch.int32)
        ids = [rng.randrange(152576) for _ in range(B)]
        head_rows = [[rng.randrange(152576) for _ in range(B)] for _ in range(MAX_DRAFTS)]
        verify_upload = (spec.verify_host(tok, pos, chain_t), spec.verify_packed)
        want_mtp, header = chain_step(pack_verify(tok, pos, chain_t).tolist(), acc, ids)
        want_result = header + drafts_step(want_mtp, head_rows)
        assert len(want_result) == RESULT_WORDS
        for runner in (run, replay):
            ttnn.copy_host_to_device_tensor(*verify_upload)
            upload_words(plan, acc, accept)
            upload_words(plan, ids, argmax)
            for h, words in zip(heads, head_rows):
                upload_words(plan, words, h)
            runner()
            for c, got in enumerate(words_of(spec.mtp_packed)):
                assert got == [w & 0xFFFFFFFF for w in want_mtp], f"case {k} chip {c}: draft upload"
            for c, got in enumerate(words_of(chain.result)):
                assert got == want_result, f"case {k} chip {c}: result page\n{got}\n{want_result}"

    # A host-written upload (a host-planned draft): the drafts step finds its anchor alone.
    host_plan = plan_mtp(DraftSource(row=0, slots=[2]), [700, -1, -1, -1], 1)
    ttnn.copy_host_to_device_tensor(spec.mtp_host([0, 0, 9, 0], host_plan), spec.mtp_packed)
    head_rows = [[rng.randrange(152576) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    for h, words in zip(heads, head_rows):
        upload_words(plan, words, h)
    chain.drafts()
    got = words_of(chain.result)[0]
    assert got[DRAFTS_AT : DRAFTS_AT + 1 + MAX_DRAFTS] == [2] + [r[2] for r in head_rows]


def u32_buffer(plan, n):
    """A persistent replicated uint32 [1, n] row-major DRAM buffer of zeros."""
    return ttnn.from_torch(
        torch.zeros(1, n, dtype=torch.int32),
        device=plan.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.replicate(),
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_chain_dflash_tp8(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    spec = SpecInputs(plan, B, REP)
    layout = DFlashLayout(ring=1024, block=8, heads=8, last=MAX_SEQ_LEN - 1)
    accept, argmax, head = u32_buffer(plan, ACCEPT_WORDS), u32_buffer(plan, B), u32_buffer(plan, B)
    packed = u32_buffer(plan, layout.words)
    chain = SpecChain(plan, spec, accept, argmax, [head], dflash=(packed, layout))

    def run():
        chain.chain()
        chain.drafts()

    replay = replayed(mesh, run)
    rng = SeededRng(1)
    cases = []  # (verify pass, chain row or None, accept words)
    for bonus in (1, 600, 1023, 1024, 1025, 2047, MAX_SEQ_LEN - 1 - MAX_DRAFTS, MAX_SEQ_LEN - 2):
        for n in range(MAX_DRAFTS + 1):
            for m in range(1, n + 2):
                if bonus - m < 0:
                    continue
                toks = [rng.randrange(152576) for _ in range(n + 1)]
                (vp,) = plan_verify_passes([(3, toks, list(range(bonus - m, bonus - m + n + 1)), n)])
                committed = [rng.randrange(152576) for _ in range(m)]
                row = chain_row(B, draft_source([vp]), MAX_SEQ_LEN - MAX_DRAFTS)
                cases.append((vp, row, accept_page(m, committed)))
    (two,) = plan_verify_passes([(0, [1, 2], [40, 41], 1), (1, [3], [90], 0)])
    cases.append((two, None, accept_page(2, [5, 6])))
    for k, (vp, row, acc) in enumerate(cases):
        tok = torch.zeros(B, dtype=torch.int32)
        pos = torch.full((B,), -1, dtype=torch.int32)
        tok[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
        pos[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
        chain_t = None if row is None else torch.tensor(row, dtype=torch.int32)
        ids = [rng.randrange(152576) for _ in range(B)]
        head_row = [rng.randrange(152576) for _ in range(B)]
        verify_upload = (spec.verify_host(tok, pos, chain_t), spec.verify_packed)
        want_packed, header = dflash_chain_step(pack_verify(tok, pos, chain_t).tolist(), acc, ids, layout)
        want_result = header + dflash_drafts_step(want_packed, head_row, layout)
        assert len(want_packed) == layout.words and len(want_result) == RESULT_WORDS
        for runner in (run, replay):
            ttnn.copy_host_to_device_tensor(*verify_upload)
            upload_words(plan, acc, accept)
            upload_words(plan, ids, argmax)
            upload_words(plan, head_row, head)
            runner()
            for c, got in enumerate(words_of(packed)):
                assert got == want_packed, f"case {k} chip {c}: DFlash upload\n{got}\n{want_packed}"
            for c, got in enumerate(words_of(chain.result)):
                assert got == want_result, f"case {k} chip {c}: result page\n{got}\n{want_result}"

    # A host-written upload (a host-planned draft): the drafts step reports the anchor slot it names.
    upload_words(plan, dflash_inputs(layout, [(0, [(0, 700), (1, 701)], 1234, 702, 1)]), packed)
    head_row = [rng.randrange(152576) for _ in range(B)]
    upload_words(plan, head_row, head)
    chain.drafts()
    got = words_of(chain.result)[0]
    assert got[DRAFTS_AT : DRAFTS_AT + 1 + MAX_DRAFTS] == [1] + head_row[:MAX_DRAFTS]


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_greedy_verify_tp8(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    gv = GreedyVerify(plan, B)
    argmax = u32_buffer(plan, B)

    def run():
        gv(None, argmax)

    replay = replayed(mesh, run)
    rng = SeededRng(2)
    cases = []
    for requests in range(1, min(4, SPEC_SLOTS) + 1):
        for _ in range(12):
            active = []
            for r in range(requests):
                n = rng.randrange(0, MAX_DRAFTS + 1)
                start = rng.randrange(0, 2000)
                active.append((r, [rng.randrange(1000) for _ in range(n + 1)], list(range(start, start + n + 1)), n))
            cases += plan_verify_passes(active)
    for k, vp in enumerate(cases):
        slots, groups = pass_inputs(vp, None)
        ctrl = control_words(groups)
        # Each slot's argmax takes the next slot's token (an accepted draft) or another, at random.
        ids = [rng.randrange(1000) for _ in range(B)]
        for s in range(SPEC_SLOTS - 1):
            if rng.random() < 0.7:
                ids[s] = vp.tokens[s + 1]
        want = verify_pass(None, None, ids, ctrl, 0)
        for runner in (run, replay):
            gv.stage(slots, groups)
            upload_words(plan, ids, argmax)
            runner()
            for c, got in enumerate(words_of(gv.out)):
                assert got == [w & 0xFFFFFFFF for w in want], f"case {k} chip {c}\n{got}\n{want}"


def int_words(t):
    """A replicated row-major int32 tensor's values on every chip (signed)."""
    return [ttnn.to_torch(d).reshape(-1).to(torch.int64).tolist() for d in ttnn.get_device_tensors(t)]


def upload_ints(plan, values, t):
    """Copy int32 ``values`` into the replicated row-major tensor ``t`` (reshaped by ``tuple(t.shape)``: older ttnn
    builds' Shape is not a sequence torch's reshape accepts)."""
    host = torch.tensor(values, dtype=torch.int32).reshape(tuple(t.shape))
    rep = plan.replicate()
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(host, dtype=t.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), t
    )


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_chain_layers_tp8(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    spec = SpecInputs(plan, B, REP)

    def u32(n):
        return ttnn.from_torch(
            torch.zeros(1, n, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=plan.replicate(),
        )

    accept, argmax, heads = u32(ACCEPT_WORDS), u32(B), [u32(B) for _ in range(MAX_DRAFTS)]
    li = make_layer_inputs(plan, B, 64)
    chain = SpecChain(plan, spec, accept, argmax, heads, li)
    chain.warm()
    rng = SeededRng(1)

    # The chain and drafts steps.
    def run_chain():
        chain.chain()
        chain.drafts()

    replay = replayed(mesh, run_chain)
    for bonus in (1, 255, 256, 257, MAX_SEQ_LEN - 1 - MAX_DRAFTS, MAX_SEQ_LEN - 2):
        for n in range(MAX_DRAFTS + 1):
            for m in range(1, n + 2):
                if bonus - m < 0:
                    continue
                toks = [rng.randrange(152576) for _ in range(n + 1)]
                (vp,) = plan_verify_passes([(3, toks, list(range(bonus - m, bonus - m + n + 1)), n)])
                committed = [rng.randrange(152576) for _ in range(m)]
                row = chain_row(B, draft_source([vp]), MAX_SEQ_LEN - MAX_DRAFTS)
                tok = torch.zeros(B, dtype=torch.int32)
                pos = torch.full((B,), -1, dtype=torch.int32)
                tok[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
                pos[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
                chain_t = torch.tensor(row, dtype=torch.int32)
                acc = accept_page(m, committed)
                ids = [rng.randrange(152576) for _ in range(B)]
                head_rows = [[rng.randrange(152576) for _ in range(B)] for _ in range(MAX_DRAFTS)]
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
                        assert got == [w & 0xFFFFFFFF for w in want_mtp], f"bonus {bonus} n {n} m {m} chip {c}"
                    for c, got in enumerate(words_of(chain.result)):
                        assert got == want_result, f"bonus {bonus} n {n} m {m} chip {c}: result page"

    # The plan step, then the tokens steps.
    def run_plan():
        chain.plan_layers()
        for layer in range(1, MAX_DRAFTS):
            chain.tokens(layer)

    replay = replayed(mesh, run_plan)
    outs = {"pos": li.pos, "rope": li.rope, "hidden": li.hidden, "tokens0": li.tokens[0]}
    for bonus in (1, 3, 255, 256, 257, 4000, MAX_SEQ_LEN - 1):
        sources = [(DraftSource(row=1, slots=[0, 1, 2, 3]), m) for m in range(1, SPEC_SLOTS + 1)]
        sources += [(DraftSource(row=1, slots=[2]), 1)]
        for source, m in sources:
            if bonus - m < 0:
                continue
            positions = [bonus - m + 1 + c for c in range(m)] + [-1] * (1 + MAX_DRAFTS - m)
            committed = [rng.randrange(152576) for _ in range(m)]
            a = positions[0] - 1
            stashes = [[-1] * (SPEC_SLOTS - r) + [a - r + j for j in range(r)] for r in range(SPEC_SLOTS + 1)]
            stashes += [[-1, a - 3, -1, a - 1], [a - 40, a - 39, a - 38, a - 37]]
            uploads = [pack_mtp_layers(B, source, committed, positions, m).tolist(), pack_mtp_layers(B).tolist()]
            for stash_positions in stashes:
                stash_positions = [p if p >= 0 else -1 for p in stash_positions]
                page = stash_page(B, stash_positions, [rng.randrange(152576) for _ in range(SPEC_SLOTS)])
                head_rows = [[rng.randrange(152576) for _ in range(B)] for _ in range(MAX_DRAFTS)]
                for mtp in uploads:
                    want, new = plan_step(mtp, page, B)
                    want_tokens = [want["tokens0"]] + [
                        tokens_step(want["tokens0"], head_rows, layer) for layer in range(1, MAX_DRAFTS)
                    ]
                    for runner in (run_plan, replay):
                        upload_ints(plan, mtp, spec.mtp_packed)
                        upload_ints(plan, page, li.stash)
                        for h, words in zip(heads, head_rows):
                            upload_words(plan, words, h)
                        runner()
                        what = f"bonus {bonus} m {m} slots {source.slots} stash {stash_positions} idle {mtp[0] == 0}"
                        for name, t in outs.items():
                            for c, got in enumerate(int_words(t)):
                                assert got == want[name], f"{what}: {name} chip {c}"
                        for j, t in enumerate(li.serial):
                            for c, got in enumerate(int_words(t)):
                                assert got == want["serial"][j], f"{what}: serial {j} chip {c}"
                        for layer, t in enumerate(li.tokens):
                            for c, got in enumerate(int_words(t)):
                                assert got == want_tokens[layer], f"{what}: tokens {layer} chip {c}"
                        for c, got in enumerate(int_words(li.stash)):
                            assert got == new, f"{what}: stash page chip {c}"


@pytest.mark.parametrize("layers", [3, 1])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_mtp_history_tp8(mesh_device, layers):
    hidden_size = 256
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    rep = plan.replicate()
    hist = MtpHistory(plan, B, hidden_size, layers)
    li = make_layer_inputs(plan, B, hidden_size) if layers > 1 else None
    hist.warm(li)
    mirror = History(hidden_size)
    rng = SeededRng(layers)
    g = torch.Generator().manual_seed(layers)

    def device_rows(rows):
        return ttnn.from_torch(
            rows.reshape(1, 1, -1, hidden_size),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )

    def check(what):
        for c, got in enumerate(int_words(hist.page)):
            assert got == mirror.page, f"{what}: page chip {c}"
        for c, got in enumerate(chips(hist.rows)):
            assert torch.equal(
                got.reshape(mirror.hidden.shape).float(), mirror.hidden.bfloat16().float()
            ), f"{what}: rows chip {c}"

    cur_pos = ttnn.from_torch(
        torch.full((B,), -1, dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=rep,
    )
    tokens_tt = ttnn.from_torch(
        torch.zeros(1, B, dtype=torch.int32),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=rep,
    )
    src = device_rows(torch.zeros(B, hidden_size))
    spec = SpecInputs(plan, B, REP)

    def decode():
        hist.stage(tokens_tt)
        hist.decode(cur_pos, src)

    def verify():
        hist.verify(spec.verify_packed, src)

    replay_decode, replay_verify = replayed(mesh, decode), replayed(mesh, verify)
    for k in range(6):
        rows = torch.randn(B, hidden_size, generator=g).bfloat16().float()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                rows.reshape(1, 1, B, hidden_size), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
            ),
            src,
        )
        pos = [rng.choice([-1, rng.randrange(1 << 20)]) for _ in range(B)]
        tok = [rng.randrange(152576) for _ in range(B)]
        upload_ints(plan, pos, cur_pos)
        upload_words(plan, tok, tokens_tt)
        (decode if k % 2 else replay_decode)()
        mirror.apply(decode_writes(pos, tok), rows)
        check(f"decode {k}")
        vtok = torch.tensor([rng.randrange(152576) for _ in range(B)], dtype=torch.int32)
        vpos = torch.tensor([rng.randrange(1 << 20) if j < 4 else -1 for j in range(B)], dtype=torch.int32)
        chain = chain_row(B, None, 0, hslots=[rng.choice([None, 0, 1, 2, 3]) for _ in range(SPEC_SLOTS)])
        ttnn.copy_host_to_device_tensor(
            spec.verify_host(vtok, vpos, torch.tensor(chain, dtype=torch.int32)), spec.verify_packed
        )
        (verify if k % 2 else replay_verify)()
        verify_words = pack_verify(vtok, vpos, torch.tensor(chain, dtype=torch.int32)).tolist()
        mirror.apply(verify_writes(verify_words, chain[CHAIN_HSLOTS : CHAIN_HSLOTS + SPEC_SLOTS]), rows)
        check(f"verify {k}")
    # A prefill chunk's list (eager, as the prefill runs it), then backfills of every slot.
    for slot, start, n, cont in ((1, 0, 200, False), (2, 64, 40, True), (3, 0, 7, False)):
        x = torch.randn(1, 1, 256, hidden_size, generator=g).bfloat16().float()
        writes = prefill_writes(slot, start, [rng.randrange(152576) for _ in range(n)], 0, cont)
        norm = lambda t, prefill: ttnn.clone(t)  # noqa: E731 (the identity: the list's copy is what is checked)
        lst = hist.prefill_list(writes)
        hist.prefill(device_rows(x.reshape(-1, hidden_size)), norm, lst, len(writes))
        mirror.apply(writes, x.reshape(-1, hidden_size))
        check(f"prefill slot {slot}")
    run = lambda: hist.backfill_plan(li)  # noqa: E731
    replay = replayed(mesh, run)
    for slot in range(SPEC_SLOTS):
        for a in (5, 130, 300, 4000):
            # The window before a and a verify pass's rows at a .. a + 3 (the plan's tokens past a).
            for q in range(max(0, a - MTP_WINDOW + 1), a + SPEC_SLOTS):
                if rng.random() < 0.9:
                    e = entry(slot, q)
                    mirror.page[e], mirror.page[len(mirror.page) // 2 + e] = q, rng.randrange(152576)
            upload_ints(plan, mirror.page, hist.page)
            for upload in (backfill_row(B, slot, a), backfill_row(B)):
                stash = stash_page(B, [1, 2, 3, 4], [5, 6, 7, 8]) if li is not None else None
                if li is not None:
                    upload_ints(plan, stash, li.stash)
                want, new = backfill_plan(upload, mirror.page, layers, stash)
                upload_ints(plan, upload, hist.upload)
                (run if a % 2 else replay)()
                for name, t in (("hidden", hist.hidx), ("rope", hist.rope), ("table", hist.table)):
                    for c, got in enumerate(int_words(t)):
                        assert got == want[name], f"slot {slot} a {a} {name} chip {c}"
                for i, t in enumerate(hist.tokens):
                    for c, got in enumerate(int_words(t)):
                        assert got == want["tokens"][i], f"slot {slot} a {a} tokens {i} chip {c}"
                if li is not None:
                    for c, got in enumerate(int_words(li.stash)):
                        assert got == new, f"slot {slot} a {a} stash chip {c}"
                    stash_rows = chips(li.stash_hidden)[0].reshape(B, hidden_size).float()
                    for k, e in want["stash_rows"]:
                        assert torch.equal(stash_rows[k], mirror.hidden[e].bfloat16().float())


@pytest.mark.parametrize("compact", [True, False], ids=["compact", "whole"])
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_rope_table_rows_1m_tp8(mesh_device, compact):
    """1M-position tables (537 MB per chip compact, 1.61 GB whole): the rows a step reads at its RoPE index
    (max(position, 0)) equal the cos / sin the host uploads for its positions, all four tables, eager and traced."""
    max_seq_len = 1 << 20
    cfg = MiMoConfig.from_model_path(model_path())
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    stub = model_stub(plan, cfg, "compact" if compact else "whole", max_seq_len)
    end = max_seq_len + ROPE_TABLE_MARGIN
    g = torch.Generator().manual_seed(0)
    cases = [torch.arange(base, base + B, dtype=torch.int32) for base in (0, 131072 - 16, max_seq_len - B, end - B)]
    cases += [random_positions(g, a, 0, max_seq_len) for a in (1, 4, 17, B)]
    idx = kv_lib.replicated(plan, torch.zeros(1, B, dtype=torch.int32), ttnn.uint32)
    made = []

    def gather():
        made[:] = TtMiMoModel.rope_table_rows(stub, idx, stub.rope_tables)

    gather()
    for t in made:
        t.deallocate(True)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    gather()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    traced = list(made)
    for k, positions in enumerate(cases):
        host_idx = kv_lib.replicated(plan, positions.clamp(min=0).reshape(1, B), ttnn.uint32, device=False)
        ttnn.copy_host_to_device_tensor(host_idx, idx)
        # The host cos / sin of these positions (decode_host_inputs: what the step would upload without the tables).
        swa, full = stub.rope_setups["swa"], stub.rope_setups["full"]
        host = [*swa.decode_host_inputs(positions), *full.decode_host_inputs(positions)]
        same_rows(stub, idx, stub.rope_tables, host, f"case {k} eager")
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        for j, (row, h) in enumerate(zip(traced, host)):
            uploaded = ttnn.to_device(h, mesh)
            same(row, chips(uploaded)[0], f"case {k} traced table {j}")
            uploaded.deallocate(True)
    ttnn.release_trace(mesh, tid)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_spec_relay_tp8(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    mesh = plan.mesh_device
    spec = SpecInputs(plan, B, REP)

    def u32(n):
        return ttnn.from_torch(
            torch.zeros(1, n, dtype=torch.int32),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=plan.replicate(),
        )

    accept, argmax, heads = u32(ACCEPT_WORDS), u32(B), [u32(B) for _ in range(MAX_DRAFTS)]
    ctrl = u32(GROUPS * GROUP_WORDS)
    chain = SpecChain(plan, spec, accept, argmax, heads)
    replay = replayed(mesh, lambda: chain.relay(ctrl))
    rng = SeededRng(1)
    limit = MAX_SEQ_LEN - MAX_DRAFTS

    def one_pass(tokens, first, n, relay=False):
        """(tok, pos, chain row) tensors of a pass of one request at ``first`` with n drafts, and its control page."""
        (vp,) = plan_verify_passes([(3, tokens, list(range(first, first + n + 1)), n)])
        tok = torch.zeros(B, dtype=torch.int32)
        pos = torch.full((B,), -1, dtype=torch.int32)
        tok[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
        pos[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
        row = torch.tensor(chain_row(B, draft_source([vp]), limit, relay=relay), dtype=torch.int32)
        return (tok, pos, row), control_words(pass_inputs(vp, None)[1])

    def page_of(p, m, page_limit=limit, layers=1):
        """The result page of a pass at p with MAX_DRAFTS drafts that committed m tokens (the chain and drafts steps'
        mirrors; with every MTP layer the drafts are slot 3's)."""
        (tok, pos, row), _ = one_pass([rng.randrange(152576) for _ in range(1 + MAX_DRAFTS)], p, MAX_DRAFTS)
        row[2] = page_limit
        committed = [rng.randrange(152576) for _ in range(m)]
        ids = [rng.randrange(152576) for _ in range(B)]
        mtp, header = chain_step(pack_verify(tok, pos, row).tolist(), accept_page(m, committed), ids, layers)
        heads = [[rng.randrange(152576) for _ in range(B)] for _ in range(MAX_DRAFTS)]
        return header + drafts_step(mtp, heads, layers)

    cases = []  # (verify upload tensors, page words, control words)
    for bonus, m, layers in itertools.product(
        (1, 255, 256, 257, 4000, MAX_SEQ_LEN - 1 - MAX_DRAFTS), range(1, MAX_DRAFTS + 2), (1, MAX_DRAFTS)
    ):
        if bonus - m >= 0:
            page = page_of(bonus - m, m, layers=layers)
            for n in range(1, MAX_DRAFTS + 1):
                upload, ctrl_words = one_pass([0] * (n + 1), bonus + 2 * MAX_DRAFTS, n, relay=True)
                cases.append((upload, page, ctrl_words))
    declined = page_of(limit - 2, 2)
    assert declined[STATUS_AT] == 0
    cases.append((one_pass([0] * (1 + MAX_DRAFTS), limit + 4, MAX_DRAFTS, relay=True)[0], declined, control_words([])))
    # Host content beside the relayed slots (one draft relayed into slots 0 .. 1).
    (tok, pos, row), ctrl_words = one_pass([0, 0], 600, 1, relay=True)
    tok[2:4], pos[2:4], tok[20], pos[20] = torch.tensor([77, 78]), torch.tensor([900, 901]), 79, 902
    cases.append(((tok, pos, row), page_of(590, 3), ctrl_words))
    # No relay word: nothing written.
    cases.append((one_pass([5, 6, 7], 700, 2)[0], page_of(690, 4), control_words([])))
    for k, ((tok, pos, row), page, ctrl_words) in enumerate(cases):
        want_verify, want_ctrl = relay_step(pack_verify(tok, pos, row).tolist(), page, ctrl_words)
        for runner in (lambda: chain.relay(ctrl), replay):
            ttnn.copy_host_to_device_tensor(spec.verify_host(tok, pos, row), spec.verify_packed)
            upload_words(plan, page, chain.result)
            upload_words(plan, ctrl_words, ctrl)
            runner()
            for c, got in enumerate(words_of(spec.verify_packed)):
                assert got == [w & 0xFFFFFFFF for w in want_verify], f"case {k} chip {c}: verify upload"
            for c, got in enumerate(words_of(ctrl)):
                assert got == want_ctrl, f"case {k} chip {c}: control page"
