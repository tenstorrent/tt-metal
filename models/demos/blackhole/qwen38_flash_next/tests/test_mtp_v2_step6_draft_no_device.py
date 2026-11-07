# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2 step 6: the draft body, the device token chain and the pass loop on the torch-backed ttnn (no device).

The pass loop's committed stream must equal the greedy MTP's for every acceptance pattern at any k, including the
EOS cut; the draft body's ids equal eager MTP rows fed the same tokens, bitwise.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    DRAFT_VOCAB,
    DRAFT_WIDTH,
    FP32,
    HOST_TENSOR_CALLS,
    ROW_MAJOR,
    TILE,
    TP,
    U32,
    VERIFY_BLOCKS,
    VERIFY_CONTEXT,
    DraftWorld,
    FakeContract,
    FakeTensor,
    RowsMixer,
    bf16_randn,
    cat_shards,
    draft_body_model,
    draft_history,
    eager_draft_rows,
    fp32_lanes,
    gdn_rows_selectors,
    make_qsa_module,
    ordered_calls,
    patch_ttnn,
    replicated,
    same_zero,
    sharded,
    source_functions,
    source_methods,
    source_segment,
    u32_scalar,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNLayerNamespace

ROOT = Path(__file__).resolve().parents[1]
MTP_V2_SOURCE = ROOT / "ttnn" / "mtp_v2.py"
QSA_SOURCE = ROOT / "ttnn" / "qsa.py"
PROMPT = [5, 9, 2]


@pytest.fixture
def fake(verify_fake, monkeypatch):
    patch_ttnn(monkeypatch, verify_fake, contracts_module)
    return verify_fake


# --------------------------------------------------------------------------- the draft raw history


@pytest.mark.parametrize("rows", (4, 5, 6))
def test_commit_verify_into_the_draft_state_leaves_the_alignment_window_and_advances_by_one(
    expect_error, fake, rows: int
) -> None:
    module = make_qsa_module()
    alignment_state, draft_state = module.allocate_verify_state(), module.allocate_verify_state()
    constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", FakeContract(), rows=rows)
    advance = gdn_rows_selectors(0, constants)
    torch.manual_seed(7)
    history, raw_rows = bf16_randn(32, 128), bf16_randn(32, 128)
    history[3:] = 0
    for accepted in range(rows):
        for target in alignment_state.raw_history.locals:
            target[0, 0] = history
        for target in alignment_state.raw_rows.locals:
            target[0, 0] = raw_rows
        for target in draft_state.raw_history.locals + draft_state.raw_rows.locals:
            target[0, 0] = 5.0
        selectors = gdn_rows_selectors(accepted, constants)
        module.commit_verify(alignment_state, selectors, target=draft_state)
        expected = draft_history(history, raw_rows, accepted, 0)
        assert same_zero(draft_state.raw_history.torch_shards()[0][0, 0], expected), accepted
        assert torch.equal(alignment_state.raw_history.torch_shards()[0][0, 0], history), "alignment history touched"
        assert torch.equal(alignment_state.raw_rows.torch_shards()[0][0, 0], raw_rows), "alignment raw rows touched"
        assert torch.all(draft_state.raw_rows.torch_shards()[0] == 5.0), "the draft raw rows are not written here"
        # One draft row lands in raw_rows row 0; the a = 0 advance moves the window by one: [h1, h2, row0].
        row0 = bf16_randn(128)
        for target in draft_state.raw_rows.locals:
            target[0, 0] = 0
            target[0, 0, 0] = row0
        module.commit_verify(draft_state, advance)
        advanced = torch.zeros(32, 128, dtype=torch.bfloat16)
        advanced[:2] = expected[1:3]
        advanced[2] = row0
        assert same_zero(draft_state.raw_history.torch_shards()[0][0, 0], advanced), accepted
        selectors.deallocate()
    released = module.allocate_verify_state()
    module.release_verify_state(released)
    with expect_error(ValueError):  # allow-pytest.raises: the target must be a live verify state of this module
        module.commit_verify(alignment_state, advance, target=released)


# --------------------------------------------------------------------------- the stub MTP model on the fake


def _build(fake, monkeypatch, world: DraftWorld, *, drafts: int, position: int, accepted: int):
    """The verify/draft/state objects the draft body reads, over the real QSA module, constants and selectors."""

    rows = drafts + 1
    contract = FakeContract()
    qsa = make_qsa_module()
    generic = qsa.allocate_generic_state()
    alignment_qsa = qsa.allocate_verify_state()
    torch.manual_seed(21)
    for target in alignment_qsa.raw_history.locals:
        target[0, 0, :3] = bf16_randn(3, 128)
    for target in alignment_qsa.raw_rows.locals:
        target[0, 0] = bf16_randn(32, 128)
    chunk_constants = qsa_module.Qwen38TTNNQSAChunkConstants.build("mesh", contract, VERIFY_BLOCKS)
    verify_constants = qsa_module.Qwen38TTNNQSAVerifyConstants.build("mesh", contract, chunk_constants, rows=rows)
    rows_constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", contract, rows=rows)
    accept_constants = mtp_v2.Qwen38TTNNAcceptConstants.build("mesh", contract, drafts=drafts)
    mlp = SimpleNamespace(weights=object(), rows=1, mesh_device="mesh", mesh_contract=contract)
    layer = SimpleNamespace(
        namespace=Qwen38TTNNLayerNamespace.MTP, layer_index=0, attention=qsa, ple=None, mlp=mlp, mesh_contract=contract
    )
    alignment_layer_state = mtp_v2.Qwen38TTNNVerifyLayerState(layer.namespace, 0, rows, alignment_qsa, None, mlp, None)
    residual = sharded(bf16_randn(1, 4, 1, DRAFT_WIDTH * TP))
    alignment = mtp_v2.Qwen38TTNNVerifyAlignment(
        layer,
        RowsMixer(world.mix),
        RowsMixer(world.final),
        SimpleNamespace(attention=generic),
        alignment_layer_state,
        residual,
    )
    owner = object()
    verify = SimpleNamespace(
        drafts=drafts,
        rows=rows,
        rows_constants=rows_constants,
        qsa_chunk_constants=chunk_constants,
        qsa_verify_constants=verify_constants,
        accept_constants=accept_constants,
        layers=(),
        token_row=replicated(torch.full((1, 1, 1, 32), -1.0), FP32),
        draft_lanes=fp32_lanes([]),
        accepted=replicated(torch.full((1, 1, 1, 1), float(accepted)), FP32),
        alignment=alignment,
        _owner=owner,
    )

    def rows_chunk(index_rows, block_start_rows):
        lanes = index_rows.torch_shards()[0].reshape(-1).tolist()
        world.rope_requests.append(lanes)
        assert block_start_rows.torch_shards()[0].reshape(-1)[:8].tolist() == [
            4 * (lanes[0] // 4) + 4 * i for i in range(8)
        ]
        tile = replicated(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), BF16)
        return SimpleNamespace(cos=tile, sin=tile, block_start_cos=tile, block_start_sin=tile, deallocate=lambda: None)

    model = draft_body_model(world, contract, owner, rows_chunk)
    device_position = contracts_module.Qwen38TTNNDevicePosition(
        u32_scalar(position),
        replicated(torch.ones(1, 1, 1, 32, dtype=torch.int64), U32, ROW_MAJOR),
        replicated(
            torch.full((1, 1, 1, 32), contracts_module.BLOCK_START_LANE_MASK, dtype=torch.int64), U32, ROW_MAJOR
        ),
        "mesh",
        contract,
    )
    state = SimpleNamespace(position=device_position)

    def forward_layer_verify(
        layer_,
        residual_,
        generic_state,
        layer_verify,
        *,
        prepared_ple_rows,
        rope_rows,
        qsa_verify,
        qsa_chunk_constants,
        selectors,
    ):
        assert layer_ is layer and generic_state is alignment.generic_state and layer_verify.rows == 1
        assert prepared_ple_rows is None and selectors is None and qsa_chunk_constants is chunk_constants
        assert residual_.shape == (1, 4, 32, DRAFT_WIDTH) and residual_.dtype is BF16
        assert torch.equal(
            cat_shards(residual_)[:, :, 1:], torch.zeros(1, 4, 31, DRAFT_WIDTH * TP, dtype=torch.bfloat16)
        )
        world.positions.append(
            {
                "kv_block_start": int(qsa_verify.chunk.kv_block_start.torch_shards()[0].reshape(-1)[0]),
                "block": int(qsa_verify.chunk.block_index_i32[0].torch_shards()[0].reshape(-1)[0]),
                "remainder": int(qsa_verify.stage_keep.torch_shards()[0].float().sum()),
            }
        )
        out = world.layer(residual_)
        fake.deallocate(residual_)
        return out

    monkeypatch.setattr(mtp_v2, "_validate_verify_state", lambda model_, verify_: None)
    monkeypatch.setattr(mtp_v2, "_forward_layer_verify", forward_layer_verify)
    return model, verify, state


@pytest.mark.parametrize("drafts", (3, 4, 5))
@pytest.mark.parametrize("position,accepted", ((5, 0), (29, 2), (31, 1), (32, 3), (60, 0), (100, 2)))
def test_forward_draft_matches_eager_rows_and_assembles_the_host_images(fake, monkeypatch, drafts, position, accepted):
    accepted = min(accepted, drafts)
    world = DraftWorld()
    model, verify, state = _build(fake, monkeypatch, world, drafts=drafts, position=position, accepted=accepted)
    draft = mtp_v2.allocate_draft_state(model, verify)
    assert draft.qsa_constants.rows == 1 and draft.layer_state.moe is verify.alignment.layer.mlp
    assert draft.layer_state.moe_input.shape == (1, 1, 1, DRAFT_WIDTH) and draft.advance_selectors.rows == drafts + 1
    next_token, first_draft = 17, 41
    readback = mtp_v2.Qwen38TTNNVerifyOutput(
        fp32_lanes([float(accepted), next_token, first_draft] + list(range(32)), width=mtp_v2.READBACK_WIDTH)
    )
    alignment_qsa = verify.alignment.verify_state.attention
    history_before = alignment_qsa.raw_history.torch_shards()[0].clone()
    raw_rows_before = alignment_qsa.raw_rows.torch_shards()[0].clone()

    mtp_v2.forward_draft(model, verify, draft, state, readback)

    # The k - 1 ids are the eager rows' ids, bitwise, and the chain starts from d_1 = the readback's first draft.
    expected_ids = eager_draft_rows(world, verify.alignment.residual, [first_draft], drafts - 1)
    pass_readback, tokens = mtp_v2.read_pass_row(verify, draft)
    assert tokens == (next_token, first_draft, *expected_ids), (drafts, position, accepted)
    host_row = embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows(list(tokens))
    assert torch.equal(verify.token_row.torch_shards()[0], host_row) and verify.token_row.layout == TILE
    host_drafts = torch.full((1, 1, 1, 32), -1.0)
    host_drafts[..., :drafts] = torch.tensor(tokens[1:], dtype=torch.float32)
    assert torch.equal(verify.draft_lanes.torch_shards()[0], host_drafts) and verify.draft_lanes.layout == ROW_MAJOR
    # The pass row: the verify readback row the draft followed, then the token lanes (one host readback per pass).
    assert mtp_v2.PASS_ROW_WIDTH == 67 and draft.pass_row.layout == ROW_MAJOR
    pass_row = draft.pass_row.torch_shards()[0]
    assert torch.equal(pass_row[..., : mtp_v2.READBACK_WIDTH], readback.readback.torch_shards()[0])
    assert torch.equal(pass_row[..., mtp_v2.READBACK_WIDTH :], host_row)
    assert (pass_readback.accepted, pass_readback.next_token, pass_readback.first_draft) == (accepted, 17, 41)
    assert pass_readback.argmaxes == tuple(range(drafts + 1))
    # The rows ran at MTP positions P .. P + k - 2 (KV block, compressed block and block row of each).
    assert [entry["block"] for entry in world.positions] == [(position + i) // 4 for i in range(drafts - 1)]
    assert [entry["kv_block_start"] for entry in world.positions] == [(position + i) & ~31 for i in range(drafts - 1)]
    assert [entry["remainder"] for entry in world.positions] == [(position + i) % 32 for i in range(drafts - 1)]
    assert [lanes[0] for lanes in world.rope_requests] == [position + i for i in range(drafts - 1)]
    assert all(lanes == list(range(lanes[0], lanes[0] + 32)) for lanes in world.rope_requests)
    # The alignment window is untouched; the draft history started as window[a + 1 : a + 4] and advanced once per
    # row after the first (the stub layer writes no raw rows, so the advances shift zeros in).
    assert torch.equal(alignment_qsa.raw_history.torch_shards()[0], history_before)
    assert torch.equal(alignment_qsa.raw_rows.torch_shards()[0], raw_rows_before)
    expected_history = draft_history(history_before[0, 0], raw_rows_before[0, 0], accepted, drafts - 2)
    assert same_zero(draft.qsa_state.raw_history.torch_shards()[0][0, 0], expected_history)
    # The position scalar is untouched (the verify body owns P).
    assert state.position.read() == position
    mtp_v2.release_draft_state(model, verify, draft)


def test_forward_draft_refuses_a_released_readback_and_a_foreign_draft_state(expect_error, fake, monkeypatch) -> None:
    world = DraftWorld()
    model, verify, state = _build(fake, monkeypatch, world, drafts=4, position=8, accepted=1)
    draft = mtp_v2.allocate_draft_state(model, verify)
    output = mtp_v2.Qwen38TTNNVerifyOutput(fp32_lanes([1.0, 3, 4] + [0] * 32, width=mtp_v2.READBACK_WIDTH))
    output.release_tensors()
    with expect_error(ValueError):  # allow-pytest.raises: the readback address must be live
        mtp_v2.forward_draft(model, verify, draft, state, output)
    other = SimpleNamespace(**vars(verify))
    other.drafts, other.rows = 3, 4
    with expect_error(ValueError):  # allow-pytest.raises: k must match
        mtp_v2._validate_draft_state(model, other, draft)
    assert mtp_v2.verify_pass_fits(VERIFY_CONTEXT - 33, VERIFY_CONTEXT) and not mtp_v2.verify_pass_fits(
        VERIFY_CONTEXT - 32, VERIFY_CONTEXT
    )
    assert mtp_v2.verify_pass_fits(0, 64) and not mtp_v2.verify_pass_fits(32, 64)
    with expect_error(ValueError):  # allow-pytest.raises: exactly one second trace form
        mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2)
    with expect_error(ValueError):  # allow-pytest.raises: exactly one second trace form
        mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, verify_catch_up=3, commit=4)


# --------------------------------------------------------------------------- the pass loop vs the greedy MTP stream


def _greedy_mtp_stream(target, drafter, prompt: list[int], *, k: int, passes: int, eos=()) -> list[int]:
    """The greedy MTP's committed stream at draft count k: the matching draft prefix, then the target's replacement
    (or bonus) token; cut at the first EOS."""

    stream = list(prompt)
    emitted: list[int] = []
    for _ in range(passes):
        drafts = []
        for _ in range(k):
            drafts.append(drafter(stream + drafts))
        targets = [target(stream + drafts[:j]) for j in range(k + 1)]
        matched = 0
        while matched < k and drafts[matched] == targets[matched]:
            matched += 1
        committed = targets[: matched + 1]
        for token in committed:
            emitted.append(token)
            if token in eos:
                return emitted
        stream.extend(committed)
    return emitted


class _FakeDevice:
    """The traces as the device would run them, over the chain's fake buffers: the verify trace reads the token
    row, resolves the targets from its own committed stream, accepts, writes the readback row and advances; the
    draft trace reads the readback row, extends the drafts and assembles the next token row."""

    def __init__(self, verify, draft, output, target, drafter, prompt: list[int]) -> None:
        self.verify, self.draft, self.output = verify, draft, output
        self.target, self.drafter = target, drafter
        self.stream = list(prompt)
        self.committed_count = 0
        self.replays: list[str] = []
        self.traces = {1: self._verify, 2: self._draft, 3: self._commit}

    def replay(self, trace_id: int) -> None:
        self.traces[trace_id]()

    def _verify(self) -> None:
        self.replays.append("verify")
        k = self.verify.drafts
        tokens = self.verify.token_row.torch_shards()[0].reshape(-1)[: k + 1].long().tolist()
        drafts = self.verify.draft_lanes.torch_shards()[0].reshape(-1)[:k].long().tolist()
        assert tokens[0] == self.stream[-1] and tokens[1:] == drafts
        targets = [self.target(self.stream + drafts[:j]) for j in range(k + 1)]
        accepted = 0
        while accepted < k and drafts[accepted] == targets[accepted]:
            accepted += 1
        self.stream.extend(targets[: accepted + 1])
        first_draft = self.drafter(self.stream)
        lanes = [float(accepted), float(targets[accepted]), float(first_draft)] + [float(t) for t in targets]
        lanes += [-1.0] * (mtp_v2.READBACK_WIDTH - len(lanes))
        for local in self.output.readback.locals:
            local.copy_(torch.tensor(lanes).reshape(1, 1, 1, -1))
        for local in self.verify.accepted.locals:
            local.fill_(float(accepted))

    def _draft(self) -> None:
        self.replays.append("draft")
        k = self.verify.drafts
        row = self.output.readback.torch_shards()[0].reshape(-1)
        next_token, drafts = int(row[1]), [int(row[2])]
        assert next_token == self.stream[-1]
        for _ in range(k - 1):
            drafts.append(self.drafter(self.stream + drafts))
        token_row = embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows([next_token, *drafts])
        draft_lanes = torch.full((1, 1, 1, 32), -1.0)
        draft_lanes[..., :k] = torch.tensor(drafts, dtype=torch.float32)
        for local in self.verify.token_row.locals:
            local.copy_(token_row)
        for local in self.draft.pass_row.locals:  # the pass row: the verify row this draft read, then the tokens
            local.copy_(torch.cat([row.reshape(1, 1, 1, -1), token_row], dim=3))
        for local in self.verify.draft_lanes.locals:
            local.copy_(draft_lanes)

    def _commit(self) -> None:
        self.replays.append("commit")

    def enqueue(self, trace_id: int) -> None:
        self.traces[trace_id]()
        self.replays[-1] += "-enqueued"


class _RecordingPLE:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], object]] = []

    def host_rows(self, tokens, context):
        tokens = tuple(int(t) for t in tokens)
        self.calls.append((tokens, context))
        contexts = [context]
        for token in tokens:
            context = (context[1] if context else -1, token)
            contexts.append(context)
        return torch.zeros(1, 1, len(tokens), 2560, dtype=torch.bfloat16), tuple(contexts)


def _oracles(pattern: tuple[int, ...], k: int):
    """A target and a drafter whose pass n accepts exactly ``pattern[n]`` drafts: the drafter copies the target on
    the first ``pattern[n]`` rows of pass n and misses on the next one.  The pass index is the count of committed
    tokens so far (a function of the stream length), so both oracles are pure functions of their context."""

    def target(context):
        return (sum(context) * 7 + len(context) * 3) % DRAFT_VOCAB

    boundaries = list(itertools.accumulate(a + 1 for a in pattern))  # committed count after pass n

    def drafter(context):
        # The stream inside a pass: committed tokens (prompt + emissions) plus the drafts so far.
        committed = len(context) - len(PROMPT)
        pass_index = next((n for n, b in enumerate(boundaries) if committed < b), len(pattern))
        drafts_so_far = committed - (boundaries[pass_index - 1] if pass_index else 0)
        wanted = pattern[pass_index] if pass_index < len(pattern) else 0
        value = target(context)
        return value if drafts_so_far < wanted else (value + 1) % DRAFT_VOCAB

    return target, drafter


def _chain(fake, monkeypatch, pattern, *, k: int, split: bool, eos=(), enqueue=False, observer=None):
    """The pass loop over ``_FakeDevice`` with the oracles of ``pattern``, and its bootstrap tokens: the prompt's last
    token and the greedy MTP's own first drafts (the host-side draft of pass 0)."""

    monkeypatch.setattr(mtp_v2, "_validate_verify_state", lambda model_, verify_: None)
    monkeypatch.setattr(mtp_v2, "_validate_draft_state", lambda model_, verify_, draft_: None)
    target, drafter = _oracles(pattern, k)
    ple = _RecordingPLE()
    ple_state = SimpleNamespace(rows=k + 1, token_context=None, validate=lambda: None)
    ple_rows = SimpleNamespace(
        embedding_rows=FakeTensor(
            [torch.zeros(1, 1, k + 1, 640, dtype=torch.bfloat16) for _ in range(TP)], BF16, ROW_MAJOR, 3
        ),
        tokens=(),
        contexts=(),
        active=True,
    )
    layers = [SimpleNamespace(ple=None)] * 48
    layers[1] = SimpleNamespace(ple=ple)
    host_verify_token_rows = embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows
    model = SimpleNamespace(
        mesh_device="mesh",
        allocated_context=VERIFY_CONTEXT,
        layers=layers,
        model_io=SimpleNamespace(embedding=SimpleNamespace(host_verify_token_rows=host_verify_token_rows)),
    )
    verify_layers = [SimpleNamespace(ple=None)] * 48
    verify_layers[1] = SimpleNamespace(ple=ple_state)
    verify = SimpleNamespace(
        drafts=k,
        rows=k + 1,
        layers=verify_layers,
        token_row=replicated(torch.full((1, 1, 1, 32), -1.0), FP32),
        draft_lanes=fp32_lanes([]),
        ple_rows=ple_rows,
        accepted=replicated(torch.full((1, 1, 1, 1), float(k)), FP32),
    )
    draft = SimpleNamespace(pass_row=fp32_lanes([], width=mtp_v2.PASS_ROW_WIDTH))
    output = mtp_v2.Qwen38TTNNVerifyOutput(fp32_lanes([0.0] * mtp_v2.READBACK_WIDTH, width=mtp_v2.READBACK_WIDTH))
    device = _FakeDevice(verify, draft, output, target, drafter, PROMPT)
    traces = (
        mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, commit=3)
        if split
        else mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, verify_catch_up=1)
    )
    chain = mtp_v2.Qwen38TTNNMTPChain(
        model,
        verify,
        draft,
        traces,
        output,
        replay=device.replay,
        position=len(PROMPT) - 1,
        eos_token_ids=eos,
        enqueue=device.enqueue if enqueue else None,
        observer=observer,
    )
    first_drafts = []
    for _ in range(k):
        first_drafts.append(drafter(PROMPT + first_drafts))
    return SimpleNamespace(
        chain=chain, device=device, ple=ple, target=target, drafter=drafter, bootstrap=[PROMPT[-1], *first_drafts]
    )


def _run_passes(run, passes: int) -> list[int]:
    """Bootstrap, then the traced passes: the committed stream."""

    emitted = list(run.chain.bootstrap(run.bootstrap).committed)
    for _ in range(passes - 1):
        emitted.extend(run.chain.step().committed)
    return emitted


@pytest.mark.parametrize("form", ("single", "split", "enqueue"))
@pytest.mark.parametrize("k", (3, 4, 5))
def test_pass_loop_commits_the_greedy_mtp_stream_for_every_acceptance_pattern(fake, monkeypatch, k, form) -> None:
    """The three pass forms: single trace (verify with catch-up -> the next draft), split (blocking commit -> verify
    -> draft) and the production form (commit enqueued non-blocking -> PLE lookup -> verify and the next pass's draft
    enqueued back to back -> one pass-row readback)."""

    split, enqueue = form != "single", form == "enqueue"
    patterns = list(itertools.product(range(k + 1), repeat=2)) + [(a,) * 3 for a in range(k + 1)]
    for pattern in patterns:
        passes = len(pattern)
        segments_seen: list[str] = []
        run = _chain(fake, monkeypatch, pattern, k=k, split=split, enqueue=enqueue, observer=segments_seen.append)
        chain, device, ple = run.chain, run.device, run.ple
        emitted = _run_passes(run, passes)
        assert emitted == _greedy_mtp_stream(run.target, run.drafter, PROMPT, k=k, passes=passes), (k, split, pattern)
        assert [record.accepted for record in chain.records] == list(pattern), (k, split)
        assert chain.position == len(PROMPT) - 1 + len(emitted)
        # Every traced pass looked its R tokens up from the committed context: pass n's tokens are
        # [t' of pass n-1, its first draft, the draft trace's k-1 ids]; the contexts chain through the commits.
        assert len(ple.calls) == passes
        for index, (tokens, _) in enumerate(ple.calls):
            record = chain.records[index]
            assert tokens == record.tokens and tokens[0] == (
                PROMPT[-1] if index == 0 else chain.records[index - 1].next_token
            )
            if index:
                assert tokens[1] == chain.records[index - 1].first_draft
        contexts = [None]
        for record in chain.records:
            token_context = contexts[-1]
            for token in record.tokens[: record.accepted + 1]:
                token_context = (token_context[1] if token_context else -1, token)
            contexts.append(token_context)
        assert [call[1] for call in ple.calls] == contexts[:-1]
        # Every pass runs the verify then the next pass's draft (the draft after the last pass goes unused); the
        # split forms run the commit first.  The pipelined form enqueues every trace and blocks on the one readback.
        suffix = "-enqueued" if enqueue else ""
        commit = ["commit" + suffix] if split else []
        steady_replays = [*commit, "verify" + suffix, "draft" + suffix]
        assert device.replays == ["verify" + suffix, "draft" + suffix] + steady_replays * (passes - 1)
        segments = chain.records[-1].segments_ns
        commit_segment = {"commit_enqueue"} if enqueue else {"commit_replay"} if split else set()
        launch = "enqueue" if enqueue else "replay"
        first = ["host_inputs", f"verify_{launch}", f"draft_{launch}", "readback"]
        if passes > 1:
            assert set(segments) == {"ple_rows", f"verify_{launch}", f"draft_{launch}", "readback"} | commit_segment
        else:
            assert set(segments) == set(first)
        # The observer names every segment as it begins, in the pass order: the commit is enqueued (or replayed)
        # before the PLE lookup, the verify and the draft go out back to back, the readback is last.
        steady = [*sorted(commit_segment), "ple_rows", f"verify_{launch}", f"draft_{launch}", "readback"]
        assert segments_seen == first + steady * (passes - 1)
        assert chain.next_tokens[0] == chain.records[-1].next_token  # the next pass's tokens are already read


def test_pass_loop_cuts_at_eos_and_run_stops_at_the_token_budget(expect_error, fake, monkeypatch) -> None:
    k, pattern = 4, (2, 4, 1, 0)
    reference = _greedy_mtp_stream(*_oracles(pattern, k), PROMPT, k=k, passes=4)
    eos = (reference[4],)  # inside pass 1's committed run
    run = _chain(fake, monkeypatch, pattern, k=k, split=True, eos=eos)
    emitted = run.chain.run(40, bootstrap_drafts=run.bootstrap)
    assert emitted == _greedy_mtp_stream(run.target, run.drafter, PROMPT, k=k, passes=4, eos=eos)
    assert run.chain.finished and run.chain.records[-1].finished and emitted[-1] == eos[0]
    with expect_error(RuntimeError):  # allow-pytest.raises: a finished chain takes no pass
        run.chain.step()
    # Without EOS the run stops at the budget, cutting the last pass's emissions.
    run = _chain(fake, monkeypatch, pattern, k=k, split=False)
    emitted = run.chain.run(6, bootstrap_drafts=run.bootstrap)
    assert emitted == reference[:6] and len(run.chain.records) == 2
    with expect_error(RuntimeError):  # allow-pytest.raises: bootstrap once
        run.chain.bootstrap(run.bootstrap)


def test_pass_loop_refuses_a_draft_token_chain_that_disagrees_with_its_verify_row(
    expect_error, fake, monkeypatch
) -> None:
    run = _chain(fake, monkeypatch, (1, 1), k=3, split=True)
    run.chain.bootstrap(run.bootstrap)
    device = run.device
    original = device._draft

    def corrupted():
        original()
        for local in device.draft.pass_row.locals:
            local[..., mtp_v2.READBACK_WIDTH + 1] += 1.0  # d_1 no longer the verify row's first draft

    device.traces[2] = corrupted
    with expect_error(RuntimeError, match="its verify row says"):  # allow-pytest.raises: the exactness check
        run.chain.step()


class _FakeCommitQueue(mtp_v2.Qwen38TTNNCommitQueue):
    """The second queue's fences as a log: the device's commit runs at the enqueue (the fake device is serial)."""

    def __init__(self, device: _FakeDevice) -> None:
        super().__init__("mesh", cq_id=1, main_cq_id=0)
        self.device, self.ops = device, []

    def history_derived(self) -> None:
        self.ops.append("history_derived")

    def enqueue_commit(self, trace_id: int) -> None:
        assert self.ops and self.ops[-1] == "history_derived", "the commit needs the history fence first"
        self.device.replay(trace_id)
        self.device.replays[-1] += "-cq1"
        self.ops.append("commit_enqueued")

    def replay_commit(self, trace_id: int) -> None:
        self.device.replay(trace_id)
        self.device.replays[-1] += "-cq1-blocking"
        self.ops.append("commit_replayed")

    def wait_committed(self) -> None:
        self.ops.append("wait_committed")


@pytest.mark.parametrize("enqueue", (True, False))
def test_pass_loop_on_two_command_queues_fences_the_commit_behind_the_draft_history(
    expect_error, fake, monkeypatch, enqueue
):
    k, pattern = 4, (2, 4, 0, 1)
    segments_seen: list[str] = []
    run = _chain(fake, monkeypatch, pattern, k=k, split=True, enqueue=enqueue, observer=segments_seen.append)
    chain, device = run.chain, run.device
    device.traces[4] = lambda: device.replays.append("draft-history")  # the split-off derivation (a no-op here)
    queue = _FakeCommitQueue(device)
    chain.traces = mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, commit=3, draft_history=4)
    chain.commit_queue = queue
    emitted = _run_passes(run, len(pattern))
    assert emitted == _greedy_mtp_stream(run.target, run.drafter, PROMPT, k=k, passes=len(pattern))
    assert [record.accepted for record in chain.records] == list(pattern)
    suffix = "-enqueued" if enqueue else ""
    commit = "commit-cq1" if enqueue else "commit-cq1-blocking"
    launch = "enqueue" if enqueue else "replay"
    # Device order per pass: verify, draft history (the fence), draft rows; the commit of the previous pass on its
    # queue right after the fence; the verify waits for it.
    pass_replays = ["verify" + suffix, "draft-history" + suffix, "draft" + suffix]
    assert device.replays == pass_replays + [commit, *pass_replays] * (len(pattern) - 1)
    committed = "commit_enqueued" if enqueue else "commit_replayed"
    assert queue.ops == ["history_derived"] + [committed, "wait_committed", "history_derived"] * (len(pattern) - 1)
    steady = [
        "commit_enqueue" if enqueue else "commit_replay",
        "ple_rows",
        "commit_wait",
        f"verify_{launch}",
        f"draft_history_{launch}",
        f"draft_{launch}",
        "readback",
    ]
    first = ["host_inputs", f"verify_{launch}", f"draft_history_{launch}", f"draft_{launch}", "readback"]
    assert segments_seen == first + steady * (len(pattern) - 1)
    with expect_error(ValueError):  # allow-pytest.raises: a commit queue needs the split draft history
        mtp_v2.Qwen38TTNNMTPChain(
            chain.model,
            chain.verify,
            chain.draft,
            mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, commit=3),
            chain.verify_output,
            replay=device.replay,
            position=3,
            commit_queue=queue,
        )
    with expect_error(ValueError):  # allow-pytest.raises: two distinct queues
        mtp_v2.Qwen38TTNNCommitQueue("mesh", cq_id=0, main_cq_id=0)
    with expect_error(RuntimeError):  # allow-pytest.raises: the fence comes first
        mtp_v2.Qwen38TTNNCommitQueue("mesh").enqueue_commit(3)


class _FakeEarlyReader(mtp_v2.Qwen38TTNNEarlyRowsReader):
    """The second queue's read as a log: the fake device is serial, so the verify's lanes are in the readback row the
    moment the verify replayed; the event and the read are recorded beside the device's replays."""

    def __init__(self, device: _FakeDevice) -> None:
        super().__init__("mesh", device.output.readback, cq_id=1, main_cq_id=0)
        self.device, self.ops = device, []

    def verify_launched(self) -> None:
        self.ops.append("verify_launched")
        self.device.replays.append("event-cq0")

    def read_fixed_lanes(self) -> tuple[int, int, int]:
        assert self.ops and self.ops[-1] == "verify_launched", "the read waits for the verify's event"
        self.ops.append("read")
        self.device.replays.append("lanes-cq1")
        row = self.device.output.readback.torch_shards()[0].reshape(-1)
        return int(row[0]), int(row[1]), int(row[2])


@pytest.mark.parametrize("enqueue", (True, False))
def test_pass_loop_with_the_early_rows_reader_splits_the_lookup_and_keeps_the_stream(
    expect_error, fake, monkeypatch, enqueue
):
    """The early rows reader: the next pass's rows 0-1 are looked up after the draft launch and rows 2..k by the next
    step; the committed stream, the PLE tokens and the contexts are the plain loop's, row for row."""

    k, pattern = 4, (2, 4, 0, 1, 3)
    segments_seen: list[str] = []
    run = _chain(fake, monkeypatch, pattern, k=k, split=True, enqueue=enqueue, observer=segments_seen.append)
    chain, device, ple = run.chain, run.device, run.ple
    reader = _FakeEarlyReader(device)
    chain.early_reader = reader
    plain = _chain(fake, monkeypatch, pattern, k=k, split=True, enqueue=enqueue)
    emitted = _run_passes(run, len(pattern))
    assert emitted == _run_passes(plain, len(pattern))
    assert emitted == _greedy_mtp_stream(run.target, run.drafter, PROMPT, k=k, passes=len(pattern))
    assert [record.accepted for record in chain.records] == list(pattern)
    # The PLE lookups: the bootstrap's whole row, then per pass rows 0-1 (early, from the committed context) and rows
    # 2..k (from the early chain's context[2]); tokens and contexts equal the plain loop's whole-row lookups.
    assert ple.calls[0] == plain.ple.calls[0]
    # Every pass (the last one too) looks the next rows 0-1 up; the steady steps consume all but the last.
    early_calls, late_calls = ple.calls[1::2], ple.calls[2::2]
    assert len(early_calls) == len(pattern) and len(late_calls) == len(pattern) - 1
    for (early_tokens, early_context), (late_tokens, late_context), (tokens, context) in zip(
        early_calls, late_calls, plain.ple.calls[1:]
    ):
        assert early_tokens + late_tokens == tokens and early_context == context
        chained = context
        for token in early_tokens:
            chained = (chained[1] if chained else -1, token)
        assert late_context == chained  # rows 2..k continue where rows 0-1 left the chain
    assert chain.verify.ple_rows.contexts == plain.chain.verify.ple_rows.contexts
    assert chain.verify.ple_rows.tokens == plain.chain.verify.ple_rows.tokens
    # Device order per pass: verify, its event, the draft, the lanes read on the second queue; the commit first.
    suffix = "-enqueued" if enqueue else ""
    pass_replays = ["verify" + suffix, "event-cq0", "draft" + suffix, "lanes-cq1"]
    assert device.replays == pass_replays + ["commit" + suffix, *pass_replays] * (len(pattern) - 1)
    assert reader.ops == ["verify_launched", "read"] * len(pattern)
    launch = "enqueue" if enqueue else "replay"
    commit = "commit_enqueue" if enqueue else "commit_replay"
    early = [f"verify_{launch}", f"draft_{launch}", "early_readback", "ple_rows_early", "readback"]
    assert segments_seen == ["host_inputs", *early] + [commit, "ple_rows_late", *early] * (len(pattern) - 1)
    assert "ple_rows" not in chain.records[-1].segments_ns and "ple_rows_late" in chain.records[-1].segments_ns
    # The reader serves the fused and the device-decided forms only; the split form keeps its order.
    chain.verify.split = SimpleNamespace()  # the split form's buffers, as far as the chain's checks look
    with expect_error(ValueError, match="not the split"):  # allow-pytest.raises: the form rule
        mtp_v2.Qwen38TTNNMTPChain(
            chain.model,
            chain.verify,
            chain.draft,
            mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, verify_head=1, verify_tail=5, draft=2, commit=3),
            chain.verify_output,
            replay=device.replay,
            position=3,
            head_output=SimpleNamespace(),
            early_reader=reader,
        )
    with expect_error(TypeError):  # allow-pytest.raises: the reader's type
        mtp_v2.Qwen38TTNNMTPChain(
            chain.model,
            chain.verify,
            chain.draft,
            chain.traces,
            chain.verify_output,
            replay=device.replay,
            position=3,
            early_reader=object(),
        )
    with expect_error(ValueError):  # allow-pytest.raises: two distinct queues
        mtp_v2.Qwen38TTNNEarlyRowsReader("mesh", device.output.readback, cq_id=0, main_cq_id=0)
    with expect_error(RuntimeError, match="must be launched"):  # allow-pytest.raises: the event comes first
        mtp_v2.Qwen38TTNNEarlyRowsReader.read_fixed_lanes(
            mtp_v2.Qwen38TTNNEarlyRowsReader("mesh", device.output.readback)
        )


# --------------------------------------------------------------------------- source pins
def test_draft_and_commit_bodies_create_no_host_tensors_and_move_ids_by_copy() -> None:
    functions = source_functions(MTP_V2_SOURCE)
    for name in ("forward_draft", "forward_commit", "_token_row_from_lane", "_lane", "_land"):
        for called in ordered_calls(functions[name]):
            assert not called.startswith(HOST_TENSOR_CALLS), (name, called)
            assert "synchronize" not in called and ".item" not in called and "read()" not in called, (name, called)
    draft = source_segment(MTP_V2_SOURCE, functions["forward_draft"])
    for forbidden in (
        "ttnn.sum(",
        "ttnn.matmul(",
        "ttnn.gather(",
        "ttnn.typecast(",
        "state.position.advance",
        "ttnn.copy(advanced",
    ):
        assert forbidden not in draft, forbidden
    calls = ordered_calls(functions["forward_draft"])
    order = {name: calls.index(name) for name in calls}
    # The draft history is derived (alignment state read only) before any row, by forward_draft_history (the same
    # body captured on its own in the two-queue form); the a = 0 advance sits inside the loop.
    history = source_segment(MTP_V2_SOURCE, functions["forward_draft_history"])
    assert (
        "alignment.layer.attention.commit_verify(alignment.verify_state.attention, selectors, target=draft.qsa_state)"
        in history
    )
    for called in ordered_calls(functions["forward_draft_history"]):
        assert not called.startswith(HOST_TENSOR_CALLS), called
    assert "if derive_history: forward_draft_history(model, verify, draft)" in draft
    assert "qsa.commit_verify(draft.qsa_state, draft.advance_selectors)" in draft
    assert order["forward_draft_history"] < order["_lane"] < order["_token_row_from_lane"] < order["_pad_rows"]
    # One real row per draft step: the 1-row embedding, mixers, LM head and resolve around the 32-row MTP layer.
    assert (
        order["model.model_io.embedding.embed_device_token"]
        < order["alignment.input_mixer"]
        < order["_pad_rows"]
        < order["_forward_layer_verify"]
        < order["ttnn.slice"]
        < order["alignment.final_mixer"]
        < order["lm_head"]
        < order["lm_head.greedy_candidates"]
        < order["lm_head.resolve_greedy_on_device"]
        < order["ttnn.to_layout"]
        < order["ttnn.concat"]
        < order["_land"]
    )
    for rows_form in ("embed_device_token_rows", "input_mixer.rows", "final_mixer.rows", "_resolve_rows"):
        assert rows_form not in draft, rows_form
    assert "residual_row is not alignment.residual" in draft and "_deallocate(residual)" in draft
    assert draft.count("_land(") == 3 and "ttnn.to_layout(token_lanes, ttnn.TILE_LAYOUT" in draft
    # The pass row is the verify readback row followed by the token lanes, landed in the draft state's buffer.
    assert "ttnn.concat([verify_output.readback, token_lanes], dim=3" in draft
    assert '_land(pass_row, draft.pass_row, label="assembled pass row")' in draft
    # The commit body commits PLE, then the attention, per layer, then the alignment layer; no forward.
    commit = source_segment(MTP_V2_SOURCE, functions["forward_commit"])
    commit_calls = ordered_calls(functions["forward_commit"])
    assert (
        commit_calls.index("layer.ple.commit_rows")
        < commit_calls.index("layer.attention.commit_rows")
        < commit_calls.index("layer.attention.commit_verify")
    )
    assert (
        "verify.alignment.layer.attention.commit_verify(verify.alignment.verify_state.attention, selectors)" in commit
    )
    for forbidden in ("forward_rows", "forward_verify_generic", "inject_rows", "state.position"):
        assert forbidden not in commit, forbidden
    # commit_verify keeps its 1-row-of-history contract, with a keyword-only target.
    qsa_functions = source_functions(QSA_SOURCE)
    commit_verify = qsa_functions["commit_verify"]
    assert [arg.arg for arg in commit_verify.args.kwonlyargs] == ["target"] and commit_verify.args.kw_defaults[
        0
    ].value is None
    assert [arg.arg for arg in commit_verify.args.args] == ["self", "verify_state", "selectors"]


def test_pass_loop_step_order_and_the_verify_body_are_pinned() -> None:
    methods = source_methods(MTP_V2_SOURCE, "Qwen38TTNNMTPChain")
    step = ordered_calls(methods["step"])
    order = {name: step.index(name) for name in step}
    assert order["self.enqueue"] < order["self.replay"] < order["write_verify_ple_rows"] < order["self._finish_pass"]
    step_source = source_segment(MTP_V2_SOURCE, methods["step"])
    assert "tokens = self.next_tokens" in step_source and "read_" not in step_source  # read with the previous pass
    assert step_source.index("self.replay(self.traces.commit)") < step_source.index("write_verify_ple_rows(")
    assert step_source.index("self.enqueue(self.traces.commit)") < step_source.index("self.replay(self.traces.commit)")
    assert "commit_enqueue" in step_source and "commit_replay" in step_source
    # The verify and the next pass's draft go out back to back; the one readback follows, then the PLE commit.
    finish_source = source_segment(MTP_V2_SOURCE, methods["_finish_pass"])
    assert (
        finish_source.index("launch(verify_trace)")
        < finish_source.index("launch(self.traces.draft)")
        < finish_source.index("read_pass_row(self.verify, self.draft)")
        < finish_source.index("commit_verify_host(")
    )
    assert "read_verify_output" not in finish_source and "read_verify_output" not in step_source
    # write_verify_inputs delegates the PLE rows; the token row and draft lanes are assembled on device.
    functions = source_functions(MTP_V2_SOURCE)
    inputs = source_segment(MTP_V2_SOURCE, functions["write_verify_inputs"])
    assert inputs.endswith("return write_verify_ple_rows(model, verify, tokens)")
    ple_rows = source_segment(MTP_V2_SOURCE, functions["write_verify_ple_rows"])
    assert "ple.host_rows(tokens, ple_state.token_context)" in ple_rows and "verify.ple_rows.embedding_rows" in ple_rows
    assert "verify.token_row" not in ple_rows and "verify.draft_lanes" not in ple_rows  # assembled on device
    exported = {
        "forward_draft",
        "capture_draft",
        "forward_commit",
        "capture_commit",
        "allocate_draft_state",
        "release_draft_state",
        "read_pass_row",
        "PASS_ROW_WIDTH",
        "write_verify_ple_rows",
        "verify_pass_fits",
        "Qwen38TTNNMTPChain",
        "Qwen38TTNNMTPTraces",
        "Qwen38TTNNMTPPassRecord",
        "Qwen38TTNNDraftState",
    }
    assert exported <= set(mtp_v2.__all__)
