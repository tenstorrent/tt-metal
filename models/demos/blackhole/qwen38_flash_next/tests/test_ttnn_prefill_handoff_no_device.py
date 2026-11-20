# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The padded tail chunk and the hand-off to the 1-row decode on the torch-backed ttnn, no device.

A GDN layer and the PLE layer run the driver's sequence against N 1-row steps: the GDN recurrent state within the
tolerance model, the ring slots, phase, PLE slots and n-gram context bitwise. The QSA hand-off is checked on
position-encoded kept rows.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    ROW_MAJOR,
    TILE,
    TP,
    FakeContract,
    FakeTensor,
    cat_shards,
    clone_gdn_state,
    gdn_output_tolerance,
    gdn_ring_history,
    gdn_rows_selectors,
    hidden_sharded,
    make_gdn_module,
    make_ple_module,
    residual_rows,
    run_gdn_steps,
    seed_gdn_state,
    seed_ple_state,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS

TAIL_ROWS = (0, 1, 2, 8, 9, 31)
PREFILL_CASES = [(tail, tail + 32 * chunks) for tail in TAIL_ROWS for chunks in range(3)]
PAD_TOKEN = 0


def gdn_state_tolerance(reference: torch.Tensor) -> float:
    return 3e-3 + 2.0**-7 * reference.abs().max().item()


def _chunk_plan(count: int) -> list[int]:
    """The accept count of every chunk of a ``count``-token prefill: 31 per full chunk, r - 1 for the padded tail."""

    full, tail = divmod(count, CHUNK_ROWS)
    return [CHUNK_ROWS - 1] * full + ([tail - 1] if tail else [])


# --------------------------------------------------------------------------- GDN


@pytest.mark.parametrize("decoded", (0, 32))
@pytest.mark.parametrize("tail,count", PREFILL_CASES)
def test_gdn_chunks_tail_and_handoff_match_the_one_row_steps(
    rows_fake, gdn_weights, tail: int, count: int, decoded: int
):
    _, weights = gdn_weights
    module = make_gdn_module(weights)
    constants = module.allocate_rows_constants(CHUNK_ROWS)
    torch.manual_seed(1000 + count)
    hidden = torch.randn(1, decoded + count, 2560).to(torch.bfloat16)
    pad_hidden = torch.randn(1, 1, 2560).to(torch.bfloat16)
    state_step = seed_gdn_state(module, 50 + count)
    state_rows = clone_gdn_state(module, state_step)

    # The alignment steps (teacher-forced 1-row decode) run on both sides before the chunks.
    run_gdn_steps(module, state_step, hidden[:, :decoded])
    run_gdn_steps(module, state_rows, hidden[:, :decoded])
    # The reverse seed: the chunk carry from the decode ring (the reset path after decode steps).
    rows_state = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state_rows, rows_state)

    run_gdn_steps(module, state_step, hidden[:, decoded:])
    for chunk, accepted in enumerate(_chunk_plan(count)):
        rows = hidden[:, decoded + CHUNK_ROWS * chunk : decoded + CHUNK_ROWS * (chunk + 1)]
        if rows.shape[1] < CHUNK_ROWS:
            rows = torch.cat([rows, pad_hidden.expand(1, CHUNK_ROWS - rows.shape[1], 2560)], dim=1)
        result = module.forward_rows(hidden_sharded(rows), state_rows, rows_state)
        module.commit_rows(state_rows, rows_state, gdn_rows_selectors(accepted, constants))
        rows_fake.ttnn.deallocate(result.final_state)
    prefilled = decoded + count
    state_rows.conv_phase = prefilled % gdn_module.CONV_KERNEL_SIZE
    module.sync_state_from_rows_history(rows_state, state_rows)

    expected_state = cat_shards(state_step.recurrent, 1)
    state_err = (cat_shards(state_rows.recurrent, 1) - expected_state).abs().max().item()
    print(
        f"r={tail} N={count} decoded={decoded}: state max abs {state_err:.2e} "
        f"(tol {gdn_state_tolerance(expected_state):.2e})"
    )
    assert state_err <= gdn_state_tolerance(expected_state)
    assert state_rows.conv_phase == state_step.conv_phase == prefilled % 4
    assert torch.equal(gdn_ring_history(state_rows), gdn_ring_history(state_step))

    torch.manual_seed(2000 + count)
    next_hidden = torch.randn(1, 1, 2560).to(torch.bfloat16)
    step_next = run_gdn_steps(module, state_step, next_hidden)[0]
    rows_next = run_gdn_steps(module, state_rows, next_hidden)[0]
    assert (rows_next.float() - step_next.float()).abs().max().item() <= gdn_output_tolerance(step_next)
    for slot, expected_slot in zip(state_rows.conv, state_step.conv):
        assert torch.equal(cat_shards(slot, 3), cat_shards(expected_slot, 3))


# --------------------------------------------------------------------------- PLE


@pytest.mark.parametrize("decoded", (0, 32))
@pytest.mark.parametrize("tail,count", PREFILL_CASES)
def test_ple_chunks_tail_and_handoff_match_the_one_row_steps(rows_fake, tail: int, count: int, decoded: int) -> None:
    module = make_ple_module()
    constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", module.mesh_contract, rows=CHUNK_ROWS)
    torch.manual_seed(3000 + count)
    residual = torch.randn(decoded + count, 4, 2560).to(torch.bfloat16)
    pad_residual = torch.randn(1, 4, 2560).to(torch.bfloat16)
    tokens = tuple(int(v) for v in torch.randint(0, 200_000, (decoded + count,)))
    state_step = seed_ple_state(module)
    state_rows = seed_ple_state(module)  # seeds itself: the same nine slots on both sides

    def step(state, row: int):
        prepared = module.prepare_decode_input(torch.tensor([[tokens[row]]], dtype=torch.long), state)
        return cat_shards(
            module.forward_prepared(residual_rows(residual[row : row + 1]), prepared, state).residual_delta, 3
        )

    for row in range(decoded):
        step(state_step, row)
        step(state_rows, row)
    rows_state = module.allocate_rows_state(CHUNK_ROWS)
    rows_state.load_from_state(state_rows)  # the reverse seed (the reset path after decode steps)
    context = rows_state.token_context

    for row in range(decoded, decoded + count):
        step(state_step, row)
    for chunk, accepted in enumerate(_chunk_plan(count)):
        start = decoded + CHUNK_ROWS * chunk
        chunk_tokens = tokens[start : start + CHUNK_ROWS]
        rows = residual[start : start + CHUNK_ROWS]
        if len(chunk_tokens) < CHUNK_ROWS:
            rows = torch.cat([rows, pad_residual.expand(CHUNK_ROWS - len(chunk_tokens), 4, 2560)], dim=0)
            chunk_tokens = chunk_tokens + (PAD_TOKEN,) * (CHUNK_ROWS - len(chunk_tokens))
        # The driver looks the 32 rows up from the running context and keeps contexts[accepted + 1].
        _, contexts = module.host_rows(chunk_tokens, context)
        assert contexts[0] == context
        prepared = module.prepare_rows_input(chunk_tokens, rows_state)
        assert prepared.contexts == contexts
        module.forward_prepared_rows(residual_rows(rows), prepared, rows_state)
        module.commit_rows(rows_state, gdn_rows_selectors(accepted, constants))
        module.commit_rows_host(rows_state, prepared, accepted)
        context = contexts[accepted + 1]
        assert rows_state.token_context == context
    rows_state.store_to_state(state_rows)

    for index, (slot, expected_slot) in enumerate(zip(state_rows.conv, state_step.conv)):
        assert torch.equal(cat_shards(slot, 3), cat_shards(expected_slot, 3)), f"PLE slot {index}"
    if decoded + count == 0:
        assert state_rows.token_context is None and state_step.token_context is None
    else:
        assert torch.equal(state_rows.token_context, state_step.token_context)
        assert tuple(int(v) for v in state_step.token_context[0]) == context

    next_token = 17
    torch.manual_seed(4000 + count)
    next_residual = torch.randn(1, 4, 2560).to(torch.bfloat16)
    deltas = []
    for state in (state_step, state_rows):
        prepared = module.prepare_decode_input(torch.tensor([[next_token]], dtype=torch.long), state)
        deltas.append(
            cat_shards(module.forward_prepared(residual_rows(next_residual), prepared, state).residual_delta, 3)
        )
    assert torch.equal(deltas[0], deltas[1])


# --------------------------------------------------------------------------- QSA

QSA_CONTEXT = 128


@pytest.fixture
def fake_qsa(rows_fake, monkeypatch):
    monkeypatch.setattr(qsa_module, "ttnn", rows_fake.ttnn)
    module = object.__new__(qsa_module.Qwen38TTNNQSA)
    module.mesh_device = "mesh"
    module.mesh_contract = FakeContract()
    module.layer_index = 3
    module.allocated_context = QSA_CONTEXT
    module.allocated_compressed_blocks = QSA_CONTEXT // qsa_module.COMPRESS_RATIO
    module._live_generic_epochs = {1, 2}
    module.compute_config = "compute_config"
    module.projection_compute_config = "compute_config"
    return module


def _zeros(shape, layout=TILE, shard_dim=None) -> FakeTensor:
    return FakeTensor([torch.zeros(shape, dtype=torch.bfloat16) for _ in range(TP)], BF16, layout, shard_dim)


def _rows_valued(rows: int, width: int, values) -> FakeTensor:
    """``[1,1,rows,width]`` BF16 whose row i holds ``values[i]`` in every column (exact bf16 integers / halves)."""

    host = (
        torch.tensor([float(v) for v in values], dtype=torch.bfloat16).reshape(1, 1, rows, 1).expand(1, 1, rows, width)
    )
    return FakeTensor([host.clone() for _ in range(TP)], BF16, TILE)


@pytest.mark.parametrize("tail,count", PREFILL_CASES)
def test_qsa_handoff_fills_the_staging_and_the_ring_from_the_kept_rows(fake_qsa, tail: int, count: int) -> None:
    module = fake_qsa
    head_dim, index_dim = qsa_module.HEAD_DIM, qsa_module.INDEX_HEAD_DIM
    generic = qsa_module.Qwen38TTNNQSAGenericState(
        layer_index=3,
        epoch=1,
        packed_kv_cache=_zeros((1, 1, QSA_CONTEXT, 2 * head_dim), ROW_MAJOR, 1),
        compressed_index_cache=_zeros((1, 1, module.allocated_compressed_blocks + 32, index_dim)),
        kv_staging=_rows_valued(CHUNK_ROWS, 2 * head_dim, [-1.0] * CHUNK_ROWS),  # stale rows of an earlier block
        raw_key_ring=_rows_valued(CHUNK_ROWS, index_dim, [-1.0] * CHUNK_ROWS),
    )
    # The last chunk covered positions block_start .. block_start + 31: kept row i encodes position block_start + i.
    block_start = count - tail
    chunk = qsa_module.Qwen38TTNNQSAChunkState(
        layer_index=3,
        epoch=2,
        rows=CHUNK_ROWS,
        kept_kv=_rows_valued(CHUNK_ROWS, 2 * head_dim, [block_start + i for i in range(CHUNK_ROWS)]),
        kept_raw=_rows_valued(CHUNK_ROWS, index_dim, [block_start + i + 0.5 for i in range(CHUNK_ROWS)]),
    )
    select_host = qsa_module.chunk_handoff_ring_select_rows(count)
    assert select_host.shape == (1, 1, CHUNK_ROWS, CHUNK_ROWS) and select_host.dtype == torch.bfloat16
    assert int(select_host.sum()) == count % 4
    ring_select = FakeTensor([select_host.clone() for _ in range(TP)], BF16, TILE)

    module.handoff_chunk_state(generic, chunk, ring_select=ring_select, open_block=tail != 0)

    staging = generic.kv_staging.torch_shards()[0].float()
    if tail:
        # The open block [count & ~31, count): rows < r are positions block_start + i, the rest padded-row values.
        assert torch.equal(staging, chunk.kept_kv.torch_shards()[0].float())
        assert staging[0, 0, :tail, 0].tolist() == [float(block_start + i) for i in range(tail)]
    else:
        assert torch.count_nonzero(staging) == 0
    ring = generic.raw_key_ring.torch_shards()[0].float()
    open_rows = count % 4
    # Ring row j holds the raw key of position (count & ~3) + j; every other row is exactly zero.
    assert ring[0, 0, :open_rows, 0].tolist() == [float((count - open_rows) + j + 0.5) for j in range(open_rows)]
    assert torch.count_nonzero(ring[0, 0, open_rows:]) == 0
    assert torch.equal(ring[0, 0, :open_rows], ring[0, 0, :open_rows, :1].expand(open_rows, index_dim))
    for local in generic.kv_staging.torch_shards()[1:] + generic.raw_key_ring.torch_shards()[1:]:
        assert local.dtype == torch.bfloat16
    assert generic.kv_staging.alive and generic.raw_key_ring.alive and chunk.kept_kv.alive and chunk.kept_raw.alive


def test_ring_select_rejects_bad_counts(expect_error) -> None:
    for value in (-1, True, 2.0, "32"):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            qsa_module.chunk_handoff_ring_select_rows(value)
