# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2 step 5: the verify pass's device paths on the step-4 torch-backed ttnn (no device).

The state rule: after a partial accept and the next pass, every KV row and compressed block that pass can name
equals the sequential 1-row stream, and the raw history holds the three raw keys before the new position.
"""

from __future__ import annotations

import ast
import inspect
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.mtp_verify import accept_select
from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    FP32,
    HOST_TENSOR_CALLS,
    I32,
    LARGE_IDS,
    ROW_MAJOR,
    SMALL_IDS,
    TILE,
    TP,
    U32,
    VERIFY_BLOCKS,
    VERIFY_CONTEXT,
    FakeContract,
    FakeTensor,
    KVStream,
    accept_pattern_rows,
    bf16_randn,
    cat_shards,
    fp32_lanes,
    gdn_rows_selectors,
    make_qsa_module,
    ordered_calls,
    qsa_verify_constants,
    replicated,
    run_qsa_verify_pass,
    same_zero,
    seed_qsa_caches,
    sharded,
    source_functions,
    source_segment,
    u32_scalar,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as final_mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module

ROOT = Path(__file__).resolve().parents[1]
QSA_SOURCE = ROOT / "ttnn" / "qsa.py"
MTP_V2_SOURCE = ROOT / "ttnn" / "mtp_v2.py"
EMBEDDING_SOURCE = ROOT / "ttnn" / "embedding.py"
POSITIONS = (0, 3, 5, 29, 30, 31, 32, 33, 61, 96, 127)  # both KV-block cases and every P % 4
PASS_POSITIONS = (5, 29, 30, 31, 32, 60, 63, 100)


def _assert_caches_follow_the_stream(module, fake, stream: KVStream, state, visible_rows: int, label) -> None:
    """Every KV row a sparse row can name and every block complete below ``visible_rows`` equal the sequential
    stream; the rest of the compressed cache is the seed's finite filler or never-visible finite garbage."""

    for device in range(TP):
        cache = state.packed_kv_cache.torch_shards()[device][0, 0]
        assert torch.equal(cache[:visible_rows], stream.packed[device][:visible_rows]), (*label, device)
    for block in range(visible_rows // 4):
        actual = state.compressed_index_cache.torch_shards()[0][0, 0, block]
        assert torch.equal(actual, stream.compressed_block(module, fake, block)), (*label, block)
    assert torch.isfinite(state.compressed_index_cache.torch_shards()[0].float()).all()


# --------------------------------------------------------------------------- QSA verify inputs


@pytest.mark.parametrize("rows", (1, 4, 5, 6))
@pytest.mark.parametrize("position", POSITIONS)
def test_verify_inputs_match_the_torch_reference_at_every_position(verify_fake, position: int, rows: int) -> None:
    position_constants, _, verify_constants = qsa_verify_constants(rows)
    derived = qsa_module.derive_qsa_verify_inputs(u32_scalar(position), position_constants, verify_constants)
    expected = qsa_module.emulate_qsa_verify_inputs(position, rows=rows, allocated_compressed_blocks=VERIFY_BLOCKS)
    # The layer's own validation admits the derived inputs (the two completed block indices of a verify pass).
    module = make_qsa_module()
    module._validate_verify_inputs(derived)
    chunk = derived.chunk
    assert len(chunk.block_index_i32) == qsa_module.VERIFY_COMPLETED_BLOCKS == 2
    assert torch.equal(chunk.kv_block_start.torch_shards()[0], expected["chunk"]["kv_block_start"])
    assert [int(t.torch_shards()[0].reshape(-1)[0]) for t in chunk.block_index_i32] == [
        int(t.reshape(-1)[0]) for t in expected["chunk"]["block_index_i32"]
    ]
    assert all(t.dtype is I32 for t in chunk.block_index_i32)
    assert same_zero(chunk.indexer_neg_mask.torch_shards()[0], expected["chunk"]["indexer_neg_mask"])
    for name in ("row_keep_bits", "row_fill"):
        assert torch.equal(getattr(chunk, name).torch_shards()[0], expected["chunk"][name]), name
    for name in (
        "kv_block_start_next",
        "kv_read_indices",
        "stage_keep",
        "stage_a_select",
        "stage_b_select",
        "pool_select",
    ):
        actual = getattr(derived, name).torch_shards()[0]
        assert torch.equal(actual.float(), expected[name].float()), (name, position, rows)
    # Rows past R carry the geometry of row R - 1 (finite attention, no position past the pass).
    mask = expected["chunk"]["indexer_neg_mask"]
    assert all(torch.equal(mask[0, 0, row], mask[0, 0, rows - 1]) for row in range(rows, 32))
    # Every real row lands exactly once across the two staging tiles; rows past R land nowhere.
    stage_a = derived.stage_a_select.torch_shards()[0][0, 0].float()
    both = stage_a + derived.stage_b_select.torch_shards()[0][0, 0].float()
    assert both.sum(dim=0).tolist() == [1.0] * rows + [0.0] * (32 - rows)
    for row in range(rows):
        (landing,) = both[:, row].nonzero().reshape(-1).tolist()
        block_rows = stage_a[:, row].nonzero().numel()
        assert landing == (position % 32 + row) % 32 and block_rows == int(position % 32 + row < 32), (position, row)
    # The block at P & ~31 keeps exactly its rows below P % 32.
    assert derived.stage_keep.torch_shards()[0].reshape(-1).float().tolist() == [
        float(i < position % 32) for i in range(32)
    ]
    # The pool select reads the four positions of block P // 4 (row 0) and P // 4 + 1 (row 1) out of the window.
    pool = derived.pool_select.torch_shards()[0][0, 0].float()
    assert torch.count_nonzero(pool[2:]) == 0 and set(pool.unique().tolist()) <= {0.0, 0.25}
    for block_row, block in ((0, position // 4), (1, position // 4 + 1)):
        columns = pool[block_row].nonzero().reshape(-1).tolist()
        positions = [position - 3 + w if w < 32 else position + w - 32 for w in columns]
        wanted = [p for p in range(4 * block, 4 * block + 4) if p < position + rows]
        assert positions == wanted, (position, rows, block_row)
    if rows == 1:
        # The draft rows' single_row inputs: no next KV block index, no next-block select, P // 4 alone; every other
        # field is the two-block derivation's, bitwise.
        single = qsa_module.derive_qsa_verify_inputs(
            u32_scalar(position), position_constants, verify_constants, single_row=True
        )
        module._validate_verify_inputs(single)
        assert single.single_row and not derived.single_row
        assert single.kv_block_start_next is None and single.stage_b_select is None
        assert len(single.chunk.block_index_i32) == 1
        for name in ("kv_read_indices", "stage_keep", "stage_a_select", "pool_select"):
            assert torch.equal(getattr(single, name).torch_shards()[0], getattr(derived, name).torch_shards()[0]), name
        for name in ("kv_block_start", "indexer_neg_mask", "row_keep_bits", "row_fill"):
            assert torch.equal(getattr(single.chunk, name).torch_shards()[0], getattr(chunk, name).torch_shards()[0])
        assert torch.equal(
            single.chunk.block_index_i32[0].torch_shards()[0], chunk.block_index_i32[0].torch_shards()[0]
        )
        single.deallocate()
    derived.deallocate()


def test_verify_constants_are_exact_zero_one_selects(expect_error) -> None:
    for rows in (4, 5, 6):
        host = qsa_module.qsa_verify_constant_rows(rows, VERIFY_BLOCKS)
        assert host["row_index_blocks"][0, 0, :, 0].tolist() == [min(j, rows - 1) for j in range(32)]
        assert torch.equal(host["row_index_slots"][0, 0, :, 0], host["row_index_blocks"][0, 0, :, 0])
        for name, bias in (("stage_a_lanes", 32), ("stage_b_lanes", 64)):
            lanes = host[name][0, 0]
            assert torch.count_nonzero(lanes[:, rows:]) == 0
            for remainder in range(32):
                select = lanes == remainder + 32
                for row in range(rows):
                    landing = remainder + row - (bias - 32)
                    assert select[:, row].sum() == int(0 <= landing < 32), (name, remainder, row)
        stack = host["pool_select_stack"][0, 0]
        assert torch.count_nonzero(stack[4:]) == 0 and torch.equal(stack.to(torch.bfloat16).float(), stack)
    with expect_error(ValueError):  # allow-pytest.raises: the row bound is a contract
        qsa_module.qsa_verify_constant_rows(7, VERIFY_BLOCKS)
    assert (
        qsa_module.VERIFY_MAX_ROWS == 6 and qsa_module.VERIFY_COMPLETED_BLOCKS == 2 and qsa_module.RAW_HISTORY_ROWS == 3
    )


# --------------------------------------------------------------------------- the state rule after a partial accept


@pytest.mark.parametrize("rows", (4, 5, 6))
@pytest.mark.parametrize("position", PASS_POSITIONS)
def test_kv_and_compressed_writes_commit_the_accepted_prefix_for_every_accept(
    verify_fake, position: int, rows: int
) -> None:
    """Pass N at P writes rows P .. P + R - 1; the host accepts a drafts; pass N + 1 at P' = P + a + 1 writes its own
    rows.  Afterwards every KV row a sparse row of pass N + 1 can name (positions < P' + R) and every compressed
    block its indexer mask leaves visible equal the sequential stream, and the raw history holds P' - 3 .. P' - 1."""

    constants = qsa_verify_constants(rows)
    gdn_constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", FakeContract(), rows=rows)
    for accepted in range(rows):  # a = 0 .. k (row 0 is always committed)
        module = make_qsa_module()
        stream = KVStream(seed=100 + accepted)
        state, verify_state = module.allocate_generic_state(), module.allocate_verify_state()
        seed_qsa_caches(module, verify_fake, stream, state, verify_state, position)
        run_qsa_verify_pass(module, stream, state, verify_state, position, rows, constants)
        # Rows past the accepted prefix and the blocks they completed are garbage from here on: scramble the
        # stream there so the check below cannot pass by accident.
        next_position = position + accepted + 1
        torch.manual_seed(500 + accepted)
        stream.raw[next_position:] = bf16_randn(VERIFY_CONTEXT - next_position, 128)
        for device in range(TP):
            stream.packed[device][next_position:] = bf16_randn(VERIFY_CONTEXT - next_position, 512)
        module.commit_verify(verify_state, gdn_rows_selectors(accepted, gdn_constants))
        history = verify_state.raw_history.torch_shards()[0][0, 0]
        for back in range(1, 4):
            expected = stream.raw[next_position - back] if next_position >= back else torch.zeros(128).bfloat16()
            assert torch.equal(history[3 - back], expected), (position, rows, accepted, back)
        assert torch.count_nonzero(history[3:].float()) == 0
        run_qsa_verify_pass(module, stream, state, verify_state, next_position, rows, constants)
        # Blocks visible to the last row of pass N + 1: complete_blocks(P' + R - 1) = (P' + R) // 4.
        _assert_caches_follow_the_stream(
            module, verify_fake, stream, state, next_position + rows, (position, rows, accepted)
        )
        module.release_verify_state(verify_state)
        module.release_generic_state(state)


@pytest.mark.parametrize("position", PASS_POSITIONS)
def test_single_row_passes_skip_the_next_block_writes_and_keep_every_visible_row_and_block(verify_fake, position: int):
    """The draft rows: one row per pass on ``single_row`` inputs (no next KV block, no P // 4 + 1 block) at P, P + 1,
    P + 2, the previous row joining the raw history between passes.  Afterwards every KV row a sparse row can name and
    every completed block equal the sequential stream: the skipped writes were never read."""

    constants = qsa_verify_constants(1)
    module = make_qsa_module()
    stream = KVStream(seed=300 + position)
    state, verify_state = module.allocate_generic_state(), module.allocate_verify_state()
    seed_qsa_caches(module, verify_fake, stream, state, verify_state, position)
    advance = gdn_rows_selectors(0, gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", FakeContract(), rows=5))
    for step in range(3):
        if step:
            module.commit_verify(verify_state, advance)
        run_qsa_verify_pass(module, stream, state, verify_state, position + step, 1, constants, single_row=True)
    _assert_caches_follow_the_stream(module, verify_fake, stream, state, position + 3, (position,))
    history = verify_state.raw_rows.torch_shards()[0][0, 0]
    assert torch.equal(history[0], stream.raw[position + 2]) and torch.count_nonzero(history[1:].float()) == 0
    module.release_verify_state(verify_state)
    module.release_generic_state(state)


def test_raw_history_sync_from_the_ring_takes_the_last_three_positions(verify_fake) -> None:
    module = make_qsa_module()
    state, verify_state = module.allocate_generic_state(), module.allocate_verify_state()
    torch.manual_seed(61)
    ring = bf16_randn(4, 128)
    for target in state.raw_key_ring.locals:
        target[0, 0, :4] = ring
    for position in (0, 1, 2, 3, 5, 30, 31, 32, 33):
        module.sync_verify_raw_history_from_ring(state, verify_state, position=position)
        history = verify_state.raw_history.torch_shards()[0][0, 0]
        for back in range(1, 4):
            assert torch.equal(history[3 - back], ring[(position - back) % 4]), (position, back)
        assert torch.count_nonzero(history[3:].float()) == 0


def test_qsa_verify_state_is_two_distinct_zero_tiles_and_validates(expect_error, verify_fake) -> None:
    module = make_qsa_module()
    verify_state = module.allocate_verify_state()
    assert verify_state.raw_history.shape == (1, 1, 32, 128) and verify_state.raw_rows.shape == (1, 1, 32, 128)
    assert verify_state.raw_history is not verify_state.raw_rows
    module._validate_verify_state(verify_state)
    module.release_verify_state(verify_state)
    with expect_error(ValueError):  # allow-pytest.raises: a released epoch is refused
        module._validate_verify_state(verify_state)


# --------------------------------------------------------------------------- device accept


@pytest.mark.parametrize("drafts", (3, 4, 5))
def test_accept_rows_is_exact_for_every_pattern_and_large_ids(verify_fake, drafts: int) -> None:
    constants = mtp_v2.Qwen38TTNNAcceptConstants.build("mesh", FakeContract(), drafts=drafts)
    assert constants.sentinel_tail.shape == (1, 1, 1, 31 - drafts)
    garbage = list(LARGE_IDS) * 8  # the padding rows' argmaxes: real ids, never compared
    for pattern in itertools.product((0, 1), repeat=drafts):
        for ids in (SMALL_IDS, LARGE_IDS, SMALL_IDS + LARGE_IDS):
            targets, draft_ids, alignment = accept_pattern_rows(pattern, ids)
            reference = accept_select(
                torch.tensor(targets, dtype=torch.float32),
                torch.tensor(draft_ids, dtype=torch.float32),
                torch.tensor(alignment, dtype=torch.float32),
            )
            argmax_lanes = fp32_lanes(targets + garbage[: 32 - len(targets)])
            result = mtp_v2.accept_rows(argmax_lanes, fp32_lanes(draft_ids), constants)
            assert result.accepted_tile.dtype is FP32 and result.accepted_index.dtype is U32
            assert int(result.accepted_tile.torch_shards()[0].item()) == reference["accepted"], (pattern, ids)
            assert int(result.accepted_index.torch_shards()[0].item()) == reference["accepted"]
            assert int(result.next_token.torch_shards()[0].item()) == reference["next_token_gather"], (pattern, ids)
            assert result.next_token.layout == ROW_MAJOR and result.next_token.shape == (1, 1, 1, 1)
            # The alignment argmaxes' lane a is the same gather the body runs for d_1'.
            first_draft = verify_fake.gather(
                fp32_lanes(alignment + garbage[: 32 - len(alignment)]), 3, result.accepted_index
            )
            assert int(first_draft.torch_shards()[0].item()) == reference["first_draft_gather"]


def test_select_residual_row_is_exact_for_every_accept(verify_fake) -> None:
    constants = mtp_v2.Qwen38TTNNAcceptConstants.build("mesh", FakeContract(), drafts=4)
    torch.manual_seed(71)
    residual = FakeTensor([bf16_randn(1, 4, 32, 640, scale=3.0) for _ in range(TP)], BF16, TILE, 3)
    output = FakeTensor([torch.zeros(1, 4, 1, 640, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE, 3)
    for accepted in range(5):
        accepted_tile = replicated(torch.full((1, 1, 1, 1), float(accepted)), FP32)
        mtp_v2.select_residual_row(residual, accepted_tile, constants, output=output)
        for device in range(TP):
            expected = residual.torch_shards()[device][:, :, accepted : accepted + 1]
            assert torch.equal(output.torch_shards()[device].view(torch.int16), expected.view(torch.int16)), accepted


# --------------------------------------------------------------------------- rows forms vs 1-row forms


def _lm_head(verify_fake) -> embedding_module.Qwen38TTNNLMHead:
    head = object.__new__(embedding_module.Qwen38TTNNLMHead)
    head.mesh_device = "mesh"
    head.mesh_contract = FakeContract()
    head.collective_topology = verify_fake.Topology.Linear
    local = embedding_module.LOCAL_VOCAB_SIZE
    owners = torch.arange(TP, dtype=torch.float32).reshape(1, 1, 1, TP)
    head.weights = SimpleNamespace(
        vocab_ranges=tuple((d * local, (d + 1) * local) for d in range(TP)),
        token_row=SimpleNamespace(
            owner_tie_break=replicated(owners * embedding_module.GREEDY_TIE_BREAK_EPS, FP32, ROW_MAJOR),
            lm_head_vocab_starts=replicated(owners * local, FP32, ROW_MAJOR),
            unit_column=replicated(torch.eye(1, 32), FP32, ROW_MAJOR),
        ),
        replicated_anchor=replicated(torch.zeros(1, 1, 1, 1), BF16),
    )
    return head


@pytest.mark.parametrize("rows", (5, 1))
def test_greedy_candidates_by_gather_are_the_grid_reduce_candidates_bitwise(verify_fake, rows: int) -> None:
    """The gathered row maximum is the row's own argmax element: the padded grid reduce's value, bitwise."""

    head = _lm_head(verify_fake)
    local = embedding_module.LOCAL_VOCAB_SIZE
    torch.manual_seed(83)
    shards = [(torch.randn(1, 1, rows, local) * 4).to(torch.bfloat16) for _ in range(TP)]
    shards[0][0, 0, 0, 7] = shards[0][0, 0, 0].max()  # a tie inside a row: both forms report the first index's value

    def logits() -> embedding_module.Qwen38ShardedLogits:
        return embedding_module.Qwen38ShardedLogits(
            tensor=FakeTensor([shard.clone() for shard in shards], BF16, TILE, 3),
            vocab_ranges=head.weights.vocab_ranges,
            global_shape=(1, 1, rows, embedding_module.VOCAB_SIZE),
        )

    reduced = head.greedy_candidates(logits())
    gathered = head.greedy_candidates(logits(), values_by_gather=True)
    assert gathered.rows == reduced.rows == rows
    assert gathered.local_values.shape == (1, 1, rows, 1) and gathered.local_values.layout == TILE
    assert gathered.local_values.dtype is BF16 and gathered.local_indices.shape == (1, 1, rows)
    for device in range(TP):
        assert torch.equal(gathered.local_indices.torch_shards()[device], reduced.local_indices.torch_shards()[device])
        assert torch.equal(
            gathered.local_values.torch_shards()[device].view(torch.int16),
            reduced.local_values.torch_shards()[device].view(torch.int16),
        )
        assert torch.equal(
            reduced.local_values.torch_shards()[device].reshape(rows), shards[device][0, 0].max(dim=-1).values
        )


@pytest.mark.parametrize("rows", (32, 5, 1))
def test_resolve_greedy_rows_on_device_matches_the_per_row_owner_rule_with_ties(
    expect_error, verify_fake, rows: int
) -> None:
    head = _lm_head(verify_fake)
    local = embedding_module.LOCAL_VOCAB_SIZE
    torch.manual_seed(81)
    for trial in range(20):
        values = (torch.randn(TP, rows) * 8).to(torch.bfloat16)
        values[:, 0] = values[0, 0]  # all owners tie on row 0
        if rows > 2:
            values[2, 1] = values[1, 1]  # a two-way tie on row 1
        indices = torch.randint(0, local, (TP, rows))
        if rows > 2:
            indices[1, 2] = local - 1  # the last id of a shard (95859 < 62080 * 2 lives on shard 1)
        candidates = embedding_module.Qwen38GreedyCandidates(
            local_indices=FakeTensor([indices[d].reshape(1, 1, rows).clone() for d in range(TP)], U32, ROW_MAJOR),
            local_values=FakeTensor([values[d].reshape(1, 1, rows, 1).clone() for d in range(TP)], BF16, TILE),
            rows=rows,
            vocab_ranges=head.weights.vocab_ranges,
        )
        lanes = head.resolve_greedy_rows_on_device(candidates)
        assert lanes.shape == (1, 1, 1, rows) and lanes.dtype is FP32 and lanes.layout == ROW_MAJOR
        owners = torch.argmax(values.float(), dim=0)  # first owner on ties: resolve_greedy's rule
        expected = torch.tensor([d * local + int(indices[d, row]) for row, d in enumerate(owners.tolist())])
        assert lanes.torch_shards()[0].reshape(-1).long().tolist() == expected.tolist(), trial
    # The 1-row resolve is untouched and still refuses rows; the rows resolve refuses more than a tile.
    if rows != 1:
        with expect_error(TypeError):  # allow-pytest.raises: the 1-row contract
            head.resolve_greedy_on_device(candidates)
    with expect_error(TypeError):  # allow-pytest.raises: the rows contract
        head.resolve_greedy_rows_on_device(
            embedding_module.Qwen38GreedyCandidates(
                candidates.local_indices, candidates.local_values, 33, head.weights.vocab_ranges
            )
        )


@pytest.mark.parametrize("flat_views", (False, True))
def test_final_mixer_rows_are_bitwise_the_one_row_mixer_per_row(verify_fake, flat_views: bool) -> None:
    mixer = object.__new__(final_mixer_module.Qwen38TTNNFinalMixer)
    mixer.mesh_device = "mesh"
    mixer.mesh_contract = FakeContract()
    mixer.collective_topology = "linear"
    mixer.compute_config = "compute_config"
    mixer.weight_compute_config = "compute_config"
    # The fake reports every tensor in DRAM, so the activation-shard configs the mixer checks are that config.
    mixer.down_act_memory_config, mixer.down_program_config = "DRAM_MEMORY_CONFIG", "down_cfg"
    mixer.up_act_memory_config, mixer.up_program_config = "DRAM_MEMORY_CONFIG", "up_cfg"
    torch.manual_seed(91)
    norm = ((1.0 + 0.1 * torch.randn(1, 4, 1, 2560)) / 4).float()
    down = bf16_randn(1, 1, 4 * 2560, 320, scale=2560**-0.5)
    up = bf16_randn(1, 1, 320, 4 * 2560, scale=320**-0.5)
    mixer.weights = SimpleNamespace(
        norm_scale=sharded(norm, dtype=FP32),
        down=sharded(down, dim=2),
        up=sharded(up),
        replicated_anchor=replicated(torch.zeros(1, 1, 1, 1), BF16),
        namespace="backbone",
    )
    mixer.norm_scale_flat = verify_fake.experimental.view(mixer.weights.norm_scale, (1, 1, 1, 2560))
    torch.manual_seed(92)
    residual = bf16_randn(1, 4, 32, 2560)
    rows = mixer.rows(sharded(residual), flat_views=flat_views)
    assert rows.shape == (1, 1, 32, 640) and rows.dtype is BF16
    for row in range(32):
        one = mixer(sharded(residual[:, :, row : row + 1]))
        assert one.shape == (1, 1, 1, 640)
        assert torch.equal(
            cat_shards(rows, 3)[:, :, row : row + 1].view(torch.int16), cat_shards(one, 3).view(torch.int16)
        ), row


# --------------------------------------------------------------------------- host images, readback, rows admission


def test_verify_token_rows_pad_with_the_zero_embedding_token_and_readback_parses(expect_error, verify_fake) -> None:
    host = embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows([17, 15, 16, 95859, 20])
    assert host.shape == (1, 1, 1, 32) and host[..., :5].tolist() == [[[[17.0, 15.0, 16.0, 95859.0, 20.0]]]]
    assert (host[..., 5:] == embedding_module.ZERO_EMBEDDING_TOKEN).all()
    assert embedding_module.ZERO_EMBEDDING_TOKEN == -1
    # -1 localizes below every shard's range: clamp(-1 - (d * LOCAL - 1), 0, LOCAL + 1) = 0, the zero sentinel row.
    local = embedding_module.LOCAL_VOCAB_SIZE
    assert all(max(0, min(-1 - (d * local - 1), local + 1)) == 0 for d in range(TP))
    with expect_error(ValueError):  # allow-pytest.raises: exact ids only
        embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows([17, -1])
    lanes = [2.0, 248319.0, -1.0] + [float(v) for v in range(32)]
    output = mtp_v2.Qwen38TTNNVerifyOutput(replicated(torch.tensor(lanes).reshape(1, 1, 1, 35), FP32, ROW_MAJOR))
    readback = mtp_v2.read_verify_output(output, rows=5)
    assert (readback.accepted, readback.next_token, readback.first_draft) == (2, 248319, None)
    assert readback.argmaxes == (0, 1, 2, 3, 4) and mtp_v2.READBACK_WIDTH == 35


def test_moe_row_admission_and_flags_are_untouched(expect_error) -> None:
    assert moe_module.SUPPORTED_ROWS == (*range(1, 33), 128)
    assert moe_module.ROWS5_HARDWARE_PROVEN is True and moe_module.ROWS32_HARDWARE_PROVEN is True
    # k = 5 runs the 6-row form (ROWS6_HARDWARE_PROVEN)
    assert [mtp_v2.moe_rows_for(k + 1) for k in mtp_v2.SUPPORTED_DRAFTS] == [5, 5, 6]
    assert mtp_v2.SUPPORTED_DRAFTS == (3, 4, 5) and mtp_v2.DEFAULT_DRAFTS == 4
    # The 32-row form holds every k up to 31; the QSA verify path caps a pass at VERIFY_MAX_ROWS = 6 rows, so k = 5 is
    # the largest draft count the verify runs.
    assert [mtp_v2.moe_rows_for(k + 1) for k in range(5, 9)] == [6, 32, 32, 32] and mtp_v2.moe_rows_for(32) == 32
    assert max(mtp_v2.SUPPORTED_DRAFTS) + 1 == qsa_module.VERIFY_MAX_ROWS
    with expect_error(ValueError):  # allow-pytest.raises: the row bound is a contract
        mtp_v2.moe_rows_for(33)
    # An explicit override (the runner's argument) is admitted for the instance it constructs, nothing else.
    assert [mtp_v2.resolve_moe_rows(k + 1, None) for k in mtp_v2.SUPPORTED_DRAFTS] == [5, 5, 6]
    assert mtp_v2.resolve_moe_rows(6, 6) == 6 and mtp_v2.resolve_moe_rows(4, 5) == 5
    source = inspect.getsource(mtp_v2._allocate_layer_verify_state)
    assert "admitted_rows=SUPPORTED_ROWS if moe_rows in SUPPORTED_ROWS else (moe_rows,)," in source
    # The GDN state re-anchor is per layer state and reaches both commit sites as the commit_rows keyword.
    for name in ("_forward_layer_verify", "forward_commit"):
        assert "step_committed_rows=layer_verify.gdn_step_anchor," in inspect.getsource(getattr(mtp_v2, name)), name
    assert "step_committed_rows" not in inspect.getsource(mtp_v2.forward_draft)


# --------------------------------------------------------------------------- source pins


def test_verify_body_creates_no_host_tensors_and_updates_the_position_last() -> None:
    functions = source_functions(MTP_V2_SOURCE)
    body = (
        "forward_verify",
        "_forward_layer_verify",
        "_forward_alignment",
        "_embed_rows",
        "_resolve_rows",
        "accept_rows",
        "select_residual_row",
    )
    for name in body:
        for called in ordered_calls(functions[name]):
            assert not called.startswith(HOST_TENSOR_CALLS), (name, called)
            assert "synchronize" not in called and ".item" not in called, (name, called)
    # The position update is the body's last device op: after it only the deallocations and the return.
    source = source_segment(MTP_V2_SOURCE, functions["forward_verify"])
    tail = source[source.index("ttnn.copy(advanced_next, state.position.scalar)") :]
    assert "ttnn." not in tail.replace("ttnn.copy(advanced_next", "").replace("_deallocate", "")
    assert source.index("ttnn.copy(accept.accepted_tile, verify.accepted)") < source.index(
        "ttnn.add(state.position.scalar"
    )
    # Selects by gather (ids) and by one-hot multiply + sum (bf16 rows); never a reduce over ids.
    accept = source_segment(MTP_V2_SOURCE, functions["accept_rows"])
    assert "ttnn.gather(argmax_lanes, 3, accepted_index" in accept and "ttnn.sum(prefix" in accept
    assert "ttnn.matmul(flags_tile, constants.prefix_upper" in accept
    # The heads run the LM head, the candidates and the resolve on the R real rows (a row-0 slice copy), then pad the
    # lanes to 32 with the zero-embedding sentinel: the accept flags past lane k stay 0 (lane k's draft is the
    # sentinel).
    resolve = source_segment(MTP_V2_SOURCE, functions["_resolve_rows"])
    assert "ttnn.slice(hidden_rows, (0, 0, 0, 0), (1, 1, rows, LOCAL_HIDDEN_SIZE)" in resolve
    assert "ttnn.concat([lanes, sentinel_tail], dim=3" in resolve and "if rows == CHUNK_ROWS:" in resolve
    assert "rows=verify.rows, sentinel_tail=verify.accept_constants.sentinel_tail" in source
    alignment = source_segment(MTP_V2_SOURCE, functions["_forward_alignment"])
    assert "rows=verify.rows, sentinel_tail=constants.sentinel_tail" in alignment
    assert "ttnn.gather(lanes, 3, accept.accepted_index" in alignment and "select_residual_row(" in alignment


def test_every_layer_commits_the_previous_pass_before_its_new_rows() -> None:
    functions = source_functions(MTP_V2_SOURCE)
    calls = ordered_calls(functions["_forward_layer_verify"])
    order = {name: calls.index(name) for name in calls}
    assert order["layer.ple.commit_rows"] < order["layer.ple.inject_rows"]
    assert order["layer.attention.commit_rows"] < order["layer.attention.forward_rows"]
    assert order["layer.attention.commit_verify"] < order["layer.attention.forward_verify_generic"]
    assert (
        order["layer.attention_gr.read_rows"]
        < order["layer.attention.forward_rows"]
        < order["layer.attention_gr.write_rows"]
    )
    assert order["layer.mlp_gr.read_rows"] < order["layer_verify.moe.forward"] < order["layer.mlp_gr.write_rows"]
    # The pad results are views: the owner is deallocated, never the pad result.
    source = source_segment(MTP_V2_SOURCE, functions["_forward_layer_verify"])
    assert "_deallocate(attention_owner, residual_owner" in source and "_deallocate(mlp_result.hidden_sharded" in source
    assert "_deallocate(attention_hidden" not in source and "_deallocate(moe_hidden" not in source
    # The verify body of the model runs the 48 layers then the alignment, and forward_verify is the only writer of P.
    model_calls = ordered_calls(functions["forward_verify"])
    assert model_calls.index("gdn_module.build_rows_selectors") < model_calls.index(
        "qsa_module.derive_qsa_verify_inputs"
    )
    assert (
        model_calls.index("_embed_rows")
        < model_calls.index("_forward_layer_verify")
        < model_calls.index("model.final_mixer.rows")
    )
    assert model_calls.index("accept_rows") < model_calls.index("_forward_alignment")


def test_qsa_verify_methods_are_the_chunk_ops_plus_two_block_writes_without_host_tensors() -> None:
    functions = source_functions(QSA_SOURCE)
    for name in (
        "forward_verify_generic",
        "_write_compressed_index_verify",
        "_write_packed_kv_verify",
        "_write_kv_block_verify",
        "_raw_window_verify",
        "commit_verify",
    ):
        for called in ordered_calls(functions[name]):
            assert not called.startswith(HOST_TENSOR_CALLS), (name, called)

    def walk(name: str) -> list[str]:
        return [c.removeprefix("self.") for c in ordered_calls(functions[name]) if c.startswith("self._")]

    assert walk("forward_verify_generic") == [
        "_validate_generic_state",
        "_validate_verify_state",
        "_validate_rope_rows",
        "_validate_rope_rows",
        "_validate_verify_inputs",
        "_all_gather_hidden_rows",
        "_index_projection_rows",
        "_write_compressed_index_verify",
        "_score_blocks_chunk",
        "_materialize_rows_chunk",
        # the qsa_rows family: the fused main tail of the verify rows, then the attention on its query; the chain's
        # projection + KV write + attention stay as the fallback branch
        "_main_tail_rows_step",
        "_sparse_value_attention_rows",
        "_main_projection_rows",
        "_write_packed_kv_verify",
        "_sparse_value_attention_rows",
        "_project_output_rows",
    ]
    # The shared row helpers take the chunk constants (32 or 128 rows); the fake ttnn never reaches the verify
    # body, so its calls are pinned to the helpers' signatures here.
    for call in (n for n in ast.walk(functions["forward_verify_generic"]) if isinstance(n, ast.Call)):
        if not ast.unparse(call.func).startswith("self._"):
            continue
        arguments = functions[call.func.attr].args
        positional = [a.arg for a in arguments.args[1:]]
        keywords = {k.arg for k in call.keywords}
        required_keywords = {a.arg for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults) if d is None}
        label = (call.func.attr, call.lineno)
        assert len(call.args) <= len(positional), label
        assert len(call.args) + len(keywords & set(positional)) >= len(positional) - len(arguments.defaults), label
        assert required_keywords <= keywords, label
    compressed = source_segment(QSA_SOURCE, functions["_write_compressed_index_verify"])
    assert compressed.count("ttnn.experimental.paged_update_cache(") == 1
    assert "enumerate(verify.chunk.block_index_i32)" in compressed
    assert "ttnn.copy(raw_key, verify_state.raw_rows)" in compressed and "ttnn.sum" not in compressed
    kv = source_segment(QSA_SOURCE, functions["_write_packed_kv_verify"])
    assert kv.count("self._write_kv_block_verify(") == 2 and "ttnn.embedding(" in kv and "kv_staging" not in kv
    assert "verify.stage_a_select" in kv and "verify.stage_b_select" in kv and "verify.stage_keep" in kv
    assert source_segment(QSA_SOURCE, functions["_write_kv_block_verify"]).count("update_padded_kv_cache(") == 1
    commit = source_segment(QSA_SOURCE, functions["commit_verify"])
    assert "optional_output_tensor=target.raw_history" in commit and "ttnn.slice" not in commit
    assert "target = verify_state if target is None else target" in commit
    derive = source_segment(QSA_SOURCE, functions["derive_qsa_verify_inputs"])
    for forbidden in ("to_torch", ".item()", "position %", "position //", "synchronize", "for row in"):
        assert forbidden not in derive, forbidden
    # The verify pass completes two compressed blocks at most: only their indices are derived (the chunk's eight
    # stay the prefill path's default).
    assert "completed_blocks=1 if single_row else VERIFY_COMPLETED_BLOCKS" in derive
    chunk_derive = source_segment(QSA_SOURCE, functions["derive_qsa_chunk_inputs"])
    assert "completed_blocks: int | None = None" in chunk_derive
    assert "for block in range(completed_blocks):" in chunk_derive
    # The 1-row generic and the chunk bodies are untouched (their walks are the pinned ones).
    generic = ["_main_projection", "_write_packed_kv_generic", "_sparse_value_attention", "_project_output"]
    assert walk("forward_decode_generic")[-4:] == generic
    chunk = ["_main_projection_rows", "_write_packed_kv_chunk", "_sparse_value_attention_rows", "_project_output_rows"]
    assert walk("forward_chunk_generic")[-4:] == chunk


def test_rows_resolve_and_host_row_images_keep_the_one_row_paths() -> None:
    functions = source_functions(EMBEDDING_SOURCE)
    rows = source_segment(EMBEDDING_SOURCE, functions["resolve_greedy_rows_on_device"])
    assert rows.count("ttnn.gather(") == 1 and rows.count("ttnn.argmax(") == 1
    assert "ttnn.transpose(token_column, 2, 3" in rows
    assert "ttnn.sum" not in rows and "ttnn.matmul" not in rows  # no id through a reduce or the FPU
    one = source_segment(EMBEDDING_SOURCE, functions["resolve_greedy_on_device"])
    assert "candidates.rows != 1" in one and "ttnn.gather(candidate_tokens, 3, owner" in one


def test_alignment_rows_consume_the_target_predictions_so_row_a_sees_the_committed_token() -> None:
    """Alignment row j takes argmax_j, the token at P + j + 1 (the accepted draft d_{j+1} for j < a, t' for j = a).

    The shifted verify tokens ``[d_1 .. d_k, t']`` would hand row a the rejected draft ``d_{a+1}`` whenever a < k.
    """

    for k in mtp_v2.SUPPORTED_DRAFTS:
        for pattern in itertools.product((0, 1), repeat=k):
            targets, draft_ids, _ = accept_pattern_rows(pattern, SMALL_IDS + LARGE_IDS)
            reference = accept_select(
                torch.tensor(targets, dtype=torch.float32),
                torch.tensor(draft_ids, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32),
            )
            a = reference["accepted"]
            body_tokens = targets[: k + 1]  # what _forward_alignment embeds: argmax lanes 0 .. k
            assert body_tokens[a] == reference["next_token_gather"] == targets[a]
            assert body_tokens[:a] == draft_ids[:a]  # the accepted drafts are the targets by definition
            shifted = [*draft_ids, reference["next_token_gather"]]
            assert (shifted[a] == body_tokens[a]) == (a == k), (k, pattern)
    functions = source_functions(MTP_V2_SOURCE)
    alignment = source_segment(MTP_V2_SOURCE, functions["_forward_alignment"])
    assert "ttnn.slice(argmax_lanes, (0, 0, 0, 0), (1, 1, 1, verify.rows)" in alignment
    assert "ttnn.concat([predicted, constants.sentinel_tail], dim=3" in alignment
    assert "accept.next_token" not in alignment and "verify.token_row" not in alignment
    body = source_segment(MTP_V2_SOURCE, functions["forward_verify"])
    call = body[body.index("_forward_alignment(") :]
    assert body.index("accept_rows(argmax_lanes") < body.index("_forward_alignment(")
    assert "argmax_lanes," in call[: call.index("rope_rows=")]
    assert body.index("_forward_alignment(") < body.index("_deallocate(argmax_lanes)")
