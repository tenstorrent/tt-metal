# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The 128-row prefill chunk form against four consecutive 32-row chunks on the same rows, without a device.

The fakes are row-serial, so any drift from the four 32-row chunks is a row-mixing bug, not arithmetic. The GR and
QSA 128-row forms are checked with their 32-row forms (test_gr_rows_no_device, test_ttnn_qsa_chunk_no_device).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.prefill import prefill_next_context
from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    FP32,
    TILE,
    TP,
    FakeTensor,
    bits_equal,
    cat_shards,
    clone_gdn_state,
    deallocate,
    device_gdn_weights,
    gdn_oracle_weights,
    hidden_sharded,
    make_gdn_module,
    make_ple_module,
    residual_rows,
    seed_gdn_state,
    seed_ple_state,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROW_COUNTS, CHUNK_ROWS, LONG_CHUNK_ROWS

TILES = LONG_CHUNK_ROWS // CHUNK_ROWS


def _full_accept(constants):
    accepted = FakeTensor([torch.full((1, 1, 1, 1), float(CHUNK_ROWS - 1)) for _ in range(TP)], FP32, TILE)
    return gdn_module.build_rows_selectors(accepted, constants)


# --------------------------------------------------------------------------- contracts


def test_long_chunk_contract(expect_error) -> None:
    assert (CHUNK_ROWS, LONG_CHUNK_ROWS, CHUNK_ROW_COUNTS) == (32, 128, (32, 128))
    assert contracts_module.chunk_row_tiles(32) == 1 and contracts_module.chunk_row_tiles(128) == 4
    for rows in (0, 31, 64, 96, 192):  # 256 .. 2048 in steps of 128 are the slab row counts
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            contracts_module.chunk_row_tiles(rows)
    assert gdn_module.rows_tile_count(1) == gdn_module.rows_tile_count(32) == 32
    assert gdn_module.rows_tile_count(128) == 128 and gdn_module.LONG_CONV_WINDOW_TILE_ROWS == 160
    assert moe_module.SUPPORTED_ROWS == (*range(1, 33), 128) and moe_module.LONG_PREFILL_CHUNK_ROWS == 128
    long_moe = moe_module.Qwen38TTNNMoERowContract(128)
    assert long_moe.row_tiles == 4 and moe_module.Qwen38TTNNMoERowContract(32).row_tiles == 1
    assert long_moe.local_combine == (10, 128, 2560)
    assert moe_module.Qwen38TTNNRouting.__dataclass_fields__["tiles"].default is None
    # moe_compute admits 32 x d x output_height_shard_dim tokens; d = 4 on an 8-bank part, 1 on a 7-bank part.
    for ring, dims in ((8, [1, 1, 1, 1]), (7, [1, 1, 1, 4])):
        got = [moe_module.moe_compute_output_height_shard_dim(r, matmul_ring_size=ring) for r in (1, 5, 32, 128)]
        assert got == dims, ring
    assert qsa_module.chunk_blocks(32) == 8 and qsa_module.chunk_blocks(128) == 32


def test_gdn_long_select_tiles_pick_the_window_rows_and_the_last_three_new_rows() -> None:
    tiles = gdn_module.rows_window_select_tiles(LONG_CHUNK_ROWS)
    taps, full = tiles["conv_taps"], tiles["history_select_full"]
    assert taps.shape == (3, 128, 160) and full.shape == (32, 160)
    assert (taps != 0).sum(dim=-1).eq(1).all()
    for tap in range(3):
        for row in range(128):
            logical = tap + row  # window row t + j: history rows 0..2, then buffer row 32 + (m - 3)
            assert taps[tap, row, logical if logical < 3 else 32 + logical - 3] == 1.0
    assert torch.count_nonzero(full[3:]) == 0
    for index in range(3):
        assert full[index].nonzero().reshape(-1).tolist() == [32 + 125 + index]  # qkv rows 125..127
    short = gdn_module.rows_window_select_tiles(CHUNK_ROWS)
    assert short["conv_taps"].shape == (3, 32, 64) and short["history_select_full"].shape == (32, 64)
    for index in range(3):
        assert short["history_select_full"][index].nonzero().reshape(-1).tolist() == [32 + 29 + index]


# --------------------------------------------------------------------------- GDN, PLE: 128 rows vs four 32-row chunks


def test_gdn_long_chunk_matches_four_chunks_and_commits_every_row(expect_error, rows_fake) -> None:
    module = make_gdn_module(device_gdn_weights(gdn_oracle_weights()))
    torch.manual_seed(128)
    hidden = torch.randn(1, LONG_CHUNK_ROWS, 2560).to(torch.bfloat16)
    state_four = seed_gdn_state(module, 41)
    state_long = clone_gdn_state(module, state_four)

    four_constants = module.allocate_rows_constants(CHUNK_ROWS)
    four_rows = module.allocate_rows_state(four_constants)
    module.sync_rows_history_from_state(state_four, four_rows)
    four_outputs = []
    for tile in range(TILES):
        rows = hidden[:, CHUNK_ROWS * tile : CHUNK_ROWS * (tile + 1)]
        four = module.forward_rows(hidden_sharded(rows), state_four, four_rows)
        four_outputs.append(cat_shards(four.hidden_rows, 3).clone())
        module.commit_rows(state_four, four_rows, _full_accept(four_constants))
        deallocate(four.final_state)
    four_output = torch.cat(four_outputs, dim=2)
    four_kernel_calls = len(rows_fake.chunk.calls)

    short_constants = module.allocate_rows_constants(CHUNK_ROWS)
    short_state = module.allocate_rows_state(short_constants)
    module.sync_rows_history_from_state(state_long, short_state)  # the 32-row state seeds the shared history
    for rows in (0, 33, 64, 192):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            module.allocate_rows_constants(rows)
    constants = module.allocate_rows_constants(LONG_CHUNK_ROWS)
    assert constants.tile_rows == 128 and constants.window_rows == 160
    assert torch.count_nonzero(constants.row_mask_fp32.torch_shards()[0]) == 128
    rows_state = module.allocate_rows_state(constants, history=short_state.history)
    assert rows_state.history is short_state.history and not rows_state.owns_history
    assert rows_state.qkv.shape == (1, 1, 128, 2560) and rows_state.q.shape == (1, 128, 12, 128)
    with expect_error(RuntimeError):  # allow-pytest.raises: a 32-row hidden tile is not the 128-row input
        module.forward_rows(hidden_sharded(torch.zeros(1, 32, 2560).to(torch.bfloat16)), state_long, rows_state)
    result = module.forward_rows(hidden_sharded(hidden), state_long, rows_state)
    assert len(rows_fake.chunk.calls) == four_kernel_calls + 1 and rows_fake.chunk.calls[-1]["rows"] == 128
    assert rows_fake.chunk.calls[-1]["chunk_size"] == 32 and rows_fake.chunk.calls[-1]["flat_v"]
    long_output = cat_shards(result.hidden_rows, 3)
    assert long_output.shape == (1, 1, 128, 2560) and result.hidden_rows is rows_state.output
    with expect_error(RuntimeError):  # allow-pytest.raises: the full commit needs the pass's final state
        module.commit_rows_full(state_long, rows_state, FakeTensor([torch.zeros(1, 12, 1, 128)] * TP, FP32, TILE))
    module.commit_rows_full(state_long, rows_state, result.final_state)
    assert len(rows_fake.chunk.calls) == four_kernel_calls + 1  # no re-run at the commit

    # Everything around the recurrence is row-serial on the fake: bitwise. The kernel's NC = 4 carry is the same
    # sequential fp32 carry as four calls.
    committed, expected = cat_shards(state_long.recurrent, 1), cat_shards(state_four.recurrent, 1)
    assert bits_equal(long_output, four_output), (long_output.float() - four_output.float()).abs().max().item()
    assert torch.equal(committed, expected), (committed - expected).abs().max().item()
    # The FIR history after 128 rows is the last three new rows (rows 125..127), bitwise the four chunks'.
    history = cat_shards(rows_state.history, 3)
    assert bits_equal(history, cat_shards(four_rows.history, 3))
    assert torch.count_nonzero(history[:, :, 3:].float()) == 0
    # The shared history carries into a following 32-row chunk without a sync.
    torch.manual_seed(129)
    tail = torch.randn(1, CHUNK_ROWS, 2560).to(torch.bfloat16)
    next_four = module.forward_rows(hidden_sharded(tail), state_four, four_rows)
    next_long = module.forward_rows(hidden_sharded(tail), state_long, short_state)
    assert bits_equal(cat_shards(next_long.hidden_rows, 3), cat_shards(next_four.hidden_rows, 3))
    for state in (rows_state, short_state):
        state.deallocate()
    assert short_state.history.alive is False and rows_state.output.alive is False


def test_ple_long_chunk_matches_four_chunks_and_commits_the_last_nine_rows(expect_error, rows_fake) -> None:
    module = make_ple_module()
    torch.manual_seed(130)
    residual = torch.randn(LONG_CHUNK_ROWS, 4, 2560).to(torch.bfloat16)
    tokens = tuple(int(v) for v in torch.randint(0, 200_000, (LONG_CHUNK_ROWS,)))
    constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", module.mesh_contract, rows=CHUNK_ROWS)

    four_state = module.allocate_rows_state(CHUNK_ROWS)
    four_state.load_from_state(seed_ple_state(module))
    deltas = []
    for tile in range(TILES):
        rows = slice(CHUNK_ROWS * tile, CHUNK_ROWS * (tile + 1))
        prepared = module.prepare_rows_input(tokens[rows], four_state)
        result = module.forward_prepared_rows(residual_rows(residual[rows]), prepared, four_state)
        deltas.append(cat_shards(result.residual_delta, 3))
        module.commit_rows(four_state, _full_accept(constants))
        module.commit_rows_host(four_state, prepared, CHUNK_ROWS - 1)

    short_state = module.allocate_rows_state(CHUNK_ROWS)
    short_state.load_from_state(seed_ple_state(module))
    long_state = module.allocate_rows_state(LONG_CHUNK_ROWS, history=short_state.history)
    assert long_state.history is short_state.history and not long_state.owns_history and long_state.rows == 128
    prepared = module.prepare_rows_input(tokens, long_state)
    assert prepared.embedding_rows.shape == (1, 1, 128, 640) and len(prepared.contexts) == 129
    result = module.forward_prepared_rows(residual_rows(residual), prepared, long_state)
    module.commit_rows_full(long_state)
    long_state.token_context = prepared.contexts[-1]
    delta = cat_shards(result.residual_delta, 3)
    assert delta.shape == (1, 128, 4, 2560) and bits_equal(delta, torch.cat(deltas, dim=1))
    assert bits_equal(cat_shards(long_state.history, 3), cat_shards(four_state.history, 3))
    assert long_state.token_context == four_state.token_context
    long_state.deallocate()
    assert short_state.history.alive
    with expect_error(ValueError):  # allow-pytest.raises: fewer than nine rows cannot fill the history
        module.commit_rows_full(module.allocate_rows_state(5))


# --------------------------------------------------------------------------- the driver


LONG_TRACE_ID, TRACE_ID = 128, 77
SHORT, LONG = SimpleNamespace(rows=CHUNK_ROWS), SimpleNamespace(rows=LONG_CHUNK_ROWS)  # the chunk states by rows


@pytest.fixture
def driver(prefill_fakes):
    prefill_fakes.model.allocated_context = 4096
    prefill_fakes.prefill = driver_module.Qwen38ChunkPrefill(
        prefill_fakes.model,
        "mesh",
        "state",
        SHORT,
        TRACE_ID,
        forced_step=lambda token, context: prefill_next_context(context, token),
        long_chunk_state=LONG,
        long_chunk_trace_id=LONG_TRACE_ID,
    )
    return prefill_fakes


@pytest.mark.parametrize("start", (0, 5, 32))
@pytest.mark.parametrize("count", (0, 31, 127, 128, 129, 160, 255, 256, 300, 600))
def test_driver_runs_the_long_chunks_ahead_of_the_32_row_chunks_and_the_tail(driver, start: int, count: int) -> None:
    tokens = [1000 + index for index in range(count)]
    result = driver.prefill.run(tokens, start_position=start, ple_context=None)
    aligned = driver_module.alignment_steps(start, count)
    remaining = count - aligned
    long_chunks = remaining // 128
    accepts = driver_module.chunk_accepts(remaining - 128 * long_chunks)
    assert result.position == start + count
    timing = result.timing
    assert (timing.long_chunks, timing.chunks, timing.tail_rows) == (long_chunks, len(accepts), remaining % 32)
    log = [entry for entry in driver.log if entry[0] not in ("record", "wait")]
    if not long_chunks and not accepts:
        assert log == []
        return
    assert log[:2] == [("synchronize",), ("reset", "state", 32)]
    expected_verify = [("verify", TRACE_ID)] + ([("verify", LONG_TRACE_ID)] if long_chunks else [])
    head = 2 + (1 if long_chunks else 0)
    assert log[2:head] == ([("reset", "state", 128)] if long_chunks else [])
    assert log[head : head + len(expected_verify)] == expected_verify
    body = log[head + len(expected_verify) : -3]
    rows_written = [(entry[1], entry[2]) for entry in body if entry[0] == "prepare"]
    replays = [entry[1] for entry in body if entry[0] == "replay"]
    assert replays == [LONG_TRACE_ID] * long_chunks + [TRACE_ID] * len(accepts)
    assert [rows for rows, _ in rows_written] == [128] * long_chunks + [32] * len(accepts)
    assert all(entry[3] is None for entry in body if entry[0] == "eager")  # no MTP rows in this driver
    written = [token for _, rows in rows_written for token in rows]
    real = tokens[aligned:]
    assert written[: len(real)] == real
    assert all(token == driver_module.CHUNK_PAD_TOKEN_ID for token in written[len(real) :])
    assert len(written) == 128 * long_chunks + 32 * len(accepts)
    assert log[-3:] == [("synchronize",), ("finish", 32, start + count), ("synchronize",)]
    # The accept scalar is written once, before the padded tail, on the 32-row state only.
    assert [entry for entry in log if entry[0] == "accepted"] == (
        [("accepted", 32, remaining % 32 - 1)] if remaining % 32 else []
    )


def test_driver_without_the_long_trace_runs_32_row_chunks_only_and_rejects_bad_pairs(expect_error, driver) -> None:
    short = driver_module.Qwen38ChunkPrefill(driver.model, "mesh", "state", SHORT, TRACE_ID, forced_step=lambda t, c: c)
    result = short.run(list(range(300)), start_position=0, ple_context=None)
    assert result.timing.long_chunks == 0 and result.timing.chunks == 10 and result.timing.tail_rows == 12
    assert [e[1] for e in driver.log if e[0] == "replay"] == [TRACE_ID] * 10
    # The long trace needs the long state; the anchor is a 32-row option.
    for bad in ({"long_chunk_trace_id": LONG_TRACE_ID}, {"long_chunk_state": "l", "gdn_step_anchor": True}):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            driver_module.Qwen38ChunkPrefill(
                driver.model, "mesh", "s", "c", TRACE_ID, forced_step=lambda t, c: c, **bad
            )
    driver.log.clear()
    timed = driver.prefill.run(list(range(300)), start_position=0, ple_context=None, time_each_chunk=True)
    assert len(timed.timing.long_chunk_replay_ms) == 2 and len(timed.timing.chunk_replay_ms) == 2
    long_replays, short_replays = [("replay", LONG_TRACE_ID, True)] * 2, [("replay", TRACE_ID, True)] * 2
    assert [e for e in driver.log if e[0] == "replay"] == long_replays + short_replays
