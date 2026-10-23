# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The 128-row prefill chunk form against four consecutive 32-row chunks on the same rows, without a device.

The C = 32 chunk body is the pinned reference: on the torch-backed fakes of the step-4 / GR / QSA tests the GDN,
PLE and GR 128-row forms must reproduce four 32-row chunks row for row (the fakes are row-serial, so any drift is a
row-mixing bug, not arithmetic), the committed state after 128 rows must be the four chunks' state, the QSA chunk
inputs at 128 rows must be the four 32-row derivations stacked, and the driver must schedule the 128-row chunks ahead
of the 32-row chunks and the padded tail.  The source pins hold the forms apart: the 32-row body's ops are unchanged
(its own pins), the 128-row body adds only the row-tile loops around the DRAM-sharded linears and the full commits.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROW_COUNTS, CHUNK_ROWS, LONG_CHUNK_ROWS


def _load(name: str):
    # The tests directory is not a package: load the sibling fakes by path.
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step4 = _load("test_mtp_v2_step4_rows_no_device")
gr_test = _load("test_gr_rows_no_device")
qsa_test = _load("test_ttnn_qsa_chunk_no_device")
BF16, FP32, TILE, TP, FakeTensor = step4.BF16, step4.FP32, step4.TILE, step4.TP, step4.FakeTensor
TILES = LONG_CHUNK_ROWS // CHUNK_ROWS


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(left.view(torch.int16), right.view(torch.int16))
    )


def _full_accept(constants):
    accepted = FakeTensor([torch.full((1, 1, 1, 1), float(CHUNK_ROWS - 1)) for _ in range(TP)], FP32, TILE)
    return gdn_module.build_rows_selectors(accepted, constants)


# --------------------------------------------------------------------------- contracts


def test_long_chunk_contract(expect_error) -> None:
    assert (CHUNK_ROWS, LONG_CHUNK_ROWS, CHUNK_ROW_COUNTS) == (32, 128, (32, 128))
    assert contracts_module.chunk_row_tiles(32) == 1 and contracts_module.chunk_row_tiles(128) == 4
    for rows in (0, 31, 64, 96, 192):  # 256 .. 4096 in steps of 128 are the slab row counts
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            contracts_module.chunk_row_tiles(rows)
    assert (
        gdn_module.rows_tile_count(1) == gdn_module.rows_tile_count(32) == 32 and gdn_module.rows_tile_count(128) == 128
    )
    assert gdn_module.LONG_CONV_WINDOW_TILE_ROWS == 160
    assert moe_module.SUPPORTED_ROWS == (*range(1, 33), 128) and moe_module.LONG_PREFILL_CHUNK_ROWS == 128
    assert (
        moe_module.Qwen38TTNNMoERowContract(128).row_tiles == 4
        and moe_module.Qwen38TTNNMoERowContract(32).row_tiles == 1
    )
    assert moe_module.Qwen38TTNNMoERowContract(128).local_combine == (10, 128, 2560)
    assert moe_module.Qwen38TTNNRouting.__dataclass_fields__["tiles"].default is None
    # moe_compute admits 32 x d x output_height_shard_dim tokens; d = 4 on an 8-bank part, 1 on a 7-bank part.
    assert [moe_module.moe_compute_output_height_shard_dim(rows, matmul_ring_size=8) for rows in (1, 5, 32, 128)] == [
        1
    ] * 4
    assert [moe_module.moe_compute_output_height_shard_dim(rows, matmul_ring_size=7) for rows in (1, 5, 32, 128)] == [
        1,
        1,
        1,
        4,
    ]
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


# --------------------------------------------------------------------------- GDN: 128 rows vs four 32-row chunks


@pytest.fixture
def fake(monkeypatch):
    chunk = step4.FakeChunk()
    fake_ttnn = step4.make_fake_ttnn(chunk)
    step4.install_source_chunk_fake(monkeypatch, chunk)
    for module in (gdn_module, ple_module, layer_module, decode_matmul_module):
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    return SimpleNamespace(ttnn=fake_ttnn, chunk=chunk)


def _four_gdn_chunks(module, state, hidden: torch.Tensor):
    constants = module.allocate_rows_constants(CHUNK_ROWS)
    rows_state = module.allocate_rows_state(constants)
    module.sync_rows_history_from_state(state, rows_state)
    outputs = []
    for tile in range(TILES):
        rows = hidden[:, CHUNK_ROWS * tile : CHUNK_ROWS * (tile + 1)]
        result = module.forward_rows(step4._hidden_sharded(rows), state, rows_state)
        outputs.append(step4._cat(result.hidden_rows, 3).clone())
        module.commit_rows(state, rows_state, _full_accept(constants))
        step4._deallocate(result.final_state)
    return torch.cat(outputs, dim=2), rows_state


def test_gdn_long_chunk_matches_four_chunks_and_commits_every_row(fake) -> None:
    oracle = step4._gdn_oracle_weights()
    module = step4._gdn_module(step4._device_gdn_weights(oracle))
    torch.manual_seed(128)
    hidden = torch.randn(1, LONG_CHUNK_ROWS, 2560).to(torch.bfloat16)
    state_four = step4._seed_state(module, 41)
    state_long = step4._clone_state(module, state_four)

    four_output, four_rows = _four_gdn_chunks(module, state_four, hidden)
    four_kernel_calls = len(fake.chunk.calls)

    short_constants = module.allocate_rows_constants(CHUNK_ROWS)
    short_state = module.allocate_rows_state(short_constants)
    module.sync_rows_history_from_state(state_long, short_state)  # the 32-row state seeds the shared history
    constants = module.allocate_rows_constants(LONG_CHUNK_ROWS)
    assert constants.tile_rows == 128 and constants.window_rows == 160
    assert torch.count_nonzero(constants.row_mask_fp32.torch_shards()[0]) == 128
    rows_state = module.allocate_rows_state(constants, history=short_state.history)
    assert rows_state.history is short_state.history and not rows_state.owns_history
    assert rows_state.qkv.shape == (1, 1, 128, 2560) and rows_state.q.shape == (1, 128, 12, 128)
    result = module.forward_rows(step4._hidden_sharded(hidden), state_long, rows_state)
    assert len(fake.chunk.calls) == four_kernel_calls + 1 and fake.chunk.calls[-1]["rows"] == 128
    assert fake.chunk.calls[-1]["chunk_size"] == 32 and fake.chunk.calls[-1]["flat_v"]
    long_output = step4._cat(result.hidden_rows, 3)
    assert long_output.shape == (1, 1, 128, 2560) and result.hidden_rows is rows_state.output
    module.commit_rows_full(state_long, rows_state, result.final_state)
    assert len(fake.chunk.calls) == four_kernel_calls + 1  # no re-run at the commit

    output_error = (long_output.float() - four_output.float()).abs().max().item()
    committed, expected = step4._cat(state_long.recurrent, 1), step4._cat(state_four.recurrent, 1)
    state_error = (committed - expected).abs().max().item()
    print(
        f"GDN 128 rows vs four 32-row chunks (fake): output bitwise={_bits_equal(long_output, four_output)} max abs "
        f"{output_error:.2e}; committed state bitwise={torch.equal(committed, expected)} max abs {state_error:.2e}"
    )
    # Everything around the recurrence is row-serial on the fake: bitwise.  The kernel's NC = 4 carry is the torch
    # chunk reference's sequential fp32 carry, the same arithmetic as four calls: bitwise on the fake, pinned on the
    # device by the admission micro-test (tolerance-class fallback = the stage-1 state tolerance).
    assert _bits_equal(long_output, four_output)
    assert torch.equal(committed, expected)
    # The FIR history after 128 rows is the last three new rows (rows 125..127), bitwise the four chunks'.
    history = step4._cat(rows_state.history, 3)
    assert _bits_equal(history, step4._cat(four_rows.history, 3))
    assert torch.count_nonzero(history[:, :, 3:].float()) == 0
    # The shared history carries into a following 32-row chunk without a sync.
    torch.manual_seed(129)
    tail = torch.randn(1, CHUNK_ROWS, 2560).to(torch.bfloat16)
    next_four = module.forward_rows(step4._hidden_sharded(tail), state_four, four_rows)
    next_long = module.forward_rows(step4._hidden_sharded(tail), state_long, short_state)
    assert _bits_equal(step4._cat(next_long.hidden_rows, 3), step4._cat(next_four.hidden_rows, 3))
    for state in (rows_state, short_state):
        state.deallocate()
    assert short_state.history.alive is False and rows_state.output.alive is False


def test_gdn_long_rows_state_and_full_commit_reject_the_wrong_forms(expect_error, fake) -> None:
    module = step4._gdn_module(step4._device_gdn_weights(step4._gdn_oracle_weights()))
    for rows in (0, 33, 64, 192):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            module.allocate_rows_constants(rows)
    constants = module.allocate_rows_constants(LONG_CHUNK_ROWS)
    rows_state = module.allocate_rows_state(constants)
    state = step4._seed_state(module, 42)
    with expect_error(RuntimeError):  # allow-pytest.raises: a 32-row hidden tile is not the 128-row input
        module.forward_rows(step4._hidden_sharded(torch.zeros(1, 32, 2560).to(torch.bfloat16)), state, rows_state)
    with expect_error(RuntimeError):  # allow-pytest.raises: the full commit needs the pass's final state
        module.commit_rows_full(state, rows_state, FakeTensor([torch.zeros(1, 12, 1, 128)] * TP, FP32, TILE))


# --------------------------------------------------------------------------- PLE


def test_ple_long_chunk_matches_four_chunks_and_commits_the_last_nine_rows(expect_error, fake) -> None:
    module = step4._ple_module()
    torch.manual_seed(130)
    residual = torch.randn(LONG_CHUNK_ROWS, 4, 2560).to(torch.bfloat16)
    tokens = tuple(int(v) for v in torch.randint(0, 200_000, (LONG_CHUNK_ROWS,)))
    constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", module.mesh_contract, rows=CHUNK_ROWS)

    four_state = module.allocate_rows_state(CHUNK_ROWS)
    four_state.load_from_state(step4._seed_ple_state(module))
    deltas = []
    for tile in range(TILES):
        rows = slice(CHUNK_ROWS * tile, CHUNK_ROWS * (tile + 1))
        prepared = module.prepare_rows_input(tokens[rows], four_state)
        result = module.forward_prepared_rows(step4._residual_rows(residual[rows]), prepared, four_state)
        deltas.append(step4._cat(result.residual_delta, 3))
        module.commit_rows(four_state, _full_accept(constants))
        module.commit_rows_host(four_state, prepared, CHUNK_ROWS - 1)

    short_state = module.allocate_rows_state(CHUNK_ROWS)
    short_state.load_from_state(step4._seed_ple_state(module))
    long_state = module.allocate_rows_state(LONG_CHUNK_ROWS, history=short_state.history)
    assert long_state.history is short_state.history and not long_state.owns_history and long_state.rows == 128
    prepared = module.prepare_rows_input(tokens, long_state)
    assert prepared.embedding_rows.shape == (1, 1, 128, 640) and len(prepared.contexts) == 129
    result = module.forward_prepared_rows(step4._residual_rows(residual), prepared, long_state)
    module.commit_rows_full(long_state)
    long_state.token_context = prepared.contexts[-1]
    delta = step4._cat(result.residual_delta, 3)
    assert delta.shape == (1, 128, 4, 2560) and _bits_equal(delta, torch.cat(deltas, dim=1))
    assert _bits_equal(step4._cat(long_state.history, 3), step4._cat(four_state.history, 3))
    assert long_state.token_context == four_state.token_context
    long_state.deallocate()
    assert short_state.history.alive
    with expect_error(ValueError):  # allow-pytest.raises: fewer than nine rows cannot fill the history
        module.commit_rows_full(module.allocate_rows_state(5))


# --------------------------------------------------------------------------- GR


@pytest.fixture
def gr_fake(monkeypatch):
    fake_ttnn = gr_test._gr_fake()
    monkeypatch.setattr(gr_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(decode_matmul_module, "ttnn", fake_ttnn)
    return fake_ttnn


def test_gr_long_rows_equal_four_32_row_reads_and_writes_row_for_row(expect_error, gr_fake) -> None:
    module = gr_test._gr_module(gr_fake)
    torch.manual_seed(131)
    # The GR fake is its own copy of the step-4 fake: its dtype/layout tags, not this module's.
    tensor, bf16, tile = gr_test.FakeTensor, gr_test.BF16, gr_test.TILE
    residual = tensor([gr_test._bf16(1, 4, 128, 640) for _ in range(TP)], bf16, tile, 3)
    block_rows = tensor([gr_test._bf16(1, 1, 128, 640) for _ in range(TP)], bf16, tile, 3)
    block_input, state = module.read_rows(residual)
    written = module.write_rows(block_rows, state)
    assert block_input.shape == (1, 1, 128, 640) and state.injection.shape == (1, 1, 128, 4)
    assert written.shape == (1, 4, 128, 640)

    def tile(whole, index):
        return tensor([x.narrow(2, 32 * index, 32).clone() for x in whole.torch_shards()], whole.dtype, whole.layout, 3)

    per_tile = [module.read_rows(tile(residual, index)) for index in range(TILES)]
    per_tile_written = [module.write_rows(tile(block_rows, index), per_tile[index][1]) for index in range(TILES)]
    for device in range(TP):
        for label, whole, parts in (
            ("block input", block_input, [t[0] for t in per_tile]),
            ("injection", state.injection, [t[1].injection for t in per_tile]),
            ("written", written, per_tile_written),
        ):
            stacked = torch.cat([part.torch_shards()[device] for part in parts], dim=2)
            assert _bits_equal(whole.torch_shards()[device], stacked), (label, device)
    with expect_error(ValueError):  # allow-pytest.raises: 64 rows are not a chunk form
        module.read_rows(tensor([gr_test._bf16(1, 4, 64, 640) for _ in range(TP)], bf16, tile, 3))


# --------------------------------------------------------------------------- QSA


LONG_POSITIONS = (0, 128, 1920, 2016, 2048, 8064, 32640)


@pytest.mark.parametrize("position", LONG_POSITIONS)
def test_qsa_long_chunk_emulation_is_the_four_chunk_emulations_stacked(position: int) -> None:
    blocks = qsa_test.RESIDENT_BLOCKS
    long = qsa_module.emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=blocks, rows=LONG_CHUNK_ROWS)
    short = [
        qsa_module.emulate_qsa_chunk_inputs(position + CHUNK_ROWS * tile, allocated_compressed_blocks=blocks)
        for tile in range(TILES)
    ]
    assert torch.equal(long["kv_block_start"], short[0]["kv_block_start"])
    # The four 32-row chunks write compressed blocks P/4 .. P/4 + 31 one by one; the 128-row chunk writes them as the
    # cache's tile P/128 in one page (no per-block indices).
    assert [t.item() for chunk in short for t in chunk["block_index_i32"]] == [position // 4 + b for b in range(32)]
    assert long["block_index_i32"] == () and long["compressed_tile_i32"].tolist() == [[position // 128]]
    assert long["compressed_tile_i32"].dtype == torch.int32 and "compressed_tile_i32" not in short[0]
    for name in ("indexer_neg_mask", "row_keep_bits", "row_fill"):
        assert torch.equal(long[name], torch.cat([chunk[name] for chunk in short], dim=2)), name
    host = qsa_module.qsa_chunk_constant_rows(blocks, LONG_CHUNK_ROWS)
    assert host["arange32_lanes"].shape == (1, 1, 4, 32) and host["arange32_lanes"].reshape(-1).tolist() == list(
        range(128)
    )
    assert host["block_start_lanes"].reshape(-1).tolist() == [4 * i for i in range(32)]
    assert host["pool_select"].shape == (1, 1, 32, 128) and host["row_selects"].shape == (32, 1, 1, 32, 32)
    for block in range(32):
        assert host["pool_select"][0, 0, block].nonzero().reshape(-1).tolist() == list(range(4 * block, 4 * block + 4))
        assert host["row_selects"][block, 0, 0, 0, block] == 1.0 and host["row_selects"][block].sum() == 1.0
    for name in ("row_index_blocks", "arange_blocks_rows", "row_index_slots", "arange_slots_rows", "all_ones_rows"):
        assert host[name].shape[2] == 128, name
    short_host = qsa_module.qsa_chunk_constant_rows(blocks)
    assert torch.equal(host["pool_select"][..., :8, :32], short_host["pool_select"][..., :8, :])
    assert torch.equal(host["row_index_slots"][:, :, :32], short_host["row_index_slots"])


@pytest.mark.parametrize("position", (0, 2048, 32640))
def test_qsa_long_chunk_inputs_derive_on_the_integer_fake(expect_error, monkeypatch, position: int) -> None:
    fake = qsa_test._integer_fake()
    monkeypatch.setattr(qsa_module, "ttnn", fake)
    blocks = qsa_test.RESIDENT_BLOCKS
    host = qsa_module.qsa_chunk_constant_rows(blocks, LONG_CHUNK_ROWS)
    constants = SimpleNamespace(
        allocated_compressed_blocks=blocks,
        high27_mask=qsa_test._u32(torch.full((1, 1, 1, 1), qsa_module.KV_BLOCK_START_MASK)),
    )
    chunk = SimpleNamespace(
        allocated_compressed_blocks=blocks,
        rows=LONG_CHUNK_ROWS,
        **{name: qsa_test._u32(host[name]) for name in qsa_test.CHUNK_UINT32_TEMPLATES},
    )
    inputs = qsa_module.derive_qsa_chunk_inputs(qsa_test._u32(torch.full((1, 1, 1, 1), position)), constants, chunk)
    expected = qsa_module.emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=blocks, rows=LONG_CHUNK_ROWS)
    # The 128-row chunk writes its 32 compressed rows as one tile: the page table P / 128, no per-block indices.
    assert inputs.rows == 128 and len(inputs.block_index_i32) == len(expected["block_index_i32"]) == 0
    assert inputs.kv_block_start._read().tolist() == expected["kv_block_start"].tolist()
    assert (
        inputs.compressed_tile_i32._read().tolist() == expected["compressed_tile_i32"].tolist() == [[position // 128]]
    )
    assert inputs.compressed_tile_i32.dtype is qsa_test.I32 and inputs.compressed_tile_i32.shape == (1, 1)
    # The layer's admission of the 128-row inputs: the page table, no per-block indices.
    layer = qsa_module.Qwen38TTNNQSA.__new__(qsa_module.Qwen38TTNNQSA)
    layer.allocated_compressed_blocks = blocks
    layer._validate_chunk_inputs(inputs)
    with expect_error(ValueError):  # allow-pytest.raises: the 128-row inputs without their page table
        layer._validate_chunk_inputs(
            qsa_module.Qwen38TTNNQSAChunkInputs(
                inputs.rows, inputs.kv_block_start, (), inputs.indexer_neg_mask, inputs.row_keep_bits, inputs.row_fill
            )
        )
    assert inputs.indexer_neg_mask.shape == (1, 1, 128, blocks)
    assert torch.equal(inputs.indexer_neg_mask._read(), expected["indexer_neg_mask"])
    assert torch.equal(inputs.row_keep_bits._read(), expected["row_keep_bits"])
    assert torch.equal(inputs.row_fill._read(), expected["row_fill"])


# --------------------------------------------------------------------------- the driver


LONG_TRACE_ID, TRACE_ID = 128, 77


class _FakeModel:
    allocated_context = 4096

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def reset_chunk_state_inplace(self, state, chunk_state) -> None:
        self.calls.append(("reset", chunk_state))

    def write_chunk_accepted(self, chunk_state, accepted: int) -> None:
        self.calls.append(("accepted", chunk_state, accepted))

    def prepare_chunk_inputs(self, chunk_state, token_ids, *, ple_context, positions=None, features=None):
        tokens = list(token_ids)
        assert len(tokens) == {"chunk_state": 32, "long_state": 128}[chunk_state]
        contexts = [ple_context]
        for token in tokens:
            contexts.append((2 if contexts[-1] is None else contexts[-1][1], token))
        self.calls.append(("inputs", chunk_state, tuple(tokens), ple_context))
        return SimpleNamespace(chunk_state=chunk_state, contexts=tuple(contexts))

    def upload_chunk_inputs(self, chunk_state, prepared) -> None:
        assert prepared.chunk_state == chunk_state  # the upload lands in the state the rows were prepared for

    def finish_prefill(self, state, chunk_state, prefilled: int, *, rope_shift: int = 0) -> None:
        self.calls.append(("finish", chunk_state, prefilled))

    def forward_prefill_chunk_generic(self, chunk_state, state, *, gdn_step_anchor: bool = False) -> None:
        self.calls.append(("eager", chunk_state, gdn_step_anchor))


@pytest.fixture
def driver(monkeypatch):
    model = _FakeModel()
    log = model.calls
    fake_ttnn = SimpleNamespace(
        _ttnn_execute_trace=lambda mesh, trace_id, *, cq_id, blocking: log.append(("replay", trace_id, blocking)),
        record_event=lambda mesh, cq_id: "event",
        event_synchronize=lambda event: log.append(("event",)),
        synchronize_device=lambda mesh: log.append(("sync",)),
    )
    monkeypatch.setattr(driver_module, "ttnn", fake_ttnn)

    class Tracker:
        @staticmethod
        def verify_before_replay(mesh, trace_id) -> None:
            log.append(("verify", trace_id))

    monkeypatch.setattr(driver_module, "TraceAllocationTracker", Tracker)
    prefill = driver_module.Qwen38ChunkPrefill(
        model,
        "mesh",
        "state",
        "chunk_state",
        TRACE_ID,
        forced_step=lambda token, context: (2 if context is None else context[1], token),
        long_chunk_state="long_state",
        long_chunk_trace_id=LONG_TRACE_ID,
    )
    return SimpleNamespace(model=model, log=log, prefill=prefill)


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
    assert (result.timing.long_chunks, result.timing.chunks, result.timing.tail_rows) == (
        long_chunks,
        len(accepts),
        remaining % 32,
    )
    log = [entry for entry in driver.log if entry[0] != "event"]
    if not long_chunks and not accepts:
        assert log == []
        return
    assert log[:2] == [("sync",), ("reset", "chunk_state")]
    expected_verify = [("verify", TRACE_ID)] + ([("verify", LONG_TRACE_ID)] if long_chunks else [])
    head = 2 + (1 if long_chunks else 0)
    assert log[2:head] == ([("reset", "long_state")] if long_chunks else [])
    assert log[head : head + len(expected_verify)] == expected_verify
    body = log[head + len(expected_verify) : -3]
    rows_written = []
    replays = []
    for entry in body:
        if entry[0] == "inputs":
            rows_written.append((entry[1], entry[2]))
        elif entry[0] == "replay":
            replays.append(entry[1])
    assert replays == [LONG_TRACE_ID] * long_chunks + [TRACE_ID] * len(accepts)
    assert [state for state, _ in rows_written] == ["long_state"] * long_chunks + ["chunk_state"] * len(accepts)
    written = [token for _, rows in rows_written for token in rows]
    real = tokens[aligned:]
    assert written[: len(real)] == real and all(
        token == driver_module.CHUNK_PAD_TOKEN_ID for token in written[len(real) :]
    )
    assert len(written) == 128 * long_chunks + 32 * len(accepts)
    assert log[-3:] == [("sync",), ("finish", "chunk_state", start + count), ("sync",)]
    # The accept scalar is written once, before the padded tail, on the 32-row state only.
    assert [entry for entry in log if entry[0] == "accepted"] == (
        [("accepted", "chunk_state", remaining % 32 - 1)] if remaining % 32 else []
    )


def test_driver_without_the_long_trace_runs_32_row_chunks_only_and_rejects_bad_pairs(expect_error, driver) -> None:
    short = driver_module.Qwen38ChunkPrefill(
        driver.model, "mesh", "state", "chunk_state", TRACE_ID, forced_step=lambda t, c: c
    )
    result = short.run(list(range(300)), start_position=0, ple_context=None)
    assert result.timing.long_chunks == 0 and result.timing.chunks == 10 and result.timing.tail_rows == 12
    assert [e[1] for e in driver.log if e[0] == "replay"] == [TRACE_ID] * 10
    with expect_error(ValueError):  # allow-pytest.raises: the long trace needs the long state
        driver_module.Qwen38ChunkPrefill(
            driver.model, "mesh", "s", "c", TRACE_ID, forced_step=lambda t, c: c, long_chunk_trace_id=LONG_TRACE_ID
        )
    with expect_error(ValueError):  # allow-pytest.raises: the anchor is a 32-row option
        driver_module.Qwen38ChunkPrefill(
            driver.model,
            "mesh",
            "s",
            "c",
            TRACE_ID,
            forced_step=lambda t, c: c,
            long_chunk_state="l",
            gdn_step_anchor=True,
        )
    driver.log.clear()
    timed = driver.prefill.run(list(range(300)), start_position=0, ple_context=None, time_each_chunk=True)
    assert len(timed.timing.long_chunk_replay_ms) == 2 and len(timed.timing.chunk_replay_ms) == 2
    assert [e for e in driver.log if e[0] == "replay"] == [("replay", LONG_TRACE_ID, True)] * 2 + [
        ("replay", TRACE_ID, True)
    ] * 2


# --------------------------------------------------------------------------- source pins


def test_long_chunk_source_pins() -> None:
    layer = inspect.getsource(layer_module.Qwen38TTNNDecoderLayer.forward_chunk_generic)
    assert "if selectors is None:\n                    self.ple.commit_rows_full(chunk_state.ple)" in layer
    assert (
        "self.attention.commit_rows_full(generic_state.attention, chunk_state.attention, result.final_state)" in layer
    )
    assert 'raise ValueError("the GDN step anchor is a 32-row chunk option")' in layer
    model = inspect.getsource(model_module.Qwen38TTNNTextModel.forward_prefill_chunk_generic)
    assert "rope = chunk_state.rope_rows" in model and "state.position.advance_by(rows)" in model
    allocate = inspect.getsource(model_module.Qwen38TTNNTextModel.allocate_chunk_state)
    assert (
        "moe_module.allocate_local_combine_output(" in allocate
        and "self.mesh_device, self.mesh_contract, moe_module.routed_tokens_per_call_for(rows)" in allocate
    )
    # The DRAM-sharded linears run one row tile per call on every 128-row body; the hidden row tiles are moved
    # into the activation shard once per layer (MoE: router + shared chain; QSA: the five linears).
    for function in (
        gdn_module.Qwen38TTNNGDN._project_rows_linear,  # the projection's linear half (_project_rows slices its result)
        gr_module.Qwen38TTNNGatedResidual.read_rows,
        moe_module.Qwen38TTNNMoE.forward,
        qsa_module.Qwen38TTNNQSA._hidden_row_tiles,
        qsa_module.Qwen38TTNNQSA._project_output_rows,
    ):
        assert "dram_sharded_row_tiles(" in inspect.getsource(function), function.__name__
    for function in (
        gdn_module.Qwen38TTNNGDN._gate_and_project_rows,
        moe_module.Qwen38TTNNMoE._shared_partial,
        qsa_module.Qwen38TTNNQSA._linear_rows,
    ):
        assert "dram_sharded_row_tiles(" not in inspect.getsource(function), function.__name__
    tiles = inspect.getsource(decode_matmul_module.dram_sharded_row_tiles)
    assert "ttnn.slice(" in tiles and "ttnn.to_memory_config(tile, activation_memory_config)" in tiles
    # The router tail (softmax, top-k, normalization, the ROW_MAJOR routing) runs once per layer on the
    # concatenated logits; the 128-row GR read folds by branch slices + concats and reduces its four FP32
    # partials in one collective; the GDN out-proj folds one head-major tile at a time (the 32-row form's view).
    route = inspect.getsource(moe_module.Qwen38TTNNMoE._route)
    assert route.count("ttnn.softmax(") == route.count("ttnn.topk(") == 1 and "logits_tiles, dim=2" in route
    read = inspect.getsource(gr_module.Qwen38TTNNGatedResidual.read_rows)
    assert (
        read.count("ttnn.experimental.all_gather_async(") == 1 and read.count("ttnn.experimental.fast_reduce_nc(") == 2
    )
    assert read.count("ttnn.concat(branches, dim=3") == read.count("ttnn.concat(branches, dim=1") == 1
    gate = inspect.getsource(gdn_module.Qwen38TTNNGDN._gate_and_project_rows)
    assert (
        gate.count("ttnn.experimental.view(") == 2
        and '_copy_inplace(output, rows_state.output, label="GDN rows output")' in gate
    )
    assert not hasattr(gdn_module.Qwen38TTNNGDN, "_fold_head_rows_long")
    routed = inspect.getsource(moe_module.Qwen38TTNNMoE._routed_partial)
    assert (
        "if self.rows != LONG_PREFILL_CHUNK_ROWS and not self.slab_one_call:\n            zeroed = ttnn.fill(" in routed
    )
    # The long chunk's combine is tilized as one [1280, 2560] tile grid and viewed back as the reduce's rank-4 input.
    assert "ttnn.reshape(outputs[5], (TOP_K * self.rows, HIDDEN_SIZE))" in routed
    assert "ttnn.experimental.view(combine_flat, self.row_contract.fast_reduce_input)" in routed
    commit = inspect.getsource(gdn_module.Qwen38TTNNGDN.commit_rows_full)
    assert "_copy_inplace(final_state, state.recurrent" in commit and "history_select_full" in commit
    assert "chunk_gated_delta_rule" not in commit
    # The indexer scores one 32-row query tile per call; the sparse attention takes all rows at once.
    score = inspect.getsource(qsa_module.Qwen38TTNNQSA._score_blocks_chunk)
    assert "for start in range(0, rows, CHUNK_ROWS):" in score and score.count("indexer_score_dsa(") == 1
    attention = inspect.getsource(qsa_module.Qwen38TTNNQSA._sparse_value_attention_rows)
    assert attention.count("sparse_sdpa(") == 1 and "rows = constants.rows" in attention
    # The hand-off reads the 32-row state only.
    for function in (
        model_module.Qwen38TTNNTextModel.finish_prefill,
        layer_module.Qwen38TTNNDecoderLayer.finish_chunk_state_inplace,
        qsa_module.Qwen38TTNNQSA.handoff_chunk_state,
    ):
        assert "!= CHUNK_ROWS" in inspect.getsource(function), function.__name__
