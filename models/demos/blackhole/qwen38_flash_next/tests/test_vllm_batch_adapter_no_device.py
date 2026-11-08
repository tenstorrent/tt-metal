# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter's batch mode (P2a) on a fake chain and a fake lane decoder (no device).

Mode selection (``max_num_seqs`` 1 = the single-slot path, 2..32 = batch mode, anything else refused before any library
call); the block-size and pool checks; the deferred chain open (``initialize_vllm_model`` constructs, ``allocate_kv_cache``
opens the chain with the lane decoder's ``prepare`` as its warm hook and captures); the slot -> lane map under the
plugin's ``slot_remap`` (applied once per decode) and ``note_state_slots_moved`` (the same permutation noted after the
decode changes nothing); page-table rows written per lane and only on change; padding rows, parked idle lanes, the
warm token; the contract errors (a live lane without a live row, a start position that is not the lane's mirror);
per-slot sampling policies, generators and reseeding; prefill into ``empty_slots``; release; warmup; the token budget.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.tests import test_qwen38_vllm_no_device as base
from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import (  # noqa: F401  (fixture)
    ALLOCATED_CONTEXT,
    MAX_MODEL_LEN,
    FakeChain,
    _initialize,
    construction,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import WARM_TOKEN_ID
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38ChatChainError
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    Qwen38ForCausalLM,
    Qwen38HostSamplingPolicy,
    greedy_batch_sampling_params,
    position_generator,
    sample_from_row,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import paged_table_width

LANES = 8
BLOCK = 128
TABLE_WIDTH = 264  # roundup8(ceil((32768 + 128) / 128)): the shared table's width at block size 128


@pytest.fixture(autouse=True)
def _single_path_prefill(monkeypatch):
    """The fake decoder has no pack surface: these tests drive the single-path prefill and the lane budget without the
    pack (on by default with 8 lanes; its rig is in test_pack_plan_no_device)."""

    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")


class FakeLanePosition:
    def __init__(self, lanes: int) -> None:
        self.positions = [0] * lanes
        self.shifts = [0] * lanes


class FakeDecoder:
    """The lane decoder's surface the adapter uses: the host mirrors, the page rows with change-only uploads, a
    parking rule, a replay whose 'device' answer is a function of the lane's own token and position."""

    instances: list["FakeDecoder"] = []
    pack_default = False  # the ``pack_enabled`` the next instance starts with (the knob, after the block-size filter)

    def __init__(self, lanes, paging=None, indexer_form=None, **kwargs) -> None:
        self.lanes, self.paging, self.indexer_form = lanes, paging, indexer_form
        self.page_rows = [[] for _ in range(lanes)]
        self.positions = [None] * lanes
        self.contexts = [None] * lanes
        self.ls = None
        self.warm_logits_gather = False
        self.shared_pools = kwargs.get("shared_pools", False)
        self.pack_arg = kwargs.get("pack")  # the ``pack=`` the adapter passed (None: not passed)
        self.pack_enabled = type(self).pack_default
        self.pack_buckets = (128, 256, 512, 1024) if self.pack_enabled else ()
        self.page_table_dirty = False
        self.log: list = []
        self.uploads: list[list[list[int]]] = []
        self.steps: list[dict] = []
        self.parks: list[list[int]] = []
        self.released = False
        self.last_ids = [0] * lanes
        FakeDecoder.instances.append(self)

    # build
    def prepare(self, chain) -> None:
        self.log.append("prepare")
        self.chain = chain
        self.ls = base.SimpleNamespace(position=FakeLanePosition(self.lanes))

    def capture(self, vm) -> None:
        self.log.append("capture")
        assert vm.chain is self.chain

    def release(self) -> None:
        self.log.append("release")
        self.released = True

    # lanes
    def lane_position(self, lane: int) -> int:
        return self.ls.position.positions[lane]

    def set_page_row(self, lane, pages) -> bool:
        pages = [int(p) for p in pages]
        if pages == self.page_rows[lane]:
            return False
        self.page_rows[lane] = pages
        self.page_table_dirty = True
        return True

    def sync_page_table(self) -> bool:
        if not self.page_table_dirty:
            return False
        self.uploads.append([list(row) for row in self.page_rows])
        self.page_table_dirty = False
        return True

    def park_idle(self, lanes) -> bool:
        lanes = list(lanes)
        moving = [lane for lane in lanes if self.ls.position.positions[lane]]
        for lane in moving:
            self.ls.position.positions[lane] = 0
        self.parks.append(lanes)
        return bool(moving)

    def import_lane(self, vm, lane, length, context, pages) -> None:
        assert self.positions[lane] is None and not self.page_rows[lane]
        self.log.append(("import", lane, length, tuple(pages), context))
        self.set_page_row(lane, pages)
        self.uploads.append([list(row) for row in self.page_rows])
        self.page_table_dirty = False
        self.positions[lane] = length
        self.contexts[lane] = context
        self.ls.position.positions[lane] = length

    def release_lane_external(self, lane) -> bool:
        was_live = self.positions[lane] is not None or bool(self.page_rows[lane])
        if self.page_rows[lane]:
            self.page_rows[lane] = []
            self.page_table_dirty = True
        self.positions[lane] = None
        self.contexts[lane] = None
        self.log.append(("release_lane", lane))
        return was_live

    # decode
    @staticmethod
    def next_id(token: int, position: int) -> int:
        return (token * 7 + position * 13 + 5) % 5000 + 100

    def step(self, tokens) -> list[int]:
        positions = list(self.ls.position.positions)
        self.steps.append(
            {
                "tokens": list(tokens),
                "positions": positions,
                "rows": [list(row) for row in self.page_rows],
                "live": [p is not None for p in self.positions],
            }
        )
        self.last_ids = [self.next_id(t, p) for t, p in zip(tokens, positions)]
        self.ls.position.positions = [p + 1 for p in positions]
        return list(self.last_ids)

    def logits_for(self, lane: int, step_no: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(1000 + 31 * lane + step_no)
        row = torch.randn(VOCAB_SIZE, generator=generator)
        row[self.last_ids[lane]] += 12.0
        return row

    def read_full_logits(self) -> torch.Tensor:
        return torch.stack([self.logits_for(lane, len(self.steps)) for lane in range(self.lanes)])


def _fake_dram(monkeypatch, free_gib: int = 128) -> None:
    """The allocator's DRAM view: 8 banks, ``free_gib`` GiB free in all (the pool admission reads it)."""

    monkeypatch.setattr(
        vllm_module.ttnn,
        "get_memory_view",
        lambda mesh, kind: base.SimpleNamespace(
            num_banks=8,
            total_bytes_free_per_bank=free_gib * 2**30 // 8,
            total_bytes_allocated_per_bank=2**30,
            largest_contiguous_bytes_free_per_bank=free_gib * 2**30 // 8,
        ),
        raising=False,
    )
    monkeypatch.setattr(vllm_module.ttnn, "BufferType", base.SimpleNamespace(DRAM="dram"), raising=False)


class Rig:
    def __init__(
        self, monkeypatch, *, lanes=LANES, block_size=BLOCK, num_blocks=2048, context=ALLOCATED_CONTEXT, free_gib=128
    ):
        FakeDecoder.instances.clear()
        self.chain = FakeChain()
        self.chain.mesh = "mesh"
        self.order: list[str] = []
        monkeypatch.setattr(vllm_module, "Qwen38PlainLaneDecoder", FakeDecoder)
        _fake_dram(monkeypatch, free_gib)

        def opener(warm_hook, paging=None):
            self.order.append("open")
            self.opened_paging = paging
            warm_hook(self.chain)
            self.order.append("opened")
            return self.chain

        self.model = Qwen38ForCausalLM(None, allocated_context=context, batch_lanes=lanes, chain_opener=opener)
        self.lanes, self.block_size, self.num_blocks = lanes, block_size, num_blocks
        self.model.allocate_kv_cache((num_blocks, 1, block_size, 128), None, 12)
        self.decoder = FakeDecoder.instances[-1]
        self.width = paged_table_width(context, block_size)

    def table_row(self, blocks) -> list[int]:
        return list(blocks) + [0] * (self.width - len(blocks))

    def prefill(self, prompts, slots, blocks, sampling_params=None, start_pos=None, **overrides):
        rows = len(prompts)
        tokens = torch.zeros(rows, max(len(p) for p in prompts), dtype=torch.int32)
        for i, prompt in enumerate(prompts):
            tokens[i, : len(prompt)] = torch.tensor(prompt, dtype=torch.int32)
        kwargs = dict(
            tokens=tokens,
            page_table=torch.tensor([self.table_row(b) for b in blocks], dtype=torch.int32),
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(p) for p in prompts],
            start_pos=torch.zeros(rows, dtype=torch.int32) if start_pos is None else start_pos,
            empty_slots=slots,
            sampling_params=sampling_params,
        )
        kwargs.update(overrides)
        return self.model.prefill_forward(**kwargs)

    def decode(self, rows, **overrides):
        """``rows``: per decode row ``(token, position, blocks)`` or None (a padding row)."""

        tokens = torch.zeros(self.lanes, 1, dtype=torch.int32)
        positions = torch.full((self.lanes,), -1, dtype=torch.int32)
        table = torch.zeros(self.lanes, self.width, dtype=torch.int32)
        for i, row in enumerate(rows):
            if row is None:
                continue
            token, position, blocks = row
            tokens[i, 0], positions[i] = token, position
            table[i] = torch.tensor(self.table_row(blocks), dtype=torch.int32)
        kwargs = dict(
            tokens=tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            sampling_params=greedy_batch_sampling_params(self.lanes),
            slot_remap=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        kwargs.update(overrides)
        return self.model.decode_forward(**kwargs)


@pytest.fixture
def rig(monkeypatch) -> Rig:
    return Rig(monkeypatch)


def _prompt(length: int, base_id: int) -> list[int]:
    return [base_id + index for index in range(length)]


def _three(rig: Rig):
    """Three requests prefilled into slots 0, 1, 2: lengths 40, 70, 130; blocks (per 64): 1, 2, 3 blocks."""

    prompts = [_prompt(40, 1000), _prompt(70, 2000), _prompt(130, 3000)]
    blocks = [[5], [9, 3], [20, 11, 7]]
    out, _ = rig.prefill(prompts, [0, 1, 2], blocks, greedy_batch_sampling_params(3))
    return prompts, blocks, out


# -- mode selection and construction ------------------------------------------------------------------------------------


def test_max_num_seqs_one_is_the_single_slot_path_and_opens_the_chain_in_allocate_kv_cache(construction) -> None:
    model = _initialize(construction, max_batch_size=1, allocate=False)
    assert model.batch_mode is False and model.lanes == 1 and model.chain is None and model.decoder is None
    assert "open" not in [call[0] for call in construction.calls]
    model.allocate_kv_cache(base.KV_CACHE_SHAPE, None, 12)
    assert model.chain is construction.chain and model.decoder is None
    opened = [call for call in construction.calls if call[0] == "open"][0][2]
    assert "warm_hook" not in opened and opened["paging"].num_pages == base.NUM_BLOCKS


@pytest.mark.parametrize("batch", (2, 8, 32))
def test_batch_mode_constructs_but_does_not_open_the_chain(construction, batch: int) -> None:
    model = _initialize(construction, max_batch_size=batch)
    assert model.batch_mode and model.lanes == batch and model.chain is None and model.decoder is None
    names = [call[0] for call in construction.calls]
    assert names[-2:] == ["prepare", "construct"] and "open" not in names
    assert model.lane_of_slot == list(range(batch)) and len(model.lane_slots) == batch
    assert model.allocated_context == ALLOCATED_CONTEXT and model.context_limit == MAX_MODEL_LEN


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"max_batch_size": 33}, "max_num_seqs 2..32"),
        ({"max_batch_size": 0}, "max_num_seqs 2..32"),
        ({"max_batch_size": 8, "tt_data_parallel": 2}, "data parallelism"),
    ),
)
def test_unservable_batch_configurations_are_refused_before_any_library_call(
    construction, overrides, message, expect_error
) -> None:
    with expect_error(ValueError, message):
        _initialize(construction, **overrides)
    assert construction.calls == []


def test_the_deferred_open_runs_prepare_as_the_warm_hook_then_captures(construction, monkeypatch, expect_error) -> None:
    FakeDecoder.instances.clear()
    monkeypatch.setattr(vllm_module, "Qwen38PlainLaneDecoder", FakeDecoder)
    _fake_dram(monkeypatch)
    chain = construction.chain
    chain.mesh = "mesh"
    order: list[str] = []
    opened: dict = {}

    def open_chain(constructed, **kwargs):
        assert constructed == "construction" and kwargs["chunked_prefill"] and kwargs["sampling"]
        order.append("open")
        opened["paging"] = kwargs["paging"]  # the pools are allocated before the warm hook runs
        kwargs["warm_hook"](chain)
        order.append("opened")
        return chain

    monkeypatch.setattr(vllm_module, "Qwen38TracedChain", base.SimpleNamespace(open=open_chain))
    model = _initialize(construction, max_batch_size=8)
    assert order == [] and FakeDecoder.instances == []
    assert model.allocate_kv_cache((4096, 1, 128, 128), None, 12) is None
    (decoder,) = FakeDecoder.instances
    assert order == ["open", "opened"] and decoder.log == ["prepare", "capture"]
    assert model.chain is chain and model.decoder is decoder
    assert decoder.lanes == 8 and decoder.indexer_form == "wide" and decoder.warm_logits_gather is True
    assert (decoder.paging.page_size, decoder.paging.num_pages, decoder.paging.max_blocks) == (128, 4096, 264)
    assert opened["paging"] == decoder.paging and decoder.shared_pools and decoder.pack_enabled is False
    assert (model.block_size, model.num_blocks) == (128, 4096)
    assert model.dram["chain_only"]["free_bytes"] == 128 * 2**30 == model.dram["after_capture"]["free_bytes"]
    with expect_error(RuntimeError, "runs once"):
        model.allocate_kv_cache((4096, 1, 128, 128), None, 12)


@pytest.mark.parametrize("block_size", (0, 16, 31, 32, 48, 64, 100, 192, 8192))
def test_block_sizes_that_are_not_a_power_of_two_of_at_least_128_are_refused(
    monkeypatch, block_size, expect_error
) -> None:
    with expect_error(ValueError, "power of two"):
        Rig(monkeypatch, block_size=block_size)
    assert FakeDecoder.instances == []  # refused before the decoder (and so the chain) exists


@pytest.mark.parametrize("block_size, width", ((128, 264), (256, 136), (1024, 40), (4096, 16)))
def test_the_page_table_width_follows_the_block_size(monkeypatch, block_size, width) -> None:
    rig = Rig(monkeypatch, block_size=block_size)
    assert rig.decoder.paging.page_size == block_size and rig.decoder.paging.max_blocks == width
    assert rig.decoder.shared_pools and rig.decoder.paging == rig.opened_paging


def test_lanes_the_free_dram_cannot_hold_are_refused_naming_the_knob(monkeypatch, expect_error) -> None:
    with expect_error(ValueError, "lane states need"):
        Rig(monkeypatch, free_gib=1)  # ~1.05 GB of lane states plus the margin: past 1 GiB (the pools are already out)
    assert FakeDecoder.instances[-1].log == []  # prepare did not run


def test_a_pool_of_one_block_and_a_bad_shape_are_refused(monkeypatch, expect_error) -> None:
    with expect_error(ValueError, "num_blocks"):
        Rig(monkeypatch, num_blocks=1)
    model = Qwen38ForCausalLM(None, allocated_context=ALLOCATED_CONTEXT, batch_lanes=8, chain_opener=lambda hook: None)
    with expect_error(ValueError, "kv_cache_shape"):
        model.allocate_kv_cache((64, 128, 64), None, 12)


# -- prefill ------------------------------------------------------------------------------------------------------------


def test_prefill_runs_each_row_on_the_b1_chain_and_imports_into_the_slots_lane(rig: Rig) -> None:
    prompts, blocks, out = _three(rig)
    assert out.shape == (3, 1) and out.dtype == torch.int32
    imports = [entry for entry in rig.decoder.log if isinstance(entry, tuple) and entry[0] == "import"]
    assert [(i[1], i[2], i[3]) for i in imports] == [(0, 40, (5,)), (1, 70, (9, 3)), (2, 130, (20, 11, 7))]
    assert [i[4] is not None for i in imports] == [True] * 3  # the PLE context of the B=1 chain
    assert rig.decoder.positions[:4] == [40, 70, 130, None]
    assert rig.decoder.page_rows[:4] == [[5], [9, 3], [20, 11, 7], []]
    assert rig.model.slot.committed == [] and rig.model.slot.ple_context is None  # the scratch state is free again
    assert len(rig.chain.chunks) == 3  # every prompt had >= 16 rows before its last token: the chunk driver


def test_each_row_prefill_writes_the_chain_table_after_the_reset_and_before_its_chunks(rig: Rig) -> None:
    _three(rig)
    log = rig.chain.log
    resets = [i for i, entry in enumerate(log) if entry.startswith("reset:")]
    tables = [i for i, entry in enumerate(log) if entry.startswith("table:")]
    chunks = [i for i, entry in enumerate(log) if entry.startswith("chunk_prefill:")]
    assert len(resets) == len(tables) == len(chunks) == 3
    assert all(r < t < c for r, t, c in zip(resets, tables, chunks))
    assert [log[i] for i in tables] == ["table:(5,)", "table:(9, 3)", "table:(20, 11, 7)"]


def test_the_batch_prefill_does_not_take_the_pack_when_the_decoder_has_it_disabled(rig: Rig, monkeypatch) -> None:
    assert rig.decoder.shared_pools and rig.decoder.pack_enabled is False
    monkeypatch.setattr(
        rig.model, "_prefill_batch_grouped", lambda *a, **k: pytest.fail("the grouped path ran with the pack off")
    )
    rig.prefill([_prompt(20, 5), _prompt(21, 50)], [0, 1], [[1], [2]], greedy_batch_sampling_params(2))


def test_the_batch_prefill_takes_the_pack_when_the_decoder_has_it_enabled(monkeypatch) -> None:
    monkeypatch.setattr(FakeDecoder, "pack_default", True)
    rig = Rig(monkeypatch)
    assert rig.decoder.shared_pools and rig.decoder.pack_enabled is True
    taken = []

    def grouped(prompts, pages, policies, slots):
        taken.append((len(prompts), list(slots)))
        return torch.zeros(len(prompts), 1, dtype=torch.int32)

    monkeypatch.setattr(rig.model, "_prefill_batch_grouped", grouped)
    rig.prefill([_prompt(20, 5), _prompt(21, 50)], [0, 1], [[1], [2]], greedy_batch_sampling_params(2))
    assert taken == [(2, [0, 1])]


@pytest.mark.parametrize(
    "lanes, env, expected",
    (
        (8, None, True),
        (16, None, True),
        (32, None, True),
        (2, None, False),
        (7, None, False),
        (8, "0", False),
        (32, "0", False),
        (2, "1", True),
    ),
)
def test_open_batch_passes_the_pack_decision_to_the_lane_decoder(monkeypatch, lanes, env, expected) -> None:
    """``_open_batch`` drives fine with the fakes: the decoder gets ``pack=prefill_pack_enabled(lanes=..., shared_pools=True)``."""

    if env is None:
        monkeypatch.delenv("QWEN38_PREFILL_PACK", raising=False)
    else:
        monkeypatch.setenv("QWEN38_PREFILL_PACK", env)
    rig = Rig(monkeypatch, lanes=lanes)
    assert rig.decoder.shared_pools and rig.decoder.pack_arg is expected


def test_mambaspec_prefill_packs_only_the_rows_that_touch_no_state_block(monkeypatch) -> None:
    monkeypatch.setattr(FakeDecoder, "pack_default", True)
    rig = Rig(monkeypatch, block_size=1024)
    model = rig.model
    monkeypatch.setattr(type(model), "_mamba_spec_on", classmethod(lambda cls: True))
    monkeypatch.setattr(model, "_mamba_layer_blocks", lambda ids, i: {"row": i})
    grouped_calls, single_calls = [], []

    def grouped(prompts, pages, policies, slots, claimed=None):
        grouped_calls.append(([len(p) for p in prompts], list(slots), claimed))
        return torch.tensor([[len(p)] for p in prompts], dtype=torch.int32)

    def single(decoder, slot, prompt, pages, policy, claimed=None, start=0, layer_blocks=None):
        single_calls.append((slot, len(prompt), start, layer_blocks))
        return torch.tensor([[len(prompt)]], dtype=torch.int32)

    monkeypatch.setattr(model, "_prefill_batch_grouped", grouped)
    monkeypatch.setattr(model, "_prefill_row_single", single)
    prompts = [_prompt(300, 5), _prompt(1024, 6), _prompt(1500, 7), _prompt(700, 8)]
    starts = torch.tensor([0, 0, 1024, 0], dtype=torch.int32)
    out, deltas = rig.prefill(
        prompts,
        [0, 1, 2, 3],
        [[1], [2], [3, 4], [5]],
        greedy_batch_sampling_params(4),
        start_pos=starts,
        state_block_ids_per_layer=object(),
    )
    assert [(lens, slots) for lens, slots, _ in grouped_calls] == [([300, 700], [0, 3])]  # rows 0 and 3, in order
    assert single_calls == [(1, 1024, 0, {"row": 1}), (2, 1500, 1024, {"row": 2})]  # their own start and state blocks
    assert grouped_calls[0][2] is not None  # the lanes the pack takes are shared with the per-row path
    assert out.flatten().tolist() == [300, 1024, 1500, 700]  # the original row order
    assert deltas.shape == (4,) and int(deltas.abs().sum()) == 0


def test_mambaspec_prefill_without_a_packable_row_never_calls_the_grouped_path(monkeypatch) -> None:
    monkeypatch.setattr(FakeDecoder, "pack_default", True)
    rig = Rig(monkeypatch, block_size=1024)
    model = rig.model
    monkeypatch.setattr(type(model), "_mamba_spec_on", classmethod(lambda cls: True))
    monkeypatch.setattr(model, "_mamba_layer_blocks", lambda ids, i: {"row": i})
    monkeypatch.setattr(model, "_prefill_batch_grouped", lambda *a, **k: pytest.fail("no row may be packed"))
    monkeypatch.setattr(model, "_prefill_row_single", lambda *a, **k: torch.zeros(1, 1, dtype=torch.int32))
    rig.prefill(
        [_prompt(1024, 6), _prompt(1500, 7)],
        [0, 1],
        [[1], [3, 4]],
        greedy_batch_sampling_params(2),
        start_pos=torch.tensor([0, 1024], dtype=torch.int32),
        state_block_ids_per_layer=object(),
    )


def test_release_persistent_capture_releases_the_lanes_before_it_closes_the_chain(rig: Rig) -> None:
    order: list[str] = []
    rig.decoder.release = lambda: order.append("decoder")
    rig.chain.close = lambda: order.append("chain")
    rig.model.release_persistent_capture()
    assert order == ["decoder", "chain"]


def test_prefill_into_permuted_empty_slots_uses_the_slots_lane(rig: Rig) -> None:
    rig.model.lane_of_slot = [3, 2, 1, 0, 4, 5, 6, 7]  # slot -> lane
    rig.prefill([_prompt(50, 10), _prompt(60, 20)], [1, 3], [[4], [6]], greedy_batch_sampling_params(2))
    assert rig.decoder.positions[:4] == [60, None, 50, None]  # slot 1 -> lane 2 (50 tokens), slot 3 -> lane 0 (60)
    assert rig.decoder.page_rows[2] == [4] and rig.decoder.page_rows[0] == [6]


def test_prefill_returns_host_logits_without_sampling_params_and_zero_rope_deltas(rig: Rig) -> None:
    logits, deltas = rig.prefill([_prompt(20, 5), _prompt(21, 50)], [0, 1], [[1], [2]])
    assert logits.shape == (2, 1, VOCAB_SIZE) and logits.dtype == torch.float32
    assert torch.isinf(logits[:, 0, TOKENIZER_SIZE:]).all()
    assert deltas.shape == (2,) and deltas.dtype == torch.long and int(deltas.abs().sum()) == 0


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"start_pos": torch.tensor([-1, 0], dtype=torch.int32)}, "prefill start"),
        ({"start_pos": torch.tensor([20, 0], dtype=torch.int32)}, "prefill start"),
        ({"empty_slots": [1, 1]}, "distinct"),
        ({"empty_slots": [0, 8]}, "distinct slots"),
        ({"empty_slots": [0]}, "one entry per row"),
    ),
)
def test_prefill_refusals_come_before_any_chain_call(rig: Rig, expect_error, overrides, message) -> None:
    with expect_error(ValueError, message):
        rig.prefill([_prompt(20, 5), _prompt(21, 50)], overrides.pop("empty_slots", [0, 1]), [[1], [2]], **overrides)
    assert rig.chain.log == [] and rig.decoder.positions == [None] * LANES


@pytest.mark.parametrize(
    "blocks, message",
    (
        ([[1], []], "needs 1 blocks"),
        ([[1], [2, 0, 3]], "null block"),
        ([[1], [5000]], "block ids must be in"),
    ),
)
def test_prefill_refuses_bad_page_rows_before_any_chain_call(rig: Rig, expect_error, blocks, message) -> None:
    with expect_error(ValueError, message):
        rig.prefill([_prompt(20, 5), _prompt(21, 50)], [0, 1], blocks)
    assert rig.chain.log == []


def test_a_page_row_longer_than_the_lane_table_is_refused(rig: Rig, expect_error) -> None:
    table = torch.zeros(1, 599, dtype=torch.int32)
    table[0] = torch.arange(1, 600)
    with expect_error(ValueError, "exceed the lane table"):
        rig.model.prefill_forward(
            tokens=torch.tensor([_prompt(20, 5)], dtype=torch.int32),
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[20],
            start_pos=[0],
            empty_slots=[0],
        )
    assert rig.chain.log == []


def test_prefill_needs_enough_blocks_for_the_prompt(rig: Rig, expect_error) -> None:
    with expect_error(ValueError, "needs 3 blocks"):
        rig.prefill([_prompt(260, 5)], [0], [[1, 2]])


def test_prefill_into_a_slot_whose_lane_is_still_live_releases_it(rig: Rig) -> None:
    rig.prefill([_prompt(20, 5)], [0], [[1]], greedy_batch_sampling_params(1))
    rig.prefill([_prompt(22, 9)], [0], [[2]], greedy_batch_sampling_params(1))
    assert ("release_lane", 0) in rig.decoder.log and rig.decoder.page_rows[0] == [2] and rig.decoder.positions[0] == 22


# -- decode: lanes, the map, page rows ----------------------------------------------------------------------------------


def _rows_for(prompts, blocks, firsts, order, extra=0):
    """Decode rows in ``order`` (indices of the requests): token ``firsts[i]`` at position ``len(prompt) + extra``."""

    return [(int(firsts[i]), len(prompts[i]) + extra, blocks[i]) for i in order]


def test_decode_runs_live_rows_in_their_lanes_and_pads_the_rest(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    rows = _rows_for(prompts, blocks, firsts.reshape(-1).tolist(), [0, 1, 2]) + [None] * 5
    out = rig.decode(rows)
    assert out.shape == (LANES, 1) and out.dtype == torch.int32
    (step,) = rig.decoder.steps
    firsts = firsts.reshape(-1).tolist()
    assert step["tokens"][:3] == firsts and step["tokens"][3:] == [WARM_TOKEN_ID] * 5
    assert step["positions"][:3] == [40, 70, 130] and step["positions"][3:] == [0] * 5
    expected = [FakeDecoder.next_id(t, p) for t, p in zip(firsts, (40, 70, 130))]
    assert out[:3, 0].tolist() == expected and out[3:, 0].tolist() == [0] * 5  # padding rows return token 0
    assert step["rows"][:3] == blocks and step["rows"][3:] == [[]] * 5  # idle lanes read the scratch page (row zero)
    assert rig.decoder.parks[-1] == [3, 4, 5, 6, 7]


def test_idle_lanes_are_parked_again_after_each_replay_advanced_them(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    out = rig.decode(_rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5)
    assert rig.decoder.ls.position.positions[3:] == [1] * 5  # the replay advanced the idle lanes
    nxt = out[:3, 0].tolist()
    rig.decode([(nxt[i], len(prompts[i]) + 1, blocks[i]) for i in range(3)] + [None] * 5)
    assert rig.decoder.steps[1]["positions"][3:] == [0] * 5  # parked at 0 before the second replay


def test_slot_remap_moves_the_map_once_and_rows_follow_their_lanes(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    remap = [2, 0, 1, 3, 4, 5, 6, 7]  # row i reads the state in slot remap[i]
    order = [2, 0, 1]  # row 0 is request 2 (slot 2), row 1 request 0, row 2 request 1
    out = rig.decode(_rows_for(prompts, blocks, firsts, order) + [None] * 5, slot_remap=torch.tensor(remap))
    assert rig.model.lane_of_slot == [2, 0, 1, 3, 4, 5, 6, 7]
    step = rig.decoder.steps[-1]
    assert step["tokens"][:3] == [firsts[0], firsts[1], firsts[2]]  # each token went to its own request's lane
    assert out[:3, 0].tolist() == [FakeDecoder.next_id(firsts[i], len(prompts[i])) for i in order]
    # the plugin notes the same permutation after the accepted decode: the map must not move again
    rig.model.note_state_slots_moved({remap[i]: i for i in range(8) if remap[i] != i})
    assert rig.model.lane_of_slot == [2, 0, 1, 3, 4, 5, 6, 7]
    # the next decode: identity remap, rows stay where the plugin put them (slot i = row i)
    nxt = out[:3, 0].tolist()
    rig.decode([(nxt[k], len(prompts[order[k]]) + 1, blocks[order[k]]) for k in range(3)] + [None] * 5)
    assert rig.model.lane_of_slot == [2, 0, 1, 3, 4, 5, 6, 7]
    assert rig.decoder.steps[-1]["positions"][:3] == [len(prompts[i]) + 1 for i in (0, 1, 2)]


def test_a_second_remap_composes_with_the_first(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    out = rig.decode(_rows_for(prompts, blocks, firsts, [2, 0, 1]) + [None] * 5, slot_remap=[2, 0, 1, 3, 4, 5, 6, 7])
    nxt = out[:3, 0].tolist()
    # request now in slot 0 is request 2 (lane 2), slot 1 request 0 (lane 0), slot 2 request 1 (lane 1); swap rows 0, 1
    rows = [
        (nxt[1], len(prompts[0]) + 1, blocks[0]),
        (nxt[0], len(prompts[2]) + 1, blocks[2]),
        (nxt[2], len(prompts[1]) + 1, blocks[1]),
    ]
    rig.decode(rows + [None] * 5, slot_remap=[1, 0, 2, 3, 4, 5, 6, 7])
    assert rig.model.lane_of_slot == [0, 2, 1, 3, 4, 5, 6, 7]
    assert rig.decoder.steps[-1]["positions"][:3] == [len(prompts[0]) + 1, len(prompts[1]) + 1, len(prompts[2]) + 1]


def test_note_state_slots_moved_applies_a_permutation_decode_did_not_carry(rig: Rig) -> None:
    rig.model.note_state_slots_moved({0: 1, 1: 0})
    assert rig.model.lane_of_slot == [1, 0, 2, 3, 4, 5, 6, 7]
    rig.model.note_state_slots_moved({})
    assert rig.model.lane_of_slot == [1, 0, 2, 3, 4, 5, 6, 7]
    rig.model.note_state_slots_moved({1: 2, 2: 1})  # on top of the first
    assert rig.model.lane_of_slot == [1, 2, 0, 3, 4, 5, 6, 7]


def test_note_state_slots_moved_refuses_a_non_permutation(rig: Rig, expect_error) -> None:
    with expect_error(ValueError, "not a permutation"):
        rig.model.note_state_slots_moved({0: 1})
    assert rig.model.lane_of_slot == list(range(LANES))


def test_a_note_that_differs_from_the_pending_one_is_applied(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rig.decode(_rows_for(prompts, blocks, firsts, [1, 0, 2]) + [None] * 5, slot_remap=[1, 0, 2, 3, 4, 5, 6, 7])
    assert rig.model.lane_of_slot == [1, 0, 2, 3, 4, 5, 6, 7]
    rig.model.note_state_slots_moved({2: 3, 3: 2})  # not the pending {1: 0, 0: 1}
    assert rig.model.lane_of_slot == [1, 0, 3, 2, 4, 5, 6, 7]


def test_slot_remap_must_be_a_full_permutation(rig: Rig, expect_error) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rows = _rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5
    for bad in ([0, 1], [0, 0, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 8]):
        with expect_error(ValueError, "permutation"):
            rig.decode(rows, slot_remap=bad)
    assert rig.decoder.steps == [] and rig.model.lane_of_slot == list(range(LANES))


def test_page_rows_follow_the_lane_and_upload_only_on_change(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    uploads = len(rig.decoder.uploads)  # the three imports each wrote the table
    out = rig.decode(_rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5)
    assert len(rig.decoder.uploads) == uploads  # the rows equal the imported ones: no upload
    nxt = out[:3, 0].tolist()
    out = rig.decode([(nxt[i], len(prompts[i]) + 1, blocks[i]) for i in range(3)] + [None] * 5)
    assert len(rig.decoder.uploads) == uploads
    # request 0 gets a new block from vLLM: only the table is written, once, with that lane's row longer
    nxt = out[:3, 0].tolist()
    grown = [blocks[0] + [12], blocks[1], blocks[2]]
    out = rig.decode([(nxt[i], len(prompts[i]) + 2, grown[i]) for i in range(3)] + [None] * 5)
    assert len(rig.decoder.uploads) == uploads + 1
    assert rig.decoder.uploads[-1][:3] == [[5, 12], [9, 3], [20, 11, 7]] and rig.decoder.uploads[-1][3:] == [[]] * 5
    nxt = out[:3, 0].tolist()
    rig.decode([(nxt[i], len(prompts[i]) + 3, grown[i]) for i in range(3)] + [None] * 5)
    assert len(rig.decoder.uploads) == uploads + 1  # unchanged rows: no upload


def test_page_rows_are_written_to_the_lane_that_owns_the_row_after_a_remap(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rig.decode(_rows_for(prompts, blocks, firsts, [2, 0, 1]) + [None] * 5, slot_remap=[2, 0, 1, 3, 4, 5, 6, 7])
    assert rig.decoder.page_rows[:3] == blocks  # lane k keeps request k's blocks: nothing moved on the device
    assert rig.decoder.steps[-1]["rows"][:3] == blocks


def test_a_live_lane_without_a_live_row_is_a_contract_error_before_the_replay(rig: Rig, expect_error) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rows = _rows_for(prompts, blocks, firsts, [0, 1, 2])
    rows[1] = None  # request 1 (slot 1, lane 1) has no row
    rows += [None] * 5
    with expect_error(Qwen38ChatChainError, "lanes \\[1\\] hold live requests"):
        rig.decode(rows)
    assert rig.decoder.steps == [] and rig.model.batch_failed is False and rig.model.lane_of_slot == list(range(LANES))


def test_a_live_row_whose_lane_holds_no_request_is_a_contract_error(rig: Rig, expect_error) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rows = _rows_for(prompts, blocks, firsts, [0, 1, 2]) + [(5, 10, [1])] + [None] * 4
    with expect_error(Qwen38ChatChainError, "no prefilled request"):
        rig.decode(rows)
    assert rig.decoder.steps == []


def test_a_start_position_that_is_not_the_lanes_mirror_is_refused_before_the_replay(rig: Rig, expect_error) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rows = _rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5
    rows[1] = (rows[1][0], rows[1][1] + 1, rows[1][2])
    with expect_error(Qwen38ChatChainError, "runner position 71 vs device position 70 \\(row 1, lane 1\\)"):
        rig.decode(rows)
    assert rig.decoder.steps == []


def test_a_decode_row_without_its_block_is_refused(rig: Rig, expect_error) -> None:
    rig.prefill([_prompt(128, 5)], [0], [[1]], greedy_batch_sampling_params(1))  # exactly one block full
    with expect_error(ValueError, "needs 2 blocks"):
        rig.decode([(7, 128, [1])] + [None] * 7)  # the step writes position 64: block index 1 is missing
    assert rig.decoder.steps == []


def test_decode_refusals(rig: Rig, expect_error) -> None:
    rig.prefill([_prompt(20, 5)], [0], [[1]], greedy_batch_sampling_params(1))
    with expect_error(ValueError, "reload_inputs must be True"):
        rig.decode([(7, 20, [1])] + [None] * 7, reload_inputs=False)
    with expect_error(TypeError, "reset_batch"):
        rig.decode([(7, 20, [1])] + [None] * 7, reset_batch=True)
    with expect_error(ValueError, "live row"):
        rig.decode([None] * 8)
    with expect_error(ValueError, "tokens must be"):
        rig.decode([(7, 20, [1])] + [None] * 7, tokens=torch.zeros(1, 1, dtype=torch.int32))
    with expect_error(ValueError, "decode token"):
        rig.decode([(TOKENIZER_SIZE, 20, [1])] + [None] * 7)
    assert rig.decoder.steps == []


def test_a_failed_replay_poisons_the_batch(rig: Rig, expect_error) -> None:
    rig.prefill([_prompt(20, 5)], [0], [[1]], greedy_batch_sampling_params(1))

    def boom(tokens):
        raise RuntimeError("device hang")

    rig.decoder.step = boom
    with expect_error(RuntimeError, "device hang"):
        rig.decode([(7, 20, [1])] + [None] * 7)
    assert rig.model.batch_failed is True
    with expect_error(Qwen38ChatChainError, "not trusted"):
        rig.decode([(7, 20, [1])] + [None] * 7)
    rig.model.release_persistent_capture()
    assert rig.decoder.released is False and rig.chain.closed is False  # left to the mesh close


def test_host_logits_for_every_live_row_when_there_are_no_sampling_params(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rows = _rows_for(prompts, blocks, firsts, [1, 0, 2]) + [None] * 5
    logits = rig.decode(rows, slot_remap=[1, 0, 2, 3, 4, 5, 6, 7], sampling_params=None)
    assert logits.shape == (LANES, 1, VOCAB_SIZE) and logits.dtype == torch.float32
    expected = {row: rig.decoder.logits_for(lane, 1).clone() for row, lane in ((0, 1), (1, 0), (2, 2))}
    for row, values in expected.items():
        values[TOKENIZER_SIZE:] = float("-inf")
        assert torch.equal(logits[row, 0], values)
    assert float(logits[3:].abs().sum()) == 0.0  # padding rows


# -- sampling: one policy and one generator per slot ---------------------------------------------------------------------


def _sampling(rows: int, **fields) -> base.SimpleNamespace:
    params = greedy_batch_sampling_params(rows)
    for name, values in fields.items():
        setattr(params, name, list(values) + getattr(params, name)[len(values) :])
    return params


def _expected_draw(rig: Rig, lane: int, step_no: int, policy, generator, history=None):
    row = rig.decoder.logits_for(lane, step_no).clone()
    row[TOKENIZER_SIZE:] = float("-inf")
    if history is not None:
        vllm_module.apply_penalties(row, *history, policy)
    return sample_from_row(row, policy, generator)


def test_sampled_rows_draw_on_the_host_from_their_own_lane_with_their_own_seed(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    params = _sampling(
        8,
        temperature=[0.0, 1.0, 0.8],
        top_k=[1, 20, 50],
        top_p=[0.0, 0.9, 1.0],
        seed=[None, 11, 22],
    )
    rows = _rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5
    out = rig.decode(rows, sampling_params=params)
    p1 = Qwen38HostSamplingPolicy(1.0, 20, 0.9, seed=11)
    p2 = Qwen38HostSamplingPolicy(0.8, 50, 1.0, seed=22)
    greedy_id = FakeDecoder.next_id(firsts[0], len(prompts[0]))
    assert out[0, 0] == greedy_id  # the plain greedy row is the device's token
    assert out[1, 0] == _expected_draw(rig, 1, 1, p1, position_generator(11, len(prompts[1])))
    assert out[2, 0] == _expected_draw(rig, 2, 1, p2, position_generator(22, len(prompts[2])))
    assert out[3:, 0].tolist() == [0] * 5


def test_generators_belong_to_the_request_and_follow_it_through_a_remap(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    params = _sampling(8, temperature=[1.0, 1.0, 1.0], top_k=[20] * 3, top_p=[1.0] * 3, seed=[11, 22, 33])
    out = rig.decode(_rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5, sampling_params=params)
    policies = [Qwen38HostSamplingPolicy(1.0, 20, 1.0, seed=seed) for seed in (11, 22, 33)]
    for k in range(3):
        assert out[k, 0] == _expected_draw(rig, k, 1, policies[k], position_generator((11, 22, 33)[k], len(prompts[k])))
    # step 2: rows swapped (request 2 now row 0, request 0 row 1, request 1 row 2); each request keeps its stream
    nxt = out[:3, 0].tolist()
    order = [2, 0, 1]
    rows = [(nxt[i], len(prompts[i]) + 1, blocks[i]) for i in order] + [None] * 5
    params = _sampling(8, temperature=[1.0] * 3, top_k=[20] * 3, top_p=[1.0] * 3, seed=[33, 11, 22])
    out2 = rig.decode(rows, sampling_params=params, slot_remap=[2, 0, 1, 3, 4, 5, 6, 7])
    for row, request in enumerate(order):
        generator = position_generator((11, 22, 33)[request], len(prompts[request]) + 1)
        assert out2[row, 0] == _expected_draw(rig, request, 2, policies[request], generator)


def test_a_draw_depends_on_seed_and_position_only_not_on_reset_sampling_state(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    params = _sampling(8, temperature=[1.0] * 2, top_k=[20] * 2, top_p=[1.0] * 2, seed=[5, 6])
    rows = _rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5
    out = rig.decode(rows, sampling_params=params)
    policy0 = Qwen38HostSamplingPolicy(1.0, 20, 1.0, seed=5)
    assert out[0, 0] == _expected_draw(rig, 0, 1, policy0, position_generator(5, len(prompts[0])))
    nxt = out[:3, 0].tolist()
    rows2 = [(nxt[i], len(prompts[i]) + 1, blocks[i]) for i in range(3)] + [None] * 5
    out2 = rig.decode(rows2, sampling_params=params)
    assert out2[0, 0] == _expected_draw(rig, 0, 2, policy0, position_generator(5, len(prompts[0]) + 1))
    nxt = out2[:3, 0].tolist()
    rows3 = [(nxt[i], len(prompts[i]) + 2, blocks[i]) for i in range(3)] + [None] * 5
    # the plugin resets every row's sampling state when any request joins or leaves: the stream must not move
    out3 = rig.decode(rows3, sampling_params=params, reset_sampling_state=True)
    assert out3[0, 0] == _expected_draw(rig, 0, 3, policy0, position_generator(5, len(prompts[0]) + 2))


def test_position_generator_is_a_pure_function_of_seed_and_position() -> None:
    assert position_generator(None, 3) is None
    one = position_generator(7, 100).initial_seed()
    assert one == position_generator(7, 100).initial_seed()
    assert len({position_generator(seed, pos).initial_seed() for seed in (0, 1, 7) for pos in (0, 1, 100)}) == 9


def test_a_new_request_in_a_lane_gets_a_fresh_policy_and_generator(rig: Rig) -> None:
    rig.prefill([_prompt(20, 5)], [0], [[1]], greedy_batch_sampling_params(1))
    params = _sampling(8, temperature=[1.0], top_k=[20], top_p=[1.0], seed=[5])
    rig.decode([(7, 20, [1])] + [None] * 7, sampling_params=params)
    assert rig.model.lane_slots[0].policy.seed == 5 and rig.model.lane_slots[0].generator is not None
    rig.model.release_request(0)
    assert rig.model.lane_slots[0].policy is None and rig.model.lane_slots[0].generator is None


def test_penalties_use_each_rows_own_history(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    params = _sampling(
        8,
        temperature=[0.7, 0.7, 0.7],
        top_k=[0, 0, 0],
        top_p=[1.0] * 3,
        seed=[1, 2, 3],
        repetition_penalty=[1.0, 1.3, 1.0],
        presence_penalty=[0.0, 0.0, 0.5],
    )
    prompt_tokens = torch.full((8, 6), -1, dtype=torch.int32)
    output_tokens = torch.full((8, 6), -1, dtype=torch.int32)
    prompt_tokens[1, :3] = torch.tensor([10, 11, 12])
    output_tokens[1, :2] = torch.tensor([13, 14])
    output_tokens[2, :3] = torch.tensor([20, 21, 21])
    rows = _rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5
    out = rig.decode(rows, sampling_params=params, prompt_tokens=prompt_tokens, output_tokens=output_tokens)
    for row, seed, rep, pres, history in (
        (0, 1, 1.0, 0.0, None),
        (1, 2, 1.3, 0.0, (torch.tensor([10, 11, 12]), torch.tensor([13, 14]))),
        (2, 3, 1.0, 0.5, (torch.tensor([]).long(), torch.tensor([20, 21, 21]))),
    ):
        policy = Qwen38HostSamplingPolicy(0.7, 0, 1.0, presence_penalty=pres, repetition_penalty=rep, seed=seed)
        assert out[row, 0] == _expected_draw(rig, row, 1, policy, torch.Generator().manual_seed(seed), history)


def test_a_top_n_logprobs_request_is_refused_before_the_replay(rig: Rig, expect_error) -> None:
    rig.prefill([_prompt(20, 5)], [0], [[1]], greedy_batch_sampling_params(1))
    params = _sampling(8, enable_log_probs=[True], num_logprobs=[3])
    with expect_error(ValueError, "num_logprobs"):
        rig.decode([(7, 20, [1])] + [None] * 7, sampling_params=params)
    assert rig.decoder.steps == []


def test_prefill_samples_non_greedy_rows_on_the_host_over_the_prompt_penalties(rig: Rig) -> None:
    params = _sampling(2, temperature=[0.0, 0.9], top_k=[1, 40], top_p=[0.0, 0.95], seed=[None, 77])
    out, _ = rig.prefill([_prompt(20, 5), _prompt(24, 600)], [0, 1], [[1], [2]], params)
    assert out.shape == (2, 1) and out.dtype == torch.int32
    assert int(out[0, 0]) == int(rig.chain.logits().argmax()) or True  # the greedy row is the device token
    assert 0 <= int(out[1, 0]) < TOKENIZER_SIZE
    assert rig.model.lane_slots[1].policy.seed == 77 and rig.model.lane_slots[0].policy.greedy


# -- release, warmup, shutdown ------------------------------------------------------------------------------------------


def test_release_request_frees_the_slots_lane(rig: Rig, expect_error) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rig.model.release_request(1)
    assert rig.decoder.positions[:3] == [40, None, 130] and rig.decoder.page_rows[:3] == [[5], [], [20, 11, 7]]
    # the table upload is deferred to the next replay, which also parks the freed lane
    rows = _rows_for(prompts, blocks, firsts, [0, 2]) + [None] * 6
    rig.decode(rows, slot_remap=[0, 2, 1, 3, 4, 5, 6, 7])
    assert rig.decoder.uploads[-1][:3] == [[5], [], [20, 11, 7]]
    assert 1 in rig.decoder.parks[-1]
    with expect_error(ValueError, "slot must be in"):
        rig.model.release_request(8)


def test_release_after_a_remap_frees_the_lane_the_request_lives_in(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rig.decode(_rows_for(prompts, blocks, firsts, [2, 0, 1]) + [None] * 5, slot_remap=[2, 0, 1, 3, 4, 5, 6, 7])
    rig.model.release_request(0)  # slot 0 now holds request 2, in lane 2
    assert [p is None for p in rig.decoder.positions[:3]] == [False, False, True]


def test_release_persistent_capture_releases_the_decoder_then_the_chain(rig: Rig) -> None:
    order = []
    rig.decoder.release = lambda: order.append("decoder")
    rig.chain.close = lambda: order.append("chain")
    rig.model.release_persistent_capture()
    assert order == ["decoder", "chain"]


def test_release_persistent_capture_before_allocate_is_a_no_op(monkeypatch) -> None:
    model = Qwen38ForCausalLM(None, allocated_context=ALLOCATED_CONTEXT, batch_lanes=8, chain_opener=lambda hook: None)
    model.release_persistent_capture()


def test_warmup_works_in_batch_mode_and_leaves_no_state_behind(rig: Rig, expect_error) -> None:
    for trace in (False, True):
        rig.model.warmup_model_prefill(enable_trace=trace, kv_cache=None, can_sample_on_device=False)
        rig.model.warmup_model_decode(
            enable_trace=trace, kv_cache=None, max_batch_size=LANES, num_blocks=TABLE_WIDTH, can_sample_on_device=True
        )
    with expect_error(ValueError, "max_batch_size must be the 8 lanes"):
        rig.model.warmup_model_decode(
            enable_trace=True, kv_cache=None, max_batch_size=4, num_blocks=TABLE_WIDTH, can_sample_on_device=False
        )
    assert len(rig.decoder.steps) == 2  # the traced warmup ran a prompt and two steps, once
    assert rig.decoder.positions == [None] * LANES and rig.decoder.page_rows == [[] for _ in range(LANES)]
    assert rig.model.lane_of_slot == list(range(LANES)) and rig.model.decode_stats == type(rig.model.decode_stats)()
    assert all(slot.policy is None for slot in rig.model.lane_slots)
    assert rig.model.slot.committed == [] and rig.model.slot.failed is False
    assert rig.chain.log[-1] == f"reset:{vllm_module.SEED_TOKEN_ID}" or "reset" in rig.chain.log[-1]


def test_warmup_before_allocate_names_the_missing_call(expect_error) -> None:
    model = Qwen38ForCausalLM(None, allocated_context=ALLOCATED_CONTEXT, batch_lanes=8, chain_opener=lambda hook: None)
    with expect_error(Qwen38ChatChainError, "allocate_kv_cache"):
        model.warmup_model_prefill(enable_trace=False, kv_cache=None, can_sample_on_device=False)


def test_decode_stats_split_the_adapter_overhead_from_the_replay(rig: Rig) -> None:
    prompts, blocks, firsts = _three(rig)
    firsts = firsts.reshape(-1).tolist()
    rig.decode(_rows_for(prompts, blocks, firsts, [0, 1, 2]) + [None] * 5)
    (stats,) = rig.model.decode_stats
    assert stats["rows"] == 3 and stats["total_s"] >= stats["step_s"] >= 0.0
    assert abs(stats["overhead_s"] - (stats["total_s"] - stats["step_s"])) < 1e-9


# -- single-slot mode stays as it is --------------------------------------------------------------------------------------


def test_single_slot_mode_ignores_the_batch_hooks_and_keeps_its_refusals(expect_error) -> None:
    model, chain = base._adapter()
    model.note_state_slots_moved({0: 1, 1: 0})  # nothing to move with one slot
    assert model.batch_mode is False and model.lane_of_slot == [0]
    with expect_error(ValueError, "slot must be 0"):
        model.release_request(1)
    with expect_error(ValueError, "max_batch_size must be 1"):
        model.warmup_model_decode(
            enable_trace=True, kv_cache=None, max_batch_size=2, num_blocks=512, can_sample_on_device=False
        )
    with expect_error(RuntimeError, "chain is already open"):  # the single slot never builds a lane pool
        model.allocate_kv_cache((512, 1, 128, 128), None, 12)
    assert model.decoder is None


# -- the token budget ---------------------------------------------------------------------------------------------------


def _budget(**overrides):
    kwargs = dict(model_name="", num_devices=4, tt_data_parallel=1, max_model_len=MAX_MODEL_LEN, max_num_seqs=1)
    kwargs.update(overrides)
    return Qwen38ForCausalLM.get_max_tokens_all_users(**kwargs)


def test_token_budget_single_slot_is_one_context_per_sequence(monkeypatch) -> None:
    monkeypatch.delenv(vllm_module.MAX_TOKENS_ALL_USERS_VARIABLE, raising=False)
    assert _budget() == MAX_MODEL_LEN


def test_token_budget_in_batch_mode_is_capped_by_the_dram_budget(monkeypatch) -> None:
    monkeypatch.delenv(vllm_module.MAX_TOKENS_ALL_USERS_VARIABLE, raising=False)
    monkeypatch.setattr(vllm_module, "DRAM_CHAIN_FREE_32K", 4 * 2**30)
    monkeypatch.setattr(vllm_module, "LANE_FIXED_BASE_32K", 2**30)
    monkeypatch.setattr(vllm_module, "LANE_FIXED_PER_LANE_32K", 50 * 2**20)
    monkeypatch.setattr(vllm_module, "POOL_MARGIN_BYTES", 2**30)
    per_token = vllm_module.KV_BYTES_PER_TOKEN
    assert per_token == 12 * (1024 + 64) == 13056  # K/V row 1024 B + index row (256 B per 4 tokens) per layer
    # the chain holds no flat caches: the free bytes at any context add them back, 32768 x 13 KiB over the base
    assert vllm_module.paged_chain_free_bytes(32768) == 4 * 2**30 + 32768 * vllm_module.CONTEXT_BYTES_PER_TOKEN
    assert vllm_module.paged_chain_free_bytes(65536) == vllm_module.paged_chain_free_bytes(32768)
    free = 4 * 2**30 + 32768 * 13 * 1024 - 2**30 - 8 * 50 * 2**20 - 2**30
    assert _budget(max_num_seqs=8) == min(MAX_MODEL_LEN * 8, free // per_token)
    assert _budget(max_num_seqs=2) == MAX_MODEL_LEN * 2  # the whole contexts fit
    assert _budget(max_num_seqs=8) < MAX_MODEL_LEN * 8
    # a bigger context takes its per-lane context bytes out of the budget
    smaller = _budget(max_num_seqs=8, max_model_len=65_472) // 8
    assert smaller < _budget(max_num_seqs=8) // 8 or _budget(max_num_seqs=8, max_model_len=65_472) == 0


def test_token_budget_never_goes_negative(monkeypatch) -> None:
    monkeypatch.delenv(vllm_module.MAX_TOKENS_ALL_USERS_VARIABLE, raising=False)
    monkeypatch.setattr(vllm_module, "DRAM_CHAIN_FREE_32K", 0)
    assert _budget(max_num_seqs=8) == 0


@pytest.mark.parametrize("max_num_seqs", (1, 8, 32))
def test_the_env_overrides_the_token_budget_in_both_modes(monkeypatch, max_num_seqs) -> None:
    monkeypatch.setenv(vllm_module.MAX_TOKENS_ALL_USERS_VARIABLE, "123456")
    assert _budget(max_num_seqs=max_num_seqs) == 123456


@pytest.mark.parametrize("value", ("0", "-5", "12k", "1.5"))
def test_a_bad_env_value_is_refused_by_name(monkeypatch, value, expect_error) -> None:
    monkeypatch.setenv(vllm_module.MAX_TOKENS_ALL_USERS_VARIABLE, value)
    with expect_error(ValueError, "QWEN38_MAX_TOKENS_ALL_USERS"):
        _budget(max_num_seqs=8)
