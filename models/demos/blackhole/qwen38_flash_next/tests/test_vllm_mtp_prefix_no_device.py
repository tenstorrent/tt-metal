# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``QWEN38_MAMBA_SPEC=1`` on the MTP adapter: prefix caching and chunked prefill (no device, no vLLM).

Pinned: the capabilities and the MambaSpec (``num_speculative_blocks`` = K) with the env on, unchanged with it off; the
per-layer allocation that opens the lanes chain with the lanes' warm hook and then the block programs' warm; the chain
primitives (the restore resets the MTP alignment layer, the session's restore, the forced step's known next token); the
driver's ``prefill_range`` (reset or restore, the checkpoint splits, the saves, the intermediate chunk that reads no
token); the adapter's rows in batch 1 and on the lanes (an intermediate chunk takes no lane, the last chunk imports the
chain's state, the hybrid single moves first, decode between chunks, refusals before any device call); the warm-up of the
restore path.  The chain, the lanes and the vLLM spec classes are fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_mambaspec_no_device as mamba_tests
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_mtp_no_device as mtp_tests
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_mambaspec_no_device import (  # noqa: F401  (fixtures)
    fake_vllm,
    pool_calls,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm_mtp as mtp_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38ChatChainError, Qwen38ChatSession
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import Qwen38MTPForCausalLM, Qwen38VLLMMTPDriver

fakes = mtp_tests.fakes
lane_fakes = mtp_tests.lane_fakes
spec_decode = mtp_tests.spec_decode
BLOCK = 1024
LK = mtp_tests.LK
LROWS = mtp_tests.LROWS
ALLOCATED = mtp_tests.ALLOCATED
NUM_BLOCKS = 16
STATE_BASE = 8  # the first state block of the tests' ``mamba_state`` (the KV blocks stay below it)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.ttnn import contracts

    monkeypatch.setenv("QWEN38_HOST_FAST", "0")
    monkeypatch.setenv("QWEN38_STEP_SPLIT", "0")
    monkeypatch.setenv("QWEN38_RING", "0")
    monkeypatch.setattr(contracts, "_RING", None)
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", str(LK))
    monkeypatch.setenv("QWEN38_MTP_HYBRID_SINGLE", "0")
    for name in ("QWEN38_VLLM_MTP_PLE_EARLY", "QWEN38_MAMBA_SPEC", "QWEN38_VLLM_CHUNKED_PREFILL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fakes, "K", LK)
    monkeypatch.setattr(mtp_tests, "K", LK)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("QWEN38_MAMBA_SPEC", "1")


# -- the fakes ---------------------------------------------------------------------------------------------------------


class RangeChain(mtp_tests.PagedChain):
    """The paged scripted chain with the QSA pools, the chunk driver, and the two state-block calls, all logged as
    ``chunk:<rows>@<start>``, ``restore:<position>``, ``save:<position>`` next to the chain's own entries."""

    def __init__(self) -> None:
        super().__init__()
        self.mtp.drafts = LK
        self.pools = {layer: mamba_tests.FakePool(layer) for layer in range(3, 48, 4)}
        self.fed_next: list[int | None] = []  # the next token of every MTP row the 1-row steps wrote
        self.chunks: list[tuple[int, int, int | None]] = []  # (start, rows, following_token)
        self.saves: list[tuple[int, dict, int]] = []
        self.restores: list[tuple[int, dict, int]] = []

    def write_mtp_next_token(self, next_token_id) -> None:
        self.fed_next.append(next_token_id)
        super().write_mtp_next_token(next_token_id)

    def restore_prefix_state(self, position, token_ids, layer_blocks, block_size):
        self.log.append(f"restore:{position}")
        self.restores.append((position, layer_blocks, block_size))
        self.inputs = [int(token) for token in token_ids[:position]]
        self.row = fakes._argmax(self.inputs)
        return (int(token_ids[position - 2]), int(token_ids[position - 1]))

    def save_prefix_state(self, position, layer_blocks, block_size) -> None:
        self.log.append(f"save:{position}")
        self.saves.append((position, layer_blocks, block_size))

    def chunk_prefill(self, token_ids, *, start_position, ple_context, forced_step, following_token=None, **hooks):
        ids = [int(token) for token in token_ids]
        assert start_position == len(self.inputs) and start_position % 32 == 0
        self.log.append(f"chunk:{len(ids)}@{start_position}")
        self.chunks.append((start_position, len(ids), following_token))
        self.inputs.extend(ids)
        return SimpleNamespace(
            position=start_position + len(ids),
            ple_context=(ids[-1], ids[-2]) if len(ids) > 1 else ple_context,
            stopped=None,
            timing=SimpleNamespace(alignment_steps=0),
        )


def _ids(length: int, base: int = 100) -> list[int]:
    return mtp_tests._prompt(length, base)


def _ops(chain: RangeChain, *names: str) -> list[str]:
    names = names or ("reset", "restore", "table", "chunk", "save", "read_token_row", "import")
    return [entry for entry in chain.log if entry.split(":")[0] in names]


def _state(rows: int = 1, checkpoints=None):
    return mamba_tests._mamba_state(base_id=STATE_BASE, rows=rows, checkpoints=checkpoints)


def _blocks(chain: RangeChain, row: int = 0) -> dict:
    return mamba_tests._expected_blocks(chain, base_id=STATE_BASE, row=row)


def _lanes_model(monkeypatch, lanes: int = 3, hybrid: bool = False, num_blocks: int = NUM_BLOCKS, events=None):
    """A lanes-mode adapter in MambaSpec mode, opened through ``allocate_kv_cache_per_layer`` on a fake lanes session and a
    ``RangeChain``; the chain-state import of the lanes driver is a record."""

    from models.demos.blackhole.qwen38_flash_next.demo import mtp_demo
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_lanes_session

    monkeypatch.setenv("QWEN38_MAMBA_SPEC", "1")
    monkeypatch.setenv("QWEN38_MTP_HYBRID_SINGLE", "1" if hybrid else "0")
    device = mtp_tests.PagedFakeLanes(lanes)
    chain = RangeChain()
    opened: list[dict] = []

    def opener(warm_hook, paging=None):
        opened.append({"warm_hook": warm_hook, "paging": paging})
        return chain

    monkeypatch.setattr(qwen38_lanes_session, "Qwen38LanesSession", lambda **kwargs: device)
    marker = (lambda chain_: chain_.log.append("lanes_warm")) if events is None else (lambda _: events.append("lanes"))
    monkeypatch.setattr(mtp_demo, "_lanes_warm_hook", lambda *args: marker)
    model = Qwen38MTPForCausalLM(None, allocated_context=ALLOCATED, batch_lanes=lanes, chain_opener=opener)
    model.kv_cache = model.allocate_kv_cache_per_layer(mamba_tests._entries(num_blocks, BLOCK))

    def import_chain_state(lane, position, ple_context):
        chain.log.append("import")
        device.imports.append((lane, position, ple_context, len(model.session.committed)))
        device.sequence[lane] = list(model.session.committed)
        device.positions[lane] = position
        device.last_rows[lane] = None

    model.lane_driver._import_chain_state = import_chain_state
    model.opened = opened
    chain.log.clear()
    return model, device, chain


def _single_model(monkeypatch, num_blocks: int = NUM_BLOCKS):
    monkeypatch.setenv("QWEN38_MAMBA_SPEC", "1")
    chain = RangeChain()
    opened: list[dict] = []

    def opener(warm_hook, paging=None):
        opened.append({"warm_hook": warm_hook, "paging": paging})
        return chain

    model = Qwen38MTPForCausalLM(None, allocated_context=ALLOCATED, chain_opener=opener)
    model.allocate_kv_cache_per_layer(mamba_tests._entries(num_blocks, BLOCK))
    model.opened = opened
    chain.log.clear()
    return model, chain


def _call(model, rows, slots, table, *, state=None, mask=None, width=None):
    """One ``prefill_forward``: ``rows`` = [(the row's ids up to its chunk end, its start)]."""

    width = width or max(len(ids) for ids, _ in rows)
    tokens = torch.zeros(len(rows), width, dtype=torch.int32)
    for i, (ids, _) in enumerate(rows):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    extra = {}
    if state is not None:
        extra["mamba_state"] = state
    if mask is not None:
        extra["intermediate_prefill_mask"] = mask
    return model.prefill_forward(
        tokens=tokens,
        page_table=table,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(ids) for ids, _ in rows],
        start_pos=[start for _, start in rows],
        empty_slots=slots,
        **extra,
    )


def _table(model, rows: dict[int, list[int]], height: int | None = None) -> torch.Tensor:
    return mtp_tests._page_table(model, rows, height or model.lanes)


def _first(out: torch.Tensor, row: int = 0) -> int:
    return int(out[row, 0].argmax())


def _driver(chain: RangeChain) -> Qwen38VLLMMTPDriver:
    return Qwen38VLLMMTPDriver(chain, Qwen38ChatSession(chain, None, allocated_context=ALLOCATED), LK)


def _range(driver, ids, *, start=0, blocks=None, checkpoints=(), final=True, pages=None):
    return driver.prefill_range(
        ids,
        start=start,
        page_ids=pages,
        layer_blocks=blocks,
        checkpoints=checkpoints,
        block_size=BLOCK,
        final=final,
    )


# -- capabilities and the spec -----------------------------------------------------------------------------------------


def test_capabilities_on(on):
    caps = Qwen38MTPForCausalLM.model_capabilities
    assert caps["supports_prefix_caching"] is True and caps["supports_chunked_prefill"] is True
    assert caps["accepts_intermediate_prefill_mask"] is True
    assert caps["supports_spec_decode"] is True and caps["spec_requirements"] == ["device_propose"]
    assert caps["spec_hidden_handoff"] == [] and caps["output_tokens_per_step"] == 1
    assert caps["supports_async_decode"] is False and caps["supports_sample_on_device"] is False


def test_capabilities_off_have_no_new_key():
    caps = Qwen38MTPForCausalLM.model_capabilities
    assert "accepts_intermediate_prefill_mask" not in caps
    assert caps["supports_prefix_caching"] is False and caps["supports_chunked_prefill"] is False


@pytest.mark.parametrize("drafts", (3, 4, 5))
def test_spec_blocks_follow_the_draft_env(drafts, on, fake_vllm, monkeypatch):
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", str(drafts))
    specs = Qwen38MTPForCausalLM.get_kv_cache_spec(mamba_tests._config())
    mambas = [spec for spec in specs.values() if isinstance(spec, mamba_tests.FakeMambaSpec)]
    assert len(mambas) == 36
    assert {spec.num_speculative_blocks for spec in mambas} == {drafts}
    assert {spec.num_prefill_checkpoint_blocks for spec in mambas} == {1}
    assert {spec.prefill_checkpoint_alignment for spec in mambas} == {BLOCK}
    attention = [spec for spec in specs.values() if isinstance(spec, mamba_tests.FakeFullAttentionSpec)]
    assert len(attention) == 12 and {spec.page_size_bytes for spec in attention} == {1088 * BLOCK}


def test_plain_class_speculative_blocks_stay_zero(on, fake_vllm, monkeypatch):
    assert Qwen38ForCausalLM._mamba_spec_speculative_blocks() == 0
    specs = Qwen38ForCausalLM.get_kv_cache_spec(mamba_tests._config())
    assert {s.num_speculative_blocks for s in specs.values() if isinstance(s, mamba_tests.FakeMambaSpec)} == {0}
    monkeypatch.setattr(Qwen38ForCausalLM, "_mamba_speculative_blocks", 2)
    assert Qwen38ForCausalLM._mamba_spec_speculative_blocks() == 2


# -- the allocation ----------------------------------------------------------------------------------------------------


def test_lanes_per_layer_allocation_opens_the_lanes_chain(monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3)
    (opened,) = model.opened
    assert model.chain is chain and model.lanes_session is device and model.decoder is None
    assert model.slot_qsa_layer == {j: 4 * j + 3 for j in range(12)}
    assert model.gdn_slot == {layer: layer // 4 for layer in range(48) if layer % 4 != 3}
    assert model.kv_cache["block_size"] == BLOCK and model.kv_cache["num_blocks"] == NUM_BLOCKS
    assert set(model.kv_cache) == {"slot_qsa_layer", "gdn_slot", "block_size", "num_blocks"}
    assert opened["paging"] is model.paging and model.paging.num_pages == NUM_BLOCKS
    assert (model.block_size, model.num_blocks) == (BLOCK, NUM_BLOCKS)
    assert model.lane_driver.paged and model.driver is not None and model.session is not None


def test_lanes_warm_hook_runs_the_block_warm_after_the_lanes_hook(monkeypatch, pool_calls):
    monkeypatch.setattr(vllm_module.ttnn, "synchronize_device", lambda mesh: pool_calls.append(("sync", mesh)))
    model, _, _ = _lanes_model(monkeypatch, 3, events=pool_calls)
    hook = model.opened[0]["warm_hook"]
    chain = mamba_tests._warm_chain(pool_calls)
    hook(chain)
    gdn = {layer: (f"kv{4 * (layer // 4) + 3}", 1) for layer in range(48) if layer % 4 != 3}
    assert pool_calls[:3] == [
        "lanes",
        ("save", chain.state, gdn, BLOCK, 0),
        ("restore", chain.state, gdn, BLOCK, 0),
    ]
    assert [call[0] for call in pool_calls[3:-1]].count("zero") == 12 and pool_calls[-1] == ("sync", "mesh")
    count = len(pool_calls)
    hook(chain)  # the block programs compile once; the lanes' allocation is the hook's
    assert pool_calls[count:] == ["lanes"]


def test_lanes_per_layer_allocation_off_is_unchanged(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.demo import mtp_demo
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_lanes_session

    device = mtp_tests.PagedFakeLanes(3)
    chain = RangeChain()
    opened: list[dict] = []
    monkeypatch.setattr(qwen38_lanes_session, "Qwen38LanesSession", lambda **kwargs: device)
    monkeypatch.setattr(mtp_demo, "_lanes_warm_hook", lambda *args: lambda chain_: chain_.log.append("lanes_warm"))
    model = Qwen38MTPForCausalLM(
        None,
        allocated_context=ALLOCATED,
        batch_lanes=3,
        chain_opener=lambda hook, paging=None: opened.append(hook) or chain,
    )
    shape = (NUM_BLOCKS, 1, 128, 256)
    model.allocate_kv_cache_per_layer([(shape, torch.bfloat16, i) for i in range(12)])
    assert model._block_warm_hook() is None and model.slot_qsa_layer == {} and model.gdn_slot == {}
    assert (model.block_size, model.num_blocks) == (128, NUM_BLOCKS) and model.lanes_session is device
    chain.log.clear()
    opened[0](SimpleNamespace(mesh="mesh", log=chain.log))
    assert chain.log == ["lanes_warm"]  # no block part
    model.zero_kv_cache_blocks(kv_cache=None, block_ids=[3])
    model.copy_kv_cache_blocks(kv_cache=None, copies=[(1, 2)])


def test_batch_one_per_layer_allocation_passes_the_block_warm(monkeypatch):
    model, chain = _single_model(monkeypatch)
    (opened,) = model.opened
    assert callable(opened["warm_hook"]) and opened["paging"] is model.paging
    assert model.chain is chain and model.driver is not None and model.slot_qsa_layer[0] == 3


# -- the chain and the session ----------------------------------------------------------------------------------------


def _shell_chain(monkeypatch, with_mtp: bool):
    log: list = []
    monkeypatch.setattr(
        session_module, "ttnn", SimpleNamespace(synchronize_device=lambda mesh: log.append(("sync", mesh)))
    )
    model = SimpleNamespace(
        restore_boundary_state=lambda state, blocks, *, block_size, position: log.append(("restore", position))
    )

    class Shell(session_module.Qwen38TracedChain):
        mesh = "mesh"

    chain = object.__new__(Shell)
    chain.built_target = SimpleNamespace(model=model)
    chain.state = SimpleNamespace(position=SimpleNamespace(read=lambda: 2048))
    chain.mtp = None
    if with_mtp:
        layer = SimpleNamespace(reset_generic_state_inplace=lambda generic: log.append(("alignment_reset", generic)))
        chain.mtp = SimpleNamespace(
            alignment=SimpleNamespace(layer=layer, generic_state="alignment_state"), step_written=True
        )
    return chain, log


def test_restore_prefix_state_resets_the_alignment_layer(monkeypatch):
    chain, log = _shell_chain(monkeypatch, with_mtp=True)
    context = chain.restore_prefix_state(2048, _ids(3000), {}, BLOCK)
    assert context == (_ids(3000)[2046], _ids(3000)[2047])
    assert log == [("restore", 2048), ("alignment_reset", "alignment_state"), ("sync", "mesh")]
    assert chain.mtp.step_written is False


def test_restore_prefix_state_of_a_plain_chain_has_no_alignment_call(monkeypatch):
    chain, log = _shell_chain(monkeypatch, with_mtp=False)
    chain.restore_prefix_state(2048, _ids(3000), {}, BLOCK)
    assert log == [("restore", 2048), ("sync", "mesh")]


def test_session_restore_prefix_sets_the_host_state():
    chain = RangeChain()
    session = Qwen38ChatSession(chain, None, allocated_context=ALLOCATED)
    ids = _ids(3000)
    session.committed, session.snapshot, session.row_unconsumed = [1, 2, 3], object(), True
    session.last_finish, session.row_token, session.committed_vision = "stop", 5, ((1, 2, "x"),)
    session.restore_prefix(2048, ids, {}, BLOCK)
    assert session.committed == ids[:2048] and session.ple_context == (ids[2046], ids[2047])
    assert session.snapshot is None and session.last_finish is None and session.row_unconsumed is False
    assert session.row_token is None and session.committed_vision == ()
    assert chain.log == ["restore:2048"]
    assert session.committed is not ids


def test_session_prefill_last_next():
    chain = RangeChain()
    session = Qwen38ChatSession(chain, None, allocated_context=ALLOCATED)
    session._prefill([11, 22, 33], None, None, last_next=44)
    assert chain.fed_next == [22, 33, 44]
    chain.fed_next.clear()
    session._prefill([55, 66], None, None)
    assert chain.fed_next == [66, None]


# -- the driver's prefill_range --------------------------------------------------------------------------------------


def test_prefill_range_cold_unaligned():
    chain = RangeChain()
    driver = _driver(chain)
    ids = _ids(10)
    token = _range(driver, ids, pages=[1])
    assert _ops(chain) == ["reset", "table:(1,)", "read_token_row"]
    assert chain.chunks == [] and chain.saves == [] and chain.restores == []
    assert token == fakes._argmax(ids) and driver.session.committed == ids
    assert chain.fed_next == ids[1:] + [None]


def test_prefill_range_cold_aligned_final_saves_at_the_end():
    chain = RangeChain()
    driver = _driver(chain)
    ids = _ids(2048)
    blocks = _blocks(chain)
    token = _range(driver, ids, blocks=blocks, pages=[1, 2])
    assert _ops(chain) == ["reset", "table:(1, 2)", "chunk:2047@0", "save:2048", "read_token_row"]
    assert chain.chunks == [(0, 2047, ids[2047])]
    assert chain.saves == [(2048, blocks, BLOCK)] and chain.fed_next == [None]
    assert token == fakes._argmax(ids)


def test_prefill_range_restore_order():
    chain = RangeChain()
    driver = _driver(chain)
    ids = _ids(3000)
    blocks = _blocks(chain)
    token = _range(driver, ids, start=1024, blocks=blocks, pages=[1, 2, 3])
    assert _ops(chain) == ["restore:1024", "table:(1, 2, 3)", "chunk:1975@1024", "read_token_row"]
    assert "reset" not in chain.log and chain.restores == [(1024, blocks, BLOCK)]
    assert chain.chunks == [(1024, 1975, ids[2999])] and chain.saves == []  # 3000 is not a block boundary
    assert driver.session.committed == ids and token == fakes._argmax(ids)


def test_prefill_range_checkpoint_split_feeds_the_exact_next_token():
    chain = RangeChain()
    driver = _driver(chain)
    ids = _ids(5000)
    ckpt = {layer: (pool, 9) for layer, (pool, _) in _blocks(chain).items()}
    _range(driver, ids, checkpoints=[(4096, ckpt)], pages=[1, 2, 3, 4, 5])
    assert _ops(chain) == [
        "reset",
        "table:(1, 2, 3, 4, 5)",
        "chunk:4095@0",
        "save:4096",
        "chunk:903@4096",
        "read_token_row",
    ]
    assert chain.chunks == [(0, 4095, ids[4095]), (4096, 903, ids[4999])]
    assert chain.fed_next == [ids[4096], None]  # the part's last row knows its next token; the last part's does not
    assert chain.saves == [(4096, ckpt, BLOCK)]  # none at 5000


def test_prefill_range_intermediate_returns_no_token():
    chain = RangeChain()
    driver = _driver(chain)
    ids = _ids(2048)
    blocks = _blocks(chain)
    assert _range(driver, ids, blocks=blocks, final=False) is None
    assert chain.saves == [(2048, blocks, BLOCK)] and "read_token_row" not in chain.log
    assert driver.session.committed == ids


def test_intermediate_forced_step_uses_no_next_token():
    chain = RangeChain()
    driver = _driver(chain)
    _range(driver, _ids(1024), blocks=_blocks(chain), final=False)
    assert chain.fed_next == [None]  # ids[E] is not in the prefill tokens: the step takes the model's own argmax


def test_prefill_range_refusals_touch_no_chain_call(expect_error):
    chain = RangeChain()
    driver = _driver(chain)
    with expect_error(Qwen38ChatChainError, "intermediate chunk must end on a block boundary"):
        _range(driver, _ids(1500), blocks=_blocks(chain), final=False)
    with expect_error(Qwen38ChatChainError, "intermediate chunk must end on a block boundary"):
        _range(driver, _ids(2048), final=False)  # no state blocks to save into
    with expect_error(Qwen38ChatChainError, "block boundary with the state blocks"):
        _range(driver, _ids(3000), start=1024)
    with expect_error(Qwen38ChatChainError, "block boundary with the state blocks"):
        _range(driver, _ids(3000), start=1500, blocks=_blocks(chain))
    assert chain.log == []


# -- the adapter: batch 1 ----------------------------------------------------------------------------------------------


def test_batch_one_intermediate_then_last_chunk(monkeypatch):
    model, chain = _single_model(monkeypatch)
    ids = _ids(3000)
    table = _table(model, {0: [1, 2, 3]}, height=1)
    out, _ = _call(model, [(ids[:2048], 0)], [0], table, state=_state(), mask=[True])
    assert out.shape == (1, 1, VOCAB_SIZE) and _first(out) == 0 and model.slot.failed is False
    assert chain.saves == [(2048, _blocks(chain), BLOCK)] and "read_token_row" not in chain.log
    out, _ = _call(model, [(ids, 2048)], [0], table, state=_state(), mask=[False])
    assert _first(out) == fakes._argmax(ids) and out[0, 0, TOKENIZER_SIZE:].isinf().all()
    assert _ops(chain, "restore", "chunk") == ["chunk:2047@0", "restore:2048", "chunk:951@2048"]
    assert model.driver.session.committed == ids


def test_batch_one_greedy_contract_and_dummy_row(monkeypatch):
    model, chain = _single_model(monkeypatch)
    ids = _ids(2048)
    table = _table(model, {0: [1, 2]}, height=1)
    params = vllm_module.greedy_batch_sampling_params(1)
    out, _ = model.prefill_forward(
        tokens=torch.tensor([ids], dtype=torch.int32),
        page_table=table,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[2048],
        start_pos=[0],
        empty_slots=[0],
        sampling_params=params,
        mamba_state=_state(),
        intermediate_prefill_mask=[True],
    )
    assert out.dtype == torch.int32 and out.tolist() == [[0]]


def test_intermediate_chunk_must_end_on_a_block_boundary(monkeypatch, expect_error):
    model, chain = _single_model(monkeypatch)
    table = _table(model, {0: [1, 2]}, height=1)
    with expect_error(ValueError, "intermediate chunk ends at a block boundary"):
        _call(model, [(_ids(1500), 0)], [0], table, state=_state(), mask=[True])
    with expect_error(ValueError, "starts at a block boundary"):
        _call(model, [(_ids(3000), 1500)], [0], table, state=_state(), mask=[False])
    assert chain.log == []


def test_restore_without_blocks_is_refused(monkeypatch, expect_error):
    model, chain = _single_model(monkeypatch)
    table = _table(model, {0: [1, 2, 3]}, height=1)
    with expect_error(ValueError, "mamba_state is missing"):
        _call(model, [(_ids(3000), 1024)], [0], table)
    assert chain.log == []


def test_page_ids_use_the_chunk_end(monkeypatch, expect_error):
    model, chain = _single_model(monkeypatch)
    ids = _ids(2048)
    wide = _table(model, {0: [1, 2, 3, 4, 5]}, height=1)  # the request's table holds more blocks than the chunk needs
    _call(model, [(ids, 0)], [0], wide, state=_state(), mask=[True], width=3000)
    assert chain.page_table_ids == (1, 2, 3, 4, 5)
    short = _table(model, {0: [1]}, height=1)
    with expect_error(ValueError, "needs 2 blocks"):
        _call(model, [(ids, 0)], [0], short, state=_state(), mask=[True], width=3000)


def test_checkpoint_outside_the_chunk_is_refused(monkeypatch, expect_error):
    model, chain = _single_model(monkeypatch)
    ids = _ids(3000)
    table = _table(model, {0: [1, 2, 3]}, height=1)
    for n in (1024, 3000, 1500):  # at the start, at the end, off the block grid
        with expect_error(ValueError, "checkpoint position"):
            _call(model, [(ids, 1024)], [0], table, state=_state(checkpoints=[[(n, 12)]]), mask=[False])
    assert chain.log == []
    _call(model, [(ids, 1024)], [0], table, state=_state(checkpoints=[[(2048, 12)]]), mask=[False])
    assert [save[0] for save in chain.saves] == [2048]
    assert chain.saves[0][1] == {layer: (pool, 12 + layer % 4) for layer, (pool, _) in _blocks(chain).items()}


# -- the adapter: lanes ------------------------------------------------------------------------------------------------


def test_prefill_lanes_intermediate_row_takes_no_lane(monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3)
    table = _table(model, {0: [1, 2]})
    out, rope = _call(model, [(_ids(2048), 0)], [1], table, state=_state(), mask=[True])
    assert out.shape == (1, 1, VOCAB_SIZE) and _first(out) == 0 and rope.shape == (1,)
    assert model.lane_driver.runs == [None] * 3 and model.lane_of_slot == [0, 1, 2]
    assert not device.imports and not _kinds(device, "set_pages", "admit_blocks")
    assert chain.saves == [(2048, _blocks(chain), BLOCK)] and "read_token_row" not in chain.log
    assert model.mtp_hybrid_slot is None


def _kinds(device, *names):
    return mtp_tests._kinds(device, *names)


def test_prefill_lanes_final_chunk_imports_from_the_chain_without_a_second_prefill(monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3)
    ids = _ids(3000)
    _call(model, [(ids[:2048], 0)], [1], _table(model, {0: [1, 2]}), state=_state(), mask=[True])
    out, _ = _call(model, [(ids, 2048)], [1], _table(model, {0: [1, 2, 3]}), state=_state(), mask=[False])
    token = fakes._argmax(ids)
    assert _first(out) == token
    assert chain.chunks == [(0, 2047, ids[2047]), (2048, 951, ids[2999])]  # one chain prefill per call
    assert device.imports == [(0, 3000, (ids[2999], ids[2998]), 3000)]  # lane 0: the lowest free lane
    assert model.lane_of_slot == [1, 0, 2] and model.lane_driver.runs[0].pending == token
    assert _kinds(device, "set_pages") == [("set_pages", 0, [1, 2, 3])]
    assert not _kinds(device, "admit_blocks") and model.lane_driver.holders == [0]
    assert _ops(chain, "reset", "restore") == ["reset", "restore:2048"]


def test_prefill_lanes_hit_row(monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3)
    ids = _ids(2048)
    out, _ = _call(model, [(ids, 1024)], [2], _table(model, {0: [1, 2]}), state=_state(), mask=[False])
    assert _ops(chain) == ["restore:1024", "table:(1, 2)", "chunk:1023@1024", "save:2048", "read_token_row", "import"]
    assert device.imports == [(0, 2048, (ids[2047], ids[2046]), 2048)] and _first(out) == fakes._argmax(ids)
    assert model.lane_of_slot == [2, 1, 0]


@pytest.mark.parametrize("start", (0, 1024))
def test_chunk_continuation_moves_the_hybrid_first(start, monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3, hybrid=True)
    _call(model, [(_ids(10), 0)], [0], _table(model, {0: [1]}))
    assert model.mtp_hybrid_slot == 0 and not device.imports
    chain.log.clear()
    ids = _ids(2048, 900)
    _call(model, [(ids, start)], [1], _table(model, {0: [2, 3]}), state=_state(), mask=[True])
    assert model.mtp_hybrid_slot is None and device.imports[0][:2] == (0, 10)
    after = "restore:1024" if start else "reset"
    assert chain.log.index("import") < chain.log.index(after)
    assert model.lane_driver.holders == [0] and model.lane_of_slot[1] == 1  # the hybrid took lane 0; B has none


def test_lone_final_chunk_starts_a_hybrid_and_a_lone_intermediate_does_not(monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3, hybrid=True)
    ids = _ids(3000)
    _call(model, [(ids[:2048], 0)], [1], _table(model, {0: [1, 2]}), state=_state(), mask=[True])
    assert model.mtp_hybrid_slot is None and model.lane_driver.holders == []
    out, _ = _call(model, [(ids, 2048)], [1], _table(model, {0: [1, 2, 3]}), state=_state(), mask=[False])
    assert model.mtp_hybrid_slot == 1 and model.mtp_hybrid_pending == _first(out) == fakes._argmax(ids)
    assert model.mtp_hybrid_pages == [1, 2, 3] and not device.imports and model.lane_driver.holders == []


def test_decode_runs_between_chunks(monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3)
    prompt = _ids(40)
    out, _ = _call(model, [(prompt, 0)], [0], _table(model, {0: [1]}))
    committed, drafts, accepted = [_first(out)], [], 1
    decode_table = _table(model, {0: [1, 2], 1: [3, 4, 5]})
    first = committed[0]

    def step():
        nonlocal committed, drafts, accepted
        position = len(prompt) + len(committed) - 1
        _, new, offered = mtp_tests._hybrid_step(model, 3, 0, committed, position, drafts, accepted, decode_table)
        committed, accepted = committed + new, len(new)
        drafts = offered.draft_token_ids[0, : int(offered.num_valid[0])].tolist()

    step()
    ids = _ids(1500, 900)
    _call(model, [(ids[:1024], 0)], [1], _table(model, {0: [3, 4]}), state=_state(), mask=[True])
    step()
    step()
    assert model.lane_driver.holders == [0] and model.lane_of_slot == [0, 1, 2]
    out, _ = _call(model, [(ids, 1024)], [1], _table(model, {0: [3, 4]}), state=_state(), mask=[False])
    assert model.lane_driver.holders == [0, 1] and sorted(model.lane_of_slot) == [0, 1, 2]
    assert committed[1:] == mtp_tests._lane_stream(prompt + [first], len(committed) - 1)


def test_aborted_chunked_request_release_is_harmless(monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3)
    _call(model, [(_ids(40), 0)], [0], _table(model, {0: [1]}))
    run = model.lane_driver.runs[0]
    _call(model, [(_ids(1024, 900), 0)], [1], _table(model, {0: [3]}), state=_state(), mask=[True])
    model.release_request(1)
    assert model.lane_driver.runs[0] is run and model.lane_driver.holders == [0]
    assert model.lane_of_slot == [0, 1, 2] and not model.batch_failed


def test_two_rows_in_one_call(monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3)
    first, second = _ids(1024, 100), _ids(40, 500)
    out, rope = _call(
        model,
        [(first, 0), (second, 0)],
        [0, 2],
        _table(model, {0: [1], 1: [2]}),
        state=_state(rows=2),
        mask=[True, False],
    )
    assert out.shape == (2, 1, VOCAB_SIZE) and rope.shape == (2,)
    assert _first(out, 0) == 0 and _first(out, 1) == fakes._argmax(second)
    assert chain.saves == [(1024, _blocks(chain), BLOCK)]
    assert device.imports == [(0, 40, (second[39], second[38]), 40)] and model.lane_of_slot == [2, 1, 0]
    assert _ops(chain, "reset", "save", "import") == ["reset", "save:1024", "reset", "import"]


def test_off_mode_prefill_is_unchanged(monkeypatch, expect_error):
    monkeypatch.setattr(
        Qwen38VLLMMTPDriver, "prefill_range", lambda *args, **kwargs: pytest.fail("prefill_range ran with the env off")
    )
    model, device = mtp_tests._lanes_model(monkeypatch, 2)
    ids = _ids(6)
    with expect_error(ValueError, "prefill must start at position 0"):
        model.prefill_forward(
            tokens=torch.tensor([ids], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[6],
            start_pos=[1024],
            empty_slots=[0],
        )
    out, _ = mtp_tests._hybrid_prefill(model, ids, 0)
    assert device.calls[0] == ("admit", 0) and out.shape == (1, 1, VOCAB_SIZE)
    single, chain = mtp_tests._model()
    with expect_error(ValueError, "prefill must start at position 0"):
        single.prefill_forward(
            tokens=torch.tensor([ids], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[6],
            start_pos=[1024],
            empty_slots=[0],
        )
    assert chain.log == []


# -- the warm-up -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("hybrid", (False, True))
def test_warmup_runs_the_restore_path_in_mamba_mode(hybrid, monkeypatch):
    model, device, chain = _lanes_model(monkeypatch, 3, hybrid=hybrid)
    model.warmup_model_decode(
        enable_trace=True, kv_cache=None, max_batch_size=3, num_blocks=NUM_BLOCKS, can_sample_on_device=False
    )
    length = BLOCK + len(mtp_module.WARMUP_PROMPT_TOKEN_IDS)
    assert chain.restores[0][0] == BLOCK and chain.saves[0][0] == BLOCK
    # the state blocks (3 groups) follow the 2 KV blocks of the prefix request, none inside the warm table's blocks
    state_blocks = {block for _, block in chain.saves[0][1].values()}
    assert state_blocks == {3, 4, 5} and {block for _, block in chain.restores[0][1].values()} == state_blocks
    assert any(imp[1] == length for imp in device.imports)
    assert model.lane_driver.holders == [] and model.mtp_hybrid_slot is None and not model.batch_failed
    assert "reset" in chain.log


def test_warmup_prefix_needs_enough_blocks(monkeypatch, expect_error):
    model, device, chain = _lanes_model(monkeypatch, 3, num_blocks=5)
    with expect_error(ValueError, "the prefix warm-up needs 6 blocks, the pool has 5"):
        model.warmup_model_decode(
            enable_trace=True, kv_cache=None, max_batch_size=3, num_blocks=5, can_sample_on_device=False
        )
    assert chain.restores == []


def test_warmup_off_runs_no_restore(monkeypatch):
    model, device, chain = mtp_tests._paged_lanes_model(monkeypatch, 3, hybrid=False)
    model.warmup_model_decode(
        enable_trace=True, kv_cache=None, max_batch_size=3, num_blocks=64, can_sample_on_device=False
    )
    assert not any(entry.startswith(("restore", "save")) for entry in chain.log)
