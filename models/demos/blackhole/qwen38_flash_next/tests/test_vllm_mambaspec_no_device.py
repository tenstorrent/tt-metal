# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``QWEN38_MAMBA_SPEC=1``: the plain adapter's vLLM-side hybrid KV cache contract (no device, no vLLM).

Pinned: off = today's behaviour (no spec, uniform allocation, no-op block hooks, capabilities); the 48 specs (names the
plugin parses, the attention page of 1088 x block_size bytes, the one MambaSpec of 905,216 bytes padded to it); the block-size
refusals; ``allocate_kv_cache_per_layer`` (the slot maps from the layer-ordered entries, the chain open exactly as
``allocate_kv_cache`` opens it); the zero / copy hooks (every QSA layer's pools, pairs in list order split into runs the op
takes, padding pairs skipped, block 0 refused); the capabilities (the MTP class's follow the env); ``start_pos > 0`` without the
state on the chain.
The vLLM spec classes are replaced by fakes with the page-size rules of vLLM 0.30.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests import test_qwen38_vllm_no_device as base_tests
from models.demos.blackhole.qwen38_flash_next.tests import test_vllm_batch_adapter_no_device as batch_tests
from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import ALLOCATED_CONTEXT
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38ChatChainError
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    MAMBA_STATE_BYTES,
    Qwen38ForCausalLM,
    split_block_copies,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import Qwen38MTPForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging, paged_table_width

BLOCK = 1024
NUM_BLOCKS = 12
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")  # the plugin's _LAYER_NAME_RE


@dataclass(frozen=True)
class FakeFullAttentionSpec:
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: Any

    @property
    def page_size_bytes(self) -> int:  # vLLM's AttentionSpec: K and V
        return (
            2 * self.block_size * self.num_kv_heads * self.head_size * torch.empty((), dtype=self.dtype).element_size()
        )


@dataclass(frozen=True)
class FakeMambaSpec:
    block_size: int
    shapes: tuple
    dtypes: tuple
    page_size_padded: int | None = None
    mamba_type: Any = None
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0
    num_prefill_checkpoint_blocks: int = 0
    prefill_checkpoint_alignment: int | None = None

    @property
    def page_size_bytes(self) -> int:
        state = sum(
            torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= state
            return self.page_size_padded
        return state


@pytest.fixture(autouse=True)
def _line_fabric(monkeypatch):
    """The capabilities read the fabric choice; QWEN38_RING=auto would ask the cluster (a device) for its type."""

    from models.demos.blackhole.qwen38_flash_next.ttnn import contracts

    monkeypatch.setenv("QWEN38_RING", "0")
    monkeypatch.setattr(contracts, "_RING", None)


@pytest.fixture
def fake_vllm(monkeypatch):
    interface = ModuleType("vllm.v1.kv_cache_interface")
    interface.FullAttentionSpec, interface.MambaSpec = FakeFullAttentionSpec, FakeMambaSpec
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.MambaAttentionBackendEnum = SimpleNamespace(GDN_ATTN="gdn")
    for name, module in (
        ("vllm.v1.kv_cache_interface", interface),
        ("vllm.v1.attention.backends.registry", registry),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("QWEN38_MAMBA_SPEC", "1")


def _config(block_size: int = BLOCK, mode: str = "align") -> SimpleNamespace:
    return SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size, mamba_cache_mode=mode))


class FakePool:
    def __init__(self, layer: int) -> None:
        self.kv, self.index = f"kv{layer}", f"index{layer}"


class FakeChain:
    def __init__(self, layers=range(3, 48, 4)) -> None:
        self.pools = {layer: FakePool(layer) for layer in layers}


def _entries(num_blocks: int = NUM_BLOCKS, block_size: int = BLOCK, slot_of=lambda layer: layer // 4):
    """The plugin's per-layer list for the spec dict above: slot j = position j within its group."""

    out = []
    for layer in range(48):
        if layer % 4 == 3:
            out.append(((num_blocks, 1, block_size, 272), torch.bfloat16, slot_of(layer)))
        else:
            out.append((((MAMBA_STATE_BYTES // 4,),), (torch.float32,), slot_of(layer)))
    return out


def _model(chain=None) -> tuple[Qwen38ForCausalLM, FakeChain, list]:
    chain = FakeChain() if chain is None else chain
    opened: list = []

    def opener(hook, paging=None):
        opened.append((hook, paging))
        return chain

    return Qwen38ForCausalLM(None, allocated_context=ALLOCATED_CONTEXT, chain_opener=opener), chain, opened


def _slot_of(layer: int) -> int:
    return layer // 4  # 12 slots; each slot j is QSA layer 4j+3 and GDN layers 4j, 4j+1, 4j+2


def _open_model() -> tuple[Qwen38ForCausalLM, FakeChain, list]:
    model, chain, opened = _model()
    model.allocate_kv_cache_per_layer(_entries())
    return model, chain, opened


@pytest.fixture
def pool_calls(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        vllm_module,
        "pool_blocks",
        SimpleNamespace(
            zero_blocks=lambda kv, index, ids, block_size: calls.append(("zero", kv, index, list(ids), block_size)),
            copy_blocks=lambda kv, index, pairs, block_size: calls.append(("copy", kv, index, list(pairs), block_size)),
        ),
    )
    return calls


# -- off ---------------------------------------------------------------------------------------------------------------


def test_off_is_todays_behaviour(monkeypatch, fake_vllm, pool_calls) -> None:
    monkeypatch.delenv("QWEN38_MAMBA_SPEC", raising=False)
    assert Qwen38ForCausalLM.get_kv_cache_spec(_config()) is None
    capabilities = Qwen38ForCausalLM.model_capabilities
    assert capabilities["supports_prefix_caching"] is False
    model, chain, opened = _model()
    shape = (NUM_BLOCKS, 1, 128, 256)
    model.allocate_kv_cache_per_layer(
        [(shape, torch.bfloat16, i) for i in range(12)]
    )  # the uniform form: allocate_kv_cache
    assert opened and opened[0][1].page_size == 128
    assert model.slot_qsa_layer == {} and model.gdn_slot == {}
    model.zero_kv_cache_blocks(kv_cache=None, block_ids=[3])
    model.copy_kv_cache_blocks(kv_cache=None, copies=[(1, 2)])
    assert pool_calls == []


def test_off_refuses_non_uniform_per_layer_specs(monkeypatch, expect_error) -> None:
    monkeypatch.delenv("QWEN38_MAMBA_SPEC", raising=False)
    model, _, _ = _model()
    with expect_error(ValueError, "per-layer KV specs differ"):
        model.allocate_kv_cache_per_layer(
            [((NUM_BLOCKS, 1, 128, 256), torch.bfloat16, 0), ((NUM_BLOCKS, 1, 128, 272), torch.bfloat16, 1)]
        )


def test_mtp_follows_the_mamba_spec_env(monkeypatch, fake_vllm) -> None:
    monkeypatch.delenv("QWEN38_MAMBA_SPEC", raising=False)
    assert Qwen38MTPForCausalLM.get_kv_cache_spec(_config()) is None
    assert Qwen38MTPForCausalLM.model_capabilities["supports_prefix_caching"] is False
    monkeypatch.setenv("QWEN38_MAMBA_SPEC", "1")
    specs = Qwen38MTPForCausalLM.get_kv_cache_spec(_config())
    assert specs is not None and len(specs) == 48
    assert Qwen38MTPForCausalLM.model_capabilities["supports_prefix_caching"] is True


# -- the spec dict -----------------------------------------------------------------------------------------------------


def test_the_spec_dict(on, fake_vllm) -> None:
    specs = Qwen38ForCausalLM.get_kv_cache_spec(_config())
    assert len(specs) == 48
    indexes = [int(LAYER_RE.search(name).group(1)) for name in specs]
    assert indexes == list(range(48))
    attention_page = 1088 * BLOCK
    for name, spec in specs.items():
        layer = int(LAYER_RE.search(name).group(1))
        if layer % 4 == 3:
            assert isinstance(spec, FakeFullAttentionSpec)
            assert (spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype) == (BLOCK, 1, 272, torch.bfloat16)
            assert spec.page_size_bytes == attention_page
        else:
            assert isinstance(spec, FakeMambaSpec)
            assert spec.shapes == ((226304,),) and spec.dtypes == (torch.float32,)
            assert 226304 * 4 == MAMBA_STATE_BYTES == 905_216
            assert spec.page_size_bytes == attention_page and spec.block_size == BLOCK
            assert spec.mamba_cache_mode == "align" and spec.num_speculative_blocks == 0 and spec.mamba_type == "gdn"
            assert spec.num_prefill_checkpoint_blocks == 1 and spec.prefill_checkpoint_alignment == BLOCK
    assert len({id(spec) for spec in specs.values() if isinstance(spec, FakeMambaSpec)}) == 1


def test_the_page_follows_a_larger_block(on, fake_vllm) -> None:
    specs = Qwen38ForCausalLM.get_kv_cache_spec(_config(2048))
    assert {spec.page_size_bytes for spec in specs.values()} == {1088 * 2048}
    assert {spec.prefill_checkpoint_alignment for spec in specs.values() if isinstance(spec, FakeMambaSpec)} == {2048}


def test_a_subclass_sets_the_speculative_blocks(on, fake_vllm, monkeypatch) -> None:
    monkeypatch.setattr(Qwen38ForCausalLM, "_mamba_speculative_blocks", 3)
    specs = Qwen38ForCausalLM.get_kv_cache_spec(_config())
    assert {spec.num_speculative_blocks for spec in specs.values() if isinstance(spec, FakeMambaSpec)} == {3}


@pytest.mark.parametrize("block_size", (128, 512))
def test_a_small_block_is_refused(block_size, on, fake_vllm, expect_error) -> None:
    with expect_error(ValueError, "--block-size must be >= 1024"):
        Qwen38ForCausalLM.get_kv_cache_spec(_config(block_size))


@pytest.mark.parametrize("block_size", (1536, 8192, 0))
def test_a_block_outside_the_pool_form_is_refused(block_size, on, fake_vllm, expect_error) -> None:
    with expect_error(ValueError, "power of two"):
        Qwen38ForCausalLM.get_kv_cache_spec(_config(block_size))


def test_mode_none_is_refused(on, fake_vllm, expect_error) -> None:
    with expect_error(ValueError, "mamba_cache_mode 'align'"):
        Qwen38ForCausalLM.get_kv_cache_spec(_config(mode="none"))


# -- allocate_kv_cache_per_layer ---------------------------------------------------------------------------------------


def test_allocate_learns_the_slot_maps_and_opens_the_paged_single_slot(on) -> None:
    model, chain, opened = _model()
    kv_cache = model.allocate_kv_cache_per_layer(_entries())
    paging = Qwen38LanePaging(
        page_size=BLOCK, num_pages=NUM_BLOCKS, max_blocks=paged_table_width(ALLOCATED_CONTEXT, BLOCK)
    )
    assert len(opened) == 1 and opened[0][1] == paging and callable(opened[0][0]) and model.chain is chain
    assert (model.paging, model.block_size, model.num_blocks) == (paging, BLOCK, NUM_BLOCKS)
    assert model.slot_qsa_layer == {j: 4 * j + 3 for j in range(12)}
    assert model.gdn_slot == {layer: layer // 4 for layer in range(48) if layer % 4 != 3}
    assert kv_cache["block_size"] == BLOCK and kv_cache["num_blocks"] == NUM_BLOCKS


def test_allocate_maps_slots_by_tensor_idx_not_by_position(on) -> None:
    model, _, _ = _model()
    shuffled = lambda layer: (11 - layer // 4)  # noqa: E731  slot numbers in reverse order
    model.allocate_kv_cache_per_layer(_entries(slot_of=shuffled))
    assert model.slot_qsa_layer == {11 - j: 4 * j + 3 for j in range(12)}
    assert model.gdn_slot[0] == 11 and model.gdn_slot[45] == 0


def test_allocate_in_batch_mode_opens_the_lane_pool(on, monkeypatch) -> None:
    model = Qwen38ForCausalLM(
        None, allocated_context=ALLOCATED_CONTEXT, batch_lanes=8, chain_opener=lambda hook, paging=None: None
    )
    opens: list = []
    monkeypatch.setattr(model, "_open_batch", lambda **kwargs: opens.append(kwargs))
    model.allocate_kv_cache_per_layer(_entries())
    assert opens == [{"num_blocks": NUM_BLOCKS, "block_size": BLOCK}]
    assert model.slot_qsa_layer[0] == 3


def test_allocate_refusals(on, expect_error) -> None:
    model, _, opened = _model()
    with expect_error(ValueError, "48 per-layer entries"):
        model.allocate_kv_cache_per_layer(_entries()[:12])
    wrong = _entries()
    wrong[3] = ((NUM_BLOCKS, 1, BLOCK, 256), torch.bfloat16, 0)
    with expect_error(ValueError, "attention shape must be"):
        model.allocate_kv_cache_per_layer(wrong)
    wrong = _entries()
    wrong[0] = wrong[3]
    with expect_error(ValueError, "must be a GDN entry"):
        model.allocate_kv_cache_per_layer(wrong)
    wrong = _entries()
    wrong[0] = (wrong[0][0], wrong[0][1], 99)
    with expect_error(ValueError, "no attention layer backs"):
        model.allocate_kv_cache_per_layer(wrong)
    with expect_error(ValueError, "--block-size must be >= 1024"):
        model.allocate_kv_cache_per_layer(_entries(block_size=512))
    assert opened == []


# -- the open's warm hook ----------------------------------------------------------------------------------------------


class FakePosition:
    def __init__(self, value: int) -> None:
        self.value = value

    def read(self) -> int:
        return self.value

    def reset(self, position: int, rope_shift: int) -> None:
        assert rope_shift == 0
        self.value = position


class WarmModel:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def save_boundary_state(self, state, layer_blocks, *, block_size, position):
        self.calls.append(("save", state, dict(layer_blocks), block_size, position))

    def restore_boundary_state(self, state, layer_blocks, *, block_size, position):
        self.calls.append(("restore", state, dict(layer_blocks), block_size, position))
        state.position.reset(position, 0)  # the real restore ends the same way


def _warm_chain(calls: list, warm_position: int = 8) -> FakeChain:
    chain = FakeChain()
    chain.state, chain.mesh = SimpleNamespace(position=FakePosition(warm_position)), "mesh"
    chain.built_target = SimpleNamespace(model=WarmModel(calls))
    return chain


def test_the_warm_hook_compiles_every_block_program_once(on, pool_calls, monkeypatch) -> None:
    monkeypatch.setattr(vllm_module.ttnn, "synchronize_device", lambda mesh: pool_calls.append(("sync", mesh)))
    model, _, opened = _model()
    model.allocate_kv_cache_per_layer(_entries())
    hook = opened[0][0]
    chain = _warm_chain(pool_calls)
    hook(chain)
    hook(chain)
    gdn = {layer: (f"kv{4 * (layer // 4) + 3}", 1) for layer in range(48) if layer % 4 != 3}
    assert pool_calls[:2] == [("save", chain.state, gdn, BLOCK, 0), ("restore", chain.state, gdn, BLOCK, 0)]
    rest = pool_calls[2:]
    assert rest[-1] == ("sync", "mesh") and len(rest) == 2 * 12 + 1
    for i, layer in enumerate(range(3, 48, 4)):
        assert rest[2 * i] == ("zero", f"kv{layer}", f"index{layer}", [1], BLOCK)
        assert rest[2 * i + 1] == ("copy", f"kv{layer}", f"index{layer}", [(1, 2)], BLOCK)


def test_the_warm_hook_leaves_the_device_position_as_it_found_it(on, pool_calls, monkeypatch) -> None:
    monkeypatch.setattr(vllm_module.ttnn, "synchronize_device", lambda mesh: None)
    model, _, opened = _model()
    model.allocate_kv_cache_per_layer(_entries())
    chain = _warm_chain(pool_calls, warm_position=8)
    opened[0][0](chain)
    assert chain.state.position.read() == 8


def test_the_warm_hook_is_absent_when_off(pool_calls) -> None:
    model, _, opened = _model()
    assert model._block_warm_hook() is None


def test_the_batch_warm_hook_composes_the_block_warm(on, pool_calls, monkeypatch) -> None:
    monkeypatch.setattr(vllm_module.ttnn, "synchronize_device", lambda mesh: None)
    model, _, _ = _model()
    model.slot_qsa_layer = {j: 4 * j + 3 for j in range(12)}
    model.gdn_slot = {layer: layer // 4 for layer in range(48) if layer % 4 != 3}
    model.block_size = BLOCK
    hook = model._block_warm_hook()
    hook(_warm_chain(pool_calls))
    assert [c[0] for c in pool_calls].count("save") == 1 and [c[0] for c in pool_calls].count("copy") == 12


# -- the block hooks ---------------------------------------------------------------------------------------------------


def test_zero_hook_zeroes_every_qsa_layers_pools(on, pool_calls) -> None:
    model, chain, _ = _open_model()
    model.zero_kv_cache_blocks(kv_cache=None, block_ids=[5, 3, 5])
    assert pool_calls == [("zero", f"kv{layer}", f"index{layer}", [5, 3], BLOCK) for layer in range(3, 48, 4)]
    pool_calls.clear()
    model.zero_kv_cache_blocks(block_ids=[])
    assert pool_calls == []


def test_copy_hook_copies_in_every_pool_and_splits_overlapping_pairs(on, pool_calls) -> None:
    model, chain, _ = _open_model()
    model.copy_kv_cache_blocks(kv_cache=None, copies=[(1, 2), (3, 4), (2, 5), (6, 6)])
    runs = [[(1, 2), (3, 4)], [(2, 5)]]
    assert pool_calls == [
        ("copy", f"kv{layer}", f"index{layer}", run, BLOCK) for run in runs for layer in range(3, 48, 4)
    ]


def test_copy_hook_skips_padding_pairs_before_validation(on, pool_calls) -> None:
    model, chain, _ = _open_model()
    model.copy_kv_cache_blocks(kv_cache=None, copies=[(0, 0), (3, 3), (1, 2), (0, 0)])
    assert pool_calls == [("copy", f"kv{layer}", f"index{layer}", [(1, 2)], BLOCK) for layer in range(3, 48, 4)]
    pool_calls.clear()
    model.copy_kv_cache_blocks(kv_cache=None, copies=[(0, 0), (NUM_BLOCKS + 5, NUM_BLOCKS + 5)])
    assert pool_calls == []


def test_copy_hook_still_refuses_another_pair_with_block_zero(on, pool_calls, expect_error) -> None:
    model, _, _ = _open_model()
    for pair in ((0, 2), (2, 0)):
        with expect_error(ValueError, "null block"):
            model.copy_kv_cache_blocks(kv_cache=None, copies=[(0, 0), pair])
    assert pool_calls == []


def test_split_block_copies() -> None:
    assert split_block_copies([(1, 2), (3, 4)]) == [[(1, 2), (3, 4)]]
    assert split_block_copies([(1, 2), (2, 3)]) == [[(1, 2)], [(2, 3)]]  # a src that an earlier pair wrote
    assert split_block_copies([(1, 2), (3, 1)]) == [[(1, 2)], [(3, 1)]]  # a dst that an earlier pair read
    assert split_block_copies([(1, 2), (3, 2)]) == [[(1, 2)], [(3, 2)]]  # the same dst twice
    assert split_block_copies([(4, 4)]) == []


@pytest.mark.parametrize("hook", ("zero", "copy"))
def test_block_zero_and_out_of_range_are_refused(hook, on, pool_calls, expect_error) -> None:
    model, _, _ = _open_model()
    call = (
        (lambda ids: model.zero_kv_cache_blocks(block_ids=ids))
        if hook == "zero"
        else (lambda ids: model.copy_kv_cache_blocks(copies=[(ids[0], ids[-1])]))
    )
    with expect_error(ValueError, "null block"):
        call([0, 2] if hook == "zero" else [2, 0])
    with expect_error(ValueError, "outside the"):
        call([NUM_BLOCKS + 1, 2] if hook == "zero" else [2, NUM_BLOCKS])
    assert pool_calls == []


def test_the_hooks_need_the_open_chain(on, expect_error) -> None:
    model, _, _ = _model()
    with expect_error(RuntimeError, "allocate_kv_cache_per_layer runs first"):
        model.zero_kv_cache_blocks(block_ids=[1])


# -- capabilities and prefill ------------------------------------------------------------------------------------------


def test_capabilities(on, monkeypatch) -> None:
    capabilities = Qwen38ForCausalLM.model_capabilities
    assert capabilities["supports_prefix_caching"] is True and capabilities["supports_chunked_prefill"] is True
    monkeypatch.delenv("QWEN38_MAMBA_SPEC")
    monkeypatch.delenv("QWEN38_VLLM_CHUNKED_PREFILL", raising=False)
    capabilities = Qwen38ForCausalLM.model_capabilities
    assert capabilities["supports_prefix_caching"] is False and capabilities["supports_chunked_prefill"] is False


# -- prefill: the state blocks ------------------------------------------------------------------------------------------


class StateChain(base_tests.FakeChain):
    """The adapter's fake chain with the QSA pools and the two state-block calls recorded."""

    def __init__(self) -> None:
        super().__init__()
        self.pools = {layer: FakePool(layer) for layer in range(3, 48, 4)}

    def restore_prefix_state(self, position, token_ids, layer_blocks, block_size):
        self.log.append(f"restore:{position}")
        self.restores.append((position, list(token_ids), layer_blocks, block_size))
        self.inputs = [int(token) for token in token_ids[:position]]
        return (int(token_ids[position - 2]), int(token_ids[position - 1]))

    def save_prefix_state(self, position, layer_blocks, block_size):
        self.log.append(f"save:{position}")
        self.saves.append((position, layer_blocks, block_size))

    restores: list = []
    saves: list = []


def _state_model():
    chain = StateChain()
    chain.restores, chain.saves = [], []
    model, _, _ = _model(chain)
    model.allocate_kv_cache_per_layer(_entries())
    return model, chain


def _group_of_stride(layer: int) -> int:
    return layer % 4  # group g holds GDN layers 4j+g


def _group_of_run(layer: int) -> int:
    return (layer - layer // 4) // 12  # vLLM's grouping: runs of 12 GDN layers in layer order


def _mamba_state(
    base_id: int = 3,
    rows: int = 1,
    group_of=_group_of_stride,
    checkpoints: list[list[tuple[int, int]]] | None = None,
    names=None,
):
    """The plugin's ``mamba_state`` for K = 0: three Mamba groups of 12 GDN layers (``layers.<i>.linear_attn`` names) whose
    windows column 0 of row r is ``base_id + g + 4 r``.  ``checkpoints[r]``: row r's ``[(n, block base)]``; the group-g
    checkpoint block is ``base + g``; ``None``: no checkpoint fields (``checkpoint_positions`` None)."""

    layers = [[layer for layer in range(48) if layer % 4 != 3 and group_of(layer) == g] for g in range(3)]
    width = 1 if checkpoints is None else max(1, *(len(row) for row in checkpoints))
    groups = []
    for g in range(3):
        blocks = None
        if checkpoints is not None:
            blocks = torch.zeros(rows, width, dtype=torch.int32)
            for r, row in enumerate(checkpoints):
                for j, (_, base) in enumerate(row):
                    blocks[r, j] = base + g
        groups.append(
            SimpleNamespace(
                layer_names=tuple(
                    names(layer) if names else f"model.layers.{layer}.linear_attn" for layer in layers[g]
                ),
                windows=torch.tensor([[base_id + g + 4 * row] for row in range(rows)], dtype=torch.int32),
                checkpoint_blocks=blocks,
                chain=None,
            )
        )
    positions = None
    if checkpoints is not None:
        positions = torch.zeros(rows, width, dtype=torch.int32)
        for r, row in enumerate(checkpoints):
            for j, (n, _) in enumerate(row):
                positions[r, j] = n
    return SimpleNamespace(groups=tuple(groups), accepted_counts=None, checkpoint_positions=positions)


def _expected_blocks(chain, base_id: int = 3, row: int = 0, group_of=_group_of_stride):
    return {
        layer: (chain.pools[4 * (layer // 4) + 3].kv, base_id + group_of(layer) + 4 * row)
        for layer in range(48)
        if layer % 4 != 3
    }


def _ids(length: int) -> list[int]:
    return [1 + token % 1000 for token in range(length)]


def test_mamba_layer_blocks_maps_layers_to_their_slots_pools(on) -> None:
    model, chain = _state_model()
    assert model._mamba_layer_blocks(_mamba_state(rows=2), 1) == _expected_blocks(chain, row=1)


def test_the_layer_names_group_three_runs_of_twelve_gdn_layers(on) -> None:
    model, chain = _state_model()
    state = _mamba_state(group_of=_group_of_run)
    assert [len(group.layer_names) for group in state.groups] == [12, 12, 12]
    assert state.groups[0].layer_names[:4] == (
        "model.layers.0.linear_attn",
        "model.layers.1.linear_attn",
        "model.layers.2.linear_attn",
        "model.layers.4.linear_attn",
    )
    assert model._mamba_layer_blocks(state, 0) == _expected_blocks(chain, group_of=_group_of_run)
    assert [layers for _, layers in model._mamba_group_layers(state)][2][-1] == 46


def test_bare_layer_names_group_too(on) -> None:
    model, chain = _state_model()
    state = _mamba_state(names=lambda layer: f"layers.{layer}")
    assert model._mamba_layer_blocks(state, 0) == _expected_blocks(chain)


def _grouping_error(model, mutate) -> str:
    state = _mamba_state()
    groups = [list(group.layer_names) for group in state.groups]
    mutate(groups)
    for group, names in zip(state.groups, groups):
        group.layer_names = tuple(names)
    try:
        model._mamba_layer_blocks(state, 0)
    except ValueError as error:
        return str(error)
    raise AssertionError("the malformed grouping was not refused")


def test_a_malformed_grouping_is_refused(on, expect_error) -> None:
    model, _ = _state_model()
    assert "no Mamba group" in _grouping_error(model, lambda groups: groups[2].pop())
    assert "is in groups 0 and 1" in _grouping_error(model, lambda groups: groups[1].append(groups[0][0]))
    assert "not a GDN layer" in _grouping_error(model, lambda groups: groups[0].append("model.layers.3.self_attn"))
    assert "no 'layers.<index>' part" in _grouping_error(model, lambda groups: groups[0].append("lm_head"))
    assert "not a GDN layer" in _grouping_error(model, lambda groups: groups[0].append("model.layers.48.linear_attn"))
    model_before_allocation, _, _ = _model()
    with expect_error(ValueError, "slot maps are missing"):
        model_before_allocation._mamba_layer_blocks(_mamba_state(), 0)


def test_mamba_layer_blocks_validation(on, expect_error) -> None:
    model, _ = _state_model()
    with expect_error(ValueError, "mamba_state is missing"):
        model._mamba_layer_blocks(None, 0)
    for block, message in ((0, "null block"), (NUM_BLOCKS, "outside")):
        state = _mamba_state()
        state.groups[0].windows[0, 0] = block
        with expect_error(ValueError, message):
            model._mamba_layer_blocks(state, 0)
    with expect_error(ValueError, "no row 1"):
        model._mamba_layer_blocks(_mamba_state(), 1)


def test_the_group_check_runs_once_per_grouping(on) -> None:
    model, _ = _state_model()
    model._mamba_layer_blocks(_mamba_state(), 0)
    model._mamba_layer_blocks(_mamba_state(base_id=5), 0)
    assert len(model._mamba_group_cache) == 1
    model._mamba_layer_blocks(_mamba_state(group_of=_group_of_run), 0)
    assert len(model._mamba_group_cache) == 2


def test_a_start_of_zero_resets_and_never_restores(on) -> None:
    model, chain = _state_model()
    ids = _ids(1500)
    model._prefill_one(ids, greedy_token=True, layer_blocks=None)
    assert chain.restores == [] and chain.saves == []
    assert chain.log[0] == f"reset:{base_tests.SEED_TOKEN_ID}"


def test_a_continuation_on_the_chain_does_not_restore(on) -> None:
    model, chain = _state_model()
    ids = _ids(3000)
    model._prefill_one(ids[:1500], greedy_token=True)
    chain.log.clear()
    model._prefill_one(ids[:2100], greedy_token=True, start=1500)  # the slot holds ids[:1500]
    assert chain.restores == [] and chain.saves == []
    assert not any(entry.startswith("reset") for entry in chain.log)


def test_a_prefix_hit_restores_the_blocks_then_prefills_from_the_boundary(on) -> None:
    model, chain = _state_model()
    ids = _ids(1500)
    blocks = _expected_blocks(chain)
    model._prefill_one(ids, greedy_token=True, page_ids=[5, 6], start=1024, layer_blocks=blocks)
    assert chain.restores == [(1024, ids, blocks, BLOCK)] and chain.saves == []
    assert not any(entry.startswith("reset") for entry in chain.log)
    assert chain.log.index("table:(5, 6)") < chain.log.index("restore:1024")
    assert chain.chunks[0] == (tuple(ids[1024:1499]), 1024)
    assert len(model.slot.committed) == 1499 + 1 and model.slot.committed[:1024] == ids[:1024]
    assert not model.slot.failed


def test_a_prefix_hit_leaves_the_ple_context_the_chunk_prefill_expects(on) -> None:
    model, chain = _state_model()
    ids = _ids(1500)
    seen: list = []
    chunk_prefill = chain.chunk_prefill
    chain.chunk_prefill = lambda token_ids, *, start_position, ple_context, forced_step: (
        seen.append(ple_context)
        or chunk_prefill(token_ids, start_position=start_position, ple_context=ple_context, forced_step=forced_step)
    )
    model._prefill_one(ids, greedy_token=True, start=1024, layer_blocks=_expected_blocks(chain))
    assert seen == [(ids[1022], ids[1023])]


def test_a_start_that_is_not_a_block_boundary_is_refused(on, expect_error) -> None:
    model, chain = _state_model()
    with expect_error(ValueError, "block boundary"):
        model._prefill_one(_ids(1500), greedy_token=True, start=1000, layer_blocks=_expected_blocks(chain))
    assert chain.restores == [] and chain.log == []


def test_a_restore_without_the_blocks_is_refused(on, expect_error) -> None:
    model, chain = _state_model()
    with expect_error(ValueError, "layer_blocks is missing"):
        model._prefill_one(_ids(1500), greedy_token=True, start=1024)
    assert chain.log == []


@pytest.mark.parametrize("length, saved", [(1024, True), (2048, True), (1500, False), (1023, False), (1025, False)])
def test_the_state_is_saved_only_when_the_chunk_ends_on_a_boundary(length, saved, on) -> None:
    model, chain = _state_model()
    blocks = _expected_blocks(chain)
    model._prefill_one(_ids(length), greedy_token=True, layer_blocks=blocks)
    assert chain.saves == ([(length, blocks, BLOCK)] if saved else [])
    if saved:
        assert chain.log[-1] == f"save:{length}"  # after the forced last token's step


def test_a_save_without_the_blocks_is_refused_before_the_prefill(on, expect_error) -> None:
    model, chain = _state_model()
    with expect_error(ValueError, "layer_blocks is missing"):
        model._prefill_one(_ids(2048), greedy_token=True)
    assert chain.log == []


def test_a_restore_that_also_ends_on_a_boundary_saves(on) -> None:
    model, chain = _state_model()
    ids = _ids(2048)
    blocks = _expected_blocks(chain)
    model._prefill_one(ids, greedy_token=True, start=1024, layer_blocks=blocks)
    assert [entry for entry in chain.log if entry.startswith(("restore", "save"))] == ["restore:1024", "save:2048"]


def test_prefill_forward_passes_the_rows_blocks(on) -> None:
    model, chain = _state_model()
    ids = _ids(2048)
    model.prefill_forward(
        tokens=torch.tensor([ids], dtype=torch.int32),
        page_table=torch.tensor([[5, 6] + [0] * (paged_table_width(ALLOCATED_CONTEXT, BLOCK) - 2)], dtype=torch.int32),
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[2048],
        start_pos=[1024],
        empty_slots=[0],
        sampling_params=None,
        mamba_state=_mamba_state(base_id=7),
    )
    blocks = _expected_blocks(chain, base_id=7)
    assert chain.restores[0][:1] + chain.restores[0][2:] == (1024, blocks, BLOCK)
    assert chain.saves == [(2048, blocks, BLOCK)]


def _prefill_rows(model, ids, start, state):
    return model.prefill_forward(
        tokens=torch.tensor([ids], dtype=torch.int32),
        page_table=torch.tensor(
            [[5, 6, 7, 8] + [0] * (paged_table_width(ALLOCATED_CONTEXT, BLOCK) - 4)], dtype=torch.int32
        ),
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(ids)],
        start_pos=[start],
        empty_slots=[0],
        mamba_state=state,
    )


def test_prefill_forward_without_mamba_state_is_refused_when_a_block_is_needed(on, expect_error) -> None:
    model, chain = _state_model()
    with expect_error(ValueError, "mamba_state is missing"):
        _prefill_rows(model, _ids(2048), 0, None)
    assert chain.log == []


def test_a_start_of_zero_never_reads_a_block(on) -> None:
    model, chain = _state_model()
    state = _mamba_state()
    for group in state.groups:  # a block that would fail every check if the prefill read it
        group.windows[0, 0] = 0
    _prefill_rows(model, _ids(1500), 0, state)
    assert chain.restores == [] and chain.saves == []


def test_a_checkpoint_splits_the_chunk_in_order(on) -> None:
    model, chain = _state_model()
    ids = _ids(3072)
    _prefill_rows(model, ids, 1024, _mamba_state(base_id=7, checkpoints=[[(2048, 9)]]))
    window, checkpoint = _expected_blocks(chain, base_id=7), _expected_blocks(chain, base_id=9)
    order = [e for e in chain.log if e.startswith(("restore", "chunk_prefill", "save", "feed"))]
    # restore, chunk to n - 1, the forced token n - 1, save the checkpoint, chunk to E - 1, its forced token, save at E
    assert order == [
        "restore:1024",
        "chunk_prefill:1023@1024",
        f"feed:{ids[2047]}",
        "save:2048",
        "chunk_prefill:1023@2048",
        f"feed:{ids[3071]}",
        "save:3072",
    ]
    assert chain.saves == [(2048, checkpoint, BLOCK), (3072, window, BLOCK)]
    assert [(r[0], r[2]) for r in chain.restores] == [(1024, window)]
    assert len(model.slot.committed) == 3072 and not model.slot.failed


def test_a_checkpoint_in_an_unaligned_chunk_saves_only_the_checkpoint(on) -> None:
    model, chain = _state_model()
    ids = _ids(2500)
    _prefill_rows(model, ids, 0, _mamba_state(base_id=7, checkpoints=[[(2048, 9)]]))
    assert chain.restores == []
    assert chain.saves == [(2048, _expected_blocks(chain, base_id=9), BLOCK)]
    assert chain.chunks[-1] == (tuple(ids[2048:2499]), 2048)
    assert len(model.slot.committed) == 2500


def test_a_start_of_zero_with_a_checkpoint_never_reads_the_window(on) -> None:
    model, chain = _state_model()
    state = _mamba_state(base_id=7, checkpoints=[[(1024, 9)]])
    for group in state.groups:
        group.windows[0, 0] = 0
    _prefill_rows(model, _ids(1500), 0, state)
    assert chain.restores == [] and chain.saves == [(1024, _expected_blocks(chain, base_id=9), BLOCK)]
    assert chain.log[0] == f"reset:{base_tests.SEED_TOKEN_ID}"


def test_two_checkpoints_save_in_position_order(on) -> None:
    model, chain = _state_model()
    _prefill_rows(model, _ids(3500), 0, _mamba_state(base_id=7, checkpoints=[[(2048, 9), (1024, 4)]]))
    assert [save[0] for save in chain.saves] == [1024, 2048]
    assert chain.saves[0][1] == _expected_blocks(chain, base_id=4)


def test_unused_checkpoint_entries_are_ignored(on) -> None:
    model, chain = _state_model()
    _prefill_rows(model, _ids(1500), 0, _mamba_state(base_id=7, checkpoints=[[(0, 0)]]))
    assert chain.saves == []


@pytest.mark.parametrize("position, start, length", [(1500, 0, 3000), (1024, 1024, 3000), (3072, 0, 3072)])
def test_a_misplaced_checkpoint_is_refused(position, start, length, on, expect_error) -> None:
    model, chain = _state_model()
    with expect_error(ValueError, "block boundary"):
        _prefill_rows(model, _ids(length), start, _mamba_state(base_id=7, checkpoints=[[(position, 9)]]))
    assert chain.log == []


def test_a_checkpoint_without_blocks_is_refused(on, expect_error) -> None:
    model, chain = _state_model()
    state = _mamba_state(base_id=7, checkpoints=[[(1024, 9)]])
    state.groups[1].checkpoint_blocks = None
    with expect_error(ValueError, "without checkpoint blocks"):
        _prefill_rows(model, _ids(1500), 0, state)
    state = _mamba_state(base_id=7, checkpoints=[[(1024, 0)]])
    state.groups[0].checkpoint_blocks[0, 0] = 0
    with expect_error(ValueError, "null block"):
        _prefill_rows(model, _ids(1500), 0, state)
    assert chain.log == []


# -- prefill: batch mode ----------------------------------------------------------------------------------------------


@pytest.fixture
def batch_rig(monkeypatch, on):
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
    monkeypatch.setattr(
        Qwen38ForCausalLM, "_block_warm_hook", lambda self: None
    )  # the rig's chain is not a device chain
    rig = batch_tests.Rig(monkeypatch, block_size=BLOCK)
    chain = rig.chain
    chain.pools = {layer: FakePool(layer) for layer in range(3, 48, 4)}
    chain.restores, chain.saves = [], []
    chain.restore_prefix_state = lambda *args: StateChain.restore_prefix_state(chain, *args)
    chain.save_prefix_state = lambda *args: StateChain.save_prefix_state(chain, *args)
    rig.model.slot_qsa_layer = {j: 4 * j + 3 for j in range(12)}
    rig.model.gdn_slot = {layer: layer // 4 for layer in range(48) if layer % 4 != 3}
    rig.model.hybrid_single = False
    return rig


def test_a_batch_rows_second_chunk_restores_instead_of_raising(batch_rig) -> None:
    rig, chain = batch_rig, batch_rig.chain
    ids = _ids(2048)
    rig.prefill([ids[:1024]], [2], [[5]], mamba_state=_mamba_state(base_id=7))
    assert chain.saves == [(1024, _expected_blocks(chain, base_id=7), BLOCK)] and chain.restores == []
    rig.prefill([ids], [2], [[5, 6]], start_pos=torch.tensor([1024]), mamba_state=_mamba_state(base_id=9))
    blocks = _expected_blocks(chain, base_id=9)
    assert [(r[0], r[2], r[3]) for r in chain.restores] == [(1024, blocks, BLOCK)]
    assert chain.saves[-1] == (2048, blocks, BLOCK)
    assert chain.chunks[-1][1] == 1024


def test_a_batch_row_saves_its_checkpoint_before_the_import(batch_rig) -> None:
    rig, chain = batch_rig, batch_rig.chain
    rig.prefill([_ids(2500)], [2], [[5, 6, 7]], mamba_state=_mamba_state(base_id=7, checkpoints=[[(2048, 9)]]))
    assert chain.saves == [(2048, _expected_blocks(chain, base_id=9), BLOCK)]


def test_a_batch_prefill_at_zero_with_an_unaligned_end_needs_no_blocks(batch_rig) -> None:
    rig, chain = batch_rig, batch_rig.chain
    rig.prefill([_ids(1500)], [0], [[5, 6]])
    assert chain.restores == [] and chain.saves == []


def test_a_hybrid_chain_continuation_does_not_restore(batch_rig) -> None:
    rig, chain = batch_rig, batch_rig.chain
    rig.model.hybrid_single = True
    ids = _ids(2048)
    rig.prefill(
        [ids[:1024]],
        [0],
        [[5]],
        sampling_params=vllm_module.greedy_batch_sampling_params(1),
        mamba_state=_mamba_state(),
    )
    assert rig.model.hybrid_slot == 0
    rig.prefill(
        [ids],
        [0],
        [[5, 6]],
        start_pos=torch.tensor([1024]),
        sampling_params=vllm_module.greedy_batch_sampling_params(1),
        mamba_state=_mamba_state(),
    )
    assert chain.restores == []


def test_a_batch_continuation_off_the_chain_is_refused_when_the_mode_is_off(monkeypatch, expect_error) -> None:
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
    monkeypatch.delenv("QWEN38_MAMBA_SPEC", raising=False)
    rig = batch_tests.Rig(monkeypatch)
    with expect_error(NotImplementedError, "chunked prefill in batch mode"):
        rig.prefill([_ids(300)], [0], [[5, 6, 7]], start_pos=torch.tensor([128]))


def test_off_mode_prefill_one_is_unchanged(monkeypatch, expect_error) -> None:
    monkeypatch.delenv("QWEN38_MAMBA_SPEC", raising=False)
    chain = StateChain()
    chain.restores, chain.saves = [], []
    model, _, _ = _model(chain)
    model.chain = chain
    with expect_error(Qwen38ChatChainError, "chunked prefill starts at position"):
        model._prefill_one(_ids(300), greedy_token=True, start=128)
    model._prefill_one(_ids(1024), greedy_token=True)
    assert chain.saves == [] and chain.restores == []
