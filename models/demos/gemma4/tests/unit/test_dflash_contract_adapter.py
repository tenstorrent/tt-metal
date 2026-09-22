# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host coverage for Gemma's actual request-owned dFlash contract adapter.

The adapter retains committed prefixes, associates completions with request
objects, and rebuilds the fused decoder after ordinary decode. These tests
import the production adapter and exercise its production state transitions.
Only lower model/device operations and unavailable plugin types are stubbed;
TT buffer correctness and captured-trace behavior require device validation.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _unused(*args, **kwargs):
    raise AssertionError("A host contract test reached an unstubbed device operation")


def _padded_length(length):
    return 128 if length <= 128 else max(1024, 1 << (length - 1).bit_length())


def _serving_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=4096,
            hf_config=SimpleNamespace(
                layer_types=["sliding_attention", "full_attention"], num_key_value_heads=8, head_dim=32
            ),
        ),
        cache_config=SimpleNamespace(block_size=64),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


class ScratchTensor:
    def __init__(self, value):
        self.value = value.clone()
        self.shape = self.padded_shape = tuple(value.shape)
        self.dtype = value.dtype
        self.releases = 0

    def deallocate(self, force):
        assert force
        self.releases += 1


def _module(name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    return module


@pytest.fixture(scope="module")
def adapter():
    """Import production Python while isolating unavailable TT dependencies."""
    with pytest.MonkeyPatch.context() as patch:
        root = Path(__file__).resolve().parents[5]
        patch.syspath_prepend(str(root))
        patch.delenv("GEMMA4_CONTRACT_ASYNC", raising=False)
        patch.delenv("GEMMA4_DFLASH_MAX_SPEC_ISL", raising=False)
        patch.setenv("GEMMA4_DFLASH_VERIFY", "5")
        patch.setitem(
            sys.modules,
            "ttnn",
            _module(
                "ttnn",
                synchronize_device=lambda mesh: mesh.append(("sync",)),
                MemoryConfig=object,
                DRAM_MEMORY_CONFIG=object(),
            ),
        )
        lower_modules = {
            "models.demos.gemma4.tt.common": {"create_tt_model": _unused},
            "models.demos.gemma4.tt.generator": {
                "SDPA_CHUNK_ALIGN": 128,
                "ChunkedPrefillPageTableGuardMixin": type("ChunkedPrefillPageTableGuardMixin", (), {}),
                "align_num_cached_tokens_to_sdpa": _unused,
                "max_batched_prefill_users": _unused,
                "resolve_batched_prefill_chunk_users": _unused,
            },
            "models.demos.gemma4.tt.generator_trace": {
                name: _unused
                for name in (
                    "maybe_disable_pli_prefill_trace",
                    "patch_gemma4_trace_model_args",
                    "resolve_gemma4_prefill_chunk_size",
                    "resolve_gemma4_prefill_trace_enable",
                    "should_auto_enable_bounded_sliding",
                    "warmup_gemma4_model_prefill",
                )
            },
            "models.tt_transformers.tt.common": {"get_padded_prefill_len": _padded_length},
            "models.tt_transformers.tt.generator": {
                "SUPPORTED_PREFILL_BATCH_SIZES": (1, 2, 4, 8, 16, 32),
                "create_submeshes": _unused,
            },
            "models.tt_transformers.tt.generator_vllm": {
                "HybridAttentionForCausalLM": type("HybridAttentionForCausalLM", (), {}),
                "allocate_vllm_kv_cache": _unused,
            },
            "models.demos.gemma4.tt.dflash_drafter": {"DFlashFusedDecoder": _unused},
            "models.demos.gemma4.tt.attention": {"__path__": [str(root / "models/demos/gemma4/tt/attention")]},
            "models.demos.gemma4.tt.attention.weights": {"AttentionWeights": object},
            "models.demos.gemma4.tt.dram_sharded": {"DramShardedLinear": object},
            "models.demos.gemma4.tt.ccl": {"ccl_allreduce": _unused},
        }
        for name, members in lower_modules.items():
            patch.setitem(sys.modules, name, _module(name, **members))
        try:
            importlib.import_module("loguru")
        except ImportError:
            logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
            patch.setitem(sys.modules, "loguru", _module("loguru", logger=logger))
        try:
            importlib.import_module("vllm_tt_plugin.spec_decode")
        except ImportError:

            @dataclass
            class DraftOutput:
                draft_token_ids: object
                num_valid: object = None
                draft_scores: object = None

            @dataclass
            class VerifyOutput:
                spec_mode: str
                argmax_ids: object = None
                hidden: object = None

            @dataclass
            class SpecPlan:
                effective_k: int
                lanes_per_request: int
                extra_bytes_per_seq: int
                extra_bytes_per_token: int
                accept_modes: tuple
                drafter_state: str
                drafter_target_cache_requires: tuple
                supports_narrow_decode: bool = False

            @dataclass
            class SpecReject:
                reason: str
                supported_k: tuple

            patch.setitem(sys.modules, "vllm_tt_plugin", _module("vllm_tt_plugin", __path__=[]))
            patch.setitem(
                sys.modules,
                "vllm_tt_plugin.spec_decode",
                _module(
                    "vllm_tt_plugin.spec_decode",
                    DraftOutput=DraftOutput,
                    VerifyOutput=VerifyOutput,
                    SpecPlan=SpecPlan,
                    SpecReject=SpecReject,
                    PLACEHOLDER_TOKEN_ID=-1,
                ),
            )
        name = "models.demos.gemma4.tt.generator_vllm"
        parent = importlib.import_module("models.demos.gemma4.tt")
        patch.setattr(parent, "generator_vllm", None, raising=False)
        patch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
        yield module
        sys.modules.pop(name, None)
        sys.modules.pop("models.demos.gemma4.tt.attention.operations", None)


@pytest.fixture
def expect_error():
    # The repository fixture imports TT runtime state outside this host harness.
    return pytest.raises  # allow-pytest.raises: root conftest requires hardware runtime


@pytest.fixture(scope="module")
def decoder_width_for(adapter):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys.modules["ttnn"], "bfloat16", object(), raising=False)
        patch.setitem(
            sys.modules,
            "models.demos.gemma4.tt.ccl",
            _module("models.demos.gemma4.tt.ccl", ccl_allgather=_unused, ccl_allreduce=_unused),
        )
        name = "models.demos.gemma4.tt.dflash_drafter"
        parent = importlib.import_module("models.demos.gemma4.tt")
        patch.setattr(parent, "dflash_drafter", None, raising=False)
        patch.delitem(sys.modules, name, raising=False)
        return importlib.import_module(name).DFlashFusedDecoder.width_for


@dataclass
class DeviceResult:
    value: torch.Tensor


@dataclass
class HostResult:
    value: torch.Tensor


class Target:
    def __init__(self, events):
        self.events = events
        self.layers = [
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    config=SimpleNamespace(head_dim=8, num_key_value_heads=1, cache_position_modulo=None),
                    weights=SimpleNamespace(kv_replicated=False),
                )
            )
            for _ in range(2)
        ]
        self.mesh_config = SimpleNamespace(tp=1)
        self.tap_layers = None
        self.taps = []
        self._dflash_sharded_logits = False

    def dflash_capture_taps(self, layers, **kwargs):
        self.tap_layers = layers
        self.taps = []
        self.events.append(("capture", None if layers is None else tuple(layers)))

    def pop_dflash_taps(self):
        self.events.append(("pop_taps",))
        result, self.taps = self.taps, []
        return result


class Decoder:
    def __init__(self, target, events):
        self.target = target
        self.events = events
        self.start = 0
        self.anchor = 0
        self.script = []
        self.P_v = 6
        self.pv_sk = 1024
        self.use_packed = True
        self._pv_widths = {1024: {"trace": 1}}
        self.refreshes = []

    def restore_model_logits_mode(self):
        self.target._dflash_sharded_logits = False
        self.events.append(("restore_logits",))

    def capture_widths(self, widths):
        self.events.append(("capture_widths", tuple(sorted(widths))))
        for width in widths:
            self._pv_widths[width]["trace"] = width

    def select_width(self, start):
        self.events.append(("select_width", start))
        return self.width_for(start)

    def refresh_page_tables(self, page_table):
        self.refreshes.append(page_table.clone())
        self.events.append(("refresh", page_table.tolist()))

    def prefill_ingest(self, taps, length):
        self.events.append(("ingest", length, [tap.tolist() for tap in taps]))

    def reseed(self, anchor, start):
        self.anchor, self.start = int(anchor), int(start)
        self.events.append(("reseed", self.anchor, self.start))

    def contract_replay(self, first=False):
        self.events.append(("replay", first, self.start))
        self.target._dflash_sharded_logits = True
        if self.script:
            return self.script.pop(0)
        return [21, 22, 23, 24, 25], [21, 22, 99, 24, 25, 26]

    def contract_commit(self, produced, anchor):
        self.events.append(("commit", int(produced), int(anchor)))
        self.start += int(produced)
        self.anchor = int(anchor)


@pytest.fixture
def model(adapter, decoder_width_for, monkeypatch):
    monkeypatch.setattr(Decoder, "width_for", decoder_width_for, raising=False)
    model = adapter.Gemma4DFlashContractForCausalLM.__new__(adapter.Gemma4DFlashContractForCausalLM)
    model._contract_init()
    model.events = []
    model.model = [Target(model.events)]
    model.model_args = [SimpleNamespace(max_seq_len=1024, max_batch_size=32)]
    model.mesh_device = model.events
    model._spec_decoder = Decoder(model.model[0], model.events)
    model._spec_width_set = True
    model._spec_width_ladder = [1024]
    model._spec_horizon = 256
    model._spec_active = False
    model._spec_active_owner = None
    model._spec_owner_slot = None
    model._spec_pending = None
    model._spec_pending_owner = None
    model._spec_first_step = True
    model._bounded_sliding_kv_cache = False
    model._decode_warmup_complete = True
    model._SPEC_CONTRACT_K = 5
    model.mode = "DECODE"
    model._slots_prefilled_since_decode = set()
    model.results = []
    model.readback_events = [object(), object()]
    cache = ScratchTensor(torch.zeros(16, 1, 64, 8, dtype=torch.bfloat16))
    model.kv_cache = [[(cache, cache), (cache, cache)]]
    runtime = sys.modules["ttnn"]
    monkeypatch.setattr(runtime, "from_torch", lambda value, **kwargs: ScratchTensor(value), raising=False)
    monkeypatch.setattr(runtime, "TILE_LAYOUT", object(), raising=False)
    monkeypatch.setattr(runtime, "ReplicateTensorToMesh", lambda mesh: mesh, raising=False)
    target = model.model[0]
    target.mesh_device = model.mesh_device
    target._page_table_torch_to_ttnn = ScratchTensor
    target._page_tables_to_ttnn = lambda tables: target._persistent_pt_by_batch[tables[0].shape[0]]
    for layer in target.layers:
        layer.self_attn._release_sliding_prefill_tail = lambda **kwargs: None
    model._contract_prepare_rebuild_storage(model.kv_cache)
    model._spec_get_drafter = lambda: SimpleNamespace(target_layer_ids=[1, 2])
    model._build_per_layer_page_tables = (
        lambda per_layer, table: per_layer if per_layer is not None else [table] * len(model.model[0].layers)
    )
    model._pad_sliding_page_tables_for_bounded = lambda per_layer, cache, authoritative: per_layer

    def prefill(*args, **kwargs):
        tokens = kwargs.get("tokens", args[0] if args else None)
        assert model.model[0]._dflash_sharded_logits is False
        model.events.append(("prefill", kwargs))
        model.mode = "PREFILL"
        model._slots_prefilled_since_decode.update(kwargs.get("empty_slots", range(tokens.shape[0])))
        if model.model[0].tap_layers is not None:
            model.model[0].taps.append(tokens.clone())
        return torch.zeros(tokens.shape[0], 1, dtype=torch.int32)

    def decode(*args, **kwargs):
        assert model.model[0]._dflash_sharded_logits is False
        assert model.model[0].tap_layers is None
        model.events.append(("decode", kwargs))
        assert model.results, "Supply a native ordinary result before submitting decode"
        return model.results.pop(0)

    def read(payload, async_read=False):
        model.events.append(("read", payload, async_read))
        result = HostResult(payload.value)
        return (result, model.readback_events) if async_read else result

    def process(payload, is_tokens=False):
        model.events.append(("process", payload, is_tokens))
        return payload.value if isinstance(payload, HostResult) else payload

    monkeypatch.setattr(model, "_contract_target_prefill", prefill)
    monkeypatch.setattr(model, "_contract_target_decode", decode)
    monkeypatch.setattr(model, "_contract_target_read", read)
    monkeypatch.setattr(model, "_contract_target_process", process)
    return model


def _tensor(values):
    return torch.tensor(values, dtype=torch.int32)


def _prefill(model, keys=(10,), slots=None, prompts=None):
    prompts = prompts if prompts is not None else [[1, 2] for _ in keys]
    slots = list(range(len(keys))) if slots is None else slots
    return model.prefill_forward(
        tokens=_tensor(prompts),
        prompt_lens=[len(prompt) for prompt in prompts],
        empty_slots=slots,
        page_table=_tensor([[key, key + 1] for key in keys]),
        kv_cache=model.kv_cache,
        warmup_prefill=False,
    )


def _ordinary(model, anchors, starts, keys, result, **kwargs):
    model.results.append(result)
    return model.decode_forward(
        tokens=_tensor(anchors).reshape(-1, 1),
        start_pos=_tensor(starts).reshape(-1, 1),
        page_table=_tensor([[key, key + 1] for key in keys]),
        kv_cache=model.kv_cache,
        **kwargs,
    )


def _complete(model, tokens, positions, counts=None, hidden=None):
    tokens = _tensor(tokens)
    positions = _tensor(positions)
    if counts is None:
        counts = [tokens.shape[1]] * tokens.shape[0]
    return model.propose_draft_tokens(5, tokens, positions, _tensor(counts), hidden=hidden)


def _start_solo(model):
    _prefill(model)
    output = _ordinary(model, [3], [2], [10], DeviceResult(_tensor([[4]])))
    proposal = _complete(model, [[4]], [[3]])
    assert proposal.num_valid.tolist() == [5]
    return output, proposal


def _verify(model, keys=(10,), blocks=None, positions=None, valid=None, **kwargs):
    blocks = blocks if blocks is not None else [[4, 21, 22, 23, 24, 25]]
    positions = positions if positions is not None else [list(range(3, 9))]
    valid = valid if valid is not None else [5]
    return model.decode_forward(
        tokens=_tensor(blocks),
        start_pos=_tensor(positions),
        spec_mode="argmax_ids",
        num_valid_drafts=_tensor(valid),
        page_table=_tensor([[key, key + 1] for key in keys]),
        kv_cache=model.kv_cache,
        **kwargs,
    )


def test_initial_narrow_completion_bootstraps_real_adapter(model):
    _, proposal = _start_solo(model)
    owner = model._ct_requests[10]
    assert model._ct_decoder_owner is owner
    assert owner.tokens == [1, 2, 3, 4]
    assert model._spec_active is True
    assert model._spec_decoder.start == 3
    assert model._spec_decoder.anchor == 4
    assert proposal.draft_token_ids.tolist() == [[21, 22, 23, 24, 25]]
    rebuild = [event[1] for event in model.events if event[0] == "prefill"][-1]
    assert rebuild["tokens"].tolist() == [[1, 2, 3]]
    assert rebuild["prompt_lens"] == [3]
    assert rebuild["empty_slots"] == [0]
    assert rebuild["start_pos"] == [0]
    assert rebuild["sampling_params"] is None
    assert rebuild["enable_trace"] is False
    assert rebuild["warmup_prefill"] is False
    assert model.mode == "PREFILL"
    assert model._slots_prefilled_since_decode == {0}
    names = [event[0] for event in model.events]
    assert names.index("sync") < names.index("ingest") < names.index("reseed") < names.index("replay")


def test_prefill_masks_stale_aliases_without_mutating_request_tables(model):
    table = _tensor([[5, 6, 7, 7, 6, 5]])
    per_layer = [table, table.clone()]
    model.prefill_forward(
        tokens=_tensor([list(range(157))]),
        prompt_lens=[157],
        start_pos=[0],
        page_table=table,
        page_tables_per_layer=per_layer,
        kv_cache=model.kv_cache,
    )
    forwarded = [event[1] for event in model.events if event[0] == "prefill"][-1]
    assert forwarded["page_table"].tolist() == [[5, 6, 7, 0, 0, 0]]
    assert all(item.tolist() == [[5, 6, 7, 0, 0, 0]] for item in forwarded["page_tables_per_layer"])
    assert table.tolist() == [[5, 6, 7, 7, 6, 5]]
    assert all(item.tolist() == table.tolist() for item in per_layer)
    assert forwarded["page_table"].data_ptr() != table.data_ptr()
    assert all(new.data_ptr() != old.data_ptr() for new, old in zip(forwarded["page_tables_per_layer"], per_layer))
    assert model._ct_requests[5].tokens == list(range(157))


def test_prefill_masks_each_batch_row_at_its_absolute_prefix_length(model):
    table = _tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]])
    model.prefill_forward(
        tokens=torch.zeros(4, 157, dtype=torch.int32),
        prompt_lens=[63, 64, 65, 157],
        empty_slots=[3, 0, 2, 1],
        page_table=table,
        kv_cache=model.kv_cache,
    )
    forwarded = [event[1] for event in model.events if event[0] == "prefill"][-1]
    expected = [[10, 0, 0, 0], [20, 0, 0, 0], [30, 31, 0, 0], [40, 41, 42, 0]]
    assert forwarded["page_table"].tolist() == expected
    assert all(item.tolist() == expected for item in forwarded["page_tables_per_layer"])
    assert forwarded["empty_slots"] == [3, 0, 2, 1]
    assert [model._ct_requests[key].slot for key in (10, 20, 30, 40)] == [3, 0, 2, 1]


def test_resumed_prefill_uses_chunk_end_without_adding_start_position(model):
    table = _tensor([[5, 6, 7, 8, 5, 6]])
    model.prefill_forward(
        tokens=_tensor([list(range(256))]),
        prompt_lens=[193],
        start_pos=[128],
        page_table=table,
        kv_cache=model.kv_cache,
    )
    forwarded = [event[1] for event in model.events if event[0] == "prefill"][-1]
    assert forwarded["page_table"].tolist() == [[5, 6, 7, 8, 0, 0]]
    assert forwarded["start_pos"] == [0]
    assert forwarded["enable_trace"] is False
    assert forwarded["prompt_lens"] == [193]
    assert model._ct_requests[5].tokens == list(range(193))
    assert table.tolist() == [[5, 6, 7, 8, 5, 6]]


def test_reconstruction_masks_prefix_tables_but_preserves_verification_tables(model):
    table = _tensor([[5, 6, 7, 8, 6, 5]])
    per_layer = [table.clone(), table.clone()]
    _prefill(model, keys=(5,), prompts=[list(range(188))])
    model.results.append(DeviceResult(_tensor([[42]])))
    ordinary = model.decode_forward(
        tokens=_tensor([[41]]),
        start_pos=_tensor([188]),
        page_table=table,
        page_tables_per_layer=per_layer,
        kv_cache=model.kv_cache,
    )
    proposal = _complete(model, [[42]], [[189]])
    assert proposal.num_valid.tolist() == [5]
    reconstruction = [event[1] for event in model.events if event[0] == "prefill"][-1]
    assert reconstruction["prompt_lens"] == [189]
    expected = [[1, 2, 3] + [0] * 13]
    assert reconstruction["page_table"].tolist() == expected
    assert all(item.tolist() == expected for item in reconstruction["page_tables_per_layer"])
    assert reconstruction["kv_cache"] is model._ct_rebuild_storage["kv_cache"]
    assert ordinary.step.page_table.tolist() == [[5, 6, 7, 8, 6, 5]]
    assert model._spec_decoder.refreshes[-1].tolist() == [[5, 6, 7, 8, 6, 5]]
    assert all(item.tolist() == [[5, 6, 7, 8, 6, 5]] for item in model.model[0]._active_page_tables_per_layer)
    assert table.tolist() == [[5, 6, 7, 8, 6, 5]]


def test_prefill_mask_uses_each_layer_cache_view_and_replicated_heads(model):
    target = model.model[0]
    target.mesh_config.tp = 8
    target.layers[0].self_attn.config.num_key_value_heads = 8
    second = target.layers[1].self_attn
    second.config.num_key_value_heads = 4
    second.config.head_dim = 16
    second.weights.kv_replicated = True
    cache = SimpleNamespace(padded_shape=(16, 4, 64, 8))
    model.kv_cache[0][1] = (cache, cache)
    table = _tensor([[5, 6, 7, 8]])
    model.prefill_forward(
        tokens=torch.zeros(1, 157, dtype=torch.int32),
        prompt_lens=[157],
        page_table=table,
        page_tables_per_layer=[table, table],
        kv_cache=model.kv_cache,
    )
    forwarded = [event[1] for event in model.events if event[0] == "prefill"][-1]
    assert forwarded["page_table"].tolist() == [[5, 6, 7, 0]]
    assert [item.tolist() for item in forwarded["page_tables_per_layer"]] == [
        [[5, 6, 7, 0]],
        [[5, 6, 0, 0]],
    ]


def test_prefill_preserves_bounded_ring_columns_and_masks_full_attention(model):
    model._bounded_sliding_kv_cache = True
    model.model[0].layers[0].self_attn.config.cache_position_modulo = 256
    table = _tensor([[5, 6, 7, 8]])
    model.prefill_forward(
        tokens=torch.zeros(1, 65, dtype=torch.int32),
        prompt_lens=[65],
        page_table=table,
        kv_cache=model.kv_cache,
    )
    forwarded = [event[1] for event in model.events if event[0] == "prefill"][-1]
    assert forwarded["page_table"].tolist() == [[5, 6, 7, 8]]
    assert forwarded["page_tables_per_layer"][0].tolist() == [[5, 6, 7, 8]]
    assert forwarded["page_tables_per_layer"][1].tolist() == [[5, 6, 0, 0]]
    assert forwarded["page_tables_per_layer"][0].data_ptr() != table.data_ptr()
    assert table.tolist() == [[5, 6, 7, 8]]


def test_initial_batched_ordinary_decode_resumes_original_survivor(model):
    _prefill(model, keys=(10, 20), prompts=[[1, 2], [5, 6]])
    _ordinary(model, [3, 7], [2, 2], [10, 20], DeviceResult(_tensor([[4], [8]])))
    first = _complete(model, [[4], [8]], [[3], [3]])
    assert first.num_valid.tolist() == [0, 0]
    assert not any(event[0] == "replay" for event in model.events)
    original = model._ct_requests[10]
    model.release_request(1)
    _ordinary(model, [4], [3], [10], DeviceResult(_tensor([[9]])))
    resumed = _complete(model, [[9]], [[4]])
    assert resumed.num_valid.tolist() == [5]
    assert model._ct_decoder_owner is original
    assert original.tokens == [1, 2, 3, 4, 9]
    assert model._spec_decoder.start == 4


def test_peer_prefill_preserves_outstanding_proposal_for_owner_at_row_one(model):
    _start_solo(model)
    original = model._ct_requests[10]
    held = model._ct_proposal
    _prefill(model, keys=(20,), slots=[1], prompts=[[5, 6]])
    assert model._ct_proposal is held
    model.results.append(DeviceResult(_tensor([[55], [0]])))
    verified = _verify(
        model,
        keys=(20, 10),
        blocks=[[7, 0, 0, 0, 0, 0], [4, 21, 22, 23, 24, 25]],
        positions=[list(range(2, 8)), list(range(3, 9))],
        valid=[0, 5],
        sampling_params=object(),
        slot_remap=[1, 0],
    )
    assert verified.argmax_ids.tolist() == [[55, -1, -1, -1, -1, -1], held.posterior]
    assert verified.hidden.verified is original
    assert verified.hidden.owners == (model._ct_requests[20], original)
    ordinary = [event[1] for event in model.events if event[0] == "decode"][-1]
    assert ordinary["start_pos"].tolist() == [2, -1]
    assert ordinary["tokens"].tolist() == [7, 4]
    decline = _complete(model, [[55, 0, 0], [21, 22, 99]], [[3, -1, -1], [4, 5, 6]], [1, 3], verified.hidden)
    assert decline.num_valid.tolist() == [0, 0]
    assert model._ct_proposal is None
    assert original.tokens == [1, 2, 3, 4, 21, 22, 99]
    model.results.append(DeviceResult(_tensor([[56], [100]])))
    resolved = _verify(
        model,
        keys=(20, 10),
        blocks=[[55, 0, 0, 0, 0, 0], [99, 0, 0, 0, 0, 0]],
        positions=[list(range(3, 9)), list(range(6, 12))],
        valid=[0, 0],
        accepted_counts=_tensor([1, 3]),
        sampling_params=object(),
    )
    assert resolved.hidden.verified is None
    assert resolved.argmax_ids.tolist() == [[56, -1, -1, -1, -1, -1], [100, -1, -1, -1, -1, -1]]
    count_only = [event[1] for event in model.events if event[0] == "decode"][-1]
    assert count_only["start_pos"].tolist() == [3, 6]
    assert count_only["tokens"].tolist() == [55, 99]
    assert _complete(model, [[56], [100]], [[4], [7]], hidden=resolved.hidden).num_valid.tolist() == [0, 0]
    assert model._ct_proposal is None
    _ordinary(model, [56, 100], [4, 7], [20, 10], DeviceResult(_tensor([[57], [101]])))
    assert _complete(model, [[57], [101]], [[5], [8]]).num_valid.tolist() == [0, 0]
    assert model._ct_proposal is None
    assert not any(event[0] == "commit" for event in model.events)
    assert len([event for event in model.events if event[0] == "replay"]) == 1
    model.release_request(0)
    _ordinary(model, [101], [8], [10], DeviceResult(_tensor([[102]])), slot_remap=[1])
    resumed = _complete(model, [[102]], [[9]])
    assert resumed.num_valid.tolist() == [5]
    assert model._ct_decoder_owner is original
    assert original.slot == 0
    assert original.tokens == [1, 2, 3, 4, 21, 22, 99, 100, 101, 102]
    assert model._spec_decoder.start == 9
    assert model._ct_proposal is not held
    assert model._ct_proposal.anchor == 102
    assert not any(event[0] == "commit" for event in model.events)


@pytest.mark.parametrize("change", ["draft", "anchor", "position", "owner", "count"])
def test_verify_mismatch_fails_before_device_operations(model, change, expect_error):
    _start_solo(model)
    block, positions, keys, valid = [4, 21, 22, 23, 24, 25], list(range(3, 9)), (10,), [5]
    if change == "draft":
        block[2] = 999
    elif change == "anchor":
        block[0] = 999
    elif change == "position":
        positions = list(range(4, 10))
    elif change == "owner":
        keys = (20,)
    else:
        valid = [6]
    before = list(model.events)
    with expect_error((ValueError, RuntimeError), match="did not propose|no device posterior|more drafts"):
        _verify(model, keys=keys, blocks=[block], positions=[positions], valid=valid)
    assert model.events == before


def test_page_tables_are_snapshotted_and_refreshed_before_commit_and_replay(model):
    _start_solo(model)
    page_table = _tensor([[10, 71]])
    per_layer = [_tensor([[80, 81]]), _tensor([[90, 91]])]
    verified = model.decode_forward(
        tokens=_tensor([[4, 21, 22, 23, 24, 25]]),
        start_pos=_tensor([list(range(3, 9))]),
        page_table=page_table,
        page_tables_per_layer=per_layer,
        kv_cache=model.kv_cache,
        spec_mode="argmax_ids",
        num_valid_drafts=_tensor([5]),
    )
    page_table[0, 1] = 777
    per_layer[0][0, 1] = 888
    model.events.clear()
    proposed = _complete(model, [[21, 22, 99]], [[4, 5, 6]], hidden=verified.hidden)
    assert proposed.num_valid.tolist() == [5]
    assert model._spec_decoder.refreshes[-1].tolist() == [[10, 71]]
    assert model.model[0]._active_page_tables_per_layer[0].tolist() == [[80, 81]]
    names = [event[0] for event in model.events]
    assert names.index("refresh") < names.index("commit") < names.index("replay")
    assert ("commit", 3, 99) in model.events
    assert model._spec_decoder.start == 6


def test_two_pending_ordinary_results_keep_context_and_decline_older_completion(model):
    _prefill(model)
    first = _ordinary(model, [3], [2], [10], DeviceResult(_tensor([[4]])))
    second = _ordinary(model, [3], [3], [10], DeviceResult(_tensor([[5]])))
    assert first.step is not second.step
    assert first.step.owners[0] is second.step.owners[0]
    assert len(model._ct_ordinary) == 2
    read_second, events = model.read_decode_output(second, async_read=True)
    read_first = model.read_decode_output(first)
    assert events is model.readback_events
    assert read_first.step is first.step
    assert read_second.step is second.step
    assert model.process_decode_output_host(read_first, is_tokens=True).tolist() == [[4]]
    assert model.process_decode_output_host(read_second, is_tokens=True).tolist() == [[5]]
    assert len(model._ct_ordinary) == 2
    older = _complete(model, [[4]], [[3]])
    assert older.num_valid.tolist() == [0]
    assert not any(event[0] == "replay" for event in model.events)
    newest = _complete(model, [[5]], [[4]])
    assert newest.num_valid.tolist() == [5]
    assert model._ct_requests[10].tokens == [1, 2, 3, 4, 5]
    assert model._spec_decoder.start == 4


def test_pending_ordinary_completions_keep_submission_row_order_after_remap(model):
    _prefill(model, keys=(10, 20), prompts=[[1, 2], [5, 6]])
    first = _ordinary(model, [3, 7], [2, 2], [10, 20], DeviceResult(_tensor([[4], [8]])))
    second = _ordinary(model, [7, 3], [3, 3], [20, 10], DeviceResult(_tensor([[9], [5]])), slot_remap=[1, 0])
    assert first.step.owners == tuple(reversed(second.step.owners))
    _complete(model, [[4], [8]], [[3], [3]])
    _complete(model, [[9], [5]], [[4], [4]])
    assert model._ct_requests[10].tokens == [1, 2, 3, 4, 5]
    assert model._ct_requests[20].tokens == [5, 6, 7, 8, 9]
    assert model._ct_requests[10].slot == 1
    assert model._ct_requests[20].slot == 0


def test_padded_solo_at_nonzero_row_rebuilds_only_its_own_prefix(model):
    _prefill(model, slots=[1])
    output = _ordinary(model, [0, 3], [-1, 2], [0, 10], DeviceResult(_tensor([[0], [4]])))
    proposal = _complete(model, [[0], [4]], [[-1], [3]], [0, 1])
    assert output.step.owners[0] is None
    assert proposal.num_valid.tolist() == [0, 5]
    assert proposal.draft_token_ids[1].tolist() == [21, 22, 23, 24, 25]
    rebuild = [event[1] for event in model.events if event[0] == "prefill"][-1]
    assert rebuild["page_table"].tolist() == [[1] + [0] * 15]
    assert rebuild["empty_slots"] == [0]
    assert rebuild["tokens"].tolist() == [[1, 2, 3]]


def test_cancelled_ordinary_completion_cannot_update_reused_slot_or_pages(model):
    _prefill(model)
    old_output = _ordinary(model, [3], [2], [10], DeviceResult(_tensor([[4]])))
    old_owner = old_output.step.owners[0]
    model.release_request(0)
    _prefill(model, prompts=[[50, 51]])
    replacement = model._ct_requests[10]
    assert old_owner.live is False
    assert replacement is not old_owner
    _ordinary(model, [52], [2], [10], DeviceResult(_tensor([[53]])))
    stale = _complete(model, [[4]], [[3]])
    assert stale.num_valid.tolist() == [0]
    assert replacement.tokens == [50, 51, 52]
    current = _complete(model, [[53]], [[3]])
    assert current.num_valid.tolist() == [5]
    assert replacement.tokens == [50, 51, 52, 53]
    assert model._ct_decoder_owner is replacement


def test_slot_permutation_updates_each_owner_once_and_preserves_pending_context(model):
    _prefill(model, keys=(10, 20, 30), prompts=[[1, 2], [3, 4], [5, 6]])
    output = _ordinary(model, [7, 8, 9], [2, 2, 2], [10, 20, 30], DeviceResult(_tensor([[10], [11], [12]])))
    owners = tuple(model._ct_requests[key] for key in (10, 20, 30))
    model._ct_decoder_owner = owners[0]
    model.note_state_slots_moved({0: 1, 1: 2, 2: 0})
    assert [owner.slot for owner in owners] == [1, 2, 0]
    assert model._spec_owner_slot == 1
    assert output.step.owners == owners
    model.release_request(0)
    assert [owner.live for owner in owners] == [True, True, False]


@pytest.mark.parametrize("acknowledge_gather", [False, True])
def test_decode_remap_and_optional_lifecycle_callback_apply_one_permutation(model, acknowledge_gather):
    _prefill(model, keys=(10, 20, 30), prompts=[[1, 2], [3, 4], [5, 6]])
    owners = tuple(model._ct_requests[key] for key in (10, 20, 30))
    model._ct_decoder_owner = owners[0]
    output = _ordinary(model, [8, 7], [2, 2], [20, 10], DeviceResult(_tensor([[9], [10]])), slot_remap=[1, 0, -1, 2])
    assert [owner.slot for owner in owners] == [1, 0, 3]
    assert output.step.owners == (owners[1], owners[0])
    if acknowledge_gather:
        model.note_state_slots_moved({0: 1, 1: 0, 2: 3})
    assert [owner.slot for owner in owners] == [1, 0, 3]
    assert model._spec_owner_slot == 1
    model.release_request(3)
    assert [owner.live for owner in owners] == [True, True, False]


@pytest.mark.parametrize("async_read", [False, True])
def test_readback_wrapper_forwards_native_events_and_same_completion(model, async_read):
    _prefill(model)
    payload = DeviceResult(_tensor([[4]]))
    submitted = _ordinary(model, [3], [2], [10], payload)
    read = model.read_decode_output(submitted, async_read=async_read)
    if async_read:
        read, events = read
        assert events is model.readback_events
    assert read.step is submitted.step
    assert read.on_host is True
    assert model.process_decode_output_host(read, is_tokens=True).tolist() == [[4]]
    assert ("read", payload, async_read) in model.events
    assert len(model._ct_ordinary) == 1


def test_synchronous_finalization_materializes_raw_ordinary_output_once(model):
    _prefill(model)
    payload = DeviceResult(_tensor([[4]]))
    submitted = _ordinary(model, [3], [2], [10], payload, read_from_device=False)
    owner = submitted.step.owners[0]
    assert submitted.on_host is False
    committed = model.process_decode_output_host(submitted, is_tokens=True)
    assert committed.tolist() == [[4]]
    reads = [event for event in model.events if event[0] == "read"]
    conversions = [event for event in model.events if event[0] == "process"]
    assert reads == [("read", payload, False)]
    assert len(conversions) == 1
    assert conversions[0][2] is True
    assert model._ct_ordinary[0] is submitted.step
    assert submitted.step.consumed is False
    proposal = model.propose_draft_tokens(5, committed, _tensor([[3]]), _tensor([1]))
    assert proposal.num_valid.tolist() == [5]
    assert submitted.step.consumed is True
    assert model._ct_decoder_owner is owner
    assert owner.tokens == [1, 2, 3, 4]
    assert len([event for event in model.events if event[0] == "read"]) == 1
    assert len([event for event in model.events if event[0] == "process"]) == 1


def test_already_host_ordinary_output_does_not_read_device(model):
    _prefill(model)
    payload = (_tensor([[4]]), None)
    submitted = _ordinary(model, [3], [2], [10], payload)
    read, events = model.read_decode_output(submitted, async_read=True)
    assert read is submitted
    assert events == []
    assert model.process_decode_output_host(read, is_tokens=True) is payload
    assert not any(event[0] == "read" for event in model.events)


def test_deferred_verify_context_survives_peer_prefill_then_cancellation(model, expect_error):
    _start_solo(model)
    verified = _verify(model)
    owner = verified.hidden.owners[0]
    _prefill(model, keys=(20,), slots=[1], prompts=[[5, 6]])
    model.release_request(0)
    _prefill(model, keys=(10,), slots=[0], prompts=[[50, 51]])
    replacement = model._ct_requests[10]
    before = len([event for event in model.events if event[0] == "replay"])
    declined = _complete(model, [[21, 22, 99]], [[4, 5, 6]], hidden=verified.hidden)
    assert owner.live is False
    assert declined.num_valid.tolist() == [0]
    assert replacement.tokens == [50, 51]
    assert before == len([event for event in model.events if event[0] == "replay"])
    with expect_error(RuntimeError, match="twice"):
        _complete(model, [[21, 22, 99]], [[4, 5, 6]], hidden=verified.hidden)


def test_warmup_prefill_does_not_create_a_request(model):
    model.prefill_forward(
        tokens=_tensor([[0, 0]]),
        prompt_lens=[2],
        empty_slots=[0],
        page_table=_tensor([[0, 0]]),
        warmup_prefill=True,
    )
    assert model._ct_requests == {}


@pytest.mark.parametrize("warmup_name", ["warmup_model_prefill", "warmup_model_decode"])
@pytest.mark.parametrize("fail", [False, True])
def test_inherited_warmup_cannot_register_requests_or_queue_completions(
    model, adapter, monkeypatch, expect_error, warmup_name, fail
):
    _prefill(model)
    owner = model._ct_requests[10]
    native_output = DeviceResult(_tensor([[0]]))

    def inherited_warmup(instance):
        assert instance._ct_warmup_depth == 1
        _prefill(instance, keys=(30,), prompts=[[0, 0]])
        result = _ordinary(instance, [0], [2], [10], native_output)
        assert result is native_output
        if fail:
            raise RuntimeError("stub warmup failure")
        return "warmup complete"

    monkeypatch.setattr(adapter.Gemma4DFlashForCausalLM, warmup_name, inherited_warmup)
    if fail:
        with expect_error(RuntimeError, match="stub warmup failure"):
            getattr(model, warmup_name)()
    else:
        assert getattr(model, warmup_name)() == "warmup complete"
    assert model._ct_warmup_depth == 0
    assert model._ct_requests == {10: owner}
    assert owner.tokens == [1, 2]
    assert not model._ct_ordinary


def test_real_request_bootstraps_when_decode_warmup_is_disabled(model):
    model._decode_warmup_complete = False
    output, proposal = _start_solo(model)
    assert output.step.owners[0] is model._ct_requests[10]
    assert proposal.num_valid.tolist() == [5]


def test_first_rejection_after_peer_join_forces_next_ordinary_host_anchor_reload(model):
    model._spec_decoder.script = [([21, 22, 23, 24, 25], [99, 22, 23, 24, 25, 26])]
    _start_solo(model)
    _prefill(model, keys=(20,), slots=[1], prompts=[[5, 6]])
    model.results.append(DeviceResult(_tensor([[0], [8]])))
    verified = _verify(
        model,
        keys=(10, 20),
        blocks=[[4, 21, 22, 23, 24, 25], [7, 0, 0, 0, 0, 0]],
        positions=[list(range(3, 9)), list(range(2, 8))],
        valid=[5, 0],
        sampling_params=object(),
    )
    assert verified.argmax_ids[:, 0].tolist() == [99, 8]
    mixed = [event[1] for event in model.events if event[0] == "decode"][-1]
    assert mixed["start_pos"].tolist() == [-1, 2]
    _complete(model, [[99], [8]], [[4], [3]], hidden=verified.hidden)
    model.mode = "DECODE"
    model._slots_prefilled_since_decode = set()
    _ordinary(model, [99, 8], [4, 3], [10, 20], DeviceResult(_tensor([[100], [9]])), reset_batch=False)
    reloaded = [event[1] for event in model.events if event[0] == "decode"][-1]
    assert reloaded["reset_batch"] is True
    assert reloaded["tokens"].reshape(-1).tolist() == [99, 8]
    assert model._slots_prefilled_since_decode == {0, 1}
    _ordinary(model, [99, 8], [5, 4], [10, 20], DeviceResult(_tensor([[101], [10]])), reset_batch=False)
    steady = [event[1] for event in model.events if event[0] == "decode"][-1]
    assert steady["reset_batch"] is False


def test_reconstruction_ceiling_declines_without_device_rebuild(model, monkeypatch):
    monkeypatch.setenv("GEMMA4_DFLASH_MAX_SPEC_ISL", "2")
    _prefill(model)
    _ordinary(model, [3], [2], [10], DeviceResult(_tensor([[4]])))
    proposal = _complete(model, [[4]], [[3]])
    assert proposal.num_valid.tolist() == [0]
    assert not any(event[0] in ("sync", "reseed", "replay") for event in model.events)


@pytest.mark.parametrize("max_seq_len, expected_valid", [(8, 0), (9, 5)])
def test_target_sequence_limit_bounds_physical_verify_extent(model, max_seq_len, expected_valid):
    model.model_args[0].max_seq_len = max_seq_len
    _prefill(model)
    _ordinary(model, [3], [2], [10], DeviceResult(_tensor([[4]])))
    proposal = _complete(model, [[4]], [[3]])
    assert proposal.num_valid.tolist() == [expected_valid]
    assert any(event[0] == "replay" for event in model.events) is bool(expected_valid)


@pytest.mark.parametrize("max_seq_len, expected_valid", [(10, 0), (11, 5)])
def test_target_sequence_limit_uses_decoder_physical_width(model, max_seq_len, expected_valid):
    model.model_args[0].max_seq_len = max_seq_len
    model._spec_decoder.P_v = 8
    _prefill(model)
    _ordinary(model, [3], [2], [10], DeviceResult(_tensor([[4]])))
    model.events.clear()
    proposal = _complete(model, [[4]], [[3]])
    assert proposal.num_valid.tolist() == [expected_valid]
    if not expected_valid:
        assert model.events == []


def _verified_completion_at(model, position):
    if position > model._ct_rebuild_storage["capacity"]:
        model._contract_release_rebuild_storage()
        model.model_args[0].max_seq_len = 4096
        model._contract_prepare_rebuild_storage(model.kv_cache)
    _start_solo(model)
    verified = _verify(model)
    owner = model._ct_requests[10]
    owner.tokens = list(range(position))
    model._spec_decoder.start = position - 1
    model.events.clear()
    return verified.hidden


@pytest.mark.parametrize(
    "width, position, expected_valid",
    [(3072, 3002, 5), (3072, 3003, 0), (4096, 4026, 5), (4096, 4027, 0), (4096, 4090, 0), (4096, 4091, 0)],
)
def test_captured_width_limit_declines_before_commit_or_device_work(model, width, position, expected_valid):
    hidden = _verified_completion_at(model, position)
    model.model_args[0].max_seq_len = 4096
    model._spec_decoder._pv_widths = {width: {"trace": 1}}
    proposal = _complete(model, [[42]], [[position]], hidden=hidden)
    assert proposal.num_valid.tolist() == [expected_valid]
    assert model._ct_requests[10].tokens[-1] == 42
    if expected_valid:
        assert any(event[0] == "commit" for event in model.events)
        assert any(event[0] == "replay" for event in model.events)
    else:
        assert model.events == []
        assert model._ct_proposal is None


@pytest.mark.parametrize(
    "budget_end, position, expected_valid",
    [(2199, 2198, 5), (2199, 2199, 0), (4096, 3002, 5), (4096, 3003, 5), (4096, 3066, 5), (4096, 3067, 0)],
)
def test_fixed_capture_checks_policy_and_physical_extent_before_commit(model, budget_end, position, expected_valid):
    hidden = _verified_completion_at(model, position)
    model.model_args[0].max_seq_len = 4096
    model._spec_width_set = False
    model._spec_decoder.pv_sk = 3072
    model._spec_budget_end = budget_end
    proposal = _complete(model, [[42]], [[position]], hidden=hidden)
    assert proposal.num_valid.tolist() == [expected_valid]
    assert model._ct_requests[10].tokens[-1] == 42
    if expected_valid:
        assert any(event[0] == "commit" for event in model.events)
        assert any(event[0] == "replay" for event in model.events)
    else:
        assert model.events == []
        assert model._ct_proposal is None


@pytest.mark.parametrize("width_set", [False, True])
def test_exhausted_capture_keeps_ordinary_progress_without_rebuilding(model, width_set):
    position = 4027 if width_set else 2199
    hidden = _verified_completion_at(model, position)
    model.model_args[0].max_seq_len = 8192
    model._spec_width_set = width_set
    model._spec_decoder._pv_widths = {4096: {"trace": 1}}
    model._spec_decoder.pv_sk = 4096
    model._spec_budget_end = 2199
    assert _complete(model, [[42]], [[position]], hidden=hidden).num_valid.tolist() == [0]
    for offset in (1, 3):
        _ordinary(model, [43], [position + offset], [10], DeviceResult(_tensor([[44]])))
        model.events.clear()
        proposal = _complete(model, [[44]], [[position + offset + 1]])
        assert proposal.num_valid.tolist() == [0]
        assert model.events == []
        assert model._ct_requests[10].tokens[-2:] == [43, 44]


def test_prepared_widths_capture_once_before_first_request_reconstruction(model):
    decoder = model._spec_decoder
    decoder._pv_widths = {1024: {"trace": None}}
    decoder._prepared_widths = {1024}
    _start_solo(model)
    captures = [event for event in model.events if event[0] == "capture_widths"]
    assert captures == [("capture_widths", (1024,))]
    names = [event[0] for event in model.events]
    assert names.index("sync") < names.index("capture_widths") < names.index("ingest") < names.index("replay")
    reconstruction = next(
        index for index, event in enumerate(model.events) if event[0] == "prefill" and event[1]["prompt_lens"] == [3]
    )
    assert names.index("capture_widths") < reconstruction
    _ordinary(model, [21], [4], [10], DeviceResult(_tensor([[22]])))
    _complete(model, [[22]], [[5]])
    assert sum(event[0] == "capture_widths" for event in model.events) == 1


def test_prepared_widths_do_not_capture_an_unsupported_request(model):
    decoder = model._spec_decoder
    decoder._pv_widths = {1024: {"trace": None}}
    decoder._prepared_widths = {1024}
    _prefill(model, prompts=[list(range(954))])
    _ordinary(model, [3], [954], [10], DeviceResult(_tensor([[4]])))
    model.events.clear()
    proposal = _complete(model, [[4]], [[955]])
    assert proposal.num_valid.tolist() == [0]
    assert model.events == []


def test_contract_capabilities_default_to_synchronous(adapter):
    capabilities = adapter.Gemma4DFlashContractForCausalLM.model_capabilities
    assert capabilities["output_tokens_per_step"] == 1
    assert capabilities["supports_spec_decode"] is True
    assert capabilities["supports_sample_on_device"] is True
    assert capabilities["supports_async_decode"] is False
    assert capabilities["supports_async_spec_decode"] is False
    assert "tt_adaptive_block_output" not in capabilities
    assert capabilities["spec_hidden_handoff"] == ("on_device",)


def test_actual_spec_plan_admits_narrow_concurrent_decode_with_drafter_accounting(adapter, monkeypatch):
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", "/unused/host-test-drafter")
    monkeypatch.setattr(
        adapter,
        "_dflash_drafter_config",
        lambda snapshot: {
            "num_hidden_layers": 5,
            "hidden_size": 128,
            "head_dim": 32,
            "num_key_value_heads": 8,
            "block_size": 16,
        },
    )
    monkeypatch.setattr(adapter, "_dflash_mesh_tp", lambda: 8)
    plan = adapter.Gemma4DFlashContractForCausalLM.spec_plan(_serving_config(), 4, 5)
    assert plan.supports_narrow_decode is True
    assert plan.effective_k == 5
    assert plan.lanes_per_request == 1
    assert plan.accept_modes == ("argmax_ids",)
    assert plan.extra_bytes_per_seq > 0
