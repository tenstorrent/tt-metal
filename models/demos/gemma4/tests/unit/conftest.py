# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the Gemma4 dFlash contract host tests.

``adapter`` imports the production ``generator_vllm`` module with the TT
runtime and the lower model modules replaced by stubs; ``drafter_module``
imports the production ``dflash_drafter`` module the same way; ``model``
builds a ``Gemma4DFlashContractForCausalLM`` whose target and fused decoder
are recording stubs. ``expect_error`` stays per test module: the repository
fixture of that name is excluded by ``--confcutdir`` and would import the TT
runtime, and a copy here would shadow it for the device tests in this
directory.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.demos.gemma4.tests.unit.dflash_contract_harness import (  # noqa: E402
    Decoder,
    DeviceResult,
    Target,
    _align_down,
    _module,
    _padded_length,
    _unused,
)

_ENV_CLEARED = (
    "GEMMA4_CONTRACT_ASYNC",
    "GEMMA4_DFLASH_MAX_SPEC_ISL",
    "GEMMA4_DFLASH_SERVE_BLOCK",
    "GEMMA4_DFLASH_WIDTH_SET",
    "GEMMA4_DFLASH_PACKED",
    "GEMMA4_DFLASH_WARMUP_DECODE",
    "GEMMA4_DFLASH_CTX_CAP",
    "GEMMA4_BOUNDED_SLIDING_KV_CACHE",
    "GEMMA4_SPEC_RING_HEADROOM_BLOCKS",
)


def _stub_runtime(patch):
    """Replace ``ttnn`` and the lower model modules for a host-only import."""
    patch.syspath_prepend(str(_ROOT))
    for name in _ENV_CLEARED:
        patch.delenv(name, raising=False)
    patch.setenv("GEMMA4_DFLASH_VERIFY", "5")
    runtime = _module(
        "ttnn",
        synchronize_device=lambda mesh: None,
        begin_trace_capture=_unused,
        end_trace_capture=_unused,
        release_trace=_unused,
        execute_trace=_unused,
        Tensor=type("Tensor", (), {}),
        MemoryConfig=object,
        DRAM_MEMORY_CONFIG=object(),
        ROW_MAJOR_LAYOUT=object(),
        TILE_LAYOUT=object(),
        uint32=object(),
        int32=object(),
        bfloat16=object(),
    )
    patch.setitem(sys.modules, "ttnn", runtime)
    lower_modules = {
        "models.demos.gemma4.tt.common": {"create_tt_model": _unused},
        "models.demos.gemma4.tt.generator": {
            "SDPA_CHUNK_ALIGN": 128,
            "ChunkedPrefillPageTableGuardMixin": type("ChunkedPrefillPageTableGuardMixin", (), {}),
            "align_num_cached_tokens_to_sdpa": _align_down,
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
            "HybridAttentionForCausalLM": type(
                "HybridAttentionForCausalLM",
                (),
                {"release_persistent_capture": lambda self: self.__dict__.setdefault("base_releases", []).append(1)},
            ),
            "allocate_vllm_kv_cache": _unused,
        },
        "models.demos.gemma4.tt.attention": {
            "_RING_HEADROOM_BLOCK": 64,
            "SPEC_RING_HEADROOM_ENV": "GEMMA4_SPEC_RING_HEADROOM_BLOCKS",
            "bounded_ring_modulo": lambda window: window,
        },
        "models.demos.gemma4.tt.ccl": {"ccl_allgather": _unused, "ccl_allreduce": _unused},
        "models.demos.gemma4.utils.general_utils": {"get_cache_file_name": _unused},
    }
    for name, members in lower_modules.items():
        patch.setitem(sys.modules, name, _module(name, **members))
    try:
        importlib.import_module("loguru")
    except ImportError:
        logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
        patch.setitem(sys.modules, "loguru", _module("loguru", logger=logger))
    try:
        importlib.import_module("vllm_tt_plugin.spec_decode")
    except ImportError:
        warnings.warn(
            "vllm_tt_plugin is not importable; the host tests run against stub SpecPlan, DraftOutput and "
            "VerifyOutput types. Put the plugin checkout's src on PYTHONPATH for the production types.",
            stacklevel=1,
        )

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


def _fresh_import(patch, name):
    parent_name, _, attr = name.rpartition(".")
    parent = importlib.import_module(parent_name)
    patch.setattr(parent, attr, None, raising=False)
    patch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def import_adapter(patch, env=None):
    """The production ``generator_vllm`` under stubs, with ``env`` applied first."""
    _stub_runtime(patch)
    for key, value in (env or {}).items():
        patch.setenv(key, value)
    patch.setitem(
        sys.modules, "models.demos.gemma4.tt.dflash_drafter", _module("models.demos.gemma4.tt.dflash_drafter")
    )
    return _fresh_import(patch, "models.demos.gemma4.tt.generator_vllm")


@pytest.fixture(scope="module")
def adapter():
    with pytest.MonkeyPatch.context() as patch:
        yield import_adapter(patch)


@pytest.fixture(scope="module")
def drafter_module():
    """The production ``dflash_drafter`` under stubs (no device, no weights)."""
    with pytest.MonkeyPatch.context() as patch:
        _stub_runtime(patch)
        yield _fresh_import(patch, "models.demos.gemma4.tt.dflash_drafter")


def build_model(adapter, monkeypatch, *, ring=None, widths=(1024,)):
    """A contract-rail model over recording stubs, after a solo-capable warmup."""
    cls = adapter.Gemma4DFlashContractForCausalLM
    model = cls.__new__(cls)
    model.events = []
    model.model = [Target(model.events, ring=ring)]
    model.model_args = [SimpleNamespace(max_seq_len=1024, max_batch_size=32)]
    model.kv_cache = [[("k", "v"), ("k", "v")]]
    model._spec_decoder = Decoder(model.events, widths=widths)
    model._spec_width_set = True
    model._spec_width_ladder = list(widths)
    model._spec_horizon = 256
    model._spec_active = False
    model._spec_active_owner = None
    model._spec_owner_slot = None
    model._spec_pending = None
    model._spec_pending_owner = None
    model._spec_carry = []
    model._spec_first_step = True
    model._spec_decoder_bucket = None
    model._bounded_sliding_kv_cache = ring is not None
    model._SPEC_CONTRACT_K = 5
    model._slots_prefilled_since_decode = set()
    model.results = []
    model._dflash_init_state()
    model._spec_get_drafter = lambda: SimpleNamespace(target_layer_ids=[1, 2])
    model._effective_paged_block_size = lambda kv_cache: 64
    model._sliding_layer_indices = lambda: [0]
    model._build_per_layer_page_tables = lambda per_layer, table: [table, table]
    model._pad_sliding_page_tables_for_bounded = lambda per_layer, kv_cache, authoritative=False: per_layer

    def bootstrap(anchor, start, page_table, kv_cache, page_tables_per_layer=None):
        taps, n = model._spec_pending
        model._spec_pending = None
        model._spec_pending_owner = None
        dec = model._spec_decoder
        row = page_table[:1]
        dec.refresh_page_tables(row)
        dec.select_width(int(start))
        dec.prefill_ingest(taps, n)
        dec.reseed(int(anchor), int(start))
        model._spec_active = True
        model._spec_first_step = True
        model.events.append(("bootstrap", int(anchor), int(start)))

    def release_decoder(drop_page_tables=True, teardown=False):
        model._spec_active = False
        model._spec_active_owner = None
        model.events.append(("release_decoder", bool(teardown)))

    monkeypatch.setattr(model, "_spec_bootstrap", bootstrap)
    monkeypatch.setattr(model, "_spec_release_decoder", release_decoder)

    def prefill(self, *args, page_tables_per_layer=None, **kwargs):
        tokens = kwargs["tokens"]
        model.events.append(("prefill", int(tokens.shape[0]), kwargs.get("prompt_lens"), kwargs.get("enable_trace")))
        model.last_prefill_tables = (kwargs.get("page_table"), page_tables_per_layer)
        target = model.model[0]
        if target.tap_layers is not None:
            group = len(target.tap_layers)
            rows = int(tokens.shape[1]) - int(list(kwargs.get("start_pos") or [0])[0])
            target.taps.extend(model.harness_taps(rows, group))
        return torch.zeros(int(tokens.shape[0]), 1, dtype=torch.int32)

    def decode(self, *args, page_tables_per_layer=None, **kwargs):
        model.events.append(("decode", kwargs["tokens"].reshape(-1).tolist(), kwargs["start_pos"].reshape(-1).tolist()))
        assert model.results, "Supply a plain decode result before submitting the step"
        return model.results.pop(0)

    def read(self, tt_out, async_read=False):
        model.events.append(("read", bool(async_read)))
        return [tt_out]

    base = adapter.Gemma4ForCausalLM
    monkeypatch.setattr(base, "prefill_forward", prefill)
    monkeypatch.setattr(base, "decode_forward", decode)
    monkeypatch.setattr(base, "read_decode_output", read, raising=False)
    model.tap_counter = [0]

    def harness_taps(rows, group):
        from models.demos.gemma4.tests.unit.dflash_contract_harness import Tap

        model.tap_counter[0] += 1
        return [Tap(rows, (model.tap_counter[0], layer)) for layer in range(group)]

    model.harness_taps = harness_taps
    return model


@pytest.fixture
def model(adapter, monkeypatch):
    return build_model(adapter, monkeypatch)


@pytest.fixture
def device_result():
    return DeviceResult
