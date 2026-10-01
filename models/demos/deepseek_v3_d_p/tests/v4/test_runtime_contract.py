# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The seam between the shared prefill runtime, the V4 adapters and TtV4Transformer.

`TtPrefillRuntime` builds and drives `MODEL_CLS` with fixed keyword lists, and a knob the transformer
does not name is swept into `**block_kwargs` and raises in `TtV4Block` only after mesh bring-up and the
checkpoint load. These tests bind the call sites, read off the runtime's source, against the V4
signatures, and check the runtime's and adapter's V4-specific rules.

Hardware-free: no mesh, no weights, no checkpoint.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

from models.demos.common.prefill.adapter import ADAPTER_PATHS, get_adapter
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime
from models.demos.deepseek_v3_d_p.tt.v4.block import TtV4Block
from models.demos.deepseek_v3_d_p.tt.v4.runtime import TtV4Runtime
from models.demos.deepseek_v3_d_p.tt.v4.transformer import TtV4Transformer

_RUNTIME_SRC = Path(__file__).parents[2] / "tt" / "tt_prefill_runtime.py"
_ADAPTERS = [
    pytest.param("deepseek_v4_pro", DeepSeekV4ProConfig, id="pro"),
    pytest.param("deepseek_v4_flash", DeepSeekV4FlashConfig, id="flash"),
]


def _call_sites(callee: str, expected: int = 1) -> list[ast.Call]:
    """The calls to `callee` in `TtPrefillRuntime`, read off the source."""
    tree = ast.parse(_RUNTIME_SRC.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]
    assert len(calls) == expected, f"expected {expected} {callee} call site(s), found {len(calls)}"
    for call in calls:
        assert not any(isinstance(arg, ast.Starred) for arg in call.args) and all(
            kw.arg for kw in call.keywords
        ), f"{callee} is called with */** expansion, which this test cannot bind"
    return calls


def _bind(func, call: ast.Call, *leading) -> None:
    inspect.signature(func).bind(*leading, *[None] * len(call.args), **{kw.arg: None for kw in call.keywords})


def test_every_runtime_kwarg_binds_to_the_transformer_or_the_block():
    passed = [kw.arg for kw in _call_sites("self.MODEL_CLS")[0].keywords]
    named = set(inspect.signature(TtV4Transformer.__init__).parameters)
    accepted = set(inspect.signature(TtV4Block.__init__).parameters)
    orphans = [k for k in passed if k not in named and k not in accepted]
    assert not orphans, f"{orphans} reach TtV4Transformer from the runtime and bind to neither it nor TtV4Block"


def test_the_cache_check_call_binds_to_the_transformer():
    _bind(TtV4Transformer.check_cache_complete, _call_sites("self.MODEL_CLS.check_cache_complete")[0])


def test_every_forward_call_binds_to_the_transformer():
    for call in _call_sites("self.model.forward", expected=2):
        _bind(TtV4Transformer.forward, call, None)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"mtp_union": object()}, ValueError, "no MTP predictor"),
        ({"input_is_embedded": True}, ValueError, "no MTP predictor"),
        ({"kvpe_cache": object()}, ValueError, "owns its state"),
        ({"metadata": object()}, NotImplementedError, "no traced path"),
        ({"d2h_service": object()}, NotImplementedError, "host layer acks"),
        ({"cache_user_id": 1}, ValueError, "single-user"),
    ],
)
def test_forward_rejects_unsupported_arguments_before_device_work(expect_error, kwargs, error, message):
    model = object.__new__(TtV4Transformer)
    model.position = 0
    with expect_error(error, message):
        model.forward(object(), **kwargs)


def test_forward_rejects_a_chunk_that_does_not_continue_the_state(expect_error):
    model = object.__new__(TtV4Transformer)
    model.position = 5120
    with expect_error(ValueError, "attention state is at 5120"):
        model.forward(object(), actual_start=0)


@pytest.mark.parametrize(
    "config, message",
    [
        (dict(mtp_levels=1, use_trace=False, num_users=1, dflash_enabled=False), "no MTP predictor"),
        (dict(mtp_levels=0, use_trace=True, num_users=1, dflash_enabled=False), "eager-only"),
        (dict(mtp_levels=0, use_trace=False, num_users=2, dflash_enabled=False), "single-user"),
        (dict(mtp_levels=0, use_trace=False, num_users=1, dflash_enabled=True), "no DFlash"),
    ],
)
def test_runtime_rejects_unsupported_config_before_the_shared_build(monkeypatch, expect_error, config, message):
    runtime = object.__new__(TtV4Runtime)
    runtime.config = SimpleNamespace(**config)
    parent_build = create_autospec(TtPrefillRuntime._build_model)
    monkeypatch.setattr(TtPrefillRuntime, "_build_model", parent_build)
    with expect_error(ValueError, message):
        runtime._build_model({})
    parent_build.assert_not_called()


@pytest.mark.parametrize("actual_start, resets", [(0, True), (5120, False)])
def test_runtime_resets_the_attention_state_only_at_request_head(monkeypatch, actual_start, resets):
    runtime = object.__new__(TtV4Runtime)
    runtime.model = Mock()
    parent_chunk = create_autospec(TtPrefillRuntime.prefill_chunk)
    monkeypatch.setattr(TtPrefillRuntime, "prefill_chunk", parent_chunk)
    # Positional, as the runner calls it: slot_id is the third argument and actual_start the fourth.
    runtime.prefill_chunk(object(), object(), 0, actual_start, actual_start + 5120)
    assert runtime.model.reset_streams.called == resets
    parent_chunk.assert_called_once()


@pytest.mark.parametrize("name, model_cfg", _ADAPTERS)
def test_adapter_is_registered_and_builds_the_model_schedule(name, model_cfg):
    assert name in ADAPTER_PATHS
    adapter = get_adapter(name)
    assert adapter.name == name and adapter.model_config is model_cfg
    assert adapter.transformer_cls is TtV4Transformer

    config = adapter.load_hf_config()
    assert config.num_hidden_layers == model_cfg.NUM_LAYERS
    assert config.hidden_size == model_cfg.EMB_SIZE and config.q_lora_rank == model_cfg.Q_LORA_RANK
    kinds = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}
    assert config.layer_types == [kinds[r] for r in model_cfg.COMPRESS_RATIOS]
    assert config.mlp_layer_types[: model_cfg.NUM_HASH_LAYERS] == ["hash_moe"] * model_cfg.NUM_HASH_LAYERS
    assert set(config.mlp_layer_types[model_cfg.NUM_HASH_LAYERS :]) == {"moe"}


@pytest.mark.parametrize("name, model_cfg", _ADAPTERS)
def test_adapter_keeps_the_hash_layers_on_the_first_rank(name, model_cfg):
    boundaries = get_adapter(name).layer_split_boundaries(model_cfg.NUM_LAYERS)
    assert 0 in boundaries
    assert not boundaries & set(range(1, model_cfg.NUM_HASH_LAYERS))
    assert set(range(model_cfg.NUM_HASH_LAYERS, model_cfg.NUM_LAYERS)) <= boundaries


@pytest.mark.parametrize("name, model_cfg", _ADAPTERS)
def test_adapter_allocates_no_kv_cache_and_no_weight_cache(name, model_cfg):
    adapter = get_adapter(name)
    assert adapter.allocate_kv_cache(mesh_device=None, hf_config=None, params=None).kvpe is None
    assert adapter.weight_cache_path((8, 4)) is None
