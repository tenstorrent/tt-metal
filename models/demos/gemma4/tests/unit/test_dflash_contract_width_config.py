# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks that admitted dFlash draft counts match physical decoder rows.

The production plan and decoder constructor execute with the tensor runtime
from the width preparation tests. Fused computation and checkpoint loading
are stubbed; these tests do not establish device correctness.
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import _padded_length, _serving_config
from models.demos.gemma4.tests.unit.test_dflash_width_prepare import Target, _drafter
from models.demos.gemma4.tests.unit.test_dflash_width_prepare import generator_module as _generator_module_fixture
from models.demos.gemma4.tests.unit.test_dflash_width_prepare import production as _production_fixture

production = _production_fixture


@pytest.fixture
def width_config(request, production, monkeypatch):
    verify, checkpoint_block, override_block = request.param
    if verify is None:
        monkeypatch.delenv("GEMMA4_DFLASH_VERIFY", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", verify)
    if override_block is None:
        monkeypatch.delenv("GEMMA4_DFLASH_BLOCK", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_DFLASH_BLOCK", override_block)
    monkeypatch.setenv("GEMMA4_DFLASH_WIDTH_SET", "1")
    monkeypatch.setenv("GEMMA4_DFLASH_WARMUP_DECODE", "1")
    monkeypatch.setenv("GEMMA4_DFLASH_PACKED", "1")
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", "/unused/host-test-drafter")
    monkeypatch.setenv("GEMMA4_DFLASH_DECODER_REUSE", "0")
    generator_import = _generator_module_fixture.__wrapped__(production, monkeypatch)
    module = next(generator_import)
    monkeypatch.setattr(module, "get_padded_prefill_len", _padded_length)
    checkpoint = {
        "num_hidden_layers": 1,
        "hidden_size": 8,
        "head_dim": 2,
        "num_key_value_heads": 1,
        "block_size": checkpoint_block,
    }
    monkeypatch.setattr(module, "_dflash_drafter_config", lambda snapshot: checkpoint)
    monkeypatch.setattr(module, "_dflash_mesh_tp", lambda: 1)
    drafter = _drafter(production.runtime)
    drafter.block_size = int(override_block) if override_block is not None else checkpoint_block
    yield SimpleNamespace(
        module=module,
        cls=module.Gemma4DFlashContractForCausalLM,
        production=production,
        drafter=drafter,
        expected_verify=5 if verify is None else int(verify),
    )
    next(generator_import, None)


def _model(width_config):
    model = width_config.cls.__new__(width_config.cls)
    model._contract_init()
    model.model = [Target(width_config.production.runtime)]
    model.model_args = [SimpleNamespace(max_seq_len=4096)]
    model.mesh_device = width_config.production.runtime
    model._spec_decoder = None
    model._spec_decoder_bucket = None
    model._spec_width_set = True
    model._spec_width_ladder = None
    model._spec_horizon = 2048
    model._bounded_sliding_kv_cache = False
    model._spec_get_drafter = lambda: width_config.drafter
    return model


@pytest.mark.parametrize(
    "width_config",
    [(None, 16, None), ("5", 16, None), ("7", 16, None), ("7", 16, "8"), ("15", 16, None)],
    indirect=True,
)
@pytest.mark.parametrize("construction", ["startup", "fresh"])
def test_admitted_count_matches_real_decoder_constructor(width_config, construction, monkeypatch):
    from vllm_tt_plugin.spec_decode import SpecPlan

    plan = width_config.cls.spec_plan(_serving_config(), 4, 15)
    assert isinstance(plan, SpecPlan)
    assert plan.supports_narrow_decode
    assert plan.effective_k == width_config.expected_verify
    assert width_config.cls._SPEC_CONTRACT_K == width_config.cls._SPEC_V == plan.effective_k
    assert width_config.cls._SPEC_N == plan.effective_k + 1
    model = _model(width_config)
    if construction == "startup":
        model._spec_capture_width_set(object(), 64, prepare_only=True)
    else:
        decoder_cls = width_config.production.module.DFlashFusedDecoder
        monkeypatch.setattr(decoder_cls, "prefill_ingest", lambda decoder, taps, length: None)
        monkeypatch.setattr(decoder_cls, "capture", lambda decoder, anchor, start, max_new: None)
        model._spec_pending = ([], 3)
        model._spec_bootstrap(17, 3, torch.zeros(1, 64, dtype=torch.int32), object())
    decoder = model._spec_decoder
    assert decoder.rotate_ring_reads is True
    assert decoder.V == plan.effective_k
    assert decoder.P_v == plan.effective_k + 1
    assert decoder.fc_prev.shape[2] == decoder.P_v
    assert decoder.commit_pos.shape[1] == decoder.P_v


@pytest.mark.parametrize("width_config", [("0", 16, None), ("16", 16, None), ("8", 16, "8")], indirect=True)
def test_contract_rejects_count_outside_actual_drafter_block(width_config):
    from vllm_tt_plugin.spec_decode import SpecReject

    plan = width_config.cls.spec_plan(_serving_config(), 4, 32)
    assert isinstance(plan, SpecReject)
    assert "outside" in plan.reason
    assert f"block_size={width_config.drafter.block_size}" in plan.reason
    assert width_config.production.runtime.events == []


@pytest.mark.parametrize("width_config", [(None, 16, None)], indirect=True)
@pytest.mark.parametrize(
    "setting,value",
    [
        ("GEMMA4_DFLASH_PACKED", "0"),
        ("GEMMA4_DFLASH_WIDTH_SET", "0"),
        ("GEMMA4_DFLASH_WIDTH_SET", "false"),
        ("GEMMA4_DFLASH_WARMUP_DECODE", "0"),
        ("GEMMA4_DFLASH_WARMUP_DECODE", "no"),
    ],
)
def test_contract_rejects_diagnostic_paths_without_preparation(width_config, monkeypatch, setting, value):
    from vllm_tt_plugin.spec_decode import SpecReject

    monkeypatch.setenv(setting, value)
    plan = width_config.cls.spec_plan(_serving_config(), 4, 5)
    assert isinstance(plan, SpecReject)
    assert f"{setting}=1" in plan.reason
    assert width_config.production.runtime.events == []


@pytest.mark.parametrize("width_config", [(None, 16, None)], indirect=True)
def test_contract_admits_unset_preparation_defaults(width_config, monkeypatch):
    from vllm_tt_plugin.spec_decode import SpecPlan

    for setting in ("GEMMA4_DFLASH_PACKED", "GEMMA4_DFLASH_WIDTH_SET", "GEMMA4_DFLASH_WARMUP_DECODE"):
        monkeypatch.delenv(setting, raising=False)
    plan = width_config.cls.spec_plan(_serving_config(), 4, 5)
    assert isinstance(plan, SpecPlan)
    assert plan.effective_k == 5 and plan.supports_narrow_decode


@pytest.mark.parametrize("width_config", [(None, 16, None)], indirect=True)
@pytest.mark.parametrize("value", ["true", "yes", "TRUE"])
def test_contract_accepts_runtime_preparation_boolean_values(width_config, monkeypatch, value):
    from vllm_tt_plugin.spec_decode import SpecPlan

    monkeypatch.setenv("GEMMA4_DFLASH_WIDTH_SET", value)
    monkeypatch.setenv("GEMMA4_DFLASH_WARMUP_DECODE", value)
    assert isinstance(width_config.cls.spec_plan(_serving_config(), 4, 5), SpecPlan)


@pytest.mark.parametrize("width_config", [(None, 16, None)], indirect=True)
def test_standalone_decoder_preserves_unset_environment_default(width_config):
    decoder = width_config.production.module.DFlashFusedDecoder(
        Target(width_config.production.runtime),
        width_config.drafter,
        object(),
        torch.zeros(1, 64, dtype=torch.int32),
        ctx_cap=32,
    )
    assert decoder.rotate_ring_reads is False
    assert decoder.V == width_config.drafter.block_size - 1
    assert decoder.P_v == width_config.drafter.block_size


@pytest.mark.parametrize("width_config", [(None, 16, None)], indirect=True)
def test_legacy_block_decoder_preserves_nonpacked_fresh_construction(width_config, monkeypatch):
    monkeypatch.setenv("GEMMA4_DFLASH_PACKED", "0")
    decoder_cls = width_config.production.module.DFlashFusedDecoder
    monkeypatch.setattr(decoder_cls, "prefill_ingest", lambda decoder, taps, length: None)
    monkeypatch.setattr(decoder_cls, "capture", lambda decoder, anchor, start, max_new: None)
    legacy_cls = width_config.module.Gemma4DFlashForCausalLM
    model = legacy_cls.__new__(legacy_cls)
    model.model = [Target(width_config.production.runtime)]
    model._spec_decoder = None
    model._spec_width_set = False
    model._spec_pending = ([], 3)
    model._spec_horizon = 2048
    model._bounded_sliding_kv_cache = False
    model._spec_get_drafter = lambda: width_config.drafter
    model._spec_bootstrap(17, 3, torch.zeros(1, 64, dtype=torch.int32), object())
    assert model._spec_decoder.rotate_ring_reads is False
    assert model._spec_decoder.P_v == width_config.drafter.block_size


@pytest.mark.parametrize("width_config", [(None, 16, None)], indirect=True)
@pytest.mark.parametrize("verify_count", [0, 16])
def test_explicit_decoder_count_fails_before_device_allocation(width_config, verify_count):
    expect_error = pytest.raises  # allow-pytest.raises: host harness excludes device fixtures
    with expect_error(ValueError, match="verify_count must be"):
        width_config.production.module.DFlashFusedDecoder(
            Target(width_config.production.runtime),
            width_config.drafter,
            object(),
            torch.zeros(1, 64, dtype=torch.int32),
            verify_count=verify_count,
        )
    assert width_config.production.runtime.events == []
