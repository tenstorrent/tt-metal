# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression for the prefill lm_head / device-sampler boundary."""

from types import SimpleNamespace

import pytest

import models.demos.gemma4.tt.ccl as ccl_module
import models.demos.gemma4.tt.model as model_module
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.generator import Generator


@pytest.fixture
def prefill_projection(monkeypatch):
    shard, gathered = object(), object()
    hidden = SimpleNamespace(shape=(1, 1, 32, 8), deallocate=lambda force: None)
    model = object.__new__(model_module.Gemma4Model)
    model.mesh_device = object()
    model.mesh_config = SimpleNamespace(tp=2)
    model.ccl_manager = object()
    model.hidden_size = 8
    model.lm_head_weight = SimpleNamespace(shape=(1, 1, 8, 16))
    model.final_logit_softcapping = None
    model.sampling = object()
    model._supports_on_device_sampling = True
    generator = object.__new__(Gemma4Generator)
    generator.model = [model]
    generator.data_parallel = 1
    monkeypatch.setattr(model_module, "_get_lm_head_program_config", lambda *a, **kw: None)
    monkeypatch.setattr(model_module.ttnn, "linear", lambda *a, **kw: shard)
    monkeypatch.setattr(ccl_module, "ccl_allgather", lambda *a, **kw: gathered)
    return generator, model, hidden, shard, gathered


def test_single_user_prefill_preserves_sampler_shards_and_host_vocabulary(monkeypatch, prefill_projection):
    generator, model, hidden, shard, gathered = prefill_projection

    def forward(self, *args, **kwargs):
        # Both eager prefill and traced prefill's deferred lm_head call this
        # without an explicit keep_sharded_for_sampling argument.
        return model._apply_lm_head(hidden, is_decode=False)

    monkeypatch.setattr(Generator, "_prefill_forward_text_impl", forward)
    assert generator._prefill_forward_text_impl(sampling_params=object()) is shard
    assert generator._prefill_forward_text_impl(sampling_params=None) is gathered
    assert model._prefill_keep_logits_sharded is False
    model._supports_on_device_sampling = False
    assert generator._prefill_forward_text_impl(sampling_params=object()) is gathered


def test_nested_host_warmup_restores_sampling_layout(monkeypatch, prefill_projection):
    generator, model, hidden, shard, gathered = prefill_projection

    def forward(self, *args, sampling_params=None, **kwargs):
        if sampling_params is not None:
            assert self._prefill_forward_text_impl(sampling_params=None) is gathered
        return model._apply_lm_head(hidden, is_decode=False)

    monkeypatch.setattr(Generator, "_prefill_forward_text_impl", forward)
    assert generator._prefill_forward_text_impl(sampling_params=object()) is shard
    assert model._prefill_keep_logits_sharded is False


def test_failed_sampling_prefill_restores_host_layout(monkeypatch, prefill_projection):
    generator, model, hidden, shard, gathered = prefill_projection

    def forward(self, *args, **kwargs):
        assert model._apply_lm_head(hidden, is_decode=False) is shard
        raise RuntimeError("prefill failed")

    monkeypatch.setattr(Generator, "_prefill_forward_text_impl", forward)
    with pytest.raises(RuntimeError, match="prefill failed"):  # allow-pytest.raises: host-only layout regression
        generator._prefill_forward_text_impl(sampling_params=object())
    assert model._apply_lm_head(hidden, is_decode=False) is gathered
