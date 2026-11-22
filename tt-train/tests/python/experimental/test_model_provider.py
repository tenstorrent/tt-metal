# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The provider ``get_provider`` picks for ``model.name``, and where it loads the model from. Device-free, and
offline: no hub id is looked up."""

import json
from types import SimpleNamespace

import pytest

import ttml.experimental.models.provider as model_provider
from ttml.experimental.errors import UnsupportedTrainingConfig
from ttml.experimental.models.provider import (
    LlamaProvider,
    get_provider,
    providers,
    register_provider,
)


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    """Registrations a test makes don't outlive it."""
    monkeypatch.setattr(model_provider, "_PROVIDERS", dict(model_provider._PROVIDERS))


def _config(name: str) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(name=name))


def _local_model(directory) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 64}))
    return str(directory)


def test_the_models_ttml_ships_are_registered_by_hub_id():
    assert providers() == {
        "meta-llama/Llama-3.2-1B": LlamaProvider,
        "meta-llama/Llama-3.2-1B-Instruct": LlamaProvider,
        "meta-llama/Llama-3.1-8B": LlamaProvider,
        "meta-llama/Llama-3.1-8B-Instruct": LlamaProvider,
    }


def test_a_provider_registered_from_outside_builds_its_model(tmp_path):
    source = _local_model(tmp_path)

    class AcmeProvider(LlamaProvider):
        def model_source(self) -> str:
            return source

    register_provider("acme/acme-7b", AcmeProvider)
    provider = get_provider(_config("acme/acme-7b"))
    assert isinstance(provider, AcmeProvider)
    assert provider.hf_config["hidden_size"] == 64


def test_a_provider_registered_from_outside_takes_back_checkpoints_of_its_model(tmp_path, expect_error):
    source = _local_model(tmp_path / "Mistral-7B-v0.1")

    class MistralProvider(LlamaProvider):
        def model_source(self) -> str:
            return source

    register_provider("mistralai/Mistral-7B-v0.1", MistralProvider)
    provider = get_provider(_config("mistralai/Mistral-7B-v0.1"))
    # A PEFT adapter records model_source(), a tt checkpoint model.name.
    provider._check_base_model(tmp_path, source)
    provider._check_base_model(tmp_path, "mistralai/Mistral-7B-v0.1")
    with expect_error(UnsupportedTrainingConfig, r"^checkpoint: .* is for 'meta-llama/Llama-3.2-1B'"):
        provider._check_base_model(tmp_path, "meta-llama/Llama-3.2-1B")


def test_registering_the_same_provider_again_is_a_no_op():
    register_provider("meta-llama/Llama-3.2-1B", LlamaProvider)
    assert providers()["meta-llama/Llama-3.2-1B"] is LlamaProvider


def test_a_hub_id_has_one_provider(expect_error):
    class OtherProvider(LlamaProvider):
        pass

    with expect_error(ValueError, "already has a provider, LlamaProvider"):
        register_provider("meta-llama/Llama-3.2-1B", OtherProvider)


@pytest.mark.parametrize("name", ["Llama-3.2-1B", "someone/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"])
def test_model_name_is_a_registered_hub_id(name, expect_error):
    with expect_error(UnsupportedTrainingConfig, r"^model\.name: "):
        get_provider(_config(name))


def test_model_name_is_not_a_local_directory(tmp_path, expect_error):
    with expect_error(UnsupportedTrainingConfig, r"^model\.name: "):
        get_provider(_config(_local_model(tmp_path / "Llama-3.2-1B")))
