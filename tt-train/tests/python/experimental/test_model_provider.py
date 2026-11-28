# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The provider ``get_provider`` picks for ``model.name``, where it loads the model from, and the checkpoint files it
writes and loads back. Device-free, and offline: no hub id is looked up."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ttml
import ttml.experimental.models.provider as model_provider
from ttml import checkpointing
from ttml.experimental.errors import UnsupportedTrainingConfig
from ttml.experimental.models.provider import (
    LlamaProvider,
    get_provider,
    providers,
    register_provider,
)
from ttml.experimental.models.utils.checkpoint import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    TT_CHECKPOINT,
    AdapterState,
    read_adapter,
)

LORA = SimpleNamespace(r=8, alpha=16, dropout=0.0, target_modules=["v_proj", "q_proj"], task_type="CAUSAL_LM")
OTHER_RANK = SimpleNamespace(**{**vars(LORA), "r": 16})
# Parameters as ``Llama`` names them, and the adapters ``LoraModel`` adds.
BASE_PARAMETERS = ["Llama/blocks/0/attention/qkv_linear/weight", "Llama/tok_emb/weight"]
LORA_PARAMETERS = ["Llama/blocks/0/attention/qkv_linear/lora_A", "Llama/blocks/0/attention/qkv_linear/lora_B"]


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


@pytest.mark.parametrize("pattern", ["rank_pattern", "alpha_pattern"])
def test_a_peft_adapter_with_a_per_projection_rank_or_alpha_is_not_loaded(tmp_path, expect_error, pattern):
    source = _local_model(tmp_path / "acme-7b")

    class AcmeProvider(LlamaProvider):
        def model_source(self) -> str:
            return source

    register_provider("acme/acme-7b", AcmeProvider)
    config = _config("acme/acme-7b")
    config.model.lora = SimpleNamespace(r=8, alpha=16, target_modules=["q_proj", "v_proj"])
    provider = get_provider(config)
    adapter = AdapterState(
        tensors={},
        config={
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            pattern: {"q_proj": 32},
        },
    )

    with expect_error(UnsupportedTrainingConfig, rf"^checkpoint: the adapter sets a per-projection .*{pattern}"):
        provider._import_adapter(object(), adapter)


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


def _acme_provider(tmp_path, format_type, lora=None, name="acme/acme-7b"):
    """A ``LlamaProvider`` of the model at ``tmp_path/acme-7b``, saving checkpoints in ``format_type``."""
    source = _local_model(tmp_path / "acme-7b")

    class AcmeProvider(LlamaProvider):
        def model_source(self) -> str:
            return source

    config = _config(name)
    config.model.lora = lora
    config.checkpoint = SimpleNamespace(format=SimpleNamespace(type=format_type))
    return AcmeProvider(config)


def _model(lora):
    """A model with the parameter names of a Llama, with LoRA adapters if ``lora``; the values are placeholders."""
    names = BASE_PARAMETERS + (LORA_PARAMETERS if lora else [])
    return SimpleNamespace(parameters=lambda: dict.fromkeys(names))


def _checkpoint_dir(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir(exist_ok=True)
    return directory


class _Checkpointing:
    """``ttml.checkpointing`` in memory: a save writes an empty file and keeps the header and the names of the
    parameters; a load records the file and the names of the parameters it fills."""

    def __init__(self):
        self.saved, self.loaded = {}, []

    def save_checkpoint(self, path, header, model_params):
        Path(path).write_text("")
        self.saved[path] = (header, sorted(model_params))

    def read_header(self, path):
        return self.saved[path][0]

    def load_checkpoint(self, path, model_params):
        self.loaded.append((path, sorted(model_params)))


@pytest.fixture
def tt_checkpointing(monkeypatch):
    fake = _Checkpointing()
    for name in ("save_checkpoint", "read_header", "load_checkpoint"):
        monkeypatch.setattr(checkpointing, name, getattr(fake, name))
    # NamedParameters holds only tensors, and the model's parameters are placeholders.
    monkeypatch.setattr(ttml, "NamedParameters", dict)
    return fake


@pytest.mark.parametrize("lora", [LORA, None], ids=["lora", "full"])
def test_a_tt_checkpoint_holds_the_adapters_of_a_lora_run_or_every_parameter_and_loads_back(
    tmp_path, tt_checkpointing, lora
):
    provider = _acme_provider(tmp_path, "tt", lora)
    checkpoint = _checkpoint_dir(tmp_path)
    file = str(checkpoint / TT_CHECKPOINT)
    parameters = LORA_PARAMETERS if lora else BASE_PARAMETERS

    provider.export_weights(_model(lora), checkpoint)
    header = {
        "base_model": {"name_or_path": "acme/acme-7b"},
        "model_type": "llama",
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]} if lora else None,
    }
    assert tt_checkpointing.saved == {file: (header, parameters)}

    provider.load_weights(_model(lora), checkpoint)
    assert tt_checkpointing.loaded == [(file, parameters)]


@pytest.mark.parametrize(
    "saved, lora, name, error",
    [
        (LORA, OTHER_RANK, "acme/acme-7b", r"doesn't match this model \(lora: checkpoint \{'r': 8, .* run \{'r': 16"),
        (LORA, None, "acme/acme-7b", r"doesn't match this model \(lora: checkpoint \{'r': 8, .* run None\)"),
        (None, LORA, "acme/acme-7b", r"doesn't match this model \(lora: checkpoint None, run \{'r': 8"),
        (LORA, LORA, "acme/acme-13b", r"is for 'acme/acme-7b', this model is 'acme/acme-13b'"),
    ],
    ids=["other-rank", "lora-onto-full", "full-onto-lora", "other-model"],
)
def test_a_tt_checkpoint_loads_only_onto_a_run_it_matches(
    tmp_path, tt_checkpointing, expect_error, saved, lora, name, error
):
    checkpoint = _checkpoint_dir(tmp_path)
    _acme_provider(tmp_path, "tt", saved).export_weights(_model(saved), checkpoint)
    provider = _acme_provider(tmp_path, "tt", lora, name)

    with expect_error(UnsupportedTrainingConfig, rf"^checkpoint: .* {error}"):
        provider.load_weights(_model(lora), checkpoint)
    assert tt_checkpointing.loaded == []


def test_a_directory_without_a_tt_checkpoint_is_not_loaded(tmp_path, tt_checkpointing, expect_error):
    checkpoint = _checkpoint_dir(tmp_path)
    with expect_error(FileNotFoundError, f"is not a tt checkpoint: no {TT_CHECKPOINT}"):
        _acme_provider(tmp_path, "tt").load_weights(_model(None), checkpoint)
    assert tt_checkpointing.loaded == []


def test_an_hf_checkpoint_of_a_lora_run_is_a_peft_adapter_that_loads_back(tmp_path, monkeypatch, expect_error):
    # A model with no LoRA layers, so the adapter has no tensors: test_peft converts them, this test the files.
    monkeypatch.setattr(LlamaProvider, "_adapted_layers", lambda self, model: iter(()))
    provider = _acme_provider(tmp_path, "hf", LORA)
    checkpoint = _checkpoint_dir(tmp_path)

    provider.export_weights(object(), checkpoint)
    assert sorted(path.name for path in checkpoint.iterdir()) == sorted([ADAPTER_CONFIG, ADAPTER_WEIGHTS])
    config = read_adapter(checkpoint).config
    assert config["base_model_name_or_path"] == provider.model_source()
    assert (config["r"], config["lora_alpha"], config["target_modules"]) == (8, 16, ["q_proj", "v_proj"])

    provider.load_weights(object(), checkpoint)
    with expect_error(UnsupportedTrainingConfig, r"^checkpoint: the adapter doesn't match this run's LoRA \(r: "):
        _acme_provider(tmp_path, "hf", OTHER_RANK).load_weights(object(), checkpoint)
