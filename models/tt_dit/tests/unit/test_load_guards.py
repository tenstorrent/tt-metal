# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A loaded module evicts its coresident exclusions and is not loaded again. Needs no device."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest
from loguru import logger

import ttnn

from ...layers.module import IncompatibleKeys, Module
from ...models.vae.vae_mochi import MochiVAEDecoderAdapter
from ...models.vae.vae_sd35 import VAEDecoderAdapter
from ...utils import cache

_NoParallelism = namedtuple("_NoParallelism", [])


class _Weight:
    """Stands in for a `Parameter`: records what loaded it and how often it was freed."""

    def __init__(self) -> None:
        self.loads: list[str] = []
        self.deallocations = 0

    def load_torch_tensor(self, value: object) -> None:
        self.loads.append("torch")

    def load(self, path: Path) -> None:
        self.loads.append("cache")

    def deallocate(self) -> None:
        self.deallocations += 1


class _Model(Module):
    def __init__(self, *children: _Model) -> None:
        super().__init__()
        self.weight = _Weight()
        # A stand-in is not registered by attribute assignment.
        self._parameters["weight"] = self.weight
        for index, child in enumerate(children):
            self.add_module(f"child{index}", child)

    def forward(self) -> None:
        pass


def _state(model: _Model, prefix: str = "") -> dict[str, object]:
    state: dict[str, object] = {f"{prefix}weight": object()}
    for name, child in model.named_children():
        state.update(_state(child, f"{prefix}{name}."))
    return state


def _loaded(*children: _Model) -> _Model:
    model = _Model(*children)
    model.load_torch_state_dict(_state(model))
    model.weight.loads.clear()
    for child in children:
        child.weight.loads.clear()
    return model


def _with_exclusion(model: _Model) -> _Model:
    """A loaded module that `model` cannot be resident beside."""
    excluded = _loaded()
    model.register_coresident_exclusions(excluded)
    return excluded


def _logged(action, level: str) -> list[str]:  # noqa: ANN001
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level=level)
    try:
        action()
    finally:
        logger.remove(sink)
    return messages


@pytest.fixture
def load_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`cache.load_model` for a model on one host, with or without a complete cache on disk."""
    monkeypatch.setattr(ttnn, "distributed_context_barrier", lambda: None)
    monkeypatch.setattr(cache, "_distributed_world_size", lambda: 1)
    reads: list[str] = []

    def state_of(model: _Model):
        def read() -> dict[str, object]:
            reads.append("read")
            return _state(model)

        return read

    def load(model: _Model, *, cached: bool) -> list[str]:
        if cached:
            monkeypatch.setenv("TT_DIT_CACHE_DIR", str(tmp_path))
        else:
            monkeypatch.delenv("TT_DIT_CACHE_DIR", raising=False)
        arguments = {
            "model_name": "model",
            "subfolder": "transformer",
            "parallel_config": _NoParallelism(),
            "mesh_shape": (1, 1),
            "mesh_device": None,
        }
        if cached:
            directory = cache.model_cache_dir(**arguments)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / cache.CACHE_DICT_FILE).touch()
        cache.load_model(model, get_torch_state_dict=state_of(model), create_cache=False, **arguments)
        return reads

    return load


@pytest.mark.parametrize("cached", [False, True], ids=["no_cache", "cache"])
def test_load_model_on_a_resident_model_evicts_and_does_not_load(load_model, cached: bool) -> None:
    model = _loaded()
    excluded = _with_exclusion(model)
    reads: list[str] = []
    messages = _logged(lambda: reads.extend(load_model(model, cached=cached)), "WARNING")
    assert not excluded.is_loaded()
    assert excluded.weight.deallocations == 1
    assert reads == []
    assert model.weight.loads == []
    assert model.is_loaded()
    assert messages == []


@pytest.mark.parametrize(("cached", "source"), [(False, "torch"), (True, "cache")], ids=["no_cache", "cache"])
def test_load_model_on_an_unloaded_model_evicts_and_loads(load_model, cached: bool, source: str) -> None:
    model = _Model()
    excluded = _with_exclusion(model)
    reads: list[str] = []
    messages = _logged(lambda: reads.extend(load_model(model, cached=cached)), "WARNING")
    assert not excluded.is_loaded()
    assert model.weight.loads == [source]
    assert reads == ([] if cached else ["read"])
    assert model.is_loaded()
    assert messages == []


def test_load_torch_state_dict_on_a_loaded_module_evicts_skips_and_warns() -> None:
    model = _loaded()
    excluded = _with_exclusion(model)
    result: list[IncompatibleKeys] = []
    # A state the module would refuse: it is not read at all.
    messages = _logged(lambda: result.append(model.load_torch_state_dict({"unexpected": object()})), "WARNING")
    assert result == [IncompatibleKeys([], [])]
    assert not excluded.is_loaded()
    assert model.weight.loads == []
    assert len(messages) == 1
    assert "load_torch_state_dict skipped, the module is already loaded" in messages[0]
    assert "deallocate_weights()" in messages[0]


def test_a_first_load_torch_state_dict_loads_without_a_warning() -> None:
    model = _Model()
    excluded = _with_exclusion(model)
    messages = _logged(lambda: model.load_torch_state_dict(_state(model)), "WARNING")
    assert not excluded.is_loaded()
    assert model.weight.loads == ["torch"]
    assert messages == []


def test_module_load_on_a_loaded_module_evicts_and_skips_quietly(tmp_path: Path) -> None:
    child = _Model()
    model = _loaded(child)
    excluded = _with_exclusion(model)
    messages = _logged(lambda: model.load(tmp_path), "DEBUG")
    assert not excluded.is_loaded()
    assert model.weight.loads == []
    assert child.weight.loads == []
    assert messages == ["_Model: cache load skipped, the module is already loaded"]
    assert _logged(lambda: model.load(tmp_path), "WARNING") == []


def test_a_first_module_load_loads_and_announces_it(tmp_path: Path) -> None:
    child = _Model()
    model = _Model(child)
    messages = _logged(lambda: model.load(tmp_path), "INFO")
    assert model.weight.loads == ["cache"]
    assert child.weight.loads == ["cache"]
    assert model.is_loaded()
    assert child.is_loaded()
    assert any("loading 2 cached weight tensors" in message for message in messages)
    assert _logged(lambda: _Model().load(tmp_path), "WARNING") == []


def test_module_load_loads_only_the_children_that_are_not_loaded(tmp_path: Path) -> None:
    resident, evicted = _loaded(), _Model()
    model = _Model(resident, evicted)
    messages = _logged(lambda: model.load(tmp_path), "DEBUG")
    assert resident.weight.loads == []
    assert evicted.weight.loads == ["cache"]
    assert model.weight.loads == ["cache"]
    assert model.is_loaded()
    # The skip of a child is not reported: only the top-level call is.
    assert not any("cache load skipped" in message for message in messages)


def test_a_deallocated_module_loads_again(load_model, tmp_path: Path) -> None:
    model = _loaded()

    model.deallocate_weights()
    model.load_torch_state_dict(_state(model))
    assert model.weight.loads == ["torch"]

    model.deallocate_weights()
    model.load(tmp_path)
    assert model.weight.loads == ["torch", "cache"]

    model.deallocate_weights()
    load_model(model, cached=False)
    assert model.weight.loads == ["torch", "cache", "torch"]


class _Decoder:
    def __init__(self) -> None:
        self.states: list[object] = []

    def load_torch_state_dict(self, state: object) -> None:
        self.states.append(state)


@pytest.mark.parametrize("adapter_class", [MochiVAEDecoderAdapter, VAEDecoderAdapter], ids=["mochi", "sd35"])
def test_reload_weights_without_a_device_decoder_does_nothing(adapter_class: type) -> None:
    adapter = object.__new__(adapter_class)
    adapter._decoder = None
    adapter.reload_weights()


@pytest.mark.parametrize("adapter_class", [MochiVAEDecoderAdapter, VAEDecoderAdapter], ids=["mochi", "sd35"])
def test_reload_weights_leaves_the_skip_to_the_decoder(adapter_class: type) -> None:
    adapter = object.__new__(adapter_class)
    adapter._decoder, adapter._decoder_state_dict = _Decoder(), "state"
    adapter.reload_weights()
    assert adapter._decoder.states == ["state"]
