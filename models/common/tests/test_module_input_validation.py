# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.common.models import module_input_validation as validation


class FakeTensor:
    def __init__(self, memory_config="dram"):
        self._memory_config = memory_config
        self.memory_config_calls = 0

    def is_allocated(self):
        return True

    def memory_config(self):
        self.memory_config_calls += 1
        return self._memory_config


class FakeConfig:
    prefill_input_memcfg = "dram"


class FakeModule:
    config = FakeConfig()

    def prefill_forward(self, x):
        return x


class FakeModel:
    def __init__(self):
        self.module = FakeModule()


def iter_named_modules(model):
    yield "module", model.module


def test_validate_module_input_configs_reads_memory_config_when_active(monkeypatch):
    monkeypatch.setattr(validation, "ttnn", SimpleNamespace(Tensor=FakeTensor))
    model = FakeModel()
    tensor = FakeTensor()

    with validation.validate_module_input_configs(
        model=model,
        iter_named_modules=iter_named_modules,
        mode="prefill",
    ):
        assert model.module.prefill_forward(tensor) is tensor

    assert tensor.memory_config_calls == 1


def test_suspend_module_input_validation_bypasses_memory_config(monkeypatch):
    monkeypatch.setattr(validation, "ttnn", SimpleNamespace(Tensor=FakeTensor))
    model = FakeModel()
    tensor = FakeTensor()

    with validation.validate_module_input_configs(
        model=model,
        iter_named_modules=iter_named_modules,
        mode="prefill",
    ):
        with validation.suspend_module_input_validation():
            assert model.module.prefill_forward(tensor) is tensor

    assert tensor.memory_config_calls == 0


def test_suspend_module_input_validation_restores_after_nested_contexts(monkeypatch):
    monkeypatch.setattr(validation, "ttnn", SimpleNamespace(Tensor=FakeTensor))
    model = FakeModel()
    tensor = FakeTensor()

    with validation.validate_module_input_configs(
        model=model,
        iter_named_modules=iter_named_modules,
        mode="prefill",
    ):
        with validation.suspend_module_input_validation():
            with validation.suspend_module_input_validation():
                model.module.prefill_forward(tensor)
            model.module.prefill_forward(tensor)
        model.module.prefill_forward(tensor)

    model.module.prefill_forward(tensor)

    assert tensor.memory_config_calls == 1
