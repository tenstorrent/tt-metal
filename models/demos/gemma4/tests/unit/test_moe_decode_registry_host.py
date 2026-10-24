# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The decode MoE module's registry rule (``Gemma4MoEDecode``): one shared module per mesh, built once every layer of
the model has registered (``num_layers`` from the config's ``num_hidden_layers``). A unit that builds one layer must
register with ``num_layers=1`` (a one-layer config), or the module is never built and the first MoE call asserts
(window 5, 2026-10-24: "decode MoE module not built: not every layer registered"). No device: the build is recorded.

    pytest models/demos/gemma4/tests/unit/test_moe_decode_registry_host.py -p no:cacheprovider
"""

from types import SimpleNamespace

import pytest

from models.demos.gemma4.tt.experts import moe_decode as md

CONFIG = SimpleNamespace(hidden_size=2560, top_k=2, num_experts=8, moe_intermediate_size=2560)


@pytest.fixture
def mesh(monkeypatch):
    """A stand-in mesh (the registry keys on its id) whose module build is recorded, not run."""
    device = SimpleNamespace(shape=(8, 4))
    builds = []

    def _build(self, config):
        builds.append(self.layer_idx)
        self._reg["module"] = f"module for {self.num_layers} layers"

    monkeypatch.setattr(md.Gemma4MoEDecode, "_build", _build)
    yield device, builds
    md._SHARED.pop(id(device), None)


def _register(device, layer_idx, num_layers):
    return md.Gemma4MoEDecode(
        device, CONFIG, object(), object(), layer_idx=layer_idx, num_layers=num_layers, frame_rows=256
    )


def test_a_one_layer_unit_registers_with_num_layers_one(mesh):
    device, builds = mesh
    layer = _register(device, 0, 1)
    assert builds == [0] and layer.module == "module for 1 layers"


def test_one_registered_layer_of_a_thirty_layer_config_never_builds(mesh, expect_error):
    device, builds = mesh
    layer = _register(device, 0, 30)
    assert builds == []
    with expect_error(AssertionError, "not every layer registered"):
        layer.module


def test_the_module_is_built_once_when_the_last_layer_registers(mesh):
    device, builds = mesh
    first = _register(device, 0, 3)
    second = _register(device, 1, 3)
    assert builds == []
    third = _register(device, 2, 3)
    assert builds == [2]
    assert first.module is second.module is third.module
    _register(device, 2, 3)  # a re-registration does not rebuild
    assert builds == [2]
