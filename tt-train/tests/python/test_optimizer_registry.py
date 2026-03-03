# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optimizer registry with Python-defined optimizers."""

import pytest
import ttnn
import ttml


def _make_python_optimizer_class():
    """Create a minimal Python optimizer subclass."""

    class PyOptimizer(ttml.optimizers.OptimizerBase):
        def __init__(self, params):
            super().__init__(params)
            self._lr = 0.042
            self._steps = 0

        def get_name(self):
            return "PyOptimizer"

        def step(self):
            self._steps += 1

        def zero_grad(self):
            pass

        def get_state_dict(self):
            return {}

        def set_state_dict(self, d):
            pass

        def get_steps(self):
            return self._steps

        def set_steps(self, s):
            self._steps = s

        def get_lr(self):
            return self._lr

        def set_lr(self, lr):
            self._lr = lr

    return PyOptimizer


def test_register_and_create_python_optimizer():
    """Test registering a Python optimizer and creating it via create_optimizer."""
    PyOptimizer = _make_python_optimizer_class()

    ttml.optimizers.register_optimizer(
        "PyOpt", lambda config, params: PyOptimizer(params)
    )

    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "PyOpt"}, params)

    assert opt.get_name() == "PyOptimizer"
    assert opt.get_lr() == pytest.approx(0.042)


def test_python_optimizer_methods():
    """Test that virtual method dispatch works for Python optimizers."""
    PyOptimizer = _make_python_optimizer_class()

    ttml.optimizers.register_optimizer(
        "PyOptMethods", lambda config, params: PyOptimizer(params)
    )

    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "PyOptMethods"}, params)

    assert opt.get_steps() == 0
    opt.step()
    opt.step()
    assert opt.get_steps() == 2

    opt.set_lr(0.1)
    assert opt.get_lr() == pytest.approx(0.1)


def test_python_optimizer_receives_config():
    """Test that the Python creator receives the config dict."""
    received = {}

    def creator(config, params):
        received.update(config)
        PyOpt = _make_python_optimizer_class()
        opt = PyOpt(params)
        opt.set_lr(config.get("lr", 0.001))
        return opt

    ttml.optimizers.register_optimizer("PyOptConfig", creator)

    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "PyOptConfig", "lr": 0.05}, params)

    assert received["type"] == "PyOptConfig"
    assert received["lr"] == pytest.approx(0.05)
    assert opt.get_lr() == pytest.approx(0.05)


def test_cpp_optimizer_still_works():
    """Test that C++ optimizers are unaffected by Python registrations."""
    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": 0.001}, params)
    assert opt.get_name() == "AdamW"
    assert opt.get_lr() == pytest.approx(0.001, abs=1e-6)
