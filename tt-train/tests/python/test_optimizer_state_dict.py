# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for optimizer ``get_state_dict`` / ``set_state_dict`` round-trip from Python.

These tests pin behaviour around a subtle nanobind variant-routing issue
discovered while implementing checkpoint resume for a TinyLlama pretraining
example.

# Background: the variant-routing issue

``serialization::ValueType`` is a ``std::variant`` with both ``int`` and
``size_t`` as separate alternatives:

    using ValueType = std::variant<bool, char, int, float, double, uint32_t,
                                   size_t, bfloat16, std::string, ...>;

Optimizers store their step counter as ``size_t m_steps``, write
``dict["steps"] = m_steps`` (lands in the ``size_t`` variant slot), and read
it back via ``serialization::get_value_type<size_t>(dict, "steps")``.

When the state dict crosses into Python and back, nanobind's default
variant caster picks the *first* matching alternative when converting a
Python ``int`` -- which is ``int``, not ``size_t``. ``std::get<size_t>``
then throws ``"std::get: wrong index for variant"`` and ``set_state_dict``
fails.

The fix lives in ``tt-train/sources/ttml/nanobind/nb_optimizers.cpp``,
where ``set_state_dict`` is bound with a wrapper that explicitly routes
the ``"steps"`` value into the ``size_t`` alternative before delegating to
the C++ implementation. Because ``set_state_dict`` is bound on
``OptimizerBase`` and dispatched virtually, a single wrapper covers every
optimizer subclass.

# What these tests cover

1. ``test_adamw_get_state_dict_keys`` -- sanity-checks the state dict has
   the documented shape (``steps``, ``exp_avg``, ``exp_avg_sq``,
   ``amsgrad``).
2. ``test_adamw_set_state_dict_inprocess_roundtrip`` -- minimal repro that
   ``set_state_dict(get_state_dict())`` works in-process with no pickling.
   Pre-fix this throws; post-fix it must succeed.
3. ``test_set_state_dict_after_steps_inprocess_roundtrip`` (parametrized)
   -- runs the same round-trip with a non-zero ``m_steps`` for every
   Python-bound optimizer (AdamW, AdamWFullPrecision, SGD, MuonComposite,
   NoOp). A regression in the binding wrapper would surface as a uniform
   failure across this matrix.
"""

import numpy as np
import pytest
import ttml
import ttnn


def _make_simple_params() -> "ttml.NamedParameters":
    """Build a small NamedParameters with one trainable bf16 weight tensor."""
    from ttml.modules import AbstractModuleBase, Parameter

    class SingleParamModule(AbstractModuleBase):
        def __init__(self) -> None:
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np, new_type=ttnn.DataType.BFLOAT16))

        def forward(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SingleParamModule()
    return model.parameters()


def _make_adamw(params: "ttml.NamedParameters", *, lr: float = 1e-3) -> "ttml.optimizers.AdamW":
    cfg = ttml.optimizers.AdamWConfig.make(lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    return ttml.optimizers.AdamW(params, cfg)


def _make_optimizer(name: str, params: "ttml.NamedParameters") -> "ttml.optimizers.OptimizerBase":
    """Construct a Python-bound optimizer by short name.

    Covers every optimizer subclass that ``nb_optimizers.cpp`` currently
    exposes with a Python binding *and* whose constructor + ``step()`` can
    run in a single-process unit test.

    Deliberately excluded:

    * ``RemoteOptimizer`` -- bound to Python (``nb_optimizers.cpp:83``) but
      its constructor calls
      ``autograd::ctx().get_distributed_context()->create_sub_context(...)``
      (see ``remote_optimizer.cpp:27``), which requires a live distributed
      context. The fix in ``OptimizerBase::set_state_dict``'s wrapper still
      applies to ``RemoteOptimizer`` via virtual dispatch -- there's just no
      way to construct one here to verify it.
    * ``SGDComposite``, ``AdamWComposite``, ``MorehAdamW`` -- defined in C++
      under ``tt-train/sources/ttml/optimizers/`` but **not** bound to
      Python. No Python entry point exists, so the variant-routing bug
      cannot be reached from Python for these. If/when they get a binding
      added to ``nb_optimizers.cpp:py_module_types``, they will inherit the
      fixed ``set_state_dict`` wrapper from ``OptimizerBase`` automatically
      and should be added to ``_PY_BOUND_OPTIMIZERS`` below.
    """
    if name == "AdamW":
        cfg = ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
        return ttml.optimizers.AdamW(params, cfg)
    if name == "AdamWFullPrecision":
        cfg = ttml.optimizers.AdamWFullPrecisionConfig.make(
            lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0
        )
        return ttml.optimizers.AdamWFullPrecision(params, cfg)
    if name == "SGD":
        cfg = ttml.optimizers.SGDConfig.make(lr=1e-3, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False)
        return ttml.optimizers.SGD(params, cfg)
    if name == "MuonComposite":
        cfg = ttml.optimizers.MuonConfig.make(lr=1e-3, momentum=0.95, ns_steps=5)
        return ttml.optimizers.MuonComposite(params, cfg)
    if name == "NoOp":
        return ttml.optimizers.NoOp(params)
    raise ValueError(f"Unknown optimizer name: {name!r}")


# Every optimizer bound to Python in nb_optimizers.cpp that we can construct
# in-process. Keep this list in sync with the ``nb::class_<..., OptimizerBase>``
# declarations in nb_optimizers.cpp:py_module_types. See ``_make_optimizer``
# above for what is intentionally excluded and why.
_PY_BOUND_OPTIMIZERS = ["AdamW", "AdamWFullPrecision", "SGD", "MuonComposite", "NoOp"]

# Optimizers that store and restore lr in their state dict. NoOp is excluded
# because its get_lr() returns a hardcoded 0.0 and set_lr() is a no-op.
_LR_OPTIMIZERS = ["AdamW", "AdamWFullPrecision", "SGD", "MuonComposite"]


@pytest.mark.requires_device
def test_adamw_get_state_dict_keys():
    """``AdamW.get_state_dict()`` returns the documented keys."""
    params = _make_simple_params()
    opt = _make_adamw(params)

    state = opt.get_state_dict()

    assert "steps" in state
    assert "lr" in state
    assert "beta1" in state
    assert "beta2" in state
    assert "epsilon" in state
    assert "weight_decay" in state
    assert "amsgrad" in state
    assert "stochastic_rounding" in state
    assert "exp_avg" in state
    assert "exp_avg_sq" in state

    # ``steps`` should be a Python int. When non-AmsGrad, ``max_exp_avg_sq``
    # must not be present (per ``AdamW::get_state_dict`` in C++).
    assert isinstance(state["steps"], int)
    assert state["steps"] == 0
    assert state["amsgrad"] is False
    assert "max_exp_avg_sq" not in state


@pytest.mark.requires_device
def test_adamw_set_state_dict_inprocess_roundtrip():
    """In-process ``set_state_dict(get_state_dict())`` round-trips cleanly.

    Pre-fix this would throw ``std::get: wrong index for variant`` because
    nanobind routes Python ``int`` to the ``int`` variant alternative while
    ``AdamW::set_state_dict`` does ``std::get<size_t>``. The fix in
    ``nb_optimizers.cpp::set_state_dict`` reroutes the ``"steps"`` value
    into the ``size_t`` alternative explicitly. This test pins the
    post-fix behaviour.
    """
    params = _make_simple_params()
    opt = _make_adamw(params)

    # Take the dict straight out of the same optimizer and feed it back. No
    # pickling, no Python-side mutation — this is the simplest possible
    # round-trip and it should be a no-op semantically.
    state = opt.get_state_dict()

    opt.set_state_dict(state)

    assert opt.get_state_dict()["steps"] == 0


@pytest.mark.requires_device
@pytest.mark.parametrize("optimizer_name", _PY_BOUND_OPTIMIZERS)
def test_set_state_dict_after_steps_inprocess_roundtrip(optimizer_name):
    """``set_state_dict(get_state_dict())`` must round-trip with non-zero steps.

    Replicated across every Python-bound optimizer (AdamW, AdamWFullPrecision,
    SGD, MuonComposite, NoOp) because all of them store their step counter as
    a ``size_t m_steps`` field and read it back via
    ``serialization::get_value_type<size_t>(dict, "steps")``. The variant
    routing fix lives in ``nb_optimizers.cpp::set_state_dict``, which is
    bound on ``OptimizerBase`` and inherited by every subclass via virtual
    dispatch -- so a regression in that single wrapper would surface as a
    uniform failure across the whole parametrize matrix.

    Each optimizer's ``step()`` short-circuits its per-parameter inner loop
    when no gradient has been backwarded but still bumps ``m_steps``
    (sometimes at the start of ``step()``, sometimes at the end), so we can
    advance the step counter without running a real forward/backward. See
    the implementations in ``tt-train/sources/ttml/optimizers/`` for the
    exact placement of ``m_steps++`` per optimizer.

    ``RemoteOptimizer`` is intentionally excluded: its ``step()`` performs
    distributed I/O and can't run in a unit-test process.
    """
    params = _make_simple_params()
    opt = _make_optimizer(optimizer_name, params)

    for _ in range(7):
        opt.step()

    state = opt.get_state_dict()
    assert state["steps"] == 7, f"{optimizer_name}.get_state_dict()['steps'] expected 7, got {state['steps']}"

    # The actual round-trip. Pre-fix this throws ``std::get: wrong index for
    # variant`` because nanobind routes Python int -> int alternative while
    # the C++ side does ``std::get<size_t>``. With the binding wrapper in
    # ``nb_optimizers.cpp::set_state_dict`` in place, this should succeed.
    opt.set_state_dict(state)

    # Steps survives the round-trip.
    assert opt.get_state_dict()["steps"] == 7


@pytest.mark.requires_device
@pytest.mark.parametrize("optimizer_name", _LR_OPTIMIZERS)
def test_lr_present_in_state_dict(optimizer_name):
    """``get_state_dict()`` includes ``lr`` as a float for all lr-bearing optimizers."""
    params = _make_simple_params()
    opt = _make_optimizer(optimizer_name, params)

    state = opt.get_state_dict()

    assert "lr" in state, f"{optimizer_name}.get_state_dict() missing 'lr' key"
    assert isinstance(state["lr"], float), f"{optimizer_name}: expected float lr, got {type(state['lr'])}"
    assert state["lr"] == pytest.approx(1e-3)


@pytest.mark.requires_device
@pytest.mark.parametrize("optimizer_name", _LR_OPTIMIZERS)
def test_lr_restored_after_set_state_dict(optimizer_name):
    """``set_state_dict`` correctly restores a mutated lr value.

    Constructs an optimizer with lr=1e-3, then round-trips a state dict with
    lr overridden to 5e-4. Verifies that ``get_lr()`` reflects the restored
    value, exercising the ``float`` variant routing through the schema.
    """
    params = _make_simple_params()
    opt = _make_optimizer(optimizer_name, params)

    state = dict(opt.get_state_dict())
    state["lr"] = 5e-4

    opt.set_state_dict(state)

    assert opt.get_lr() == pytest.approx(
        5e-4
    ), f"{optimizer_name}: expected lr=5e-4 after set_state_dict, got {opt.get_lr()}"


# Per-optimizer hyperparameter keys (excluding lr which is tested separately)
# and the mutated values to round-trip through set_state_dict.
_HYPERPARAMS_BY_OPTIMIZER = {
    "AdamW": {"beta1": 0.8, "beta2": 0.99, "epsilon": 1e-6, "weight_decay": 1e-2, "stochastic_rounding": False},
    "AdamWFullPrecision": {"beta1": 0.8, "beta2": 0.99, "epsilon": 1e-6, "weight_decay": 1e-2},
    "SGD": {"momentum_factor": 0.9, "dampening": 0.1, "weight_decay": 1e-4, "nesterov": False},
    "MuonComposite": {"momentum": 0.85, "ns_steps": 3},
}


@pytest.mark.requires_device
@pytest.mark.parametrize("optimizer_name", list(_HYPERPARAMS_BY_OPTIMIZER))
def test_hyperparams_restored_after_set_state_dict(optimizer_name):
    """All config hyperparameters survive a get/set_state_dict round-trip.

    Mutates each hyperparameter key in the state dict and verifies that
    set_state_dict restores the mutated value, exercising the ValueType
    schema routing for float, bool, and int fields.
    """
    params = _make_simple_params()
    opt = _make_optimizer(optimizer_name, params)
    overrides = _HYPERPARAMS_BY_OPTIMIZER[optimizer_name]

    state = dict(opt.get_state_dict())
    state.update(overrides)
    opt.set_state_dict(state)

    restored = opt.get_state_dict()
    for key, expected in overrides.items():
        actual = restored[key]
        if isinstance(expected, float):
            assert actual == pytest.approx(expected), f"{optimizer_name}['{key}']: expected {expected}, got {actual}"
        else:
            assert actual == expected, f"{optimizer_name}['{key}']: expected {expected}, got {actual}"
