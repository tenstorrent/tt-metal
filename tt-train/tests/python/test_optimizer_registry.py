# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optimizer registry with Python-defined optimizers."""

import numpy as np
import pytest
import ttml
import ttnn
from ttml.modules import AbstractModuleBase, Parameter


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

        def get_group_lrs(self):
            return [self._lr]

        def set_group_lrs(self, lrs):
            (self._lr,) = lrs

    return PyOptimizer


def test_register_and_create_python_optimizer():
    """Test registering a Python optimizer and creating it via create_optimizer."""
    PyOptimizer = _make_python_optimizer_class()

    ttml.optimizers.register_optimizer("PyOpt", lambda config, params: PyOptimizer(params))

    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "PyOpt"}, params)

    assert opt.get_name() == "PyOptimizer"
    assert opt.get_group_lrs() == pytest.approx([0.042])


def test_python_optimizer_methods():
    """Test that virtual method dispatch works for Python optimizers."""
    PyOptimizer = _make_python_optimizer_class()

    ttml.optimizers.register_optimizer("PyOptMethods", lambda config, params: PyOptimizer(params))

    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "PyOptMethods"}, params)

    assert opt.get_steps() == 0
    opt.step()
    opt.step()
    assert opt.get_steps() == 2

    opt.set_group_lrs([0.1])
    assert opt.get_group_lrs() == pytest.approx([0.1])
    # get_initial_group_lrs is inherited: it records the LRs through the Python get_group_lrs.
    assert opt.get_initial_group_lrs() == pytest.approx([0.1])


def test_python_optimizer_receives_config():
    """Test that the Python creator receives the config dict."""
    received = {}

    def creator(config, params):
        received.update(config)
        PyOpt = _make_python_optimizer_class()
        opt = PyOpt(params)
        opt.set_group_lrs([config.get("lr", 0.001)])
        return opt

    ttml.optimizers.register_optimizer("PyOptConfig", creator)

    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "PyOptConfig", "lr": 0.05}, params)

    assert received["type"] == "PyOptConfig"
    assert received["lr"] == pytest.approx(0.05)
    assert opt.get_group_lrs() == pytest.approx([0.05])


def test_cpp_optimizer_still_works():
    """Test that C++ optimizers are unaffected by Python registrations."""
    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": 0.001}, params)
    assert opt.get_name() == "AdamW"
    assert opt.get_group_lrs() == pytest.approx([0.001], abs=1e-6)


NO_DECAY_GROUPS = [{"modules": ["gamma", "bias"], "weight_decay": 0.0}]


def test_param_groups_reject_unsupported_keys(expect_error):
    params = ttml.NamedParameters()
    with expect_error(ValueError, "Unknown param group option 'weight_dacay'"):
        ttml.optimizers.create_optimizer(
            {"type": "AdamW", "param_groups": [{"modules": ["gamma"], "weight_dacay": 0.0}]}, params
        )


def test_param_groups_reject_non_numeric_options(expect_error):
    params = ttml.NamedParameters()
    with expect_error(RuntimeError, "Param group option 'amsgrad' must be a number"):
        ttml.optimizers.create_optimizer(
            {"type": "AdamW", "param_groups": [{"modules": ["gamma"], "amsgrad": "yes"}]}, params
        )


def test_param_groups_require_modules(expect_error):
    params = ttml.NamedParameters()
    with expect_error(RuntimeError, 'Every param group needs a "modules" list'):
        ttml.optimizers.create_optimizer({"type": "AdamW", "param_groups": [{"weight_decay": 0.0}]}, params)


@pytest.mark.parametrize("optimizer_type", ["AdamW", "AdamWFullPrecision", "SGD", "MuonComposite"])
def test_param_groups_accepted_by_ported_optimizers(optimizer_type):
    params = ttml.NamedParameters()
    # "lr" is a per-group option of every ported optimizer (Muon has no weight_decay).
    groups = [{"modules": ["gamma", "bias"], "lr": 0.002}]
    opt = ttml.optimizers.create_optimizer({"type": optimizer_type, "param_groups": groups}, params)
    assert opt.get_name() == optimizer_type


def test_param_groups_ignored_by_no_op():
    """NoOp updates nothing, so a config with param_groups is accepted and the groups are ignored."""
    params = ttml.NamedParameters()
    opt = ttml.optimizers.create_optimizer({"type": "NoOp", "param_groups": NO_DECAY_GROUPS}, params)
    assert opt.get_name() == "NoOp"
    # Like every C++ optimizer it has one group (holding all params), with only an (unused) lr.
    assert opt.get_param_groups() == [{"params": [], "lr": 0.0}]


def test_python_optimizer_must_implement_group_lrs(expect_error):
    """A Python optimizer has no C++ param groups, so the plural LR API is its to implement."""

    class NoLrOptimizer(ttml.optimizers.OptimizerBase):
        def get_name(self):
            return "NoLrOptimizer"

    opt = NoLrOptimizer(ttml.NamedParameters())
    with expect_error(RuntimeError, "get_group_lrs"):
        opt.get_group_lrs()


def test_param_groups_rejected_for_python_optimizer(expect_error):
    PyOptimizer = _make_python_optimizer_class()
    ttml.optimizers.register_optimizer("PyOptNoDecay", lambda config, params: PyOptimizer(params))

    params = ttml.NamedParameters()
    with expect_error(RuntimeError, "param_groups is not supported by optimizer type: PyOptNoDecay"):
        ttml.optimizers.create_optimizer({"type": "PyOptNoDecay", "param_groups": NO_DECAY_GROUPS}, params)


@pytest.mark.parametrize("optimizer_type", ["AdamW", "AdamWFullPrecision", "SGD", "MuonComposite"])
@pytest.mark.parametrize("value", [True, False])
def test_removed_weight_decay_skip_1d_is_rejected(expect_error, optimizer_type, value):
    """yaml-cpp ignores unknown keys, so without an explicit check an old config would load and silently start
    decaying the 1-D params it used to exempt. Any value is rejected, so stale configs get cleaned up."""
    params = ttml.NamedParameters()
    with expect_error(RuntimeError, "weight_decay_skip_1d was removed"):
        ttml.optimizers.create_optimizer({"type": optimizer_type, "weight_decay_skip_1d": value}, params)


def test_removed_weight_decay_skip_1d_is_rejected_for_python_optimizer(expect_error):
    """Python-registered optimizers bypass OptimizerRegistry::create, so the binding rejects the key too."""
    PyOptimizer = _make_python_optimizer_class()
    ttml.optimizers.register_optimizer("PyOptSkip1d", lambda config, params: PyOptimizer(params))

    params = ttml.NamedParameters()
    with expect_error(RuntimeError, "weight_decay_skip_1d was removed"):
        ttml.optimizers.create_optimizer({"type": "PyOptSkip1d", "weight_decay_skip_1d": True}, params)


def test_adamw_config_param_groups_from_python(expect_error):
    defaults = ttml.optimizers.AdamWConfig.make(1e-3, 0.9, 0.999, 1e-8, 0.1)
    with expect_error(ValueError, "Param group 1 lists unknown param 'missing'"):
        ttml.optimizers.AdamW(
            ttml.NamedParameters(), defaults, param_groups=[{"params": ["missing"], "weight_decay": 0.0}]
        )


def test_muon_with_adamw_resolves_param_groups_in_both_sections(expect_error):
    """Both inner optimizers are built through the registry, so neither section's param_groups is dropped."""
    from ttml.common.muon_optimizer import MuonWithAdamW

    config = {
        "muon": {"lr": 0.02, "param_groups": [{"modules": ["block"], "lr": 0.04}]},
        "adamw": {"lr": 1e-3, "param_groups": [{"modules": ["gamma"], "weight_decay": 0.0}]},
    }
    opt = MuonWithAdamW(config, ttml.NamedParameters())
    assert opt.muon_optimizer().get_group_lrs() == pytest.approx([0.02, 0.04])
    assert opt.adamw_optimizer().get_param_groups()[1]["weight_decay"] == 0.0

    # Options an inner optimizer does not have are rejected rather than ignored.
    with expect_error(ValueError, "Unknown param group option 'weight_decay'"):
        MuonWithAdamW({"muon": {"param_groups": [{"modules": ["block"], "weight_decay": 0.0}]}}, ttml.NamedParameters())


def _param():
    w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
    return Parameter(ttml.autograd.Tensor.from_numpy(w_np, new_type=ttnn.DataType.BFLOAT16))


def _make_stage_model():
    """A pipeline stage's model: a ``weight`` and a ``bias``, but none of the other stages' modules (tok_emb, fc)."""

    class StageModule(AbstractModuleBase):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _param()
            self.bias = _param()

        def forward(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    return StageModule()


def _make_block_model():
    """A transformer-block-like model: attention/weight, norm/gamma, mlp/weight and mlp/bias."""

    class Leaf(AbstractModuleBase):
        def __init__(self, *names) -> None:
            super().__init__()
            for name in names:
                setattr(self, name, _param())

    class Block(AbstractModuleBase):
        def __init__(self) -> None:
            super().__init__()
            self.attention = Leaf("weight")
            self.norm = Leaf("gamma")
            self.mlp = Leaf("weight", "bias")

    return Block()


def _group_members(opt):
    """Each group's param names without the model's own name prefix, e.g. {"norm/gamma", "mlp/bias"}."""
    return [{name.split("/", 1)[1] for name in group["params"]} for group in opt.get_param_groups()]


def test_registry_resolves_param_groups_from_modules(expect_error):
    """``modules`` matches any "/"-separated segment of a param name, so module names (mlp) and tensor names (gamma,
    bias) both work. A param joins the first matching group, and params in no group form group 0 with the top-level
    values."""
    params = _make_block_model().parameters()
    config = {
        "type": "AdamW",
        "lr": 0.01,
        "weight_decay": 0.1,
        "param_groups": [
            {"modules": ["gamma", "bias"], "weight_decay": 0.0},
            {"modules": ["mlp"], "lr": 0.02},  # mlp/bias is already in the group above, so only mlp/weight
        ],
    }
    opt = ttml.optimizers.create_optimizer(config, params)

    assert _group_members(opt) == [{"attention/weight"}, {"norm/gamma", "mlp/bias"}, {"mlp/weight"}]
    groups = opt.get_param_groups()
    assert [group["weight_decay"] for group in groups] == pytest.approx([0.1, 0.0, 0.1])
    assert opt.get_group_lrs() == pytest.approx([0.01, 0.01, 0.02])

    # A group whose modules match nothing (here a typo) is an error, not a silently empty group.
    with expect_error(RuntimeError, r"Param group 1 \(modules: gama\) matches no parameters"):
        ttml.optimizers.create_optimizer({"type": "AdamW", "param_groups": [{"modules": ["gama"]}]}, params)


def test_pipeline_stage_allows_param_groups_empty_on_this_stage(expect_error, tmp_path):
    """Under pipeline parallelism a stage holds only its own params, so a group may match nothing there."""
    import yaml
    from ttml.common.utils import create_optimizer

    model = _make_stage_model()
    # "tok_emb" lives on another stage, so the group is empty on this one.
    optimizer_cfg = {"type": "AdamW", "lr": 1e-3, "param_groups": [{"modules": ["tok_emb"], "lr": 2e-3}]}

    with expect_error(RuntimeError, "matches no parameters"):
        create_optimizer(model, {"training_config": {"optimizer": optimizer_cfg}})

    # The training configs point training_config.multihost_config at a separate multihost YAML file.
    multihost_path = tmp_path / "pp_multihost.yaml"
    multihost_path.write_text(
        yaml.safe_dump({"multihost_config": {"enabled": True, "pipeline_parallel_config": {"num_blocks": 2}}})
    )
    opt = create_optimizer(
        model, {"training_config": {"optimizer": optimizer_cfg, "multihost_config": str(multihost_path)}}
    )
    assert opt.get_group_lrs() == pytest.approx([1e-3, 2e-3])
    assert "allow_empty_param_groups" not in optimizer_cfg  # the caller's config is not mutated

    # A multihost config without pipeline parallelism keeps the empty-group error.
    no_pp_path = tmp_path / "ddp_multihost.yaml"
    no_pp_path.write_text(yaml.safe_dump({"multihost_config": {"enabled": True, "num_workers": 2}}))
    with expect_error(RuntimeError, "matches no parameters"):
        create_optimizer(model, {"training_config": {"optimizer": optimizer_cfg, "multihost_config": str(no_pp_path)}})


def test_muon_with_adamw_passes_allow_empty_param_groups_to_both_sections(expect_error):
    """The top-level allow_empty_param_groups (set by trainers under pipeline parallelism) reaches the inner sections,
    where the param groups are."""
    from ttml.common.muon_optimizer import MuonWithAdamW

    params = _make_stage_model().parameters()  # "weight" goes to Muon, "bias" to AdamW
    config = {
        "muon": {"param_groups": [{"modules": ["attention"], "lr": 0.04}]},
        "adamw": {"param_groups": [{"modules": ["tok_emb", "fc"], "weight_decay": 0.0}]},
    }
    with expect_error(RuntimeError, "matches no parameters"):
        MuonWithAdamW(config, params)
    opt = MuonWithAdamW({**config, "allow_empty_param_groups": True}, params)
    assert len(opt.get_param_groups()) == 4  # each section: the ungrouped group and the (empty) configured one


def test_muon_with_adamw_group_options_cover_both_inner_optimizers(expect_error):
    """Every per-group API of the wrapper works over Muon's groups followed by AdamW's."""
    from ttml.common.muon_optimizer import MuonWithAdamW

    opt = MuonWithAdamW({"muon": {"lr": 0.02}, "adamw": {"lr": 3e-4}}, ttml.NamedParameters())
    assert opt.get_group_options("lr") == pytest.approx([0.02, 3e-4])
    # An option only one inner optimizer has is an error, not a partial list.
    with expect_error(ValueError, "Unknown param group option 'weight_decay'"):
        opt.get_group_options("weight_decay")

    opt.set_group_options("lr", [0.01, 1e-4])
    assert opt.muon_optimizer().get_group_lrs() == pytest.approx([0.01])
    assert opt.adamw_optimizer().get_group_lrs() == pytest.approx([1e-4])
    with expect_error(ValueError, "MuonWithAdamW has 2 param groups, but got 3 values for 'lr'"):
        opt.set_group_options("lr", [0.1, 0.1, 0.1])
    # An option only Muon has is rejected before Muon is changed.
    with expect_error(ValueError, "Unknown param group option 'momentum'"):
        opt.set_group_options("momentum", [0.5, 0.5])
    assert opt.muon_optimizer().get_param_groups()[0]["momentum"] == pytest.approx(0.95)

    opt.set_param_group_option(0, "momentum", 0.9)  # Muon's group
    opt.set_param_group_option(1, "weight_decay", 0.05)  # AdamW's group
    assert opt.muon_optimizer().get_param_groups()[0]["momentum"] == pytest.approx(0.9)
    assert opt.adamw_optimizer().get_param_groups()[0]["weight_decay"] == pytest.approx(0.05)


def test_registry_reads_variant_only_adamw_options():
    """Stochastic Rounding (AdamW) and Kahan Summation (AdamWComposite) are parsed only for the config that has the
    field, so a renamed field would silently drop its YAML key."""
    params = _make_stage_model().parameters()

    for stochastic_rounding in (True, False):
        opt = ttml.optimizers.create_optimizer({"type": "AdamW", "stochastic_rounding": stochastic_rounding}, params)
        assert opt.get_state_dict()["stochastic_rounding"] == stochastic_rounding

    for kahan_summation, num_buffers in ((True, len(params)), (False, 0)):
        opt = ttml.optimizers.create_optimizer({"type": "AdamWComposite", "kahan_summation": kahan_summation}, params)
        assert len(opt.get_state_dict()["kahan_compensation"]) == num_buffers
