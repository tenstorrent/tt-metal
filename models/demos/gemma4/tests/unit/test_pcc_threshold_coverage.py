"""Wormhole full-model PCC gates must exist, and must be able to fail.

Device-free: reads the threshold table and the pure key-parsing helper.

Background. Every gemma-4-12B-it full_model / full_model_decode entry used to be
blackhole-only, so on a WH T3K the lookup missed and fell back to
get_pcc_threshold's 0.99 "unmeasured" default. The test then reported failure at
any real value, which made it useless as a gate AND hid a real 1x8 prefill
regression (0.9514 -> 0.8953) for an entire porting effort. A gate that is red
at every value carries no signal.
"""
import json
import os

import pytest

from models.demos.gemma4.tests.test_factory import _mesh_key_from_node_name, get_pcc_threshold

_TABLE = os.path.join(os.path.dirname(__file__), "..", "pcc_thresholds.json")
_MESHES = ["1x2", "1x4", "1x8"]

# Measured on a real T3K, bit-reproducible (paired runs agreed to every decimal
# and survived a board reset). (clean base 9d83ad5c8c7, this branch).
_WH_MEASURED = {
    "test_full_model": {"1x2": (0.9550, 0.9780), "1x4": (0.9613, 0.9591), "1x8": (0.9514, 0.9507)},
    "test_full_model_decode": {"1x2": (0.9610, 0.9742), "1x4": (0.9737, 0.9643), "1x8": (0.9735, 0.9647)},
}
# The regression this coverage exists to catch: prefill island ON.
_REGRESSED = {
    "test_full_model": {"1x2": 0.9188, "1x8": 0.8953},
    "test_full_model_decode": {"1x2": 0.9241, "1x8": 0.9459},
}


def _table():
    with open(_TABLE) as fh:
        return json.load(fh)["gemma-4-12B-it"]


def _gate(test, mesh):
    return _table()[mesh][f"{test}[wormhole_b0-{mesh}]"]


@pytest.mark.parametrize("test", sorted(_WH_MEASURED))
@pytest.mark.parametrize("mesh", _MESHES)
def test_wormhole_gate_exists(test, mesh):
    entry = _table()[mesh]
    key = f"{test}[wormhole_b0-{mesh}]"
    assert key in entry, f"12B {mesh} has no wormhole_b0 {test} gate; WH runs fall back to 0.99"
    assert 0.5 < entry[key] < 1.0


@pytest.mark.parametrize("test", sorted(_WH_MEASURED))
@pytest.mark.parametrize("mesh", _MESHES)
def test_gate_passes_a_healthy_tree(test, mesh):
    """A gate above the clean measurement is permanently red and therefore useless."""
    gate = _gate(test, mesh)
    for label, measured in zip(("base", "branch"), _WH_MEASURED[test][mesh]):
        assert gate < measured, f"{test}/{mesh}: gate {gate} >= {label} measurement {measured}"


@pytest.mark.parametrize("test", sorted(_REGRESSED))
@pytest.mark.parametrize("mesh", ["1x2", "1x8"])
def test_gate_actually_fails_the_known_regression(test, mesh):
    """The point of the gate: prefill-island-ON values must NOT pass."""
    gate, regressed = _gate(test, mesh), _REGRESSED[test][mesh]
    assert regressed < gate, f"{test}/{mesh}: gate {gate} would let the known regression {regressed} pass"


@pytest.mark.parametrize("test", sorted(_WH_MEASURED))
@pytest.mark.parametrize("mesh", _MESHES)
def test_key_resolves_to_its_own_mesh_section(test, mesh):
    """The node name must parse to the section it is filed under, or the lookup
    looks in the wrong place and silently returns the 0.99 default."""
    assert _mesh_key_from_node_name(f"{test}[wormhole_b0-{mesh}]") == mesh


def test_blackhole_gates_are_untouched():
    e = _table()
    assert e["1x1"]["test_full_model[blackhole-1x1]"] == 0.92
    assert e["1x4"]["test_full_model[blackhole-1x4]"] == 0.935
    assert e["1x8"]["test_full_model[blackhole-1x8]"] == 0.97
    assert e["1x8"]["test_full_model_decode[blackhole-1x8]"] == 0.97


def test_unmeasured_combination_still_defaults_to_099():
    """This change adds coverage, not slack."""
    assert get_pcc_threshold.__defaults__[0] == 0.99
