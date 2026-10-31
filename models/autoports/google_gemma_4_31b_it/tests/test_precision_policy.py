"""Reject ambiguous or unimplemented policies before device construction."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("precision", Path(__file__).parents[1] / "tt/precision.py")
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["weight_groups"].pop("down"),
        lambda p: p["compute_fidelities"].update(attention="HiFi2"),
        lambda p: p.update(residual_dtype="bfloat8_b"),
        lambda p: p["ccl_dtype"].update(decode="float32"),
        lambda p: p["layer_exceptions"].update({"60": {"mlp": "bfp4"}}),
        lambda p: p.update(max_seq_len=32768),
    ],
)
def test_reject_unconstructible_policy(mutate):
    policy = copy.deepcopy(precision.BASELINE)
    mutate(policy)
    with pytest.raises(ValueError):  # allow-pytest.raises: Host-only schema test runs without TTNN fixtures.
        precision.load_precision(policy)


def test_policy_does_not_mutate_caller():
    policy = precision.load_precision(precision.BASELINE)
    policy["weight_groups"]["down"] = "bf16"
    assert precision.BASELINE["weight_groups"]["down"] == "bfp4"
