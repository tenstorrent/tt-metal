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


def test_optional_sdpa_policy_preserves_default_and_rejects_unknown(expect_error):
    assert "sdpa_compute_fidelity" not in precision.load_precision(precision.BASELINE)
    for fidelity in ("HiFi2", "HiFi4"):
        candidate = dict(precision.BASELINE, sdpa_compute_fidelity=fidelity)
        assert precision.load_precision(candidate)["sdpa_compute_fidelity"] == fidelity
    with expect_error(ValueError, "SDPA"):
        precision.load_precision(dict(precision.BASELINE, sdpa_compute_fidelity="LoFi"))


@pytest.mark.parametrize("fidelity", ["HiFi2", "HiFi4"])
def test_sdpa_policy_reaches_both_kernel_constructors(fidelity):
    """Execute actual constructor expressions with host stubs, without loading TTNN."""
    import ast
    from types import SimpleNamespace

    root = Path(__file__).parents[1] / "tt"
    model_tree = ast.parse((root / "model.py").read_text())
    forwarded = next(
        kw.value
        for node in ast.walk(model_tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "sdpa_compute_fidelity"
    )
    value = eval(
        compile(ast.Expression(forwarded), "model.py", "eval"), {"policy": {"sdpa_compute_fidelity": fidelity}}
    )
    tree = ast.parse((root / "multichip_decoder.py").read_text())
    target_names = {"sdpa_fidelity", "sdpa_kernel", "prefill_sdpa_kernel"}
    statements = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            (isinstance(t, ast.Name) and t.id in target_names)
            or (isinstance(t, ast.Attribute) and t.attr in target_names)
            for t in node.targets
        )
    ]
    statements.sort(key=lambda node: node.lineno)
    obj = SimpleNamespace()
    env = {
        "kwargs": {"sdpa_compute_fidelity": value},
        "obj": obj,
        "ttnn": SimpleNamespace(MathFidelity=SimpleNamespace(HiFi2="HiFi2", HiFi4="HiFi4")),
        "BlackholeComputeKernelConfig": lambda **kwargs: kwargs,
    }
    exec(compile(ast.Module(body=statements, type_ignores=[]), "multichip_decoder.py", "exec"), env)
    assert obj.sdpa_kernel["math_fidelity"] == obj.prefill_sdpa_kernel["math_fidelity"] == fidelity
    assert obj.sdpa_kernel["fp32_dest_acc_en"] is True
    assert obj.prefill_sdpa_kernel["fp32_dest_acc_en"] is (fidelity == "HiFi4")
    assert not env["kwargs"]
