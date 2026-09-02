"""The captured golden must be used, and the captured cache must not be dropped by name.

`_captured/<comp>/output.pt` is the output the real module produced for the exact inputs capture
recorded. The generated test loaded it into `_cap_output` and never mentioned that variable again,
so the one artefact that could say "the inputs this test rebuilt actually drive the module the way
the real forward did" was thrown away. Meanwhile the cache kwargs were dropped by a hardcoded name
list, which guaranteed the reference ran without the state the real forward had.

These exercise the injected helpers for real -- the loader source is exec'd into a namespace -- so
they fail if the behaviour regresses, not merely if the source text is reworded.
"""
from __future__ import annotations

import importlib.util

import pytest

from scripts.tt_hw_planner.capture_inputs import _CAPTURED_SHORT_CIRCUIT_BLOCK, CAPTURE_LOADER_SOURCE

requires_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")


def _loader_ns(**extra):
    """Exec the injected loader source, standing in for the generated test module around it."""
    ns = dict(extra)
    exec(CAPTURE_LOADER_SOURCE, ns)  # noqa: S102 -- this source is what really lands in the test
    return ns


class _CacheLike:
    """Stands in for a live cache object: tensors held in attributes, not passed as tensors."""

    def __init__(self, tensors):
        self.key_cache = list(tensors)


# --- what counts as data vs live state, decided from the value ---------------------------------


@requires_torch
@pytest.mark.parametrize(
    ("value", "plain"),
    [
        (None, True),
        (True, True),
        (3, True),
        (1.5, True),
        ("text", True),
        ("tensor", True),
        ("tensor_list", True),
        ("cache", False),
    ],
    ids=["none", "bool", "int", "float", "str", "tensor", "tensor-list", "cache-object"],
)
def test_plain_inputs_are_told_apart_from_live_state(value, plain) -> None:
    import torch

    ns = _loader_ns()
    if value == "tensor":
        value = torch.zeros(2)
    elif value == "tensor_list":
        value = [torch.zeros(2), torch.ones(2)]
    elif value == "cache":
        value = _CacheLike([torch.zeros(2)])
    assert ns["_is_plain_input"](value) is plain


def test_a_cache_class_nobody_named_is_still_recognised_as_state() -> None:
    """The point of asking the value: an unknown cache class must not need adding to a list."""
    ns = _loader_ns()
    unknown = type("SomeFutureCacheNobodyHasHeardOf", (), {"__init__": lambda self: None})()
    assert ns["_is_plain_input"](unknown) is False


# --- the cache is no longer dropped by name ----------------------------------------------------


def test_captured_cache_is_not_filtered_out_by_a_name_list() -> None:
    block = _CAPTURED_SHORT_CIRCUIT_BLOCK
    for name in ("past_key_values", "past_key_value", "cache_position", "use_cache"):
        assert f'"{name}"' not in block, f"{name} must not be dropped by name any more"
    assert '_CAPTURED_STATE["stateful_keys"]' in block, "live state must be recorded for the retry"
    assert '_CAPTURED_STATE["golden_out"]' in block, "the captured golden must be handed on"


def test_the_captured_golden_is_no_longer_discarded() -> None:
    """`_cap_output` was unpacked and then never referenced again anywhere."""
    block = _CAPTURED_SHORT_CIRCUIT_BLOCK
    assert block.count("_cap_output") >= 2, "unpacking the golden without using it is the old bug"


# --- the golden actually gates the reference ---------------------------------------------------


def _fidelity_ns(target=0.99, component="enc.blk.0"):
    """Loader namespace wired to the names the generated test module provides around it."""
    import torch

    def _normalize_out(out):
        return out[0] if isinstance(out, tuple) else out

    def comp_pcc(golden, calculated, pcc):
        # Stand-in with the real signature: correlation of two flattened tensors.
        a = golden.flatten().to(torch.float64)
        b = calculated.flatten().to(torch.float64)
        if a.std() == 0 or b.std() == 0:
            value = 1.0 if torch.allclose(a, b) else 0.0
        else:
            value = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        return value >= pcc, value

    return _loader_ns(
        _normalize_out=_normalize_out,
        comp_pcc=comp_pcc,
        PCC_TARGET=target,
        COMPONENT_NAME=component,
    )


@requires_torch
def test_reference_that_reproduces_the_golden_passes() -> None:
    import torch

    ns = _fidelity_ns()
    out = torch.randn(4, 8)
    ns["_CAPTURED_STATE"]["golden_out"] = out.clone()
    ns["_check_captured_fidelity"](out)  # must not raise


@requires_torch
def test_reference_that_does_not_reproduce_the_golden_fails_the_test(expect_error) -> None:
    """THE case: the rebuilt inputs drive the module differently than the real forward did.

    Any PCC printed after this point would describe a computation the model never performed, so a
    green result would be meaningless -- exactly the false pass this check exists to stop.
    """
    import torch

    ns = _fidelity_ns()
    ns["_CAPTURED_STATE"]["golden_out"] = torch.randn(4, 8)
    with expect_error(pytest.fail.Exception, "does not reproduce the captured output"):
        ns["_check_captured_fidelity"](torch.randn(4, 8))


@requires_torch
def test_fidelity_is_held_to_the_components_own_bar(expect_error) -> None:
    """No second magic number: the reconstruction must meet the same PCC the component must meet."""
    import torch

    golden = torch.randn(64)
    nudged = golden + torch.randn(64) * 0.05  # close, but not exact

    lenient = _fidelity_ns(target=0.5)
    lenient["_CAPTURED_STATE"]["golden_out"] = golden.clone()
    lenient["_check_captured_fidelity"](nudged.clone())  # passes a low bar

    strict = _fidelity_ns(target=0.999)
    strict["_CAPTURED_STATE"]["golden_out"] = golden.clone()
    with expect_error(pytest.fail.Exception, "fidelity pcc"):
        strict["_check_captured_fidelity"](nudged.clone())


@requires_torch
def test_no_golden_and_shape_mismatch_are_not_failures() -> None:
    """Absence of evidence must not fail a component that is fine."""
    import torch

    ns = _fidelity_ns()
    ns["_check_captured_fidelity"](torch.randn(4, 8))  # nothing captured at all

    ns["_CAPTURED_STATE"]["golden_out"] = torch.randn(2, 2)
    ns["_check_captured_fidelity"](torch.randn(4, 8))  # shape differs -> reported, not failed


@requires_torch
def test_a_tuple_golden_is_normalised_before_comparing() -> None:
    """Components that return tuples must compare against the same element the test compares."""
    import torch

    ns = _fidelity_ns()
    out = torch.randn(4, 8)
    ns["_CAPTURED_STATE"]["golden_out"] = (out.clone(), torch.randn(4, 4))
    ns["_check_captured_fidelity"](out)  # must not raise


# --- the captured state is offered first, then fallen back on ----------------------------------


@requires_torch
def test_captured_state_is_offered_to_the_reference_first() -> None:
    """The real forward ran WITH its cache; the reference has to be given the chance to as well."""
    import torch

    ns = _loader_ns()
    ns["_CAPTURED_STATE"]["stateful_keys"] = ("past_key_values",)
    seen = {}

    def module(**kwargs):
        seen.update(kwargs)
        return torch.zeros(2)

    _, used = ns["_reference_forward"](module, {"x": torch.zeros(2), "past_key_values": _CacheLike([])})
    assert "past_key_values" in seen, "the captured state must be offered, not dropped up front"
    assert "past_key_values" in used


@requires_torch
def test_a_module_that_rejects_the_state_is_retried_without_it() -> None:
    """Modules that rebuild their own state must stay testable rather than skipping."""
    import torch

    ns = _loader_ns()
    ns["_CAPTURED_STATE"]["stateful_keys"] = ("past_key_values",)
    calls = []

    def picky(**kwargs):
        calls.append(dict(kwargs))
        if "past_key_values" in kwargs:
            raise TypeError("this module builds its own cache")
        return torch.zeros(2)

    out, used = ns["_reference_forward"](picky, {"x": torch.zeros(2), "past_key_values": _CacheLike([])})
    assert out is not None
    assert len(calls) == 2, "must try with the state before falling back"
    assert "past_key_values" not in used, "the fallback kwargs are what the caller must go on to use"


@requires_torch
def test_a_forward_that_fails_both_ways_reports_the_real_error(expect_error) -> None:
    import torch

    ns = _loader_ns()
    ns["_CAPTURED_STATE"]["stateful_keys"] = ("past_key_values",)

    def broken(**kwargs):
        raise RuntimeError("shape mismatch somewhere")

    with expect_error(RuntimeError, "shape mismatch somewhere"):
        ns["_reference_forward"](broken, {"x": torch.zeros(2), "past_key_values": _CacheLike([])})


@requires_torch
def test_no_captured_state_means_a_single_attempt() -> None:
    """Components on the synthetic path must not pay for a retry that cannot help them."""
    import torch

    ns = _loader_ns()
    calls = []

    def module(**kwargs):
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):  # allow-pytest.raises: asserting the raw error propagates
        ns["_reference_forward"](module, {"x": torch.zeros(2)})
    assert len(calls) == 1, "nothing stateful to drop, so there is nothing to retry"
