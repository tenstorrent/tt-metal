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


@pytest.fixture(autouse=True)
def _deterministic():
    """These assert on PCC thresholds, so an unseeded draw makes them pass or fail by luck."""
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(0)


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
    assert '_CAPTURED_STATE["golden_out"]' in block, "the captured golden must be handed on"
    assert "_captured_kwargs_and_primary(" in block, "inputs must be prepared by the shared helper"


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
    nudged = golden + torch.randn(64) * 0.05  # close, but not exact: lands well inside 0.5..0.9999

    lenient = _fidelity_ns(target=0.5)
    lenient["_CAPTURED_STATE"]["golden_out"] = golden.clone()
    lenient["_check_captured_fidelity"](nudged.clone())  # passes a low bar

    strict = _fidelity_ns(target=0.9999)
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


# --- more than one input set -------------------------------------------------------------------
# A component can be right for the shape it was captured at and wrong for the next one. Gating on a
# single input cannot tell those apart, so every captured set is now held to the same target.


def _demo_with_samples(tmp_path, component, count):
    """A demo tree with `count` captured sets: the primary in place, extras under samples/."""
    import torch

    safe = component.replace(".", "_").lower()
    comp = tmp_path / "_captured" / safe
    for idx in range(count):
        d = comp if idx == 0 else comp / "samples" / f"{idx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save((torch.full((4,), float(idx)),), d / "args.pt")
        torch.save({}, d / "kwargs.pt")
        torch.save(torch.full((4,), float(idx)), d / "output.pt")
    test_file = tmp_path / "tests" / "pcc" / "test_x.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("")
    return test_file


@requires_torch
def test_the_primary_sample_stays_exactly_where_it_was(tmp_path) -> None:
    """Extras must not move the primary: everything else in the tree reads it in place."""
    test_file = _demo_with_samples(tmp_path, "enc.blk", 3)
    ns = _loader_ns(__file__=str(test_file))
    dirs = ns["_captured_sample_dirs"]("enc.blk")
    assert len(dirs) == 3
    assert dirs[0] == tmp_path / "_captured" / "enc_blk", "the primary must come first, in place"
    assert all(d.parent.name == "samples" for d in dirs[1:])


@requires_torch
def test_every_captured_set_is_offered_to_the_gate(tmp_path) -> None:
    import torch

    test_file = _demo_with_samples(tmp_path, "enc.blk", 3)
    ns = _loader_ns(__file__=str(test_file))

    class _Mod(torch.nn.Module):
        def forward(self, hidden):  # noqa: D401 -- name is what the mapping uses
            return hidden

    extras = ns["_captured_extra_samples"]("enc.blk", _Mod())
    assert len(extras) == 2, "the two sets beyond the gating one must be returned"
    for name, kwargs, primary, golden in extras:
        assert primary[0] == "hidden", "prepared through the same mapping as the gating sample"
        assert isinstance(golden, torch.Tensor)


@requires_torch
def test_a_single_captured_set_behaves_exactly_as_before(tmp_path) -> None:
    """No samples/ directory must mean no extra work and no extra requirements."""
    import torch

    test_file = _demo_with_samples(tmp_path, "enc.blk", 1)
    ns = _loader_ns(__file__=str(test_file))
    assert ns["_captured_sample_dirs"]("enc.blk") == [tmp_path / "_captured" / "enc_blk"]
    assert ns["_captured_extra_samples"]("enc.blk", torch.nn.Identity()) == []


@requires_torch
def test_an_incomplete_sample_is_skipped_not_crashed_on(tmp_path) -> None:
    test_file = _demo_with_samples(tmp_path, "enc.blk", 2)
    (tmp_path / "_captured" / "enc_blk" / "samples" / "01" / "output.pt").unlink()
    ns = _loader_ns(__file__=str(test_file))
    assert ns["_load_sample"](tmp_path / "_captured" / "enc_blk" / "samples" / "01") is None
    assert ns["_captured_extra_samples"]("enc.blk", None) == []


def test_the_generated_test_gates_on_every_sample() -> None:
    """The extra samples must be asserted on, not merely printed."""
    import scripts.tt_hw_planner.bringup_loop as bl

    body = bl._PCC_TEST_TEMPLATE
    assert "_captured_extra_samples(COMPONENT_NAME" in body, "the test must look for extra samples"
    tail = body[body.find("_captured_extra_samples(COMPONENT_NAME") :]
    assert "assert not _failures" in tail, "a failing extra sample must fail the test"


# --- the capture side has to produce them ------------------------------------------------------


def test_extra_rounds_default_to_more_than_one_sample(monkeypatch) -> None:
    from scripts.tt_hw_planner import capture_inputs as ci

    monkeypatch.delenv(ci._SAMPLES_ENV, raising=False)
    assert ci._extra_sample_rounds() >= 1, "the default must actually exercise generalisation"
    monkeypatch.setenv(ci._SAMPLES_ENV, "1")
    assert ci._extra_sample_rounds() == 0, "single-sample capture must remain available"
    monkeypatch.setenv(ci._SAMPLES_ENV, "not-a-number")
    assert ci._extra_sample_rounds() >= 1, "a bad value must not disable capture"


@requires_torch
def test_extra_capture_rounds_never_endanger_the_primary_set() -> None:
    """A failed extra round must leave capture exactly as it would have been without the feature."""
    import torch

    from scripts.tt_hw_planner import capture_inputs as ci

    state = {"enc.blk": {"args": (), "kwargs": {}, "output": torch.zeros(2)}}
    resolved = [("enc.blk", None, "enc.blk")]

    def exploding_driver(_model, _inputs):
        raise RuntimeError("driver blew up on the extra round")

    extras = ci._capture_extra_samples(
        model=None,
        pixel_values=torch.zeros(1, 3, 4, 4),
        resolved=resolved,
        state=state,
        seed=0,
        rounds=2,
        driver=exploding_driver,
        verbose=False,
    )
    assert extras == {}
    assert "enc.blk" in state and "output" in state["enc.blk"], "the primary capture must survive"


@requires_torch
def test_extra_rounds_collect_a_set_per_round_and_restore_the_primary() -> None:
    import torch

    from scripts.tt_hw_planner import capture_inputs as ci

    primary = {"args": (), "kwargs": {}, "output": torch.zeros(2)}
    state = {"enc.blk": dict(primary)}
    rounds_seen = []

    def driver(_model, inputs):
        rounds_seen.append(inputs)
        state["enc.blk"] = {"args": (), "kwargs": {}, "output": torch.full((2,), float(len(rounds_seen)))}
        return True, []

    extras = ci._capture_extra_samples(
        model=None,
        pixel_values=torch.zeros(1, 3, 4, 4),
        resolved=[("enc.blk", None, "enc.blk")],
        state=state,
        seed=0,
        rounds=2,
        driver=driver,
        verbose=False,
    )
    assert len(extras["enc.blk"]) == 2, "one set per round"
    assert torch.equal(state["enc.blk"]["output"], primary["output"]), "primary restored afterwards"


def test_no_driver_means_no_extra_rounds() -> None:
    from scripts.tt_hw_planner import capture_inputs as ci

    assert (
        ci._capture_extra_samples(
            model=None, pixel_values=None, resolved=[], state={}, seed=0, rounds=3, driver=None, verbose=False
        )
        == {}
    )
