# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
    collect_accuracy_and_determinism_results,
)


@pytest.mark.parametrize(
    "value,label",
    [(float("nan"), "nan=1"), (float("inf"), r"\+inf=1"), (-float("inf"), "-inf=1")],
)
@pytest.mark.parametrize("side", ["expected", "actual"])
@pytest.mark.parametrize("assertion", [assert_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_nonfinite_values(assertion, side: str, value: float, label: str, expect_error) -> None:
    expected = torch.zeros(1)
    actual = expected.clone()
    target = expected if side == "expected" else actual
    target[0] = value
    with expect_error(AssertionError, label):
        assertion(expected, actual)


@pytest.mark.parametrize("assertion", [assert_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_shape_metadata_mismatch(assertion, expect_error) -> None:
    with expect_error(AssertionError, "shape"):
        assertion(torch.zeros(2), torch.zeros(1, 2))


@pytest.mark.parametrize("assertion", [assert_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_dtype_metadata_mismatch(assertion, expect_error) -> None:
    with expect_error(AssertionError, "dtype"):
        assertion(torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.bfloat16))


def test_assert_accurate_rejects_pcc_below_threshold(expect_error) -> None:
    with expect_error(AssertionError, "PCC"):
        assert_accurate(torch.arange(8.0), torch.arange(7, -1, -1.0), pcc_threshold=0.99)


def test_assert_equal_rejects_value_mismatch(expect_error) -> None:
    with expect_error(AssertionError, "values differ"):
        assert_equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))


def test_assert_bit_identical_rejects_bit_mismatch(expect_error) -> None:
    expected = torch.tensor([1.0], dtype=torch.float32)
    actual = torch.nextafter(expected, torch.tensor([2.0]))
    with expect_error(AssertionError, "values differ"):
        assert_bit_identical(expected, actual)


def test_assert_bit_identical_rejects_signed_zero_mismatch(expect_error) -> None:
    with expect_error(AssertionError, "bit patterns differ"):
        assert_bit_identical(torch.tensor([0.0]), torch.tensor([-0.0]))


@pytest.mark.parametrize("assertion", [assert_equal, assert_bit_identical])
def test_exact_assertions_accept_identical_finite_tensors(assertion) -> None:
    expected = torch.tensor([1.0, 2.0])
    assertion(expected, expected.clone())


def test_assert_accurate_returns_measured_pcc() -> None:
    expected = torch.arange(8.0)
    actual = expected + torch.linspace(0.0, 1e-4, 8)
    pcc = assert_accurate(expected, actual, pcc_threshold=0.999999)
    assert 0.999999 <= pcc <= 1.0


class _FakeDeviceTensor:
    def __init__(
        self,
        values: torch.Tensor,
        *,
        dtype: object = "float32",
        layout: object = "tile",
        memory_config: object = "dram",
    ) -> None:
        self.values = values.clone()
        self.shape = self.values.shape
        self.dtype = dtype
        self.layout = layout
        self._memory_config = memory_config

    def memory_config(self) -> object:
        return self._memory_config


@pytest.fixture
def fake_device_ttnn(monkeypatch):
    calls = {"empty": [], "deallocate": [], "to_torch": []}

    def make(
        values: list[float],
        *,
        shape: tuple[int, ...] | None = None,
        dtype: object = "float32",
        layout: object = "tile",
        memory_config: object = "dram",
    ) -> _FakeDeviceTensor:
        tensor = torch.tensor(values, dtype=torch.float32)
        return _FakeDeviceTensor(
            tensor.reshape(shape or tensor.shape), dtype=dtype, layout=layout, memory_config=memory_config
        )

    def empty(shape, *, dtype, layout, device, memory_config):
        del device
        output = _FakeDeviceTensor(torch.zeros(tuple(shape)), dtype=dtype, layout=layout, memory_config=memory_config)
        calls["empty"].append(output)
        return output

    def ne(reference, actual, *, dtype, output_tensor):
        del dtype
        output_tensor.values.copy_(reference.values.ne(actual.values))
        return output_tensor

    def maximum(lhs, rhs):
        return make([max(lhs.values.item(), rhs.values.item())])

    def to_torch(tensor):
        calls["to_torch"].append(tensor)
        return tensor.values

    monkeypatch.setattr(ttnn, "empty", empty)
    monkeypatch.setattr(ttnn, "ne", ne)
    monkeypatch.setattr(ttnn, "max", lambda tensor: make([tensor.values.max().item()]))
    monkeypatch.setattr(ttnn, "maximum", maximum)
    monkeypatch.setattr(ttnn, "deallocate", calls["deallocate"].append)
    monkeypatch.setattr(ttnn, "to_torch", to_torch)
    return make, calls


def test_collect_accuracy_and_determinism_rejects_invalid_count_before_run(fake_device_ttnn, expect_error) -> None:
    del fake_device_ttnn
    invoked = False

    def run():
        nonlocal invoked
        invoked = True
        return ()

    with expect_error(ValueError, "greater than one"):
        collect_accuracy_and_determinism_results(object(), run, count=1)
    assert not invoked


def test_collect_accuracy_and_determinism_rejects_empty_outputs(fake_device_ttnn, expect_error) -> None:
    del fake_device_ttnn
    with expect_error(ValueError, "at least one"):
        collect_accuracy_and_determinism_results(object(), lambda: ())


def test_collect_accuracy_and_determinism_rejects_changed_arity(fake_device_ttnn, expect_error) -> None:
    make, calls = fake_device_ttnn
    repeat_outputs = (make([1.0]), make([2.0]))
    runs = iter(((make([1.0]),), repeat_outputs))
    with expect_error(ValueError, "different number"):
        collect_accuracy_and_determinism_results(object(), lambda: next(runs), count=2)
    assert all(output in calls["deallocate"] for output in repeat_outputs)


@pytest.mark.parametrize("field", ["shape", "dtype", "layout", "memory_config"])
def test_collect_accuracy_and_determinism_rejects_changed_metadata(fake_device_ttnn, field: str, expect_error) -> None:
    make, calls = fake_device_ttnn
    reference = make([1.0, 2.0])
    kwargs = {}
    values = [1.0, 2.0]
    if field == "shape":
        values = [1.0]
    else:
        kwargs[field] = "different"
    repeat = make(values, **kwargs)
    runs = iter(((reference,), (repeat,)))
    with expect_error(ValueError, "different metadata"):
        collect_accuracy_and_determinism_results(object(), lambda: next(runs), count=2)
    assert repeat in calls["deallocate"]


def test_collect_accuracy_and_determinism_aggregates_multiple_outputs(fake_device_ttnn) -> None:
    make, calls = fake_device_ttnn
    reference = (make([1.0, 2.0]), make([3.0], shape=(1, 1)))
    repeat_one = (make([1.0, 2.0]), make([3.0], shape=(1, 1)))
    repeat_two = (make([1.0, 2.0]), make([3.0], shape=(1, 1)))
    runs = iter((reference, repeat_one, repeat_two))

    device_outputs, host_outputs, marker = collect_accuracy_and_determinism_results(
        object(), lambda: next(runs), count=3
    )

    assert device_outputs == reference
    assert all(torch.equal(host, expected.values) for host, expected in zip(host_outputs, reference, strict=True))
    assert marker.item() == 0
    assert len(calls["empty"]) == len(reference)
    assert calls["to_torch"][: len(reference)] == list(reference)
    assert all(output in calls["deallocate"] for output in (*repeat_one, *repeat_two))


def test_collect_accuracy_and_determinism_reports_any_output_mismatch(fake_device_ttnn) -> None:
    make, _ = fake_device_ttnn
    runs = iter(((make([1.0]), make([2.0])), (make([1.0]), make([3.0]))))

    _, _, marker = collect_accuracy_and_determinism_results(object(), lambda: next(runs), count=2)

    assert marker.item() == 1
