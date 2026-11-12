# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    TokenErrorAccumulator,
    golden_floor,
    nrmse,
    assert_accurate,
    assert_bit_identical,
    assert_equal,
    collect_accuracy_and_determinism_results,
)


def _ungated_accurate(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return assert_accurate(expected, actual, nrmse_threshold=None)  # these tests exercise the checks before the metrics


@pytest.mark.parametrize(
    "value,label",
    [(float("nan"), "nan=1"), (float("inf"), r"\+inf=1"), (-float("inf"), "-inf=1")],
)
@pytest.mark.parametrize("assertion", [_ungated_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_nonfinite_values(assertion, value: float, label: str, expect_error) -> None:
    tensor = torch.tensor([value])
    with expect_error(AssertionError, label):
        assertion(tensor, tensor.clone())


@pytest.mark.parametrize("assertion", [_ungated_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_shape_metadata_mismatch(assertion, expect_error) -> None:
    with expect_error(AssertionError, "shape"):
        assertion(torch.zeros(2), torch.zeros(1, 2))


@pytest.mark.parametrize("assertion", [_ungated_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_dtype_metadata_mismatch(assertion, expect_error) -> None:
    with expect_error(AssertionError, "dtype"):
        assertion(torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.bfloat16))


def test_assert_accurate_rejects_pcc_below_threshold(expect_error) -> None:
    with expect_error(AssertionError, "PCC"):
        assert_accurate(torch.arange(8.0), torch.arange(7, -1, -1.0), pcc_threshold=0.99, nrmse_threshold=None)


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


def test_assert_accurate_returns_measured_pcc() -> None:
    expected = torch.arange(8.0)
    actual = expected + torch.linspace(0.0, 1e-4, 8)
    pcc = assert_accurate(expected, actual, pcc_threshold=0.999999, nrmse_threshold=None)
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


def test_nrmse_gate_fails_a_scale_error_that_passes_pcc(expect_error) -> None:
    expected = torch.arange(1024, dtype=torch.float32)
    actual = expected * 2
    assert_accurate(expected, actual, pcc_threshold=0.999999, nrmse_threshold=None)
    with expect_error(AssertionError, "NRMSE 1.000e\\+00 > 5.000e-02"):
        assert_accurate(expected, actual, nrmse_threshold=0.05)


def test_nrmse_keep_dims_gates_the_worst_slice(expect_error) -> None:
    expected = torch.ones(8, 64)
    expected[:, ::2] = -1.0  # every head the same pattern of +-1, norm 8
    actual = expected.clone()
    actual[5] *= 1.2  # one head 20% off: whole-tensor NRMSE 0.2 * 8 / sqrt(8 * 64) = 0.0707
    assert_accurate(expected, actual, pcc_threshold=0.99, nrmse_threshold=0.1)
    with expect_error(AssertionError, "NRMSE 2.000e-01 \\(worst slice 5 over dims \\(0,\\)\\)"):
        assert_accurate(expected, actual, pcc_threshold=0.99, nrmse_threshold=0.1, nrmse_keep_dims=(0,))


def test_pcc_keep_dims_gates_the_worst_slice(expect_error) -> None:
    signal = torch.linspace(-1.0, 1.0, 8)
    expected = torch.stack([100.0 * signal, signal])  # a large correct head next to a small one
    actual = torch.stack([100.0 * signal, -signal])  # the small head is anti-correlated
    assert assert_accurate(expected, actual, pcc_threshold=0.999, nrmse_threshold=None) > 0.999
    with expect_error(AssertionError, "worst slice 1"):
        assert_accurate(expected, actual, pcc_threshold=0.999, nrmse_threshold=None, pcc_keep_dims=(0,))


def test_offset_peak_gate_detects_local_carry_fault(expect_error) -> None:
    expected = torch.linspace(-1, 1, 1280 * 32).reshape(1280, 32)
    actual = expected.clone()
    actual[640, :] += 0.2
    assert_accurate(expected, actual, pcc_threshold=0.999, nrmse_threshold=0.05)
    with expect_error(AssertionError, "relative L-inf"):
        assert_accurate(expected, actual, pcc_threshold=0.999, nrmse_threshold=0.05, linf_threshold=0.25)


def test_nrmse_is_error_norm_over_expected_norm() -> None:
    expected = torch.tensor([3.0, 4.0])  # norm 5
    actual = torch.tensor([3.0, 5.0])  # error norm 1
    assert nrmse(expected, actual).item() == pytest.approx(0.2, rel=1e-15)


def test_nrmse_sees_a_uniform_scale_error_that_pcc_does_not() -> None:
    expected = torch.linspace(-1.0, 1.0, 64)
    actual = expected * 0.95
    assert assert_accurate(expected, actual, pcc_threshold=0.999999, nrmse_threshold=None) == pytest.approx(
        1.0, abs=1e-12
    )
    assert nrmse(expected, actual).item() == pytest.approx(0.05, rel=1e-6)


def test_nrmse_is_reported_per_kept_slice() -> None:
    expected = torch.tensor([[3.0, 4.0], [6.0, 8.0]])  # row norms 5, 10
    actual = expected + torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    torch.testing.assert_close(
        nrmse(expected, actual, keep_dims=(0,)), torch.tensor([0.2, 0.1], dtype=torch.float64), rtol=1e-15, atol=0
    )


def test_nrmse_keeping_every_dimension_is_per_element() -> None:
    expected = torch.tensor([2.0, 4.0])
    actual = torch.tensor([2.0, 5.0])
    torch.testing.assert_close(
        nrmse(expected, actual, keep_dims=(0,)), torch.tensor([0.0, 0.25], dtype=torch.float64), rtol=1e-15, atol=0
    )


def test_nrmse_floor_belongs_to_the_whole_golden() -> None:
    expected = torch.tensor([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0], [30.0, 40.0]], dtype=torch.float64)  # 5, 0, 10, 50
    actual = expected + torch.tensor([[0.0, 0.0], [0.0, 0.01], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    floor = golden_floor(expected.norm(dim=1))
    assert floor == pytest.approx(0.05, rel=1e-15)  # 1% of the lower median, 5

    whole = nrmse(expected, actual, keep_dims=(0,))
    window = nrmse(expected[1:], actual[1:], keep_dims=(0,), floor=floor)

    assert whole[1].item() == window[0].item() == pytest.approx(0.2, rel=1e-12)
    assert whole[0].item() == whole[2].item() == whole[3].item() == 0.0
    # The window's own floor (1% of median(0, 10, 50) = 0.1) would have scored the same token 0.1.
    assert nrmse(expected[1:], actual[1:], keep_dims=(0,))[0].item() == pytest.approx(0.1, rel=1e-12)


def test_nrmse_of_an_all_zero_golden_is_zero_only_when_exact() -> None:
    assert nrmse(torch.zeros(4), torch.zeros(4)).item() == 0.0
    assert nrmse(torch.zeros(4), torch.full((4,), 1e-6)).item() == float("inf")


def test_nrmse_arithmetic_is_float64() -> None:
    expected = torch.tensor([1.0], dtype=torch.float32)
    actual = torch.tensor([1.0 + 2**-23], dtype=torch.float32)  # one FP32 ulp above 1
    assert nrmse(expected, actual).item() == 2**-23


def test_nrmse_rejects_non_finite_and_mismatched_shapes(expect_error) -> None:
    with expect_error(ValueError, "1 non-finite"):
        nrmse(torch.tensor([1.0, 1.0]), torch.tensor([1.0, float("nan")]))
    with expect_error(ValueError, "shape"):
        nrmse(torch.zeros(2, 3), torch.zeros(3, 2))


def test_streamed_token_errors_equal_the_batch_nrmse() -> None:
    generator = torch.Generator().manual_seed(0)
    expected = torch.randn(23, 6, generator=generator, dtype=torch.float64)
    expected[7] *= 1e-4  # one near-silent token, so the floor matters
    actual = expected + 1e-3 * torch.randn(23, 6, generator=generator, dtype=torch.float64)
    accumulator = TokenErrorAccumulator()
    for start, end in ((0, 5), (5, 6), (6, 17), (17, 23)):  # chunk edges off the 4-token window grid
        accumulator.update(expected[start:end], actual[start:end])
    floor = golden_floor(accumulator.expected_norms())

    assert accumulator.tokens == 23
    assert accumulator.nrmse() == pytest.approx(nrmse(expected, actual).item(), rel=1e-14)
    windows = [nrmse(expected[i : i + 4], actual[i : i + 4]).item() for i in range(0, 23, 4)]
    assert accumulator.window_nrmse(4).tolist() == pytest.approx(windows, rel=1e-14)
    torch.testing.assert_close(
        accumulator.token_nrmse(floor), nrmse(expected, actual, keep_dims=(0,)), rtol=1e-14, atol=0
    )


def test_streamed_token_errors_on_exact_zeros_equal_the_batch_nrmse() -> None:
    silent = TokenErrorAccumulator()
    silent.update(torch.zeros(4, 3), torch.zeros(4, 3))
    assert silent.nrmse() == nrmse(torch.zeros(4, 3), torch.zeros(4, 3)).item() == 0.0
    assert silent.window_nrmse(2).tolist() == [0.0, 0.0]

    expected = torch.tensor([[3.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # a matching silent window, then
    actual = torch.tensor([[3.0, 5.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])  # an error on a silent token
    mixed = TokenErrorAccumulator()
    mixed.update(expected, actual)
    batch = [nrmse(expected[i : i + 1], actual[i : i + 1]).item() for i in range(4)]
    assert mixed.window_nrmse(1).tolist() == batch == [0.2, 0.0, 0.0, float("inf")]


def test_token_error_accumulator_rejects_mismatched_chunks_and_empty_reads(expect_error) -> None:
    accumulator = TokenErrorAccumulator()
    with expect_error(ValueError, "no tokens"):
        accumulator.nrmse()
    with expect_error(ValueError, "matching"):
        accumulator.update(torch.zeros(2, 3), torch.zeros(3, 2))


def test_nrmse_threshold_is_required(expect_error) -> None:
    with expect_error(TypeError, "nrmse_threshold"):
        assert_accurate(torch.ones(2), torch.ones(2))
