# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.common.modules.lazy_buffer import LazyBuffer
from models.common.modules.sampling import sampling_1d
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.sampling.tt_log_probs import LogProbsCalculator


class FakeTensor:
    def __init__(self, name):
        self.name = name


class FakeLogProbsCalculator:
    instances = []

    def __init__(self, *args):
        self.buffer = FakeTensor(f"log-probs-{len(self.instances)}")
        self.instances.append(self)

    def release(self):
        if self.buffer is not None:
            sampling_1d.ttnn.deallocate(self.buffer)
            self.buffer = None


def _lazy_buffer(name):
    return LazyBuffer(
        source=name,
        dtype=object(),
        layout=object(),
        device=object(),
        mesh_mapper=object(),
    )


def _sampler_config(index_offsets, owned_specs):
    return SimpleNamespace(
        index_offsets=index_offsets,
        local_indices=owned_specs["local_indices"],
        invalid_vocab_mask=owned_specs.get("invalid_vocab_mask"),
        invalid_vocab_tail_mask=owned_specs.get("invalid_vocab_tail_mask"),
        invalid_vocab_tail_width=32,
        seeds=owned_specs["seeds"],
        user_ids=owned_specs["user_ids"],
        mesh_device=object(),
        tt_ccl=None,
        sub_core_grids=None,
        start_core=None,
        max_batch_size=32,
        is_resolved=lambda: True,
    )


def test_sampling_release_is_idempotent_preserves_borrowed_and_allows_reload(monkeypatch):
    deallocated = []
    allocations = []

    def from_torch(source, **kwargs):
        tensor = FakeTensor(f"{source}-{len(allocations)}")
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(sampling_1d.ttnn, "Tensor", FakeTensor)
    monkeypatch.setattr(sampling_1d.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(sampling_1d.ttnn, "deallocate", deallocated.append)

    from models.common import utils as common_utils

    FakeLogProbsCalculator.instances = []
    monkeypatch.setattr(common_utils, "LogProbsCalculator", FakeLogProbsCalculator)

    borrowed_index_offsets = FakeTensor("borrowed-index-offsets")
    owned_specs = {
        "local_indices": _lazy_buffer("local-indices"),
        "invalid_vocab_mask": _lazy_buffer("invalid-vocab-mask"),
        "invalid_vocab_tail_mask": _lazy_buffer("invalid-vocab-tail-mask"),
        "seeds": _lazy_buffer("seeds"),
        "user_ids": _lazy_buffer("user-ids"),
    }
    config = _sampler_config(borrowed_index_offsets, owned_specs)
    sampler = object.__new__(Sampling1D)
    sampler.config = config
    sampler._device_buffers_loaded = False

    sampler.load_device_buffers()
    first_owned = {name: getattr(sampler, f"_{name}") for name in owned_specs}
    first_calculator = sampler._log_probs_calculator
    assert sampler._index_offsets is borrowed_index_offsets

    sampler.release()
    deallocations_after_first_release = list(deallocated)
    sampler.release()

    assert deallocated == deallocations_after_first_release
    assert borrowed_index_offsets not in deallocated
    assert set(first_owned.values()).issubset(deallocated)
    assert first_calculator.buffer is None
    assert not sampler._device_buffers_loaded
    assert all(spec._value is None for spec in owned_specs.values())

    sampler.load_device_buffers()

    assert sampler._device_buffers_loaded
    assert sampler._index_offsets is borrowed_index_offsets
    assert sampler._log_probs_calculator is not first_calculator
    for name, old_tensor in first_owned.items():
        assert getattr(sampler, f"_{name}") is not old_tensor


def test_partial_load_failure_preserves_primary_and_releases_before_reload(monkeypatch, expect_error):
    allocations = []
    deallocated = []
    allocation_error = RuntimeError("seed allocation failed")
    cleanup_error = RuntimeError("local cleanup failed once")
    fail_allocation = True
    fail_cleanup = True

    def from_torch(source, **kwargs):
        if source == "seeds" and fail_allocation:
            raise allocation_error
        tensor = FakeTensor(f"{source}-{len(allocations)}")
        allocations.append(tensor)
        return tensor

    def deallocate(value):
        nonlocal fail_cleanup
        deallocated.append(value)
        if value.name.startswith("local-indices") and fail_cleanup:
            fail_cleanup = False
            raise cleanup_error

    monkeypatch.setattr(sampling_1d.ttnn, "Tensor", FakeTensor)
    monkeypatch.setattr(sampling_1d.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(sampling_1d.ttnn, "deallocate", deallocate)

    from models.common import utils as common_utils

    FakeLogProbsCalculator.instances = []
    monkeypatch.setattr(common_utils, "LogProbsCalculator", FakeLogProbsCalculator)

    borrowed_index_offsets = FakeTensor("borrowed-index-offsets")
    owned_specs = {
        "local_indices": _lazy_buffer("local-indices"),
        "seeds": _lazy_buffer("seeds"),
        "user_ids": _lazy_buffer("user-ids"),
    }
    sampler = object.__new__(Sampling1D)
    sampler.config = _sampler_config(borrowed_index_offsets, owned_specs)
    sampler._device_buffers_loaded = False

    with expect_error(RuntimeError, "seed allocation failed") as caught:
        sampler.load_device_buffers()

    assert caught.value is allocation_error
    assert cleanup_error in caught.value.cleanup_failures
    assert owned_specs["local_indices"]._value is not None
    assert not sampler._device_buffers_loaded
    assert borrowed_index_offsets not in deallocated

    sampler.release()
    assert owned_specs["local_indices"]._value is None

    fail_allocation = False
    sampler.load_device_buffers()
    assert sampler._device_buffers_loaded
    sampler.release()


def test_sampling_release_is_best_effort_and_retries_only_failed_buffer(monkeypatch, expect_error):
    attempts = []
    cleanup_error = RuntimeError("seed cleanup failed once")

    monkeypatch.setattr(sampling_1d.ttnn, "Tensor", FakeTensor)
    monkeypatch.setattr(
        sampling_1d.ttnn,
        "from_torch",
        lambda source, **kwargs: FakeTensor(source),
    )

    from models.common import utils as common_utils

    FakeLogProbsCalculator.instances = []
    monkeypatch.setattr(common_utils, "LogProbsCalculator", FakeLogProbsCalculator)

    borrowed_index_offsets = FakeTensor("borrowed-index-offsets")
    owned_specs = {
        "local_indices": _lazy_buffer("local-indices"),
        "seeds": _lazy_buffer("seeds"),
        "user_ids": _lazy_buffer("user-ids"),
    }
    sampler = object.__new__(Sampling1D)
    sampler.config = _sampler_config(borrowed_index_offsets, owned_specs)
    sampler._device_buffers_loaded = False
    sampler.load_device_buffers()
    failed = sampler._seeds

    def deallocate(value):
        attempts.append(value)
        if value is failed and attempts.count(value) == 1:
            raise cleanup_error

    monkeypatch.setattr(sampling_1d.ttnn, "deallocate", deallocate)

    with expect_error(RuntimeError, "seed cleanup failed once") as caught:
        sampler.release()

    assert caught.value is cleanup_error
    assert sampler._seeds is failed
    assert owned_specs["seeds"]._value is failed
    assert owned_specs["local_indices"]._value is None
    assert owned_specs["user_ids"]._value is None

    sampler.release()

    assert attempts.count(failed) == 2
    assert attempts.count(borrowed_index_offsets) == 0
    assert owned_specs["seeds"]._value is None


def test_log_probs_calculator_release_deallocates_unique_owned_tensors_once(monkeypatch):
    deallocated = []
    monkeypatch.setattr(sampling_1d.ttnn, "deallocate", deallocated.append)

    shared = FakeTensor("shared")
    tensors = {
        "global_max": shared,
        "global_exp_sum": FakeTensor("global-exp-sum"),
        "mask": FakeTensor("mask"),
        "output_tensor": shared,
        "topk_logprobs_output": FakeTensor("topk-logprobs"),
        "topk_indices_output": FakeTensor("topk-indices"),
    }
    calculator = object.__new__(LogProbsCalculator)
    for name, tensor in tensors.items():
        setattr(calculator, name, tensor)

    calculator.release()
    calculator.release()

    assert len(deallocated) == len({id(tensor) for tensor in tensors.values()})
    assert {id(tensor) for tensor in deallocated} == {id(tensor) for tensor in tensors.values()}
    assert all(getattr(calculator, name) is None for name in tensors)
