# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig
from models.common.llm_runtime.paged_kv_cache import PagedKVCacheManager, PagedKVCacheState, torch_dtype_for_ttnn


class FakeMesh:
    def __init__(self, num_devices=2):
        self._num_devices = num_devices

    def get_num_devices(self):
        return self._num_devices


class FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    def volume(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self):
        return 1 if self.dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) else 2


class FakeModel:
    def __init__(self, dtypes=(ttnn.bfloat8_b, ttnn.bfloat8_b), *, bind_fails=False):
        mesh = FakeMesh(2)
        blocks = []
        for dtype in dtypes:
            attention = SimpleNamespace(
                n_kv_heads=8,
                head_dim=16,
                kv_cache_dtype=dtype,
                paged_attention_config=SimpleNamespace(block_size=32, max_num_blocks=8),
            )
            blocks.append(SimpleNamespace(attention_config=attention))
        self.config = SimpleNamespace(
            block_configs=blocks,
            n_layers=len(blocks),
            num_devices=2,
            mesh_device=mesh,
        )
        self.set_calls = []
        self.bound_cache = None
        self.bind_fails = bind_fails

    def set_kv_cache(self, cache):
        self.set_calls.append(cache)
        if cache is not None and self.bind_fails:
            raise RuntimeError("bind failed")
        self.bound_cache = cache


def cache_config(**overrides):
    values = {
        "block_size": 32,
        "max_num_blocks": 8,
        "dtype": ttnn.bfloat8_b,
    }
    values.update(overrides)
    return PagedKVCacheConfig(**values)


@pytest.fixture
def fake_allocator(monkeypatch):
    allocated = []
    deallocated = []

    def as_tensor(host_tensor, **kwargs):
        tensor = FakeTensor(host_tensor.shape, kwargs["dtype"])
        allocated.append((tensor, host_tensor, kwargs))
        return tensor

    monkeypatch.setattr(ttnn, "as_tensor", as_tensor)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: ("replicate", mesh))
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: deallocated.append(tensor))
    return allocated, deallocated


def test_unresolved_config_accepts_one_resolved_replacement(expect_error):
    manager = PagedKVCacheManager(FakeModel(), cache_config())

    assert manager.state == PagedKVCacheState.UNRESOLVED
    assert manager.capacity_tokens is None

    resolved = replace(manager.config, num_blocks=4)
    manager.configure(resolved)

    assert manager.state == PagedKVCacheState.CONFIGURED
    assert manager.config is resolved
    assert manager.num_blocks == 4
    assert manager.capacity_tokens == 128
    with expect_error(RuntimeError, "only once"):
        manager.configure(replace(resolved, num_blocks=5))


def test_resolved_replacement_may_change_only_num_blocks(expect_error):
    manager = PagedKVCacheManager(FakeModel(), cache_config())

    with expect_error(ValueError, "block_size changed"):
        manager.configure(cache_config(block_size=16, num_blocks=4))
    assert manager.state == PagedKVCacheState.UNRESOLVED


def test_pre_resolved_config_starts_configured_and_cannot_be_replaced(expect_error):
    manager = PagedKVCacheManager(FakeModel(), cache_config(num_blocks=4))

    assert manager.state == PagedKVCacheState.CONFIGURED
    with expect_error(RuntimeError, "only once"):
        manager.configure(cache_config(num_blocks=5))


def test_model_paged_attention_policy_and_uniform_dtype_are_validated(expect_error):
    model = FakeModel()
    model.config.block_configs[1].attention_config.paged_attention_config.block_size = 16
    with expect_error(ValueError, "block_size"):
        PagedKVCacheManager(model, cache_config())

    with expect_error(ValueError, "model-owned dtype"):
        PagedKVCacheManager(FakeModel(), cache_config(dtype=ttnn.bfloat16))


def test_vllm_torch_dtype_mapping_includes_quantized_surrogate(expect_error):
    manager = PagedKVCacheManager(FakeModel(), cache_config())

    assert torch_dtype_for_ttnn(ttnn.bfloat8_b) == torch.bfloat16
    manager.validate_vllm_cache_spec(block_size=32, dtype=torch.bfloat16, num_blocks=8)
    with expect_error(ValueError, "incompatible"):
        manager.validate_vllm_cache_spec(block_size=32, dtype=torch.float32)
    with expect_error(ValueError, "block_size"):
        manager.validate_vllm_cache_spec(block_size=16, dtype=torch.bfloat16)
    with expect_error(ValueError, "exceeds configured maximum"):
        manager.validate_vllm_cache_spec(block_size=32, dtype=torch.bfloat16, num_blocks=9)


def test_nonuniform_model_dtypes_remain_per_layer():
    manager = PagedKVCacheManager(
        FakeModel(dtypes=(ttnn.bfloat8_b, ttnn.bfloat16)),
        cache_config(dtype=ttnn.bfloat8_b),
    )

    assert manager.per_layer_dtypes == (ttnn.bfloat8_b, ttnn.bfloat16)
    manager.validate_vllm_cache_spec(block_size=32, dtype=torch.bfloat16)


def test_allocate_derives_shapes_and_dtypes_binds_exact_borrowed_handle(fake_allocator, expect_error):
    allocated, _ = fake_allocator
    model = FakeModel(dtypes=(ttnn.bfloat8_b, ttnn.bfloat16))
    manager = PagedKVCacheManager(model, cache_config(num_blocks=4))

    cache = manager.allocate()

    assert manager.state == PagedKVCacheState.BOUND
    assert cache is manager.bound_cache
    assert cache is model.bound_cache
    assert manager.cache_shapes == ((4, 4, 32, 16), (4, 4, 32, 16))
    assert manager.cache_shape == (4, 4, 32, 16)
    assert [entry[0].dtype for entry in allocated] == [
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.bfloat16,
        ttnn.bfloat16,
    ]
    assert all(tuple(entry[1].shape) == (4, 4, 32, 16) for entry in allocated)
    assert len({id(entry[1]) for entry in allocated}) == 1
    assert all(entry[2]["memory_config"] == ttnn.DRAM_MEMORY_CONFIG for entry in allocated)
    assert all(entry[2]["cache_file_name"] is None for entry in allocated)
    assert manager.allocated_bytes == 2 * (4 * 4 * 32 * 16) * (1 + 2)

    manager.validate_borrowed_handle(cache)
    with expect_error(ValueError, "exact manager-owned"):
        manager.validate_borrowed_handle([pair[:] for pair in cache])
    assert manager.bound_context.config is manager.config
    assert manager.bound_context.tensors[0][0] is cache[0][0]


def test_allocate_reuses_legacy_cache_files_when_dtype_is_unambiguous(fake_allocator, tmp_path):
    allocated, _ = fake_allocator
    model = FakeModel()
    model.model_args = SimpleNamespace(model_cache_path=tmp_path)
    manager = PagedKVCacheManager(model, cache_config(num_blocks=4))

    manager.allocate()

    shape = (4, 4, 32, 16)
    expected = [
        tmp_path / f"empty_kcache_paged_attention{shape}",
        tmp_path / f"empty_vcache_paged_attention{shape}",
    ]
    assert [entry[2]["cache_file_name"] for entry in allocated] == expected * 2
    assert len({id(entry[1]) for entry in allocated}) == 1


def test_allocate_avoids_cache_file_collision_for_nonuniform_device_dtypes(fake_allocator, tmp_path):
    allocated, _ = fake_allocator
    model = FakeModel(dtypes=(ttnn.bfloat8_b, ttnn.bfloat16))
    model.model_args = SimpleNamespace(model_cache_path=tmp_path)
    manager = PagedKVCacheManager(model, cache_config(num_blocks=4))

    manager.allocate()

    assert all(entry[2]["cache_file_name"] is None for entry in allocated)


def test_allocate_accounts_for_packed_device_buffers_without_element_size(monkeypatch):
    class FakePackedTensor(FakeTensor):
        def element_size(self):
            raise ValueError("packed dtype has no datum size")

        def buffer_page_size(self):
            return 1088

        def buffer_num_pages(self):
            return self.volume() // 1024

    monkeypatch.setattr(
        ttnn,
        "as_tensor",
        lambda host_tensor, **kwargs: FakePackedTensor(host_tensor.shape, kwargs["dtype"]),
    )
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    manager = PagedKVCacheManager(FakeModel(), cache_config(num_blocks=4))

    manager.allocate()

    assert manager.allocated_bytes == 4 * 1088 * 8


def test_allocation_requires_resolved_capacity_and_happens_once(fake_allocator, expect_error):
    manager = PagedKVCacheManager(FakeModel(), cache_config())
    with expect_error(RuntimeError, "must be resolved"):
        manager.allocate()

    manager.configure(replace(manager.config, num_blocks=4))
    cache = manager.allocate()
    with expect_error(RuntimeError, "already been allocated"):
        manager.allocate()
    assert cache is manager.bound_cache


def test_partial_allocation_failure_deallocates_created_tensors(monkeypatch, expect_error):
    first = FakeTensor((4, 4, 32, 16), ttnn.bfloat8_b)
    calls = 0
    deallocated = []

    def fail_second_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("allocation failed")
        return first

    monkeypatch.setattr(ttnn, "as_tensor", fail_second_allocation)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: deallocated.append(tensor))
    model = FakeModel()
    manager = PagedKVCacheManager(model, cache_config(num_blocks=4))

    with expect_error(RuntimeError, "allocation failed"):
        manager.allocate()

    assert deallocated == [first]
    assert manager.state == PagedKVCacheState.CONFIGURED
    assert manager.bound_cache is None
    assert model.set_calls == []


def test_bind_failure_unbinds_then_deallocates_all_tensors(fake_allocator, expect_error):
    allocated, deallocated = fake_allocator
    model = FakeModel(bind_fails=True)
    manager = PagedKVCacheManager(model, cache_config(num_blocks=4))

    with expect_error(RuntimeError, "bind failed"):
        manager.allocate()

    assert model.set_calls[-1] is None
    assert deallocated == [entry[0] for entry in reversed(allocated)]
    assert manager.state == PagedKVCacheState.CONFIGURED
    assert manager.bound_cache is None


def test_release_unbinds_before_deallocating_and_is_idempotent(monkeypatch, expect_error):
    operations = []
    model = FakeModel()
    original_set = model.set_kv_cache

    def set_kv_cache(cache):
        operations.append(("bind", cache))
        original_set(cache)

    model.set_kv_cache = set_kv_cache
    monkeypatch.setattr(
        ttnn,
        "as_tensor",
        lambda host_tensor, **kwargs: FakeTensor(host_tensor.shape, kwargs["dtype"]),
    )
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: operations.append(("deallocate", tensor)))
    manager = PagedKVCacheManager(model, cache_config(num_blocks=4))
    cache = manager.allocate()
    operations.clear()

    manager.release()

    assert operations[0] == ("bind", None)
    assert [operation[1] for operation in operations[1:]] == [tensor for pair in cache for tensor in pair]
    assert manager.state == PagedKVCacheState.RELEASED
    assert manager.bound_cache is None
    assert manager.bound_context is None
    assert manager.allocated_bytes == 0

    manager.release()
    assert len(operations) == 5
    with expect_error(RuntimeError, "terminal"):
        manager.allocate()


def test_borrowed_handle_mutation_cannot_redirect_owned_tensor_release(fake_allocator, expect_error):
    allocated, deallocated = fake_allocator
    model = FakeModel()
    manager = PagedKVCacheManager(model, cache_config(num_blocks=4))
    cache = manager.allocate()
    owned_tensors = [entry[0] for entry in allocated]
    replacement = FakeTensor(cache[0][0].shape, cache[0][0].dtype)

    cache[0][0] = replacement

    with expect_error(ValueError, "exact manager-owned K/V tensors"):
        manager.validate_borrowed_handle(cache)

    manager.release()

    assert deallocated == owned_tensors
    assert replacement not in deallocated
    assert manager.state == PagedKVCacheState.RELEASED


def test_release_failure_retains_only_failed_tensor_and_retries(monkeypatch, expect_error):
    tensors = []

    def as_tensor(host_tensor, **kwargs):
        tensor = FakeTensor(host_tensor.shape, kwargs["dtype"])
        tensors.append(tensor)
        return tensor

    monkeypatch.setattr(ttnn, "as_tensor", as_tensor)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    model = FakeModel()
    manager = PagedKVCacheManager(model, cache_config(num_blocks=4))
    manager.allocate()
    failed_tensor = tensors[0]
    attempts = []

    def fail_once(tensor):
        attempts.append(tensor)
        if tensor is failed_tensor and attempts.count(tensor) == 1:
            raise RuntimeError("deallocate failed")

    monkeypatch.setattr(ttnn, "deallocate", fail_once)

    with expect_error(RuntimeError, "Failed to deallocate 1"):
        manager.release()

    assert manager.state == PagedKVCacheState.BOUND
    assert manager.bound_cache is None
    assert manager.bound_context is None
    assert model.bound_cache is None
    assert manager.allocated_bytes == failed_tensor.volume() * failed_tensor.element_size()
    assert attempts == tensors

    manager.release()

    assert attempts.count(failed_tensor) == 2
    assert all(attempts.count(tensor) == 1 for tensor in tensors[1:])
    assert manager.state == PagedKVCacheState.RELEASED
    assert manager.allocated_bytes == 0


def test_partial_allocation_cleanup_failure_preserves_tensor_for_release_retry(monkeypatch, expect_error):
    tensor = FakeTensor((4, 4, 32, 16), ttnn.bfloat8_b)
    allocation_calls = 0
    deallocation_calls = 0

    def fail_second_allocation(*args, **kwargs):
        nonlocal allocation_calls
        allocation_calls += 1
        if allocation_calls == 2:
            raise RuntimeError("allocation failed")
        return tensor

    def fail_first_deallocation(value):
        nonlocal deallocation_calls
        deallocation_calls += 1
        if deallocation_calls == 1:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(ttnn, "as_tensor", fail_second_allocation)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "deallocate", fail_first_deallocation)
    manager = PagedKVCacheManager(FakeModel(), cache_config(num_blocks=4))

    with expect_error(RuntimeError, "allocation failed") as exc_info:
        manager.allocate()

    assert [str(error) for error in exc_info.value.cleanup_failures] == ["cleanup failed"]
    assert manager.state == PagedKVCacheState.CONFIGURED
    assert manager.allocated_bytes == tensor.volume() * tensor.element_size()

    manager.release()

    assert deallocation_calls == 2
    assert manager.state == PagedKVCacheState.RELEASED
    assert manager.allocated_bytes == 0


def test_release_before_allocation_is_terminal_and_idempotent():
    manager = PagedKVCacheManager(FakeModel(), cache_config())

    manager.release()
    manager.release()

    assert manager.state == PagedKVCacheState.RELEASED
    assert manager.bound_cache is None
