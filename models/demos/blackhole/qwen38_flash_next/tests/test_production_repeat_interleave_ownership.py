# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNModelPoisonedError, Qwen38TTNNTextModel


class _Tensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tensor_id = id(self)


def _local_names(tensor: _Tensor) -> list[str]:
    return [f"{tensor.name}-local-{index}" for index in range(4)]


class _Embedding:
    def __init__(
        self,
        token: _Tensor,
        hidden: _Tensor,
        events: list[str],
        *,
        fail: bool = False,
        poisoned: bool = False,
    ) -> None:
        self.token = token
        self.hidden = hidden
        self.events = events
        self.fail = fail
        self.poisoned = poisoned
        self.poisoned_device_owners = (_Tensor("embedding-async-owner"),) if poisoned else ()

    def upload_tokens(self, _host_token):
        self.events.append("upload")
        return SimpleNamespace(tensor=self.token)

    def __call__(self, _validated):
        self.events.append("embedding")
        if self.fail:
            raise RuntimeError("synthetic embedding failure")
        return self.hidden


def _owner(events: list[str], *, fail_embedding: bool = False, poisoned_embedding: bool = False):
    token = _Tensor("token")
    hidden = _Tensor("hidden")
    residual = _Tensor("residual")
    owner = object.__new__(Qwen38TTNNTextModel)
    owner.mesh_device = object()
    owner.model_io = SimpleNamespace(
        embedding=_Embedding(token, hidden, events, fail=fail_embedding, poisoned=poisoned_embedding),
    )
    owner._poisoned_error = None
    owner._poisoned_device_owners = []
    owner._active_snapshot = None
    owner._validate_hidden = lambda tensor, **_kwargs: events.append(f"validate-{tensor.name}")
    owner._validate_residual = lambda tensor, **_kwargs: events.append(f"validate-{tensor.name}")
    return owner, token, hidden, residual


def _install_runtime(
    monkeypatch,
    events: list[str],
    token: _Tensor,
    hidden: _Tensor,
    residual: _Tensor,
    *,
    fail_repeat=False,
    fail_sync=False,
):
    released: list[_Tensor] = []
    locals_by_mesh = {
        tensor: tuple(_Tensor(name) for name in _local_names(tensor)) for tensor in (token, hidden, residual)
    }

    def repeat_interleave(tensor, *, repeats, dim, memory_config):
        del memory_config
        events.append(f"repeat-{tensor.name}-{repeats}-{dim}")
        if fail_repeat:
            raise RuntimeError("synthetic repeat-interleave enqueue failure")
        return residual

    def synchronize_device(_mesh):
        events.append("synchronize")
        if fail_sync:
            raise RuntimeError("synthetic repeat-interleave synchronize failure")

    def get_device_tensors(tensor):
        events.append(f"locals-{tensor.name}")
        return locals_by_mesh.setdefault(tensor, tuple(_Tensor(name) for name in _local_names(tensor)))

    def deallocate(tensor):
        events.append(f"deallocate-{tensor.name}")
        released.append(tensor)

    monkeypatch.setattr(model_module.ttnn, "repeat_interleave", repeat_interleave)
    monkeypatch.setattr(model_module.ttnn, "get_device_tensors", get_device_tensors)
    monkeypatch.setattr(model_module.ttnn, "synchronize_device", synchronize_device)
    monkeypatch.setattr(model_module.ttnn, "deallocate", deallocate)
    return released, locals_by_mesh


def test_success_enqueues_repeat_before_releasing_producer_owners(monkeypatch) -> None:
    events: list[str] = []
    owner, token, hidden, residual = _owner(events)
    released, _locals = _install_runtime(monkeypatch, events, token, hidden, residual)

    assert owner._embed_residual(object()) is residual

    assert events == [
        "upload",
        "embedding",
        "validate-hidden",
        "locals-token",
        "locals-hidden",
        "repeat-hidden-4-1",
        "locals-residual",
        "validate-residual",
        "deallocate-token",
        "deallocate-hidden",
    ]
    assert released == [token, hidden]
    assert owner._poisoned_device_owners == []
    assert not owner.poisoned


def test_success_is_trace_capturable_even_if_synchronize_would_fail(monkeypatch) -> None:
    events: list[str] = []
    owner, token, hidden, residual = _owner(events)
    released, locals_by_mesh = _install_runtime(monkeypatch, events, token, hidden, residual, fail_sync=True)

    assert owner._embed_residual(object()) is residual

    assert "synchronize" not in events
    assert released == [token, hidden]
    assert owner._poisoned_device_owners == []
    assert not owner.poisoned


def test_repeat_enqueue_failure_retains_sources_without_cleanup_or_retry(monkeypatch, expect_error) -> None:
    events: list[str] = []
    owner, token, hidden, residual = _owner(events)
    released, locals_by_mesh = _install_runtime(monkeypatch, events, token, hidden, residual, fail_repeat=True)

    with expect_error(Qwen38TTNNModelPoisonedError, "initial residual repeat-interleave"):
        owner._embed_residual(object())

    assert released == []
    assert owner._poisoned_device_owners == [token, *locals_by_mesh[token], hidden, *locals_by_mesh[hidden]]
    assert events.count("repeat-hidden-4-1") == 1
    assert "synchronize" not in events


def test_post_enqueue_contract_failure_retains_output_and_skips_fence(monkeypatch, expect_error) -> None:
    events: list[str] = []
    owner, token, hidden, residual = _owner(events)
    owner._validate_residual = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic residual contract failure")
    )
    released, locals_by_mesh = _install_runtime(monkeypatch, events, token, hidden, residual)

    with expect_error(Qwen38TTNNModelPoisonedError, "initial residual repeat-interleave"):
        owner._embed_residual(object())

    assert released == []
    assert owner._poisoned_device_owners == [
        token,
        *locals_by_mesh[token],
        hidden,
        *locals_by_mesh[hidden],
        residual,
        *locals_by_mesh[residual],
    ]
    assert "synchronize" not in events


def test_pre_repeat_embedding_failure_keeps_prior_cleanup_semantics(monkeypatch, expect_error) -> None:
    events: list[str] = []
    owner, token, hidden, residual = _owner(events, fail_embedding=True)
    released, _locals = _install_runtime(monkeypatch, events, token, hidden, residual)

    with expect_error(RuntimeError, "synthetic embedding failure"):
        owner._embed_residual(object())

    assert released == [token]
    assert owner._poisoned_device_owners == []
    assert not owner.poisoned
    assert not any(event.startswith("repeat-") for event in events)
    assert "synchronize" not in events


def test_async_embedding_failure_retains_uploaded_token_and_poisons_model(monkeypatch, expect_error) -> None:
    events: list[str] = []
    owner, token, hidden, residual = _owner(events, fail_embedding=True, poisoned_embedding=True)
    released, locals_by_mesh = _install_runtime(monkeypatch, events, token, hidden, residual)
    embedding_owner = owner.model_io.embedding.poisoned_device_owners[0]

    with expect_error(Qwen38TTNNModelPoisonedError, "token embedding asynchronous chain"):
        owner._embed_residual(object())

    assert released == []
    assert owner._poisoned_device_owners == [embedding_owner, token, *locals_by_mesh[token]]
    assert owner.poisoned
    assert not any(event.startswith("repeat-") for event in events)
