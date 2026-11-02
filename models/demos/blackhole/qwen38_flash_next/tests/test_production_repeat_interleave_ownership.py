# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNModelPoisonedError, Qwen38TTNNTextModel


class _Tensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tensor_id = id(self)


class _Embedding:
    def __init__(self, token: _Tensor, hidden: _Tensor, events: list[str], *, fail: bool, poisoned: bool) -> None:
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


def _embed_residual_fakes(
    monkeypatch, *, fail_embedding=False, poisoned_embedding=False, fail_repeat=False, fail_sync=False
):
    """A bare text model whose ``_embed_residual`` runs on a fake embedding and ttnn; ``events`` logs the calls,
    ``released`` the deallocated tensors and ``locals`` each mesh tensor's four device-local tensors."""
    events: list[str] = []
    released: list[_Tensor] = []
    token, hidden, residual = _Tensor("token"), _Tensor("hidden"), _Tensor("residual")
    owner = object.__new__(Qwen38TTNNTextModel)
    owner.mesh_device = object()
    owner.model_io = SimpleNamespace(
        embedding=_Embedding(token, hidden, events, fail=fail_embedding, poisoned=poisoned_embedding)
    )
    owner._poisoned_error = None
    owner._poisoned_device_owners = []
    owner._active_snapshot = None
    owner._validate_hidden = lambda tensor, **_kwargs: events.append(f"validate-{tensor.name}")
    owner._validate_residual = lambda tensor, **_kwargs: events.append(f"validate-{tensor.name}")

    def device_locals(tensor):
        return tuple(_Tensor(f"{tensor.name}-local-{index}") for index in range(4))

    locals_by_mesh = {tensor: device_locals(tensor) for tensor in (token, hidden, residual)}

    def repeat_interleave(tensor, *, repeats, dim, memory_config):
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
        return locals_by_mesh.setdefault(tensor, device_locals(tensor))

    def deallocate(tensor):
        events.append(f"deallocate-{tensor.name}")
        released.append(tensor)

    monkeypatch.setattr(model_module.ttnn, "repeat_interleave", repeat_interleave)
    monkeypatch.setattr(model_module.ttnn, "get_device_tensors", get_device_tensors)
    monkeypatch.setattr(model_module.ttnn, "synchronize_device", synchronize_device)
    monkeypatch.setattr(model_module.ttnn, "deallocate", deallocate)
    return SimpleNamespace(
        owner=owner,
        token=token,
        hidden=hidden,
        residual=residual,
        events=events,
        released=released,
        locals=locals_by_mesh,
    )


@pytest.mark.parametrize("fail_sync", [False, True], ids=["success", "synchronize would fail"])
def test_success_enqueues_repeat_before_releasing_producer_owners_and_is_trace_capturable(
    monkeypatch, fail_sync: bool
) -> None:
    """The repeat is enqueued before the producers are released and nothing synchronizes, so the path is trace
    capturable even where a synchronize would fail."""
    fakes = _embed_residual_fakes(monkeypatch, fail_sync=fail_sync)

    assert fakes.owner._embed_residual(object()) is fakes.residual

    assert fakes.events == [
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
    assert fakes.released == [fakes.token, fakes.hidden]
    assert fakes.owner._poisoned_device_owners == []
    assert not fakes.owner.poisoned


@pytest.mark.parametrize("failure", ["repeat enqueue", "post-enqueue contract"])
def test_repeat_failure_retains_the_live_owners_without_cleanup_retry_or_fence(
    monkeypatch, expect_error, failure: str
) -> None:
    """A repeat enqueue failure retains the sources; a residual contract failure after the enqueue retains the output
    too; neither releases, retries the repeat or synchronizes."""
    fakes = _embed_residual_fakes(monkeypatch, fail_repeat=failure == "repeat enqueue")
    if failure == "post-enqueue contract":
        fakes.owner._validate_residual = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic residual contract failure")
        )

    with expect_error(Qwen38TTNNModelPoisonedError, "initial residual repeat-interleave"):
        fakes.owner._embed_residual(object())

    owners = (fakes.token, fakes.hidden) + ((fakes.residual,) if failure == "post-enqueue contract" else ())
    assert fakes.released == []
    assert fakes.owner._poisoned_device_owners == [
        item for tensor in owners for item in (tensor, *fakes.locals[tensor])
    ]
    assert fakes.events.count("repeat-hidden-4-1") == 1
    assert "synchronize" not in fakes.events


def test_pre_repeat_embedding_failure_keeps_prior_cleanup_semantics(monkeypatch, expect_error) -> None:
    fakes = _embed_residual_fakes(monkeypatch, fail_embedding=True)

    with expect_error(RuntimeError, "synthetic embedding failure"):
        fakes.owner._embed_residual(object())

    assert fakes.released == [fakes.token]
    assert fakes.owner._poisoned_device_owners == []
    assert not fakes.owner.poisoned
    assert not any(event.startswith("repeat-") for event in fakes.events)
    assert "synchronize" not in fakes.events


def test_async_embedding_failure_retains_uploaded_token_and_poisons_model(monkeypatch, expect_error) -> None:
    fakes = _embed_residual_fakes(monkeypatch, fail_embedding=True, poisoned_embedding=True)
    (embedding_owner,) = fakes.owner.model_io.embedding.poisoned_device_owners

    with expect_error(Qwen38TTNNModelPoisonedError, "token embedding asynchronous chain"):
        fakes.owner._embed_residual(object())

    assert fakes.released == []
    assert fakes.owner._poisoned_device_owners == [embedding_owner, fakes.token, *fakes.locals[fakes.token]]
    assert fakes.owner.poisoned
    assert not any(event.startswith("repeat-") for event in fakes.events)
