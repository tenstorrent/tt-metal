# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import Qwen38ShardedLogits
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNTextModelOutput


class _Tensor:
    def __init__(self, tensor_id: int) -> None:
        self.tensor_id = tensor_id


def _logits(tensor: _Tensor) -> Qwen38ShardedLogits:
    return Qwen38ShardedLogits(
        tensor=tensor,
        vocab_ranges=((0, 62080), (62080, 124160), (124160, 186240), (186240, 248320)),
        global_shape=(1, 1, 1, 248320),
    )


def _output(*, root=None, hidden=None, logits=None) -> Qwen38TTNNTextModelOutput:
    return Qwen38TTNNTextModelOutput(
        input_token_id=7,
        position=3,
        hyper_residual_sharded=root,
        hidden_sharded=hidden,
        logits=logits,
        greedy_token=None,
        state=object(),
        layer_aux=(),
    )


def test_hyper_root_transfer_updates_cleanup_ledger(monkeypatch):
    root, hidden, logit_tensor = _Tensor(1), _Tensor(2), _Tensor(3)
    output = _output(root=root, hidden=hidden, logits=_logits(logit_tensor))
    released = []
    monkeypatch.setattr(
        "models.demos.blackhole.qwen38_flash_next.ttnn.model.ttnn.deallocate",
        released.append,
    )

    assert output.take_hyper_residual() is root
    assert output.hyper_residual_sharded is None
    assert output.active
    output.release_tensors()

    assert released == [hidden, logit_tensor]
    assert not output.active


def test_logits_transfer_leaves_root_owned_and_never_releases_logits(monkeypatch):
    root, logit_tensor = _Tensor(11), _Tensor(12)
    logits = _logits(logit_tensor)
    output = _output(root=root, logits=logits)
    released = []
    monkeypatch.setattr(
        "models.demos.blackhole.qwen38_flash_next.ttnn.model.ttnn.deallocate",
        released.append,
    )

    assert output.take_logits() is logits
    assert output.logits is None
    output.release_tensors()

    assert released == [root]
    assert not output.active


def test_single_root_transfer_consumes_output_owner_without_deallocation(monkeypatch, expect_error):
    root = _Tensor(21)
    output = _output(root=root)
    released = []
    monkeypatch.setattr(
        "models.demos.blackhole.qwen38_flash_next.ttnn.model.ttnn.deallocate",
        released.append,
    )

    assert output.take_hyper_residual() is root
    assert not output.active
    assert released == []
    with expect_error(RuntimeError, "does not own"):
        output.take_hyper_residual()
    with expect_error(RuntimeError, "already released"):
        output.release_tensors()


def test_transfer_rejects_cross_slot_alias_without_mutating_owner(expect_error):
    alias = _Tensor(31)
    output = _output(root=alias, hidden=alias)

    with expect_error(RuntimeError, "aliased hyper-residual"):
        output.take_hyper_residual()

    assert output.hyper_residual_sharded is alias
    assert output.hidden_sharded is alias
    assert output.active
