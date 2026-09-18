from types import SimpleNamespace

import pytest
import torch

from models.common.sampling.full_vocab_device import sample_unrestricted_top_p_one


class TorchOps:
    """Tiny operation-compatible oracle; no host-readback method exists."""

    float32 = torch.float32

    @staticmethod
    def transpose(tensor, first, second):
        return torch.transpose(tensor, first, second)

    @staticmethod
    def slice(tensor, begin, end):
        return tensor[tuple(slice(start, stop) for start, stop in zip(begin, end))]

    @staticmethod
    def multiply(left, right):
        return torch.multiply(left, right)

    @staticmethod
    def concat(tensors, dim):
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def add(left, right, dtype=None):
        result = torch.add(left, right)
        return result.to(dtype) if dtype is not None else result

    @staticmethod
    def max(tensor, dim, keepdim):
        return torch.max(tensor, dim=dim, keepdim=keepdim).values

    @staticmethod
    def subtract(left, right):
        return torch.subtract(left, right)

    @staticmethod
    def exp(tensor):
        return torch.exp(tensor)

    @staticmethod
    def uniform(tensor, low, high, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.rand(tensor.shape, generator=generator, dtype=torch.float32) * (high - low) + low

    @staticmethod
    def gt(left, right):
        return torch.gt(left, right)

    @staticmethod
    def argmax(tensor, dim, keepdim):
        return torch.argmax(tensor.to(torch.int32), dim=dim, keepdim=keepdim)


def _run(logits, *, seeds, active=None, trace_enabled=False, vocab_size=None):
    batch, width = logits.shape[2:]
    return sample_unrestricted_top_p_one(
        logits,
        inverse_temperature=torch.ones(1, 1, batch, 1),
        row_scratch=[torch.zeros(1, 1, 1, 1) for _ in range(batch)],
        seed_values=seeds,
        active_rows=[True] * batch if active is None else active,
        vocab_size=width if vocab_size is None else vocab_size,
        ops=TorchOps,
        trace_enabled=trace_enabled,
    )


def test_device_categorical_returns_one_token_per_row_without_host_readback():
    logits = torch.linspace(-3.0, 3.0, 3 * 32).reshape(1, 1, 3, 32)
    result = _run(logits, seeds=[11, 22, 33])

    assert result.token_ids.shape == (1, 1, 3, 1)
    assert result.valid_distribution.shape == (1, 1, 3, 1)
    assert result.valid_distribution.all()
    assert ((0 <= result.token_ids) & (result.token_ids < 32)).all()
    assert len(result.owned_tensors) > 0


def test_device_categorical_is_reproducible_per_row_and_seed_changes_only_that_row():
    logits = torch.zeros(1, 1, 3, 32)
    first = _run(logits, seeds=[101, 202, 303]).token_ids
    repeated = _run(logits, seeds=[101, 202, 303]).token_ids
    changed = _run(logits, seeds=[101, 999, 303]).token_ids

    assert torch.equal(first, repeated)
    assert first[0, 0, 0, 0] == changed[0, 0, 0, 0]
    assert first[0, 0, 2, 0] == changed[0, 0, 2, 0]


def test_inactive_row_requires_skip_sentinel_and_does_not_call_uniform(monkeypatch):
    calls = []

    def uniform(tensor, low, high, seed):
        calls.append(seed)
        return TorchOps.uniform(tensor, low, high, seed)

    ops = SimpleNamespace(**{name: getattr(TorchOps, name) for name in dir(TorchOps) if not name.startswith("_")})
    ops.uniform = uniform
    result = sample_unrestricted_top_p_one(
        torch.zeros(1, 1, 2, 32),
        inverse_temperature=torch.ones(1, 1, 2, 1),
        row_scratch=[torch.zeros(1, 1, 1, 1) for _ in range(2)],
        seed_values=[7, 2**32 - 1],
        active_rows=[True, False],
        vocab_size=32,
        ops=ops,
    )
    assert calls == [7]
    assert result.token_ids.shape == (1, 1, 2, 1)


def test_padded_tail_cannot_be_selected_when_producer_masks_it():
    logits = torch.zeros(1, 1, 2, 64)
    logits[..., 37:] = -torch.inf
    result = _run(logits, seeds=[17, 29], vocab_size=37)
    assert (result.token_ids < 37).all()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"trace_enabled": True}, "not trace-safe"),
        ({"seeds": [0]}, "excluding zero"),
        ({"seeds": [2**32 - 1]}, "excluding zero"),
    ],
)
def test_device_categorical_fails_closed(kwargs, message):
    values = {"seeds": [7], "trace_enabled": False}
    values.update(kwargs)
    with pytest.raises((ValueError, RuntimeError), match=message):
        _run(torch.zeros(1, 1, 1, 32), **values)
