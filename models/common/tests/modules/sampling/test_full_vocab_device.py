from types import SimpleNamespace

import pytest
import torch

from models.common.sampling.full_vocab_device import (
    _release_owned,
    _normalize_gather_index_layout,
    hierarchical_stable_topk,
    merge_unrestricted_rows,
    plan_exact_nucleus_candidates,
    sample_unrestricted_nucleus,
    sample_unrestricted_top_p_one,
)
import models.common.sampling.full_vocab_device as full_vocab_device


class TorchOps:
    """Tiny operation-compatible oracle; no host-readback method exists."""

    float32 = torch.float32
    bfloat16 = torch.bfloat16
    uint16 = torch.uint16
    uint32 = torch.uint32

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
    def zeros_like(tensor):
        return torch.zeros_like(tensor)

    @staticmethod
    def typecast(tensor, dtype):
        return tensor.to(dtype)

    @staticmethod
    def concat(tensors, dim):
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def add(left, right, dtype=None):
        # PyTorch's CPU backend lacks uint32 add, while the public TTNN op
        # supports an explicitly typed uint32 result for global token IDs.
        arithmetic_left = left.to(torch.int64) if left.dtype in (torch.uint16, torch.uint32) else left
        result = torch.add(arithmetic_left, right)
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
    def lt(left, right):
        arithmetic_left = left.to(torch.int64) if left.dtype in (torch.uint16, torch.uint32) else left
        return torch.lt(arithmetic_left, right)

    @staticmethod
    def isfinite(tensor):
        return torch.isfinite(tensor)

    @staticmethod
    def logical_and(left, right):
        return torch.logical_and(left, right)

    @staticmethod
    def logical_not(tensor):
        return torch.logical_not(tensor)

    @staticmethod
    def argmax(tensor, dim, keepdim):
        return torch.argmax(tensor.to(torch.int32), dim=dim, keepdim=keepdim)

    @staticmethod
    def reshape(tensor, shape):
        return torch.reshape(tensor, shape)

    @staticmethod
    def to_layout(tensor, layout):
        assert layout == tensor.layout
        return tensor.clone()

    @staticmethod
    def where(condition, left, right):
        return torch.where(condition, left, right)

    @staticmethod
    def topk(tensor, k, dim, largest, sorted, stable):
        assert stable
        # torch.argsort(stable=True) pins the nucleus tie order explicitly.
        order = torch.argsort(tensor, dim=dim, descending=largest, stable=True)
        indices = order.narrow(dim, 0, k)
        return torch.gather(tensor, dim, indices), indices.to(torch.uint16)

    @staticmethod
    def gather(tensor, dim, index):
        return torch.gather(tensor, dim, index.to(torch.int64))


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


def test_device_categorical_forces_fp32_probability_math_from_bfloat16_logits():
    logits = torch.linspace(-30.0, 30.0, 32, dtype=torch.bfloat16).reshape(1, 1, 1, 32)
    result = _run(logits, seeds=[17])
    assert result.valid_distribution.all()
    assert result.token_ids.dtype == torch.int64


@pytest.mark.parametrize(
    "bad_logits",
    [
        torch.full((1, 1, 1, 32), -torch.inf),
        torch.full((1, 1, 1, 32), torch.inf),
        torch.full((1, 1, 1, 32), torch.nan),
    ],
)
def test_device_categorical_marks_nonfinite_distribution_invalid(bad_logits):
    result = _run(bad_logits, seeds=[17])
    assert not result.valid_distribution.any()


def test_only_last_valid_token_is_selected_and_tail_stays_masked():
    logits = torch.full((1, 1, 1, 64), -torch.inf)
    logits[..., 36] = 0.0
    result = _run(logits, seeds=[17], vocab_size=37)
    assert result.valid_distribution.all()
    assert result.token_ids.item() == 36


def test_inactive_nan_scratch_is_zeroed_without_nan_propagation():
    result = sample_unrestricted_top_p_one(
        torch.zeros(1, 1, 2, 32),
        inverse_temperature=torch.ones(1, 1, 2, 1),
        row_scratch=[torch.zeros(1, 1, 1, 1), torch.full((1, 1, 1, 1), torch.nan)],
        seed_values=[7, 2**32 - 1],
        active_rows=[True, False],
        vocab_size=32,
        ops=TorchOps,
    )
    assert result.valid_distribution.all()


def test_active_uniform_scratch_is_borrowed_and_reusable_across_steps():
    scratch = torch.zeros(1, 1, 1, 1)
    first = sample_unrestricted_top_p_one(
        torch.zeros(1, 1, 1, 32),
        inverse_temperature=torch.ones(1, 1, 1, 1),
        row_scratch=[scratch],
        seed_values=[7],
        active_rows=[True],
        vocab_size=32,
        ops=TorchOps,
    )
    second = sample_unrestricted_top_p_one(
        torch.zeros(1, 1, 1, 32),
        inverse_temperature=torch.ones(1, 1, 1, 1),
        row_scratch=[scratch],
        seed_values=[8],
        active_rows=[True],
        vocab_size=32,
        ops=TorchOps,
    )
    assert all(tensor is not scratch for tensor in first.owned_tensors)
    assert all(tensor is not scratch for tensor in second.owned_tensors)
    assert first.valid_distribution.all() and second.valid_distribution.all()


def test_release_owned_is_exact_and_does_not_suppress_release_errors():
    class Allocation:
        def __init__(self, allocated=True):
            self.allocated = allocated

        def is_allocated(self):
            return self.allocated

    released = []

    def deallocate(tensor):
        released.append(tensor)
        tensor.allocated = False

    ops = SimpleNamespace(deallocate=deallocate)
    first = Allocation()
    already_free = Allocation(allocated=False)
    _release_owned([first, first, already_free], ops=ops)
    assert released == [first]

    def failing_deallocate(_tensor):
        raise RuntimeError("ownership failure")

    with pytest.raises(RuntimeError, match="ownership failure"):
        _release_owned([Allocation()], ops=SimpleNamespace(deallocate=failing_deallocate))


def test_mixed_merge_preserves_native_rows_and_uses_invalid_sentinel():
    categorical = _run(torch.zeros(1, 1, 4, 32), seeds=[7, 8, 9, 10])
    # Inject one invalid unrestricted row to exercise the device sentinel.
    injected_valid = torch.tensor([[[[True], [False], [True], [True]]]])
    categorical = type(categorical)(
        categorical.token_ids,
        injected_valid,
        (*categorical.owned_tensors, injected_valid),
    )
    native = torch.tensor([[[[10, 11, 12, 13]]]], dtype=torch.int64)
    merged = merge_unrestricted_rows(
        categorical,
        native,
        unrestricted_selector=torch.tensor([[[[0, 1, 1, 0]]]], dtype=torch.int32),
        invalid_token_ids=torch.full((1, 1, 1, 4), 32, dtype=torch.int64),
        ops=TorchOps,
    )
    assert merged.token_ids.tolist() == [[[[10, 32, int(categorical.token_ids[0, 0, 2, 0]), 13]]]]
    # reshape views are not recorded as independent allocation owners.
    assert sum(tensor is categorical.token_ids for tensor in merged.owned_tensors) == 1
    assert sum(tensor is categorical.valid_distribution for tensor in merged.owned_tensors) == 1


def test_mixed_merge_releases_only_new_intermediates_on_second_add_failure(monkeypatch):
    categorical = _run(torch.zeros(1, 1, 4, 32), seeds=[7, 8, 9, 10])
    native = torch.tensor([[[[10, 11, 12, 13]]]], dtype=torch.int64)
    released = []
    monkeypatch.setattr(full_vocab_device, "_release_owned", lambda tensors, *, ops: released.extend(tensors))

    class FailSecondAddOps(TorchOps):
        add_calls = 0

        @classmethod
        def add(cls, left, right, dtype=None):
            cls.add_calls += 1
            if cls.add_calls == 2:
                raise RuntimeError("injected second add failure")
            return super().add(left, right, dtype=dtype)

        @staticmethod
        def gt(tensor, value):
            return torch.gt(tensor, value)

    with pytest.raises(RuntimeError, match="injected second add failure"):
        merge_unrestricted_rows(
            categorical,
            native,
            unrestricted_selector=torch.tensor([[[[0, 1, 1, 0]]]], dtype=torch.int32),
            invalid_token_ids=torch.full((1, 1, 1, 4), 32, dtype=torch.int64),
            ops=FailSecondAddOps,
        )

    assert released
    assert not any(tensor is owned for tensor in released for owned in categorical.owned_tensors)


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


def test_exact_nucleus_plan_uses_distribution_independent_mass_bound():
    plan = plan_exact_nucleus_candidates(
        0.01,
        vocab_size=201088,
        public_topk_max_candidates=2048,
    )
    assert plan.candidate_count == 2016
    assert plan.maximum_supported_top_p == pytest.approx(2016 / 201088)


def test_exact_nucleus_plan_rejects_domain_beyond_public_topk_envelope():
    with pytest.raises(ValueError, match="exceeds the exact public top-k nucleus envelope"):
        plan_exact_nucleus_candidates(
            0.95,
            vocab_size=201088,
            public_topk_max_candidates=2048,
        )


@pytest.mark.parametrize("width,k,max_local_width", [(96, 16, 32), (224, 32, 64)])
def test_hierarchical_stable_topk_matches_global_stable_oracle(width, k, max_local_width):
    logits = torch.arange(width, dtype=torch.float32).remainder(7).reshape(1, 1, 1, width)
    result = hierarchical_stable_topk(
        logits,
        k=k,
        max_local_width=max_local_width,
        ops=TorchOps,
    )
    expected_ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[..., :k]
    expected_values = torch.gather(logits, -1, expected_ids)
    assert torch.equal(result.global_indices.to(torch.int64), expected_ids)
    assert torch.equal(result.values, expected_values)
    # Equal-value ties cross both local and recursive merge boundaries; the
    # resulting IDs remain globally ascending within each value group.
    for value in torch.unique(result.values):
        ids = result.global_indices[result.values == value]
        assert torch.equal(ids, torch.sort(ids).values)


def test_hierarchical_stable_topk_preserves_global_ids_and_masked_tail():
    logits = torch.zeros(1, 1, 2, 160)
    logits[..., 129:] = -torch.inf
    logits[0, 0, 0, 97] = 11
    logits[0, 0, 1, 128] = 12
    result = hierarchical_stable_topk(
        logits,
        k=32,
        max_local_width=64,
        ops=TorchOps,
    )
    assert result.global_indices[0, 0, 0, 0] == 97
    assert result.global_indices[0, 0, 1, 0] == 128
    assert (result.global_indices.to(torch.int64) < 129).all()


def test_hierarchical_stable_topk_rejects_unqualified_chunk_envelope():
    with pytest.raises(ValueError, match="does not fit"):
        hierarchical_stable_topk(
            torch.zeros(1, 1, 1, 96),
            k=48,
            max_local_width=32,
            ops=TorchOps,
        )

    with pytest.raises(ValueError, match="merge exceeds"):
        hierarchical_stable_topk(
            torch.zeros(1, 1, 1, 256),
            k=96,
            max_local_width=128,
            ops=TorchOps,
        )


def test_hierarchical_stable_topk_rejects_wide_local_indices():
    class WideIndexOps(TorchOps):
        @staticmethod
        def topk(tensor, k, dim, largest, sorted, stable):
            values, indices = TorchOps.topk(tensor, k, dim, largest, sorted, stable)
            return values, indices.to(torch.uint32)

    with pytest.raises(RuntimeError, match="requires uint16 local indices"):
        hierarchical_stable_topk(
            torch.zeros(1, 1, 1, 96),
            k=16,
            max_local_width=32,
            ops=WideIndexOps,
        )

def test_exact_nucleus_plan_allows_non_tile_vocab_at_full_candidate_envelope():
    plan = plan_exact_nucleus_candidates(
        0.999,
        vocab_size=65,
        public_topk_max_candidates=65,
    )
    assert plan.candidate_count == 65

    logits = torch.zeros(1, 1, 1, 96)
    logits[..., 65:] = -torch.inf
    result = _run_nucleus(
        logits,
        top_p=0.999,
        seeds=[17],
        candidate_count=plan.candidate_count,
        vocab_size=65,
    )
    assert result.valid_distribution.all()
    assert result.token_ids.item() < 65


def _run_nucleus(
    logits,
    *,
    top_p,
    seeds,
    candidate_count,
    vocab_size=None,
    stable_topk_max_local_width=0,
):
    logits = logits.to(torch.bfloat16)
    batch, width = logits.shape[2:]
    return sample_unrestricted_nucleus(
        logits,
        inverse_temperature=torch.ones(1, 1, batch, 1),
        top_p=torch.full((1, 1, batch, 1), top_p, dtype=torch.float32),
        row_scratch=[torch.zeros(1, 1, 1, 1) for _ in range(batch)],
        seed_values=seeds,
        active_rows=[True] * batch,
        vocab_size=width if vocab_size is None else vocab_size,
        candidate_count=candidate_count,
        stable_topk_max_local_width=stable_topk_max_local_width,
        ops=TorchOps,
    )


def test_exact_nucleus_matches_fp64_first_crossing_oracle_with_ties():
    # Candidate count 32 covers p<=.5 for V=64.  Equal leading logits pin the
    # stable index-order boundary, while the near-threshold tail exercises the
    # first-crossing rule instead of candidate renormalization.
    logits = torch.linspace(-8.0, -20.0, 64, dtype=torch.float64)
    logits[:4] = 2.0
    logits[4] = 1.999999
    logits = logits.to(torch.bfloat16).reshape(1, 1, 1, 64)
    p = 0.49
    seed = 1729
    result = _run_nucleus(logits, top_p=p, seeds=[seed], candidate_count=32)

    probs = torch.softmax(logits.double().flatten(), dim=0)
    order = torch.argsort(logits.double().flatten(), descending=True, stable=True)
    sorted_probs = probs[order]
    keep = (torch.cumsum(sorted_probs, 0) - sorted_probs) < p
    nucleus = sorted_probs * keep
    generator = torch.Generator().manual_seed(seed)
    draw = torch.rand((1,), generator=generator, dtype=torch.float32).double().item()
    expected_pos = int(torch.argmax((torch.cumsum(nucleus, 0) > nucleus.sum() * draw).to(torch.int32)))
    assert result.valid_distribution.all()
    assert result.token_ids.item() == order[expected_pos].item()


def test_exact_nucleus_uniform_worst_case_and_padded_tail():
    logits = torch.zeros(1, 1, 2, 96)
    logits[..., 65:] = -torch.inf
    # ceil(.49 * 65)=32, exactly the smallest guaranteed candidate envelope.
    result = _run_nucleus(logits, top_p=0.49, seeds=[11, 19], candidate_count=32, vocab_size=65)
    assert result.valid_distribution.all()
    assert (result.token_ids.to(torch.int64) < 65).all()


def test_exact_nucleus_hierarchical_stable_topk_matches_direct_route():
    logits = torch.linspace(-8, 8, 224).reshape(1, 1, 1, 224)
    logits[..., 17:24] = 3.0
    direct = _run_nucleus(logits, top_p=0.14, seeds=[919], candidate_count=32)
    hierarchical = _run_nucleus(
        logits,
        top_p=0.14,
        seeds=[919],
        candidate_count=32,
        stable_topk_max_local_width=64,
    )
    assert direct.valid_distribution.all() and hierarchical.valid_distribution.all()
    assert torch.equal(direct.token_ids.to(torch.int64), hierarchical.token_ids.to(torch.int64))


def test_exact_nucleus_normalizes_mixed_predicate_dtypes_before_logical_and():
    class StrictPredicateOps(TorchOps):
        @staticmethod
        def _predicate(value, dtype):
            return value.to(dtype)

        @classmethod
        def gt(cls, left, right):
            return cls._predicate(TorchOps.gt(left, right), left.dtype)

        @classmethod
        def lt(cls, left, right):
            return cls._predicate(TorchOps.lt(left, right), left.dtype)

        @classmethod
        def isfinite(cls, tensor):
            return cls._predicate(TorchOps.isfinite(tensor), tensor.dtype)

        @classmethod
        def logical_not(cls, tensor):
            return cls._predicate(torch.logical_not(tensor), tensor.dtype)

        @classmethod
        def logical_and(cls, left, right):
            assert left.dtype == right.dtype
            return cls._predicate(torch.logical_and(left, right), left.dtype)

    logits = torch.linspace(-2, 2, 64).reshape(1, 1, 1, 64).to(torch.bfloat16)
    result = sample_unrestricted_nucleus(
        logits,
        inverse_temperature=torch.ones(1, 1, 1, 1),
        top_p=torch.full((1, 1, 1, 1), 0.49),
        row_scratch=[torch.zeros(1, 1, 1, 1)],
        seed_values=[17],
        active_rows=[True],
        vocab_size=64,
        candidate_count=32,
        ops=StrictPredicateOps,
    )
    assert result.valid_distribution.to(torch.bool).all()


def test_gather_index_layout_normalization_uses_values_layout():
    index = SimpleNamespace(layout="row-major")
    values = SimpleNamespace(layout="tile")
    converted = SimpleNamespace(layout="tile")
    calls = []
    ops = SimpleNamespace(
        to_layout=lambda tensor, layout: calls.append((tensor, layout)) or converted
    )

    assert _normalize_gather_index_layout(index, values, ops=ops) is converted
    assert calls == [(index, "tile")]
    assert _normalize_gather_index_layout(converted, values, ops=ops) is converted
    assert calls == [(index, "tile")]


def test_exact_nucleus_marks_unmasked_high_padded_tail_invalid():
    logits = torch.zeros(1, 1, 1, 96)
    logits[..., 65:] = 100.0
    result = _run_nucleus(logits, top_p=0.49, seeds=[11], candidate_count=32, vocab_size=65)
    assert not result.valid_distribution.any()


@pytest.mark.parametrize(
    "bad_logits",
    [
        torch.full((1, 1, 1, 32), -torch.inf),
        torch.full((1, 1, 1, 32), torch.inf),
        torch.full((1, 1, 1, 32), torch.nan),
    ],
)
def test_exact_nucleus_invalid_distributions_fail_device_validity(bad_logits):
    result = _run_nucleus(bad_logits, top_p=0.5, seeds=[17], candidate_count=32)
    assert not result.valid_distribution.any()
