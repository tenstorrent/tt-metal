import pytest

from models.common.sampling.full_vocab_contract import ManagedPerSlotDrawSeeds, classify_sampling_batch


def contract(*, temperature, top_p, top_k, active_slots=range(32), seeds=None, vocab_size=201088):
    return classify_sampling_batch(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seeds=[None] * 32 if seeds is None else seeds,
        active_slots=list(active_slots),
        batch_size=32,
        vocab_size=vocab_size,
        max_bounded_top_k=32,
    )


@pytest.mark.parametrize("sentinel", [0, -1, 201088, 262144])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_unrestricted_sentinels_survive_classification(sentinel, temperature):
    result = contract(
        temperature=[temperature] * 32,
        top_p=[1.0] * 32,
        top_k=[sentinel] * 32,
    )
    assert result.needs_full_vocabulary
    assert result.active_modes == {"unrestricted"}
    assert [row.top_k for row in result.rows] == [sentinel] * 32


def test_mixed_greedy_bounded_unrestricted_and_inactive_rows_are_explicit():
    result = contract(
        temperature=[0.0, 1.0, 1.0, 1.0] + [0.0] * 28,
        top_p=[1.0, 0.95, 1.0, 1.0] + [1.0] * 28,
        top_k=[0, 20, -1, 0] + [0] * 28,
        active_slots=[0, 1, 2],
        seeds=[7, 11, 13, 17] + [None] * 28,
    )
    assert [row.mode for row in result.rows[:4]] == ["greedy", "bounded", "unrestricted", "inactive"]
    assert result.is_mixed
    assert [row.seed for row in result.rows[:4]] == [7, 11, 13, 17]


def test_unrestricted_nucleus_requirement_is_not_conflated_with_top_p_one():
    result = contract(
        temperature=[1.0] * 32,
        top_p=[1.0, 0.9] + [1.0] * 30,
        top_k=[0, -1] + [20] * 30,
    )
    assert result.unrestricted_slots == (0, 1)
    assert result.unrestricted_nucleus_slots == (1,)


def test_top_k_between_device_limit_and_vocab_fails_closed():
    with pytest.raises(ValueError, match="no exact device algorithm"):
        contract(temperature=[1.0] * 32, top_p=[1.0] * 32, top_k=[33] * 32)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", float("nan"), "temperature"),
        ("temperature", -0.1, "temperature"),
        ("top_p", 0.0, "top_p"),
        ("top_p", float("inf"), "top_p"),
        ("top_k", 2.5, "top_k"),
        ("top_k", True, "top_k"),
    ],
)
def test_invalid_policy_fails_closed(field, value, message):
    values = {
        "temperature": [1.0] * 32,
        "top_p": [1.0] * 32,
        "top_k": [0] * 32,
    }
    values[field][0] = value
    with pytest.raises(ValueError, match=message):
        contract(**values)


def test_active_slots_must_be_unique_and_in_range():
    values = dict(temperature=[1.0] * 32, top_p=[1.0] * 32, top_k=[0] * 32)
    with pytest.raises(ValueError, match="distinct in-range"):
        contract(**values, active_slots=[0, 0])
    with pytest.raises(ValueError, match="distinct in-range"):
        contract(**values, active_slots=[32])
    with pytest.raises(ValueError, match="integer slot indices"):
        contract(**values, active_slots=[0.0])
    with pytest.raises(ValueError, match="integer slot indices"):
        contract(**values, active_slots=[True])


@pytest.mark.parametrize("seed", [True, 1.5, float("nan")])
def test_seed_must_be_an_integer_or_none(seed):
    seeds = [None] * 32
    seeds[0] = seed
    with pytest.raises(ValueError, match=r"seeds\[0\]"):
        contract(
            temperature=[1.0] * 32,
            top_p=[1.0] * 32,
            top_k=[0] * 32,
            seeds=seeds,
        )


def test_managed_draws_are_per_slot_and_independent_of_other_row_algorithms():
    left = ManagedPerSlotDrawSeeds(4, entropy=iter([101, 202]).__next__)
    right = ManagedPerSlotDrawSeeds(4, entropy=iter([101, 202]).__next__)
    for state in (left, right):
        state.reset_slot(0, 7)
        state.reset_slot(1, None)

    mixed = left.next_plan([1, 2, 0, 0])
    bounded_only = right.next_plan([1, 1, 0, 0])
    assert mixed.seeds_by_subdraw[0] == bounded_only.seeds_by_subdraw[0]
    assert mixed.seeds_by_subdraw[1][0] == 2**32 - 1
    # The next token's first draw is unchanged by another row taking an extra
    # subdraw in the prior token.
    assert left.next_plan([1, 1, 0, 0]).seeds_by_subdraw == right.next_plan([1, 1, 0, 0]).seeds_by_subdraw


def test_managed_draw_state_follows_slot_remap_and_clears_vacated_source():
    baseline = ManagedPerSlotDrawSeeds(4, entropy=lambda: 99)
    remapped = ManagedPerSlotDrawSeeds(4, entropy=lambda: 99)
    for state in (baseline, remapped):
        state.reset_slot(3, 1234, salt=2)
        state.next_plan([0, 0, 0, 2])
    remapped.remap([3, 1, 2, 3])
    expected = baseline.next_plan([0, 0, 0, 2])
    actual = remapped.next_plan([2, 0, 0, 0])
    assert actual.seeds_by_subdraw[0][0] == expected.seeds_by_subdraw[0][3]
    assert actual.seeds_by_subdraw[1][0] == expected.seeds_by_subdraw[1][3]
    with pytest.raises(RuntimeError, match="slot 3"):
        remapped.next_plan([0, 0, 0, 1])


def test_unseeded_request_root_is_owned_until_departure_then_replaced():
    entropy = iter([11, 22]).__next__
    state = ManagedPerSlotDrawSeeds(2, entropy=entropy)
    state.reset_slot(0, None)
    first = state.next_plan([1, 0]).seeds_by_subdraw[0][0]
    second = state.next_plan([1, 0]).seeds_by_subdraw[0][0]
    assert first != second
    state.deactivate_slots_except([])
    with pytest.raises(RuntimeError, match="slot 0"):
        state.next_plan([1, 0])
    state.reset_slot(0, None)
    assert state.next_plan([1, 0]).seeds_by_subdraw[0][0] != first


def test_managed_draw_alignment_is_absolute_and_fail_closed():
    state = ManagedPerSlotDrawSeeds(2, entropy=lambda: 44)
    state.reset_slot(0, 5)
    state.align_token_counters([17, -1], [0])
    aligned = state.next_plan([1, 0]).seeds_by_subdraw

    reference = ManagedPerSlotDrawSeeds(2, entropy=lambda: 44)
    reference.reset_slot(0, 5)
    for _ in range(19):
        expected = reference.next_plan([1, 0]).seeds_by_subdraw
    assert aligned == expected
    with pytest.raises(RuntimeError, match="slot 1"):
        state.align_token_counters([17, 0], [1])


def test_managed_draw_contract_rejects_unbounded_internal_work():
    state = ManagedPerSlotDrawSeeds(1, entropy=lambda: 1)
    state.reset_slot(0, 1)
    with pytest.raises(ValueError, match=r"\[0, 16\]"):
        state.next_plan([17])


def test_one_request_subdraw_has_no_seed_repeat_in_bounded_prefix():
    state = ManagedPerSlotDrawSeeds(1, entropy=lambda: 1)
    state.reset_slot(0, 123)
    values = [state.next_plan([1]).seeds_by_subdraw[0][0] for _ in range(10_000)]
    assert len(set(values)) == len(values)
    assert 0 not in values
    assert 2**32 - 1 not in values
