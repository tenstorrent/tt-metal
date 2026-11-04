# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Check that the model scenarios reject plausible integration failures."""

import pytest

from tests.model_behavior.driver import RequestDriver
from tests.model_behavior.test_sampling_behavior import test_identical_seed_requests_match as identical_seed_scenario
from tests.model_behavior.test_sampling_behavior import (
    test_neighbor_penalties_preserve_unpenalized_request as neighbor_scenario,
)
from tests.model_behavior.test_sampling_behavior import test_penalty_changes_model_output as penalty_scenario
from tests.model_behavior.test_sampling_behavior import (
    test_seeded_requests_replay_after_slot_permutation as permutation_scenario,
)
from tests.model_behavior.test_sampling_behavior import test_stochastic_requests_vary as stochastic_scenario
from tests.model_behavior.test_sampling_behavior import test_top_p_changes_model_sampling as top_p_scenario
from tests.model_behavior.test_sampling_behavior import test_unseeded_single_request_varies as unseeded_scenario
from tests.model_behavior.unit.test_driver import RecordingAdapter


class SlotDependentAdapter(RecordingAdapter):
    duplicate_seed_policy = "identical"

    def prefill(self, admitted):
        tokens = super().prefill(admitted)
        return {slot: (token + slot) % self.vocab_size for slot, token in tokens.items()}


class NeighborPenaltyLeakAdapter(RecordingAdapter):
    def prefill(self, admitted):
        tokens = super().prefill(admitted)
        any_penalty = any(state.request.sampling.presence_penalty for state in admitted)
        return {slot: (token + int(any_penalty)) % self.vocab_size for slot, token in tokens.items()}


@pytest.mark.parametrize("penalty", ["repetition_penalty", "presence_penalty", "frequency_penalty"])
def test_model_effect_checks_reject_ignored_penalties(penalty):
    with pytest.raises(AssertionError, match="had no effect"):  # allow-pytest.raises: host-only isolated suite
        penalty_scenario(RequestDriver(RecordingAdapter()), penalty, 2.0)


def test_diversity_check_rejects_deterministic_sampling():
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        AssertionError, match="No prefill token diversity"
    ):
        stochastic_scenario(RequestDriver(RecordingAdapter()), seeded=True)


def test_seed_change_check_rejects_slot_randomness_without_request_seeds():
    # Variation across slots and exact replay alone are insufficient: this
    # adapter satisfies both while completely ignoring the requested seeds.
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        AssertionError, match="Changing seeds had no effect"
    ):
        stochastic_scenario(RequestDriver(SlotDependentAdapter()), seeded=True)


def test_unseeded_batch_check_rejects_fixed_per_slot_streams():
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        AssertionError, match="Unseeded requests replayed exactly"
    ):
        stochastic_scenario(RequestDriver(SlotDependentAdapter()), seeded=False)


def test_unseeded_single_check_rejects_fixed_stream():
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        AssertionError, match="No prefill token diversity"
    ):
        unseeded_scenario(RequestDriver(RecordingAdapter()))


def test_top_p_effect_check_rejects_ignored_cutoff():
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        AssertionError, match="Restricting top_p had no effect"
    ):
        top_p_scenario(RequestDriver(RecordingAdapter()))


@pytest.mark.parametrize("scenario,kwargs", [(identical_seed_scenario, {"seed": 0}), (permutation_scenario, {})])
def test_seed_contract_checks_reject_slot_dependent_streams(scenario, kwargs):
    with pytest.raises(AssertionError, match="first difference"):  # allow-pytest.raises: host-only isolated suite
        scenario(RequestDriver(SlotDependentAdapter()), **kwargs)


def test_neighbor_control_rejects_batch_wide_penalty_leak():
    with pytest.raises(AssertionError, match="first difference"):  # allow-pytest.raises: host-only isolated suite
        neighbor_scenario(RequestDriver(NeighborPenaltyLeakAdapter()), "presence_penalty", 2.0)


def test_single_user_penalty_sensitivity_keeps_every_prompt():
    adapter = RecordingAdapter()
    adapter.capacity = 1
    driver = RequestDriver(adapter)
    with pytest.raises(AssertionError, match="had no effect"):  # allow-pytest.raises: host-only isolated suite
        penalty_scenario(driver, "presence_penalty", 2.0)
    assert len(driver.requests) == 24  # Eight baselines, controls, and penalized requests.
    assert {state.slot for state in driver.requests.values()} == {0}


def test_single_user_top_p_check_still_rejects_an_ignored_cutoff():
    adapter = RecordingAdapter()
    adapter.capacity = 1
    driver = RequestDriver(adapter)
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        AssertionError, match="Restricting top_p had no effect"
    ):
        top_p_scenario(driver)
    assert len(driver.requests) == 16


def test_host_prefill_bootstrap_does_not_excuse_deterministic_decode():
    adapter = RecordingAdapter()
    adapter.stochastic_prefill = False
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        AssertionError, match="No generated sequence diversity"
    ):
        unseeded_scenario(RequestDriver(adapter))
