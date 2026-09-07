# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only state contracts for TT sampled-token log-probability mode."""

from models.common.sampling.tt_log_probs import LogProbsCalculator


def _calculator(batch_size=4):
    calculator = object.__new__(LogProbsCalculator)
    calculator.batch_size = batch_size
    calculator.logprobs_enabled = [False] * batch_size
    calculator.num_logprobs = [0] * batch_size
    calculator.enable_log_probs = False
    calculator.topk_logprobs_needed = False
    return calculator


def test_all_false_tuple_keeps_log_probs_disabled():
    calculator = _calculator()

    calculator.set_log_probs_mode((False, False, False, False), num_logprobs=(0, 0, 0, 0))

    assert calculator.logprobs_enabled == [False, False, False, False]
    assert calculator.num_logprobs == [0, 0, 0, 0]
    assert calculator.enable_log_probs is False
    assert calculator.topk_logprobs_needed is False


def test_tuple_log_probs_and_counts_map_per_user_exactly():
    calculator = _calculator()

    calculator.set_log_probs_mode((True, False, True, False), num_logprobs=(0, 4, 7, 2))

    assert calculator.logprobs_enabled == [True, False, True, False]
    assert calculator.num_logprobs == [0, 4, 7, 2]
    assert calculator.enable_log_probs is True
    assert calculator.topk_logprobs_needed is True


def test_tuple_partial_update_respects_empty_slots():
    calculator = _calculator(batch_size=6)

    calculator.set_log_probs_mode((True, False), num_logprobs=(10, 15), empty_slots=[2, 5])

    assert calculator.logprobs_enabled == [False, False, True, False, False, False]
    assert calculator.num_logprobs == [0, 0, 10, 0, 0, 15]
    assert calculator.enable_log_probs is True


def test_existing_scalar_and_list_modes_remain_unchanged():
    calculator = _calculator()

    calculator.set_log_probs_mode(True, num_logprobs=5)
    assert calculator.logprobs_enabled == [True, True, True, True]
    assert calculator.num_logprobs == [5, 5, 5, 5]

    calculator.set_log_probs_mode([True, False, False, True], num_logprobs=[1, 2, 3, 4])
    assert calculator.logprobs_enabled == [True, False, False, True]
    assert calculator.num_logprobs == [1, 2, 3, 4]
    assert calculator.enable_log_probs is True
    assert calculator.topk_logprobs_needed is True
