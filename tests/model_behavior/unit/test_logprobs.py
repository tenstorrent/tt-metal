# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.model_behavior.driver import Request, RequestDriver, RequestState, Sample, Sampling
from tests.model_behavior.test_logprobs import assert_logprobs_align
from tests.model_behavior.unit.test_driver import RecordingAdapter


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.0])
def test_driver_rejects_invalid_logprob_and_missing_value(value):
    driver = RequestDriver(RecordingAdapter())
    state = RequestState(Request("a", "prompt", Sampling(enable_log_probs=True)), 0, (1,))
    for output in (Sample(2, value, -0.5), 2):
        with pytest.raises(ValueError, match="logprob"):  # allow-pytest.raises: host-only isolated suite
            driver._record([state], {0: output}, phase="prefill")


@pytest.mark.parametrize("samples", [[], [Sample(8, -1.0, -1.0)], [Sample(7, -2.0, -1.0)]])
def test_oracle_rejects_missing_wrong_token_or_wrong_value(samples):
    state = RequestState(Request("a", "prompt", max_tokens=1), 0, (1,), [7], samples)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only isolated suite
        assert_logprobs_align(state)
