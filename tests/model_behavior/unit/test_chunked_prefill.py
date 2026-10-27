# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tests.model_behavior.driver import Request, RequestState
from tests.model_behavior.test_chunked_prefill import assert_needle_recalled


def test_needle_oracle_rejects_another_requests_answer():
    state = RequestState(Request("a", "prompt", max_tokens=1), 31, (1,), [2])
    adapter = SimpleNamespace(decode_tokens=lambda tokens: "copper lantern 4826")
    with pytest.raises(AssertionError, match="violet compass"):  # allow-pytest.raises: host-only isolated suite
        assert_needle_recalled(adapter, state, "violet compass 7319")
