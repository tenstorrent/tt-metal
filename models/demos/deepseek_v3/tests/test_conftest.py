# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.deepseek_v3.conftest import _clear_requested_state_dict_cache, clear_state_dict_cache


class _StateDict:
    def __init__(self):
        self.cache_clear_count = 0

    def clear_cache(self):
        self.cache_clear_count += 1


class _Request:
    def __init__(self, state_dict):
        self.fixturenames = []
        self.state_dict = state_dict
        self.request_count = 0

    def getfixturevalue(self, name):
        assert name == "state_dict"
        self.request_count += 1
        return self.state_dict


def test_state_dict_cache_cleanup_detects_dynamic_fixture_request():
    state_dict = _StateDict()
    request = _Request(state_dict)

    # Dynamic fixture requests become visible only after autouse setup has run.
    request.fixturenames.append("state_dict")
    _clear_requested_state_dict_cache(request)

    assert request.request_count == 1
    assert state_dict.cache_clear_count == 1


def test_state_dict_cache_cleanup_checks_dynamic_request_after_yield():
    state_dict = _StateDict()
    request = _Request(state_dict)
    fixture = clear_state_dict_cache.__wrapped__(request)

    assert next(fixture) is None
    assert request.request_count == 0

    request.fixturenames.append("state_dict")
    sentinel = object()
    assert next(fixture, sentinel) is sentinel
    assert request.request_count == 1
    assert state_dict.cache_clear_count == 1


def test_state_dict_cache_cleanup_keeps_random_tests_lazy():
    state_dict = _StateDict()
    request = _Request(state_dict)

    _clear_requested_state_dict_cache(request)

    assert request.request_count == 0
    assert state_dict.cache_clear_count == 0
