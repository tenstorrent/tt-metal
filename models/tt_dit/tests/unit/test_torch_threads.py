# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.tt_dit.utils import torch_threads
from models.tt_dit.utils.torch_threads import cap_torch_threads_to_half_the_cpus


@pytest.fixture
def torch_pool(monkeypatch):
    pool = SimpleNamespace(count=64, sets=[])

    def set_num_threads(count):
        pool.sets.append(count)
        pool.count = count

    fake_torch = SimpleNamespace(get_num_threads=lambda: pool.count, set_num_threads=set_num_threads)
    monkeypatch.setattr(torch_threads, "torch", fake_torch)
    return pool


@pytest.fixture
def affinity(monkeypatch):
    mask = SimpleNamespace(cpus=set(range(8)))
    monkeypatch.setattr(torch_threads.os, "sched_getaffinity", lambda pid: mask.cpus)
    return mask


def test_lowers_to_half_the_affinity_mask_and_restores(torch_pool, affinity):
    with cap_torch_threads_to_half_the_cpus():
        assert torch_pool.count == 4
    assert torch_pool.count == 64


@pytest.mark.parametrize("count", [1, 4])
def test_leaves_a_count_at_or_below_the_cap_untouched(torch_pool, affinity, count):
    torch_pool.count = count
    with cap_torch_threads_to_half_the_cpus():
        assert torch_pool.count == count
    assert torch_pool.sets == []


def test_nested_caps_restore_the_callers_count_after_an_exception(torch_pool, affinity, expect_error):
    with expect_error(RuntimeError, "body failed"):
        with cap_torch_threads_to_half_the_cpus():
            with cap_torch_threads_to_half_the_cpus():
                assert torch_pool.count == 4
            assert torch_pool.count == 4
            raise RuntimeError("body failed")
    assert torch_pool.count == 64


def test_rereads_the_affinity_mask_on_every_entry(torch_pool, affinity):
    with cap_torch_threads_to_half_the_cpus():
        assert torch_pool.count == 4
    affinity.cpus = {0, 1, 2, 3}
    with cap_torch_threads_to_half_the_cpus():
        assert torch_pool.count == 2
    assert torch_pool.count == 64


def test_single_cpu_mask_caps_to_one_thread_not_zero(torch_pool, affinity):
    affinity.cpus = {0}
    with cap_torch_threads_to_half_the_cpus():
        assert torch_pool.count == 1


@pytest.mark.parametrize("cpu_count, expected", [(6, 3), (None, 1)])
def test_falls_back_to_cpu_count_without_sched_getaffinity(monkeypatch, cpu_count, expected):
    monkeypatch.delattr(torch_threads.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(torch_threads.os, "cpu_count", lambda: cpu_count)
    assert torch_threads.half_of_affinity_mask() == expected


def test_decorated_function_is_capped_on_every_call_and_keeps_its_name(torch_pool, affinity):
    @cap_torch_threads_to_half_the_cpus()
    def read_back():
        return torch_pool.count

    assert [read_back(), read_back()] == [4, 4]
    assert read_back.__name__ == "read_back"
    assert torch_pool.count == 64
