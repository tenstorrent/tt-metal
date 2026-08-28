# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the shared Qwen2/Qwen2.5 executor policy."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.common.models import qwen2_executor
from models.common.models.executor import ModelExecutor


def test_qwen2_public_executors_are_unique_composition_facades():
    executor_classes = (
        qwen2_executor.Qwen2Executor,
        qwen2_executor.Qwen25Executor,
        qwen2_executor.Qwen25_72BExecutor,
        qwen2_executor.Qwen25Coder32BExecutor,
    )

    assert len(set(executor_classes)) == 4
    assert all(executor_class.__bases__ == (object,) for executor_class in executor_classes)
    assert all(not issubclass(executor_class, ModelExecutor) for executor_class in executor_classes)


@pytest.mark.parametrize(
    "executor_class",
    (qwen2_executor.Qwen2Executor, qwen2_executor.Qwen25Executor),
)
def test_qwen2_7b_warms_every_q128_topk_tile_end_once_per_execution_mode(executor_class):
    executor = object.__new__(executor_class)
    executor._q128_topk_tile_ends_warmed = set()
    executor.eager_executor = object()
    executor.traced_executor = object()
    executor.page_table_layout = SimpleNamespace(block_size=32)
    executor.prefill_runtime = SimpleNamespace(config=SimpleNamespace(static_q128_topk_supported=True))
    executor.warmup = SimpleNamespace(
        config=SimpleNamespace(
            prefill_sequence_lengths=(128,),
            prime_q128_tile_ends=False,
        )
    )
    executor.compile_prefill = MagicMock()
    kv_cache = object()

    for enable_trace in (False, False, True, True):
        executor._warmup_q128_topk_tile_ends(
            kv_cache=kv_cache,
            can_sample_on_device=True,
            enable_trace=enable_trace,
        )

    assert executor.compile_prefill.call_count == 6
    calls = executor.compile_prefill.call_args_list
    assert [call.kwargs["tokens"].shape[1] for call in calls] == [32, 64, 96, 32, 64, 96]
    assert [call.kwargs["page_table"].shape[1] for call in calls] == [1, 2, 3, 1, 2, 3]
    assert all(call.kwargs["kv_cache"] is kv_cache for call in calls)
    assert all(call.kwargs["execution"] is executor.eager_executor for call in calls[:3])
    assert all(call.kwargs["execution"] is executor.traced_executor for call in calls[3:])
    assert executor._q128_topk_tile_ends_warmed == {False, True}


@pytest.mark.parametrize(
    ("enable_trace", "expected_events"),
    (
        (False, ("default", "prime-eager")),
        (True, ("prime-traced", "default")),
    ),
)
def test_qwen2_7b_q128_warmup_preserves_mode_specific_order(monkeypatch, enable_trace, expected_events):
    events = []

    def record_prime(executor, **kwargs):
        del executor
        events.append("prime-traced" if kwargs["enable_trace"] else "prime-eager")

    monkeypatch.setattr(qwen2_executor, "_warmup_q128_topk_tile_ends", record_prime)
    qwen2_executor._warmup_q128_around_prefill(
        object(),
        lambda: events.append("default"),
        kv_cache=object(),
        can_sample_on_device=True,
        enable_trace=enable_trace,
    )

    assert tuple(events) == expected_events
