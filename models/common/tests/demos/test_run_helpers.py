# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.common.tests.demos import run_helpers
from models.common.tests.demos.run_helpers import run_perf_benchmark, run_teacher_forcing


def _logits(token_ids, vocab_size=8):
    output = torch.zeros(len(token_ids), 1, vocab_size)
    for row, token_id in enumerate(token_ids):
        output[row, 0, token_id] = 1
    return output


class FakeExecutionTarget:
    def __init__(self, *, compile_prefill_output, prefill_output, decode_outputs):
        self.compile_prefill_output = compile_prefill_output
        self.prefill_output = prefill_output
        self.decode_outputs = list(decode_outputs)
        self.calls = []

    @property
    def _engine(self):
        raise AssertionError("execution helpers must not inspect a private wrapped engine")

    def compile_prefill(self, **kwargs):
        self.calls.append(("compile_prefill", kwargs))
        return self.compile_prefill_output

    def compile_decode(self, **kwargs):
        self.calls.append(("compile_decode", kwargs))

    def prefill_forward(self, *args, **kwargs):
        self.calls.append(("prefill_forward", kwargs))
        return self.prefill_output

    def decode_forward(self, *args, **kwargs):
        self.calls.append(("decode_forward", kwargs))
        return self.decode_outputs.pop(0)


def test_teacher_forcing_uses_public_target_surface_and_preserves_user_order():
    target = FakeExecutionTarget(
        compile_prefill_output=_logits([3, 4]),
        prefill_output=_logits([3, 4]),
        decode_outputs=[(_logits([5, 1]), None)],
    )
    top5_tokens = torch.tensor([[3, 0, 1, 2, 4], [5, 0, 1, 2, 3]])

    result = run_teacher_forcing(
        executor=target,
        prompt_tokens=torch.tensor([[1, 2], [1, 2]]),
        reference_tokens=torch.tensor([1, 2, 6, 7]),
        top5_tokens=top5_tokens,
        kv_cache=[],
        page_table=torch.zeros(2, 1, dtype=torch.int32),
        max_batch_size=2,
    )

    assert result.predicted_tokens_per_user == [[3, 5], [4, 1]]
    assert result.top1_accuracy() == 1.0
    assert result.top5_accuracy() == 1.0
    assert [name for name, _ in target.calls] == [
        "compile_prefill",
        "compile_decode",
        "prefill_forward",
        "decode_forward",
    ]


def test_perf_benchmark_host_argmax_path_preserves_timing_and_tokens(monkeypatch):
    target = FakeExecutionTarget(
        compile_prefill_output=_logits([2]),
        prefill_output=_logits([2]),
        decode_outputs=[(_logits([3]), None), (_logits([4]), None), (_logits([5]), None)],
    )
    times = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.25, 3.0, 3.3])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))

    result = run_perf_benchmark(
        target,
        tokens=torch.tensor([[1, 2]]),
        kv_cache=[],
        page_table=torch.zeros(1, 1, dtype=torch.int32),
        num_decode_tokens=3,
    )

    assert result.prefill_time_s == pytest.approx(0.1)
    assert result.compile_decode_time_s == pytest.approx(0.2)
    assert result.decode_times_s == pytest.approx([0.25, 0.3])
    assert result.generated_token_ids == [[2, 3, 4, 5]]


class PublicReadbackTarget(FakeExecutionTarget):
    def __init__(self):
        super().__init__(
            compile_prefill_output=(torch.tensor([2]), None),
            prefill_output=(torch.tensor([2]), None),
            decode_outputs=[
                (torch.tensor([3]), None),
                (torch.tensor([4]), None),
                (torch.tensor([5]), None),
            ],
        )
        self.mesh_device = SimpleNamespace(shape=(1, 1))
        self.next_event = 0

    def read_decode_output(self, output, async_read=False):
        assert async_read
        event = self.next_event
        self.next_event += 1
        self.calls.append(("read_decode_output", {"event": event}))
        return output, [event]

    def process_decode_output_host(self, output, is_tokens=False):
        assert is_tokens
        self.calls.append(("process_decode_output_host", {}))
        return output


def test_perf_benchmark_uses_public_async_readback_without_trace_introspection(monkeypatch):
    target = PublicReadbackTarget()
    synchronized_events = []
    times = iter([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 3.4])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(run_helpers.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(run_helpers.ttnn, "event_synchronize", synchronized_events.append)

    result = run_perf_benchmark(
        target,
        tokens=torch.tensor([[1, 2]]),
        kv_cache=[],
        page_table=torch.zeros(1, 1, dtype=torch.int32),
        num_decode_tokens=3,
        sampling_params=object(),
        pipeline_readback=True,
    )

    assert [name for name, _ in target.calls[:2]] == ["compile_decode", "compile_prefill"]
    assert synchronized_events == [0, 1, 2]
    assert result.generated_token_ids == [[2, 3, 4, 5]]
    assert len(result.decode_times_s) == 2


def test_perf_benchmark_does_not_reprocess_blocking_sampled_output(monkeypatch):
    target = PublicReadbackTarget()
    times = iter([0.0, 0.1, 1.0, 1.1])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(run_helpers.ttnn, "synchronize_device", lambda mesh: None)

    result = run_perf_benchmark(
        target,
        tokens=torch.tensor([[1, 2]]),
        kv_cache=[],
        page_table=torch.zeros(1, 1, dtype=torch.int32),
        num_decode_tokens=1,
        sampling_params=object(),
        pipeline_readback=False,
    )

    assert result.generated_token_ids == [[2, 3]]
    assert "process_decode_output_host" not in [name for name, _ in target.calls]


def test_loop_policy_is_not_exported_from_production_executor():
    from models.common.llm_runtime import executor as production_executor

    for name in ("TeacherForceResult", "PerfBenchmarkResult", "run_teacher_forcing", "run_perf_benchmark"):
        assert not hasattr(production_executor, name)
