# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from models.common.llm_runtime.lane_group import LaneGroupExecutor


@dataclass(frozen=True)
class _SamplingParams:
    temperature: list[float]
    top_k: torch.Tensor
    top_p: float


@dataclass(frozen=True)
class _SingletonSamplingParams:
    temperature: list[float]
    top_k: torch.Tensor
    top_p: tuple[float, ...]


class _Lane:
    def __init__(self, lane_idx, *, capacity=2):
        self.lane_idx = lane_idx
        self.model = SimpleNamespace(config=SimpleNamespace(max_batch_size=capacity))
        self.model_args = f"args-{lane_idx}"
        self.mesh_device = f"mesh-{lane_idx}"
        self.cache_path = f"cache-{lane_idx}"
        self.already_warmed_up_prefill = False
        self.calls = []
        self.cleanup_calls = 0
        self.fail_method = None
        self.cleanup_error = None

    def _call(self, method, kwargs):
        self.calls.append((method, kwargs))
        if self.fail_method == method:
            raise RuntimeError(f"{method} boom {self.lane_idx}")

    def configure_paged_kv_cache(self, config):
        self._call("configure", {"config": config})
        self.paged_kv_cache_config = config

    def allocate_kv_cache(self):
        self._call("allocate", {})
        return f"cache-handle-{self.lane_idx}"

    def compile_prefill(self, **kwargs):
        self._call("compile_prefill", kwargs)

    def compile_decode(self, **kwargs):
        self._call("compile_decode", kwargs)

    def warmup_model_prefill(self, **kwargs):
        self._call("warmup_prefill", kwargs)
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, **kwargs):
        self._call("warmup_decode", kwargs)

    def prefill_forward(self, **kwargs):
        self._call("prefill", kwargs)
        values = kwargs["tokens"][:, 0] + self.lane_idx * 100
        if kwargs.get("sampling_params") is not None:
            return values.to(torch.int64), None
        return values.float().view(-1, 1, 1)

    def decode_forward(self, **kwargs):
        self._call("decode", kwargs)
        values = kwargs["tokens"] + self.lane_idx * 100
        if not kwargs.get("read_from_device", True):
            return f"raw-{self.lane_idx}", None
        if kwargs.get("sampling_params") is not None:
            return values.to(torch.int64), None
        return values.float().view(-1, 1, 1), None

    def read_decode_output(self, lane_output, async_read=False):
        self._call("read", {"lane_output": lane_output, "async_read": async_read})
        host = (torch.tensor([self.lane_idx * 2, self.lane_idx * 2 + 1]), None)
        if async_read:
            return host, [f"event-{self.lane_idx}"]
        return host

    def process_decode_output_host(self, lane_output, is_tokens=False):
        self._call("process", {"lane_output": lane_output, "is_tokens": is_tokens})
        if is_tokens:
            return torch.tensor([self.lane_idx * 2, self.lane_idx * 2 + 1], dtype=torch.int64), None
        return torch.full((2, 1, 3), float(self.lane_idx)), None

    def cleanup(self):
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


def _sampling():
    return _SamplingParams(
        temperature=[0.1, 0.2, 0.3, 0.4],
        top_k=torch.tensor([1, 2, 3, 4]),
        top_p=0.9,
    )


def test_prefill_routes_global_slots_and_restores_source_row_order():
    lanes = [_Lane(0), _Lane(1)]
    group = LaneGroupExecutor(lanes)
    tokens = torch.tensor([[10], [11], [12]])
    page_table = torch.tensor([[0], [1], [2]], dtype=torch.int32)

    output = group.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        empty_slots=[3, 0, 2],
        prompt_lens=torch.tensor([1, 1, 1]),
        sampling_params=_sampling(),
        kv_cache=["kv-0", "kv-1"],
    )

    assert isinstance(output, tuple)
    assert output[0].tolist() == [110, 11, 112]
    assert output[0].dtype == torch.int64
    assert output[1] is None

    lane0_kwargs = next(kwargs for method, kwargs in lanes[0].calls if method == "prefill")
    lane1_kwargs = next(kwargs for method, kwargs in lanes[1].calls if method == "prefill")
    assert lane0_kwargs["empty_slots"] == [0]
    assert lane1_kwargs["empty_slots"] == [1, 0]
    assert lane0_kwargs["tokens"].flatten().tolist() == [11]
    assert lane1_kwargs["tokens"].flatten().tolist() == [10, 12]
    assert lane0_kwargs["sampling_params"].temperature == [0.2]
    assert lane1_kwargs["sampling_params"].temperature == [0.1, 0.3]
    assert lane0_kwargs["sampling_params"].top_k.tolist() == [2]
    assert lane1_kwargs["sampling_params"].top_k.tolist() == [1, 3]
    assert lane0_kwargs["sampling_params"].top_p == 0.9
    assert lane0_kwargs["kv_cache"] == "kv-0"
    assert lane1_kwargs["kv_cache"] == "kv-1"


def test_decode_slices_fixed_lane_capacity_and_normalizes_sampled_tokens():
    lanes = [_Lane(0), _Lane(1)]
    group = LaneGroupExecutor(lanes)

    output = group.decode_forward(
        tokens=torch.tensor([10, 11, 12, 13]),
        start_pos=torch.tensor([1, 2, 3, 4]),
        page_table=torch.arange(4, dtype=torch.int32).view(4, 1),
        sampling_params=_sampling(),
        kv_cache=["kv-0", "kv-1"],
    )

    assert output[0].tolist() == [10, 11, 112, 113]
    assert output[0].dtype == torch.int64
    assert output[1] is None
    lane0_kwargs = next(kwargs for method, kwargs in lanes[0].calls if method == "decode")
    lane1_kwargs = next(kwargs for method, kwargs in lanes[1].calls if method == "decode")
    assert lane0_kwargs["tokens"].tolist() == [10, 11]
    assert lane1_kwargs["tokens"].tolist() == [12, 13]
    assert lane0_kwargs["sampling_params"].temperature == [0.1, 0.2]
    assert lane1_kwargs["sampling_params"].temperature == [0.3, 0.4]


def test_decode_broadcasts_singleton_sampling_values_to_later_dp_lanes():
    lanes = [_Lane(0, capacity=1), _Lane(1, capacity=1)]
    group = LaneGroupExecutor(lanes)
    sampling_params = _SingletonSamplingParams(
        temperature=[0.7],
        top_k=torch.tensor([8]),
        top_p=(0.4,),
    )

    output = group.decode_forward(
        tokens=torch.tensor([10, 11]),
        start_pos=torch.tensor([1, 2]),
        page_table=torch.arange(2, dtype=torch.int32).view(2, 1),
        sampling_params=sampling_params,
    )

    assert output[0].tolist() == [10, 111]
    lane0_kwargs = next(kwargs for method, kwargs in lanes[0].calls if method == "decode")
    lane1_kwargs = next(kwargs for method, kwargs in lanes[1].calls if method == "decode")
    for lane_kwargs in (lane0_kwargs, lane1_kwargs):
        sliced = lane_kwargs["sampling_params"]
        assert sliced.temperature == [0.7]
        assert sliced.top_k.tolist() == [8]
        assert sliced.top_p == (0.4,)


def test_decode_without_readback_returns_lane_local_outputs():
    group = LaneGroupExecutor([_Lane(0), _Lane(1)])

    output = group.decode_forward(
        tokens=torch.tensor([0, 1, 2, 3]),
        start_pos=torch.tensor([0, 0, 0, 0]),
        page_table=torch.zeros((4, 1), dtype=torch.int32),
        read_from_device=False,
    )

    assert output == [("raw-0", None), ("raw-1", None)]


def test_async_read_and_host_processing_run_per_lane_and_preserve_lane_order():
    group = LaneGroupExecutor([_Lane(0), _Lane(1)])

    host_outputs, events = group.read_decode_output(
        [("raw-0", None), ("raw-1", None)],
        async_read=True,
    )
    processed = group.process_decode_output_host(host_outputs, is_tokens=True)

    assert events == ["event-0", "event-1"]
    assert processed[0].tolist() == [0, 1, 2, 3]
    assert processed[0].dtype == torch.int64
    assert processed[1] is None


def test_warmup_replicates_lane_local_case_and_cache_to_every_lane():
    lanes = [_Lane(0), _Lane(1)]
    group = LaneGroupExecutor(lanes)

    group.warmup_model_prefill(kv_cache=["kv-0", "kv-1"], enable_trace=True)
    group.warmup_model_decode(
        kv_cache=["kv-0", "kv-1"],
        enable_trace=True,
        max_batch_size=4,
        num_blocks=8,
        can_sample_on_device=True,
    )

    for lane_idx, lane in enumerate(lanes):
        prefill = next(kwargs for method, kwargs in lane.calls if method == "warmup_prefill")
        decode = next(kwargs for method, kwargs in lane.calls if method == "warmup_decode")
        assert prefill["kv_cache"] == f"kv-{lane_idx}"
        assert decode["kv_cache"] == f"kv-{lane_idx}"
        assert decode["max_batch_size"] == 2
    assert group.already_warmed_up_prefill


def test_compile_methods_slice_requests_to_lane_executors():
    lanes = [_Lane(0), _Lane(1)]
    group = LaneGroupExecutor(lanes)

    group.compile_prefill(
        tokens=torch.tensor([[10], [11], [12]]),
        page_table=torch.tensor([[0], [1], [2]], dtype=torch.int32),
        empty_slots=[3, 0, 2],
        kv_cache=["kv-0", "kv-1"],
    )
    group.compile_decode(
        tokens=torch.tensor([10, 11, 12, 13]),
        start_pos=torch.tensor([1, 2, 3, 4]),
        page_table=torch.arange(4, dtype=torch.int32).view(4, 1),
        kv_cache=["kv-0", "kv-1"],
    )

    lane0_prefill = next(kwargs for method, kwargs in lanes[0].calls if method == "compile_prefill")
    lane1_prefill = next(kwargs for method, kwargs in lanes[1].calls if method == "compile_prefill")
    lane0_decode = next(kwargs for method, kwargs in lanes[0].calls if method == "compile_decode")
    lane1_decode = next(kwargs for method, kwargs in lanes[1].calls if method == "compile_decode")
    assert lane0_prefill["tokens"].flatten().tolist() == [11]
    assert lane1_prefill["tokens"].flatten().tolist() == [10, 12]
    assert lane0_decode["tokens"].tolist() == [10, 11]
    assert lane1_decode["tokens"].tolist() == [12, 13]


def test_configure_and_allocate_fan_out_with_distinct_lane_configs():
    lanes = [_Lane(0), _Lane(1)]
    group = LaneGroupExecutor(lanes)
    config = SimpleNamespace(num_blocks=8)

    group.configure_paged_kv_cache(config)
    handles = group.allocate_kv_cache()

    assert handles == ["cache-handle-0", "cache-handle-1"]
    lane_configs = [next(kwargs["config"] for method, kwargs in lane.calls if method == "configure") for lane in lanes]
    assert lane_configs[0] is not lane_configs[1]
    assert lane_configs[0].num_blocks == lane_configs[1].num_blocks == 8


def test_side_effect_execution_target_methods_return_none_even_if_lanes_return_values():
    class ReturningLane(_Lane):
        def configure_paged_kv_cache(self, config):
            super().configure_paged_kv_cache(config)
            return self.lane_idx

        def compile_prefill(self, **kwargs):
            super().compile_prefill(**kwargs)
            return self.lane_idx

        def compile_decode(self, **kwargs):
            super().compile_decode(**kwargs)
            return self.lane_idx

        def warmup_model_prefill(self, **kwargs):
            super().warmup_model_prefill(**kwargs)
            return self.lane_idx

        def warmup_model_decode(self, **kwargs):
            super().warmup_model_decode(**kwargs)
            return self.lane_idx

    group = LaneGroupExecutor([ReturningLane(0), ReturningLane(1)])
    config = SimpleNamespace(num_blocks=8)
    results = (
        group.configure_paged_kv_cache(config),
        group.compile_prefill(
            tokens=torch.tensor([[10], [11]]),
            page_table=torch.tensor([[0], [1]], dtype=torch.int32),
            empty_slots=[0, 2],
        ),
        group.compile_decode(
            tokens=torch.tensor([10, 11, 12, 13]),
            start_pos=torch.tensor([1, 2, 3, 4]),
            page_table=torch.arange(4, dtype=torch.int32).view(4, 1),
        ),
        group.warmup_model_prefill(enable_trace=False),
        group.warmup_model_decode(enable_trace=False, max_batch_size=4),
    )

    assert results == (None, None, None, None, None)


def test_constructor_validation_failure_cleans_every_supplied_lane(expect_error):
    lanes = [_Lane(0, capacity=2), _Lane(1, capacity=4)]

    with expect_error(ValueError, "same fixed capacity"):
        LaneGroupExecutor(lanes)

    assert [lane.cleanup_calls for lane in lanes] == [1, 1]


def test_operation_failure_is_primary_group_becomes_terminal_and_all_lanes_cleanup(expect_error):
    lanes = [_Lane(0), _Lane(1)]
    lanes[1].fail_method = "allocate"
    lanes[0].cleanup_error = RuntimeError("cleanup boom")
    group = LaneGroupExecutor(lanes)

    with expect_error(RuntimeError, "allocate boom 1") as exc_info:
        group.allocate_kv_cache()

    assert lanes[0].cleanup_calls == 1
    assert lanes[1].cleanup_calls == 1
    assert [str(error) for error in exc_info.value.cleanup_failures] == ["cleanup boom"]
    with expect_error(RuntimeError, "terminal"):
        group.allocate_kv_cache()


def test_non_null_lane_log_probs_fail_explicitly_and_cleanup_group(expect_error):
    lanes = [_Lane(0), _Lane(1)]

    def decode_with_log_probs(**kwargs):
        return torch.zeros(2, dtype=torch.int64), torch.ones(2)

    lanes[1].decode_forward = decode_with_log_probs
    group = LaneGroupExecutor(lanes)

    with expect_error(NotImplementedError, "log probabilities"):
        group.decode_forward(
            tokens=torch.tensor([0, 1, 2, 3]),
            start_pos=torch.tensor([0, 0, 0, 0]),
            page_table=torch.zeros((4, 1), dtype=torch.int32),
            sampling_params=_sampling(),
        )

    assert [lane.cleanup_calls for lane in lanes] == [1, 1]


def test_cleanup_is_idempotent_and_terminal(expect_error):
    lanes = [_Lane(0), _Lane(1)]
    group = LaneGroupExecutor(lanes)

    group.cleanup()
    group.cleanup()

    assert [lane.cleanup_calls for lane in lanes] == [1, 1]
    with expect_error(RuntimeError, "terminal"):
        group.prefill_forward(
            tokens=torch.zeros((1, 1), dtype=torch.long),
            page_table=torch.zeros((1, 1), dtype=torch.int32),
        )


def test_cleanup_retries_only_lanes_that_failed_cleanup_and_stays_terminal(expect_error):
    class FailOnceCleanupLane(_Lane):
        def cleanup(self):
            self.cleanup_calls += 1
            if self.cleanup_calls == 1:
                raise RuntimeError("lane cleanup failed once")

    failing_lane = FailOnceCleanupLane(0)
    successful_lane = _Lane(1)
    group = LaneGroupExecutor([failing_lane, successful_lane])

    with expect_error(RuntimeError, "lane cleanup failed once"):
        group.cleanup()

    assert group.terminal
    assert [failing_lane.cleanup_calls, successful_lane.cleanup_calls] == [1, 1]
    with expect_error(RuntimeError, "terminal"):
        group.allocate_kv_cache()

    group.cleanup()
    group.cleanup()

    assert group.terminal
    assert [failing_lane.cleanup_calls, successful_lane.cleanup_calls] == [2, 1]


def test_cleanup_retries_failed_pool_shutdown_without_recleaning_lanes(expect_error):
    class FailOnceShutdownPool:
        def __init__(self, pool):
            self.pool = pool
            self.shutdown_calls = 0

        def submit(self, *args, **kwargs):
            return self.pool.submit(*args, **kwargs)

        def shutdown(self, *, wait):
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise RuntimeError("pool shutdown failed once")
            self.pool.shutdown(wait=wait)

    lanes = [_Lane(0), _Lane(1)]
    group = LaneGroupExecutor(lanes)
    assert group._output_pool is not None
    pool = FailOnceShutdownPool(group._output_pool)
    group._output_pool = pool

    with expect_error(RuntimeError, "pool shutdown failed once"):
        group.cleanup()

    assert group.terminal
    assert [lane.cleanup_calls for lane in lanes] == [1, 1]

    group.cleanup()
    group.cleanup()

    assert pool.shutdown_calls == 2
    assert [lane.cleanup_calls for lane in lanes] == [1, 1]


def test_explicit_mesh_device_is_preserved_by_identity():
    mesh_device = object()
    group = LaneGroupExecutor([_Lane(0), _Lane(1)], mesh_device=mesh_device)

    assert group.mesh_device is mesh_device

    group.cleanup()
