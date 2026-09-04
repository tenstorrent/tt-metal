# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.common.tests.demos import run_helpers
from models.common.tests.demos.run_helpers import run_perf_benchmark, run_teacher_forcing


class FakeProfiler:
    def __init__(self):
        self.events = []

    def start(self, name):
        self.events.append(("start", name))

    def end(self, name):
        self.events.append(("end", name))


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

    def _record(self, method_name, arguments):
        self.calls.append(
            (
                method_name,
                {name: value for name, value in arguments.items() if name != "self"},
            )
        )

    @property
    def _engine(self):
        raise AssertionError("execution helpers must not inspect a private wrapped engine")

    def compile_prefill(
        self,
        *,
        tokens,  # ↓ Core request
        page_table,
        prompt_lens=None,  # ↓ Sequence metadata
        start_pos=None,
        empty_slots=None,  # ↓ Lane routing
        kv_cache=None,  # ↓ Borrowed resources
        sampling_params=None,  # ↓ Sampling
        execution=None,  # ↓ Internal dispatch
    ):
        self._record("compile_prefill", locals())
        return self.compile_prefill_output

    def compile_decode(
        self,
        *,
        tokens,  # ↓ Core request
        start_pos,
        page_table,
        kv_cache=None,  # ↓ Borrowed resources
        sampling_params=None,  # ↓ Sampling
        reset_batch=False,  # ↓ State transition
        execution=None,  # ↓ Internal dispatch
    ):
        self._record("compile_decode", locals())

    def prefill_forward(
        self,
        tokens,
        page_table,
        *,
        prompt_lens=None,  # ↓ Sequence metadata
        start_pos=None,
        empty_slots=None,  # ↓ Lane routing
        kv_cache=None,  # ↓ Borrowed resources
        sampling_params=None,  # ↓ Sampling
        execution=None,  # ↓ Internal dispatch
    ):
        self._record("prefill_forward", locals())
        return self.prefill_output

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        *,
        kv_cache=None,  # ↓ Borrowed resources
        sampling_params=None,  # ↓ Sampling
        reset_batch=False,  # ↓ State transition
        read_from_device=True,  # ↓ Output policy
        execution=None,  # ↓ Internal dispatch
    ):
        self._record("decode_forward", locals())
        return self.decode_outputs.pop(0)


def test_compile_only_helper_ignores_compiled_prefill_programs():
    target = FakeExecutionTarget(
        compile_prefill_output=(object(),),
        prefill_output=None,
        decode_outputs=[],
    )

    run_helpers._compile_prefill_and_decode(
        target,
        prefill_tokens=torch.tensor([[1, 2], [3, 4]]),
        prefill_page_table=torch.zeros(2, 1, dtype=torch.int32),
    )

    assert [name for name, _ in target.calls] == ["compile_prefill", "compile_decode"]
    assert torch.equal(target.calls[1][1]["tokens"], torch.zeros(2, dtype=torch.long))


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


def test_teacher_forcing_times_prefill_excludes_first_decode_and_brackets_profiler(monkeypatch):
    target = FakeExecutionTarget(
        compile_prefill_output=_logits([3, 4]),
        prefill_output=_logits([3, 4]),
        decode_outputs=[(_logits([5, 1]), None), (_logits([6, 2]), None)],
    )
    profiler = FakeProfiler()
    times = iter([0.0, 0.2, 1.0, 1.1, 2.0, 2.25])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))

    result = run_teacher_forcing(
        executor=target,
        prompt_tokens=torch.tensor([[1, 2], [1, 2]]),
        reference_tokens=torch.tensor([1, 2, 8, 9, 10]),
        top5_tokens=torch.tensor([[3, 0, 1, 2, 4], [5, 0, 1, 2, 3], [6, 0, 1, 2, 3]]),
        kv_cache=[],
        page_table=torch.zeros(2, 1, dtype=torch.int32),
        max_batch_size=2,
        profiler=profiler,
    )

    assert result.predicted_tokens_per_user == [[3, 5, 6], [4, 1, 2]]
    assert result.prefill_time_s == pytest.approx(0.2)
    assert result.compile_decode_time_s == pytest.approx(0.1)
    assert result.decode_times_s == pytest.approx([0.25])
    assert result.ttft_ms == pytest.approx(100.0)
    assert result.prefill_tok_s == pytest.approx(20.0)
    assert result.decode_tok_s_u == pytest.approx(4.0)
    assert result.decode_tok_s == pytest.approx(8.0)
    assert profiler.events == [
        ("start", "inference_prefill"),
        ("end", "inference_prefill"),
        ("start", "inference_decode"),
        ("end", "inference_decode"),
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
    assert result.decode_iteration_times_s == pytest.approx([0.25, 0.3])
    assert result.generated_token_ids == [[2, 3, 4, 5]]


def test_perf_benchmark_slices_prefill_page_rows_but_keeps_full_decode_capacity(monkeypatch):
    target = FakeExecutionTarget(
        compile_prefill_output=_logits([2, 2, 2, 2]),
        prefill_output=_logits([2, 2]),
        decode_outputs=[(_logits([3, 3, 0, 0]), None)],
    )
    times = iter([0.0, 0.1, 1.0, 1.1])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))
    page_table = torch.arange(8, dtype=torch.int32).reshape(4, 2)

    run_perf_benchmark(
        target,
        tokens=torch.tensor([[1, 2], [1, 2]]),
        kv_cache=[],
        page_table=page_table,
        num_decode_tokens=1,
        max_batch_size=4,
    )

    calls = {name: arguments for name, arguments in target.calls}
    assert calls["compile_prefill"]["page_table"].shape[0] == 2
    torch.testing.assert_close(calls["prefill_forward"]["page_table"], page_table[:2])
    torch.testing.assert_close(calls["decode_forward"]["page_table"], page_table)


def test_perf_benchmark_brackets_profiler_without_changing_host_argmax(monkeypatch):
    target = FakeExecutionTarget(
        compile_prefill_output=_logits([2]),
        prefill_output=_logits([2]),
        decode_outputs=[(_logits([3]), None), (_logits([4]), None)],
    )
    profiler = FakeProfiler()
    times = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.25])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))

    result = run_perf_benchmark(
        target,
        tokens=torch.tensor([[1, 2]]),
        kv_cache=[],
        page_table=torch.zeros(1, 1, dtype=torch.int32),
        num_decode_tokens=2,
        profiler=profiler,
    )

    assert result.generated_token_ids == [[2, 3, 4]]
    assert result.prefill_time_s == pytest.approx(0.1)
    assert result.compile_decode_time_s == pytest.approx(0.2)
    assert result.decode_times_s == pytest.approx([0.25])
    assert profiler.events == [
        ("start", "inference_prefill"),
        ("end", "inference_prefill"),
        ("start", "inference_decode"),
        ("end", "inference_decode"),
    ]


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
        self.pending_events = []

    def read_decode_output(self, tt_out, *, async_read=False):
        assert async_read
        event = self.next_event
        self.next_event += 1
        self.pending_events.append(event)
        self.calls.append(("read_decode_output", {"event": event}))
        return tt_out, [event]

    def process_decode_output_host(self, tt_out, *, is_tokens=False):
        assert is_tokens
        run_helpers.ttnn.event_synchronize(self.pending_events.pop(0))
        self.calls.append(("process_decode_output_host", {}))
        return tt_out


def test_perf_benchmark_uses_public_async_readback_without_trace_introspection(monkeypatch):
    target = PublicReadbackTarget()
    profiler = FakeProfiler()
    synchronized_events = []
    times = iter([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 3.4])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(run_helpers.ttnn, "synchronize_device", lambda mesh: None, raising=False)
    monkeypatch.setattr(run_helpers.ttnn, "event_synchronize", synchronized_events.append, raising=False)

    result = run_perf_benchmark(
        target,
        tokens=torch.tensor([[1, 2]]),
        kv_cache=[],
        page_table=torch.zeros(1, 1, dtype=torch.int32),
        num_decode_tokens=3,
        sampling_params=object(),
        pipeline_readback=True,
        profiler=profiler,
    )

    assert [name for name, _ in target.calls[:2]] == ["compile_decode", "compile_prefill"]
    # The benchmark owns the device-paced wait; the public normalizer may
    # defensively observe the already-completed event again while retiring it.
    assert synchronized_events == [0, 0, 1, 1, 2, 2]
    assert result.generated_token_ids == [[2, 3, 4, 5]]
    assert len(result.decode_times_s) == 2
    assert result.decode_iteration_times_s == pytest.approx([0.1, 0.1])
    assert profiler.events == [
        ("start", "inference_prefill"),
        ("end", "inference_prefill"),
        ("start", "inference_decode"),
        ("end", "inference_decode"),
    ]


def test_perf_benchmark_does_not_reprocess_blocking_sampled_output(monkeypatch):
    target = PublicReadbackTarget()
    times = iter([0.0, 0.1, 1.0, 1.1])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(run_helpers.ttnn, "synchronize_device", lambda mesh: None, raising=False)

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


def test_perf_benchmark_can_use_host_prefill_with_sampled_decode(monkeypatch):
    target = FakeExecutionTarget(
        compile_prefill_output=_logits([2]),
        prefill_output=_logits([2]),
        decode_outputs=[(torch.tensor([3]), None)],
    )
    times = iter([0.0, 0.1, 1.0, 1.1])
    monkeypatch.setattr(run_helpers.time, "perf_counter", lambda: next(times))

    result = run_perf_benchmark(
        target,
        tokens=torch.tensor([[1, 2]]),
        kv_cache=[],
        page_table=torch.zeros(1, 1, dtype=torch.int32),
        num_decode_tokens=1,
        sampling_params=object(),
        prefill_sampling_params=None,
    )

    calls = {name: arguments for name, arguments in target.calls}
    assert calls["compile_prefill"]["sampling_params"] is None
    assert calls["prefill_forward"]["sampling_params"] is None
    assert calls["compile_decode"]["sampling_params"] is not None
    assert calls["decode_forward"]["sampling_params"] is not None
    assert result.generated_token_ids == [[2, 3]]


def test_composite_target_synchronizes_each_lane_mesh(monkeypatch):
    target = SimpleNamespace(mesh_device="parent", mesh_devices=("lane-0", "lane-1"))
    synchronized = []
    monkeypatch.setattr(
        run_helpers.ttnn,
        "synchronize_device",
        lambda mesh_device: synchronized.append(mesh_device),
        raising=False,
    )

    run_helpers._synchronize_target(target)

    assert synchronized == ["lane-0", "lane-1"]


def test_special_token_guard_truncates_at_stop_tokens_warns_locally_and_fails_in_ci(monkeypatch, expect_error):
    tokenizer = SimpleNamespace(
        all_special_ids=[0, 1, 2, 99],
        eos_token_id=2,
        stop_tokens=[3],
        convert_tokens_to_ids=lambda token: 99 if token == "<|eot_id|>" else -1,
    )
    generated_token_ids = [
        [5, 2, 0],
        [5, 3, 0],
        [6, 99, 1],
        [7, 0, 8],
        [9, 1, 2],
    ]
    warnings = []
    monkeypatch.setattr(run_helpers.logger, "warning", warnings.append)

    run_helpers.assert_no_special_tokens(generated_token_ids, tokenizer, case_name="batch-1", is_ci_env=False)

    assert warnings == ["[batch-1] model produced special tokens (2/5 users)"]
    with expect_error(AssertionError, "2/5 users"):
        run_helpers.assert_no_special_tokens(generated_token_ids, tokenizer, case_name="batch-1", is_ci_env=True)


def test_special_token_guard_without_stop_tokens_keeps_generated_tail_visible(expect_error):
    tokenizer = SimpleNamespace(
        all_special_ids=[0],
        eos_token_id=2,
        convert_tokens_to_ids=lambda _token: -1,
    )

    with expect_error(AssertionError, "1/1 users"):
        run_helpers.assert_no_special_tokens([[5, 3, 0]], tokenizer, is_ci_env=True)


def test_eval_repeat_compares_decoded_text_like_tttv1_despite_different_bpe_segmentations():
    pieces = {10: "12", 11: "3", 12: "1", 13: "23", 99: "<eos>", 77: "ignored"}
    tokenizer = SimpleNamespace(decode=lambda token_ids: "".join(pieces[token] for token in token_ids))

    first_segmentation = run_helpers.decode_eval_output(tokenizer, [10, 11, 99, 77], {99})
    second_segmentation = run_helpers.decode_eval_output(tokenizer, [12, 13], {99})

    assert first_segmentation == second_segmentation == "123"
    # Repeat 1 rotates prompt 1 into slot 0 and prompt 0 into slot 1.
    run_helpers.assert_cross_batch_consistency(
        [
            [first_segmentation, "456"],
            ["456", second_segmentation],
        ]
    )


def test_eval_page_table_ab_preserves_slots_or_prompt_physical_blocks(expect_error):
    page_table = torch.tensor([[0, 1], [10, 11], [20, 21]], dtype=torch.int32)

    assert run_helpers.eval_page_table_for_repeat(page_table, 2, mode="slot-stable") is page_table
    torch.testing.assert_close(
        run_helpers.eval_page_table_for_repeat(page_table, 1, mode="prompt-stable"),
        torch.tensor([[10, 11], [20, 21], [0, 1]], dtype=torch.int32),
    )
    with expect_error(ValueError, "slot-stable.*prompt-stable"):
        run_helpers.eval_page_table_for_repeat(page_table, 0, mode="unsupported")


def test_eval_decode_ab_defaults_to_trace_and_can_isolate_eager_execution(expect_error):
    assert run_helpers.eval_decode_trace_mode("traced") == "decode_only"
    assert run_helpers.eval_decode_trace_mode("eager") == "none"
    with expect_error(ValueError, "traced.*eager"):
        run_helpers.eval_decode_trace_mode("unsupported")


@pytest.mark.parametrize(
    "override",
    [
        {"EVAL_DECODE_MODE": "eager"},
        {"EVAL_PAGE_TABLE_MODE": "prompt-stable"},
        {"EVAL_IDENTICAL_PROMPT_INDEX": "8"},
        {"EVAL_ACTIVE_BATCH_SIZE": "24"},
    ],
)
def test_ci_rejects_diagnostic_eval_modes(override, expect_error):
    with expect_error(RuntimeError, "diagnostic eval modes cannot replace the canonical CI gate"):
        run_helpers.require_canonical_eval_modes_in_ci({"CI": "true", **override})


def test_non_ci_diagnostics_and_canonical_ci_are_allowed():
    run_helpers.require_canonical_eval_modes_in_ci({"EVAL_DECODE_MODE": "eager", "EVAL_IDENTICAL_PROMPT_INDEX": "8"})
    run_helpers.require_canonical_eval_modes_in_ci(
        {"CI": "true", "EVAL_DECODE_MODE": "traced", "EVAL_PAGE_TABLE_MODE": "slot-stable"}
    )


def test_host_argmax_diagnostic_reports_top1_minus_top2_margin():
    logits = torch.tensor(
        [
            [[0.0, 4.0, 3.5]],
            [[9.0, 1.0, 8.875]],
        ]
    )

    tokens, margins = run_helpers._host_argmax_with_margins(logits, 2)

    torch.testing.assert_close(tokens, torch.tensor([1, 0]))
    torch.testing.assert_close(margins, torch.tensor([0.5, 0.125]))


def test_eval_repeat_driver_canonicalizes_each_rotated_output_through_tokenizer(monkeypatch):
    pieces = {10: "12", 11: "3", 12: "1", 13: "23", 20: "456"}
    tokenizer = SimpleNamespace(
        decode=lambda token_ids: "".join(pieces[token] for token in token_ids),
        eos_token_id=None,
        stop_tokens=[],
        all_special_ids=[],
    )
    generated = iter(
        [
            [[10, 11], [20]],
            [[20], [12, 13]],
        ]
    )
    monkeypatch.setattr(
        run_helpers,
        "run_perf_benchmark",
        lambda *_args, **_kwargs: SimpleNamespace(generated_token_ids=next(generated)),
    )

    class FakeEvalExecutor:
        def cleanup(self):
            pass

    result = run_helpers.run_eval_repeat_batch32(
        make_executor=FakeEvalExecutor,
        allocate_kv_cache=lambda _executor: [],
        page_table=torch.zeros(2, 1, dtype=torch.int32),
        prompts=["prompt-0", "prompt-1"],
        tokenizer=tokenizer,
        tokenize_fn=lambda prompts: (torch.zeros(len(prompts), 1, dtype=torch.long), torch.ones(len(prompts))),
        num_decode_tokens=2,
        max_batch_size=2,
        repeat_batches=2,
    )

    assert result.generated_token_ids == [[10, 11], [20]]


def test_eval_repeat_still_rejects_genuine_decoded_text_divergence(expect_error):
    with expect_error(AssertionError, "1/2 cross-batch consistency checks failed"):
        run_helpers.assert_cross_batch_consistency(
            [
                ["123", "456"],
                ["different", "123"],
            ]
        )


def test_eval_repeat_failure_localizes_prompt_slots_and_first_token_divergence(expect_error):
    with expect_error(
        AssertionError,
        r"repeat 0 slot 1 -> repeat 1 slot 0, prompt index 1; "
        r"first token divergence at generation step 1 \(12 != 99\); "
        r"top2 margins 0.125 and 0.25; prompt lengths 80 and 80",
    ):
        run_helpers.assert_cross_batch_consistency(
            [
                ["alpha", "beta-left"],
                ["beta-right", "alpha"],
            ],
            per_repeat_token_ids=[
                [[1], [11, 12, 13]],
                [[11, 99, 13], [1]],
            ],
            per_repeat_prompt_lens=[
                [64, 80],
                [80, 64],
            ],
            per_repeat_argmax_margins=[
                [[1.0], [0.5, 0.125, 0.75]],
                [[0.5, 0.25, 0.75], [1.0]],
            ],
        )


def test_identical_request_diagnostic_localizes_logical_slot_divergence(expect_error):
    with expect_error(
        AssertionError,
        r"prompt index 8 differs between logical slots 0 and 1 at generation step 2 "
        r"\(13 != 99\); top2 margins 0.125 and 0.25",
    ):
        run_helpers.assert_within_batch_slot_consistency(
            ["same prefix left", "same prefix right"],
            token_ids=[[11, 12, 13], [11, 12, 99]],
            argmax_margins=[[1.0, 0.5, 0.125], [1.0, 0.5, 0.25]],
            prompt_index=8,
        )


def test_identical_request_diagnostic_accepts_slot_invariant_decoded_text():
    run_helpers.assert_within_batch_slot_consistency(
        ["same", "same", "same"],
        token_ids=[[1], [1], [1]],
        argmax_margins=[[0.5], [0.5], [0.5]],
        prompt_index=8,
    )


def test_identical_request_driver_uses_one_fixed_prompt_and_needs_no_repeat(monkeypatch):
    seen_prompts = []
    tokenizer = SimpleNamespace(
        decode=lambda token_ids: "same",
        eos_token_id=None,
        stop_tokens=[],
        all_special_ids=[],
    )
    monkeypatch.setattr(
        run_helpers,
        "run_perf_benchmark",
        lambda *_args, **_kwargs: SimpleNamespace(
            generated_token_ids=[[1]],
            argmax_top2_margins=[[0.5]],
        ),
    )

    class FakeEvalExecutor:
        def cleanup(self):
            pass

    run_helpers.run_eval_repeat_batch32(
        make_executor=FakeEvalExecutor,
        allocate_kv_cache=lambda _executor: [],
        page_table=torch.zeros(2, 1, dtype=torch.int32),
        prompts=["prompt-0", "prompt-1"],
        tokenizer=tokenizer,
        tokenize_fn=lambda prompts: (
            seen_prompts.append(prompts) or torch.zeros(len(prompts), 1, dtype=torch.long),
            torch.ones(len(prompts)),
        ),
        num_decode_tokens=1,
        max_batch_size=2,
        repeat_batches=1,
        identical_prompt_index=1,
        active_batch_size=1,
    )

    assert seen_prompts == [["prompt-1"]]


def test_active_batch_diagnostic_requires_identical_request(expect_error):
    with expect_error(ValueError, "active_batch_size requires identical_prompt_index"):
        run_helpers.run_eval_repeat_batch32(
            make_executor=lambda: None,
            allocate_kv_cache=lambda _executor: [],
            page_table=torch.zeros(2, 1, dtype=torch.int32),
            prompts=["prompt-0", "prompt-1"],
            tokenizer=SimpleNamespace(eos_token_id=None, stop_tokens=[], all_special_ids=[]),
            tokenize_fn=lambda prompts: (torch.zeros(len(prompts), 1), torch.ones(len(prompts))),
            num_decode_tokens=1,
            max_batch_size=2,
            repeat_batches=1,
            active_batch_size=1,
        )


def test_cross_cardinality_consistency_accepts_fixed_request_prefixes():
    run_helpers.assert_cross_cardinality_consistency(
        {
            1: {"r0": "a"},
            2: {"r0": "a", "r1": "b"},
            4: {"r0": "a", "r1": "b", "r2": "c", "r3": "d"},
            32: {f"r{i}": chr(97 + i) for i in range(32)},
        }
    )


def test_cross_cardinality_consistency_reports_request_and_cardinalities(expect_error):
    with expect_error(AssertionError, "request 'r0' differs at cardinality 1->2"):
        run_helpers.assert_cross_cardinality_consistency(
            {1: {"r0": "same"}, 2: {"r0": "different", "r1": "x"}},
            expected_cardinalities=(1, 2),
        )


def test_loop_policy_is_not_exported_from_production_executor():
    try:
        from models.common.llm_runtime import execution as production_executor
    except AttributeError as exc:
        pytest.skip(f"production executor import requires full ttnn runtime: {exc}")

    for name in ("TeacherForceResult", "PerfBenchmarkResult", "run_teacher_forcing", "run_perf_benchmark"):
        assert not hasattr(production_executor, name)
