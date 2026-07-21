# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

import models.common.llm_runtime.graph_compiler as graph_compiler_module
import ttnn
from models.common.llm_runtime.graph_compiler import (
    GraphKey,
    GraphState,
    InputRefreshPolicy,
    LLMGraphCompiler,
    OutputSpec,
    TraceCapturePlan,
)


@dataclass(frozen=True)
class _TraceConfig:
    mode: str

    def enables(self, graph_mode):
        return self.mode == "all" or self.mode == "decode_only" and graph_mode == "decode"


def _key(*, mode="decode", batch=4, page_width=8, sampling="logits", seq_len=None, chunk_width=None):
    return GraphKey(
        mode=mode,
        batch_size=batch,
        page_table_width=page_width,
        sampling_path=sampling,
        sequence_length=seq_len,
        chunk_page_table_width=chunk_width,
    )


def _patch_trace_backend(monkeypatch, events):
    next_id = iter(range(100, 200))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append(("sync", mesh)))
    monkeypatch.setattr(
        ttnn,
        "begin_trace_capture",
        lambda mesh, cq_id: events.append(("begin", mesh, cq_id)) or next(next_id),
    )
    monkeypatch.setattr(
        ttnn,
        "end_trace_capture",
        lambda mesh, trace_id, cq_id: events.append(("end", trace_id, cq_id)),
    )
    monkeypatch.setattr(
        ttnn,
        "execute_trace",
        lambda mesh, trace_id, cq_id, blocking: events.append(("execute", trace_id, cq_id, blocking)),
    )
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, trace_id: events.append(("release", trace_id)))
    monkeypatch.setattr(graph_compiler_module, "_trim_host_allocator", lambda: events.append(("trim",)))


def test_compile_requires_bound_kv_cache(expect_error):
    compiler = LLMGraphCompiler("mesh", _TraceConfig("none"), lambda: None)
    with expect_error(RuntimeError, "allocated and bound"):
        compiler.compile(_key(), lambda context: torch.zeros(4, 1, 16))


def test_compile_receives_exact_context_and_releases_output_before_return(monkeypatch):
    events = []
    _patch_trace_backend(monkeypatch, events)
    context = object()
    seen = []
    compiler = LLMGraphCompiler("mesh", _TraceConfig("none"), lambda: context)

    graph = compiler.compile(
        _key(),
        lambda supplied: seen.append(supplied) or torch.zeros(4, 1, 16, dtype=torch.bfloat16),
    )

    assert seen == [context]
    assert graph.output_spec.shape == (4, 1, 16)
    assert graph.output_spec.dtype is torch.bfloat16
    assert graph.state is GraphState.COMPILED
    assert events == [("sync", "mesh"), ("sync", "mesh")]


def test_output_spec_reads_tensor_spec_without_calling_memory_config(monkeypatch):
    class Tensor:
        shape = (4, 1, 16)
        dtype = "bfloat16"
        layout = "tile"
        spec = type("Spec", (), {"memory_config": "dram"})()

        def __init__(self):
            self.memory_config_calls = 0

        def is_allocated(self):
            return True

        def memory_config(self):
            self.memory_config_calls += 1
            return "dram"

    monkeypatch.setattr(ttnn, "Tensor", Tensor)
    tensor = Tensor()

    output_spec = OutputSpec.from_value(tensor)

    assert output_spec.memory_config == "dram"
    assert tensor.memory_config_calls == 0


def test_canonical_keys_include_all_required_identity_fields():
    base = _key(mode="prefill", seq_len=128, batch=1, page_width=4, sampling="topk")
    assert base != _key(mode="prefill", seq_len=1024, batch=1, page_width=4, sampling="topk")
    assert base != _key(mode="prefill", seq_len=128, batch=2, page_width=4, sampling="topk")
    assert base != _key(mode="prefill", seq_len=128, batch=1, page_width=8, sampling="topk")
    assert base != _key(mode="prefill", seq_len=128, batch=1, page_width=4, chunk_width=2, sampling="topk")
    assert base != _key(mode="prefill", seq_len=128, batch=1, page_width=4, sampling="logits")


def test_capture_prepares_every_graph_before_first_trace_and_seals_compilation(monkeypatch, expect_error):
    events = []
    _patch_trace_backend(monkeypatch, events)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("all"), lambda: object())
    keys = [
        _key(mode="prefill", seq_len=128, batch=1),
        _key(mode="decode", batch=4, sampling="argmax"),
    ]
    for key in keys:
        compiler.compile(key, lambda context: torch.zeros(1))
    events.clear()

    plans = {
        key: TraceCapturePlan(
            prepare_inputs=lambda key=key: events.append(("prepare", key)) or (key,),
            capture=lambda persistent, key=key: events.append(("capture", key)) or ("output", key),
        )
        for key in keys
    }
    compiler.capture_all(plans)

    first_begin = next(i for i, event in enumerate(events) if event[0] == "begin")
    assert [event[0] for event in events[:first_begin]] == ["prepare", "prepare"]
    for end_index in (i for i, event in enumerate(events) if event[0] == "end"):
        assert events[end_index + 1] == ("sync", "mesh")
    assert events[-1] == ("trim",)
    assert all(compiler.get(key).state is GraphState.CAPTURED for key in keys)
    assert compiler.trace_active
    with expect_error(RuntimeError, "after trace activation"):
        compiler.compile(_key(mode="decode", page_width=16), lambda context: torch.zeros(1))


def test_later_capture_failure_releases_each_prepared_input_once(monkeypatch, expect_error):
    events = []
    _patch_trace_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    deallocated = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)

    compiler = LLMGraphCompiler("mesh", _TraceConfig("all"), lambda: object())
    first_key = _key(mode="prefill", seq_len=128, batch=1)
    second_key = _key(mode="decode", batch=4, sampling="argmax")
    for key in (first_key, second_key):
        compiler.compile(key, lambda context: torch.zeros(1))

    first_input = OwnedTensor()
    second_input = OwnedTensor()
    first_output = OwnedTensor()

    with expect_error(RuntimeError, "second capture failed"):
        compiler.capture_all(
            {
                first_key: TraceCapturePlan(lambda: first_input, lambda persistent: first_output),
                second_key: TraceCapturePlan(
                    lambda: second_input,
                    lambda persistent: (_ for _ in ()).throw(RuntimeError("second capture failed")),
                ),
            }
        )

    assert deallocated.count(first_input) == 1
    assert deallocated.count(first_output) == 1
    assert deallocated.count(second_input) == 1
    assert compiler.get(first_key).state is GraphState.COMPILED
    assert compiler.get(second_key).state is GraphState.COMPILED
    assert not compiler.trace_active


def test_replay_refreshes_on_reset_feedback_loss_page_change_and_key_switch(monkeypatch):
    events = []
    _patch_trace_backend(monkeypatch, events)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("decode_only"), lambda: object())
    keys = [_key(sampling="argmax"), _key(sampling="topk")]
    for key in keys:
        compiler.compile(key, lambda context: torch.zeros(1))
    policy = InputRefreshPolicy(every_replay=("position", "sampling"))
    compiler.capture_all(
        {
            key: TraceCapturePlan(lambda key=key: (key,), lambda persistent, key=key: ("output", key), policy)
            for key in keys
        }
    )

    decisions = []
    compiler.replay(
        keys[0],
        lambda artifact, decision: decisions.append(decision),
        reset_batch=True,
        device_feedback_enabled=True,
        feedback_compatible=True,
    )
    compiler.replay(
        keys[0],
        lambda artifact, decision: decisions.append(decision),
        device_feedback_enabled=True,
        feedback_compatible=True,
        page_table_changed=True,
    )
    compiler.replay(
        keys[1],
        lambda artifact, decision: decisions.append(decision),
        device_feedback_enabled=True,
        feedback_compatible=True,
    )
    compiler.replay(
        keys[1],
        lambda artifact, decision: decisions.append(decision),
        device_feedback_enabled=False,
        feedback_compatible=False,
    )

    assert [decision.full for decision in decisions] == [True, False, True, True]
    assert decisions[1].page_table is True
    assert all(decision.fields == ("position", "sampling") for decision in decisions)


def test_cleanup_releases_each_trace_once_and_is_terminal(monkeypatch, expect_error):
    events = []
    _patch_trace_backend(monkeypatch, events)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("decode_only"), lambda: object())
    key = _key()
    compiler.compile(key, lambda context: torch.zeros(1))
    compiler.capture_all({key: TraceCapturePlan(lambda: (), lambda persistent: torch.zeros(1))})
    events.clear()

    compiler.cleanup()
    compiler.cleanup()

    assert [event[0] for event in events] == ["release"]
    assert compiler.get(key).state is GraphState.RELEASED
    with expect_error(RuntimeError, "released"):
        compiler.assert_executable(key)


def test_capture_seals_compilation_while_callbacks_are_running(monkeypatch, expect_error):
    events = []
    _patch_trace_backend(monkeypatch, events)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("all"), lambda: object())
    key = _key()
    compiler.compile(key, lambda context: torch.zeros(1))

    def capture(persistent):
        with expect_error(RuntimeError, "after trace activation"):
            compiler.compile(_key(page_width=16), lambda context: torch.zeros(1))
        return torch.zeros(1)

    compiler.capture_all({key: TraceCapturePlan(lambda: (), capture)})
    assert compiler.trace_active


def test_capture_skips_compiled_trace_ineligible_fallback(monkeypatch):
    events = []
    _patch_trace_backend(monkeypatch, events)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("all"), lambda: object())
    traced_key = _key()
    fallback_key = _key(mode="prefill", seq_len=1024, chunk_width=4)
    compiler.compile(traced_key, lambda context: torch.zeros(1))
    compiler.compile(fallback_key, lambda context: torch.zeros(1), trace_eligible=False)

    compiler.capture_all({traced_key: TraceCapturePlan(lambda: (), lambda persistent: torch.zeros(1))})

    assert compiler.get(traced_key).state is GraphState.CAPTURED
    assert compiler.get(fallback_key).state is GraphState.COMPILED
    assert compiler.assert_executable(fallback_key).trace is None


def test_end_capture_failure_releases_trace_before_owned_output(monkeypatch, expect_error):
    events = []
    _patch_trace_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    deallocated = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    monkeypatch.setattr(
        ttnn,
        "end_trace_capture",
        lambda mesh, trace_id, cq_id: (_ for _ in ()).throw(RuntimeError("end failed")),
    )

    compiler = LLMGraphCompiler("mesh", _TraceConfig("all"), lambda: object())
    key = _key()
    compiler.compile(key, lambda context: torch.zeros(1))
    persistent = OwnedTensor()
    output = OwnedTensor()

    with expect_error(RuntimeError, "end failed"):
        compiler.capture_all({key: TraceCapturePlan(lambda: persistent, lambda values: output)})

    assert deallocated.count(persistent) == 1
    assert deallocated.count(output) == 1
    assert compiler.get(key).trace is None
    assert compiler.get(key).state is GraphState.COMPILED
    assert not compiler.trace_active


def test_cleanup_retries_release_before_deallocating_live_trace_resources(monkeypatch, expect_error):
    events = []
    _patch_trace_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    deallocated = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    release_attempts = []

    def fail_once(mesh, trace_id):
        release_attempts.append(trace_id)
        if len(release_attempts) == 1:
            raise RuntimeError("release failed once")

    monkeypatch.setattr(ttnn, "release_trace", fail_once)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("decode_only"), lambda: object())
    key = _key()
    compiler.compile(key, lambda context: torch.zeros(1))
    persistent = OwnedTensor()
    output = OwnedTensor()
    compiler.capture_all({key: TraceCapturePlan(lambda: persistent, lambda values: output)})

    with expect_error(RuntimeError, "Failed to release"):
        compiler.cleanup()

    assert compiler.get(key).trace is not None
    assert compiler.get(key).state is GraphState.CAPTURED
    assert deallocated == []

    compiler.cleanup()

    assert len(release_attempts) == 2
    assert deallocated.count(persistent) == 1
    assert deallocated.count(output) == 1
    assert compiler.get(key).state is GraphState.RELEASED


def test_capture_rollback_preserves_primary_and_retains_failed_release(monkeypatch, expect_error):
    events = []
    _patch_trace_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    deallocated = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    release_first_trace = False
    release_error = RuntimeError("first trace release failed")

    def release(mesh, trace_id):
        if trace_id == 100 and not release_first_trace:
            raise release_error
        events.append(("release", trace_id))

    monkeypatch.setattr(ttnn, "release_trace", release)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("all"), lambda: object())
    first_key = _key(mode="prefill", seq_len=128, batch=1)
    second_key = _key(mode="decode", batch=4, sampling="argmax")
    for key in (first_key, second_key):
        compiler.compile(key, lambda context: torch.zeros(1))

    first_input = OwnedTensor()
    first_output = OwnedTensor()
    second_input = OwnedTensor()
    primary = RuntimeError("second capture failed")

    with expect_error(RuntimeError, "second capture failed") as caught:
        compiler.capture_all(
            {
                first_key: TraceCapturePlan(lambda: first_input, lambda values: first_output),
                second_key: TraceCapturePlan(
                    lambda: second_input,
                    lambda values: (_ for _ in ()).throw(primary),
                ),
            }
        )

    assert caught.value is primary
    assert release_error in caught.value.cleanup_failures
    assert compiler.get(first_key).trace is not None
    assert compiler.trace_active
    assert first_input not in deallocated
    assert first_output not in deallocated
    assert deallocated.count(second_input) == 1

    release_first_trace = True
    compiler.cleanup()
    assert deallocated.count(first_input) == 1
    assert deallocated.count(first_output) == 1


def test_compile_first_sync_failure_cleans_owned_output_and_preserves_primary(monkeypatch, expect_error):
    class OwnedTensor:
        pass

    first = OwnedTensor()
    second = OwnedTensor()
    primary = RuntimeError("compile sync failed")
    sync_calls = 0
    deallocated = []

    def synchronize(mesh):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise primary

    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocated.append)
    monkeypatch.setattr(ttnn, "synchronize_device", synchronize)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("none"), lambda: object())

    with expect_error(RuntimeError, "compile sync failed") as caught:
        compiler.compile(_key(), lambda context: {"first": first, "second": second})

    assert caught.value is primary
    assert sync_calls == 2
    assert deallocated.count(first) == 1
    assert deallocated.count(second) == 1
    assert compiler.get(_key()) is None


def test_compile_cleanup_retains_failed_output_and_retries_only_failed(monkeypatch, expect_error):
    class OwnedTensor:
        pass

    first = OwnedTensor()
    retry = OwnedTensor()
    attempts = []
    cleanup_error = RuntimeError("compile output deallocation failed once")

    def deallocate(value):
        attempts.append(value)
        if value is retry and attempts.count(retry) == 1:
            raise cleanup_error

    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("none"), lambda: object())

    with expect_error(RuntimeError, "Failed to deallocate") as caught:
        compiler.compile(
            _key(),
            lambda context: (first, retry),
            output_spec=lambda output: object(),
        )

    assert caught.value.cleanup_failures == (cleanup_error,)
    assert attempts == [first, retry]
    with expect_error(RuntimeError, "unreleased compile output"):
        compiler.compile(_key(page_width=16), lambda context: torch.zeros(1))
    with expect_error(RuntimeError, "unreleased compile output"):
        compiler.capture_all({})

    compiler.cleanup()

    assert attempts.count(first) == 1
    assert attempts.count(retry) == 2


def test_compile_primary_error_retains_failed_cleanup_for_retry(monkeypatch, expect_error):
    class OwnedTensor:
        pass

    first = OwnedTensor()
    retry = OwnedTensor()
    attempts = []
    primary = RuntimeError("compile synchronization failed")
    cleanup_error = RuntimeError("compile output deallocation failed once")
    sync_calls = 0

    def synchronize(mesh):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise primary

    def deallocate(value):
        attempts.append(value)
        if value is retry and attempts.count(retry) == 1:
            raise cleanup_error

    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    monkeypatch.setattr(ttnn, "synchronize_device", synchronize)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("none"), lambda: object())

    with expect_error(RuntimeError, "compile synchronization failed") as caught:
        compiler.compile(_key(), lambda context: (first, retry))

    assert caught.value is primary
    assert caught.value.cleanup_failures == (cleanup_error,)
    assert sync_calls == 2
    assert attempts == [first, retry]

    compiler.cleanup()

    assert attempts.count(first) == 1
    assert attempts.count(retry) == 2


def test_prepare_failure_retains_only_failed_orphan_and_cleanup_retries(monkeypatch, expect_error):
    events = []
    _patch_trace_backend(monkeypatch, events)

    class OwnedTensor:
        pass

    retry = OwnedTensor()
    sibling = OwnedTensor()
    attempts = []
    cleanup_error = RuntimeError("deallocate failed once")

    def deallocate(value):
        attempts.append(value)
        if value is retry and attempts.count(retry) == 1:
            raise cleanup_error

    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("all"), lambda: object())
    first_key = _key(mode="prefill", seq_len=128, batch=1)
    second_key = _key(mode="decode", batch=4, sampling="argmax")
    for key in (first_key, second_key):
        compiler.compile(key, lambda context: torch.zeros(1))

    primary = RuntimeError("second prepare failed")
    with expect_error(RuntimeError, "second prepare failed") as caught:
        compiler.capture_all(
            {
                first_key: TraceCapturePlan(lambda: (retry, sibling), lambda persistent: torch.zeros(1)),
                second_key: TraceCapturePlan(
                    lambda: (_ for _ in ()).throw(primary),
                    lambda persistent: torch.zeros(1),
                ),
            }
        )

    assert caught.value is primary
    assert caught.value.cleanup_failures == (cleanup_error,)
    assert compiler.trace_active
    assert attempts.count(retry) == 1
    assert attempts.count(sibling) == 1
    with expect_error(RuntimeError, "after trace activation"):
        compiler.compile(_key(page_width=16), lambda context: torch.zeros(1))

    compiler.cleanup()

    assert attempts.count(retry) == 2
    assert attempts.count(sibling) == 1
    assert compiler.get(first_key).state is GraphState.RELEASED
    assert compiler.get(second_key).state is GraphState.RELEASED


def test_first_replay_fully_refreshes_without_reset(monkeypatch):
    events = []
    _patch_trace_backend(monkeypatch, events)
    compiler = LLMGraphCompiler("mesh", _TraceConfig("decode_only"), lambda: object())
    key = _key()
    compiler.compile(key, lambda context: torch.zeros(1))
    compiler.capture_all(
        {
            key: TraceCapturePlan(
                lambda: (),
                lambda persistent: torch.zeros(1),
                InputRefreshPolicy(full_without_device_feedback=False),
            )
        }
    )

    decisions = []
    compiler.replay(
        key,
        lambda artifact, decision: decisions.append(decision),
        reset_batch=False,
        device_feedback_enabled=True,
        feedback_compatible=True,
    )

    assert len(decisions) == 1
    assert decisions[0].full
