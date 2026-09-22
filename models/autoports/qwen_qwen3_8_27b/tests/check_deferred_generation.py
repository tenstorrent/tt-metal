# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact output-delivery checks; reduced layers by default, full model with --full.

Prompt lengths below are shape fixtures, not qualitative evaluation. Guarded
runs prove the host boundary; unguarded repeated runs provide warmed timings.
"""

import argparse
import inspect
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder import device_only
from models.autoports.qwen_qwen3_8_27b.tt.generator import QwenGenerator, build_generator, configure_fabric
from models.common.readiness_check.schema import load_reference

FORBIDDEN_REPLAY_APIS = (
    "to_torch",
    "from_torch",
    "as_tensor",
    "copy_host_to_device_tensor",
    "synchronize_device",
)


def forbidden(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"Forbidden operation during guarded decode: {name}")

    return fail


def assert_tokens(actual, expected, count, label):
    assert len(actual) == len(expected) == count, (label, len(actual), len(expected), count)
    if actual != expected:
        index = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise AssertionError(f"{label}: output {index} differs: {actual[index]} != {expected[index]}")


def assert_perf(perf, count, *, deferred):
    steps = count - 1
    effective_deferred = deferred and steps > 0
    steady = perf["steady_state_counters"]
    total = perf["counters"]
    assert perf["decode_tokens"] == steps
    assert perf["deferred_token_readback"] is effective_deferred
    assert perf["ttft_s"] > 0 and perf["decode_s"] >= 0
    assert not perf["host_sampling"]
    for name in ("model_replays", "sampling_replays"):
        assert steady.get(name, 0) == steps, (name, steady, steps)
    for name in (
        "token_refreshes",
        "position_refreshes",
        "rope_refreshes",
        "page_table_refreshes",
        "seed_refreshes",
        "history_cursor_refreshes",
        "history_allocations",
        "trace_captures",
        "full_logits_readbacks",
        "history_readbacks",
    ):
        assert steady.get(name, 0) == 0, (name, steady)
    assert total.get("full_logits_readbacks", 0) == 0
    assert steady.get("history_appends", 0) == (steps if effective_deferred else 0)
    assert steady.get("token_readbacks", 0) == (0 if effective_deferred else steps)
    assert total.get("token_readbacks", 0) == (1 if effective_deferred else count)
    assert total.get("history_readbacks", 0) == int(effective_deferred)
    assert perf["delivery_counters"] == ({"history_readbacks": 1} if effective_deferred else {})
    if effective_deferred:
        assert perf["history_capacity"] >= steps


def guarded_generate(gen, prompt, count, **sampling):
    """Guard only warmed decode calls, allowing the first and final output reads."""
    decode = gen.decode_forward
    execute = ttnn.execute_trace
    read_tokens, read_history = gen._read_tokens, gen._read_history
    expected_traces = (gen.trace, gen.sample_trace)
    observed = dict(decode_calls=0, nonblocking_replays=0, reads=[])

    def nonblocking_execute(*args, **kwargs):
        assert kwargs.get("blocking", True) is False, "Trace replay must explicitly use blocking=False"
        result = execute(*args, **kwargs)
        observed["nonblocking_replays"] += 1
        return result

    def guarded_decode(*args, **kwargs):
        assert kwargs.get("read_from_device") is False
        assert kwargs.get("record_history") is True
        assert gen.trace is not None and gen.sample_trace is not None, "Warmed trace was invalidated"
        assert (gen.trace, gen.sample_trace) == expected_traces, "Warmed trace pair changed"
        with ExitStack() as stack:
            for name in FORBIDDEN_REPLAY_APIS:
                stack.enter_context(patch.object(ttnn, name, forbidden(name)))
            stack.enter_context(patch.object(ttnn, "execute_trace", nonblocking_execute))
            result = decode(*args, **kwargs)
        assert result is gen.tokens, "Non-reading decode must return the persistent feedback tensor"
        assert (gen.trace, gen.sample_trace) == expected_traces, "Decode recaptured a warmed trace"
        observed["decode_calls"] += 1
        return result

    def observe_read(kind, reader):
        begin = time.perf_counter()
        value = reader()
        observed["reads"].append(
            dict(kind=kind, after_decode_calls=observed["decode_calls"], seconds=time.perf_counter() - begin)
        )
        return value

    with (
        patch.object(gen, "decode_forward", guarded_decode),
        patch.object(gen, "_read_tokens", lambda: observe_read("first_token", read_tokens)),
        patch.object(gen, "_read_history", lambda: observe_read("history", read_history)),
    ):
        output = gen.generate(prompt, count, defer_token_readback=True, **sampling)
    steps = count - 1
    assert observed["decode_calls"] == steps
    assert observed["nonblocking_replays"] == 2 * steps
    expected_reads = [("first_token", 0)] + ([("history", steps)] if steps else [])
    assert [(row["kind"], row["after_decode_calls"]) for row in observed["reads"]] == expected_reads
    for row in observed["reads"]:
        metric = "ttft_s" if row["kind"] == "first_token" else "decode_s"
        assert gen.last_perf[metric] >= row["seconds"], (metric, gen.last_perf, row)
    assert_perf(gen.last_perf, count, deferred=True)
    observed["steady_state_counters"] = gen.last_perf["steady_state_counters"]
    observed["delivery_counters"] = gen.last_perf["delivery_counters"]
    return output, observed


def check_workload(gen, prompt, count):
    expected = gen.generate(prompt, count, defer_token_readback=False)
    immediate = gen.generate(prompt, count, defer_token_readback=False)
    assert_tokens(immediate, expected, count, "Repeated immediate output")
    immediate_perf = gen.last_perf
    assert_perf(immediate_perf, count, deferred=False)
    cold_deferred = gen.generate(prompt, count, defer_token_readback=True)
    assert_tokens(cold_deferred, expected, count, "Deferred warmup output")
    deferred = gen.generate(prompt, count, defer_token_readback=True)
    assert_tokens(deferred, expected, count, "Repeated deferred output")
    deferred_perf = gen.last_perf
    assert_perf(deferred_perf, count, deferred=True)
    guarded, evidence = guarded_generate(gen, prompt, count)
    assert_tokens(guarded, expected, count, "Guarded deferred output")
    return dict(
        prompt_length=len(prompt),
        generated=count,
        all_tokens_equal=True,
        tokens=deferred,
        warmed_immediate=immediate_perf,
        warmed_deferred=deferred_perf,
        guarded_replay=evidence,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (
        "defer_token_readback" in inspect.signature(QwenGenerator.generate).parameters
    ), "Apply the deferred patch first"
    root = Path("models/autoports/qwen_qwen3_8_27b")
    entry = load_reference(root / "readiness_aime24_chat.refpt").entries[0]
    raw = entry.prompt_tokens[0].tolist()

    def prompt(length):
        return (raw * ((length + len(raw) - 1) // len(raw)))[:length]

    report = dict(
        status="running",
        full=args.full,
        mesh=[1, 4],
        workloads=[],
        sampling_transitions=[],
        forbidden_replay_apis=list(FORBIDDEN_REPLAY_APIS) + ["execute_trace(blocking=True or omitted)"],
        timing_scope="Warmed timings exclude guard instrumentation; guarded runs provide correctness evidence only",
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(root, mesh, layer_indices=None if args.full else [0, 3])
        report["layers"] = gen.model.layer_indices
        append = gen._append_history
        append_calls = 0

        def guarded_append():
            nonlocal append_calls
            with (
                device_only(),
                patch.object(ttnn, "copy_host_to_device_tensor", forbidden("copy_host_to_device_tensor")),
                patch.object(ttnn, "synchronize_device", forbidden("synchronize_device")),
            ):
                append()
            append_calls += 1

        with patch.object(gen, "_append_history", guarded_append):
            for length, count in ((1, 1), (33, 2), (128, 128), (129, 129), (33, 257), (31, 128)):
                history_before = gen.token_history, gen.history_cursor, gen.history_capacity
                row = check_workload(gen, prompt(length), count)
                if (length, count) == (31, 128):
                    assert gen.token_history is history_before[0] and gen.history_cursor is history_before[1]
                    assert gen.history_capacity == history_before[2] >= 256
                    row["long_to_short_history_reused"] = True
                report["workloads"].append(row)
                save()
                print("DEFERRED_WORKLOAD_PASSED", length, count, flush=True)

            # Prepare both immediate oracles, then alternate sampling parameters
            # while retaining the deferred trace mode across whole requests.
            modes = dict(
                greedy=dict(top_k=1, top_p=0.0, temperature=1.0, seed=37),
                sampled=dict(top_k=8, top_p=0.9, temperature=0.8, seed=37),
            )
            oracles = {
                name: gen.generate(prompt(33), 16, defer_token_readback=False, **params)
                for name, params in modes.items()
            }
            for index, name in enumerate(("greedy", "sampled", "greedy", "sampled")):
                if index == 0:
                    tokens = gen.generate(prompt(33), 16, defer_token_readback=True, **modes[name])
                    guard = None  # This request establishes the recording trace.
                else:
                    tokens, guard = guarded_generate(gen, prompt(33), 16, **modes[name])
                assert_tokens(tokens, oracles[name], 16, f"Sampling transition {index}: {name}")
                report["sampling_transitions"].append(
                    dict(mode=name, parameters=modes[name], tokens=tokens, all_tokens_equal=True, guarded_replay=guard)
                )
                save()

            forced = entry.generated_tokens[0].tolist()[:8]
            assert len(forced) == 8, "The callback fixture requires eight reference tokens"
            callback_runs = []
            for defer in (False, True):
                seen = []
                before_models = gen.counters["model_replays"]

                def next_input(step, predicted):
                    assert isinstance(predicted, int)
                    assert gen.counters["model_replays"] - before_models == step, "Callback was delayed past its step"
                    seen.append((step, predicted))
                    return forced[step]

                tokens = gen.generate(prompt(33), 8, defer_token_readback=defer, next_input=next_input, seed=23)
                assert seen == list(enumerate(tokens)), "Callbacks must observe each returned prediction immediately"
                assert len(tokens) == 8
                perf = gen.last_perf
                assert perf["teacher_forcing"] and not perf["deferred_token_readback"]
                assert perf["counters"].get("token_readbacks", 0) == 8
                assert perf["counters"].get("history_readbacks", 0) == 0
                assert perf["steady_state_counters"].get("token_refreshes", 0) == 6
                assert perf["steady_state_counters"].get("history_appends", 0) == 0
                callback_runs.append(dict(requested_defer=defer, callbacks=seen, tokens=tokens, perf=perf))
            assert_tokens(callback_runs[1]["tokens"], callback_runs[0]["tokens"], 8, "Automatic immediate callbacks")
            report["callbacks"] = dict(automatic_immediate=True, all_tokens_equal=True, runs=callback_runs)

            counters_before = gen.counters.copy()
            traces_before = gen.trace, gen.sample_trace
            history_count_before = gen.history_count
            with patch.object(gen, "decode_forward", forbidden("decode for zero tokens")):
                assert gen.generate(prompt(33), 0) == []
            assert gen.counters == counters_before
            assert (gen.trace, gen.sample_trace) == traces_before
            assert gen.history_count == history_count_before
            report["zero_generation"] = dict(empty_output=True, no_work=True)
            report["append_device_only_guard_calls"] = append_calls
            assert append_calls > 0
        report["status"] = "passed"
        save()
        print("DEFERRED_GENERATION_CHECK_PASSED", flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
