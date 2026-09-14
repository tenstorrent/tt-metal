# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Owned-prefill trace correctness and lifetime checks on the serialized TP4 mesh.

Reduced layers cover chunk boundaries by default. --full defaults to short
lengths; --lengths can explicitly request the longer full-model checks. Prompt
IDs are shape fixtures, not qualitative model evidence. Guarded runs are not
performance measurements.
"""

import argparse
import inspect
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.check_deferred_generation import assert_tokens, forbidden, guarded_generate
from models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder import device_only
from models.autoports.qwen_qwen3_8_27b.tt.generator import QwenGenerator, build_generator, configure_fabric


def host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def addresses(tensor):
    return [part.buffer_address() for part in ttnn.get_device_tensors(tensor)]


def identity(gen):
    tensors = dict(
        feedback=gen.tokens,
        positions=gen.positions,
        rope=gen.rope_indices,
        table=gen.page_table,
        history=gen.token_history,
        cursor=gen.history_cursor,
        decode_logits=gen.logits,
    )
    if gen.prefill_prepared is not None:
        tensors.update({f"prefill_{key}": gen.prefill_prepared[key] for key in ("tokens", "positions", "output")})
    return dict(
        traces=[str(t) for t in (gen.trace, gen.sample_trace, gen.prefill_trace)],
        tensors={name: dict(identity=id(tensor), addresses=addresses(tensor)) for name, tensor in tensors.items()},
        preparation=id(gen.prefill_prepared),
    )


@contextmanager
def capture_guards(gen, evidence):
    """Forbid fallback inside prefill and program-cache misses only while recording."""
    begin, end = ttnn.begin_trace_capture, ttnn.end_trace_capture
    set_misses = getattr(gen.mesh, "set_program_cache_misses_allowed", None)
    evidence["program_cache_miss_guard_available"] = callable(set_misses)
    evidence["guarded_captures"] = 0
    evidence["device_only_prefill_calls"] = 0

    def guarded_begin(*args, **kwargs):
        if callable(set_misses):
            set_misses(False)
        try:
            trace = begin(*args, **kwargs)
        except BaseException:
            if callable(set_misses):
                set_misses(True)
            raise
        evidence["guarded_captures"] += 1
        return trace

    def guarded_end(*args, **kwargs):
        try:
            return end(*args, **kwargs)
        finally:
            if callable(set_misses):
                set_misses(True)

    def wrap_forward(forward):
        def guarded(*args, **kwargs):
            with (
                device_only(),
                patch.object(ttnn, "copy_host_to_device_tensor", forbidden("prefill host upload")),
                patch.object(ttnn, "synchronize_device", forbidden("prefill synchronization")),
            ):
                result = forward(*args, **kwargs)
            evidence["device_only_prefill_calls"] += 1
            return result

        return guarded

    with ExitStack() as stack:
        stack.enter_context(patch.object(ttnn, "begin_trace_capture", guarded_begin))
        stack.enter_context(patch.object(ttnn, "end_trace_capture", guarded_end))
        for name in ("_prefill_trace_logits", "_prefill_trace_step"):
            stack.enter_context(patch.object(gen, name, wrap_forward(getattr(gen, name))))
        try:
            yield
        finally:
            if callable(set_misses):
                set_misses(True)


def cached_run(gen, prompt, count, **sampling):
    """Guard a previously warmed key, including its decode feedback loop."""
    eligible = len(prompt) <= 4096
    before = identity(gen)
    traces = gen.trace, gen.sample_trace, gen.prefill_trace
    assert traces[0] is not None and traces[1] is not None
    assert (traces[2] is not None) is eligible
    if eligible:
        assert len({str(t) for t in traces}) == 3, "Prefill, model and sampling need distinct traces"
    execute = ttnn.execute_trace
    replayed = []

    def nonblocking_execute(*args, **kwargs):
        assert kwargs.get("blocking", True) is False, "Replay must explicitly use blocking=False"
        trace = args[1] if len(args) > 1 else kwargs["trace_id"]
        replayed.append(str(trace))
        return execute(*args, **kwargs)

    with (
        patch.object(ttnn, "execute_trace", nonblocking_execute),
        patch.object(gen, "_capture", forbidden("recapture of a warmed key")),
        patch.object(gen, "_prefill_trace_logits", forbidden("eager owned prefill on a warmed key")),
    ):
        tokens, decode_guard = guarded_generate(gen, prompt, count, **sampling)
    assert identity(gen) == before, "A warmed request changed persistent tensors or trace identities"
    expected_replays = ([str(traces[2])] if eligible else []) + [
        str(trace) for _ in range(count - 1) for trace in traces[:2]
    ]
    assert replayed == expected_replays, (replayed, expected_replays)
    perf = gen.last_perf
    counters = perf["counters"]
    assert perf["prefill_trace_eligible"] is eligible
    for name in ("trace_captures", "prefill_trace_captures", "history_allocations", "page_table_allocations"):
        assert counters.get(name, 0) == 0, (name, counters)
    assert counters.get("prefill_replays", 0) == int(eligible), counters
    assert counters.get("prefill_eager_calls", 0) == int(not eligible), counters
    assert counters.get("prefill_token_refreshes", 0) == int(eligible), counters
    assert counters.get("page_table_refreshes", 0) == 0, counters
    return tokens, dict(identity=before, replay_order=replayed, perf=perf, decode_guard=decode_guard)


def check_length(gen, prompt, count):
    expected = gen.generate(prompt, count, trace_prefill=False, defer_token_readback=False)
    cold = gen.generate(prompt, count, trace_prefill=True)
    assert_tokens(cold, expected, count, "Cold owned prefill versus eager oracle")
    cold_perf = gen.last_perf
    eligible = len(prompt) <= 4096
    assert cold_perf["counters"].get("prefill_trace_captures", 0) == int(eligible)
    warmed = gen.generate(prompt, count, trace_prefill=True)
    assert_tokens(warmed, expected, count, "Repeated owned prefill versus eager oracle")
    warmed_perf = gen.last_perf  # Uninstrumented timing, separate from guarded evidence.
    guarded, evidence = cached_run(gen, prompt, count)
    assert_tokens(guarded, expected, count, "Guarded owned prefill versus eager oracle")
    position = int(host(gen.positions).reshape(-1)[0])
    assert position == len(prompt) + count - 1, position
    return dict(
        length=len(prompt),
        generated=count,
        tokens=guarded,
        all_tokens_equal=True,
        final_position=position,
        cold=cold_perf,
        warmed=warmed_perf,
        guarded=evidence,
    )


def check_changed_inputs(gen, first, second, count, save_row):
    modes = dict(
        greedy=dict(top_k=1, top_p=0.0, temperature=1.0, seed=37),
        sampled=dict(top_k=8, top_p=0.9, temperature=0.8, seed=37),
    )
    prompts = dict(first=first, second=second)
    oracles = {
        (label, mode): gen.generate(prompt, count, trace_prefill=False, defer_token_readback=False, **params)
        for label, prompt in prompts.items()
        for mode, params in modes.items()
    }
    warm = gen.generate(first, count, **modes["greedy"])
    assert_tokens(warm, oracles["first", "greedy"], count, "Changed-input warmup")
    first_logits = host(gen.prefill_prepared["output"])[0, 0, 0].clone()
    for label, mode in (("second", "sampled"), ("first", "greedy"), ("first", "sampled"), ("second", "greedy")):
        tokens, evidence = cached_run(gen, prompts[label], count, **modes[mode])
        assert_tokens(tokens, oracles[label, mode], count, f"Changed prompt {label}, mode {mode}")
        logits = host(gen.prefill_prepared["output"])[0, 0, 0]
        if label == "first":
            assert torch.equal(logits, first_logits), "Same prompt did not restore exact prefill logits"
        else:
            assert not torch.equal(logits, first_logits), "Changed token buffer produced stale prefill logits"
        save_row(dict(prompt=label, mode=mode, parameters=modes[mode], tokens=tokens, guarded=evidence))


def check_public_ownership(gen, first, second, count):
    assert gen.prefill_trace is not None
    outputs = []
    snapshots = []
    for prompt in (first, second):
        gen.reset()
        output = gen.prefill_forward(
            torch.tensor([prompt]), page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=[len(prompt)]
        )[0]
        assert gen.prefill_prepared is None and gen.prefill_trace is None
        assert gen.trace is None and gen.sample_trace is None
        outputs.append(output)
        snapshots.append(host(output).clone())
    assert outputs[0] is not outputs[1]
    assert all(a != b for a, b in zip(addresses(outputs[0]), addresses(outputs[1])))
    assert not torch.equal(snapshots[0], snapshots[1]), "Ownership fixture must produce different logits"
    gen.generate(first, count)
    cached_run(gen, first, count)
    for output, snapshot in zip(outputs, snapshots):
        assert torch.equal(host(output), snapshot), "A retained public output was overwritten by later prefill"
    return dict(
        independent_addresses=[addresses(output) for output in outputs],
        public_call_discarded_owned_traces=True,
        outputs_survive_other_public_and_traced_requests=True,
    )


def check_page_table(gen, prompt, count):
    """Check physical KV writes, since equal tokens alone cannot expose a stale table."""
    state = next(state for state in gen.cache.layers if state.key is not None)
    original = host(gen.page_table).int().clone()
    changed = gen.cache.num_pages - 1 - original
    used = (len(prompt) + count - 1 + 31) // 32
    original_ids, changed_ids = original[0, :used].long(), changed[0, :used].long()
    assert set(original_ids.tolist()).isdisjoint(changed_ids.tolist())

    def logical_pages(table_ids, empty_ids):
        result = {}
        for name in ("key", "value"):
            tensor = host(getattr(state, name))
            assert torch.count_nonzero(tensor[empty_ids]).item() == 0, f"{name} wrote stale physical page IDs"
            result[name] = tensor[table_ids].clone()
            assert torch.count_nonzero(result[name]).item() > 0, f"{name} fixture has no visible KV writes"
        return result

    def assert_pages(actual, expected):
        for name in ("key", "value"):
            assert torch.equal(actual[name], expected[name]), f"Physical remapping changed {name} values"

    expected = gen.generate(prompt, count, trace_prefill=False, defer_token_readback=False)
    canonical = logical_pages(original_ids, changed_ids)
    table_identity, table_addresses = id(gen.page_table), addresses(gen.page_table)
    gen._refresh_table(changed)
    changed_oracle = gen.generate(prompt, count, trace_prefill=False, defer_token_readback=False)
    assert_tokens(changed_oracle, expected, count, "Eager page permutation")
    assert_pages(logical_pages(changed_ids, original_ids), canonical)
    gen._refresh_table(original)
    warm = gen.generate(prompt, count)
    assert_tokens(warm, expected, count, "Canonical traced table warmup")
    assert_pages(logical_pages(original_ids, changed_ids), canonical)
    runs = []
    for label, table, table_ids, empty_ids in (
        ("permuted", changed, changed_ids, original_ids),
        ("restored", original, original_ids, changed_ids),
    ):
        before = gen.counters["page_table_refreshes"]
        gen._refresh_table(table)
        assert gen.counters["page_table_refreshes"] == before + 1
        tokens, evidence = cached_run(gen, prompt, count)
        assert_tokens(tokens, expected, count, f"Cached {label} table")
        assert_pages(logical_pages(table_ids, empty_ids), canonical)
        assert id(gen.page_table) == table_identity and addresses(gen.page_table) == table_addresses
        runs.append(dict(mapping=label, physical_pages=table_ids.tolist(), tokens=tokens, guarded=evidence))
    return dict(runs=runs, physical_kv_writes_follow_changed_table=True, exact_kv_values=True)


def check_small_generation(gen, prompt):
    gen._release_traces()
    first = gen.generate(prompt, 1)
    assert len(first) == 1 and gen.prefill_trace is None and gen.trace is None and gen.sample_trace is None
    repeated = gen.generate(prompt, 1)
    assert repeated == first
    assert gen.prefill_trace is None and gen.last_perf["counters"].get("prefill_eager_calls", 0) == 1
    gen.generate(prompt, 2)
    cached, evidence = cached_run(gen, prompt, 1)
    assert cached == first
    before, counters, history_count = identity(gen), gen.counters.copy(), gen.history_count
    assert gen.generate(prompt, 0) == []
    assert identity(gen) == before and gen.counters == counters and gen.history_count == history_count
    return dict(cold_g1_eager=True, cached_g1_equal=True, zero_generation_no_work=True, guarded=evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--lengths", help="Comma-separated lengths; reduced default includes 4095/4096/4097")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert "trace_prefill" in inspect.signature(QwenGenerator.generate).parameters, "Apply prefill tracing first"
    lengths = (
        [int(value) for value in args.lengths.split(",")]
        if args.lengths
        else ([31, 32, 33] if args.full else [31, 32, 33, 4095, 4096, 4097])
    )
    assert lengths and min(lengths) >= 1
    count = 4
    report = dict(
        status="running",
        full=args.full,
        mesh=[1, 4],
        lengths=lengths,
        generated=count,
        boundaries=[],
        changed_inputs=[],
        capture_guard={},
        timing_scope="Only uninstrumented warmed timings are performance evidence",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def save_transition(row):
        report["changed_inputs"].append(row)
        save()
        print("PREFILL_CHANGED_INPUT_PASS", row["prompt"], row["mode"], flush=True)

    def prompt(length, offset=0):
        return [1596 + (i + offset) % 17 for i in range(length)]

    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=None if args.full else [0, 3])
        required = max(8192, max(lengths) + count - 1)
        assert required <= gen.model.context, "Requested shape exceeds advertised context"
        gen._ensure_cache(1, required)
        gen._ensure_history(8)
        report["layers"] = gen.model.layer_indices
        report["cache_capacity"] = gen.cache.capacity
        report["history_capacity"] = gen.history_capacity
        save()
        with capture_guards(gen, report["capture_guard"]):
            for length in lengths:
                print("PREFILL_BOUNDARY_BEGIN", length, flush=True)
                row = check_length(gen, prompt(length), count)
                report["boundaries"].append(row)
                save()
                print("PREFILL_BOUNDARY_PASS", length, flush=True)
            check_changed_inputs(gen, prompt(33), prompt(33, offset=7), count, save_transition)
            report["public_ownership"] = check_public_ownership(gen, prompt(33), prompt(33, offset=7), count)
            save()
            print("PREFILL_PUBLIC_OWNERSHIP_PASS", flush=True)
            report["page_table"] = check_page_table(gen, prompt(33), count)
            save()
            print("PREFILL_CHANGED_TABLE_PASS", flush=True)
            report["small_generation"] = check_small_generation(gen, prompt(1))
            assert report["capture_guard"]["guarded_captures"] > 0
            assert report["capture_guard"]["device_only_prefill_calls"] > 0
        report["status"] = "passed"
        save()
        print("PREFILL_TRACING_CHECK_PASSED", flush=True)
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
