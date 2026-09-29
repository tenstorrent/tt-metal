"""Public generation after mixed-batch external-cache requests in one process."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.common.modules.tt_ccl import TT_CCL

from ..tt.generator import K2Generator
from .probe_full_batch import run as batch_probe


def all_logits_retirement_regression(gen, ids):
    """Retired short-head scratch must invalidate its exact prepared signature."""
    prompt = torch.tensor([ids[:17]])
    cache_ids = tuple(id(tensor) for pair in gen.kv_cache for tensor in pair)
    cache_addresses = tuple(tensor.buffer_address() for pair in gen.kv_cache for tensor in pair)
    table = gen.page_table.clone()
    signature = gen._prefill_signature([17], [0], [0], gen.page_table, gen.kv_cache, True, "host")

    def pool_rows():
        return {key[1][-2] for key in gen.model.pool.tensors}

    def trace_ids():
        return gen.state["trace"], dict(gen.state["sample_traces"])

    def pool_addresses():
        return {
            key: tuple(t.buffer_address() for t in (value if isinstance(value, list) else [value]))
            for key, value in gen.model.pool.tensors.items()
        }

    def all_logits():
        gen.reset()
        return gen.prefill_forward(
            prompt,
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[17],
            return_all_logits=True,
            sampling_mode="host",
        )

    assert gen.state is not None and gen.state["trace"] is not None
    first = all_logits()
    assert first.shape == (1, 17, gen.model.vocab_size)
    assert torch.isfinite(first).all()
    assert signature in gen.prepared_prefills
    assert 17 in pool_rows(), "All-logits17 must retain its short final-head gather"
    assert gen.state["trace"] is None and not gen.state["sample_traces"]

    # Public generation must retire diagnostic-only scratch even when the
    # preceding prefill preserved this exact cache/state key with trace=None.
    before_retirement = gen.counters["collective_buckets_retired"]
    expected_tokens = gen.generate(ids[:33], 4)
    assert 17 not in pool_rows()
    assert signature not in gen.prepared_prefills, "Retired all-logits scratch must lose prepared status"
    assert gen.counters["collective_buckets_retired"] > before_retirement

    live_traces = trace_ids()
    live_pool = pool_addresses()
    assert live_traces[0] is not None and set(live_traces[1]) == (
        {"split", "argmax", "split_history", "argmax_history"} if gen.token_history is not None else {"split", "argmax"}
    )
    assert signature not in gen.prepared_prefills
    assert 17 not in pool_rows()
    repeated_tokens = gen.generate(ids[:33], 4)
    assert repeated_tokens == expected_tokens
    assert trace_ids() == live_traces, "Prepared generate33 must preserve both trace sets"
    assert pool_addresses() == live_pool, "Prepared generate33 must preserve collective addresses"

    assert signature == gen._prefill_signature([17], [0], [0], gen.page_table, gen.kv_cache, True, "host")
    before_invalidation = gen.counters["release_synchronizations"]
    second = all_logits()
    assert gen.counters["release_synchronizations"] > before_invalidation
    assert (
        gen.state["trace"] is None and not gen.state["sample_traces"]
    ), "Unprepared all-logits must retire live traces"
    assert signature in gen.prepared_prefills and 17 in pool_rows()
    assert torch.equal(first, second), "Same all-logits input must survive pool retirement and native rebind exactly"

    resumed_tokens = gen.generate(ids[:33], 4)
    assert resumed_tokens == expected_tokens
    assert gen.state["trace"] is not None and set(gen.state["sample_traces"]) == (
        {"split", "argmax", "split_history", "argmax_history"} if gen.token_history is not None else {"split", "argmax"}
    )
    assert 17 not in pool_rows() and signature not in gen.prepared_prefills
    assert tuple(id(tensor) for pair in gen.kv_cache for tensor in pair) == cache_ids
    assert tuple(tensor.buffer_address() for pair in gen.kv_cache for tensor in pair) == cache_addresses
    assert torch.equal(gen.page_table, table)
    return {
        "pass": True,
        "all_logits_length": 17,
        "generation_length": 33,
        "all_logits_equal": True,
        "all_logits_max_abs_delta": float((first.float() - second.float()).abs().max()),
        "retired_signature_invalidated": True,
        "exact_signature_reused": True,
        "repeat_invalidated_model_and_sample_traces": True,
        "public_generation_retired_diagnostic_bucket_before_recapture": True,
        "warmed_generation_trace_and_pool_addresses_stable": True,
        "owned_cache_identity_and_addresses_preserved": True,
        "resumed_tokens": resumed_tokens,
        "pool_rows_after_resume": sorted(pool_rows()),
    }


def prepared_prefill_regression(gen, ids, record, save, batch=2):
    """Pool eviction invalidates prefill readiness across B2/B32 decode."""
    cache, table = gen.model.allocate_cache(batch_size=batch, capacity=384)
    tokens = torch.tensor([ids[:257]] * batch)
    assert tokens.shape == (batch, 257)
    lengths, starts, slots = [257] * batch, [0] * batch, list(range(batch))
    cache_ids = tuple(id(tensor) for pair in cache for tensor in pair)
    cache_addresses = tuple(tensor.buffer_address() for pair in cache for tensor in pair)
    original_table = table.clone()
    signature = gen._prefill_signature(lengths, starts, slots, table, cache, False, "host")

    def prefill():
        return gen.prefill_forward(
            tokens,
            page_table=table,
            kv_cache=cache,
            prompt_lens=lengths,
            start_pos=starts,
            active_slots=slots,
            sampling_mode="host",
        )

    def trace_snapshot():
        return {
            "model": None if gen.state["trace"] is None else str(gen.state["trace"]),
            "samplers": {name: str(trace) for name, trace in gen.state["sample_traces"].items()},
            "batch": gen.state["batch"],
        }

    # The raw-clear control can leave only row1 resident. Warm a legitimate
    # short request if needed so both entry paths create the same prefill scratch.
    # Only the fixed decode batch remains resident after trace preparation.
    short_bucket_warmup = not any(key[1][-2] == 32 for key in gen.model.pool.tensors)
    if short_bucket_warmup:
        gen.prefill_forward(
            tokens[:, :31], page_table=table, kv_cache=cache, prompt_lens=[31] * batch, sampling_mode="host"
        )
    first = prefill()
    assert first.shape == (batch, 1, gen.model.vocab_size) and torch.isfinite(first).all()
    assert signature in gen.prepared_prefills
    decode_tokens = first[:, 0].argmax(dim=-1).reshape(batch, 1)
    positions = torch.tensor(lengths)
    retired_before_decode = gen.counters["collective_buckets_retired"]
    first_decode = gen.decode_forward(decode_tokens, positions, page_table=table, kv_cache=cache, sampling_mode="host")
    assert first_decode.shape == (batch, gen.model.vocab_size) and torch.isfinite(first_decode).all()
    assert gen.state["batch"] == batch and gen.state["trace"] is not None
    assert set(gen.state["sample_traces"]) == (
        {"split", "argmax", "split_history", "argmax_history"} if gen.token_history is not None else {"split", "argmax"}
    )
    decode_buckets_retired = gen.counters["collective_buckets_retired"] - retired_before_decode
    assert decode_buckets_retired > 0, "Decode preparation must evict the prefill's row-1 scratch"
    assert signature not in gen.prepared_prefills, "Pool eviction must invalidate last-only prefill readiness"
    assert {key[1][-2] for key in gen.model.pool.tensors} == {batch}
    before_retirement = gen.counters["collective_buckets_retired"]
    before_invalidation = gen.counters["release_synchronizations"]
    record.update(
        {
            "pass": False,
            "stage": "before_prepared_prefill_repeat",
            "batch": batch,
            "prompt_lens": lengths,
            "short_bucket_warmup": short_bucket_warmup,
            "initial_decode_buckets_retired": decode_buckets_retired,
            "signature_invalidated_by_pool_retirement": True,
            "signature_hit_before_repeat": signature in gen.prepared_prefills,
            "signature_before_repeat": signature,
            "pool_before_repeat": gen.model.pool.inventory(),
            "traces_before_repeat": trace_snapshot(),
        }
    )
    # Persist the suspected collision's preconditions even if the old code
    # aborts inside minimal matmul. This is evidence capture, not a fallback.
    save()
    print(
        f"PREPARED_PREFILL_REPEAT signature_hit={signature in gen.prepared_prefills} "
        f"buckets_retired={decode_buckets_retired} batch={batch} rows=257",
        flush=True,
    )
    try:
        second = prefill()
    except Exception as error:
        record.update(stage="prepared_prefill_repeat_failed", error=str(error).split("backtrace:")[0][:2000])
        save()
        raise
    assert gen.counters["release_synchronizations"] > before_invalidation
    if batch != 32:
        assert gen.counters["collective_buckets_retired"] >= before_retirement + 2
    assert gen.state["trace"] is None and not gen.state["sample_traces"]
    if batch != 32:
        assert not any(key[1][-2] == batch for key in gen.model.pool.tensors)
    assert signature in gen.prepared_prefills
    assert torch.equal(first, second), f"Prefill must remain numerically identical across B{batch} retirement"
    record.update(
        stage="prepared_prefill_repeat_passed",
        pool_after_repeat=gen.model.pool.inventory(),
        traces_after_repeat=trace_snapshot(),
        prefill_logits_equal=True,
        prefill_max_abs_delta=float((first.float() - second.float()).abs().max()),
    )
    save()

    # Re-establish the external batch state with explicit scheduler inputs; the
    # same cache pages now again contain the identical 257-token prefix.
    resumed = gen.decode_forward(decode_tokens, positions, page_table=table, kv_cache=cache, sampling_mode="host")
    assert torch.equal(first_decode, resumed), f"Explicit B{batch} decode must resume identically after recapture"
    assert gen.state["trace"] is not None and set(gen.state["sample_traces"]) == (
        {"split", "argmax", "split_history", "argmax_history"} if gen.token_history is not None else {"split", "argmax"}
    )
    assert {key[1][-2] for key in gen.model.pool.tensors} == {batch}
    assert signature not in gen.prepared_prefills, "Resumed decode retires the recreated prefill scratch"
    assert tuple(id(tensor) for pair in cache for tensor in pair) == cache_ids
    assert tuple(tensor.buffer_address() for pair in cache for tensor in pair) == cache_addresses
    assert torch.equal(table, original_table)
    record.update(
        {
            "pass": True,
            "stage": "explicit_decode_resumed",
            "explicit_decode_logits_equal": True,
            "external_cache_identity_and_addresses_preserved": True,
            "pool_after_resume": gen.model.pool.inventory(),
            "traces_after_resume": trace_snapshot(),
        }
    )
    save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--resident-layers", type=int, default=1)
    parser.add_argument("--clear-pool-control", action="store_true")
    parser.add_argument("--old-head", action="store_true")
    parser.add_argument("--all-logits-regression", action="store_true")
    parser.add_argument("--repeat-batch-probe", action="store_true")
    parser.add_argument("--prepared-prefill-regression", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(
            mesh,
            override_num_layers=args.layers,
            **({"head_split_size": 32768, "head_workers": 2, "head_k": 2} if args.old_head else {}),
        )
        # Match full-stack semaphore residency without the other layers' compute.
        resident_collectives = [TT_CCL(mesh) for _ in range(max(0, args.resident_layers - args.layers))]
        batch_probe(gen, args.output.with_name(args.output.stem + "_batch.json"))
        result = {
            "layers": args.layers,
            "old_head": args.old_head,
            "resident_layers": max(args.layers, args.resident_layers),
            "clear_pool_control": args.clear_pool_control,
            "pool_before": gen.model.pool.inventory(),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.clear_pool_control:
            gen._release_traces()
            gen.model.pool.tensors.clear()
        ids = (gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. ") * 30)[:257]
        output = gen.generate(ids, 4)
        result.update(pass_transition=True, tokens=output, pool_after=gen.model.pool.inventory(), perf=gen.last_perf)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.all_logits_regression:
            result["all_logits_retirement"] = all_logits_retirement_regression(gen, ids)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("ALL_LOGITS_RETIREMENT_PASS", flush=True)
        if args.repeat_batch_probe:
            assert not any(key[1][-2] == 2 for key in gen.model.pool.tensors)
            retired_before = gen.counters["collective_buckets_retired"]
            records = batch_probe(gen, args.output.with_name(args.output.stem + "_batch_rebound.json"))
            retired = gen.counters["collective_buckets_retired"] - retired_before
            assert retired >= 2, "The re-created B2 gather/reduce buckets must retire at the next B1 boundary"
            assert not any(key[1][-2] == 2 for key in gen.model.pool.tensors)
            result["batch_rebind"] = {
                "pass": True,
                "batches": [r["batch"] for r in records],
                "new_buckets_retired": retired,
                "pool_after": gen.model.pool.inventory(),
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("BATCH_REBIND_PASS", flush=True)
        if args.prepared_prefill_regression:
            result["prepared_prefill"] = {}
            prepared_prefill_regression(
                gen,
                ids,
                result["prepared_prefill"],
                lambda: args.output.write_text(json.dumps(result, indent=2) + "\n"),
            )
            result["prepared_prefill_batch32"] = {}
            prepared_prefill_regression(
                gen,
                ids,
                result["prepared_prefill_batch32"],
                lambda: args.output.write_text(json.dumps(result, indent=2) + "\n"),
                batch=32,
            )
            print("PREPARED_PREFILL_RETIREMENT_PASS", flush=True)
        print("BATCH_TRANSITION_PASS", flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
