# SPDX-License-Identifier: Apache-2.0
"""Full-model precision experiment using the unchanged readiness entry checks."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import weakref
from dataclasses import asdict
from pathlib import Path

import torch
from readiness_check.run_prefill_check import _run_one_entry_prefill
from readiness_check.run_teacher_forcing import _run_one_entry
from readiness_check.schema import load_reference
from readiness_check.teacher_forcing import TokenAccuracy

import ttnn

from ..tt.generator import KolibriGenerator, build_generator
from ..tt.precision import load_precision_config
from .full_memory import memory_views
from .full_provenance import provenance


def runtime_summary(gen):
    model = gen.model
    runtime = model.runtime_precision
    for v in model.head.output_weights:
        assert v.dtype == getattr(ttnn, runtime["head_weight_dtype"])
    assert model.head.config.compute_kernel_config.math_fidelity == getattr(ttnn.MathFidelity, runtime["head_fidelity"])
    assert model.head.config.compute_kernel_config.fp32_dest_acc_en == runtime["head_fp32"]
    assert gen.logits.dtype == getattr(ttnn, runtime["sampling_logits_dtype"])
    assert gen.tokens.dtype == gen.k.dtype == getattr(ttnn, runtime["sampling_index_dtype"])
    assert all(v["tokens"].dtype == gen.tokens.dtype for v in gen.prefill_buffers.values())
    assert all(v.dtype == getattr(ttnn, runtime["residual_dtype"]) for v in gen.prefill_outputs.values())
    assert model.embedding.dtype == getattr(ttnn, runtime["embedding_dtype"])
    assert model.norm.dtype == getattr(ttnn, runtime["norm_dtype"])
    assert model.compute.math_fidelity == getattr(ttnn.MathFidelity, runtime["norm_fidelity"])
    assert all(v.dtype == getattr(ttnn, runtime["sampling_parameter_dtype"]) for v in (gen.p, gen.temperature))
    layers = []
    for layer, cache in zip(model.layers, gen.state.layers):
        policy = layer.policy
        for info in layer.projection_info.values():
            attention = info["role"] in ("qkv", "o_proj")
            assert info["weight"].dtype == getattr(ttnn, policy.attention_dtype if attention else policy.shared_dtype)
            assert info["compute"].math_fidelity == getattr(
                ttnn.MathFidelity, policy.attention_fidelity if attention else policy.shared_fidelity
            )
        assert layer.experts["gate_up"].dtype == getattr(ttnn, policy.expert_gate_up_dtype)
        assert layer.experts["down_proj"].dtype == getattr(ttnn, policy.expert_down_dtype)
        assert all(
            v.dtype == getattr(ttnn, policy.prefill_expert_dtype)
            for values in layer.prefill_experts.values()
            for v in values
        )
        assert layer.expert_compute.math_fidelity == getattr(ttnn.MathFidelity, policy.prefill_expert_fidelity)
        assert layer.compute.math_fidelity == getattr(ttnn.MathFidelity, policy.safe_compute_fidelity)
        assert layer.sparse_compute.math_fidelity == getattr(ttnn.MathFidelity, policy.expert_fidelity)
        assert layer.sparse_compute.fp32_dest_acc_en == policy.expert_fp32
        assert layer.router.dtype == getattr(ttnn, policy.router_dtype)
        packed_router = getattr(layer, "decode_router", None)
        if packed_router is not None:
            assert packed_router.dtype == getattr(ttnn, policy.router_dtype)
        assert layer.router_compute.math_fidelity == getattr(ttnn.MathFidelity, policy.router_fidelity)
        assert layer.sdpa_compute.math_fidelity == getattr(ttnn.MathFidelity, policy.sdpa_fidelity)
        assert layer.sdpa_compute.fp32_dest_acc_en == policy.sdpa_fp32
        assert layer.long_prefill_compute.math_fidelity == getattr(ttnn.MathFidelity, policy.prefill_long_fidelity)
        assert layer.long_prefill_compute.fp32_dest_acc_en == policy.prefill_long_fp32
        assert all(v.dtype == getattr(ttnn, runtime["kv_cache_dtype"]) for v in cache)
        layers.append(
            dict(
                layer=layer.layer_idx,
                policy=asdict(layer.policy),
                projections={
                    info["role"]: dict(
                        dtype=str(info["weight"].dtype),
                        fidelity=str(info["compute"].math_fidelity),
                        fp32=info["compute"].fp32_dest_acc_en,
                    )
                    for info in layer.projection_info.values()
                },
                experts={key: str(value.dtype) for key, value in layer.experts.items()},
                expert_fidelity=str(layer.sparse_compute.math_fidelity),
                expert_fp32=layer.sparse_compute.fp32_dest_acc_en,
                prefill_experts={
                    key: sorted({str(v.dtype) for v in values}) for key, values in layer.prefill_experts.items()
                },
                prefill_expert_fidelity=str(layer.expert_compute.math_fidelity),
                prefill_expert_fp32=layer.expert_compute.fp32_dest_acc_en,
                router_dtype=str(layer.router.dtype),
                packed_router_dtype=str(packed_router.dtype) if packed_router is not None else None,
                router_fidelity=str(layer.router_compute.math_fidelity),
                router_fp32=layer.router_compute.fp32_dest_acc_en,
                norm_dtypes=sorted({str(v.dtype) for v in layer.norms.values()}),
                norm_fidelity=str(layer.compute.math_fidelity),
                sdpa_fidelity=str(layer.sdpa_compute.math_fidelity),
                sdpa_fp32=layer.sdpa_compute.fp32_dest_acc_en,
                long_prefill_fidelity=str(layer.long_prefill_compute.math_fidelity),
                long_prefill_fp32=layer.long_prefill_compute.fp32_dest_acc_en,
                kv_cache_dtypes=[str(v.dtype) for v in cache],
            )
        )
    return dict(
        layers=layers,
        config=model.precision_config,
        active_experts_per_token=6,
        expert_count=384,
        head_weights=[str(v.dtype) for v in model.head.output_weights],
        head_fidelity=str(model.head.config.compute_kernel_config.math_fidelity),
        head_fp32=model.head.config.compute_kernel_config.fp32_dest_acc_en,
        head_output_dtype=str(model.head.config.lm_head_dtype),
        embedding_dtype=str(model.embedding.dtype),
        final_norm_dtype=str(model.norm.dtype),
        logits_dtype=str(gen.logits.dtype),
        tokens_dtype=str(gen.tokens.dtype),
        topk_parameter_dtype=str(gen.k.dtype),
        prefill_token_dtypes={str(k): str(v["tokens"].dtype) for k, v in gen.prefill_buffers.items()},
        parameter_dtypes=[str(v.dtype) for v in (gen.p, gen.temperature)],
        prefill_output_dtypes={str(k): str(v.dtype) for k, v in gen.prefill_outputs.items()},
        ccl_buffer_dtypes=[str(v.dtype) for v in model.workspace.ar_buffers],
        prefill_buckets=list(gen.buckets),
        capacity=gen.logical_capacity,
        mesh=list(gen.mesh_device.shape),
    )


def token_out(gen, prompt):
    prior = Path(__file__).resolve().parents[1] / "doc/optimized_full_model/reference_metadata.json"
    assert prompt == json.loads(prior.read_text())["prompt_token_ids"][:128]
    assert len(prompt) == 128
    primary = []
    for repeat in range(3):
        gen.generate(prompt, 128, stop_on_eos=False)
        primary.append(dict(repeat=repeat, **gen.last_generation_metrics))
    gen.reset()
    gen._prefill(prompt[:-1])
    gen.bind([prompt[-1]], [127])
    gen.replay()
    ttnn.synchronize_device(gen.mesh_device)
    before = gen.counters.copy()
    tick = time.monotonic()
    for _ in range(128):
        gen.decode_forward(None, None, page_table=None, kv_cache=gen.state, read_from_device=False)
    ttnn.synchronize_device(gen.mesh_device)
    elapsed = time.monotonic() - tick
    counters = dict(gen.counters - before)
    assert counters.get("decode_replays") == 128 and counters.get("split_replays") == 128, counters
    for name in (
        "token_refreshes",
        "position_refreshes",
        "rope_refreshes",
        "page_table_refreshes",
        "token_readbacks",
        "logit_readbacks",
        "synchronizations",
    ):
        assert counters.get(name, 0) == 0, (name, counters)
    return dict(
        regime="warmed token-out no-readback; nonblocking model+split-sampler traces",
        prompt_tokens=128,
        prompt_ids=prompt,
        prompt_sha256=hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
        generated_tokens=128,
        batch=1,
        public_samples=primary,
        ttft_ms=statistics.median(sample["ttft_ms"] for sample in primary),
        decode_ms=elapsed * 1000 / 128,
        decode_t_s_u=128 / elapsed,
        counters=counters,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--token-out", action="store_true")
    parser.add_argument("--token-out-repeats", type=int, choices=(1, 3, 5), default=1)
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--capability", action="store_true")
    parser.add_argument("--long-prefix", action="store_true")
    parser.add_argument("--batch-reconfigure", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    config = load_precision_config(args.config)
    out = Path(os.environ["FULL_ARTIFACT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    reference_path = root / "readiness_aime24_chat.refpt"
    result = dict(
        config_id=config["config_id"],
        precision_config=config,
        provenance=provenance(),
        hardware="QB2 / P300x2 / four Blackhole chips",
        mesh=[1, 4],
        reference=str(reference_path),
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        thresholds=dict(top1=0.90, top5=0.98, top100=1.0),
        status="running",
    )
    harness_source = Path(__file__).read_bytes()
    harness_hash = hashlib.sha256(harness_source).hexdigest()
    (out / "source_snapshots" / (harness_hash + ".py.txt")).write_bytes(harness_source)
    result["harness_sha256"] = harness_hash

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(
            root,
            mesh,
            layer_indices=[0, 4] if args.reduced else None,
            **({"precision_config": config} if args.config else {}),
        )
        gen.prepare()
        result["runtime_summary"] = runtime_summary(gen)
        result["allocator"] = memory_views(mesh)
        saved_traces = dict(gen.traces)
        save()
        if args.reduced:
            # Dtype/API and awkward-tail smoke; no reduced accuracy is used in selection.
            rows = []
            for length in (31, 33, 129, 197, 8193):
                ids = gen.generate([42] * length, 4, stop_on_eos=False)
                rows.append(dict(length=length, tokens=ids))
            result["smoke"] = rows
            result["status"] = "smoke-pass"
        else:
            reference = load_reference(reference_path)
            assert all(entry.generated_tokens.shape[-1] == 100 for entry in reference.entries)
            result["prefill"] = [
                _run_one_entry_prefill(generator=gen, entry=e, reference=reference) for e in reference.entries
            ]
            save()
            samples = []
            for repeat in range(args.repeat):
                acc = TokenAccuracy(reference_path)
                before = gen.counters.copy()
                rows = [_run_one_entry(generator=gen, acc=acc, entry_idx=i) for i in range(acc.num_entries)]
                counters = dict(gen.counters - before)
                assert counters.get("decode_replays", 0) >= 99, counters
                samples.append(dict(repeat=repeat, rows=rows, counters=counters, metrics=gen.last_generation_metrics))
                result["teacher_forcing_samples"] = samples
                save()
            result["decode_t_s_u"] = statistics.median(s["rows"][0]["decode_t/s/u"] for s in samples)
            result["ttft_ms"] = statistics.median(s["rows"][0]["ttft_ms"] for s in samples)
            result[
                "measurement_regime"
            ] = "warmed trace-verified AIME24 teacher-forcing; host reference input and token readback; B1 P197 G100; full 1M cache; median of repeats"
            result["accuracy"] = samples[0]["rows"][0]
            result["status"] = (
                "pass"
                if all(
                    row[k] >= v
                    for row in result["prefill"] + [s["rows"][0] for s in samples]
                    for k, v in result["thresholds"].items()
                )
                else "accuracy-fail"
            )
            if args.token_out:
                token_out_samples = [
                    token_out(gen, reference.entries[0].prompt_tokens[0].tolist()[:128])
                    for _ in range(args.token_out_repeats)
                ]
                result["token_out_repetitions"] = token_out_samples
                median_sample = sorted(token_out_samples, key=lambda sample: sample["decode_t_s_u"])[
                    len(token_out_samples) // 2
                ]
                result["token_out"] = dict(
                    median_sample,
                    repetitions=len(token_out_samples),
                    aggregation="median no-readback decode rate; TTFT from that repetition's three public warm samples",
                    decode_t_s_u_samples=[sample["decode_t_s_u"] for sample in token_out_samples],
                )
                save()
        if args.qualitative:
            from .full_qualitative import run_suite

            result["qualitative"] = run_suite(gen, "tt")
            save()
        if args.capability:
            checks = []
            seed_prompt = load_reference(reference_path).entries[0].prompt_tokens[0].tolist()
            for length in (
                31,
                32,
                33,
                34,
                127,
                128,
                129,
                130,
                511,
                512,
                513,
                514,
                2047,
                2048,
                2049,
                2050,
                8191,
                8192,
                8193,
                8194,
                8209,
            ):
                outputs = []
                logits = []
                request = (seed_prompt * ((length + len(seed_prompt) - 1) // len(seed_prompt)))[:length]
                for traced in (False, True):
                    gen.trace_prefill = traced
                    outputs.append(gen.generate(request, 4, stop_on_eos=False))
                    logits.append(gen.read_logits().clone())
                assert outputs[0] == outputs[1], (length, outputs)
                assert torch.equal(logits[0], logits[1]), (length, (logits[0] - logits[1]).abs().max())
                checks.append(dict(length=length, eager_traced_equal=True, exact_logits_equal=True, tokens=outputs[1]))
            gen.trace_prefill = True
            gen.reset()
            tail = gen.prefill_forward(
                torch.full((1, 33), 42, dtype=torch.long),
                page_table=gen.state.host_page_tables,
                kv_cache=gen.state,
                prompt_lens=[33],
                start_pos=[gen.logical_capacity - 33],
                slots=[0],
            )
            result["capability"] = dict(
                non_aligned_checks=checks,
                input_kind="Pinned AIME24 prompt prefix repeated to requested length; capability input, not a quality prompt",
                max_address_probe=dict(
                    start=gen.logical_capacity - 33, length=33, initialized_prefix=False, tokens=tail.tolist()
                ),
                capacity=gen.logical_capacity,
            )
            save()
        if args.long_prefix:
            length = gen.logical_capacity - 33
            original_prefill = gen._prefill_bucket

            def progress_prefill(n, bound, alignment=32):
                value = original_prefill(n, bound, alignment)
                print("LONG_PREFILL_SUBMITTED_END", bound, flush=True)
                return value

            gen._prefill_bucket = progress_prefill
            tick = time.monotonic()
            try:
                ids = gen.generate([42] * length, 34, stop_on_eos=False)
            finally:
                del gen._prefill_bucket
                # A bound method keeps the complete generator/cache alive even
                # after `del gen` during the later B1 -> B32 transition.
                del progress_prefill, original_prefill
            assert len(ids) == 34
            final_positions = {
                name: ttnn.to_torch(ttnn.get_device_tensors(value)[0]).flatten().tolist()
                for name, value in (("kv", gen.positions), ("rope", gen.rope_positions))
            }
            assert final_positions == {"kv": [gen.logical_capacity], "rope": [gen.logical_capacity]}
            result["long_prefix"] = dict(
                length=length,
                ids=ids,
                last_consumed_position=length + len(ids) - 2,
                seconds=time.monotonic() - tick,
                initialized_full_prefix=True,
                native_next_positions=final_positions,
                input_kind="Repeated token42 capacity workload; not a quality prompt",
            )
            save()
        assert gen.traces == saved_traces
        result["traces"] = {k: str(v) for k, v in gen.traces.items()}
        result["counters"] = dict(gen.counters)
        save()
        if args.batch_reconfigure:
            first_b1 = gen.generate([42] * 65, 1, stop_on_eos=False)[0]
            retained_model = gen.model
            retired_generator = weakref.ref(gen)
            result["before_b1_release_allocator"] = memory_views(mesh)
            ttnn.synchronize_device(mesh)
            gen.close()
            result["b1_teardown_counters"] = dict(gen.counters)
            assert gen.counters["teardown_releases"] == len(saved_traces)
            del gen
            gen = None
            gc.collect()
            assert retired_generator() is None, "Retired B1 generator still owns its cache"
            result["after_b1_release_allocator"] = memory_views(mesh)
            result["b1_generator_released"] = True
            save()
            gen = KolibriGenerator(retained_model, batch_size=32, capacity=8192)
            gen.prepare()
            batch_traces = dict(gen.traces)
            prompts = torch.full((32, 129), 42, dtype=torch.long)
            lengths = [65 + i % 3 * 32 for i in range(32)]
            lengths[-1] = 0
            first = gen.prefill_forward(
                prompts, page_table=gen.state.host_page_tables, kv_cache=gen.state, prompt_lens=lengths
            )
            first_initial = first.clone()
            assert int(first[0]) == first_b1 and first[0] == first[3]
            logits = gen.decode_forward(
                [42] * 32,
                [n if n else -1 for n in lengths],
                page_table=gen.state.host_page_tables,
                kv_cache=gen.state,
                sample_on_device=False,
            )
            assert torch.equal(logits[0], logits[3])
            counts = gen.counters.copy()
            for _ in range(4):
                gen.decode_forward(None, None, page_table=None, kv_cache=gen.state)
            for key in ("token_refreshes", "position_refreshes", "rope_refreshes", "page_table_refreshes"):
                assert gen.counters[key] == counts[key]
            gen.reset()
            repeated = gen.prefill_forward(
                prompts, page_table=gen.state.host_page_tables, kv_cache=gen.state, prompt_lens=lengths
            )
            result["batch32_attempt"] = dict(
                prompt_lengths=lengths,
                first_initial=first_initial.tolist(),
                first_after=first.tolist(),
                repeated=repeated.tolist(),
                changed_slots=[i for i in range(32) if int(first[i]) != int(repeated[i])],
                readback_unchanged=torch.equal(first, first_initial),
                trace_ids_unchanged=gen.traces == batch_traces,
                counters=dict(gen.counters),
            )
            save()
            assert torch.equal(first, repeated) and gen.traces == batch_traces
            repeated_logits = gen.decode_forward(
                [42] * 32,
                [n if n else -1 for n in lengths],
                page_table=gen.state.host_page_tables,
                kv_cache=gen.state,
                sample_on_device=False,
            )
            assert torch.isfinite(logits).all() and torch.isfinite(repeated_logits).all()
            assert torch.equal(logits, repeated_logits), "Batch logits changed after reset and identical prefill"
            batch_runtime = runtime_summary(gen)
            batch_runtime["layers"] = [
                dict(layer=row["layer"], kv_cache_dtypes=row["kv_cache_dtypes"]) for row in batch_runtime["layers"]
            ]
            result["batch32"] = dict(
                capacity=8192,
                layer_count=len(gen.model.layers),
                prompt_lengths=lengths,
                first=first.tolist(),
                repeat=repeated.tolist(),
                b1_token_match=True,
                equal_slot_logits=True,
                repeated_all_slot_logits_exact=True,
                all_slot_logits_finite=True,
                feedback_without_host_refresh=True,
                allocator=memory_views(mesh),
                trace_ids={k: str(v) for k, v in batch_traces.items()},
                runtime_summary=batch_runtime,
            )
            save()
        print("DATATYPE_RESULT", result["config_id"], result["status"], result.get("decode_t_s_u"), flush=True)
    except Exception as error:
        result.update(status="runtime-fail", error=repr(error))
        save()
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
