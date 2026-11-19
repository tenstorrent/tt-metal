# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic TP4 logits through the canonical adapter's split sampling traces.

Run serially under an external timeout after capacity integration. This does not
modify capabilities or sampler implementations. A capacity32 run records greedy
controls for a later capacity64 run; no second generator is allocated in-process.
Token reads are observations only. They never feed model decode or sample inputs.
"""

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_adapter_decode_probe import build_adapter
from models.common.sampling.generator import SamplingParams, _hash_request_seed_to_device_seed


def candidate_ids(vocab):
    shard = vocab // 4
    assert vocab % 4 == 0 and shard > 2048
    return [
        [1000 + rank for rank in range(64)],
        [3 * shard + 1000 + rank for rank in range(64)],
        [(rank % 4) * shard + 1000 + rank // 4 for rank in range(64)],
    ]


def synthetic_logits(vocab, candidates, *, tie=False):
    logits = torch.full((1, 1, 32, vocab), -80.0, dtype=torch.bfloat16)
    for row, ids in enumerate(candidates):
        logits[0, 0, row, ids] = -torch.arange(64).float().to(torch.bfloat16) / 128
        if tie:
            # Cross the global rank32 boundary on row2 (16 ties per shard).
            # Keep at most32 maxima on a single shard so this is inside the
            # documented legacy32 greedy tie contract as well.
            logits[0, 0, row, ids[: 64 if row == 2 else 32]] = 0
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-capacity", type=int, choices=(32, 64), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-reference", type=Path)
    parser.add_argument("--draws", type=int, default=64)
    args = parser.parse_args()
    if args.draws < 64 or args.draws > 128:
        parser.error("--draws must be 64..128 for tail coverage and context256")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "capacity": args.expected_capacity,
        "layers": [0, 3],
        "batch_capacity": 3,
        "physical_sampling_rows": 32,
        "context": 256,
        "trace_allocation_tracking": os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "unset"),
        "scope": "Synthetic sampling correctness and isolated sampling latency; not full-model or serving evidence.",
        "cases": [],
        "primitive_calls": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_adapter(None, mesh, max_batch_size=3, max_seq_len=256)
        assert gen.cache_is_external
        assert gen.sampler.tt_sampling.max_top_k == args.expected_capacity, "Production capacity is not integrated"
        assert not gen.sampler.tt_sampling._allow_force_argmax_sampling
        candidates = candidate_ids(gen.model.vocab_size)
        rank_maps = [{token: rank for rank, token in enumerate(ids)} for ids in candidates]
        report["candidate_ids"] = candidates
        prompts = [[ids[0], ids[1], ids[1]] for ids in candidates]
        original_sampling = ttnn.sampling

        def observed_sampling(*positional, **keywords):
            capacity = keywords.get("max_top_k", 32)
            assert capacity == args.expected_capacity, "Common primitive silently used a different capacity"
            shape = list(positional[0].shape)
            assert shape[-1] == args.expected_capacity * 4, shape
            report["primitive_calls"].append({"capacity": capacity, "gathered_shape": shape})
            return original_sampling(*positional, **keywords)

        with patch.object(ttnn, "sampling", side_effect=observed_sampling):
            state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=(0, 1, 2))
            source = gen.model.upload(synthetic_logits(gen.model.vocab_size, candidates), dim=3)
            tie_source = gen.model.upload(synthetic_logits(gen.model.vocab_size, candidates, tie=True), dim=3)
            # Warm diagnostic copies and state resets against the eager model
            # output before any live trace can own their program-cache buffers.
            target = state.logits

            def reset(k, penalties=False, greedy=False):
                params = SamplingParams(
                    temperature=[0.0 if greedy else 1.0] * 3,
                    top_k=[k] * 3,
                    top_p=[1.0] * 3,
                    seed=[42, 43, 44],
                    presence_penalty=0.75 if penalties else 0.0,
                    frequency_penalty=0.5 if penalties else 0.0,
                    repetition_penalty=1.1 if penalties else 1.0,
                )
                gen.configure_sampling(params)
                gen._reset_sampling_state(slots=(0, 1, 2), prompt_tokens=prompts, output_tokens=[[], [], []])
                gen.sampling_state.refresh_active_slots((0, 1, 2))

            def read_tokens():
                replicas = [
                    ttnn.to_torch(shard).reshape(-1)[:3].long() for shard in ttnn.get_device_tensors(state.tokens)
                ]
                assert all(torch.equal(replicas[0], other) for other in replicas[1:])
                return replicas[0].tolist()

            def draw(mode, penalties=False, logits_source=None):
                ttnn.copy(source if logits_source is None else logits_source, target)
                if mode == "eager":
                    gen._sample_logits(target, state.tokens)
                else:
                    ttnn.execute_trace(mesh, state.sample_traces[(False, penalties)], cq_id=0, blocking=True)
                return read_tokens()

            def check_state(outputs, penalties):
                count = len(outputs[0])
                bases = [(_hash_request_seed_to_device_seed(seed, 0) % 500_000) + 1 for seed in (42, 43, 44)]
                for shard in ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor):
                    actual = ttnn.to_torch(shard).reshape(-1)[:3].long().tolist()
                    assert actual == [base + count for base in bases], (actual, bases, count)
                vocab = gen.model.vocab_size
                expected_counts = (
                    torch.stack([torch.bincount(torch.tensor(row), minlength=vocab) for row in outputs])
                    if penalties
                    else torch.zeros(3, vocab, dtype=torch.int64)
                )
                expected_prompt = torch.stack(
                    [torch.bincount(torch.tensor(row), minlength=vocab) > 0 for row in prompts]
                ).long()
                expected = {
                    "prompt_mask": expected_prompt,
                    "output_mask": (expected_counts > 0).long(),
                    "output_counts": expected_counts,
                    "output_counts_gathered": expected_counts,
                }
                for field, reference in expected.items():
                    for rank, shard in enumerate(ttnn.get_device_tensors(getattr(gen.sampler.tt_penalties, field))):
                        actual = ttnn.to_torch(shard).reshape(32, -1)[:3].long()
                        wanted = reference if field == "output_counts_gathered" else reference.chunk(4, dim=-1)[rank]
                        assert torch.equal(actual, wanted), (field, rank)

            programs_before_diagnostics = mesh.num_program_cache_entries()
            warm_ks = (1, 20) if args.expected_capacity == 32 else (1, 20, 40, 64)
            for k in warm_ks:
                for penalties in (False, True):
                    reset(k, penalties, greedy=k == 1)
                    sampled = draw("eager", penalties)
                    check_state([[token] for token in sampled], penalties)
                    gen._reset_sampling_state(
                        slots=(0, 1, 2),
                        prompt_tokens=prompts,
                        output_tokens=[[ids[0]] for ids in candidates] if penalties else [[], [], []],
                        output_counts=[1, 1, 1],
                    )
            reset(1, greedy=True)
            draw("eager", logits_source=tie_source)
            reset(20)
            programs_after_diagnostics = mesh.num_program_cache_entries()
            eager_output_spec = (tuple(target.shape), target.dtype, target.layout, target.memory_config())
            gen.capture(state)
            assert len(state.sample_traces) == 4
            target = state.trace_logits[False]
            assert (tuple(target.shape), target.dtype, target.layout, target.memory_config()) == eager_output_spec
            programs_after_capture = mesh.num_program_cache_entries()
            report["diagnostic_preparation"] = {
                "programs_before": programs_before_diagnostics,
                "programs_after_warmup": programs_after_diagnostics,
                "programs_after_capture": programs_after_capture,
                "copy_shape": list(target.shape),
                "warmed_k_values": list(warm_ks),
                "warmed_penalty_modes": [False, True],
                "tracking_exemptions_added": False,
            }
            greedy = {}
            for tie, name in ((False, "unique"), (True, "ties_across_shards")):
                reset(1, greedy=True)
                eager = draw("eager", logits_source=tie_source if tie else source)
                reset(1, greedy=True)
                traced = draw("trace", logits_source=tie_source if tie else source)
                expected = [min(ids) if tie else ids[0] for ids in candidates]
                assert eager == traced == expected, (name, eager, traced, expected)
                greedy[name] = traced
            report["greedy_outputs"] = greedy
            if args.compare_reference:
                reference = json.loads(args.compare_reference.read_text())
                assert reference["status"] == "pass" and reference["capacity"] == 32
                assert reference["candidate_ids"] == candidates
                assert reference["greedy_outputs"] == greedy
                report["greedy_capacity32_reference"] = str(args.compare_reference)

            for k in (20,) if args.expected_capacity == 32 else (40, 64):
                for penalties in (False, True):
                    trajectories = {}
                    for mode in ("eager", "trace", "resumed_trace"):
                        reset(k, penalties)
                        outputs = [[], [], []]
                        for step in range(args.draws):
                            if mode == "resumed_trace" and step == args.draws // 2:
                                gen._reset_sampling_state(
                                    slots=(0, 1, 2),
                                    prompt_tokens=prompts,
                                    output_tokens=outputs if penalties else [[], [], []],
                                    output_counts=[step] * 3,
                                )
                            values = draw(mode, penalties)
                            for row, token in enumerate(values):
                                assert token in rank_maps[row], (row, token)
                                if not penalties:
                                    assert rank_maps[row][token] < k, (row, token, k)
                                outputs[row].append(token)
                        check_state(outputs, penalties)
                        trajectories[mode] = outputs
                    assert trajectories["eager"] == trajectories["trace"] == trajectories["resumed_trace"]
                    tails = [
                        sum(rank_maps[row][token] >= 32 for token in tokens)
                        for row, tokens in enumerate(trajectories["trace"])
                    ]
                    if k > 32 and not penalties:
                        assert all(tails), tails
                    report["cases"].append(
                        {
                            "k": k,
                            "penalties": penalties,
                            "status": "pass",
                            "draws_per_user": args.draws,
                            "tail_rank32_plus_counts": tails,
                            "outputs": trajectories["trace"],
                            "eager_trace_resume_exact": True,
                            "seed_and_history_exact": True,
                        }
                    )
                    save()

            # Isolated steady sampling trace: no token reads or logits uploads
            # inside the timed batch, and no penalties that mutate input logits.
            trace = state.sample_traces[(False, False)]
            report["isolated_sampling_trace"] = []
            for k in (1, 20) if args.expected_capacity == 32 else (1, 20, 40):
                reset(k, greedy=k == 1)
                ttnn.copy(source, target)
                for _ in range(8):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                started = time.perf_counter()
                for _ in range(64):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                elapsed = time.perf_counter() - started
                report["isolated_sampling_trace"].append(
                    {
                        "k": k,
                        "temperature": 0.0 if k == 1 else 1.0,
                        "active_users": 3,
                        "physical_rows": 32,
                        "steps": 64,
                        "warmup_steps": 8,
                        "seconds": elapsed,
                        "mean_ms_per_step": elapsed * 1000 / 64,
                        "trace_allocation_tracking": report["trace_allocation_tracking"],
                        "auxiliary": True,
                        "scope": "Host wall for queued sampling traces only; excludes model, logits copies, token reads, serving and scheduler. Correctness-run tracking settings are recorded; not full-model performance.",
                    }
                )
            report["diagnostic_preparation"]["programs_after_probes"] = mesh.num_program_cache_entries()
            assert (
                mesh.num_program_cache_entries() == programs_after_capture
            ), "Diagnostic compiled a new post-capture program"
        report["status"] = "pass"
    except BaseException as exc:
        report["status"] = "fail"
        report["error"] = repr(exc)
        raise
    finally:
        save()
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
    print("SAMPLING_CAPACITY_PROBE_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
