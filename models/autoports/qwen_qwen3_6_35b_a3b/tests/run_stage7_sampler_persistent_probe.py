# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full split-sampler A/B: persistent broadcast outputs, changing inputs, queued replay.

This is a hardware runner, never called implicitly by importing the probe helper.
Acceptance requires allocation tracking; clean timing is a separate process.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel
from models.autoports.qwen_qwen3_6_35b_a3b.tt.sampler_persistent_probe import install_persistent_sampler_probe

PARAMETERS = ("k_tensor", "p_tensor", "temp_tensor", "_greedy_col", "seeds_tt_tensor")


def read_ranks(tensor):
    return [ttnn.to_torch(part).reshape(-1).clone() for part in ttnn.get_device_tensors(tensor)]


def copied(model, source):
    # All sampler parameter and token buffers in this TP4 contract are replicated.
    result = model.upload(torch.zeros(tuple(source.shape)), source.dtype, layout=source.layout)
    ttnn.copy(source, result)
    return result


def restore(generator, parameters):
    for name, value in parameters.items():
        ttnn.copy(value, getattr(generator.sampler.tt_sampling, name))


def rank_diagnostics(tensor, expected=None):
    reports = []
    for rank, value in enumerate(read_ranks(tensor)):
        flat = value.float()
        report = dict(
            rank=rank,
            shape=list(tensor.shape),
            address=tensor.buffer_address(),
            nan_count=int(torch.isnan(flat).sum()),
            poison_65535_count=int((flat == 65535).sum()),
            first_16=flat[:16].tolist(),
            last_16=flat[-16:].tolist(),
        )
        if expected is not None:
            target = expected.reshape(-1).to(value.dtype)
            mismatch = value != target
            report["mismatch_count"] = int(mismatch.sum())
            report["first_mismatch_offsets"] = torch.nonzero(mismatch).reshape(-1)[:16].tolist()
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("acceptance", "timing"), default="acceptance")
    parser.add_argument("--num-links", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=64, help="Queued changing-input replay steps per parameter case")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert args.steps >= 8 and args.iterations > 0 and args.repeats > 0
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == ("1" if args.mode == "acceptance" else "0")
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
    if args.mode == "timing":
        for name in (
            "TT_METAL_WATCHER",
            "TT_METAL_DEVICE_PROFILER",
            "TTNN_OP_PROFILER",
            "TT_METAL_PROFILER_TRACE_TRACKING",
        ):
            assert os.environ.get(name, "0") in ("", "0"), name
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    checkout = root.parents[2]
    sources = [
        Path(__file__),
        root / "tt/sampler_persistent_probe.py",
        root / "tt/generator.py",
        root / "tt/model.py",
        checkout / "models/common/sampling/tt_sampling.py",
    ]
    ccl_source = checkout / "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async"
    sources += [
        ccl_source / "all_gather_async_nanobind.cpp",
        ccl_source / "device/all_gather_async_device_operation.cpp",
        ccl_source / "device/all_gather_async_default_program_factory.cpp",
        ccl_source / "device/all_gather_via_broadcast_factory.cpp",
        ccl_source / "device/kernels/broadcast_rm_writer.cpp",
        ccl_source / "device/kernels/minimal_default_reader.cpp",
        ccl_source / "device/kernels/minimal_default_writer.cpp",
    ]
    provenance = {str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    diagnostics = []
    diagnostic_path = Path(args.output).with_suffix(".diagnostics.json")
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)

    def record_diagnostic(kind, **data):
        entry = dict(kind=kind, **data)
        diagnostics.append(entry)
        diagnostic_path.write_text(json.dumps(dict(source_sha256=provenance, records=diagnostics), indent=2) + "\n")
        print("PERSISTENT_SAMPLER_DIAGNOSTIC", entry, flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    traces = {}
    try:
        model = QwenModel(mesh, layer_indices=(), max_seq_len=256)
        generators = {name: QwenGenerator(model, max_batch_size=32) for name in ("baseline", "persistent_broadcast")}
        for generator in generators.values():
            generator._set_active_slots(tuple(range(32)))
            generator.configure_sampling()
        ccl = install_persistent_sampler_probe(generators["persistent_broadcast"], num_links=args.num_links)
        ids = model.upload(
            torch.arange(32, dtype=torch.int32).reshape(1, 32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        real = model.terminal(ttnn.reshape(model.embed(ids), (1, 32, 2048)))
        reference = model.host_logits(real).reshape(32, model.vocab_size).float()
        hosts = [reference]
        local_vocab = model.padded_vocab_size // 4
        for variant in range(3):
            # Every row changes; peaks rotate across every rank and local position.
            host = torch.full_like(reference, -64.0 - variant)
            for row in range(32):
                for peak in range(64):
                    rank = (row + variant + peak) % 4
                    index = rank * local_vocab + (row * 193 + variant * 4099 + peak * 37 + 17) % local_vocab
                    if index < model.vocab_size:
                        host[row, index] = -1.0 - peak / 8 if variant == 0 else 16.0 - peak / 8
                if variant == 2:
                    host[row, 7 + row] = 32
                    host[row, local_vocab + 117 + row] = 32
            hosts.append(host.bfloat16().float())
        fixtures = []
        for host in hosts:
            value = torch.nn.functional.pad(host, (0, model.padded_vocab_size - model.vocab_size), value=-float("inf"))
            fixtures.append(model.upload(value.reshape(1, 1, 32, -1), ttnn.bfloat16, shard_dim=3))
        logits_poison = model.upload(
            torch.full((1, 1, 32, model.padded_vocab_size), float("nan")), ttnn.bfloat16, shard_dim=3
        )
        cases = {
            "greedy": dict(temperature=0.0, top_k=1, top_p=1.0),
            "topk8": dict(temperature=0.8, top_k=8, top_p=1.0),
            "topk32_topp": dict(temperature=1.0, top_k=32, top_p=0.75),
            "mixed_rows": dict(
                temperature=[0.0 if row % 3 == 0 else 0.8 for row in range(32)],
                top_k=[(1, 8, 32)[row % 3] for row in range(32)],
                top_p=[(1.0, 1.0, 0.75)[row % 3] for row in range(32)],
            ),
        }
        parameters = {}
        for name, config in cases.items():
            generator = generators["baseline"]
            generator.configure_sampling(**config, seed=[1907 + row * 23 for row in range(32)])
            parameters[name] = {
                field: copied(model, getattr(generator.sampler.tt_sampling, field)) for field in PARAMETERS
            }
        histories = {
            name: [
                (copied(model, gen.tokens), copied(model, gen.sampler.tt_sampling.seeds_tt_tensor))
                for _ in range(args.steps)
            ]
            for name, gen in generators.items()
        }
        poisons = {
            "SAMPLING_VALUES": model.upload(torch.full((1, 1, 32, 128), float("nan")), ttnn.bfloat16),
            "SAMPLING_INDICES": model.upload(torch.full((1, 1, 32, 128), 65535, dtype=torch.int32), ttnn.uint16),
        }
        # Isolate transport from sampling, including first-call vs cache-hit
        # address rebinding. Distinct source allocations carry distinct rank,
        # row, and column sentinels. Read all ranks before any trace capture.
        raw_sources = []
        for key, dtype in (("SAMPLING_VALUES", ttnn.bfloat16), ("SAMPLING_INDICES", ttnn.uint16)):
            for variant in range(2):
                columns = torch.arange(128).reshape(1, 128)
                rows = torch.arange(32).reshape(32, 1)
                expected = rows * 128 + columns + variant * 8192
                if dtype == ttnn.bfloat16:
                    expected = (expected / 16).bfloat16().float()
                raw = model.upload(expected.reshape(1, 1, 32, 128), dtype, shard_dim=3)
                raw_sources.append(raw)
                ttnn.copy(poisons[key], ccl.buffers[key])
                output = ccl.line_all_gather(
                    raw, dim=-1, cluster_axis=None, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_links=2, buffer_key=key
                )
                ttnn.synchronize_device(mesh)
                report = rank_diagnostics(output, expected)
                record_diagnostic(
                    "raw_gather",
                    buffer_key=key,
                    variant=variant,
                    input_address=raw.buffer_address(),
                    output_address=output.buffer_address(),
                    ranks=report,
                )
                assert all(item["mismatch_count"] == 0 for item in report), ("raw gather failed", key, variant)

        def sampler_intermediates(kind):
            values = read_ranks(ccl.buffers["SAMPLING_VALUES"])
            indices = read_ranks(ccl.buffers["SAMPLING_INDICES"])
            reports = []
            for rank, (value, index) in enumerate(zip(values, indices)):
                index = index.long().reshape(32, 128)
                value = value.float().reshape(32, 128)
                offsets = torch.arange(4).repeat_interleave(32) * local_vocab
                global_ids = index + offsets
                valid = (index < local_vocab) & (global_ids < model.vocab_size)
                expected = hosts[0].gather(1, global_ids.clamp(0, model.vocab_size - 1))
                matches = valid & (value == expected)
                reports.append(
                    dict(
                        rank=rank,
                        valid_indices=int(valid.sum()),
                        exact_value_index_pairs=int(matches.sum()),
                        total_pairs=value.numel(),
                    )
                )
            record_diagnostic(
                kind,
                values=rank_diagnostics(ccl.buffers["SAMPLING_VALUES"]),
                indices=rank_diagnostics(ccl.buffers["SAMPLING_INDICES"]),
                pairs=reports,
            )
            assert all(item["exact_value_index_pairs"] == item["total_pairs"] for item in reports), kind

        # One eager full-sampler comparison isolates native ownership/transport
        # from trace capture. Inspect intermediates before allocating captures.
        eager_tokens = {}
        for name, generator in generators.items():
            restore(generator, parameters["greedy"])
            ttnn.copy(fixtures[0], generator.sampler_logits)
            if name == "persistent_broadcast":
                for key, poison in poisons.items():
                    ttnn.copy(poison, ccl.buffers[key])
            generator._sampling_forward()
            ttnn.synchronize_device(mesh)
            eager_tokens[name] = read_ranks(generator.tokens)
        sampler_intermediates("eager_sampler_intermediates")
        record_diagnostic(
            "eager_sampler_tokens",
            tokens={name: [rank.tolist() for rank in values] for name, values in eager_tokens.items()},
        )
        assert all(torch.equal(left, right) for left, right in zip(*eager_tokens.values())), "eager sampler A/B"
        # Warm every operation signature and every per-rank read layout BEFORE
        # the first capture. No cache-skipping/late-allocation exemptions.
        for name, generator in generators.items():
            for parameter_set in parameters.values():
                restore(generator, parameter_set)
            ttnn.copy(logits_poison, generator.sampler_logits)
            for fixture in fixtures:
                ttnn.copy(fixture, generator.sampler_logits)
                generator._sampling_forward()
            for tokens, seeds in histories[name]:
                ttnn.copy(generator.tokens, tokens)
                ttnn.copy(generator.sampler.tt_sampling.seeds_tt_tensor, seeds)
                read_ranks(tokens)
                read_ranks(seeds)
            restore(generator, parameters["greedy"])
            ttnn.copy(fixtures[0], generator.sampler_logits)
        for key, poison in poisons.items():
            ttnn.copy(poison, ccl.buffers[key])
        generators["persistent_broadcast"]._sampling_forward()
        ccl.freeze()
        ttnn.synchronize_device(mesh)
        for name, generator in generators.items():
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                generator._sampling_forward()
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces[name] = trace

        # Single replay comes before queued stress. Reads are already warmed by
        # the eager diagnostics; this discriminates captured address bindings.
        replay_tokens = {}
        for name, generator in generators.items():
            restore(generator, parameters["greedy"])
            ttnn.copy(fixtures[0], generator.sampler_logits)
            if name == "persistent_broadcast":
                for key, poison in poisons.items():
                    ttnn.copy(poison, ccl.buffers[key])
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            replay_tokens[name] = read_ranks(generator.tokens)
        sampler_intermediates("single_replay_intermediates")
        record_diagnostic(
            "single_replay_tokens",
            tokens={name: [rank.tolist() for rank in values] for name, values in replay_tokens.items()},
        )
        assert all(torch.equal(left, right) for left, right in zip(*replay_tokens.values())), "single replay A/B"

        records = []
        for case, config in cases.items():
            results = {}
            for name, generator in generators.items():
                restore(generator, parameters[case])
                # One queued burst, no per-token host synchronization/readback.
                # Poison old gather outputs before EVERY candidate invocation.
                for step in range(args.steps):
                    fixture_id = (step * 3 + step // 4) % len(fixtures)
                    ttnn.copy(logits_poison, generator.sampler_logits)
                    ttnn.copy(fixtures[fixture_id], generator.sampler_logits)
                    if name == "persistent_broadcast":
                        for key, poison in poisons.items():
                            ttnn.copy(poison, ccl.buffers[key])
                    ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                    tokens, seeds = histories[name][step]
                    ttnn.copy(generator.tokens, tokens)
                    ttnn.copy(generator.sampler.tt_sampling.seeds_tt_tensor, seeds)
                ttnn.synchronize_device(mesh)
                results[name] = [(read_ranks(tokens), read_ranks(seeds)) for tokens, seeds in histories[name]]
            for step in range(args.steps):
                fixture_id = (step * 3 + step // 4) % len(fixtures)
                host = hosts[fixture_id]
                baseline, candidate = (results[name][step] for name in generators)
                for state_kind, (left, right) in enumerate(zip(baseline, candidate)):
                    assert len(left) == len(right) == 4
                    for rank, (expected, actual) in enumerate(zip(left, right)):
                        assert torch.equal(expected, actual), (case, step, state_kind, rank, expected, actual)
                        assert torch.equal(actual, right[0]), ("rank disagreement", case, step, state_kind, rank)
                tokens = candidate[0][0].long()
                expected_seeds = torch.arange(32, dtype=torch.int64) * 23 + 1908 + step + 1
                assert torch.equal(candidate[1][0].long(), expected_seeds), ("device seed advance", case, step)
                assert tokens.numel() == 32 and torch.all((tokens >= 0) & (tokens < model.vocab_size))
                for row in range(32):
                    k = config["top_k"][row] if isinstance(config["top_k"], list) else config["top_k"]
                    threshold = host[row].topk(k).values[-1]
                    assert host[row, tokens[row]] >= threshold, ("outside global top-k", case, step, row, tokens[row])
                    if k == 1:
                        assert tokens[row] == host[row].argmax(), ("greedy lowest-id mismatch", case, step, row)
            ccl.assert_live()
            records.append(
                dict(
                    case=case,
                    queued_steps=args.steps,
                    all_32_rows=True,
                    all_4_ranks=True,
                    exact_tokens_and_seeds=True,
                    poisoned_previous_outputs=True,
                    poisoned_previous_logits=True,
                )
            )
            print("PERSISTENT_SAMPLER_GATE", records[-1], flush=True)

        timings = {}
        if args.mode == "timing":
            for name, generator in generators.items():
                restore(generator, parameters["greedy"])
                ttnn.copy(fixtures[0], generator.sampler_logits)
                for _ in range(5):
                    ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - start) * 1000 / args.iterations)
                timings[name] = dict(median_ms=statistics.median(samples), samples_ms=samples)
                print("PERSISTENT_SAMPLER_TIMING", name, timings[name], flush=True)
        ccl.assert_live()
        result = dict(
            passed=True,
            mode=args.mode,
            timing_eligible=args.mode == "timing",
            mesh=[1, 4],
            records=records,
            timings=timings,
            candidate_num_links=args.num_links,
            baseline_num_links=2,
            persistent_bytes_per_rank=16384,
            persistent_addresses=ccl.addresses,
            trace_tracking=args.mode == "acceptance",
            program_cache_tracking=True,
            scope="Full selected split sampler, tt_out_tok feedback and device seed advance; excludes model forward",
            timing_scope="Nonblocking repeated full sampler trace only; no poison, input copies or history snapshots",
            ownership_hook="candidate-instance forward only; skip force deallocation of CCL-owned gathered values",
            rank_skew_scope="Queued stress plus native unconditional init/final barrier source audit; no injected rank delay",
            source_sha256=provenance,
            forward_source_sha256=ccl.forward_source_sha256,
            patched_forward_sha256=ccl.patched_forward_sha256,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
