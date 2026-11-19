# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""No-model BF16 terminal row assembly candidate; no production policy change.

Device owner only. Clean timing:
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
  timeout 240 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_terminal_row_assembly \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_rows_v1.json

Use --tracked-correctness with TT_METAL_TRACE_ALLOC_TRACKING=1 and
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 separately; that mode emits no timing.
All tensors/programs are prepared before first capture. Baseline is the actual
logical-row concat/copy expression. Candidate copies an immutable inactive
physical32 template then uses the existing tiled BF16 write_token_chunk helper.
Both keep TP4 vocab sharding and common split greedy; no model weights loaded.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch


def preflight(tracked):
    assert "TT_METAL_WATCHER" not in os.environ, "Watcher must be absent, including value0"
    for name in (
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_PROFILE_PERF_COUNTERS",
        "TT_METAL_STREAMING_PROFILER",
        "TT_METAL_DEVICE_PROFILER_STREAMING",
        "TT_METAL_TRACE_PROFILER",
    ):
        assert os.environ.get(name) in (None, "", "0"), name
    assert os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if tracked:
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    else:
        for name in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS"):
            assert os.environ.get(name) in (None, "", "0"), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--tracked-correctness", action="store_true")
    parser.add_argument(
        "--fill-candidate",
        action="store_true",
        help="Also compare native in-place BF16 fill plus tiled row writes (no template allocation needed in production)",
    )
    args = parser.parse_args()
    assert args.iterations > 0 and not args.output.exists()
    preflight(args.tracked_correctness)
    from ttnn.tools import trace_allocation_tracker as tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import Collectives
    from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params

    if not args.tracked_correctness:
        assert not tracker.TRACE_ALLOC_TRACKING and not tracker.TRACE_ALLOC_DIAGNOSTICS
    report = {
        "passed": False,
        "scope": "no_model_terminal_row_assembly_candidate",
        "acceptance_eligible": False,
        "tracked_correctness": args.tracked_correctness,
        "policy": "TP4 BF16 tiled DRAM vocab-sharded logits; unchanged common split greedy",
        "checks": [],
        "timings": {},
        "iterations": args.iterations,
    }
    mesh, samplers, traces = None, {}, {}
    started = time.monotonic()
    phase = "setup"

    def mark(value):
        nonlocal phase
        phase = value
        print(json.dumps({"phase": value, "wall_seconds": time.monotonic() - started}), flush=True)

    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        vocab = 248320
        host_rows = (torch.arange(vocab).remainder(17).float() / 17).repeat(32, 1).to(torch.bfloat16)
        for row in range(32):
            host_rows[row, (row * 32003 + 37) % vocab] = 20

        def upload(value):
            return ttnn.from_torch(
                value,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )

        inputs = [upload(row.reshape(1, 1, 1, vocab)) for row in host_rows]
        inactive = upload(torch.full((1, 1, 1, vocab), -100.0, dtype=torch.bfloat16))
        template_host = torch.full((1, 1, 32, vocab), -100.0, dtype=torch.bfloat16)
        template = upload(template_host)
        strategies = ("concat", "tiled_row_copy") + (("tiled_row_fill",) if args.fill_candidate else ())
        report["strategies"] = list(strategies)
        report[
            "fill_dispatch"
        ] = "BF16 TILE DEVICE full_like -> native fill with persistent optional_tensor; no host write"
        outputs = {name: upload(template_host) for name in strategies}
        # Device-resident poison is allocated before capture. Correctness writes
        # it into every destination to prove inactive rows are overwritten even
        # after NaN/Inf; this copy is never part of the measured candidate path.
        poison_host = template_host.clone()
        poison_host[:, :, 0::2] = float("nan")
        poison_host[:, :, 1::2] = float("inf")
        poison = upload(poison_host)
        tokens = {
            name: ttnn.from_torch(
                torch.zeros(1, 1, 1, 32, dtype=torch.int32),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for name in outputs
        }
        cases = {
            "b1_slot0": ((0, 0),),
            "b1_slot31": ((31, 7),),
            "b2_mixed": ((0, 1), (7, 3)),
            "b2_reordered": ((7, 1), (0, 3)),
            **{f"b{width}": tuple((row, row) for row in range(width)) for width in (8, 16, 24, 28)},
            "b32": tuple((row, row) for row in range(32)),
        }
        ccl = Collectives(mesh)
        ccl.prepare_sampling_collectives()
        config = SimpleNamespace(
            vocab_size=vocab,
            padded_vocab_size=vocab,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=(1, 4),
            sampling_all_gather_axis=1,
            model_config={
                "SAMPLING_AG_CONFIG": {"allow_force_argmax": False, "num_links": 1, "topology": ttnn.Topology.Ring}
            },
        )
        for name in outputs:
            sampler = SamplingGenerator(args=config, mesh_device=mesh, tt_ccl=ccl)
            sampler.tt_sampling._allow_penalties_sampling = False
            sampler.reset_sampling_params(
                format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=0.0), 32)
            )
            samplers[name] = sampler

        def assemble(name, mapping):
            if name == "concat":
                rows = {slot: inputs[row] for slot, row in mapping}
                combined = ttnn.concat([rows.get(slot, inactive) for slot in range(32)], dim=2)
                ttnn.copy(combined, outputs[name])
            else:
                if name == "tiled_row_fill":
                    ttnn.full_like(outputs[name], -100.0, optional_tensor=outputs[name])
                else:
                    ttnn.copy(template, outputs[name])
                for slot, row in mapping:
                    write_token_chunk(inputs[row], outputs[name], slot)

        def sample(name):
            samplers[name]._run_sampling(outputs[name], penalties_on=False, grammar_on=False, tt_out_tok=tokens[name])

        mark("warm_all_variants")
        for name in outputs:
            for mapping in cases.values():
                assemble(name, mapping)
            samplers[name].precompile(
                outputs[name], tt_out_tok=tokens[name], all_configs=False, compile_token_update=True
            )
            sample(name)
        ttnn.synchronize_device(mesh)
        cache_count = mesh.num_program_cache_entries()

        def capture(key, body):
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                body()
            except BaseException:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.release_trace(mesh, trace)
                raise
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[key] = trace

        mark("capture")
        for name in outputs:
            for label, mapping in cases.items():
                capture((name, label), lambda name=name, mapping=mapping: assemble(name, mapping))
            capture((name, "sample"), lambda name=name: sample(name))
        assert mesh.num_program_cache_entries() == cache_count

        def replay(key):
            ttnn.execute_trace(mesh, traces[key], cq_id=0, blocking=False)

        def read_logits(name):
            return ttnn.to_torch(outputs[name], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))

        mark("exact_logits_and_common_greedy")
        # Revisit small active sets after B32 to prove inactive rows are cleared.
        for label in [*cases, "b1_slot31", "b2_mixed", "b1_slot0"]:
            expected = template_host.clone()
            for slot, row in cases[label]:
                expected[0, 0, slot] = host_rows[row]
            observed = {}
            for name in outputs:
                for _ in range(2):
                    ttnn.copy(poison, outputs[name])
                    replay((name, label))
                    assert torch.equal(read_logits(name), expected), (name, label, "logit mismatch")
                    replay((name, "sample"))
                    shards = [ttnn.to_torch(t).flatten().long() for t in ttnn.get_device_tensors(tokens[name])]
                    assert all(torch.equal(shards[0], shard) for shard in shards[1:])
                    selected = expected[0, 0, torch.arange(32), shards[0]]
                    assert torch.equal(selected, expected[0, 0].max(-1).values)
                    observed[name] = shards[0].tolist()
            assert all(value == observed["concat"] for value in observed.values())
            report["checks"].append(
                {
                    "case": label,
                    "logits_exact": True,
                    "nonfinite_poison_overwritten": True,
                    "tokens": observed["concat"],
                }
            )
        if not args.tracked_correctness:
            mark("clean_traced_timing")

            def timing(operation):
                for _ in range(2):
                    operation()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for _ in range(args.iterations):
                    operation()
                ttnn.synchronize_device(mesh)
                return (time.perf_counter() - begin) * 1000 / args.iterations

            for label in cases:
                report["timings"][label] = {}
                for name in outputs:

                    def assembly_sample(name=name, label=label):
                        replay((name, label))
                        replay((name, "sample"))

                    report["timings"][label][name] = {
                        "assembly_ms": timing(lambda name=name, label=label: replay((name, label))),
                        "assembly_plus_common_split_greedy_ms": timing(assembly_sample),
                    }
            report["timing_boundary"] = "Warmed trace submissions plus amortized boundary sync; no model or token read"
        assert mesh.num_program_cache_entries() == cache_count
        report["program_cache_entries"] = cache_count
        report["trace_count"] = len(traces)
        # Finish with actual replay after every correctness read/timing window.
        replay((strategies[-1], "b1_slot0"))
        replay((strategies[-1], "sample"))
        ttnn.synchronize_device(mesh)
        report["passed"] = True
    except BaseException as error:
        report.update(failed_phase=phase, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            for sampler in samplers.values():
                sampler.reset_trace()
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
