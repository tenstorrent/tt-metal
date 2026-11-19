# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare common greedy implementations on the real TP4 terminal contract."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("acceptance", "timing"), default="acceptance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == ("1" if args.mode == "acceptance" else "0")
    if args.mode == "timing":
        assert os.environ.get("TT_METAL_WATCHER", "0") in ("0", "")
        assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("0", "")
        assert os.environ.get("TTNN_OP_PROFILER", "0") in ("0", "")
        assert os.environ.get("TT_METAL_PROFILER_TRACE_TRACKING", "0") in ("0", "")
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
    root = Path(__file__).parents[1]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    traces = {}
    try:
        model = QwenModel(mesh, layer_indices=(), max_seq_len=256)
        gen = QwenGenerator(model, max_batch_size=32)
        gen._set_active_slots(tuple(range(32)))
        gen.configure_sampling()
        ids = model.upload(
            torch.arange(32, dtype=torch.int32).reshape(1, 32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        logits = model.terminal(ttnn.reshape(model.embed(ids), (1, 32, 2048)))
        reference = model.host_logits(logits).reshape(32, model.vocab_size).float()
        ttnn.copy(ttnn.reshape(logits, (1, 1, 32, logits.shape[-1])), gen.sampler_logits)
        del logits, ids
        modes = (("split_greedy", False, False), ("split_greedy_power2", False, True), ("force_argmax", True, False))
        for name, force, power2 in modes:
            gen.sampler.tt_sampling._force_argmax_sampling = force
            gen.sampler.tt_sampling.pad_to_power_of_2 = power2
            gen._sampling_forward()
        ttnn.synchronize_device(mesh)
        for name, force, power2 in modes:
            gen.sampler.tt_sampling._force_argmax_sampling = force
            gen.sampler.tt_sampling.pad_to_power_of_2 = power2
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                gen._sampling_forward()
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces[name] = trace
        records = []
        fixtures = [("real_terminal", reference)]
        negative = torch.full_like(reference, -20)
        for row in range(32):
            negative[row, (row * 62081 + 17) % model.vocab_size] = -1
        fixtures.append(("all_negative_unique_max", negative))
        tied = negative.clone()
        tied[:, 7] = 3
        tied[:, 70000] = 3
        fixtures.append(("cross_shard_ties", tied))
        for fixture, host in fixtures:
            if fixture != "real_terminal":
                value = torch.nn.functional.pad(host, (0, model.padded_vocab_size - model.vocab_size))
                staging = ttnn.from_torch(
                    value.reshape(1, 1, 32, -1).bfloat16(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
                )
                ttnn.copy_host_to_device_tensor(staging, gen.sampler_logits)
            maxima = host.max(-1).values
            for name, trace in traces.items():
                for _ in range(3):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                selected = torch.tensor(gen._read_tokens(range(32)))
                greedy = torch.equal(host[torch.arange(32), selected], maxima)
                assert greedy, (fixture, name, selected.tolist(), host.argmax(-1).tolist())
                begin = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - begin) * 1000 / args.iterations
                records.append(
                    dict(
                        fixture=fixture,
                        strategy=name,
                        semantically_greedy=greedy,
                        matches_lowest_id=torch.equal(selected, host.argmax(-1)),
                        milliseconds=ms,
                    )
                )
                print("SAMPLER_CONTRACT", records[-1], flush=True)
        result = {
            "passed": True,
            "records": records,
            "mesh": [1, 4],
            "vocab_size": model.vocab_size,
            "physical_rows": 32,
            "iterations": args.iterations,
            "mode": args.mode,
            "timing_eligible": args.mode == "timing",
            "scope": "Terminal-contract sampler only; no decoder layers, not full-model headline throughput",
            "trace_tracking": args.mode == "acceptance",
            "program_cache_tracking": True,
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), root / "tt/generator.py", root / "tt/model.py"]
            },
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
