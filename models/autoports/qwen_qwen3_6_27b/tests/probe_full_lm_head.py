# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused real-weight BF16/HiFi4 TP4 terminal comparison; never a policy override.

Parent-owned hardware command, after normal environment/device checks:
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_full_lm_head \\
    --candidates whole_r2_k1 split2_r1_k1 split2_r2_k2 --output <result.json>

Only reduced Model[0,3] weights and its unchanged common sampler are loaded.
All candidates, graph signatures, and stable inputs are prepared before capture.
The trace includes final norm/gather, optional input reshard, head, output
interleave/trim/concat, and the exact stable sampler logits staging. Timing
also measures the common sample trace plus token readback. No device execution
occurs on import or during py_compile. A failed candidate aborts this process;
its failure is recorded, never silently replaced with another implementation.
"""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_model_evidence import source_identity
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model

CANDIDATES = {
    "whole_r2_k1": dict(splits=1, readers=2, block=1),
    "whole_r3_k1": dict(splits=1, readers=3, block=1),
    "split2_r1_k1": dict(splits=2, readers=1, block=1),
    "split2_r2_k2": dict(splits=2, readers=2, block=2),
}


def prepare_candidate(model, name):
    """Mirror LMHead1D explicit DRAM path without changing its default fidelity."""
    spec = CANDIDATES[name]
    mesh = model.mesh
    k, logical_n = tuple(model.lm_head.shape)
    assert (k, logical_n) == (5120, 62080)
    grid = mesh.dram_grid_size()
    banks = grid.x * grid.y
    assert banks == 8, "This focused candidate geometry targets the measured eight-bank Blackhole rank"
    cores = 8
    storage_tiles = 32
    assert k % (32 * cores) == 0 and (k // 32) % spec["block"] == 0
    assert logical_n % spec["splits"] == 0
    width = logical_n // spec["splits"]
    alignment = 32 * math.lcm(banks * spec["readers"], cores)
    physical = math.ceil(width / alignment) * alignment
    output_cores = math.ceil(physical / (32 * storage_tiles))
    workers_grid = mesh.compute_with_storage_grid_size()
    assert output_cores <= workers_grid.x * workers_grid.y
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    weight_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (k, physical // banks), ttnn.ShardOrientation.ROW_MAJOR),
    )
    workers = ttnn.num_cores_to_corerangeset(cores, workers_grid, row_wise=True)
    input_mem = ttnn.create_sharded_memory_config(
        (32, k // cores), workers, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
    )
    program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=spec["block"],
        per_core_M=1,
        # Output storage is independent of the eight input shards/readers.
        per_core_N=storage_tiles,
        num_workers_per_dram_bank=spec["readers"],
        use_multicore_pipeline=False,
    )
    weights = []
    for part in range(spec["splits"]):
        # Slice each mesh-local vocabulary shard, never the global TP axis.
        weight = ttnn.slice(model.lm_head, (0, part * width), (k, (part + 1) * width))
        if physical != width:
            weight = ttnn.pad(weight, ((0, 0), (0, physical - width)), value=0.0)
        weights.append(ttnn.to_memory_config(weight, weight_mem))
    geometry = dict(
        **spec,
        input_cores=cores,
        dram_banks=banks,
        local_logical_width=logical_n,
        split_logical_width=width,
        split_physical_width=physical,
        output_storage_cores=output_cores,
        output_storage_tiles=storage_tiles,
        input_shard_shape=[32, k // cores],
        weight_shard_shape=[k, physical // banks],
        program=str(program),
        weight_dtype=str(weights[0].dtype),
        compute_kernel_config=str(model.io.ckc),
    )

    def head(hidden):
        normalized = model.io._norm(hidden, "final_norm", decode=True)
        x = ttnn.to_memory_config(ttnn.reshape(normalized, (1, 1, hidden.shape[1], k)), input_mem)
        pieces = []
        for weight in weights:
            output = ttnn.linear(
                x,
                weight,
                compute_kernel_config=model.io.ckc,
                program_config=program,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if physical != width:
                output = ttnn.slice(output, (0, 0, 0, 0), (1, 1, hidden.shape[1], width))
            pieces.append(output)
        # Padded zero columns are removed before common sampling. Leaving them
        # present could let an invalid ID beat negative legitimate logits.
        return pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return head, geometry


def compare_logits(reference, actual):
    ref, got = reference.float(), actual.float()
    rc, gc = ref - ref.mean(-1, keepdim=True), got - got.mean(-1, keepdim=True)
    denominator = rc.square().sum(-1).sqrt() * gc.square().sum(-1).sqrt()
    pcc = (rc * gc).sum(-1) / denominator
    target = ref.argmax(-1)
    metrics = dict(
        finite=bool(torch.isfinite(got).all()),
        minimum_row_pcc=float(pcc.min()),
        max_abs_error=float((ref - got).abs().max()),
        reference_greedy=target.tolist(),
        candidate_greedy=got.argmax(-1).tolist(),
        top1_agreement=float((got.argmax(-1) == target).float().mean()),
        top5_contains_reference_greedy=float((got.topk(5).indices == target[:, None]).any(-1).float().mean()),
        top100_contains_reference_greedy=float((got.topk(100).indices == target[:, None]).any(-1).float().mean()),
    )
    metrics["passed"] = metrics["finite"] and metrics["minimum_row_pcc"] >= 0.9999 and metrics["top1_agreement"] == 1.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    torch.set_num_threads(8)
    report = dict(
        passed=False,
        source_sha256=source_identity(),
        scope="Focused terminal candidate experiment; no production policy change or full-model acceptance",
        batch=args.batch,
        layers=[0, 3],
        iterations=args.iterations,
        repeats=args.repeats,
        dtype="BF16 weights, activations and output",
        fidelity="Same model.io.ckc: HiFi4, FP32 dest, packer accumulation, math_approx=False",
        environment={
            k: os.environ.get(k)
            for k in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
        candidates={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    mesh = gen = None
    phase = "open"
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
        print("LM_HEAD LOAD reduced model", flush=True)
        model = Model(mesh, layer_indices=[0, 3])
        gen = QwenGenerator(model, batch_size=args.batch, max_seq_len=512)
        # Real embeddings plus three independent deterministic BF16 activation
        # cases. Real model weights are retained unchanged on every candidate.
        ids = gen._upload(torch.arange(100, 100 + args.batch, dtype=torch.int32).reshape(1, args.batch), ttnn.uint32)
        cases = [("embedding_rows", model.embeddings(ids))]
        for seed in (11, 29, 47):
            host = torch.randn(
                (1, args.batch, 5120), generator=torch.Generator().manual_seed(seed), dtype=torch.bfloat16
            )
            cases.append((f"bf16_seed{seed}", model.io._upload_shards(list(host.chunk(4, -1)))))
        hidden = ttnn.clone(cases[0][1])
        heads = {"current": lambda x: model.terminal(x, decode=True)}
        for name in args.candidates:
            phase = "prepare:" + name
            print("LM_HEAD " + phase, flush=True)
            head, geometry = prepare_candidate(model, name)
            heads[name] = head
            report["candidates"][name] = dict(geometry=geometry, accuracy=[], timing=[])
        report["candidates"]["current"] = dict(geometry="unchanged Model.terminal", accuracy=[], timing=[])
        graphs = {}
        for name, head in heads.items():

            def graph(head=head):
                value = head(hidden)
                value = ttnn.reshape(value, (1, 1, args.batch, 62080))
                if args.batch < 32:
                    value = ttnn.pad(value, ((0, 0), (0, 0), (0, 32 - args.batch), (0, 0)), value=0)
                ttnn.copy(value, gen.logits)

            graphs[name] = graph
        phase = "warm_all_before_capture"
        print("LM_HEAD " + phase, flush=True)
        for _, source in cases:
            ttnn.copy(source, hidden)
        for name, graph in graphs.items():
            phase = "warm:" + name
            print("LM_HEAD " + phase, flush=True)
            graph()
            graph()
        gen.configure_sampling(temperature=0.0, top_k=1, top_p=0.0, seed=1700)
        gen._sample_graph()
        gen._sample_graph()
        for name, graph in graphs.items():
            phase = "capture:" + name
            gen._capture(name, graph)
        gen._capture("sample", gen._sample_graph)
        programs = int(mesh.num_program_cache_entries())
        trace_ids = {key: str(value) for key, value in gen.traces.items()}
        report["program_cache_before"] = programs
        report["trace_ids_before"] = trace_ids
        phase = "accuracy"
        for case, source in cases:
            print("LM_HEAD accuracy " + case, flush=True)
            ttnn.copy(source, hidden)
            gen._execute("current")
            reference = gen._read_logits(gen.logits)[0, 0, : args.batch].clone()
            gen._execute("sample")
            baseline_tokens = gen._read_tokens()
            for name in heads:
                gen._execute(name)
                actual = gen._read_logits(gen.logits)[0, 0, : args.batch]
                metrics = compare_logits(reference, actual)
                gen._execute("sample")
                tokens = gen._read_tokens()
                metrics.update(case=case, common_sampled_ids=tokens, sample_matches_current=tokens == baseline_tokens)
                selected = actual.float()[torch.arange(args.batch), torch.tensor(tokens)]
                metrics["sample_semantically_greedy"] = bool(torch.equal(selected, actual.float().max(-1).values))
                metrics["passed"] &= metrics["sample_matches_current"] and metrics["sample_semantically_greedy"]
                report["candidates"][name]["accuracy"].append(metrics)
                save()
            del reference, actual
        phase = "timing"
        names = list(heads)
        for repeat in range(args.repeats):
            for name in names[repeat % len(names) :] + names[: repeat % len(names)]:
                for token_out in (False, True):
                    print(f"LM_HEAD timing {name} repeat={repeat} token_out={token_out}", flush=True)
                    ttnn.synchronize_device(mesh)
                    before = dict(gen.counters)
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        gen._execute(name)
                        if token_out:
                            gen._execute("sample")
                            gen._read_tokens()
                    ttnn.synchronize_device(mesh)
                    ms = (time.perf_counter() - start) * 1000 / args.iterations
                    counters = {key: gen.counters[key] - before.get(key, 0) for key in gen.counters}
                    assert counters.get("full_logits_readbacks", 0) == 0
                    report["candidates"][name]["timing"].append(
                        dict(repeat=repeat, token_out=token_out, ms_per_iteration=ms, counters=counters)
                    )
                    save()
        for row in report["candidates"].values():
            row["median_head_and_staging_ms"] = statistics.median(
                x["ms_per_iteration"] for x in row["timing"] if not x["token_out"]
            )
            row["median_head_sample_readback_ms"] = statistics.median(
                x["ms_per_iteration"] for x in row["timing"] if x["token_out"]
            )
        report["program_cache_after"] = int(mesh.num_program_cache_entries())
        report["trace_ids_after"] = {key: str(value) for key, value in gen.traces.items()}
        assert report["program_cache_after"] == programs
        assert report["trace_ids_after"] == trace_ids
        report["passed"] = all(x["passed"] for row in report["candidates"].values() for x in row["accuracy"])
        save()
        assert report["passed"], "Candidate numerical/greedy gate failed; retain results without promotion"
    except BaseException as error:
        report["failure"] = dict(phase=phase, type=type(error).__name__, message=str(error))
        save()
        raise
    finally:
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
