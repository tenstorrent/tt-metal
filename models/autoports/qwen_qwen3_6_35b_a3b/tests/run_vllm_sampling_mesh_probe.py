# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Weight-free TP4 canonical Qwen sampling probe; exclusive device ownership required.

Requires common sampler wiring from canonical_sampling64_wiring.patch, including
for width32 so an ignored opt-in cannot produce misleading results. This calls
QwenGenerator._sampling_forward unchanged. Host token reads are diagnostic only;
no sampled token is fed back from host. Timing excludes reads and parameter writes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VOCAB = 248320
ROWS = 32


def fixture(torch):
    values = torch.full((ROWS, VOCAB), -80, dtype=torch.bfloat16)
    ids = torch.empty((ROWS, 64), dtype=torch.int64)
    for row in range(ROWS):
        for rank in range(64):
            token = (rank % 4) * (VOCAB // 4) + (row * 137 + rank * 997 + 17) % (VOCAB // 4)
            ids[row, rank] = token
            values[row, token] = -rank / 64
    assert all(len(set(row)) == 64 for row in ids.tolist())
    assert torch.equal(values.argmax(-1), ids[:, 0])
    return values.reshape(1, 1, ROWS, VOCAB), ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-width", type=int, choices=(32, 64), default=64)
    parser.add_argument("--draw-widths", default="32,64")
    parser.add_argument("--mode", choices=("acceptance", "timing"), default="acceptance")
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    widths = tuple(int(x) for x in args.draw_widths.split(","))
    if not widths or len(set(widths)) != len(widths) or any(w not in (32, 64) for w in widths):
        parser.error("--draw-widths must be 32, 64, or 32,64")
    if max(widths) > args.candidate_width or args.draws < 32 or args.iterations < 1:
        parser.error("Draw width cannot exceed candidate width; draws >=32 and iterations >=1")
    import torch

    torch.set_num_threads(8)
    logits, ids = fixture(torch)
    # Logical requests retain logits, seeds, params and token state under permutation.
    permutations = {"identity": torch.arange(ROWS), "reordered": (torch.arange(ROWS) * 13 + 7) % ROWS}
    profiles = {"greedy": (1, 0.0, 1.0), "top20": (20, 0.95, 1.0), "top40": (40, 0.95, 1.0 / 0.7)}
    for perm in permutations.values():
        assert torch.equal(perm.sort().values, torch.arange(ROWS))
        assert torch.equal(logits[:, :, perm].reshape(ROWS, VOCAB).argmax(-1), ids[perm, 0])
    result = dict(
        status="RUNNING",
        command=sys.argv,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        candidate_width=args.candidate_width,
        draw_widths=widths,
        mesh=[1, 4],
        logits_shape=list(logits.shape),
        mode=args.mode,
        draws=args.draws,
        iterations=args.iterations,
        captures=[],
        releases=[],
        cases=[],
        timing_eligible=args.mode == "timing" and not args.cpu_only,
        limitations=[
            "Sampler-only diagnostic, not model accuracy, adapter remap acceptance or serving throughput.",
            "Top20 nucleus support checked conservatively against top20; not a distribution proof.",
            "No host token feedback occurs; reads/reset writes are isolated correctness diagnostics.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    if args.cpu_only:
        result["status"] = "PASS_CPU_FIXTURES_ONLY"
        save()
        return
    expected_tracking = "1" if args.mode == "acceptance" else "0"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == expected_tracking
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
    if args.mode == "timing":
        for key in (
            "TT_METAL_WATCHER",
            "TT_METAL_DEVICE_PROFILER",
            "TTNN_OP_PROFILER",
            "TT_METAL_PROFILER_TRACE_TRACKING",
        ):
            assert os.environ.get(key, "0") in ("0", ""), key
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator, SamplerCCL
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import CollectiveWorkspace
    from models.common.sampling.generator import SamplingGenerator, SamplingParams

    mesh = None
    traces = {}
    raw_path = args.output.with_suffix(".tokens.raw.jsonl")
    result["raw_tokens"] = str(raw_path)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        shard_mapper = ttnn.ShardTensorToMesh(mesh, dim=3)

        def upload(value, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, distribution=mapper):
            return ttnn.from_torch(
                value.contiguous(), dtype=dtype, layout=layout, device=mesh, mesh_mapper=distribution
            )

        def refresh(value, dest, distribution=mapper):
            host = ttnn.from_torch(value.contiguous(), dtype=dest.dtype, layout=dest.layout, mesh_mapper=distribution)
            ttnn.copy_host_to_device_tensor(host, dest)

        def read(value):
            replicas = [ttnn.to_torch(chip).reshape(-1).to(torch.int64) for chip in ttnn.get_device_tensors(value)]
            assert len(replicas) == 4
            assert all(torch.equal(replicas[0], item) for item in replicas[1:]), "TP4 replica mismatch"
            return replicas[0]

        def observe(value, key, label):
            assert value.dtype == ttnn.bfloat16, (key, label, value.dtype)

        model = SimpleNamespace(mesh=mesh, workspace=CollectiveWorkspace(mesh), observe_boundary=observe)
        gen = QwenGenerator.__new__(QwenGenerator)
        gen.mesh, gen.model = mesh, model
        gen.sampler = SamplingGenerator(
            args=SimpleNamespace(
                vocab_size=VOCAB,
                padded_vocab_size=VOCAB,
                max_batch_size=ROWS,
                max_top_k=args.candidate_width,
                device_sampling_draw_width=widths[0],
                cluster_shape=(1, 4),
                sampling_all_gather_axis=1,
            ),
            mesh_device=mesh,
            tt_ccl=SamplerCCL(model),
        )
        sampler = gen.sampler.tt_sampling
        assert getattr(sampler, "_sampling_draw_width", None) == widths[0], "Common wiring patch is required"
        gen.sampler_logits = upload(logits, ttnn.bfloat16, ttnn.TILE_LAYOUT, shard_mapper)
        sentinel = (torch.arange(ROWS, dtype=torch.int32) + 1000000).reshape(1, 1, 1, ROWS)
        gen.tokens = upload(sentinel, ttnn.uint32)
        gen.previous_tokens = upload(sentinel, ttnn.uint32)
        gen.active_mask = upload(torch.ones(ROWS, dtype=torch.int32), ttnn.uint32)

        def set_params(profile):
            k, p, temp = profiles[profile]
            gen.sampler.reset_sampling_params(
                SamplingParams(
                    temperature=torch.full((ROWS,), temp),
                    top_k=torch.full((ROWS,), k),
                    top_p=[p] * ROWS,
                    presence_penalty=[0.0] * ROWS,
                    frequency_penalty=[0.0] * ROWS,
                    repetition_penalty=[1.0] * ROWS,
                )
            )
            assert not sampler.force_argmax_sampling, "Must use canonical split sampling"

        set_params("top20")
        # Compile both signatures before either capture; persistent params and seeds are shared.
        for width in widths:
            sampler._sampling_draw_width = width
            gen.sampler.precompile(gen.sampler_logits, tt_out_tok=gen.tokens)
            gen._sampling_forward()
        ttnn.synchronize_device(mesh)
        warm_entries = int(mesh.num_program_cache_entries())
        result["program_cache_entries_after_warmup"] = warm_entries
        for width in widths:
            sampler._sampling_draw_width = width
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                gen._sampling_forward()
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces[width] = trace
            capture_entries = int(mesh.num_program_cache_entries())
            assert capture_entries == warm_entries, "Cold program compilation during capture"
            result["captures"].append(
                {
                    "candidate_width": args.candidate_width,
                    "draw_width": width,
                    "trace_id": str(trace),
                    "program_cache_entries": capture_entries,
                }
            )
        entries = int(mesh.num_program_cache_entries())
        signatures = {}
        with raw_path.open("w") as raw:
            for profile, (k, _, _) in profiles.items():
                for width in widths:
                    if k > width:
                        continue
                    sampler._sampling_draw_width = width
                    set_params(profile)
                    for active_count in (1, 32):
                        for permutation, perm in permutations.items():
                            refresh(logits[:, :, perm], gen.sampler_logits, shard_mapper)
                            active = (torch.arange(ROWS) < active_count).to(torch.int32)[perm]
                            refresh(active, gen.active_mask)
                            before = sentinel.reshape(-1)[perm].reshape(sentinel.shape)
                            ranks = [{int(t): r for r, t in enumerate(row)} for row in ids[perm].tolist()]
                            selected_rows = []
                            tail = 0
                            for draw in range(args.draws if args.mode == "acceptance" else 1):
                                seed = ((torch.arange(ROWS) * 104729 + (draw + 1) * 7919) % 999983 + 1).to(torch.int32)[
                                    perm
                                ]
                                refresh(seed, sampler.seeds_tt_tensor)
                                refresh(before, gen.tokens)
                                gen._sampling_forward()
                                eager = read(gen.tokens)
                                eager_seeds = read(sampler.seeds_tt_tensor)
                                refresh(seed, sampler.seeds_tt_tensor)
                                refresh(before, gen.tokens)
                                ttnn.execute_trace(mesh, traces[width], cq_id=0, blocking=False)
                                actual = read(gen.tokens)
                                assert torch.equal(eager, actual), (profile, width, permutation, draw)
                                after_seed = read(sampler.seeds_tt_tensor)
                                assert torch.equal(eager_seeds, after_seed)
                                assert torch.equal(after_seed, seed.to(torch.int64) + active)
                                for row, token in enumerate(actual.tolist()):
                                    if active[row]:
                                        rank = ranks[row].get(token)
                                        assert rank is not None and rank < k, (profile, row, token, rank)
                                        tail += int(rank >= 32)
                                    else:
                                        assert token == int(before.reshape(-1)[row]), "Inactive token changed"
                                canonical = actual[torch.argsort(perm)].tolist()
                                selected_rows.append(canonical)
                                raw.write(
                                    json.dumps(
                                        dict(
                                            profile=profile,
                                            width=width,
                                            active=active_count,
                                            permutation=permutation,
                                            draw=draw,
                                            tokens=canonical,
                                        )
                                    )
                                    + "\n"
                                )
                            key = (profile, width, active_count)
                            if permutation == "identity":
                                signatures[key] = selected_rows
                            else:
                                assert signatures[key] == selected_rows, ("row remap changed request draw", key)
                            if profile == "top40" and active_count == 32 and args.mode == "acceptance":
                                assert tail > 0, "No rank33-40 selected; wider support unproved"
                            record = dict(
                                profile=profile,
                                top_k=profiles[profile][0],
                                top_p=profiles[profile][1],
                                inverse_temperature=profiles[profile][2],
                                draw_width=width,
                                active_rows=active_count,
                                physical_rows=ROWS,
                                permutation=permutation,
                                tail33_40=tail,
                                token_digest=hashlib.sha256(json.dumps(selected_rows).encode()).hexdigest(),
                            )
                            if args.mode == "timing" and permutation == "identity":
                                for _ in range(3):
                                    ttnn.execute_trace(mesh, traces[width], cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                begin = time.perf_counter()
                                for _ in range(args.iterations):
                                    ttnn.execute_trace(mesh, traces[width], cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                record["sampler_ms"] = (time.perf_counter() - begin) * 1000 / args.iterations
                            result["cases"].append(record)
                            save()
            if set(widths) == {32, 64}:
                result["width_token_identity"] = {
                    f"{profile}_{count}u": signatures[(profile, 32, count)] == signatures[(profile, 64, count)]
                    for profile in ("greedy", "top20")
                    for count in (1, 32)
                }
                # Greedy must be identical. Stochastic identity is measured, not a mathematical requirement.
                assert all(result["width_token_identity"][f"greedy_{n}u"] for n in (1, 32))
        assert int(mesh.num_program_cache_entries()) == entries, "New program compilation during replay checks"
        result["program_cache_entries"] = entries
        assert len(result["captures"]) == len(widths), "Unexpected trace capture count"
        result["status"] = "PASS"
    except Exception:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        raise
    finally:
        if mesh is not None:
            for width, trace in traces.items():
                ttnn.release_trace(mesh, trace)
                result["releases"].append({"draw_width": width, "trace_id": str(trace)})
            ttnn.close_mesh_device(mesh)
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()


if __name__ == "__main__":
    main()
