# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded native sampling-width probe; run only with exclusive device ownership.

The 64 route requires the unapplied native_sampling64_candidate.patch. This is
an isolated native kernel test, not model quality or serving performance proof.
Use --cpu-only to validate fixtures without importing TTNN or opening a device.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fixture(torch, width, users, reverse=False):
    ranks = torch.arange(width, dtype=torch.float32)
    values = torch.empty((users, width), dtype=torch.bfloat16)
    for row in range(users):
        order = torch.randperm(width, generator=torch.Generator().manual_seed(917 + row))
        values[row, order] = (-ranks / 64).to(torch.bfloat16)
    if reverse:
        values = values.flip(-1).contiguous()
    ids = torch.arange(width, dtype=torch.int32)[None, :] * 7 + torch.arange(users)[:, None] * 4096 + 3
    order = torch.argsort(values.float(), dim=-1, descending=True, stable=True)
    sorted_ids = torch.gather(ids, 1, order)
    maps = [{int(token): rank for rank, token in enumerate(row)} for row in sorted_ids.tolist()]
    assert all(len(row) == width for row in maps)
    assert len(set(ids.reshape(-1).tolist())) == users * width
    return values.reshape(1, 1, users, width), ids.reshape(1, 1, users, width), maps


def profiles(torch, draw_width, users):
    mixed = [min(k, draw_width) for k in (1, 16, 31, 32, 33, 40, 63, 64)]
    return [
        ("top40", [min(40, draw_width)] * users, [1.0] * users, [1.0] * users, False),
        ("full_width", [draw_width] * users, [1.0] * users, [1.0] * users, False),
        ("mixed", [mixed[i % len(mixed)] for i in range(users)], [1.0] * users, [1.0] * users, False),
        ("greedy", [draw_width] * users, [0.0] * users, [1.0] * users, False),
        ("nucleus", [draw_width] * users, None, [2.0] * users, False),
        ("reversed_logits", [min(40, draw_width)] * users, [1.0] * users, [1.0] * users, True),
    ]


def expected_support(torch, values, ks, ps, temps):
    sorted_values = values.reshape(len(ks), -1).float().sort(dim=-1, descending=True).values
    limits, thresholds, margins = [], [], []
    for row, k in enumerate(ks):
        if ps is not None:
            thresholds.append(ps[row])
            limits.append(1 if ps[row] == 0 else k)
            continue
        # Put the BF16 threshold midway between two CDF steps, avoiding an
        # exact-cutoff numerical tie while testing a real nucleus restriction.
        probs = torch.softmax(sorted_values[row, :k] * temps[row], dim=-1).bfloat16().float()
        cdf = probs.cumsum(0)
        count = max(2, k // 2)
        left, right = float(cdf[count - 2]), float(cdf[count - 1])
        threshold = float(torch.tensor((left + right) / 2).bfloat16())
        margin = min(threshold - left, right - threshold)
        assert margin > 0.003, (k, left, right, threshold, margin)
        thresholds.append(threshold)
        limits.append(count)
        margins.append(margin)
    return limits, thresholds, margins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw-width", type=int, choices=(32, 64), default=64)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/sampling64_probe/result.json")
    args = parser.parse_args()
    if not 128 <= args.draws <= 1024:
        parser.error("--draws must be in [128, 1024]")
    import torch

    result = dict(
        status="RUNNING",
        command=sys.argv,
        draw_width=args.draw_width,
        draws=args.draws,
        cpu_only=args.cpu_only,
        mesh=[1, 1],
        device_id=args.device_id,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        captures=0,
        releases=0,
        fixtures=[],
        limitations=[
            "Native sampling only: no model weights, four-chip CCL, adapter, or serving latency.",
            "Nucleus fixture uses a margin away from the BF16 cutoff; not a complete distribution test.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".tokens.raw.log")
    result["raw_tokens"] = str(raw_path)

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.output)

    save()
    cases = []
    for width in (64, 256):
        for users in (1, 32):
            for name, ks, ps, temps, reverse in profiles(torch, args.draw_width, users):
                values, ids, maps = fixture(torch, width, users, reverse)
                limits, thresholds, margins = expected_support(torch, values, ks, ps, temps)
                assert all(limit <= k <= args.draw_width for limit, k in zip(limits, ks))
                cases.append((width, users, name, ks, thresholds, temps, reverse, limits, margins))
    result["cpu_fixture_checks"] = len(cases)
    if args.cpu_only:
        result["status"] = "PASS_CPU_FIXTURES_ONLY"
        result["raw_tokens"] = None
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        print(json.dumps(result, indent=2))
        return

    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    if (
        os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] != "1"
        or os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] != "0"
    ):
        raise RuntimeError("Probe requires trace allocation tracking=1 and skip_program_cache=0")
    import ttnn

    mesh = trace = None
    capture_open = False
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[args.device_id], trace_region_size=64 * 1024 * 1024
        )
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload(value, dtype, layout):
            return ttnn.from_torch(value.contiguous(), device=mesh, dtype=dtype, layout=layout, mesh_mapper=mapper)

        def refresh(value, dest):
            host = ttnn.from_torch(value.contiguous(), dtype=dest.dtype, layout=dest.layout, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(host, dest)

        def read(output):
            chips = ttnn.get_device_tensors(output)
            assert len(chips) == 1
            return ttnn.to_torch(chips[0]).reshape(-1).to(torch.int64).tolist()

        with raw_path.open("w") as raw:
            for width in (64, 256):
                for users in (1, 32):
                    values, ids, _ = fixture(torch, width, users)
                    vt = upload(values, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    it = upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                    kt = upload(
                        torch.full((users,), min(40, args.draw_width), dtype=torch.int32),
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                    pt = upload(torch.ones(users), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                    temp = upload(torch.ones(users), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                    seeds = upload(torch.arange(1, users + 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                    user_ids = upload(torch.arange(users, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                    output = upload(
                        torch.full((1, 1, 1, users), 2147483647, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                    )
                    output_address = output.buffer_address()

                    def forward():
                        ttnn.manual_seed(seeds=seeds, user_ids=user_ids)
                        kwargs = {} if args.draw_width == 32 else {"max_top_k": 64}
                        answer = ttnn.sampling(vt, it, k=kt, p=pt, temp=temp, output_tensor=output, **kwargs)
                        assert answer.buffer_address() == output_address

                    forward()
                    read(output)
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    forward()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    result["captures"] += 1
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    read(output)
                    entries = int(mesh.num_program_cache_entries())
                    for case in [c for c in cases if c[:2] == (width, users)]:
                        _, _, name, ks, ps, temps, reverse, limits, margins = case
                        values, _, maps = fixture(torch, width, users, reverse)
                        refresh(values, vt)
                        refresh(torch.tensor(ks, dtype=torch.int32), kt)
                        refresh(torch.tensor(ps), pt)
                        refresh(torch.tensor(temps), temp)
                        tail = [0] * users
                        far_tail = [0] * users
                        digest = hashlib.sha256()
                        for draw in range(args.draws):
                            seed_values = (
                                (torch.arange(users, dtype=torch.int64) * 104729 + (draw + 1) * 7919) % 999983 + 1
                            ).to(torch.int32)
                            refresh(seed_values, seeds)
                            forward()
                            eager = read(output)
                            refresh(torch.full((1, 1, 1, users), 2147483647, dtype=torch.int32), output)
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            actual = read(output)
                            assert actual == eager, (width, users, name, draw, actual, eager)
                            for row, token in enumerate(actual):
                                rank = maps[row].get(token)
                                assert rank is not None and rank < limits[row], (
                                    width,
                                    users,
                                    name,
                                    draw,
                                    row,
                                    token,
                                    rank,
                                    limits[row],
                                )
                                tail[row] += int(rank >= 32)
                                far_tail[row] += int(rank >= 40)
                            payload = json.dumps(
                                dict(
                                    width=width,
                                    users=users,
                                    profile=name,
                                    draw=draw,
                                    seeds=seed_values.tolist(),
                                    eager=eager,
                                    trace=actual,
                                )
                            )
                            digest.update(payload.encode())
                            raw.write(payload + "\n")
                        if name in ("top40", "full_width", "mixed", "reversed_logits"):
                            for row, k in enumerate(ks):
                                if k >= 40:
                                    assert tail[row] > 0, (name, row, k, "no rank beyond 32 reached")
                                if k >= 63:
                                    assert far_tail[row] > 0, (name, row, k, "no rank beyond 40 reached")
                        assert int(mesh.num_program_cache_entries()) == entries
                        record = dict(
                            width=width,
                            users=users,
                            profile=name,
                            k=ks,
                            p=ps,
                            inverse_temperature=temps,
                            allowed_rank_counts=limits,
                            cdf_cutoff_margins=margins,
                            tail_draws_per_row=tail,
                            rank40_plus_draws_per_row=far_tail,
                            eager_trace_exact=True,
                            output_address=output_address,
                            token_digest=digest.hexdigest(),
                            program_entries=entries,
                            trace=str(trace),
                            status="PASS",
                        )
                        result["fixtures"].append(record)
                        raw.flush()
                        save()
                        print(
                            f"SAMPLING_PROBE_PASS width={width} users={users} profile={name} draws={args.draws}",
                            flush=True,
                        )
                    ttnn.release_trace(mesh, trace)
                    trace = None
                    result["releases"] += 1
                    for tensor in (vt, it, kt, pt, temp, seeds, user_ids, output):
                        ttnn.deallocate(tensor)
                    save()
        result["status"] = "PASS_NATIVE_PROBE"
    except BaseException:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if mesh is not None:
            if capture_open:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            if trace is not None:
                ttnn.release_trace(mesh, trace)
                result["releases"] += 1
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()


if __name__ == "__main__":
    main()
