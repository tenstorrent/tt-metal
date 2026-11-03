# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-instance paged/continuation/changed-input trace and cache-ownership tests.

Run --baseline first with matching arguments to record optimized TTNN outputs.
All prefill variants and persistent inputs are prepared before trace capture.
"""

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder
from .fused_reference import ROOT, pcc, reference
from .multichip_activations import recorded_inputs
from .multichip_test_utils import device_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 31, 32, 33, 257, 2047, 2048, 2049])
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--cache-dtype", default="bfloat8_b")
    ap.add_argument("--options", type=json.loads, default={})
    ap.add_argument("--tag", default="validation")
    ap.add_argument("--stress-seconds", type=float, default=0)
    ap.add_argument("--artifact-dir", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(112)
    c, hf, rope, sd = reference(args.layer)
    b, p = args.batch, args.page_size
    spec = [(s, 0, False) for s in args.lengths] + [(p + 3, 31, False), (67, 0, True)]
    blocks = (max(s + prefix for s, prefix, _ in spec) + 64 + p - 1) // p
    baseline_root = ROOT / "doc/multichip_decoder"
    root = args.artifact_dir or baseline_root
    root.mkdir(parents=True, exist_ok=True)
    local = baseline_root / "local"
    local.mkdir(exist_ok=True)
    key = f"l{args.layer}_b{b}_p{p}_{args.cache_dtype}_" + hashlib.sha256(str(spec).encode()).hexdigest()[:8]
    path = local / f"validation_baseline_{key}.pt"
    expected = {} if args.baseline else torch.load(path, weights_only=True)
    cases = []
    for i, (s, prefix, random_pos) in enumerate(spec):
        total = s + prefix
        x = recorded_inputs(args.layer, b, total + 1, i % 4)
        cs = rope(x, torch.arange(total).unsqueeze(0).expand(b, -1))
        positions = torch.full((b,), total, dtype=torch.int32)
        if random_pos:
            positions = total - 1 - (torch.arange(b, dtype=torch.int32) * 17 + 5) % min(total, 64)
        dc = rope(x, positions[:, None])
        cases.append(
            dict(
                length=s,
                prefix=prefix,
                total=total,
                pages=torch.randperm(b * blocks).reshape(b, blocks).int(),
                x=x,
                cs=cs,
                positions=positions,
                dc=dc,
            )
        )
    if not args.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if args.baseline else 4),
        physical_device_ids=[0] if args.baseline else [],
        trace_region_size=32 * 1024 * 1024,
    )
    tid = None
    try:
        ttnn.CONFIG.throw_exception_on_fallback = True
        sharded = not args.baseline and args.options.get("residual_layout") == "sharded"

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None, device=mesh):
            mapper = ttnn.ReplicateTensorToMesh(mesh) if shard is None else ttnn.ShardTensorToMesh(mesh, dim=shard)
            return ttnn.from_torch(v.contiguous(), device=device, mesh_mapper=mapper, dtype=dtype, layout=layout)

        def cpu(v, shard=None):
            result = ttnn.to_torch(
                v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=shard if shard is not None else 0)
            )
            return result if args.baseline or shard is not None else result.chunk(4, dim=0)[0]

        def copy(v, to, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None):
            ttnn.copy_host_to_device_tensor(tt(v, dtype, layout, shard, device=None), to)

        klass = OptimizedDecoder if args.baseline else MultichipDecoder
        model = klass.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh, options=args.options)
        kv = [
            tt(torch.zeros(b * blocks, 8, p, 128), getattr(ttnn, args.cache_dtype), shard=None if args.baseline else 1)
            for _ in range(2)
        ]
        table = tt(torch.zeros(b, blocks, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        dx = tt(torch.zeros(1, 1, b, 2560), shard=3 if sharded else None)
        dcos, dsin = [tt(torch.zeros(1, b, 1, 128)) for _ in range(2)]
        pos = tt(torch.zeros(b, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        dk = dict(cos=dcos, sin=dsin, page_table=table, current_pos=pos, kv_cache=kv)
        for case in cases:
            ends = [(0, case["prefix"]), (case["prefix"], case["total"])] if case["prefix"] else [(0, case["total"])]
            case["parts"] = [
                (
                    start,
                    tt(case["x"][:, start:end].unsqueeze(0), shard=3 if sharded else None),
                    *[tt(v[:, start:end].unsqueeze(1)) for v in case["cs"]],
                )
                for start, end in ends
            ]

        def prefill(case):
            copy(case["pages"], table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for start, x, co, si in case["parts"]:
                with device_only():
                    out = model.prefill_forward(x, cos=co, sin=si, page_table=table, kv_cache=kv, start_pos=start)
                if start + out.shape[2] < case["total"]:
                    out.deallocate(True)
            return out

        def refresh(case):
            copy(case["x"][:, case["total"] :].reshape(1, 1, b, 2560), dx, shard=3 if sharded else None)
            for v, to in zip(case["dc"], (dcos, dsin)):
                copy(v.reshape(1, b, 1, 128), to)
            copy(case["positions"], pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def caches():
            return [cpu(v, None if args.baseline else 1) for v in kv]

        def check_untouched(before, case, decode=False):
            allowed = torch.zeros(b * blocks, p, dtype=torch.bool)
            for u in range(b):
                tokens = case["positions"][u : u + 1] if decode else torch.arange(case["total"])
                allowed[case["pages"][u, tokens // p].long(), tokens % p] = True
            after = caches()
            for old, new in zip(before, after):
                changed = (old != new).any(dim=3).any(dim=1)
                assert not torch.any(changed & ~allowed), "Unrelated physical KV row modified"

        scores = []
        for i, case in enumerate(cases):
            before = caches()
            out = prefill(case)
            check_untouched(before, case)
            pre = cpu(out, 3 if sharded else None).clone()
            out.deallocate(True)
            refresh(case)
            before = caches()
            with device_only():
                out = model.decode_forward(dx, **dk)
            dec = cpu(out, 3 if sharded else None).clone()
            out.deallocate(True)
            check_untouched(before, case, True)
            if args.baseline:
                expected[str(i)] = dict(prefill=pre, decode=dec)
            row = dict(
                length=case["length"],
                prefix=case["prefix"],
                positions=case["positions"].tolist(),
                prefill_pcc=pcc(pre, expected[str(i)]["prefill"]),
                decode_pcc=pcc(dec, expected[str(i)]["decode"]),
            )
            assert min(row["prefill_pcc"], row["decode_pcc"]) >= 0.995, row
            scores.append(row)
            print("WARM", row, flush=True)
        if args.baseline:
            torch.save(expected, path)
        # Ping-pong semaphore variants must both be warmed.
        for _ in range(2):
            out = model.decode_forward(dx, **dk)
            out.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            with device_only():
                traced = model.decode_forward(dx, **dk)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        replay = []
        for i in list(range(len(cases))) + [0]:
            case = cases[i]
            out = prefill(case)
            out.deallocate(True)
            refresh(case)
            previous = None
            for rep in range(3):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                actual = cpu(traced, 3 if sharded else None)
                score = pcc(actual, expected[str(i)]["decode"])
                assert score >= 0.995
                if previous is not None:
                    assert torch.equal(previous, actual), "Nondeterministic trace"
                previous = actual.clone()
            replay.append(dict(case=i, trace_id=str(tid), pcc=score))
            print("REPLAY", i, score, flush=True)
        count = 0
        start = time.monotonic()
        while time.monotonic() - start < args.stress_seconds:
            for _ in range(100):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            count += 100
        if count:
            assert torch.equal(previous, cpu(traced, 3 if sharded else None))
        result = dict(
            key=key,
            chunk_size=model.chunk_size,
            baseline=args.baseline,
            options=args.options,
            cases=scores,
            replays=replay,
            cache_ownership="pass",
            determinism="bitwise",
            stress_replays=count,
            capture_count=1,
            tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
            skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
            watcher=os.environ.get("TT_METAL_WATCHER"),
            source_sha256=hashlib.sha256(Path(inspect.getfile(klass)).read_bytes()).hexdigest(),
        )
        (root / f"{args.tag}_{key}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("PASS", key, flush=True)
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
