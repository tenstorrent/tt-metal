# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parent-serialized TP4 compact Q/K norm correctness and total-path timing."""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 3, 17, 32])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--weight-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (
        all(1 <= b <= 32 for b in args.batches)
        and args.steps >= 2
        and args.samples > 0
        and args.replays > 0
        and args.epsilon > 0
    )
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert not args.output.exists()
    watcher = os.getenv("TT_METAL_WATCHER", "0") != "0"
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    assert not (watcher and profiler)
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt.compact_qk_norm_candidate import compact_qk_norm
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import _local_qkv_split
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
    from types import SimpleNamespace

    torch.set_num_threads(4)
    report = {
        "status": "running",
        "mesh": [1, 4],
        "watcher": watcher,
        "profiler": profiler,
        "cases": [],
        "arguments": vars(args) | {"output": str(args.output)},
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "tt/compact_qk_norm_candidate.py", ROOT / "tt/multichip_decoder.py")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            packer_l1_acc=True,
        )
        wdtype = getattr(ttnn, args.weight_dtype)
        torch_wdtype = getattr(torch, args.weight_dtype)

        def upload(values, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        for batch in args.batches:
            entry = {"batch": batch, "checks": [], "timings_ms": {}}
            report["cases"].append(entry)
            save()
            generators = [torch.Generator().manual_seed(8123 + batch * 19 + rank) for rank in range(4)]
            weights_host = [
                [(torch.randn(1, 1, 256, generator=g) * 0.15 + 1).to(torch_wdtype) for g in generators]
                for _ in range(2)
            ]
            weights = [upload(w, wdtype) for w in weights_host]
            owner = SimpleNamespace()
            cores = OptimizedDecoder._compact_cores(owner, batch)
            memory = OptimizedDecoder._compact_height_mem(owner, cores, 32, 256)
            bank = []
            expected = []
            for step in range(args.steps):
                host = [
                    (torch.randn(1, 1, batch, 2048, generator=g) * (0.125 if step else 0.5) + rank * 0.0625).bfloat16()
                    for rank, g in enumerate(generators)
                ]
                # A valid zero head tests epsilon without contaminating other heads.
                if step == 1:
                    for value in host:
                        value[0, 0, 0, :256] = 0
                split = _local_qkv_split(upload(host), memory_config=memory)
                bank.append(split[:2])
                refs = []
                for branch, (start, stop, heads) in enumerate(((0, 1536, 6), (1536, 1792, 1))):
                    values = []
                    for rank, t in enumerate(host):
                        x = t[:, :, :, start:stop].reshape(1, batch, heads, 256).float()
                        values.append(
                            (
                                x
                                * torch.rsqrt((x * x).mean(-1, keepdim=True) + args.epsilon)
                                * weights_host[branch][rank].float()
                            ).bfloat16()
                        )
                    refs.append(values)
                expected.append(refs)
            q, k = (ttnn.empty_like(t) for t in bank[0])

            def update(step):
                ttnn.copy(bank[step][0], q)
                ttnn.copy(bank[step][1], k)

            def invoke(index):
                if index:
                    return compact_qk_norm(q, k, *weights, epsilon=args.epsilon)
                return tuple(
                    ttnn.rms_norm(
                        ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG),
                        weight=w,
                        epsilon=args.epsilon,
                        compute_kernel_config=ckc,
                    )
                    for t, w in zip((q, k), weights)
                )

            eager = []
            for step in range(args.steps):
                update(step)
                outputs = [invoke(i) for i in range(2)]
                values = [[read(t) for t in branch] for branch in outputs]
                eager.append(values)
                scores = []
                oracle = []
                finite = True
                for branch in range(2):
                    for rank in range(4):
                        a, b = values[0][branch][rank], values[1][branch][rank]
                        scores.append(pcc(a, b))
                        oracle.append(pcc(b, expected[step][branch][rank]))
                        finite = finite and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                entry["checks"].append({"step": step, "native_pcc": scores, "torch_pcc": oracle, "finite": finite})
                save()
                assert finite and min(scores) >= 0.995 and min(oracle) >= 0.995
                # Exact input ownership / no in-place mutation.
                assert all(
                    torch.equal(a, b) for t, source in zip((q, k), bank[step]) for a, b in zip(read(t), read(source))
                )
            for index in range(2):
                update(0)
                invoke(index)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = invoke(index)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                try:
                    for step in range(args.steps):
                        update(step)
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        errors = []
                        for branch, t in enumerate(output):
                            for rank, actual in enumerate(read(t)):
                                errors.append(
                                    int(
                                        (
                                            actual.view(torch.int16)
                                            != eager[step][index][branch][rank].view(torch.int16)
                                        ).sum()
                                    )
                                )
                        entry.setdefault("trace_vs_eager_bits", []).append(
                            {"path": index, "step": step, "errors": errors}
                        )
                        save()
                        assert max(errors) == 0
                    if not watcher:
                        samples = []
                        for _ in range(args.samples):
                            ttnn.synchronize_device(mesh)
                            before = time.perf_counter()
                            for _ in range(args.replays):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples.append((time.perf_counter() - before) * 1000 / args.replays)
                        entry["timings_ms"]["candidate" if index else "native"] = {
                            "samples": samples,
                            "median": statistics.median(samples),
                        }
                finally:
                    ttnn.release_trace(mesh, trace)
                    del output
            if profiler:
                ttnn.ReadDeviceProfiler(mesh)
            save()
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
