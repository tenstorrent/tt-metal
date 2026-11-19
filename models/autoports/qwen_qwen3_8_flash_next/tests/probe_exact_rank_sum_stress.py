# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced correctness reproducer for watcher-sensitive replicated MLP output."""

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import Collectives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--preceding-matmul", action="store_true")
    parser.add_argument("--alternate-inputs", action="store_true")
    parser.add_argument("--gather-tap", action="store_true")
    parser.add_argument("--persistent-gather", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    report = dict(
        passed=False,
        iterations=args.iterations,
        completed=0,
        preceding_matmul=args.preceding_matmul,
        alternate_inputs=args.alternate_inputs,
        gather_tap=args.gather_tap,
        persistent_gather=args.persistent_gather,
        scope="reduced_collective_correctness_not_serving",
    )
    mesh = trace = None
    started = time.monotonic()
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        source = (torch.arange(4 * 2560).reshape(1, 4, 1, 2560) % 31 - 15).to(torch.bfloat16) / 32
        inp = ttnn.from_torch(
            source,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        out = ttnn.zeros(
            (1, 1, 1, 2560),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ccl = Collectives(mesh)
        if args.persistent_gather:
            gathered = ttnn.zeros(
                (1, 4, 1, 2560),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ccl.gather = lambda value, dim=-1: ttnn.all_gather(
                value, dim=dim, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=gathered
            )
        gather_out = None
        if args.gather_tap:
            gather_out = ttnn.zeros(
                (1, 4, 1, 2560),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            original_gather = ccl.gather

            def tapped_gather(value, dim=-1):
                result = original_gather(value, dim)
                ttnn.copy(result, gather_out)
                return result

            ccl.gather = tapped_gather
        weight = compute = None
        if args.preceding_matmul:
            torch.manual_seed(42)
            weight = ttnn.from_torch(
                torch.randn(1, 1, 2560, 2560).bfloat16() / 64,
                device=mesh,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
                math_approx_mode=False,
            )

        def body():
            value = (
                inp
                if weight is None
                else ttnn.matmul(
                    inp,
                    weight,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=compute,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            value = ttnn.typecast(ccl.sum(value), ttnn.bfloat16)
            ttnn.copy(value, out)

        body()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        body()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        expected = None if args.preceding_matmul else source.float().sum(dim=1, keepdim=True).bfloat16()
        for step in range(args.iterations):
            if args.alternate_inputs:
                current = source + (step % 7) / 8
                host = ttnn.from_torch(
                    current,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                )
                ttnn.copy_host_to_device_tensor(host, inp)
                expected = None if args.preceding_matmul else current.float().sum(dim=1, keepdim=True).bfloat16()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            if gather_out is not None and not args.preceding_matmul:
                for rank, tensor in enumerate(ttnn.get_device_tensors(gather_out)):
                    actual = ttnn.to_torch(tensor)
                    reference = current if args.alternate_inputs else source
                    if not torch.equal(actual, reference):
                        report["first_failure"] = dict(step=step, rank=rank, stage="gather")
                        torch.save(dict(actual=actual, expected=reference), args.output.with_suffix(".pt"))
                        raise AssertionError(f"Gather differs at step{step} rank{rank}")
            for rank, tensor in enumerate(ttnn.get_device_tensors(out)):
                actual = ttnn.to_torch(tensor)
                if expected is None:
                    expected = actual.clone()
                if not torch.equal(actual, expected):
                    ids = (actual != expected).nonzero()
                    report["first_failure"] = dict(
                        step=step,
                        rank=rank,
                        differing=len(ids),
                        indices=ids[:100].tolist(),
                        max_abs=float((actual.float() - expected.float()).abs().max()),
                    )
                    torch.save(dict(actual=actual, expected=expected), args.output.with_suffix(".pt"))
                    raise AssertionError(f"Replicated reduction differs at step{step} rank{rank}")
            report["completed"] = step + 1
            if step % 100 == 0:
                print(f"REDUCTION_STRESS {step}", flush=True)
        report["passed"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
