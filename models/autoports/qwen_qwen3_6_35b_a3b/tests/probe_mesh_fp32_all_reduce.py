# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Cancellation-sensitive direct TP4 all-reduce component; parent runs hardware.

Uses production 32-core [32,64] input shards, [32,256] scratch shards,
FP32 scratch/input, BF16 output, cluster axis 1 and two fabric links. Includes
an unchanged BF16-input control and changing-input trace replay. Failures are
written to the requested JSON before a nonzero exit; pre-repair failure is
expected evidence, never recategorized as passing.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import ttnn as t
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MeshCCL


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[4]
    kernel_root = root / "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device"
    sources = [kernel_root / "all_reduce_async_program_factory.cpp", kernel_root / "kernels/compute/reduction.cpp"]
    native = sorted(
        {
            Path(line.split()[-1]).resolve()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.split()[-1].endswith(("/_ttnncpp.so", "/_ttnn.so", "/libtt_metal.so"))
        }
    )
    return {
        "sources": {str(path): digest(path) for path in sources},
        "loaded_native_binaries": {str(path): {"sha256": digest(path), "inode": path.stat().st_ino} for path in native},
        "ttnn_version": getattr(t, "__version__", "unknown"),
    }


def values(shift=0):
    # Columns hold independent rank tuples; residuals remain representable in
    # BF16 after FP32 cancellation but are lost by TF32 input narrowing.
    cases = torch.tensor(
        [
            [1 + 2**-12, -1, 0, 0],
            [-1 - 2**-12, 1, 0, 0],
            [1 + 2**-12, -1, 1 + 2**-13, -1],
            [2**16 + 2**-6, -(2**16), 2**-10, 0],
            [3, -2, 4, -1],
            [0, 0, 0, 0],
            [2**24, 1, -(2**24), 2**-8],
            [2**24, -(2**24), 1, 2**-8],
        ],
        dtype=torch.float32,
    ).T
    return (
        torch.roll(cases.repeat(1, 256), shifts=shift, dims=-1)
        .reshape(1, 4, 1, 2048)
        .expand(1, 4, 32, 2048)
        .contiguous()
    )


def pack_bf16(values):
    """Finite FP32 -> BF16 packer rounding: nearest, ties away from zero.

    PyTorch uses ties-to-even and is not the exact pack oracle. This follows
    the Blackhole routed-combine test's bfloat16_ties_away helper:
    tests/ttnn/nightly/unit_tests/operations/experimental/
    nomic_embed_text_v2_moe/test_routed_combine.py:29.
    Keep the final cancellation tuple: its FP32 sum 1.00390625 must pack to
    1.0078125, not PyTorch's 1.0. All other existing tuples are unchanged.
    """
    if not torch.isfinite(values).all():
        raise ValueError("this component oracle is defined for finite inputs")
    bits = values.float().view(torch.int32)
    return ((bits + 0x8000) & ~0xFFFF).view(torch.float32).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=4)
    args = parser.parse_args()
    if args.replays < 2:
        parser.error("--replays must include changed input")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("run with TT_METAL_TRACE_ALLOC_TRACKING=1")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") not in ("", "0"):
        parser.error("program-cache allocations must remain included in trace allocation tracking")
    identity = provenance()
    records = []
    error = None
    trace_ids = []
    copy_warmup = {}
    mesh = None
    try:
        t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=1 << 20)
        ccl = MeshCCL(mesh, 2, t.Topology.Linear)
        semaphore = t.create_global_semaphore(mesh, ccl.ccl_cores, 0)

        def memory(width):
            return t.create_sharded_memory_config(
                (32, width),
                t.CoreGrid(x=8, y=4),
                t.ShardStrategy.WIDTH,
                t.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        input_memory, scratch_memory = memory(64), memory(256)

        def upload(host, dtype):
            return t.from_torch(
                host,
                dtype=dtype,
                layout=t.TILE_LAYOUT,
                device=mesh,
                memory_config=input_memory,
                mesh_mapper=t.ShardTensorToMesh(mesh, dim=1),
            )

        # Keep all source tensors alive until every trace has been released.
        # Upload and warm both dtype families before the first capture, so
        # replay refresh cannot introduce device or copy-program allocations.
        prepared = []
        for dtype_name, dtype in (("fp32", t.float32), ("bf16_control", t.bfloat16)):
            host = values()
            if dtype == t.bfloat16:
                # Exact integer inputs isolate the unchanged BF16 branch rather
                # than asserting a new association contract for arbitrary sums.
                host = host.round().clamp(-16, 16).bfloat16().float()
            inp = upload(host, dtype)
            scratch = t.empty(
                (1, 1, 32, 8192), device=mesh, dtype=dtype, layout=t.TILE_LAYOUT, memory_config=scratch_memory
            )
            hosts = [host] + [torch.roll(host, shifts=step + 1, dims=-1) for step in range(args.replays)]
            sources = [upload(current, dtype) for current in hosts]
            for source in sources:
                t.copy(source, inp)
            t.copy(sources[0], inp)
            prepared.append((dtype_name, inp, scratch, hosts, sources))
        t.synchronize_device(mesh)
        copy_warmup = {
            "before_any_capture": True,
            "source_destination_pairs": 2 * (args.replays + 1),
            "dtypes": ["float32", "bfloat16"],
            "initial_inputs_restored": True,
            "sources_retained_until_all_traces_released": True,
        }
        from ttnn.tools import trace_allocation_tracker

        for dtype_name, inp, scratch, hosts, sources in prepared:

            def run():
                return t.experimental.all_reduce_async(
                    inp,
                    scratch,
                    cluster_axis=1,
                    mesh_device=mesh,
                    multi_device_global_semaphore=semaphore,
                    memory_config=input_memory,
                    dtype=t.bfloat16,
                    topology=t.Topology.Linear,
                    num_links=2,
                    fp32_dest_acc=True,
                )

            def check(output, current, mode, step):
                expected = current[:, :1].clone()
                for rank in range(1, 4):
                    expected = expected + current[:, rank : rank + 1]
                torch_rne = expected.bfloat16()
                expected = pack_bf16(expected)
                t.synchronize_device(mesh)
                for rank, local in enumerate(t.get_device_tensors(output)):
                    actual = t.to_torch(local)
                    equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                    records.append(
                        {
                            "dtype": dtype_name,
                            "mode": mode,
                            "step": step,
                            "rank": rank,
                            "bit_equal": equal,
                            "mismatch_count": int((actual != expected).sum()),
                            "max_abs": float((actual.float() - expected.float()).abs().max()),
                            "actual_first_8": actual.reshape(-1)[:8].float().tolist(),
                            "expected_first_8": expected.reshape(-1)[:8].float().tolist(),
                            "torch_rne_first_8": torch_rne.reshape(-1)[:8].float().tolist(),
                            "packer_tie_difference_count": int((expected != torch_rne).sum()),
                        }
                    )

            check(run(), hosts[0], "eager", 0)
            run()
            t.synchronize_device(mesh)
            trace = t.begin_trace_capture(mesh, cq_id=0)
            captured = run()
            t.end_trace_capture(mesh, trace, cq_id=0)
            trace_ids.append(trace)
            # Only this exact captured output is acknowledged. Its reference
            # remains live until release; no program-cache storage is excluded.
            trace_allocation_tracker.acknowledge_corruptible(captured)
            for step, (current, source) in enumerate(zip(hosts[1:], sources[1:])):
                t.copy(source, inp)
                t.execute_trace(mesh, trace, cq_id=0, blocking=True)
                check(captured, current, "trace", step)
            t.release_trace(mesh, trace)
            trace_ids.remove(trace)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result = {
            "passed": error is None
            and len(records) == 8 * (args.replays + 1)
            and all(row["bit_equal"] for row in records),
            "geometry": {
                "mesh": [1, 4],
                "axis": 1,
                "input_shard": [32, 64],
                "scratch_shard": [32, 256],
                "cores": 32,
                "output_dtype": "BF16",
                "fp32_dest_acc": True,
                "links": 2,
            },
            "reference": "serial full-FP32 rank order 0,1,2,3 followed by one hardware BF16 pack (nearest, ties away from zero)",
            "reference_rounding_source": "tests/ttnn/nightly/unit_tests/operations/experimental/nomic_embed_text_v2_moe/test_routed_combine.py:29",
            "historical_oracle_artifact": "doc/optimized_multichip_decoder/fp32_ar_after.json retains the prior PyTorch ties-to-even failure; not rewritten",
            "provenance": identity,
            "trace_allocation_tracking": True,
            "program_cache_excluded": False,
            "copy_warmup": copy_warmup,
            "error": error,
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Preserve numerical/compile failure evidence before device cleanup.
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if mesh is not None:
            for trace in trace_ids:
                t.release_trace(mesh, trace)
            t.close_mesh_device(mesh)
    print(json.dumps({"passed": result["passed"], "comparisons": len(records), "error": error}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
