"""Reproduce bulk-prefill TP4 reduce-scatter corruption without the model.

Run with a free mesh; the calling agent owns device serialization and recovery.
Every replay contains 72 collectives with no intervening host synchronization.
Blocking reads happen only after that entire replay, for diagnostic validation.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# This is a single trace family with a persistent input. Do not suppress the
# production allocation guard or acknowledge newly allocated tensors as safe.
os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
if os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] != "1":
    raise RuntimeError("This probe requires TT_METAL_TRACE_ALLOC_TRACKING=1")

import torch

import ttnn
from models.common.modules.tt_ccl import TT_CCL

LAYERS = 36
RANKS = 4
ROWS = 4096
LOCAL_INPUT_WIDTH = 4096
LOCAL_OUTPUT_WIDTH = LOCAL_INPUT_WIDTH // RANKS


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def tensor_range(tensor):
    return {
        "min": json_number(tensor.min().item()),
        "max": json_number(tensor.max().item()),
        "nonfinite_count": int((~torch.isfinite(tensor)).sum().item()),
    }


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def check_outputs(outputs, expected, result, *, phase, replay, first_host=None):
    """Read the parent mesh tensors first, keeping their blocking CQ semantics."""
    expected_shape = (1, 1, ROWS, LOCAL_OUTPUT_WIDTH)
    for output_index, output in enumerate(outputs):
        host_output = first_host if output_index == 0 and first_host is not None else output.cpu(blocking=True)
        shards = ttnn.get_device_tensors(host_output)
        if len(shards) != RANKS:
            raise AssertionError(f"Expected {RANKS} output shards, got {len(shards)}")
        for rank, shard in enumerate(shards):
            actual = ttnn.to_torch(shard)
            reference = expected[..., rank * LOCAL_OUTPUT_WIDTH : (rank + 1) * LOCAL_OUTPUT_WIDTH]
            result["rank_outputs_checked"] += 1
            location = {
                "phase": phase,
                "replay": replay,
                "output_index": output_index,
                "layer": output_index // 2,
                "role": "attention" if output_index % 2 == 0 else "down",
                "rank": rank,
                "actual_shape": list(actual.shape),
                "expected_shape": list(expected_shape),
            }
            if tuple(actual.shape) != expected_shape:
                result["failure"] = {**location, "kind": "shape"}
                raise AssertionError(f"RS output shape mismatch: {location}")
            if torch.equal(actual, reference):
                continue
            mismatch = actual != reference
            flat_indices = mismatch.reshape(-1).nonzero().flatten()[:128]
            actual_flat = actual.reshape(-1)
            expected_flat = reference.reshape(-1)
            examples = []
            for index in flat_indices.tolist():
                row, column = divmod(index, LOCAL_OUTPUT_WIDTH)
                examples.append(
                    {
                        "coordinate": [0, 0, row, column],
                        "global_column": rank * LOCAL_OUTPUT_WIDTH + column,
                        "tile_id": (row // 32) * (LOCAL_OUTPUT_WIDTH // 32) + column // 32,
                        "actual": json_number(actual_flat[index].item()),
                        "expected": json_number(expected_flat[index].item()),
                    }
                )
            result["failure"] = {
                **location,
                "kind": "value",
                "mismatch_count": int(mismatch.sum().item()),
                "actual_range": tensor_range(actual),
                "expected_range": tensor_range(reference),
                "first_bad_elements": examples,
            }
            raise AssertionError(
                f"RS mismatch in {phase}, replay {replay}, layer {output_index // 2}, "
                f"role {location['role']}, rank {rank}: {result['failure']['mismatch_count']} elements"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--chunks-per-sync", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/ifm_k2_horizon_7b/doc/optimized_full_model/reduce_scatter_transport.json"),
    )
    args = parser.parse_args()
    if min(args.repeats, args.chunks_per_sync, args.workers) < 1:
        parser.error("repeats, chunks-per-sync, and workers must be positive")

    torch.set_num_threads(16)
    seed = 314159
    rng = torch.Generator().manual_seed(seed)
    # Sharding the host width gives independent, deterministic integer BF16
    # inputs to the four ranks. All reduction orders are exact within [-8, 8].
    host_input = torch.randint(-2, 3, (1, 1, ROWS, RANKS * LOCAL_INPUT_WIDTH), generator=rng, dtype=torch.int8).to(
        torch.bfloat16
    )
    expected = torch.zeros((1, 1, ROWS, LOCAL_INPUT_WIDTH), dtype=torch.float32)
    for shard in host_input.chunk(RANKS, dim=-1):
        expected.add_(shard)
    del shard
    expected = expected.to(torch.bfloat16)

    result = {
        "pass": False,
        "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "mesh_shape": [1, RANKS],
        "local_input_shape": [1, 1, ROWS, LOCAL_INPUT_WIDTH],
        "local_output_shape": [1, 1, ROWS, LOCAL_OUTPUT_WIDTH],
        "input_dtype": "bfloat16",
        "input_integer_range": [-2, 2],
        "expected_integer_range": [-8, 8],
        "comparison": "exact; integer inputs make every BF16 ring reduction order exact",
        "tt_ccl_owners": LAYERS,
        "collectives_per_replay": 2 * LAYERS,
        "barrier_handle_control": {
            "omitted_ag_handle_indices_per_layer": [0, 0],
            "rs_handle_indices_per_layer": [1, 1],
            "ag_operations_enqueued": False,
            "explanation": "Consume the preceding AG barrier handle to match the bulk model's O/down RS parity",
        },
        "topology": "Ring",
        "num_links": 2,
        "num_workers_per_link": args.workers,
        "chunks_per_sync": args.chunks_per_sync,
        "num_buffers_per_channel": 2,
        "persistent_output_buffers": False,
        "trace_allocation_tracking": True,
        "inter_collective_host_syncs": 0,
        "readback_boundary": "after each entire 72-collective replay; parent tensor cpu(blocking=True)",
        "requested_replays": args.repeats,
        "completed_replays": 0,
        "initial_replay_exact": False,
        "rank_outputs_checked": 0,
        "device_close_complete": False,
        "replay_to_first_read_seconds": [],
    }
    mesh = None
    trace = None
    outputs = []
    warm_outputs = []
    started = time.perf_counter()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, RANKS), trace_region_size=200_000_000)
        input_tensor = ttnn.from_torch(
            host_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        del host_input
        ccls = [TT_CCL(mesh) for _ in range(LAYERS)]

        def run_collectives():
            retained = []
            for ccl in ccls:
                for _ in range(2):
                    # Bulk QKV/SwiGLU AG each consume barrier slot 0 before
                    # O/down RS uses slot 1. Match those handle indices while
                    # keeping this reproduction limited to RS device work.
                    ccl.get_and_cycle_barrier_semaphore_handle()
                    retained.append(
                        ttnn.experimental.reduce_scatter_minimal_async(
                            input_tensor,
                            dim=3,
                            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
                            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
                            num_links=2,
                            topology=ttnn.Topology.Ring,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            chunks_per_sync=args.chunks_per_sync,
                            num_workers_per_link=args.workers,
                            num_buffers_per_channel=2,
                        )
                    )
            return retained

        print("RS_TRANSPORT_WARMUP", flush=True)
        warm_outputs = run_collectives()
        check_outputs(warm_outputs, expected, result, phase="warmup", replay=0)
        warm_outputs.clear()
        gc.collect()
        ttnn.synchronize_device(mesh)
        result["warmup_exact"] = True

        print("RS_TRANSPORT_CAPTURE", flush=True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        outputs = run_collectives()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        # Fast-dispatch capture records trace nodes and returns without issuing
        # their workload (FDMeshCommandQueue::enqueue_mesh_workload bypass path).
        # These new output buffers have no valid result until an actual replay.
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        check_outputs(outputs, expected, result, phase="initial_replay", replay=0)
        result["initial_replay_exact"] = True
        write_result(args.output, result)

        for replay in range(1, args.repeats + 1):
            replay_started = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            # This first blocking parent read waits for the complete trace on
            # its queue. It is an elapsed diagnostic, not kernel-only latency.
            first_host = outputs[0].cpu(blocking=True)
            result["replay_to_first_read_seconds"].append(time.perf_counter() - replay_started)
            check_outputs(outputs, expected, result, phase="replay", replay=replay, first_host=first_host)
            del first_host
            result["completed_replays"] = replay
            if replay == 1 or replay % 10 == 0 or replay == args.repeats:
                print(f"RS_TRANSPORT_EXACT {replay}/{args.repeats}", flush=True)
                write_result(args.output, result)
        result["pass"] = True
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_result(args.output, result)
        if mesh is not None:
            try:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                outputs.clear()
                warm_outputs.clear()
                ttnn.close_mesh_device(mesh)
                result["device_close_complete"] = True
            except BaseException as error:
                result["pass"] = False
                result["close_error"] = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                result["elapsed_seconds"] = time.perf_counter() - started
                write_result(args.output, result)
    print(json.dumps({"pass": result["pass"], "completed_replays": result["completed_replays"]}), flush=True)


if __name__ == "__main__":
    main()
