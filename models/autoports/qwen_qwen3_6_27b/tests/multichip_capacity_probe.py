# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process TP4 full-stack DRAM capacity probe.

The controller binary-searches context in 64-token increments.  Every candidate
is delegated to a fresh worker process because an out-of-memory allocation can
leave a device allocator unsuitable for the next measurement.  A worker reserves
the selected per-device model categories, then allocates the 16 local K and 16
local V caches separately, matching the full model's cache object cardinality.

This is a capacity test, not an execution test, so it deliberately does not open
fabric.  The model-resident placeholders use BF16 storage rounded to a tile-sized
byte boundary; KV placeholders use the decoder's real BFP8 dtype and local shape.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import ttnn

MESH_SHAPE = (1, 4)
CONTEXT_QUANTUM = 64
FULL_ATTENTION_LAYERS = 16
LOCAL_KV_HEADS = 1
HEAD_DIM = 256
PAGE_SIZE = 64

# Actual current layer-stack residency at C=262144, including physical BFP tile
# overhead, separate decode/prefill/long-prefill objects, norms/misc tensors,
# and per-layer RoPE.  This is intentionally more conservative than the unique
# semantic-weight estimate because it models objects the current loader keeps.
DEFAULT_WEIGHT_BYTES_PER_DEVICE = 10_599_141_888
DEFAULT_LINEAR_STATE_BYTES_PER_DEVICE_B32 = 572_522_496
DEFAULT_WORKSPACE_RESERVE_BYTES_PER_DEVICE = 1_969_152
WEIGHT_RESIDENCY_BREAKDOWN = {
    "norms": 1_310_720,
    "duplicated_tp_mlp_decode_prefill": 4_812_963_840,
    "full_attention_projection_objects": 2_264_924_160,
    "full_attention_qk_norms": 524_288,
    "per_layer_rope_c262144": 2_147_483_648,
    "linear_projection_objects": 1_362_493_440,
    "linear_misc_constants": 9_441_792,
}
LINEAR_STATE_BREAKDOWN_B32 = {
    "linear_convolution_state": 251_658_240,
    "linear_recurrent_state": 320_864_256,
}
assert sum(WEIGHT_RESIDENCY_BREAKDOWN.values()) == DEFAULT_WEIGHT_BYTES_PER_DEVICE
assert sum(LINEAR_STATE_BREAKDOWN_B32.values()) == DEFAULT_LINEAR_STATE_BYTES_PER_DEVICE_B32


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def _json_dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _memory_view(device) -> dict:
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    fields = (
        "num_banks",
        "total_bytes_per_bank",
        "total_bytes_allocated_per_bank",
        "total_bytes_free_per_bank",
        "largest_contiguous_bytes_free_per_bank",
    )
    result = {name: int(getattr(view, name)) for name in fields}
    for suffix in ("total_bytes_per_bank", "total_bytes_allocated_per_bank", "total_bytes_free_per_bank"):
        result[suffix.replace("_per_bank", "")] = result["num_banks"] * result[suffix]
    result["largest_contiguous_bytes_free_per_bank"] = int(view.largest_contiguous_bytes_free_per_bank)
    return result


def _mesh_memory_views(mesh) -> list[dict]:
    ttnn.synchronize_device(mesh)
    # Python exposes the mesh-level MemoryView (the symmetric allocator view),
    # not the underlying IDevice objects.  Every placeholder is replicated, so
    # the same per-device allocator state applies to all four ranks.
    view = _memory_view(mesh)
    return [dict(device_index=index, symmetric_mesh_view=True, **view) for index in range(len(mesh.get_device_ids()))]


def _byte_placeholders(mesh, requested_bytes: int):
    """Allocate at least requested_bytes/device without constructing host data."""

    # A BF16 32x32 tile occupies 2048 payload bytes.  Keep dimensions modest
    # while forcing the requested physical payload to a whole-tile boundary.
    tensors, allocated_bytes = [], 0
    remaining = requested_bytes
    # Model residency is many tensors, not one monolithic allocation.  Bound
    # placeholders to 512 MiB to avoid tensor-size limits masquerading as OOM.
    while remaining:
        chunk = min(remaining, 512 << 20)
        chunk = _round_up(chunk, 2048)
        width = chunk // (32 * 2)
        tensors.append(
            ttnn.empty(
                (1, 1, 32, width),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        allocated_bytes += chunk
        remaining -= min(remaining, 512 << 20)
    return tensors, allocated_bytes


def _worker(args: argparse.Namespace) -> int:
    result_path = Path(args.worker_result_json)
    result = {
        "schema_version": 1,
        "mode": "worker",
        "pid": os.getpid(),
        "batch": args.batch,
        "context": args.worker_context,
        "mesh_shape": list(MESH_SHAPE),
        "fabric_opened": False,
        "success": False,
        "categories": {},
        "memory_views": {},
    }
    mesh = None
    allocations = []
    started = time.perf_counter()
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE), trace_region_size=args.trace_region_size)
        result["device_ids"] = [int(device_id) for device_id in mesh.get_device_ids()]
        result["memory_views"]["opened"] = _mesh_memory_views(mesh)

        if args.weight_bytes == DEFAULT_WEIGHT_BYTES_PER_DEVICE:
            weight_categories = tuple((f"weight_{name}", value) for name, value in WEIGHT_RESIDENCY_BREAKDOWN.items())
        else:
            weight_categories = (("selected_tp4_weights_override", args.weight_bytes),)
        if args.linear_state_bytes_b32 == DEFAULT_LINEAR_STATE_BYTES_PER_DEVICE_B32:
            state_categories = tuple(
                (f"state_{name}", value * args.batch // 32) for name, value in LINEAR_STATE_BREAKDOWN_B32.items()
            )
        else:
            state_categories = (("linear_state_override", args.linear_state_bytes_b32 * args.batch // 32),)
        categories = (
            weight_categories + state_categories + (("measured_peak_warmed_trace_workspace", args.workspace_bytes),)
        )
        for name, requested in categories:
            tensors, allocated = _byte_placeholders(mesh, requested)
            allocations.extend(tensors)
            result["categories"][name] = {
                "requested_bytes_per_device": requested,
                "allocated_payload_bytes_per_device": allocated,
            }
            result["memory_views"][f"after_{name}"] = _mesh_memory_views(mesh)

        # Local TP4 cache shape is [B, C, one KV head, D=256].  Allocate K/V
        # independently for every full-attention layer, as the decoder stack does.
        cache_blocks = args.batch * (args.worker_context // PAGE_SIZE)
        cache_shape = (cache_blocks, LOCAL_KV_HEADS, PAGE_SIZE, HEAD_DIM)
        for layer in range(FULL_ATTENTION_LAYERS):
            for cache_kind in ("key", "value"):
                allocations.append(
                    ttnn.empty(
                        cache_shape,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
            result["memory_views"][f"after_layer_{layer:02d}_kv"] = _mesh_memory_views(mesh)

        result["kv"] = {
            "tensor_count": 2 * FULL_ATTENTION_LAYERS,
            "shape_per_tensor_per_device": list(cache_shape),
            "dtype": "bfloat8_b",
            "logical_payload_bytes_per_tensor_per_device": args.batch * args.worker_context * HEAD_DIM,
            "logical_payload_bytes_all_32_tensors_per_device": (
                2 * FULL_ATTENTION_LAYERS * args.batch * args.worker_context * HEAD_DIM
            ),
        }
        result["memory_views"]["complete"] = _mesh_memory_views(mesh)
        result["success"] = True
        return_code = 0
    except BaseException as error:  # Persist evidence for allocation errors as well as Python exceptions.
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        if mesh is not None:
            try:
                result["memory_views"]["at_failure"] = _mesh_memory_views(mesh)
            except BaseException as view_error:
                result["memory_view_failure"] = f"{type(view_error).__name__}: {view_error}"
        return_code = 1
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        _json_dump(result_path, result)
        allocations.clear()
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except BaseException:
                pass
    return return_code


def _git_revision() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_candidate(args: argparse.Namespace, context: int, artifact_dir: Path) -> dict:
    stem = f"b{args.batch}_c{context}"
    worker_json = artifact_dir / "workers" / f"{stem}.json"
    log_path = artifact_dir / "workers" / f"{stem}.log"
    worker_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-context",
        str(context),
        "--worker-result-json",
        str(worker_json),
        "--batch",
        str(args.batch),
        "--weight-bytes",
        str(args.weight_bytes),
        "--linear-state-bytes-b32",
        str(args.linear_state_bytes_b32),
        "--workspace-bytes",
        str(args.workspace_bytes),
        "--trace-region-size",
        str(args.trace_region_size),
    ]
    started = time.perf_counter()
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    record = {
        "context": context,
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "worker_json": str(worker_json),
        "combined_log": str(log_path),
        "success": completed.returncode == 0,
    }
    if worker_json.exists():
        worker_result = json.loads(worker_json.read_text())
        record["success"] = completed.returncode == 0 and bool(worker_result.get("success"))
        record["worker_result"] = worker_result
    else:
        record["worker_result_missing"] = True
    return record


def _controller(args: argparse.Namespace) -> int:
    if args.max_context < CONTEXT_QUANTUM:
        raise ValueError(f"--max-context must be at least {CONTEXT_QUANTUM}")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    attempted: dict[int, dict] = {}

    def probe(step: int) -> bool:
        context = step * CONTEXT_QUANTUM
        if context not in attempted:
            attempted[context] = _run_candidate(args, context, artifact_dir)
            print(f"B{args.batch} C{context}: {'PASS' if attempted[context]['success'] else 'FAIL'}", flush=True)
        return attempted[context]["success"]

    maximum_step = args.max_context // CONTEXT_QUANTUM
    low, high = 0, maximum_step + 1  # low passes (zero needs no process); high is exclusive/failing sentinel.
    if probe(maximum_step):
        low = maximum_step
    else:
        high = maximum_step
        while low + 1 < high:
            middle = (low + high) // 2
            if probe(middle):
                low = middle
            else:
                high = middle

    largest = low * CONTEXT_QUANTUM
    smallest_failure = high * CONTEXT_QUANTUM if high <= maximum_step else None
    summary = {
        "schema_version": 1,
        "mode": "controller",
        "git_revision": _git_revision(),
        "python": sys.executable,
        "script": str(Path(__file__).resolve()),
        "mesh_shape": list(MESH_SHAPE),
        "fabric_opened": False,
        "batch": args.batch,
        "context_quantum": CONTEXT_QUANTUM,
        "maximum_context_requested": args.max_context,
        "largest_feasible_context": largest,
        "smallest_failed_context": smallest_failure,
        "model_resident_bytes_per_device": {
            "selected_tp4_weights": args.weight_bytes,
            "linear_recurrent_state_at_batch_32": args.linear_state_bytes_b32,
            "conservative_workspace_and_safety_reserve": args.workspace_bytes,
        },
        "weight_residency_breakdown_per_device": WEIGHT_RESIDENCY_BREAKDOWN,
        "linear_state_breakdown_batch_32_per_device": LINEAR_STATE_BREAKDOWN_B32,
        "workspace_reserve_is_measured": args.workspace_bytes == DEFAULT_WORKSPACE_RESERVE_BYTES_PER_DEVICE,
        "workspace_reserve_note": "Default is the measured maximum load-to-warmed/captured/replayed DRAM allocation delta from final B32 full/linear trace artifacts.",
        "worker_isolation": "one fresh subprocess and mesh open/close per context candidate",
        "attempts": [attempted[context] for context in sorted(attempted)],
    }
    summary_path = Path(args.result_json) if args.result_json else artifact_dir / f"capacity_b{args.batch}.json"
    _json_dump(summary_path, summary)
    print(
        json.dumps(
            {key: summary[key] for key in ("batch", "largest_feasible_context", "smallest_failed_context")}, indent=2
        )
    )
    return 1 if args.require_max_pass and largest != maximum_step * CONTEXT_QUANTUM else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=32)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument(
        "--artifact-dir", default="models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/artifacts/capacity"
    )
    parser.add_argument("--result-json")
    parser.add_argument(
        "--require-max-pass",
        action="store_true",
        help="Fail unless the rounded maximum context passes (use for B1 C262144).",
    )
    parser.add_argument("--weight-bytes", type=int, default=DEFAULT_WEIGHT_BYTES_PER_DEVICE)
    parser.add_argument("--linear-state-bytes-b32", type=int, default=DEFAULT_LINEAR_STATE_BYTES_PER_DEVICE_B32)
    parser.add_argument("--workspace-bytes", type=int, default=DEFAULT_WORKSPACE_RESERVE_BYTES_PER_DEVICE)
    parser.add_argument("--trace-region-size", type=int, default=4_000_000)
    parser.add_argument("--worker-context", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_context is not None and not args.worker_result_json:
        parser.error("--worker-result-json is required with --worker-context")
    return args


def main() -> int:
    args = _parse_args()
    if args.worker_context is not None:
        return _worker(args)
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
