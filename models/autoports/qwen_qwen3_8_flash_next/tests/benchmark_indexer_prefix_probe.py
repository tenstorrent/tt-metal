# SPDX-License-Identifier: Apache-2.0
"""Classic indexer device prefix-cap regression and isolated wall-time probe.

Requires a rebuilt ttnn binding/backend and exclusive device ownership. Uses
ordinary trace completion synchronization only in this standalone experiment;
never install these measurement boundaries in a live serving phase collector.

Build after all device clients have exited, using the checkout's activated env:
  cmake --build build_Release --target ttnn --parallel 4
  cmake --install build_Release --component ttnn-runtime
  cp build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
The build products are build_Release/ttnn/{_ttnncpp.so,_ttnn.so}; installation
updates build_Release/lib/_ttnncpp.so, while the final copy updates the imported
Python binding. Check the actual import and loaded shared-library paths before
running this probe. No binary replacement is safe during another device job.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch

import ttnn


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upload(value, mesh, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        value.contiguous(),
        device=mesh,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def copy_host(value, target, mesh):
    source = ttnn.from_torch(
        value.contiguous(),
        dtype=target.dtype,
        layout=target.layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ttnn.copy_host_to_device_tensor(source, target)


def download(value):
    return [ttnn.to_torch(device_tensor).clone() for device_tensor in ttnn.get_device_tensors(value)]


def timing(mesh, trace_id, repeats):
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(repeats):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        samples.append((time.perf_counter() - start) * 1000 / repeats)
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def prefix_metrics(expected, actual, end):
    rows = []
    for ref, got in zip(expected, actual):
        a, b = ref[..., :end], got[..., :end]
        same = torch.equal(a, b)
        finite = torch.isfinite(a) & torch.isfinite(b)
        error = (a[finite].float() - b[finite].float()).abs()
        rows.append(
            {
                "exact_valid_prefix": same,
                "prefix_elements": a.numel(),
                "finite_elements": int(finite.sum()),
                "max_finite_error": float(error.max()) if error.numel() else 0.0,
                "nonfinite_pattern_equal": bool(torch.equal(torch.isfinite(a), torch.isfinite(b))),
            }
        )
        if not same:
            raise AssertionError(("Valid indexer prefix changed", end, rows[-1]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--head-groups", nargs="+", type=int, default=[4, 2])
    parser.add_argument("--bounds", nargs="+", type=int, default=[0, 1, 31, 32, 33, 1024, 65536])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--helper", action="store_true", help="Also qualify the unreferenced model prefix-bound helper")
    args = parser.parse_args()
    if args.capacity < 1024 or args.capacity % 128 or args.repeats < 1:
        parser.error("capacity must be >=1024 and aligned to128; repeats must be positive")
    if any(not 0 <= end <= args.capacity for end in args.bounds):
        parser.error("bounds must lie within full capacity")
    if any(group not in (1, 2, 4) for group in args.head_groups):
        parser.error("head-groups must divide the four input heads")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") not in (None, "0"):
        raise RuntimeError("Disable the device profiler for this isolated host-wall probe")
    if args.output.exists():
        raise FileExistsError("Use a fresh output path to preserve previous evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.with_suffix("").with_name(args.output.stem + "_raw")
    raw_dir.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(1701)
    q_host = torch.randn(1, 4, 32, 128, dtype=torch.bfloat16)
    k_host = torch.randn(1, 1, args.capacity, 128, dtype=torch.bfloat16)
    weights_host = torch.ones(1, 1, 32, 4, dtype=torch.bfloat16)
    input_path = raw_dir / "inputs.pt"
    torch.save({"q": q_host, "k_finite": k_host, "weights": weights_host}, input_path)
    native_root = Path("ttnn/cpp/ttnn/operations/experimental/indexer_score")
    sources = [
        native_root / "device/indexer_score_device_operation.hpp",
        native_root / "device/indexer_score_device_operation.cpp",
        native_root / "device/indexer_score_program_factory.cpp",
        native_root / "device/kernels/reader_indexer_score.cpp",
        native_root / "indexer_score_nanobind.cpp",
        Path(__file__),
        Path(__file__).resolve().parents[1] / "tt/indexer_prefix.py",
        Path(__file__).resolve().parents[1] / "tt/kernels/indexer_prefix.cpp",
    ]
    report = {
        "status": "started",
        "args": vars(args),
        "scope": "Synthetic BF16 Q/K, Qwen-shaped all-one gates; native operation regression, not model accuracy",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {str(path): digest(path) for path in sources},
        "binding": {"path": ttnn._ttnn.__file__, "sha256": digest(ttnn._ttnn.__file__)},
        "loaded_ttnn_libraries": [line for line in Path("/proc/self/maps").read_text().splitlines() if "_ttnn" in line],
        "inputs": {"path": str(input_path), "sha256": digest(input_path)},
        "timing": "3 warmed host perf_counter samples, asynchronous trace replays then completion sync",
        "cases": [],
        "cache_hit_checks": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    save()
    mesh = None
    traces = []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16000000)
        report["device_ids"] = list(mesh.get_device_ids())
        q = upload(q_host, mesh)
        keys = upload(k_host, mesh)
        weights = upload(weights_host, mesh)
        bound = upload(
            torch.tensor([args.capacity], dtype=torch.int32), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        report["bound_address"] = bound.buffer_address()
        report["key_shape"] = list(keys.shape)
        report["key_padded_shape"] = list(keys.padded_shape)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        for head_group in args.head_groups:
            config = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=head_group)

            def score(cap=None, chunk_start=None):
                return ttnn.experimental.indexer_score_dsa(
                    q,
                    keys,
                    weights,
                    chunk_start_idx=args.capacity - 32 if chunk_start is None else chunk_start,
                    cache_batch_idx=0,
                    program_config=config,
                    compute_kernel_config=compute,
                    valid_end_tensor=cap,
                )

            # Warm both programs, then measure only one live trace at a time.
            # A later capture's output allocation can overlap an earlier trace's
            # scratch range even when both Python output owners remain alive.
            for _ in range(2):
                warm = score()
                del warm
                warm = score(bound)
                del warm
            ttnn.synchronize_device(mesh)
            baseline_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            baseline = score()
            ttnn.end_trace_capture(mesh, baseline_trace, cq_id=0)
            traces.append(baseline_trace)
            trace_row = {"head_group": head_group, "baseline": str(baseline_trace)}
            report.setdefault("traces", []).append(trace_row)
            case_rows = []
            for poison in (False, True):
                for end in args.bounds:
                    row = {"head_group": head_group, "end": end, "poison_suffix": poison}
                    report["cases"].append(row)
                    case_rows.append(row)
                    save()
                    live_keys = k_host.clone()
                    if poison:
                        live_keys[..., end:, :] = float("nan")
                    copy_host(live_keys, keys, mesh)
                    ttnn.execute_trace(mesh, baseline_trace, cq_id=0, blocking=False)
                    reference = download(baseline)
                    raw_path = raw_dir / f"g{head_group}_end{end}_poison{int(poison)}_baseline.pt"
                    torch.save(reference, raw_path)
                    row["baseline_raw"] = {"path": str(raw_path), "sha256": digest(raw_path)}
                    row["baseline_wall"] = timing(mesh, baseline_trace, args.repeats)
                    del reference
                    save()
            ttnn.release_trace(mesh, baseline_trace)
            traces.remove(baseline_trace)
            del baseline

            # No baseline device output/trace survives into candidate capture.
            copy_host(k_host, keys, mesh)
            copy_host(torch.tensor([args.capacity], dtype=torch.int32), bound, mesh)
            candidate_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            candidate = score(bound)
            ttnn.end_trace_capture(mesh, candidate_trace, cq_id=0)
            traces.append(candidate_trace)
            assert tuple(candidate.shape) == (1, 1, 32, args.capacity)
            trace_row["candidate"] = str(candidate_trace)
            for row in case_rows:
                end, poison = row["end"], row["poison_suffix"]
                live_keys = k_host.clone()
                if poison:
                    live_keys[..., end:, :] = float("nan")
                copy_host(live_keys, keys, mesh)
                copy_host(torch.tensor([end], dtype=torch.int32), bound, mesh)
                # Sentinel makes skipped output writes observable, including
                # end=0 where every worker has an empty K/output interval.
                copy_host(torch.full(tuple(candidate.shape), -123.0, dtype=torch.bfloat16), candidate, mesh)
                ttnn.execute_trace(mesh, candidate_trace, cq_id=0, blocking=False)
                reference = torch.load(row["baseline_raw"]["path"], weights_only=True)
                actual = download(candidate)
                raw_path = raw_dir / f"g{head_group}_end{end}_poison{int(poison)}.pt"
                torch.save({"baseline": reference, "candidate": actual}, raw_path)
                row["raw"] = {"path": str(raw_path), "sha256": digest(raw_path)}
                row["per_device"] = prefix_metrics(reference, actual, end)
                rounded = (end + 31) // 32 * 32
                row["unwritten_suffix_preserved"] = all(bool((x[..., rounded:] == -123).all()) for x in actual)
                assert row["unwritten_suffix_preserved"], "Writer touched columns outside rounded cap"
                # Explicit CPU post-mask compares the complete consumer-facing
                # score matrix, including poisoned and unwritten suffixes.
                for ref, got in zip(reference, actual):
                    ref[..., end:] = -float("inf")
                    got[..., end:] = -float("inf")
                    assert torch.equal(ref, got)
                row["masked_full_output_equal"] = True
                row["candidate_wall"] = timing(mesh, candidate_trace, args.repeats)
                row["speedup"] = row["baseline_wall"]["median_ms"] / row["candidate_wall"]["median_ms"]
                save()
                print("INDEXER_PREFIX", json.dumps(row), flush=True)
            ttnn.release_trace(mesh, candidate_trace)
            traces.remove(candidate_trace)
            del candidate

            # Same shape/program cache, different bound buffer AND scalar causal
            # position: catches stale pointer or geometry in the cache-hit hook.
            # Both traces have been released before any new device allocation.
            copy_host(k_host, keys, mesh)
            alternate = upload(
                torch.tensor([33], dtype=torch.int32), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            expected = download(score(chunk_start=0))
            actual = download(score(alternate, chunk_start=0))
            check = {
                "head_group": head_group,
                "address": alternate.buffer_address(),
                "per_device": prefix_metrics(expected, actual, 33),
            }
            assert alternate.buffer_address() != bound.buffer_address()
            raw_path = raw_dir / f"g{head_group}_cache_hit.pt"
            torch.save({"baseline": expected, "candidate": actual}, raw_path)
            check["raw"] = {"path": str(raw_path), "sha256": digest(raw_path)}
            report["cache_hit_checks"].append(check)
            save()
            del alternate
            if args.helper and head_group == args.head_groups[0]:
                from models.autoports.qwen_qwen3_8_flash_next.tt.indexer_prefix import prepare_indexer_prefix

                packed_host = torch.zeros(32, 32, dtype=torch.int32)
                positions = upload(packed_host, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                copy_host(k_host, keys, mesh)
                helper_reference = download(score())
                for seq in (1, 128):

                    def helper_score():
                        prepare_indexer_prefix(mesh, positions, bound, seq=seq, lane=31, blocks=args.capacity)
                        return score(bound)

                    helper_output = helper_score()
                    ttnn.synchronize_device(mesh)
                    del helper_output
                    helper_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    helper_output = helper_score()
                    ttnn.end_trace_capture(mesh, helper_trace, cq_id=0)
                    traces.append(helper_trace)
                    starts = [0, 31, 32, 127, 128, 2046, 2047, 2048, args.capacity * 4 - seq, 0]
                    for start in starts:
                        # Other lanes differ from lane31 to expose accidental
                        # lane-zero reads; update device input without recapture.
                        packed_host[:, 0] = 256
                        packed_host[31, 0] = start
                        copy_host(packed_host, positions, mesh)
                        ttnn.execute_trace(mesh, helper_trace, cq_id=0, blocking=False)
                        cap = max(512, (((start + seq) // 4 + 31) // 32) * 32)
                        device_caps = [int(item.reshape(-1)[0]) for item in download(bound)]
                        assert device_caps == [cap] * len(report["device_ids"]), (start, seq, cap, device_caps)
                        actual = download(helper_output)
                        helper_row = {
                            "start": start,
                            "seq": seq,
                            "lane": 31,
                            "expected_cap": cap,
                            "device_caps": device_caps,
                            "per_device": prefix_metrics(helper_reference, actual, cap),
                            "wall": timing(mesh, helper_trace, args.repeats),
                        }
                        raw_path = raw_dir / f"helper_seq{seq}_start{start}_{len(report.get('helper_cases', []))}.pt"
                        torch.save(
                            {"baseline": helper_reference, "candidate": actual, "positions": packed_host.clone()},
                            raw_path,
                        )
                        helper_row["raw"] = {"path": str(raw_path), "sha256": digest(raw_path)}
                        report.setdefault("helper_cases", []).append(helper_row)
                        save()
                        print("INDEXER_HELPER", json.dumps(helper_row), flush=True)
                    ttnn.release_trace(mesh, helper_trace)
                    traces.remove(helper_trace)
                    del helper_output
                del positions
        report["status"] = "pass"
    except Exception:
        report.update(status="error", failure=traceback.format_exc())
        raise
    finally:
        save()
        if mesh is not None:
            for trace_id in traces:
                ttnn.release_trace(mesh, trace_id)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
