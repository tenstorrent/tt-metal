# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 exact post-gather sum control; all Q1..32, no production monkey-patching."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker as tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tt.exact_rank_sum import exact_rank_sum
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import Collectives, MeshLoader


def fixtures(count):
    families = []
    for stress in (False, True):
        devices = []
        for device_rank in range(4):
            rng = torch.Generator().manual_seed(910 + count * 100 + device_rank)
            value = torch.randn((1, 4, count, 2560), generator=rng).bfloat16()
            if stress:
                # Exponent-gap cancellation distinguishes right nesting from a left fold.
                # The second feature is four negative zeros, so omitted +0 is visible.
                columns = torch.tensor(
                    [
                        [1e8, -0.0, 1, 1e-8, -1e8, -0.0, 1, 0],
                        [1, -0.0, 1e8, -1e-8, 1, -0.0, -1, 0],
                        [-1e8, -0.0, -1e8, 1e-16, 1e8, 0, 1e-8, 0],
                        [1, -0.0, 1, 0, 1, 0, -1e-8, 0],
                    ],
                    dtype=torch.bfloat16,
                ).roll(device_rank, dims=0)
                value = columns[:, None, :].repeat(1, count, 320).unsqueeze(0)
            devices.append(value)
        families.append(devices)
    # Identical live stress values, different unused token padding.
    return [families[0], families[1], families[1], families[1]]


def baseline(value):
    raw = ttnn.sum(ttnn.typecast(value, ttnn.float32), dim=1, keepdim=True)
    return raw, ttnn.typecast(ttnn.multiply(raw, 0.25), ttnn.bfloat16)


def candidate(value):
    raw = exact_rank_sum(value)
    return raw, ttnn.typecast(ttnn.multiply(raw, 0.25), ttnn.bfloat16)


def read_ranks(value, physical=False):
    result = []
    for device_tensor in ttnn.get_device_tensors(value):
        host = ttnn.from_device(device_tensor)
        if physical:
            host = host.reshape(host.padded_shape)
        result.append(ttnn.to_torch(host))
    return result


def read_outputs(outputs):
    return [read_ranks(value) for value in outputs]


def upload_poisoned(family, count, pad, mesh):
    """Upload poison as logical data, then hide it with a metadata-only view.

    preserve_nan_values forces host construction in py_to_tt_tensor.cpp, avoiding
    device tilize/typecast transformations of NaN payloads. Fully aligned source
    tensors do not rely on from_torch's implicit pad_value handling at all.
    """
    physical = []
    for live in family:
        full = torch.full((1, 4, 32, 2560), pad, dtype=torch.bfloat16)
        full[:, :, :count, :] = live
        physical.append(full)
    backing = ttnn.from_torch(
        torch.cat(physical, dim=3).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        preserve_nan_values=True,
    )
    addresses = [part.buffer_address() for part in ttnn.get_device_tensors(backing)]
    # Python Tensor.reshape delegates to routed ttnn::reshape, unlike the C++
    # Tensor::reshape method. Its one-shape overload passes Q as padded H too,
    # and the device volume-change guard rejects Q<32. The public TWO-shape
    # overload supplies the actual padded32 geometry: codegen rejects explicit
    # padding/volume change and reshape.cpp:778 takes its metadata-only view.
    # Host-only reshape success does not validate this device branch.
    view = ttnn.reshape(
        backing,
        ttnn.Shape((1, 4, count, 2560)),
        ttnn.Shape((1, 4, 32, 2560)),
    )
    assert addresses == [part.buffer_address() for part in ttnn.get_device_tensors(view)]
    assert tuple(view.shape) == (1, 4, count, 2560) and tuple(view.padded_shape) == (1, 4, 32, 2560)
    return view, physical


def padding_observation(actual, expected, count, pad, family, rank):
    padding = actual[:, :, count:, :].contiguous()
    bits, frequencies = torch.unique(padding.view(torch.int16), return_counts=True)
    values = padding.float()
    return {
        "count": count,
        "family": family,
        "rank": rank,
        "expected_padding": repr(pad),
        "physical_shape": list(actual.shape),
        "expected_physical_shape": list(expected.shape),
        "padding_elements": padding.numel(),
        "nan_count": int(torch.isnan(values).sum()),
        "positive_inf_count": int(torch.isposinf(values).sum()),
        "negative_inf_count": int(torch.isneginf(values).sum()),
        "unique_bf16_bits": [f"0x{int(value) & 0xffff:04x}" for value in bits[:16]],
        "unique_bit_counts": frequencies[:16].tolist(),
        "unique_bit_count": bits.numel(),
        "observed_values_sample": [repr(float(value)) for value in values.flatten()[:8]],
        "full_physical_bits_exact": torch.equal(actual.view(torch.int16), expected.view(torch.int16)),
        "padding_verified": bool(torch.isnan(values).all()) if pad != pad else bool((values == pad).all()),
    }


def exact(actual, expected):
    result = []
    for label, actual_ranks, expected_ranks, dtype in zip(
        ("raw_fp32", "downstream_bf16"), actual, expected, (torch.int32, torch.int16), strict=True
    ):
        assert len(actual_ranks) == len(expected_ranks) == 4
        for rank, (a, b) in enumerate(zip(actual_ranks, expected_ranks, strict=True)):
            result.append({"output": label, "rank": rank, "exact": torch.equal(a.view(dtype), b.view(dtype))})
    assert all(item["exact"] for item in result), result
    return result


def assert_input_unchanged(value, expected):
    actual = read_ranks(value, physical=True)
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual, expected, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-timing", action="store_true")
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    assert args.repeats > 0 and args.rounds > 0
    assert bool(tracker.TRACE_ALLOC_TRACKING) != args.clean_timing
    assert not args.clean_timing or not os.environ.get("TT_METAL_WATCHER")
    assert not args.clean_timing or os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("", "0")
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "tt/exact_rank_sum.py",
        root / "tt/multichip_decoder.py",
        Path("ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp"),
        Path("ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl"),
        Path("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h"),
        Path("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h"),
        Path("ttnn/cpp/ttnn-nanobind/pytensor.cpp"),
        Path("ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp"),
        Path("ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_nanobind.cpp"),
        Path("ttnn/cpp/ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_supported.cpp"),
        Path("ttnn/core/tensor/tensor_ops.cpp"),
        Path(
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp"
        ),
    ]
    sources += sorted((root / "tt/kernels").glob("exact_rank_sum_*.cpp"))
    report = {
        "passed": False,
        "rows": [],
        "padding_observations": [],
        "padding_construction": "Aligned host BF16 logical data with preserve_nan_values=True, then explicit ttnn.reshape(backing, logical_Q_shape, physical_32_shape); same-buffer assertion and full physical readback required",
        "padding_fixture_revision": "v3 explicitly supplies padded32. v2 used Python Tensor.reshape single shape and failed device volume validation; host-only tests did not exercise the device branch.",
        "mesh": [1, 4],
        "fabric_config": "FABRIC_1D_RING",
        "l1_small_size": 32768,
        "scope": "Synthetic actual-shape component control, not full-model acceptance or speedup",
        "tracking": bool(tracker.TRACE_ALLOC_TRACKING),
        "clean_timing": args.clean_timing,
        "watcher": os.environ.get("TT_METAL_WATCHER"),
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("TT_", "QWEN_"))},
        "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "boundary": "Preloaded persistent inputs; nonblocking replay loop, one synchronization at window end; no per-replay read/copy/sync",
        "preserved_math": "u_j=(FP32(rank_j)+positive_zero)+positive_zero; raw=u0+(u1+(u2+u3)); downstream=BF16(raw*0.25)",
        "equality": "Raw FP32 and downstream BF16 storage bits, including signed zero; live finite operands, padding includes NaN/Inf",
    }
    mesh, trace = None, None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def phase(label):
        report["last_phase"] = label
        print(f"EXACT_RANK_SUM_PHASE {label}", flush=True)

    try:
        phase("open_mesh")
        assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore the selected mesh descriptor before opening devices"
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        loader = MeshLoader({}, mesh)
        owner = Collectives(mesh)
        for count in range(1, 33):
            phase(f"q{count}:setup_and_baseline_eager")
            hosts = fixtures(count)
            prepared = [
                upload_poisoned(family, count, pad, mesh)
                for family, pad in zip(hosts, (0.0, 137.0, float("nan"), float("inf")), strict=True)
            ]
            inputs = [value for value, _ in prepared]
            snapshots = [read_ranks(value, physical=True) for value in inputs]
            for family, (snapshot, (_, expected_physical), pad) in enumerate(
                zip(snapshots, prepared, (0.0, 137.0, float("nan"), float("inf")), strict=True)
            ):
                assert len(snapshot) == len(expected_physical) == 4
                for rank, (actual, expected_input) in enumerate(zip(snapshot, expected_physical, strict=True)):
                    observed = padding_observation(actual, expected_input, count, pad, family, rank)
                    report["padding_observations"].append(observed)
                    print("EXACT_RANK_SUM_PADDING " + json.dumps(observed), flush=True)
                    assert observed["full_physical_bits_exact"] and observed["padding_verified"], observed
            del prepared
            expected = []
            for values in inputs:
                with runtime_audit():
                    output = baseline(values)
                expected.append(read_outputs(output))
                del output
            exact(expected[1], expected[2])
            exact(expected[1], expected[3])
            row = {"count": count, "eager": [], "traces": {}, "padding_invariance": True}
            report["rows"].append(row)
            if count in (1, 32):
                # Exercise the unchanged real synchronous collective separately from local timing.
                phase(f"q{count}:synchronous_gather_native_eager")
                local = loader.shards([hosts[0][r][:, r : r + 1] for r in range(4)], dim=3)
                with runtime_audit():
                    expected_sum = owner.sum(local)
                    gathered = owner.gather(local, dim=1)
                    actual_sum = exact_rank_sum(gathered)
                a, b = read_ranks(actual_sum), read_ranks(expected_sum)
                assert all(torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(a, b, strict=True))
                row["synchronous_gather_control"] = "raw_fp32_exact_all_four_devices"
                del local, expected_sum, gathered, actual_sum
            for index, values in enumerate(inputs):
                phase(f"q{count}:native_eager:family{index}")
                with runtime_audit():
                    output = candidate(values)
                row["eager"].append(
                    {
                        "family": (
                            "random_zero_pad",
                            "cancellation_finite_pad",
                            "cancellation_nan_pad",
                            "cancellation_inf_pad",
                        )[index],
                        "ranks": exact(read_outputs(output), expected[index]),
                    }
                )
                del output
            # All immutable input sets and persistent replay inputs predate capture.
            for value, snapshot in zip(inputs, snapshots, strict=True):
                assert_input_unchanged(value, snapshot)
            live = ttnn.empty_like(inputs[0])
            for name, function in (("post_gather_baseline", baseline), ("native_exact_rank_sum", candidate)):
                phase(f"q{count}:{name}:warm")
                ttnn.copy(inputs[0], live)
                with runtime_audit():
                    warm = function(live)
                del warm
                gc.collect()
                phase(f"q{count}:{name}:capture")
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    with runtime_audit():
                        output = function(live)
                finally:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                item = {
                    "changed_input_replays": [],
                    "windows_ms_per_replay": [],
                    "runtime_audit": "passed_eager_warm_capture",
                }
                row["traces"][name] = item
                if tracker.TRACE_ALLOC_TRACKING:
                    phase(f"q{count}:{name}:allocation_guard_positive_control")
                    # This trace owns and overwrites both retained capture outputs.
                    for value in output:
                        tracker.acknowledge_corruptible(value)
                    del value
                    sentinel = ttnn.empty_like(live)
                    try:
                        try:
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        except RuntimeError as error:
                            assert "still alive before trace replay" in str(error), str(error)
                            item["allocation_guard_positive_control"] = {"rejected": True, "error": str(error)}
                        else:
                            raise AssertionError("Allocation guard did not reject retained post-capture sentinel")
                    finally:
                        ttnn.deallocate(sentinel)
                        del sentinel
                phase(f"q{count}:{name}:changed_input_replay")
                for repeat in range(12):
                    family = repeat % 4
                    ttnn.copy(inputs[family], live)
                    if repeat < 4:
                        # Prove the changed-input copy preserved genuine poison
                        # in the persistent trace input, outside timed replay.
                        assert_input_unchanged(live, snapshots[family])
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    item["changed_input_replays"].append(
                        {"family": family, "ranks": exact(read_outputs(output), expected[family])}
                    )
                # Keep the last stress family resident; correctness checked after every window.
                phase(f"q{count}:{name}:timing")
                for _ in range(args.rounds):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.repeats):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    item["windows_ms_per_replay"].append((time.perf_counter() - start) * 1000 / args.repeats)
                    exact(read_outputs(output), expected[3])
                item["median_ms"] = statistics.median(item["windows_ms_per_replay"])
                item["repeats_per_window"] = args.repeats
                ttnn.release_trace(mesh, trace)
                trace = None
                del output
            row["speedup"] = (
                row["traces"]["post_gather_baseline"]["median_ms"] / row["traces"]["native_exact_rank_sum"]["median_ms"]
            )
            print(json.dumps({"count": count, "speedup": row["speedup"]}), flush=True)
            assert_input_unchanged(live, snapshots[3])
            del live, inputs, values, snapshots
        report["passed"] = True
        phase("passed")
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
