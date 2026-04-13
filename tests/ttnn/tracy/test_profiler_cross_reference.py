# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Cross-reference test: real-time profiler vs device profiler.

Runs ResNet50 inference with both profiling systems active, then verifies
that the kernel durations reported by each system agree within tolerance.

Real-time profiler: streams per-program start/end timestamps over a D2H
socket; host receives them via RegisterProgramRealtimeProfilerCallback.

Device profiler: firmware writes zone markers to L1/DRAM; host reads them
back via ttnn.ReadDeviceProfiler() and post-processes into per-program
"DEVICE KERNEL DURATION [ns]" via ttnn.get_all_programs_perf_data().
"""

import ast
import json
import threading
import time
from collections import defaultdict

import pytest

import ttnn


DEVICE_ID_NUM_BITS = 10
RELATIVE_TOLERANCE = 0.20
DEVICE_KERNEL_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Programs with device kernel duration below this threshold are excluded
# from the tolerance check.  Very short kernels (<10 µs) have large
# relative overhead from firmware dispatch boundaries, making the
# real-time profiler's wall-clock measurement diverge significantly
# from the device profiler's kernel-zone measurement.
MIN_DEVICE_KERNEL_DURATION_NS = 10_000


def decode_runtime_id(encoded_runtime_id):
    """Decode a per-device encoded runtime_id to extract the base program ID."""
    return (encoded_runtime_id >> DEVICE_ID_NUM_BITS) & 0x1FFFFF


def test_profiler_cross_reference(monkeypatch, tmp_path):
    """
    Run ResNet50 inference with both profiling systems, then cross-reference
    per-program kernel durations between the real-time profiler and the
    device profiler.

    Device profiler env vars must be set before device creation, so this
    test manages its own device lifecycle rather than using the ``device``
    fixture.
    """
    profiler_mod = getattr(ttnn, "_ttnn", None)
    if profiler_mod is None or not hasattr(profiler_mod, "profiler"):
        pytest.skip("Profiler bindings not available in this build")

    # Env vars must be set BEFORE any call that triggers MetalContext
    # construction (including GetNumAvailableDevices), because the
    # RuntimeOptions singleton caches TT_METAL_DEVICE_PROFILER on first
    # creation and never re-reads the environment.
    for var, value in {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_SYNC": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
        "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
    }.items():
        monkeypatch.setenv(var, value)

    if ttnn.GetNumAvailableDevices() < 1:
        pytest.skip("No devices available")

    device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=24576,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )

    try:
        _run_cross_reference(device, tmp_path)
    finally:
        ttnn.close_mesh_device(device)


def _run_resnet_inference(device):
    """Run a single ResNet50 inference pass (batch_size=16)."""
    labels_path = "models/sample_data/imagenet_class_labels.txt"
    with open(labels_path, "r") as f:
        imagenet_label_dict = ast.literal_eval(f.read())

    input_loc = "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images/"

    from models.demos.vision.classification.resnet50.ttnn_resnet.demo.demo import run_resnet_inference

    run_resnet_inference(
        batch_size_per_device=16,
        input_loc=input_loc,
        imagenet_label_dict=imagenet_label_dict,
        device=device,
        model_location_generator=None,
    )


def _run_cross_reference(device, tmp_path):
    # -- 1. Register real-time profiler callback --
    rt_records = []
    lock = threading.Lock()

    def collect_record(record):
        entry = {
            "program_id": record.program_id,
            "chip_id": record.chip_id,
            "start_timestamp": record.start_timestamp,
            "end_timestamp": record.end_timestamp,
            "frequency_ghz": record.frequency,
        }
        with lock:
            rt_records.append(entry)

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect_record)

    # -- 2. Run ResNet50 inference workload --
    try:
        _run_resnet_inference(device)
        ttnn.synchronize_device(device)
    finally:
        time.sleep(1.0)
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)

    # -- 3. Read device profiler results --
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception as exc:
        pytest.skip(f"ReadDeviceProfiler failed (profiling may be disabled): {exc}")

    try:
        dev_perf = ttnn.get_all_programs_perf_data()
    except RuntimeError as exc:
        if "profiler_state_manager is nullptr" in str(exc):
            pytest.skip("Profiler state manager not initialized")
        raise

    # -- 4. Snapshot data for diagnostics --
    with lock:
        rt_snapshot = list(rt_records)

    assert len(rt_snapshot) > 0, "No real-time profiler records collected"
    assert dev_perf, "No device profiler data returned"

    # -- 5. Build device profiler duration map --
    dev_durations_by_raw_id = defaultdict(list)
    dev_durations_by_decoded_id = defaultdict(list)

    for chip_id, programs in dev_perf.items():
        for program in programs:
            uid = program.program_execution_uid
            runtime_id = uid.runtime_id
            analyses = program.program_analyses_results

            if DEVICE_KERNEL_DURATION_KEY not in analyses:
                continue

            duration_ns = analyses[DEVICE_KERNEL_DURATION_KEY].duration
            if duration_ns <= 0:
                continue

            dev_durations_by_raw_id[runtime_id].append(duration_ns)
            decoded_base = decode_runtime_id(runtime_id)
            dev_durations_by_decoded_id[decoded_base].append(duration_ns)

    assert (
        dev_durations_by_raw_id or dev_durations_by_decoded_id
    ), "No device profiler programs with kernel duration data"

    # -- 6. Build real-time profiler duration map --
    rt_durations_by_id = defaultdict(list)
    for rec in rt_snapshot:
        pid = rec["program_id"]
        if pid == 0:
            continue
        freq = rec["frequency_ghz"]
        if freq <= 0:
            continue
        duration_ns = (rec["end_timestamp"] - rec["start_timestamp"]) / freq
        if duration_ns <= 0:
            continue
        rt_durations_by_id[pid].append(duration_ns)

    assert rt_durations_by_id, "No valid real-time profiler records with nonzero program_id and duration"

    # -- 7. Determine best ID matching strategy --
    raw_match_count = sum(1 for pid in rt_durations_by_id if pid in dev_durations_by_raw_id)
    decoded_match_count = sum(1 for pid in rt_durations_by_id if pid in dev_durations_by_decoded_id)

    if raw_match_count >= decoded_match_count:
        dev_durations = dev_durations_by_raw_id
        match_strategy = "raw"
    else:
        dev_durations = dev_durations_by_decoded_id
        match_strategy = "decoded"

    # -- 8. Cross-reference matched programs --
    matched = 0
    within_tolerance = 0
    skipped_short = 0
    comparison_details = []

    for pid, rt_durs in sorted(rt_durations_by_id.items()):
        if pid not in dev_durations:
            continue

        dev_durs = dev_durations[pid]
        pairs = min(len(rt_durs), len(dev_durs))

        for i in range(pairs):
            rt_ns = rt_durs[i]
            dev_ns = dev_durs[i]

            if dev_ns > 0:
                relative_error = abs(rt_ns - dev_ns) / dev_ns
            else:
                relative_error = float("inf")

            short = dev_ns < MIN_DEVICE_KERNEL_DURATION_NS
            if short:
                skipped_short += 1

            ok = relative_error <= RELATIVE_TOLERANCE
            if not short:
                matched += 1
                if ok:
                    within_tolerance += 1

            comparison_details.append(
                {
                    "program_id": pid,
                    "instance": i,
                    "rt_duration_ns": round(rt_ns, 1),
                    "dev_duration_ns": round(dev_ns, 1),
                    "relative_error": round(relative_error, 4),
                    "within_tolerance": ok,
                    "below_threshold": short,
                }
            )

    # -- 9. Save diagnostics --
    diagnostics = {
        "match_strategy": match_strategy,
        "rt_record_count": len(rt_snapshot),
        "dev_program_count": sum(len(progs) for progs in dev_perf.values()),
        "rt_unique_program_ids": len(rt_durations_by_id),
        "dev_unique_program_ids_raw": len(dev_durations_by_raw_id),
        "dev_unique_program_ids_decoded": len(dev_durations_by_decoded_id),
        "matched_pairs": matched,
        "skipped_short": skipped_short,
        "within_tolerance": within_tolerance,
        "tolerance": RELATIVE_TOLERANCE,
        "min_device_kernel_duration_ns": MIN_DEVICE_KERNEL_DURATION_NS,
        "comparisons": comparison_details,
    }

    out_file = tmp_path / "profiler_cross_reference.json"
    with open(out_file, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n=== Profiler Cross-Reference Results ===")
    print(f"  Match strategy:          {match_strategy}")
    print(f"  RT records:              {len(rt_snapshot)}")
    print(f"  RT unique program IDs:   {len(rt_durations_by_id)}")
    print(
        f"  Dev unique program IDs:  {len(dev_durations_by_raw_id)} (raw) / {len(dev_durations_by_decoded_id)} (decoded)"
    )
    print(f"  Matched (>={MIN_DEVICE_KERNEL_DURATION_NS}ns): {matched}  (skipped {skipped_short} short)")
    print(f"  Within {RELATIVE_TOLERANCE*100:.0f}% tolerance:  {within_tolerance}/{matched}")
    print(f"  Diagnostics:             {out_file}")

    for detail in comparison_details:
        tag = "skip" if detail["below_threshold"] else ("OK" if detail["within_tolerance"] else "FAIL")
        print(
            f"    pid={detail['program_id']:>6} [{detail['instance']}]: "
            f"rt={detail['rt_duration_ns']:>12.1f} ns  dev={detail['dev_duration_ns']:>12.1f} ns  "
            f"err={detail['relative_error']*100:>6.2f}%  {tag}"
        )

    # -- 10. Assertions --
    assert matched > 0, (
        f"No program IDs matched (above {MIN_DEVICE_KERNEL_DURATION_NS} ns threshold) "
        f"between real-time profiler ({len(rt_durations_by_id)} IDs) "
        f"and device profiler ({len(dev_durations)} IDs). "
        f"RT IDs: {sorted(rt_durations_by_id.keys())[:20]}, "
        f"Dev IDs (raw): {sorted(dev_durations_by_raw_id.keys())[:20]}, "
        f"Dev IDs (decoded): {sorted(dev_durations_by_decoded_id.keys())[:20]}"
    )

    pass_rate = within_tolerance / matched if matched > 0 else 0
    assert pass_rate >= 0.50, (
        f"Only {within_tolerance}/{matched} ({pass_rate*100:.1f}%) matched pairs are within "
        f"{RELATIVE_TOLERANCE*100:.0f}% tolerance (need >= 50%). "
        f"See {out_file} for details."
    )
