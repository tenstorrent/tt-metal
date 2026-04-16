# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Standalone workload for the profiler cross-reference tests.

Opens a mesh device with both the real-time profiler and the device
profiler active, runs a ResNet50 inference pass, and dumps

  * every real-time profiler record       -> ``RT_RECORDS_PATH`` (JSON)
  * every device profiler kernel duration -> ``DEV_PERF_PATH``   (JSON)

Run this as a subprocess from the pytest-level test so that the device
profiler env vars (``TT_METAL_DEVICE_PROFILER=1`` etc.) are already set
before ``MetalContext`` / ``RuntimeOptions`` are constructed — the
``RuntimeOptions`` singleton caches those env vars on first creation and
never re-reads them, so they have to be in place before any device call.

Environment variables consumed:
  MESH_SHAPE        "rows,cols"  (required, e.g. "1,1" or "8,4")
  RT_RECORDS_PATH   output JSON file for real-time profiler records
  DEV_PERF_PATH     output JSON file for device profiler durations
  REQUIRE_GALAXY    "1" -> exit 3 when the cluster is not GALAXY/TG

Exit codes:
  0  success
  1  bad invocation (missing env vars etc.)
  2  insufficient devices for requested mesh (pytest should skip)
  3  REQUIRE_GALAXY set but not on a Galaxy/TG cluster (pytest should skip)
  4  device profiler readback failed
"""

import ast
import json
import os
import sys
import threading
import time

import ttnn


DEVICE_KERNEL_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _run_resnet50(device):
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


def main():
    mesh_shape_raw = os.environ.get("MESH_SHAPE")
    rt_path = os.environ.get("RT_RECORDS_PATH")
    dev_path = os.environ.get("DEV_PERF_PATH")
    if not mesh_shape_raw or not rt_path or not dev_path:
        print("ERROR: MESH_SHAPE / RT_RECORDS_PATH / DEV_PERF_PATH must be set", file=sys.stderr)
        sys.exit(1)

    rows, cols = [int(x) for x in mesh_shape_raw.split(",")]

    if os.environ.get("REQUIRE_GALAXY") == "1":
        try:
            cluster_type = ttnn.cluster.get_cluster_type()
            is_galaxy = cluster_type in (ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG)
        except Exception:
            is_galaxy = False
        if not is_galaxy:
            print("SKIP: Not a Galaxy/TG system", file=sys.stderr)
            sys.exit(3)

    if ttnn.GetNumAvailableDevices() < rows * cols:
        print(f"ERROR: Need {rows*cols} devices, have {ttnn.GetNumAvailableDevices()}", file=sys.stderr)
        sys.exit(2)

    device = ttnn.open_mesh_device(
        ttnn.MeshShape(rows, cols),
        l1_small_size=24576,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )

    rt_records = []
    lock = threading.Lock()

    def collect(record):
        with lock:
            rt_records.append(
                {
                    "program_id": record.program_id,
                    "chip_id": record.chip_id,
                    "start_timestamp": record.start_timestamp,
                    "end_timestamp": record.end_timestamp,
                    "frequency_ghz": record.frequency,
                }
            )

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)

    try:
        _run_resnet50(device)
        ttnn.synchronize_device(device)
    finally:
        time.sleep(2.0 if rows * cols > 1 else 1.0)
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)

        try:
            ttnn.ReadDeviceProfiler(device)
            dev_perf = ttnn.get_all_programs_perf_data()
        except Exception as exc:
            print(f"ERROR: device profiler readback failed: {exc}", file=sys.stderr)
            ttnn.close_mesh_device(device)
            sys.exit(4)

        # Flatten device profiler data to (chip_id, runtime_id, duration_ns)
        # triples so the outer test has everything it needs without needing
        # to re-import the C++ bindings.
        dev_flat = []
        for chip_id, programs in dev_perf.items():
            for program in programs:
                runtime_id = program.program_execution_uid.runtime_id
                analyses = program.program_analyses_results
                if DEVICE_KERNEL_DURATION_KEY not in analyses:
                    continue
                duration_ns = analyses[DEVICE_KERNEL_DURATION_KEY].duration
                dev_flat.append(
                    {
                        "chip_id": chip_id,
                        "runtime_id": runtime_id,
                        "duration_ns": duration_ns,
                    }
                )

        ttnn.close_mesh_device(device)

        with lock:
            rt_snapshot = list(rt_records)

        with open(rt_path, "w") as f:
            json.dump(rt_snapshot, f)
        with open(dev_path, "w") as f:
            json.dump(dev_flat, f)

        print(f"Saved {len(rt_snapshot)} RT records -> {rt_path}")
        print(f"Saved {len(dev_flat)} device perf entries -> {dev_path}")


if __name__ == "__main__":
    main()
