# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Standalone workload script for the host-device correlation test.

Runs ResNet50 inference (same as test_demo_sample) while collecting
real-time profiler device records via callback. Saves the collected
records to a JSON file whose path is specified by the DEVICE_RECORDS_PATH
environment variable.

This script is meant to be run under Tracy capture by the outer test
(test_host_device_correlation.py).
"""

import ast
import json
import os
import sys
import threading
import time

import ttnn


def main():
    records_path = os.environ.get("DEVICE_RECORDS_PATH")
    if not records_path:
        print("ERROR: DEVICE_RECORDS_PATH environment variable not set", file=sys.stderr)
        sys.exit(1)

    records = []
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
            records.append(entry)

    # Read imagenet labels (same source as models/conftest.py)
    labels_path = "models/sample_data/imagenet_class_labels.txt"
    with open(labels_path, "r") as f:
        imagenet_label_dict = ast.literal_eval(f.read())

    input_loc = "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images/"

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)

    # Register callback to capture device-side program records
    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect_record)

    try:
        from models.demos.vision.classification.resnet50.ttnn_resnet.demo.demo import run_resnet_inference

        run_resnet_inference(
            batch_size_per_device=16,
            input_loc=input_loc,
            imagenet_label_dict=imagenet_label_dict,
            device=mesh_device,
            model_location_generator=None,
        )
    finally:
        # Give the receiver thread time to deliver remaining records
        time.sleep(1.0)
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)

        with lock:
            snapshot = list(records)

        with open(records_path, "w") as f:
            json.dump(snapshot, f, indent=2)

        print(f"Saved {len(snapshot)} device records to {records_path}")

        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
