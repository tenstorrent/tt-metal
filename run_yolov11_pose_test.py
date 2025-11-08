#!/usr/bin/env python3
import subprocess
import sys

# Run the YOLO11 pose PCC test
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "models/demos/yolov11/tests/pcc/test_ttnn_yolov11_pose_model.py::test_yolov11_pose_model",
    ],
    check=True,
)

sys.exit(result.returncode)
