#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Run the FastAPI server for YOLOv11 pose estimation
cd /home/ubuntu/pose/tt-metal
export PYTHONPATH=/home/ubuntu/pose/tt-metal:$PYTHONPATH
source python_env/bin/activate
uvicorn models.demos.yolov11.pose_web_demo.server.fast_api_yolov11_pose:app --host 0.0.0.0 --port 8000
