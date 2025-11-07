#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Simple client runner based on yolov4 web demo pattern
# Usage: ./run_simple.sh --api-url http://server-ip:8000

streamlit run yolov11_pose_simple.py -- "$@"
