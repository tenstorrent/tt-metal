# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Common constants and utilities for SFace model."""

import os

# L1 small size for SFace model
# SFace has larger intermediate tensors (112x112 input, up to 1024 channels)
# Need larger L1 allocation than YuNet
SFACE_L1_SMALL_SIZE = 32768  # 32KB - tuned for 112x112 input

# ONNX model URL and path
SFACE_ONNX_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_ONNX_FILENAME = "face_recognition_sface_2021dec.onnx"


def get_sface_onnx_path():
    """Get path to SFace ONNX model, downloading if necessary."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ttnn", "models", "sface")
    os.makedirs(cache_dir, exist_ok=True)

    onnx_path = os.path.join(cache_dir, SFACE_ONNX_FILENAME)

    if not os.path.exists(onnx_path):
        import urllib.request

        print(f"Downloading SFace ONNX model to {onnx_path}...")
        urllib.request.urlretrieve(SFACE_ONNX_URL, onnx_path)
        print("Download complete.")

    return onnx_path
