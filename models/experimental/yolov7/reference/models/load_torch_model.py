# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.experimental.yolov7.reference.models.experimental import attempt_load


def get_yolov7_fused_cpu_model(model_location_generator):
    # Get model weights
    model_path = model_location_generator("models", model_subdir="Yolo")
    weights = str(model_path / "yolov7.pt")

    # Load model
    model = attempt_load(weights, map_location="cpu")  # load FP32 model
    model = model.fuse().eval()

    return model
