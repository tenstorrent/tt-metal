# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

MODEL_REGISTRY = {}


def register_model(model_cfg_cls, pipeline_cls):
    if model_cfg_cls in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_cfg_cls}' already registered.")
    MODEL_REGISTRY[model_cfg_cls] = pipeline_cls
