# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-side bring-up hooks for qwen38_27b: the reference stage owns this file; implement jobs may not edit it (the goldens
and every test are computed from it).

reference(spec, layers, dtype), and optionally tokenizer(spec), hf_model(spec, num_layers), hf_layers(model).
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw).
"""

import torch

from models.demos.qwen38_27b.reference.qwen38_ref import Qwen38Reference


def reference(spec, layers=None, dtype=None):
    return Qwen38Reference(spec, str(spec.hf_dir), layers=layers, dtype=dtype or torch.float32)


def hf_layers(model):
    """Qwen3_5ForConditionalGeneration keeps the decoder layers under model.language_model."""
    inner = model.model
    return getattr(inner, "language_model", inner).layers
