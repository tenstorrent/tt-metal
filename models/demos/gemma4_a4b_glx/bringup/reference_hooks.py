# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-side bring-up hooks for gemma4_a4b_glx: the reference stage owns this file; implement jobs may not edit it (the goldens
and every test are computed from it).

reference(spec, layers, dtype), and optionally tokenizer(spec), hf_model(spec, num_layers), hf_layers(model).
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw).
"""


import torch


def reference(spec, layers=None, dtype=None):
    from models.demos.gemma4_a4b_glx.reference.gemma4_ref import Gemma4Reference

    return Gemma4Reference(str(spec.hf_dir), layers=layers, dtype=dtype or torch.float32)


def hf_layers(model):
    """Decoder layers of the HF oracle (Gemma4ForConditionalGeneration: model.model.language_model.layers)."""
    inner = model.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers
