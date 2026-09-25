# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for gemma4_a4b_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""


import torch


def reference(spec, layers=None, dtype=None):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import Gemma4Reference

    return Gemma4Reference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_layers(model):
    """Decoder layers of the HF oracle (Gemma4ForConditionalGeneration: model.model.language_model.layers)."""
    inner = model.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


def device_component(mesh, spec, layer, step):
    raise NotImplementedError(f"implement step: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("implement step: no device model yet")
