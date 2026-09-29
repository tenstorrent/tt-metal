# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for glm53_flash_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/glm_ref.py), the requested layers resident."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.glm53_flash_d_p.reference.glm_ref import GlmReference

    return GlmReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF glm5_next code (vendored) on the dequantized checkpoint, text model only, routed experts read from
    disk per use. num_layers=None: the whole model in bf16 (intake sanity); else the first num_layers in fp32."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.glm53_flash_d_p.reference.hf_oracle import build_hf_model

    return build_hf_model(hf_path(spec), num_layers, torch.bfloat16 if num_layers is None else torch.float32)


def device_component(mesh, spec, layer, step):
    raise NotImplementedError(f"implement step: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("implement step: no device model yet")
