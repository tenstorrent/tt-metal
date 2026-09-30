# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for xing40_a4b_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""


import torch


def reference(spec, layers=None, dtype=None):
    """The standalone chunked CPU reference (reference/xing_ref.py): mHC 4-stream residual, dense causal MLA with a
    latent cache, DeepSeek-style MoE. All requested layers stay resident (40 layers: 122 GB in fp32). The HF oracle
    is the stock loader (trust_remote_code); its [B, S, 4, H] layer output flattens to the reference's [S * 4, H]
    block output, so check_hf needs no hf_model / hf_layers hook."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.xing40_a4b_d_p.reference.xing_ref import XingReference

    return XingReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def device_component(mesh, spec, layer, step):
    raise NotImplementedError(f"implement step: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("implement step: no device model yet")
