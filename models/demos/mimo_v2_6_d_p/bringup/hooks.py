# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for mimo_v2_6_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def _physical_threads():
    """Use one torch thread per physical core: the gates set os.cpu_count() (SMT) threads, which makes the many small
    per-expert GEMMs and mxfp4 dequants 2-7x slower on this host."""
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
    except Exception:
        n = None
    if n and torch.get_num_threads() > n:
        torch.set_num_threads(n)


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/mimo_ref.py). Experts are dequantized at load for small layer sets
    (the 0-5 subset) and kept mxfp4, dequantized per use, for the 48-layer parity run."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoReference

    _physical_threads()
    return MiMoReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF oracle: the checkpoint's modeling_mimo_v2.py, text decoder only, fp32, eager attention, weights
    dequantized as the checkpoint defines them, routed experts kept mxfp4 and dequantized when they run
    (reference/hf_oracle.py). num_layers=None is the full 48-layer model (HF sanity gate)."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.hf_oracle import build_hf_model

    _physical_threads()
    return build_hf_model(hf_path(spec), num_layers, torch.float32)


def device_component(mesh, spec, layer, step):
    raise NotImplementedError(f"implement step: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("implement step: no device model yet")
