# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for hy4_preview_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def reference(spec, layers=None, dtype=None):
    """The standalone chunked sparse CPU reference (reference/hy4_ref.py). Routed experts are held in memory for small
    layer sets (the 0-5 subset: 5 MoE layers, ~39 GB each in fp32) and read per expert from the checkpoint beyond
    that."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.hy4_preview_d_p.reference.hy4_ref import Hy4Reference

    return Hy4Reference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF oracle: transformers 5.17's hy_v4 modeling code (vendored, reference/hf_hy_v4) with the checkpoint's
    weights, eager attention, routed experts read per expert from the checkpoint when they run
    (reference/hf_oracle.py). num_layers=None is the full 78-layer model in bf16 (HF sanity gate: smoke and
    next-token accuracy; HF keeps iHC, sinks, router bias, indexer k_norm / weights_proj and the LM head in fp32);
    a layer prefix is built in fp32 (check_hf parity)."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.hy4_preview_d_p.reference.hf_oracle import build_hf_model

    return build_hf_model(hf_path(spec), num_layers, torch.bfloat16 if num_layers is None else torch.float32)


class _StreamTap(torch.nn.Module):
    """Identity module called with one HF decoder layer's output streams, flattened to [B, S, 4H]."""

    def forward(self, x):
        return x


def hf_layers(model):
    """Per-layer taps for check_hf. An HF decoder layer returns (streams [B, S, 4, H], topk); the reference's block
    output is the flat [S, 4H] layout (reference/hy4_ref.py), so each tap is called from the layer's own forward hook
    with ``streams.flatten(2)`` (same element order) and check_hf's hook reads the tap."""

    def hook(tap):
        def fn(module, inputs, output):
            tap(output[0].flatten(2))  # returns None: the layer's output is unchanged

        return fn

    taps = []
    for layer in model.model.layers:
        tap = _StreamTap()
        layer.register_forward_hook(hook(tap))
        taps.append(tap)
    return taps


def device_component(mesh, spec, layer, step):
    raise NotImplementedError(f"implement step: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("implement step: no device model yet")
