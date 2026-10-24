# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for mimo_v26_flash_rl: the framework reaches the model only through these functions.

Device side (implement stages): device_component, device_model, and optionally device_params. The CPU side
(reference, tokenizer, hf_model, hf_layers) is in reference_hooks.py, the reference stage's.
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw); the framework's lint fails it if it does.
"""

import contextlib

import torch

from models.demos.mimo_v26_flash_rl.reference.weights import Checkpoint


def _checkpoint(spec):
    return Checkpoint(spec.hf_dir, torch.bfloat16)


@contextlib.contextmanager
def _op_hooks():
    """ttnn runs its pre-operation hooks (what the framework's device-execution proof counts ops with) only outside
    fast runtime mode, the C++ config default; the component adapters run in the slow runtime so the proof sees the
    ops. Same toggle as ttnn.graph's capture; the ops and their results are unchanged."""
    import ttnn

    prev = ttnn.CONFIG.enable_fast_runtime_mode
    ttnn.CONFIG.enable_fast_runtime_mode = False
    try:
        yield
    finally:
        ttnn.CONFIG.enable_fast_runtime_mode = prev


def _norm_component(mesh, spec, layer, weight_name):
    from models.demos.mimo_v26_flash_rl.tt.mesh_io import residual_to_host, residual_to_mesh
    from models.demos.mimo_v26_flash_rl.tt.rms_norm import RMSNorm

    ck = _checkpoint(spec)
    norm = RMSNorm(mesh, ck.get(f"model.layers.{layer}.{weight_name}"), ck.config["layernorm_epsilon"])

    def fn(ctx, x):
        with _op_hooks():
            tx = residual_to_mesh(mesh, x)
            ty = norm(tx)
            tx.deallocate(True)
            out = residual_to_host(mesh, ty).to(x.dtype)
            ty.deallocate(True)
        return out

    return fn


def device_component(mesh, spec, layer, step):
    if step == "attn_norm":
        return _norm_component(mesh, spec, layer, "input_layernorm.weight")
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
