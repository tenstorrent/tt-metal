# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for gemma4_31b_glx: the framework reaches the model only through these functions.

Device side (implement stages): device_component, device_model, and optionally device_params. The CPU side
(reference, tokenizer, hf_model, hf_layers) is in reference_hooks.py, the reference stage's.
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw); the framework's lint fails it if it does.
"""

import torch

NORM_WEIGHTS = {"attn_norm": "input_layernorm"}  # step -> checkpoint norm (the other norms: their own tasks)


def residual_to_device(mesh, x: torch.Tensor):
    """Harness boundary: a chunk [S, H] to the residual layout between blocks, [1, 1, S / (R * C), H] bf16 TILE per chip.
    Chip (r, t) holds chunk rows r * S/R + t * S/(R*C) + [0, S/(R*C)): the SP row split, then the TP row split."""
    import ttnn

    rows, cols = tuple(mesh.shape)
    s, h = x.shape[-2:]
    return ttnn.from_torch(
        x.reshape(rows, cols, s // (rows * cols), h).to(torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (rows, cols), dims=(0, 1)),
    )


def residual_to_host(mesh, t) -> torch.Tensor:
    """Inverse of residual_to_device: [S, H] float32."""
    import ttnn

    rows, cols = tuple(mesh.shape)
    composer = ttnn.create_mesh_composer(
        mesh, ttnn.MeshComposerConfig(dims=[0, 1], mesh_shape_override=ttnn.MeshShape(rows, cols))
    )
    out = ttnn.to_torch(t, mesh_composer=composer)
    return out.reshape(-1, out.shape[-1]).float()


class _ResidualStep:
    """A device module on the residual layout, called by the component test with torch tensors."""

    def __init__(self, mesh, module):
        self.mesh, self.module = mesh, module

    def __call__(self, ctx, *xs):
        y = self.module(*[residual_to_device(self.mesh, x) for x in xs])
        return residual_to_host(self.mesh, y)


def device_component(mesh, spec, layer, step):
    import ttnn
    from models.demos.gemma4_31b_glx.tt import weights
    from models.demos.gemma4_31b_glx.tt.rms_norm import RMSNorm
    from models.demos.gemma4_31b_glx.tt.settings import settings

    # The fast runtime skips ttnn's pre-operation hooks (the framework's device-execution proof counts ops by them).
    ttnn.CONFIG.enable_fast_runtime_mode = settings.get("COMPONENT_FAST_RUNTIME")
    ckpt = spec.get("paths.hf")
    if step in NORM_WEIGHTS:
        w = weights.load(ckpt, f"layers.{layer}.{NORM_WEIGHTS[step]}.weight")
        return _ResidualStep(mesh, RMSNorm(mesh, w, weights.text_config(ckpt)["rms_norm_eps"]))
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
