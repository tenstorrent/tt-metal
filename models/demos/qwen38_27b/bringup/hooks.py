# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for qwen38_27b: the framework reaches the model only through these functions.

Device side (implement stages): device_component, device_model, and optionally device_params. The CPU side
(reference, tokenizer, hf_model, hf_layers) is in reference_hooks.py, the reference stage's.
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw); the framework's lint fails it if it does.
"""

import contextlib

import torch

import ttnn

NORM_WEIGHTS = {"attn_norm": "input_layernorm.weight", "ffn_norm": "post_attention_layernorm.weight"}


def _to_device(mesh, x: torch.Tensor) -> ttnn.Tensor:
    """Host [S, hidden] -> [1, 1, S / sp, hidden] bf16 per chip: rows split over axis 0 (a chunk's 640-row blocks in
    order, the engine's block-cyclic split within one chunk), replicated over the TP columns (axis 1)."""
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]).to(torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None)),
    )


def _to_host(mesh, t: ttnn.Tensor, shape) -> torch.Tensor:
    """The replicated residual layout back to host [S, hidden]: rows from axis 0, TP column 0's copy."""
    out = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(2, 3), mesh_shape=tuple(mesh.shape)))
    return out[..., : shape[-1]].reshape(shape).float()


@contextlib.contextmanager
def _hooked_dispatch():
    """ttnn's fast runtime (FastOperation) skips the pre-operation hooks the framework's device-execution proof counts
    ops with; the standard runtime runs the same device ops and calls them. Scoped to one component call."""
    fast = ttnn.CONFIG.enable_fast_runtime_mode
    ttnn.CONFIG.enable_fast_runtime_mode = False
    try:
        yield
    finally:
        ttnn.CONFIG.enable_fast_runtime_mode = fast


def device_component(mesh, spec, layer, step):
    from models.demos.qwen38_27b.reference.qwen38_ref import PREFIX, Checkpoint, Config
    from models.demos.qwen38_27b.tt.norm import TtRMSNorm

    if step in NORM_WEIGHTS:
        cfg = Config.load(str(spec.hf_dir))
        w = Checkpoint(str(spec.hf_dir)).get(f"{PREFIX}layers.{layer}.{NORM_WEIGHTS[step]}", torch.bfloat16)
        norm = TtRMSNorm(mesh, w, cfg.eps)

        def run(ctx, x):
            with _hooked_dispatch():
                return _to_host(mesh, norm(_to_device(mesh, x)), x.shape)

        return run
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
