# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for qwen38_27b: the framework reaches the model only through these functions.

Device side (implement stages): device_component, device_model, and optionally device_params. The CPU side
(reference, tokenizer, hf_model, hf_layers) is in reference_hooks.py, the reference stage's.
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw); the framework's lint fails it if it does.
"""

import torch


def _layer_weights(spec, layer, names):
    from models.demos.qwen38_27b.reference.model import Checkpoint

    return Checkpoint(spec.hf_dir).layer(layer, names, torch.float32)


def _rows_to_device(mesh, x):
    """Harness boundary: a [S, H] host tensor as the residual stream, [1, 1, S/8, H] bf16 TILE per chip, the
    sequence split over mesh rows and replicated over columns. Within one chunk the server's block-cyclic layout
    (position p on row (p // 640) % 8) is the contiguous split."""
    import ttnn

    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None)),
    )


def _rows_to_host(mesh, t):
    """Harness boundary: the residual stream back as [S, H] fp32 (column 0 of each mesh row; the columns are
    replicas)."""
    import ttnn

    rows, cols = tuple(mesh.shape)
    shards = ttnn.get_device_tensors(t)
    out = torch.cat([ttnn.to_torch(shards[r * cols]) for r in range(rows)], dim=-2)
    return out.reshape(out.shape[-2], out.shape[-1]).float()


def device_component(mesh, spec, layer, step):
    from models.demos.qwen38_27b.reference.model import Config

    cfg = Config(spec.hf_dir)
    if step == "attn_norm":
        from models.demos.qwen38_27b.tt.norm import TtRmsNorm

        w = _layer_weights(spec, layer, ["input_layernorm.weight"])["input_layernorm.weight"]
        mod = TtRmsNorm(mesh, w, cfg.eps)
    else:
        raise NotImplementedError(f"implement stage: no device module for {step} yet")

    def fn(ctx, x):
        with _ops_visible():
            return _rows_to_host(mesh, mod(_rows_to_device(mesh, x)))

    return fn


class _ops_visible:
    """Harness boundary only: run a component call through ttnn's slow-runtime Operation wrapper, the only one that
    calls ttnn's pre-operation hooks (FastOperation.__call__ skips them), so the gate's device-execution proof can
    count the ops. The same ops run on the device either way; the previous mode is restored after the call."""

    def __enter__(self):
        import ttnn

        self.prev = ttnn.CONFIG.enable_fast_runtime_mode
        ttnn.CONFIG.enable_fast_runtime_mode = False

    def __exit__(self, *exc):
        import ttnn

        ttnn.CONFIG.enable_fast_runtime_mode = self.prev
        return False


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
