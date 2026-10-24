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
    if step in ("attn_norm", "mlp_norm"):
        from models.demos.qwen38_27b.tt.norm import TtRmsNorm

        name = "input_layernorm.weight" if step == "attn_norm" else "post_attention_layernorm.weight"
        mod = TtRmsNorm(mesh, _layer_weights(spec, layer, [name])[name], cfg.eps)
    elif step == "mlp":
        from models.demos.qwen38_27b.reference.model import COMMON_NAMES
        from models.demos.qwen38_27b.tt.mlp import TtMlp

        names = [n for n in COMMON_NAMES if n.startswith("mlp.")]
        mod = TtMlp(mesh, _layer_weights(spec, layer, names))
    elif step == "attention" and cfg.is_linear(layer):
        return _linear_attention_component(mesh, spec, cfg, layer)
    elif step in ("attn_residual", "mlp_residual"):
        from models.demos.qwen38_27b.tt.residual import TtResidualAdd

        add = TtResidualAdd()

        def residual_fn(ctx, a, b):
            with _ops_visible():
                return _rows_to_host(mesh, add(_rows_to_device(mesh, a), _rows_to_device(mesh, b)))

        return residual_fn
    else:
        raise NotImplementedError(f"implement stage: no device module for {step} yet")

    def fn(ctx, x):
        with _ops_visible():
            return _rows_to_host(mesh, mod(_rows_to_device(mesh, x)))

    return fn


def _linear_attention_component(mesh, spec, cfg, layer):
    """Gated DeltaNet on device. The slot state at the chunk's start comes from the context (the golden prefix
    snapshot ``state_prefix``, or none at start 0) and goes to the device once per call at the harness boundary; the
    module returns the state after the chunk (``ctx.extra["state_out"]``, conv [conv_dim, 3] and recurrent fp32)."""
    from models.demos.qwen38_27b.reference.model import LINEAR_NAMES
    from models.demos.qwen38_27b.tt.linear_attention import TtGatedDeltaNet

    rows = tuple(mesh.shape)[0]
    mod = None

    def fn(ctx, x):
        nonlocal mod
        S = x.shape[-2]
        if mod is None:
            mod = TtGatedDeltaNet(mesh, _layer_weights(spec, layer, LINEAR_NAMES), cfg, S // rows)
        assert S == rows * mod.local_rows, f"chunk of {S} rows, the module was built for {rows} x {mod.local_rows}"
        prefix = (ctx.extra or {}).get("state_prefix") if ctx.start else None
        state = mod.state_from_host(
            None if prefix is None else prefix["conv"], None if prefix is None else prefix["recurrent"]
        )
        start = mod.scalar(ctx.start)
        with _ops_visible():
            out, new = mod(_rows_to_device(mesh, x), state, start)
        ctx.extra["state_out"] = {
            "conv": _conv_to_host(mesh, mod, new.conv),
            "recurrent": _rec_to_host(mesh, new.recurrent),
        }
        return _rows_to_host(mesh, out)

    return fn


def _conv_to_host(mesh, mod, t):
    """The conv history [1, 3, conv_tp] per column back in the reference's [conv_dim, 3] channel order."""
    import ttnn

    cols = tuple(mesh.shape)[1]
    shards = ttnn.get_device_tensors(t)
    per = torch.cat([ttnn.to_torch(shards[c]).float().reshape(3, -1) for c in range(cols)], dim=-1)  # [3, cols*conv_tp]
    kd, vd = mod.kd, mod.vd
    q, k, v = [], [], []
    for c in range(cols):
        blk = per[:, c * mod.conv_tp : (c + 1) * mod.conv_tp]
        q.append(blk[:, :kd])
        k.append(blk[:, kd : 2 * kd])
        v.append(blk[:, 2 * kd :])
    return torch.cat(q + k + v, dim=-1).T.contiguous()


def _rec_to_host(mesh, t):
    import ttnn

    cols = tuple(mesh.shape)[1]
    shards = ttnn.get_device_tensors(t)
    return torch.cat([ttnn.to_torch(shards[c]).float() for c in range(cols)], dim=0)


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
