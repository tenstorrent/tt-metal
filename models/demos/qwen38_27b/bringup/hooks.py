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
RESIDUAL_STEPS = ("attn_residual", "mlp_residual")


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
    if step in RESIDUAL_STEPS:
        from models.demos.qwen38_27b.tt.residual import TtResidualAdd

        add = TtResidualAdd()

        def run(ctx, x, y):
            with _hooked_dispatch():
                return _to_host(mesh, add(_to_device(mesh, x), _to_device(mesh, y)), x.shape)

        return run
    if step == "attention" and spec.block_type_of(layer) == "linear":
        from models.demos.qwen38_27b.tt.gdn import TtGatedDeltaNet

        cfg = Config.load(str(spec.hf_dir))
        ckpt = Checkpoint(str(spec.hf_dir))
        p = f"{PREFIX}layers.{layer}.linear_attn."
        w = {"linear_attn." + n[len(p) :]: ckpt.get(n, torch.bfloat16) for n in ckpt.index if n.startswith(p)}
        gdn = TtGatedDeltaNet(mesh, cfg, w)

        def run(ctx, x):
            # the state as of ctx.start (golden prefix, or zeros at 0) uploaded at the component boundary
            prefix = ctx.extra.get("state_prefix") or {}
            conv = prefix.get("conv", torch.zeros(cfg.conv_dim, cfg.conv_k))
            rec = prefix.get("recurrent", torch.zeros(cfg.lv_heads, cfg.lk_dim, cfg.lv_dim))
            if not ctx.start:
                conv, rec = torch.zeros_like(conv), torch.zeros_like(rec)
            state = gdn.state_to_device(conv, rec)
            with _hooked_dispatch():
                out, _, _ = gdn(_to_device(mesh, x), ctx.start, state)
                return _to_host(mesh, out, x.shape)

        return run
    if step == "attention" and spec.block_type_of(layer) == "attn":
        from models.demos.qwen38_27b.tt.attention import TtGatedAttention
        from models.demos.qwen38_27b.tt.kv_cache import TtKVCache

        cfg = Config.load(str(spec.hf_dir))
        ckpt = Checkpoint(str(spec.hf_dir))
        p = f"{PREFIX}layers.{layer}.self_attn."
        w = {"self_attn." + n[len(p) :]: ckpt.get(n, torch.bfloat16) for n in ckpt.index if n.startswith(p)}
        max_seq, chunk = spec.get("target")["seq"], spec.get("target")["chunk"]
        attn = TtGatedAttention(mesh, cfg, w, max_seq, chunk)
        cache = TtKVCache(mesh, cfg.n_kv, cfg.head_dim, max_seq, chunk)

        def run(ctx, x):
            # the served cache holds the golden prefix [0, ctx.start) (zeros past it), uploaded at the boundary
            prefix = ctx.extra.get("state_prefix") or {}
            empty = torch.zeros(cfg.n_kv, 0, cfg.head_dim)
            cache.load_prefix(prefix.get("key", empty), prefix.get("value", empty), ctx.start)
            with _hooked_dispatch():
                return _to_host(mesh, attn(_to_device(mesh, x), ctx.start, cache), x.shape)

        return run
    if step == "mlp":
        from models.demos.qwen38_27b.tt.mlp import TtMLP

        cfg = Config.load(str(spec.hf_dir))
        ckpt = Checkpoint(str(spec.hf_dir))
        p = f"{PREFIX}layers.{layer}.mlp."
        w = {"mlp." + n[len(p) :]: ckpt.get(n, torch.bfloat16) for n in ckpt.index if n.startswith(p)}
        mlp = TtMLP(mesh, cfg, w)

        def run(ctx, x):
            with _hooked_dispatch():
                return _to_host(mesh, mlp(_to_device(mesh, x)), x.shape)

        return run
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    """The all-device model (tt/model.py): hidden state on the device from the embedding to the final norm, the
    served KV cache and GDN state (with restart snapshots) as its state."""
    from models.demos.qwen38_27b.tt.model import TtQwen38Model

    return TtQwen38Model(mesh, spec, layers, lm_head=lm_head)


def contract_state_pcc(spec, runtime, kv, slot, length, golden) -> dict:
    """K.1: the Gated DeltaNet state (not in the KV table) of the prefill runtime vs the golden snapshot at length."""
    from models.demos.qwen38_27b.tt.runners.adapter import contract_state_pcc as pcc

    return pcc(spec, runtime, kv, slot, length, golden)
