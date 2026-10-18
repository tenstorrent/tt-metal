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

NORM_WEIGHTS = {  # step -> checkpoint norm (the other norms: their own tasks)
    "attn_norm": "input_layernorm",
    "post_attn_norm": "post_attention_layernorm",
}


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


class _AttentionStep:
    """A sliding attention layer called by the component test: (ctx, x [S, H]) -> [S, H].

    Harness boundary: before the chunk it loads the ctx's KV prefix (the reference's state: key / value
    [Hkv, >= prefix_len, D], K in NeoX lane order) into the device's served cache, one chunk at a time through the same
    cache write the forward uses. K lanes go to the served adjacent-pair order; SP row r, TP column t gets chunk k's
    positions k*C + r*L + [0, L) of heads 4t..4t+3."""

    def __init__(self, mesh, module):
        self.mesh, self.module = mesh, module

    def _load_prefix(self, state, prefix_len):
        import ttnn
        from models.demos.gemma4_31b_glx.tt.attention import sliding_kv_indices

        m = self.module
        rows, cols = tuple(self.mesh.shape)
        order = sliding_kv_indices(m.d)
        for start in range(0, prefix_len, m.chunk):
            kv = []
            for name in ("key", "value"):
                x = state[name][:, start : start + m.chunk]  # [Hkv, C, D]
                if name == "key":
                    x = x[..., order]
                # [Hkv, SP * L, D] -> [SP, Hkv, L, D]; mesh (r, t) takes SP slab r, heads [t * Hkv/TP, ...)
                x = x.reshape(x.shape[0], rows, m.slab, m.d).permute(1, 0, 2, 3).contiguous()
                kv.append(
                    ttnn.from_torch(
                        x.float(),  # fp32: the cache write's fp32 -> bfp8 typecast is unbiased (bf16 -> bfp8 is not)
                        device=self.mesh,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, (rows, cols), dims=(0, 1)),
                    )
                )
            m.write_kv(kv[0], kv[1], start)
            for t in kv:
                ttnn.deallocate(t)

    def __call__(self, ctx, x):
        prefix_len = ctx.extra.get("prefix_len", ctx.start)
        assert prefix_len == ctx.start, (prefix_len, ctx.start)
        if prefix_len:
            self._load_prefix(ctx.extra["state_prefix"], prefix_len)
        y = self.module(residual_to_device(self.mesh, x), ctx.start)
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
    cfg = weights.text_config(ckpt)
    if step == "attention" and cfg["layer_types"][layer] == "sliding_attention":
        from models.demos.gemma4_31b_glx.tt.attention import SlidingAttention

        names = ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]
        w = {n: weights.load(ckpt, f"layers.{layer}.self_attn.{n}.weight").float() for n in names}
        module = SlidingAttention(mesh, w, cfg, max_seq=spec.get("target.seq"), chunk=spec.get("target.chunk"))
        return _AttentionStep(mesh, module)
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
