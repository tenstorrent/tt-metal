# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for gpt_oss_120b: the framework reaches the model only through these functions.

Device side (implement stages): device_component, device_model, and optionally device_params. The CPU side
(reference, tokenizer, hf_model, hf_layers) is in reference_hooks.py, the reference stage's.
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw); the framework's lint fails it if it does.
"""

import json
from pathlib import Path

NORM_WEIGHTS = {"attn_norm": "input_layernorm.weight", "ffn_norm": "post_attention_layernorm.weight"}


def _checkpoint(spec):
    from models.demos.gpt_oss_120b.reference.gpt_oss_ref import Checkpoint

    return Checkpoint(Path(spec.hf_dir))


def _config(spec) -> dict:
    return json.loads((Path(spec.hf_dir) / "config.json").read_text())


def _observable_ops():
    """Turn off ttnn's fast runtime mode so ttnn.register_pre_operation_hook sees every op. In fast mode (this
    tt-metal's default) FastOperation never calls PRE_OPERATION_HOOKS, so the framework's device-execution proof
    counts 0 ops for a module that does run on the device. Ops run the same; only their Python wrapper changes."""
    import ttnn

    ttnn.CONFIG.enable_fast_runtime_mode = False


class _HostBoundary:
    """Component harness boundary: each [S, H] host input -> device module on the mesh layout -> [S, H] host out."""

    def __init__(self, mesh, module):
        self.mesh, self.module = mesh, module

    def __call__(self, ctx, *xs):
        from models.demos.gpt_oss_120b.tt.layout import rows_to_device, rows_to_host

        tt_xs = [rows_to_device(self.mesh, x) for x in xs]
        tt_y = self.module(*tt_xs)
        out = rows_to_host(self.mesh, tt_y)
        for t in (*tt_xs, tt_y):
            t.deallocate(True)
        return out


ATTN_WEIGHTS = {
    "wq": "q_proj.weight",
    "bq": "q_proj.bias",
    "wk": "k_proj.weight",
    "bk": "k_proj.bias",
    "wv": "v_proj.weight",
    "bv": "v_proj.bias",
    "wo": "o_proj.weight",
    "bo": "o_proj.bias",
    "sinks": "sinks",
}


def _rope(mesh, spec, cfg):
    from models.demos.gpt_oss_120b.reference.gpt_oss_ref import yarn_inv_freq
    from models.demos.gpt_oss_120b.tt.attention import TtRope

    inv_freq, att = yarn_inv_freq(cfg)
    return TtRope(mesh, inv_freq, att, int(spec.get("target.seq")), int(spec.get("target.chunk")))


def _attention(mesh, spec, layer):
    from models.demos.gpt_oss_120b.tt.attention import TtAttention
    from models.demos.gpt_oss_120b.tt.ccl import TtCcl

    cfg = _config(spec)
    ck = _checkpoint(spec)
    w = {k: ck.get(f"model.layers.{layer}.self_attn.{n}") for k, n in ATTN_WEIGHTS.items()}
    ck.close()
    window = int(cfg["sliding_window"]) if cfg["layer_types"][layer] == "sliding_attention" else None
    return TtAttention(mesh, TtCcl(mesh), w, cfg, _rope(mesh, spec, cfg), window)


class _AttentionBoundary:
    """Component harness boundary of the stateful attention step: [S, H] host in, the golden KV prefix
    (ctx.extra state_prefix, positions [0, prefix_len)) loaded into a fresh served-layout cache, the device module,
    [S, H] host out. Every call starts from its own cache, so repeated calls see the same state."""

    def __init__(self, mesh, module, chunk: int):
        self.mesh, self.module, self.chunk = mesh, module, chunk

    def __call__(self, ctx, x):
        import torch

        from models.demos.gpt_oss_120b.tt.attention import meta_permute_cols
        from models.demos.gpt_oss_120b.tt.kv_cache import cache_from_host
        from models.demos.gpt_oss_120b.tt.layout import rows_to_device, rows_to_host

        assert x.shape[0] == self.chunk, f"one chunk of {self.chunk} rows, got {x.shape[0]}"
        n, seq = ctx.extra["prefix_len"], ctx.extra["max_seq"]
        caches = []
        for name in ("key", "value"):
            src = ctx.extra["state_prefix"][name][:n].float()  # [n, n_kv, 64], K half-split (the reference's)
            if name == "key":
                src = meta_permute_cols(src)  # the served cache holds K in meta order
            host = torch.zeros(1, src.shape[1], seq, src.shape[2])
            host[0, :, :n] = src.permute(1, 0, 2)
            caches.append(cache_from_host(self.mesh, host, self.chunk))
        tt_x = rows_to_device(self.mesh, x)
        tt_y = self.module(tt_x, ctx.start, *caches)
        out = rows_to_host(self.mesh, tt_y)
        for t in (tt_x, tt_y, *caches):
            t.deallocate(True)
        return out


def _experts(mesh, spec, layer):
    import torch

    from models.demos.gpt_oss_120b.reference.gpt_oss_ref import LayerWeights
    from models.demos.gpt_oss_120b.tt.ccl import TtCcl
    from models.demos.gpt_oss_120b.tt.experts import TtExperts

    cfg = _config(spec)
    ck = _checkpoint(spec)
    lw = LayerWeights(ck, layer, cfg, torch.float32, torch.bfloat16)  # mxfp4 dequantized (exact in bf16)
    ck.close()
    seq_local = int(spec.get("target.chunk")) // tuple(mesh.shape)[0]
    return TtExperts(
        mesh,
        TtCcl(mesh),
        lw.w_gu,
        lw.b_gu,
        lw.w_d,
        lw.b_d,
        cfg["num_experts_per_tok"],
        float(cfg.get("swiglu_limit", 7.0)),
        seq_local,
    )


class _ExpertsBoundary:
    """Component harness boundary of the experts step: x [S, H] (bf16 rows) and the dense routing weights [S, E]
    (fp32: the router's output dtype) to the residual-stream layout, the device module, [S, H] host out."""

    def __init__(self, mesh, module):
        self.mesh, self.module = mesh, module

    def __call__(self, ctx, x, routing):
        import ttnn
        from models.demos.gpt_oss_120b.tt.layout import rows_to_device, rows_to_host

        tt_x = rows_to_device(self.mesh, x)
        tt_r = ttnn.from_torch(
            routing.float().reshape(1, 1, *routing.shape[-2:]),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None)),
        )
        tt_y = self.module(tt_x, tt_r)
        out = rows_to_host(self.mesh, tt_y)
        for t in (tt_x, tt_r, tt_y):
            t.deallocate(True)
        return out


def device_component(mesh, spec, layer, step):
    _observable_ops()
    if step in NORM_WEIGHTS:
        from models.demos.gpt_oss_120b.tt.norm import TtRmsNorm

        ck = _checkpoint(spec)
        w = ck.get(f"model.layers.{layer}.{NORM_WEIGHTS[step]}")
        ck.close()
        return _HostBoundary(mesh, TtRmsNorm(mesh, w, _config(spec)["rms_norm_eps"]))
    if step == "attention":
        return _AttentionBoundary(mesh, _attention(mesh, spec, layer), int(spec.get("target.chunk")))
    if step in ("attn_residual", "mlp_residual"):
        from models.demos.gpt_oss_120b.tt.residual import TtResidualAdd

        return _HostBoundary(mesh, TtResidualAdd())
    if step == "router":
        from models.demos.gpt_oss_120b.tt.router import TtRouter

        ck = _checkpoint(spec)
        w, b = (ck.get(f"model.layers.{layer}.mlp.router.{n}") for n in ("weight", "bias"))
        ck.close()
        router = TtRouter(mesh, w, b, _config(spec)["num_experts_per_tok"])
        router.prepare(int(spec.get("target.chunk")) // tuple(mesh.shape)[0])
        return _HostBoundary(mesh, router)
    if step == "experts":
        return _ExpertsBoundary(mesh, _experts(mesh, spec, layer))
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
