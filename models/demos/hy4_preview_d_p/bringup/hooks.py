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


# Device steps of the hybrid harness, per block type: every step passed its component gate on the device (and its
# swap gate, once run). Steps not listed run on the CPU reference.
DEVICE_STEPS = {
    "dense_full": {
        "attn_hc",
        "attn_hc_pre",
        "attn_norm",
        "q_a",
        "indexer",
        "attention",
        "attn_residual",
        "ffn_hc",
        "ffn_hc_pre",
        "ffn_norm",
        "mlp",
        "ffn_residual",
    },
    "moe_full": {"router", "experts", "shared_expert", "moe_combine"},
    "moe_shared": set(),
}

# iHC gate steps -> checkpoint prefix under model.layers.<i>. (tt/ihc.py:TtHcGates).
_HC_STEPS = {"attn_hc": "hc_attn_layer", "ffn_hc": "hc_mlp_layer"}
# iHC pre-mix steps (tt/ihc.py:TtHcPre): no weights; attn (streams "in") and ffn (streams "h_mid").
_HC_PRE_STEPS = {"attn_hc_pre", "ffn_hc_pre"}
# iHC post / residual steps (tt/ihc.py:TtHcPost): h_j = stream_j + post_j * y, no weights, no collective.
_HC_POST_STEPS = {"attn_residual", "ffn_residual"}
# Column-split distributed RMSNorm steps (tt/norm.py:TtDistributedRmsNorm) -> weight under model.layers.<i>.
_NORM_STEPS = {"attn_norm": "input_layernorm"}
# Gathered RMSNorm steps (tt/norm.py:TtGatheredRmsNorm): all_gather over axis 1 -> ttnn.bringup.rms_norm on the full
# hidden; output [S/2, H] replicated over the 2 columns of a row.
_GATHERED_NORM_STEPS = {"ffn_norm": "post_attention_layernorm"}
# q_a stem (tt/q_a.py:TtQa): K-split q_a_proj -> all_reduce over axis 1 -> q_a_layernorm (eps 1e-6).
_QA_STEPS = {"q_a"}
# DSA indexer (tt/indexer.py:TtHy4Indexer), stateful: owns the layer's device index-key cache.
_INDEXER_STEPS = {"indexer"}
# Gated sparse MLA (tt/attention.py:TtHy4Attention), stateful: owns the layer's device MLA latent cache.
_ATTENTION_STEPS = {"attention"}
# Dense SwiGLU MLP (tt/mlp.py:TtDenseMLP), layer 0: TP=2 over axis 1, fp32 intermediates, reduce_scatter over axis 1.
_MLP_STEPS = {"mlp"}
# Shared expert (tt/mlp.py:TtDenseMLP on mlp.shared_experts.*), MoE layers: intermediate 2048 -> 1024 per chip column,
# same TP=2 / fp32 intermediates / reduce_scatter over axis 1 as the dense MLP.
_SHARED_STEPS = {"shared_expert": "mlp.shared_experts."}
# MoE router (tt/router.py:TtHy4Router), replicated fp32 gate + bias, on each row's S/2 tokens; no collective.
_ROUTER_STEPS = {"router"}
# Routed experts (tt/experts.py:TtHy4Experts): DeepSeek 2D EP (dispatch over axis 0 within each column, 64 experts per
# chip, bfp8), fused ClampedSiluGlu experts at HiFi4, combine, reduce_scatter over axis 1.
_EXPERTS_STEPS = {"experts"}
# MoE combine (tt/mlp.py:TtMoeCombine): mlp_out = experts_out + shared_out, fp32 ttnn.add on the column split
# [S/2, H/2] per chip (both inputs already reduce-scattered over axis 1); no collective.
_MOE_COMBINE_STEPS = {"moe_combine"}


def _loader(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.hy4_preview_d_p.reference.weights import WeightLoader

    return WeightLoader(hf_path(spec))


def _cfg(loader):
    import os

    from models.demos.hy4_preview_d_p.reference.hy4_ref import Hy4Config

    return Hy4Config.from_json(os.path.join(loader.model_path, "config.json"))


def _hc_module(mesh, spec, layer, step, loader=None, cfg=None):
    """TtHcGates (fn split by mesh column in the chip's stream-column order, one [S/2, 32] fp32 all_reduce over
    axis 1) for one layer's iHC gate step; loads only that step's fn / base / scale."""
    from models.demos.hy4_preview_d_p.tt.ihc import TtHcGates

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    p = f"model.layers.{layer}.{_HC_STEPS[step]}.hc_pre.hc_"
    fn, base, scale = (loader.get(p + n).float() for n in ("fn", "base", "scale"))
    return TtHcGates(
        mesh,
        fn,
        base,
        scale,
        cfg.hidden_size,
        norm_eps=cfg.rms_norm_eps,
        hc_eps=cfg.hc_eps,
        magnitude=cfg.hc_magnitude,
    )


def _hc_host_fn(mesh, module, hidden):
    """fn(ctx, streams_host [S, 4H]) -> gates host [S, 8] fp32 (harness boundary: streams in, gates out)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import row_split_to_host, streams_to_device

    def fn(ctx, x):
        xd = streams_to_device(mesh, x, hidden)
        gd = module(xd)
        g = row_split_to_host(mesh, gd).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(gd)
        return g

    return fn


def _hc_pre_host_fn(mesh, module, hidden):
    """fn(ctx, streams_host [S, 4H], gates_host [S, 8]) -> sublayer input host [S, H] fp32 (harness boundary)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import col_split_to_host, row_split_to_device, streams_to_device

    def fn(ctx, x, gates):
        xd = streams_to_device(mesh, x, hidden)
        gd = row_split_to_device(mesh, gates)
        yd = module(xd, gd)
        y = col_split_to_host(mesh, yd).float()
        for t in (xd, gd, yd):
            ttnn.deallocate(t)
        return y

    return fn


def _hc_post_host_fn(mesh, module, hidden):
    """fn(ctx, streams_host [S, 4H], gates_host [S, 8], y_host [S, H]) -> new streams host [S, 4H] fp32 (harness
    boundary: streams / gates / column-split sublayer output in, streams out)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import (
        col_split_to_device,
        row_split_to_device,
        streams_to_device,
        streams_to_host,
    )

    def fn(ctx, x, gates, y):
        xd = streams_to_device(mesh, x, hidden)
        gd = row_split_to_device(mesh, gates)
        yd = col_split_to_device(mesh, y)
        hd = module(xd, gd, yd)
        h = streams_to_host(mesh, hd, hidden).float()
        for t in (xd, gd, yd, hd):
            ttnn.deallocate(t)
        return h

    return fn


def _norm_module(mesh, spec, layer, step, loader=None, cfg=None):
    """TtDistributedRmsNorm (weight split by mesh column, [S/2, 32] fp32 stats all_gather over axis 1)."""
    from models.demos.hy4_preview_d_p.tt.norm import TtDistributedRmsNorm

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    w = loader.get(f"model.layers.{layer}.{_NORM_STEPS[step]}.weight").float()
    return TtDistributedRmsNorm(mesh, w, cfg.rms_norm_eps, cluster_axis=1)


def _gathered_norm_module(mesh, spec, layer, step, loader=None, cfg=None):
    """TtGatheredRmsNorm (all_gather the column-split input over axis 1, ttnn.bringup.rms_norm with the full weight)."""
    from models.demos.hy4_preview_d_p.tt.norm import TtGatheredRmsNorm

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    w = loader.get(f"model.layers.{layer}.{_GATHERED_NORM_STEPS[step]}.weight").float()
    return TtGatheredRmsNorm(mesh, w, cfg.rms_norm_eps, cluster_axis=1)


def _col_in_row_out_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, W] fp32 for a module taking the column-split [1, 1, S/2, H/2] input and
    returning a row-split tensor replicated over axis 1 (harness boundary; column 0's copy is read back)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import col_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = col_split_to_device(mesh, x)
        yd = module(xd)
        y = row_split_to_host(mesh, yd).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y

    return fn


def _qa_module(mesh, spec, layer, loader=None, cfg=None):
    """TtQa (q_a_proj [2048, 6144] K-split over mesh columns, bf16; all_reduce over axis 1; q_a_layernorm eps 1e-6)."""
    from models.demos.hy4_preview_d_p.reference.hy4_ref import LATENT_NORM_EPS
    from models.demos.hy4_preview_d_p.tt.q_a import TtQa

    loader = loader or _loader(spec)
    p = f"model.layers.{layer}.self_attn."
    return TtQa(
        mesh,
        loader.get(p + "q_a_proj.weight").float(),
        loader.get(p + "q_a_layernorm.weight").float(),
        LATENT_NORM_EPS,
        cluster_axis=1,
    )


def _qa_host_fn(mesh, module):
    """fn(ctx, attn_norm_host [S, H]) -> q_resid host [S, 2048] fp32 (harness boundary: column-split bf16 in, the
    row-split output replicated over axis 1 read back from column 0)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import col_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = col_split_to_device(mesh, x, dtype=ttnn.bfloat16)
        yd = module(xd)
        y = row_split_to_host(mesh, yd).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y

    return fn


def _indexer_module(mesh, spec, layer, loader=None, cfg=None):
    """TtHy4Indexer: wq_b replicated (bf16), wk / weights_proj K-split over mesh columns (bf16 / fp32, 1/64 folded),
    k_norm LayerNorm eps rms_norm_eps (1e-5), RoPE on the last 64 of 128 dims, bf16 block-cyclic key cache."""
    import os

    from models.demos.hy4_preview_d_p.tt.indexer import TtHy4Indexer

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    p = f"model.layers.{layer}.self_attn.indexer."
    return TtHy4Indexer(
        mesh,
        loader.get(p + "wq_b.weight").float(),
        loader.get(p + "wk.weight").float(),
        loader.get(p + "k_norm.weight").float(),
        loader.get(p + "k_norm.bias").float(),
        loader.get(p + "weights_proj.weight").float(),
        n_heads=cfg.index_n_heads,
        head_dim=cfg.index_head_dim,
        rope_dim=cfg.qk_rope_head_dim,
        rope_theta=cfg.rope_theta,
        eps=cfg.rms_norm_eps,
        topk=cfg.index_topk,
        score_impl=os.environ.get("HY4_INDEXER_SCORE", "bringup"),  # "native": ttnn.experimental's score op
    )


class _IndexerHostFn:
    """fn(ctx, attn_norm_host [S, H], q_resid_host [S, 2048]) -> topk host [S, 2048] int64 (unsorted, -1 pads).

    Harness boundary around TtHy4Indexer. The module keeps the layer's index-key cache on the device. With a device
    ctx (component / swap tests: ``state_prefix``, ``prefix_len``, ``max_seq`` in ctx.extra) every call reloads the
    golden prefix. In the hybrid model the cache persists across chunks; the hybrid state calls ``reset`` /
    ``load_prefix`` / ``read_state``."""

    stateful = True
    state_key = "index_key"

    def __init__(self, mesh, module):
        self.mesh, self.mod = mesh, module
        self.reset()

    def reset(self):
        self._pending, self._fresh = None, True

    def load_prefix(self, index_key):
        self._pending, self._fresh = index_key, True

    def read_state(self, length):
        return self.mod.read_state(length)

    def __call__(self, ctx, x, q_resid):
        import ttnn
        from models.demos.hy4_preview_d_p.tt.layout import col_split_to_device, row_split_to_device, row_split_to_host

        if "state_prefix" in ctx.extra:
            self.mod.setup(ctx.length, ctx.extra["max_seq"])
            self.mod.load_state(ctx.extra["state_prefix"]["index_key"][: ctx.extra["prefix_len"]])
        else:
            self.mod.setup(ctx.length, ctx.state.max_seq)
            if self._fresh:
                self.mod.load_state(self._pending)
                self._fresh = False
        xd = col_split_to_device(self.mesh, x, dtype=ttnn.bfloat16)
        qd = row_split_to_device(self.mesh, q_resid, dtype=ttnn.bfloat16)
        od = self.mod(xd, qd, ctx.start)
        out = row_split_to_host(self.mesh, od).to(torch.int64) & 0xFFFFFFFF
        ttnn.deallocate(xd)
        ttnn.deallocate(qd)
        return torch.where(out == 0xFFFFFFFF, torch.full_like(out, -1), out)


def _attention_module(mesh, spec, layer, loader=None, cfg=None):
    """TtHy4Attention: q_b / linear_gate column (head) split, kv_b per head, o_proj row-parallel, kv_a K-split (all
    bf16 as stored); kv_a_layernorm eps 1e-6; sink x 16 with scale 1/16; bf16 block-cyclic latent cache."""
    from models.demos.hy4_preview_d_p.reference.hy4_ref import LATENT_NORM_EPS
    from models.demos.hy4_preview_d_p.tt.attention import TtHy4Attention

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    p = f"model.layers.{layer}.self_attn."
    w = lambda n: loader.get(p + n).float()  # noqa: E731
    return TtHy4Attention(
        mesh,
        w("q_b_proj.weight"),
        w("kv_a_proj_with_mqa.weight"),
        w("kv_a_layernorm.weight"),
        w("kv_b_proj.weight"),
        w("linear_gate.weight"),
        w("o_proj.weight"),
        w("learnable_sink_param"),
        n_heads=cfg.num_attention_heads,
        nope_dim=cfg.qk_nope_head_dim,
        rope_dim=cfg.qk_rope_head_dim,
        v_dim=cfg.v_head_dim,
        kv_lora_rank=cfg.kv_lora_rank,
        rope_theta=cfg.rope_theta,
        eps=LATENT_NORM_EPS,
        scale=cfg.scale,
    )


class _AttentionHostFn:
    """fn(ctx, attn_norm_host [S, H], q_resid_host [S, 2048], topk_host [S, 2048] int64 -1 padded) -> attn_out host
    [S, H] fp32.

    Harness boundary around TtHy4Attention, which keeps the layer's MLA latent cache on the device. With a device
    ctx (component / swap tests: ``state_prefix``, ``prefix_len``, ``max_seq`` in ctx.extra) every call reloads the
    golden kv_latent prefix. In the hybrid model the cache persists across chunks; the hybrid state calls ``reset``
    / ``load_prefix`` / ``read_state``."""

    stateful = True
    state_key = "kv_latent"

    def __init__(self, mesh, module):
        self.mesh, self.mod = mesh, module
        self.reset()

    def reset(self):
        self._pending, self._fresh = None, True

    def load_prefix(self, kv_latent):
        self._pending, self._fresh = kv_latent, True

    def read_state(self, length):
        return self.mod.read_state(length)

    def __call__(self, ctx, x, q_resid, topk):
        import ttnn
        from models.demos.hy4_preview_d_p.tt.layout import col_split_to_device, col_split_to_host, row_split_to_device

        if "state_prefix" in ctx.extra:
            self.mod.setup(ctx.length, ctx.extra["max_seq"])
            self.mod.load_state(ctx.extra["state_prefix"]["kv_latent"][: ctx.extra["prefix_len"]])
        else:
            self.mod.setup(ctx.length, ctx.state.max_seq)
            if self._fresh:
                self.mod.load_state(self._pending)
                self._fresh = False
        xd = col_split_to_device(self.mesh, x, dtype=ttnn.bfloat16)
        qd = row_split_to_device(self.mesh, q_resid, dtype=ttnn.bfloat16)
        s, k = topk.shape
        tk = torch.where(topk < 0, torch.full_like(topk, -1), topk).to(torch.int32).reshape(1, 1, s, k)
        td = ttnn.from_torch(
            tk,  # -1 -> 0xFFFFFFFF, the sparse_sdpa sentinel (a contiguous tail, as the indexer emits)
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None)),
        )
        od = self.mod(xd, qd, td, ctx.start)
        out = col_split_to_host(self.mesh, od).float()
        for t in (xd, qd, td, od):
            ttnn.deallocate(t)
        return out


def _mlp_module(mesh, spec, layer, loader=None, cfg=None, prefix="mlp."):
    """TtDenseMLP: gate / up column-parallel and down row-parallel over mesh columns (9216 of 18432 per chip for the
    dense MLP, 1024 of 2048 for the shared expert, ``prefix="mlp.shared_experts."``), bf16 as stored; HiFi4 + fp32
    dest, fp32 gate / up / h (``HY4_MLP_MID=bf16`` for comparison); unclamped."""
    import os

    import ttnn
    from models.demos.hy4_preview_d_p.tt.mlp import TtDenseMLP

    loader = loader or _loader(spec)
    p = f"model.layers.{layer}.{prefix}"
    w = lambda n: loader.get(p + n + ".weight")  # noqa: E731
    mid = ttnn.bfloat16 if os.environ.get("HY4_MLP_MID") == "bf16" else ttnn.float32
    return TtDenseMLP(mesh, w("gate_proj"), w("up_proj"), w("down_proj"), tp_axis=1, mid=mid)


def _row_in_col_out_host_fn(mesh, module, dtype=None):
    """fn(ctx, x_host [S, H]) -> host [S, W] fp32 for a module taking the row-split input replicated over axis 1
    ([1, 1, S/2, H] per chip, the TtGatheredRmsNorm layout) and returning the column split [1, 1, S/2, W/2] (harness
    boundary)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import col_split_to_host, row_split_to_device

    def fn(ctx, x):
        xd = row_split_to_device(mesh, x, dtype=dtype or ttnn.bfloat16)
        yd = module(xd)
        y = col_split_to_host(mesh, yd).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y

    return fn


def _col_split_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, H] fp32 for a module on column-split [1, 1, S/2, H/2] tensors (harness
    boundary)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import col_split_to_device, col_split_to_host

    def fn(ctx, x):
        xd = col_split_to_device(mesh, x)
        yd = module(xd)
        y = col_split_to_host(mesh, yd).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y

    return fn


def _router_module(mesh, spec, layer, loader=None, cfg=None):
    """TtHy4Router: mlp.gate.weight [256, 6144] and e_score_correction_bias [256] replicated in fp32; fp32 logits
    (HiFi4 + fp32 acc), sigmoid, + bias, ttnn.topk 8, gather / renorm, x routed_scaling_factor."""
    from models.demos.hy4_preview_d_p.tt.router import TtHy4Router

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    p = f"model.layers.{layer}.mlp.gate."
    assert cfg.norm_topk_prob, "TtHy4Router always renormalizes the top-k weights"
    return TtHy4Router(
        mesh,
        loader.get(p + "weight").float(),
        loader.get(p + "e_score_correction_bias").float(),
        top_k=cfg.num_experts_per_tok,
        route_scale=cfg.routed_scaling_factor,
    )


def _router_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H]) -> dense routing host [S, E] fp32 (harness boundary: the row-split input
    replicated over axis 1 in; (idx, wts) [S/2, 8] per chip read back from column 0 and scattered on the host)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import row_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = row_split_to_device(mesh, x, dtype=ttnn.float32)
        idx, wts = module(xd)
        ti = row_split_to_host(mesh, idx).to(torch.int64)
        tw = row_split_to_host(mesh, wts).float()
        for t in (xd, idx, wts):
            ttnn.deallocate(t)
        return torch.zeros(ti.shape[0], module.num_experts, dtype=torch.float32).scatter_(1, ti, tw)

    return fn


def _max_chunk(spec):
    """The longest chunk any ladder rung (or the target) runs: the experts' per-expert cap and dispatch sizing."""
    return max([int(r["chunk"]) for r in spec.data["ladder"]] + [int(spec.data["target"]["chunk"])])


def _experts_module(mesh, spec, layer, loader=None, cfg=None):
    """TtHy4Experts: gate_up_proj split on the host into gate (rows 0-2047) / up (rows 2048-4095), bfp8 on device,
    read one expert at a time from the checkpoint (cache under generated/hy4_preview_d_p/tt_cache/experts).
    ``HY4_EXPERTS_MODE=loop`` selects the per-expert ttnn.linear path (fp32 intermediates, ttnn.clamp) for comparison.
    """
    import os

    from models.demos.hy4_preview_d_p.tt.experts import LazyExpertWeights, TtHy4Experts

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    weights = LazyExpertWeights(
        loader, f"model.layers.{layer}.mlp.experts.", cfg.n_routed_experts, cfg.moe_intermediate_size
    )
    return TtHy4Experts(
        mesh,
        layer,
        weights,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        top_k=cfg.num_experts_per_tok,
        max_seq_len=_max_chunk(spec),
        swiglu_limit=cfg.swiglu_limit,
        mode=os.environ.get("HY4_EXPERTS_MODE", "unified"),
    )


def _experts_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H], dense_routing_host [S, E]) -> experts_out host [S, H] fp32 (harness boundary:
    the dense routing is turned back into the router's (idx, wts) [S, 8] here, both row-split over axis 0 and
    replicated over axis 1 like x; the column-split [S/2, H/2] output is read back)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import col_split_to_host, row_split_to_device

    def fn(ctx, x, routing):
        s = x.shape[0]
        tw, ti = torch.topk(routing.float(), k=module.K, dim=-1, sorted=True)
        rep = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))
        xd = row_split_to_device(mesh, x, dtype=ttnn.bfloat16)
        idd = ttnn.from_torch(
            ti.to(torch.int32).reshape(1, 1, s, module.K),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        wd = row_split_to_device(mesh, tw, dtype=ttnn.float32)
        yd = module(xd, idd, wd)
        y = col_split_to_host(mesh, yd).float()
        for t in (xd, idd, wd, yd):
            ttnn.deallocate(t)
        return y

    return fn


def _col_split2_host_fn(mesh, module):
    """fn(ctx, a_host [S, W], b_host [S, W]) -> host [S, W] fp32 for a module on two column-split [1, 1, S/2, W/2]
    fp32 tensors (harness boundary)."""
    import ttnn
    from models.demos.hy4_preview_d_p.tt.layout import col_split_to_device, col_split_to_host

    def fn(ctx, a, b):
        ad = col_split_to_device(mesh, a)
        bd = col_split_to_device(mesh, b)
        yd = module(ad, bd)
        y = col_split_to_host(mesh, yd).float()
        for t in (ad, bd, yd):
            ttnn.deallocate(t)
        return y

    return fn


def _device_step_fn(mesh, spec, layer, step, loader, cfg):
    if step in _NORM_STEPS:
        return _col_split_host_fn(mesh, _norm_module(mesh, spec, layer, step, loader, cfg))
    if step in _GATHERED_NORM_STEPS:
        return _col_in_row_out_host_fn(mesh, _gathered_norm_module(mesh, spec, layer, step, loader, cfg))
    if step in _HC_STEPS:
        return _hc_host_fn(mesh, _hc_module(mesh, spec, layer, step, loader, cfg), cfg.hidden_size)
    if step in _QA_STEPS:
        return _qa_host_fn(mesh, _qa_module(mesh, spec, layer, loader, cfg))
    if step in _INDEXER_STEPS:
        return _IndexerHostFn(mesh, _indexer_module(mesh, spec, layer, loader, cfg))
    if step in _ATTENTION_STEPS:
        return _AttentionHostFn(mesh, _attention_module(mesh, spec, layer, loader, cfg))
    if step in _MLP_STEPS:
        return _row_in_col_out_host_fn(mesh, _mlp_module(mesh, spec, layer, loader, cfg))
    if step in _SHARED_STEPS:
        return _row_in_col_out_host_fn(mesh, _mlp_module(mesh, spec, layer, loader, cfg, prefix=_SHARED_STEPS[step]))
    if step in _ROUTER_STEPS:
        return _router_host_fn(mesh, _router_module(mesh, spec, layer, loader, cfg))
    if step in _EXPERTS_STEPS:
        return _experts_host_fn(mesh, _experts_module(mesh, spec, layer, loader, cfg))
    if step in _MOE_COMBINE_STEPS:
        from models.demos.hy4_preview_d_p.tt.mlp import TtMoeCombine

        return _col_split2_host_fn(mesh, TtMoeCombine(mesh))
    if step in _HC_PRE_STEPS:
        from models.demos.hy4_preview_d_p.tt.ihc import TtHcPre

        return _hc_pre_host_fn(mesh, TtHcPre(mesh, cfg.hidden_size), cfg.hidden_size)
    if step in _HC_POST_STEPS:
        from models.demos.hy4_preview_d_p.tt.ihc import TtHcPost

        return _hc_post_host_fn(mesh, TtHcPost(mesh, cfg.hidden_size), cfg.hidden_size)
    return None


def device_component(mesh, spec, layer, step):
    if any(
        step in steps
        for steps in (
            _HC_STEPS,
            _HC_PRE_STEPS,
            _HC_POST_STEPS,
            _NORM_STEPS,
            _GATHERED_NORM_STEPS,
            _QA_STEPS,
            _INDEXER_STEPS,
            _ATTENTION_STEPS,
            _MLP_STEPS,
            _SHARED_STEPS,
            _ROUTER_STEPS,
            _EXPERTS_STEPS,
            _MOE_COMBINE_STEPS,
        )
    ):
        loader = _loader(spec)
        return _device_step_fn(mesh, spec, layer, step, loader, _cfg(loader))
    raise NotImplementedError(f"implement step: no device module for {step} yet")


class _HybridState:
    """The CPU reference state, with the caches of the device's stateful steps (``_IndexerHostFn``: index_key,
    ``_AttentionHostFn``: kv_latent) on the device."""

    def __init__(self, ref, max_seq, device_state=None):
        self.ref, self.s = ref, ref.new_state(max_seq)
        self.dev = device_state or {}  # layer -> [stateful host fns], each owning ``state_key``
        for fns in self.dev.values():
            for fn in fns:
                fn.reset()

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)
        for fn in self.dev.get(layer, ()):
            fn.load_prefix(tensors[fn.state_key][:length])

    def to_torch(self, layer, length):
        d = self.ref.state_tensors(self.s, layer, length)
        for fn in self.dev.get(layer, ()):
            d[fn.state_key] = fn.read_state(length)
        return d


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    Hidden states (the 4 iHC streams, [S, 4H] fp32) stay on the host. The all-device model is the assemble step's."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        t0 = time.time()
        self.spec, self.mesh = spec, mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = _loader(spec)
        self.overrides = {}
        for i in self.ref.layer_ids:
            steps = DEVICE_STEPS.get(spec.block_type_of(i), ())
            self.overrides[i] = {s: _device_step_fn(mesh, spec, i, s, loader, self.cfg) for s in steps}
            missing = [s for s, f in self.overrides[i].items() if f is None]
            assert not missing, f"layer {i}: no device module for {missing}"
            if "indexer" in self.overrides[i]:
                self.overrides[i]["indexer"] = self._record_topk(i, self.overrides[i]["indexer"])
        self.load_seconds = time.time() - t0

    def _record_topk(self, i, fn):
        """Keep the reference's top-k record (shared layers read the latest full layer's top-k from it)."""

        def run(ctx, x, qr):
            tk = fn(ctx, x, qr)
            self.ref._topk = {k: v for k, v in self.ref._topk.items() if k[1:] == (ctx.start, ctx.length)}
            self.ref._topk[(i, ctx.start, ctx.length)] = tk
            return tk

        run.device_fn = fn
        return run

    def new_state(self, max_seq):
        dev = {}
        for i, o in self.overrides.items():
            fns = [getattr(f, "device_fn", f) for f in o.values()]
            dev[i] = [f for f in fns if getattr(f, "stateful", False)]
        return _HybridState(self.ref, max_seq, dev)

    def embed(self, tokens):
        import torch.nn.functional as F

        return F.embedding(tokens.long(), self.ref.embed).repeat(1, self.cfg.hc_mult)  # [S, 4H]: 4 identical streams

    def from_host(self, h):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state):
        from models.demos.common.bringup.reference.interface import run_block

        ctx = self.ref.chunk_context(i, start, h.shape[0], state.s)
        return run_block(
            self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h, overrides=self.overrides[i]
        )

    def final_norm(self, h):
        from models.demos.hy4_preview_d_p.reference.hy4_ref import hc_head, rms_norm

        return rms_norm(hc_head(h, *self.ref.hc_head_w, self.cfg), self.ref.final_norm_w, self.cfg.rms_norm_eps)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    """The hybrid harness (CPU reference + DEVICE_STEPS on the device, host in / host out per step) until the
    assemble step builds the all-device model."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
