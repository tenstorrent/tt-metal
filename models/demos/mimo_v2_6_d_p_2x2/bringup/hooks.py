# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for mimo_v2_6_d_p_2x2: the same checkpoint as the prior bring-up mimo_v2_6_d_p (spec ``prior``), on another mesh.

CPU side: the prior's hooks (same reference, goldens and HF loader); change them there only for a bug.
Device side (implement role): device_component, device_model, written for this bring-up's plan (tt/ of this model only).
"""

import torch

from models.demos.mimo_v2_6_d_p.bringup import hooks as _prior

reference = _prior.reference
for _name in ("tokenizer", "hf_model", "hf_layers"):
    if hasattr(_prior, _name):
        globals()[_name] = getattr(_prior, _name)


# Device steps of the hybrid harness, per block type: every step passed its component gate on the device (and its
# swap gate, once run). Steps not listed run on the CPU reference.
DEVICE_STEPS = {
    "full_dense": {"attn_norm", "attention", "attn_residual", "mlp"},
    "sliding_moe": {"attention", "router", "experts"},
    "full_moe": set(),
}

# Norm steps -> checkpoint weight name (under model.layers.<i>.); tt/model.py:NORM_WEIGHTS.
_NORM_STEPS = {"attn_norm", "ffn_norm"}

# Residual steps (replicated bf16 add, no collective): h_mid = in + attn_out; out = h_mid + mlp_out (dense)
# or out = h_mid + experts_out (MoE, ffn_residual).
_RESIDUAL_STEPS = {"attn_residual", "mlp_residual", "ffn_residual"}


def _loader(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader

    return WeightLoader(hf_path(spec))


def _norm_module(mesh, spec, layer, step, loader=None):
    """TtRMSNorm (replicated on 2x2, HiFi4 + fp32 acc, plain w) for one layer's norm step; loads only that weight."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.model import build_norm

    return build_norm(mesh, loader or _loader(spec), layer, step, eps=1e-6)


def _host_fn(mesh, module):
    """Wrap a device module as fn(ctx, x_host [S, H]) -> host [S, H] (component/swap/hybrid harness boundary)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        yd = module(xd)
        y = replicated_to_host(yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def _residual_host_fn(mesh):
    """fn(ctx, a_host [S, H], b_host [S, H]) -> host [S, H] via TtResidualAdd (a + b on the device)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_2x2.tt.residual import TtResidualAdd
    from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import replicated_to_host, to_device_replicated

    module = TtResidualAdd(mesh)

    def fn(ctx, a, b):
        ad = to_device_replicated(mesh, a)
        bd = to_device_replicated(mesh, b)
        yd = module(ad, bd)
        y = replicated_to_host(yd)
        for t in (ad, bd, yd):
            ttnn.deallocate(t)
        return y.to(a.dtype)

    return fn


def _mlp_module(mesh, spec, layer, loader=None):
    """TtDenseMLP (TP=4 SwiGLU over the 2x2 mesh, all_reduce axis 1 then axis 0) for the dense layer; fp8 + 128x128
    block scale dequantized to bf16 at load."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.model import build_mlp

    return build_mlp(mesh, loader or _loader(spec), layer)


def _rope_max_seq(spec):
    """Longest sequence any rung or the target runs: the RoPE tables are built once for it at load."""
    seqs = [int(r.get("seq", 0)) for r in spec.data.get("ladder", [])]
    seqs.append(int((spec.data.get("target") or {}).get("seq", 0)))
    return max(seqs)


def _cfg(loader):
    import os

    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoConfig

    return MiMoConfig.from_json(os.path.join(loader.model_path, "config.json"))


def _attention_module(mesh, spec, layer, loader=None, cfg=None):
    """TtFullAttention (full layers) or TtSlidingAttention (sliding layers: window, per-head sink), TP=4 over the 2x2
    mesh, all_reduce axis 1 then axis 0; loads only its attention weights (fused qkv dequantized per TP rank, bf16
    o_proj, bf16 sink bias)."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.model import build_attention

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    return build_attention(mesh, loader, cfg, layer, _rope_max_seq(spec)), cfg


def _max_chunk(spec):
    """Largest chunk any rung or the target runs: the router's bias / zero tables are built once for it at load."""
    chunks = [int(r.get("chunk", 0)) for r in spec.data.get("ladder", [])]
    chunks.append(int((spec.data.get("target") or {}).get("chunk", 0)))
    return max(chunks)


def _router_module(mesh, spec, layer, loader=None, cfg=None):
    """TtRouter (replicated on 2x2, fp32 logits, fp32 sigmoid + bias choice, ttnn.topk, no CCL) for one MoE layer.
    MIMO_ROUTER_MODE=fused selects moe_grouped_topk (TF32 keys) for comparison."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.model import build_router

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    return build_router(mesh, loader, cfg, layer, _max_chunk(spec))


def _router_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> dense routing host [S, E] (chip 0's copy of the replicated result)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        dense, idx, wts = module(xd)
        y = replicated_to_host(dense)
        for t in (xd, dense, idx, wts):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def _experts_module(mesh, spec, layer, loader=None, cfg=None):
    """TtExperts (2x2 DeepSeek 2D EP: row-half split, 2-chip fabric dispatch per column, unified_routed_expert_moe
    high_precision at HiFi4, combine, all_reduce axis 1, all_gather axis 0) for one MoE layer."""
    from models.demos.mimo_v2_6_d_p_2x2.tt.model import build_experts

    loader = loader or _loader(spec)
    cfg = cfg or _cfg(loader)
    return build_experts(mesh, loader, cfg, layer, _max_chunk(spec))


def _experts_host_fn(mesh, module):
    """fn(ctx, ffn_norm_host [S, H], dense_routing_host [S, E]) -> experts_out host [S, H]."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x, r):
        xd = to_device_replicated(mesh, x)
        rd = to_device_replicated(mesh, r)
        yd = module(xd, dense=rd)
        y = replicated_to_host(yd)
        for t in (xd, rd, yd):
            ttnn.deallocate(t)
        return y.to(x.dtype)

    return fn


def _new_kv_cache(mesh, cfg, layer, max_seq):
    from models.demos.mimo_v2_6_d_p_2x2.tt.model import new_kv_cache

    return new_kv_cache(mesh, cfg, layer, max_seq)


def _attention_host_fn(mesh, module, cache_of):
    """fn(ctx, x_host [S, H]) -> host [S, H]; cache_of(ctx) returns the layer's device KV cache."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        yd = module(xd, ctx.start, cache_of(ctx))
        y = replicated_to_host(yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def device_component(mesh, spec, layer, step):
    if step in _NORM_STEPS:
        return _host_fn(mesh, _norm_module(mesh, spec, layer, step))
    if step in _RESIDUAL_STEPS:
        return _residual_host_fn(mesh)
    if step == "attention":
        module, cfg = _attention_module(mesh, spec, layer)
        caches = {}

        def cache_of(ctx):
            # Component/swap tests: a fresh device cache holding the golden prefix [0, prefix_len).
            ex = ctx.extra or {}
            max_seq = int(ex.get("max_seq", ctx.start + ctx.length))
            if "c" in caches:
                caches.pop("c").free()
            c = _new_kv_cache(mesh, cfg, layer, max_seq)
            n = int(ex.get("prefix_len", ctx.start))
            if n:
                c.load_prefix(ex["state_prefix"]["key"], ex["state_prefix"]["value"], n)
            caches["c"] = c
            return c

        return _attention_host_fn(mesh, module, cache_of)
    if step == "mlp":
        return _host_fn(mesh, _mlp_module(mesh, spec, layer))
    if step == "router":
        return _router_host_fn(mesh, _router_module(mesh, spec, layer))
    if step == "experts":
        return _experts_host_fn(mesh, _experts_module(mesh, spec, layer))
    raise NotImplementedError(f"implement step: no device module for {step} yet")


class _HybridState:
    """CPU reference state, except layers whose attention runs on the device: their K/V live in a device cache."""

    def __init__(self, ref, max_seq, mesh=None, device_attn_layers=(), cfg=None):
        self.ref, self.s = ref, ref.new_state(max_seq)
        self.dev = {i: _new_kv_cache(mesh, cfg, i, max_seq) for i in device_attn_layers}

    def load_prefix(self, layer, tensors, length):
        if layer in self.dev:
            self.dev[layer].load_prefix(tensors["key"], tensors["value"], length)
        else:
            self.ref.load_state(self.s, layer, tensors, length)

    def to_torch(self, layer, length):
        if layer in self.dev:
            return self.dev[layer].to_torch(length)
        return self.ref.state_tensors(self.s, layer, length)


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    Hidden states stay on the host (BRINGUP_HYBRID=1, debugging)."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader

        t0 = time.time()
        self.spec, self.mesh = spec, mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = WeightLoader(self.ref.loader.model_path)
        self.overrides = {}
        self.attn_layers = []
        for i in self.ref.layer_ids:
            steps = DEVICE_STEPS.get(spec.block_type_of(i), ())
            ov = {s: _host_fn(mesh, _norm_module(mesh, spec, i, s, loader)) for s in steps if s in _NORM_STEPS}
            ov.update({s: _residual_host_fn(mesh) for s in steps if s in _RESIDUAL_STEPS})
            if "attention" in steps:
                module, _ = _attention_module(mesh, spec, i, loader, self.cfg)
                ov["attention"] = _attention_host_fn(mesh, module, lambda ctx: ctx.extra["dev_cache"])
                self.attn_layers.append(i)
            if "mlp" in steps:
                ov["mlp"] = _host_fn(mesh, _mlp_module(mesh, spec, i, loader))
            if "router" in steps:
                ov["router"] = _router_host_fn(mesh, _router_module(mesh, spec, i, loader, self.cfg))
            if "experts" in steps:
                ov["experts"] = _experts_host_fn(mesh, _experts_module(mesh, spec, i, loader, self.cfg))
            self.overrides[i] = ov
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq, self.mesh, self.attn_layers, self.cfg)

    def embed(self, tokens):
        import torch.nn.functional as F

        return F.embedding(tokens.long(), self.ref.embed)

    def from_host(self, h):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state):
        from models.demos.common.bringup.reference.interface import Ctx, run_block

        ctx = self.ref.chunk_context(i, start, h.shape[0], state.s)
        if i in state.dev:
            ctx = Ctx(ctx.layer, ctx.start, ctx.length, ctx.state, {**ctx.extra, "dev_cache": state.dev[i]})
        return run_block(
            self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h, overrides=self.overrides[i]
        )

    def final_norm(self, h):
        from models.demos.mimo_v2_6_d_p.reference.mimo_ref import rms_norm

        return rms_norm(h, self.ref.final_norm_w, self.cfg.layernorm_epsilon)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


class _DeviceState:
    """Per-layer device KV caches (tt/model.py:new_attention_caches)."""

    def __init__(self, mesh, cfg, layers, max_seq):
        from models.demos.mimo_v2_6_d_p_2x2.tt.model import new_attention_caches

        self.caches = new_attention_caches(mesh, cfg, layers, max_seq)

    def load_prefix(self, layer, tensors, length):
        self.caches[layer].load_prefix(tensors["key"], tensors["value"], length)

    def to_torch(self, layer, length):
        return self.caches[layer].to_torch(length)


class MiMoDeviceModel:
    """Ladder/profile adapter over tt/model.py:TtMiMoModel (2x2).

    The hidden state is a replicated [1, 1, S, H] bf16 device tensor from the embedding to the final norm. Each layer
    is TtMiMoBlock.__call__: run_block over the reference block graph with the validated device modules (one profiler
    section per step). RoPE tables, page tables and router / dispatch tables are built once at load and sliced on the
    device. Only the LM head runs on the host (ladder logits on sampled rows, when the stack ends at the last layer)."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        from models.demos.common.bringup.reference.golden import hf_path
        from models.demos.mimo_v2_6_d_p_2x2.tt.model import TtMiMoModel

        t0 = time.time()
        self.mesh, self.spec = mesh, spec
        self.path = hf_path(spec)
        self.model = TtMiMoModel(
            mesh, self.path, max_seq=_rope_max_seq(spec), max_chunk=_max_chunk(spec), layers=list(layers)
        )
        self.cfg = self.model.cfg
        self.blocks = {b.i: b for b in self.model.blocks}
        self._lm_head = None
        if lm_head:
            from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader

            self._lm_head = WeightLoader(self.path).get("lm_head.weight").float()  # untied
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _DeviceState(self.mesh, self.cfg, list(self.blocks), max_seq)

    def embed(self, tokens):
        import ttnn

        ids = ttnn.from_torch(
            tokens.reshape(1, 1, -1).to(torch.int64).to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        h = self.model.embed(ids)
        ttnn.deallocate(ids)
        return h

    def from_host(self, h):
        from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import to_device_replicated

        return to_device_replicated(self.mesh, h)

    def to_host(self, h):
        from models.demos.mimo_v2_6_d_p_2x2.tt.rms_norm import replicated_to_host

        return replicated_to_host(h).float()

    def layer(self, i, h, start, state):
        return self.blocks[i](h, start, state.caches[i])

    def final_norm(self, h):
        return self.model.final_norm(h)

    def logits(self, hidden, rows):
        return torch.nn.functional.linear(self.to_host(hidden)[rows], self._lm_head)

    def free(self, h):
        import ttnn

        if isinstance(h, ttnn.Tensor) and h.is_allocated():
            ttnn.deallocate(h)

    def sync(self):
        import ttnn

        ttnn.synchronize_device(self.mesh)

    def perf_settings(self):
        """Recorded in the profile: the active SDPA presets and the experts / router modes."""
        import os

        from models.demos.mimo_v2_6_d_p_2x2.tt.attention import sdpa_settings, v_pad_enabled

        full, sl = sdpa_settings(False), sdpa_settings(True)
        return {
            "sdpa_full_cfg": full["name"],
            "sdpa_sliding_cfg": sl["name"],
            "sdpa_full_chunks": list(full["chunks"]),
            "sdpa_sliding_chunks": list(sl["chunks"]),
            "experts_mode": os.environ.get("MIMO_EXPERTS_MODE", "unified"),
            "router_mode": os.environ.get("MIMO_ROUTER_MODE", "fp32"),
            "fuse_residual_norm": os.environ.get("MIMO_FUSE_RESIDUAL_NORM", "1") != "0",
            "attn_v_pad": v_pad_enabled(),
        }


def device_model(mesh, spec, layers, lm_head=True):
    """All-device model (default); BRINGUP_HYBRID=1 selects the hybrid harness (CPU reference + DEVICE_STEPS on the
    device, host in / host out per step) for debugging."""
    import os

    if os.environ.get("BRINGUP_HYBRID") == "1":
        return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
    return MiMoDeviceModel(mesh, spec, layers, lm_head=lm_head)
