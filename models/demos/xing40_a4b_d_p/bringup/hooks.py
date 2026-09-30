# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for xing40_a4b_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def reference(spec, layers=None, dtype=None):
    """The standalone chunked CPU reference (reference/xing_ref.py): mHC 4-stream residual, dense causal MLA with a
    latent cache, DeepSeek-style MoE. All requested layers stay resident (40 layers: 122 GB in fp32). The HF oracle
    is the stock loader (trust_remote_code); its [B, S, 4, H] layer output flattens to the reference's [S * 4, H]
    block output, so check_hf needs no hf_model / hf_layers hook."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.xing40_a4b_d_p.reference.xing_ref import XingReference

    return XingReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


# Device steps of the hybrid harness, per block type: every step passed its component gate on the device (and its
# swap gate, once run). Steps not listed run on the CPU reference.
DEVICE_STEPS = {
    "dense": {"attn_hc", "attn_collapse", "attn_norm", "q_a", "attention"},
}

# mHC coefficient steps (tt/mhc.py:TtHcWeights) -> checkpoint prefix under model.layers.<i>.
_HC_STEPS = {"attn_hc", "ffn_hc"}
# mHC collapse steps (tt/collapse.py:TtHcCollapse): (streams, hc) -> [S, H], no weights.
_COLLAPSE_STEPS = {"attn_collapse", "ffn_collapse"}
# Distributed RMSNorm steps (tt/norm.py:TtDistributedRmsNorm): column-split [S, H] in -> column-split [S, H] out.
_NORM_STEPS = {"attn_norm"}
# q_a stem (tt/q_a.py:TtQa): column-split [S, H] in -> K-split q_a_proj -> all_reduce axis 1 -> q_a_layernorm ->
# row-split q_resid [S, 768], replicated over axis 1.
_QA_STEPS = {"q_a"}
# Dense causal MLA (tt/attention.py:TtMlaAttention), stateful: owns the layer's block-cyclic device latent cache.
_ATTENTION_STEPS = {"attention"}


def _loader(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.xing40_a4b_d_p.reference.weights import WeightLoader

    return WeightLoader(hf_path(spec))


def _cfg(spec):
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.xing40_a4b_d_p.reference.xing_ref import XingConfig

    return XingConfig.from_json(os.path.join(hf_path(spec), "config.json"))


def _hc_host_fn(mesh, module, hidden):
    """fn(ctx, streams_host [S * 4, H]) -> coefficients host [S, 24] fp32 (harness boundary: streams in, out back)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import row_split_to_host, streams_to_device

    def fn(ctx, x):
        xd = streams_to_device(mesh, x, hidden)
        od = module(xd)
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _collapse_host_fn(mesh, module, hidden):
    """fn(ctx, streams_host [S * 4, H], hc_host [S, 24]) -> host [S, H] fp32 (harness boundary)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_host, row_split_to_device, streams_to_device

    def fn(ctx, x, hc):
        xd = streams_to_device(mesh, x, hidden)
        hd = row_split_to_device(mesh, hc)
        od = module(xd, hd)
        out = col_split_to_host(mesh, od).float()
        for t in (xd, hd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _norm_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, H] fp32 (harness boundary: column-split in, column-split out)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, col_split_to_host

    def fn(ctx, x):
        xd = col_split_to_device(mesh, x)
        od = module(xd)
        out = col_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _qa_host_fn(mesh, module):
    """fn(ctx, attn_norm_host [S, H]) -> q_resid host [S, 768] fp32 (harness boundary: column-split bf16 in, the
    row-split output replicated over axis 1 read back from column 0)."""
    import ttnn
    from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = col_split_to_device(mesh, x, dtype=ttnn.bfloat16)
        od = module(xd)
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


class _AttentionHostFn:
    """fn(ctx, attn_norm_host [S, H], q_resid_host [S, 768]) -> attn_out host [S, H] fp32.

    Harness boundary around TtMlaAttention, which keeps the layer's MLA latent cache on the device. With a device
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

    def __call__(self, ctx, x, q_resid):
        import ttnn
        from models.demos.xing40_a4b_d_p.tt.layout import col_split_to_device, col_split_to_host, row_split_to_device

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
        od = self.mod(xd, qd, ctx.start)
        out = col_split_to_host(self.mesh, od).float()
        for t in (xd, qd, od):
            ttnn.deallocate(t)
        return out


def _device_step_fn(mesh, spec, layer, step, loader, cfg):
    if step in _ATTENTION_STEPS:
        from models.demos.xing40_a4b_d_p.tt.attention import build_attention

        return _AttentionHostFn(mesh, build_attention(mesh, loader, cfg, layer))
    if step in _COLLAPSE_STEPS:
        from models.demos.xing40_a4b_d_p.tt.collapse import build_collapse

        return _collapse_host_fn(mesh, build_collapse(cfg), cfg.hidden_size)
    if step in _HC_STEPS:
        from models.demos.xing40_a4b_d_p.tt.mhc import build_hc

        return _hc_host_fn(mesh, build_hc(mesh, loader, cfg, layer, step), cfg.hidden_size)
    if step in _NORM_STEPS:
        from models.demos.xing40_a4b_d_p.tt.norm import build_norm

        return _norm_host_fn(mesh, build_norm(mesh, loader, cfg, layer, step))
    if step in _QA_STEPS:
        from models.demos.xing40_a4b_d_p.tt.q_a import build_q_a

        return _qa_host_fn(mesh, build_q_a(mesh, loader, cfg, layer))
    return None


def device_component(mesh, spec, layer, step):
    fn = _device_step_fn(mesh, spec, layer, step, _loader(spec), _cfg(spec))
    if fn is None:
        raise NotImplementedError(f"implement step: no device module for {step} yet")
    return fn


class _HybridState:
    """The CPU reference state, with the caches of the device's stateful steps (``_AttentionHostFn``: kv_latent) on
    the device."""

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
    Hidden states (the 4 mHC streams, [S * 4, H] fp32 token-major) stay on the host. The all-device model is the
    assemble step's."""

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
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        dev = {i: [f for f in o.values() if getattr(f, "stateful", False)] for i, o in self.overrides.items()}
        return _HybridState(self.ref, max_seq, dev)

    def embed(self, tokens):
        import torch.nn.functional as F

        e = F.embedding(tokens.long().reshape(-1), self.ref.embed)
        return e.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).reshape(-1, e.shape[-1]).contiguous()

    def from_host(self, h):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state):
        from models.demos.common.bringup.reference.interface import run_block

        ctx = self.ref.chunk_context(i, start, h.shape[0] // self.cfg.hc_mult, state.s)
        return run_block(
            self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h, overrides=self.overrides[i]
        )

    def final_norm(self, h):
        from models.demos.xing40_a4b_d_p.reference.xing_ref import rms_norm

        mean = h.view(-1, self.cfg.hc_mult, h.shape[-1]).mean(dim=1)
        return rms_norm(mean, self.ref.final_norm_w, self.cfg.rms_norm_eps)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    """The hybrid model (CPU reference with DEVICE_STEPS on the device) until the assemble step."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
