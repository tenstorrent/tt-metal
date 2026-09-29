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
    "dense_full": {"attn_hc", "attn_hc_pre"},
    "moe_full": set(),
    "moe_shared": set(),
}

# iHC gate steps -> checkpoint prefix under model.layers.<i>. (tt/ihc.py:TtHcGates).
_HC_STEPS = {"attn_hc": "hc_attn_layer"}
# iHC pre-mix steps (tt/ihc.py:TtHcPre): no weights.
_HC_PRE_STEPS = {"attn_hc_pre"}


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


def _device_step_fn(mesh, spec, layer, step, loader, cfg):
    if step in _HC_STEPS:
        return _hc_host_fn(mesh, _hc_module(mesh, spec, layer, step, loader, cfg), cfg.hidden_size)
    if step in _HC_PRE_STEPS:
        from models.demos.hy4_preview_d_p.tt.ihc import TtHcPre

        return _hc_pre_host_fn(mesh, TtHcPre(mesh, cfg.hidden_size), cfg.hidden_size)
    return None


def device_component(mesh, spec, layer, step):
    if step in _HC_STEPS or step in _HC_PRE_STEPS:
        loader = _loader(spec)
        return _device_step_fn(mesh, spec, layer, step, loader, _cfg(loader))
    raise NotImplementedError(f"implement step: no device module for {step} yet")


class _HybridState:
    """The CPU reference state (every stateful step still runs on the CPU)."""

    def __init__(self, ref, max_seq):
        self.ref, self.s = ref, ref.new_state(max_seq)

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)

    def to_torch(self, layer, length):
        return self.ref.state_tensors(self.s, layer, length)


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
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq)

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
