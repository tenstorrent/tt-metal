# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for glm53_flash_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/glm_ref.py), the requested layers resident."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.glm53_flash_d_p.reference.glm_ref import GlmReference

    return GlmReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF glm5_next code (vendored) on the dequantized checkpoint, text model only, routed experts read from
    disk per use. num_layers=None: the whole model in bf16 (intake sanity); else the first num_layers in fp32."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.glm53_flash_d_p.reference.hf_oracle import build_hf_model

    return build_hf_model(hf_path(spec), num_layers, torch.bfloat16 if num_layers is None else torch.float32)


# Device steps of the hybrid model, per block type: each passed its component gate on the device. Every other step
# runs on the CPU reference.
DEVICE_STEPS = {
    "kda_dense": {"attn_hc", "attn_collapse"},
    "dsa_moe": set(),
    "kda_moe": set(),
}

_HC_STEPS = {"attn_hc": "attn", "ffn_hc": "ffn"}
_COLLAPSE_STEPS = {"attn_collapse", "ffn_collapse"}


def _loader_cfg(spec):
    import os

    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig
    from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

    path = hf_path(spec)
    return WeightLoader(path), GlmConfig.from_json(os.path.join(path, "config.json"))


def _hc_host_fn(mesh, module, n):
    """fn(ctx, x_host [S * n, H]) -> host [S, (2 + n) * n] fp32 (harness boundary: upload bf16, read chip 0)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, x):
        s = x.shape[0] // n
        xd = replicate(mesh, x.reshape(1, 1, s, n * x.shape[-1]).to(torch.bfloat16))
        yd = module(xd)
        y = replicated_to_host(yd).reshape(s, -1).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y

    return fn


def _collapse_host_fn(mesh, module, n):
    """fn(ctx, x_host [S * n, H], hc_host [S, K]) -> host [S, H] (harness boundary: x bf16, hc fp32, read chip 0)."""
    import ttnn
    from models.demos.glm53_flash_d_p.tt.common import replicate, replicated_to_host

    def fn(ctx, x, hc):
        s = x.shape[0] // n
        xd = replicate(mesh, x.reshape(1, 1, s, n * x.shape[-1]).to(torch.bfloat16))
        hd = replicate(mesh, hc.reshape(1, 1, s, hc.shape[-1]).float(), dtype=ttnn.float32)
        yd = module(xd, hd)
        y = replicated_to_host(yd).reshape(s, -1)
        for t in (xd, hd, yd):
            ttnn.deallocate(t)
        return y

    return fn


def _device_step(mesh, spec, layer, step, loader, cfg):
    if step in _HC_STEPS:
        from models.demos.glm53_flash_d_p.tt.mhc import build_hc

        return _hc_host_fn(mesh, build_hc(mesh, loader, cfg, layer, _HC_STEPS[step]), cfg.hc_mult)
    if step in _COLLAPSE_STEPS:
        from models.demos.glm53_flash_d_p.tt.collapse import build_collapse

        return _collapse_host_fn(mesh, build_collapse(cfg), cfg.hc_mult)
    raise NotImplementedError(f"implement step: no device module for {step} yet")


def device_component(mesh, spec, layer, step):
    loader, cfg = _loader_cfg(spec)
    return _device_step(mesh, spec, layer, step, loader, cfg)


class _RefState:
    """CPU reference state (no device-resident state yet)."""

    def __init__(self, ref, max_seq):
        self.ref, self.s = ref, ref.new_state(max_seq)

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)

    def to_torch(self, layer, length):
        return self.ref.state_tensors(self.s, layer, length)


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    Hidden states stay on the host until the assemble step builds the all-device model."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        t0 = time.time()
        self.mesh, self.spec = mesh, spec
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader, cfg = _loader_cfg(spec)
        self.overrides = {
            i: {s: _device_step(mesh, spec, i, s, loader, cfg) for s in DEVICE_STEPS.get(spec.block_type_of(i), ())}
            for i in self.ref.layer_ids
        }
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _RefState(self.ref, max_seq)

    def embed(self, tokens):
        import torch.nn.functional as F

        e = F.embedding(tokens.long(), self.ref.embed)
        return e.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).reshape(-1, e.shape[-1])

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
        from models.demos.glm53_flash_d_p.reference.glm_ref import rms_norm

        n = self.cfg.hc_mult
        return rms_norm(h.view(-1, n, h.shape[-1]).mean(dim=1), self.ref.final_norm_w, self.cfg.rms_norm_eps)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    """Hybrid model (CPU reference + DEVICE_STEPS on the device) until the assemble step."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
