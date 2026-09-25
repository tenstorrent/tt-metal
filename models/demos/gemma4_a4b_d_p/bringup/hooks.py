# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for gemma4_a4b_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""


import torch


def reference(spec, layers=None, dtype=None):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import Gemma4Reference

    return Gemma4Reference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_layers(model):
    """Decoder layers of the HF oracle (Gemma4ForConditionalGeneration: model.model.language_model.layers)."""
    inner = model.model
    return inner.language_model.layers if hasattr(inner, "language_model") else inner.layers


# Device steps swapped in so far, per block type. Every other step runs on the CPU reference.
DEVICE_STEPS = {"sliding": {"attn_norm"}, "global": set()}

# Norm steps -> checkpoint weight name (under model.language_model.layers.<i>.).
_NORM_WEIGHTS = {"attn_norm": "input_layernorm.weight"}


def _norm_module(mesh, spec, layer, step, loader=None):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import PREFIX, WeightLoader
    from models.demos.gemma4_a4b_d_p.tt.rms_norm import TtRMSNorm

    loader = loader or WeightLoader(hf_path(spec))
    w = loader.get(f"{PREFIX}layers.{layer}.{_NORM_WEIGHTS[step]}")
    return TtRMSNorm(mesh, w, eps=1e-6)


def _host_fn(mesh, module):
    """Wrap a device module as fn(ctx, x_host [S, H]) -> host [S, H]."""
    import ttnn
    from models.demos.gemma4_a4b_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

    def fn(ctx, x):
        xd = to_device_replicated(mesh, x)
        yd = module(xd)
        y = replicated_to_host(yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def device_component(mesh, spec, layer, step):
    if step in _NORM_WEIGHTS:
        return _host_fn(mesh, _norm_module(mesh, spec, layer, step))
    raise NotImplementedError(f"implement step: no device module for {step} yet")


class _HybridState:
    def __init__(self, ref, max_seq):
        self.ref, self.s = ref, ref.new_state(max_seq)

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)

    def to_torch(self, layer, length):
        return self.ref.state_tensors(self.s, layer, length)


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    Hidden states stay on the host until more of the block is on the device."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import WeightLoader

        t0 = time.time()
        self.spec = spec
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = WeightLoader(self.ref.loader.model_path)
        self.overrides = {}
        for i in self.ref.layer_ids:
            bt = spec.block_type_of(i)
            self.overrides[i] = {
                step: _host_fn(mesh, _norm_module(mesh, spec, i, step, loader)) for step in DEVICE_STEPS.get(bt, ())
            }
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq)

    def embed(self, tokens):
        import torch.nn.functional as F

        return F.embedding(tokens.long(), self.ref.embed) * self.ref.embed_scale

    def from_host(self, h):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state):
        from models.demos.common.bringup.reference.interface import run_block

        ctx = self.ref.chunk_context(i, start, h.shape[0], state.s)
        return run_block(
            self.ref.block_graph(i),
            lambda n: self.ref.component(i, n),
            ctx,
            h,
            overrides=self.overrides[i],
        )

    def final_norm(self, h):
        from models.demos.gemma4_a4b_d_p.reference.gemma4_ref import rms_norm

        return rms_norm(h, self.ref.final_norm_w, self.cfg.rms_norm_eps)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
