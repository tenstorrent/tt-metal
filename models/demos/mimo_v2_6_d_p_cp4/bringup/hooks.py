# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for mimo_v2_6_d_p_cp4: the same checkpoint as the prior bring-up mimo_v2_6_d_p (spec ``prior``), on another mesh.

CPU side: the prior's hooks (same reference, goldens and HF loader); change them there only for a bug.
Device side (implement role): device_component, device_model, written for this bring-up's plan.
"""

from models.demos.mimo_v2_6_d_p.bringup import hooks as _prior

reference = _prior.reference
for _name in ("tokenizer", "hf_model", "hf_layers"):
    if hasattr(_prior, _name):
        globals()[_name] = getattr(_prior, _name)


# Norm steps -> checkpoint weight name (under model.layers.<i>.).
_NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}


def _norm_module(mesh, spec, layer, step, loader=None):
    """TtRMSNorm (CP slice per chip, replicated weight, HiFi4 + fp32 acc, plain w); loads only that weight."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p_cp4.tt.rms_norm import TtRMSNorm

    loader = loader or WeightLoader(hf_path(spec))
    return TtRMSNorm(mesh, loader.get(f"model.layers.{layer}.{_NORM_WEIGHTS[step]}"), eps=1e-6)


def _cp_host_fn(mesh, module):
    """Wrap a device module as fn(ctx, x_host [S, H]) -> host [S, H]: the input is split into the 4 CP slices
    (chip c gets rows [c*S/4, (c+1)*S/4)) and the outputs are concatenated back (component/swap harness boundary)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p_cp4.tt.rms_norm import cp_to_host, to_device_cp

    def fn(ctx, x):
        xd = to_device_cp(mesh, x)
        yd = module(xd)
        y = cp_to_host(mesh, yd)
        ttnn.deallocate(xd)
        ttnn.deallocate(yd)
        return y.to(x.dtype)

    return fn


def device_component(mesh, spec, layer, step):
    if step in _NORM_WEIGHTS:
        return _cp_host_fn(mesh, _norm_module(mesh, spec, layer, step))
    raise NotImplementedError(f"implement step: no device module for {step} yet")


# Device steps of the hybrid harness, per block type: each passed its component gate on the device (CP slices).
# Steps not listed run on the CPU reference.
DEVICE_STEPS = {
    "full_dense": {"attn_norm"},
    "sliding_moe": set(),
    "full_moe": set(),
}


class _HybridState:
    """CPU reference state (no device-resident state until attention is swapped)."""

    def __init__(self, ref, max_seq):
        self.ref, self.s = ref, ref.new_state(max_seq)

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)

    def to_torch(self, layer, length):
        return self.ref.state_tensors(self.s, layer, length)


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step,
    the input split into the 4 CP slices). Hidden states stay on the host until the assemble step."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        import torch

        from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader

        t0 = time.time()
        self.spec, self.mesh = spec, mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = WeightLoader(self.ref.loader.model_path)
        self.overrides = {}
        for i in self.ref.layer_ids:
            steps = DEVICE_STEPS.get(spec.block_type_of(i), ())
            self.overrides[i] = {
                s: _cp_host_fn(mesh, _norm_module(mesh, spec, i, s, loader)) for s in steps if s in _NORM_WEIGHTS
            }
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq)

    def embed(self, tokens):
        import torch.nn.functional as F

        return F.embedding(tokens.long(), self.ref.embed)

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
        from models.demos.mimo_v2_6_d_p.reference.mimo_ref import rms_norm

        return rms_norm(h, self.ref.final_norm_w, self.cfg.layernorm_epsilon)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    """The hybrid harness (CPU reference + DEVICE_STEPS on the device) until the assemble step adds the all-device
    model."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
