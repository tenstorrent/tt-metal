# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for mimo_v2_6_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, and optionally tokenizer, hf_model, hf_layers.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch


def _physical_threads():
    """Use one torch thread per physical core: the gates set os.cpu_count() (SMT) threads, which makes the many small
    per-expert GEMMs and mxfp4 dequants 2-7x slower on this host."""
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
    except Exception:
        n = None
    if n and torch.get_num_threads() > n:
        torch.set_num_threads(n)


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/mimo_ref.py). Experts are dequantized at load for small layer sets
    (the 0-5 subset) and kept mxfp4, dequantized per use, for the 48-layer parity run."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.mimo_ref import MiMoReference

    _physical_threads()
    return MiMoReference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF oracle: the checkpoint's modeling_mimo_v2.py, text decoder only, fp32, eager attention, weights
    dequantized as the checkpoint defines them, routed experts kept mxfp4 and dequantized when they run
    (reference/hf_oracle.py). num_layers=None is the full 48-layer model (HF sanity gate)."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.hf_oracle import build_hf_model

    _physical_threads()
    return build_hf_model(hf_path(spec), num_layers, torch.float32)


# Device steps swapped in so far, per block type. Every other step runs on the CPU reference.
DEVICE_STEPS = {
    "full_dense": {"attn_norm"},
}

# Norm steps -> checkpoint weight name (under model.layers.<i>.).
_NORM_WEIGHTS = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
}


def _norm_module(mesh, spec, layer, step, loader=None):
    """TtRMSNorm (replicated, HiFi4 + fp32 acc, plain w) for one layer's norm step; loads only that weight."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader
    from models.demos.mimo_v2_6_d_p.tt.rms_norm import TtRMSNorm

    loader = loader or WeightLoader(hf_path(spec))
    w = loader.get(f"model.layers.{layer}.{_NORM_WEIGHTS[step]}")
    return TtRMSNorm(mesh, w, eps=1e-6)


def _host_fn(mesh, module):
    """Wrap a device module as fn(ctx, x_host [S, H]) -> host [S, H] (component/swap/hybrid harness boundary)."""
    import ttnn
    from models.demos.mimo_v2_6_d_p.tt.rms_norm import replicated_to_host, to_device_replicated

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
    """CPU reference state (no device attention yet)."""

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

        from models.demos.mimo_v2_6_d_p.reference.weights import WeightLoader

        t0 = time.time()
        self.spec = spec
        self.mesh = mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        loader = WeightLoader(self.ref.loader.model_path)
        self.overrides = {}
        for i in self.ref.layer_ids:
            steps = DEVICE_STEPS.get(spec.block_type_of(i), ())
            self.overrides[i] = {
                s: _host_fn(mesh, _norm_module(mesh, spec, i, s, loader)) for s in steps if s in _NORM_WEIGHTS
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
    """Hybrid for now (CPU reference + DEVICE_STEPS on the device); the assemble step replaces it."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
