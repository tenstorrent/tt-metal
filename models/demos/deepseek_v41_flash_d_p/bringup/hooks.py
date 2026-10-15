# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for deepseek_v41_flash_d_p: the framework reaches the model only through these functions.

CPU side (reference role): reference, hf_model, hf_layers, state_sources.
Device side (implement role): device_params, device_component, device_model.
Contract side (contract role): contract_independent_pcc (optional).
See models/demos/common/bringup/reference/interface.py and testing/harness.py for the contracts.
"""

import torch

ORACLE_MAX_SEQ = 65536  # positions the oracle's caches hold (parity and sanity prompts are far shorter)


def _config(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config

    return Config.from_dir(hf_path(spec))


def reference(spec, layers=None, dtype=None):
    """The standalone CPU reference (reference/ds41_ref.py), the requested layers resident (fp32)."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import DS41Reference

    ref = DS41Reference(hf_path(spec), layers=layers, dtype=dtype or torch.float32)
    for i in ref.layer_ids:  # the state each layer returns is what the spec's block type declares
        assert ref.cfg.state_names(i) == spec.state_names(i), (i, ref.cfg.state_names(i), spec.state_names(i))
    return ref


def hf_model(spec, num_layers):
    """DeepSeek's inference/model.py on the checkpoint, on the CPU (reference/hf_oracle.py). num_layers=None: the
    whole text model in bf16 (the GPU configuration; intake sanity); else the first num_layers blocks in fp32."""
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.hf_oracle import build_oracle

    dtype = torch.bfloat16 if num_layers is None else torch.float32
    return build_oracle(hf_path(spec), num_layers, dtype, max_seq_len=ORACLE_MAX_SEQ)


def hf_layers(model):
    """Per-block taps: the block output packed like the reference's boundary, [S, 4 * H + 4] (streams, pending pre)."""
    return model.taps


def state_sources(spec, layer):
    """Earlier layers whose state `layer` reads: its compressed-KV source, its top-k source, layer 20's candidates."""
    return _config(spec).sources(layer)


# Device steps of the hybrid model, per block type: every step listed passed its component gate on the device.
# Steps not listed run on the CPU reference.
DEVICE_STEPS = {
    "swa": {"attn_hc", "attn_collapse", "attn_norm", "q_a"},
}

# mHC coefficient steps (tt/mhc.py:TtHcMixes): packed boundary [S, 4H + 4] (or streams [S, 4H]) in -> [S, 24].
_HC_STEPS = {"attn_hc", "ffn_hc"}


def _checkpoint(spec):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint

    return Checkpoint(hf_path(spec))


def _hc_host_fn(mesh, module, hidden):
    """fn(ctx, x_host [S, 4H + 4] or [S, 4H]) -> mixes host [S, 24] fp32 (harness boundary: the streams in, the
    row-split, TP-replicated coefficients back from mesh column 0)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import row_split_to_host, streams_to_device

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
    """fn(ctx, x_host [S, 4H + 4]) -> attn_in host [S, H] fp32 (harness boundary: the streams and the pending pre in,
    the row- and hidden-split output back)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import (
        HC,
        hidden_split_to_host,
        row_split_to_device,
        streams_to_device,
    )

    def fn(ctx, x):
        assert x.shape[-1] == HC * hidden + HC, x.shape
        xd = streams_to_device(mesh, x, hidden)
        pd = row_split_to_device(mesh, x[:, HC * hidden :])
        od = module(xd, pd)
        out = hidden_split_to_host(mesh, od).float()
        for t in (xd, pd, od):
            ttnn.deallocate(t)
        return out

    fn.module = module
    return fn


def _hidden_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, H] fp32 (harness boundary: rows split over axis 0, hidden columns over
    axis 1, in and out)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_device, hidden_split_to_host

    def fn(ctx, x):
        xd = hidden_split_to_device(mesh, x)
        od = module(xd)
        out = hidden_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _hidden_to_rows_host_fn(mesh, module):
    """fn(ctx, x_host [S, H]) -> host [S, W] fp32 (harness boundary: x rows split over axis 0 and hidden columns over
    axis 1 in; the output rows split over axis 0, replicated over axis 1, read back from mesh column 0)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_device, row_split_to_host

    def fn(ctx, x):
        xd = hidden_split_to_device(mesh, x)
        od = module(xd)
        out = row_split_to_host(mesh, od).float()
        ttnn.deallocate(xd)
        ttnn.deallocate(od)
        return out

    fn.module = module
    return fn


def _device_step_fn(mesh, spec, layer, step, ck, cfg):
    if step == "q_a":
        from models.demos.deepseek_v41_flash_d_p.tt.q_a import build_q_a

        return _hidden_to_rows_host_fn(mesh, build_q_a(mesh, ck, cfg, layer))
    if step == "attn_norm":
        from models.demos.deepseek_v41_flash_d_p.tt.norm import build_norm

        return _hidden_host_fn(mesh, build_norm(mesh, ck, cfg, layer, step))
    if step == "attn_collapse":
        from models.demos.deepseek_v41_flash_d_p.tt.collapse import build_collapse

        return _collapse_host_fn(mesh, build_collapse(cfg), cfg.dim)
    if step in _HC_STEPS:
        from models.demos.deepseek_v41_flash_d_p.tt.mhc import build_hc

        return _hc_host_fn(mesh, build_hc(mesh, ck, cfg, layer, step), cfg.dim)
    return None


def device_component(mesh, spec, layer, step):
    fn = _device_step_fn(mesh, spec, layer, step, _checkpoint(spec), _config(spec))
    if fn is None:
        raise NotImplementedError(f"implement step: no device module for {step} yet")
    return fn


class _HybridState:
    """The CPU reference state (every cache on the host until the stateful steps move to the device)."""

    def __init__(self, ref, max_seq):
        self.ref, self.s = ref, ref.new_state(max_seq)

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)

    def to_torch(self, layer, length):
        return self.ref.state_tensors(self.s, layer, length)


class HybridDeviceModel:
    """CPU reference model with the steps in DEVICE_STEPS swapped for device modules (host in / host out per step).
    The block boundary (the packed [S, 4H + 4] streams + pending pre) stays on the host. The device model until the
    assemble step builds the all-device one."""

    def __init__(self, mesh, spec, layers, lm_head=True):
        import time

        t0 = time.time()
        self.spec, self.mesh = spec, mesh
        self.ref = reference(spec, layers=layers, dtype=torch.float32)
        self.cfg = self.ref.cfg
        ck = _checkpoint(spec)
        self.overrides = {}
        for i in self.ref.layer_ids:
            steps = sorted(DEVICE_STEPS.get(spec.block_type_of(i), ()))
            self.overrides[i] = {st: _device_step_fn(mesh, spec, i, st, ck, self.cfg) for st in steps}
            missing = [st for st, f in self.overrides[i].items() if f is None]
            assert not missing, f"layer {i}: no device module for {missing}"
        self._tokens = None
        self.load_seconds = time.time() - t0

    def new_state(self, max_seq):
        return _HybridState(self.ref, max_seq)

    def begin_chunk(self, tokens, start):
        self._tokens = tokens.reshape(-1).long()

    def embed(self, tokens):
        return self.ref.first_block_input(self.ref.embed(tokens.reshape(-1).long()))

    def from_host(self, h):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state):
        from models.demos.common.bringup.reference.interface import run_block

        ctx = self.ref.chunk_context(i, start, h.shape[0], state.s)
        if self._tokens is not None:
            ctx.extra["tokens"] = self._tokens
        return run_block(
            self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h, overrides=self.overrides[i]
        )

    def final_norm(self, h):
        return self.ref.final_norm(h)

    def logits(self, hidden, rows):
        return self.ref.logits(hidden[rows])

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    """The hybrid model (CPU reference + the validated device steps); the all-device model comes with assemble."""
    return HybridDeviceModel(mesh, spec, layers, lm_head=lm_head)
