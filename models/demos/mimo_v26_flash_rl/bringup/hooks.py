# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for mimo_v26_flash_rl: the framework reaches the model only through these functions.

Device side (implement stages): device_component, device_model, and optionally device_params. The CPU side
(reference, tokenizer, hf_model, hf_layers) is in reference_hooks.py, the reference stage's.
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw); the framework's lint fails it if it does.
"""

import contextlib

import torch

from models.demos.mimo_v26_flash_rl.reference.weights import Checkpoint


def _checkpoint(spec):
    return Checkpoint(spec.hf_dir, torch.bfloat16)


@contextlib.contextmanager
def _op_hooks():
    """ttnn runs its pre-operation hooks (what the framework's device-execution proof counts ops with) only outside
    fast runtime mode, the C++ config default; the component adapters run in the slow runtime so the proof sees the
    ops. Same toggle as ttnn.graph's capture; the ops and their results are unchanged."""
    import ttnn

    prev = ttnn.CONFIG.enable_fast_runtime_mode
    ttnn.CONFIG.enable_fast_runtime_mode = False
    try:
        yield
    finally:
        ttnn.CONFIG.enable_fast_runtime_mode = prev


def _norm_component(mesh, spec, layer, weight_name):
    from models.demos.mimo_v26_flash_rl.tt.mesh_io import residual_to_host, residual_to_mesh
    from models.demos.mimo_v26_flash_rl.tt.rms_norm import RMSNorm

    ck = _checkpoint(spec)
    norm = RMSNorm(mesh, ck.get(f"model.layers.{layer}.{weight_name}"), ck.config["layernorm_epsilon"])

    def fn(ctx, x):
        with _op_hooks():
            tx = residual_to_mesh(mesh, x)
            ty = norm(tx)
            tx.deallocate(True)
            out = residual_to_host(mesh, ty).to(x.dtype)
            ty.deallocate(True)
        return out

    return fn


def _residual_component(mesh):
    from models.demos.mimo_v26_flash_rl.tt.mesh_io import residual_to_host, residual_to_mesh
    from models.demos.mimo_v26_flash_rl.tt.residual import ResidualAdd

    add = ResidualAdd()

    def fn(ctx, a, b):
        with _op_hooks():
            ta, tb = residual_to_mesh(mesh, a), residual_to_mesh(mesh, b)
            ty = add(ta, tb)
            ta.deallocate(True)
            tb.deallocate(True)
            out = residual_to_host(mesh, ty).to(a.dtype)
            ty.deallocate(True)
        return out

    return fn


SLOTS = 2  # serving_answers.yaml slots


def _attention_component(mesh, spec, layer):
    from models.demos.mimo_v26_flash_rl.tt.attention import ChunkMeta, FullAttention, RopeTables
    from models.demos.mimo_v26_flash_rl.tt.ccl import RingResources
    from models.demos.mimo_v26_flash_rl.tt.mesh_io import residual_to_host, residual_to_mesh

    ck = Checkpoint(spec.hf_dir, torch.float32)
    cfg = ck.config
    if cfg["hybrid_layer_pattern"][layer] != 0:
        raise NotImplementedError(f"layer {layer} is a sliding-window layer: no device module yet")
    chunk, max_seq = spec.get("target.chunk"), spec.get("target.seq")
    rope_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])  # the reference Layer's inv_freq formula
    inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).float() / rope_dim))
    meta = ChunkMeta(mesh, chunk)
    attn = FullAttention(
        mesh,
        ck,
        layer,
        cfg,
        max_seq=max_seq,
        chunk=chunk,
        slots=SLOTS,
        ring=RingResources(mesh),
        meta=meta,
        rope=RopeTables(mesh, inv_freq, max_seq),
    )

    def fn(ctx, x):
        with _op_hooks():
            start, length = ctx.start, x.shape[0]
            assert length == chunk, f"the device attention runs whole {chunk}-token chunks, got {length}"
            pre = ctx.extra["state_prefix"]
            attn.load_prefix(0, pre["key"], pre["value"], start)
            meta.stage(0, start, start + length)
            tx = residual_to_mesh(mesh, x)
            ty = attn(tx)
            tx.deallocate(True)
            out = residual_to_host(mesh, ty).to(x.dtype)
            ty.deallocate(True)
        return out

    fn.module = attn
    return fn


def device_component(mesh, spec, layer, step):
    if step == "attn_norm":
        return _norm_component(mesh, spec, layer, "input_layernorm.weight")
    if step == "attention":
        return _attention_component(mesh, spec, layer)
    if step == "attn_residual":
        return _residual_component(mesh)
    raise NotImplementedError(f"implement stage: no device module for {step} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("assemble stage: no device model yet")
