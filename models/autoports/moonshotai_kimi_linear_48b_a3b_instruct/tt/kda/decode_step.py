# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-token Kimi Delta Attention recurrence with PER-CHANNEL decay.

Forked from models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py::recurrent_gated_delta_rule_decode_ttnn
(Gated DeltaNet, per-head scalar decay). KDA's log-decay ``g`` has shape [B,1,H,K]; the state decay is therefore a
row-broadcast multiply ``S[b,h,k,:] *= exp(g[b,h,k])`` instead of a scalar per head. Everything else (L2 norm, scale,
delta rule write, read-out) is identical to the Gated DeltaNet step and matches transformers'
``recurrent_kimi_delta_attention`` / fla ``naive_recurrent_kda``:
    q,k <- l2norm; q *= K^-0.5; S <- S*exp(g)[...,None]; delta = beta*(v - k^T S); S += k (x) delta; o = q^T S.
"""

from __future__ import annotations

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_read_query_program_config,
    l2_norm_ttnn,
)

_L1 = ttnn.L1_MEMORY_CONFIG


def recurrent_kda_decode_step(
    q, k, v, beta, g, state, *, scale=None, device=None, l2_weight_q=None, l2_weight_k=None, state_out=None
):
    """Pre-shaped fast path (what the decode layer and the decode trace use).

    Args: q, k [B,H,1,K] bf16/fp32; v [B,H,1,V]; beta [B,H,1,1] fp32 (sigmoid already applied); g [B,H,K,1] fp32 (log decay
    <= 0); state [B,H,K,V] fp32 TILE (read only). Returns (o [B,H,1,V] fp32, new_state [B,H,K,V] fp32).
    ``l2_weight_q`` / ``l2_weight_k`` (fp32 [1,1,1,K], constant K^-0.5*scale resp. K^-0.5) fold the L2 normalisation and the
    query scale into one ``rms_norm`` each; ``state_out`` (a [B,H,K,V] fp32 DRAM tensor, may alias ``state``) receives the new
    state in place, which removes the separate copy-back. Same math as ``recurrent_kda_decode_ttnn`` with the [B,1,H,*] <->
    [B,H,*] reshapes, redundant fp32 typecasts, no-op ``to_memory_config`` calls and two full passes over the 16 MB state
    (beta applied to the [B,H,1,V] delta instead of the [B,H,K,V] outer product; decay read straight from DRAM) removed.
    """
    B, H, _, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    f32 = ttnn.float32
    if q.dtype != f32:
        q = ttnn.typecast(q, f32)
    if k.dtype != f32:
        k = ttnn.typecast(k, f32)
    if v.dtype != f32:
        v = ttnn.typecast(v, f32)
    if beta.dtype != f32:
        beta = ttnn.typecast(beta, f32)
    if g.dtype != f32:
        g = ttnn.typecast(g, f32)

    # l2 norm = rms_norm(eps/K) * K^-0.5 (models/experimental/.../l2_norm_ttnn); the constant (and the query scale) ride
    # along as the norm's weight when the caller provides it
    if l2_weight_q is not None:
        q = ttnn.rms_norm(q, epsilon=1e-6 / K, weight=l2_weight_q, memory_config=_L1)
    else:
        q = ttnn.multiply(l2_norm_ttnn(q, dim=-1), scale, memory_config=_L1)
    if l2_weight_k is not None:
        k = ttnn.rms_norm(k, epsilon=1e-6 / K, weight=l2_weight_k, memory_config=_L1)
    else:
        k = l2_norm_ttnn(k, dim=-1)

    h = state if state.dtype == f32 else ttnn.typecast(state, f32)
    # per-channel decay: rows of the [K,V] state decay independently (exp fused into the multiply); reads the DRAM state
    # directly and lands the decayed copy in L1
    h = ttnn.multiply(h, g, input_tensor_b_activations=[ttnn.UnaryOpType.EXP], memory_config=_L1)

    arch = device.arch() if device is not None else h.device().arch()
    read_cfg = ttnn.init_device_compute_kernel_config(
        arch, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    # outer product k (x) (beta*delta): identical config to models/experimental/.../fused_decay_and_write_ttnn
    write_cfg = ttnn.init_device_compute_kernel_config(
        arch, math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    prog_cfg = None
    if device is not None:
        try:
            prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            prog_cfg = None

    v_read = ttnn.matmul(k, h, memory_config=_L1, program_config=prog_cfg, compute_kernel_config=read_cfg)  # [B,H,1,V]
    delta = ttnn.subtract(v, v_read, memory_config=_L1)  # [B,H,1,V]
    ttnn.deallocate(v_read)
    delta = ttnn.multiply(
        delta, beta, memory_config=_L1
    )  # beta is a per-(b,h) scalar: apply it to delta, not to k (x) delta
    k_col = ttnn.reshape(k, [B, H, K, 1], memory_config=_L1)
    outer = ttnn.matmul(k_col, delta, memory_config=_L1, compute_kernel_config=write_cfg)  # [B,H,K,V]
    ttnn.deallocate(k_col)
    ttnn.deallocate(delta)
    if state_out is not None:
        h_new = ttnn.add(
            h, outer, output_tensor=state_out
        )  # in place into the persistent state (address-stable for traces)
        ttnn.deallocate(h)
    else:
        h_new = ttnn.add(h, outer, memory_config=_L1)
    ttnn.deallocate(outer)
    o = ttnn.matmul(q, h_new, memory_config=_L1, program_config=prog_cfg, compute_kernel_config=read_cfg)  # [B,H,1,V]
    return o, h_new


def recurrent_kda_decode_ttnn(q, k, v, beta, g, state, *, scale=None, device=None):
    """Generic layout (unit tests / callers with [B,1,H,*] tensors); reshapes into ``recurrent_kda_decode_step``.

    Args: q,k [B,1,H,K]; v [B,1,H,V]; beta [B,1,H]; g [B,1,H,K] (log decay, <= 0); state [B,H,K,V] fp32 TILE.
    Returns (o [B,1,H,V] fp32, new_state [B,H,K,V] fp32). ``state`` is read only."""
    B, _, H, K = q.shape
    V = v.shape[-1]
    o, h = recurrent_kda_decode_step(
        ttnn.reshape(q, [B, H, 1, K]),
        ttnn.reshape(k, [B, H, 1, K]),
        ttnn.reshape(v, [B, H, 1, V]),
        ttnn.reshape(beta, [B, H, 1, 1]),
        ttnn.reshape(g, [B, H, K, 1]),
        state,
        scale=scale,
        device=device,
    )
    return ttnn.reshape(o, [B, 1, H, V], memory_config=_L1), h
