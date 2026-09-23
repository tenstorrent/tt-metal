# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the Gated DeltaNet layer.
"""

import os

import torch

import ttnn

from models.demos.blackhole.qwen36.tt import tp_common as tpc

from .ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_ttnn,
    recurrent_gated_delta_rule_decode_ttnn,
    recurrent_gated_delta_rule_decode_inplace_ttnn,
)
from .ttnn_delta_rule_seq import chunk_gated_delta_rule_seq_adapter

_L1_SEQ_THRESHOLD = 512


def _l1_seq_threshold():
    """F10A (G2): QWEN36_GDN_L1_MAX_T can raise the chunk-prefill L1 threshold from the legacy 512
    up to the T=2048 production chunk size. Measured (step1 F10A): at 2048 this puts every chunk-
    path `mc`-gated tensor (split qkv/gab projection outputs, conv output/q,k,v slices, beta/g,
    GDN output post-processing, out-proj -- see gated_deltanet_forward_ttnn) in L1 at once, which
    overflows L1 on its own (independent of the other F10A groups) -- "Statically allocated
    circular buffers ... clash with L1 buffers" inside the native conv1d call. So this defaults to
    the legacy 512 (unset AND "0" both mean legacy); set it explicitly (e.g. =2048) only to
    reproduce/investigate the clash or after a finer-grained (per-tensor, not blanket) L1 split."""
    val = int(os.environ.get("QWEN36_GDN_L1_MAX_T", "0"))
    return _L1_SEQ_THRESHOLD if val == 0 else val


def _seq_memory_config(seq_len):
    """L1 for short sequences (faster), DRAM for long (avoids OOM)."""
    return ttnn.L1_MEMORY_CONFIG if seq_len <= _l1_seq_threshold() else None


def _gdn_post_l1_groups():
    """F10B item B (QWEN36_GDN_POST_L1, default "1"): per-tensor L1 policy for the chunk-prefill
    path at T <= 2048, INSTEAD of the blanket _seq_memory_config threshold above (which overflows
    L1 at T=2048 -- the [2048,6144]-ish qkv/gab projection outputs and conv slices don't fit
    alongside everything else; see _l1_seq_threshold's docstring). Only three small/cheap groups
    move to L1 here; the qkv/gab projection outputs, the conv path (governed separately by
    QWEN36_GDN_CONV_XIN_L1_MAX_T, conv1d_native.py) and the ChunkGdnPrep/Scan op's own I/O are
    untouched -- see gated_deltanet_forward_ttnn's mc_small/mc_scan/mc_outproj uses.

    Sub-overrides (only meaningful with the master flag on) reproduce a partial fit without a code
    change, per the measured fallback order: drop the post-scan group ([1,16,2048,128] typecast +
    per-head rms_norm + nlp_concat_heads + fused gate multiply, ~8 MB) first if the full combination
    clashes, then the out-proj group, keeping only the small beta/g/a/b chain in L1.
      QWEN36_GDN_POST_L1=0          -- all three groups off (mc_* fall back to mc, pre-F10B).
      QWEN36_GDN_POST_L1_SCAN=0     -- drop the post-scan (typecast/norm/concat/gate) group only.
      QWEN36_GDN_POST_L1_OUTPROJ=0  -- drop the out-proj linear output only.
    (the small beta/g/a/b chain has no sub-override; it never needed to be dropped in testing.)
    """
    master = os.environ.get("QWEN36_GDN_POST_L1", "1") != "0"
    if not master:
        return False, False, False
    post_scan = os.environ.get("QWEN36_GDN_POST_L1_SCAN", "1") != "0"
    out_proj = os.environ.get("QWEN36_GDN_POST_L1_OUTPROJ", "1") != "0"
    return True, post_scan, out_proj


def _pad_rows(t, extra):
    """Zero-pad a [B, T, ...] TILE tensor with `extra` rows on the T (dim 1) axis."""
    pad = [(0, 0)] * len(t.shape)
    pad[1] = (0, extra)
    return ttnn.pad(t, pad, value=0.0)


# F6-B (QWEN36_GDN_SPLIT_PROJ): split the mega in-projection [qkv|g|a|b] matmul into two
# ([qkv] and [g|a|b]) so the gate/a/b slices come off a much narrower ~2080-wide tensor instead
# of the full ~8224-wide mega_out. There is no plumbing from gdn/weights.py's precompute helpers
# through gdn/decode.py into this function for a dedicated `mega_w_qkv`/`mega_w_gab` weight pair
# (decode.py enumerates every gated_deltanet_forward_ttnn kwarg explicitly and is out of scope for
# this change), so the two split weights are instead derived on-device from the existing
# `mega_fused_weight` tensor the first time each layer's weight is seen, then cached by identity —
# both slices are tile-aligned (mega_qkv_dim and mega_g_dim are tile-width multiples) so this is a
# cheap one-time ttnn.slice per layer, not a per-call cost. Values are bit-identical to slicing
# mega_fused_weight fresh every call; only the (one-time) op count differs.
_mega_split_cache = {}

# step2 (2026-09-22): the qkv minimal_matmul's `config=` now comes from
# tpc.prefill_minimal_matmul_config() (QWEN36_PREFILL_MINIMAL_CFG / QWEN36_PREFILL_MM_FP32_ACC in
# tp_common.py) instead of this fixed None. `_mm_grid_cache` below caches the per-device
# ttnn.CoreCoord that call needs, the same grid mlp.py's `self._mm_grid` already captures.
_mm_grid_cache = {}


def _get_mm_grid(device):
    """Per-device ttnn.CoreCoord from compute_with_storage_grid_size(), cached by device identity
    (mirrors _get_mega_split_weights' id()-keyed cache below -- `device` is a fixed object for the
    process lifetime)."""
    key = id(device)
    grid = _mm_grid_cache.get(key)
    if grid is None:
        g = device.compute_with_storage_grid_size()
        grid = ttnn.CoreCoord(g.x, g.y)
        _mm_grid_cache[key] = grid
    return grid


def _get_mega_split_weights(mega_fused_weight, mega_qkv_dim, mega_g_dim, mega_a_dim, mega_b_dim):
    """Derive (w_qkv, w_gab) from the mega-fused [qkv|g|a|b] weight, cached by tensor identity.

    mega_fused_weight is a fixed per-layer device weight (never reallocated for the process
    lifetime), so caching by id() is safe and keeps this to one slice pair per layer.
    """
    key = id(mega_fused_weight)
    cached = _mega_split_cache.get(key)
    if cached is not None:
        return cached
    gab_dim = mega_g_dim + mega_a_dim + mega_b_dim
    w_qkv = mega_fused_weight[:, :mega_qkv_dim]
    w_gab = mega_fused_weight[:, mega_qkv_dim : mega_qkv_dim + gab_dim]
    _mega_split_cache[key] = (w_qkv, w_gab)
    return w_qkv, w_gab


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-5, memory_config=None):
    """RMSNorm + SiLU gate (trace-compatible). Clips gate to avoid overflow at long T."""
    mc = memory_config
    x_normed = ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=mc)
    gate_act = ttnn.silu(gate, memory_config=mc)
    gate_act = ttnn.clip(gate_act, min=-1e4, max=1e4)
    return ttnn.multiply(x_normed, gate_act, memory_config=mc)


def rms_norm_ttnn(x, weight, eps=1e-5, memory_config=None):
    """Standard RMSNorm (trace-compatible)."""
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=memory_config)


def _causal_conv1d_decode_t1_split(
    x, conv_state_list, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """T=1 decode conv+SiLU with split state (list of [B,1,D]); avoids slice ops.

    Returns output [B,1,D], new_state_list.
    """
    mc = memory_config

    # out = sum(weight_taps[k] * state[k]) + weight_taps[K-1] * x
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        term = ttnn.multiply(conv_state_list[k], weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Shift state left, append x
    new_state_list = conv_state_list[1:] + [x]

    return ttnn.silu(out, memory_config=mc), new_state_list


def _causal_conv1d_decode_t1_split_inplace(
    x, conv_state_list, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """Split-state T=1 conv; inplace copy for trace-stable addresses."""
    mc = memory_config

    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        term = ttnn.multiply(conv_state_list[k], weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Inplace shift via ttnn.copy
    for k in range(kernel_size - 2):
        ttnn.copy(conv_state_list[k + 1], conv_state_list[k])
    ttnn.copy(x, conv_state_list[kernel_size - 2])

    return ttnn.silu(out, memory_config=mc), conv_state_list


def _causal_conv1d_decode_t1(x, conv_state, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None):
    """T=1 decode conv+SiLU; taps 0..K-2 from state[:,k], tap K-1 from x."""
    mc = memory_config

    # out = sum(weight_taps[k] * state[k]) + weight_taps[K-1] * x
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        s_k = conv_state[:, k : k + 1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(s_k, weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Drop oldest, append x
    new_state = ttnn.concat([conv_state[:, 1:, :], x], dim=1, memory_config=mc)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)

    return ttnn.silu(out, memory_config=mc), new_state


def _causal_conv1d_decode_t1_inplace(
    x, conv_buffer, kernel_size, device, memory_config=None, weight_taps=None, bias_dev=None
):
    """T=1 conv; copy-back to pre-allocated conv_buffer for trace capture."""
    mc = memory_config

    # out = sum(weight_taps[k] * state[k]) + weight_taps[K-1] * x
    out = ttnn.multiply(x, weight_taps[kernel_size - 1], memory_config=mc)
    for k in range(kernel_size - 1):
        s_k = conv_buffer[:, k : k + 1, :]
        s_k = ttnn.to_layout(s_k, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(s_k, weight_taps[k], memory_config=mc)
        out = ttnn.add(out, term, memory_config=mc)

    if bias_dev is not None:
        out = ttnn.add(out, bias_dev, memory_config=mc)

    # Update conv_buffer in-place
    new_state = ttnn.concat([conv_buffer[:, 1:, :], x], dim=1, memory_config=mc)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
    ttnn.copy(new_state, conv_buffer)
    ttnn.deallocate(new_state)

    return ttnn.silu(out, memory_config=mc), conv_buffer


def _causal_conv1d_fir(
    x,
    weight,
    bias,
    kernel_size,
    device,
    memory_config=None,
    conv_state=None,
    weight_taps=None,
    bias_dev=None,
    valid_len=None,
):
    """Depthwise causal conv1d + SiLU via K shifted multiply-accumulate slices.

    x [B,T,D]; conv_state [B,K-1,D] or list of [B,1,D]; weight_taps/bias_dev optional.
    Returns output [B,T,D], new_state [B,K-1,D].
    """
    mc = memory_config
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    # Fast path: T=1 decode with state + pre-sliced taps
    if T == 1 and conv_state is not None and weight_taps is not None:
        return _causal_conv1d_decode_t1(
            x, conv_state, kernel_size, device, memory_config=mc, weight_taps=weight_taps, bias_dev=bias_dev
        )

    if conv_state is not None:
        x_padded = ttnn.concat([conv_state, x], dim=1, memory_config=mc)
    else:
        pad = ttnn.zeros(
            [B, kernel_size - 1, D],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )
        x_padded = ttnn.concat([pad, x], dim=1, memory_config=mc)

    # new_state: last K-1 tokens; land in DRAM (carry alive across downstream kernel CBs).
    total_len = (kernel_size - 1) + T
    if valid_len is None:
        new_state = x_padded[:, total_len - (kernel_size - 1) :, :]
        # to_layout then to_memory_config: slice keeps L1 if memory_config passed to to_layout
        new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
        new_state = ttnn.to_memory_config(new_state, ttnn.DRAM_MEMORY_CONFIG)
    else:
        # Fixed-bucket masking: x is right-padded to a bucket length T but only the first
        # valid_len positions are real; the decode conv window must come from the real tail
        # x[valid_len-(K-1):valid_len], i.e. x_padded[:, valid_len : valid_len+(K-1)] (x[i]
        # is at x_padded index (K-1)+i). A static slice there would compile a new program per
        # valid_len value — defeating the bounded-program goal — so select those rows with a
        # one-hot matmul instead: the program depends only on shapes (fixed per bucket), and
        # only the one-hot VALUES depend on valid_len.
        # valid_len may be a scalar (one length for all B rows) or a per-row list/tuple of length
        # B (batched prefill: each user's own real length picks that user's decode conv window).
        sel = torch.zeros(B, kernel_size - 1, total_len, dtype=torch.float32)
        if isinstance(valid_len, (list, tuple)):
            for bi in range(B):
                for j in range(kernel_size - 1):
                    sel[bi, j, int(valid_len[bi]) + j] = 1.0
        else:
            for j in range(kernel_size - 1):
                sel[:, j, valid_len + j] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=x_padded.dtype, layout=ttnn.TILE_LAYOUT, device=device)
        xp = ttnn.to_layout(x_padded, ttnn.TILE_LAYOUT)
        # cross-chunk carry -> DRAM
        new_state = ttnn.matmul(sel_tt, xp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sel_tt)

    # Precompute weight taps if not provided
    if weight_taps is None:
        pass

        weight_torch = ttnn.to_torch(weight)
        weight_taps = []
        for k in range(kernel_size):
            w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
            weight_taps.append(
                ttnn.from_torch(w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            )

    total_len = (kernel_size - 1) + T
    _dram = ttnn.DRAM_MEMORY_CONFIG
    # Depthwise K-tap FIR via multiply + addcmul; re-tilize k>=1 slices (only k=0 is tile-aligned).
    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k : k + T]
        if k != 0:
            x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)
        if out is None:
            out = ttnn.multiply(x_slice, weight_taps[k], memory_config=mc)
        else:
            out = ttnn.addcmul(out, x_slice, weight_taps[k], memory_config=mc)

    # Bias (+ fused SiLU when a bias is present) else standalone SiLU. Conv output lands in DRAM.
    _silu = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]
    if bias_dev is not None:
        return ttnn.add(out, bias_dev, activations=_silu, memory_config=_dram), new_state
    if bias is not None:
        bias_torch = ttnn.to_torch(bias).reshape(1, 1, D).contiguous()
        bias_dev_tmp = ttnn.from_torch(
            bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        return ttnn.add(out, bias_dev_tmp, activations=_silu, memory_config=_dram), new_state
    # Conv output in DRAM (feeds gated_delta_attn_seq; MAC still ran in L1 when mc=L1)
    return ttnn.silu(out, memory_config=_dram), new_state


def causal_conv1d_ttnn(
    x,
    weight,
    bias,
    kernel_size,
    device,
    max_conv_len=512,
    memory_config=None,
    conv_state=None,
    weight_taps=None,
    bias_dev=None,
):
    """Depthwise causal conv1d + SiLU. FIR fallback when conv_state, T>max_conv_len, or D>2048."""
    B, T, D = x.shape[0], x.shape[1], x.shape[2]
    mc = memory_config

    # FIR when conv_state, T>max_conv_len, or D>2048 (native conv1d CBs overflow L1 at D=4096)
    if conv_state is not None or T > max_conv_len or D > 2048:
        return _causal_conv1d_fir(
            x,
            weight,
            bias,
            kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state,
            weight_taps=weight_taps,
            bias_dev=bias_dev,
        )

    # No state: native conv1d with zero padding
    if mc is not None:
        x = ttnn.to_memory_config(x, mc)
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    pad_zeros = ttnn.zeros(
        [B, kernel_size - 1, D],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=mc,
    )
    x_padded = ttnn.concat([pad_zeros, x_rm], dim=1, memory_config=mc)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        config_tensors_in_dram=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
    )

    [out, out_length, _] = ttnn.conv1d(
        input_tensor=x_padded,
        weight_tensor=weight,
        in_channels=D,
        out_channels=D,
        device=device,
        bias_tensor=bias,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        batch_size=B,
        input_length=T + kernel_size - 1,
        groups=D,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    out = ttnn.sharded_to_interleaved(out, memory_config=mc)
    out = ttnn.reshape(out, [B, T, D])
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=mc)

    # Save last K-1 input tokens as conv state
    x_tile = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    if T >= kernel_size - 1:
        new_state = x_tile[:, -(kernel_size - 1) :, :]
        new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT)
    else:
        pad_needed = kernel_size - 1 - T
        pad_state = ttnn.zeros(
            [B, pad_needed, D],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )
        new_state = ttnn.concat([pad_state, x_tile], dim=1, memory_config=mc)

    return out, new_state


def gated_deltanet_forward_ttnn(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    a_proj_weight,
    b_proj_weight,
    o_proj_weight,
    q_conv_weight,
    k_conv_weight,
    v_conv_weight,
    q_conv_bias,
    k_conv_bias,
    v_conv_bias,
    A_log,
    dt_bias,
    o_norm_weight,
    g_proj_weight=None,
    num_heads=4,
    num_v_heads=None,
    head_k_dim=256,
    head_v_dim=512,
    conv_kernel_size=4,
    use_gate=True,
    allow_neg_eigval=False,
    norm_eps=1e-5,
    device=None,
    recurrent_state=None,
    conv_state_q=None,
    conv_state_k=None,
    conv_state_v=None,
    mode="recurrent",
    chunk_size=64,
    q_weight_taps=None,
    k_weight_taps=None,
    v_weight_taps=None,
    q_bias_dev=None,
    k_bias_dev=None,
    v_bias_dev=None,
    qkv_proj_weight=None,
    q_dim=None,
    k_dim=None,
    compute_kernel_config=None,
    A_neg_precomputed=None,
    fused_conv_weight_taps=None,
    fused_conv_bias_dev=None,
    fused_conv_state=None,
    fused_conv_state_split=None,  # list of [B,1,D] for decode (no slice)
    ab_proj_weight=None,  # fused a+b (1 matmul)
    mega_fused_weight=None,  # QKV+a+b+g in one matmul
    mega_qkv_dim=None,
    mega_a_dim=None,
    mega_b_dim=None,
    mega_g_dim=None,
    use_inplace_state=False,  # ttnn.copy for trace-stable state
    chunk_seq_masks=None,  # cached masks for gated_delta_attn_seq prefill
    valid_len=None,  # fixed-bucket padding; zeros padded positions in scan
    chunk_delta_fn=None,
    prefill_progcfg_fn=None,
    decode_progcfg_fn=None,
    native_conv1d_fn=None,
):
    """Gated DeltaNet forward. mode: recurrent (decode T=1) or chunk (prefill T>1).

    Returns output [B,T,hidden], new_state [B,H,K,V], conv states.

    chunk_delta_fn: optional callable replacing the chunk-prefill delta-rule core; receives
    FLAT q/k/v ([B,T,H*D]) plus qkv_head_dims and must return (o [B,T,Nv,Dv] TILE,
    new_state [B,Nv,Dk,Dv]).

    prefill_progcfg_fn: optional `(m,k,n,in0_dtype,in1_dtype,fp32_acc)->progcfg` used for the
    chunk-prefill projections when T > 512 (DRAM outputs).

    decode_progcfg_fn: optional `(k,n)->progcfg` used for T==1 (decode) projections; takes
    priority over prefill_progcfg_fn (which never fires at T==1 anyway, see `_pc`).

    native_conv1d_fn: optional (x [B,T,C] TILE, conv_state [B,K-1,C]|None) -> (silu(conv(x))
    [B,T,C] TILE DRAM, new_state [B,K-1,C]) used for chunk prefill when valid_len is None.
    """
    if num_v_heads is None:
        num_v_heads = num_heads

    B = hidden_states.shape[0]
    T = hidden_states.shape[1]
    # valid_len path forces DRAM (bucket 512 hits L1 CB clash on _seq_memory_config threshold)
    mc = None if valid_len is not None else _seq_memory_config(T)

    # F10B item B: independent per-tensor L1 config for three small/cheap groups (beta/g/a/b chain,
    # post-scan chain, out-proj output) at T <= 2048 -- see _gdn_post_l1_groups' docstring for why
    # this is separate from `mc` above. Masked/bucket calls (valid_len is not None) keep `mc`
    # (DRAM) for these too, matching mc's own guard on the line above.
    _post_small, _post_scan, _post_outproj = _gdn_post_l1_groups()
    _post_ok = valid_len is None and T <= 2048
    mc_small = ttnn.L1_MEMORY_CONFIG if (_post_small and _post_ok) else mc
    mc_scan = ttnn.L1_MEMORY_CONFIG if (_post_scan and _post_ok) else mc
    mc_outproj = ttnn.L1_MEMORY_CONFIG if (_post_outproj and _post_ok) else mc

    ckc = compute_kernel_config

    def _pc(x_in, w_in):
        if T == 1 and decode_progcfg_fn is not None:
            return decode_progcfg_fn(x_in.shape[-1], w_in.shape[-1])
        if prefill_progcfg_fn is None or mode != "chunk":
            return None
        return prefill_progcfg_fn(
            T, x_in.shape[-1], w_in.shape[-1], x_in.dtype, w_in.dtype, getattr(ckc, "fp32_dest_acc_en", True)
        )

    # Mega-fused: one matmul for QKV+a+b+g (decode needs conv state; prefill needs fused taps)
    use_mega_fused = (
        mega_fused_weight is not None
        and mega_qkv_dim is not None
        and (T == 1 or fused_conv_weight_taps is not None)
        and (T > 1 or (fused_conv_state is not None and fused_conv_weight_taps is not None))
    )

    # Fused conv decode: QKV -> fused conv -> split (skip mega path)
    use_fused_conv = (
        not use_mega_fused
        and T == 1
        and fused_conv_weight_taps is not None
        and fused_conv_state is not None
        and qkv_proj_weight is not None
        and q_dim is not None
    )

    if use_mega_fused:
        # F6-B: T>1 (chunk prefill) can run two matmuls off the same x instead of one mega matmul
        # + 3 slices of the full [*, D_total] output — qkv comes back tile-native at full width
        # (no slicing at all) and gate/a/b are sliced from a ~2080-wide tensor instead of the full
        # ~8224-wide mega_out. Decode (T==1) always takes the legacy single-matmul path (mega-fused
        # is a bigger win there — one matmul beats two at M=1), matching today's behavior exactly;
        # QWEN36_GDN_SPLIT_PROJ=0 also restores the legacy path at any T.
        if T > 1 and os.environ.get("QWEN36_GDN_SPLIT_PROJ", "1") != "0":
            w_qkv, w_gab = _get_mega_split_weights(mega_fused_weight, mega_qkv_dim, mega_g_dim, mega_a_dim, mega_b_dim)
            qkv = ttnn.experimental.minimal_matmul(
                hidden_states,
                w_qkv,
                config=tpc.prefill_minimal_matmul_config(
                    T, hidden_states.shape[-1], w_qkv.shape[-1], _get_mm_grid(device)
                ),
                compute_kernel_config=ckc,
                memory_config=mc,
            )
            gab = ttnn.linear(
                hidden_states,
                w_gab,
                memory_config=mc,
                compute_kernel_config=ckc,
                program_config=_pc(hidden_states, w_gab),
            )
            # gab is laid out g|a|b (columns mega_qkv_dim..end of the original mega weight); g_dim
            # is a tile-width multiple so this begin lands on a tile boundary (tile-native, no
            # untilize). a/b are half a tile each, pulled out together as before.
            gate_raw = gab[:, :, :mega_g_dim]
            ab_raw = gab[:, :, mega_g_dim : mega_g_dim + mega_a_dim + mega_b_dim]
            a_raw = ab_raw[:, :, :mega_a_dim]
            a_raw = ttnn.to_layout(a_raw, ttnn.TILE_LAYOUT)
            b_raw = ab_raw[:, :, mega_a_dim : mega_a_dim + mega_b_dim]
            b_raw = ttnn.to_layout(b_raw, ttnn.TILE_LAYOUT)
            ttnn.deallocate(ab_raw)
            ttnn.deallocate(gab)
        else:
            mega_out = ttnn.linear(
                hidden_states,
                mega_fused_weight,
                memory_config=mc,
                compute_kernel_config=ckc,
                program_config=_pc(hidden_states, mega_fused_weight),
            )
            # Split QKV | gate | (a|b). mega_fused_weight is laid out qkv|g|a|b (gdn/weights.py):
            # qkv_dim and g_dim are both tile-width multiples, so both begins below land on a
            # tile boundary and ttnn.slice keeps them in TILE layout natively (no to_layout, no
            # untilize of mega_out). a/b are half a tile each, so they're pulled out together as one
            # tile-aligned, 1-tile-wide slice and only that small slice pays the untilize needed to
            # split it in half below — not the whole [*, D_total] mega_out.
            qkv = mega_out[:, :, :mega_qkv_dim]
            gate_raw = mega_out[:, :, mega_qkv_dim : mega_qkv_dim + mega_g_dim]
            ab_raw = mega_out[:, :, mega_qkv_dim + mega_g_dim : mega_qkv_dim + mega_g_dim + mega_a_dim + mega_b_dim]
            a_raw = ab_raw[:, :, :mega_a_dim]
            a_raw = ttnn.to_layout(a_raw, ttnn.TILE_LAYOUT)
            b_raw = ab_raw[:, :, mega_a_dim : mega_a_dim + mega_b_dim]
            b_raw = ttnn.to_layout(b_raw, ttnn.TILE_LAYOUT)
            ttnn.deallocate(ab_raw)
            ttnn.deallocate(mega_out)

        # Fused conv on QKV
        if T > 1:
            if native_conv1d_fn is not None and mode == "chunk" and T > 1 and valid_len is None:
                # Native ttnn.conv1d depthwise + SiLU (height-sharded, L1_FULL slice) — replaces the
                # FIR MAC fallback for chunk prefill. SiLU is applied inside native_conv1d_fn,
                # matching this FIR call's bias_dev=None branch (fused_conv_bias_dev is None for
                # this model: Qwen3.5 GDN has no conv1d bias — see gdn/weights.py). May return qkv
                # as a (q, k, v) tuple directly (conv1d_native.py's n_cc=3 fast path, only taken
                # when chunk width == q_dim == k_dim == v_dim) — see "Split QKV after conv" below.
                qkv, new_fused_conv_state_raw = native_conv1d_fn(qkv, fused_conv_state)
            else:
                # Prefill FIR conv
                qkv, new_fused_conv_state_raw = _causal_conv1d_fir(
                    qkv,
                    None,
                    None,
                    conv_kernel_size,
                    device,
                    memory_config=mc,
                    conv_state=fused_conv_state,
                    weight_taps=fused_conv_weight_taps,
                    bias_dev=fused_conv_bias_dev,
                    valid_len=valid_len,
                )
            new_fused_conv_state = new_fused_conv_state_raw
            # Per-stream conv states for decode handoff: dead here (fused_conv_state carries the
            # cross-chunk conv history instead), so the old slice+to_layout split into
            # new_conv_q/k/v — 3 Slice + 3 to_layout per GDN layer, unused — is removed.
            new_conv_q = None
            new_conv_k = None
            new_conv_v = None
        elif fused_conv_state_split is not None:
            # Split-state decode (no slice+to_layout)
            conv_fn = _causal_conv1d_decode_t1_split_inplace if use_inplace_state else _causal_conv1d_decode_t1_split
            qkv, new_fused_conv_state = conv_fn(
                qkv,
                fused_conv_state_split,
                conv_kernel_size,
                device,
                memory_config=mc,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
            )
            new_conv_q = None
            new_conv_k = None
            new_conv_v = None
        else:
            # Decode with fused state
            conv_fn = _causal_conv1d_decode_t1_inplace if use_inplace_state else _causal_conv1d_decode_t1
            qkv, new_fused_conv_state = conv_fn(
                qkv,
                fused_conv_state,
                conv_kernel_size,
                device,
                memory_config=mc,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
            )
            new_conv_q = None
            new_conv_k = None
            new_conv_v = None

        # Split QKV after conv — skipped when native_conv1d_fn already returned (q, k, v) directly
        # (its n_cc=3 fast path: the 3 conv outputs ARE q/k/v, so no concat happened above and
        # there is nothing left to slice).
        if isinstance(qkv, tuple):
            q, k, v = qkv
        else:
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + k_dim]
            v = qkv[:, :, q_dim + k_dim :]
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv)

        # a/b/g already from mega projection
        _mega_extracted = True
    elif use_fused_conv:
        # Fused decode: QKV proj -> fused conv -> split
        qkv = ttnn.linear(
            hidden_states,
            qkv_proj_weight,
            memory_config=mc,
            compute_kernel_config=ckc,
            program_config=_pc(hidden_states, qkv_proj_weight),
        )
        # Fused conv1d on concatenated QKV
        conv_fn = _causal_conv1d_decode_t1_inplace if use_inplace_state else _causal_conv1d_decode_t1
        qkv, new_fused_conv_state = conv_fn(
            qkv,
            fused_conv_state,
            conv_kernel_size,
            device,
            memory_config=mc,
            weight_taps=fused_conv_weight_taps,
            bias_dev=fused_conv_bias_dev,
        )
        # Split after conv
        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim : q_dim + k_dim]
        v = qkv[:, :, q_dim + k_dim :]
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        ttnn.deallocate(qkv)
        new_conv_q = None
        new_conv_k = None
        new_conv_v = None
        _mega_extracted = False
    elif qkv_proj_weight is not None and q_dim is not None:
        qkv = ttnn.linear(
            hidden_states,
            qkv_proj_weight,
            memory_config=mc,
            compute_kernel_config=ckc,
            program_config=_pc(hidden_states, qkv_proj_weight),
        )
        # Fused conv prefill on concatenated QKV
        if T > 1 and fused_conv_weight_taps is not None:
            qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
            qkv, new_fused_conv_state_raw = _causal_conv1d_fir(
                qkv,
                None,
                None,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=fused_conv_state,
                weight_taps=fused_conv_weight_taps,
                bias_dev=fused_conv_bias_dev,
                valid_len=valid_len,
            )
            # Split after conv
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + k_dim]
            v = qkv[:, :, q_dim + k_dim :]
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv)
            # Per-stream conv states from fused state
            D_total = (
                q_dim + k_dim + (qkv_proj_weight.shape[-1] - q_dim - k_dim)
                if hasattr(qkv_proj_weight, "shape")
                else None
            )
            new_conv_q = new_fused_conv_state_raw[:, :, :q_dim]
            new_conv_q = ttnn.to_layout(new_conv_q, ttnn.TILE_LAYOUT)
            new_conv_k = new_fused_conv_state_raw[:, :, q_dim : q_dim + k_dim]
            new_conv_k = ttnn.to_layout(new_conv_k, ttnn.TILE_LAYOUT)
            new_conv_v = new_fused_conv_state_raw[:, :, q_dim + k_dim :]
            new_conv_v = ttnn.to_layout(new_conv_v, ttnn.TILE_LAYOUT)
            new_fused_conv_state = new_fused_conv_state_raw
            _mega_extracted = False
        else:
            q = qkv[:, :, :q_dim]
            k = qkv[:, :, q_dim : q_dim + k_dim]
            v = qkv[:, :, q_dim + k_dim :]
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
            v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            ttnn.deallocate(qkv)
            new_fused_conv_state = None
            _mega_extracted = False
            q, new_conv_q = causal_conv1d_ttnn(
                q,
                q_conv_weight,
                q_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_q,
                weight_taps=q_weight_taps,
                bias_dev=q_bias_dev,
            )
            k, new_conv_k = causal_conv1d_ttnn(
                k,
                k_conv_weight,
                k_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_k,
                weight_taps=k_weight_taps,
                bias_dev=k_bias_dev,
            )
            v, new_conv_v = causal_conv1d_ttnn(
                v,
                v_conv_weight,
                v_conv_bias,
                conv_kernel_size,
                device,
                memory_config=mc,
                conv_state=conv_state_v,
                weight_taps=v_weight_taps,
                bias_dev=v_bias_dev,
            )
    else:
        q = ttnn.linear(
            hidden_states,
            q_proj_weight,
            memory_config=mc,
            compute_kernel_config=ckc,
            program_config=_pc(hidden_states, q_proj_weight),
        )
        k = ttnn.linear(
            hidden_states,
            k_proj_weight,
            memory_config=mc,
            compute_kernel_config=ckc,
            program_config=_pc(hidden_states, k_proj_weight),
        )
        v = ttnn.linear(
            hidden_states,
            v_proj_weight,
            memory_config=mc,
            compute_kernel_config=ckc,
            program_config=_pc(hidden_states, v_proj_weight),
        )
        new_fused_conv_state = None
        _mega_extracted = False
        q, new_conv_q = causal_conv1d_ttnn(
            q,
            q_conv_weight,
            q_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_q,
            weight_taps=q_weight_taps,
            bias_dev=q_bias_dev,
        )
        k, new_conv_k = causal_conv1d_ttnn(
            k,
            k_conv_weight,
            k_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_k,
            weight_taps=k_weight_taps,
            bias_dev=k_bias_dev,
        )
        v, new_conv_v = causal_conv1d_ttnn(
            v,
            v_conv_weight,
            v_conv_bias,
            conv_kernel_size,
            device,
            memory_config=mc,
            conv_state=conv_state_v,
            weight_taps=v_weight_taps,
            bias_dev=v_bias_dev,
        )

    _use_chunk_fn = chunk_delta_fn is not None and mode == "chunk" and T > 1
    _o_head_major = False  # True only on the fused chunk_delta_fn path (o stays [B*Nv, T, Dv])

    # QWEN36_GDN_FLA_INPUTS_DRAM (step2, 2026-09-22; default "1" when QWEN_GDN_PATH=="fused", else
    # "0" -- same flag/default as gdn/conv1d_kda.py's, read again here since this file doesn't
    # import that module). On the fused chunk_delta_fn path only: beta/g (and the a_biased/sp
    # intermediates that feed g, all normally `mc_small` = L1 at T <= 2048 per F10B) are alive in
    # L1 from here until the chunk_delta_fn call below returns -- exactly like conv1d_kda.py's
    # q/k/v, they are the FLA op's own inputs, produced with no op in between. At T=2048 they are
    # small (~1 KB/bank each) next to q/k/v's ~193.6 KB/bank, but on the specific core(s) where the
    # FLA op's static CBs (ending at 1,213,440 B/bank, of 1,436,672 B/bank total -- 223,232 B free)
    # are laid out, this chain is enough to tip an already-nearly-full core over the edge. Moving
    # it to DRAM alongside q/k/v removes it from that budget entirely. Scoped to `_use_chunk_fn`
    # only -- decode (T==1) and the seq-adapter path keep beta/g in L1 (mc_small), unaffected,
    # since their own kernel CBs are smaller and don't clash (see conv1d_kda.py's flag docstring).
    if (
        _use_chunk_fn
        and os.environ.get("QWEN36_GDN_FLA_INPUTS_DRAM", "1" if os.environ.get("QWEN_GDN_PATH") == "fused" else "0")
        != "0"
    ):
        mc_small = ttnn.DRAM_MEMORY_CONFIG

    if not _use_chunk_fn:
        # Reshape to heads (explicit mc keeps decode in L1)
        q = ttnn.reshape(q, [B, T, num_heads, head_k_dim], memory_config=mc)
        k = ttnn.reshape(k, [B, T, num_heads, head_k_dim], memory_config=mc)
        v = ttnn.reshape(v, [B, T, num_v_heads, head_v_dim], memory_config=mc)

        # GVA: repeat q,k
        if num_v_heads > num_heads:
            repeats = num_v_heads // num_heads
            q = ttnn.repeat_interleave(q, repeats, dim=2)
            k = ttnn.repeat_interleave(k, repeats, dim=2)

    # Beta and g. F10B item B: the small (num_v_heads-wide) beta/g/a elementwise chain below uses
    # mc_small (independent L1 policy), not mc -- the qkv/gab/ab/a/b PROJECTION matmuls just above
    # and below are untouched (still `mc`, i.e. DRAM at T=2048).
    if _mega_extracted:
        a = a_raw
        beta = ttnn.sigmoid(b_raw, memory_config=mc_small)
    elif ab_proj_weight is not None:
        ab = ttnn.linear(
            hidden_states,
            ab_proj_weight,
            memory_config=mc,
            compute_kernel_config=ckc,
            program_config=_pc(hidden_states, ab_proj_weight),
        )
        num_v = num_v_heads if num_v_heads is not None else num_heads
        a = ab[:, :, :num_v]
        a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
        b_raw = ab[:, :, num_v:]
        b_raw = ttnn.to_layout(b_raw, ttnn.TILE_LAYOUT)
        ttnn.deallocate(ab)
        beta = ttnn.sigmoid(b_raw, memory_config=mc_small)
    else:
        beta = ttnn.sigmoid(
            ttnn.linear(
                hidden_states,
                b_proj_weight,
                memory_config=mc,
                compute_kernel_config=ckc,
                program_config=_pc(hidden_states, b_proj_weight),
            ),
            memory_config=mc_small,
        )
        a = ttnn.linear(
            hidden_states,
            a_proj_weight,
            memory_config=mc,
            compute_kernel_config=ckc,
            program_config=_pc(hidden_states, a_proj_weight),
        )
    if allow_neg_eigval:
        beta = ttnn.multiply(beta, 2.0, memory_config=mc_small)
    a_biased = ttnn.add(a, dt_bias, memory_config=mc_small)
    sp = ttnn.softplus(a_biased, memory_config=mc_small)
    if A_neg_precomputed is not None:
        g = ttnn.multiply(A_neg_precomputed, sp, memory_config=mc_small)
    else:
        A = ttnn.exp(A_log, memory_config=mc_small)
        A_neg = ttnn.neg(A, memory_config=mc_small)
        g = ttnn.multiply(A_neg, sp, memory_config=mc_small)

    # Gated delta rule: chunk prefill (fp32 seq kernel) vs decode (optimized T=1) vs recurrent fallback
    if mode == "chunk" and T > 1:
        if _use_chunk_fn:
            # F6-D: the fused op's C++ composition (gdn/fused_chunk.py ->
            # ttnn.transformer.chunk_gated_delta_rule -> chunk_gated_delta_rule.cpp's
            # headvec_split_tile()) unconditionally typecasts g/beta to FLOAT32 before its own
            # permute+reshape to [BH,T]; g/beta land here as bf16 (sigmoid/softplus/exp/multiply
            # all run in bf16 above), so casting to fp32 here removes that redundant internal
            # typecast (numerically identical: same cast, just done once, earlier, on the smaller
            # pre-pad [B,T,Nv] tensor). NOTE: this does NOT eliminate the two
            # TransposeDeviceOperation + ReshapeViewDeviceOperation pairs the profile flags for
            # g/beta — those live inside chunk_gated_delta_rule.cpp's headvec_split_tile() and the
            # g_c/beta_c per-chunk reshape, neither of which is in this change's file scope (only
            # ttnn_gated_deltanet.py/ttnn_gated_attention.py/conv1d_native.py/weights.py are
            # editable here); see the F6 handoff report. QWEN36_GDN_GB_LAYOUT=0 skips the cast,
            # matching today's dtype exactly. Default OFF (phase-2 measurement, 2026-09-20): the
            # pre-cast only replaces an equally-cheap ~2us internal typecast with an equally-cheap
            # ~2us external one -- no measurable val_b or per-op benefit (T=2048/4096 deltas were
            # within run-to-run noise, +-0.4us out of ~90ms total). Set QWEN36_GDN_GB_LAYOUT=1 to
            # re-enable; kept for anyone revisiting (D) once fused_chunk.py/the C++ op are in scope.
            if os.environ.get("QWEN36_GDN_GB_LAYOUT", "0") != "0":
                if beta.dtype != ttnn.float32:
                    beta = ttnn.typecast(beta, ttnn.float32, memory_config=mc)
                if g.dtype != ttnn.float32:
                    g = ttnn.typecast(g, ttnn.float32, memory_config=mc)
            # Fused chunk kernel on flat q/k/v; it L2-normalizes q/k in-kernel and handles GQA itself.
            # OPT-A flat-v requires T to be a whole number of 32-row chunks, so pad up to the next
            # chunk boundary. Zero rows (beta=g=0) leave the recurrent state untouched at those
            # positions (decay factor exp(0)=1, update weight 0), so only the output needs slicing back.
            T_pad = ((T + 31) // 32) * 32
            if T_pad != T:
                q, k, v, beta, g = (_pad_rows(t, T_pad - T) for t in (q, k, v, beta, g))
            # Head-major output ([B*Nv, T, Dv] TILE) skips the untilize + [B,Nv,T,Dv]->[B,T,Nv,Dv]
            # permute + tilize the adapter otherwise does to hand back token-major o.
            o, new_state = chunk_delta_fn(
                q,
                k,
                v,
                beta,
                g,
                chunk_size=chunk_size,
                initial_state=recurrent_state,
                device=device,
                valid_len=valid_len,
                qkv_head_dims=(num_heads, head_k_dim, num_v_heads, head_v_dim),
                return_o_bh=True,
            )
            _o_head_major = True
            if T_pad != T:
                # o is [B*Nv, T_pad, Dv]; T is dim 1 here too, same slice as the token-major shape.
                o = o[:, :T]
        else:
            o, new_state = chunk_gated_delta_rule_seq_adapter(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
                chunk_size=chunk_size,
                initial_state=recurrent_state,
                device=device,
                cached_masks=chunk_seq_masks,
                valid_len=valid_len,
            )
    elif T == 1:
        if use_inplace_state and recurrent_state is not None:
            o, new_state = recurrent_gated_delta_rule_decode_inplace_ttnn(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
                state_buffer=recurrent_state,
                device=device,
            )
        else:
            o, new_state = recurrent_gated_delta_rule_decode_ttnn(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
                initial_state=recurrent_state,
                device=device,
            )
    else:
        o, new_state = recurrent_gated_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            initial_state=recurrent_state,
            device=device,
        )

    # Output norm + projection (clip before o_proj to avoid sparse overflow)
    if _o_head_major:
        # Fused-chunk output is head-major [B*Nv, T, Dv]: norm per head in place, fold heads with the
        # TILE-native concat, and gate on the flat [B, T, Nv*Dv] tensor. Same math as
        # rms_norm_gated_ttnn without the token<->head relayouts and the two copying reshapes.
        o = ttnn.reshape(o, [B, num_v_heads, T, head_v_dim])  # metadata only
        # scan emits fp32; the rest of the layer is bf16 — cast once here instead of running
        # norm/concat/gate/clip at 2x the bytes
        # F10B item B: this typecast + per-head rms_norm + nlp_concat_heads is the "post-scan chain"
        # group -- mc_scan (independent L1 policy), not mc.
        o = ttnn.typecast(o, ttnn.bfloat16, memory_config=mc_scan)
        o = ttnn.rms_norm(o, weight=o_norm_weight, epsilon=norm_eps, memory_config=mc_scan)
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=mc_scan)  # [B, 1, T, Nv*Dv]
        o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim])  # metadata only (last dim kept)
        if use_gate and g_proj_weight is not None:
            if _mega_extracted:
                gate = gate_raw  # already flat [B, T, Nv*Dv]
            else:
                gate = ttnn.linear(
                    hidden_states,
                    g_proj_weight,
                    memory_config=mc,
                    compute_kernel_config=ckc,
                    program_config=_pc(hidden_states, g_proj_weight),
                )
            # F6-A: fuse silu(gate) into the multiply's second-operand activation (one op instead
            # of silu+clip+multiply) and make the post-multiply clip optional. QWEN36_GDN_GATE_FUSED=0
            # restores exactly the legacy 4-op sequence (silu, clip, multiply, clip) byte-for-byte.
            # F10B item B: the fused gating multiply is also part of the post-scan group -> mc_scan.
            if os.environ.get("QWEN36_GDN_GATE_FUSED", "1") != "0":
                o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=mc_scan)
                if os.environ.get("QWEN36_GDN_GATE_CLIP", "0") == "1":
                    o = ttnn.clip(o, min=-1e4, max=1e4, memory_config=mc)
            else:
                gate_act = ttnn.silu(gate, memory_config=mc)
                gate_act = ttnn.clip(gate_act, min=-1e4, max=1e4)
                o = ttnn.multiply(o, gate_act, memory_config=mc)
                o = ttnn.clip(o, min=-1e4, max=1e4, memory_config=mc)
        else:
            o = ttnn.clip(o, min=-1e4, max=1e4, memory_config=mc)
        # already [B, T, Nv*Dv]; skip the reshape below
    else:
        if use_gate and g_proj_weight is not None:
            if _mega_extracted:
                gate = ttnn.reshape(gate_raw, [B, T, num_v_heads, head_v_dim], memory_config=mc)
            else:
                gate = ttnn.linear(
                    hidden_states,
                    g_proj_weight,
                    memory_config=mc,
                    compute_kernel_config=ckc,
                    program_config=_pc(hidden_states, g_proj_weight),
                )
                gate = ttnn.reshape(gate, [B, T, num_v_heads, head_v_dim], memory_config=mc)
            o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps, memory_config=mc)
        else:
            o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps, memory_config=mc)

        o = ttnn.clip(o, min=-1e4, max=1e4, memory_config=mc)
        o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim], memory_config=mc)

    if mc is not None:
        o = ttnn.to_memory_config(o, mc)
    # F10B item B: out-proj linear OUTPUT uses mc_outproj (independent L1 policy); the input staging
    # copy above and the matmul's own program config are untouched.
    o = ttnn.linear(
        o,
        o_proj_weight,
        memory_config=mc_outproj,
        compute_kernel_config=ckc,
        program_config=_pc(o, o_proj_weight),
    )

    return o, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv_state
