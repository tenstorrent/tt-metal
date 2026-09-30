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
_C2_LOGGED = set()  # C2 items whose one-time "[C2] ... active" line was printed


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
    """Per-device ttnn.CoreCoord from compute_with_storage_grid_size(), cached by id(device). A
    reused id (a new device after a close) is harmless here: the value is a plain CoreCoord that
    holds no device resource, and it is the same grid on the same hardware."""
    key = id(device)
    grid = _mm_grid_cache.get(key)
    if grid is None:
        g = device.compute_with_storage_grid_size()
        grid = ttnn.CoreCoord(g.x, g.y)
        _mm_grid_cache[key] = grid
    return grid


def _mega_split_entry_live(entry, mega_fused_weight):
    """A cache entry (src, w_qkv, w_gab) is reusable only for the same source tensor object and while
    its slices are still allocated."""
    src, w_qkv, w_gab = entry
    return src is mega_fused_weight and w_qkv.is_allocated() and (w_gab is None or w_gab.is_allocated())


def _get_mega_split_weights(mega_fused_weight, mega_qkv_dim, mega_g_dim, mega_a_dim, mega_b_dim, need_gab=True):
    """Derive (w_qkv, w_gab) from the mega-fused [qkv|g|a|b] weight, cached per source tensor.

    Keyed by id(mega_fused_weight). id() alone is not safe: a new model object (e.g. per-test
    pytest devices) can get the id of a freed weight and would receive that weight's deallocated
    slices. So each entry holds the source tensor itself (the id cannot be reused while the entry
    lives) and a hit must be the same object with its slices still allocated; otherwise the slices
    are made again. Entries whose slices are no longer allocated (closed device) are dropped when a
    new entry is added, so the cache does not grow across devices.
    need_gab=False (a separate prefill [g|a|b] weight is used): w_gab is not sliced (returned None
    unless an earlier call already made it).
    """
    key = id(mega_fused_weight)
    cached = _mega_split_cache.get(key)
    if cached is not None and not _mega_split_entry_live(cached, mega_fused_weight):
        cached = None
    if cached is not None and (cached[2] is not None or not need_gab):
        return cached[1], cached[2]
    if cached is None:
        for k in [k for k, e in _mega_split_cache.items() if not (e[1].is_allocated() and e[0].is_allocated())]:
            del _mega_split_cache[k]
    gab_dim = mega_g_dim + mega_a_dim + mega_b_dim
    w_qkv = cached[1] if cached is not None else mega_fused_weight[:, :mega_qkv_dim]
    w_gab = mega_fused_weight[:, mega_qkv_dim : mega_qkv_dim + gab_dim] if need_gab else None
    _mega_split_cache[key] = (mega_fused_weight, w_qkv, w_gab)
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
    prefill_gab_pad_weight=None,
    prefill_gab_a_off=None,
    prefill_gab_b_off=None,
    m3_qkv_zero_bias=None,
    sp_hooks=None,
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

    prefill_gab_pad_weight: optional PREFILL-only (T > 1, split-projection path) [g | a | 0 | b | 0]
    weight whose a and b column blocks are each padded to a whole tile; a/b are then the
    tile-aligned slices at prefill_gab_a_off / prefill_gab_b_off of its output (no untilize).
    Same values as the [g|a|b] columns of mega_fused_weight. Decode (T == 1) never uses it.

    m3_qkv_zero_bias: optional (QWEN36_M3_ZB, qwen36 tp_common M3 table) bf16 TILE [1, N] DRAM zero
    bias, allocated once at model load and shared across layers. When the M1 S3 program applies, the
    q|k|v in-proj runs as ttnn.linear with this bias (bit-identical to minimal_matmul). None = unchanged.

    sp_hooks: optional dict (qwen36 sequence-parallel prefill, tt/sp_prefill_sc.py QWEN36_SP_EARLY_SEND /
    QWEN36_SP_LATE_RECV; chunk prefill with native_conv1d_fn + chunk_delta_fn only) of callables fired at
    the points the cross-die GDN state transfer needs: "pre_conv"() before the causal conv, "post_conv"(
    new_conv_state) right after it, "pre_scan"() before the chunk delta-rule core, "post_scan"(new_state)
    -> new_state | None right after it (None: the hook consumed the state, the caller skips its write).
    None (default) = no call, identical behavior.
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
    # C2 SGRN: the L1 z|a|0|b|0 in-proj output kept alive for sigmoid_gated_rms_norm (None = flag off / n.a.).
    _c2_gab = None
    # P300 D1 L1-gab path (11x10 grid, T in tpc.R3_T_SET; set below): C2 SGRN there runs with fp32 dest off, and the
    # a/b slices go to L1 for the lean pre-scan (see _prescan_lean).
    _sgrn_p300 = False
    _p300_ab_l1 = False
    _gg_gab = None  # P10_GDNGATE: gab kept alive for ttnn.experimental.gdn_gates (None = flag off / not applicable)

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
        # M1 S3: True when the q|k|v in-proj output below is L1 (freed right after the conv).
        _m1_qkv_l1 = False
        # F6-B: T>1 (chunk prefill) can run two matmuls off the same x instead of one mega matmul
        # + 3 slices of the full [*, D_total] output — qkv comes back tile-native at full width
        # (no slicing at all) and gate/a/b are sliced from a ~2080-wide tensor instead of the full
        # ~8224-wide mega_out. Decode (T==1) always takes the legacy single-matmul path (mega-fused
        # is a bigger win there — one matmul beats two at M=1), matching today's behavior exactly;
        # QWEN36_GDN_SPLIT_PROJ=0 also restores the legacy path at any T.
        if T > 1 and os.environ.get("QWEN36_GDN_SPLIT_PROJ", "1") != "0":
            _gab_pad = prefill_gab_pad_weight is not None
            w_qkv, w_gab = _get_mega_split_weights(
                mega_fused_weight, mega_qkv_dim, mega_g_dim, mega_a_dim, mega_b_dim, need_gab=not _gab_pad
            )
            if _gab_pad:
                w_gab = prefill_gab_pad_weight
            # M1 S3 / S4 (QWEN36_M1_S3 / _S4 = 1; tp_common M1 table): unmasked T == 2048 chunk only.
            # S3: q|k|v in-proj as the H V2_bw8 2D-mcast ttnn.matmul, output L1 interleaved (the
            # tiled KDA conv reads it; freed right after the conv). S4: z|a|0|b|0 in-proj as the same
            # program family, output L1; its slices keep today's DRAM placement. With S3 or S4 on,
            # layer.py writes the attention_norm output (hidden_states) to L1: `_m1_x_l1`. A call that
            # M1 does not switch then keeps its current DRAM output placement (minimal_matmul would
            # otherwise inherit in0's L1 placement), and hidden_states is freed after its last
            # consumer (the z|a|0|b|0 matmul) instead of after the layer (layer.py; that deallocate
            # is then a no-op). All None / False when the flags are off -> the calls below unchanged.
            _m1_ok = mode == "chunk" and valid_len is None
            _m1_grid = _get_mm_grid(device)
            _m1_qkv_pc = (
                tpc.m1_prefill_2d_progcfg("S3", T, hidden_states.shape[-1], w_qkv.shape[-1], _m1_grid)
                if _m1_ok and native_conv1d_fn is not None
                else None
            )
            _m1_gab_pc = (
                tpc.m1_prefill_2d_progcfg("S4", T, hidden_states.shape[-1], w_gab.shape[-1], _m1_grid)
                if _m1_ok and _gab_pad
                else None
            )
            _m1_x_l1 = (
                _m1_ok
                and tpc.m1_gdn_norm_l1(T, _m1_grid)
                and hidden_states.memory_config().buffer_type == ttnn.BufferType.L1
            )
            # I-2 S3 (QWEN36_I2_S3=1, T <= 2048): 2D-mcast ttnn.linear (T2 S3-D008) instead of the
            # minimal_matmul; the output placement is unchanged (mc, or in0's placement when mc is
            # None, as minimal_matmul inherits it). None -> the pre-I-2 minimal_matmul.
            _i2_qkv_pc = tpc.i2_prefill_2d_progcfg(
                "S3", T, hidden_states.shape[-1], w_qkv.shape[-1], _get_mm_grid(device)
            )
            # P300 C / D1 (QWEN36_P300_MM=1, 11x10 grid, T == 1024; tp_common P300 table): (progcfg, out mc)
            # for the q|k|v and z|a|0|b|0 in-projs, or None. Same guards as M1 S3 / S4 (13x10 only, so
            # disjoint): C needs the native / KDA conv (it frees an L1 q|k|v output right after the conv),
            # D1 the padded [g | a | 0 | b | 0] weight (tile-aligned slices).
            _p300_qkv = (
                tpc.p300_prefill_mm(
                    "C", _m1_grid, T, hidden_states.shape[-1], w_qkv.shape[-1], in1_dtype=w_qkv.dtype, ckc=ckc
                )
                if _m1_ok and native_conv1d_fn is not None and _m1_qkv_pc is None
                else None
            )
            _p300_gab = (
                tpc.p300_prefill_mm(
                    "D1", _m1_grid, T, hidden_states.shape[-1], w_gab.shape[-1], in1_dtype=w_gab.dtype, ckc=ckc
                )
                if _m1_ok and _gab_pad and _m1_gab_pc is None
                else None
            )
            # P300 D1 with an L1 output: gate/a/b are then sliced to DRAM (their placement without P300) and
            # gab freed right away, as the M1 S4 branch below does, so no L1 gab slice stays alive through
            # ChunkGdnFused (C4: an L1 gab alive there clashes with its static CBs). C2 SGRN stays M1-only.
            _p300_gab_l1 = _p300_gab is not None and _p300_gab[1] is not None
            if _m1_qkv_pc is not None and m3_qkv_zero_bias is not None:
                # M3 ZB (QWEN36_M3_ZB=1): the same M1 S3 program via ttnn.linear + the shared zero bias
                # (FUSE_BIAS path; N1 N-f: bit-identical to minimal_matmul). Allocated at model load.
                assert (
                    m3_qkv_zero_bias.shape[-1] == w_qkv.shape[-1]
                ), f"M3 ZB: zero bias width {m3_qkv_zero_bias.shape[-1]} != q|k|v in-proj N {w_qkv.shape[-1]}"
                qkv = ttnn.linear(
                    hidden_states,
                    w_qkv,
                    bias=m3_qkv_zero_bias,
                    program_config=_m1_qkv_pc,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )
                _m1_qkv_l1 = True
            elif _m1_qkv_pc is not None:
                assert not tpc.m3_enabled("ZB"), (
                    "QWEN36_M3_ZB=1 but the M1 S3 zero bias was not passed (Qwen36Model allocates it at "
                    "model load; the forward never allocates it)"
                )
                qkv = ttnn.matmul(
                    hidden_states,
                    w_qkv,
                    program_config=_m1_qkv_pc,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )
                _m1_qkv_l1 = True
            elif _p300_qkv is not None:
                # P300 C: 2D-mcast ttnn.linear 11x10 bw16 pcM4 pcN18 1x6 (same [K, N] weight as the
                # minimal_matmul). L1 output (QWEN36_P300_MM_C_L1=1): read by the tiled KDA conv and freed
                # right after it (the M1 S3 handling, _m1_qkv_l1); DRAM (=0): today's placement.
                qkv = ttnn.linear(
                    hidden_states,
                    w_qkv,
                    program_config=_p300_qkv[0],
                    compute_kernel_config=ckc,
                    memory_config=_p300_qkv[1],
                    dtype=ttnn.bfloat16,
                )
                _m1_qkv_l1 = _p300_qkv[1].buffer_type == ttnn.BufferType.L1
            elif _i2_qkv_pc is not None:
                qkv = ttnn.linear(
                    hidden_states,
                    w_qkv,
                    compute_kernel_config=ckc,
                    memory_config=(
                        mc
                        if mc is not None
                        else (ttnn.DRAM_MEMORY_CONFIG if _m1_x_l1 else hidden_states.memory_config())
                    ),
                    program_config=_i2_qkv_pc,
                )
            else:
                qkv = ttnn.experimental.minimal_matmul(
                    hidden_states,
                    w_qkv,
                    config=tpc.prefill_minimal_matmul_config(
                        T, hidden_states.shape[-1], w_qkv.shape[-1], _get_mm_grid(device)
                    ),
                    compute_kernel_config=ckc,
                    memory_config=mc if (mc is not None or not _m1_x_l1) else ttnn.DRAM_MEMORY_CONFIG,
                )
            # P10_GDNGATE (QWEN36_GDN_GATES_OP, code default 0): fused beta/g op, only in the M1 S4 gab branch
            # (gab L1/DRAM [g | a | 0 | b | 0]) of an unmasked-or-masked chunk prefill on the fused FLA path;
            # also on the P300 L1-gab path (_p300_gab_l1), which takes the same gab branch.
            _gg_gab = None
            _gg_op = (
                os.environ.get("QWEN36_GDN_GATES_OP", "0") != "0"
                and (_m1_gab_pc is not None or _p300_gab_l1)
                and A_neg_precomputed is not None
                and chunk_delta_fn is not None
                and mode == "chunk"
                and T > 1
                and _gab_pad
            )
            # C2 SGRN (QWEN36_C2_SGRN=1; tp_common C2 table): only where M1 S4 applies (unmasked chunk,
            # T == 2048) on the fused FLA path with the fused SILU gate. gab (z = columns
            # 0..mega_g_dim-1) is then read in place by sigmoid_gated_rms_norm after ChunkGdnFused: no
            # z slice here and gab stays alive until that op (freed right after it).
            # C2 SGRN is also taken on the P300 D1 L1-gab path (11x10 grid, unmasked chunk with T in tpc.R3_T_SET,
            # e.g. the SP per-die span 1024). The op itself is grid-agnostic (it spreads B*Nv*Mt rows over the
            # device grid). gab placement follows the same C4 / R3 SGRN_GAB_L1 rule as the M1 path (_c2_gab_dram).
            _sgrn_p300 = _p300_gab_l1 and T in tpc.R3_T_SET
            _c2_sgrn = (
                (_m1_gab_pc is not None or _sgrn_p300)
                and tpc.c2_enabled("SGRN")
                and chunk_delta_fn is not None
                and use_gate
                and g_proj_weight is not None
                and os.environ.get("QWEN36_GDN_GATE_FUSED", "1") != "0"
                and os.environ.get("QWEN36_GDN_GATE_CLIP", "0") != "1"
            )
            # C4 (tp_common.c2_sgrn_gab_dram): with SGRN and QWEN36_LAYER_RESID_L1=1, "variant b": the M1 S4
            # program writes gab DRAM interleaved (an L1 gab alive through ChunkGdnFused clashes with its
            # static CBs when the residual stream is also L1). Otherwise gab L1 as before ("variant a").
            _c2_gab_dram = _c2_sgrn and tpc.c2_sgrn_gab_dram()
            if _m1_gab_pc is not None:
                gab = ttnn.matmul(
                    hidden_states,
                    w_gab,
                    program_config=_m1_gab_pc,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG if _c2_gab_dram else ttnn.L1_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                )
            elif _p300_gab is not None:
                # P300 D1: 2D-mcast 11x10 bw16 pcM4 pcN6 1x6; L1 output (QWEN36_P300_MM_D1_L1=1) or mc (=0).
                gab = ttnn.linear(
                    hidden_states,
                    w_gab,
                    memory_config=((ttnn.DRAM_MEMORY_CONFIG if _c2_gab_dram else _p300_gab[1]) if _p300_gab_l1 else mc),
                    compute_kernel_config=ckc,
                    program_config=_p300_gab[0],
                    dtype=ttnn.bfloat16,
                )
            else:
                gab = ttnn.linear(
                    hidden_states,
                    w_gab,
                    memory_config=mc,
                    compute_kernel_config=ckc,
                    program_config=_pc(hidden_states, w_gab),
                )
            if _m1_x_l1:
                # M1: last consumer of the L1 norm output -> free it now (not after the layer).
                ttnn.deallocate(hidden_states)
            if _m1_gab_pc is not None or _p300_gab_l1:
                # M1 S4: gab is L1 ([g | a | 0 | b | 0], _gab_pad is True here; DRAM in C2 SGRN variant b);
                # the three slices go to DRAM, the placement they have without M1 (gab DRAM, slices
                # inherit it). P300 D1 with an L1 gab takes the same branch (_c2_sgrn is False then).
                _gs = list(gab.shape)
                _dram = ttnn.DRAM_MEMORY_CONFIG
                if _c2_sgrn:
                    gate_raw = None
                    _c2_gab = gab
                else:
                    gate_raw = ttnn.slice(gab, [0, 0, 0], [_gs[0], _gs[1], mega_g_dim], memory_config=_dram)
                if _gg_op:
                    # P10_GDNGATE: a/b are read in place from gab by ttnn.experimental.gdn_gates (no slices);
                    # gab stays alive until that op (freed there unless SGRN owns it).
                    _gg_gab = gab
                    a_raw = None
                    b_raw = None
                else:
                    # P300 L1-gab path: the a/b slices go to L1 (read by the lean pre-scan below).
                    _p300_ab_l1 = _p300_gab_l1 and _m1_gab_pc is None
                    _ab_mc = ttnn.L1_MEMORY_CONFIG if _p300_ab_l1 else _dram
                    a_raw = ttnn.slice(
                        gab,
                        [0, 0, prefill_gab_a_off],
                        [_gs[0], _gs[1], prefill_gab_a_off + mega_a_dim],
                        memory_config=_ab_mc,
                    )
                    b_raw = ttnn.slice(
                        gab,
                        [0, 0, prefill_gab_b_off],
                        [_gs[0], _gs[1], prefill_gab_b_off + mega_b_dim],
                        memory_config=_ab_mc,
                    )
                    if not _c2_sgrn:
                        ttnn.deallocate(gab)
            else:
                # gab is laid out g|a|b (columns mega_qkv_dim..end of the original mega weight); g_dim
                # is a tile-width multiple so this begin lands on a tile boundary (tile-native, no
                # untilize). a/b are half a tile each, pulled out together as before.
                gate_raw = gab[:, :, :mega_g_dim]
                if _gab_pad:
                    # [g | a | 0 | b | 0]: a and b each start on a tile boundary -> tile-native slices.
                    a_raw = gab[:, :, prefill_gab_a_off : prefill_gab_a_off + mega_a_dim]
                    b_raw = gab[:, :, prefill_gab_b_off : prefill_gab_b_off + mega_b_dim]
                else:
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
                if sp_hooks is not None and "pre_conv" in sp_hooks:
                    sp_hooks["pre_conv"]()
                if _m1_qkv_l1:
                    # M1 S3: free the L1 in-proj output right after the conv (the DRAM one is freed at
                    # the same point, when `qkv` is rebound). Skipped if a conv output shares its buffer.
                    _m1_qkv_in = qkv
                    qkv, new_fused_conv_state_raw = native_conv1d_fn(_m1_qkv_in, fused_conv_state)
                    _m1_addr = _m1_qkv_in.buffer_address()
                    if all(o.buffer_address() != _m1_addr for o in (qkv if isinstance(qkv, tuple) else (qkv,))):
                        ttnn.deallocate(_m1_qkv_in)
                    del _m1_qkv_in
                else:
                    qkv, new_fused_conv_state_raw = native_conv1d_fn(qkv, fused_conv_state)
                if sp_hooks is not None and "post_conv" in sp_hooks:
                    sp_hooks["post_conv"](new_fused_conv_state_raw)
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
    # R3 FLA_IN_L1 (qwen36 tp_common R3 table): unmasked T in R3_T_SET chunks keep the chain in mc_small (L1).
    _r3_fla_in_l1 = _use_chunk_fn and valid_len is None and T in tpc.R3_T_SET and tpc.r3_enabled("FLA_IN_L1")
    if (
        _use_chunk_fn
        and os.environ.get("QWEN36_GDN_FLA_INPUTS_DRAM", "1" if os.environ.get("QWEN_GDN_PATH") == "fused" else "0")
        != "0"
        and not _r3_fla_in_l1
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
    # Lean pre-scan (P300 L1-gab path: fused chunk prefill, unmasked, T in tpc.R3_T_SET): beta = sigmoid(b)
    # written straight to an fp32 tensor (the fused op's dtype; its internal typecast is then skipped),
    # softplus fused into the dt_bias add as a post-activation, and g cast to fp32 here (6 ops instead of 8).
    # Takes precedence over P5_GATING (QWEN36_GDN_GATE_FUSE) below.
    _prescan_lean = (
        _p300_ab_l1
        and _mega_extracted
        and chunk_delta_fn is not None
        and mode == "chunk"
        and valid_len is None
        and T in tpc.R3_T_SET
        and not allow_neg_eigval
        and A_neg_precomputed is not None
        and mc_small is not None
        and os.environ.get("QWEN36_GDN_GB_LAYOUT", "0") == "0"
    )

    # P5_GATING (QWEN36_GDN_GATE_FUSE=1, default off): cut this chain's op count with existing
    # ttnn fused-activation / output-dtype kwargs, same math:
    #   - beta: BinaryNg ttnn.multiply(b_raw, scale, input_tensor_a_activations=[SIGMOID], dtype=...)
    #     runs sigmoid(b_raw) then the *2.0 (allow_neg_eigval) or *1.0 scale in ONE device op
    #     (folds in the old `if allow_neg_eigval: beta = multiply(beta, 2.0)` step unconditionally,
    #     so it always costs one multiply, not a no-op skip -- verified bit-exact below either way),
    #     with the output dtype set straight to FLOAT32 when the fused-chunk path needs it.
    #   - sp: BinaryNg ttnn.add(a, dt_bias, activations=[softplus_param]) runs the add and the
    #     softplus in ONE op via the SAME UnaryOpType.SOFTPLUS -> softplus_tile() LLK path that
    #     ttnn.softplus itself uses (see unary_op_utils.cpp add_activation_defines /
    #     string_to_unary_with_param): this keeps the x < -5 accuracy fix, it is not a different
    #     implementation.
    #   - g: the existing ttnn.multiply(A_neg, sp) just gets dtype=FLOAT32 added when needed.
    # Where fp32 output isn't needed (QWEN36_GDN_GB_LAYOUT=0, or off the fused-chunk path), dtype
    # is left None (default bf16), matching today's dtype exactly.
    # The old code's typecast block below (~1190) already guards on `beta.dtype != ttnn.float32`
    # / `g.dtype != ttnn.float32`, so when this path already produced FLOAT32 those typecasts are
    # skipped automatically -- no separate edit needed there.
    _gf_enabled = os.environ.get("QWEN36_GDN_GATE_FUSE", "0") != "0"
    _gf_need_fp32 = (
        _gf_enabled
        and _use_chunk_fn
        and mode == "chunk"
        and T > 1
        and os.environ.get("QWEN36_GDN_GB_LAYOUT", "0") != "0"
    )
    _gf_dtype = ttnn.float32 if _gf_need_fp32 else None
    _gf_beta_out_mc = mc if _gf_need_fp32 else mc_small
    _gf_beta_scale = 2.0 if allow_neg_eigval else 1.0
    _gf_softplus_param = ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)  # matches ttnn.softplus defaults

    def _gf_beta_from_b_raw(braw):
        return ttnn.multiply(
            braw,
            _gf_beta_scale,
            input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID],
            dtype=_gf_dtype,
            memory_config=_gf_beta_out_mc,
        )

    # P10_GDNGATE (QWEN36_GDN_GATES_OP=1, code default 0): ONE op (ttnn.experimental.gdn_gates) makes beta and g
    # in fp32 straight from the a and b columns of gab, bit-identical to the chain below (bf16 sigmoid / add +
    # softplus / multiply, then the fp32 widening the FLA op does itself). Only where the M1 S4 gab branch above
    # kept gab alive (_gg_gab) and the fused FLA op consumes beta/g.
    if _gg_gab is not None:
        beta, g = ttnn.experimental.gdn_gates(
            _gg_gab,
            dt_bias,
            A_neg_precomputed,
            a_col_offset=prefill_gab_a_off,
            b_col_offset=prefill_gab_b_off,
            num_heads=mega_a_dim,
            beta_scale=2.0 if allow_neg_eigval else 1.0,
            memory_config=mc_small,
        )
        if not _c2_sgrn:
            ttnn.deallocate(_gg_gab)
        _gg_gab = None
    else:
        if _prescan_lean:
            a = a_raw
            beta = ttnn.allocate_tensor_on_device(
                ttnn.TensorSpec(list(b_raw.shape), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=mc_small.buffer_type),
                b_raw.device(),
            )
            ttnn.sigmoid(b_raw, memory_config=mc_small, output_tensor=beta)
        elif _mega_extracted:
            a = a_raw
            if _gf_enabled:
                beta = _gf_beta_from_b_raw(b_raw)
            else:
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
            if _gf_enabled:
                beta = _gf_beta_from_b_raw(b_raw)
            else:
                beta = ttnn.sigmoid(b_raw, memory_config=mc_small)
        else:
            b_raw = ttnn.linear(
                hidden_states,
                b_proj_weight,
                memory_config=mc,
                compute_kernel_config=ckc,
                program_config=_pc(hidden_states, b_proj_weight),
            )
            if _gf_enabled:
                beta = _gf_beta_from_b_raw(b_raw)
            else:
                beta = ttnn.sigmoid(b_raw, memory_config=mc_small)
            a = ttnn.linear(
                hidden_states,
                a_proj_weight,
                memory_config=mc,
                compute_kernel_config=ckc,
                program_config=_pc(hidden_states, a_proj_weight),
            )
        if allow_neg_eigval and not _gf_enabled:
            beta = ttnn.multiply(beta, 2.0, memory_config=mc_small)
        if _prescan_lean:
            sp = ttnn.add(
                a,
                dt_bias,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
                memory_config=mc_small,
            )
            _g16 = ttnn.multiply(A_neg_precomputed, sp, memory_config=mc_small)
            ttnn.deallocate(sp)
            g = ttnn.typecast(_g16, ttnn.float32, memory_config=mc_small)
            ttnn.deallocate(_g16)
        else:
            if _gf_enabled:
                sp = ttnn.add(a, dt_bias, activations=[_gf_softplus_param], memory_config=mc_small)
            else:
                a_biased = ttnn.add(a, dt_bias, memory_config=mc_small)
                sp = ttnn.softplus(a_biased, memory_config=mc_small)
            _gf_g_out_mc = mc if _gf_need_fp32 else mc_small
            if A_neg_precomputed is not None:
                g = ttnn.multiply(A_neg_precomputed, sp, dtype=_gf_dtype, memory_config=_gf_g_out_mc)
            else:
                A = ttnn.exp(A_log, memory_config=mc_small)
                A_neg = ttnn.neg(A, memory_config=mc_small)
                g = ttnn.multiply(A_neg, sp, dtype=_gf_dtype, memory_config=_gf_g_out_mc)

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
            if sp_hooks is not None and "pre_scan" in sp_hooks:
                sp_hooks["pre_scan"]()
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
            if sp_hooks is not None and "post_scan" in sp_hooks:
                new_state = sp_hooks["post_scan"](new_state)
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
    if _c2_gab is not None:
        # C2 SGRN (QWEN36_C2_SGRN=1): the post-scan chain (z slice, typecast, per-head rms_norm,
        # nlp_concat_heads, multiply with SILU gate) as ONE op: rms_norm(o per head; fp32 in) * weight *
        # silu(z), z read in place from gab columns [0, Nv*Dv) (gate_col_offset_tiles 0); bf16 out in the
        # placement the gate multiply wrote (mc_scan); gab freed right after.
        assert _o_head_major and mega_g_dim == num_v_heads * head_v_dim, (
            f"C2 SGRN: expected the head-major fused FLA output and z width {mega_g_dim} == Nv*Dv "
            f"{num_v_heads * head_v_dim}"
        )
        if len(o.shape) != 3:
            o = ttnn.reshape(o, [B * num_v_heads, T, head_v_dim])  # metadata only
        if "SGRN" not in _C2_LOGGED:
            _C2_LOGGED.add("SGRN")
            print(
                f"[C2] QWEN36_C2_SGRN=1 active: sigmoid_gated_rms_norm(o {list(o.shape)} {o.dtype} "
                f"{o.memory_config().buffer_type}, gab {list(_c2_gab.shape)} {_c2_gab.dtype} "
                f"{_c2_gab.memory_config().buffer_type}, weight {list(o_norm_weight.shape)} {o_norm_weight.dtype} "
                f"{o_norm_weight.layout}, H={num_v_heads}, epsilon={norm_eps}, silu, gate_col_offset_tiles=0, "
                f"out bf16 {mc_scan}, HiFi4 approx=F fp32_dest={'T' if tpc.sgrn_compute_config(_sgrn_p300).fp32_dest_acc_en else 'F'} packer_l1_acc=F, kernel_variant={tpc.sgrn_kernel_variant()}) gab placement variant "
                f"{'b (DRAM)' if _c2_gab.memory_config().buffer_type == ttnn.BufferType.DRAM else 'a (L1)'}",
                flush=True,
            )
        o = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o,
            _c2_gab,
            o_norm_weight,
            num_v_heads,
            epsilon=norm_eps,
            memory_config=mc_scan,
            # P300 L1-gab path: fp32 dest off (HiFi4), ~16% faster (48.8 -> 41.2 us per GDN layer at T=1024).
            # (QWEN36_SGRN_FP32_DEST=1 turns fp32 dest back on there, tpc.sgrn_compute_config.)
            compute_kernel_config=tpc.sgrn_compute_config(_sgrn_p300),
            output_dtype=ttnn.bfloat16,
            gate_activation="silu",
            gate_col_offset_tiles=0,
            kernel_variant=tpc.sgrn_kernel_variant(),
        )
        ttnn.deallocate(_c2_gab)
        _c2_gab = None
        # already [B, T, Nv*Dv]; skip the reshape below
    elif _o_head_major:
        # Fused-chunk output is head-major [B*Nv, T, Dv]: norm per head in place, fold heads with the
        # TILE-native concat, and gate on the flat [B, T, Nv*Dv] tensor. Same math as
        # rms_norm_gated_ttnn without the token<->head relayouts and the two copying reshapes.
        o = ttnn.reshape(o, [B, num_v_heads, T, head_v_dim])  # metadata only
        # scan emits fp32; the rest of the layer is bf16 — cast once here instead of running
        # norm/concat/gate/clip at 2x the bytes
        # F10B item B: this typecast + per-head rms_norm + nlp_concat_heads is the "post-scan chain"
        # group -- mc_scan (independent L1 policy), not mc.
        # step2 (2026-09-23): default flipped "1" -> "0" per explicit user scope-change (this
        # worktree's default behavior should exclude the norm-output-dtype change; the step2 FLA
        # L1 fix validation runs with this OFF). Set to "1" to re-enable the bf16-out RMSNorm path.
        if os.environ.get("QWEN36_GDN_POST_NORM_BF16OUT", "0") != "0":
            # rms_norm's LayerNorm primitive now honors an explicit output dtype (fp32 in, bf16
            # out), so the standalone typecast (~47 us/2048-token chunk) is folded into the norm's
            # pack stage instead of running as a separate op. Requires the ttnn dtype-plumbing
            # patch in scratchpad/rmsnorm_out_dtype.patch.
            o = ttnn.rms_norm(o, weight=o_norm_weight, epsilon=norm_eps, memory_config=mc_scan, dtype=ttnn.bfloat16)
        else:
            o = ttnn.typecast(o, ttnn.bfloat16, memory_config=mc_scan)
            o = ttnn.rms_norm(o, weight=o_norm_weight, epsilon=norm_eps, memory_config=mc_scan)
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=mc_scan, head_split=True)  # [B, 1, T, Nv*Dv]
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
        memory_config=(tpc.resid_hs_mc() if (tpc.resid_hs_active() and T == tpc.RESID_HS_T) else mc_outproj),
        compute_kernel_config=ckc,
        program_config=_pc(o, o_proj_weight),
        **({"dtype": ttnn.bfloat8_b} if (T > 1 and os.environ.get("QWEN36_ACT_BF8_RESID", "0") == "1") else {}),
    )

    return o, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv_state
