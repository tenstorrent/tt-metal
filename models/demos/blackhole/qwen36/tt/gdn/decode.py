# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent / chunked DeltaNet forward (the `forward` dispatch).

Behavior-preserving extraction of the original `Qwen36GatedDeltaNet.forward` body.
Operates on the gdn instance: reads weights from `gdn.weights`, config dims from
`gdn.cfg`, mirrored scalar attrs + runtime state from `gdn`. Every ttnn op,
memory_config, and the `gated_deltanet_forward_ttnn` kwargs are verbatim.
"""
import functools
import os

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.state import init_recurrent_state, split_fused_conv_state
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn


def _gdn_wy_inverse_kwargs():
    """QWEN36_GDN_WYINV (PR #57445 selector for the fused FLA prefill): unset/"auto" -> wy_inverse not passed
    (op default AUTO); "horner" -> ttnn.ChunkGdnWyInverse.HORNER; "sfpu" -> ttnn.ChunkGdnWyInverse.SFPU."""
    name = os.environ.get("QWEN36_GDN_WYINV", "auto").strip().lower()
    if name in ("", "auto"):
        return {}
    if name == "horner":
        return {"wy_inverse": ttnn.ChunkGdnWyInverse.HORNER}
    if name == "sfpu":
        return {"wy_inverse": ttnn.ChunkGdnWyInverse.SFPU}
    raise ValueError(f"QWEN36_GDN_WYINV={name!r}: expected auto, horner or sfpu")


def _gdn_state_inplace_enabled():
    """QWEN36_GDN_STATE_INPLACE (default "0"): on the traced chunked prefill (gdn._chunk_inplace_state), the
    fused FLA op and the tiled KDA conv op write the final recurrent state / new conv state straight into
    the persistent state buffers (final_state_output / conv_state_output), which removes the two per-layer
    ttnn.copy write-backs below. "0" = the current path (fresh state tensors + copy), unchanged."""
    return os.environ.get("QWEN36_GDN_STATE_INPLACE", "0") == "1"


def _same_buffer(a, b):
    return a is not None and b is not None and a.buffer_address() == b.buffer_address()


def recurrent_forward(gdn, x, mode="recurrent", chunk_size=None, valid_len=None):
    """Non-kernel GDN forward. mode='chunk' (prefill, may delegate to the prefill kernel) or
    'recurrent' (single-token decode). Reads weights/state/dims off the gdn instance; updates
    gdn's recurrent + conv state in place or by reassignment per the trace-capture flags.

    valid_len: for fixed-bucket masked prefill — x is right-padded to T but only the first
    valid_len positions are real (see gated_deltanet_forward_ttnn). None = no padding."""
    w = gdn.weights
    if chunk_size is None:
        chunk_size = gdn.long_prefill_chunk_size if mode == "chunk" else 64

    if gdn.recurrent_state is None:
        shape = x.shape
        batch_size = shape[0] if len(shape) == 3 else 1
        init_recurrent_state(gdn, batch_size)

    T = x.shape[1]

    # After prefill, fuse separate conv states into one for efficient decode
    if T == 1 and gdn.fused_conv_state is None and gdn.conv_state_q is not None:
        gdn.fused_conv_state = ttnn.concat([gdn.conv_state_q, gdn.conv_state_k, gdn.conv_state_v], dim=2)
        gdn.fused_conv_state = ttnn.to_layout(gdn.fused_conv_state, ttnn.TILE_LAYOUT)
        split_fused_conv_state(gdn)
        # Fused GDN decode (INT-2): the end-of-prefill conv_hist rebuild saw no fused_conv_state (zero
        # history); rebuild it from the conv state that exists now. Device ops only (no host reads).
        # Eager paths only: the traced flows bind fused_conv_state, so this branch never runs there.
        if getattr(gdn, "_decode_fused", False) and gdn.fused_conv_state.shape[0] == 1:
            gdn.refresh_conv_hist()

    # QWEN36_GDN_DECODE_FUSED=2: single-token decode through the fused op (gdn/decode_fused.py).
    if T == 1 and mode == "recurrent" and getattr(gdn, "_decode_fused", False):
        from models.demos.blackhole.qwen36.tt.gdn import decode_fused as _df

        if _df.fused_decode_applicable(gdn, x):
            return _df.fused_decode_forward(gdn, x)

    # Chunk-parallel prefill via the C++ gated_delta_attn_seq kernel (float32, chunk_size=128).
    seq_masks = w.chunk_seq_masks_long

    # QWEN36_GDN_STATE_INPLACE: the ops write the state into the persistent buffers (traced chunked prefill
    # only). The recurrent buffer must be fp32 (the fused FLA op's state dtype); else that write-back stays.
    _state_inplace = mode == "chunk" and gdn._chunk_inplace_state and _gdn_state_inplace_enabled()
    _rec_out = (
        gdn.recurrent_state
        if _state_inplace and gdn.recurrent_state is not None and gdn.recurrent_state.dtype == ttnn.float32
        else None
    )
    _conv_out = gdn.fused_conv_state if _state_inplace else None

    chunk_delta_fn = None
    if mode == "chunk" and os.environ.get("QWEN36_GDN_FUSED_PREFILL", "1") != "0":
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import chunk_gated_delta_rule_fused_adapter

        chunk_delta_fn = functools.partial(
            chunk_gated_delta_rule_fused_adapter,
            const_tiles=getattr(gdn, "_fused_const_tiles", None),
            program_config=getattr(gdn, "gdn_program_config", None),
            **_gdn_wy_inverse_kwargs(),
            # Single-device path only (the TP path keeps the in-kernel norm and the default decay chain):
            # qk_prenormed when the KDA conv produced normalized q/k (gated_deltanet.py _conv_qk_prenormed);
            # decay_sfpu on the P300 SP dies with a pinned fused geometry (gated_deltanet.py _fused_sp_die).
            **({"qk_prenormed": True} if getattr(gdn, "_conv_qk_prenormed", False) else {}),
            **({"decay_sfpu": True} if getattr(gdn, "_fused_sp_die", False) else {}),
            **({"final_state_out": _rec_out} if _rec_out is not None else {}),
        )

    native_conv1d_fn = getattr(gdn, "_native_conv1d_fn", None) if mode == "chunk" else None
    if _conv_out is not None and getattr(native_conv1d_fn, "accepts_conv_state_out", False):
        native_conv1d_fn = functools.partial(native_conv1d_fn, conv_state_out=_conv_out)

    output, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv = gated_deltanet_forward_ttnn(
        hidden_states=x,
        q_proj_weight=w.q_proj_weight,
        k_proj_weight=w.k_proj_weight,
        v_proj_weight=w.v_proj_weight,
        a_proj_weight=w.a_proj_weight,
        b_proj_weight=w.b_proj_weight,
        o_proj_weight=w.o_proj_weight,
        q_conv_weight=w.q_conv_weight,
        k_conv_weight=w.k_conv_weight,
        v_conv_weight=w.v_conv_weight,
        q_conv_bias=w.q_conv_bias,
        k_conv_bias=w.k_conv_bias,
        v_conv_bias=w.v_conv_bias,
        A_log=w.A_log,
        dt_bias=w.dt_bias,
        o_norm_weight=w.o_norm_weight,
        g_proj_weight=w.g_proj_weight,
        num_heads=gdn.num_heads,
        num_v_heads=gdn.num_v_heads,
        head_k_dim=gdn.head_k_dim,
        head_v_dim=gdn.head_v_dim,
        conv_kernel_size=gdn.conv_kernel_size,
        use_gate=True,
        norm_eps=gdn.norm_eps,
        device=gdn.device,
        recurrent_state=gdn.recurrent_state,
        conv_state_q=gdn.conv_state_q,
        conv_state_k=gdn.conv_state_k,
        conv_state_v=gdn.conv_state_v,
        mode=mode,
        chunk_size=chunk_size,
        q_weight_taps=w.q_weight_taps,
        k_weight_taps=w.k_weight_taps,
        v_weight_taps=w.v_weight_taps,
        q_bias_dev=w.q_bias_dev,
        k_bias_dev=w.k_bias_dev,
        v_bias_dev=w.v_bias_dev,
        qkv_proj_weight=w.qkv_proj_weight,
        q_dim=gdn.cfg.q_dim,
        k_dim=gdn.cfg.k_dim,
        compute_kernel_config=gdn.compute_kernel_config_decode if mode == "recurrent" else gdn.compute_kernel_config,
        A_neg_precomputed=w.A_neg,
        fused_conv_weight_taps=w.fused_conv_weight_taps,
        fused_conv_bias_dev=w.fused_conv_bias_dev,
        fused_conv_state=gdn.fused_conv_state,
        fused_conv_state_split=getattr(gdn, "split_conv_state", None),
        ab_proj_weight=w.ab_proj_weight,
        mega_fused_weight=w.mega_fused_weight,
        mega_qkv_dim=w.mega_qkv_dim,
        mega_a_dim=w.mega_a_dim,
        mega_b_dim=w.mega_b_dim,
        mega_g_dim=w.mega_g_dim,
        use_inplace_state=gdn.use_inplace_state,
        chunk_seq_masks=seq_masks,
        valid_len=valid_len,
        chunk_delta_fn=chunk_delta_fn,
        prefill_progcfg_fn=getattr(gdn, "_prefill_progcfg_fn", None),
        decode_progcfg_fn=getattr(gdn, "_decode_progcfg_fn", None),
        native_conv1d_fn=native_conv1d_fn,
        # I-1 P5 (QWEN36_I1_P5): prefill-only tile-padded [g|a|0|b|0] weight (None = off).
        **(
            dict(
                prefill_gab_pad_weight=w.prefill_gab_pad_weight,
                prefill_gab_a_off=w.prefill_gab_a_off,
                prefill_gab_b_off=w.prefill_gab_b_off,
            )
            if mode == "chunk" and getattr(w, "prefill_gab_pad_weight", None) is not None
            else {}
        ),
        # M3 ZB (QWEN36_M3_ZB): shared zero bias for the M1 S3 q|k|v in-proj, set by Qwen36Model at load.
        **(
            dict(m3_qkv_zero_bias=gdn._m3_qkv_zero_bias)
            if mode == "chunk" and getattr(gdn, "_m3_qkv_zero_bias", None) is not None
            else {}
        ),
        # SP prefill (tt/sp_prefill_sc.py QWEN36_SP_EARLY_SEND / QWEN36_SP_LATE_RECV): in-mixer cross-die
        # state transfer hooks, set on the layer's GDN by the SP harness around its forward (None = off).
        **(dict(sp_hooks=gdn._sp_hooks) if mode == "chunk" and getattr(gdn, "_sp_hooks", None) else {}),
    )

    if chunk_delta_fn is not None and new_state is not None:
        # Fused adapter returns fp32 state; the seq adapter returns bf16 unless
        # QWEN_GDN_FP32_STATE=1. Match whichever convention the destination expects: the
        # preallocated recurrent_state buffer's dtype when writing in place below (ttnn.copy
        # writes into that fixed-dtype buffer), else the seq adapter's env-selected convention.
        if gdn._chunk_inplace_state and mode == "chunk" and gdn.recurrent_state is not None:
            target_dtype = gdn.recurrent_state.dtype
        else:
            target_dtype = ttnn.float32 if os.environ.get("QWEN_GDN_FP32_STATE") == "1" else ttnn.bfloat16
            if getattr(gdn, "_decode_fused", False) and new_state.shape[0] == 1:
                target_dtype = ttnn.float32  # fused decode keeps the whole GDN state FP32
        if new_state.dtype != target_dtype:
            new_state = ttnn.typecast(new_state, target_dtype)

    if gdn._chunk_inplace_state and mode == "chunk":
        # Per-chunk traced-prefill replay: write state into the persistent external
        # buffers in place so it carries across execute_trace() calls. gdn.recurrent_state
        # and gdn.fused_conv_state keep pointing at the same (baked) buffer addresses.
        # (new_state None: an SP post_scan hook already wrote gdn.recurrent_state.)
        # QWEN36_GDN_STATE_INPLACE: an op that wrote straight into the persistent buffer returned that
        # buffer itself -> no copy, and no deallocate (it is the persistent state).
        if new_state is not None and not (_state_inplace and _same_buffer(new_state, gdn.recurrent_state)):
            if list(new_state.shape) != list(gdn.recurrent_state.shape):
                new_state = ttnn.reshape(new_state, list(gdn.recurrent_state.shape))
            ttnn.copy(new_state, gdn.recurrent_state)
            ttnn.deallocate(new_state)
        if new_fused_conv is not None and not isinstance(new_fused_conv, list):
            if not (_state_inplace and _same_buffer(new_fused_conv, gdn.fused_conv_state)):
                if new_fused_conv.layout != ttnn.TILE_LAYOUT:
                    new_fused_conv = ttnn.to_layout(new_fused_conv, ttnn.TILE_LAYOUT)
                ttnn.copy(new_fused_conv, gdn.fused_conv_state)
                ttnn.deallocate(new_fused_conv)
        return output

    if (
        mode == "chunk"
        and chunk_delta_fn is not None
        and new_state is not None
        and new_state.memory_config().buffer_type == ttnn.BufferType.L1
    ):
        # R3 O_L1: the fused FLA op wrote the final state to L1. The in-place (traced) branch above copies it
        # into the persistent state and frees it; here it would stay alive as gdn.recurrent_state -> DRAM.
        new_state = ttnn.to_memory_config(new_state, ttnn.DRAM_MEMORY_CONFIG)
    gdn.recurrent_state = new_state
    if isinstance(new_fused_conv, list):
        gdn.split_conv_state = new_fused_conv
    elif new_fused_conv is not None:
        gdn.fused_conv_state = new_fused_conv
    else:
        gdn.conv_state_q = new_conv_q
        gdn.conv_state_k = new_conv_k
        gdn.conv_state_v = new_conv_v
    return output
