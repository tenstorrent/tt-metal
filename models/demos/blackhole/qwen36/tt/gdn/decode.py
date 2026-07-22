# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent / chunked DeltaNet forward (the `forward` dispatch).

Behavior-preserving extraction of the original `Qwen36GatedDeltaNet.forward` body.
Operates on the gdn instance: reads weights from `gdn.weights`, config dims from
`gdn.cfg`, mirrored scalar attrs + runtime state from `gdn`. Every ttnn op,
memory_config, and the `gated_deltanet_forward_ttnn` kwargs are verbatim.
"""
import ttnn
from models.demos.blackhole.qwen36.tt.gdn.state import init_recurrent_state, split_fused_conv_state
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn


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

    # Chunk-parallel prefill via the C++ gated_delta_attn_seq kernel (float32, chunk_size=128).
    seq_masks = w.chunk_seq_masks_long

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
    )

    if gdn._chunk_inplace_state and mode == "chunk":
        # Per-chunk traced-prefill replay: write state into the persistent external
        # buffers in place so it carries across execute_trace() calls. gdn.recurrent_state
        # and gdn.fused_conv_state keep pointing at the same (baked) buffer addresses.
        if list(new_state.shape) != list(gdn.recurrent_state.shape):
            new_state = ttnn.reshape(new_state, list(gdn.recurrent_state.shape))
        ttnn.copy(new_state, gdn.recurrent_state)
        ttnn.deallocate(new_state)
        if new_fused_conv is not None and not isinstance(new_fused_conv, list):
            if new_fused_conv.layout != ttnn.TILE_LAYOUT:
                new_fused_conv = ttnn.to_layout(new_fused_conv, ttnn.TILE_LAYOUT)
            ttnn.copy(new_fused_conv, gdn.fused_conv_state)
            ttnn.deallocate(new_fused_conv)
        return output

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
