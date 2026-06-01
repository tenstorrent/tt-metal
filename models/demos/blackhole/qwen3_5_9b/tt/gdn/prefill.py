# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""On-device GDN recurrence prefill kernel forward.

Behavior-preserving extraction of the original
`Qwen35GatedDeltaNet.forward_prefill_kernel` body. Reads weights/kernel-constants
from `gdn.weights`, config dims from `gdn.cfg`, runtime state from `gdn`. Every
ttnn op, memory_config, and the `gdn_prefill_fused` call (incl. `v_split`) is verbatim.
"""
import os

# Install the experimental backend on sys.path BEFORE importing its ops.
import models.demos.blackhole.qwen3_5_9b.tt.gdn._experimental_path  # noqa: F401
import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import rms_norm_gated_ttnn


def prefill_kernel_forward(gdn, x, prefill_output=None):
    """Prefill using on-device GDN recurrence kernel.

    Replaces the chunked delta rule with a single kernel dispatch that
    processes all tokens sequentially on-device. State stays in L1 across
    tokens — no CPU round-trips, no chunking approximation.

    Input:  x [B, T, hidden_size]
            prefill_output: optional pre-allocated [num_pairs * T, 1, Dv] buffer
              in DRAM, TILE_LAYOUT, bfloat16. When None, allocated per call.
              Pass a persistent buffer for trace-capture compatibility.
    Output: [B, T, hidden_size]
    """
    w = gdn.weights
    B = x.shape[0]
    T = x.shape[1]
    Nv = gdn.num_v_heads
    Nk = gdn.num_heads
    Dk = gdn.head_k_dim
    Dv = gdn.head_v_dim
    q_dim = gdn.cfg.q_dim
    k_dim = gdn.cfg.k_dim
    v_dim = gdn.cfg.v_dim
    qkv_dim = q_dim + k_dim + v_dim
    num_pairs = B * Nv
    K = gdn.conv_kernel_size

    # Match old path: L1 for short sequences, DRAM for long.
    # Kernel inputs must be in DRAM (NOC reads), but projections/conv use mc.
    mc = ttnn.DRAM_MEMORY_CONFIG if T > 512 else None
    ckc = gdn.compute_kernel_config

    # ---- 1. Projections (single mega-fused matmul for QKV+a+b+g) ----
    mega_out = ttnn.linear(x, w.mega_fused_weight, memory_config=mc, compute_kernel_config=ckc)
    # Split: QKV | a | b | g
    qkv = mega_out[:, :, : w.mega_qkv_dim]
    qkv = ttnn.to_layout(qkv, ttnn.TILE_LAYOUT)
    a_fused = mega_out[:, :, w.mega_qkv_dim : w.mega_qkv_dim + w.mega_a_dim]
    a_fused = ttnn.to_layout(a_fused, ttnn.TILE_LAYOUT)
    b_fused = mega_out[:, :, w.mega_qkv_dim + w.mega_a_dim : w.mega_qkv_dim + w.mega_a_dim + w.mega_b_dim]
    b_fused = ttnn.to_layout(b_fused, ttnn.TILE_LAYOUT)
    gate_raw = mega_out[:, :, w.mega_qkv_dim + w.mega_a_dim + w.mega_b_dim :]
    gate_raw = ttnn.to_layout(gate_raw, ttnn.TILE_LAYOUT)
    ttnn.deallocate(mega_out)

    # ---- 2. Fused causal conv1d on QKV ----
    if gdn.fused_conv_state is not None:
        x_padded = ttnn.concat([gdn.fused_conv_state, qkv], dim=1, memory_config=mc)
    elif gdn.conv_state_q is not None:
        fused_state = ttnn.concat([gdn.conv_state_q, gdn.conv_state_k, gdn.conv_state_v], dim=2)
        fused_state = ttnn.to_layout(fused_state, ttnn.TILE_LAYOUT)
        x_padded = ttnn.concat([fused_state, qkv], dim=1, memory_config=mc)
    else:
        pad = ttnn.zeros(
            [B, K - 1, qkv_dim],
            device=gdn.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
        )
        x_padded = ttnn.concat([pad, qkv], dim=1, memory_config=mc)

    # Save new conv state: last K-1 tokens from padded (pre-conv, pre-silu)
    total_len = (K - 1) + T
    start = total_len - (K - 1)
    new_fused_conv = x_padded[:, start:, :]
    new_fused_conv = ttnn.to_layout(new_fused_conv, ttnn.TILE_LAYOUT)
    ttnn.deallocate(qkv)

    # FIR conv1d: sum of shifted slices weighted by conv taps
    conv_out = None
    for j in range(K):
        x_slice = x_padded[:, j : j + T]
        x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)
        term = ttnn.multiply(x_slice, w.fused_conv_weight_taps[j], memory_config=mc)
        conv_out = term if conv_out is None else ttnn.add(conv_out, term, memory_config=mc)
    ttnn.deallocate(x_padded)

    if w.fused_conv_bias_dev is not None:
        conv_out = ttnn.add(conv_out, w.fused_conv_bias_dev, memory_config=mc)

    conv_out = ttnn.silu(conv_out, memory_config=mc)
    # conv_out: [B, T, qkv_dim] — post-conv+silu

    # ---- 3. Prepare state for kernel ----
    # Reshape: [B, Nv, Dk, Dv] → [B*Nv, Dk, Dv] for kernel tile addressing
    # State is already TILE_LAYOUT + DRAM from _init_recurrent_state or prior kernel write.
    state_3d = ttnn.reshape(gdn.recurrent_state, [num_pairs, Dk, Dv])

    # Output buffer: caller-supplied (trace-safe) or fresh per call.
    owns_prefill_output = prefill_output is None
    if owns_prefill_output:
        prefill_output = ttnn.zeros(
            [num_pairs * T, 1, Dv],
            device=gdn.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Ensure kernel inputs are in DRAM for NOC reads (may be in L1 when mc=None)
    if mc != ttnn.DRAM_MEMORY_CONFIG:
        conv_out = ttnn.to_memory_config(conv_out, ttnn.DRAM_MEMORY_CONFIG)
        a_fused = ttnn.to_memory_config(a_fused, ttnn.DRAM_MEMORY_CONFIG)
        b_fused = ttnn.to_memory_config(b_fused, ttnn.DRAM_MEMORY_CONFIG)

    # ---- 4. Call prefill kernel ----
    # GDN_PREFILL_V_SPLIT env var controls V-dim parallelism: each pair's V-tiles
    # split across this many cores. Default 1 (32 active cores). Setting 4 uses
    # 128 cores for the prefill kernel — only meaningful when num_pairs * v_split
    # ≤ 130 (P150 worker grid).
    v_split = int(os.environ.get("GDN_PREFILL_V_SPLIT", "1"))
    gdn_prefill_fused(
        conv_out,
        a_fused,
        b_fused,
        w.kernel_neg_exp_A,
        w.kernel_dt_bias,
        w.kernel_norm_w,
        w.kernel_scale_tt,
        w.kernel_rms_scale_tt,
        w.kernel_rms_eps_tt,
        state_3d,
        prefill_output,
        num_pairs=num_pairs,
        num_tokens=T,
        Nv_TP=Nv,
        Nk_TP=Nk,
        repeat_factor=Nv // Nk,
        key_dim_tp=q_dim,
        v_split=v_split,
    )
    ttnn.deallocate(conv_out)
    ttnn.deallocate(a_fused)
    ttnn.deallocate(b_fused)

    # ---- 5. Update recurrent state ----
    gdn.recurrent_state = ttnn.reshape(state_3d, [B, Nv, Dk, Dv])

    # ---- 6. Reshape kernel output ----
    # Flat [num_pairs * T, 1, Dv] → [B, Nv, T, Dv] → transpose → [B, T, Nv, Dv]
    out_4d = ttnn.reshape(prefill_output, [B, Nv, T, Dv])
    if owns_prefill_output:
        ttnn.deallocate(prefill_output)
    out_4d = ttnn.transpose(out_4d, 1, 2)  # [B, T, Nv, Dv]

    # ---- 7. Gated RMS norm with z ----
    gate = ttnn.reshape(gate_raw, [B, T, Nv, Dv])
    o = rms_norm_gated_ttnn(out_4d, gate, w.o_norm_weight, eps=gdn.norm_eps, memory_config=mc)
    ttnn.deallocate(out_4d)
    ttnn.deallocate(gate_raw)

    # ---- 8. Reshape and output projection ----
    o = ttnn.clip(o, min=-1e4, max=1e4)
    o = ttnn.reshape(o, [B, T, Nv * Dv])
    if mc is not None:
        o = ttnn.to_memory_config(o, mc)
    o = ttnn.linear(o, w.o_proj_weight, memory_config=mc, compute_kernel_config=ckc)

    # ---- 9. Update conv state for next chunk / decode ----
    # new_fused_conv is already TILE_LAYOUT (set where it was sliced above).
    # Don't split here — splitting is deferred to first decode call
    # (either via prefill_paged post-processing or lazy on first T=1 forward).
    if gdn.use_inplace_state and gdn.fused_conv_state is not None:
        # Trace-safe path: write into the persistent buffer so its address is stable.
        ttnn.copy(new_fused_conv, gdn.fused_conv_state)
        ttnn.deallocate(new_fused_conv)
    else:
        gdn.fused_conv_state = new_fused_conv
    gdn.conv_state_q = None
    gdn.conv_state_k = None
    gdn.conv_state_v = None

    return o
