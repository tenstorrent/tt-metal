# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Coherent opt-in compact GDN packed producer/consumer, no default wiring.

Importing this module uses only the Python standard library. Device calls are
lazy and reserved for the parent's serialized hardware test. No host readback,
runtime output cache, policy mutation, or state publication between chunks.
"""


def flat_packed_gdn(layer, packed, state, *, batch, seq, cores=None):
    """Consume [1,B*S,4160]; preserve compact Z and output projection rows.

    Q/K/V and the small beta/g scalars alone acquire per-user padded sequence
    axes. The existing FP32 chunk recurrence and normalization are unchanged.
    The scheduler still owns logical chunking, list/resume, and BF16 publication.
    """
    import ttnn

    assert layer.kind == "linear_attention" and batch > 1 and seq > 1
    assert tuple(packed.shape) == (1, batch * seq, 4160)
    assert packed.dtype == ttnn.bfloat16 and packed.layout == ttnn.TILE_LAYOUT
    assert tuple(state.conv.shape) == (batch, 3, 2560)
    assert tuple(state.recurrent.shape) == (batch, 12, 128, 128)
    assert state.recurrent.dtype == ttnn.float32, "Caller must retain live FP32 state across prefill chunks"
    from .tiled_prefill_conv_candidate import tiled_prefill_conv

    q, k, v = tiled_prefill_conv(
        packed,
        state.conv,
        layer.taps,
        batch=batch,
        seq=seq,
        cores=cores,
        block_ct=layer.policy.get("prefill_conv_block_ct", 8),
    )
    beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, 4096:4108]), (batch, seq, 12))
    g = ttnn.reshape(layer._decay(packed[:, :, 4128:4140]), (batch, seq, 12))
    padded = ((seq + 31) // 32) * 32
    if seq != padded:
        beta, g = (ttnn.pad(t, ((0, 0), (0, padded - seq), (0, 0)), value=0.0) for t in (beta, g))
    out, updated = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state.recurrent,
        output_final_state=True,
        chunk_size=32,
        num_key_heads=4,
        output_head_major=True,
        eye=layer.eye,
        tril=layer.lower,
        ones=layer.ones,
        masks=layer.masks,
        compute_kernel_config=layer.ckc,
    )
    # FP32 to FP32 copy matches the existing live chunk state. Publication to
    # the externally owned BF16 state remains in flat_prefill_forward.
    ttnn.typecast(updated, state.recurrent.dtype, output_tensor=state.recurrent)
    out = ttnn.reshape(out, ttnn.Shape((batch, 12, seq, 128)), ttnn.Shape((batch, 12, padded, 128)))
    out = ttnn.rms_norm(
        ttnn.typecast(out, ttnn.bfloat16),
        weight=layer.weights["linear_attn.norm.weight"],
        epsilon=layer.config.rms_norm_eps,
        compute_kernel_config=layer.ckc,
    )
    # Gather heads once, drop per-user tile padding before token-independent
    # gating, and keep the out projection and following residual compact.
    out = ttnn.reshape(ttnn.transformer.concatenate_heads(out), (1, batch * seq, 1536))
    out = ttnn.multiply(ttnn.silu(packed[:, :, 2560:4096]), out)
    return layer._linear(out, "linear_attn.out_proj.weight")
