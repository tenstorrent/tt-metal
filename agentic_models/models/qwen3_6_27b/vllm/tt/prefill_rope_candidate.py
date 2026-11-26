# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optional TP4 prefill RoPE batching candidate; no TTNN import at module load.

MultichipDecoder._rope calls it for multi-user, multi-token prefill only (B>1, S>1);
B=1 and S=1 keep the inherited RoPE.
"""


def batched_prefill_rope(layer, x, rope):
    """Rotate each user's own positions with one native prefill RoPE call.

    Input [B,H,S,D] and trig [B,1,S,64] become [1,H,B*T,64] and
    [1,1,B*T,64], where T=ceil(S/32)*32. Explicit per-user padding keeps
    tile reshape semantics unambiguous; no public alignment is required.
    """
    import ttnn

    batch, heads, seq, width = tuple(x.shape)
    if batch <= 1 or seq <= 1:
        return layer._prefill_rope_original(x, rope)
    if width < 64:
        raise ValueError(f"Qwen partial rotary width requires at least 64 channels: {x.shape}")
    cos, sin = rope
    expected = (batch, 1, seq, 64)
    if tuple(cos.shape) != expected or tuple(sin.shape) != expected:
        raise ValueError(f"RoPE trig must match each user's logical positions: expected {expected}")
    padded_seq = ((seq + 31) // 32) * 32

    def padded(t):
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        if seq != padded_seq:
            t = ttnn.pad(t, ((0, 0), (0, 0), (0, padded_seq - seq), (0, 0)), value=0.0)
        return t

    prefix = padded(x[:, :, :, :64])
    # H,B,T ordering repeats the complete B*T trig sequence once per head.
    prefix = ttnn.reshape(ttnn.permute(prefix, (1, 0, 2, 3)), (1, heads, batch * padded_seq, 64))
    trig = [ttnn.reshape(padded(t), (1, 1, batch * padded_seq, 64)) for t in (cos, sin)]
    rotated = ttnn.experimental.rotary_embedding_hf(
        prefix,
        *trig,
        compute_kernel_config=layer.ckc,
    )
    rotated = ttnn.permute(ttnn.reshape(rotated, (heads, batch, padded_seq, 64)), (1, 0, 2, 3))
    if seq != padded_seq:
        rotated = rotated[:, :, :seq, :]
    if width == 64:
        return rotated
    return ttnn.concat([rotated, x[:, :, :, 64:]], dim=-1)
