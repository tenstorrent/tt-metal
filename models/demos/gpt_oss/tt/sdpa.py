import ttnn


def softmax(x: ttnn.Tensor, stable=False):
    """
    Performs Softmax on a ``ttnn.Tensor``.
    """
    if stable:
        sumsW = ttnn.max(x, -1)
        sumsW = ttnn.unsqueeze(sumsW, -1)
        z = ttnn.subtract(x, sumsW)  # x-max(x)
    else:
        z = x
    numerator = ttnn.exp(z)  # exp(z)
    denom1 = ttnn.sum(numerator, 3)  # torch.sum(x, 3)
    denom = ttnn.reciprocal(denom1)
    denom = ttnn.unsqueeze(denom, -1)
    output = ttnn.multiply(numerator, denom)

    return output


def sdpa(
    tt_q: ttnn.Tensor,
    tt_k: ttnn.Tensor,
    tt_v: ttnn.Tensor,
    tt_sink: ttnn.Tensor,
    sm_scale: float,
    tt_mask: ttnn.Tensor = None,
    tt_cache: ttnn.Tensor = None,
    position_idx: int = None,
) -> ttnn.Tensor:
    """
    Perform a single attention operation using the provided tensors.

    Args:
        tt_q (ttnn.Tensor): Query tensor.
        tt_k (ttnn.Tensor): Key tensor.
        tt_v (ttnn.Tensor): Value tensor.
        tt_sink (ttnn.Tensor): Output tensor.
        sm_scale (float): Scaling factor for softmax.
        sliding_window (int, optional): Size of the sliding window for attention. Defaults to 0.

    Returns:
        ttnn.Tensor: The result of the attention operation.
    """

    assert tt_q.shape[-1] == tt_k.shape[-1] == tt_v.shape[-1], "Dim size mismatch"

    num_tokens, _, nh, dim = tt_q.shape
    _, nkv, _ = tt_k.shape

    # Prepare inputs
    tt_q = ttnn.permute(tt_q, [1, 2, 0, 3])  # (1, nh, num_tokens, dim)
    tt_q = ttnn.reshape(tt_q, [nkv, nh // nkv, num_tokens, dim])

    # KV Cache handling
    tt_k = ttnn.transpose(tt_k, 0, 1)  # [nkv, num_tokens, dim]
    tt_k = ttnn.unsqueeze(tt_k, 1)  # [nkv, 1, num_tokens, dim]

    tt_v = ttnn.transpose(tt_v, 0, 1)  # [nkv, num_tokens, dim]
    tt_v = ttnn.unsqueeze(tt_v, 1)  # [nkv, 1, num_tokens, dim]

    if tt_cache is None:
        tt_cache = [tt_k, tt_v]  # Cache for keys and values
    else:
        tt_k_back, tt_v_back = tt_cache

        if position_idx is not None:
            assert position_idx <= tt_k_back.shape[2], "position_idx exceeds cache length"
            tt_k_back = tt_k_back[:, :, :position_idx, :]
            tt_v_back = tt_v_back[:, :, :position_idx, :]

        tt_k = ttnn.concat([tt_k_back, tt_k], dim=2)  # (nkv, 1, cache_len + num_tokens, dim)
        tt_v = ttnn.concat([tt_v_back, tt_v], dim=2)  # (nkv, 1, cache_len + num_tokens, dim)

        tt_cache = [tt_k, tt_v]  # Update cache with new keys and values
        # ttnn.deallocate(tt_k_back)
        # ttnn.deallocate(tt_v_back)

    kv_len = tt_k.shape[2]  # Length of keys/values in the cache

    tt_k = ttnn.repeat(tt_k, [1, nh // nkv, 1, 1])  # (nkv, nh // nkv, kv_len, dim)
    tt_k = ttnn.transpose(tt_k, -1, -2)  # (nkv, nh // nkv, dim, kv_len)

    tt_v = ttnn.repeat(tt_v, [1, nh // nkv, 1, 1])  # (nkv, nh // nkv, kv_len, dim)

    # QK + scale
    tt_qk = ttnn.matmul(tt_q, tt_k)  # (nkv, nh // nkv, num_tokens, kv_len)
    tt_qk *= sm_scale

    # Mask
    if tt_mask is not None:
        tt_qk += tt_mask

    # Sink
    tt_sink = ttnn.reshape(tt_sink, [nkv, nh // nkv, 1, 1])  # (nkv, nh // nkv, 1, 1)
    tt_sink = ttnn.repeat(tt_sink, [1, 1, num_tokens, 1])  # (nkv, nh // nkv, num_tokens, 1)
    tt_qk = ttnn.concat([tt_qk, tt_sink], dim=-1)  # (nkv, nh // nkv, num_tokens, kv_len + 1)

    # Softmax
    # FIXME: Program cache issue!!
    # tt_qk = ttnn.softmax(tt_qk, dim=-1, numeric_stable=True)  # (nkv, nh // nkv, num_tokens, kv_len + 1)
    tt_qk = softmax(tt_qk, stable=True)  # (nkv, nh // nkv, num_tokens, kv_len + 1)
    tt_qk = tt_qk[:, :, :, :kv_len]

    # Out stuff
    out = ttnn.matmul(tt_qk, tt_v)  # (nkv, nh // nkv, num_tokens, dim)
    out = ttnn.reshape(out, [1, nh, num_tokens, dim])

    out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, num_tokens, dim * nh]

    return out, tt_cache
