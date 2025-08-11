import ttnn


def sdpa(
    tt_q: ttnn.Tensor,
    tt_k: ttnn.Tensor,
    tt_v: ttnn.Tensor,
    tt_sink: ttnn.Tensor,
    sm_scale: float,
    tt_mask: ttnn.Tensor,
    tt_cache: ttnn.Tensor = None,
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
    tt_q = ttnn.reshape(tt_q, [num_tokens, nkv, nh // nkv, dim])
    tt_q = ttnn.permute(tt_q, [1, 2, 0, 3])  # (nkv, nh // nkv, num_tokens, dim)

    # KV Cache handling
    tt_k = ttnn.reshape(tt_k, [num_tokens, nkv, 1, dim])  # unsqueeze
    tt_v = ttnn.reshape(tt_v, [num_tokens, nkv, 1, dim])  # unsqueeze
    tt_k = ttnn.permute(tt_k, [1, 2, 0, 3])  # (nkv, 1, num_tokens, dim)
    tt_v = ttnn.permute(tt_v, [1, 2, 0, 3])  # (nkv, 1, num_tokens, dim)

    slice_out = False
    if tt_cache is None:
        tt_cache = [tt_k, tt_v]  # Cache for keys and values
    else:
        tt_k_back, tt_v_back = tt_cache

        tt_k = ttnn.concat([tt_k_back, tt_k], dim=2)  # (nkv, 1, cache_len + num_tokens, dim)
        tt_v = ttnn.concat([tt_v_back, tt_v], dim=2)  # (nkv, 1, cache_len + num_tokens, dim)

        tt_cache = [tt_k, tt_v]  # Update cache with new keys and values
        slice_out = True

    kv_len = tt_k.shape[2]  # Length of keys/values in the cache

    tt_k = ttnn.repeat(tt_k, [1, nh // nkv, 1, 1])  # (nkv, nh // nkv, kv_len, dim)
    tt_k = ttnn.transpose(tt_k, -1, -2)  # (nkv, nh // nkv, dim, kv_len)

    tt_v = ttnn.repeat(tt_v, [1, nh // nkv, 1, 1])  # (nkv, nh // nkv, kv_len, dim)

    # QK + scale
    tt_qk = ttnn.matmul(tt_q, tt_k)  # (nkv, nh // nkv, kv_len, kv_len)
    tt_qk *= sm_scale

    # Mask
    tt_qk += tt_mask

    # Sink
    tt_sink = ttnn.reshape(tt_sink, [1, nkv, nh // nkv, 1])
    tt_sink = ttnn.permute(tt_sink, [1, 2, 0, 3])  # (nkv, nh // nkv, 1, 1)
    tt_sink = ttnn.repeat(tt_sink, [1, 1, kv_len, 1])  # (nkv, nh // nkv, kv_len, 1)
    tt_qk = ttnn.concat([tt_qk, tt_sink], dim=-1)  # (nkv, nh // nkv, kv_len, kv_len + 1)

    # Softmax
    tt_qk = ttnn.softmax(tt_qk, dim=-1, numeric_stable=True)  # (nkv, nh // nkv, kv_len, kv_len + 1)

    tt_qk = ttnn.slice(
        tt_qk, [0, 0, 0, 0], [nkv, nh // nkv, kv_len, kv_len]
    )  # (nkv, nh // nkv, num_tokens, num_tokens)

    # Out stuff
    out = ttnn.matmul(tt_qk, tt_v)  # (nkv, nh // nkv, kv_len, dim)
    out = ttnn.reshape(out, [1, nh, kv_len, dim])
    out = ttnn.permute(out, [0, 2, 1, 3])  # (1, kv_len, nh, dim)

    if slice_out:
        # if kv_len > 132:
        #     breakpoint()
        out = ttnn.slice(out, [0, kv_len - num_tokens, 0, 0], [1, kv_len, nh, dim])  # (1, num_tokens, nh, dim)

    # FIXME: This reshape hangs after a few iterations (GH Issue)
    out = ttnn.reshape(out, [num_tokens, dim * nh])

    return out, tt_cache
