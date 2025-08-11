import ttnn


def sdpa(
    tt_q: ttnn.Tensor,
    tt_k: ttnn.Tensor,
    tt_v: ttnn.Tensor,
    tt_sink: ttnn.Tensor,
    sm_scale: float,
    tt_mask: ttnn.Tensor,
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

    assert tt_q.shape[0] == tt_k.shape[0] == tt_v.shape[0], "Number of tokens mismatch"
    assert tt_q.shape[-1] == tt_k.shape[-1] == tt_v.shape[-1], "Dim size mismatch"

    num_tokens, _, nh, dim = tt_q.shape
    _, nkv, _ = tt_k.shape

    # Prepare inputs
    tt_q = ttnn.reshape(tt_q, [num_tokens, nkv, nh // nkv, dim])
    tt_q = ttnn.permute(tt_q, [1, 2, 0, 3])  # (nkv, nh // nkv, num_tokens, dim)

    tt_k = ttnn.reshape(tt_k, [num_tokens, nkv, 1, dim])
    tt_k = ttnn.repeat(tt_k, [1, 1, nh // nkv, 1])
    tt_k = ttnn.permute(tt_k, [1, 2, 3, 0])  # (nh // nkv, dim, num_tokens)

    tt_v = ttnn.reshape(tt_v, [num_tokens, nkv, 1, dim])
    tt_v = ttnn.repeat(tt_v, [1, 1, nh // nkv, 1])
    tt_v = ttnn.permute(tt_v, [1, 2, 0, 3])  # (nh // nkv, num_tokens dim)

    # QK + scale
    tt_qk = ttnn.matmul(tt_q, tt_k)  # (nkv, nh // nkv, num_tokens, num_tokens)
    tt_qk *= sm_scale

    # Mask
    tt_qk += tt_mask

    # Sink
    tt_sink = ttnn.reshape(tt_sink, [1, nkv, nh // nkv, 1])
    tt_sink = ttnn.permute(tt_sink, [1, 2, 0, 3])  # (nkv, nh // nkv, 1, 1)
    tt_sink = ttnn.repeat(tt_sink, [1, 1, num_tokens, 1])  # (nkv, nh // nkv, num_tokens, 1)
    tt_qk = ttnn.concat([tt_qk, tt_sink], dim=-1)  # (nkv, nh // nkv, num_tokens, num_tokens + 1)

    # Softmax
    tt_qk = ttnn.softmax(tt_qk, dim=-1, numeric_stable=True)  # (nkv, nh // nkv, num_tokens, num_tokens + 1)

    tt_qk = ttnn.slice(
        tt_qk, [0, 0, 0, 0], [nkv, nh // nkv, num_tokens, num_tokens]
    )  # (nkv, nh // nkv, num_tokens, num_tokens)

    # Out stuff
    out = ttnn.matmul(tt_qk, tt_v)  # (nkv, nh // nkv, num_tokens, dim)
    out = ttnn.reshape(out, [1, nh, num_tokens, dim])
    out = ttnn.permute(out, [0, 2, 1, 3])  # (1, num_tokens, nh, dim)
    out = ttnn.reshape(out, [num_tokens, dim * nh])

    return out
