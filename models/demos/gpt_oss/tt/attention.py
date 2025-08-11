import ttnn


def sdpa(
    tt_q: ttnn.Tensor,
    tt_k: ttnn.Tensor,
    tt_v: ttnn.Tensor,
    tt_sink: ttnn.Tensor,
    sm_scale: float,
    sliding_window: int = 0,
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

    bsz, gqa_factor, nh, dim = tt_q.shape
    assert tt_k.shape == (bsz, 1, nh, dim)

    tt_k = ttnn.repeat(tt_k, [1, gqa_factor, 1, 1])

    return tt_k
