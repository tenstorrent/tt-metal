import ttnn


def qwen35_rms_norm(x: ttnn.Tensor, weight: ttnn.Tensor, eps: float = 1e-6, scale=False):
    if scale:
        weight = weight + 1.0
    return ttnn.rms_norm(x, epsilon=eps) * weight
