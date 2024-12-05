import ttnn


def modulated_rmsnorm(x, scale, eps=1e-6):
    weight = 1.0 + scale
    x = ttnn.rms_norm(x, weight=weight, epsilon=eps)
    return x
