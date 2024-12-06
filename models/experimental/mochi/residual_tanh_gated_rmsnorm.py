import ttnn


def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
    tanh_gate = ttnn.tanh(gate)
    x_normed = ttnn.rms_norm(x_res, weight=tanh_gate, epsilon=eps)
    output = x + x_normed
    return output
