"""Iteration 10 wiring: route the rank-wide LoRA products (x @ A^T, g @ B) through ttml.ops.metal.skinny_matmul.
Run from tt-train/."""


def sub(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(path, "w").write(s.replace(old, new))


P = "sources/ttml/ttml/modules/lora.py"
sub(
    P,
    '''def lora_matmul_config():
    """Compute config matching ttml's linear (HiFi4, fp32 accumulation)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )
''',
    '''def lora_matmul_config():
    """Compute config matching ttml's linear (HiFi4, fp32 accumulation)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )


# `x @ A^T` / `g @ B` produce rank-wide outputs ([T, r], one tile column): ttnn's matmul parallelises over output
# tiles and runs them on T/32 cores (33 of 110 at T=1024, 75-120 us for K=5120-8192). `skinny_matmul` splits K over
# the cores instead (iteration 10); below this K the two are equal within dispatch noise.
SKINNY_MIN_K = 2048


def skinny_matmul(a, b, transpose_b=False):
    """``a @ (b^T if transpose_b else b)`` (HiFi4, fp32 accumulation) for a rank-wide ``b``: the K-split op when it
    pays, ttnn's matmul otherwise."""
    k = a.shape[-1]
    n = b.shape[-2] if transpose_b else b.shape[-1]
    if n <= 64 and k >= SKINNY_MIN_K and k % 32 == 0 and b.shape[0] == 1 and b.shape[1] == 1:
        return ttml.ops.metal.skinny_matmul(a, b, transpose_b)
    return ttnn.matmul(a, b, transpose_b=transpose_b, compute_kernel_config=lora_matmul_config())
''',
)
sub(
    P,
    """        h = ttnn.matmul(x_val, a_val, transpose_b=True, compute_kernel_config=cfg)
        update = ttnn.matmul(h, b_val, transpose_b=True, compute_kernel_config=cfg)
        ctx.save_for_backward(x_val, a_val, b_val, h)""",
    """        h = skinny_matmul(x_val, a_val, transpose_b=True)
        update = ttnn.matmul(h, b_val, transpose_b=True, compute_kernel_config=cfg)
        ctx.save_for_backward(x_val, a_val, b_val, h)""",
)
sub(
    P,
    """        s = ctx.scaling
        grad_h = ttnn.multiply(ttnn.matmul(grad_output, b_val, compute_kernel_config=cfg), s)""",
    """        s = ctx.scaling
        grad_h = ttnn.multiply(skinny_matmul(grad_output, b_val), s)""",
)
sub(
    P,
    """        cfg = lora_matmul_config()
        grad_h = ttnn.multiply(ttnn.matmul(grad_output, b_val, compute_kernel_config=cfg), ctx.scaling)""",
    """        cfg = lora_matmul_config()
        grad_h = ttnn.multiply(skinny_matmul(grad_output, b_val), ctx.scaling)""",
)

P = "sources/ttml/ttml/modules/fused_lora.py"
sub(
    P,
    """        h = ttnn.matmul(x_val, a_val, transpose_b=True, compute_kernel_config=cfg)  # [T, rank]""",
    """        h = skinny_matmul(x_val, a_val, transpose_b=True)  # [T, rank], K split over the cores (iteration 10)""",
)
sub(
    P,
    "from .frozen_linear import TRANSPOSED_SHAPES, _compute_config, forward_matmul_kwargs, full_core_grid, local_in_out\n",
    "from .frozen_linear import TRANSPOSED_SHAPES, _compute_config, forward_matmul_kwargs, full_core_grid, local_in_out\nfrom .lora import skinny_matmul\n",
)
print("iteration 10 wired")
