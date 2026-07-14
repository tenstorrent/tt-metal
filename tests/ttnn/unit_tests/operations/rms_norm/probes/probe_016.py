import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def ref(x, eps=1e-6):
    xf = x.to(torch.float32)
    return xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


def run_bypass(name, shape, min_pcc=0.99):
    torch.manual_seed(0)
    ti = torch.randn(shape, dtype=torch.bfloat16)
    xi = ttnn.from_torch(
        ti, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_t = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(xi.shape)), xi.dtype, xi.layout, device, ttnn.DRAM_MEMORY_CONFIG
    )
    pd = create_program_descriptor(xi, out_t, gamma=None, epsilon=1e-6, compute_kernel_config=cfg())
    res = ttnn.generic_op([xi, out_t], pd)
    exp = ref(ti)
    act = ttnn.to_torch(res).reshape(exp.shape)
    p = pcc(exp, act)
    print(f"[{name}] shape={shape} PCC={p:.5f} {'(would PASS)' if p>=min_pcc else '(FAILS -> exclusion justified)'}")


device = ttnn.open_device(device_id=0)
try:
    run_bypass("bf8b_W_aligned_ctrl", (1, 1, 64, 128))  # control (aligned) — should pass
    run_bypass("bf8b_w_nonalign_50", (1, 1, 64, 50))
    run_bypass("bf8b_w_nonalign_100", (1, 1, 64, 100))
    run_bypass("bf8b_h_nonalign_50", (1, 1, 50, 128))
    run_bypass("bf8b_h_nonalign_17", (1, 1, 17, 64))
    run_bypass("bf8b_both_nonalign", (1, 1, 50, 100))
finally:
    ttnn.close_device(device)
