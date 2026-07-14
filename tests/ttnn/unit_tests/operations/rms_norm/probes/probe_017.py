import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor
from eval.metrics import check_output, CheckOutputError


def ref(x, eps=1e-6):
    xf = x.to(torch.float32)
    return (xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)).to(torch.bfloat16)


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


# Exactly the golden bf8b tolerance
TOL = (0.99, 0.10)


def run_bypass(name, shape):
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
    try:
        check_output(res, exp, shape=shape, dtype=ttnn.bfloat8_b, expected_layout=ttnn.TILE_LAYOUT, tolerance=TOL)
        print(f"[{name}] shape={shape} -> golden check_output PASS")
    except CheckOutputError as e:
        print(f"[{name}] shape={shape} -> golden check_output FAIL: {str(e)[:150]}")


device = ttnn.open_device(device_id=0)
try:
    # all the non-aligned golden shapes that would be bf8b cells
    for nm, sh in [
        ("w50", (1, 1, 32, 50)),
        ("w17", (1, 1, 64, 17)),
        ("w47", (4, 8, 32, 47)),
        ("w100", (2, 1, 128, 100)),
        ("h17", (1, 1, 17, 64)),
        ("h50", (1, 1, 50, 128)),
        ("h47", (4, 8, 47, 256)),
        ("both_17_50", (1, 1, 17, 50)),
        ("both_100_47", (2, 1, 100, 47)),
    ]:
        run_bypass(nm, sh)
finally:
    ttnn.close_device(device)
