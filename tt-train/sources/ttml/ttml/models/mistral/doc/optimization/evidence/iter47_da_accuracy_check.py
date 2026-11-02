"""Iteration 47: the gate_up dA product on the owned op ([r, N] mode, the overlap window's route) against ttnn's
transposed-a matmul (the route outside the window) and the fp64 reference on bf16 inputs; one chip, the step's shape
(s_dh [T, 16], x [T, 5120])."""
import numpy as np
import torch
import ttnn
import ttml

ttml.autograd.AutoContext.get_instance().open_device()
rng = np.random.default_rng(47)


def bf16(a):
    return torch.tensor(a, dtype=torch.float32).to(torch.bfloat16).to(torch.float64).numpy()


def dev(a):
    return ttml.autograd.Tensor.from_numpy(a.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16).get_value()


def host(t):
    return ttml.autograd.create_tensor(t).to_numpy(ttnn.DataType.FLOAT32).astype(np.float64)


cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
for trial in range(3):
    s_dh = bf16(rng.normal(0, 0.01, (1, 1, 1024, 16)))
    x = bf16(rng.normal(0, 1, (1, 1, 1024, 5120)))
    ref = s_dh[0, 0].T @ x[0, 0]  # fp64
    op = host(
        ttml.ops.metal.skinny_contract(
            dev(np.ascontiguousarray(np.swapaxes(s_dh, -1, -2))),
            dev(x),
            1.0,
            False,
            big_transposed=False,
            small_transposed=True,
        )
    )[0, 0, :16]
    op7 = host(
        ttml.ops.metal.skinny_contract(
            dev(np.ascontiguousarray(np.swapaxes(s_dh, -1, -2))),
            dev(x),
            1.0,
            False,
            big_transposed=False,
            small_transposed=True,
            grid_rows=7,
        )
    )[0, 0, :16]
    tt = host(ttnn.matmul(dev(s_dh), dev(x), transpose_a=True, compute_kernel_config=cfg))[0, 0, :16]

    def rel(a):
        return np.linalg.norm(a - ref) / np.linalg.norm(ref)

    bias = lambda a: float(np.mean(a - ref) / np.mean(np.abs(ref)))
    print(
        f"trial {trial}: rel-to-fp64 op {rel(op):.3e} (7 rows {rel(op7):.3e}, bit-equal {np.array_equal(op, op7)}), ttnn {rel(tt):.3e}; "
        f"op-vs-ttnn {np.linalg.norm(op - tt) / np.linalg.norm(tt):.3e}; mean bias op {bias(op):+.2e} ttnn {bias(tt):+.2e}",
        flush=True,
    )
ttml.autograd.AutoContext.get_instance().close_device()
