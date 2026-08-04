import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

dev = ttnn.open_device(device_id=0)


def cfg(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


def ref(x, g):
    xf = x.to(torch.float32)
    return (xf / torch.sqrt(torch.mean(xf**2, -1, True) + 1e-6)) * g.to(torch.float32).reshape(-1)


torch.manual_seed(0)
for shape in [(1, 1, 32, 4064), (1, 1, 32, 2848)]:
    for lay in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        for acc in (True, False):
            for thr in (4, 10**9):
                pd.REDUCE_ACC_VIA_ADD_MIN_WT = thr
                x = torch.randn(*shape, dtype=torch.bfloat16)
                g = torch.randn(shape[-1], dtype=torch.bfloat16)
                xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=lay, device=dev)
                gt = g.reshape(1, 1, 1, -1) if lay == ttnn.TILE_LAYOUT else g
                gd = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=lay, device=dev)
                a = ttnn.to_torch(rms_norm(xd, gamma=gd, compute_kernel_config=cfg(acc))).float()
                e = ref(x, g)
                rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
                print(
                    f"RES {shape} {'RM' if lay==ttnn.ROW_MAJOR_LAYOUT else 'TIL'} acc={acc} algo={'AccViaAdd' if thr==4 else 'ReduceTile'} rms={rms:.5f}",
                    flush=True,
                )
pd.REDUCE_ACC_VIA_ADD_MIN_WT = 4
ttnn.close_device(dev)
