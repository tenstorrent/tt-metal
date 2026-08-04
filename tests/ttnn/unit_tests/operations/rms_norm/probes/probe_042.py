import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
_ML = ttnn.TensorMemoryLayout


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


# band INPUT + a NON-matching output placement: interleaved DRAM (page == stick,
# so the writer addresses it through the accessor at the band's byte offset), and
# the default (None -> inherits the input's shard, the local path).
for shape in [(1, 1, 256, 512), (1, 1, 224, 3072)]:
    for ml in (_ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED):
        for omc_name in ("dram", "l1_interleaved", "inherit"):
            torch.manual_seed(0)
            x = torch.randn(*shape, dtype=torch.bfloat16)
            g = torch.randn(shape[-1], dtype=torch.bfloat16)
            mc = auto_shard_config(list(shape), ml, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=dev)
            xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
            gd = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            omc = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1_interleaved": ttnn.L1_MEMORY_CONFIG, "inherit": None}[omc_name]
            try:
                out = ttnn.to_torch(rms_norm(xd, gamma=gd, compute_kernel_config=cfg(), memory_config=omc)).float()
            except Exception as ex:
                print(
                    f"RES {shape} {str(ml).split('.')[-1]} out={omc_name}: EXC {type(ex).__name__}: {str(ex)[:100]}",
                    flush=True,
                )
                continue
            xf = x.float()
            e = xf / torch.sqrt(torch.mean(xf**2, -1, True) + 1e-6) * g.float()
            a = out.flatten()
            ee = e.flatten()
            pcc = torch.corrcoef(torch.stack([a, ee]))[0, 1].item()
            rms = ((a - ee).pow(2).mean().sqrt() / ee.pow(2).mean().sqrt()).item()
            print(
                f"RES {shape} {str(ml).split('.')[-1]:14s} out={omc_name:14s}: pcc={pcc:.6f} rms={rms:.5f}", flush=True
            )
ttnn.close_device(dev)
