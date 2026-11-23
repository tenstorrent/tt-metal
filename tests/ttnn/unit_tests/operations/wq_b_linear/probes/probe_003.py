import torch, ttnn
import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd
from ttnn.operations.wq_b_linear import wq_b_linear
from models.common.utility_functions import comp_pcc

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 160, 1536).bfloat16()
w = (torch.randn(1536, 4096) * 0.02).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
e = torch.matmul(x.double(), ttnn.to_torch(tw).double())


def stats(tag, g):
    g = g.double()
    slope = (g * e).sum() / (e * e).sum()
    _, p = comp_pcc(e, g)
    rms = (g - e).pow(2).mean().sqrt() / e.std()
    print(f"P: {tag} pcc={p:.6f} rms={rms:.5f} slope={slope:.5f}")


for kb in [1, 2, 4, 8, 12, 16, 24, 48]:
    pd.BLOCK_K_TILES = kb
    try:
        stats(f"op kb={kb}", ttnn.to_torch(wq_b_linear(tx, tw)))
    except Exception as ex:
        print(f"P: kb={kb} err {str(ex)[:120]}")
pd.BLOCK_K_TILES = 8
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
)
stats("stock matmul l1acc", ttnn.to_torch(ttnn.matmul(tx, tw, compute_kernel_config=ckc)))
ckc2 = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=False
)
stats("stock matmul nol1acc", ttnn.to_torch(ttnn.matmul(tx, tw, compute_kernel_config=ckc2)))
ckc3 = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
)
stats("stock matmul fp32dest", ttnn.to_torch(ttnn.matmul(tx, tw, compute_kernel_config=ckc3)))
print("P: arch", dev.arch(), "grid", dev.compute_with_storage_grid_size(), "dram", dev.dram_grid_size())
ttnn.close_device(dev)
