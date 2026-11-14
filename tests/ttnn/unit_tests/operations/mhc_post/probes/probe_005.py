import torch, ttnn

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
K = 32
w = torch.randn(K, 32).bfloat16()
a = torch.zeros(32, K)
for r in range(32):
    a[r, r] = torch.randn(1).item()
a = a.bfloat16()
ex = a.double() @ w.double()
cfg = ttnn.init_device_compute_kernel_config(
    dev.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
    math_approx_mode=False,
)
ad = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
wd = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
o = ttnn.to_torch(ttnn.matmul(ad, wd, compute_kernel_config=cfg, dtype=ttnn.float32)).double()
print("onehot exact frac", (o == ex).double().mean().item(), "rel", ((o - ex).norm() / ex.norm()).item())
# two products
a2 = torch.zeros(32, K)
for r in range(32):
    a2[r, r] = torch.randn(1).item()
    a2[r, (r + 5) % 32] = torch.randn(1).item()
a2 = a2.bfloat16()
ex2 = a2.double() @ w.double()
o2 = ttnn.to_torch(
    ttnn.matmul(
        ttnn.from_torch(a2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
        wd,
        compute_kernel_config=cfg,
        dtype=ttnn.float32,
    )
).double()
print("twohot exact frac", (o2 == ex2).double().mean().item(), "rel", ((o2 - ex2).norm() / ex2.norm()).item())
ttnn.close_device(dev)
