import sys, torch, ttnn
from ttnn.operations import normalization as N

dev = ttnn.open_device(device_id=0)
x = torch.randn(1, 1, 64, 128, dtype=torch.float32).to(torch.bfloat16)
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

GEN = "ttnn.operations.rms_norm_ttnn"
print("generated module imported at start :", GEN in sys.modules)

out_native = ttnn.to_torch(N._native_rms_norm(tx, epsilon=1e-12))
print("after calling _native_rms_norm     :", GEN in sys.modules, "<- must stay False")

out_switch = ttnn.to_torch(ttnn.rms_norm(tx, epsilon=1e-12))
print("after calling ttnn.rms_norm        :", GEN in sys.modules, "<- must be True")

ref = x.to(torch.float32) * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-12)
for name, o in (("native", out_native), ("switched", out_switch)):
    d = (o.to(torch.float32) - ref).abs().max().item()
    print(f"{name:9s} max abs diff vs torch: {d:.5f}")
agree = (out_native.to(torch.float32) - out_switch.to(torch.float32)).abs().max().item()
print(f"native vs switched agreement     : {agree:.5f}")
ttnn.close_device(dev)
print("PROBE_OK")
