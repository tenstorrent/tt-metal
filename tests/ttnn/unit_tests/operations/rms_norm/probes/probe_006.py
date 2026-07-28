# Re-run the deterministic partial-W mask probe after routing the reduce
# datapath (cb_x_squared) to a maskable format for block-float inputs.
import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
exp = 1.0 / (1.0 + 1e-6) ** 0.5

for dt, name in [(ttnn.bfloat8_b, "bfp8"), (ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")]:
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
    for W in (33, 49, 63, 100, 4097):
        x = torch.ones((1, 1, 32, W), dtype=tdt)
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        got = ttnn.to_torch(rms_norm(tx, epsilon=1e-6)).float()
        dev = (got - exp).abs().max().item()
        # recover the sum the kernel actually accumulated: out = 1/sqrt(S/W + eps)
        S = W * (1.0 / got[0, 0, 0, 0].item() ** 2 - 1e-6)
        print(
            f"{'OK ' if dev < 0.02 else 'BAD'} {name} W={W:<5d} max|out-1|={dev:.5f}  "
            f"recovered_sum={S:8.2f} (true {W})",
            flush=True,
        )
    print(flush=True)

# H non-aligned (W aligned -> no mask at all) — separate path, confirm too.
for dt, name in [(ttnn.bfloat8_b, "bfp8"), (ttnn.bfloat16, "bf16")]:
    for H in (17, 47, 50):
        x = torch.ones((1, 1, H, 128), dtype=torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        got = ttnn.to_torch(rms_norm(tx, epsilon=1e-6)).float()
        dev = (got - exp).abs().max().item()
        print(f"{'OK ' if dev < 0.02 else 'BAD'} {name} H={H} (h_non_aligned) max|out-1|={dev:.5f}", flush=True)

ttnn.close_device(device)
