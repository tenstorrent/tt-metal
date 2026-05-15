import torch, ttnn
from ttnn.operations.rms_norm.rms_norm_program_descriptor import create_program_descriptor

dev = ttnn.open_device(device_id=0)
try:

    def run(W, input_dt, gamma_dt, label):
        H = 32
        shape = (1, 1, H, W)
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.float32) * 0.5
        g = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.3 + 1.0
        x_t = ttnn.from_torch(
            x, dtype=input_dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        g_t = ttnn.from_torch(
            g, dtype=gamma_dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), input_dt, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        pd = create_program_descriptor(x_t, g_t, out_t, epsilon=1e-6)
        result = ttnn.generic_op([x_t, g_t, out_t], pd)
        y = ttnn.to_torch(result).to(torch.float32)
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + 1e-6)
        golden = (x / rms) * g
        max_abs = (y - golden).abs().max().item()
        pcc = torch.corrcoef(torch.stack([y.flatten(), golden.flatten()]))[0, 1].item()
        print(f"W={W:5d} {label}: max_abs={max_abs:.5f}  pcc={pcc:.6f}", flush=True)

    print("--- Stage A as BinaryFpu Mul(x,x); random data ---")
    for W in [32, 64, 128, 256]:
        for it, gt, lbl in [
            (ttnn.bfloat16, ttnn.bfloat16, "bf16+bf16"),
            (ttnn.float32, ttnn.float32, "fp32+fp32"),
            (ttnn.bfloat16, ttnn.float32, "bf16+fp32"),
            (ttnn.float32, ttnn.bfloat16, "fp32+bf16"),
        ]:
            run(W, it, gt, lbl)
finally:
    ttnn.close_device(dev)
