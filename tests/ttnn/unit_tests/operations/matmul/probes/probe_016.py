import torch, ttnn
from ttnn.operations.matmul.matmul_program_descriptor import create_program_descriptor

device = ttnn.open_device(device_id=0)


def run_raw(A_shape, B_shape, dtype=ttnn.float32, wdtype=ttnn.float32, fp32_acc=True):
    torch.manual_seed(0)
    tdt = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
    A = torch.randn(A_shape, dtype=tdt[dtype])
    B = torch.randn(B_shape, dtype=tdt[wdtype])
    expected = torch.matmul(A.float(), B.float())
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc, math_approx_mode=False
    )
    ttA = ttnn.from_torch(A, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttB = ttnn.from_torch(
        B, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_shape = list(A_shape[:-1]) + [B_shape[-1]]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    pd = create_program_descriptor(ttA, ttB, out, cfg)
    res = ttnn.generic_op([ttA, ttB, out], pd)
    res_t = ttnn.to_torch(res).float()
    exp = expected.float()
    diff = (res_t - exp).abs()
    rf, ef = res_t.flatten(), exp.flatten()
    pcc = torch.corrcoef(torch.stack([rf, ef]))[0, 1].item()
    rms = (diff.pow(2).mean().sqrt() / ef.std()).item()
    print(
        f"  A{A_shape} B{B_shape} dt={dtype} wdt={wdtype} acc={fp32_acc}: max_diff={diff.max():.4f} PCC={pcc:.6f} relRMS={rms:.4f}"
    )
    return pcc, rms


try:
    print("=== K non-aligned (load-bearing) ===")
    run_raw((64, 50), (50, 128))
    run_raw((128, 100), (100, 256))
    print("=== N non-aligned ===")
    run_raw((64, 128), (128, 50))
    print("=== M non-aligned ===")
    run_raw((50, 128), (128, 256))
    print("=== multi non-aligned ===")
    run_raw((50, 100), (100, 47))
finally:
    ttnn.close_device(device)
