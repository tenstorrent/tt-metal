import torch, ttnn
from ttnn.operations.matmul.matmul_program_descriptor import create_program_descriptor

device = ttnn.open_device(device_id=0)


def run_raw(A_shape, B_shape, dtype, wdtype, fp32_acc):
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
    dn = {ttnn.float32: "fp32", ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bf8b"}
    print(f"  A{A_shape}B{B_shape} {dn[dtype]}/{dn[wdtype]} acc={fp32_acc}: PCC={pcc:.6f} relRMS={rms:.4f}")
    return pcc, rms


shapes = {
    "K_nonalign": ((64, 50), (50, 128)),
    "N_nonalign": ((64, 128), (128, 50)),
    "M_nonalign": ((50, 128), (128, 256)),
    "multi(K>N>M)": ((50, 100), (100, 47)),
}
try:
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        for acc in (True, False):
            print(f"=== dtype={dt} acc={acc} ===")
            for name, (a, b) in shapes.items():
                run_raw(a, b, dt, dt, acc)
    # mixed bf16-act x fp32-weight, K non-aligned
    print("=== mixed bf16/fp32 ===")
    run_raw((64, 50), (50, 128), ttnn.bfloat16, ttnn.float32, True)
finally:
    ttnn.close_device(device)
