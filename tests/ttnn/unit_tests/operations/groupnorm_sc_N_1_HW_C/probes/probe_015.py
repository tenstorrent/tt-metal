"""Isolate NaN/Inf in fp32 sdxl_C320_G32_Cg10."""
import torch
import torch.nn.functional as F
import ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)


def torch_gn(x, num_groups, gamma, beta, eps=1e-5):
    N, _, HW, C = x.shape
    x_ncl = x.reshape(N, HW, C).permute(0, 2, 1).contiguous()
    gw = gamma.reshape(C) if gamma is not None else None
    bw = beta.reshape(C) if beta is not None else None
    y = F.group_norm(x_ncl, num_groups=num_groups, weight=gw, bias=bw, eps=eps)
    return y.permute(0, 2, 1).reshape(N, 1, HW, C).contiguous()


def run_case(shape, G, dtype, fp32_acc, label):
    N, _, HW, C = shape
    x_torch = torch.randn(shape, dtype=torch.float32)
    g_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)
    b_torch = torch.randn((1, 1, 1, C), dtype=torch.float32)
    y_ref = torch_gn(x_torch, G, g_torch, b_torch)

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x_tt = ttnn.from_torch(
        x_torch.to(torch_dtype),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g_tt = ttnn.from_torch(
        g_torch.to(torch_dtype),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_tt = ttnn.from_torch(
        b_torch.to(torch_dtype),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    y_tt = groupnorm_sc_N_1_HW_C(x_tt, G, gamma=g_tt, beta=b_tt, eps=1e-5, compute_kernel_config=cfg)
    y = ttnn.to_torch(y_tt).to(torch.float32)
    nan_count = torch.isnan(y).sum().item()
    inf_count = torch.isinf(y).sum().item()
    max_abs = (y - y_ref).abs().max().item() if nan_count == 0 and inf_count == 0 else float("nan")
    print(f"{label:55s} | NaN={nan_count} Inf={inf_count} max_abs={max_abs:.4f}")


run_case((1, 1, 64, 320), 32, ttnn.float32, True, "FAIL CASE      (1,1,64,320) G=32 Cg=10 fp32 fp32_acc")
run_case((1, 1, 64, 320), 32, ttnn.float32, False, "PARTIAL FAIL   (1,1,64,320) G=32 Cg=10 fp32 bf16_acc")
run_case((1, 1, 64, 320), 32, ttnn.bfloat16, True, "SANITY (bf16)  (1,1,64,320) G=32 Cg=10 bf16 fp32_acc")
run_case((1, 1, 32, 320), 32, ttnn.float32, True, "HW=32          (1,1,32,320) G=32 Cg=10 fp32 fp32_acc")
run_case((1, 1, 64, 320), 10, ttnn.float32, True, "G=10 Cg=32     (1,1,64,320) G=10 Cg=32 fp32 fp32_acc")
run_case((1, 1, 64, 320), 20, ttnn.float32, True, "G=20 Cg=16     (1,1,64,320) G=20 Cg=16 fp32 fp32_acc")
run_case((1, 1, 64, 320), 16, ttnn.float32, True, "G=16 Cg=20     (1,1,64,320) G=16 Cg=20 fp32 fp32_acc")
run_case((1, 1, 64, 320), 8, ttnn.float32, True, "G=8  Cg=40     (1,1,64,320) G=8  Cg=40 fp32 fp32_acc")
run_case((1, 1, 64, 64), 2, ttnn.float32, True, "C=64 G=2 Cg=32 (1,1,64,64)  fp32 fp32_acc")
run_case((1, 1, 64, 64), 4, ttnn.float32, True, "C=64 G=4 Cg=16 (1,1,64,64)  fp32 fp32_acc")
run_case((1, 1, 32, 64), 4, ttnn.float32, True, "C=64 G=4 Cg=16 HW=32 fp32 fp32_acc")

ttnn.close_device(device)
