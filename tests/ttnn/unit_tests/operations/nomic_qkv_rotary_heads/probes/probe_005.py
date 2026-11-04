import math, os, torch, ttnn
from ttnn.operations.nomic_qkv_rotary_heads import nomic_qkv_rotary_heads, prepare_qkv_weights, prepare_rotary_tables
from ttnn.operations.nomic_residual_layer_norm import input_memory_config as nin, output_memory_config as nout

device = ttnn.open_device(device_id=0)


def dur():
    ttnn.ReadDeviceProfiler(device)
    data = ttnn.get_latest_programs_perf_data()
    out = []
    for _c, progs in (data or {}).items():
        for p in progs:
            r = p.program_analyses_results
            if "DEVICE KERNEL DURATION [ns]" in r:
                out.append(int(r["DEVICE KERNEL DURATION [ns]"].duration))
    return out


shape = (8, 1, 512, 768)
x = torch.randn(shape).to(torch.bfloat16)
w = (torch.randn(2304, 768) / 28).to(torch.bfloat16)
b = (0.02 * torch.randn(2304)).to(torch.bfloat16)
tx = ttnn.from_torch(
    x,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=nout(shape, device, nin(shape, device)),
)
W = prepare_qkv_weights(w, b, device)
T = prepare_rotary_tables(512, device)
hs = ttnn.Shape((8, 12, 512, 64))
v, k, q = (
    ttnn.allocate_tensor_on_device(hs, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG) for _ in range(3)
)
dur()
for fid in ["LoFi", "HiFi2", "HiFi3", "HiFi4"]:
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
    )
    nomic_qkv_rotary_heads(
        tx, W, T, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensors=[q, k, v], compute_kernel_config=cfg
    )
    ttnn.synchronize_device(device)
    print("FID", fid, dur())
ttnn.close_device(device)
