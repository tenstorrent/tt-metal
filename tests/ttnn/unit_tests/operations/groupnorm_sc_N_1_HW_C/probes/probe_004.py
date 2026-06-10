import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for shape, G in (((1, 1, 64, 64), 2), ((1, 1, 1024, 256), 8)):
        N, _, HW, C = shape
        x = torch.randn(shape)
        g = torch.randn(1, 1, 1, C)
        ref = (
            torch.nn.functional.group_norm(x.squeeze(1).permute(0, 2, 1), G, weight=g.reshape(C))
            .permute(0, 2, 1)
            .unsqueeze(1)
        )
        tt_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        for acc in (False, True):
            cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=acc, math_approx_mode=False
            )
            out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, G, gamma=tt_g, compute_kernel_config=cfg)).float()
            rms = ((out - ref).pow(2).mean().sqrt() / ref.std()).item()
            print(f"shape={shape} G={G} fp32_acc={acc}: rel_rms={rms:.5f} (target 0.01)")
finally:
    ttnn.close_device(device)
