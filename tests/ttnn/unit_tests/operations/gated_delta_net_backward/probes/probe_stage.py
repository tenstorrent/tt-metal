import os
import torch
import torch.nn.functional as F
import ttnn
import ttnn.operations.gated_delta_net_backward.gated_delta_net_backward_program_descriptor as pd
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward


def test_stage(device):
    pd.STAGE_MASK = int(os.environ.get("GDN_STAGE_MASK", "7"))
    B, T, H, K, V, C = 1, 32, 1, 32, 32, 32
    torch.manual_seed(0)

    def l2(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def dev(t):
        return ttnn.from_torch(
            t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    q = dev(l2(torch.randn(B, T, H, K)))
    k = dev(l2(torch.randn(B, T, H, K)))
    v = dev(torch.randn(B, T, H, V))
    g = dev(F.logsigmoid(torch.randn(B, T, H)) * 0.5)
    beta = dev(torch.rand(B, T, H))
    do = dev(torch.randn(B, T, H, V))
    out = gated_delta_net_backward(q, k, v, g, beta, do, chunk_size=C)
    print("STAGE_MASK", pd.STAGE_MASK, "ok, dq shape", ttnn.to_torch(out[0]).shape)
