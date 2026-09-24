import os

import pytest
import torch
import torch.nn.functional as F
import ttnn
import ttnn.operations.gated_delta_net_backward.gated_delta_net_backward_program_descriptor as pd
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

CASES = {
    "small": ((1, 128, 2, 64, 64), 32),
    "widev": ((1, 128, 2, 64, 128), 64),
    "large": ((1, 256, 4, 128, 256), 64),
    "long": ((1, 512, 2, 64, 64), 32),
}


@pytest.mark.parametrize("case", list(CASES))
def test_perf(case, device):
    pd.STAGE_MASK = int(os.environ.get("GDN_STAGE_MASK", "7"))
    shape, chunk = CASES[case]
    B, T, H, K, V = shape
    torch.manual_seed(0)

    def l2(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def dev(t):
        return ttnn.from_torch(
            t.to(torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    args = (
        dev(l2(torch.randn(B, T, H, K))),
        dev(l2(torch.randn(B, T, H, K))),
        dev(torch.randn(B, T, H, V)),
        dev(F.logsigmoid(torch.randn(B, T, H)) * 0.5),
        dev(torch.rand(B, T, H)),
        dev(torch.randn(B, T, H, V)),
    )
    gated_delta_net_backward(*args, chunk_size=chunk)
    ttnn.synchronize_device(device)
