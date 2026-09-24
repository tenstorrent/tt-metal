import sys

import torch
import torch.nn.functional as F
import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

gdn = sys.modules["ttnn.operations.gated_delta_net_backward.gated_delta_net_backward"]

SHAPES = [
    ((1, 32, 1, 32, 32), 32),
    ((1, 128, 2, 64, 64), 32),
    ((1, 128, 2, 64, 128), 64),
    ((2, 64, 4, 64, 64), 32),
    ((1, 100, 2, 64, 64), 64),
    ((1, 256, 4, 128, 256), 64),
    ((1, 512, 2, 64, 64), 32),
]


def test_distribution(device):
    grid = device.compute_with_storage_grid_size()
    print(f"GRID {grid.x}x{grid.y} = {grid.x * grid.y} cores")
    for shape, chunk in SHAPES:
        B, T, H, K, V = shape

        def dev(t):
            return ttnn.from_torch(
                t.to(torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def l2(x):
            return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        torch.manual_seed(0)
        gated_delta_net_backward(
            dev(l2(torch.randn(B, T, H, K))),
            dev(l2(torch.randn(B, T, H, K))),
            dev(torch.randn(B, T, H, V)),
            dev(F.logsigmoid(torch.randn(B, T, H)) * 0.5),
            dev(torch.rand(B, T, H)),
            dev(torch.randn(B, T, H, V)),
            chunk_size=chunk,
        )
        sc = gdn._LAST_SCRATCH
        geo = sc["geo"]
        print(
            f"DIST {shape} c{chunk}: BH={geo.BH} NC={geo.NC} items={geo.NI} "
            f"cores={sc['num_cores']} Vt={geo.Vt} Vb={sc['Vb']} NVB={sc['NVB']} "
            f"scan_cores={geo.BH}"
        )
