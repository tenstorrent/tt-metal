import ttnn
import torch


def test_div(device):
    vox_feats = torch.randn((1, 256, 3, 25281), dtype=torch.bfloat16)
    area = torch.randn((1, 1, 3, 25281), dtype=torch.bfloat16)

    vox_feats = ttnn.from_torch(vox_feats, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    area = ttnn.from_torch(
        area, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    vox_feats = ttnn.div(vox_feats, area, dtype=ttnn.bfloat8_b)


def test_multiply(device):
    vox_feats = torch.randn((1, 256, 3, 25281), dtype=torch.bfloat16)
    visible = torch.randn((1, 1, 3, 25281), dtype=torch.bfloat16)

    vox_feats = ttnn.from_torch(vox_feats, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    visible = ttnn.from_torch(
        visible, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    vox_feats = ttnn.multiply(vox_feats, visible, dtype=ttnn.bfloat8_b)
