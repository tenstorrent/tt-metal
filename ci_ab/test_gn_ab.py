# Scratch, not for merge. Block sharded group norm (SDXL shapes, tile statistics, groupnorm_sharded_v2.cpp); each output's
# sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

SHAPES = {  # name: (C, H, W)
    "gn1280_64": (1280, 64, 64),
    "gn1920_32": (1920, 32, 32),
    "gn640_64": (640, 64, 64),
    "gn320_128": (320, 128, 128),
    "gn2560_32": (2560, 32, 32),
    "gn512_128_tile": (512, 128, 128),
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True, ids=["l1"])
@pytest.mark.parametrize("case", list(SHAPES))
def test_gn_sharded(device, case, request):
    C, H, W = SHAPES[case]
    N, groups = 1, 32
    layout = ttnn.TILE_LAYOUT if case.endswith("_tile") else ttnn.ROW_MAJOR_LAYOUT
    inplace = layout != ttnn.TILE_LAYOUT
    torch.manual_seed(0)
    core_grid = ttnn.CoreGrid(y=8, x=8)
    x = torch.rand((N, C, H, W), dtype=torch.bfloat16).permute(0, 2, 3, 1).reshape(N, 1, W * H, C)
    mem = ttnn.create_sharded_memory_config(shape=x.shape, core_grid=core_grid, strategy=ttnn.ShardStrategy.BLOCK,
                                            orientation=ttnn.ShardOrientation.ROW_MAJOR)
    mask = ttnn.to_device(ttnn.create_group_norm_input_mask(C, groups, core_grid.x, ttnn.DataType.BFLOAT8_B), device)
    out = None
    for _ in range(2):
        t = ttnn.from_torch(x, dtype=ttnn.DataType.BFLOAT16, layout=layout, memory_config=mem, device=device)
        out = ttnn.group_norm(t, num_groups=groups, input_mask=mask, memory_config=t.memory_config(),
                              core_grid=core_grid, inplace=inplace, use_welford=False)
    ttnn.synchronize_device(device)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")
