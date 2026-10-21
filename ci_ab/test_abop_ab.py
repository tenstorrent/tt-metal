# Scratch, not for merge. Op level callers of _llk_unpack_AB_: binary_ng (per-tile and mixed-dtype forms, broadcast forms) and
# deepseek_moe_gate; each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

BF, F32, B8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
# name: (op, shape a, shape b, dtype a, dtype b, memory)
BNG = {
    "add_bf16_dram": ("add", (1, 1, 2048, 2048), (1, 1, 2048, 2048), BF, BF, DRAM),
    "mul_bf16_l1": ("multiply", (1, 1, 1024, 1024), (1, 1, 1024, 1024), BF, BF, L1),
    "sub_bf16_l1": ("subtract", (1, 1, 1024, 1024), (1, 1, 1024, 1024), BF, BF, L1),
    "add_f32_l1": ("add", (1, 1, 1024, 1024), (1, 1, 1024, 1024), F32, F32, L1),
    "mul_f32_l1": ("multiply", (1, 1, 1024, 1024), (1, 1, 1024, 1024), F32, F32, L1),
    "add_mixed_l1": ("add", (1, 1, 1024, 1024), (1, 1, 1024, 1024), BF, F32, L1),
    "add_bfp8_l1": ("add", (1, 1, 1024, 1024), (1, 1, 1024, 1024), B8, B8, L1),
    "mul_colb_l1": ("multiply", (1, 1, 1024, 1024), (1, 1, 1024, 1), BF, BF, L1),
    "add_rowb_l1": ("add", (1, 1, 1024, 1024), (1, 1, 1, 1024), BF, BF, L1),
    "sub_batch_l1": ("subtract", (4, 1, 512, 512), (1, 1, 512, 512), BF, BF, L1),
}


def record(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


@pytest.mark.parametrize("case", list(BNG))
def test_bng(device, case, request):
    op, sa, sb, da, db, mem = BNG[case]
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(*sa).bfloat16(), dtype=da, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    b = ttnn.from_torch(torch.randn(*sb).bfloat16(), dtype=db, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    out = None
    for _ in range(2):
        out = getattr(ttnn, op)(a, b, memory_config=mem)
    ttnn.synchronize_device(device)
    record(request, out)


@pytest.mark.parametrize("batch", [1, 16])
def test_moe_gate(device, batch, request):
    eps, scaling = 1e-20, 2.5
    shard = (32, 32)
    tile = ttnn.Tile(shard)
    shp = (batch, 16, 16)
    torch.manual_seed(42)
    x = (2 * torch.rand((batch, 8, 32), dtype=torch.bfloat16)) - 1
    bias = (2 * torch.rand((batch, 8, 32), dtype=torch.bfloat16)) - 1
    grid = device.compute_with_storage_grid_size()
    cg = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(cg, shard, ttnn.ShardOrientation.ROW_MAJOR))
    mk = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, tile=tile)
    t_in = mk(x.reshape(shp), BF)
    t_bias = mk(bias.reshape(shp).transpose(-2, -1), BF)
    idx = torch.arange(256, dtype=torch.int32).unsqueeze(0).expand(batch, -1).reshape(shp).transpose(-2, -1).to(torch.uint16)
    t_idx = mk(idx, ttnn.uint16)
    t_out = mk(torch.zeros((batch, 1, 16), dtype=torch.bfloat16), BF)
    t_oidx = mk(torch.zeros((batch, 1, 16), dtype=torch.uint16), ttnn.uint16)
    s = i = None
    for _ in range(2):
        s, i = ttnn.experimental.deepseek.moe.deepseek_moe_gate(
            t_in, bias_tensor=t_bias, input_indices_tensor=t_idx, output_tensor=t_out, output_indices_tensor=t_oidx,
            eps=eps, scaling_factor=scaling, enable_sigmoid=True)
    ttnn.synchronize_device(device)
    record(request, s)
    record(request, i)
