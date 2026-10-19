# Scratch, not for merge. binary_ng sharded no-broadcast add, sub and mul (FPU: fast_and_approximate_mode) at one-op
# replicas of model shard specs, every bf16/bfp8_b mix the widened block rule takes, plus LoFi and HiFi4 layer norm and
# softmax (their kernels opt into the per-tile hand-off). Each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
# name: (shape, strategy, grid x, grid y); tiles per core in the comment
CASES = {
    "dec_w32": ((1, 1, 32, 4096), ttnn.ShardStrategy.WIDTH, 8, 4),  # 4 x 32, Llama 3.1-8B decode residual
    "dec_w16": ((1, 1, 32, 4096), ttnn.ShardStrategy.WIDTH, 8, 2),  # 8 x 16
    "dec_w10": ((1, 1, 32, 1280), ttnn.ShardStrategy.WIDTH, 5, 2),  # 4 x 10, Qwen3-32B Galaxy decode residual
    "h8_16": ((1, 1, 4096, 32), ttnn.ShardStrategy.HEIGHT, 8, 1),  # 16 x 8
    "pre_b32": ((1, 1, 1024, 896), ttnn.ShardStrategy.BLOCK, 4, 8),  # 28 x 32, gemma4_d_p prefill residual
    "blk64": ((1, 1, 2048, 1024), ttnn.ShardStrategy.BLOCK, 8, 8),  # 32 x 64, SDXL-sized
}
MIX = ["bf16-bf16-bf16", "bfp8-bfp8-bfp8", "bfp8-bf16-bf16", "bf16-bfp8-bf16", "bf16-bf16-bfp8"]
OPS = {"add": ttnn.add, "sub": ttnn.subtract, "mul": ttnn.multiply}


def _bits(node, t):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(t).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {node} {h}\n")


@pytest.mark.parametrize("mix", MIX)
@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("case", list(CASES))
def test_bng(device, case, op, mix, request):
    shape, strat, gx, gy = CASES[case]
    da, db, dc = (DT[x] for x in mix.split("-"))
    mem = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=strat)
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(shape) * 0.5, dtype=da, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    b = ttnn.from_torch(torch.randn(shape) * 0.5, dtype=db, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    out = None
    for _ in range(4):
        out = OPS[op](a, b, dtype=dc, memory_config=mem, fast_and_approximate_mode=True)
    _bits(request.node.nodeid, out)


@pytest.mark.parametrize("fid", ["LoFi", "HiFi4"])
@pytest.mark.parametrize("kind", ["layernorm", "softmax"])
def test_gate(device, kind, fid, request):
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=False, math_approx_mode=False)
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, 1, 256, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if kind == "layernorm":
        w = ttnn.from_torch(torch.randn(1, 1, 1, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.layer_norm(x, weight=w, compute_kernel_config=cfg)
    else:
        out = ttnn.softmax(x, dim=-1, compute_kernel_config=cfg)
    _bits(request.node.nodeid, out)
