# Scratch, not for merge. ttnn copy / typecast / tilize / untilize ops on DRAM interleaved and L1 sharded tensors, one op call per
# test function call; each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

BF, F32, B8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
T, R = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG


def shard_mc(shape, cores):
    return ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=1, x=cores), strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=False)


# name: (op, in dtype, out dtype, shape, sharded cores or 0)
CASES = {}
for dn, d in (("bf16", BF), ("fp32", F32)):
    for shp, tag in (((1, 1, 2048, 2048), "d"), ((1, 1, 32, 1024), "s1")):
        CASES[f"tilize_{dn}_{tag}"] = ("tilize", d, d, shp, 0)
    CASES[f"tilize_{dn}_l1"] = ("tilize", d, d, (1, 1, 2048, 256), 8)
    CASES[f"untilize_{dn}_d"] = ("untilize", d, d, (1, 1, 2048, 2048), 0)
    CASES[f"untilize_{dn}_w"] = ("untilize", d, d, (1, 1, 64, 8192), 0)
    CASES[f"untilize_{dn}_l1"] = ("untilize", d, d, (1, 1, 2048, 256), 8)
    CASES[f"untilize_{dn}_s1"] = ("untilize", d, d, (1, 1, 32, 1024), 0)
    CASES[f"tilize_vp_{dn}"] = ("tilize_vp", d, d, (1, 1, 2000, 2000), 0)
    CASES[f"untilize_unpad_{dn}"] = ("untilize_unpad", d, d, (1, 1, 2048, 2048), 0)
CASES["untilize_bfp8_d"] = ("untilize", B8, B8, (1, 1, 2048, 2048), 0)
for a, b, n in ((BF, F32, "bf16_fp32"), (F32, BF, "fp32_bf16"), (BF, B8, "bf16_bfp8"), (B8, BF, "bfp8_bf16"), (F32, B8, "fp32_bfp8")):
    CASES[f"typecast_{n}_d"] = ("typecast", a, b, (1, 1, 2048, 2048), 0)
    CASES[f"typecast_{n}_l1"] = ("typecast", a, b, (1, 1, 2048, 256), 8)
    CASES[f"typecast_{n}_s1"] = ("typecast", a, b, (1, 1, 32, 1024), 0)
CASES["clone_bf16_fp32"] = ("clone", BF, F32, (1, 1, 2048, 2048), 0)
CASES["clone_fp32_bf16_l1"] = ("clone", F32, BF, (1, 1, 2048, 256), 8)


def sha(t):
    x = ttnn.to_torch(t)
    return hashlib.sha1(x.contiguous().view(torch.uint8).numpy().tobytes() if x.dtype != torch.bfloat16 else x.view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def run_case(device, case):
    op, din, dout, shp, cores = CASES[case]
    torch.manual_seed(1)
    x = torch.rand(shp) - 0.5
    in_layout = R if op in ("tilize", "tilize_vp") else T
    mc_in = DRAM
    if cores:
        h = shp[-2] // cores
        mc_in = shard_mc((h, shp[-1]), cores)
    t = ttnn.from_torch(x, dtype=din, layout=in_layout, device=device, memory_config=mc_in)
    if op == "tilize":
        out = ttnn.tilize(t, memory_config=mc_in)
    elif op == "untilize":
        out = ttnn.untilize(t, memory_config=mc_in)
    elif op == "tilize_vp":
        out = ttnn.tilize_with_val_padding(t, [1, 1, 2048, 2048], 0.0)
    elif op == "untilize_unpad":
        out = ttnn.untilize_with_unpadding(t, [0, 0, 1999, 1999])
    elif op == "typecast":
        out = ttnn.typecast(t, dout, memory_config=mc_in)
    elif op == "clone":
        out = ttnn.clone(t, dtype=dout, memory_config=mc_in)
    ttnn.synchronize_device(device)
    return out


@pytest.mark.parametrize("case", list(CASES))
def test_cls(device, case, request):
    out = run_case(device, case)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid.split('::')[-1]} {sha(out)}\n")
