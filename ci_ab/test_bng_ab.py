# Scratch, not for merge. binary_ng column and scalar broadcast on L1 sharded tensors: per shape, dtype mix and op, the
# sha1 of the output goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

BF16, BFP8 = ttnn.bfloat16, ttnn.bfloat8_b
OPS = {"add": ttnn.add, "sub": ttnn.subtract, "mul": ttnn.multiply}
MIX = {"bb": (BF16, BF16), "ee": (BFP8, BFP8), "be": (BF16, BFP8), "eb": (BFP8, BF16), "bo": (BF16, BF16), "eo": (BFP8, BFP8)}
OUT = {"bo": BFP8, "eo": BF16}  # output dtype differs from the inputs
# name: (a shape, b shape, shard height in elements, shard width)
SHAPES = {
    "col64": ((1, 1, 4096, 128), (1, 1, 4096, 1), 512, 128),   # 64 tiles per core
    "col16": ((1, 1, 1024, 128), (1, 1, 1024, 1), 128, 128),   # 16 tiles
    "col4": ((1, 1, 256, 128), (1, 1, 256, 1), 32, 128),       # 4 tiles
    "colw": ((1, 1, 2048, 512), (1, 1, 2048, 1), 256, 512),    # 128 tiles
    "colb": ((2, 4, 256, 256), (2, 4, 256, 1), 256, 256),      # batch heads, 64 tiles
    "sc64": ((1, 1, 4096, 128), (1, 1, 1, 1), 512, 128),
    "sc16": ((1, 1, 1024, 128), (1, 1, 1, 1), 128, 128),
    "sc4": ((1, 1, 256, 128), (1, 1, 1, 1), 32, 128),
}
CASES = [(s, m, o) for s in SHAPES for m in ("bb", "ee", "be") for o in ("add", "mul")] + \
        [(s, "eb", "sub") for s in ("col64", "col16", "sc64")] + [(s, m, o) for s in ("col64", "sc16") for m in ("bo", "eo") for o in ("add", "mul")] + [(s, "bb", "sub") for s in ("col64", "col4", "sc16")]


@pytest.mark.parametrize("swap", [False, True], ids=["bB", "bA"])
@pytest.mark.parametrize("case", [f"{s}-{m}-{o}" for s, m, o in CASES])
def test_bng_bcast(device, case, swap, request):
    s, m, o = case.split("-")
    ash, bsh, sh, sw = SHAPES[s]
    da, db = MIX[m]
    torch.manual_seed(0)
    ta, tb = torch.randn(ash) * 3, torch.randn(bsh) * 3
    if swap:
        ta, tb, da, db = tb, ta, db, da
    h = ash[0] * ash[1] * ash[2]
    ncores = h // sh * (ash[3] // sw)
    grid = device.compute_with_storage_grid_size()
    if ncores > grid.x * grid.y:
        pytest.skip("grid")
    cr = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    mem = ttnn.create_sharded_memory_config([sh, sw], core_grid=cr, strategy=ttnn.ShardStrategy.HEIGHT,
                                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                            use_height_and_width_as_shard_shape=True)
    a = ttnn.from_torch(ta, dtype=da, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=mem if not swap else ttnn.L1_MEMORY_CONFIG)
    b = ttnn.from_torch(tb, dtype=db, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=mem if swap else ttnn.L1_MEMORY_CONFIG)
    out = OPS[o](a, b, memory_config=mem, fast_and_approximate_mode=True, dtype=OUT.get(m))
    p = os.environ.get("EB_BITS_OUT")
    if p:
        hh = hashlib.sha1(ttnn.to_torch(out).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {hh}\n")
