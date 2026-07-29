"""Refinement 2 first light: B8 (trid double-issue), B10 (per-core VC), A3 (bank
placement). arange inputs -> every element unique, so a misplaced row cannot cancel."""
import os
import torch
import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

SHAPES = [
    ((1, 1, 128, 256), False),  # 1 core, chunk 8, 4 blocks
    ((1, 1, 128, 1024), False),  # 1 core, chunk 16, 2 chunks x 4 rows = 8 blocks (flattened)
    ((1, 1, 512, 512), False),  # 1 core, chunk 16, 16 blocks (c_single_core geometry)
    ((1, 1, 4096, 32), True),  # 64 cores, chunk 1, 2 blocks/core
    ((1, 1, 32, 4096), True),  # 64 cores, 1 block/core (B8 structurally off)
    ((1, 1, 96, 96), True),  # awkward Wt=3
]

COMBOS = [
    ("baseline", dict(B8="0", B10="0", A3="0")),
    ("b8", dict(B8="2", B10="0", A3="0")),
    ("b10", dict(B8="0", B10="2", A3="0")),
    ("a3", dict(B8="0", B10="0", A3="2")),
    ("all", dict(B8="2", B10="2", A3="2")),
]

fails = 0
for shape, mc in SHAPES:
    t = torch.arange(shape[2] * shape[3], dtype=torch.int32).reshape(shape).to(torch.bfloat16)
    for name, env in COMBOS:
        for k, v in env.items():
            os.environ[f"TILIZE_LEVER_{k}"] = v
        dev_in = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        probe_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        plan = tpd.build_plan(dev_in, probe_out, device, use_multicore=mc)
        out = tilize(dev_in, use_multicore=mc)
        got = ttnn.to_torch(out)
        ok = torch.equal(got.float(), t.float())
        if not ok:
            fails += 1
        print(
            f"  {str(shape):<20} {name:<9} cores={plan['ncores']:>3} chk={plan['chunk_wt']:>2} "
            f"d={plan['depth']} blk={plan['blocks_per_core']:>2} b8={plan['prefetch_blocks']} "
            f"vc={plan['vc_spread']} a3={plan['bank_placement']}  {'OK' if ok else 'MISMATCH'}"
        )
        ttnn.deallocate(out)
        ttnn.deallocate(dev_in)
        ttnn.deallocate(probe_out)

for k in ("B8", "B10", "A3"):
    os.environ[f"TILIZE_LEVER_{k}"] = "1"
print("FAILS:", fails)
assert fails == 0
