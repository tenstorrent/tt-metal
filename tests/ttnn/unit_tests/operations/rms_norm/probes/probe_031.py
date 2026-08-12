import torch, ttnn
from eval.sharding import shard_config
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

ML = ttnn.TensorMemoryLayout
CASES = [
    (32, 1024, ML.WIDTH_SHARDED, [32, 128], (8, 1)),
    (32, 2304, ML.WIDTH_SHARDED, [32, 256], (9, 1)),
    (32, 5120, ML.WIDTH_SHARDED, [32, 160], (8, 4)),
    (32, 7168, ML.WIDTH_SHARDED, [32, 256], (7, 4)),
    (8192, 1024, ML.BLOCK_SHARDED, [1024, 128], (8, 8)),
]
device = ttnn.open_device(device_id=0)
try:
    for rows, W, ml, ss, grid in CASES:
        shape = (1, 1, rows, W)
        mc = shard_config(ss, grid, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        x = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        g = ttnn.from_torch(
            torch.zeros(1, 1, 1, W, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        geo = pd._Geometry(x, g)
        sv = pd._ShardView(x, geo, False)
        infos = pd._shard_core_infos(geo, sv)
        groups = pd._sharded_groups(sv, infos)
        G = len(groups[0]["members"])
        C = sv.shard_w_tiles
        for gg in groups:
            for m in gg["members"]:
                m["w_start_tiles"] = m["w_start_elems"] // 32
        tree = pd._combine_tree(groups, G)
        budget = pd._l1_cb_budget() - 2 * sv.bank_bytes
        max_row_count = max(gg["row_count"] for gg in groups)
        R = 0
        for depths in pd._depth_ladder():
            R = pd._max_block_row_tiles(
                geo,
                C,
                tree[0],
                max_row_count,
                False,
                0,
                budget,
                depths=depths,
                pin_in=True,
                pin_out=True,
                stage2_span=tree[1],
            )
            if R:
                break
        fixed, per_row = pd._cb_bytes(
            geo, C, tree[0], False, 0, depths=depths, pin_in=True, pin_out=True, stage2_span=tree[1]
        )
        print(
            f"{shape} {str(ml).split('.')[-1]}: ngroups={len(groups)} G={G} C={C} rows/group={max_row_count} R={R} nblocks={-(-max_row_count//R)} tree={tree} budget={budget} bank={sv.bank_bytes} fixed={fixed} per_row={per_row}"
        )
finally:
    ttnn.close_device(device)
