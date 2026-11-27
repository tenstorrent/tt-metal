import ttnn
from ttnn.operations.g6_embedding.g6_embedding_program_descriptor import make_plan, plan_report

grid = ttnn.CoreCoord(11, 10)
cases = [
    ((16, 4096), (248320, 1280)),
    ((8, 4096), (248320, 1280)),
    ((16, 1024), (248320, 1280)),
    ((1, 3844), (248320, 1280)),
    ((16, 4096), (262144, 64)),
    ((16, 128), (248320, 1280)),
    ((1, 512), (131072, 2560)),
    ((1, 1), (32, 5120)),
    ((1, 1), (1, 256, 128)),
    ((1, 1000), (248320, 1280)),
    ((1, 1), (248320, 1280)),
    ((2, 128), (262144, 64)),
    ((1, 33), (248320, 1280)),
    ((1,), (131072, 2560)),
]
for i, w in cases:
    r = plan_report(make_plan(i, w, grid))
    print(
        i,
        w,
        r["regime"],
        "ct",
        r["col_block_tiles"],
        "P",
        r["col_pieces"],
        "bt",
        r["block_tile_rows"],
        "U",
        r["total_units"],
        "cores",
        r["num_cores"],
        "L1",
        r["l1_bytes_per_core"],
        r["core_ranges"],
    )
