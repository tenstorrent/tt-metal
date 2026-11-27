import torch, ttnn
import ttnn.operations.g6_to_memory_config.g6_to_memory_config_program_descriptor as dm
from ttnn.operations.g6_to_memory_config import g6_to_memory_config, plan_of


def mc(buffer, scheme="interleaved", grid=None, shard=None, orient=ttnn.ShardOrientation.ROW_MAJOR):
    b = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}[buffer]
    s = {
        "interleaved": ttnn.TensorMemoryLayout.INTERLEAVED,
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }[scheme]
    if scheme == "interleaved":
        return ttnn.MemoryConfig(s, b)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, b_), ttnn.CoreCoord(c, d)) for a, b_, c, d in grid])
    return ttnn.MemoryConfig(s, b, ttnn.ShardSpec(crs, tuple(shard), orient))


def make(device, shape, m, seed=0):
    torch.manual_seed(seed)
    return ttnn.from_torch(
        torch.randn(shape).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=m,
    )


device = ttnn.open_device(device_id=0)
try:
    # validation reasons
    q2058_out = [
        [0, 2, 0, 2],
        [1, 2, 1, 2],
        [2, 2, 2, 2],
        [3, 2, 3, 2],
        [4, 2, 4, 2],
        [5, 1, 5, 1],
        [5, 2, 5, 2],
        [6, 1, 6, 1],
        [6, 2, 6, 2],
        [7, 1, 7, 1],
        [7, 2, 7, 2],
        [8, 1, 8, 1],
        [8, 2, 8, 2],
        [9, 1, 9, 1],
        [9, 2, 9, 2],
        [10, 1, 10, 1],
    ]
    x = make(device, [1, 16, 1, 256], mc("dram"))
    for name, cfg in [
        ("colmajor", mc("l1", "height", q2058_out, [32, 256], ttnn.ShardOrientation.COL_MAJOR)),
        ("shard16", mc("l1", "height", q2058_out, [16, 256])),
        ("fewcores", mc("l1", "height", q2058_out[:8], [32, 256])),
    ]:
        try:
            g6_to_memory_config(x, cfg)
            print("BAD no error", name)
        except Exception as e:
            print("ERR", name, type(e).__name__, str(e)[:110])
    try:
        g6_to_memory_config(
            ttnn.from_torch(
                torch.randn(1, 16, 1, 256).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
            ttnn.L1_MEMORY_CONFIG,
        )
    except Exception as e:
        print("ERR host", type(e).__name__, str(e)[:100])
    for shape, im, om in [
        ([16, 128, 1536], mc("dram"), mc("dram")),
        ([1, 64, 1536], mc("dram"), mc("l1")),
        ([1, 1, 16, 12928], mc("l1", "width", [[0, 0, 7, 0]], [32, 1664]), mc("dram")),
    ]:
        t = make(device, shape, im)
        p = plan_of(t, om)
        print(
            "PLAN",
            shape,
            p["regime"],
            "movers",
            len(p["movers"]),
            "box",
            p["kernel_cores"],
            "blk",
            p["bounce_block_pages"],
            "cb",
            p["cb_bounce_pages"],
            "l1",
            p["l1_op_bytes"],
            list(p["roles"].items())[:2],
        )

    # knobs: movers_per_owner=2 and bounce depth/block variants, bit-exact
    cases = {
        "pull_shard": (
            [1, 1, 16, 5120],
            mc("l1", "width", [[0, 0, 7, 0]], [32, 640]),
            mc("l1", "width", [[0, 0, 9, 3]], [32, 128]),
        ),
        "pull_shard_dram": ([1, 16, 6, 256], mc("dram"), mc("l1", "height", [[0, 0, 10, 0], [0, 1, 4, 1]], [32, 256])),
        "pull_bank": ([1, 77, 4352], mc("dram"), mc("l1")),
        "push_shard": ([1, 1, 16, 12928], mc("l1", "width", [[0, 0, 7, 0]], [32, 1664]), mc("dram")),
        "push_bank": ([1, 1, 16, 4352], mc("l1"), mc("dram")),
        "bounce": ([16, 128, 1536], mc("dram"), mc("dram")),
    }
    for movers, depth in [(2, 2), (1, 3), (2, 4)]:
        dm.MOVERS_PER_OWNER = movers
        dm.BOUNCE_DEPTH = depth
        for name, (shape, im, om) in cases.items():
            t = make(device, shape, im, seed=3)
            out = g6_to_memory_config(t, om)
            p = plan_of(t, om)
            ok = torch.equal(ttnn.to_torch(out), ttnn.to_torch(t))
            print(
                "KNOB" if ok else "BAD",
                f"movers={movers} depth={depth}",
                name,
                p["regime"],
                p["movers_per_owner"],
                p["bounce_block_pages"],
                ok,
            )
finally:
    ttnn.close_device(device)
