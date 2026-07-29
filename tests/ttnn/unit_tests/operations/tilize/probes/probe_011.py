import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

device = ttnn.open_device(device_id=0)
try:
    for shape, W in [((1, 1, 2048, 32), 32), ((1, 1, 128, 64), 64), ((1, 1, 64, 128), 128)]:
        t = torch.arange(shape[2] * shape[3], dtype=torch.float32).reshape(shape).bfloat16()
        d = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = tilize(d)
        back = ttnn.to_torch(out)
        probe = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(d.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        plan = tpd.build_plan(d, probe, device)
        print(
            shape,
            "stateful",
            plan["stateful_read"],
            "split",
            plan["split_read"],
            "chunk_row_bytes",
            plan["chunk_row_bytes"],
            "depth",
            plan["depth"],
            "cores",
            plan["ncores"],
            "blk",
            plan["blocks_per_core"],
            "EXACT"
            if torch.equal(back.float(), t.float())
            else "MISMATCH maxdiff=%s" % (back.float() - t.float()).abs().max(),
        )
finally:
    ttnn.close_device(device)
