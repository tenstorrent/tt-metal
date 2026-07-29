import os, torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

device = ttnn.open_device(device_id=0)
try:
    for shape, lever in [
        ((1, 1, 32, 16384), "1"),
        ((1, 1, 32, 8192), "1"),
        ((1, 1, 32, 4096), "1"),
        ((1, 1, 2048, 2048), "1"),
        ((1, 1, 2048, 32), "1"),
    ]:
        os.environ["TILIZE_LEVER_R2B"] = lever
        os.environ["TILIZE_LEVER_B13"] = "0"
        t = torch.arange(shape[2] * shape[3], dtype=torch.int32).reshape(shape).to(torch.bfloat16)
        ti = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        probe_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(ti.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        plan = tpd.build_plan(ti, probe_out, device, use_multicore=True, use_double_buffer=None)
        print(
            f"{shape} ncores={plan['ncores']} n_h={plan['n_h']} n_w={plan['n_w']} chunk={plan['chunk_wt']} "
            f"chunk_row_bytes={plan['chunk_row_bytes']} blk={plan['blocks_per_core']} depth={plan['depth']} "
            f"fanin={plan['fanin_mode']} groups={plan['fanin_groups']} piece={plan['piece_bytes']} cbB={plan['cb_bytes_per_core']}"
        )
        out = tilize(ti)
        back = ttnn.to_torch(out)
        ok = torch.equal(back.float(), t.float())
        print(f"   bit-exact={ok}" + ("" if ok else f"  maxdiff={(back.float()-t.float()).abs().max()}"))
        ttnn.deallocate(out)
        ttnn.deallocate(ti)
        ttnn.deallocate(probe_out)
finally:
    ttnn.close_device(device)
