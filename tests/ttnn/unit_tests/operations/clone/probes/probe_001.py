import torch, ttnn, struct
from ttnn.operations.clone.clone_program_artifacts import create_program_artifacts
device = ttnn.open_device(device_id=0)
try:
    cases = []
    for dt, tdt in [(ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)]:
        for layout in [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]:
            for shape in [(1,1,17,50), (2,1,100,47), (1,32,4096), (32,17)]:
                for in_mem, out_mem in [(ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG), (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG), (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)]:
                    cases.append((dt, tdt, layout, shape, in_mem, out_mem))
    bad = 0
    for dt, tdt, layout, shape, in_mem, out_mem in cases:
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.float32)
        if tdt == torch.float32:
            x = x + 1e-4  # values not representable in bf16
        x = x.to(tdt)
        tin = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=in_mem)
        spec = ttnn.TensorSpec(tin.shape, tin.dtype, tin.layout, out_mem, tile=tin.tile)
        tout = ttnn.allocate_tensor_on_device(spec, device)
        s, ra, idx = create_program_artifacts(tin, tout)
        got = ttnn.to_torch(ttnn.generic_op([tin, tout], s, ra, idx))
        ok = torch.equal(got.view(torch.int16 if tdt==torch.bfloat16 else torch.int32), x.view(torch.int16 if tdt==torch.bfloat16 else torch.int32))
        bad += not ok
        if not ok:
            print("MISMATCH", dt, layout, shape, in_mem.buffer_type, out_mem.buffer_type)
    print(f"bitwise-equal {len(cases)-bad}/{len(cases)}")
finally:
    ttnn.close_device(device)
