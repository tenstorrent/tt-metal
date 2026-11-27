import torch, ttnn
from ttnn.operations.g6_to_layout import g6_to_layout_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:

    def bits(t):
        return [hex(int(v) & 0xFFFF) for v in t.contiguous().view(torch.int16).reshape(-1)]

    specials = [0x7FC1, 0xFFFF, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x3F80, 0x7FFF]
    for width, label in [(32, "pack_untilize w=1"), (64, "fast_untilize w=2")]:
        x = torch.zeros(1, 32, width, dtype=torch.bfloat16)
        x[0, 0, :8] = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in specials], dtype=torch.int16).view(
            torch.bfloat16
        )
        for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
            t = ttnn.from_torch(
                x,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **({"preserve_nan_values": True} if dtype == ttnn.bfloat16 else {}),
            )
            grid = device.compute_with_storage_grid_size()
            plan = pd.compute_plan(
                tuple(x.shape),
                (grid.x, grid.y),
                output_in_l1=False,
                num_l1_banks=1,
                out_alignment=ttnn.get_dram_alignment(),
                dtype=ttnn.bfloat8_b,
                in_tile_bytes=ttnn.tile_size(dtype),
            )
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            prog = pd.G6ToLayoutProgram(plan, t, out)
            y = ttnn.to_torch(ttnn.generic_op([t, out], prog.descriptor_for(t, out)))
            print(label, dtype, "in ", [hex(b) for b in specials])
            print(label, dtype, "out", bits(y[0, 0, :8]))
finally:
    ttnn.close_device(device)
