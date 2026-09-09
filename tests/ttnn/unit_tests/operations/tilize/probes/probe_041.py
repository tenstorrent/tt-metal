"""Measured per-core L1 across the dtype pair, [1,1,2048,2048] ROW_MAJOR -> 32."""
import importlib, torch, ttnn

M = importlib.import_module("ttnn.operations.tilize.tilize")
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
shape = (1, 1, 2048, 2048)
TD = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.uint8: torch.uint8}
for i, o in (
    (ttnn.bfloat16, ttnn.bfloat16),
    (ttnn.bfloat16, ttnn.float32),
    (ttnn.float32, ttnn.float32),
    (ttnn.float32, ttnn.bfloat4_b),
    (ttnn.uint8, ttnn.uint8),
):
    x = torch.zeros(shape, dtype=TD[i])
    tt = ttnn.from_torch(x, dtype=i, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    spec = M._output_tensor_spec(shape, o, ttnn.DRAM_MEMORY_CONFIG, ttnn.Tile([32, 32]))
    out = ttnn.allocate_tensor_on_device(spec, device)
    p = pd.derive_plan(tt, out, low_l1=False, grid=grid)
    print(
        f"{str(i)[9:]:9s}->{str(o)[9:]:10s} tb_in={p.in_page_bytes:5d} tb_out={p.out_page_bytes:5d} "
        f"bw={p.block_width_tiles:3d} wrpb={p.write_rows_per_barrier} L1={p.l1_per_core_bytes}",
        flush=True,
    )
    out.deallocate()
    tt.deallocate()
ttnn.close_device(device)
