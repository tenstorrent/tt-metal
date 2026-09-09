"""Empty (0-element) tensor: what do the plan's inputs look like?"""
import importlib, torch, ttnn, traceback

M = importlib.import_module("ttnn.operations.tilize.tilize")
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
x = torch.rand(0)
tt = ttnn.from_torch(
    x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
print(
    "in shape",
    list(tt.shape),
    "padded",
    list(tt.padded_shape),
    "page",
    tt.buffer_page_size(),
    "npages",
    tt.buffer_num_pages(),
    "addr",
    tt.buffer_address(),
    flush=True,
)
spec = M._output_tensor_spec([0], ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, ttnn.Tile([32, 32]), [32, 0], None)
out = ttnn.allocate_tensor_on_device(spec, device)
print(
    "out shape",
    list(out.shape),
    "padded",
    list(out.padded_shape),
    "page",
    out.buffer_page_size(),
    "npages",
    out.buffer_num_pages(),
    "addr",
    out.buffer_address(),
    flush=True,
)
try:
    M.tilize(tt, ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0, output_padded_shape=[32, 0])
except Exception:
    traceback.print_exc()
ttnn.close_device(device)
