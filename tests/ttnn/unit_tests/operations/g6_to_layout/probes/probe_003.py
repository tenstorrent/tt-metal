import sys, torch, ttnn
import ttnn.operations.g6_to_layout

pd = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor"]
op = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout"]
device = ttnn.open_device(device_id=0)
try:
    for extra in (0, 65536):
        pd.L1_BUDGET_BYTES = 1347776 + extra
        op._PLAN_CACHE.clear()
        op._DESCRIPTOR_CACHE.clear()
        shape = (1, 1024, 1024, 3)
        x = torch.randn(shape, dtype=torch.bfloat16)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
        print("free", view.largest_contiguous_bytes_free_per_bank)
        pages = (view.largest_contiguous_bytes_free_per_bank - 1347776) // 2048
        print("filler pages per bank", pages)
        filler = ttnn.allocate_tensor_on_device(
            ttnn.Shape([pages * view.num_banks, 1024]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG,
        )
        try:
            out = op.g6_to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
            ok = torch.equal(ttnn.to_torch(out).reshape(shape).view(torch.int16), x.view(torch.int16))
            print("RESULT extra", extra, "ran, exact", ok)
        except Exception as e:
            print("RESULT extra", extra, "raised", str(e).replace("\n", " ")[:250])
        del filler
finally:
    ttnn.close_device(device)
