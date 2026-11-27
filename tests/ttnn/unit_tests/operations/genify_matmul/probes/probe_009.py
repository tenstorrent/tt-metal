import math, torch, ttnn
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd


def old_budget(device, op_tensors):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    return view.total_bytes_per_bank - max(
        pd._l1_tensor_bytes_per_core(op_tensors, view.num_banks), view.total_bytes_allocated_per_bank
    )


device = ttnn.open_device(device_id=0)
try:
    for shape in [((1, 64, 16384), (16384, 512)), ((1, 384, 1536), (1536, 1024))]:
        for label, fn in [("old", old_budget), ("new", pd.l1_cb_budget)]:
            small = ttnn.from_torch(
                torch.randn(8192, 512),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            big = ttnn.from_torch(
                torch.randn(2048, 512),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            small.deallocate()
            ta = ttnn.from_torch(torch.randn(*shape[0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tb = ttnn.from_torch(torch.randn(*shape[1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            v = ttnn.get_memory_view(device, ttnn.BufferType.L1)
            saved = pd.l1_cb_budget
            pd.l1_cb_budget = fn
            try:
                genify_matmul(ta, tb)
                print("PROBE", shape[0], label, "OK", fn(device, []), v.total_bytes_allocated_per_bank)
            except Exception as e:
                print("PROBE", shape[0], label, "EXC clash=", "clash" in str(e))
            finally:
                pd.l1_cb_budget = saved
            big.deallocate()
finally:
    ttnn.close_device(device)
