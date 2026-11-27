import torch, ttnn
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)
bf16 = ttnn.bfloat16
print("grid", device.compute_with_storage_grid_size(), "dram banks", device.dram_grid_size())
for m, k, n, epi in [
    (256, 6144, 18432, "swiglu"),
    (256, 6144, 9216, "none"),
    (1, 6144, 6144, "none"),
    (256, 9216, 6144, "none"),
    (256, 15360, 3072, "none"),
    (256, 6144, 3072, "none"),
]:
    ta = ttnn.allocate_tensor_on_device(ttnn.Shape([1, m, k]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    tb = ttnn.allocate_tensor_on_device(ttnn.Shape([k, n]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    nout = n // 2 if epi == "swiglu" else n
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, m, nout]), bf16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    g, grid, blk, _ = pd.plan(ta, tb, out, True, device, epilogue=epi)
    print(
        (m, k, n, epi),
        grid,
        blk,
        "mcast",
        pd.uses_operand_mcast(g, grid, blk),
        "traffic MB",
        pd.read_traffic(g, grid, blk) / 1e6,
    )
    for t in (ta, tb, out):
        ttnn.deallocate(t)
ttnn.close_device(device)
