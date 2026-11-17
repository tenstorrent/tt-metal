import math, torch, ttnn
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)
try:
    a_shape, b_shape = (1, 64, 16384), (16384, 512)
    A = torch.randn(a_shape)
    B = torch.randn(b_shape)
    ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 64, 512]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    geom, grid, blk, _ = pd.plan(ta, tb, out, True, device)
    print(
        "PROBE",
        grid,
        blk,
        pd.l1_cb_budget(device, [ta, tb, out]),
        pd._footprint(geom, blk.block_m_tiles, blk.block_n_tiles, blk.block_k_tiles, blk.use_partials),
    )
    try:
        genify_matmul(ta, tb, dtype=ttnn.float32)
    except Exception as e:
        print("PROBE EXC", str(e)[:1500])
finally:
    ttnn.close_device(device)
